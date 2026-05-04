#!/usr/bin/env python3
"""
Evaluate the current goals pick engine by reconstructed league rounds.

The historical backtest CSVs in reports/ are daily/as-of prediction snapshots
without a matchweek column. For strategy tuning we reconstruct league rounds by
sorting fixtures chronologically and chunking by the league's usual fixtures per
round, then we run the pick engine once per reconstructed round.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgm.pick_engine_goals import build_picks


DEFAULT_INPUTS = {
    "Premier League": "reports/backtest_premier_fullseason.csv",
    "Serie A": "reports/backtest_serie_a_fullseason.csv",
    "Ligue 1": "reports/backtest_ligue1_fullseason.csv",
}
DEFAULT_ROUND_SIZES = {
    "Premier League": 10,
    "Serie A": 10,
    "Ligue 1": 9,
}


def _settle(row: pd.Series) -> bool | None:
    if pd.isna(row.get("ft_home")) or pd.isna(row.get("ft_away")):
        return None
    home_goals = int(float(row["ft_home"]))
    away_goals = int(float(row["ft_away"]))
    total_goals = home_goals + away_goals
    market = str(row.get("market", ""))
    if market == "OU25_OVER":
        return total_goals > 2.5
    if market == "OU25_UNDER":
        return total_goals < 2.5
    if market == "BTTS_YES":
        return home_goals > 0 and away_goals > 0
    if market == "BTTS_NO":
        return not (home_goals > 0 and away_goals > 0)
    return None


def _odds_bucket(odds: object) -> str:
    try:
        value = float(odds)
    except Exception:
        return "missing"
    if not pd.notna(value):
        return "missing"
    if value < 1.70:
        return "<1.70"
    if value < 2.10:
        return "1.70-2.09"
    if value < 2.60:
        return "2.10-2.59"
    return "2.60+"


def _prob_bucket(prob: object) -> str:
    try:
        value = float(prob)
    except Exception:
        return "missing"
    if not pd.notna(value):
        return "missing"
    if value < 0.52:
        return "<0.52"
    if value < 0.56:
        return "0.52-0.55"
    if value < 0.60:
        return "0.56-0.59"
    if value < 0.65:
        return "0.60-0.64"
    return "0.65+"


def _max_loss_run(hits: pd.Series) -> int:
    longest = 0
    current = 0
    for hit in hits.astype(bool).tolist():
        if hit:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def _max_drawdown(hit_series: pd.Series) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for hit in hit_series.astype(bool).tolist():
        equity += 1.0 if hit else -1.0
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return float(max_dd)


def _group_summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=by + ["picks", "wins", "losses", "hit_rate"])
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(by, keys)}
        wins = int(group["hit"].sum())
        row.update(
            {
                "picks": int(len(group)),
                "wins": wins,
                "losses": int(len(group) - wins),
                "hit_rate": float(group["hit"].mean()),
                "avg_p_model": float(pd.to_numeric(group["p_model"], errors="coerce").mean()),
                "primary_gate_share": float((group.get("gate_tier", "") == "primary").mean())
                if "gate_tier" in group.columns
                else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["picks"] + by, ascending=[False] + [True] * len(by), kind="mergesort")


def _round_chunks(df: pd.DataFrame, *, league: str, round_size: int) -> Iterable[tuple[int, pd.DataFrame]]:
    key_cols = ["fixture_datetime", "home", "away"]
    fixtures = (
        df[key_cols]
        .drop_duplicates()
        .sort_values(["fixture_datetime", "home", "away"], kind="mergesort")
        .reset_index(drop=True)
    )
    fixtures["round_idx"] = (fixtures.index // int(round_size)) + 1
    tagged = df.merge(fixtures, on=key_cols, how="left")
    for round_idx, chunk in tagged.groupby("round_idx", sort=True):
        yield int(round_idx), chunk.copy()


def _prepare_round_input(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk.copy()
    fixture_dt = pd.to_datetime(out["fixture_datetime"], errors="coerce")
    first_dt = fixture_dt.dropna().min()
    if pd.isna(first_dt):
        asof = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
    else:
        asof = first_dt.normalize() - pd.Timedelta(microseconds=1)
    out["run_asof_datetime"] = asof.isoformat()
    out["horizon_days"] = 14
    out["next_round_only"] = 1
    return out


def _summarize(scored: pd.DataFrame, *, total_rounds: int, label: str) -> dict[str, object]:
    if scored.empty:
        return {
            "window": label,
            "rounds": total_rounds,
            "picks": 0,
            "avg_picks_per_round": 0.0,
            "min_picks_per_round": 0,
            "rounds_ge_4_picks": 0,
            "wins": 0,
            "losses": 0,
            "hit_rate": float("nan"),
            "max_loss_run": 0,
            "max_drawdown_units": 0.0,
            "priced_picks_odds_gt_1_7": 0,
            "priced_roi_odds_gt_1_7": float("nan"),
            "primary_gate_share": float("nan"),
        }

    round_counts = scored.groupby("round_idx").size()
    ordered = scored.sort_values(["round_idx", "fixture_datetime", "home", "away"], kind="mergesort")
    priced = scored[pd.to_numeric(scored["odds"], errors="coerce") > 1.70].copy()
    if not priced.empty:
        priced["profit"] = priced.apply(
            lambda row: float(row["odds"]) - 1.0 if bool(row["hit"]) else -1.0,
            axis=1,
        )
    return {
        "window": label,
        "rounds": total_rounds,
        "picks": int(len(scored)),
        "avg_picks_per_round": float(round_counts.mean()),
        "min_picks_per_round": int(round_counts.min()),
        "rounds_ge_4_picks": int((round_counts >= 4).sum()),
        "wins": int(scored["hit"].sum()),
        "losses": int(len(scored) - int(scored["hit"].sum())),
        "hit_rate": float(scored["hit"].mean()),
        "max_loss_run": _max_loss_run(ordered["hit"]),
        "max_drawdown_units": _max_drawdown(ordered["hit"]),
        "priced_picks_odds_gt_1_7": int(len(priced)),
        "priced_roi_odds_gt_1_7": float(priced["profit"].mean()) if not priced.empty else float("nan"),
        "primary_gate_share": float((scored.get("gate_tier", "") == "primary").mean())
        if "gate_tier" in scored.columns
        else float("nan"),
    }


def evaluate_league(league: str, path: Path, *, round_size: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    actuals = df[["fixture_datetime", "home", "away", "ft_home", "ft_away"]].drop_duplicates()
    scored_parts: list[pd.DataFrame] = []

    for round_idx, chunk in _round_chunks(df, league=league, round_size=round_size):
        prepared = _prepare_round_input(chunk)
        picks, _debug = build_picks(prepared, input_hash=f"{league}-{round_idx}", run_id="round_eval")
        if picks.empty:
            continue
        picks["round_idx"] = round_idx
        scored = picks.merge(actuals, on=["fixture_datetime", "home", "away"], how="left")
        scored["hit"] = scored.apply(_settle, axis=1)
        scored = scored[scored["hit"].notna()].copy()
        scored["odds_bucket"] = scored["odds"].apply(_odds_bucket)
        scored["prob_bucket"] = scored["p_model"].apply(_prob_bucket)
        scored_parts.append(scored)

    scored_all = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    if not scored_all.empty:
        scored_all.insert(0, "league_group", league)
    total_rounds = int((df[["fixture_datetime", "home", "away"]].drop_duplicates().shape[0] + round_size - 1) // round_size)

    windows: list[dict[str, object]] = []
    if not scored_all.empty:
        windows.append(_summarize(scored_all[scored_all["round_idx"] == 1], total_rounds=1, label="round_1"))
        windows.append(_summarize(scored_all[scored_all["round_idx"].between(1, 5)], total_rounds=min(5, total_rounds), label="rounds_1_5"))
        windows.append(_summarize(scored_all, total_rounds=total_rounds, label="full_season"))
    else:
        windows.extend(
            [
                _summarize(scored_all, total_rounds=1, label="round_1"),
                _summarize(scored_all, total_rounds=min(5, total_rounds), label="rounds_1_5"),
                _summarize(scored_all, total_rounds=total_rounds, label="full_season"),
            ]
        )

    summary = pd.DataFrame(windows)
    scored_path = out_dir / f"{league.lower().replace(' ', '_')}_round_picks.csv"
    summary_path = out_dir / f"{league.lower().replace(' ', '_')}_round_summary.csv"
    scored_all.to_csv(scored_path, index=False)
    summary.insert(0, "league", league)
    summary.to_csv(summary_path, index=False)
    return summary, scored_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pick engine by reconstructed league rounds")
    parser.add_argument("--out-dir", default="reports/round_strategy_eval", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    scored: list[pd.DataFrame] = []
    for league, input_path in DEFAULT_INPUTS.items():
        summary, scored_league = evaluate_league(
            league,
            Path(input_path),
            round_size=DEFAULT_ROUND_SIZES[league],
            out_dir=out_dir,
        )
        summaries.append(summary)
        if not scored_league.empty:
            scored.append(scored_league)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(out_dir / "summary.csv", index=False)
    if scored:
        all_scored = pd.concat(scored, ignore_index=True)
        all_scored.to_csv(out_dir / "all_round_picks.csv", index=False)
        _group_summary(all_scored, ["league_group", "market"]).to_csv(out_dir / "market_summary.csv", index=False)
        _group_summary(all_scored, ["league_group", "odds_bucket"]).to_csv(out_dir / "odds_bucket_summary.csv", index=False)
        _group_summary(all_scored, ["league_group", "prob_bucket"]).to_csv(out_dir / "prob_bucket_summary.csv", index=False)
        round_counts = (
            all_scored.groupby(["league_group", "round_idx"], as_index=False)
            .agg(picks=("hit", "size"), wins=("hit", "sum"))
            .sort_values(["league_group", "round_idx"], kind="mergesort")
        )
        round_counts["hit_rate"] = round_counts["wins"] / round_counts["picks"]
        round_counts.to_csv(out_dir / "round_counts.csv", index=False)
    print(all_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
