"""
Generate a plain-language business report focused on model probabilities only.

Sections:
1) Last N played rounds (results)
2) Next round predictions (OU2.5 + BTTS) from internal model probabilities
3) Backtest accuracy summary (probability threshold 0.5, no EV/odds dependency)
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "reports" / "cgm_upcoming_predictions.csv"
DEFAULT_REPORT_STEM = ROOT / "reports" / "business_report"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)


def _choose_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    return float(v)


def _to_int(value: Any) -> int | None:
    v = _to_num(value)
    if v is None:
        return None
    return int(round(v))


def _norm_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _history_source_score(path: Path) -> tuple[pd.Timestamp, int]:
    df = _safe_read_csv(path)
    if df.empty:
        return pd.Timestamp.min, 0
    date_col = _choose_col(df, ["date", "fixture_date", "datameci"])
    gh_col = _choose_col(df, ["ft_home", "FTHG", "scor1", "score_home"])
    ga_col = _choose_col(df, ["ft_away", "FTAG", "scor2", "score_away"])
    if date_col is None:
        return pd.Timestamp.min, 0
    dt_series = pd.to_datetime(df[date_col], errors="coerce")
    valid_mask = dt_series.notna()
    if gh_col and ga_col:
        gh = pd.to_numeric(df[gh_col], errors="coerce")
        ga = pd.to_numeric(df[ga_col], errors="coerce")
        valid_mask = valid_mask & gh.notna() & ga.notna()
    valid_rows = int(valid_mask.sum())
    if valid_rows <= 0:
        return pd.Timestamp.min, 0
    latest_dt = pd.to_datetime(dt_series[valid_mask], errors="coerce").max()
    if pd.isna(latest_dt):
        return pd.Timestamp.min, valid_rows
    return pd.Timestamp(latest_dt), valid_rows


def _resolve_history_path(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        return None

    candidates = [
        ROOT / "data" / "api_football" / "history_fixtures.csv",
        ROOT / "data" / "enhanced" / "cgm_match_history_with_elo_stats_xg.csv",
        ROOT / "data" / "enhanced" / "cgm_match_history_with_elo_stats.csv",
        ROOT / "data" / "enhanced" / "cgm_match_history.csv",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    ranked: list[tuple[pd.Timestamp, int, float, Path]] = []
    for path in existing:
        latest_dt, valid_rows = _history_source_score(path)
        ranked.append((latest_dt, valid_rows, path.stat().st_mtime, path))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return ranked[0][3]


def _resolve_backtest_path(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    reports_dir = ROOT / "reports"
    candidates: list[Path] = []
    for pattern in ("backtest*.csv", "full_backtest*.csv", "latest_backtest*.csv"):
        candidates.extend(reports_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_results_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    date_col = _choose_col(df, ["date", "fixture_date", "datameci"])
    home_col = _choose_col(df, ["home", "home_name", "txtechipa1"])
    away_col = _choose_col(df, ["away", "away_name", "txtechipa2"])
    gh_col = _choose_col(df, ["ft_home", "FTHG", "scor1", "score_home"])
    ga_col = _choose_col(df, ["ft_away", "FTAG", "scor2", "score_away"])
    return date_col, home_col, away_col, gh_col, ga_col


def _outcome_from_score(ft_home: Any, ft_away: Any) -> dict[str, str] | None:
    h = _to_int(ft_home)
    a = _to_int(ft_away)
    if h is None or a is None:
        return None
    total = h + a
    btts_yes = (h > 0 and a > 0)
    return {
        "ou_actual": "OU25_OVER" if total > 2 else "OU25_UNDER",
        "btts_actual": "BTTS_YES" if btts_yes else "BTTS_NO",
    }


def _classify_ou_from_probs(p_over: float | None, p_under: float | None) -> tuple[str, float | None]:
    """Classify O/U side directly from model probabilities (no fixed threshold)."""
    if p_over is None and p_under is None:
        return "", None
    po = p_over if p_over is not None else 0.0
    pu = p_under if p_under is not None else (1.0 - po)
    pred = "OU25_OVER" if po >= pu else "OU25_UNDER"
    return pred, max(po, pu)


def _build_backtest_lookup(df: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    if df.empty:
        return {}

    date_col = _choose_col(df, ["date", "fixture_date", "_backtest_date"])
    home_col = _choose_col(df, ["home"])
    away_col = _choose_col(df, ["away"])
    if date_col is None or home_col is None or away_col is None:
        return {}

    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    bt = df.copy()
    bt["_date"] = pd.to_datetime(bt[date_col], errors="coerce").dt.date.astype(str)

    for _, row in bt.iterrows():
        p_over = _to_num(row.get("p_over25", row.get("p_over_2_5")))
        p_under = _to_num(row.get("p_under25", row.get("p_under_2_5")))
        p_btts = _to_num(row.get("p_btts_yes"))
        if p_over is None and p_btts is None:
            continue

        ou_pred, _ = _classify_ou_from_probs(p_over, p_under)
        btts_pred = "BTTS_YES" if (p_btts is not None and p_btts >= 0.5) else "BTTS_NO"

        outcomes = _outcome_from_score(
            row.get("ft_home", row.get("FTHG")),
            row.get("ft_away", row.get("FTAG")),
        )
        ou_hit = ""
        btts_hit = ""
        if outcomes is not None:
            ou_hit = "won" if ou_pred == outcomes["ou_actual"] else "lost"
            btts_hit = "won" if btts_pred == outcomes["btts_actual"] else "lost"

        key = (
            str(row["_date"]),
            _norm_name(row.get(home_col)),
            _norm_name(row.get(away_col)),
        )
        out[key] = {
            "ou_pred": ou_pred,
            "btts_pred": btts_pred,
            "ou_hit": ou_hit,
            "btts_hit": btts_hit,
        }
    return out


def _build_recent_results(
    history_df: pd.DataFrame,
    *,
    rounds: int,
    league_filter: str | None,
    bt_lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if history_df.empty:
        return pd.DataFrame(), None

    date_col, home_col, away_col, gh_col, ga_col = _extract_results_columns(history_df)
    if None in (date_col, home_col, away_col, gh_col, ga_col):
        return pd.DataFrame(), None

    league_col = _choose_col(history_df, ["league"])
    season_col = _choose_col(history_df, ["season"])

    df = history_df.copy()
    df["_date_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["_date_dt"].notna()].copy()
    df["_gh"] = pd.to_numeric(df[gh_col], errors="coerce")
    df["_ga"] = pd.to_numeric(df[ga_col], errors="coerce")
    df = df[df["_gh"].notna() & df["_ga"].notna()].copy()

    latest_played_dt = df["_date_dt"].max() if not df.empty else None

    if league_filter and league_col:
        needle = league_filter.strip().lower()
        df = df[df[league_col].astype(str).str.lower().str.contains(needle, na=False)]

    if df.empty:
        return pd.DataFrame(), latest_played_dt

    recent_dates = sorted(df["_date_dt"].dt.date.unique(), reverse=True)[: max(1, rounds)]
    df = df[df["_date_dt"].dt.date.isin(recent_dates)].copy()
    df = df.sort_values(["_date_dt", league_col or home_col, home_col, away_col], ascending=[False, True, True, True])

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        date_iso = row["_date_dt"].date().isoformat()
        home = str(row[home_col])
        away = str(row[away_col])
        gh = _to_int(row["_gh"])
        ga = _to_int(row["_ga"])
        score = f"{gh}-{ga}" if gh is not None and ga is not None else ""

        bt = bt_lookup.get((date_iso, _norm_name(home), _norm_name(away)))
        rows.append(
            {
                "date": date_iso,
                "league": str(row[league_col]) if league_col else "",
                "season": str(row[season_col]) if season_col else "",
                "home": home,
                "away": away,
                "score": score,
                "bot_prediction_available": "yes" if bt else "no",
                "bot_ou_pred": (bt or {}).get("ou_pred", ""),
                "bot_btts_pred": (bt or {}).get("btts_pred", ""),
                "bot_ou_result": (bt or {}).get("ou_hit", ""),
                "bot_btts_result": (bt or {}).get("btts_hit", ""),
            }
        )

    return pd.DataFrame(rows), latest_played_dt


def _build_upcoming_summary(
    predictions_df: pd.DataFrame,
    *,
    league_filter: str | None,
    limit: int,
    next_round_span_days: int,
) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    df = predictions_df.copy()
    dt_col = _choose_col(df, ["fixture_datetime", "datetime", "kickoff_utc"])
    home_col = _choose_col(df, ["home", "home_name", "txtechipa1"])
    away_col = _choose_col(df, ["away", "away_name", "txtechipa2"])
    league_col = _choose_col(df, ["league"])
    if None in (dt_col, home_col, away_col):
        return pd.DataFrame()

    df["_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df[df["_dt"].notna()].copy()

    now_utc = pd.Timestamp.now(tz="UTC")
    if df["_dt"].dt.tz is None:
        df = df[df["_dt"] >= pd.Timestamp(now_utc.date())]
    else:
        df = df[df["_dt"] >= now_utc]

    if league_filter and league_col:
        needle = league_filter.strip().lower()
        df = df[df[league_col].astype(str).str.lower().str.contains(needle, na=False)]

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["_dt", league_col or home_col, home_col, away_col], ascending=[True, True, True, True])

    first_dt = df["_dt"].min()
    if pd.notna(first_dt):
        round_end = first_dt + pd.Timedelta(days=max(0, int(next_round_span_days)))
        df = df[df["_dt"] <= round_end].copy()

    df = df.head(max(1, limit))

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        p_over = _to_num(row.get("p_over25", row.get("p_over_2_5")))
        p_under = _to_num(row.get("p_under25", row.get("p_under_2_5")))
        p_btts_yes = _to_num(row.get("p_btts_yes"))
        p_btts_no = _to_num(row.get("p_btts_no"))

        ou_pred, ou_conf = _classify_ou_from_probs(p_over, p_under)

        if p_btts_yes is None and p_btts_no is None:
            btts_pred = ""
            btts_conf = None
        else:
            py = p_btts_yes if p_btts_yes is not None else 0.0
            pn = p_btts_no if p_btts_no is not None else (1.0 - py)
            btts_pred = "BTTS_YES" if py >= pn else "BTTS_NO"
            btts_conf = max(py, pn)

        rows.append(
            {
                "fixture_datetime": pd.to_datetime(row["_dt"]).isoformat(),
                "date": pd.to_datetime(row["_dt"]).date().isoformat(),
                "league": str(row[league_col]) if league_col else "",
                "home": str(row[home_col]),
                "away": str(row[away_col]),
                "pred_ou25": ou_pred,
                "pred_ou25_confidence_pct": round(float(ou_conf) * 100.0, 2) if ou_conf is not None else "",
                "pred_btts": btts_pred,
                "pred_btts_confidence_pct": round(float(btts_conf) * 100.0, 2) if btts_conf is not None else "",
            }
        )
    return pd.DataFrame(rows)


def _backtest_summary(backtest_df: pd.DataFrame) -> dict[str, Any]:
    if backtest_df.empty:
        return {"available": False, "message": "No backtest file found."}

    rows = []
    for _, row in backtest_df.iterrows():
        score = _outcome_from_score(row.get("ft_home", row.get("FTHG")), row.get("ft_away", row.get("FTAG")))
        if score is None:
            continue
        p_over = _to_num(row.get("p_over25", row.get("p_over_2_5")))
        p_under = _to_num(row.get("p_under25", row.get("p_under_2_5")))
        p_btts_yes = _to_num(row.get("p_btts_yes"))
        if p_over is None and p_btts_yes is None:
            continue

        ou_pred, _ = _classify_ou_from_probs(p_over, p_under)
        btts_pred = "BTTS_YES" if (p_btts_yes is not None and p_btts_yes >= 0.5) else "BTTS_NO"

        rows.append(
            {
                "ou_hit": int(ou_pred == score["ou_actual"]),
                "btts_hit": int(btts_pred == score["btts_actual"]),
            }
        )

    if not rows:
        return {
            "available": False,
            "message": "Backtest exists but has no rows with both predictions and final scores.",
        }

    df = pd.DataFrame(rows)
    both = (df["ou_hit"] & df["btts_hit"]).astype(int)
    return {
        "available": True,
        "rows": int(len(df)),
        "ou_accuracy": float(df["ou_hit"].mean()),
        "btts_accuracy": float(df["btts_hit"].mean()),
        "both_accuracy": float(both.mean()),
    }


def _render_text(
    *,
    recent_df: pd.DataFrame,
    upcoming_df: pd.DataFrame,
    backtest_info: dict[str, Any],
    rounds: int,
    league_filter: str | None,
    history_path: Path | None,
    predictions_path: Path,
    backtest_path: Path | None,
    latest_played_dt: pd.Timestamp | None,
) -> str:
    lines: list[str] = []
    now_text = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("BUSINESS REPORT (PROBABILITY-ONLY)")
    lines.append(f"Generated: {now_text}")
    lines.append("")
    lines.append("Data source files:")
    lines.append(f"- History: {history_path if history_path else 'not found'}")
    lines.append(f"- Predictions: {predictions_path}")
    lines.append(f"- Backtest: {backtest_path if backtest_path else 'not found'}")
    if league_filter:
        lines.append(f"- League filter: {league_filter}")
    lines.append("")

    lines.append(f"SECTION 1 - Last {rounds} played rounds")
    if latest_played_dt is not None and not pd.isna(latest_played_dt):
        latest_str = pd.Timestamp(latest_played_dt).date().isoformat()
        lines.append(f"Latest played date in source history: {latest_str}")
    if recent_df.empty:
        lines.append("No played matches found in selected history source.")
    else:
        lines.append(f"Rows listed: {len(recent_df)}")
        show_cols = [
            "date",
            "league",
            "home",
            "away",
            "score",
            "bot_prediction_available",
            "bot_ou_pred",
            "bot_btts_pred",
            "bot_ou_result",
            "bot_btts_result",
        ]
        lines.append(recent_df[[c for c in show_cols if c in recent_df.columns]].head(50).to_string(index=False))
    lines.append("")

    lines.append("SECTION 2 - Next round predictions (no odds, model-only)")
    if upcoming_df.empty:
        lines.append("No upcoming fixtures found in prediction output.")
    else:
        lines.append(f"Rows listed: {len(upcoming_df)}")
        show_cols = [
            "date",
            "league",
            "home",
            "away",
            "pred_ou25",
            "pred_ou25_confidence_pct",
            "pred_btts",
            "pred_btts_confidence_pct",
        ]
        lines.append(upcoming_df[[c for c in show_cols if c in upcoming_df.columns]].to_string(index=False))
    lines.append("")

    lines.append("SECTION 3 - Backtest accuracy summary (0.5 probability threshold)")
    if not backtest_info.get("available"):
        lines.append(str(backtest_info.get("message", "No backtest summary available.")))
    else:
        lines.append(f"Rows evaluated: {backtest_info['rows']}")
        lines.append(f"OU2.5 accuracy: {backtest_info['ou_accuracy'] * 100.0:.2f}%")
        lines.append(f"BTTS accuracy: {backtest_info['btts_accuracy'] * 100.0:.2f}%")
        lines.append(f"Both-correct accuracy: {backtest_info['both_accuracy'] * 100.0:.2f}%")
    lines.append("")

    lines.append("Conclusion:")
    if latest_played_dt is None or pd.isna(latest_played_dt):
        lines.append("- History source has no valid played-date rows.")
    else:
        now_naive = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        latest_naive = pd.Timestamp(latest_played_dt).tz_localize(None).normalize()
        lag_days = (now_naive - latest_naive).days
        if lag_days > 7:
            lines.append(
                f"- History looks stale (latest played date is {int(lag_days)} days old). "
                "This is why recent weekends may not appear."
            )
        else:
            lines.append("- History looks recent enough for last-round reporting.")
    if upcoming_df.empty:
        lines.append("- No next-round fixtures were available in the prediction file.")
    else:
        lines.append("- Next round predictions are available and limited to one round window.")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate non-technical probability-only business report")
    ap.add_argument("--history", default=None, help="Optional explicit history CSV path. If omitted, auto-select freshest source.")
    ap.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS), help="Upcoming predictions CSV path")
    ap.add_argument("--backtest", default=None, help="Optional explicit backtest CSV path")
    ap.add_argument("--rounds", type=int, default=5, help="How many recent matchdays to show")
    ap.add_argument("--upcoming-limit", type=int, default=20, help="Max upcoming fixtures to list")
    ap.add_argument("--league", default=None, help="Optional league filter (contains match)")
    ap.add_argument("--next-round-span-days", type=int, default=3, help="Round window days from earliest upcoming fixture")
    ap.add_argument("--output", default=str(DEFAULT_REPORT_STEM), help="Output path stem (without extension)")
    args = ap.parse_args()

    history_path = _resolve_history_path(args.history)
    predictions_path = Path(args.predictions)
    backtest_path = _resolve_backtest_path(args.backtest)
    output_stem = Path(args.output)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    history_df = _safe_read_csv(history_path) if history_path else pd.DataFrame()
    predictions_df = _safe_read_csv(predictions_path)
    backtest_df = _safe_read_csv(backtest_path) if backtest_path else pd.DataFrame()

    bt_lookup = _build_backtest_lookup(backtest_df)
    recent_df, latest_played_dt = _build_recent_results(
        history_df,
        rounds=max(1, int(args.rounds)),
        league_filter=args.league,
        bt_lookup=bt_lookup,
    )
    upcoming_df = _build_upcoming_summary(
        predictions_df,
        league_filter=args.league,
        limit=max(1, int(args.upcoming_limit)),
        next_round_span_days=max(0, int(args.next_round_span_days)),
    )
    backtest_info = _backtest_summary(backtest_df)

    text = _render_text(
        recent_df=recent_df,
        upcoming_df=upcoming_df,
        backtest_info=backtest_info,
        rounds=max(1, int(args.rounds)),
        league_filter=args.league,
        history_path=history_path,
        predictions_path=predictions_path,
        backtest_path=backtest_path,
        latest_played_dt=latest_played_dt,
    )

    txt_path = output_stem.with_suffix(".txt")
    md_path = output_stem.with_suffix(".md")
    recent_csv = output_stem.parent / f"{output_stem.stem}_recent_results.csv"
    upcoming_csv = output_stem.parent / f"{output_stem.stem}_upcoming_summary.csv"

    txt_path.write_text(text, encoding="utf-8")
    md_path.write_text("```\n" + text + "\n```\n", encoding="utf-8")
    recent_df.to_csv(recent_csv, index=False)
    upcoming_df.to_csv(upcoming_csv, index=False)

    print(f"[ok] wrote {txt_path}")
    print(f"[ok] wrote {md_path}")
    print(f"[ok] wrote {recent_csv}")
    print(f"[ok] wrote {upcoming_csv}")


if __name__ == "__main__":
    main()
