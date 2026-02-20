#!/usr/bin/env python3
"""
Detailed audit report for non-technical review.

Produces a human-readable log that explains:
- What data was used and whether files exist
- How Elo, xG proxy, and Pressure are calculated (plain language)
- Coverage of required stats
- Per-team game counts and evidence windows
- Per-fixture evidence for upcoming predictions
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from cgm.pressure_inputs import ensure_pressure_inputs



def _now_stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _parse_fixture_datetime(df: pd.DataFrame) -> pd.Series:
    if "fixture_datetime_utc" in df.columns:
        return pd.to_datetime(df["fixture_datetime_utc"], errors="coerce")
    if "fixture_datetime" in df.columns:
        return pd.to_datetime(df["fixture_datetime"], errors="coerce")
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")
    if "date" in df.columns:
        if "time" in df.columns:
            dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
            if dt.notna().any():
                return dt
        return pd.to_datetime(df["date"], errors="coerce")
    if "datameci" in df.columns:
        return pd.to_datetime(df["datameci"], errors="coerce")
    return pd.to_datetime(pd.Series([pd.NaT] * len(df)))


def _league_col(df: pd.DataFrame) -> str | None:
    for cand in ("league", "txtliga"):
        if cand in df.columns:
            return cand
    return None


def _team_counts(hist: pd.DataFrame, dt: pd.Series) -> pd.DataFrame:
    hist = hist.copy()
    hist["_dt"] = dt
    home_counts = hist.groupby("home").size().rename("home_matches")
    away_counts = hist.groupby("away").size().rename("away_matches")
    total = home_counts.add(away_counts, fill_value=0).rename("total_matches")
    out = pd.concat([home_counts, away_counts, total], axis=1).fillna(0).reset_index()
    out = out.rename(columns={"index": "team"})
    last_home = hist.groupby("home")["_dt"].max().rename("last_home_date")
    last_away = hist.groupby("away")["_dt"].max().rename("last_away_date")
    last_any = pd.concat([last_home, last_away], axis=1).max(axis=1).rename("last_any_date")
    out = out.merge(last_home.reset_index().rename(columns={"home": "team"}), on="team", how="left")
    out = out.merge(last_away.reset_index().rename(columns={"away": "team"}), on="team", how="left")
    out = out.merge(last_any.reset_index().rename(columns={"index": "team"}), on="team", how="left")
    return out


def _pressure_usable_mask(hist: pd.DataFrame) -> pd.Series:
    if "pressure_usable" in hist.columns:
        return pd.to_numeric(hist["pressure_usable"], errors="coerce").fillna(0.0) > 0
    h = ensure_pressure_inputs(hist)
    required = ["shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A"]
    ok = h[required].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    return ok


def _xg_usable_mask(hist: pd.DataFrame) -> pd.Series:
    if "xg_usable" in hist.columns:
        return pd.to_numeric(hist["xg_usable"], errors="coerce").fillna(0.0) > 0
    if {"xg_proxy_H", "xg_proxy_A"}.issubset(hist.columns):
        return hist[["xg_proxy_H", "xg_proxy_A"]].notna().all(axis=1)
    return pd.Series(False, index=hist.index)


def _write_line(lines: list[str], text: str = "") -> None:
    lines.append(text)


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _minmax_date(dt: pd.Series) -> tuple[str, str]:
    dmin = pd.to_datetime(dt, errors="coerce").min()
    dmax = pd.to_datetime(dt, errors="coerce").max()
    return (str(dmin) if pd.notna(dmin) else "N/A", str(dmax) if pd.notna(dmax) else "N/A")


def _league_summary(df: pd.DataFrame, dt: pd.Series, source: str) -> pd.DataFrame:
    league_col = _league_col(df)
    if league_col is None:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["_dt"] = dt
    summary = (
        tmp.groupby(league_col)
        .agg(
            rows=("_dt", "count"),
            min_date=("_dt", "min"),
            max_date=("_dt", "max"),
        )
        .reset_index()
        .rename(columns={league_col: "league"})
    )
    summary["source"] = source
    summary = summary[["source", "league", "rows", "min_date", "max_date"]]
    return summary


def _describe_calcs(lines: list[str]) -> None:
    _write_line(lines, "How calculations work (plain language):")
    _write_line(lines, "- Elo:")
    _write_line(lines, f"  - Start rating: {getattr(config, 'START_ELO_DEFAULT', 1500)} (from calc_cgm_elo defaults)")
    _write_line(lines, "  - Update rule: expected score uses a logistic curve with home advantage.")
    _write_line(lines, "  - Home advantage: 65 Elo points.")
    _write_line(lines, "  - K-factor: 20, scaled by goal-difference multiplier.")
    _write_line(lines, "  - Margin multiplier: 1.0 for 0-1 goals, 1.5 for 2, 1.75 for 3, then +0.125 per extra goal.")
    _write_line(lines, "- Pressure (team dominance):")
    _write_line(lines, "  - Inputs: shots, shots on target, corners, possession.")
    _write_line(lines, "  - Weights: shots 0.45, shots on target 0.30, corners 0.15, possession 0.10.")
    _write_line(lines, "  - Pressure is only usable when all required inputs exist for a match.")
    _write_line(lines, "- xG proxy (internal, not true xG):")
    _write_line(lines, "  - Inputs: shots_for, shots_on_target_for, shots_against, shots_on_target_against, is_home.")
    _write_line(lines, "  - Model: Poisson regression trained walk-forward by ISO week.")
    _write_line(lines, "  - If not enough training rows, a prior mean (1.25 goals) is used.")
    _write_line(lines, "")


def _audit_history(hist: pd.DataFrame, lines: list[str]) -> tuple[dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    out: dict[str, Any] = {"ok": True, "issues": []}
    dt = _parse_fixture_datetime(hist)
    dmin, dmax = _minmax_date(dt)
    league_col = _league_col(hist)

    _write_line(lines, "History data")
    _write_line(lines, f"- Rows: {len(hist)}")
    _write_line(lines, f"- Date range: {dmin} -> {dmax}")
    if league_col:
        leagues = hist[league_col].astype(str).value_counts().head(10)
        _write_line(lines, "- Top leagues by rows:")
        for league, count in leagues.items():
            _write_line(lines, f"  - {league}: {count}")
    else:
        _write_line(lines, "- League column missing.")
        out["ok"] = False
        out["issues"].append("History missing league column")

    if "home" not in hist.columns or "away" not in hist.columns:
        out["ok"] = False
        out["issues"].append("History missing home/away team columns")

    # Elo coverage
    elo_cols = ["elo_home", "elo_away"]
    missing_elo_cols = [c for c in elo_cols if c not in hist.columns]
    if missing_elo_cols:
        out["ok"] = False
        out["issues"].append(f"Missing Elo columns: {missing_elo_cols}")
    else:
        elo_home = pd.to_numeric(hist["elo_home"], errors="coerce")
        elo_away = pd.to_numeric(hist["elo_away"], errors="coerce")
        _write_line(lines, f"- Elo coverage: home missing={elo_home.isna().mean():.1%}, away missing={elo_away.isna().mean():.1%}")
        _write_line(lines, f"  - Elo home min/med/max: {elo_home.min():.1f} / {elo_home.median():.1f} / {elo_home.max():.1f}")
        _write_line(lines, f"  - Elo away min/med/max: {elo_away.min():.1f} / {elo_away.median():.1f} / {elo_away.max():.1f}")

    # Pressure coverage
    pressure_mask = _pressure_usable_mask(hist)
    pressure_rate = float(pressure_mask.mean()) if len(hist) else 0.0
    _write_line(lines, f"- Pressure usable rate: {_fmt_pct(pressure_rate)}")

    # xG proxy coverage
    xg_mask = _xg_usable_mask(hist)
    xg_rate = float(xg_mask.mean()) if len(hist) else 0.0
    _write_line(lines, f"- xG proxy usable rate: {_fmt_pct(xg_rate)}")

    # Team counts
    counts = None
    if "home" in hist.columns and "away" in hist.columns:
        counts = _team_counts(hist, dt)
        low = counts[counts["total_matches"] < 10].sort_values("total_matches").head(15)
        _write_line(lines, "- Teams with fewer than 10 total history matches (sample):")
        if low.empty:
            _write_line(lines, "  - None")
        else:
            for _, row in low.iterrows():
                _write_line(lines, f"  - {row['team']}: total={int(row['total_matches'])} home={int(row['home_matches'])} away={int(row['away_matches'])}")

    _write_line(lines, "")
    league_summary = _league_summary(hist, dt, "history")
    return out, counts, league_summary


def _audit_upcoming(up: pd.DataFrame, lines: list[str]) -> tuple[dict[str, Any], pd.DataFrame | None]:
    out: dict[str, Any] = {"ok": True, "issues": []}
    dt = _parse_fixture_datetime(up)
    dmin, dmax = _minmax_date(dt)
    league_col = _league_col(up)

    _write_line(lines, "Upcoming fixtures data")
    _write_line(lines, f"- Rows: {len(up)}")
    _write_line(lines, f"- Date range: {dmin} -> {dmax}")
    if league_col:
        leagues = up[league_col].astype(str).value_counts().head(10)
        _write_line(lines, "- Top leagues by rows:")
        for league, count in leagues.items():
            _write_line(lines, f"  - {league}: {count}")
    else:
        _write_line(lines, "- League column missing.")
        out["ok"] = False
        out["issues"].append("Upcoming missing league column")
    _write_line(lines, "")
    league_summary = _league_summary(up, dt, "upcoming")
    return out, league_summary


def _audit_predictions(
    preds: pd.DataFrame,
    hist: pd.DataFrame,
    lines: list[str],
    *,
    sample_limit: int,
    window_n: int,
) -> tuple[dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    out: dict[str, Any] = {"ok": True, "issues": []}
    if preds.empty:
        _write_line(lines, "Predictions data: empty")
        out["ok"] = False
        out["issues"].append("Predictions are empty")
        return out, None, None

    dt = _parse_fixture_datetime(preds)
    dmin, dmax = _minmax_date(dt)
    _write_line(lines, "Predictions data")
    _write_line(lines, f"- Rows: {len(preds)}")
    _write_line(lines, f"- Date range: {dmin} -> {dmax}")

    # Evidence stats
    hist_dt = _parse_fixture_datetime(hist)
    hist = hist.copy()
    hist["_dt"] = hist_dt
    xg_mask = _xg_usable_mask(hist)
    pressure_mask = _pressure_usable_mask(hist)
    hist["_xg_ok"] = xg_mask
    hist["_press_ok"] = pressure_mask

    _write_line(lines, f"- Evidence window: last {window_n} matches (context-specific)")
    _write_line(lines, "- Sample per fixture (first N):")
    shown = 0
    evidence_rows: list[dict[str, Any]] = []
    xg_min = float(getattr(config, "XG_EVID_MIN_GOALS", 3.0))
    press_min = float(getattr(config, "PRESS_EVID_MIN_GOALS", 3.0))
    low_xg_fixtures = 0
    low_press_fixtures = 0
    league_col = _league_col(preds)
    for idx, row in preds.iterrows():
        if shown >= sample_limit:
            break
        home = str(row.get("home", ""))
        away = str(row.get("away", ""))
        fdt = pd.to_datetime(row.get("fixture_datetime") or row.get("fixture_datetime_utc") or row.get("date"), errors="coerce")
        if pd.isna(fdt):
            fdt = hist["_dt"].max() if hist["_dt"].notna().any() else pd.Timestamp.utcnow()

        h_hist = hist[(hist["home"] == home) & (hist["_dt"] < fdt)]
        a_hist = hist[(hist["away"] == away) & (hist["_dt"] < fdt)]

        h_n = int(len(h_hist))
        a_n = int(len(a_hist))
        h_last = h_hist.sort_values("_dt").tail(window_n)
        a_last = a_hist.sort_values("_dt").tail(window_n)

        h_xg = int(h_last["_xg_ok"].sum()) if not h_last.empty else 0
        a_xg = int(a_last["_xg_ok"].sum()) if not a_last.empty else 0
        h_press = int(h_last["_press_ok"].sum()) if not h_last.empty else 0
        a_press = int(a_last["_press_ok"].sum()) if not a_last.empty else 0

        if min(h_xg, a_xg) < xg_min:
            low_xg_fixtures += 1
        if min(h_press, a_press) < press_min:
            low_press_fixtures += 1

        _write_line(
            lines,
            f"  - {home} vs {away}:",
        )
        _write_line(lines, f"    - Home matches available: {h_n} (xG ok last {window_n}: {h_xg}, Pressure ok: {h_press})")
        _write_line(lines, f"    - Away matches available: {a_n} (xG ok last {window_n}: {a_xg}, Pressure ok: {a_press})")
        shown += 1

        evidence_rows.append(
            {
                "fixture_datetime": str(fdt) if not pd.isna(fdt) else "",
                "home": home,
                "away": away,
                "league": str(row.get(league_col)) if league_col else "",
                "home_matches_total": h_n,
                "away_matches_total": a_n,
                "home_xg_ok_last_window": h_xg,
                "away_xg_ok_last_window": a_xg,
                "home_pressure_ok_last_window": h_press,
                "away_pressure_ok_last_window": a_press,
                "window_n": window_n,
            }
        )

    _write_line(lines, "")
    _write_line(lines, f"- Fixtures with low xG evidence (min < {xg_min} in last {window_n}): {low_xg_fixtures}")
    _write_line(lines, f"- Fixtures with low Pressure evidence (min < {press_min} in last {window_n}): {low_press_fixtures}")
    _write_line(lines, "")
    league_summary = _league_summary(preds, dt, "predictions")
    return out, pd.DataFrame(evidence_rows) if evidence_rows else None, league_summary


def _load_quality_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Detailed audit log for non-technical review")
    ap.add_argument("--history", default=str(ROOT / "data" / "enhanced" / "cgm_match_history_with_elo_stats_xg.csv"))
    ap.add_argument("--upcoming", default=str(ROOT / "data" / "api_football" / "upcoming_fixtures.csv"))
    ap.add_argument("--predictions", default=str(ROOT / "reports" / "cgm_upcoming_predictions.csv"))
    ap.add_argument("--quality", default=str(ROOT / "data" / "api_football" / "fixture_quality_report.json"))
    ap.add_argument("--out", default=None, help="Output log path (txt).")
    ap.add_argument("--json", default=None, help="Optional JSON summary path (defaults next to txt).")
    ap.add_argument("--sample", type=int, default=20, help="Number of fixtures to sample in the log.")
    ap.add_argument("--window", type=int, default=10, help="Evidence window (matches).")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (ROOT / "reports" / "audits" / f"audit_detailed_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt")
    json_path = Path(args.json) if args.json else out_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    summary: dict[str, Any] = {
        "generated_at_utc": _now_stamp(),
        "paths": {
            "history": str(Path(args.history)),
            "upcoming": str(Path(args.upcoming)),
            "predictions": str(Path(args.predictions)),
            "quality": str(Path(args.quality)),
        },
        "issues": [],
    }
    team_counts_df: pd.DataFrame | None = None
    evidence_df: pd.DataFrame | None = None
    league_summaries: list[pd.DataFrame] = []

    _write_line(lines, "DETAILED AUDIT LOG")
    _write_line(lines, f"Generated: {_now_stamp()}")
    _write_line(lines, "")

    _describe_calcs(lines)

    # Quality report (API sync)
    q = _load_quality_report(Path(args.quality))
    _write_line(lines, "API sync quality report")
    if q is None:
        _write_line(lines, "- Not found or unreadable.")
    else:
        _write_line(lines, f"- Quality gate passed: {q.get('quality_gate_passed')}")
        _write_line(lines, f"- Stats coverage: {q.get('quality_metrics', {}).get('stats_coverage')}")
        _write_line(lines, f"- Odds coverage: {q.get('quality_metrics', {}).get('odds_coverage')}")
        _write_line(lines, f"- Errors: {len(q.get('errors') or [])}")
        if q.get("quality_gate_reasons"):
            _write_line(lines, "- Quality gate reasons:")
            for r in q.get("quality_gate_reasons"):
                _write_line(lines, f"  - {r}")
    _write_line(lines, "")

    # History
    history_path = Path(args.history)
    hist = _safe_read_csv(history_path)
    if hist is None:
        _write_line(lines, f"History data: MISSING ({history_path})")
        summary["issues"].append("History file missing")
    else:
        summary.update({"history_rows": int(len(hist))})
        res, team_counts_df, league_summary = _audit_history(hist, lines)
        if not res["ok"]:
            summary["issues"].extend(res["issues"])
        if league_summary is not None and not league_summary.empty:
            league_summaries.append(league_summary)
        # Raw metrics
        hist_dt = _parse_fixture_datetime(hist)
        summary["history_metrics"] = {
            "rows": int(len(hist)),
            "date_min": str(pd.to_datetime(hist_dt, errors="coerce").min()),
            "date_max": str(pd.to_datetime(hist_dt, errors="coerce").max()),
            "pressure_usable_rate": float(_pressure_usable_mask(hist).mean()) if len(hist) else 0.0,
            "xg_usable_rate": float(_xg_usable_mask(hist).mean()) if len(hist) else 0.0,
        }

    # Upcoming
    upcoming_path = Path(args.upcoming)
    upcoming = _safe_read_csv(upcoming_path)
    if upcoming is None:
        _write_line(lines, f"Upcoming data: MISSING ({upcoming_path})")
        summary["issues"].append("Upcoming file missing")
    else:
        summary.update({"upcoming_rows": int(len(upcoming))})
        res, league_summary = _audit_upcoming(upcoming, lines)
        if not res["ok"]:
            summary["issues"].extend(res["issues"])
        if league_summary is not None and not league_summary.empty:
            league_summaries.append(league_summary)
        up_dt = _parse_fixture_datetime(upcoming)
        summary["upcoming_metrics"] = {
            "rows": int(len(upcoming)),
            "date_min": str(pd.to_datetime(up_dt, errors="coerce").min()),
            "date_max": str(pd.to_datetime(up_dt, errors="coerce").max()),
        }

    # Predictions
    pred_path = Path(args.predictions)
    preds = _safe_read_csv(pred_path)
    if preds is None:
        _write_line(lines, f"Predictions data: MISSING ({pred_path})")
        summary["issues"].append("Predictions file missing")
    else:
        summary.update({"prediction_rows": int(len(preds))})
        if hist is not None:
            res, evidence_df, league_summary = _audit_predictions(preds, hist, lines, sample_limit=int(args.sample), window_n=int(args.window))
            if not res["ok"]:
                summary["issues"].extend(res["issues"])
            if league_summary is not None and not league_summary.empty:
                league_summaries.append(league_summary)
        pred_dt = _parse_fixture_datetime(preds)
        summary["prediction_metrics"] = {
            "rows": int(len(preds)),
            "date_min": str(pd.to_datetime(pred_dt, errors="coerce").min()),
            "date_max": str(pd.to_datetime(pred_dt, errors="coerce").max()),
        }

    _write_line(lines, "Summary")
    if summary["issues"]:
        _write_line(lines, "- Status: ISSUES FOUND")
        _write_line(lines, "- Issues:")
        for item in summary["issues"]:
            _write_line(lines, f"  - {item}")
    else:
        _write_line(lines, "- Status: OK")

    extra_outputs: list[str] = []
    if team_counts_df is not None and not team_counts_df.empty:
        team_counts_path = out_path.with_name("audit_team_counts.csv")
        team_counts_df.to_csv(team_counts_path, index=False)
        summary["team_counts_csv"] = str(team_counts_path)
        extra_outputs.append(str(team_counts_path))

    if evidence_df is not None and not evidence_df.empty:
        evidence_path = out_path.with_name("audit_fixture_evidence.csv")
        evidence_df.to_csv(evidence_path, index=False)
        summary["fixture_evidence_csv"] = str(evidence_path)
        extra_outputs.append(str(evidence_path))

    if league_summaries:
        league_df = pd.concat(league_summaries, ignore_index=True)
        league_path = out_path.with_name("audit_league_summary.csv")
        league_df.to_csv(league_path, index=False)
        summary["league_summary_csv"] = str(league_path)
        extra_outputs.append(str(league_path))

    if extra_outputs:
        _write_line(lines, "")
        _write_line(lines, "Extra outputs")
        for p in extra_outputs:
            _write_line(lines, f"- {p}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[ok] wrote detailed audit -> {out_path}")
    if json_path:
        print(f"[ok] wrote audit summary -> {json_path}")


if __name__ == "__main__":
    main()
