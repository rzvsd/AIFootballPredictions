#!/usr/bin/env python3
"""
Build isolated LATAM API dataset (Argentina + Brazil) from two season snapshots.

Why this exists:
  - South American schedules can straddle calendar years.
  - We keep LATAM isolated so core Europe strategy/files are untouched.
  - We rename Brazil "Serie A" -> "Serie A Brazil" in this isolated dataset
    to avoid collisions with Italy "Serie A" thresholds/backtests.

Inputs (default):
  - data/api_football_latam_s2025/history_fixtures.csv
  - data/api_football_latam_s2026/history_fixtures.csv
  - data/api_football_latam_s2026/upcoming_fixtures.csv

Outputs (default):
  - data/api_football_latam/history_fixtures.csv
  - data/api_football_latam/upcoming_fixtures.csv
  - data/api_football_latam/fixture_quality_report.json
  - data/api_football_latam/multiple leagues and seasons/{history_fixtures.csv, upcoming.csv, allratingv.csv}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _coverage(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "stats_coverage": 0.0, "odds_coverage": 0.0}
    rows = len(df)
    stats_cols = [
        "shots_home",
        "shots_away",
        "shots_on_target_home",
        "shots_on_target_away",
        "corners_home",
        "corners_away",
        "possession_home",
        "possession_away",
    ]
    odds_cols = [
        "odds_home",
        "odds_draw",
        "odds_away",
        "odds_over_2_5",
        "odds_under_2_5",
        "odds_btts_yes",
        "odds_btts_no",
    ]
    stats_cov = 1.0
    odds_cov = 1.0
    for col in stats_cols:
        if col in df.columns:
            stats_cov = min(stats_cov, _safe_ratio(pd.to_numeric(df[col], errors="coerce").notna().sum(), rows))
        else:
            stats_cov = 0.0
    for col in odds_cols:
        if col in df.columns:
            odds_cov = min(odds_cov, _safe_ratio(pd.to_numeric(df[col], errors="coerce").notna().sum(), rows))
        else:
            odds_cov = 0.0
    return {"rows": rows, "stats_coverage": round(stats_cov, 4), "odds_coverage": round(odds_cov, 4)}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build merged LATAM dataset from season snapshots")
    ap.add_argument("--season-2025-dir", default="data/api_football_latam_s2025")
    ap.add_argument("--season-2026-dir", default="data/api_football_latam_s2026")
    ap.add_argument("--out-dir", default="data/api_football_latam")
    ap.add_argument("--rename-brazil-serie-a", action="store_true", default=True)
    ap.add_argument("--no-rename-brazil-serie-a", dest="rename_brazil_serie_a", action="store_false")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    s2025 = Path(args.season_2025_dir)
    s2026 = Path(args.season_2026_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h25_path = s2025 / "history_fixtures.csv"
    h26_path = s2026 / "history_fixtures.csv"
    up_path = s2026 / "upcoming_fixtures.csv"

    if not h25_path.exists() or not h26_path.exists() or not up_path.exists():
        missing = [str(p) for p in (h25_path, h26_path, up_path) if not p.exists()]
        raise SystemExit(f"Missing required files: {missing}")

    h25 = _read_csv(h25_path)
    h26 = _read_csv(h26_path)
    upcoming = _read_csv(up_path)

    history = pd.concat([h25, h26], ignore_index=True)
    if "fixture_id" in history.columns:
        history = history.drop_duplicates(subset=["fixture_id"], keep="last").copy()
    else:
        history = history.drop_duplicates().copy()

    if bool(args.rename_brazil_serie_a):
        mask_hist = history.get("country", "").astype(str).eq("Brazil") & history.get("league", "").astype(str).eq("Serie A")
        mask_up = upcoming.get("country", "").astype(str).eq("Brazil") & upcoming.get("league", "").astype(str).eq("Serie A")
        history.loc[mask_hist, "league"] = "Serie A Brazil"
        upcoming.loc[mask_up, "league"] = "Serie A Brazil"

    history_out = out_dir / "history_fixtures.csv"
    upcoming_out = out_dir / "upcoming_fixtures.csv"
    history.to_csv(history_out, index=False)
    upcoming.to_csv(upcoming_out, index=False)

    multi = out_dir / "multiple leagues and seasons"
    multi.mkdir(parents=True, exist_ok=True)
    history.to_csv(multi / "history_fixtures.csv", index=False)
    upcoming.to_csv(multi / "upcoming.csv", index=False)

    allrating = multi / "allratingv.csv"
    with open(allrating, "wb") as out:
        hb = history_out.read_bytes().splitlines(keepends=True)
        ub = upcoming_out.read_bytes().splitlines(keepends=True)
        out.writelines(hb)
        if ub:
            start = 1 if hb and ub and hb[0].strip() == ub[0].strip() else 0
            out.writelines(ub[start:])

    hist_cov = _coverage(history)
    up_cov = _coverage(upcoming)
    quality = {
        "quality_gate_passed": True,
        "quality_gate_reasons": [],
        "min_odds_coverage": 0.0,
        "min_stats_coverage": 0.0,
        "coverage": {
            "history_stats_coverage": hist_cov["stats_coverage"],
            "upcoming_odds_coverage": up_cov["odds_coverage"],
        },
        "sync_summary": {
            "finished_rows": int(len(history)),
            "upcoming_rows": int(len(upcoming)),
            "source_seasons": [2025, 2026],
        },
        "dataset_scope": {
            "countries": sorted(set(history.get("country", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())),
            "leagues": sorted(set(history.get("league", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())),
        },
    }
    (out_dir / "fixture_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print(f"[ok] wrote merged history -> {history_out} (rows={len(history)})")
    print(f"[ok] wrote merged upcoming -> {upcoming_out} (rows={len(upcoming)})")
    print(f"[ok] wrote quality report -> {out_dir / 'fixture_quality_report.json'}")
    print(f"[ok] bridge files -> {multi}")


if __name__ == "__main__":
    main()

