#!/usr/bin/env python3
"""
Audit GG/NG (BTTS) availability and gating.

Confirms:
  - BTTS probabilities exist in predictions.
  - BTTS odds are present (or missing).
  - Pick engine reasons for BTTS rejection.
  - Upcoming feed coverage for future fixtures (gg/ng odds source).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Ensure repo root is on path for local imports.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

try:
    from cgm.predict_upcoming import _parse_upcoming_datetime  # type: ignore
except Exception:
    _parse_upcoming_datetime = None


def _count_non_null(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").notna().sum())


def _parse_fixture_dt(df: pd.DataFrame) -> pd.Series:
    if "datameci" not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    if _parse_upcoming_datetime is not None:
        return df.apply(lambda r: _parse_upcoming_datetime(r.get("datameci"), r.get("orameci")), axis=1)
    # Fallback parser (date-only + optional HHMM time).
    dates = pd.to_datetime(df["datameci"].astype(str), errors="coerce", dayfirst=False)
    times = pd.to_numeric(df.get("orameci"), errors="coerce").fillna(0)
    hours = (times // 100).astype(int).clip(lower=0, upper=23)
    mins = (times % 100).astype(int).clip(lower=0, upper=59)
    return dates.dt.normalize() + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(mins, unit="m")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit GG/NG (BTTS) availability and gating")
    ap.add_argument("--predictions", default="reports/cgm_upcoming_predictions.csv", help="Predictions CSV path")
    ap.add_argument("--picks-debug", default="reports/picks_debug.csv", help="Pick debug CSV path")
    ap.add_argument("--upcoming", default="CGM data/multiple leagues and seasons/upcoming.csv", help="Upcoming odds feed")
    ap.add_argument("--allrating", default="CGM data/multiple leagues and seasons/allratingv.csv", help="Fixtures feed")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"[error] predictions not found: {pred_path}")
        return 1
    preds = pd.read_csv(pred_path, low_memory=False)

    print(f"[predictions] rows={len(preds)}")
    for col in ("p_btts_yes", "p_btts_no"):
        if col in preds.columns:
            print(f"[predictions] {col} non-null={_count_non_null(preds, col)}")
        else:
            print(f"[predictions] {col} missing")

    for col in ("odds_btts_yes", "odds_btts_no", "gg", "ng"):
        if col in preds.columns:
            print(f"[predictions] {col} non-null={_count_non_null(preds, col)}")

    debug_path = Path(args.picks_debug)
    if debug_path.exists():
        dbg = pd.read_csv(debug_path, low_memory=False)
        btts = dbg[dbg["market"].isin(["BTTS_YES", "BTTS_NO"])].copy()
        print(f"[picks_debug] BTTS rows={len(btts)} eligible={int(btts['eligible'].sum())} selected={int(btts['selected'].sum())}")
        if "reason_codes" in btts.columns:
            top = btts["reason_codes"].value_counts().head(8)
            print("[picks_debug] top reasons:")
            for reason, count in top.items():
                print(f"  {reason}: {count}")
    else:
        print(f"[picks_debug] not found: {debug_path}")

    # Feed coverage check (GG/NG odds live in upcoming.csv)
    upcoming_path = Path(args.upcoming)
    allrating_path = Path(args.allrating)
    if upcoming_path.exists():
        up = pd.read_csv(upcoming_path, low_memory=False)
        up["_fixture_dt"] = _parse_fixture_dt(up)
        max_up = up["_fixture_dt"].max()
        print(f"[feed] upcoming rows={len(up)} max_fixture_dt={max_up}")
    else:
        up = None
        print(f"[feed] upcoming not found: {upcoming_path}")

    if allrating_path.exists():
        ar = pd.read_csv(allrating_path, low_memory=False)
        ar["_fixture_dt"] = _parse_fixture_dt(ar)
        max_ar = ar["_fixture_dt"].max()
        print(f"[feed] allrating rows={len(ar)} max_fixture_dt={max_ar}")
    else:
        ar = None
        print(f"[feed] allrating not found: {allrating_path}")

    # Compare future coverage using run_asof_datetime if present
    run_asof = None
    if "run_asof_datetime" in preds.columns:
        try:
            run_asof = pd.to_datetime(preds["run_asof_datetime"].dropna().iloc[0], errors="coerce")
        except Exception:
            run_asof = None

    if run_asof is not None:
        print(f"[feed] run_asof_datetime={run_asof}")
        if up is not None and "_fixture_dt" in up.columns:
            future_up = up[up["_fixture_dt"] > run_asof]
            gg_ok = _count_non_null(future_up, "gg")
            ng_ok = _count_non_null(future_up, "ng")
            print(f"[feed] upcoming future fixtures={len(future_up)} gg_non_null={gg_ok} ng_non_null={ng_ok}")
        if ar is not None and "_fixture_dt" in ar.columns:
            future_ar = ar[ar["_fixture_dt"] > run_asof]
            print(f"[feed] allrating future fixtures={len(future_ar)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
