#!/usr/bin/env python3
"""
Prediction Data Health Audit.

Goal:
  Fail when upcoming predictions are built from weak evidence coverage.
  This is an audit-only gate; it does not change prediction logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _ratio(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.0
    return float(mask.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit live prediction data/evidence health")
    ap.add_argument("--predictions", default="reports/cgm_upcoming_predictions.csv", help="Predictions CSV")
    ap.add_argument("--min-press-coverage", type=float, default=0.95, help="Min rows with both press_stats_n_H/A > 0")
    ap.add_argument("--min-xg-coverage", type=float, default=0.95, help="Min rows with both xg_stats_n_H/A > 0")
    ap.add_argument("--min-odds-coverage", type=float, default=0.95, help="Min rows with valid OU+BTTS odds when model_variant=full")
    ap.add_argument("--require-nonempty", action="store_true", help="Fail if predictions file has zero rows")
    args = ap.parse_args()

    print("=" * 80)
    print("PREDICTION DATA HEALTH AUDIT")
    print("=" * 80)

    p = Path(args.predictions)
    if not p.exists():
        print(f"[FAIL] Missing predictions file: {p}")
        return 1

    df = pd.read_csv(p)
    print(f"rows={len(df)} file={p}")

    if df.empty:
        if args.require_nonempty:
            print("[FAIL] Predictions file is empty")
            return 1
        print("[WARN] Predictions file is empty; skipping coverage checks")
        return 0

    issues: list[str] = []

    # Core numeric sanity (should always hold if pipeline is healthy).
    for c in ["mu_home", "mu_away", "mu_total", "p_over25", "p_under25", "p_btts_yes", "p_btts_no"]:
        if c not in df.columns:
            issues.append(f"missing column: {c}")
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().any():
            issues.append(f"{c} has NaN values")

    if all(c in df.columns for c in ["p_over25", "p_under25"]):
        s = pd.to_numeric(df["p_over25"], errors="coerce") + pd.to_numeric(df["p_under25"], errors="coerce")
        if (s - 1.0).abs().max() > 0.02:
            issues.append("p_over25 + p_under25 deviates from 1.0 by > 0.02")

    # Evidence coverage.
    press_mask = None
    if {"press_stats_n_H", "press_stats_n_A"}.issubset(df.columns):
        press_h = pd.to_numeric(df["press_stats_n_H"], errors="coerce")
        press_a = pd.to_numeric(df["press_stats_n_A"], errors="coerce")
        press_mask = (press_h > 0) & (press_a > 0)
        press_cov = _ratio(press_mask)
        print(f"press_coverage={press_cov:.3f} min={args.min_press_coverage:.3f}")
        if press_cov < float(args.min_press_coverage):
            issues.append(f"press coverage too low: {press_cov:.3f} < {args.min_press_coverage:.3f}")
    else:
        issues.append("missing press evidence columns")

    if {"xg_stats_n_H", "xg_stats_n_A"}.issubset(df.columns):
        xg_h = pd.to_numeric(df["xg_stats_n_H"], errors="coerce")
        xg_a = pd.to_numeric(df["xg_stats_n_A"], errors="coerce")
        xg_mask = (xg_h > 0) & (xg_a > 0)
        xg_cov = _ratio(xg_mask)
        print(f"xg_coverage={xg_cov:.3f} min={args.min_xg_coverage:.3f}")
        if xg_cov < float(args.min_xg_coverage):
            issues.append(f"xG coverage too low: {xg_cov:.3f} < {args.min_xg_coverage:.3f}")
    else:
        issues.append("missing xG evidence columns")

    variants = sorted(set(df.get("model_variant", pd.Series(["full"])).astype(str).str.strip().str.lower().tolist()))
    is_full = any(v == "full" for v in variants)
    print(f"model_variants={variants}")

    if is_full:
        required_odds = ["odds_over25", "odds_under25", "odds_btts_yes", "odds_btts_no"]
        if all(c in df.columns for c in required_odds):
            odd_masks = []
            for c in required_odds:
                s = pd.to_numeric(df[c], errors="coerce")
                odd_masks.append(s.notna() & (s > 0))
            odds_cov = _ratio(odd_masks[0] & odd_masks[1] & odd_masks[2] & odd_masks[3])
            print(f"odds_coverage={odds_cov:.3f} min={args.min_odds_coverage:.3f}")
            if odds_cov < float(args.min_odds_coverage):
                issues.append(f"odds coverage too low: {odds_cov:.3f} < {args.min_odds_coverage:.3f}")
        else:
            issues.append("missing OU/BTTS odds columns for full variant")

    if issues:
        print("[FAIL] Prediction data health failed")
        for msg in issues:
            print(f"  - {msg}")
        return 1

    print("[PASS] Prediction data health passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
