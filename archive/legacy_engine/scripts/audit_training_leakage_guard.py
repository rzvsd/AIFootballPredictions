#!/usr/bin/env python3
"""
Training Leakage Guard Audit.

Goal:
  Catch leakage patterns without changing training strategy or feature engineering.
  This audit only inspects the feature matrix that train_frankenstein_mu.load_data()
  would consume for the selected variant.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cgm.train_frankenstein_mu import load_data  # noqa: E402


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit training features for leakage risk")
    ap.add_argument("--data", default="data/enhanced/frankenstein_training.csv", help="Training CSV path")
    ap.add_argument("--variant", choices=["full", "no_odds"], default="full", help="Feature variant")
    ap.add_argument("--max-abs-corr", type=float, default=0.995, help="Absolute correlation threshold for suspicious target-like features")
    ap.add_argument("--max-equality-rate", type=float, default=0.99, help="Max allowed exact-match rate vs targets")
    args = ap.parse_args()

    print("=" * 80)
    print("TRAINING LEAKAGE GUARD AUDIT")
    print("=" * 80)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[FAIL] Missing training data: {data_path}")
        return 1

    # Use the exact loader used by training to avoid strategy drift.
    X, y_home, y_away = load_data(data_path, variant=str(args.variant))

    print(f"rows={len(X)} feature_cols={len(X.columns)} variant={args.variant}")
    if X.empty:
        print("[FAIL] Feature matrix is empty")
        return 1

    issues: list[str] = []

    # 1) Name-based guard (stronger token checks than plain substring-only logic).
    banned_name_patterns = [
        re.compile(r"(^|_)ft(_|$)"),
        re.compile(r"(^|_)ht(_|$)"),
        re.compile(r"(^|_)result($|_)"),
        re.compile(r"(^|_)validated($|_)"),
        re.compile(r"(^|_)final(_|$)"),
        re.compile(r"(^|_)outcome($|_)"),
    ]
    leaked_by_name = []
    for c in X.columns:
        cl = str(c).lower()
        if any(p.search(cl) for p in banned_name_patterns):
            leaked_by_name.append(c)
    if leaked_by_name:
        issues.append(f"Suspicious feature names detected: {sorted(leaked_by_name)[:10]}")

    # 2) Value-equality guard against direct target copies.
    yh = pd.Series(y_home).astype(float).to_numpy()
    ya = pd.Series(y_away).astype(float).to_numpy()

    target_clones = []
    suspicious_corr = []
    for col in X.columns:
        x = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        eq_h = float(np.mean(np.isclose(x, yh, atol=1e-12)))
        eq_a = float(np.mean(np.isclose(x, ya, atol=1e-12)))
        if max(eq_h, eq_a) >= float(args.max_equality_rate):
            target_clones.append((col, eq_h, eq_a))
            continue

        corr_h = _safe_corr(x, yh)
        corr_a = _safe_corr(x, ya)
        max_abs = max(abs(corr_h) if np.isfinite(corr_h) else 0.0, abs(corr_a) if np.isfinite(corr_a) else 0.0)
        # High correlation alone is not proof, but this threshold catches near-copies.
        if max_abs >= float(args.max_abs_corr):
            suspicious_corr.append((col, corr_h, corr_a))

    if target_clones:
        preview = [f"{c}(eq_home={eh:.3f},eq_away={ea:.3f})" for c, eh, ea in target_clones[:10]]
        issues.append("Target-clone features detected: " + ", ".join(preview))

    if suspicious_corr:
        preview = [f"{c}(corr_home={ch:.3f},corr_away={ca:.3f})" for c, ch, ca in suspicious_corr[:10]]
        issues.append("Near-target correlated features detected: " + ", ".join(preview))

    # 3) Basic shape guard.
    if np.isnan(X.to_numpy(dtype=float)).any():
        issues.append("NaN values remained in X after load_data normalization")

    if issues:
        print("[FAIL] Leakage guard failed")
        for msg in issues:
            print(f"  - {msg}")
        return 1

    print("[PASS] Leakage guard passed")
    print(f"  - no banned name patterns in {len(X.columns)} features")
    print("  - no direct target clones / near-target copies detected")
    print("  - feature matrix numeric + NaN-free after training normalization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
