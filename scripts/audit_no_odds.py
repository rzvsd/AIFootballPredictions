"""
Audit: no-odds model invariance to odds columns.

The no-odds variant should produce identical mu/probabilities even if the upcoming feed
odds are changed, because odds/p_* market features are excluded from the model contract.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _load_feature_names(model) -> list[str]:
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None and len(cols) > 0:
        return list(cols)
    try:
        return list(model.get_booster().feature_names)
    except Exception:
        return []


def _assert_no_market_features(feature_names: list[str]) -> None:
    banned_exact = {"p_home", "p_draw", "p_away", "p_over", "p_under"}
    banned_prefixes = ("odds_", "fair_")
    leaked = sorted([c for c in feature_names if (c in banned_exact) or c.startswith(banned_prefixes)])
    if leaked:
        raise AssertionError(f"no_odds model contract still contains market features: {leaked}")


def _make_modified_odds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Deterministic, plausible placeholders for missing/zero odds.
    defaults = {"cotaa": 2.10, "cotae": 3.50, "cotad": 3.80, "cotao": 1.95, "cotau": 1.90}
    for col, default in defaults.items():
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        # If odds are missing or nonsensical, overwrite with a deterministic value.
        bad = vals.isna() | (vals <= 1.01)
        vals2 = vals.copy()
        vals2[bad] = float(default)
        # If odds are present, perturb them so they're clearly different.
        vals2[~bad] = vals2[~bad] * 1.23
        out[col] = vals2
    return out


def _run_predict(
    *,
    history: Path,
    models_dir: Path,
    data_dir: Path,
    out_csv: Path,
    as_of_date: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "cgm.predict_upcoming",
        "--history",
        str(history),
        "--models-dir",
        str(models_dir),
        "--model-variant",
        "no_odds",
        "--data-dir",
        str(data_dir),
        "--as-of-date",
        str(as_of_date),
        "--out",
        str(out_csv),
        "--log-level",
        "WARNING",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"predict_upcoming failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit that the no-odds model is invariant to odds columns in the upcoming feed")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv", help="History CSV used for inference")
    ap.add_argument("--models-dir", default="models", help="Directory with no-odds models")
    ap.add_argument("--upcoming", default="CGM data/multiple leagues and seasons/allratingv.csv", help="Raw upcoming feed CSV (any CGM export)")
    ap.add_argument("--as-of-date", default="2025-12-19", help="As-of date (YYYY-MM-DD) used by scope filter")
    args = ap.parse_args()

    history = Path(args.history)
    models_dir = Path(args.models_dir)
    upcoming = Path(args.upcoming)
    if not history.exists():
        raise FileNotFoundError(history)
    if not upcoming.exists():
        raise FileNotFoundError(upcoming)

    model_h = joblib.load(models_dir / "frankenstein_mu_home_no_odds.pkl")
    model_a = joblib.load(models_dir / "frankenstein_mu_away_no_odds.pkl")
    _assert_no_market_features(_load_feature_names(model_h))
    _assert_no_market_features(_load_feature_names(model_a))

    df = pd.read_csv(upcoming)
    df_mod = _make_modified_odds(df)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        d1 = td_path / "d1"
        d2 = td_path / "d2"
        up1 = d1 / "multiple leagues and seasons"
        up2 = d2 / "multiple leagues and seasons"
        up1.mkdir(parents=True, exist_ok=True)
        up2.mkdir(parents=True, exist_ok=True)
        (up1 / "allratingv.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        (up2 / "allratingv.csv").write_text(df_mod.to_csv(index=False), encoding="utf-8")
        out1 = td_path / "pred1.csv"
        out2 = td_path / "pred2.csv"

        _run_predict(history=history, models_dir=models_dir, data_dir=d1, out_csv=out1, as_of_date=str(args.as_of_date))
        _run_predict(history=history, models_dir=models_dir, data_dir=d2, out_csv=out2, as_of_date=str(args.as_of_date))

        p1 = pd.read_csv(out1)
        p2 = pd.read_csv(out2)

        key_cols = ["fixture_datetime", "league", "home", "away"]
        cmp_cols = [
            "mu_home",
            "mu_away",
            "p_over_2_5",
            "p_under_2_5",
            "p_btts_yes",
            "p_btts_no",
        ]
        for c in key_cols + cmp_cols:
            if c not in p1.columns or c not in p2.columns:
                raise AssertionError(f"Missing required output column for audit: {c}")

        m1 = p1[key_cols + cmp_cols].copy()
        m2 = p2[key_cols + cmp_cols].copy()
        merged = m1.merge(m2, on=key_cols, how="outer", suffixes=("_a", "_b"), indicator=True)
        if (merged["_merge"] != "both").any():
            bad = merged[merged["_merge"] != "both"][key_cols + ["_merge"]]
            raise AssertionError(f"Row mismatch between runs:\n{bad.to_string(index=False)}")

        max_abs = 0.0
        for c in cmp_cols:
            a = pd.to_numeric(merged[f"{c}_a"], errors="coerce").to_numpy()
            b = pd.to_numeric(merged[f"{c}_b"], errors="coerce").to_numpy()
            diff = np.nanmax(np.abs(a - b)) if a.size else 0.0
            max_abs = max(max_abs, float(diff))
            if not np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
                raise AssertionError(f"no_odds invariance failed for column '{c}' (max_abs_diff={diff})")

    print("[ok] no_odds invariance audit passed (mu/probs unchanged when odds change)")
    print(f"[ok] max_abs_diff={max_abs}")


if __name__ == "__main__":
    main()

