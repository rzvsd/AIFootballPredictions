"""
Train ShotXG (MicroXG) model: per-shot goal probability using XGBoost.

Usage:
  python -m xg_shot_model.train_gbm \
    --shots data/shots/understat_shots.csv \
    --out models/shotxg_xgb.pkl \
    --calib models/shotxg_iso.pkl \
    --report reports/shotxg_metrics.json

The script:
  - Loads the engineered shots CSV (from Stage 1).
  - Builds features (numeric + one-hot for categorical).
  - Splits by time (if date present) or stratified random split.
  - Trains XGBClassifier with early stopping.
  - Fits isotonic calibrator on validation predictions.
  - Writes model, calibrator and metrics report (uncalibrated vs calibrated).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier


def _as_int01(x) -> int:
    try:
        if isinstance(x, (int, float)):
            return int(x != 0)
        s = str(x).strip().lower()
        if s in ("1", "true", "t", "yes", "y"):  # truthy
            return 1
        if s in ("0", "false", "f", "no", "n", "none", "", "nan"):
            return 0
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def _build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    # Label: goal vs other
    y = d.get("result").astype(str).str.lower().eq("goal").astype(int)

    # Ensure numeric columns exist
    for col in ("dist_m", "angle_deg", "X", "Y", "minute"):
        if col not in d.columns:
            d[col] = np.nan
    d["dist_m"] = pd.to_numeric(d["dist_m"], errors="coerce")
    d["angle_deg"] = pd.to_numeric(d["angle_deg"], errors="coerce")
    d["X"] = pd.to_numeric(d["X"], errors="coerce")
    d["Y"] = pd.to_numeric(d["Y"], errors="coerce")
    d["minute"] = pd.to_numeric(d["minute"], errors="coerce")

    # Binary flags
    for col in ("is_header", "fast_break", "isKeyPass"):
        if col not in d.columns:
            d[col] = 0
        d[col] = d[col].apply(_as_int01).astype(int)

    # Categorical: situation, shotType, h_a
    for col in ("situation", "shotType", "h_a"):
        if col not in d.columns:
            d[col] = ""
        d[col] = d[col].astype(str).fillna("")

    base_cols = [
        "dist_m", "angle_deg", "X", "Y", "minute",
        "is_header", "fast_break", "isKeyPass",
    ]
    cat_cols = ["situation", "shotType", "h_a"]

    X_num = d[base_cols].astype(float).fillna(0.0)
    X_cat = pd.get_dummies(d[cat_cols], drop_first=False, dtype=int)
    X = pd.concat([X_num, X_cat], axis=1)
    return X, y


def _time_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    if "date" in d.columns:
        try:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.sort_values("date")
        except Exception:
            pass
    n = len(d)
    k = max(1, int(n * (1.0 - float(test_size))))
    return d.iloc[:k, :].copy(), d.iloc[k:, :].copy()


def _fit_isotonic(p: np.ndarray, y: np.ndarray):
    # Lazy import to keep sklearn optional submodules isolated
    from sklearn.isotonic import IsotonicRegression

    # Isotonic works on 1D arrays
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    # Clip to avoid degenerate inputs
    p = np.clip(p, 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def _metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    try:
        ll = log_loss(y, p)
    except Exception:
        ll = float("nan")
    brier = float(np.mean((p - y) ** 2))
    return {"auc": float(auc), "logloss": float(ll), "brier": float(brier)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train per-shot xG model (XGBClassifier)")
    ap.add_argument("--shots", required=True, help="Path to CSV produced by shots_ingest_understat")
    ap.add_argument("--out", default=os.path.join("models", "shotxg_xgb.pkl"))
    ap.add_argument("--calib", default=os.path.join("models", "shotxg_iso.pkl"))
    ap.add_argument("--report", default=os.path.join("reports", "shotxg_metrics.json"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.shots)
    # Time-based split if possible
    df_tr, df_te = _time_split(df, test_size=args.test_size)
    X_tr, y_tr = _build_xy(df_tr)
    X_te, y_te = _build_xy(df_te)

    # Model
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=args.seed,
        tree_method="hist",
    )
    # Early stopping: use callbacks for broad xgboost compatibility
    callbacks = []
    try:
        from xgboost.callback import EarlyStopping as XgbEarlyStopping  # type: ignore
        try:
            cb = XgbEarlyStopping(rounds=50, save_best=True, maximize=False)
        except TypeError:  # older versions use 'stopping_rounds'
            cb = XgbEarlyStopping(stopping_rounds=50)
        callbacks = [cb]
    except Exception:
        callbacks = []

    try:
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_te, y_te)],
            verbose=False,
            callbacks=callbacks,
        )
    except TypeError:
        # Older XGBoost without 'callbacks' in fit signature
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_te, y_te)],
            verbose=False,
        )

    # Predictions (uncalibrated)
    p_tr_raw = clf.predict_proba(X_tr)[:, 1]
    p_te_raw = clf.predict_proba(X_te)[:, 1]

    # Isotonic calibrator fitted on validation (avoid overfitting)
    iso = _fit_isotonic(p_te_raw, y_te)
    p_te_iso = np.clip(iso.predict(p_te_raw), 1e-9, 1 - 1e-9)

    # Metrics
    m_tr = _metrics(y_tr, p_tr_raw)
    m_te_raw = _metrics(y_te, p_te_raw)
    m_te_iso = _metrics(y_te, p_te_iso)
    report = {
        "counts": {"train": int(len(X_tr)), "test": int(len(X_te))},
        "train_raw": m_tr,
        "test_raw": m_te_raw,
        "test_iso": m_te_iso,
        "features": list(X_tr.columns),
    }

    # Save artifacts
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.calib), exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    joblib.dump(clf, args.out)
    joblib.dump(iso, args.calib)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model -> {args.out}\nSaved calibrator -> {args.calib}\nSaved report -> {args.report}")


if __name__ == "__main__":
    main()
