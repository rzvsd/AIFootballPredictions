"""
Train μ_home and μ_away models from frankenstein_training.csv.

Outputs:
  models/frankenstein_mu_home.pkl
  models/frankenstein_mu_away.pkl
Prints basic RMSE and Poisson log-likelihood on a holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib


def poisson_log_likelihood(mu: np.ndarray, y: np.ndarray) -> float:
    mu = np.clip(mu, 1e-6, None)
    y = np.clip(y, 0, None)
    return float(np.mean(y * np.log(mu) - mu))


def _model_paths(out_dir: Path, *, variant: str) -> tuple[Path, Path]:
    suffix = "" if variant == "full" else f"_{variant}"
    return (out_dir / f"frankenstein_mu_home{suffix}.pkl", out_dir / f"frankenstein_mu_away{suffix}.pkl")


def load_data(path: str | Path, *, variant: str = "full") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    # Identify target and drop non-feature cols
    y_home = df["y_home"].to_numpy()
    y_away = df["y_away"].to_numpy()
    # Aggressively drop any post-match / outcome columns to avoid leakage
    drop_cols = set([
        "y_home", "y_away",
        "date", "time", "datetime",
        "home", "away", "code_home", "code_away",
        "result", "validated",
        "ft_home", "ft_away", "ht_home", "ht_away",
        "shots", "shots_on_target", "corners",
        "possession_home", "possession_away",
    ])
    # Also drop any columns whose name contains these substrings
    leak_tokens = ["ft_", "ht_", "shots", "corners", "possession"]
    for col in df.columns:
        # Keep Milestone 2 "Pressure Cooker" features (pre-match derived)
        if col.startswith("press_") or col.startswith("div_") or col.endswith("_z_H") or col.endswith("_z_A"):
            continue
        if any(tok in col for tok in leak_tokens):
            drop_cols.add(col)

    # Optional: drop market features (odds + market probabilities) so the model is purely internal.
    if str(variant).strip().lower() in {"no_odds", "no-odds", "internal"}:
        market_exact = {
            "p_home", "p_draw", "p_away", "p_over", "p_under",
            "fair_home", "fair_draw", "fair_away", "fair_over", "fair_under",
            "odds_home", "odds_draw", "odds_away", "odds_over", "odds_under",
        }
        for col in df.columns:
            if col in market_exact or col.startswith("odds_") or col.startswith("fair_"):
                drop_cols.add(col)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Poison-pill: training must never see raw/truth columns as features.
    banned_exact = {
        "ft_home", "ft_away", "ht_home", "ht_away", "result", "validated",
        "shots", "shots_on_target", "corners", "possession_home", "possession_away",
        "shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A",
    }
    leaked = sorted([c for c in X.columns if c in banned_exact])
    if leaked:
        raise AssertionError(f"Leakage risk: banned columns in X: {leaked}")
    # Coerce numerics
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)
    return X, y_home, y_away


def train_models(X: pd.DataFrame, y_home: np.ndarray, y_away: np.ndarray, *, seed: int = 42) -> Dict[str, object]:
    # Time-based split: use the last 20% rows as validation (assumes chronological order of X)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_h_tr, y_h_val = y_home[:split_idx], y_home[split_idx:]
    y_a_tr, y_a_val = y_away[:split_idx], y_away[split_idx:]
    params = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="count:poisson",  # keep mu > 0
        eval_metric="rmse",
        random_state=int(seed),
        n_jobs=1,
    )
    model_h = XGBRegressor(**params)
    model_a = XGBRegressor(**params)
    model_h.fit(X_train, y_h_tr)
    model_a.fit(X_train, y_a_tr)
    # Metrics
    mu_h_val = model_h.predict(X_val)
    mu_a_val = model_a.predict(X_val)
    rmse_h = float(np.sqrt(mean_squared_error(y_h_val, mu_h_val)))
    rmse_a = float(np.sqrt(mean_squared_error(y_a_val, mu_a_val)))
    pll_h = poisson_log_likelihood(mu_h_val, y_h_val)
    pll_a = poisson_log_likelihood(mu_a_val, y_a_val)
    metrics = {
        "rmse_home": rmse_h,
        "rmse_away": rmse_a,
        "poisson_ll_home": pll_h,
        "poisson_ll_away": pll_a,
    }
    return {"model_home": model_h, "model_away": model_a, "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train μ_home and μ_away models from Frankenstein features")
    ap.add_argument("--data", default="data/enhanced/frankenstein_training.csv", help="Path to feature CSV")
    ap.add_argument("--out-dir", default="models", help="Directory to save models")
    ap.add_argument("--variant", choices=["full", "no_odds"], default="full", help="Model variant (feature set)")
    ap.add_argument("--seed", type=int, default=42, help="Training seed (XGBoost random_state)")
    args = ap.parse_args()

    X, y_home, y_away = load_data(args.data, variant=str(args.variant))
    out = train_models(X, y_home, y_away, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    home_path, away_path = _model_paths(out_dir, variant=str(args.variant))
    joblib.dump(out["model_home"], home_path)
    joblib.dump(out["model_away"], away_path)
    print(f"[ok] saved models -> {home_path}, {away_path}")
    print(f"[variant] {args.variant}")
    print(f"[seed] {args.seed}")
    print("[metrics]", json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
