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

import logging
_logger = logging.getLogger(__name__)

import config
from cgm.league_weighting_nlm import LeagueNLMCombiner, infer_group_columns


def poisson_log_likelihood(mu: np.ndarray, y: np.ndarray) -> float:
    """Compute Poisson log-likelihood, with logging for bad predictions."""
    # Warn if model predicted very low mu values (potential issue)
    bad_count = int(np.sum(mu < 0.01))
    if bad_count > 0:
        _logger.warning(f"poisson_log_likelihood: {bad_count} mu values < 0.01 (will be clipped)")
    mu = np.clip(mu, 1e-6, None)
    y = np.clip(y, 0, None)
    return float(np.mean(y * np.log(mu) - mu))


def _model_paths(out_dir: Path, *, variant: str) -> tuple[Path, Path]:
    suffix = "" if variant == "full" else f"_{variant}"
    return (out_dir / f"frankenstein_mu_home{suffix}.pkl", out_dir / f"frankenstein_mu_away{suffix}.pkl")


def _nlm_paths(out_dir: Path, *, variant: str) -> tuple[Path, Path]:
    suffix = "" if variant == "full" else f"_{variant}"
    return (out_dir / f"frankenstein_mu_home{suffix}_nlm.pkl", out_dir / f"frankenstein_mu_away{suffix}_nlm.pkl")


def load_data(path: str | Path, *, variant: str = "full") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    df = pd.read_csv(path)
    if "league" in df.columns:
        leagues = df["league"].astype(str).replace("", np.nan).fillna("GLOBAL")
    elif "league_id" in df.columns:
        leagues = pd.to_numeric(df["league_id"], errors="coerce").fillna(-1).astype(int).astype(str)
    else:
        leagues = pd.Series(["GLOBAL"] * len(df))
    # Identify target and drop non-feature cols
    y_home = df["y_home"].to_numpy()
    y_away = df["y_away"].to_numpy()
    # Aggressively drop any post-match / outcome columns to avoid leakage
    drop_cols = set([
        "y_home", "y_away",
        "date", "time", "datetime",
        "home", "away", "code_home", "code_away",
        "country",
        "fixture_id", "league_id", "round", "match_type", "competition_type",
        "result", "validated",
        "ft_home", "ft_away", "ht_home", "ht_away",
        "shots", "shots_on_target", "corners",
        "possession_home", "possession_away",
        "elo_k_base_used", "elo_k_matchtype_mult", "elo_k_newteam_mult", "elo_k_upset_mult",
        "elo_k_used", "elo_g_used", "elo_expected_home", "elo_actual_home", "elo_delta",
    ])
    # Also drop any columns whose name contains these substrings
    leak_tokens = [
        "ft_",
        "ht_",
        "shots",
        "corners",
        "possession",
        "attempts",
        "attacks",
        "goalkeeper_saves",
        "fouls",
        "offsides",
        "free_kicks",
        "throwins",
        "yellow_cards",
        "red_cards",
        "substitutions",
    ]
    for col in df.columns:
        # Keep Milestone 2 "Pressure Cooker" features (pre-match derived)
        if col.startswith("press_") or col.startswith("div_") or col.endswith("_z_H") or col.endswith("_z_A"):
            continue
        if any(tok in col for tok in leak_tokens):
            drop_cols.add(col)

    variant_norm = str(variant).strip().lower()

    # Optional: drop market features (odds + market probabilities) so the model is purely internal.
    if variant_norm in {"no_odds", "no-odds", "internal"}:
        market_exact = {
            "p_home", "p_draw", "p_away", "p_over", "p_under",
            "fair_home", "fair_draw", "fair_away", "fair_over", "fair_under",
            "odds_home", "odds_draw", "odds_away", "odds_over", "odds_under",
        }
        for col in df.columns:
            if col in market_exact or col.startswith("odds_") or col.startswith("fair_"):
                drop_cols.add(col)

    # Optional: drop league baseline anchors (for A/B tests against "with league-average base").
    if variant_norm in {"no_lgavg", "no-league-avg", "no_league_avg"}:
        no_lgavg_exact = {
            "Attack_H",
            "Attack_A",
            "Defense_H",
            "Defense_A",
            "Expected_Destruction_H",
            "Expected_Destruction_A",
        }
        for col in df.columns:
            if col in no_lgavg_exact or col.startswith("lg_avg_"):
                drop_cols.add(col)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Poison-pill: training must never see raw/truth columns as features.
    banned_exact = {
        "ft_home", "ft_away", "ht_home", "ht_away", "result", "validated",
        "fixture_id", "league_id", "round", "match_type", "competition_type",
        "shots", "shots_on_target", "corners", "possession_home", "possession_away",
        "elo_k_base_used", "elo_k_matchtype_mult", "elo_k_newteam_mult", "elo_k_upset_mult",
        "elo_k_used", "elo_g_used", "elo_expected_home", "elo_actual_home", "elo_delta",
        "shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A",
        "shots_off_H", "shots_off_A",
        "blocked_shots_H", "blocked_shots_A",
        "goal_attempts_H", "goal_attempts_A",
        "attacks_H", "attacks_A",
        "dangerous_attacks_H", "dangerous_attacks_A",
        "counter_attacks_H", "counter_attacks_A",
        "cross_attacks_H", "cross_attacks_A",
        "goalkeeper_saves_H", "goalkeeper_saves_A",
        "fouls_H", "fouls_A",
        "offsides_H", "offsides_A",
        "free_kicks_H", "free_kicks_A",
        "throwins_H", "throwins_A",
        "yellow_cards_H", "yellow_cards_A",
        "red_cards_H", "red_cards_A",
        "substitutions_H", "substitutions_A",
    }
    leaked = sorted([c for c in X.columns if c in banned_exact])
    if leaked:
        raise AssertionError(f"Leakage risk: banned columns in X: {leaked}")
    # Coerce numerics
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)
    return X, y_home, y_away, leagues


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
    ap.add_argument(
        "--variant",
        choices=["full", "no_odds", "no_lgavg"],
        default="no_odds",
        help="Model variant (feature set)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Training seed (XGBoost random_state)")
    ap.add_argument(
        "--nlm-enabled",
        action="store_true",
        default=bool(getattr(config, "NLM_STACKER_ENABLED", True)),
        help="Train and save league-specific NLM combiner on top of base mu models",
    )
    ap.add_argument(
        "--nlm-prior-strength",
        type=float,
        default=float(getattr(config, "NLM_PRIOR_STRENGTH", 120.0)),
        help="Shrinkage strength toward global combiner",
    )
    ap.add_argument(
        "--nlm-min-league-rows",
        type=int,
        default=int(getattr(config, "NLM_MIN_LEAGUE_ROWS", 80)),
        help="Minimum league rows to fit a league-specific residual model",
    )
    ap.add_argument(
        "--nlm-alpha-global",
        type=float,
        default=float(getattr(config, "NLM_ALPHA_GLOBAL", 1.0)),
        help="Ridge alpha for global combiner",
    )
    ap.add_argument(
        "--nlm-alpha-league",
        type=float,
        default=float(getattr(config, "NLM_ALPHA_LEAGUE", 2.0)),
        help="Ridge alpha for league residual models",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    X, y_home, y_away, leagues = load_data(args.data, variant=str(args.variant))
    out = train_models(X, y_home, y_away, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    home_path, away_path = _model_paths(out_dir, variant=str(args.variant))
    joblib.dump(out["model_home"], home_path)
    joblib.dump(out["model_away"], away_path)
    _logger.info(f"[ok] saved models -> {home_path}, {away_path}")
    _logger.info(f"[variant] {args.variant}")
    _logger.info(f"[seed] {args.seed}")
    _logger.info(f"[metrics] {json.dumps(out['metrics'], indent=2)}")

    if bool(args.nlm_enabled):
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_h_tr, y_h_val = y_home[:split_idx], y_home[split_idx:]
        y_a_tr, y_a_val = y_away[:split_idx], y_away[split_idx:]
        lg_train, lg_val = leagues.iloc[:split_idx], leagues.iloc[split_idx:]

        model_h = out["model_home"]
        model_a = out["model_away"]
        base_h_tr = np.asarray(model_h.predict(X_train), dtype=float)
        base_h_val = np.asarray(model_h.predict(X_val), dtype=float)
        base_a_tr = np.asarray(model_a.predict(X_train), dtype=float)
        base_a_val = np.asarray(model_a.predict(X_val), dtype=float)

        group_cols = infer_group_columns(list(X.columns))

        nlm_h_eval = LeagueNLMCombiner(
            group_cols=group_cols,
            prior_strength=float(args.nlm_prior_strength),
            min_league_rows=int(args.nlm_min_league_rows),
            alpha_global=float(args.nlm_alpha_global),
            alpha_league=float(args.nlm_alpha_league),
        ).fit(X=X_train, y=y_h_tr, base_mu=base_h_tr, leagues=lg_train)
        nlm_a_eval = LeagueNLMCombiner(
            group_cols=group_cols,
            prior_strength=float(args.nlm_prior_strength),
            min_league_rows=int(args.nlm_min_league_rows),
            alpha_global=float(args.nlm_alpha_global),
            alpha_league=float(args.nlm_alpha_league),
        ).fit(X=X_train, y=y_a_tr, base_mu=base_a_tr, leagues=lg_train)

        mu_h_val_nlm = nlm_h_eval.predict_batch(X=X_val, base_mu=base_h_val, leagues=lg_val)
        mu_a_val_nlm = nlm_a_eval.predict_batch(X=X_val, base_mu=base_a_val, leagues=lg_val)
        nlm_metrics = {
            "rmse_home_nlm": float(np.sqrt(mean_squared_error(y_h_val, mu_h_val_nlm))),
            "rmse_away_nlm": float(np.sqrt(mean_squared_error(y_a_val, mu_a_val_nlm))),
            "poisson_ll_home_nlm": poisson_log_likelihood(mu_h_val_nlm, y_h_val),
            "poisson_ll_away_nlm": poisson_log_likelihood(mu_a_val_nlm, y_a_val),
            "home_league_models": len(nlm_h_eval.league_params_),
            "away_league_models": len(nlm_a_eval.league_params_),
        }
        _logger.info(f"[nlm_eval] {json.dumps(nlm_metrics, indent=2)}")

        # Fit final NLM combiners on full data and persist.
        base_h_full = np.asarray(model_h.predict(X), dtype=float)
        base_a_full = np.asarray(model_a.predict(X), dtype=float)
        nlm_h_final = LeagueNLMCombiner(
            group_cols=group_cols,
            prior_strength=float(args.nlm_prior_strength),
            min_league_rows=int(args.nlm_min_league_rows),
            alpha_global=float(args.nlm_alpha_global),
            alpha_league=float(args.nlm_alpha_league),
        ).fit(X=X, y=y_home, base_mu=base_h_full, leagues=leagues)
        nlm_a_final = LeagueNLMCombiner(
            group_cols=group_cols,
            prior_strength=float(args.nlm_prior_strength),
            min_league_rows=int(args.nlm_min_league_rows),
            alpha_global=float(args.nlm_alpha_global),
            alpha_league=float(args.nlm_alpha_league),
        ).fit(X=X, y=y_away, base_mu=base_a_full, leagues=leagues)

        nlm_home_path, nlm_away_path = _nlm_paths(out_dir, variant=str(args.variant))
        joblib.dump(nlm_h_final, nlm_home_path)
        joblib.dump(nlm_a_final, nlm_away_path)
        _logger.info(f"[ok] saved NLM combiners -> {nlm_home_path}, {nlm_away_path}")
        _logger.info(f"[nlm_summary_home] {json.dumps(nlm_h_final.summary(), indent=2)}")
        _logger.info(f"[nlm_summary_away] {json.dumps(nlm_a_final.summary(), indent=2)}")


if __name__ == "__main__":
    main()
