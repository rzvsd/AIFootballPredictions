"""
Train NGBoost Poisson models for home/away goals per league and save under advanced_models/.

Requirements:
  pip install ngboost

Usage:
  python -m scripts.ngboost_trainer --league E0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import xgb_trainer
import config


def build_training_frame(league: str, start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    proc_path = os.path.join('data', 'processed', f'{league}_merged_preprocessed.csv')
    df = pd.read_csv(proc_path)
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception:
            pass
    # Ensure labels
    if not set(['FTHG','FTAG']).issubset(df.columns) or df['FTHG'].isna().all():
        raw_path = os.path.join('data','raw', f'{league}_merged.csv')
        if os.path.exists(raw_path):
            raw = pd.read_csv(raw_path, parse_dates=['Date'], dayfirst=True)
            left = df.copy()
            if 'Date' in left.columns:
                try:
                    left['Date'] = pd.to_datetime(left['Date'], errors='coerce')
                except Exception:
                    pass
            keys = [c for c in ['Date','HomeTeam','AwayTeam'] if c in left.columns and c in raw.columns]
            if keys:
                df = pd.merge(left, raw[['Date','HomeTeam','AwayTeam','FTHG','FTAG']], on=keys, how='left', suffixes=('', '_raw'))
    # Optional subset by dates (Step B)
    if start:
        try:
            df = df[df['Date'] >= pd.to_datetime(start)]
        except Exception:
            pass
    if end:
        try:
            df = df[df['Date'] <= pd.to_datetime(end)]
        except Exception:
            pass

    # Compute features
    sigma = float(getattr(config, 'ELO_SIM_SIGMA_PER_LEAGUE', {}).get(league, 50.0))
    d_feats = xgb_trainer._compute_ewma_elo_prematch(df, half_life_matches=5, elo_k=20.0, elo_home_adv=60.0,
                                                     elo_similarity_sigma=sigma)
    X = pd.DataFrame(index=d_feats.index)
    # Map to ULTIMATE_FEATURES
    X['ShotConv_H'] = d_feats['xg_home_EWMA']
    X['ShotConv_A'] = d_feats['xg_away_EWMA']
    X['ShotConvRec_H'] = d_feats['xga_home_EWMA']
    X['ShotConvRec_A'] = d_feats['xga_away_EWMA']
    X['PointsPerGame_H'] = d_feats['ppg_home_EWMA']
    X['PointsPerGame_A'] = d_feats['ppg_away_EWMA']
    X['CleanSheetStreak_H'] = 0.0
    X['CleanSheetStreak_A'] = 0.0
    X['xGDiff_H'] = X['ShotConv_H'] - X['ShotConvRec_H']
    X['xGDiff_A'] = X['ShotConv_A'] - X['ShotConvRec_A']
    X['CornersConv_H'] = 0.0
    X['CornersConv_A'] = 0.0
    X['CornersConvRec_H'] = 0.0
    X['CornersConvRec_A'] = 0.0
    X['NumMatches_H'] = 20.0
    X['NumMatches_A'] = 20.0
    X['Elo_H'] = d_feats['Elo_H']
    X['Elo_A'] = d_feats['Elo_A']
    X['EloDiff'] = d_feats['EloDiff']
    for c in ['GFvsMid_H','GAvsMid_H','GFvsHigh_H','GAvsHigh_H','GFvsMid_A','GAvsMid_A','GFvsHigh_A','GAvsHigh_A']:
        X[c] = d_feats.get(c, 0.0)
    for c in ['GFvsSim_H','GAvsSim_H','GFvsSim_A','GAvsSim_A']:
        X[c] = d_feats.get(c, 0.0) if c in d_feats.columns else 0.0
    # Reorder to ULTIMATE_FEATURES
    cols = list(config.ULTIMATE_FEATURES)
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[cols].fillna(0.0)
    yH = df['FTHG'] if 'FTHG' in df.columns else pd.Series([0]*len(df))
    yA = df['FTAG'] if 'FTAG' in df.columns else pd.Series([0]*len(df))
    y_home = yH.fillna(0).astype(int).values
    y_away = yA.fillna(0).astype(int).values
    return X, y_home, y_away


def train_ngb_poisson(X: pd.DataFrame, y: np.ndarray, *, use_requested: bool = True):
    """Train NGBoost with a robust configuration suitable for Windows/NumPy setups.

    Strategy:
      - Disable natural_gradient to avoid linear-solve issues on some platforms
      - Use moderate n_estimators, lower learning_rate, and minibatching
      - Fix random_state for reproducibility
    """
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Poisson
    except Exception:
        raise SystemExit("NGBoost not installed. Install with: pip install ngboost")
    # Pasul A: configurație cerută (natural_gradient=True, minibatch_frac=0.8)
    if use_requested:
        model = NGBRegressor(
            Dist=Poisson,
            n_estimators=400,
            learning_rate=0.03,
            minibatch_frac=0.8,
            natural_gradient=True,
            verbose=True,
            random_state=42,
        )
    else:
        # fallback robust (natural_gradient=False)
        model = NGBRegressor(
            Dist=Poisson,
            n_estimators=400,
            learning_rate=0.03,
            minibatch_frac=0.7,
            col_sample=0.8,
            natural_gradient=False,
            verbose=True,
            random_state=42,
        )
    # Ensure non-negative integer targets
    y = np.asarray(y, dtype=np.int64)
    y[y < 0] = 0
    model.fit(X.values, y)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description='Train NGBoost Poisson models for home/away goals')
    ap.add_argument('--league', required=True)
    args = ap.parse_args()
    lg = args.league
    # Step A: try requested config
    try:
        X, yH, yA = build_training_frame(lg)
        mH = train_ngb_poisson(X, yH, use_requested=True)
        mA = train_ngb_poisson(X, yA, use_requested=True)
    except Exception as e:
        print(f"[ngb] Requested config failed: {e}\n[ngb] Trying subset (last season) with robust fallback...")
        # Step B: subset last season
        subset_start = os.getenv('NGB_SUBSET_START', '2024-08-01')
        subset_end = os.getenv('NGB_SUBSET_END', None)
        X, yH, yA = build_training_frame(lg, start=subset_start, end=subset_end)
        try:
            mH = train_ngb_poisson(X, yH, use_requested=True)
            mA = train_ngb_poisson(X, yA, use_requested=True)
        except Exception as e2:
            print(f"[ngb] Requested config on subset failed: {e2}\n[ngb] Falling back to robust config (natural_gradient=False)...")
            mH = train_ngb_poisson(X, yH, use_requested=False)
            mA = train_ngb_poisson(X, yA, use_requested=False)
    Path('advanced_models').mkdir(parents=True, exist_ok=True)
    joblib.dump(mH, Path('advanced_models')/f'{lg}_ngb_poisson_home.pkl')
    joblib.dump(mA, Path('advanced_models')/f'{lg}_ngb_poisson_away.pkl')
    print(f"Saved NGBoost Poisson models for {lg} -> advanced_models/{lg}_ngb_poisson_{{home,away}}.pkl")


if __name__ == '__main__':
    main()
