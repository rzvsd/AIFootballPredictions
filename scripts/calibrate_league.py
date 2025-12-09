"""
Fit per-league probability calibrators for 1X2 and OU 2.5.

Approach:
- Build pre-match features per row (EWMA + Elo) using the same logic as trainer
- Predict μ_home/μ_away with XGB models
- Convert to Poisson probabilities for 1X2 and OU 2.5
- Fit isotonic (default) or Platt calibrators on held-out style data
- Save to calibrators/{LEAGUE}_1x2.pkl and calibrators/{LEAGUE}_ou25.pkl

Usage:
  python -m scripts.calibrate_league --league E0
  python -m scripts.calibrate_league --league E0 D1 F1 SP1 I1 --method isotonic --since 2021
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

import calibrators as cal
from models import xgb_trainer as trainer
import config
import bet_fusion as fusion


def _load_processed_with_labels(league: str) -> pd.DataFrame:
    proc_path = os.path.join('data', 'processed', f'{league}_merged_preprocessed.csv')
    df = pd.read_csv(proc_path)
    # Ensure Date exists and parse
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception:
            pass
    # Ensure labels; if missing, merge from raw
    need_labels = ('FTHG' not in df.columns) or ('FTAG' not in df.columns) or df['FTHG'].isna().all() or df['FTAG'].isna().all()
    if need_labels:
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
    return df


def _build_features_for_trainer(d_feats: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d_feats.index)
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
    # Elo-similarity dynamic features if available
    for k in ('GFvsSim_H','GAvsSim_H','GFvsSim_A','GAvsSim_A'):
        if k in d_feats.columns:
            X[k] = d_feats[k]
    for col in config.ULTIMATE_FEATURES:
        if col not in X.columns:
            X[col] = 0.0
    return X[config.ULTIMATE_FEATURES].astype(float).fillna(0.0)


def _poisson_probs(mu_h: float, mu_a: float, max_goals: int = 10) -> Tuple[float,float,float,float]:
    P = fusion._score_matrix(mu_h, mu_a, max_goals=max_goals, trim_epsilon=0.0)
    x = fusion._oneXtwo_from_matrix(P)
    # Over 2.5
    # reuse goals distribution and cutoff
    n = P.shape[0]
    grid = np.add.outer(np.arange(n), np.arange(n))
    over25 = float(P[grid >= 3].sum())
    return float(x['p_H']), float(x['p_D']), float(x['p_A']), over25


def calibrate_league(league: str, method: str, since_year: int, half_life: int, elo_k: float, elo_home_adv: float) -> None:
    print(f"[cal] League={league} method={method} since>={since_year}")
    df = _load_processed_with_labels(league)
    # Filter by date
    if 'Date' in df.columns:
        df = df.dropna(subset=['Date','HomeTeam','AwayTeam'])
        df = df[df['Date'].dt.year >= since_year]
    # Compute pre-match EWMA/Elo
    d_feats = trainer._compute_ewma_elo_prematch(df, half_life_matches=half_life, elo_k=elo_k, elo_home_adv=elo_home_adv)
    # Build X
    X = _build_features_for_trainer(d_feats)
    # Labels
    FTHG = df.get('FTHG')
    FTAG = df.get('FTAG')
    lab_ok = (FTHG is not None) and (FTAG is not None)
    if not lab_ok:
        raise RuntimeError('Labels FTHG/FTAG not available for calibration.')
    lab_df = pd.DataFrame({'FTHG': FTHG, 'FTAG': FTAG}).astype(float)
    mask = lab_df.notna().all(axis=1)
    X = X.loc[mask]
    lab_df = lab_df.loc[mask]
    # Models
    m_home = joblib.load(os.path.join('advanced_models', f'{league}_ultimate_xgb_home.pkl'))
    m_away = joblib.load(os.path.join('advanced_models', f'{league}_ultimate_xgb_away.pkl'))
    # Align features to model's expected order/names
    try:
        names_h = m_home.get_booster().feature_names
    except Exception:
        names_h = list(X.columns)
    try:
        names_a = m_away.get_booster().feature_names
    except Exception:
        names_a = list(X.columns)
    Xh = X[names_h].values if names_h else X.values
    Xa = X[names_a].values if names_a else X.values
    mu_h = m_home.predict(Xh)
    mu_a = m_away.predict(Xa)
    # Compute probabilities
    max_goals = int(config.MAX_GOALS_PER_LEAGUE.get(league, 10))
    pH_list: List[float] = []
    pD_list: List[float] = []
    pA_list: List[float] = []
    pO_list: List[float] = []
    for a, b in zip(mu_h, mu_a):
        h, d, a_, over = _poisson_probs(float(a), float(b), max_goals=max_goals)
        pH_list.append(h); pD_list.append(d); pA_list.append(a_); pO_list.append(over)
    p1x2_arr = np.column_stack([pH_list, pD_list, pA_list])
    y_1x2 = np.where(lab_df['FTHG'].values > lab_df['FTAG'].values, 'H', np.where(lab_df['FTHG'].values < lab_df['FTAG'].values, 'A', 'D'))
    y_ou = (lab_df['FTHG'].values + lab_df['FTAG'].values > 2).astype(int)
    # Fit calibrators
    cal1 = cal.multi_calibrate_1x2(p1x2_arr, y_1x2, method=method)
    cal.save_calibrators(os.path.join('calibrators', f'{league}_1x2.pkl'), cal1)
    cal_ou_model = cal.calibrate_binary(np.array(pO_list), y_ou, method=method)
    cal.save_calibrators(os.path.join('calibrators', f'{league}_ou25.pkl'), {'ou25': cal_ou_model})
    print(f"[cal] Saved: calibrators/{league}_1x2.pkl and calibrators/{league}_ou25.pkl  (n={len(X)})")


def main() -> None:
    ap = argparse.ArgumentParser(description='Fit per-league probability calibrators (1X2 and OU 2.5)')
    ap.add_argument('--league', nargs='+', required=True, help='One or more leagues (e.g., E0 D1 F1 SP1 I1)')
    ap.add_argument('--method', choices=['isotonic','platt'], default='isotonic')
    ap.add_argument('--since', type=int, default=2021, help='Use matches from this year and later')
    ap.add_argument('--half-life', type=int, default=5)
    ap.add_argument('--elo-k', type=float, default=15.0)
    ap.add_argument('--elo-home-adv', type=float, default=40.0)
    args = ap.parse_args()

    for lg in args.league:
        calibrate_league(lg, method=args.method, since_year=args.since, half_life=args.half_life, elo_k=args.elo_k, elo_home_adv=args.elo_home_adv)


if __name__ == '__main__':
    main()
