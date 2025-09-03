# xgb_trainer.py
import argparse
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor


ULTIMATE_FEATURES = [
    'ShotConv_H', 'ShotConv_A', 'ShotConvRec_H', 'ShotConvRec_A',
    'PointsPerGame_H', 'PointsPerGame_A', 'CleanSheetStreak_H', 'CleanSheetStreak_A',
    'xGDiff_H', 'xGDiff_A', 'CornersConv_H', 'CornersConv_A',
    'CornersConvRec_H', 'CornersConvRec_A', 'NumMatches_H', 'NumMatches_A',
    'Elo_H', 'Elo_A', 'EloDiff',
    'GFvsMid_H', 'GAvsMid_H', 'GFvsHigh_H', 'GAvsHigh_H',
    'GFvsMid_A', 'GAvsMid_A', 'GFvsHigh_A', 'GAvsHigh_A'
]


def _build_fallback_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map processed dataset columns to ULTIMATE_FEATURES as a fallback training set.

    This uses last-5 goals proxies and season averages where needed.
    """
    X = pd.DataFrame(index=df.index)
    X['ShotConv_H'] = df.get('AvgLast5HomeGoalsScored')
    X['ShotConv_A'] = df.get('AvgLast5AwayGoalsScored')
    X['ShotConvRec_H'] = df.get('AvgLast5HomeGoalsConceded')
    X['ShotConvRec_A'] = df.get('AvgLast5AwayGoalsConceded')
    # Proxies for points per game: season average goals scored
    X['PointsPerGame_H'] = df.get('AvgHomeGoalsScored')
    X['PointsPerGame_A'] = df.get('AvgAwayGoalsScored')
    # Clean sheet streaks not available -> zeros
    X['CleanSheetStreak_H'] = 0.0
    X['CleanSheetStreak_A'] = 0.0
    # xG differential proxy from last-5 goals
    X['xGDiff_H'] = X['ShotConv_H'] - X['ShotConvRec_H']
    X['xGDiff_A'] = X['ShotConv_A'] - X['ShotConvRec_A']
    # Corners proxies not available -> zeros
    X['CornersConv_H'] = 0.0
    X['CornersConv_A'] = 0.0
    X['CornersConvRec_H'] = 0.0
    X['CornersConvRec_A'] = 0.0
    # Match-count stabilizers
    X['NumMatches_H'] = 20.0
    X['NumMatches_A'] = 20.0
    # Elo placeholders (will be filled later if computed)
    X['Elo_H'] = 1500.0
    X['Elo_A'] = 1500.0
    X['EloDiff'] = 0.0
    return X[ULTIMATE_FEATURES]


def _compute_ewma_elo_prematch(df: pd.DataFrame, half_life_matches: int = 5,
                               elo_k: float = 20.0, elo_home_adv: float = 60.0) -> pd.DataFrame:
    """Compute per-match pre-game EWMA (home/away splits) and Elo for each row."""
    d = df.copy()
    if 'Date' in d.columns:
        try:
            d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
        except Exception:
            pass
    d = d.dropna(subset=['Date','HomeTeam','AwayTeam']).sort_values('Date')
    alpha = 1 - 0.5**(1/float(half_life_matches))
    # state per team
    state = {}
    elo = {}
    # pre-match lists
    cols = ['xg_home_EWMA','xga_home_EWMA','ppg_home_EWMA','xg_away_EWMA','xga_away_EWMA','ppg_away_EWMA','Elo_H','Elo_A','EloDiff',
            'GFvsMid_H','GAvsMid_H','GFvsHigh_H','GAvsHigh_H','GFvsMid_A','GAvsMid_A','GFvsHigh_A','GAvsHigh_A']
    store = {c: [] for c in cols}
    for _, r in d.iterrows():
        home = str(r['HomeTeam']); away = str(r['AwayTeam'])
        # init state if needed
        if home not in state:
            state[home] = {'h_gf':1.2,'h_ga':1.2,'h_ppg':1.2,'a_gf':1.0,'a_ga':1.0,'a_ppg':1.0,
                           'h_gf_mid':1.2,'h_ga_mid':1.2,'h_gf_high':1.2,'h_ga_high':1.2,
                           'a_gf_mid':1.0,'a_ga_mid':1.0,'a_gf_high':1.0,'a_ga_high':1.0}
        if away not in state:
            state[away] = {'h_gf':1.2,'h_ga':1.2,'h_ppg':1.2,'a_gf':1.0,'a_ga':1.0,'a_ppg':1.0,
                           'h_gf_mid':1.2,'h_ga_mid':1.2,'h_gf_high':1.2,'h_ga_high':1.2,
                           'a_gf_mid':1.0,'a_ga_mid':1.0,'a_gf_high':1.0,'a_ga_high':1.0}
        eh = float(elo.get(home, 1500.0)); ea = float(elo.get(away, 1500.0))
        # record pre-match
        sh = state[home]; sa = state[away]
        store['xg_home_EWMA'].append(sh['h_gf'])
        store['xga_home_EWMA'].append(sh['h_ga'])
        store['ppg_home_EWMA'].append(sh['h_ppg'])
        store['xg_away_EWMA'].append(sa['a_gf'])
        store['xga_away_EWMA'].append(sa['a_ga'])
        store['ppg_away_EWMA'].append(sa['a_ppg'])
        store['Elo_H'].append(eh)
        store['Elo_A'].append(ea)
        store['EloDiff'].append(eh - ea)
        # initialize banded stores with current pre-match EWMA (will update after match)
        store['GFvsMid_H'].append(state[home]['h_gf_mid'])
        store['GAvsMid_H'].append(state[home]['h_ga_mid'])
        store['GFvsHigh_H'].append(state[home]['h_gf_high'])
        store['GAvsHigh_H'].append(state[home]['h_ga_high'])
        store['GFvsMid_A'].append(state[away]['a_gf_mid'])
        store['GAvsMid_A'].append(state[away]['a_ga_mid'])
        store['GFvsHigh_A'].append(state[away]['a_gf_high'])
        store['GAvsHigh_A'].append(state[away]['a_ga_high'])
        # update after match
        try:
            fthg = float(r['FTHG']); ftag = float(r['FTAG'])
        except Exception:
            continue
        sh['h_gf'] = (1-alpha)*sh['h_gf'] + alpha*fthg
        sh['h_ga'] = (1-alpha)*sh['h_ga'] + alpha*ftag
        sh['h_ppg'] = (1-alpha)*sh['h_ppg'] + alpha*(3.0 if fthg>ftag else (1.0 if fthg==ftag else 0.0))
        sa['a_gf'] = (1-alpha)*sa['a_gf'] + alpha*ftag
        sa['a_ga'] = (1-alpha)*sa['a_ga'] + alpha*fthg
        sa['a_ppg'] = (1-alpha)*sa['a_ppg'] + alpha*(3.0 if ftag>fthg else (1.0 if ftag==fthg else 0.0))
        exp_home = 1.0/(1.0 + 10**(-((eh + elo_home_adv) - ea)/400.0))
        S_home = 1.0 if fthg>ftag else (0.5 if fthg==ftag else 0.0)
        elo[home] = eh + elo_k*(S_home - exp_home)
        elo[away] = ea + elo_k*((1.0 - S_home) - (1.0 - exp_home))
        # Update banded EWMA based on opponent pre-match Elo
        if 1600.0 <= ea <= 1800.0:
            state[home]['h_gf_mid'] = (1-alpha)*state[home]['h_gf_mid'] + alpha*fthg
            state[home]['h_ga_mid'] = (1-alpha)*state[home]['h_ga_mid'] + alpha*ftag
        elif ea > 1800.0:
            state[home]['h_gf_high'] = (1-alpha)*state[home]['h_gf_high'] + alpha*fthg
            state[home]['h_ga_high'] = (1-alpha)*state[home]['h_ga_high'] + alpha*ftag
        if 1600.0 <= eh <= 1800.0:
            state[away]['a_gf_mid'] = (1-alpha)*state[away]['a_gf_mid'] + alpha*ftag
            state[away]['a_ga_mid'] = (1-alpha)*state[away]['a_ga_mid'] + alpha*fthg
        elif eh > 1800.0:
            state[away]['a_gf_high'] = (1-alpha)*state[away]['a_gf_high'] + alpha*ftag
            state[away]['a_ga_high'] = (1-alpha)*state[away]['a_ga_high'] + alpha*fthg
    # attach
    for c in cols:
        d[c] = store[c]
    return d


def train_xgb_models_for_league(league: str) -> None:
    # Prefer enhanced final_features; fallback to processed merged_preprocessed
    # Always use processed data to ensure labels and chronological order for Elo/EWMA
    proc_path = os.path.join('data', 'processed', f'{league}_merged_preprocessed.csv')
    print(f"Loading processed data: {proc_path}")
    df = pd.read_csv(proc_path)
    # Ensure labels exist; if missing, try raw merge
    y_home = df.get('FTHG'); y_away = df.get('FTAG')
    if y_home is None or (hasattr(y_home,'isna') and y_home.isna().all()) or y_away is None or (hasattr(y_away,'isna') and y_away.isna().all()):
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
                y_home = df.get('FTHG')
                y_away = df.get('FTAG')
            else:
                print('Warning: could not align with raw file to get labels (missing keys).')
        else:
            print(f'Warning: raw file not found for labels: {raw_path}')
    # Compute per-match EWMA and Elo
    d_feats = _compute_ewma_elo_prematch(df, half_life_matches=5, elo_k=20.0, elo_home_adv=60.0)
    # Build features consistent with inference mapping
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
    # Ensure all expected columns exist
    for col in ULTIMATE_FEATURES:
        if col not in X.columns:
            X[col] = 0.0
    X = X[ULTIMATE_FEATURES]
    y_home = df.get('FTHG')
    y_away = df.get('FTAG')

    # Drop rows with missing labels (align safely)
    train_df = X.copy()
    train_df['FTHG'] = y_home.values if hasattr(y_home, 'values') else y_home
    train_df['FTAG'] = y_away.values if hasattr(y_away, 'values') else y_away
    train_df = train_df.dropna(subset=['FTHG','FTAG'])
    X = train_df[ULTIMATE_FEATURES].astype(float).fillna(0.0)
    y_home = train_df['FTHG'].astype(float)
    y_away = train_df['FTAG'].astype(float)

    print(f"Training XGB for {league}: {len(X)} samples, {X.shape[1]} features")
    home_model = XGBRegressor(objective='reg:squarederror', n_estimators=220, learning_rate=0.06, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42)
    away_model = XGBRegressor(objective='reg:squarederror', n_estimators=220, learning_rate=0.06, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42)
    home_model.fit(X, y_home)
    away_model.fit(X, y_away)

    out_dir = 'advanced_models'
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(home_model, os.path.join(out_dir, f'{league}_ultimate_xgb_home.pkl'))
    joblib.dump(away_model, os.path.join(out_dir, f'{league}_ultimate_xgb_away.pkl'))
    print(f"Saved models for {league} in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description='Train XGB goal models per league (fallback features)')
    ap.add_argument('--league', required=True, help='League code, e.g., E0, D1, F1, SP1, I1')
    args = ap.parse_args()
    train_xgb_models_for_league(args.league)


if __name__ == '__main__':
    main()
