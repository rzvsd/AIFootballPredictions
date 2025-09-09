"""
A/B comparison: baseline engine (XGB -> Poisson/NB) vs probabilistic model (NGBoost Poisson) using CRPS and LogLoss.

Usage:
  python -m scripts.ab_compare_prob_models --league E0 --start 2023-08-01 --end 2024-06-30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import feature_store
import bet_fusion as fusion
import joblib


def read_enhanced(league: str) -> pd.DataFrame:
    enh = Path('data')/'enhanced'/f'{league}_final_features.csv'
    if enh.exists():
        df = pd.read_csv(enh)
    else:
        df = pd.read_csv(Path('data')/'processed'/f'{league}_merged_preprocessed.csv')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


def iter_matches(df: pd.DataFrame, start: str|None, end: str|None) -> List[dict]:
    d = df.copy()
    if start:
        d = d[d['Date'] >= pd.to_datetime(start)]
    if end:
        d = d[d['Date'] <= pd.to_datetime(end)]
    d = d.dropna(subset=['HomeTeam','AwayTeam','FTHG','FTAG']).sort_values('Date')
    out = []
    for _, r in d.iterrows():
        out.append({'date': r['Date'], 'home': str(r['HomeTeam']), 'away': str(r['AwayTeam']), 'tot': int(r['FTHG'])+int(r['FTAG']), 'res': 'H' if r['FTHG']>r['FTAG'] else ('A' if r['FTAG']>r['FTHG'] else 'D')})
    return out


def crps_from_dist(dist: np.ndarray, y: int) -> float:
    cdf = np.cumsum(dist)
    val = 0.0
    for t in range(len(dist)):
        F = cdf[t]
        H = 1.0 if t >= y else 0.0
        val += (F - H) ** 2
    return float(val)


def main():
    ap = argparse.ArgumentParser(description='A/B compare baseline vs NGBoost Poisson')
    ap.add_argument('--league', required=True)
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    args = ap.parse_args()

    lg = args.league
    df = read_enhanced(lg)
    rows = iter_matches(df, args.start, args.end)
    if not rows:
        print('No matches found for window.')
        raise SystemExit(1)

    # Load models
    xgb_home = joblib.load(Path('advanced_models')/f'{lg}_ultimate_xgb_home.pkl')
    xgb_away = joblib.load(Path('advanced_models')/f'{lg}_ultimate_xgb_away.pkl')
    try:
        ngb_home = joblib.load(Path('advanced_models')/f'{lg}_ngb_poisson_home.pkl')
        ngb_away = joblib.load(Path('advanced_models')/f'{lg}_ngb_poisson_away.pkl')
        has_ngb = True
    except Exception:
        has_ngb = False

    enh_csv = str(Path('data')/'enhanced'/f'{lg}_final_features.csv')
    crps_base=[]; crps_ngb=[]; ll1_base=[]; ll1_ngb=[]
    for m in rows:
        cutoff = (pd.to_datetime(m['date']) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        snap = feature_store.build_snapshot(enhanced_csv=enh_csv, as_of=cutoff)
        feat = fusion._feature_row_from_snapshot(snap, m['home'], m['away'])
        if feat is None:
            continue
        # baseline Poisson
        mu_h = fusion._predict_xgb(xgb_home, feat); mu_a = fusion._predict_xgb(xgb_away, feat)
        P = fusion._score_matrix(mu_h, mu_a, max_goals=10)
        dist = fusion._goals_distribution(P)
        crps_base.append(crps_from_dist(dist, m['tot']))
        p1x2 = fusion._eval_1x2(P)
        ll1_base.append(-(np.log(max(1e-12, p1x2[m['res']]))) )
        # ngb
        if has_ngb:
            mu_h2 = float(max(0.05, ngb_home.predict(feat)[0]))
            mu_a2 = float(max(0.05, ngb_away.predict(feat)[0]))
            P2 = fusion._score_matrix(mu_h2, mu_a2, max_goals=10)
            dist2 = fusion._goals_distribution(P2)
            crps_ngb.append(crps_from_dist(dist2, m['tot']))
            p1x2_2 = fusion._eval_1x2(P2)
            ll1_ngb.append(-(np.log(max(1e-12, p1x2_2[m['res']]))) )

    def mean(x):
        return float(np.nanmean(x)) if x else float('nan')
    print(f"League {lg}  window=({args.start}..{args.end})")
    print(f"Baseline: CRPS={mean(crps_base):.4f}  LogLoss1X2={mean(ll1_base):.4f}  n={len(crps_base)}")
    if has_ngb:
        print(f"NGBoost : CRPS={mean(crps_ngb):.4f}  LogLoss1X2={mean(ll1_ngb):.4f}  n={len(crps_ngb)}")
        d_crps = mean(crps_ngb) - mean(crps_base)
        d_ll = mean(ll1_ngb) - mean(ll1_base)
        print(f"Δ vs baseline: CRPS={d_crps:+.4f}  LogLoss1X2={d_ll:+.4f}")
    else:
        print('NGBoost models not found. Train them with: python -m scripts.ngboost_trainer --league', lg)


if __name__ == '__main__':
    main()

