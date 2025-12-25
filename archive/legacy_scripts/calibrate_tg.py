"""
Fit per-league calibrators for Total-Goals intervals using isotonic regression per band.

Usage:
  python -m scripts.calibrate_tg --league E0 --bands 0-3 1-3 2-4 2-5 3-6 --start 2021-08-01 --end 2024-06-30
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import feature_store
import bet_fusion as fusion
import calibrators as cal


def read_enhanced(league: str) -> pd.DataFrame:
    enh = Path('data') / 'enhanced' / f'{league}_final_features.csv'
    if enh.exists():
        df = pd.read_csv(enh)
    else:
        proc = Path('data') / 'processed' / f'{league}_merged_preprocessed.csv'
        df = pd.read_csv(proc)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


def iter_matches(df: pd.DataFrame, start: str | None, end: str | None) -> List[dict]:
    d = df.copy()
    if start:
        d = d[d['Date'] >= pd.to_datetime(start)]
    if end:
        d = d[d['Date'] <= pd.to_datetime(end)]
    d = d.dropna(subset=['HomeTeam','AwayTeam','FTHG','FTAG']).sort_values('Date')
    rows = []
    for _, r in d.iterrows():
        rows.append({
            'date': r['Date'],
            'home': str(r['HomeTeam']),
            'away': str(r['AwayTeam']),
            'tot': int(r['FTHG']) + int(r['FTAG']),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description='Calibrate TG intervals via isotonic per band')
    ap.add_argument('--league', required=True)
    ap.add_argument('--bands', nargs='+', default=['0-3','1-3','2-4','2-5','3-6'])
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--method', choices=['isotonic','platt'], default='isotonic')
    args = ap.parse_args()

    league = args.league
    df = read_enhanced(league)
    matches = iter_matches(df, args.start, args.end)
    if not matches:
        print('No matches found for window.')
        raise SystemExit(1)

    # Build predictions per match at as_of date-1
    enh_csv = str(Path('data')/'enhanced'/f'{league}_final_features.csv')
    bands = [str(b) for b in args.bands]
    preds: Dict[str, List[float]] = {b: [] for b in bands}
    labels: Dict[str, List[int]] = {b: [] for b in bands}
    for m in matches:
        cutoff = (pd.to_datetime(m['date']) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        snap = feature_store.build_snapshot(enhanced_csv=enh_csv, as_of=cutoff)
        # emulate engine path
        from joblib import load
        home_path = Path('advanced_models')/f'{league}_ultimate_xgb_home.pkl'
        away_path = Path('advanced_models')/f'{league}_ultimate_xgb_away.pkl'
        xgh = load(home_path); xga = load(away_path)
        feat = fusion._feature_row_from_snapshot(snap, m['home'], m['away'])
        if feat is None:
            continue
        mu_h = fusion._predict_xgb(xgh, feat)
        mu_a = fusion._predict_xgb(xga, feat)
        P = fusion._score_matrix(mu_h, mu_a, max_goals=10)  # use Poisson baseline for calibration data
        ints = fusion._evaluate_all_markets(P, ou_lines=[2.5], intervals=[tuple(map(int,b.split('-'))) for b in bands])['Intervals']
        for b in bands:
            a,b2 = map(int, b.split('-'))
            p = float(ints.get(f'{a}-{b2}', 0.0))
            y = 1 if (m['tot'] >= a and m['tot'] <= b2) else 0
            preds[b].append(p)
            labels[b].append(y)

    out = {}
    for b in bands:
        p = np.array(preds[b], dtype=float)
        y = np.array(labels[b], dtype=int)
        if len(p) < 20:
            continue
        model = cal.calibrate_binary(p, y, method=args.method)
        out[b] = model

    if not out:
        print('No calibrators produced (insufficient data).')
        raise SystemExit(1)
    Path('calibrators').mkdir(parents=True, exist_ok=True)
    cal.save_calibrators(str(Path('calibrators')/f'{league}_tg.pkl'), out)
    print(f"[cal] Saved TG calibrators -> calibrators/{league}_tg.pkl (bands={list(out.keys())})")


if __name__ == '__main__':
    main()

