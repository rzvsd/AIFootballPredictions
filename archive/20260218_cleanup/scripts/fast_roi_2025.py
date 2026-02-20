#!/usr/bin/env python3
"""
Fast ROI Analysis for 2025-2026
===============================
Uses pre-calculated features from frankenstein_training.csv matched with 
metadata from history to perform instant backtesting and ROI analysis.

Usage:
    python scripts/fast_roi_2025.py
"""

import sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from scipy.stats import poisson

_ROOT = Path(__file__).resolve().parents[1]

# Feature columns used by the model (excluding targets/metadata)
# Based on train_frankenstein_mu.py logic
IGNORE_COLS = {
    "y_home", "y_away", "date", "time", "datetime", "home", "away", 
    "code_home", "code_away", "result", "validated", 
    "ft_home", "ft_away", "ht_home", "ht_away",
    "shots", "shots_on_target", "corners", 
    "possession_home", "possession_away"
}

def poisson_probs(mu_h, mu_a):
    """Calculate match probabilities from expected goals."""
    max_goals = 10
    home_probs = [poisson.pmf(i, mu_h) for i in range(max_goals)]
    away_probs = [poisson.pmf(i, mu_a) for i in range(max_goals)]
    
    p_home = sum(home_probs[i] * sum(away_probs[j] for j in range(i)) for i in range(max_goals))
    p_away = sum(away_probs[j] * sum(home_probs[i] for i in range(j)) for j in range(max_goals))
    p_draw = sum(home_probs[i] * away_probs[i] for i in range(max_goals))
    
    # O/U 2.5
    p_under = sum(home_probs[i] * away_probs[j] for i in range(max_goals) for j in range(max_goals) if i + j <= 2)
    p_over = 1 - p_under
    
    # BTTS
    p_btts_yes = sum(home_probs[i] * away_probs[j] for i in range(1, max_goals) for j in range(1, max_goals))
    p_btts_no = 1 - p_btts_yes
    
    return p_home, p_draw, p_away, p_over, p_under, p_btts_yes, p_btts_no

def calc_roi(df, pred_col, actual_col, odds_col, ev_col, ev_thresh=0.0):
    """Calculate ROI when betting on EV > threshold."""
    bets = df[df[ev_col] > ev_thresh].copy()
    if len(bets) == 0:
        return 0.0, 0.0, 0, 0.0
    
    stake = len(bets)
    profit = (bets[actual_col] * bets[odds_col]).sum() - stake
    roi = profit / stake * 100
    win_rate = bets[actual_col].mean() * 100
    
    return roi, win_rate, int(bets[actual_col].sum()), stake

def analyze_league(df, league_name):
    """Analyze ROI for a specific league."""
    lg = df[df['league'] == league_name].copy()
    if len(lg) == 0:
        return None
        
    res = {'league': league_name, 'matches': len(lg)}
    
    # O/U
    o_roi, _, o_w, o_n = calc_roi(lg, 'p_over25', 'actual_over25', 'odds_over', 'EV_over25')
    u_roi, _, u_w, u_n = calc_roi(lg, 'p_under25', 'actual_under25', 'odds_under', 'EV_under25')
    res['ou_over_roi'] = o_roi
    res['ou_over_bets'] = o_n
    res['ou_under_roi'] = u_roi
    res['ou_under_bets'] = u_n
    
    # BTTS (using estimated odds if missing, usually avg ~1.85)
    # We use implied odds from history if available or default
    lg['odds_btts_yes'] = 1.85
    lg['odds_btts_no'] = 1.95
    lg['EV_btts_yes'] = lg['p_btts_yes'] * lg['odds_btts_yes'] - 1
    lg['EV_btts_no'] = lg['p_btts_no'] * lg['odds_btts_no'] - 1
    
    by_roi, _, by_w, by_n = calc_roi(lg, 'p_btts_yes', 'actual_btts', 'odds_btts_yes', 'EV_btts_yes')
    bn_roi, _, bn_w, bn_n = calc_roi(lg, 'p_btts_no', 'actual_btts_no', 'odds_btts_no', 'EV_btts_no')
    
    res['btts_yes_roi'] = by_roi
    res['btts_yes_bets'] = by_n
    res['btts_no_roi'] = bn_roi
    res['btts_no_bets'] = bn_n
    
    return res

def main():
    print("Loading data...")
    # 1. Load Features
    feat_path = _ROOT / "data" / "enhanced" / "frankenstein_training.csv"
    if not feat_path.exists():
        print("Feature file missing!")
        return
    X_all = pd.read_csv(feat_path)
    
    # 2. Load Metadata (History)
    hist_path = _ROOT / "data" / "enhanced" / "cgm_match_history_with_elo_stats_xg.csv"
    if not hist_path.exists():
        print("History file missing!")
        return
    meta = pd.read_csv(hist_path, low_memory=False)
    
    if len(X_all) != len(meta):
        print(f"data mismatch! Features: {len(X_all)}, History: {len(meta)}")
        return
        
    print(f"Data aligned: {len(X_all)} matches")
    
    # 3. Merge relevant metadata
    # We trust the row order matches exactly (same generation source)
    X_all['league'] = meta['league']
    X_all['season'] = meta['season']
    X_all['home'] = meta['home']
    X_all['away'] = meta['away']
    X_all['ft_home'] = meta['ft_home']
    X_all['ft_away'] = meta['ft_away']
    X_all['odds_home'] = meta['odds_home']
    X_all['odds_draw'] = meta['odds_draw']
    X_all['odds_away'] = meta['odds_away']
    X_all['odds_over'] = meta['odds_over']
    X_all['odds_under'] = meta['odds_under']
    
    # 4. Filter for 2025-2026 Season
    target_season = "2025-2026"
    df = X_all[X_all['season'] == target_season].copy()
    print(f"Filtered for {target_season}: {len(df)} matches")
    
    if len(df) == 0:
        print("No matches found for this season!")
        return

    # 5. Load Models
    print("Loading models...")
    model_h = joblib.load(_ROOT / "models" / "frankenstein_mu_home.pkl")
    model_a = joblib.load(_ROOT / "models" / "frankenstein_mu_away.pkl")
    
    # 6. Prepare Feature Matrix
    
    # Get expected features from model
    try:
        expected_features = model_h.feature_names_in_
    except AttributeError:
        # If raw booster
        expected_features = model_h.get_booster().feature_names
        
    print(f"Model expects {len(expected_features)} features")
    
    # Filter columns to match exactly
    # We might need to handle missing columns if frankenstein_training.csv is older/newer
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        print(f"WARNING: Input missing features: {missing}")
        # Add missing features as 0
        for c in missing:
            df[c] = 0
            
    clean_X = df[expected_features].copy()

    # 7. Predict
    print("Predicting...")
    # Ensure numeric
    clean_X = clean_X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    mu_h = model_h.predict(clean_X)
    mu_a = model_a.predict(clean_X)
    
    # 8. Calculate Probabilities & EV
    print("Calculating ROI...")
    results = []
    
    # Calibration loaders (optional, skipping for speed/simplicity or can load)
    # Ideally should load callibration models here.
    
    for i, (idx, row) in enumerate(df.iterrows()):
        mh, ma = mu_h[i], mu_a[i]
        ph, pd_prob, pa, po, pu, pby, pbn = poisson_probs(mh, ma)
        
        # Calculate EV
        ev_over = po * row.get('odds_over', 1.9) - 1
        ev_under = pu * row.get('odds_under', 1.9) - 1
        
        # Actuals
        ft_h = row.get('ft_home', 0)
        ft_a = row.get('ft_away', 0)
        act_over = (ft_h + ft_a) > 2.5
        act_btts = (ft_h > 0) and (ft_a > 0)
        
        results.append({
            'league': row['league'],
            'p_over25': po,
            'p_under25': pu,
            'p_btts_yes': pby,
            'p_btts_no': pbn,
            'odds_over': row.get('odds_over', 1.9),
            'odds_under': row.get('odds_under', 1.9),
            'EV_over25': ev_over,
            'EV_under25': ev_under,
            'actual_over25': act_over,
            'actual_under25': not act_over,
            'actual_btts': act_btts,
            'actual_btts_no': not act_btts
        })
        
    res_df = pd.DataFrame(results)
    
    # 9. Aggregate per league
    leagues = sorted(res_df['league'].unique())
    print("\n" + "="*60)
    print("PER-LEAGUE ROI ANALYSIS (2025-2026)")
    print("="*60)
    
    summary_rows = []
    
    print("\n⚽ OVER/UNDER 2.5")
    print("-" * 60)
    
    for lg in leagues:
        r = analyze_league(res_df, lg)
        if r:
            summary_rows.append(r)
            print(f"{lg:15} ({r['matches']} matches)")
            print(f"  Over:  {r['ou_over_roi']:+6.1f}% ROI ({r['ou_over_bets']} bets)")
            print(f"  Under: {r['ou_under_roi']:+6.1f}% ROI ({r['ou_under_bets']} bets)")
            
    # Save to CSV
    if summary_rows:
        out_file = _ROOT / "reports" / "roi_analysis_2025.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(out_file, index=False)
        print(f"\n✅ Full report saved to: {out_file}")
            
    print("\n🎯 BTTS (Estimated Odds)")
    print("-" * 60)
    for r in summary_rows:
        if r:
             print(f"{r['league']:15}")
             print(f"  Yes:   {r['btts_yes_roi']:+6.1f}% ROI ({r['btts_yes_bets']} bets)")
             print(f"  No:    {r['btts_no_roi']:+6.1f}% ROI ({r['btts_no_bets']} bets)")

if __name__ == "__main__":
    main()
