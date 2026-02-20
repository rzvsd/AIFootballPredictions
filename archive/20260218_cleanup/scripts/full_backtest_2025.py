"""
Full Retroactive Backtest for 2025 ROI
Calculates ROI for O/U 2.5 and BTTS markets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson

def poisson_probs(mu_h, mu_a):
    """Calculate match probabilities from expected goals."""
    max_goals = 10
    home_probs = [poisson.pmf(i, mu_h) for i in range(max_goals)]
    away_probs = [poisson.pmf(i, mu_a) for i in range(max_goals)]
    
    # O/U 2.5
    p_under = sum(home_probs[i] * away_probs[j] for i in range(max_goals) for j in range(max_goals) if i + j <= 2)
    p_over = 1 - p_under
    
    # BTTS
    p_btts_yes = sum(home_probs[i] * away_probs[j] for i in range(1, max_goals) for j in range(1, max_goals))
    p_btts_no = 1 - p_btts_yes
    
    return {
        'p_over25': p_over,
        'p_under25': p_under,
        'p_btts_yes': p_btts_yes, 'p_btts_no': p_btts_no
    }

def main():
    # Load history with predictions
    df = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Filter 2025 and played matches
    df_2025 = df[df['date'].dt.year == 2025].copy()
    for c in ['ft_home', 'ft_away', 'mu_home', 'mu_away', 'odds_home', 'odds_draw', 'odds_away', 'odds_over', 'odds_under']:
        if c in df_2025.columns:
            df_2025[c] = pd.to_numeric(df_2025[c], errors='coerce')
    
    played = df_2025[~((df_2025['ft_home'] == 0) & (df_2025['ft_away'] == 0))].copy()
    print(f"2025 played matches: {len(played)}")
    
    # Load calibration models
    import joblib
    calib_models = {}
    for name in ["over", "btts"]:
        cp = Path('models/calibration') / f"calib_{name}.pkl"
        if cp.exists():
            calib_models[name] = joblib.load(cp)
            print(f"Loaded {name} calibrator")

    # Recalculate predictions from mu_home/mu_away
    results = []
    for idx, row in played.iterrows():
        mu_h = row.get('mu_home', 1.5)
        mu_a = row.get('mu_away', 1.2)
        if pd.isna(mu_h) or pd.isna(mu_a):
            mu_h, mu_a = 1.5, 1.2  # defaults
        
        probs = poisson_probs(mu_h, mu_a)
        
        # Apply Calibration
        if "over" in calib_models:
             probs["p_over25"] = float(calib_models["over"].predict([probs["p_over25"]])[0])
             probs["p_under25"] = 1.0 - probs["p_over25"]
             
        if "btts" in calib_models:
             probs["p_btts_yes"] = float(calib_models["btts"].predict([probs["p_btts_yes"]])[0])
             probs["p_btts_no"] = 1.0 - probs["p_btts_yes"]
        
        # Actual results
        actual_over25 = (row['ft_home'] + row['ft_away']) > 2.5
        actual_btts = (row['ft_home'] > 0) and (row['ft_away'] > 0)
        
        results.append({
            'date': row['date'],
            'home': row['home'],
            'away': row['away'],
            'ft_home': row['ft_home'],
            'ft_away': row['ft_away'],
            **probs,
            'odds_over': row.get('odds_over', 1.9),
            'odds_under': row.get('odds_under', 1.9),
            'actual_over25': actual_over25,
            'actual_btts': actual_btts,
        })
    
    bt = pd.DataFrame(results)
    
    # Calculate EV
    bt['EV_over25'] = bt['p_over25'] * bt['odds_over'] - 1
    bt['EV_under25'] = bt['p_under25'] * bt['odds_under'] - 1
    
    # Calculate ROI when betting on EV > threshold
    def calc_roi(df, pred_col, actual_col, odds_col, ev_col, ev_thresh=0.0):
        mask = df[ev_col] > ev_thresh
        bets = df[mask]
        if len(bets) == 0:
            return 0, 0, 0, 0
        stake = len(bets)
        returns = (bets[actual_col] * bets[odds_col]).sum()
        profit = returns - stake
        roi = profit / stake * 100
        win_rate = bets[actual_col].mean() * 100
        return roi, win_rate, int(bets[actual_col].sum()), stake
    
    print("\n" + "="*70)
    print("2025 ROI ANALYSIS - FULL BACKTEST")
    print("="*70)
    
    # O/U 2.5
    print("\nO/U 2.5 MARKET (betting when EV > 0%):")
    print("-" * 50)
    bt['actual_under25'] = ~bt['actual_over25']
    o_roi, o_wr, o_w, o_n = calc_roi(bt, 'p_over25', 'actual_over25', 'odds_over', 'EV_over25', 0)
    u_roi, u_wr, u_w, u_n = calc_roi(bt, 'p_under25', 'actual_under25', 'odds_under', 'EV_under25', 0)
    print(f"  Over 2.5:  ROI={o_roi:+.1f}%  WinRate={o_wr:.1f}%  ({o_w}/{o_n} bets)")
    print(f"  Under 2.5: ROI={u_roi:+.1f}%  WinRate={u_wr:.1f}%  ({u_w}/{u_n} bets)")
    
    # BTTS - use average odds if not available
    print("\nBTTS MARKET (estimated, avg odds ~1.8):")
    print("-" * 50)
    bt['odds_btts_yes'] = 1.85
    bt['odds_btts_no'] = 1.95
    bt['EV_btts_yes'] = bt['p_btts_yes'] * bt['odds_btts_yes'] - 1
    bt['EV_btts_no'] = bt['p_btts_no'] * bt['odds_btts_no'] - 1
    bt['actual_btts_no'] = ~bt['actual_btts']
    by_roi, by_wr, by_w, by_n = calc_roi(bt, 'p_btts_yes', 'actual_btts', 'odds_btts_yes', 'EV_btts_yes', 0)
    bn_roi, bn_wr, bn_w, bn_n = calc_roi(bt, 'p_btts_no', 'actual_btts_no', 'odds_btts_no', 'EV_btts_no', 0)
    print(f"  BTTS Yes:  ROI={by_roi:+.1f}%  WinRate={by_wr:.1f}%  ({by_w}/{by_n} bets)")
    print(f"  BTTS No:   ROI={bn_roi:+.1f}%  WinRate={bn_wr:.1f}%  ({bn_w}/{bn_n} bets)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Best Markets by ROI:")
    print("="*70)
    all_markets = [
        ('Over 2.5', o_roi, o_n),
        ('Under 2.5', u_roi, u_n),
        ('BTTS Yes', by_roi, by_n),
        ('BTTS No', bn_roi, bn_n),
    ]
    sorted_markets = sorted(all_markets, key=lambda x: x[1], reverse=True)
    for name, roi, bets in sorted_markets:
        status = "✅" if roi > 0 else "❌"
        print(f"{status} {name:12s}: ROI={roi:+.1f}%  ({bets} bets)")
    
    # Save backtest
    bt.to_csv('reports/full_backtest_2025.csv', index=False)
    print(f"\nSaved full backtest to: reports/full_backtest_2025.csv")

if __name__ == "__main__":
    main()
