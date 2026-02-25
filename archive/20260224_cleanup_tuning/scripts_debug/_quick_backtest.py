import pandas as pd
import numpy as np

import math

def _poisson_probs(mu_h, mu_a, limit=10):
    p_h = [np.exp(-mu_h) * (mu_h**k) / math.factorial(k) for k in range(limit)]
    p_a = [np.exp(-mu_a) * (mu_a**k) / math.factorial(k) for k in range(limit)]
    
    p_over = 0.0
    for h in range(limit):
        for a in range(limit):
            if h + a > 2.5:
                p_over += p_h[h] * p_a[a]
    return p_over

def backtest_bundesliga():
    df = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')
    
    # We only want to test matches where we actually have history built up.
    # The first 50-80 matches of the season are unpredictable because form is building.
    # Let's start after matchweek 10 (roughly nov 1st).
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'ft_home', 'ft_away'])
    df = df[df['date'] > '2025-11-01']
    
    results = []
    
    for _, r in df.iterrows():
        # Re-create the formula from predict_upcoming.py
        lg_avg_h = max(r.get("lg_avg_gf_home", 1.5) if not pd.isna(r.get("lg_avg_gf_home")) else 1.5, 1.0)
        lg_avg_a = max(r.get("lg_avg_gf_away", 1.2) if not pd.isna(r.get("lg_avg_gf_away")) else 1.2, 0.8)

        xg_h_adj = float(r.get("xg_for_form_H", lg_avg_h))
        xg_a_adj = float(r.get("xg_for_form_A", lg_avg_a))
        xg_def_h = float(r.get("xg_against_form_A", lg_avg_h))
        xg_def_a = float(r.get("xg_against_form_H", lg_avg_a))

        attack_h_raw = max(xg_h_adj / max(lg_avg_h, 0.5), 0.3)
        attack_a_raw = max(xg_a_adj / max(lg_avg_a, 0.5), 0.3)
        defence_leak_a_raw = max(xg_def_a / max(lg_avg_a, 0.5), 0.3)
        defence_leak_h_raw = max(xg_def_h / max(lg_avg_h, 0.5), 0.3)

        dampen = 0.6
        attack_h = dampen * attack_h_raw + (1 - dampen) * 1.0
        attack_a = dampen * attack_a_raw + (1 - dampen) * 1.0
        defence_leak_a = dampen * defence_leak_a_raw + (1 - dampen) * 1.0
        defence_leak_h = dampen * defence_leak_h_raw + (1 - dampen) * 1.0

        elo_home = float(r.get("elo_home", 1500.0) if not pd.isna(r.get("elo_home")) else 1500.0)
        elo_away = float(r.get("elo_away", 1500.0) if not pd.isna(r.get("elo_away")) else 1500.0)
        
        elo_ratio_h = float(np.clip((elo_home / 1500.0) ** 0.2, 0.85, 1.15))
        elo_ratio_a = float(np.clip((elo_away / 1500.0) ** 0.2, 0.85, 1.15))

        mu_h = lg_avg_h * attack_h * defence_leak_a * elo_ratio_h
        mu_a = lg_avg_a * attack_a * defence_leak_h * elo_ratio_a

        mu_h = float(np.clip(mu_h, 0.3, 5.0))
        mu_a = float(np.clip(mu_a, 0.3, 5.0))

        p_over = _poisson_probs(mu_h, mu_a)
        p_under = 1.0 - p_over
        
        actual_total = r['ft_home'] + r['ft_away']
        is_over = int(actual_total > 2.5)
        
        # In our dataset we only pulled 'odds_over_2_5' implicitly from the API sync (actually sometimes missing). 
        # For a quick test, let's assume mirrored odds if under is missing
        odds_over = r.get('odds_over_2_5', np.nan)
        odds_under = r.get('odds_under_2_5', np.nan)
        
        if pd.isna(odds_over): odds_over = 1.90
        if pd.isna(odds_under): odds_under = 1.90
            
        imp_over = 1.0 / odds_over
        imp_under = 1.0 / odds_under
        
        bet_over = p_over > imp_over
        bet_under = p_under > imp_under
        
        # Don't bet both sides
        if bet_over and bet_under:
            if (p_over - imp_over) > (p_under - imp_under):
                bet_under = False
            else:
                bet_over = False
        
        profit = 0.0
        bet_placed = False
        if bet_over:
            profit = (odds_over - 1.0) if is_over else -1.0
            bet_placed = True
        elif bet_under:
            profit = (odds_under - 1.0) if not is_over else -1.0
            bet_placed = True
            
        results.append({
            'date': r['date'],
            'home': r['home'],
            'away': r['away'],
            'mu_total': mu_h + mu_a,
            'p_over': p_over,
            'p_under': p_under,
            'odds_over': odds_over,
            'odds_under': odds_under,
            'bet_over': bet_over,
            'bet_under': bet_under,
            'bet_placed': bet_placed,
            'actual': actual_total,
            'is_over': is_over,
            'profit': profit
        })

    res = pd.DataFrame(results)
    
    print("\n=== QUICK BACKTEST: BUNDESLIGA (Nov 2025 - Feb 2026) ===")
    print(f"Total Matches Analyzed: {len(res)}")
    print(f"Mean goal prediction (mu_total): {res['mu_total'].mean():.2f}")
    print(f"Actual mean goals per game: {res['actual'].mean():.2f}")
    
    bets_over = res[res['bet_over'] == True]
    bets_under = res[res['bet_under'] == True]
    all_bets = res[res['bet_placed'] == True]
    
    print(f"\n--- OVER 2.5 ---")
    print(f"Bets: {len(bets_over)}")
    if len(bets_over) > 0:
        win_rate = bets_over['is_over'].mean() * 100
        roi = (bets_over['profit'].sum() / len(bets_over)) * 100
        print(f"Win Rate: {win_rate:.1f}% | Net Units: {bets_over['profit'].sum():.2f} | ROI: {roi:.1f}%")

    print(f"\n--- UNDER 2.5 ---")
    print(f"Bets: {len(bets_under)}")
    if len(bets_under) > 0:
        win_rate = (1 - bets_under['is_over']).mean() * 100
        roi = (bets_under['profit'].sum() / len(bets_under)) * 100
        print(f"Win Rate: {win_rate:.1f}% | Net Units: {bets_under['profit'].sum():.2f} | ROI: {roi:.1f}%")
        
    print(f"\n--- COMBINED ---")
    print(f"Total Bets: {len(all_bets)}")
    if len(all_bets) > 0:
        roi = (all_bets['profit'].sum() / len(all_bets)) * 100
        print(f"Total Net Units: {all_bets['profit'].sum():.2f}")
        print(f"Combined ROI: {roi:.1f}%")


if __name__ == '__main__':
    backtest_bundesliga()
