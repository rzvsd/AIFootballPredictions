#!/usr/bin/env python3
"""
Per-League ROI Analysis
=======================
Calculates ROI for each league separately across goals markets (O/U, BTTS).

Usage:
    python scripts/roi_by_league.py
    python scripts/roi_by_league.py --season 2024-2025
"""

import sys
from pathlib import Path
import pandas as pd
import argparse

_ROOT = Path(__file__).resolve().parents[1]


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


def analyze_league(bt, league_name):
    """Analyze ROI for a specific league."""
    league_data = bt[bt['league'] == league_name]
    
    if len(league_data) == 0:
        return None
    
    results = {
        'league': league_name,
        'total_matches': len(league_data),
    }
    
    # O/U
    o_roi, o_wr, o_w, o_n = calc_roi(league_data, 'p_over25', 'actual_over25', 'odds_over', 'EV_over25', 0)
    u_roi, u_wr, u_w, u_n = calc_roi(league_data, 'p_under25', 'actual_under25', 'odds_under', 'EV_under25', 0)
    
    results['ou_over_roi'] = o_roi
    results['ou_under_roi'] = u_roi
    results['ou_over_bets'] = o_n
    results['ou_under_bets'] = u_n
    
    # BTTS
    by_roi, by_wr, by_w, by_n = calc_roi(league_data, 'p_btts_yes', 'actual_btts', 'odds_btts_yes', 'EV_btts_yes', 0)
    bn_roi, bn_wr, bn_w, bn_n = calc_roi(league_data, 'p_btts_no', 'actual_btts_no', 'odds_btts_no', 'EV_btts_no', 0)
    
    results['btts_yes_roi'] = by_roi
    results['btts_no_roi'] = bn_roi
    results['btts_yes_bets'] = by_n
    results['btts_no_bets'] = bn_n
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate ROI per league")
    parser.add_argument("--backtest", default="reports/full_backtest_2025.csv", help="Path to backtest file")
    parser.add_argument("--season", help="Filter by season (e.g., 2024-2025)")
    args = parser.parse_args()
    
    bt_path = _ROOT / args.backtest
    if not bt_path.exists():
        print(f"ERROR: Backtest file not found: {bt_path}")
        print("Run a backtest first with: python scripts/run_multi_league_backtest.py")
        return 1
    
    bt = pd.read_csv(bt_path)
    print(f"Loaded {len(bt)} matches from backtest")
    
    # Add league column from history if missing
    if 'league' not in bt.columns:
        print("Adding league data from history...")
        hist = pd.read_csv(_ROOT / 'data/enhanced/cgm_match_history_with_elo_stats_xg.csv', low_memory=False)
        hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
        bt['date'] = pd.to_datetime(bt['date'], errors='coerce')
        
        # Merge league from history
        bt = bt.merge(
            hist[['date', 'home', 'away', 'league']],
            on=['date', 'home', 'away'],
            how='left'
        )
        print(f"Merged league data: {bt['league'].notna().sum()}/{len(bt)} matches matched")
    
    if args.season:
        bt = bt[bt['season'] == args.season]
        print(f"Filtered to {len(bt)} matches for season {args.season}")
    
    # Get all leagues
    leagues = sorted(bt['league'].unique())
    
    print("\n" + "="*80)
    print("PER-LEAGUE ROI ANALYSIS")
    print("="*80)
    
    all_results = []
    
    for league in leagues:
        result = analyze_league(bt, league)
        if result:
            all_results.append(result)
    
    # Create DataFrame for nice display
    df_results = pd.DataFrame(all_results)
    
    # Display by market type
    print("\n⚽ OVER/UNDER 2.5")
    print("-" * 80)
    for _, row in df_results.iterrows():
        print(f"\n{row['league']:15s}")
        print(f"  Over:  {row['ou_over_roi']:+6.1f}% ROI  ({int(row['ou_over_bets'])} bets)")
        print(f"  Under: {row['ou_under_roi']:+6.1f}% ROI  ({int(row['ou_under_bets'])} bets)")
    
    print("\n🎯 BTTS")
    print("-" * 80)
    for _, row in df_results.iterrows():
        print(f"\n{row['league']:15s}")
        print(f"  Yes: {row['btts_yes_roi']:+6.1f}% ROI  ({int(row['btts_yes_bets'])} bets)")
        print(f"  No:  {row['btts_no_roi']:+6.1f}% ROI  ({int(row['btts_no_bets'])} bets)")
    
    # Summary - best/worst leagues per market
    print("\n" + "="*80)
    print("SUMMARY - Best/Worst Leagues")
    print("="*80)
    
    print("\n🏆 Best ROI by Market:")
    print(f"  Over 2.5:  {df_results.loc[df_results['ou_over_roi'].idxmax(), 'league']} ({df_results['ou_over_roi'].max():+.1f}%)")
    print(f"  BTTS Yes:  {df_results.loc[df_results['btts_yes_roi'].idxmax(), 'league']} ({df_results['btts_yes_roi'].max():+.1f}%)")
    
    print("\n❌ Worst ROI by Market:")
    print(f"  Over 2.5:  {df_results.loc[df_results['ou_over_roi'].idxmin(), 'league']} ({df_results['ou_over_roi'].min():+.1f}%)")
    print(f"  BTTS Yes:  {df_results.loc[df_results['btts_yes_roi'].idxmin(), 'league']} ({df_results['btts_yes_roi'].min():+.1f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
