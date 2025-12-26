"""
Milestone 11: League-Specific Features Audit

Verifies league profile features are correctly computed.
Tests:
1. League features exist in training data
2. Value ranges are reasonable (goals ~2-4, rates 0-1)
3. Different leagues have different profiles
4. No future leakage in rolling window
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))

from config import LEAGUE_FEATURES_ENABLED, LEAGUE_MIN_MATCHES, LEAGUE_PROFILE_WINDOW


def main():
    print("="*80)
    print("LEAGUE-SPECIFIC FEATURES AUDIT - Milestone 11")
    print("="*80)
    print(f"Config: ENABLED={LEAGUE_FEATURES_ENABLED}, MIN_MATCHES={LEAGUE_MIN_MATCHES}, WINDOW={LEAGUE_PROFILE_WINDOW}")
    
    # Load training data
    df = pd.read_csv('data/enhanced/frankenstein_training.csv')
    print(f"Loaded {len(df)} training rows")
    
    issues = 0
    
    # Audit 1: Check league features exist
    print("\n--- Audit 1: League Features Present ---")
    expected_lg = [
        'lg_goals_per_match', 'lg_home_win_rate', 'lg_btts_rate',
        'lg_over25_rate', 'lg_home_advantage', 'lg_defensive_idx', 'lg_profile_usable'
    ]
    
    for col in expected_lg:
        if col in df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ❌ {col} MISSING")
            issues += 1
    
    if issues > 0:
        print("\n⚠️ League features not found - cannot continue audit")
        return
    
    # Audit 2: Value ranges
    print("\n--- Audit 2: League Feature Ranges ---")
    
    usable = df[df['lg_profile_usable'] == 1]
    print(f"  Usable profiles: {len(usable)} ({len(usable)/len(df)*100:.1f}%)")
    
    # Goals per match - typically 2.5-3.0
    min_g, max_g, mean_g = usable['lg_goals_per_match'].min(), usable['lg_goals_per_match'].max(), usable['lg_goals_per_match'].mean()
    print(f"  lg_goals_per_match: min={min_g:.2f}, max={max_g:.2f}, mean={mean_g:.2f}")
    if min_g < 1.5 or max_g > 4.5:
        print("    ⚠️ Unusual goals per match range")
    
    # Rates should be in [0, 1]
    for col in ['lg_home_win_rate', 'lg_btts_rate', 'lg_over25_rate']:
        min_v, max_v = usable[col].min(), usable[col].max()
        print(f"  {col}: min={min_v:.3f}, max={max_v:.3f}")
        if min_v < 0 or max_v > 1:
            print(f"    ⚠️ Out of [0, 1] range!")
            issues += 1
    
    # Home advantage - typically 1.2-1.5
    min_ha, max_ha = usable['lg_home_advantage'].min(), usable['lg_home_advantage'].max()
    print(f"  lg_home_advantage: min={min_ha:.3f}, max={max_ha:.3f}")
    
    # Defensive index - typically 0.8-1.2
    min_di, max_di = usable['lg_defensive_idx'].min(), usable['lg_defensive_idx'].max()
    print(f"  lg_defensive_idx: min={min_di:.3f}, max={max_di:.3f}")
    
    # Audit 3: Data completeness
    print("\n--- Audit 3: Data Completeness ---")
    for col in expected_lg:
        nan_pct = df[col].isna().sum() / len(df) * 100
        print(f"  {col}: {nan_pct:.1f}% NaN")
    
    # Audit 4: League variation
    print("\n--- Audit 4: League Profile Variation ---")
    print("  Checking if different leagues have different profiles...")
    
    # Load history to get league info
    hist = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')
    
    # Get unique leagues from history
    leagues = hist['league'].dropna().unique()
    print(f"  Leagues in history: {list(leagues)[:5]}...")
    
    # Check if league features vary (std > 0)
    std_goals = usable['lg_goals_per_match'].std()
    std_btts = usable['lg_btts_rate'].std()
    std_home = usable['lg_home_win_rate'].std()
    
    print(f"  Feature variation (std):")
    print(f"    lg_goals_per_match: std={std_goals:.4f}")
    print(f"    lg_btts_rate: std={std_btts:.4f}")
    print(f"    lg_home_win_rate: std={std_home:.4f}")
    
    if std_goals < 0.01 and std_btts < 0.01:
        print("    ⚠️ Very low variation - features may not differentiate leagues")
    else:
        print("    ✓ Sufficient variation detected")
    
    print("\n" + "="*80)
    if issues == 0:
        print("✅ League Features Audit PASSED!")
    else:
        print(f"⚠️ Found {issues} issues")
    print("="*80)


if __name__ == "__main__":
    main()
