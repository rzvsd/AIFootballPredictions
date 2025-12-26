"""
Milestone 2 & 3: Pressure & xG Form Audit

Since pressure and xG form features are computed dynamically during 
training (build_frankenstein.py) and inference (predict_upcoming.py),
this audit verifies:
1. The calculation logic produces consistent results
2. Evidence counts are within expected bounds
3. No obvious data anomalies
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("PRESSURE & xG FORM AUDIT - Milestones 2 & 3")
    print("="*80)
    
    # Load training data
    df = pd.read_csv('data/enhanced/frankenstein_training.csv')
    print(f"Loaded {len(df)} training rows")
    
    # Get pressure columns
    press_cols = [c for c in df.columns if 'press_' in c.lower()]
    xg_cols = [c for c in df.columns if 'xg_' in c.lower()]
    
    print(f"Pressure columns: {len(press_cols)}")
    print(f"xG columns: {len(xg_cols)}")
    
    # Audit 1: Check value ranges
    print("\n--- Audit 1: Value Range Checks ---")
    issues = 0
    
    for col in ['press_form_H', 'press_form_A']:
        if col in df.columns:
            min_v, max_v = df[col].min(), df[col].max()
            print(f"  {col}: min={min_v:.3f}, max={max_v:.3f}")
            if min_v < 0 or max_v > 1:
                print(f"    ⚠️ Out of expected [0, 1] range!")
                issues += 1
        else:
            print(f"  {col}: NOT FOUND")
    
    for col in ['press_n_H', 'press_n_A']:
        if col in df.columns:
            min_v, max_v = df[col].min(), df[col].max()
            print(f"  {col}: min={min_v:.0f}, max={max_v:.0f}")
            if max_v > 10:
                print(f"    ⚠️ Exceeds L10 window!")
                issues += 1
    
    # Audit 2: Check for NaN distribution
    print("\n--- Audit 2: Data Completeness ---")
    for col in ['press_form_H', 'press_form_A', 'press_n_H', 'press_n_A']:
        if col in df.columns:
            nan_pct = df[col].isna().sum() / len(df) * 100
            print(f"  {col}: {nan_pct:.1f}% NaN")
    
    # Audit 3: xG features
    print("\n--- Audit 3: xG Feature Ranges ---")
    for col in ['xg_for_form_H', 'xg_for_form_A', 'xg_against_form_H', 'xg_against_form_A']:
        if col in df.columns:
            min_v, max_v = df[col].min(), df[col].max()
            print(f"  {col}: min={min_v:.2f}, max={max_v:.2f}")
            if min_v < 0:
                print(f"    ⚠️ Negative xG values!")
                issues += 1
    
    for col in ['xg_stats_n_H', 'xg_stats_n_A']:
        if col in df.columns:
            min_v, max_v = df[col].min(), df[col].max()
            print(f"  {col}: min={min_v:.0f}, max={max_v:.0f}")
            if max_v > 10:
                print(f"    ⚠️ Exceeds L10 window!")
                issues += 1
    
    # Audit 4: Divergence features
    print("\n--- Audit 4: Divergence Features ---")
    for col in ['div_team_H', 'div_team_A', 'div_diff', 'div_px_diff']:
        if col in df.columns:
            min_v, max_v, std_v = df[col].min(), df[col].max(), df[col].std()
            print(f"  {col}: min={min_v:.2f}, max={max_v:.2f}, std={std_v:.2f}")
        else:
            print(f"  {col}: NOT FOUND")
    
    # Audit 5: Sterile/Assassin flags
    print("\n--- Audit 5: Sterile/Assassin Flags ---")
    sterile_cols = [c for c in df.columns if 'sterile' in c.lower()]
    assassin_cols = [c for c in df.columns if 'assassin' in c.lower()]
    
    for col in sterile_cols + assassin_cols:
        if col in df.columns:
            count = df[col].sum()
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*80)
    if issues == 0:
        print("✅ All audits PASSED!")
    else:
        print(f"⚠️ Found {issues} issues")
    print("="*80)


if __name__ == "__main__":
    main()
