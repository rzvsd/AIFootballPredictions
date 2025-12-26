"""
Milestone 3: xG Form Audit

Verifies xG form features are within expected ranges.
Since xG proxy columns are dropped from training to prevent leakage,
this audit focuses on verifying form values and evidence counts.
"""

import pandas as pd
import numpy as np
from pathlib import Path

WINDOW = 10  # Rolling window size


def main():
    print("="*80)
    print("xG FORM AUDIT - Milestone 3")
    print("="*80)
    
    # Load training data (contains xG features)
    df = pd.read_csv('data/enhanced/frankenstein_training.csv')
    print(f"Loaded {len(df)} training rows")
    
    # Check required columns
    xg_form_cols = ['xg_for_form_H', 'xg_for_form_A', 'xg_against_form_H', 'xg_against_form_A',
                    'xg_stats_n_H', 'xg_stats_n_A']
    missing = [c for c in xg_form_cols if c not in df.columns]
    if missing:
        print(f"Missing xG form columns: {missing}")
        return
    
    print("All required xG form columns present ✓")
    
    issues = 0
    
    # Audit 1: Value Range Checks
    print("\n--- Audit 1: xG Form Value Ranges ---")
    for col in ['xg_for_form_H', 'xg_for_form_A', 'xg_against_form_H', 'xg_against_form_A']:
        min_v, max_v, mean_v = df[col].min(), df[col].max(), df[col].mean()
        print(f"  {col}: min={min_v:.2f}, max={max_v:.2f}, mean={mean_v:.2f}")
        if min_v < 0:
            print(f"    ⚠️ Negative xG values!")
            issues += 1
        if max_v > 5:
            print(f"    ⚠️ Unusually high xG values!")
            issues += 1
    
    # Audit 2: Evidence Count Check
    print("\n--- Audit 2: Evidence Counts ---")
    for col in ['xg_stats_n_H', 'xg_stats_n_A']:
        min_v, max_v = df[col].min(), df[col].max()
        print(f"  {col}: min={min_v:.0f}, max={max_v:.0f}")
        if max_v > WINDOW:
            print(f"    ⚠️ Exceeds L{WINDOW} window!")
            issues += 1
    
    # Audit 3: Decay Features Present
    print("\n--- Audit 3: Decay Features ---")
    decay_cols = [c for c in df.columns if 'xg_' in c and '_decay' in c]
    print(f"  Found {len(decay_cols)} xG decay columns")
    for col in decay_cols[:5]:
        print(f"    - {col}")
    
    # Audit 4: xG Diff Form
    print("\n--- Audit 4: xG Diff Form ---")
    for col in ['xg_diff_form_H', 'xg_diff_form_A']:
        if col in df.columns:
            min_v, max_v = df[col].min(), df[col].max()
            print(f"  {col}: min={min_v:.2f}, max={max_v:.2f}")
        else:
            print(f"  {col}: NOT FOUND")
    
    # Audit 5: Data Completeness
    print("\n--- Audit 5: Data Completeness ---")
    for col in xg_form_cols:
        nan_pct = df[col].isna().sum() / len(df) * 100
        print(f"  {col}: {nan_pct:.1f}% NaN")
    
    # Audit 6: Z-scores
    print("\n--- Audit 6: Z-Score Features ---")
    for col in ['xg_z_H', 'xg_z_A']:
        if col in df.columns:
            min_v, max_v, std_v = df[col].min(), df[col].max(), df[col].std()
            print(f"  {col}: min={min_v:.2f}, max={max_v:.2f}, std={std_v:.2f}")
        else:
            print(f"  {col}: NOT FOUND")
    
    print("\n" + "="*80)
    if issues == 0:
        print("✅ xG Form Audit PASSED!")
    else:
        print(f"⚠️ Found {issues} issues")
    print("="*80)


if __name__ == "__main__":
    main()
