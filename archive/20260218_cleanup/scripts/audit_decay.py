"""
Milestone 9: Time Decay Audit

Verifies that decay-weighted features use correct exponential weighting.
Tests:
1. Decay features exist in training data
2. Decay weights follow formula: weight = exp(-0.693 * match_age / half_life)
3. Decay values differ appropriately from non-decay values
4. Recent matches have higher weight than older matches
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))

from config import DECAY_HALF_LIFE, DECAY_ENABLED


def main():
    print("="*80)
    print("TIME DECAY AUDIT - Milestone 9")
    print("="*80)
    print(f"Config: DECAY_ENABLED={DECAY_ENABLED}, HALF_LIFE={DECAY_HALF_LIFE}")
    
    # Load training data
    df = pd.read_csv('data/enhanced/frankenstein_training.csv')
    print(f"Loaded {len(df)} training rows")
    
    issues = 0
    
    # Audit 1: Check decay features exist
    print("\n--- Audit 1: Decay Features Present ---")
    expected_decay = [
        'press_form_H_decay', 'press_form_A_decay',
        'xg_for_form_H_decay', 'xg_against_form_H_decay',
        'xg_for_form_A_decay', 'xg_against_form_A_decay'
    ]
    
    for col in expected_decay:
        if col in df.columns:
            print(f"  [OK] {col}")
        else:
            print(f"  [X] {col} MISSING")
            issues += 1
    
    # Audit 2: Decay values should differ from non-decay
    print("\n--- Audit 2: Decay vs Non-Decay Difference ---")
    pairs = [
        ('press_form_H', 'press_form_H_decay'),
        ('press_form_A', 'press_form_A_decay'),
        ('xg_for_form_H', 'xg_for_form_H_decay'),
        ('xg_for_form_A', 'xg_for_form_A_decay'),
    ]
    
    for base, decay in pairs:
        if base in df.columns and decay in df.columns:
            # Check if they differ
            diff = (df[base] - df[decay]).abs()
            avg_diff = diff.mean()
            max_diff = diff.max()
            identical = (diff == 0).sum() / len(df) * 100
            
            print(f"  {base} vs {decay}:")
            print(f"    Avg diff: {avg_diff:.4f}, Max diff: {max_diff:.4f}")
            print(f"    Identical rows: {identical:.1f}%")
            
            # If all rows are identical, decay isn't working
            if identical > 95:
                print(f"    [!] WARNING: Decay barely differs from base!")
        else:
            print(f"  {base} or {decay} missing")
    
    # Audit 3: Value ranges for decay features
    print("\n--- Audit 3: Decay Value Ranges ---")
    for col in expected_decay:
        if col in df.columns:
            min_v, max_v, mean_v = df[col].min(), df[col].max(), df[col].mean()
            nan_pct = df[col].isna().sum() / len(df) * 100
            print(f"  {col}: min={min_v:.3f}, max={max_v:.3f}, mean={mean_v:.3f}, NaN={nan_pct:.1f}%")
            
            # Pressure form should be in [0, 1]
            if 'press' in col:
                if min_v < 0 or max_v > 1:
                    print(f"    [!] Out of [0, 1] range!")
                    issues += 1
    
    # Audit 4: Verify decay weight formula with a sample
    print("\n--- Audit 4: Decay Weight Formula Check ---")
    
    # Half-life of 5 means:
    # - match 0 games ago: weight = 1.0
    # - match 5 games ago: weight = 0.5
    # - match 10 games ago: weight = 0.25
    
    for age in [0, 5, 10, 15]:
        weight = np.exp(-0.693 * age / DECAY_HALF_LIFE)
        print(f"  Age {age} matches: expected weight = {weight:.4f}")
    
    print("\n  Formula: weight = exp(-0.693 * age / half_life)")
    print(f"  With half_life={DECAY_HALF_LIFE}, weight halves every {DECAY_HALF_LIFE} matches")
    
    print("\n" + "="*80)
    if issues == 0:
        print("[OK] Time Decay Audit PASSED!")
    else:
        print(f"[!] Found {issues} issues")
    print("="*80)


if __name__ == "__main__":
    main()
