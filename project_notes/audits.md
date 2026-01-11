# Feature Audits

This document tracks verification tests for core features to ensure calculation correctness and prevent regressions.

## Audit Scripts Location
`scripts/audit_*.py`

---

## Milestone 1: Elo Audit

### Script
`scripts/investigate_elo.py`

### What It Verifies
1. **Elo Calculation Consistency** - Simulates Elo from scratch and compares to stored values
2. **Team Identity Tracking** - Verifies code-based team IDs are consistent across matches
3. **Index Alignment** - Ensures pandas Series assignment doesn't misalign values after sorting

### How to Run
```bash
python scripts/investigate_elo.py
```

### Expected Output
```
Total matches: 283, Mismatches: 0
✅ All Elo values match correctly!
```

### Bug Found & Fixed (2025-12-26)
- **Issue**: `compute_elo_series()` returned Series without indices
- **Impact**: 34 mismatches for Man Utd, affected all 30 teams
- **Fix**: Added `index=df.index` to returned Series in `calc_cgm_elo.py`

---

## Milestone 2: Pressure Form Audit

### Script
`scripts/audit_pressure_form.py`

### What It Verifies
1. **Value Ranges** - `press_form_H/A` within [0, 1]
2. **Evidence Counts** - `press_n_H/A` within L10 window
3. **No NaN Values** - Data completeness check
4. **Divergence Features** - `div_team_*`, `div_diff` have reasonable z-score distributions
5. **Sterile/Assassin Flags** - Count and consistency

### How to Run
```bash
python scripts/audit_pressure_form.py
```

### Expected Output (2025-12-26)
```
press_form_H: min=0.373, max=0.669
press_form_A: min=0.335, max=0.582
press_n_H: min=0, max=10
press_n_A: min=0, max=10
✅ All audits PASSED!
```

---

## Milestone 3: xG Form Audit

### Script
`scripts/audit_xg_form.py`

### What It Verifies
1. **xG Form Ranges** - Non-negative, reasonable max values
2. **Evidence Counts** - `xg_stats_n_*` within L10 window
3. **Decay Features** - Present and correctly named
4. **Z-Score Features** - Reasonable distributions
5. **No NaN Values** - Data completeness

### How to Run
```bash
python scripts/audit_xg_form.py
```

### Expected Output (2025-12-26)
```
xg_for_form_H: min=0.00, max=2.48, mean=0.08
xg_stats_n_H: min=0, max=8
xg_z_H: min=-2.84, max=2.88, std=0.26
✅ xG Form Audit PASSED!
```

---

## Running All Audits

To run all verification scripts:
```bash
python scripts/investigate_elo.py
python scripts/audit_pressure_form.py
python scripts/audit_xg_form.py
```

All should report PASSED for a healthy pipeline.

---

## Milestone 9: Time Decay Audit

### Script
`scripts/audit_decay.py`

### What It Verifies
1. **Decay Features Present** - All 6 decay features exist
2. **Decay vs Non-Decay Difference** - Decay features differ from base features
3. **Value Ranges** - Pressure decay in [0,1], xG decay reasonable
4. **Weight Formula** - Exponential decay with correct half-life

### Expected Output (2025-12-26)
```
Decay features present: 6/6 ✓
Weight formula: exp(-0.693 * age / half_life)
With half_life=5, weight halves every 5 matches
✅ Time Decay Audit PASSED!
```

---

## Milestone 10: H2H History Audit

### Script
`scripts/audit_h2h.py`

### What It Verifies
1. **H2H Features Present** - All 6 features exist
2. **Value Ranges** - Match counts non-negative, rates in [0,1]
3. **Usability Stats** - 64.2% of fixtures have usable H2H data
4. **Rivalry Counts** - Major rivalries have expected meeting counts

### Expected Output (2025-12-26)
```
h2h_matches: min=0, max=10
Usable H2H: 1816 (64.2%)
Man Utd vs Liverpool: 15 meetings
✅ H2H History Audit PASSED!
```

---

## Milestone 11: League Features Audit

### Script
`scripts/audit_league_features.py`

### What It Verifies
1. **League Features Present** - All 7 features exist
2. **Value Ranges** - Goals 2.3-3.6, rates in [0,1]
3. **Data Completeness** - 98.2% usable profiles
4. **League Variation** - Different leagues have different profiles

### Expected Output (2025-12-26)
```
lg_goals_per_match: min=2.31, max=3.61, mean=2.87
Usable profiles: 2780 (98.2%)
Sufficient variation detected ✓
✅ League Features Audit PASSED!
```

---

## Milestone 12: Backtest Leakage Audit

### Script
`scripts/audit_backtest.py`

### What It Verifies
1. **Required Columns** - All backtest columns present
2. **Date Range** - Reasonable prediction period
3. **Prediction Coverage** - Predictions exist for results
4. **No Leakage** - Accuracy in normal range (47.7%, not 90%+)
5. **Walk-Forward** - Multiple unique run timestamps

### Expected Output (2025-12-26)
```
Date range: 2025-09-13 to 2025-12-22
O/U 2.5 accuracy (50% threshold): 47.7%
Unique run timestamps: 38 (walk-forward)
✅ Backtest Leakage Audit PASSED!
```

---

## Full Audit Command

Run all audits with one command (10 scripts, validates all features):
```bash
python scripts/run_all_audits.py
```

Or run individual audits:
```bash
python scripts/audit_multi_league.py          # NEW: Multi-league coverage
python scripts/investigate_elo.py
python scripts/audit_pressure_form.py
python scripts/audit_xg_form.py
python scripts/audit_decay.py
python scripts/audit_h2h.py
python scripts/audit_league_features.py
python scripts/audit_backtest.py
(legacy) python archive/legacy_full_engine/audit_picks.py
python -m scripts.audit_picks_goals
python -m scripts.audit_upcoming_feed --as-of-date YYYY-MM-DD
python -m scripts.audit_narrator
```

---

## Milestone 15: Multi-League Audit (Dec 2025)

### Script
`scripts/audit_multi_league.py`

### What It Verifies
1. **History Coverage** - Matches per league, date ranges
2. **Source Data** - Future fixtures per league from allratingv.csv
3. **Predictions Generated** - Predictions coverage per league
4. **Coverage Analysis** - Identifies missing leagues or low coverage

### How to Run
```bash
python scripts/audit_multi_league.py
```

### Expected Output (2025-12-27)
```
12 leagues supported
164 predictions across 10 active leagues
Liga 1 and Super Lig on winter break (next fixtures Jan 16-18)
[OK] All leagues with future fixtures have predictions
```

---

## Milestone 16: Prediction Report (Dec 2025)

### Script
`scripts/generate_predictions_report.py`

### What It Does
Generates formatted prediction tables with O/U 2.5, BTTS probabilities and EV.

### How to Run
```bash
python scripts/generate_predictions_report.py
python scripts/generate_predictions_report.py --league "Premier L"
python scripts/generate_predictions_report.py --date 2025-12-28
```

### Output
- `reports/predictions_report.md` - Markdown table
- `reports/predictions_report.txt` - Plain text table
- Console output with summary
