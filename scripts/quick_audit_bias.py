#!/usr/bin/env python3
"""Quick audit: Check if predictions show market bias."""
import pandas as pd

# Load predictions
preds = pd.read_csv('reports/cgm_upcoming_predictions.csv')
print(f'Total predictions: {len(preds)}')
print()

# Market columns
markets = {
    'Over 2.5': 'EV_over25',
    'Under 2.5': 'EV_under25',
    'BTTS Yes': 'EV_btts_yes',
    'BTTS No': 'EV_btts_no'
}

print('=== EV DISTRIBUTION (positive EV count) ===')
for name, col in markets.items():
    if col in preds.columns:
        positive = (preds[col] > 0).sum()
        strong = (preds[col] > 0.05).sum()
        avg_ev = preds[col].mean() * 100
        print(f'{name:12}: {positive:3} positive | {strong:3} strong (>5%) | avg: {avg_ev:+.1f}%')

print()

# Find best pick per match
def get_best(row):
    ev_vals = {}
    for k, v in markets.items():
        if v in row.index and pd.notna(row[v]):
            ev_vals[k] = row[v]
    if not ev_vals:
        return 'None', 0
    best = max(ev_vals, key=ev_vals.get)
    return best, ev_vals[best]

preds['best_pick'], preds['best_ev'] = zip(*preds.apply(get_best, axis=1))

print('=== BEST PICK DISTRIBUTION (per match) ===')
pick_counts = preds['best_pick'].value_counts()
for pick, count in pick_counts.items():
    pct = count / len(preds) * 100
    print(f'{pick:12}: {count:3} matches ({pct:.1f}%)')

print()
print('=== TOP 10 BY EV ===')
top10 = preds.nlargest(10, 'best_ev')[['home', 'away', 'best_pick', 'best_ev']]
for _, r in top10.iterrows():
    print(f"{r['home']:15} vs {r['away']:15} -> {r['best_pick']:12} (+{r['best_ev']*100:.1f}%)")

print()
print('=== PROBABILITY AVERAGES ===')
print(f"p_over25 avg: {preds['p_over25'].mean()*100:.1f}%")
if 'p_btts_yes' in preds.columns:
    print(f"p_btts_yes avg: {preds['p_btts_yes'].mean()*100:.1f}%")
