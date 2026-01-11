"""Generate prediction table with Faith, O/U 2.5, and BTTS."""
import pandas as pd

df = pd.read_csv('reports/cgm_upcoming_predictions.csv')
df['date'] = pd.to_datetime(df['fixture_datetime']).dt.strftime('%Y-%m-%d')

# Calculate faith/confidence (1-5 based on evidence)
def calc_faith(row):
    neff = (row.get('neff_H', 0) or 0) + (row.get('neff_A', 0) or 0)
    if neff >= 20: return 5
    elif neff >= 15: return 4
    elif neff >= 10: return 3
    elif neff >= 5: return 2
    return 1

df['faith'] = df.apply(calc_faith, axis=1)

for date in sorted(df['date'].unique())[:5]:  # First 5 dates
    print(f"\n{'='*100}")
    print(f"  {date}")
    print(f"{'='*100}")
    print(f"{'Match':<40} {'Faith':^7} {'O/U 2.5':^12} {'BTTS':^10}")
    print("-"*100)
    
    dfd = df[df['date'] == date].sort_values(['league', 'home'])
    
    for _, r in dfd.iterrows():
        match = f"{r['home'][:18]} vs {r['away'][:18]}"
        
        # Faith
        faith = int(r['faith'])
        faith_str = f"{'★'*faith}{'☆'*(5-faith)}"
        
        # O/U 2.5
        p_over = r.get('p_over25', 0.5)
        p_over = p_over * 100 if p_over < 1 else p_over
        pred_ou = f"O:{p_over:.0f} U:{100-p_over:.0f}"
        
        # BTTS
        p_btts = r.get('p_btts_yes', 0.5)
        p_btts = p_btts * 100 if p_btts and p_btts < 1 else (p_btts if p_btts else 50)
        pred_btts = f"Y:{p_btts:.0f} N:{100-p_btts:.0f}"
        
        print(f"{match:<40} {faith_str:^7} {pred_ou:^12} {pred_btts:^10}")

print(f"\n{'='*100}")
print(f"SUMMARY: {len(df)} predictions | {df['league'].nunique()} leagues | {df['date'].nunique()} dates")
print(f"{'='*100}")
