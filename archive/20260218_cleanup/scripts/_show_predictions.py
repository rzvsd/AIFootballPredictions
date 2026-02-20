"""Display predictions by date and league."""
import pandas as pd

df = pd.read_csv('reports/cgm_upcoming_predictions.csv')
df['date'] = pd.to_datetime(df['fixture_datetime']).dt.date

print("="*90)
print("PREDICTIONS BY DATE AND LEAGUE (within 14-day horizon)")
print("="*90)

for date in sorted(df['date'].unique()):
    print(f"\n### {date}")
    print("-"*90)
    dfd = df[df['date'] == date].sort_values(['league', 'home'])
    for _, r in dfd.iterrows():
        p_over = r.get("p_over25", 0.5)
        p_over = p_over * 100 if p_over < 1 else p_over
        p_under = 100 - p_over
        p_btts = r.get("p_btts_yes", 0.5)
        if pd.isna(p_btts):
            p_btts = 0.5
        p_btts = p_btts * 100 if p_btts < 1 else p_btts
        p_btts_no = 100 - p_btts
        print(
            f"  {r['league']:15} | {r['home']:22} vs {r['away']:22} | "
            f"O:{p_over:>4.0f}% U:{p_under:>4.0f}% | BTTS Y:{p_btts:>4.0f}% N:{p_btts_no:>4.0f}%"
        )

print("\n" + "="*90)
print(f"TOTAL: {len(df)} predictions across {df['league'].nunique()} leagues")
print("="*90)
