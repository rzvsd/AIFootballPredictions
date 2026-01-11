"""Display picks (best value bets)."""
import pandas as pd

df = pd.read_csv('reports/picks_explained.csv')

print("="*90)
print("TOP PICKS (Best Value Bets) - 27 Total")
print("="*90)

for _, r in df.iterrows():
    date = r['fixture_datetime'][:10]
    print(f"\n{date} | {r['league']}")
    print(f"  {r['home']} vs {r['away']}")
    print(f"  >>> {r['pick_text']}")
