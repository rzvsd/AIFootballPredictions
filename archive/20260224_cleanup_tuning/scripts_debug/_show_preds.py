import pandas as pd

df = pd.read_csv('reports/cgm_upcoming_predictions.csv')

print("=== NEW PREDICTIONS (Poisson Formula) ===")
for _, r in df.iterrows():
    total = r['mu_total']
    over_pct = r['p_over25'] * 100
    home = str(r.get('home', ''))
    away = str(r.get('away', ''))
    print(f"  {home:>25s} vs {away:<25s} mu_total={total:.2f}  Over2.5={over_pct:.0f}%")

print()
mean_mu = df['mu_total'].mean()
print(f"  Mean mu_total: {mean_mu:.2f}")
print(f"  Actual Bundesliga avg: 3.19 goals/game")
print(f"  Gap from reality: {abs(mean_mu - 3.19):.2f} goals")
print(f"  Old XGBoost mu_total avg: 2.16")
improvement = abs(3.19 - 2.16) - abs(3.19 - mean_mu)
print(f"  Improvement: {improvement:.2f} goals closer to reality")
