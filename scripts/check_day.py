import pandas as pd

df = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Check 2025-11-01 - the day with the Nottingham mismatch
day = df[(df['date'] >= '2025-11-01') & (df['date'] < '2025-11-02')]
day = day.sort_values('date')

print(f'Matches on 2025-11-01: {len(day)}')
print()
print('Row order in the CSV (determines Elo calc order):')
for i, (idx, row) in enumerate(day.iterrows()):
    home = str(row.get('home', '?'))[:12].ljust(12)
    away = str(row.get('away', '?'))[:12].ljust(12)
    fh, fa = row['ft_home'], row['ft_away']
    score = f"{int(fh)}-{int(fa)}" if pd.notna(fh) else "?-?"
    elo_h = row['elo_home_calc']
    elo_a = row['elo_away_calc']
    print(f"{i+1}. idx={idx} {home} vs {away} | {score} | elo_home={elo_h:.1f} elo_away={elo_a:.1f}")

# Check if Man Utd Elo changed during the day
mu_games = day[(day['home'] == 'Man Utd') | (day['away'] == 'Man Utd')]
print()
print("Man Utd game on this day:")
for _, row in mu_games.iterrows():
    is_home = row['home'] == 'Man Utd'
    print(f"  {'HOME' if is_home else 'AWAY'} | Elo: {row['elo_home_calc'] if is_home else row['elo_away_calc']:.1f}")
