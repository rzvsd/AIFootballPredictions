import pandas as pd

# Load history with Elo
df = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Filter to EPL 2025-2026 season
df = df[(df['league'] == 'Premier L') & (df['date'] >= '2025-08-01')].copy()
df = df.sort_values('date')

# Find Man Utd matches
mu = df[(df['home'] == 'Man Utd') | (df['away'] == 'Man Utd')].copy()

# Add result info
def get_mu_result(row):
    is_home = row['home'] == 'Man Utd'
    if is_home:
        if row['ft_home'] > row['ft_away']: return 'W'
        elif row['ft_home'] < row['ft_away']: return 'L'
        else: return 'D'
    else:
        if row['ft_away'] > row['ft_home']: return 'W'
        elif row['ft_away'] < row['ft_home']: return 'L'
        else: return 'D'

def get_mu_elo(row):
    is_home = row['home'] == 'Man Utd'
    if is_home:
        return row.get('elo_home_calc', row.get('elo_H', row.get('elo_home', None)))
    else:
        return row.get('elo_away_calc', row.get('elo_A', row.get('elo_away', None)))

def get_opponent(row):
    return row['away'] if row['home'] == 'Man Utd' else row['home']

def get_score(row):
    is_home = row['home'] == 'Man Utd'
    if is_home:
        return f"{int(row['ft_home'])}-{int(row['ft_away'])}"
    else:
        return f"{int(row['ft_away'])}-{int(row['ft_home'])}"

mu['result'] = mu.apply(get_mu_result, axis=1)
mu['mu_elo'] = mu.apply(get_mu_elo, axis=1)
mu['opponent'] = mu.apply(get_opponent, axis=1)
mu['score'] = mu.apply(get_score, axis=1)
mu['venue'] = mu.apply(lambda r: 'H' if r['home']=='Man Utd' else 'A', axis=1)

# Calculate Elo change
mu['elo_change'] = mu['mu_elo'].diff()

print('Man Utd Elo Progression - EPL 2025-2026')
print('='*70)
for i, (_, row) in enumerate(mu.iterrows()):
    date = row['date'].strftime('%Y-%m-%d')
    opp = row['opponent'][:12].ljust(12)
    score = row['score'].ljust(5)
    venue = row['venue']
    result = row['result']
    elo = row['mu_elo']
    change = row['elo_change']
    
    if pd.notna(elo):
        change_str = f"{change:+.0f}" if pd.notna(change) and i > 0 else ""
        print(f"{date} | {venue} {opp} {score} {result} | Elo: {elo:.0f} {change_str}")
    else:
        print(f"{date} | {venue} {opp} {score} {result} | Elo: N/A")
