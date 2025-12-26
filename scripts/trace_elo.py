import pandas as pd
import numpy as np

df = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date')

# Simulate Elo calculation for Man Utd (code 1020)
start_elo = 1500.0
k_factor = 20.0
home_adv = 65.0

def margin_mult(gd):
    gd = abs(int(gd))
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    if gd == 3: return 1.75
    return 1.75 + (gd - 3) / 8.0

ratings = {}
mu_elo_history = []

for _, row in df.iterrows():
    # Get team IDs (prefer code, fallback to name)
    code_home = row.get('code_home')
    code_away = row.get('code_away')
    
    if pd.notna(code_home):
        home_id = str(int(code_home)) if isinstance(code_home, (int, float)) else str(code_home)
    else:
        home_id = str(row.get('home', ''))
    
    if pd.notna(code_away):
        away_id = str(int(code_away)) if isinstance(code_away, (int, float)) else str(code_away)
    else:
        away_id = str(row.get('away', ''))
    
    fh, fa = row.get('ft_home'), row.get('ft_away')
    
    # Get current ratings
    r_home = ratings.get(home_id, start_elo)
    r_away = ratings.get(away_id, start_elo)
    
    # Track Man Utd (code 1020)
    if home_id == '1020' or away_id == '1020':
        is_home = (home_id == '1020')
        mu_elo = r_home if is_home else r_away
        date = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '?'
        opp_name = row['away'] if is_home else row['home']
        venue = 'H' if is_home else 'A'
        
        if pd.notna(fh) and pd.notna(fa):
            mu_goals = int(fh) if is_home else int(fa)
            opp_goals = int(fa) if is_home else int(fh)
            score = f"{mu_goals}-{opp_goals}"
            result = 'W' if mu_goals > opp_goals else ('L' if mu_goals < opp_goals else 'D')
        else:
            score = '?-?'
            result = '?'
        
        mu_elo_history.append({
            'date': date,
            'venue': venue,
            'opp': opp_name,
            'score': score,
            'result': result,
            'elo_before': mu_elo
        })
    
    # Update ratings if result exists
    if pd.notna(fh) and pd.notna(fa) and home_id and away_id:
        exp = 1.0 / (1.0 + 10 ** (-((r_home + home_adv) - r_away) / 400.0))
        actual = 1.0 if fh > fa else (0.5 if fh == fa else 0.0)
        gd = int(abs(fh - fa))
        mult = margin_mult(gd)
        delta = k_factor * mult * (actual - exp)
        
        ratings[home_id] = r_home + delta
        ratings[away_id] = r_away - delta
        
        # Log update for Man Utd
        if home_id == '1020' or away_id == '1020':
            mu_elo_history[-1]['delta'] = delta if home_id == '1020' else -delta
            mu_elo_history[-1]['elo_after'] = ratings['1020']

# Print last matches
print('Man Utd Elo Progression (simulated from scratch):')
print('='*80)
for h in mu_elo_history[-20:]:
    delta_str = f"{h.get('delta', 0):+.1f}" if 'delta' in h else ''
    after = h.get('elo_after', h['elo_before'])
    print(f"{h['date']} | {h['venue']} vs {h['opp'][:12].ljust(12)} | {h['score']} {h['result']} | Elo: {h['elo_before']:.1f} -> {after:.1f} ({delta_str})")
