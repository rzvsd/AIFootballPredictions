"""
Systematic Elo Investigation - Compare calculated vs stored Elo for Man Utd
"""
import pandas as pd
import numpy as np

# Load the history file
df = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)

print("="*100)
print("SYSTEMATIC ELO INVESTIGATION - Man Utd (code 1020)")
print("="*100)

# Parameters matching calc_cgm_elo.py
START_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADV = 65.0

def margin_mult(gd):
    gd = abs(int(gd))
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    if gd == 3: return 1.75
    return 1.75 + (gd - 3) / 8.0

# Simulate Elo from scratch
ratings = {}

# Track all Man Utd matches
mu_analysis = []

for idx, row in df.iterrows():
    # Get team IDs (prefer code, fallback to name)
    code_home = row.get('code_home')
    code_away = row.get('code_away')
    
    if pd.notna(code_home):
        home_id = str(int(code_home)) if isinstance(code_home, (int, float)) and not np.isnan(code_home) else str(code_home)
    else:
        home_id = str(row.get('home', ''))
    
    if pd.notna(code_away):
        away_id = str(int(code_away)) if isinstance(code_away, (int, float)) and not np.isnan(code_away) else str(code_away)
    else:
        away_id = str(row.get('away', ''))
    
    fh, fa = row.get('ft_home'), row.get('ft_away')
    
    # Get CURRENT ratings (before this match)
    r_home_before = ratings.get(home_id, START_ELO)
    r_away_before = ratings.get(away_id, START_ELO)
    
    # Get STORED ratings from the file
    stored_elo_home = row.get('elo_home_calc', np.nan)
    stored_elo_away = row.get('elo_away_calc', np.nan)
    
    # Is this a Man Utd match?
    is_mu_home = (home_id == '1020')
    is_mu_away = (away_id == '1020')
    
    if is_mu_home or is_mu_away:
        is_home = is_mu_home
        mu_calc_elo = r_home_before if is_home else r_away_before
        mu_stored_elo = stored_elo_home if is_home else stored_elo_away
        opp_name = row['away'] if is_home else row['home']
        
        if pd.notna(fh) and pd.notna(fa):
            mu_goals = int(fh) if is_home else int(fa)
            opp_goals = int(fa) if is_home else int(fh)
            score = f"{mu_goals}-{opp_goals}"
            result = 'W' if mu_goals > opp_goals else ('L' if mu_goals < opp_goals else 'D')
        else:
            score = '?-?'
            result = '?'
        
        diff = mu_stored_elo - mu_calc_elo if pd.notna(mu_stored_elo) else float('nan')
        
        mu_analysis.append({
            'idx': idx,
            'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '?',
            'venue': 'H' if is_home else 'A',
            'opp': opp_name,
            'score': score,
            'result': result,
            'calc_elo': mu_calc_elo,
            'stored_elo': mu_stored_elo,
            'diff': diff,
            'mismatch': abs(diff) > 1 if pd.notna(diff) else True
        })
    
    # Update ratings if result exists (this is what calc_cgm_elo.py does)
    if pd.notna(fh) and pd.notna(fa) and home_id and away_id:
        exp = 1.0 / (1.0 + 10 ** (-((r_home_before + HOME_ADV) - r_away_before) / 400.0))
        actual = 1.0 if fh > fa else (0.5 if fh == fa else 0.0)
        gd = int(abs(fh - fa))
        mult = margin_mult(gd)
        delta = K_FACTOR * mult * (actual - exp)
        
        ratings[home_id] = r_home_before + delta
        ratings[away_id] = r_away_before - delta

# Print analysis
print("\n")
print(f"{'Match':<4} {'Date':<12} {'V':<2} {'Opponent':<14} {'Score':<6} {'R':<2} {'Calc Elo':>10} {'Stored':>10} {'Diff':>8} {'OK?'}")
print("-"*100)

mismatches = 0
for i, m in enumerate(mu_analysis):
    ok = "✓" if not m['mismatch'] else "❌"
    if m['mismatch']:
        mismatches += 1
    print(f"{i+1:<4} {m['date']:<12} {m['venue']:<2} {m['opp'][:14]:<14} {m['score']:<6} {m['result']:<2} {m['calc_elo']:>10.1f} {m['stored_elo']:>10.1f} {m['diff']:>+8.1f} {ok}")

print("-"*100)
print(f"Total matches: {len(mu_analysis)}, Mismatches: {mismatches}")

if mismatches > 0:
    print("\n⚠️  MISMATCHES DETECTED - Stored Elo values don't match calculated!")
    print("First mismatch:")
    for m in mu_analysis:
        if m['mismatch']:
            print(f"  Match {m['idx']}: {m['date']} vs {m['opp']} - Calc: {m['calc_elo']:.1f}, Stored: {m['stored_elo']:.1f}")
            break
else:
    print("\n✅ All Elo values match correctly!")
