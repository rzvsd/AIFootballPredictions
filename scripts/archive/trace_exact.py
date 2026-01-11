"""
Run calc_cgm_elo logic EXACTLY and trace Man Utd step by step
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))

# Import the EXACT functions from calc_cgm_elo.py
from scripts.calc_cgm_elo import load_history, infer_team, margin_multiplier, expected_home

START_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADV = 65.0

# Load history exactly like the script does
hist = load_history(Path("data/enhanced/cgm_match_history.csv"), max_date="2025-12-26")
print(f"Loaded {len(hist)} rows")

# Track Elo progression
ratings = {}
mu_trace = []

for idx, row in hist.iterrows():
    home_id = infer_team(row, True)
    away_id = infer_team(row, False)
    
    fh_raw = row.get("ft_home")
    fa_raw = row.get("ft_away")
    try:
        fh = float(fh_raw)
    except:
        fh = np.nan
    try:
        fa = float(fa_raw)
    except:
        fa = np.nan

    # Get current ratings
    r_home = ratings.get(home_id, START_ELO) if home_id else START_ELO
    r_away = ratings.get(away_id, START_ELO) if away_id else START_ELO

    # Track Man Utd (1020)
    if home_id == "1020" or away_id == "1020":
        is_home = (home_id == "1020")
        mu_elo = r_home if is_home else r_away
        mu_trace.append({
            'idx': idx,
            'date': str(row.get('datetime', '?'))[:10],
            'is_home': is_home,
            'opp': row['away'] if is_home else row['home'],
            'elo_before': mu_elo,
            'r_home': r_home,
            'r_away': r_away,
            'home_id': home_id,
            'away_id': away_id,
            'fh': fh,
            'fa': fa
        })

    # Update ratings if result exists
    if pd.isna(fh) or pd.isna(fa) or home_id is None or away_id is None:
        continue

    exp = expected_home(r_home, r_away, HOME_ADV)
    if fh > fa:
        actual = 1.0
    elif fh == fa:
        actual = 0.5
    else:
        actual = 0.0

    gd = int(abs(fh - fa))
    mult = margin_multiplier(gd)
    delta = K_FACTOR * mult * (actual - exp)

    ratings[home_id] = r_home + delta
    ratings[away_id] = r_away - delta
    
    # Update trace for Man Utd
    if home_id == "1020" or away_id == "1020":
        mu_trace[-1]['delta'] = delta if is_home else -delta
        mu_trace[-1]['elo_after'] = ratings.get("1020")

# Show Oct-Nov 2025 matches
print("\nMan Utd Oct-Nov 2025 (using EXACT calc_cgm_elo functions):")
print("-"*100)
for t in mu_trace:
    if '2025-10' in t['date'] or '2025-11' in t['date']:
        venue = "H" if t['is_home'] else "A"
        delta = t.get('delta', 0)
        after = t.get('elo_after', t['elo_before'])
        score = f"{int(t['fh'])}-{int(t['fa'])}" if pd.notna(t['fh']) else "?-?"
        print(f"{t['date']} {venue} vs {t['opp'][:12].ljust(12)} | {score} | Before: {t['elo_before']:.1f} -> After: {after:.1f} ({delta:+.1f})")

print(f"\nFinal Man Utd Elo: {ratings.get('1020', 'N/A'):.1f}")
