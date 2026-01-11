"""
Trace EXACTLY what calc_cgm_elo.py does step by step for Man Utd
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))

# Load history EXACTLY like calc_cgm_elo.py does
def load_history(path, max_date=None):
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        dt = pd.to_datetime(df["date"] + " " + df.get("time", "").astype(str), errors="coerce")
        dt2 = pd.to_datetime(df["date"], errors="coerce")
        df["datetime"] = dt.fillna(dt2)
    df = df.sort_values("datetime")
    if max_date:
        cutoff = pd.to_datetime(max_date)
        df = df[df["datetime"] <= cutoff]
    return df

def _clean_team_id(val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except:
        pass
    try:
        if isinstance(val, (int, float, np.integer, np.floating)):
            if np.isnan(val):
                return None
            if float(val).is_integer():
                return str(int(val))
            return str(float(val))
    except:
        pass
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s

def infer_team(row, home):
    code = row.get("code_home") if home else row.get("code_away")
    name = row.get("home") if home else row.get("away")
    code_id = _clean_team_id(code)
    if code_id:
        return code_id
    name_id = _clean_team_id(name)
    if name_id:
        return name_id
    return None

# Load like the script does
hist = load_history("data/enhanced/cgm_match_history.csv", max_date="2025-12-26")
print(f"Loaded {len(hist)} rows")

# Simulate Elo exactly
START_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADV = 65.0

def margin_mult(gd):
    gd = abs(int(gd))
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    if gd == 3: return 1.75
    return 1.75 + (gd - 3) / 8.0

ratings = {}
mu_trace = []

for idx, row in hist.iterrows():
    home_id = infer_team(row, True)
    away_id = infer_team(row, False)
    fh, fa = row.get("ft_home"), row.get("ft_away")
    
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
            'elo_before': mu_elo
        })
    
    # Update
    if pd.notna(fh) and pd.notna(fa) and home_id and away_id:
        exp = 1.0 / (1.0 + 10 ** (-((r_home + HOME_ADV) - r_away) / 400.0))
        actual = 1.0 if fh > fa else (0.5 if fh == fa else 0.0)
        gd = int(abs(fh - fa))
        mult = margin_mult(gd)
        delta = K_FACTOR * mult * (actual - exp)
        ratings[home_id] = r_home + delta
        ratings[away_id] = r_away - delta

# Show last 10 Man Utd matches
print("\nLast 10 Man Utd matches (from simulation):")
for t in mu_trace[-10:]:
    venue = "H" if t['is_home'] else "A"
    print(f"{t['date']} {venue} vs {t['opp'][:15].ljust(15)} | Elo before: {t['elo_before']:.1f}")

print(f"\nFinal Man Utd Elo: {ratings.get('1020', 'N/A'):.1f}")
