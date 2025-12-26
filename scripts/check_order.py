"""
Check exact match ordering in cgm_match_history.csv for Oct-Nov 2025
"""
import pandas as pd

df = pd.read_csv('data/enhanced/cgm_match_history.csv')

# Add datetime exactly like calc_cgm_elo.py does
if "datetime" not in df.columns:
    dt = pd.to_datetime(df["date"] + " " + df.get("time", "").astype(str), errors="coerce")
    dt2 = pd.to_datetime(df["date"], errors="coerce")
    df["datetime"] = dt.fillna(dt2)

df = df.sort_values("datetime")
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Get Man Utd matches in Oct-Nov 2025
mu = df[(df['date'] >= '2025-10-15') & (df['date'] <= '2025-11-10') & 
        ((df['home'] == 'Man Utd') | (df['away'] == 'Man Utd'))]

print("Man Utd matches Oct-Nov 2025 (in sorted order):")
print("-"*80)
for idx, row in mu.iterrows():
    date_str = str(row['datetime'])[:10]
    time_str = row.get('time', '?')
    is_home = row['home'] == 'Man Utd'
    opp = row['away'] if is_home else row['home']
    venue = 'H' if is_home else 'A'
    code_mu = row['code_home'] if is_home else row['code_away']
    print(f"idx={idx:4d} | {date_str} {time_str} | {venue} vs {opp[:12].ljust(12)} | code={int(code_mu)}")

# Check matches BETWEEN Man Utd games on Oct 25 and Nov 01
print("\n\nAll matches between Oct 25 18:00 and Nov 01 (showing row order):")
print("-"*80)
between = df[(df['datetime'] > '2025-10-25 18:00') & (df['datetime'] <= '2025-11-01 23:59')]
for i, (idx, row) in enumerate(between.iterrows()):
    date_str = str(row['datetime'])[:16]
    home = row['home'][:12].ljust(12)
    away = row['away'][:12].ljust(12)
    print(f"{i+1:3d}. idx={idx:4d} | {date_str} | {home} vs {away}")
