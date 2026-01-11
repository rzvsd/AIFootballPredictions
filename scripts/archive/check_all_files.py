"""
Check Elo values for Man Utd across all intermediate files
"""
import pandas as pd

files = [
    ("cgm_match_history.csv", "data/enhanced/cgm_match_history.csv"),
    ("cgm_match_history_with_elo.csv", "data/enhanced/cgm_match_history_with_elo.csv"),
    ("cgm_match_history_with_elo_stats.csv", "data/enhanced/cgm_match_history_with_elo_stats.csv"),
    ("cgm_match_history_with_elo_stats_xg.csv", "data/enhanced/cgm_match_history_with_elo_stats_xg.csv"),
]

# Check Nov 01 Nottingham vs Man Utd match in each file
target_date = "2025-11-01"
target_home = "Nottingham"
target_away = "Man Utd"

print("Elo values for Nottingham vs Man Utd (2025-11-01) across files:")
print("="*80)

for name, path in files:
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        match = df[(df['date'].astype(str).str.contains('2025-11-01')) & 
                   (df['home'].str.contains('Notting', na=False)) & 
                   (df['away'] == 'Man Utd')]
        
        if len(match) > 0:
            row = match.iloc[0]
            elo_cols = [c for c in df.columns if 'elo' in c.lower()]
            print(f"\n{name}:")
            for col in sorted(elo_cols):
                val = row.get(col, "N/A")
                if pd.notna(val):
                    print(f"  {col}: {val}")
        else:
            print(f"\n{name}: Match not found")
    except Exception as e:
        print(f"\n{name}: Error - {e}")
