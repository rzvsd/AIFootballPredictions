import pandas as pd
df = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f'History rows: {len(df)}')
print(f'Date range: {df.datetime.min()} to {df.datetime.max()}')
print(f'Leagues: {list(df["league"].unique()) if "league" in df.columns else "N/A"}')
print(f'Seasons: {list(df["season"].unique()) if "season" in df.columns else "N/A"}')
early = df[df.datetime < '2025-09-01']
print(f'Matches before Sep 2025: {len(early)}')
print(f'Has odds columns: {[c for c in df.columns if "odds" in c.lower() or "cota" in c.lower()]}')
