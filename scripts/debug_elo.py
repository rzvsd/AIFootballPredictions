import pandas as pd
from xgb_trainer import _compute_ewma_elo_prematch
import config

LEAGUE = 'E0'
FILE = f'data/processed/{LEAGUE}_merged_preprocessed.csv'
TEAM = 'manchester city'  # normalized name
TEAM_ALT = 'man city'

print(f"Loading {FILE}...")
try:
    df = pd.read_csv(FILE, parse_dates=['Date'], dayfirst=True)
    print(f"Columns found: {list(df.columns)}")

    if 'FTHG' not in df.columns:
        print(" FATAL: FTHG still missing!")
    else:
        print(" Goals found. Computing ELO...")

        processed = _compute_ewma_elo_prematch(df, league_code=LEAGUE)

        norm_home = processed['HomeTeam'].map(config.normalize_team_name)
        norm_away = processed['AwayTeam'].map(config.normalize_team_name)
        city_rows = processed[
            (norm_home.isin([TEAM, TEAM_ALT])) | (norm_away.isin([TEAM, TEAM_ALT]))
        ].sort_values('Date')

        print("\n--- Man City ELO History (Last 5 Games) ---")
        if not city_rows.empty:
            cols = [c for c in ['Date','HomeTeam','AwayTeam','FTHG','FTAG','Elo_H','Elo_A'] if c in city_rows.columns]
            print(city_rows[cols].tail(5))
            last = city_rows.iloc[-1]
            home_norm = config.normalize_team_name(last['HomeTeam'])
            last_elo = last['Elo_H'] if home_norm in [TEAM, TEAM_ALT] else last['Elo_A']
            if last_elo == 1500.0:
                print(" FAIL: ELO is still 1500.0 (Static).")
            else:
                print(f" PASS: ELO is dynamic ({last_elo:.2f}).")
        else:
            print(f"No matches found for {TEAM} in processed data.")

except Exception as e:
    print(f"Error: {e}")
