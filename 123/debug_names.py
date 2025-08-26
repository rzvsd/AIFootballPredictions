# debug_names.py
import pandas as pd
import os
import asyncio
from feature_enhancer import fetch_understat_league # <-- CORRECTED IMPORT

async def find_mismatched_names():
    """
    Loads both datasets and prints the unique team names from each
    to help us identify mismatches for our team name map.
    """
    LEAGUE = 'E0'
    SEASONS = ['2021', '2022', '2023']
    
    # --- 1. Load Base Data ---
    print("--- Loading Base Data Teams ---")
    raw_data_path = os.path.join('data', 'raw', f'{LEAGUE}_merged.csv')
    base_df = pd.read_csv(raw_data_path)
    base_teams = sorted(pd.concat([base_df['HomeTeam'], base_df['AwayTeam']]).unique())
    print(f"Found {len(base_teams)} unique team names in the base data.")
    print(base_teams)

    # --- 2. Fetch xG Data ---
    print("\n--- Loading Understat xG Data Teams ---")
    all_xg_data = []
    for year in SEASONS:
        # We need to specify the league code for the fetch function
        season_xg_df = await fetch_understat_league(year, league_code=LEAGUE)
        all_xg_data.append(season_xg_df)
    
    xg_df = pd.concat(all_xg_data)
    xg_teams = sorted(pd.concat([xg_df['HomeTeam'], xg_df['AwayTeam']]).unique())
    print(f"Found {len(xg_teams)} unique team names in the Understat data.")
    print(xg_teams)
    
    # --- 3. Find the Differences ---
    print("\n--- Mismatch Report ---")
    base_set = set(base_teams)
    xg_set = set(xg_teams)
    
    print("\nTeams in Understat data but NOT in base data (Need to be mapped TO):")
    print(sorted(list(xg_set - base_set)))
    
    print("\nTeams in base data but NOT in Understat data (Need to be mapped FROM):")
    print(sorted(list(base_set - xg_set)))


if __name__ == "__main__":
    asyncio.run(find_mismatched_names())