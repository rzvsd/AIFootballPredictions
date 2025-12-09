# team_name_finder.py
#
# A helper script to discover team names from both the API and your local dataset.
# Run this script once for your league, then use its output to fill in the
# TEAM_NAME_MAP in your config.py file.

import os
import pandas as pd
import requests
from dotenv import load_dotenv

# --- SETTINGS TO CONFIGURE ---
# Make sure these match the league you are setting up in config.py
LEAGUE_ID = 39      # API League ID (e.g., 39 for Premier League)
SEASON = 2024       # A recent full season to get a complete team list
DATASET_PATH = "data/enhanced/E0_strength_adj.csv" # Path to your main training data file
# --- END OF SETTINGS ---


def get_api_team_names(league_id: int, season: int, api_key: str) -> list:
    """Connects to the API and retrieves a unique, sorted list of team names."""
    if not api_key:
        print("ERROR: API key not found. Please ensure it is in your .env file.")
        return []

    url = "https://v3.football.api-sports.io/teams"
    headers = {'x-apisports-key': api_key}
    params = {'league': league_id, 'season': season}
    
    print(f"Querying API for team names in league {league_id}, season {season}...")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json().get('response', [])
        
        if not data:
            print("API returned no teams. Check your LEAGUE_ID and SEASON.")
            return []
            
        team_names = {team_info['team']['name'] for team_info in data}
        print(f"Found {len(team_names)} unique team names from the API.")
        return sorted(list(team_names))
        
    except Exception as e:
        print(f"An error occurred while fetching from the API: {e}")
        return []

def get_dataset_team_names(file_path: str) -> list:
    """Reads your local CSV and gets a unique, sorted list of team names."""
    print(f"\nReading team names from local file: {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # Check for common column names for teams
        home_col, away_col = None, None
        if 'HomeTeam' in df.columns:
            home_col, away_col = 'HomeTeam', 'AwayTeam'
        elif 'home_team' in df.columns:
            home_col, away_col = 'home_team', 'away_team'
        else:
            print("ERROR: Could not find team name columns (e.g., 'HomeTeam') in the CSV.")
            return []

        home_teams = df[home_col].unique()
        away_teams = df[away_col].unique()
        
        all_teams = set(home_teams) | set(away_teams) # Combine and get unique
        print(f"Found {len(all_teams)} unique team names from your dataset.")
        return sorted(list(all_teams))

    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{file_path}'. Please check the path.")
        return []


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_FOOTBALL_ODDS_KEY") or os.getenv("API_FOOTBALL_KEY")
    
    api_names = get_api_team_names(LEAGUE_ID, SEASON, api_key)
    dataset_names = get_dataset_team_names(DATASET_PATH)
    
    print("\n\n--- COMPARISON ---")
    print("Use these two lists to fill out the TEAM_NAME_MAP in config.py")
    print("Format: 'API Name': 'Your Dataset Name',\n")
    
    print("="*30)
    print("   NAMES FROM API")
    print("="*30)
    if api_names:
        for name in api_names:
            print(f"'{name}'")
    
    print("\n" + "="*30)
    print("   NAMES FROM YOUR DATASET")
    print("="*30)
    if dataset_names:
        for name in dataset_names:
            print(f"'{name}'")
    
    print("\nComparison complete. You can now update your config file.")