# fixtures_autogen.py
import os
import argparse
from datetime import date, timedelta, datetime
import pandas as pd
import requests
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_ODDS_KEY") or os.getenv("API_FOOTBALL_KEY")
LEAGUE_IDS = {"E0": 39, "D1": 78}
OUT_PATH = os.path.join("data", "fixtures", "{league}_weekly_fixtures.csv")

TEAM_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton", "West Ham United": "West Ham",
    "AFC Bournemouth": "Bournemouth", "Sheffield Utd": "Sheffield United",
}
def norm_team(x: str) -> str: return TEAM_MAP.get(x, x)

def get_season_from_date(start_date_str: str) -> int:
    """
    Intelligently determines the correct season year for the API.
    Football seasons typically start in August (month 8).
    """
    start_date = datetime.fromisoformat(start_date_str).date()
    year = start_date.year
    month = start_date.month
    
    # If the date is before August, it belongs to the previous year's season
    if month < 8:
        return year - 1
    else:
        return year

def fetch_fixtures_from_api(api_key, league_id, dfrom, dto):
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': api_key}
    
    # --- FINAL, CORRECTED SEASON LOGIC ---
    season_year = get_season_from_date(dfrom)
    params = {'league': league_id, 'season': season_year, 'from': dfrom, 'to': dto}
    
    print(f"DEBUG: Requesting fixtures for season {season_year}...")

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()['response']
    except Exception as e:
        print(f"Error fetching fixtures from API: {e}")
        return []

    fixtures = []
    for match in data:
        fixtures.append({
            "date": match['fixture']['date'],
            "home_team": norm_team(match['teams']['home']['name']),
            "away_team": norm_team(match['teams']['away']['name'])
        })
    return fixtures

def main():
    parser = argparse.ArgumentParser(description="Generate a weekly fixtures CSV from the API for a specific date range.")
    parser.add_argument("--league", required=True, help="League code, e.g., E0")
    parser.add_argument("--from", dest="dfrom", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--to", dest="dto", required=True, help="End date in YYYY-MM-DD format.")
    args = parser.parse_args()

    if not API_FOOTBALL_KEY:
        print("Error: API key not found in .env file.")
        return

    league_id = LEAGUE_IDS.get(args.league)
    if not league_id:
        print(f"Error: League ID not found for {args.league}.")
        return

    print(f"Fetching fixtures for {args.league} from {args.dfrom} to {args.dto}...")
    
    fixtures = fetch_fixtures_from_api(API_FOOTBALL_KEY, league_id, args.dfrom, args.dto)
    
    if not fixtures:
        print("No fixtures found for the specified date range.")
        return

    df = pd.DataFrame(fixtures)
    
    weekly_path = OUT_PATH.format(league=args.league)
    os.makedirs(os.path.dirname(weekly_path), exist_ok=True)
    
    df.to_csv(weekly_path, index=False)
    print(f"Successfully wrote {len(df)} fixtures to {weekly_path}")

if __name__ == "__main__":
    main()