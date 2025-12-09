# fixture_fetcher_v2.py
import os
import requests
import pandas as pd
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_ODDS_KEY") or os.getenv("API_FOOTBALL_KEY")
LEAGUE = 'E0'
SEASON_YEAR = 2024 # The start year of the season we want fixtures for

# Manually found Season IDs from API-Football documentation for Premier League
SEASON_IDS = {
    2021: 3456,
    2022: 4335,
    2023: 5267,
    2024: 6280 # This is a guess, may need to be updated
}

TEAM_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton", "West Ham United": "West Ham",
    "AFC Bournemouth": "Bournemouth", "Sheffield Utd": "Sheffield United",
}
def norm_team(x: str) -> str: return TEAM_MAP.get(x, x)

def fetch_fixtures_v2(api_key, season_id):
    """
    Fetches the full season fixture list from the V2 API endpoint.
    """
    base_url = 'https://v2.api-football.com/'
    endpoint = f'fixtures/league/{season_id}'
    url = base_url + endpoint
    headers = {'X-RapidAPI-Key': api_key}

    print(f"Requesting fixtures from V2 endpoint: {url}")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()['api']['fixtures']
    except Exception as e:
        print(f"Error fetching fixtures from V2 API: {e}")
        return []

    fixtures = []
    for match in data:
        fixtures.append({
            "date": match['event_date'],
            "home_team": norm_team(match['homeTeam']['team_name']),
            "away_team": norm_team(match['awayTeam']['team_name'])
        })
    return fixtures

def main():
    if not API_KEY:
        print("Error: API key not found in .env file.")
        return

    season_id = SEASON_IDS.get(SEASON_YEAR)
    if not season_id:
        print(f"Error: Season ID not found for the year {SEASON_YEAR}.")
        return

    fixtures = fetch_fixtures_v2(API_KEY, season_id)
    
    if not fixtures:
        print("No fixtures found. The Season ID might be incorrect for the current year.")
        return

    df = pd.DataFrame(fixtures)
    
    # Save the full season master file
    output_dir = os.path.join('data', 'fixtures_master')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{LEAGUE}_{SEASON_YEAR}.csv')
    
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully wrote {len(df)} fixtures to the season master file: {output_path}")
    print("\nYou can now use fixtures_autogen.py to create your weekly file from this master list.")


if __name__ == "__main__":
    main()