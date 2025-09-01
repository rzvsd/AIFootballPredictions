# debug_api.py
import os
import requests
from dotenv import load_dotenv

def check_api_status():
    """
    Connects to the API and prints the raw data for the next few matches
    to help us debug what the API is providing.
    """
    load_dotenv()
    api_key = os.getenv('API_FOOTBALL_DATA')
    if not api_key:
        print("Error: API key not found in .env file.")
        return

    LEAGUE = 'PL' # Premier League API code
    url = f"https://api.football-data.org/v4/competitions/{LEAGUE}/matches"
    headers = {'X-Auth-Token': api_key}

    print(f"Checking API status for {LEAGUE}...")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return

    matches = data.get('matches', [])
    
    if not matches:
        print("API returned no matches at all.")
        return

    print(f"\nFound {len(matches)} total matches. Showing the first 15:")
    print("-" * 50)

    for i, match in enumerate(matches[:15]):
        home = match.get('homeTeam', {}).get('name', 'N/A')
        away = match.get('awayTeam', {}).get('name', 'N/A')
        status = match.get('status', 'N/A')
        odds_available = 'Yes' if match.get('odds') and match['odds'].get('homeWin') is not None else 'No'
        
        print(f"Match {i+1}: {home} vs {away}")
        print(f"  Status: {status}")
        print(f"  Odds Available: {odds_available}")
        print("-" * 20)

if __name__ == "__main__":
    check_api_status()