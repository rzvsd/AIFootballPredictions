# debug_master_predictor.py
import pandas as pd
import joblib
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
from scipy.stats import poisson

# --- CONFIGURATION ---
LEAGUES_TO_PREDICT = {'E0': 39} # Let's just test the Premier League for now

def get_api_football_fixtures_and_odds(api_key, league_id):
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': api_key}
    
    # --- NEW LOGIC: Get the LAST 10 FINISHED games ---
    params = {'league': league_id, 'season': 2024, 'last': 10}
    
    print(f"DEBUG: Requesting URL: {url} with params: {params}")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()['response']
        print(f"DEBUG: API returned {len(data)} raw fixtures.")
    except Exception as e:
        print(f"CRITICAL DEBUG: Error fetching live data from API-Football: {e}")
        return []

    # ... (the rest of the function can stay the same for this test) ...
    fixture_ids = [match['fixture']['id'] for match in data]
    if not fixture_ids: return []
        
    odds_url = "https://v3.football.api-sports.io/odds"
    odds_data = []
    for i in range(0, len(fixture_ids), 20):
        chunk = fixture_ids[i:i + 20]
        odds_params = {'fixture': ','.join(map(str, chunk))}
        try:
            odds_response = requests.get(odds_url, headers=headers, params=odds_params)
            odds_response.raise_for_status()
            odds_data.extend(odds_response.json()['response'])
        except Exception as e:
            print(f"CRITICAL DEBUG: Error fetching odds chunk from API-Football: {e}")
    
    print(f"DEBUG: API returned odds for {len(odds_data)} fixtures.")
    
    odds_map = {}
    for item in odds_data:
        bookmaker = next((b for b in item['bookmakers'] if b['id'] == 8), None)
        if bookmaker:
            market = next((m for m in bookmaker['bets'] if m['id'] == 1), None)
            if market:
                odds_map[item['fixture']['id']] = True

    fixtures_with_odds = []
    for match in data:
        fixture_id = match['fixture']['id']
        if fixture_id in odds_map:
            fixtures_with_odds.append({
                'home_team': match['teams']['home']['name'],
                'away_team': match['teams']['away']['name'],
            })
            
    return fixtures_with_odds

def run_debug_predictor():
    print("--- Running DEBUG Master Predictor ---")
    load_dotenv()
    api_key = os.getenv('API_FOOTBALL_ODDS_KEY')
    if not api_key:
        print("Error: API_FOOTBALL_ODDS_KEY not found in .env file.")
        return

    for league_code, league_id in LEAGUES_TO_PREDICT.items():
        print(f"\n--- Processing League: {league_code} ---")
        
        # We don't need the model for this test, just the data
        
        fixtures = get_api_football_fixtures_and_odds(api_key, league_id)
        
        if not fixtures:
            print(f"RESULT: No upcoming fixtures with odds were found for {league_code}.")
        else:
            print(f"RESULT: Successfully found {len(fixtures)} upcoming matches with odds.")
            print("--- Here are the team names from the API ---")
            for i, match in enumerate(fixtures[:5]):
                print(f"Match {i+1}: '{match['home_team']}' vs '{match['away_team']}'")


if __name__ == "__main__":
    run_debug_predictor()