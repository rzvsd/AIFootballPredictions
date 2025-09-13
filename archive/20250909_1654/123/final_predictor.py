# final_predictor.py
import pandas as pd
import joblib
import os
from scipy.stats import poisson
import numpy as np
import requests
from dotenv import load_dotenv

# --- Helper Functions from previous scripts ---

def derive_market_probabilities(model, home_team, away_team, max_goals=5):
    home_goals_avg = model.predict(pd.DataFrame(data={'team': home_team, 'opponent': away_team, 'home': 1}, index=[1])).values[0]
    away_goals_avg = model.predict(pd.DataFrame(data={'team': away_team, 'opponent': home_team, 'home': 0}, index=[1])).values[0]
    
    home_probs = [poisson.pmf(i, home_goals_avg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_goals_avg) for i in range(max_goals + 1)]

    score_matrix = np.outer(np.array(home_probs), np.array(away_probs))
    score_matrix = score_matrix / score_matrix.sum()

    return {
        'home_win': np.sum(np.tril(score_matrix, -1)),
        'draw': np.sum(np.diag(score_matrix)),
        'away_win': np.sum(np.triu(score_matrix, 1))
    }

def get_live_odds_and_fixtures(api_key, league_code='PL'):
    """ Fetches live 1X2 odds and fixtures from football-data.org """
    
    # Mapping our league codes to the API's codes
    league_map = {'E0': 'PL', 'D1': 'BL1', 'SP1': 'PD', 'I1': 'SA', 'F1': 'FL1'}
    api_league_code = league_map.get(league_code, 'PL')

    url = f"https://api.football-data.org/v4/competitions/{api_league_code}/matches"
    headers = {'X-Auth-Token': api_key}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live data: {e}")
        return []

    fixtures = []
    for match in data.get('matches', []):
        if match['status'] == 'SCHEDULED' and match.get('odds') and match['odds'].get('homeWin') is not None:
            fixtures.append({
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'odds_h': match['odds']['homeWin'],
                'odds_d': match['odds']['draw'],
                'odds_a': match['odds']['awayWin']
            })
            
    return fixtures

# --- Main Predictor Function ---
def run_final_predictor():
    LEAGUE = 'D1'
    EDGE_THRESHOLD = 0.05
    
    print("--- Running Final Prediction Pipeline ---")
    
    # 1. Load API Key
    load_dotenv()
    api_key = os.getenv('API_FOOTBALL_DATA')
    if not api_key:
        print("Error: API_FOOTBALL_DATA key not found in .env file.")
        return

    # 2. Load Trained Dixon-Coles Model
    model_path = os.path.join('advanced_models', f'{LEAGUE}_dixon_coles_model.pkl')
    try:
        model = joblib.load(model_path)
        print("Dixon-Coles model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # 3. Get Live Fixtures and Odds
    print(f"Fetching live fixtures and odds for league {LEAGUE}...")
    fixtures = get_live_odds_and_fixtures(api_key, LEAGUE)
    if not fixtures:
        print("No upcoming fixtures with odds found.")
        return
    
    print(f"Found {len(fixtures)} upcoming matches with odds.")

    # 4. Find Value Bets
    value_bets = []
    for match in fixtures:
        home_team = match['home_team']
        away_team = match['away_team']
        
        # We need to map API team names to our dataset's team names
        # This is a crucial step we will need to refine
        # For now, we'll assume they match for simplicity
        
        try:
            market_probs = derive_market_probabilities(model, home_team, away_team)
            
            implied_prob_h = 1 / match['odds_h']
            implied_prob_a = 1 / match['odds_a']
            
            edge_h = market_probs['home_win'] - implied_prob_h
            edge_a = market_probs['away_win'] - implied_prob_a

            if edge_h > EDGE_THRESHOLD:
                value_bets.append({'match': f"{home_team} vs {away_team}", 'bet': 'Home Win (1X2)', 'edge': f"{edge_h:.2%}"})
            
            if edge_a > EDGE_THRESHOLD:
                value_bets.append({'match': f"{home_team} vs {away_team}", 'bet': 'Away Win (1X2)', 'edge': f"{edge_a:.2%}"})

        except Exception as e:
            # This will happen if a team name from the API doesn't match our model's team names
            print(f"Skipping match {home_team} vs {away_team} due to error: {e}")

    # 5. Print Final Report
    print("\n--- Top Value Bets Found ---")
    if not value_bets:
        print("No value bets found for the upcoming matches.")
    else:
        for bet in sorted(value_bets, key=lambda x: x['edge'], reverse=True):
            print(f"Match: {bet['match']}, Bet: {bet['bet']}, Edge: {bet['edge']}")


if __name__ == "__main__":
    run_final_predictor()