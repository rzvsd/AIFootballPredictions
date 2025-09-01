# run_predictions.py
import os
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import date, timedelta
from scipy.stats import poisson
from dotenv import load_dotenv

# Import all settings from our new config file
import config

# --- SETUP ---
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_ODDS_KEY") or os.getenv("API_FOOTBALL_KEY")


# --- CORE FUNCTIONS (Adapted from your existing scripts) ---

def fetch_upcoming_fixtures(league_id: int, api_key: str) -> pd.DataFrame:
    """
    Fetches fixtures for the next 7 days automatically from the API.
    """
    if not api_key:
        raise ValueError("API key not found. Please set API_FOOTBALL_KEY in your .env file.")

    today = date.today()
    end_date = today + timedelta(days=7)
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': api_key}
    
    # Determine the correct season year
    season_year = today.year if today.month >= 8 else today.year - 1
    
    params = {
        'league': league_id,
        'season': season_year,
        'from': today.isoformat(),
        'to': end_date.isoformat()
    }
    
    print(f"Fetching fixtures from {today.isoformat()} to {end_date.isoformat()} for season {season_year}...")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json().get('response', [])
    except Exception as e:
        print(f"Error fetching fixtures from API: {e}")
        return pd.DataFrame()

    fixtures = []
    for match in data:
        fixtures.append({
            "date": match['fixture']['date'],
            "home_team_api": match['teams']['home']['name'],
            "away_team_api": match['teams']['away']['name']
        })
    
    df = pd.DataFrame(fixtures)
    if not df.empty:
        # IMMEDIATELY normalize the names using our config map
        df["home_team"] = df["home_team_api"].apply(config.normalize_team_name)
        df["away_team"] = df["away_team_api"].apply(config.normalize_team_name)
    
    return df

def derive_market_probabilities(home_goals_avg: float, away_goals_avg: float, max_goals=10) -> dict:
    """
    Takes predicted goals and calculates 1X2 market probabilities using a Poisson distribution.
    """
    home_probs = [poisson.pmf(i, home_goals_avg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_goals_avg) for i in range(max_goals + 1)]
    score_matrix = np.outer(np.array(home_probs), np.array(away_probs))
    
    # Normalize the matrix to ensure probabilities sum to 1
    score_matrix = score_matrix / score_matrix.sum()

    return {
        'home_win': np.sum(np.tril(score_matrix, -1)),
        'draw': np.sum(np.diag(score_matrix)),
        'away_win': np.sum(np.triu(score_matrix, 1))
    }

def create_feature_row(home_team: str, away_team: str, stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a single feature row for an upcoming match, ready for the XGB model.
    This is the critical link between your stored features and a future game.
    """
    try:
        home_stats = stats_df[stats_df['team'] == home_team].iloc[0]
        away_stats = stats_df[stats_df['team'] == away_team].iloc[0]
    except IndexError:
        # This happens if a team (e.g., newly promoted) is not in our stats file
        print(f"Warning: Could not find stats for '{home_team}' or '{away_team}'. Skipping match.")
        return None

    feature_dict = {
        'ShotConv_H': home_stats.get('xg_L5'), # Example mapping, adjust to your feature names in the parquet
        'ShotConv_A': away_stats.get('xg_L5'),
        'ShotConvRec_H': home_stats.get('xga_L5'),
        'ShotConvRec_A': away_stats.get('xga_L5'),
        'PointsPerGame_H': home_stats.get('gpg_L10'),
        'PointsPerGame_A': away_stats.get('gpg_L10'),
        'CleanSheetStreak_H': 0, # Placeholder if not available
        'CleanSheetStreak_A': 0,
        'xGDiff_H': home_stats.get('xgdiff_L10'),
        'xGDiff_A': away_stats.get('xgdiff_L10'),
        'CornersConv_H': home_stats.get('corners_L10'),
        'CornersConv_A': away_stats.get('corners_L10'),
        'CornersConvRec_H': home_stats.get('corners_allowed_L10'),
        'CornersConvRec_A': away_stats.get('corners_allowed_L10'),
        'NumMatches_H': 20, # Placeholder
        'NumMatches_A': 20,
        # 'EloDifference': home_stats.get('elo') - away_stats.get('elo') # If you add Elo
    }

    # Ensure the order of columns matches the training features
    feature_row = pd.DataFrame([feature_dict], columns=config.ULTIMATE_FEATURES)
    return feature_row


# --- MAIN EXECUTION ---
def main():
    """
    The main function that orchestrates the entire prediction process.
    """
    print("---  Prognosticator Bot Initialized ---")
    
    # 1. Load the "Brain": The pre-trained XGBoost models
    try:
        home_model = joblib.load(config.HOME_MODEL_PATH)
        away_model = joblib.load(config.AWAY_MODEL_PATH)
        print("Successfully loaded champion models.")
    except FileNotFoundError:
        print(f"ERROR: Models not found. Please run 'xgb_trainer.py' first.")
        return

    # 2. Load the latest team stats snapshot
    try:
        latest_stats = pd.read_parquet(config.STATS_SNAPSHOT_PATH)
        print(f"Successfully loaded team stats snapshot for {len(latest_stats)} teams.")
    except FileNotFoundError:
        print(f"ERROR: Stats snapshot not found. Please run 'feature_store.py' first.")
        return

    # 3. Fetch and process upcoming fixtures
    upcoming_fixtures = fetch_upcoming_fixtures(config.LEAGUE_ID_API, API_KEY)
    if upcoming_fixtures.empty:
        print("No upcoming fixtures found for the next 7 days.")
        return
    
    print(f"\nFound {len(upcoming_fixtures)} upcoming matches. Analyzing...\n")
    
    # 4. Generate and display predictions
    predictions = []
    for _, fixture in upcoming_fixtures.iterrows():
        home = fixture['home_team']
        away = fixture['away_team']

        # Create the specific feature set for this match
        feature_row = create_feature_row(home, away, latest_stats)

        if feature_row is not None:
            # Predict expected goals using the models
            pred_xg_h = home_model.predict(feature_row)[0]
            pred_xg_a = away_model.predict(feature_row)[0]

            # Translate goals into market probabilities
            market_probs = derive_market_probabilities(pred_xg_h, pred_xg_a)
            
            predictions.append({
                "Date": pd.to_datetime(fixture['date']).strftime('%Y-%m-%d %H:%M'),
                "Home": fixture['home_team_api'], # Show original API name for clarity
                "Away": fixture['away_team_api'],
                "P(H)": market_probs['home_win'],
                "P(D)": market_probs['draw'],
                "P(A)": market_probs['away_win'],
                "xG_H": pred_xg_h,
                "xG_A": pred_xg_a
            })

    if not predictions:
        print("Could not generate predictions. Check team names and stats file.")
        return
        
    # Display results in a clean table
    results_df = pd.DataFrame(predictions).sort_values("Date")
    results_df['P(H)'] = results_df['P(H)'].map('{:.2%}'.format)
    results_df['P(D)'] = results_df['P(D)'].map('{:.2%}'.format)
    results_df['P(A)'] = results_df['P(A)'].map('{:.2%}'.format)
    results_df['xG_H'] = results_df['xG_H'].map('{:.2f}'.format)
    results_df['xG_A'] = results_df['xG_A'].map('{:.2f}'.format)
    
    print("--- WEEKLY PREDICTIONS ---")
    print(results_df.to_string(index=False))
    print("--- Bot Shutdown ---")


if __name__ == "__main__":
    main()