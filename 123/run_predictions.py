# run_predictions.py
import os
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import date, timedelta
from scipy.stats import poisson
from dotenv import load_dotenv
from bet_fusion import _feature_row_from_snapshot as build_feature_row

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

def create_feature_rows(home_team: str, away_team: str, stats_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature rows aligned to the training columns for home/away XGB models
    using the snapshot-to-features mapping from bet_fusion.
    Returns (home_input_df, away_input_df) or (None, None) if teams missing.
    """
    feat_full = build_feature_row(stats_df, home_team, away_team)
    if feat_full is None or feat_full.empty:
        print(f"Warning: Could not build features for '{home_team}' vs '{away_team}'. Skipping match.")
        return None, None
    # Training columns as in xgb_trainer
    home_cols = [
        'ShotConv_H','ShotConvRec_H','PointsPerGame_H','xGDiff_H',
        'CornersConv_H','CornersConvRec_H','NumMatches_H',
        'Elo_H','Elo_A','EloDiff',
        'GFvsMid_H','GAvsMid_H','GFvsHigh_H','GAvsHigh_H',
        'GFvsSim_H','GAvsSim_H'
    ]
    away_cols = [
        'ShotConv_A','ShotConvRec_A','PointsPerGame_A','xGDiff_A',
        'CornersConv_A','CornersConvRec_A','NumMatches_A',
        'Elo_H','Elo_A','EloDiff',
        'GFvsMid_A','GAvsMid_A','GFvsHigh_A','GAvsHigh_A',
        'GFvsSim_A','GAvsSim_A'
    ]
    # Ensure columns exist and fill missing with 0.0
    for col in set(home_cols + away_cols):
        if col not in feat_full.columns:
            feat_full[col] = 0.0
    home_input = feat_full[home_cols].astype(float)
    away_input = feat_full[away_cols].astype(float)
    return home_input, away_input


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
        home_input, away_input = create_feature_rows(home, away, latest_stats)

        if home_input is not None and away_input is not None:
            # Predict expected goals using the trained input schemas
            pred_xg_h = home_model.predict(home_input)[0]
            pred_xg_a = away_model.predict(away_input)[0]

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
