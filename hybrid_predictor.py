# hybrid_predictor.py
import pandas as pd
import joblib
import os
from scipy.stats import poisson
import numpy as np

# --- Helper Functions ---
def get_latest_features(enhanced_df: pd.DataFrame, team_name: str) -> pd.Series:
    """
    Finds the most recent match for a team to get its latest features.
    """
    last_home_game = enhanced_df[enhanced_df['HomeTeam'] == team_name].tail(1)
    last_away_game = enhanced_df[enhanced_df['AwayTeam'] == team_name].tail(1)
    
    if last_home_game.empty and last_away_game.empty: return None
    if last_home_game.empty: last_game = last_away_game
    elif last_away_game.empty: last_game = last_home_game
    else:
        last_game = last_home_game if pd.to_datetime(last_home_game['Date'].iloc[0]) > pd.to_datetime(last_away_game['Date'].iloc[0]) else last_away_game
        
    return last_game.iloc[0]

def derive_market_probabilities(score_matrix: pd.DataFrame):
    """
    Analyzes the scoreline probability matrix to calculate probabilities
    for the main betting markets.
    """
    home_win_prob = np.sum(np.tril(score_matrix.values, -1))
    draw_prob = np.sum(np.diag(score_matrix.values))
    away_win_prob = np.sum(np.triu(score_matrix.values, 1))

    over_2_5_prob = np.sum(score_matrix.values[np.add.outer(np.arange(6), np.arange(6)) > 2.5])
    under_2_5_prob = 1 - over_2_5_prob

    btts_yes_prob = np.sum(score_matrix.iloc[1:, 1:]).sum()
    btts_no_prob = 1 - btts_yes_prob

    print("\n--- Derived Market Probabilities ---")
    print(f"Home Win: {home_win_prob:.2%} (Fair Odds: {1/home_win_prob:.2f})")
    print(f"Draw:     {draw_prob:.2%} (Fair Odds: {1/draw_prob:.2f})")
    print(f"Away Win: {away_win_prob:.2%} (Fair Odds: {1/away_win_prob:.2f})")
    print("-" * 35)
    print(f"Over 2.5 Goals:  {over_2_5_prob:.2%} (Fair Odds: {1/over_2_5_prob:.2f})")
    print(f"Under 2.5 Goals: {under_2_5_prob:.2%} (Fair Odds: {1/under_2_5_prob:.2f})")
    print("-" * 35)
    print(f"BTTS (Yes): {btts_yes_prob:.2%} (Fair Odds: {1/btts_yes_prob:.2f})")
    print(f"BTTS (No):  {btts_no_prob:.2%} (Fair Odds: {1/btts_no_prob:.2f})")
    print("-" * 35)

# --- Main Predictor Function ---
def run_hybrid_predictor():
    LEAGUE = 'E0'
    
    print("--- Running Hybrid Prediction Pipeline ---")
    print("Loading assets...")
    home_model = joblib.load(os.path.join('advanced_models', f'{LEAGUE}_xgb_home_model.pkl'))
    away_model = joblib.load(os.path.join('advanced_models', f'{LEAGUE}_xgb_away_model.pkl'))
    enhanced_df = pd.read_csv(os.path.join('data', 'enhanced', f'{LEAGUE}_enhanced.csv'))
    
    home_team = "Man City"
    away_team = "Arsenal"
    print(f"\n--- Predicting Match: {home_team} vs {away_team} ---")

    home_features_series = get_latest_features(enhanced_df, home_team)
    away_features_series = get_latest_features(enhanced_df, away_team)

    if home_features_series is None or away_features_series is None:
        print("Could not find one of the teams in the historical data.")
        return

    # Construct the feature vector for the match
    feature_values = {
        'HomeAvgGoalsScored_Last5': home_features_series['HomeAvgGoalsScored_Last5'],
        'HomeAvgGoalsConceded_Last5': home_features_series['HomeAvgGoalsConceded_Last5'],
        'AwayAvgGoalsScored_Last5': away_features_series['AwayAvgGoalsScored_Last5'],
        'AwayAvgGoalsConceded_Last5': away_features_series['AwayAvgGoalsConceded_Last5'],
        'HomeAvgShots_Last5': home_features_series['HomeAvgShots_Last5'],
        'AwayAvgShots_Last5': away_features_series['AwayAvgShots_Last5'],
        'HomeAvgShotsOnTarget_Last5': home_features_series['HomeAvgShotsOnTarget_Last5'],
        'AwayAvgShotsOnTarget_Last5': away_features_series['AwayAvgShotsOnTarget_Last5'],
        'HomeAvgCorners_Last5': home_features_series['HomeAvgCorners_Last5'],
        'AwayAvgCorners_Last5': away_features_series['AwayAvgCorners_Last5'],
        'EloDifference': home_features_series['HomeElo'] - away_features_series['AwayElo']
    }
    
    features_df = pd.DataFrame([feature_values])
    
    # --- Stage 1: Use XGBoost to predict Expected Goals ---
    expected_home_goals = home_model.predict(features_df)[0]
    expected_away_goals = away_model.predict(features_df)[0]
    print(f"\nXGBoost Predicted Expected Goals: Home={expected_home_goals:.2f}, Away={expected_away_goals:.2f}")

    # --- Stage 2: Use Poisson to generate probability matrix ---
    max_goals = 5
    home_probs = [poisson.pmf(i, expected_home_goals) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, expected_away_goals) for i in range(max_goals + 1)]
    score_matrix = pd.DataFrame(np.outer(np.array(home_probs), np.array(away_probs)))
    score_matrix = score_matrix / score_matrix.values.sum()
    
    # --- Final Analysis ---
    derive_market_probabilities(score_matrix)


if __name__ == "__main__":
    run_hybrid_predictor()