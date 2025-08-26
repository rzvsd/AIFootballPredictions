# dixon_coles_predictor.py
import pandas as pd
import joblib
import os
from scipy.stats import poisson
import numpy as np

def predict_scoreline_probabilities(model, home_team, away_team, max_goals=5):
    """
    Uses the trained Dixon-Coles model to predict expected goals and then
    calculates the probability for each possible scoreline.
    """
    # Predict expected goals for the home team
    home_goals_avg = model.predict(pd.DataFrame(data={'team': home_team, 'opponent': away_team, 'home': 1}, index=[1])).values[0]
    
    # Predict expected goals for the away team
    away_goals_avg = model.predict(pd.DataFrame(data={'team': away_team, 'opponent': home_team, 'home': 0}, index=[1])).values[0]

    print(f"\nPredicted Expected Goals: Home={home_goals_avg:.2f}, Away={away_goals_avg:.2f}")

    # Calculate the probability of scoring 0, 1, 2, ... max_goals
    home_probs = [poisson.pmf(i, home_goals_avg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_goals_avg) for i in range(max_goals + 1)]

    # Create the scoreline probability matrix
    score_matrix = pd.DataFrame(
        np.outer(np.array(home_probs), np.array(away_probs)),
        index=[f'Home {i}' for i in range(max_goals + 1)],
        columns=[f'Away {i}' for i in range(max_goals + 1)]
    )
    
    score_matrix = score_matrix / score_matrix.values.sum()
    
    return score_matrix

def derive_market_probabilities(score_matrix: pd.DataFrame):
    """
    Analyzes the scoreline probability matrix to calculate probabilities
    for the main betting markets (1X2, O/U 2.5, BTTS).
    """
    home_win_prob = np.sum(np.tril(score_matrix.values, -1))
    draw_prob = np.sum(np.diag(score_matrix.values))
    away_win_prob = np.sum(np.triu(score_matrix.values, 1))

    over_2_5_prob = 0
    for home_goals in range(6):
        for away_goals in range(6):
            if home_goals + away_goals > 2.5:
                over_2_5_prob += score_matrix.iloc[home_goals, away_goals]
    
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

# --- Main Function ---
def run_predictor():
    LEAGUE = 'E0'
    
    print("Loading Dixon-Coles model...")
    model_path = os.path.join('advanced_models', f'{LEAGUE}_dixon_coles_model.pkl')
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please run dixon_coles_trainer.py first.")
        return

    # --- Define the Matchup to Predict ---
    # Make sure the team names EXACTLY match those in the dataset (e.g., 'Man City', not 'Manchester City')
    home_team = "Man City"
    away_team = "Chelsea"
    print(f"\n--- Predicting Match: {home_team} vs {away_team} ---")
    
    # Generate and Display the Scoreline Probability Matrix
    score_matrix = predict_scoreline_probabilities(model, home_team, away_team)
    
    print("\nScoreline Probability Matrix (%):")
    print((score_matrix * 100).round(2))

    # Run the Analysis
    derive_market_probabilities(score_matrix)


if __name__ == "__main__":
    run_predictor()