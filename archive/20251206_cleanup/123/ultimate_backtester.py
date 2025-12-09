# ultimate_backtester.py
import pandas as pd
import joblib
import os
from scipy.stats import poisson
import numpy as np

# --- CONFIGURATION ---
LEAGUE = 'E0'
STARTING_BANKROLL = 1000
STAKE = 10
EDGE_THRESHOLD = 0.05

def derive_market_probabilities(home_goals_avg, away_goals_avg, max_goals=5):
    """
    Uses expected goals to derive probabilities for the 1X2 market.
    """
    home_probs = [poisson.pmf(i, home_goals_avg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_goals_avg) for i in range(max_goals + 1)]

    score_matrix = np.outer(np.array(home_probs), np.array(away_probs))
    score_matrix = score_matrix / score_matrix.sum()

    return {
        'home_win': np.sum(np.tril(score_matrix, -1)),
        'draw': np.sum(np.diag(score_matrix)),
        'away_win': np.sum(np.triu(score_matrix, 1))
    }

# --- Main Backtesting Function ---
def run_ultimate_backtest():
    print(f"--- Starting Ultimate Backtest for League: {LEAGUE} ---")

    # --- 1. Load Ultimate Assets ---
    home_model_path = os.path.join('advanced_models', f'{LEAGUE}_ultimate_xgb_home.pkl')
    away_model_path = os.path.join('advanced_models', f'{LEAGUE}_ultimate_xgb_away.pkl')
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_strength_adj.csv')

    try:
        home_model = joblib.load(home_model_path)
        away_model = joblib.load(away_model_path)
        df = pd.read_csv(data_path)
        print("Successfully loaded ultimate models and data.")
    except FileNotFoundError as e:
        print(f"Error: Could not find file.")
        print(e)
        return

    # --- 2. Prepare Data ---
    features = [
        'ShotConv_H', 'ShotConv_A',
        'ShotConvRec_H', 'ShotConvRec_A',
        'PointsPerGame_H', 'PointsPerGame_A',
        'CleanSheetStreak_H', 'CleanSheetStreak_A',
        'xGDiff_H', 'xGDiff_A',
        'CornersConv_H', 'CornersConv_A',
        'CornersConvRec_H', 'CornersConvRec_A',
        'NumMatches_H', 'NumMatches_A'
    ]
    
    # Use robust odds handling
    if 'B365H' in df.columns and 'B365D' in df.columns and 'B365A' in df.columns:
        odds_h_col, odds_d_col, odds_a_col = 'B365H', 'B365D', 'B365A'
        print("Using Bet365 (B365) odds for backtest.")
    elif 'BWH' in df.columns and 'BWD' in df.columns and 'BWA' in df.columns:
        odds_h_col, odds_d_col, odds_a_col = 'BWH', 'BWD', 'BWA'
        print("Using Bet&Win (BW) odds for backtest.")
    else:
        print("Error: No recognized 1X2 odds columns found.")
        return
        
    df.dropna(subset=features + [odds_h_col, odds_d_col, odds_a_col, 'FTR'], inplace=True)
    
    X = df[features]
    
    # --- 3. Run Simulation ---
    bankroll = STARTING_BANKROLL
    bets_placed = 0
    
    print(f"\nSimulating {len(df)} matches with a starting bankroll of {bankroll}...")

    expected_home_goals = home_model.predict(X)
    expected_away_goals = away_model.predict(X)

    df_indices = X.index
    
    for i, index in enumerate(df_indices):
        row = df.loc[index]
        
        home_goals_avg = expected_home_goals[i]
        away_goals_avg = expected_away_goals[i]

        market_probs = derive_market_probabilities(home_goals_avg, away_goals_avg)

        odds_h, odds_d, odds_a = row[odds_h_col], row[odds_d_col], row[odds_a_col]
        implied_prob_h, implied_prob_d, implied_prob_a = 1/odds_h, 1/odds_d, 1/odds_a
        
        actual_result = row['FTR']
        
        if market_probs['home_win'] > implied_prob_h + EDGE_THRESHOLD:
            bets_placed += 1
            bankroll -= STAKE
            if actual_result == 'H':
                bankroll += STAKE * odds_h
        
        elif market_probs['draw'] > implied_prob_d + EDGE_THRESHOLD:
            bets_placed += 1
            bankroll -= STAKE
            if actual_result == 'D':
                bankroll += STAKE * odds_d
        
        elif market_probs['away_win'] > implied_prob_a + EDGE_THRESHOLD:
            bets_placed += 1
            bankroll -= STAKE
            if actual_result == 'A':
                bankroll += STAKE * odds_a

    # --- 4. Generate Final Report ---
    print("\n--- Ultimate Model Backtest Complete ---")
    print(f"League Tested: {LEAGUE}")
    print("-" * 25)
    print(f"Starting Bankroll: {STARTING_BANKROLL:.2f}")
    print(f"Ending Bankroll:   {bankroll:.2f}")
    
    profit = bankroll - STARTING_BANKROLL
    total_wagered = bets_placed * STAKE
    roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0

    print(f"Profit / Loss:     {profit:.2f}")
    print(f"Total Bets Placed: {bets_placed}")
    print(f"ULTIMATE ROI:      {roi:.2f}%")
    print("-" * 25)


if __name__ == "__main__":
    run_ultimate_backtest()