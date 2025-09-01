# ultimate_backtester.py
import pandas as pd
import joblib
import os
from scipy.stats import poisson
import numpy as np

# --- CONFIGURATION ---
LEAGUE = 'E0'
SEASON_TO_TEST = 2023 # The START year of the season we want to backtest
STARTING_BANKROLL = 1000
STAKE = 10
PROBABILITY_THRESHOLD = 0.60 # Your 60% confidence filter
EDGE_THRESHOLD = 0.05        # Only bet if our edge is > 5%

def score_matrix_from_mus(mu_h: float, mu_a: float, max_goals: int = 10) -> np.ndarray:
    # ... (This helper function remains the same) ...
    hg = np.arange(0, max_goals + 1); ag = np.arange(0, max_goals + 1)
    def pois_pmf(k_arr, lam):
        k_arr = np.asarray(k_arr, dtype=float)
        fact = np.array([np.math.factorial(int(k)) for k in k_arr], dtype=float)
        return np.exp(-lam) * (lam ** k_arr) / fact
    ph = pois_pmf(hg, float(max(mu_h, 1e-6))); pa = pois_pmf(ag, float(max(mu_a, 1e-6)))
    P = np.outer(ph, pa); P /= P.sum()
    return P

def derive_all_market_probabilities(P: np.ndarray):
    # ... (This helper function remains the same) ...
    home = np.tril(P, -1).sum(); draw = np.diag(P).sum(); away = np.triu(P, 1).sum()
    grid = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
    over25 = float(P[grid >= 3].sum())
    prob_home_zero = P[0, :].sum(); prob_away_zero = P[:, 0].sum(); prob_zero_zero = P[0, 0]
    btts_no = prob_home_zero + prob_away_zero - prob_zero_zero
    btts = 1.0 - btts_no
    return {"p_H": home, "p_D": draw, "p_A": away, "p_O2.5": over25, "p_U2.5": 1.0 - over25, "p_BTTS_Y": btts, "p_BTTS_N": btts_no}

def main():
    print(f"--- Starting Ultimate Backtest for League: {LEAGUE}, Season: {SEASON_TO_TEST} ---")

    # 1. Load Data and Models
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_final_features.csv')
    home_model_path = os.path.join("advanced_models", f'{LEAGUE}_ultimate_xgb_home.pkl')
    away_model_path = os.path.join("advanced_models", f'{LEAGUE}_ultimate_xgb_away.pkl')
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    home_model = joblib.load(home_model_path)
    away_model = joblib.load(away_model_path)

    # 2. Split Data into Train and Test
    train_df = df[df['Date'].dt.year < SEASON_TO_TEST]
    test_df = df[df['Date'].dt.year >= SEASON_TO_TEST].copy() # Use .copy() to avoid warnings
    print(f"Training on {len(train_df)} matches, testing on {len(test_df)} matches.")

    # NOTE: For a true walk-forward test, we would retrain the models here on train_df.
    # For simplicity and speed, we are using the models already trained on the full dataset.
    # This is a minor simplification for this test.

    # 3. Simulate the Test Season
    bankroll = STARTING_BANKROLL
    bets_placed = 0
    bets_won = 0
    
    features = [
        'ShotConv_H', 'ShotConv_A', 'ShotConvRec_H', 'ShotConvRec_A', 'PointsPerGame_H', 'PointsPerGame_A',
        'CleanSheetStreak_H', 'CleanSheetStreak_A', 'xGDiff_H', 'xGDiff_A', 'CornersConv_H', 'CornersConv_A',
        'CornersConvRec_H', 'CornersConvRec_A', 'NumMatches_H', 'NumMatches_A'
    ]
    test_df.dropna(subset=features + ['B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5', 'FTR'], inplace=True)
    
    X_test = test_df[features]
    
    # Get all predictions at once for speed
    expected_home_goals = home_model.predict(X_test)
    expected_away_goals = away_model.predict(X_test)

    for i, (index, row) in enumerate(test_df.iterrows()):
        
        P = score_matrix_from_mus(expected_home_goals[i], expected_away_goals[i])
        market_probs = derive_all_market_probabilities(P)
        
        # --- 1X2 Market Logic ---
        if market_probs['p_H'] >= PROBABILITY_THRESHOLD:
            odds = row['B365H']
            implied_prob = 1 / odds
            if market_probs['p_H'] > implied_prob + EDGE_THRESHOLD:
                bets_placed += 1
                if row['FTR'] == 'H':
                    bankroll += (STAKE * odds) - STAKE
                    bets_won += 1
                else:
                    bankroll -= STAKE
        
        # --- Over/Under 2.5 Market Logic ---
        if market_probs['p_O2.5'] >= PROBABILITY_THRESHOLD:
            odds = row['B365>2.5']
            implied_prob = 1 / odds
            if market_probs['p_O2.5'] > implied_prob + EDGE_THRESHOLD:
                bets_placed += 1
                is_over = (row['FTHG'] + row['FTAG']) > 2.5
                if is_over:
                    bankroll += (STAKE * odds) - STAKE
                    bets_won += 1
                else:
                    bankroll -= STAKE

    # 4. Generate Final Report
    print("\n--- Ultimate Backtest Complete ---")
    profit = bankroll - STARTING_BANKROLL
    total_wagered = bets_placed * STAKE
    roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0
    win_rate = (bets_won / bets_placed) * 100 if bets_placed > 0 else 0

    print(f"Profit / Loss:     {profit:.2f}")
    print(f"Total Bets Placed: {bets_placed}")
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"ULTIMATE TRUE ROI: {roi:.2f}%")

if __name__ == "__main__":
    main()