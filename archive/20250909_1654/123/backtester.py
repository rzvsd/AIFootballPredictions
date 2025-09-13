import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
LEAGUE = 'E0'               # The league we want to test (E0 for Premier League)
STARTING_BANKROLL = 1000    # Start with 1000 units
STAKE = 10                  # Bet 10 units on every value bet
EDGE_THRESHOLD = 0.05       # Only bet if our edge is > 5% (Model_Prob > Implied_Prob + 5%)

# --- THE SOURCE OF TRUTH: FEATURE LISTS ---
# This is the exact list of 13 features selected by data_preprocessing.py for the EPL (E0)
E0_FEATURES = [
    'Last5HomeOver2.5Perc', 'Last5AwayOver2.5Perc', 'HST', 'AST', 
    'HomeOver2.5Perc', 'AvgLast5AwayGoalsConceded', 'AwayOver2.5Perc', 
    'AvgLast5HomeGoalsScored', 'AvgLast5HomeGoalsConceded', 
    'AvgLast5AwayGoalsScored', 'MaxC>2.5', 'AvgC<2.5', 'HR'
]

def run_backtest():
    """
    Loads a trained model and historical data to simulate a betting
    strategy for a full season, calculating the final profit and ROI.
    """
    print(f"--- Starting Backtest for League: {LEAGUE} ---")

    # --- 1. Load Assets ---
    model_path = os.path.join('models', f'{LEAGUE}_voting_classifier.pkl')
    data_path = os.path.join('data', 'processed', f'{LEAGUE}_merged_preprocessed.csv')

    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find file. Make sure you have run the training scripts.")
        print(e)
        return

    print(f"Successfully loaded model and {len(df)} matches.")

    # --- 2. Prepare Data ---
    TARGET_COL = 'Over2.5'
    ODDS_OVER_COL = 'MaxC>2.5'
    ODDS_UNDER_COL = 'AvgC<2.5'
    
    df.dropna(subset=[ODDS_OVER_COL, ODDS_UNDER_COL, TARGET_COL], inplace=True)
    
    # Use the exact feature list for the selected league
    if LEAGUE == 'E0':
        features = E0_FEATURES
    else:
        print(f"Error: No feature list defined for league {LEAGUE}. Please add it.")
        return

    if not all(feature in df.columns for feature in features):
        print("Error: The data file is missing some of the required features.")
        missing = [f for f in features if f not in df.columns]
        print(f"Missing features: {missing}")
        return
        
    X = df[features]
    
    # --- 3. Run Simulation ---
    bankroll = STARTING_BANKROLL
    bets_placed = 0
    bets_won = 0
    
    print(f"\nSimulating {len(df)} matches with a starting bankroll of {bankroll}...")

    for index, row in df.iterrows():
        
        match_features = row[features].values.reshape(1, -1)
        
        probabilities = model.predict_proba(match_features)[0]
        prob_under = probabilities[0]
        prob_over = probabilities[1]

        odds_under = row[ODDS_UNDER_COL]
        odds_over = row[ODDS_OVER_COL]

        implied_prob_under = 1 / odds_under
        implied_prob_over = 1 / odds_over

        actual_result = row[TARGET_COL]
        bet_placed_this_match = False

        # Check for value on the OVER bet
        if prob_over > implied_prob_over + EDGE_THRESHOLD:
            bets_placed += 1
            bet_placed_this_match = True

            # --- SANITY CHECK CODE ---
            if bets_placed <= 5:
                print(f"\n--- Bet #{bets_placed} ---")
                print(f"Match: {row['HomeTeam']} vs {row['AwayTeam']} on {row['Date']}")
                print(f"Betting on: OVER 2.5 Goals")
                print(f"Model Probability: {prob_over:.2%}")
                print(f"Odds: {odds_over} (Implied Prob: {implied_prob_over:.2%})")
                print(f"Edge: {(prob_over - implied_prob_over):.2%}")
                print(f"Actual Result: {'Over' if actual_result == 1 else 'Under'}")
            # --- END SANITY CHECK CODE ---

            if actual_result == 1:
                bankroll += (STAKE * odds_over) - STAKE
                bets_won += 1
            else:
                bankroll -= STAKE

        # Check for value on the UNDER bet
        elif not bet_placed_this_match and prob_under > implied_prob_under + EDGE_THRESHOLD:
            bets_placed += 1

            # --- SANITY CHECK CODE ---
            if bets_placed <= 5:
                print(f"\n--- Bet #{bets_placed} ---")
                print(f"Match: {row['HomeTeam']} vs {row['AwayTeam']} on {row['Date']}")
                print(f"Betting on: UNDER 2.5 Goals")
                print(f"Model Probability: {prob_under:.2%}")
                print(f"Odds: {odds_under} (Implied Prob: {implied_prob_under:.2%})")
                print(f"Edge: {(prob_under - implied_prob_under):.2%}")
                print(f"Actual Result: {'Under' if actual_result == 0 else 'Over'}")
            # --- END SANITY CHECK CODE ---

            if actual_result == 0:
                bankroll += (STAKE * odds_under) - STAKE
                bets_won += 1
            else:
                bankroll -= STAKE

    # --- 4. Generate Final Report ---
    print("\n--- Backtest Complete ---")
    print(f"League Tested: {LEAGUE}")
    print("-" * 25)
    print(f"Starting Bankroll: {STARTING_BANKROLL:.2f}")
    print(f"Ending Bankroll:   {bankroll:.2f}")
    
    profit = bankroll - STARTING_BANKROLL
    total_wagered = bets_placed * STAKE
    roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0
    win_rate = (bets_won / bets_placed) * 100 if bets_placed > 0 else 0

    print(f"Profit / Loss:     {profit:.2f}")
    print(f"Total Bets Placed: {bets_placed}")
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"ROI:               {roi:.2f}%")
    print("-" * 25)


if __name__ == "__main__":
    run_backtest()