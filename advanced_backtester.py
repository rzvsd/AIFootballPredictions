# advanced_backtester.py
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
LEAGUE = 'E0'
STARTING_BANKROLL = 1000
STAKE = 10
EDGE_THRESHOLD = 0.05

def run_advanced_backtest():
    """
    Loads our custom trained model and enhanced data to calculate the TRUE ROI.
    """
    print(f"--- Starting Advanced Backtest for League: {LEAGUE} ---")

    # --- 1. Load OUR Custom Assets ---
    model_path = os.path.join('advanced_models', f'{LEAGUE}_simple_goals_model.pkl')
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_enhanced.csv')

    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find file.")
        print(e)
        return

    print(f"Successfully loaded custom model and {len(df)} matches.")

    # --- 2. Prepare Data ---
    df['Over2.5'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)

    TARGET_COL = 'Over2.5'
    ODDS_OVER_COL = 'MaxC>2.5'
    ODDS_UNDER_COL = 'AvgC<2.5'
    
    features = [
        'HomeAvgGoalsScored_Last5',
        'HomeAvgGoalsConceded_Last5',
        'AwayAvgGoalsScored_Last5',
        'AwayAvgGoalsConceded_Last5'
    ]
    
    # --- CRITICAL FIX: Drop rows with missing values ---
    # This ensures the backtester sees the exact same clean data as the trainer.
    df.dropna(subset=features + [ODDS_OVER_COL, ODDS_UNDER_COL, TARGET_COL], inplace=True)
    
    X = df[features]
    
    # --- 3. Run Simulation ---
    bankroll = STARTING_BANKROLL
    bets_placed = 0
    bets_won = 0
    
    print(f"\nSimulating {len(df)} matches with a starting bankroll of {bankroll}...")

    all_probabilities = model.predict_proba(X)
    df_indices = X.index
    
    for i, index in enumerate(df_indices):
        row = df.loc[index]
        
        probabilities = all_probabilities[i]
        prob_under = probabilities[0]
        prob_over = probabilities[1]

        odds_under = row[ODDS_UNDER_COL]
        odds_over = row[ODDS_OVER_COL]

        implied_prob_under = 1 / odds_under
        implied_prob_over = 1 / odds_over

        actual_result = row[TARGET_COL]
        
        if prob_over > implied_prob_over + EDGE_THRESHOLD:
            bets_placed += 1
            if actual_result == 1:
                bankroll += (STAKE * odds_over) - STAKE
                bets_won += 1
            else:
                bankroll -= STAKE

        elif prob_under > implied_prob_under + EDGE_THRESHOLD:
            bets_placed += 1
            if actual_result == 0:
                bankroll += (STAKE * odds_under) - STAKE
                bets_won += 1
            else:
                bankroll -= STAKE

    # --- 4. Generate Final Report ---
    print("\n--- Advanced Backtest Complete ---")
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
    print(f"TRUE ROI:          {roi:.2f}%")
    print("-" * 25)


if __name__ == "__main__":
    run_advanced_backtest()