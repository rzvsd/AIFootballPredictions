# corners_backtester.py
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
LEAGUE = 'E0'
CORNER_LINE = 10.5
STARTING_BANKROLL = 1000
STAKE = 10
EDGE_THRESHOLD = 0.05

def run_corners_backtest():
    """
    Loads the trained corners model and historical data to simulate a betting
    strategy for the Over/Under corners market.
    """
    print(f"--- Starting Corners Backtest for League: {LEAGUE} ---")

    # --- 1. Load Assets ---
    model_path = os.path.join('advanced_models', f'{LEAGUE}_corners_model.pkl')
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_enhanced.csv')

    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find file.")
        print(e)
        return

    print(f"Successfully loaded corners model and {len(df)} matches.")

    # --- 2. Prepare Data ---
    # We need odds for the corners market. Let's assume the columns are B365C>10.5 and B365C<10.5
    # If these don't exist, we will get a KeyError and will need to find the correct names.
    ODDS_OVER_COL = 'B365C>10.5'
    ODDS_UNDER_COL = 'B365C<10.5'
    
    df['Over_10.5_Corners'] = ((df['HC'] + df['AC']) > CORNER_LINE).astype(int)
    TARGET_COL = 'Over_10.5_Corners'
    
    df.dropna(subset=[ODDS_OVER_COL, ODDS_UNDER_COL, TARGET_COL], inplace=True)
    
    features = [
        'HomeAvgShots_Last5', 'AwayAvgShots_Last5',
        'HomeAvgShotsOnTarget_Last5', 'AwayAvgShotsOnTarget_Last5',
        'HomeAvgCorners_Last5', 'AwayAvgCorners_Last5',
        'EloDifference'
    ]
    
    df.dropna(subset=features, inplace=True)
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
    print("\n--- Corners Backtest Complete ---")
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
    run_corners_backtest()