# fusion_backtester.py
import pandas as pd
import numpy as np
import os
import joblib
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf
from xgboost import XGBRegressor
from typing import Dict

# --- CONFIGURATION ---
LEAGUE = 'E0'
SEASON_TO_TEST = 2023
STARTING_BANKROLL = 1000
STAKE = 10

# --- FUSION WEIGHTS ---
WEIGHTS = {
    "w_dc_matrix": 0.70, "w_model_prob": 0.70, "w_signal_bias": 0.30,
    "form_w": 0.55, "tempo_w": 0.25, "elo_w": 0.20,
    "slope_strength": 1.6, "slope_tempo": 1.2, "slope_elo": 1.2,
}

# --- THRESHOLDS ---
MIN_PROBABILITY = 0.55
MIN_CONFIDENCE = 0.65

# --- TEAM NAME NORMALIZATION (add to top of fusion_backtester.py) ---
TEAM_NORMALIZE = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton",
    "Sheffield United": "Sheffield United",  # keep, but ensure training includes them
    # add any other variants you've seen
}

def norm_team_name(x):
    if pd.isna(x): return x
    s = str(x).strip()
    return TEAM_NORMALIZE.get(s, s)

# --- HELPERS ---
def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))

def score_matrix_from_mus(mu_h: float, mu_a: float, max_goals: int = 10) -> np.ndarray:
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)
    def pois_pmf(k_arr, lam):
        k_arr = np.asarray(k_arr, dtype=float)
        fact = np.array([np.math.factorial(int(k)) for k in k_arr], dtype=float)
        # safe pmf
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            result = np.exp(-lam) * (lam ** k_arr) / fact
            result[np.isnan(result)] = 0.0
        return result
    ph = pois_pmf(hg, mu_h)
    pa = pois_pmf(ag, mu_a)
    P = np.outer(ph, pa)
    total = P.sum()
    if total <= 0:
        # fallback tiny uniform
        P = np.ones_like(P)
        total = P.sum()
    P /= total
    return P

def probs_1x2_from_matrix(P: np.ndarray) -> Dict[str, float]:
    return {"home": float(np.tril(P, -1).sum()), "draw": float(np.diag(P).sum()), "away": float(np.triu(P, 1).sum())}

def team_strength(row, prefix):
    # accept either exact column names or case-insensitive variants
    xg_for = row.get(f'{prefix}AvgXgFor_Last5', 0) or row.get(f'{prefix}AvgXGFor_Last5', 0) or 0
    xg_against = row.get(f'{prefix}AvgXgAgainst_Last5', 0) or 0
    goal_for = row.get(f'{prefix}AvgGoalsScored_Last5', 0) or 0
    goal_against = row.get(f'{prefix}AvgGoalsConceded_Last5', 0) or 0
    sot = row.get(f'{prefix}AvgShotsOnTarget_Last5', 0) or 0
    xg_diff = xg_for - xg_against
    goal_diff = goal_for - goal_against
    sot_diff = sot
    return (0.5 * xg_diff) + (0.5 * goal_diff) + (0.1 * sot_diff)

def signal_bias(home_strength, away_strength, elo_diff):
    strength_diff = home_strength - away_strength
    b_strength = sigmoid(WEIGHTS["slope_strength"] * strength_diff)
    b_elo_home = sigmoid(WEIGHTS["slope_elo"] * (elo_diff / 200.0)) if not pd.isna(elo_diff) else 0.5
    return (WEIGHTS["form_w"] * b_strength) + (WEIGHTS["elo_w"] * b_elo_home)

def fuse_prob_with_bias(model_prob: float, bias_score: float) -> float:
    return float(np.clip(WEIGHTS["w_model_prob"] * model_prob + WEIGHTS["w_signal_bias"] * bias_score, 0.0, 1.0))

# --- UTILS to match columns robustly ---
def find_column(df, target_name):
    # case-insensitive match
    cols = {c.lower(): c for c in df.columns}
    return cols.get(target_name.lower())

def ensure_numeric_columns(df, feature_list):
    actual_cols = []
    missing = []
    for f in feature_list:
        c = find_column(df, f)
        if c is None:
            missing.append(f)
        else:
            # coerce to numeric
            df[c] = pd.to_numeric(df[c], errors='coerce')
            actual_cols.append(c)
    if missing:
        raise ValueError(f"Missing required feature columns in CSV: {missing}")
    return actual_cols

# --- MAIN BACKTEST FUNCTION ---
def run_fusion_backtest():
    print(f"--- Starting Fusion Backtest for League: {LEAGUE}, Season: {SEASON_TO_TEST} ---")

    # 1. Load Data
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_final_features.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Enhanced data not found: {data_path}")
    df = pd.read_csv(data_path)
    if 'Date' not in df.columns and 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Normalize team names to ensure consistency between training and test
    df['HomeTeam'] = df['HomeTeam'].map(norm_team_name)
    df['AwayTeam'] = df['AwayTeam'].map(norm_team_name)

    # canonical features (case-insensitive matching will be performed)
    FEATURES = [
        'HomeAvgGoalsScored_Last5', 'HomeAvgGoalsConceded_Last5', 'AwayAvgGoalsScored_Last5', 'AwayAvgGoalsConceded_Last5',
        'HomeAvgXgFor_Last5', 'HomeAvgXgAgainst_Last5', 'AwayAvgXgFor_Last5', 'AwayAvgXgAgainst_Last5',
        'HomeAvgShots_Last5', 'AwayAvgShots_Last5', 'HomeAvgShotsOnTarget_Last5', 'AwayAvgShotsOnTarget_Last5',
        'HomeAvgCorners_Last5', 'AwayAvgCorners_Last5', 'EloDifference'
    ]

    # make sure features are present and numeric, get actual column names used
    try:
        feature_cols = ensure_numeric_columns(df, FEATURES)
    except ValueError as e:
        print("[error] Feature check failed:", e)
        raise

    # report NaNs and drop rows missing any of the numeric features
    before_count = len(df)
    df = df.dropna(subset=feature_cols + ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], how='any')
    after_count = len(df)
    print(f"Loaded {before_count} rows; {before_count-after_count} rows removed because of missing core features. Remaining: {after_count}")

    # 2. Split Data into Train and Test
    train_df = df[df['Date'].dt.year < SEASON_TO_TEST]
    test_df = df[df['Date'].dt.year >= SEASON_TO_TEST]
    print(f"Training on {len(train_df)} matches, testing on {len(test_df)} matches.")

    if len(train_df) < 10 or len(test_df) == 0:
        raise SystemExit("Not enough data for training/testing after filtering. Check enhanced CSV and feature presence.")

    # 3. Train Models on the Training Data
    print("Training models on historical data...")
    home_data = train_df[['HomeTeam', 'AwayTeam', 'FTHG']].rename(columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'FTHG': 'goals'})
    home_data['home'] = 1
    away_data = train_df[['AwayTeam', 'HomeTeam', 'FTAG']].rename(columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'FTAG': 'goals'})
    away_data['home'] = 0
    dc_model_data = pd.concat([home_data, away_data])
    
    # after creating dc_model_data (training):
    team_levels = sorted(dc_model_data['team'].unique())
    opponent_levels = team_levels  # same set in DC formulation
    
    dc_model = smf.glm("goals ~ home + team + opponent", data=dc_model_data, family=sm.families.Poisson()).fit()

    X_train = train_df[feature_cols]
    # ensure X_train numeric dtype
    X_train = X_train.astype(float)
    xgb_home_model = XGBRegressor(objective='reg:squarederror', random_state=42).fit(X_train, train_df['FTHG'])
    xgb_away_model = XGBRegressor(objective='reg:squarederror', random_state=42).fit(X_train, train_df['FTAG'])
    print("All models trained successfully.")

    # 4. Simulate the Test Season
    bankroll = STARTING_BANKROLL
    bets_placed = 0
    bets_won = 0

    print(f"\nSimulating {len(test_df)} matches for the {SEASON_TO_TEST} season...")

    for index, row in test_df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']

        # Dixon-Coles predictions with categorical levels
        try:
            pred_df_home = pd.DataFrame({"team":[home_team], "opponent":[away_team], "home":[1]})
            pred_df_home["team"] = pd.Categorical(pred_df_home["team"], categories=team_levels)
            pred_df_home["opponent"] = pd.Categorical(pred_df_home["opponent"], categories=opponent_levels)

            pred_df_away = pd.DataFrame({"team":[away_team], "opponent":[home_team], "home":[0]})
            pred_df_away["team"] = pd.Categorical(pred_df_away["team"], categories=team_levels)
            pred_df_away["opponent"] = pd.Categorical(pred_df_away["opponent"], categories=opponent_levels)

            mu_h_dc = dc_model.predict(pred_df_home).values[0]
            mu_a_dc = dc_model.predict(pred_df_away).values[0]
        except Exception as e:
            print(f"[debug] DC predict failed for {home_team} vs {away_team}: {e}. Skipping match.")
            continue
        P_dc = score_matrix_from_mus(mu_h_dc, mu_a_dc)

        # Hybrid / XGB predictions: build numeric row for features
        X_test_row = row[feature_cols].to_frame().T
        try:
            X_test_row = X_test_row.astype(float)
        except Exception as e:
            print(f"[debug] Feature conversion failed for match {home_team} vs {away_team}: {e}. Skipping match.")
            continue

        try:
            mu_h_hyb = float(xgb_home_model.predict(X_test_row)[0])
            mu_a_hyb = float(xgb_away_model.predict(X_test_row)[0])
        except Exception as e:
            print(f"[debug] XGB predict failed for {home_team} vs {away_team}: {e}. Skipping match.")
            continue

        P_hyb = score_matrix_from_mus(mu_h_hyb, mu_a_hyb)

        w = WEIGHTS["w_dc_matrix"]
        P_fused = w * P_dc + (1.0 - w) * P_hyb

        p1x2 = probs_1x2_from_matrix(P_fused)

        home_strength = team_strength(row, 'Home')
        away_strength = team_strength(row, 'Away')
        elo_val = row.get('EloDifference', np.nan)
        bias_home = signal_bias(home_strength, away_strength, elo_val)

        home_conf = fuse_prob_with_bias(p1x2["home"], bias_home)
        away_conf = fuse_prob_with_bias(p1x2["away"], 1.0 - bias_home)

        odds_h = row.get('B365H', np.nan)
        odds_a = row.get('B365A', np.nan)
        actual_result = row.get('FTR', None)

        # Place bets per your rules
        if p1x2["home"] >= MIN_PROBABILITY and home_conf >= MIN_CONFIDENCE:
            bets_placed += 1
            bankroll -= STAKE
            if actual_result == 'H':
                if pd.notna(odds_h):
                    bankroll += STAKE * float(odds_h)
                bets_won += 1

        if p1x2["away"] >= MIN_PROBABILITY and away_conf >= MIN_CONFIDENCE:
            bets_placed += 1
            bankroll -= STAKE
            if actual_result == 'A':
                if pd.notna(odds_a):
                    bankroll += STAKE * float(odds_a)
                bets_won += 1

    # 5. Generate Final Report
    print("\n--- Fusion Model Backtest Complete ---")
    print(f"League: {LEAGUE}, Season Tested: {SEASON_TO_TEST}")
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
    print(f"TRUE FUSION ROI:   {roi:.2f}%")
    print("-" * 25)

if __name__ == "__main__":
    run_fusion_backtest()
