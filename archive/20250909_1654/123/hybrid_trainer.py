# hybrid_trainer.py
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor # We use a Regressor to predict a number (goals)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_hybrid_xgboost_models():
    """
    Loads enhanced data and trains two separate XGBoost Regressor models:
    1. To predict Full Time Home Goals (FTHG) based on our advanced features.
    2. To predict Full Time Away Goals (FTAG) based on our advanced features.
    These models will serve as the "Feature Brain" of our system.
    """
    LEAGUE = 'E0'
    
    # --- 1. Load STRENGTH-ADJUSTED Data ---
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_strength_adj.csv')
    print(f"Loading strength-adjusted data from {data_path}...")
    df = pd.read_csv(data_path)

    # --- 2. Prepare Data and ENHANCED Features ---
    features = [
        # Keep the best of the old
        'HomeAvgXgFor_Last5', 'HomeAvgXgAgainst_Last5',
        'AwayAvgXgFor_Last5', 'AwayAvgXgAgainst_Last5',
        'EloDifference',
        # Add the new strength-adjusted features
        'ShotConv_H', 'ShotConv_A',
        'ShotConvRec_H', 'ShotConvRec_A',
        'PointsPerGame_H', 'PointsPerGame_A',
        'CleanSheetStreak_H', 'CleanSheetStreak_A',
        'xGDiff_H', 'xGDiff_A',
        'CornersConv_H', 'CornersConv_A',
        'CornersConvRec_H', 'CornersConvRec_A'
    ]
    
    home_target = 'FTHG'
    away_target = 'FTAG'
    
    df.dropna(subset=features + [home_target, away_target], inplace=True)

    X = df[features]
    y_home = df[home_target]
    y_away = df[away_target]

    # --- 3. Train the Home Goals XGBoost Model ---
    print(f"\nTraining XGBoost Home Goals (FTHG) model on {len(X)} matches...")
    home_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    home_model.fit(X, y_home)
    
    # --- 4. Train the Away Goals XGBoost Model ---
    print(f"Training XGBoost Away Goals (FTAG) model on {len(X)} matches...")
    away_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    away_model.fit(X, y_away)

    # --- 5. Save Both XGBoost Models ---
    output_dir = 'advanced_models'
    os.makedirs(output_dir, exist_ok=True)
    
    home_model_path = os.path.join(output_dir, f'{LEAGUE}_xgb_home_model.pkl')
    joblib.dump(home_model, home_model_path)
    print(f"\nXGBoost Home Goals model saved to {home_model_path}")

    away_model_path = os.path.join(output_dir, f'{LEAGUE}_xgb_away_model.pkl')
    joblib.dump(away_model, away_model_path)
    print(f"XGBoost Away Goals model saved to {away_model_path}")


if __name__ == "__main__":
    train_hybrid_xgboost_models()