# xgb_trainer.py
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor

def train_xgb_models():
    """
    Loads the ultimate strength-adjusted data and trains two XGBoost models
    on the full set of advanced features.
    """
    LEAGUE = 'E0'
    
    # --- 1. Load the Ultimate Dataset ---
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_strength_adj.csv')
    print(f"Loading ultimate feature data from {data_path}...")
    df = pd.read_csv(data_path)

    # --- 2. Define the Ultimate Feature Set ---
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
    
    home_target = 'FTHG'
    away_target = 'FTAG'
    
    required_cols = features + [home_target, away_target]
    df.dropna(subset=required_cols, inplace=True)

    X = df[features]
    y_home = df[home_target]
    y_away = df[away_target]

    print(f"Training XGBoost models on {len(X)} matches with {len(features)} features...")

    # --- 3. Train the Home Goals XGBoost Model ---
    home_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    home_model.fit(X, y_home)
    
    # --- 4. Train the Away Goals XGBoost Model ---
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
    
    # Let's give these ultimate models a new name
    home_model_path = os.path.join(output_dir, f'{LEAGUE}_ultimate_xgb_home.pkl')
    joblib.dump(home_model, home_model_path)
    print(f"\nUltimate XGBoost Home Goals model saved to {home_model_path}")

    away_model_path = os.path.join(output_dir, f'{LEAGUE}_ultimate_xgb_away.pkl')
    joblib.dump(away_model, away_model_path)
    print(f"Ultimate XGBoost Away Goals model saved to {away_model_path}")


if __name__ == "__main__":
    train_xgb_models()