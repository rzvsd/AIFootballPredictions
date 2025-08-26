# poisson_trainer.py
import pandas as pd
import joblib
import os
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance

def train_poisson_models():
    """
    Loads enhanced data and trains two separate Poisson Regressor models:
    1. To predict Full Time Home Goals (FTHG)
    2. To predict Full Time Away Goals (FTAG)
    """
    LEAGUE = 'E0'
    
    # --- 1. Load Enhanced Data ---
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_enhanced.csv')
    print(f"Loading enhanced data from {data_path}...")
    df = pd.read_csv(data_path)

    # --- 2. Prepare Data and Features ---
    # Define the features we will use to predict goals
    features = [
        'HomeAvgGoalsScored_Last5', 'HomeAvgGoalsConceded_Last5',
        'AwayAvgGoalsScored_Last5', 'AwayAvgGoalsConceded_Last5',
        'HomeAvgShots_Last5', 'AwayAvgShots_Last5',
        'HomeAvgShotsOnTarget_Last5', 'AwayAvgShotsOnTarget_Last5',
        'HomeAvgCorners_Last5', 'AwayAvgCorners_Last5',
        'EloDifference'
    ]
    
    # Define our two separate targets
    home_target = 'FTHG'
    away_target = 'FTAG'
    
    df.dropna(subset=features + [home_target, away_target], inplace=True)

    X = df[features]
    y_home = df[home_target]
    y_away = df[away_target]

    # --- 3. Train the Home Goals Model ---
    print(f"\nTraining Home Goals (FTHG) model on {len(X)} matches...")
    home_model = PoissonRegressor(alpha=1.0) # alpha is for regularization
    home_model.fit(X, y_home)
    
    # --- 4. Train the Away Goals Model ---
    print(f"Training Away Goals (FTAG) model on {len(X)} matches...")
    away_model = PoissonRegressor(alpha=1.0)
    away_model.fit(X, y_away)

    # --- 5. Save Both Models ---
    output_dir = 'advanced_models'
    os.makedirs(output_dir, exist_ok=True)
    
    home_model_path = os.path.join(output_dir, f'{LEAGUE}_poisson_home_model.pkl')
    joblib.dump(home_model, home_model_path)
    print(f"\nHome goals model saved to {home_model_path}")

    away_model_path = os.path.join(output_dir, f'{LEAGUE}_poisson_away_model.pkl')
    joblib.dump(away_model, away_model_path)
    print(f"Away goals model saved to {away_model_path}")


if __name__ == "__main__":
    train_poisson_models()