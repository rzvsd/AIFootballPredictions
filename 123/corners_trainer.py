# corners_trainer.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_corners_model():
    """
    Loads enhanced data and trains an XGBoost model to predict Over/Under 10.5 corners.
    """
    LEAGUE = 'E0'
    CORNER_LINE = 10.5
    
    # --- 1. Load Enhanced Data ---
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_enhanced.csv')
    print(f"Loading enhanced data from {data_path}...")
    df = pd.read_csv(data_path)

    # --- 2. Prepare Data and Features for Corners ---
    # Create the target variable for corners
    df['Over_10.5_Corners'] = ((df['HC'] + df['AC']) > CORNER_LINE).astype(int)
    
    # Define features most relevant to predicting corners
    # We use team strength (Elo) and recent form in shots and corners
    features = [
        'HomeAvgShots_Last5', 'AwayAvgShots_Last5',
        'HomeAvgShotsOnTarget_Last5', 'AwayAvgShotsOnTarget_Last5',
        'HomeAvgCorners_Last5', 'AwayAvgCorners_Last5',
        'EloDifference'
    ]
    
    df.dropna(subset=features + ['Over_10.5_Corners'], inplace=True)

    X = df[features]
    y = df['Over_10.5_Corners']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Training Corners model on {len(X_train)} matches...")

    # --- 3. Train the XGBoost Model ---
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- 4. Evaluate the Model ---
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nCorners model trained with an accuracy of: {accuracy:.2%}")

    # --- 5. Save the Model ---
    output_dir = 'advanced_models'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{LEAGUE}_corners_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_corners_model()