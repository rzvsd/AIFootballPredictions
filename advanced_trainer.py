# advanced_trainer.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_simple_model():
    """
    Loads our enhanced data, trains a simple model on non-leaky features,
    and saves the trained model.
    """
    LEAGUE = 'E0'
    
    # --- 1. Load Enhanced Data ---
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_enhanced.csv')
    print(f"Loading enhanced data from {data_path}...")
    df = pd.read_csv(data_path)

    # --- 2. Prepare Data and Features ---
    # The target is still Over/Under 2.5 goals
    df['Over2.5'] = ((df['FTHG'] + df['FTAG']) > 2.5).astype(int)
    
    # Our new, clean features
    features = [
        'HomeAvgGoalsScored_Last5',
        'HomeAvgGoalsConceded_Last5',
        'AwayAvgGoalsScored_Last5',
        'AwayAvgGoalsConceded_Last5'
    ]
    
    # Drop any rows that might have missing values from the rolling average calculation
    df.dropna(subset=features + ['Over2.5'], inplace=True)

    X = df[features]
    y = df['Over2.5']

    # Split data into training and testing sets to evaluate performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Training model on {len(X_train)} matches...")

    # --- 3. Train the Model ---
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # --- 4. Evaluate the Model ---
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nSimple model trained with an accuracy of: {accuracy:.2%}")

    # --- 5. Save the Model ---
    output_dir = 'advanced_models'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{LEAGUE}_simple_goals_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_simple_model()