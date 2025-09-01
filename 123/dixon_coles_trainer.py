# dixon_coles_trainer.py
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import joblib
import os

def train_dixon_coles_model():
    """
    Loads enhanced data and trains a Dixon-Coles style model using
    statsmodels to predict football match outcomes.
    """
    LEAGUE = 'E0'
    
    # --- 1. Load FINAL FEATURE Data ---
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_strength_adj.csv') # <-- UPDATED
    print(f"Loading final feature data from {data_path}...")
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    # --- 2. Restructure Data for Modeling ---
    # We need to structure the data so that each match has two rows:
    # one for the home team's perspective, and one for the away team's.
    home_data = df[['HomeTeam', 'AwayTeam', 'FTHG']].rename(
        columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'FTHG': 'goals'}
    )
    home_data['home'] = 1

    away_data = df[['AwayTeam', 'HomeTeam', 'FTAG']].rename(
        columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'FTAG': 'goals'}
    )
    away_data['home'] = 0

    model_data = pd.concat([home_data, away_data])

    print(f"Data restructured for modeling. Total rows: {len(model_data)}")

    # --- 3. Define and Train the Dixon-Coles Model ---
    # The formula tells the model:
    # "Goals depend on the team's strength, the opponent's strength, and whether the team is playing at home."
    # This is a simplified Dixon-Coles model focusing on attack, defense, and home advantage.
    poisson_model = smf.glm(
        formula="goals ~ home + team + opponent", 
        data=model_data, 
        family=sm.families.Poisson()
    ).fit()

    print("\nDixon-Coles style model trained successfully.")
    print(poisson_model.summary())

    # --- 4. Save the Model ---
    output_dir = 'advanced_models'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{LEAGUE}_dixon_coles_model.pkl')
    joblib.dump(poisson_model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train_dixon_coles_model()