# custom_features/elo_calculator.py
import pandas as pd

def calculate_elo(df: pd.DataFrame, k_factor=20) -> pd.DataFrame:
    """
    Calculates Elo ratings for teams based on match results.
    """
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo_ratings = {team: 1500 for team in teams}
    
    home_elos, away_elos = [], []

    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Get current Elo ratings before the match
        home_elo_before = elo_ratings[home_team]
        away_elo_before = elo_ratings[away_team]
        
        home_elos.append(home_elo_before)
        away_elos.append(away_elo_before)

        # Calculate expected win probabilities
        prob_home_win = 1 / (1 + 10**((away_elo_before - home_elo_before) / 400))
        prob_away_win = 1 - prob_home_win

        # Determine match outcome
        if row['FTHG'] > row['FTAG']:
            home_score = 1
            away_score = 0
        elif row['FTHG'] < row['FTAG']:
            home_score = 0
            away_score = 1
        else:
            home_score = 0.5
            away_score = 0.5

        # Update Elo ratings after the match
        new_home_elo = home_elo_before + k_factor * (home_score - prob_home_win)
        new_away_elo = away_elo_before + k_factor * (away_score - prob_away_win)
        
        elo_ratings[home_team] = new_home_elo
        elo_ratings[away_team] = new_away_elo

    df['HomeElo'] = home_elos
    df['AwayElo'] = away_elos
    df['EloDifference'] = df['HomeElo'] - df['AwayElo']
    
    return df