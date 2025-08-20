# feature_enhancer.py
import pandas as pd
import os

def create_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-series-aware rolling averages for key stats.
    This function CORRECTLY calculates form without looking into the future.
    """
    
    # Ensure data is sorted chronologically
    # The dayfirst=True argument is important for UK date formats
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_stats = {}
    new_features = []

    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        if home_team not in team_stats:
            team_stats[home_team] = {'goals_scored': [], 'goals_conceded': []}
        if away_team not in team_stats:
            team_stats[away_team] = {'goals_scored': [], 'goals_conceded': []}

        home_avg_gs = pd.Series(team_stats[home_team]['goals_scored']).rolling(window=5, min_periods=1).mean().iloc[-1] if team_stats[home_team]['goals_scored'] else None
        home_avg_gc = pd.Series(team_stats[home_team]['goals_conceded']).rolling(window=5, min_periods=1).mean().iloc[-1] if team_stats[home_team]['goals_conceded'] else None
        away_avg_gs = pd.Series(team_stats[away_team]['goals_scored']).rolling(window=5, min_periods=1).mean().iloc[-1] if team_stats[away_team]['goals_scored'] else None
        away_avg_gc = pd.Series(team_stats[away_team]['goals_conceded']).rolling(window=5, min_periods=1).mean().iloc[-1] if team_stats[away_team]['goals_conceded'] else None

        new_features.append({
            'HomeAvgGoalsScored_Last5': home_avg_gs,
            'HomeAvgGoalsConceded_Last5': home_avg_gc,
            'AwayAvgGoalsScored_Last5': away_avg_gs,
            'AwayAvgGoalsConceded_Last5': away_avg_gc
        })

        team_stats[home_team]['goals_scored'].append(row['FTHG'])
        team_stats[home_team]['goals_conceded'].append(row['FTAG'])
        team_stats[away_team]['goals_scored'].append(row['FTAG'])
        team_stats[away_team]['goals_conceded'].append(row['FTHG'])

    new_features_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)
    
    return df

if __name__ == "__main__":
    
    LEAGUE = 'E0'
    raw_data_path = os.path.join('data', 'raw', f'{LEAGUE}_merged.csv')
    
    print(f"Loading raw data for {LEAGUE}...")
    raw_df = pd.read_csv(raw_data_path)

    # --- NEW: Define columns to keep from the raw data ---
    # We need identifiers, results, and the odds columns we plan to use.
    columns_to_keep = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
        'MaxC>2.5', 'AvgC<2.5' 
    ]
    
    # Filter the raw dataframe to only keep necessary columns
    filtered_df = raw_df[columns_to_keep].copy()

    print("Creating non-leaky rolling average features...")
    enhanced_df = create_rolling_averages(filtered_df)

    # Save the new, enhanced dataframe
    output_dir = os.path.join('data', 'enhanced')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{LEAGUE}_enhanced.csv')
    enhanced_df.to_csv(output_path, index=False)

    print(f"\nEnhanced data saved to {output_path}")
    print("\nShowing the last 5 rows with new features and kept columns:")
    print(enhanced_df.tail())