# feature_builder.py
import pandas as pd
import os
from custom_features.elo_calculator import calculate_elo

def create_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-series-aware rolling averages for all key stats.
    """
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_stats = {}
    new_features = []

    stats_to_average = {
        'goals_scored': ('FTHG', 'FTAG'),
        'goals_conceded': ('FTAG', 'FTHG'),
        'xg_for': ('Home_xG', 'Away_xG'),
        'xg_against': ('Away_xG', 'Home_xG'),
        'shots': ('HS', 'AS'),
        'shots_on_target': ('HST', 'AST'),
        'corners': ('HC', 'AC')
    }

    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        current_match_features = {}

        for team, is_home in [(home_team, True), (away_team, False)]:
            if team not in team_stats:
                team_stats[team] = {key: [] for key in stats_to_average.keys()}

            prefix = 'Home' if is_home else 'Away'
            
            for stat_name, (home_col, away_col) in stats_to_average.items():
                stat_series = pd.Series(team_stats[team][stat_name])
                rolling_avg = stat_series.rolling(window=5, min_periods=1).mean().iloc[-1] if not stat_series.empty else None
                current_match_features[f'{prefix}Avg{stat_name.replace("_", " ").title().replace(" ", "")}_Last5'] = rolling_avg

        new_features.append(current_match_features)

        if pd.notna(row['Home_xG']) and pd.notna(row['Away_xG']):
            for stat_name, (home_col, away_col) in stats_to_average.items():
                team_stats[home_team][stat_name].append(row[home_col])
                team_stats[away_team][stat_name].append(row[away_col])

    new_features_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)
    
    return df

if __name__ == "__main__":
    
    LEAGUE = 'E0'
    input_path = os.path.join('data', 'enhanced', f'{LEAGUE}_enhanced_with_xg.csv')
    output_path = os.path.join('data', 'enhanced', f'{LEAGUE}_final_features.csv')
    
    print(f"Loading merged data for {LEAGUE} from {input_path}...")
    merged_df = pd.read_csv(input_path)

    print("Creating rolling average features (including xG)...")
    rolling_df = create_rolling_averages(merged_df)

    print("Calculating Elo ratings...")
    final_df = calculate_elo(rolling_df)

    final_df.to_csv(output_path, index=False)

    print(f"\nFinal feature dataset saved to {output_path}")
    print("\nShowing the last 5 rows of the final dataset:")
    print(final_df.tail())