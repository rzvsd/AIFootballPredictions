# feature_store.py
import pandas as pd
import os
import json
import numpy as np

def build_feature_store():
    LEAGUE = 'E0'
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_strength_adj.csv')
    output_path = os.path.join('data', 'enhanced', f'{LEAGUE}_feature_store.json')

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    feature_store = {}

    print(f"Building feature store for {len(all_teams)} teams...")

    for team in all_teams:
        last_home_game = df[df['HomeTeam'] == team].sort_values('Date').tail(1)
        last_away_game = df[df['AwayTeam'] == team].sort_values('Date').tail(1)
        
        if last_home_game.empty and last_away_game.empty: continue

        if last_home_game.empty: latest_game, is_home = last_away_game.iloc[0], False
        elif last_away_game.empty: latest_game, is_home = last_home_game.iloc[0], True
        else:
            if last_home_game['Date'].iloc[0] > last_away_game['Date'].iloc[0]:
                latest_game, is_home = last_home_game.iloc[0], True
            else:
                latest_game, is_home = last_away_game.iloc[0], False

        team_features = {}
        prefix = 'H' if is_home else 'A'
        
        for col in df.columns:
            if col.endswith(f'_{prefix}'):
                feature_name = col[:-2]
                value = latest_game[col]
                if isinstance(value, np.integer): team_features[feature_name] = int(value)
                elif isinstance(value, np.floating): team_features[feature_name] = float(value)
                else: team_features[feature_name] = value
        
        feature_store[team] = team_features

    with open(output_path, 'w') as f:
        json.dump(feature_store, f, indent=4)
        
    print(f"Feature store successfully built and saved to {output_path}")

if __name__ == "__main__":
    build_feature_store()