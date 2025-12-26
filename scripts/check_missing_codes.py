import pandas as pd
import numpy as np

df = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')

# Check for matches where Man Utd has missing code
mu_matches = df[(df['home'] == 'Man Utd') | (df['away'] == 'Man Utd')].copy()

# Check if code_home or code_away is missing/NaN for Man Utd
def check_mu_code(row):
    is_home = row['home'] == 'Man Utd'
    if is_home:
        return pd.isna(row['code_home']) or str(row['code_home']).strip() == '' or str(row['code_home']).lower() in ['nan', 'none']
    else:
        return pd.isna(row['code_away']) or str(row['code_away']).strip() == '' or str(row['code_away']).lower() in ['nan', 'none']

mu_matches['code_missing'] = mu_matches.apply(check_mu_code, axis=1)

print("Man Utd matches with missing team code:")
missing = mu_matches[mu_matches['code_missing']]
print(f"Total: {len(missing)} out of {len(mu_matches)}")
if len(missing) > 0:
    print(missing[['date', 'home', 'away', 'code_home', 'code_away']].head(20).to_string())
else:
    print("None found - all matches have codes")

# Check the actual codes used
print("\n\nDistinct code values for Man Utd:")
home_codes = mu_matches[mu_matches['home'] == 'Man Utd']['code_home'].unique()
away_codes = mu_matches[mu_matches['away'] == 'Man Utd']['code_away'].unique()
print(f"As home: {home_codes}")
print(f"As away: {away_codes}")
