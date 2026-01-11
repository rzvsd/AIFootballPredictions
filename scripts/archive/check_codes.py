import pandas as pd

df = pd.read_csv('data/enhanced/cgm_match_history_with_elo_stats_xg.csv')

# Check the Nottingham vs Man Utd match (index 2754)
row = df.loc[2754]
print("Nottingham vs Man Utd match:")
print(f"  home={row['home']}, code_home={row['code_home']}")
print(f"  away={row['away']}, code_away={row['code_away']}")
print(f"  elo_home_calc={row['elo_home_calc']:.1f}, elo_away_calc={row['elo_away_calc']:.1f}")

# Check if 1037 (Nottingham) has separate rating from name-based
print()

# Check previous Man Utd match to see what Elo was after Brighton
prev_idx = 2743  # Brighton match
prev_row = df.loc[prev_idx]
print(f"Previous Man Utd match (idx {prev_idx}):")
print(f"  {prev_row['home']} vs {prev_row['away']}")
print(f"  elo_home_calc={prev_row['elo_home_calc']:.1f}")
print(f"  Result: {int(prev_row['ft_home'])}-{int(prev_row['ft_away'])}")

# Check all matches between Brighton and Nottingham to see Elo flow
print()
print("Matches between Man Utd's Oct 25 and Nov 01 games:")
subset = df[(df.index > 2743) & (df.index <= 2754)]
for idx, r in subset.iterrows():
    print(f"  {idx}: {r['home'][:12].ljust(12)} vs {r['away'][:12].ljust(12)} | code_h={r['code_home']}, code_a={r['code_away']}")
