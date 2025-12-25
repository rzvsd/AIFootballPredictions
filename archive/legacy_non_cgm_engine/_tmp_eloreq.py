import pandas as pd

df=pd.read_csv("data/enhanced/cgm_match_history_with_elo.csv")
df['datetime']=pd.to_datetime(df['date'])
for team in ["Liverpool","Sunderland","Arsenal"]:
    games=df[(df.home==team)|(df.away==team)].copy()
    games.sort_values('datetime', inplace=True)
    if games.empty:
        print(f"{team}: no games")
        continue
    last=games.iloc[-1]
    is_home = last.home==team
    elo = last.elo_home_calc if is_home else last.elo_away_calc
    opp = last.away if is_home else last.home
    print(f"{team}: last game {last.date} vs {opp}, elo={elo:.2f}")
