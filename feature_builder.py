"""Build final feature set for modelling.

This module previously created rolling averages and Elo ratings. It now also
adds goal-scoring rate features for fixed 15-minute intervals using historical
Understat match events.
"""

import asyncio
import os
from typing import Dict, List, Tuple

import pandas as pd
from custom_features.elo_calculator import calculate_elo
from feature_enhancer import LEAGUE_KEYS, normalize_team_names

GOAL_INTERVALS: List[Tuple[int, int]] = [
    (0, 15),
    (15, 30),
    (30, 45),
    (45, 60),
    (60, 75),
    (75, 90),
]
GOAL_INTERVAL_LABELS = [f"{a}_{b}" for a, b in GOAL_INTERVALS]


async def fetch_understat_matches(season: int, league: str) -> pd.DataFrame:
    """Return match metadata from Understat including match IDs."""
    import aiohttp
    from understat import Understat

    league_key = LEAGUE_KEYS.get(league, "epl")

    async with aiohttp.ClientSession() as session:
        u = Understat(session)
        matches = await u.get_league_results(league_key, int(season))

    rows = []
    for m in matches:
        dt = (m.get("datetime") or "")[:10]
        rows.append(
            {
                "Date": dt,
                "HomeTeam": (m.get("h") or {}).get("title"),
                "AwayTeam": (m.get("a") or {}).get("title"),
                "MatchID": m.get("id"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = normalize_team_names(df, league)
    return df


async def fetch_goal_bins(match_ids: List[int]) -> Dict[int, Tuple[Dict[str, int], Dict[str, int]]]:
    """Fetch goal timings for matches and bucket into 15-minute bins.

    Returns a mapping of match ID to a tuple: (home_bins, away_bins).
    """
    import aiohttp
    from understat import Understat

    async with aiohttp.ClientSession() as session:
        u = Understat(session)
        result: Dict[int, Tuple[Dict[str, int], Dict[str, int]]] = {}
        for mid in match_ids:
            shots = await u.get_match_shots(int(mid))
            home = {label: 0 for label in GOAL_INTERVAL_LABELS}
            away = {label: 0 for label in GOAL_INTERVAL_LABELS}
            for s in shots:
                if not s.get("isGoal"):
                    continue
                minute = int(float(s.get("minute", 0)))
                minute = min(minute, 89)  # treat stoppage time as 90th minute
                label = next(
                    (lbl for (a, b), lbl in zip(GOAL_INTERVALS, GOAL_INTERVAL_LABELS) if a <= minute < b),
                    GOAL_INTERVAL_LABELS[-1],
                )
                if s.get("h_a") == "h":
                    home[label] += 1
                else:
                    away[label] += 1
            result[mid] = (home, away)
        return result


def add_goal_rate_features(
    df: pd.DataFrame, league: str = "E0", n_matches: int = 5
) -> pd.DataFrame:
    """Add rolling goal-rate features for 15-minute intervals."""

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    seasons = sorted(df["Date"].dt.year.unique())

    # Fetch mapping to Understat match IDs
    all_matches = asyncio.run(
        asyncio.gather(*[fetch_understat_matches(season, league) for season in seasons])
    )
    mapping = pd.concat(all_matches, ignore_index=True).drop_duplicates()
    df = df.merge(mapping, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    # Fetch goal bins for all matches we have IDs for
    match_ids = df["MatchID"].dropna().astype(int).unique().tolist()
    bins_map = asyncio.run(fetch_goal_bins(match_ids))

    team_history: Dict[str, Dict[str, List[int]]] = {}
    features: List[Dict[str, float]] = []

    for _, row in df.sort_values("Date").iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        mid = row.get("MatchID")
        home_bins, away_bins = bins_map.get(mid, ({}, {}))
        match_feats: Dict[str, float] = {}

        for team, bins, prefix in [
            (home, home_bins, "Home"),
            (away, away_bins, "Away"),
        ]:
            history = team_history.setdefault(
                team, {label: [] for label in GOAL_INTERVAL_LABELS}
            )
            for label in GOAL_INTERVAL_LABELS:
                series = pd.Series(history[label])
                avg = (
                    series.rolling(window=n_matches, min_periods=1).mean().iloc[-1]
                    if not series.empty
                    else None
                )
                match_feats[
                    f"{prefix}Goals_{label}_Rate_Last{n_matches}"
                ] = avg
            for label in GOAL_INTERVAL_LABELS:
                history[label].append(bins.get(label, 0))

        features.append(match_feats)

    features_df = pd.DataFrame(features, index=df.sort_values("Date").index)
    df = pd.concat([df, features_df], axis=1)
    return df

def create_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-series-aware rolling averages for all key stats.
    """
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
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

    print("Calculating goal interval rates from Understat events...")
    goal_df = add_goal_rate_features(merged_df, league=LEAGUE)

    print("Creating rolling average features (including xG)...")
    rolling_df = create_rolling_averages(goal_df)

    print("Calculating Elo ratings...")
    final_df = calculate_elo(rolling_df)

    final_df.to_csv(output_path, index=False)

    print(f"\nFinal feature dataset saved to {output_path}")
    print("\nShowing the last 5 rows of the final dataset:")
    print(final_df.tail())