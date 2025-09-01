# config.py

# --- API & LEAGUE SETTINGS ---
# Operating under the context of the 2025-2026 Season.
LEAGUE_CODE = "E0"  # English Premier League
LEAGUE_ID_API = 39  # The ID for the Premier League in the API-Football

# --- MODEL & DATA PATHS ---
MODELS_DIR = "advanced_models"
DATA_STORE_DIR = "data/store"

HOME_MODEL_PATH = f"{MODELS_DIR}/{LEAGUE_CODE}_ultimate_xgb_home.pkl"
AWAY_MODEL_PATH = f"{MODELS_DIR}/{LEAGUE_CODE}_ultimate_xgb_away.pkl"
STATS_SNAPSHOT_PATH = f"{DATA_STORE_DIR}/{LEAGUE_CODE}_latest_team_stats.parquet"

# --- FEATURE LIST ---
# This MUST match the features the XGB models were trained on
ULTIMATE_FEATURES = [
    'ShotConv_H', 'ShotConv_A', 'ShotConvRec_H', 'ShotConvRec_A',
    'PointsPerGame_H', 'PointsPerGame_A', 'CleanSheetStreak_H', 'CleanSheetStreak_A',
    'xGDiff_H', 'xGDiff_A', 'CornersConv_H', 'CornersConv_A',
    'CornersConvRec_H', 'CornersConvRec_A', 'NumMatches_H', 'NumMatches_A'
]

# --- TEAM NAME NORMALIZATION MAP ---
# This map is CRITICAL. It translates API names (left) to your dataset names (right).
# This has been pre-filled based on your dataset list and standard API names.
TEAM_NAME_MAP = {
    # API Name (Standard)       : Your Dataset Name (From your list)
    "Arsenal":                  "Arsenal",
    "Aston Villa":              "Aston Villa",
    "AFC Bournemouth":          "Bournemouth",
    "Brentford":                "Brentford",
    "Brighton and Hove Albion": "Brighton",
    "Burnley":                  "Burnley",
    "Chelsea":                  "Chelsea",
    "Crystal Palace":           "Crystal Palace",
    "Everton":                  "Everton",
    "Fulham":                   "Fulham",
    "Leeds United":             "Leeds",
    "Leicester City":           "Leicester",
    "Liverpool":                "Liverpool",
    "Luton Town":               "Luton",
    "Manchester City":          "Man City",
    "Manchester United":        "Man United",
    "Newcastle United":         "Newcastle",
    "Norwich City":             "Norwich",
    "Nottingham Forest":        "Nott'm Forest",
    "Sheffield United":         "Sheffield United",
    "Southampton":              "Southampton",
    "Tottenham Hotspur":        "Tottenham",
    "Watford":                  "Watford",
    "West Ham United":          "West Ham",
    "Wolverhampton Wanderers":  "Wolves",
}

def normalize_team_name(api_name: str) -> str:
    """Uses the map to convert an API team name to your internal model name."""
    # Safely returns the original name if it's not found in the map
    return TEAM_NAME_MAP.get(api_name, api_name)