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
    'CornersConvRec_H', 'CornersConvRec_A', 'NumMatches_H', 'NumMatches_A',
    'Elo_H', 'Elo_A', 'EloDiff'
]

# Optional opponent-Elo band features (if present)
ELO_BAND_FEATURES = [
    'GFvsMid_H', 'GAvsMid_H', 'GFvsHigh_H', 'GAvsHigh_H',
    'GFvsMid_A', 'GAvsMid_A', 'GFvsHigh_A', 'GAvsHigh_A',
]

# Kernel-similarity features (dynamic performance vs similar-Elo opponents)
ELO_SIM_FEATURES = [
    'GFvsSim_H', 'GAvsSim_H', 'GFvsSim_A', 'GAvsSim_A'
]

# Expand the ultimate features to include Elo-band and Elo-sim features
ULTIMATE_FEATURES = ULTIMATE_FEATURES + ELO_BAND_FEATURES + ELO_SIM_FEATURES

# MicroXG enrichment features (possession, corners, absences, and derived rates)
EXTRA_FEATURES = [
    # Possession EWMAs (directional)
    'Possession_H', 'Possession_A',
    'PossessionRec_H', 'PossessionRec_A',
    # Corners totals EWMAs (directional)
    'CornersFor_H', 'CornersFor_A',
    'CornersAgainst_H', 'CornersAgainst_A',
    # Availability / absences
    'Availability_H', 'Availability_A', 'AvailabilityDiff',
    # xG per possession point (rates)
    'xGpp_H', 'xGpp_A', 'xGppRec_H', 'xGppRec_A',
    # xG generated/allowed from corners
    'xGCorners_H', 'xGCorners_A', 'xGCornersRec_H', 'xGCornersRec_A',
]

# Final feature order
ULTIMATE_FEATURES = ULTIMATE_FEATURES + EXTRA_FEATURES

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
    # Bundesliga (D1)
    "Bayern Munich":             "Bayern Munich",
    "Borussia Dortmund":         "Dortmund",
    "RB Leipzig":                "RB Leipzig",
    "Bayer Leverkusen":          "Leverkusen",
    "Borussia Monchengladbach":  "M'gladbach",
    "Union Berlin":              "Union Berlin",
    "SC Freiburg":               "Freiburg",
    "VfB Stuttgart":             "Stuttgart",
    "VfL Wolfsburg":             "Wolfsburg",
    "Werder Bremen":             "Werder Bremen",
    "Eintracht Frankfurt":       "Ein Frankfurt",
    "1. FC Koln":                "FC Koln",
    "1. FSV Mainz 05":           "Mainz",
    "TSG Hoffenheim":            "Hoffenheim",
    "Hertha BSC":                "Hertha",
    "FC Augsburg":               "Augsburg",
    "VfL Bochum":                "Bochum",
    "Arminia Bielefeld":         "Bielefeld",
    "Greuther Furth":            "Greuther Furth",
    "SV Darmstadt 98":           "Darmstadt",
    "1. FC Heidenheim":          "Heidenheim",
    "Schalke 04":                "Schalke 04",
    # Ligue 1 (F1)
    "Paris Saint Germain":       "Paris SG",
    "AS Monaco":                 "Monaco",
    "Olympique Marseille":       "Marseille",
    "Olympique Lyonnais":        "Lyon",
    "Lille":                     "Lille",
    "Stade Rennais":             "Rennes",
    "Stade de Reims":            "Reims",
    "OGC Nice":                  "Nice",
    "FC Nantes":                 "Nantes",
    "Montpellier":               "Montpellier",
    "RC Lens":                   "Lens",
    "Strasbourg":                "Strasbourg",
    "Toulouse FC":               "Toulouse",
    "FC Lorient":                "Lorient",
    "St Etienne":                "St Etienne",
    "Angers":                    "Angers",
    "Brest":                     "Brest",
    "Auxerre":                   "Auxerre",
    "Metz":                      "Metz",
    "Le Havre":                  "Le Havre",
    "Clermont Foot":             "Clermont",
    "Troyes":                    "Troyes",
    "Bordeaux":                  "Bordeaux",
    # La Liga (SP1)
    "Real Madrid":               "Real Madrid",
    "Barcelona":                 "Barcelona",
    "Atletico Madrid":           "Ath Madrid",
    "Athletic Club":             "Ath Bilbao",
    "Real Sociedad":             "Sociedad",
    "Sevilla":                   "Sevilla",
    "Valencia":                  "Valencia",
    "Villarreal":                "Villarreal",
    "Real Betis":                "Betis",
    "Osasuna":                   "Osasuna",
    "Getafe":                    "Getafe",
    "Rayo Vallecano":            "Vallecano",
    "Celta Vigo":                "Celta",
    "Granada":                   "Granada",
    "Girona":                    "Girona",
    "Las Palmas":                "Las Palmas",
    "Levante":                   "Levante",
    "Cadiz":                     "Cadiz",
    "Mallorca":                  "Mallorca",
    "Elche":                     "Elche",
    "Espanyol":                  "Espanol",
    "Deportivo Alaves":          "Alaves",
    "Real Valladolid":           "Valladolid",
    # Serie A (I1)
    "Inter Milan":               "Inter",
    "Internazionale":            "Inter",
    "AC Milan":                  "Milan",
    "Juventus":                  "Juventus",
    "Napoli":                    "Napoli",
    "AS Roma":                   "Roma",
    "SS Lazio":                  "Lazio",
    "Fiorentina":                "Fiorentina",
    "Atalanta":                  "Atalanta",
    "Torino":                    "Torino",
    "Udinese":                   "Udinese",
    "Bologna":                   "Bologna",
    "Empoli":                    "Empoli",
    "Genoa":                     "Genoa",
    "Sassuolo":                  "Sassuolo",
    "Spezia":                    "Spezia",
    "Hellas Verona":             "Verona",
    "Salernitana":               "Salernitana",
    "Monza":                     "Monza",
    "Frosinone":                 "Frosinone",
    "Cremonese":                 "Cremonese",
    "Lecce":                     "Lecce",
    "Cagliari":                  "Cagliari",
    "Venezia":                   "Venezia",
}

# --- DISPERSION AND MATRIX SETTINGS ---
# Optional per-league Poisson matrix cap (can be overridden in cfg)
MAX_GOALS_PER_LEAGUE = {
    'E0': 10,
    'D1': 9,
    'F1': 8,
    'SP1': 9,
    'I1': 8,
}

# Gaussian kernel width (sigma) per league for Elo-similarity features
ELO_SIM_SIGMA_PER_LEAGUE = {
    'E0': 50.0,
    'D1': 40.0,
    'F1': 45.0,
    'SP1': 55.0,
    'I1': 45.0,
}

def normalize_team_name(api_name: str) -> str:
    """Uses the map to convert an API team name to your internal model name."""
    # Safely returns the original name if it's not found in the map
    return TEAM_NAME_MAP.get(api_name, api_name)
