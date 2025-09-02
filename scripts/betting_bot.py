# betting_bot.py (Final, Config-Driven Version)
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import requests
from requests.exceptions import RequestException
import yaml
import joblib
import numpy as np
from scipy.stats import poisson

# --- CONFIGURATION LOADER ---
CONFIG_PATH = Path(__file__).parent / "bot_config.yaml"

def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load bot configuration from a YAML file and apply environment overrides."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # API URLs
    api_config = config.get("api_credentials", {})
    api_config["odds_url"] = os.getenv("BOT_ODDS_URL", api_config.get("odds_url"))
    api_config["bet_url"] = os.getenv("BOT_BET_URL", api_config.get("bet_url"))
    config["api_credentials"] = api_config

    # Strategy settings
    config["stake_size"] = float(os.getenv("BOT_STAKE_SIZE", config.get("stake_size", 10.0)))
    config["probability_threshold"] = float(os.getenv("BOT_PROB_THRESHOLD", config.get("probability_threshold", 0.6)))
    config["edge_requirement"] = float(os.getenv("BOT_EDGE_REQUIREMENT", config.get("edge_requirement", 0.05)))
    
    # Model settings
    config["league"] = os.getenv("BOT_LEAGUE", config.get("league", "E0"))
    config["fusion_weight_dc"] = float(os.getenv("BOT_W_DC", config.get("fusion_weight_dc", 0.3)))
    
    return config

# --- HELPER & PREDICTION FUNCTIONS (from bet_fusion.py) ---
# Note: These are simplified for this script. A real implementation would import them.
TEAM_NORMALIZE = {
    "Manchester City": "Man City", "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton", "West Ham United": "West Ham", "AFC Bournemouth": "Bournemouth",
    "Sheffield Utd": "Sheffield United",
}
def norm_team(x: str) -> str:
    s = str(x).strip()
    return TEAM_NORMALIZE.get(s, s)

def score_matrix_from_mus(mu_h, mu_a):
    hg = np.arange(0, 11); ag = np.arange(0, 11)
    def pois_pmf(k, lam): return np.exp(-lam) * (lam**k) / np.array([np.math.factorial(i) for i in k])
    ph = pois_pmf(hg, mu_h); pa = pois_pmf(ag, mu_a)
    P = np.outer(ph, pa); P /= P.sum()
    return P

def derive_all_market_probabilities(P):
    home = np.tril(P, -1).sum(); draw = np.diag(P).sum(); away = np.triu(P, 1).sum()
    return {"p_H": home, "p_D": draw, "p_A": away}

def get_features_for_match(features_df, home, away):
    try:
        h = features_df[features_df["HomeTeam"] == home].tail(1).iloc[0]
        a = features_df[features_df["AwayTeam"] == away].tail(1).iloc[0]
        features = [
            'ShotConv_H', 'ShotConv_A', 'ShotConvRec_H', 'ShotConvRec_A', 'PointsPerGame_H', 'PointsPerGame_A',
            'CleanSheetStreak_H', 'CleanSheetStreak_A', 'xGDiff_H', 'xGDiff_A', 'CornersConv_H', 'CornersConv_A',
            'CornersConvRec_H', 'CornersConvRec_A', 'NumMatches_H', 'NumMatches_A'
        ]
        vals = {feat: h.get(feat) if '_H' in feat else a.get(feat) for feat in features}
        return pd.DataFrame([vals]).astype(float)
    except IndexError:
        return None

# --- MAIN BOT LOGIC ---
def generate_predictions(config: Dict[str, Any]) -> pd.DataFrame:
    """Generate match probabilities using the fusion logic."""
    league = config['league']
    dc_model = joblib.load(os.path.join("advanced_models", f"{league}_dixon_coles_model.pkl"))
    xgb_home = joblib.load(os.path.join("advanced_models", f"{league}_ultimate_xgb_home.pkl"))
    xgb_away = joblib.load(os.path.join("advanced_models", f"{league}_ultimate_xgb_away.pkl"))
    features_df = pd.read_csv(os.path.join("data", "enhanced", f"{league}_strength_adj.csv"))
    fixtures = pd.read_csv(os.path.join("data", "fixtures", f"{league}_weekly_fixtures.csv"))

    rows = []
    for _, row in fixtures.iterrows():
        home = norm_team(row["home_team"])
        away = norm_team(row["away_team"])
        
        # Get Hybrid Prediction
        match_features = get_features_for_match(features_df, home, away)
        if match_features is None: continue
        mu_h_hyb = float(xgb_home.predict(match_features)[0])
        mu_a_hyb = float(xgb_away.predict(match_features)[0])
        
        # Get DC Prediction
        try:
            mu_h_dc = float(dc_model.predict(pd.DataFrame({'team':[home],'opponent':[away],'home':[1]})).values[0])
            mu_a_dc = float(dc_model.predict(pd.DataFrame({'team':[away],'opponent':[home],'home':[0]})).values[0])
        except Exception:
            mu_h_dc, mu_a_dc = None, None

        # Fuse
        if mu_h_dc is not None:
            P_dc = score_matrix_from_mus(mu_h_dc, mu_a_dc)
            P_hyb = score_matrix_from_mus(mu_h_hyb, mu_a_hyb)
            P = config['fusion_weight_dc'] * P_dc + (1.0 - config['fusion_weight_dc']) * P_hyb
        else:
            P = score_matrix_from_mus(mu_h_hyb, mu_a_hyb)

        probs = derive_all_market_probabilities(P)
        rows.append({"home": home, "away": away, **probs})

    return pd.DataFrame(rows)

def main() -> None:
    """Entry point for the betting bot."""
    config = load_config()
    print("--- Betting Bot Started with Configuration ---")
    print(config)
    
    predictions = generate_predictions(config)
    if predictions.empty:
        print("\nNo predictions available.")
        return

    print("\n--- Analyzing Opportunities ---")
    for _, match in predictions.iterrows():
        home, away, prob_home = match["home"], match["away"], match["p_H"]
        
        # This is where the real odds fetcher would go
        # For now, we use a placeholder
        odds = 2.0 # Placeholder odds
        
        implied_prob = 1 / odds
        edge = prob_home - implied_prob
        
        print(f"{home} vs {away}: Model Prob={prob_home:.2%}, Odds={odds:.2f} (Implied: {implied_prob:.2%}), Edge={edge:.2%}")
        
        if prob_home >= config['probability_threshold'] and edge >= config['edge_requirement']:
            print(f"  -> VALUE BET FOUND! Placing bet on {home}...")
            # In a real system, you would call a place_bet() function here
        else:
            print("  -> No value.")


if __name__ == "__main__":
    main()