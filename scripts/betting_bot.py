# betting_bot.py (Final, Config-Driven Version)
import os
import sys
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

# Ensure project root is importable when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.append(str(_SCRIPT_DIR.parent))

import config
from bet_fusion import (
    generate_market_book,
    attach_value_metrics,
    BankrollManager,
)

# --- CONFIGURATION LOADER ---
def _find_config_path() -> Path:
    """Locate bot_config.yaml in common locations.

    Search order:
      1) repo root (one level up from this script)
      2) this script's directory
      3) current working directory
    Falls back to script dir path even if missing (so a clear error occurs).
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "bot_config.yaml",
        script_dir / "bot_config.yaml",
        Path.cwd() / "bot_config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return script_dir / "bot_config.yaml"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load bot configuration from a YAML file and apply environment overrides."""
    cfg_path = Path(path) if path else _find_config_path()
    with open(cfg_path, "r", encoding="utf-8") as f:
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

# --- HELPER & PREDICTION FUNCTIONS ---
# Use shared team-name normalization from config to ensure one source of truth.
def norm_team(x: str) -> str:
    return config.normalize_team_name(str(x).strip())

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
    # Legacy helper (unused after fusion integration). Kept for reference.
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

def main() -> None:
    """Entry point for the betting bot."""
    config = load_config()
    print("--- Betting Bot Started with Configuration ---")
    print(config)
    
    market_df = generate_market_book(config)
    if market_df.empty:
        print("\nNo predictions available.")
        return

    df_val = attach_value_metrics(market_df, use_placeholders=True)
    # Simple thresholds for demo; can be moved to config['thresholds']
    thresholds = {
        '1X2': {'min_prob': 0.55, 'min_edge': 0.03},
        'DC': {'min_prob': 0.75, 'min_edge': 0.02},
        'OU': {'min_prob': 0.58, 'min_edge': 0.02},
        'TG Interval': {'min_prob': 0.30, 'min_edge': 0.05},
    }
    def base_key(m):
        return m.split()[0] if m.startswith('OU ') else m
    picks = df_val[
        df_val.apply(lambda r: (r['prob'] >= thresholds.get(base_key(r['market']), thresholds['1X2'])['min_prob']) and (r['edge'] >= thresholds.get(base_key(r['market']), thresholds['1X2'])['min_edge']), axis=1)
    ].copy()
    if picks.empty:
        print("\nNo value picks under current thresholds.")
        return
    picks.sort_values(['EV','prob'], ascending=[False, False], inplace=True)
    top = picks.head(15)
    print("\n--- Top Picks (placeholder odds) ---")
    for _, r in top.iterrows():
        print(f"{r['date']}  {r['home']} vs {r['away']} | {r['market']} {r['outcome']}  p={r['prob']:.2%}  odds={r['odds']:.2f}  edge={r['edge']:.2%}  EV={r['EV']:.2f}")


if __name__ == "__main__":
    main()
