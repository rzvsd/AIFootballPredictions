# betting_bot.py (Final Version with Config, Odds Fetcher, and Bankroll/Logging)
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import csv
import json

import pandas as pd
import requests
from requests.exceptions import RequestException

# Import our custom modules
import bet_fusion
from scripts import bookmaker_api

# --- CONFIGURATION LOADER ---
CONFIG_PATH = Path(__file__).parent / "bot_config.yaml"

def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load bot configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    # Apply environment overrides if necessary (good for servers)
    config["stake_size"] = float(os.getenv("BOT_STAKE_SIZE", config.get("stake_size", 10.0)))
    config["probability_threshold"] = float(os.getenv("BOT_PROB_THRESHOLD", config.get("probability_threshold", 0.6)))
    config["edge_requirement"] = float(os.getenv("BOT_EDGE_REQUIREMENT", config.get("edge_requirement", 0.05)))
    return config

# --- BANKROLL & LOGGING (Your new class) ---
class BankrollManager:
    """Simple bankroll tracker and logger for bets."""
    def __init__(self, bankroll_file: str = "data/bankroll.json", log_file: str = "data/bets_log.csv"):
        self.bankroll_path = Path(bankroll_file)
        self.log_path = Path(log_file)
        self.bankroll = self._load_bankroll()
        print(f"Bankroll initialized at: {self.bankroll:.2f}")

    def _load_bankroll(self) -> float:
        self.bankroll_path.parent.mkdir(parents=True, exist_ok=True)
        if self.bankroll_path.exists():
            with open(self.bankroll_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("bankroll", 1000.0)) # Default to 1000 if not set
        return 1000.0

    def _save_bankroll(self) -> None:
        with open(self.bankroll_path, "w", encoding="utf-8") as f:
            json.dump({"bankroll": self.bankroll}, f)

    def log_bet(self, date: str, league: str, home: str, away: str, market: str, odds: float, stake: float, prob: float, ev: float):
        """Deduct stake and log the placed bet."""
        if stake > self.bankroll:
            print(f"Warning: Insufficient bankroll ({self.bankroll:.2f}) for stake ({stake:.2f}). Skipping bet.")
            return
        self.bankroll -= stake
        self._save_bankroll()
        self._append_log(date, league, home, away, market, odds, stake, prob, ev)
        print(f"Bet Logged. New bankroll: {self.bankroll:.2f}")

    def _append_log(self, date, league, home, away, market, odds, stake, prob, ev):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["date", "league", "home_team", "away_team", "market", "odds", "stake", "model_prob", "expected_value"])
            writer.writerow([date, league, home, away, market, odds, stake, f"{prob:.4f}", f"{ev:.4f}"])

# --- MAIN BOT LOGIC ---
def main() -> None:
    """Entry point for the betting bot."""
    config = load_config()
    print("\n--- Betting Bot Started with Configuration ---")
    print(config)
    
    bankroll_manager = BankrollManager()
    
    # Generate predictions using our trusted fusion engine
    predictions = bet_fusion.generate_predictions(config)
    if predictions.empty:
        print("\nNo predictions available.")
        return

    print("\n--- Analyzing Opportunities ---")
    for _, match in predictions.iterrows():
        home, away, prob_home = match["home"], match["away"], match["p_H"]
        
        # Use our new, robust odds fetcher
        odds_dict = bookmaker_api.get_odds(config['league'], home, away)
        
        if not odds_dict or not odds_dict.get("home"):
            print(f"No odds for {home} vs {away}")
            continue
            
        odds = odds_dict["home"]
        implied_prob = 1 / odds
        edge = prob_home - implied_prob
        
        print(f"\nMatch: {home} vs {away}")
        print(f"  Model Prob (Home Win): {prob_home:.2%}")
        print(f"  Odds: {odds:.2f} (Implied Prob: {implied_prob:.2%})")
        print(f"  Edge: {edge:.2%}")
        
        # The final decision logic
        if prob_home >= config['probability_threshold'] and edge >= config['edge_requirement']:
            stake = config['stake_size']
            print(f"  -> VALUE BET FOUND! Logging bet on {home} with stake {stake:.2f}...")
            bankroll_manager.log_bet(
                date=match['date'],
                league=config['league'],
                home=home,
                away=away,
                market="HOME_WIN",
                odds=odds,
                stake=stake,
                prob=prob_home,
                ev=(prob_home * odds - 1) # Simplified EV for logging
            )
        else:
            print("  -> No value or probability threshold not met.")


if __name__ == "__main__":
    # We need to add bet_fusion to the path to import it correctly
    import sys
    sys.path.append(os.path.dirname(__file__))
    main()