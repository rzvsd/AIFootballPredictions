#!/usr/bin/env python3
"""
Accuracy check: compare bot predictions vs actual results for the last completed round.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load enhanced history (has actual results)
hist = pd.read_csv(ROOT / "data" / "enhanced" / "cgm_match_history_with_elo_stats_xg.csv")
hist["datetime"] = pd.to_datetime(hist["datetime"])
hist = hist.sort_values("datetime")

# Find the last 2 match-day clusters
unique_dates = sorted(hist["datetime"].dt.date.unique())
last_date = unique_dates[-1]

# Group recent rounds
from datetime import timedelta
round_start = last_date - timedelta(days=3)
last_round = hist[hist["datetime"].dt.date >= round_start].copy()

print("=" * 110)
print(f"LAST COMPLETED ROUND: {round_start} to {last_date} ({len(last_round)} matches)")
print("=" * 110)

# Load the current predictions (for the NEXT round Feb 21-22)
pred_path = ROOT / "reports" / "cgm_upcoming_predictions.csv"
preds = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()

print(f"\n--- ACTUAL RESULTS (last round with data) ---\n")

for _, r in last_round.iterrows():
    h = r["home"]
    a = r["away"]
    fh = int(r["ft_home"])
    fa = int(r["ft_away"])
    total = fh + fa
    ou = "OVER" if total > 2 else "UNDER"
    btts = "YES" if fh > 0 and fa > 0 else "NO"
    elo_h = r.get("elo_home", np.nan)
    elo_a = r.get("elo_away", np.nan)
    xg_h = r.get("xg_proxy_H", np.nan)
    xg_a = r.get("xg_proxy_A", np.nan)
    
    dt_str = r["datetime"].strftime("%m-%d %H:%M")
    print(f"{dt_str} | {h:>25s} {fh}-{fa} {a:<25s} | Tot={total} {ou:>5s} | BTTS={btts}")
    print(f"           Pre-match Elo: H={elo_h:.0f} A={elo_a:.0f}  |  xG_proxy: H={xg_h:.2f} A={xg_a:.2f}")

# Summary
total_games = len(last_round)
total_goals = (last_round["ft_home"] + last_round["ft_away"]).sum()
avg_goals = total_goals / total_games if total_games > 0 else 0
overs = ((last_round["ft_home"] + last_round["ft_away"]) > 2).sum()
btts_yes = ((last_round["ft_home"] > 0) & (last_round["ft_away"] > 0)).sum()

print(f"\n  Games={total_games} | Goals={int(total_goals)} | Avg/game={avg_goals:.1f} | OVER={overs} UNDER={total_games-overs} | BTTS_Y={btts_yes} BTTS_N={total_games-btts_yes}")

# Now show the CURRENT predictions (for the upcoming round)
print(f"\n{'='*110}")
print(f"BOT PREDICTIONS FOR NEXT ROUND (Feb 21-22) — {len(preds)} matches")
print(f"{'='*110}\n")

if not preds.empty:
    for _, p in preds.iterrows():
        h = p["home"]
        a = p["away"]
        mu_h = p["mu_home"]
        mu_a = p["mu_away"]
        mu_t = p["mu_total"]
        p_over = p.get("p_over25", 0)
        p_btts_y = p.get("p_btts_yes", 0)
        bot_ou = "OVER" if p_over > 0.5 else "UNDER"
        bot_btts = "YES" if p_btts_y > 0.5 else "NO"
        
        print(f"{h:>25s} vs {a:<25s}")
        print(f"  mu_home={mu_h:.2f}  mu_away={mu_a:.2f}  mu_total={mu_t:.2f}")
        print(f"  Predicted: {bot_ou} 2.5 (p={p_over:.1%})  BTTS={bot_btts} (p={p_btts_y:.1%})")
        print()

# KEY ANALYSIS: Check the bot's mu values against reality
print("=" * 110)
print("KEY DIAGNOSTIC: How good are the mu (expected goals) vs reality?")
print("=" * 110)
print()

# For the last round, get mu predictions (if they were stored)
# The enhanced history has the training features, not the live predictions.
# But we can see x from frankenstein_training_full.csv
frank_path = ROOT / "data" / "enhanced" / "frankenstein_training_full.csv"
if frank_path.exists():
    frank = pd.read_csv(frank_path)
    frank["datetime"] = pd.to_datetime(frank.get("datetime", ""), errors="coerce")
    frank_recent = frank[frank["datetime"].dt.date >= round_start]
    
    if not frank_recent.empty and "y_home" in frank_recent.columns:
        print("Training data for last round (features used by XGBoost to predict goals):")
        key_features = ["datetime", "home", "away", "y_home", "y_away", "EloDiff", 
                        "lg_avg_gf_home", "lg_avg_gf_away",
                        "press_form_H", "press_form_A", "xg_for_form_H", "xg_for_form_A"]
        available = [c for c in key_features if c in frank_recent.columns]
        print(frank_recent[available].to_string(index=False))
        
        # Check the feature distributions
        print("\n\nKey Feature Statistics (FULL TRAINING SET):")
        for col in ["EloDiff", "lg_avg_gf_home", "lg_avg_gf_away", 
                     "press_form_H", "press_form_A",
                     "xg_for_form_H", "xg_for_form_A"]:
            if col in frank.columns:
                vals = pd.to_numeric(frank[col], errors="coerce")
                print(f"  {col:>25s}: mean={vals.mean():7.3f}  std={vals.std():7.3f}  min={vals.min():7.3f}  max={vals.max():7.3f}")

print("\n\nMu values in current predictions:")
if not preds.empty:
    mu_h = preds["mu_home"]
    mu_a = preds["mu_away"]
    mu_t = preds["mu_total"]
    print(f"  mu_home: mean={mu_h.mean():.3f}  min={mu_h.min():.3f}  max={mu_h.max():.3f}")
    print(f"  mu_away: mean={mu_a.mean():.3f}  min={mu_a.min():.3f}  max={mu_a.max():.3f}")
    print(f"  mu_total: mean={mu_t.mean():.3f}  min={mu_t.min():.3f}  max={mu_t.max():.3f}")
    print(f"  Bundesliga league avg goals/game (real): ~2.7-2.9")
    print(f"  Bot average mu_total: {mu_t.mean():.3f}")
    
    if mu_t.mean() < 2.2:
        print("\n  >>> WARNING: Bot is SYSTEMATICALLY PREDICTING TOO FEW GOALS <<<")
        print("  The mu_total average is well below the Bundesliga average.")
        print("  This explains the poor accuracy: the bot leans UNDER on almost every game.")
