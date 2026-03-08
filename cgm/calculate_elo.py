"""
Compute football Elo ratings from CGM match history.

This module is a compatibility wrapper around scripts.calc_cgm_elo (Elo V2 core),
so standalone calls stay aligned with the production pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from scripts.calc_cgm_elo import (
    START_ELO_DEFAULT,
    K_FACTOR_DEFAULT,
    HOME_ADV_DEFAULT,
    compute_elo_series,
    infer_team,
    load_history,
)


def calculate_elo(df: pd.DataFrame, start_elo: float, k_factor: float, home_adv: float) -> Dict[str, float]:
    """Return final ratings per team using the Elo V2 core."""
    ordered = df.sort_values("datetime") if "datetime" in df.columns else df.copy()
    trace = compute_elo_series(ordered, start_elo=start_elo, k_factor=k_factor, home_adv=home_adv)

    ratings: Dict[str, float] = {}
    # Reconstruct post-match ratings from pre-match Elo + delta from V2 trace.
    for idx, row in ordered.iterrows():
        home_id = infer_team(row, home=True)
        away_id = infer_team(row, home=False)
        if home_id is None or away_id is None:
            continue
        pre_h = float(trace.loc[idx, "elo_home_calc"])
        pre_a = float(trace.loc[idx, "elo_away_calc"])
        delta = trace.loc[idx, "elo_delta"]
        if pd.notna(delta):
            d = float(delta)
            ratings[home_id] = pre_h + d
            ratings[away_id] = pre_a - d
        else:
            ratings.setdefault(home_id, pre_h)
            ratings.setdefault(away_id, pre_a)
    return ratings


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Elo ratings from CGM match history")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history.csv", help="Path to match history CSV")
    ap.add_argument("--out", default="data/enhanced/current_elo.json", help="Output JSON path")
    ap.add_argument("--start-elo", type=float, default=START_ELO_DEFAULT, help="Starting Elo for new teams")
    ap.add_argument("--k-factor", type=float, default=K_FACTOR_DEFAULT, help="Base K factor")
    ap.add_argument("--home-adv", type=float, default=HOME_ADV_DEFAULT, help="Base home advantage (points)")
    args = ap.parse_args()

    hist = load_history(Path(args.history))
    ratings = calculate_elo(hist, start_elo=args.start_elo, k_factor=args.k_factor, home_adv=args.home_adv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)
    print(f"[ok] wrote Elo ratings -> {out_path} (teams={len(ratings)})")


if __name__ == "__main__":
    main()
