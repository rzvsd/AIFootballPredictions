"""
Elo favourite strategy.

Idea: Back favourites (home or away) when Elo gap + 1X2 probability indicate strong edge.
Parameters (from config['strategies']['elo_fav']):
  - min_prob: minimum 1X2 prob to consider
  - min_edge: minimum edge (prob - implied)
  - min_odds: minimum book odds
  - elo_gap: minimum Elo difference (favourite - underdog)
"""
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd

from .base_strategy import filter_min_odds, to_candidates


def generate_candidates(context: Dict[str, Any], df_val: pd.DataFrame) -> List[Dict[str, Any]]:
    cfg = (context.get("strategies") or {}).get("elo_fav", {})
    min_prob = float(cfg.get("min_prob", 0.52))
    min_edge = float(cfg.get("min_edge", 0.01))
    min_odds = float(cfg.get("min_odds", 1.6))
    elo_gap = float(cfg.get("elo_gap", 40.0))

    if df_val.empty:
        return []
    d = df_val[df_val["market"] == "1X2"].copy()
    if d.empty:
        return []
    # Require Elo columns in df_val or skip Elo filter
    if "Elo_H" in d.columns and "Elo_A" in d.columns:
        def fav_ok(row):
            if row["outcome"] == "H":
                return (row["Elo_H"] - row["Elo_A"]) >= elo_gap
            if row["outcome"] == "A":
                return (row["Elo_A"] - row["Elo_H"]) >= elo_gap
            return False
        d = d[d.apply(fav_ok, axis=1)]
    d = d[pd.to_numeric(d["prob"], errors="coerce") >= min_prob]
    d = d[pd.to_numeric(d["edge"], errors="coerce") >= min_edge]
    d = filter_min_odds(d, min_odds)
    return to_candidates(d)
