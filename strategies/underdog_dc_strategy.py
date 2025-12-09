"""
Underdog double-chance strategy.

Idea: in tight lines, take DC on the underdog when model edge is positive.
Parameters (config['strategies']['underdog_dc']):
  - max_fav_prob: max 1X2 prob for favourite (to define "tight")
  - min_edge: min edge on DC
  - min_odds: min odds
"""
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd

from .base_strategy import filter_min_odds, to_candidates


def generate_candidates(context: Dict[str, Any], df_val: pd.DataFrame) -> List[Dict[str, Any]]:
    cfg = (context.get("strategies") or {}).get("underdog_dc", {})
    max_fav_prob = float(cfg.get("max_fav_prob", 0.6))
    min_edge = float(cfg.get("min_edge", 0.01))
    min_odds = float(cfg.get("min_odds", 1.4))

    if df_val.empty:
        return []
    one = df_val[df_val["market"] == "1X2"].copy()
    dc = df_val[df_val["market"] == "DC"].copy()
    if one.empty or dc.empty:
        return []
    # Determine favourite prob per match
    fav = one.loc[one.groupby(["date","home","away"])["prob"].idxmax()].copy()
    fav["fav_prob"] = fav["prob"]
    # Merge fav info into DC rows
    dc = dc.merge(fav[["date","home","away","fav_prob"]], on=["date","home","away"], how="left")
    dc = dc[pd.to_numeric(dc["fav_prob"], errors="coerce") <= max_fav_prob]
    dc = dc[pd.to_numeric(dc["edge"], errors="coerce") >= min_edge]
    dc = filter_min_odds(dc, min_odds)
    return to_candidates(dc)
