"""
Total-goals interval strategy (SIM-oriented unless real odds exist).

Idea: pick best TG intervals (e.g., 0-3, 2-5) by probability/edge.
Parameters (config['strategies']['tg_interval']):
  - min_prob
  - min_edge
  - min_odds
"""
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd

from .base_strategy import filter_min_odds, to_candidates


def generate_candidates(context: Dict[str, Any], df_val: pd.DataFrame) -> List[Dict[str, Any]]:
    cfg = (context.get("strategies") or {}).get("tg_interval", {})
    min_prob = float(cfg.get("min_prob", 0.40))
    min_edge = float(cfg.get("min_edge", 0.03))
    min_odds = float(cfg.get("min_odds", 1.6))

    d = df_val[df_val["market"] == "TG Interval"].copy()
    if d.empty:
        return []
    d = d[pd.to_numeric(d["prob"], errors="coerce") >= min_prob]
    d = d[pd.to_numeric(d["edge"], errors="coerce") >= min_edge]
    d = filter_min_odds(d, min_odds)
    return to_candidates(d)
