"""
OU xG strategy.

Idea: Use summed expected goals to choose OU lines (default 2.5) when probability and edge meet thresholds.
Parameters (config['strategies']['xg_ou']):
  - min_prob: minimum prob
  - min_edge: minimum edge
  - min_odds: minimum odds
  - preferred_line: default 2.5 (string)
"""
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd

from .base_strategy import filter_min_odds, to_candidates


def generate_candidates(context: Dict[str, Any], df_val: pd.DataFrame) -> List[Dict[str, Any]]:
    cfg = (context.get("strategies") or {}).get("xg_ou", {})
    min_prob = float(cfg.get("min_prob", 0.57))
    min_edge = float(cfg.get("min_edge", 0.01))
    min_odds = float(cfg.get("min_odds", 1.6))
    preferred_line = str(cfg.get("preferred_line", "2.5"))

    if df_val.empty:
        return []
    d = df_val[df_val["market"] == f"OU {preferred_line}"].copy()
    if d.empty:
        d = df_val[df_val["market"].astype(str).str.startswith("OU ")].copy()
    if d.empty:
        return []
    d = d[pd.to_numeric(d["prob"], errors="coerce") >= min_prob]
    d = d[pd.to_numeric(d["edge"], errors="coerce") >= min_edge]
    d = filter_min_odds(d, min_odds)
    return to_candidates(d)
