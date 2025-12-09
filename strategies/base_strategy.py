"""
Base strategy abstraction.

Each strategy exposes:
    generate_candidates(context, df_val) -> list[dict]

Where:
  - context: dict of config + metadata (league, mode, thresholds, etc.)
  - df_val: dataframe with markets, probabilities, odds, fair_odds, price_source, edge, EV
Returns:
  - list of dicts representing bet candidates (subset of df_val rows or enriched)
"""
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd


def filter_min_odds(df: pd.DataFrame, min_odds: float) -> pd.DataFrame:
    if df.empty or "odds" not in df.columns:
        return df
    return df[pd.to_numeric(df["odds"], errors="coerce") >= float(min_odds)].copy()


def to_candidates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return df.to_dict(orient="records")
