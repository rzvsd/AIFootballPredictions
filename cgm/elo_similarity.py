"""
Elo-based similarity features with kernel weighting and effective sample size.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

Venue = Literal["home", "away"]


def prepare_histories(df: pd.DataFrame) -> Dict[str, Dict[Venue, pd.DataFrame]]:
    """
    Build per-team histories for home/away contexts.
    Expects columns: datetime, home, away, elo_home, elo_away, ft_home, ft_away.
    Returns {team: {'home': df_home, 'away': df_away}}
    """
    hist: Dict[str, Dict[Venue, pd.DataFrame]] = {}
    if "datetime" not in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    for _, row in df.iterrows():
        dt = row.get("datetime")
        home = row.get("home")
        away = row.get("away")
        if pd.isna(dt) or pd.isna(home) or pd.isna(away):
            continue
        # home context
        h_entry = {
            "datetime": dt,
            "opp_elo": row.get("elo_away"),
            "gf": row.get("ft_home"),
            "ga": row.get("ft_away"),
            "league": row.get("league"),
        }
        # away context
        a_entry = {
            "datetime": dt,
            "opp_elo": row.get("elo_home"),
            "gf": row.get("ft_away"),
            "ga": row.get("ft_home"),
            "league": row.get("league"),
        }
        hist.setdefault(home, {}).setdefault("home", []).append(h_entry)
        hist.setdefault(away, {}).setdefault("away", []).append(a_entry)
    # convert lists to DataFrames sorted by datetime
    out: Dict[str, Dict[Venue, pd.DataFrame]] = {}
    for team, contexts in hist.items():
        out[team] = {}
        for venue, entries in contexts.items():
            df_ctx = pd.DataFrame(entries)
            if not df_ctx.empty:
                df_ctx["datetime"] = pd.to_datetime(df_ctx["datetime"], errors="coerce")
                df_ctx.sort_values("datetime", inplace=True)
            out[team][venue] = df_ctx
    return out


def kernel_similarity(
    history: pd.DataFrame,
    target_elo: float,
    sigma: float,
    as_of: pd.Timestamp | None = None,
    min_eff: float = 5.0,
    inclusive: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Compute weighted GF/GA vs opponents similar to target_elo using a Gaussian kernel.
    Returns (gf_sim, ga_sim, wsum, neff).
    Applies as_of cutoff (<=) to avoid leakage.
    Blends with unconditional mean when effective sample size is low.
    """
    if history is None or history.empty or pd.isna(target_elo) or sigma <= 0:
        return (math.nan, math.nan, 0.0, 0.0)
    h = history
    if as_of is not None:
        h = h[h["datetime"] <= as_of] if inclusive else h[h["datetime"] < as_of]
    if h.empty:
        return (math.nan, math.nan, 0.0, 0.0)
    opp = pd.to_numeric(h["opp_elo"], errors="coerce")
    gf = pd.to_numeric(h["gf"], errors="coerce")
    ga = pd.to_numeric(h["ga"], errors="coerce")
    valid = (~opp.isna()) & (~gf.isna()) & (~ga.isna())
    if not valid.any():
        return (math.nan, math.nan, 0.0, 0.0)
    opp = opp[valid]
    gf = gf[valid]
    ga = ga[valid]
    weights = np.exp(-((opp - target_elo) ** 2) / (2 * sigma ** 2))
    wsum = weights.sum()
    if wsum <= 0:
        return (math.nan, math.nan, 0.0, 0.0)
    gf_mean = float(np.sum(weights * gf) / wsum)
    ga_mean = float(np.sum(weights * ga) / wsum)
    neff = float((wsum ** 2) / np.sum(weights ** 2))
    # Blend with unconditional mean if too thin
    if neff < min_eff:
        base_gf = float(gf.mean())
        base_ga = float(ga.mean())
        alpha = min(1.0, neff / 10.0)
        gf_mean = alpha * gf_mean + (1 - alpha) * base_gf
        ga_mean = alpha * ga_mean + (1 - alpha) * base_ga
    return (gf_mean, ga_mean, wsum, neff)
