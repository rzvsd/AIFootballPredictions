"""
Milestone 10: Head-to-Head History Features

Extracts features from direct matchups between specific teams.
Only uses H2H matches BEFORE the current match date (no leakage).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# Load config
try:
    import config
    H2H_MIN_MATCHES = getattr(config, "H2H_MIN_MATCHES", 3)
    H2H_MAX_LOOKBACK_YEARS = getattr(config, "H2H_MAX_LOOKBACK_YEARS", 5)
except ImportError:
    H2H_MIN_MATCHES = 3
    H2H_MAX_LOOKBACK_YEARS = 5


def _compute_h2h_stats(
    matches: pd.DataFrame,
    *,
    home_col: str = "home",
    away_col: str = "away",
    ft_home_col: str = "ft_home",
    ft_away_col: str = "ft_away",
) -> Dict[str, float]:
    """
    Compute H2H statistics from a filtered set of past matches.
    
    Returns dict with:
    - h2h_matches: number of meetings
    - h2h_home_win_rate: % home team won
    - h2h_goals_avg: average total goals
    - h2h_btts_rate: % both teams scored
    - h2h_over25_rate: % over 2.5 goals
    """
    n = len(matches)
    if n == 0:
        return {
            "h2h_matches": 0,
            "h2h_home_win_rate": 0.5,
            "h2h_goals_avg": 2.5,
            "h2h_btts_rate": 0.5,
            "h2h_over25_rate": 0.5,
            "h2h_usable": 0,
        }
    
    ft_h = pd.to_numeric(matches[ft_home_col], errors="coerce")
    ft_a = pd.to_numeric(matches[ft_away_col], errors="coerce")
    
    # Remove rows with missing scores
    valid = ft_h.notna() & ft_a.notna()
    ft_h = ft_h[valid]
    ft_a = ft_a[valid]
    n_valid = len(ft_h)
    
    if n_valid == 0:
        return {
            "h2h_matches": n,
            "h2h_home_win_rate": 0.5,
            "h2h_goals_avg": 2.5,
            "h2h_btts_rate": 0.5,
            "h2h_over25_rate": 0.5,
            "h2h_usable": 0,
        }
    
    home_wins = (ft_h > ft_a).sum()
    total_goals = (ft_h + ft_a).mean()
    btts = ((ft_h > 0) & (ft_a > 0)).sum()
    over25 = ((ft_h + ft_a) > 2.5).sum()
    
    return {
        "h2h_matches": int(n_valid),
        "h2h_home_win_rate": float(home_wins / n_valid),
        "h2h_goals_avg": float(total_goals),
        "h2h_btts_rate": float(btts / n_valid),
        "h2h_over25_rate": float(over25 / n_valid),
        "h2h_usable": int(n_valid >= H2H_MIN_MATCHES),
    }


def add_h2h_features(
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    home_col: str = "home",
    away_col: str = "away",
    ft_home_col: str = "ft_home",
    ft_away_col: str = "ft_away",
) -> pd.DataFrame:
    """
    Add head-to-head features to a match-level dataframe.
    
    For each row, looks up past H2H matches between the same two teams
    (in either home/away configuration) and computes statistics.
    
    Strict leakage prevention: only uses matches with datetime < current row's datetime.
    """
    out = df.copy()
    
    # Ensure datetime is parsed
    if datetime_col not in out.columns:
        raise ValueError(f"Missing datetime column: {datetime_col}")
    
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    out = out.sort_values(datetime_col)
    
    # Initialize H2H columns
    h2h_cols = ["h2h_matches", "h2h_home_win_rate", "h2h_goals_avg", 
                "h2h_btts_rate", "h2h_over25_rate", "h2h_usable"]
    for col in h2h_cols:
        out[col] = np.nan
    
    # For efficiency, precompute lookback cutoff
    max_lookback = pd.Timedelta(days=H2H_MAX_LOOKBACK_YEARS * 365)
    
    _logger.info("[H2H] Computing head-to-head features for %d rows...", len(out))
    
    # Process each row
    for idx, row in out.iterrows():
        home = row[home_col]
        away = row[away_col]
        match_dt = row[datetime_col]
        
        if pd.isna(match_dt) or pd.isna(home) or pd.isna(away):
            continue
        
        # Find past H2H matches (same teams, either home/away config)
        # Match 1: home=A, away=B (current perspective)
        # Match 2: home=B, away=A (reversed perspective)
        lookback_start = match_dt - max_lookback
        
        h2h_mask = (
            (out[datetime_col] < match_dt) &  # Strict past only
            (out[datetime_col] >= lookback_start) &
            (
                ((out[home_col] == home) & (out[away_col] == away)) |
                ((out[home_col] == away) & (out[away_col] == home))
            )
        )
        
        h2h_matches = out.loc[h2h_mask]
        
        # For matches where teams are reversed, we need to flip the perspective
        # to compute stats from current home team's POV
        h2h_same = h2h_matches[(h2h_matches[home_col] == home) & (h2h_matches[away_col] == away)]
        h2h_reversed = h2h_matches[(h2h_matches[home_col] == away) & (h2h_matches[away_col] == home)]
        
        # Flip reversed matches to current perspective
        if not h2h_reversed.empty:
            h2h_reversed = h2h_reversed.copy()
            # Swap home/away scores for reversed matches
            h2h_reversed[[ft_home_col, ft_away_col]] = h2h_reversed[[ft_away_col, ft_home_col]].values
        
        # Combine
        h2h_combined = pd.concat([h2h_same, h2h_reversed], ignore_index=True)
        
        stats = _compute_h2h_stats(
            h2h_combined,
            home_col=home_col,
            away_col=away_col,
            ft_home_col=ft_home_col,
            ft_away_col=ft_away_col,
        )
        
        for col, val in stats.items():
            out.at[idx, col] = val
    
    # Fill any remaining NaNs with neutral defaults
    out["h2h_matches"] = out["h2h_matches"].fillna(0).astype(int)
    out["h2h_home_win_rate"] = out["h2h_home_win_rate"].fillna(0.5)
    out["h2h_goals_avg"] = out["h2h_goals_avg"].fillna(2.5)
    out["h2h_btts_rate"] = out["h2h_btts_rate"].fillna(0.5)
    out["h2h_over25_rate"] = out["h2h_over25_rate"].fillna(0.5)
    out["h2h_usable"] = out["h2h_usable"].fillna(0).astype(int)
    
    _logger.info("[H2H] Complete. H2H usable: %d / %d rows", 
                 int(out["h2h_usable"].sum()), len(out))
    
    return out


def get_h2h_features_for_fixture(
    history: pd.DataFrame,
    home: str,
    away: str,
    as_of_datetime: pd.Timestamp,
    *,
    datetime_col: str = "datetime",
    home_col: str = "home",
    away_col: str = "away",
    ft_home_col: str = "ft_home",
    ft_away_col: str = "ft_away",
) -> Dict[str, float]:
    """
    Get H2H features for a specific fixture at inference time.
    
    Used by predict_upcoming.py for live predictions.
    """
    max_lookback = pd.Timedelta(days=H2H_MAX_LOOKBACK_YEARS * 365)
    lookback_start = as_of_datetime - max_lookback
    
    # Find past H2H matches
    h2h_mask = (
        (history[datetime_col] < as_of_datetime) &
        (history[datetime_col] >= lookback_start) &
        (
            ((history[home_col] == home) & (history[away_col] == away)) |
            ((history[home_col] == away) & (history[away_col] == home))
        )
    )
    
    h2h_matches = history.loc[h2h_mask]
    
    # Flip reversed matches
    h2h_same = h2h_matches[(h2h_matches[home_col] == home) & (h2h_matches[away_col] == away)]
    h2h_reversed = h2h_matches[(h2h_matches[home_col] == away) & (h2h_matches[away_col] == home)]
    
    if not h2h_reversed.empty:
        h2h_reversed = h2h_reversed.copy()
        h2h_reversed[[ft_home_col, ft_away_col]] = h2h_reversed[[ft_away_col, ft_home_col]].values
    
    h2h_combined = pd.concat([h2h_same, h2h_reversed], ignore_index=True)
    
    return _compute_h2h_stats(
        h2h_combined,
        home_col=home_col,
        away_col=away_col,
        ft_home_col=ft_home_col,
        ft_away_col=ft_away_col,
    )
