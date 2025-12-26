"""
Milestone 11: League-Specific Features.

Computes league profile features that capture scoring patterns per competition:
- lg_goals_per_match: League average total goals/match (rolling)
- lg_home_win_rate: League home win rate (rolling)
- lg_btts_rate: League BTTS rate (rolling)
- lg_over25_rate: League Over 2.5 rate (rolling)
- lg_home_advantage: League home goal advantage (home - away avg)
- lg_defensive_idx: League defensive tendency (inverse of goals)
- lg_profile_usable: 1 if league has enough matches for reliable profile

Usage (training):
    from cgm.league_features import add_league_features
    df = add_league_features(df)

Usage (inference):
    from cgm.league_features import get_league_features_for_fixture
    feats = get_league_features_for_fixture(history_df, country, league, as_of)
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

try:
    import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config


def _compute_league_profile(
    matches: pd.DataFrame,
    min_matches: int = 50,
) -> Dict[str, float]:
    """
    Compute league profile stats from a set of matches.
    
    Args:
        matches: DataFrame with ft_home, ft_away columns (past matches only)
        min_matches: Minimum matches required for usable profile
        
    Returns:
        Dict with league profile features
    """
    n = len(matches)
    
    if n < min_matches:
        # Not enough data - return neutral defaults
        return {
            "lg_goals_per_match": np.nan,
            "lg_home_win_rate": np.nan,
            "lg_btts_rate": np.nan,
            "lg_over25_rate": np.nan,
            "lg_home_advantage": np.nan,
            "lg_defensive_idx": np.nan,
            "lg_profile_usable": 0,
        }
    
    ft_home = pd.to_numeric(matches["ft_home"], errors="coerce")
    ft_away = pd.to_numeric(matches["ft_away"], errors="coerce")
    
    # Total goals per match
    total_goals = ft_home + ft_away
    goals_per_match = float(total_goals.mean()) if total_goals.notna().any() else np.nan
    
    # Home win rate
    home_wins = (ft_home > ft_away).sum()
    home_win_rate = float(home_wins / n) if n > 0 else np.nan
    
    # BTTS rate (both teams score)
    btts = ((ft_home > 0) & (ft_away > 0)).sum()
    btts_rate = float(btts / n) if n > 0 else np.nan
    
    # Over 2.5 rate
    over25 = (total_goals > 2.5).sum()
    over25_rate = float(over25 / n) if n > 0 else np.nan
    
    # Home advantage (avg home goals - avg away goals)
    avg_home = float(ft_home.mean()) if ft_home.notna().any() else np.nan
    avg_away = float(ft_away.mean()) if ft_away.notna().any() else np.nan
    home_advantage = avg_home - avg_away if pd.notna(avg_home) and pd.notna(avg_away) else np.nan
    
    # Defensive index (inverse of goals - lower goals = more defensive)
    # Normalize to [0, 1] range: 2.0 goals = 0.5, 3.0 goals = 0.33, 4.0 goals = 0.25
    defensive_idx = 1.0 / goals_per_match if pd.notna(goals_per_match) and goals_per_match > 0 else np.nan
    
    return {
        "lg_goals_per_match": goals_per_match,
        "lg_home_win_rate": home_win_rate,
        "lg_btts_rate": btts_rate,
        "lg_over25_rate": over25_rate,
        "lg_home_advantage": home_advantage,
        "lg_defensive_idx": defensive_idx,
        "lg_profile_usable": 1,
    }


def add_league_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add league profile features to each row in the training data.
    
    Leakage-safe: only uses matches strictly before current match datetime.
    Uses a rolling window for recency.
    
    Args:
        df: DataFrame with datetime, country, league, ft_home, ft_away
        
    Returns:
        DataFrame with league profile features added
    """
    min_matches = getattr(config, "LEAGUE_MIN_MATCHES", 50)
    window = getattr(config, "LEAGUE_PROFILE_WINDOW", 100)
    
    df = df.copy()
    
    # Ensure datetime is parsed
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Initialize output columns
    feature_cols = [
        "lg_goals_per_match",
        "lg_home_win_rate", 
        "lg_btts_rate",
        "lg_over25_rate",
        "lg_home_advantage",
        "lg_defensive_idx",
        "lg_profile_usable",
    ]
    for col in feature_cols:
        df[col] = np.nan
    
    # Group by (country, league) for efficiency
    for (country, league), group in df.groupby(["country", "league"], dropna=False):
        group_sorted = group.sort_values("datetime")
        indices = group_sorted.index.tolist()
        
        for i, idx in enumerate(indices):
            current_dt = group_sorted.loc[idx, "datetime"]
            
            # Get past matches in this league (strictly before current)
            past_mask = group_sorted["datetime"] < current_dt
            past = group_sorted[past_mask]
            
            # Apply rolling window (most recent N matches)
            if len(past) > window:
                past = past.tail(window)
            
            # Compute profile
            profile = _compute_league_profile(past, min_matches=min_matches)
            
            # Assign to row
            for col, val in profile.items():
                df.loc[idx, col] = val
    
    return df


def get_league_features_for_fixture(
    history_df: pd.DataFrame,
    country: str,
    league: str,
    as_of: pd.Timestamp | str,
) -> Dict[str, float]:
    """
    Get league profile features for a fixture at inference time.
    
    Args:
        history_df: Full match history DataFrame
        country: Country of the fixture
        league: League of the fixture
        as_of: Datetime of the fixture (use matches strictly before this)
        
    Returns:
        Dict with league profile features
    """
    min_matches = getattr(config, "LEAGUE_MIN_MATCHES", 50)
    window = getattr(config, "LEAGUE_PROFILE_WINDOW", 100)
    
    as_of = pd.to_datetime(as_of)
    
    # Ensure datetime parsed
    if "datetime" not in history_df.columns:
        history_df = history_df.copy()
        history_df["datetime"] = pd.to_datetime(history_df["date"], errors="coerce")
    
    # Filter to league and past only
    mask = (
        (history_df["country"] == country)
        & (history_df["league"] == league)
        & (history_df["datetime"] < as_of)
    )
    past = history_df[mask].sort_values("datetime")
    
    # Apply rolling window
    if len(past) > window:
        past = past.tail(window)
    
    return _compute_league_profile(past, min_matches=min_matches)


def main() -> None:
    """CLI for testing league features."""
    import argparse
    
    ap = argparse.ArgumentParser(description="Compute league features for testing")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv")
    args = ap.parse_args()
    
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _logger = logging.getLogger(__name__)

    df = pd.read_csv(args.history)
    _logger.info(f"Loaded {len(df)} rows from {args.history}")
    
    # Check leagues
    _logger.info("\nLeague distribution:")
    _logger.info(df.groupby(["country", "league"]).size())
    
    # Add features
    df = add_league_features(df)
    
    # Show stats
    _logger.info("\nLeague feature stats:")
    for col in ["lg_goals_per_match", "lg_home_win_rate", "lg_btts_rate", 
                "lg_over25_rate", "lg_home_advantage", "lg_defensive_idx", "lg_profile_usable"]:
        if col in df.columns:
            valid = df[col].notna().sum()
            mean_val = df[col].mean()
            _logger.info(f"  {col}: valid={valid}, mean={mean_val:.3f}" if pd.notna(mean_val) else f"  {col}: valid={valid}")
    
    _logger.info("[ok] league features computed successfully")


if __name__ == "__main__":
    main()
