"""
League-Specific Probability Calibration (Milestone 13)

Analyzes backtest results to calculate optimal thresholds per league.
Outputs a calibration JSON file that the pick engine uses to adjust predictions.

Usage:
    python -m scripts.calibrate_league --input reports/backtest_epl_2025.csv --league "Premier L"
    python -m scripts.calibrate_league --input reports/backtest_*.csv  # All leagues
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = _ROOT / "data" / "league_calibration.json"


def calculate_optimal_threshold(
    probabilities: pd.Series,
    actuals: pd.Series,
    thresholds: list[float] = None
) -> tuple[float, float]:
    """
    Find the threshold that maximizes accuracy.
    
    Returns:
        (optimal_threshold, accuracy_at_optimal)
    """
    if thresholds is None:
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    
    best_threshold = 0.50
    best_accuracy = 0.0
    
    for thresh in thresholds:
        pred_over = probabilities > thresh
        pred_under = probabilities <= thresh
        correct = ((pred_over & (actuals == 1)) | (pred_under & (actuals == 0))).sum()
        accuracy = correct / len(actuals) if len(actuals) > 0 else 0
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    
    return best_threshold, best_accuracy


def calculate_bias(
    probabilities: pd.Series,
    actuals: pd.Series
) -> float:
    """
    Calculate the systematic bias between model predictions and actual outcomes.
    
    Positive bias = model underestimates (actual > predicted)
    Negative bias = model overestimates (actual < predicted)
    """
    mean_pred = probabilities.mean()
    mean_actual = actuals.mean()
    return float(mean_actual - mean_pred)


def calibrate_league(df: pd.DataFrame, league: str = None) -> Dict[str, Any]:
    """
    Calculate calibration parameters for a league from backtest data.
    """
    # Filter to league if specified
    if league and "league" in df.columns:
        df = df[df["league"] == league]
    
    if len(df) == 0:
        logger.warning(f"No data for league: {league}")
        return {}
    
    # Ensure numeric columns
    for col in ["ft_home", "ft_away", "p_over25", "p_btts_yes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Filter out 0-0 placeholder results (likely missing data)
    if "ft_home" in df.columns and "ft_away" in df.columns:
        is_zero = (df["ft_home"] == 0) & (df["ft_away"] == 0)
        zero_count = is_zero.sum()
        if len(df) > 0 and (zero_count / len(df)) > 0.15:
            logger.warning(f"Excluding {zero_count} likely-missing 0-0 results")
            df = df[~is_zero]
    
    # Calculate actuals
    df = df.dropna(subset=["ft_home", "ft_away"])
    if len(df) == 0:
        return {}
    
    df["total_goals"] = df["ft_home"] + df["ft_away"]
    df["actual_over25"] = (df["total_goals"] > 2.5).astype(int)
    df["actual_btts"] = ((df["ft_home"] > 0) & (df["ft_away"] > 0)).astype(int)
    
    result = {
        "sample_size": int(len(df)),
        "actual_over25_rate": float(df["actual_over25"].mean()),
        "actual_btts_rate": float(df["actual_btts"].mean()),
    }
    
    # O/U 2.5 calibration
    if "p_over25" in df.columns:
        valid_ou = df.dropna(subset=["p_over25"])
        if len(valid_ou) > 0:
            ou_threshold, ou_accuracy = calculate_optimal_threshold(
                valid_ou["p_over25"], valid_ou["actual_over25"]
            )
            ou_bias = calculate_bias(valid_ou["p_over25"], valid_ou["actual_over25"])
            result["ou25_optimal_threshold"] = ou_threshold
            result["ou25_accuracy_at_optimal"] = float(ou_accuracy)
            result["ou25_bias"] = ou_bias
            result["ou25_default_accuracy"] = float(
                ((valid_ou["p_over25"] > 0.5) == (valid_ou["actual_over25"] == 1)).mean()
            )
    
    # BTTS calibration
    if "p_btts_yes" in df.columns:
        valid_btts = df.dropna(subset=["p_btts_yes"])
        if len(valid_btts) > 0:
            btts_threshold, btts_accuracy = calculate_optimal_threshold(
                valid_btts["p_btts_yes"], valid_btts["actual_btts"]
            )
            btts_bias = calculate_bias(valid_btts["p_btts_yes"], valid_btts["actual_btts"])
            result["btts_optimal_threshold"] = btts_threshold
            result["btts_accuracy_at_optimal"] = float(btts_accuracy)
            result["btts_bias"] = btts_bias
            result["btts_default_accuracy"] = float(
                ((valid_btts["p_btts_yes"] > 0.5) == (valid_btts["actual_btts"] == 1)).mean()
            )
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate league-specific thresholds")
    parser.add_argument("--input", required=True, help="Path to backtest CSV")
    parser.add_argument("--league", help="League name to calibrate (optional, calibrates all if not specified)")
    parser.add_argument("--out", default=str(DEFAULT_CALIBRATION_PATH), help="Output JSON path")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum samples required")
    args = parser.parse_args()
    
    # Load backtest data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Backtest file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    
    # Load existing calibration if present
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path, "r") as f:
            calibration = json.load(f)
        logger.info(f"Loaded existing calibration from {out_path}")
    else:
        calibration = {}
    
    # Calibrate specified league or all leagues
    if args.league:
        leagues = [args.league]
    elif "league" in df.columns:
        leagues = df["league"].dropna().unique().tolist()
    else:
        leagues = ["_global"]
    
    for league in leagues:
        logger.info(f"Calibrating league: {league}")
        result = calibrate_league(df, league if league != "_global" else None)
        
        if result.get("sample_size", 0) < args.min_samples:
            logger.warning(f"Skipping {league}: only {result.get('sample_size', 0)} samples (min: {args.min_samples})")
            continue
        
        calibration[league] = result
        
        # Print summary
        logger.info(f"  Sample size: {result['sample_size']}")
        logger.info(f"  Actual Over 2.5 rate: {result['actual_over25_rate']:.1%}")
        if "ou25_optimal_threshold" in result:
            logger.info(f"  O/U 2.5: default acc={result['ou25_default_accuracy']:.1%}, "
                       f"optimal threshold={result['ou25_optimal_threshold']:.0%}, "
                       f"optimal acc={result['ou25_accuracy_at_optimal']:.1%}, "
                       f"bias={result['ou25_bias']:+.1%}")
        if "btts_optimal_threshold" in result:
            logger.info(f"  BTTS: default acc={result['btts_default_accuracy']:.1%}, "
                       f"optimal threshold={result['btts_optimal_threshold']:.0%}, "
                       f"optimal acc={result['btts_accuracy_at_optimal']:.1%}, "
                       f"bias={result['btts_bias']:+.1%}")
    
    # Save calibration
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info(f"Saved calibration to {out_path}")


if __name__ == "__main__":
    main()
