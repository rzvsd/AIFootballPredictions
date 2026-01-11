#!/usr/bin/env python3
"""
Multi-League Backtest Orchestrator (Milestone 16)
==================================================
Runs backtests for all available leagues in the history data,
aggregates results, and runs calibration.

Usage:
    python scripts/run_multi_league_backtest.py --start-date 2025-10-01
    python scripts/run_multi_league_backtest.py --start-date 2025-10-01 --leagues "Premier L,Serie A"
    python scripts/run_multi_league_backtest.py --list-leagues
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("multi_league_backtest")

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = _ROOT / "data" / "enhanced" / "cgm_match_history_with_elo_stats_xg.csv"
DEFAULT_OUTPUT_DIR = _ROOT / "reports" / "backtests"


def get_available_leagues(history_path: Path) -> list[tuple[str, str, int]]:
    """
    Get all available (league, season) combinations from history.
    Returns list of (league, season, match_count) tuples.
    """
    df = pd.read_csv(history_path, low_memory=False)
    
    # Group by league and season
    if "league" not in df.columns or "season" not in df.columns:
        logger.warning("History missing league/season columns")
        return []
    
    counts = df.groupby(["league", "season"]).size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=False)
    
    return [(r["league"], r["season"], r["count"]) for _, r in counts.iterrows()]


def run_single_backtest(
    league: str, 
    season: str, 
    start_date: str, 
    history_path: Path,
    output_dir: Path
) -> Path | None:
    """Run backtest for a single league/season."""
    
    # Clean league name for filename
    league_clean = league.replace(" ", "_").replace("/", "-")
    season_clean = season.replace("-", "_")
    output_file = output_dir / f"backtest_{league_clean}_{season_clean}.csv"
    
    cmd = [
        sys.executable, "-m", "scripts.run_backtest",
        "--league", league,
        "--season", season,
        "--start-date", start_date,
        "--history", str(history_path),
        "--out", str(output_file)
    ]
    
    logger.info(f"Running: {league} {season}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.warning(f"  Failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
        return None
    
    if output_file.exists():
        df = pd.read_csv(output_file)
        logger.info(f"  ✅ Generated {len(df)} predictions")
        return output_file
    
    return None


def aggregate_results(backtest_files: list[Path], output_path: Path) -> pd.DataFrame:
    """Combine all backtest results into single file."""
    dfs = []
    for f in backtest_files:
        if f.exists():
            df = pd.read_csv(f)
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Aggregated {len(combined)} predictions to {output_path}")
    return combined


def calculate_accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate accuracy metrics per league."""
    if df.empty or "league" not in df.columns:
        return pd.DataFrame()
    
    # Ensure we have actual results
    df = df.dropna(subset=["ft_home", "ft_away"])
    if df.empty:
        return pd.DataFrame()
    
    df["total_goals"] = df["ft_home"] + df["ft_away"]
    df["actual_over25"] = (df["total_goals"] > 2.5).astype(int)
    df["actual_btts"] = ((df["ft_home"] > 0) & (df["ft_away"] > 0)).astype(int)
    
    results = []
    for league in df["league"].unique():
        lg_df = df[df["league"] == league]
        
        row = {
            "league": league,
            "matches": len(lg_df),
            "actual_over25_rate": lg_df["actual_over25"].mean(),
            "actual_btts_rate": lg_df["actual_btts"].mean(),
        }
        
        # O/U 2.5 accuracy
        if "p_over25" in lg_df.columns:
            valid = lg_df.dropna(subset=["p_over25"])
            if len(valid) > 0:
                pred_over = valid["p_over25"] > 0.5
                correct = (pred_over == (valid["actual_over25"] == 1)).sum()
                row["ou25_accuracy"] = correct / len(valid)
        
        # BTTS accuracy
        if "p_btts_yes" in lg_df.columns:
            valid = lg_df.dropna(subset=["p_btts_yes"])
            if len(valid) > 0:
                pred_btts = valid["p_btts_yes"] > 0.5
                correct = (pred_btts == (valid["actual_btts"] == 1)).sum()
                row["btts_accuracy"] = correct / len(valid)
        
        results.append(row)
    
    return pd.DataFrame(results).sort_values("matches", ascending=False)


def run_calibration(aggregated_file: Path):
    """Run calibration on aggregated backtest results."""
    cmd = [
        sys.executable, "-m", "scripts.calibrate_league",
        "--input", str(aggregated_file)
    ]
    
    logger.info("Running calibration for all leagues...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ Calibration complete")
    else:
        logger.warning(f"Calibration failed: {result.stderr[:200] if result.stderr else ''}")


def main():
    parser = argparse.ArgumentParser(description="Multi-League Backtest Orchestrator")
    parser.add_argument("--start-date", help="Start date for backtests (YYYY-MM-DD)")
    parser.add_argument("--history", default=str(DEFAULT_HISTORY), help="History CSV path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--leagues", help="Comma-separated list of leagues (default: all)")
    parser.add_argument("--seasons", help="Comma-separated list of seasons (default: all)")
    parser.add_argument("--min-matches", type=int, default=20, help="Minimum matches to include league")
    parser.add_argument("--list-leagues", action="store_true", help="List available leagues and exit")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration step")
    args = parser.parse_args()
    
    history_path = Path(args.history)
    if not history_path.exists():
        logger.error(f"History file not found: {history_path}")
        return 1
    
    # Get available leagues
    all_leagues = get_available_leagues(history_path)
    
    if args.list_leagues:
        print("\n📊 Available Leagues in History:\n")
        print(f"{'League':<20} {'Season':<15} {'Matches':>10}")
        print("-" * 50)
        for league, season, count in all_leagues:
            print(f"{league:<20} {season:<15} {count:>10}")
        return 0
    
    if not args.start_date:
        parser.error("--start-date is required (use --list-leagues to see available data)")
    
    # Filter leagues
    target_leagues = None
    if args.leagues:
        target_leagues = [l.strip() for l in args.leagues.split(",")]
    
    target_seasons = None
    if args.seasons:
        target_seasons = [s.strip() for s in args.seasons.split(",")]
    
    # Filter by criteria
    filtered = []
    for league, season, count in all_leagues:
        if count < args.min_matches:
            continue
        if target_leagues and league not in target_leagues:
            continue
        if target_seasons and season not in target_seasons:
            continue
        filtered.append((league, season, count))
    
    if not filtered:
        logger.error("No leagues match the criteria")
        return 1
    
    logger.info(f"Running backtest for {len(filtered)} league/season combinations")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run backtests
    backtest_files = []
    for league, season, count in filtered:
        logger.info(f"\n{'='*60}")
        logger.info(f"League: {league} | Season: {season} | Matches: {count}")
        logger.info("="*60)
        
        result = run_single_backtest(
            league, season, args.start_date, 
            history_path, output_dir
        )
        if result:
            backtest_files.append(result)
    
    # Aggregate results
    if backtest_files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        aggregated_path = output_dir / f"backtest_all_leagues_{timestamp}.csv"
        combined_df = aggregate_results(backtest_files, aggregated_path)
        
        # Show accuracy summary
        summary = calculate_accuracy_summary(combined_df)
        if not summary.empty:
            print("\n" + "="*70)
            print("📊 ACCURACY SUMMARY BY LEAGUE")
            print("="*70)
            print(summary.to_string(index=False))
            print("="*70)
            
            # Save summary
            summary_path = output_dir / f"backtest_summary_{timestamp}.csv"
            summary.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to {summary_path}")
        
        # Run calibration
        if not args.skip_calibration:
            run_calibration(aggregated_path)
    else:
        logger.warning("No backtest results generated")
        return 1
    
    logger.info("\n✅ Multi-league backtest complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
