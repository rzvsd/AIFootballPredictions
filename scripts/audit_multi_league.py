#!/usr/bin/env python3
"""
Multi-League Predictions Audit Script
======================================
Verifies that predictions are generated correctly across all leagues.
Checks for:
1. Data coverage per league (history vs predictions)
2. Missing leagues
3. Date range coverage
4. Potential data skipping issues

Usage:
    python scripts/audit_multi_league.py
    python scripts/audit_multi_league.py --detailed
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("audit_multi_league")

_ROOT = Path(__file__).resolve().parents[1]


def audit_history(history_path: Path) -> pd.DataFrame:
    """Analyze history file coverage."""
    df = pd.read_csv(history_path, low_memory=False)
    
    # Parse dates
    if "datameci" in df.columns:
        df["_date"] = pd.to_datetime(df["datameci"], errors="coerce")
    elif "date" in df.columns:
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["_date"] = pd.NaT
    
    if "league" not in df.columns:
        logger.warning("History missing 'league' column")
        return pd.DataFrame()
    
    # Group by league
    summary = df.groupby("league").agg(
        total_matches=("_date", "count"),
        min_date=("_date", "min"),
        max_date=("_date", "max"),
        seasons=("season", lambda x: x.nunique() if "season" in df.columns else 0),
        teams=("home", lambda x: x.nunique() if "home" in df.columns else 0),
    ).reset_index()
    
    summary = summary.sort_values("total_matches", ascending=False)
    return summary


def audit_predictions(predictions_path: Path) -> pd.DataFrame:
    """Analyze predictions file coverage."""
    if not predictions_path.exists():
        logger.warning(f"Predictions file not found: {predictions_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(predictions_path)
    
    if "league" not in df.columns:
        logger.warning("Predictions missing 'league' column")
        return pd.DataFrame()
    
    df["_date"] = pd.to_datetime(df.get("fixture_datetime"), errors="coerce")
    
    summary = df.groupby("league").agg(
        predictions=("_date", "count"),
        min_date=("_date", "min"),
        max_date=("_date", "max"),
        teams=("home", lambda x: x.nunique()),
    ).reset_index()
    
    summary = summary.sort_values("predictions", ascending=False)
    return summary


def audit_source_data(source_path: Path) -> pd.DataFrame:
    """Analyze source allratingv.csv for future fixtures."""
    if not source_path.exists():
        logger.warning(f"Source file not found: {source_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(source_path, encoding="latin1", low_memory=False)
    
    # Parse dates
    df["_date"] = pd.to_datetime(df.get("datameci"), errors="coerce")
    
    # Current date
    now = pd.Timestamp.now().normalize()
    
    # Future fixtures
    future = df[df["_date"] > now].copy()
    
    if "league" not in future.columns and "txtliga" in future.columns:
        future["league"] = future["txtliga"]
    
    if "league" not in future.columns:
        logger.warning("Source missing league column")
        return pd.DataFrame()
    
    summary = future.groupby("league").agg(
        future_fixtures=("_date", "count"),
        next_fixture=("_date", "min"),
        last_fixture=("_date", "max"),
    ).reset_index()
    
    summary = summary.sort_values("future_fixtures", ascending=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit Multi-League Predictions")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    parser.add_argument("--history", default=str(_ROOT / "data" / "enhanced" / "cgm_match_history.csv"))
    parser.add_argument("--predictions", default=str(_ROOT / "reports" / "cgm_upcoming_predictions.csv"))
    parser.add_argument("--source", default=str(_ROOT / "CGM data" / "multiple leagues and seasons" / "allratingv.csv"))
    args = parser.parse_args()
    
    print("=" * 80)
    print("[AUDIT] MULTI-LEAGUE PREDICTIONS AUDIT")
    print("=" * 80)
    
    # Audit history
    print("\n[HISTORY] DATA COVERAGE:")
    print("-" * 80)
    hist_summary = audit_history(Path(args.history))
    if not hist_summary.empty:
        print(f"{'League':<20} {'Matches':>10} {'Seasons':>10} {'Teams':>8} {'Date Range':<25}")
        print("-" * 80)
        for _, r in hist_summary.iterrows():
            date_range = f"{r['min_date'].date() if pd.notna(r['min_date']) else 'N/A'} -> {r['max_date'].date() if pd.notna(r['max_date']) else 'N/A'}"
            print(f"{r['league']:<20} {r['total_matches']:>10} {r['seasons']:>10} {r['teams']:>8} {date_range:<25}")
        print(f"\n{'TOTAL':<20} {hist_summary['total_matches'].sum():>10}")
    
    # Audit source (future fixtures)
    print("\n[SOURCE] FUTURE FIXTURES:")
    print("-" * 80)
    source_summary = audit_source_data(Path(args.source))
    if not source_summary.empty:
        print(f"{'League':<20} {'Future':>10} {'Next Fixture':<15} {'Last Fixture':<15}")
        print("-" * 80)
        for _, r in source_summary.iterrows():
            next_fix = r['next_fixture'].date() if pd.notna(r['next_fixture']) else 'N/A'
            last_fix = r['last_fixture'].date() if pd.notna(r['last_fixture']) else 'N/A'
            print(f"{r['league']:<20} {r['future_fixtures']:>10} {str(next_fix):<15} {str(last_fix):<15}")
        print(f"\n{'TOTAL FUTURE':<20} {source_summary['future_fixtures'].sum():>10}")
    
    # Audit predictions
    print("\n[PREDICTIONS] GENERATED:")
    print("-" * 80)
    pred_summary = audit_predictions(Path(args.predictions))
    if not pred_summary.empty:
        print(f"{'League':<20} {'Predictions':>12} {'Teams':>8} {'Date Range':<25}")
        print("-" * 80)
        for _, r in pred_summary.iterrows():
            date_range = f"{r['min_date'].date() if pd.notna(r['min_date']) else 'N/A'} -> {r['max_date'].date() if pd.notna(r['max_date']) else 'N/A'}"
            print(f"{r['league']:<20} {r['predictions']:>12} {r['teams']:>8} {date_range:<25}")
        print(f"\n{'TOTAL PREDICTIONS':<20} {pred_summary['predictions'].sum():>12}")
    
    # Compare source vs predictions
    if not source_summary.empty and not pred_summary.empty:
        print("\n[WARNING] COVERAGE ANALYSIS:")
        print("-" * 80)
        
        source_leagues = set(source_summary["league"].tolist())
        pred_leagues = set(pred_summary["league"].tolist())
        
        missing_in_preds = source_leagues - pred_leagues
        if missing_in_preds:
            print(f"Leagues with future fixtures but NO predictions: {sorted(missing_in_preds)}")
        else:
            print("[OK] All leagues with future fixtures have predictions")
        
        # Coverage ratio
        merged = source_summary.merge(pred_summary, on="league", how="outer", suffixes=("_src", "_pred"))
        merged["coverage"] = merged["predictions"].fillna(0) / merged["future_fixtures"].fillna(1) * 100
        
        low_coverage = merged[merged["coverage"] < 50]
        if not low_coverage.empty:
            print(f"\n[!] Low coverage leagues (<50%):")
            for _, r in low_coverage.iterrows():
                preds = r['predictions'] if pd.notna(r['predictions']) else 0
                print(f"  {r['league']}: {preds:.0f}/{r['future_fixtures']:.0f} ({r['coverage']:.1f}%)")
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
