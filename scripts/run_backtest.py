"""
Script to run a "Time-Travel" Backtest.
It simulates the pipeline state for a range of past dates to generate predictions
without future leakage, then compares them against actual results.

Usage:
    python -m scripts.run_backtest --league "Premier League" --season 2025-2026 --start-date 2025-08-15
"""

import argparse
import glob
import json
import logging
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")


def _read_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def _create_temp_upcoming(df_day: pd.DataFrame, out_dir: Path) -> Path:
    """
    Creates a temporary 'upcoming - Copy.CSV' for a specific day.
    CRITICAL: Drops result columns so the bot CANNOT see the outcome.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "upcoming - Copy.CSV"

    # Map history columns back to the "upcoming" format expected by predict_upcoming.py
    # CGM Column Legend used for mapping:
    # datameci, txtechipa1, txtechipa2, cotaa, cotae, cotad, cotao, cotau
    
    upcoming_df = pd.DataFrame()
    upcoming_df["datameci"] = df_day["date"]
    upcoming_df["orameci"] = df_day["time"].astype(str).str.replace(":", "").str.ljust(4, "0") # rough approximation if needed
    upcoming_df["txtechipa1"] = df_day["home"] # Note: normalize might need original names, but we use normalized in history. 
    # Logic check: predict_upcoming calls normalize_team. If we pass mostly normalized names, it should be fine 
    # IF the registry maps them to themselves or if they are already standard.
    # To be safe, we use 'home'/'away' which are likely already normalized.
    
    upcoming_df["txtechipa2"] = df_day["away"]
    upcoming_df["cotaa"] = df_day["odds_home"]
    upcoming_df["cotae"] = df_day["odds_draw"]
    upcoming_df["cotad"] = df_day["odds_away"]
    upcoming_df["cotao"] = df_day["odds_over"]
    upcoming_df["cotau"] = df_day["odds_under"]
    
    # We add league/country to help scope filtering if needed
    upcoming_df["league"] = df_day["league"]
    upcoming_df["country"] = df_day["country"]
    upcoming_df["sezonul"] = df_day["season"]

    upcoming_df.to_csv(out_file, index=False)
    return out_dir


def run_backtest(league: str, season: str, start_date: str, history_path_str: str, output_file: str):
    history_path = Path(history_path_str)
    if not history_path.exists():
        logger.error(f"History file not found: {history_path}")
        return

    logger.info(f"Loading history from {history_path}...")
    df = _read_history(history_path)
    
    # Filter for target subset
    mask = (df["league"] == league) & (df["season"] == season)
    if start_date:
        mask &= (df["datetime"] >= pd.to_datetime(start_date))
    
    subset = df[mask].copy()
    if subset.empty:
        logger.error(f"No matches found for {league} {season} after {start_date}")
        return

    # Group by date to simulate "daily runs"
    # We process each unique date.
    dates = sorted(subset["datetime"].dt.date.unique())
    logger.info(f"Found {len(dates)} match dates to backtest.")

    all_preds = []
    
    temp_data_dir = Path("temp_backtest_data")
    temp_out_dir = Path("temp_backtest_results")
    temp_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, test_date in enumerate(dates):
            logger.info(f"[{i+1}/{len(dates)}] Running backtest for {test_date}...")
            
            # 1. Get matches for this day
            daily_matches = subset[subset["datetime"].dt.date == test_date]
            
            # 2. Create the BLIND input file (no results)
            _create_temp_upcoming(daily_matches, temp_data_dir)
            
            # 3. Define as-of-date. 
            # We want to predict games on 'test_date'.
            # The bot prevents same-day leakage if we say "run as of {test_date - 1 day}".
            # Actually, `predict_upcoming.py` logic is: "Strict 'no same-day' cutoff: fixtures must be strictly after {run_asof_datetime}".
            # If we set --as-of-date {test_date - 1}, then {run_asof_datetime} becomes {test_date} 00:00:00 (approx).
            # So fits are allowed.
            as_of_arg = (test_date - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # 4. Run predict.py
            # We bypass the full pipeline and call predict_upcoming directly via python -m to save time?
            # Or use predict.py? predict.py recalculates everything. predict_upcoming.py is faster if features are mostly static.
            # However, predict_upcoming needs HISTORY.
            # IMPORTANT: predict_upcoming loads history. We must tell it to IGNORE future rows.
            # It has an `--as-of-date` param for "Live Scope" but does it filter history?
            # Looking at predict_upcoming.py: 
            #   parser argument --as-of-date is used for live scope filtering.
            #   It does NOT seem to reload/filter the history CSV passed in --history.
            #   WAIT. `predict_upcoming.py` does NOT filter history rows by date?
            #   Let's check code...
            #   `hist = _load_history(Path(args.history))` -> `_rolling_stats` -> `_add_franken_features`.
            #   It seems `predict_upcoming.py` uses the WHOLE history file provided to it.
            #   
            #   CORRECTION: We need to pass a "history as of X" to avoid leakage in rolling features (e.g. form).
            #   If we pass the full history (which contains the match we are predicting), 
            #   `predict_upcoming.py` might see it?
            #   
            #   `predict_upcoming.py`:
            #   `latest_home, latest_away = _latest_side_snapshots(hist)`
            #   `latest_home = df_roll.sort_values("datetime").groupby("home").tail(1)`
            #   
            #   If `hist` contains the match on `test_date`, then "latest snapshot" will be THAT match (including its result if features use it? No, features are pre-match).
            #   BUT `cgm_match_history.csv` has `ft_home` etc.
            #   
            #   Ideally, we should filter the HISTORY file too. 
            #   We can create a filtered history file for each step.
            
            # 4a. Create FILTERED history (matches strictly BEFORE test_date)
            hist_subset = df[df["datetime"].dt.date < test_date].copy()
            temp_hist_path = temp_data_dir / "filtered_history.csv"
            hist_subset.to_csv(temp_hist_path, index=False)
            
            # 4b. Run prediction command
            pred_out_file = temp_out_dir / f"pred_{test_date}.csv"
            
            cmd = [
                sys.executable, "-m", "cgm.predict_upcoming",
                "--data-dir", str(temp_data_dir),
                "--history", str(temp_hist_path),
                "--out", str(pred_out_file),
                "--as-of-date", as_of_arg,
                "--log-level", "WARNING" # Reduce spam
            ]
            
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error(f"Prediction failed for {test_date}: {res.stderr}")
                continue
                
            # 5. Run Pick Engine (to get gates/scoring)
            picks_out_file = temp_out_dir / f"picks_{test_date}.csv"
            cmd_picks = [
                sys.executable, "-m", "cgm.pick_engine",
                "--in", str(pred_out_file),
                "--out", str(picks_out_file)
            ]
            subprocess.run(cmd_picks, capture_output=True)
            
            # Load results if they exist
            if picks_out_file.exists():
                p_df = pd.read_csv(picks_out_file)
                # We also want the raw predictions for ALL matches, not just picks? 
                # User asked "results vs bot pronostics". Usually implies all. 
                # But pick engine does the "selection".
                # Let's read the PREDICTIONS file for coverage, and maybe merge pick status.
                raw_df = pd.read_csv(pred_out_file)
                
                # Tag these with the test date (though they have fixture_datetime)
                raw_df["_backtest_date"] = str(test_date)
                all_preds.append(raw_df)

    finally:
        # Cleanup
        if temp_data_dir.exists():
            shutil.rmtree(temp_data_dir, ignore_errors=True)
        if temp_out_dir.exists():
            shutil.rmtree(temp_out_dir, ignore_errors=True)

    if not all_preds:
        logger.warning("No predictions generated.")
        return

    logger.info("Aggregating results...")
    full_preds = pd.concat(all_preds, ignore_index=True)
    
    # Merge with actual results
    # We assume 'home' and 'away' and 'date' match.
    # We used normalized names in history -> passed to inputs -> used in preds. should match.
    
    # Simplify history for merging
    actuals = df[["date", "home", "away", "ft_home", "ft_away", "result"]].copy()
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.date
    
    full_preds["fixture_date"] = pd.to_datetime(full_preds["fixture_datetime"]).dt.date
    
    merged = pd.merge(
        full_preds, 
        actuals, 
        left_on=["fixture_date", "home", "away"],
        right_on=["date", "home", "away"],
        how="left",
        suffixes=("", "_actual")
    )
    
    # Save raw
    merged.to_csv(output_file, index=False)
    logger.info(f"Backtest complete. Saved to {output_file}")
    
    # Print mini summary
    if "result" in merged.columns:
        valid = merged.dropna(subset=["result"])
        logger.info(f"Matches with known results: {len(valid)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--history", default="data/enhanced/cgm_match_history.csv")
    parser.add_argument("--out", default="reports/latest_backtest.csv")
    
    args = parser.parse_args()
    
    run_backtest(args.league, args.season, args.start_date, args.history, args.out)

if __name__ == "__main__":
    main()
