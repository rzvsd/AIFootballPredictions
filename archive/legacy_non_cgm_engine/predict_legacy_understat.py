"""
Orchestrator: Single-command predictions pipeline.

Usage:
  python predict.py --league E0 [--days 7] [--max-age-hours 24] [--retrain-days 7]
  python predict.py --leagues E0 D1 F1 [--days 7] [--max-age-hours 24] [--retrain-days 7]

What it does:
  1) Ensures raw -> processed data is fresh (downloads + preprocesses if stale)
  2) Best-effort Understat shots fetch + micro aggregates rebuild
  3) Fetches odds from The Odds API (Bet365 preferred) into data/odds/{LEAGUE}.json
  4) Ensures absences snapshot exists (seeds default if missing)
  5) Trains models if stale
  6) Prints top picks and per-match combined table using Understat fixtures + odds
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def run(cmd: List[str], env: Optional[dict] = None, cwd: Optional[Path] = None) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd or ROOT), env=env or os.environ.copy())
    return proc.returncode


def is_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    return (dt.datetime.now() - mtime) <= dt.timedelta(hours=max_age_hours)


def current_season_start_year(today: Optional[dt.date] = None) -> int:
    d = today or dt.date.today()
    return d.year if d.month >= 8 else d.year - 1


def season_code_mmz(year_start: int) -> str:
    y1 = str(year_start)[-2:]
    y2 = str(year_start + 1)[-2:]
    return f"{y1}{y2}"


def ensure_processed(league: str, max_age_hours: int) -> None:
    proc_csv = DATA / "processed" / f"{league}_merged_preprocessed.csv"
    if is_fresh(proc_csv, max_age_hours):
        print(f"[ok] Processed data fresh: {proc_csv}")
        return
    # Download latest raw and preprocess
    season_start = current_season_start_year()
    seasons = [season_code_mmz(season_start + i) for i in (0, -1, -2)]
    print(f"[step] Acquiring raw for {league} seasons {seasons}")
    rc = run([sys.executable, "scripts/data_acquisition.py", "--leagues", league, "--seasons", *seasons, "--raw_data_output_dir", "data/raw"])  # noqa: E501
    if rc != 0:
        print("[warn] data_acquisition returned non-zero; continuing if any raw exists")
    print("[step] Preprocessing raw -> processed")
    rc = run([sys.executable, "scripts/data_preprocessing.py", "--raw_data_input_dir", "data/raw", "--processed_data_output_dir", "data/processed", "--num_features", "30", "--clustering_threshold", "0.5"])  # noqa: E501
    if rc != 0:
        print("[warn] data_preprocessing returned non-zero; continuing best-effort")


def ensure_micro_aggregates(league: str) -> None:
    # Try fetching Understat (best-effort)
    season_start = current_season_start_year()
    understat_seasons = f"{season_start},{season_start-1}"
    print(f"[step] Fetching Understat shots (best-effort): {understat_seasons}")
    _ = run([sys.executable, "-m", "scripts.fetch_understat_simple", "--league", league, "--seasons", understat_seasons])
    # Build shots CSV (if any) and micro aggregates (will also inject possession/corners if available)
    print("[step] Building shots CSV and micro aggregates")
    run([sys.executable, "-m", "scripts.shots_ingest_understat", "--inputs", "data/understat/*_shots.json", "--out", "data/shots/understat_shots.csv"])  # noqa: E501
    run([sys.executable, "-m", "scripts.build_micro_aggregates", "--shots", "data/shots/understat_shots.csv", "--league", league, "--out", "data/enhanced/micro_agg.csv"])  # noqa: E501
    # Also persist per-league copy to avoid cross-league contamination
    try:
        import shutil  # local import
        src = DATA / "enhanced" / "micro_agg.csv"
        dest = DATA / "enhanced" / f"{league}_micro_agg.csv"
        if src.exists():
            shutil.copyfile(str(src), str(dest))
            print(f"[ok] Wrote {dest}")
    except Exception as e:
        print(f"[warn] Could not persist per-league micro aggregates: {e}")


def ensure_absences_seed(league: str) -> None:
    out_dir = DATA / "absences"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{league}_availability.csv"
    if dest.exists():
        print(f"[ok] Absences snapshot exists: {dest}")
        return
    # Seed a flat availability=1.0 per team based on processed file
    proc = DATA / "processed" / f"{league}_merged_preprocessed.csv"
    if not proc.exists():
        print("[warn] Processed data missing; cannot seed absences yet")
        return
    import pandas as pd  # local import
    df = pd.read_csv(proc)
    teams = sorted(set(df.get("HomeTeam").dropna().astype(str)).union(set(df.get("AwayTeam").dropna().astype(str))))
    seed = out_dir / f"{league}_availability_seed.csv"
    pd.DataFrame({"team": teams, "availability_index": [1.0]*len(teams)}).to_csv(seed, index=False)
    run([sys.executable, "-m", "scripts.absences_import", "--league", league, "--input", str(seed)])


def ensure_trained(league: str, retrain_days: int) -> None:
    models_dir = ROOT / "advanced_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    # Require both home and away models (JSON preferred, PKL fallback)
    home_candidates = [
        models_dir / f"{league}_ultimate_xgb_home.json",
        models_dir / f"{league}_ultimate_xgb_home.pkl",
    ]
    away_candidates = [
        models_dir / f"{league}_ultimate_xgb_away.json",
        models_dir / f"{league}_ultimate_xgb_away.pkl",
    ]
    def _latest(paths):
        existing = [p for p in paths if p.exists()]
        return max(existing, key=lambda p: p.stat().st_mtime) if existing else None

    latest_home = _latest(home_candidates)
    latest_away = _latest(away_candidates)
    needs_train = not (latest_home and latest_away)

    if not needs_train:
        newest_mtime = max(latest_home.stat().st_mtime, latest_away.stat().st_mtime)
        mtime = dt.datetime.fromtimestamp(newest_mtime)
        needs_train = (dt.datetime.now() - mtime) > dt.timedelta(days=retrain_days)
        if not needs_train:
            print(f"[ok] Models fresh: {latest_home.name}, {latest_away.name}")
    if needs_train:
        print(f"[step] Training XGB models for {league}")
        run([sys.executable, "xgb_trainer.py", "--league", league])


def ensure_odds(league: str) -> None:
    # In CGM/offline mode, do not fetch odds from APIs (preserve local CGM-derived odds JSON).
    use_cgm = str(os.getenv("BOT_FIXTURE_SOURCE", "")).lower() == "cgm" or str(os.getenv("BOT_USE_CGM", "")).strip().lower() in ("1", "true", "yes", "on")
    if use_cgm:
        print(f"[skip] Using CGM/local odds for {league}; skipping API fetch.")
        return
    odds_dir = DATA / "odds"
    odds_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        print("[warn] THE_ODDS_API_KEY not set; skipping odds fetch.")
        return
    cmd = [sys.executable, "-m", "scripts.fetch_odds_toa", "--leagues", league]
    run(cmd)


def show_predictions(league: str, days: int, season: Optional[int] = None) -> None:
    # Use bet_fusion top picks (placeholder odds) and per-match combined table
    env = os.environ.copy()
    env["BOT_LEAGUE"] = league
    env["BOT_FIXTURES_DAYS"] = str(days)
    if season is not None:
        env["BOT_UNDERSTAT_SEASON"] = str(season)
    print("\n=== Top Picks (by EV) ===")
    run([sys.executable, "bet_fusion.py", "--top", "20"], env=env)
    print("\n=== Best Per Match (TG, OU 2.5, 1X2) - by highest probability ===")
    run([sys.executable, "-m", "scripts.print_best_per_match", "--by", "prob"], env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command predictions orchestrator")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--league", help="League code, e.g., E0, I1, D1, SP1, F1")
    group.add_argument("--leagues", nargs="+", help="Space-separated list of league codes, e.g., E0 D1 F1 I1 SP1")
    ap.add_argument("--days", type=int, default=7, help="How many days ahead to fetch fixtures from Understat")
    ap.add_argument("--season", type=int, default=None, help="Optional Understat season start year (e.g., 2025 for 2025-26)")
    ap.add_argument("--max-age-hours", type=int, default=24, help="Max age for processed data before refresh")
    ap.add_argument("--retrain-days", type=int, default=7, help="Retrain models if older than this many days")
    ap.add_argument("--compact", action="store_true", help="Also print compact round prognostics (1X2, OU 2.5, TG)")
    args = ap.parse_args()

    if args.leagues:
        leagues: List[str] = [str(x).upper().strip() for x in args.leagues]
    else:
        leagues = [str(args.league).upper().strip()]

    print(
        f"[orchestrator] Leagues={','.join(leagues)} days={args.days} data_max_age={args.max_age_hours}h retrain_after={args.retrain_days}d"
    )

    for league in leagues:
        print("\n" + "=" * 20 + f"  {league}  " + "=" * 20)
        # 1) Data freshness (raw->processed)
        ensure_processed(league, max_age_hours=args.max_age_hours)

        # 2) Micro aggregates (Understat best-effort + enrich)
        ensure_micro_aggregates(league)

        # 3) Odds (The Odds API -> local JSON)
        ensure_odds(league)

        # 4) Absences seed
        ensure_absences_seed(league)

        # 5) Train if needed
        ensure_trained(league, retrain_days=args.retrain_days)

        # 6) Show predictions
        show_predictions(league, days=args.days, season=args.season)

        # Optional: compact per-match prognostics for the next round
        if args.compact:
            print("\n=== Round Prognostics (compact) ===")
            run([sys.executable, "-m", "scripts.round_prognostics", "--league", league])

if __name__ == "__main__":
    main()
