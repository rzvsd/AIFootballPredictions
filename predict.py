"""
Orchestrator: Single-command predictions pipeline.

Usage:
  python predict.py --league E0 [--days 7] [--max-age-hours 24] [--retrain-days 7]
  python predict.py --leagues E0 D1 F1 [--days 7] [--max-age-hours 24] [--retrain-days 7]

What it does:
  1) Loads API keys from .env if present
  2) Ensures raw -> processed data is fresh (downloads + preprocesses if stale)
  3) Best-effort Understat shots fetch + micro aggregates rebuild
  4) Fetches real odds for upcoming fixtures (API-Football)
  5) Ensures absences snapshot exists (seeds default if missing)
  6) Trains models if stale
  7) Prints top picks and per-match combined table
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*_, **__):
        return False


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


def ensure_odds(league: str, days: int, season: Optional[int] = None) -> None:
    print(f"[step] Fetching odds (API-Football) for {league}, next {days} days")
    env = os.environ.copy()
    load_dotenv()
    env.setdefault("BOT_ODDS_DIR", str(DATA / "odds"))
    env.setdefault("BOT_ODDS_MODE", "local")
    Path(env["BOT_ODDS_DIR"]).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "scripts.fetch_odds_api_football", "--league", league, "--days", str(days), "--tag", "closing"]
    if season is not None:
        cmd += ["--season", str(season)]
    rc = run(cmd, env=env)  # noqa: E501
    if rc != 0:
        print("[warn] odds fetcher returned non-zero; predictions will use placeholders if odds missing")
        # Fallback: if odds JSON has no fixtures, generate weekly fixtures via football-data.org and refetch with CSV
        try:
            odds_path = DATA / "odds" / f"{league}.json"
            fixtures_count = 0
            if odds_path.exists():
                import json as _json
                try:
                    data = _json.loads(odds_path.read_text(encoding="utf-8"))
                    fixtures_count = len(data.get("fixtures", []))
                except Exception:
                    fixtures_count = 0
            if fixtures_count == 0:
                print(f"[info] No fixtures in odds JSON for {league}. Generating weekly fixtures via football-data.org fallback...")
                run([sys.executable, "-m", "scripts.gen_weekly_fixtures_from_fd", "--league", league, "--days", str(days)], env=env)
                fx_csv = str(DATA / "fixtures" / f"{league}_weekly_fixtures.csv")
                if os.path.exists(fx_csv):
                    cmd2 = [sys.executable, "-m", "scripts.fetch_odds_api_football", "--league", league, "--fixtures-csv", fx_csv, "--days", str(days), "--tag", "closing"]
                    if season is not None:
                        cmd2 += ["--season", str(season)]
                    run(cmd2, env=env)
        except Exception as e:
            print(f"[warn] Fallback odds population failed for {league}: {e}")


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
    # Find any model file for the league
    model_files = list(models_dir.glob(f"*{league}*"))
    needs_train = True
    if model_files:
        newest = max(model_files, key=lambda p: p.stat().st_mtime)
        mtime = dt.datetime.fromtimestamp(newest.stat().st_mtime)
        needs_train = (dt.datetime.now() - mtime) > dt.timedelta(days=retrain_days)
        if not needs_train:
            print(f"[ok] Model fresh: {newest.name}")
    if needs_train:
        print(f"[step] Training XGB models for {league}")
        run([sys.executable, "xgb_trainer.py", "--league", league])


def show_predictions(league: str, days: int, season: Optional[int] = None) -> None:
    # Use bet_fusion top picks (with odds) and also per-match combined table
    env = os.environ.copy()
    env.setdefault("BOT_ODDS_MODE", "local")
    # Ensure fusion engine and helpers use the selected league
    env["BOT_LEAGUE"] = league
    # Pass fixtures horizon to fusion fallbacks
    env["BOT_FIXTURES_DAYS"] = str(days)
    if season is not None:
        env["BOT_SEASON"] = str(season)
    print("\n=== Top Picks (by EV) ===")
    run([sys.executable, "bet_fusion.py", "--with-odds", "--top", "20"], env=env)
    print("\n=== Best Per Match (TG, OU 2.5, 1X2) — by highest probability ===")
    run([sys.executable, "-m", "scripts.print_best_per_match", "--by", "prob"], env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command predictions orchestrator")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--league", help="League code, e.g., E0, I1, D1, SP1, F1")
    group.add_argument("--leagues", nargs="+", help="Space-separated list of league codes, e.g., E0 D1 F1 I1 SP1")
    ap.add_argument("--days", type=int, default=7, help="How many days ahead to fetch odds/fixtures (used for odds); fixtures can also use --from/--to")
    ap.add_argument("--from", dest="from_date", default=None, help="Optional fixtures window start date (YYYY-MM-DD)")
    ap.add_argument("--to", dest="to_date", default=None, help="Optional fixtures window end date (YYYY-MM-DD)")
    ap.add_argument("--season", type=int, default=None, help="Season start year for API-Football (e.g., 2025 for 2025-26)")
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

        # 3) Odds (include season for API-Football date windows/free plan compatibility)
        ensure_odds(league, days=args.days, season=args.season)

        # 4) Absences seed
        ensure_absences_seed(league)

        # 5) Train if needed
        ensure_trained(league, retrain_days=args.retrain_days)

        # 6) Show predictions
        # Pass fixtures window to fusion fallbacks via environment
        if args.from_date:
            os.environ["BOT_FIXTURES_FROM"] = str(args.from_date)
        if args.to_date:
            os.environ["BOT_FIXTURES_TO"] = str(args.to_date)
        show_predictions(league, days=args.days, season=args.season)

        # Optional: compact per-match prognostics for the next round
        if args.compact:
            print("\n=== Round Prognostics (compact) ===")
            run([sys.executable, "-m", "scripts.round_prognostics", "--league", league])

if __name__ == "__main__":
    main()
