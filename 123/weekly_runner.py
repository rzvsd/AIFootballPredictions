# weekly_runner.py
# Chains the weekly workflow for a league (offline picks).
#
# Usage:
#   python weekly_runner.py --league E0 --season 2025 --as-of 2025-08-18 \
#     --min-prob 0.55 --min-conf 0.65 --top-k 30
#
# If --from/--to are omitted, it builds the next Tue→Mon window from today.

import os, sys, subprocess
import argparse
from datetime import date, timedelta

# Try preferred data sources automatically (API first, then football-data.org)
def auto_fetch_master_for_window(league, season, dfrom, dto):
    # try API-Football first
    try:
        if os.environ.get("API_FOOTBALL_KEY"):
            subprocess.run([sys.executable, "fetch_fixtures_api_football.py", "--league", league, "--season", season, "--from", dfrom, "--to", dto], check=True)
            print("[fetch] api-football success")
            return
    except Exception as e:
        print("[fetch] api-football failed:", e)
    
    # try football-data.org
    try:
        if os.environ.get("FOOTBALL_DATA_KEY"):
            subprocess.run([sys.executable, "fetch_fixtures_football_data_org.py", "--league", league, "--season", season, "--from", dfrom, "--to", dto], check=True)
            print("[fetch] football-data.org success")
            return
    except Exception as e:
        print("[fetch] football-data.org failed:", e)
    
    # otherwise, fall back to scraper (fragile)
    print("[fetch] falling back to fbref scraper")
    subprocess.run([sys.executable, "fetch_fixtures_fbref.py", "--league", league, "--season", season, "--from", dfrom, "--to", dto])

def next_tue_to_mon(today: date):
    days_to_tue = (1 - today.weekday()) % 7
    start = today + timedelta(days=days_to_tue or 7)
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()

def run(cmd):
    print(">", " ".join(cmd))
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def main():
    ap = argparse.ArgumentParser(description="Weekly chain: feature snapshot → DC snapshot → fixtures → picks")
    ap.add_argument("--league", required=True)
    ap.add_argument("--season", required=True)
    ap.add_argument("--as-of", required=True, help="Round-end date (YYYY-MM-DD) to freeze snapshots")
    ap.add_argument("--from", dest="dfrom", default=None)
    ap.add_argument("--to", dest="dto", default=None)
    ap.add_argument("--tau-days", type=float, default=150.0)
    ap.add_argument("--min-prob", type=float, default=0.55)
    ap.add_argument("--min-conf", type=float, default=0.65)
    ap.add_argument("--top-k", type=int, default=30)
    args = ap.parse_args()

    if not (args.dfrom and args.dto):
        args.dfrom, args.dto = next_tue_to_mon(date.today())

    # 1) Auto-fetch fresh data for rolling horizon (28 days)
    print(f"\n[weekly] fetching fresh data for {args.league} {args.season}")
    auto_fetch_master_for_window(args.league, args.season, args.dfrom, args.dto)

    # 2) Feature snapshot
    run([sys.executable, "feature_store.py", "--league", args.league, "--as-of", args.as_of])

    # 3) Recency DC snapshot
    run([sys.executable, "dixon_coles_trainer_recency.py", "--league", args.league, "--as-of", args.as_of, "--tau-days", str(args.tau_days)])

    # 4) Weekly fixtures from master (offline)
    run([sys.executable, "fixtures_autogen.py", "--league", args.league, "--season", args.season,
         "--from", args.dfrom, "--to", args.dto])

    # 5) Picks (offline)
    run([sys.executable, "bet_fusion.py", "--league", args.league, "--from", args.dfrom, "--to", args.dto,
         "--source", "local", "--min-prob", str(args.min_prob), "--min-conf", str(args.min_conf),
         "--top-k", str(args.top_k)])

    print("\n[weekly] complete ✅")

if __name__ == "__main__":
    main()
