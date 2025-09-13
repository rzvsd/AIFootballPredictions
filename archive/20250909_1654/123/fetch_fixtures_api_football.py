# fetch_fixtures_api_football.py
# Fetch fixtures from API-Football (api-sports) and append/update data/fixtures_master/{league}_{season}.csv
#
# Set env var API_FOOTBALL_KEY with your key (x-apisports-key).
# Usage:
#  python fetch_fixtures_api_football.py --league E0 --season 2025 --from 2025-08-15 --to 2025-09-15

import os, argparse, csv, requests, datetime, time
import pandas as pd

MASTER_DIR = os.path.join("data","fixtures_master")
MASTER_TPL = os.path.join(MASTER_DIR, "{league}_{season}.csv")

# Map your internal league code to api id if you need — for PL example:
LEAGUE_ID_MAP = {"E0": {"id": 39}}  # api-sports: 39 = Premier League (example)

def norm_team(x):
    # keep your normalizer from feature_store
    MAP = {
        "Manchester City":"Man City","Manchester United":"Man United",
        "Newcastle United":"Newcastle","Nottingham Forest":"Nott'm Forest",
        "Wolverhampton Wanderers":"Wolves","Tottenham Hotspur":"Tottenham",
        "Brighton & Hove Albion":"Brighton"
    }
    return MAP.get(x, x)

def api_fetch(league_id, date_from, date_to, season, key):
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": key}
    params = {"league": league_id, "season": season, "date_from": date_from, "date_to": date_to}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def write_master(league, season, rows):
    os.makedirs(MASTER_DIR, exist_ok=True)
    path = MASTER_TPL.format(league=league, season=season)
    # read existing
    if os.path.exists(path):
        df_old = pd.read_csv(path, dtype=str)
    else:
        df_old = pd.DataFrame(columns=["date","home_team","away_team"])
    df_new = pd.DataFrame(rows)[["date","home_team","away_team"]].drop_duplicates()
    merged = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates().sort_values("date")
    merged.to_csv(path, index=False)
    print(f"[fetch] master updated: {path}  (rows={len(merged)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--from", dest="dfrom", required=True)
    ap.add_argument("--to", dest="dto", required=True)
    args = ap.parse_args()

    # prefer standard name, but accept legacy env var names
    key = (
        os.environ.get("API_FOOTBALL_KEY")
        or os.environ.get("API_FOOTBALL_DATA")
        or os.environ.get("API_FOOTBALL_ODDS_KEY")
    )
    if not key:
        raise SystemExit("Set API_FOOTBALL_KEY or API_FOOTBALL_DATA (your API-Football key) in the environment.")

    league_cfg = LEAGUE_ID_MAP.get(args.league)
    if not league_cfg:
        raise SystemExit("Missing mapping for league id in LEAGUE_ID_MAP. Add mapping for your league code.")

    payload = api_fetch(league_cfg["id"], args.dfrom, args.dto, args.season, key)
    rows=[]
    for item in payload.get("response", []):
        fx = item.get("fixture", {})
        teams = item.get("teams", {})
        dt = fx.get("date")  # ISO UTC
        home = teams.get("home", {}).get("name")
        away = teams.get("away", {}).get("name")
        if dt and home and away:
            rows.append({"date": dt, "home_team": norm_team(home), "away_team": norm_team(away)})
    if not rows:
        print("[fetch] no fixtures found in API response")
    write_master(args.league, args.season, rows)

if __name__=="__main__":
    main()
