"""
Fetch upcoming odds from The Odds API and save in internal JSON format.

Usage:
  python -m scripts.fetch_odds_toa --leagues E0 D1 F1 I1 SP1

Environment:
  THE_ODDS_API_KEY must be set (in .env or environment).

Output:
  data/odds/{LEAGUE}.json with structure:
    {"fixtures":[{"date": "...", "home": "...", "away": "...",
                  "markets": {
                    "1X2": {"main": {"home": 1.95, "draw": 3.20, "away": 4.00}},
                    "OU": {"2.5": {"main": {"Over": 1.90, "Under": 1.95}}}
                  }}]}
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import requests

import config

LEAGUE_TO_SPORT = {
    "E0": "soccer_epl",
    "D1": "soccer_germany_bundesliga",
    "F1": "soccer_france_ligue_one",
    "I1": "soccer_italy_serie_a",
    "SP1": "soccer_spain_la_liga",
}

DEFAULT_REGIONS = "eu"
DEFAULT_MARKETS = "h2h,totals"
TIMEOUT = 10


def choose_bookmaker(bms: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not bms:
        return None
    for bm in bms:
        if str(bm.get("title", "")).lower() == "bet365":
            return bm
    return bms[0]


def parse_event(ev: Dict[str, Any]) -> Dict[str, Any] | None:
    home = ev.get("home_team")
    away = ev.get("away_team")
    if not home or not away:
        return None
    date = ev.get("commence_time")
    bookmakers = ev.get("bookmakers") or []
    bm = choose_bookmaker(bookmakers)
    markets_out: Dict[str, Any] = {"1X2": {}, "OU": {}, "DC": {}, "Intervals": {}}
    if bm:
        for m in bm.get("markets", []):
            mkey = m.get("key")
            outcomes = m.get("outcomes") or []
            if mkey == "h2h":
                prices = {}
                for oc in outcomes:
                    name = str(oc.get("name", "")).strip().lower()
                    price = oc.get("price")
                    if price is None:
                        continue
                    if name == home.lower():
                        prices["home"] = float(price)
                    elif name == away.lower():
                        prices["away"] = float(price)
                    elif name == "draw":
                        prices["draw"] = float(price)
                if prices:
                    markets_out["1X2"]["main"] = prices
            elif mkey == "totals":
                for oc in outcomes:
                    try:
                        point = oc.get("point")
                        if point is None:
                            continue
                        line = str(float(point)).rstrip("0").rstrip(".")
                        price = oc.get("price")
                        name = str(oc.get("name", "")).strip().capitalize()
                        if price is None or name not in {"Over", "Under"}:
                            continue
                        bucket = markets_out["OU"].setdefault(line, {})
                        tag = bucket.setdefault("main", {})
                        tag[name] = float(price)
                    except Exception:
                        continue
    return {
        "date": date,
        "home": config.normalize_team_name(home),
        "away": config.normalize_team_name(away),
        "markets": markets_out,
    }


def fetch_for_league(league: str, api_key: str, regions: str) -> Dict[str, Any]:
    sport = LEAGUE_TO_SPORT.get(league.upper())
    if not sport:
        raise ValueError(f"No Odds API sport mapping for league {league}")
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        f"?apiKey={api_key}&regions={regions}&markets={DEFAULT_MARKETS}&oddsFormat=decimal&dateFormat=iso"
    )
    print(f"[{league}] GET {url}")
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    fixtures = []
    for ev in data:
        parsed = parse_event(ev)
        if parsed:
            fixtures.append(parsed)
    return {"fixtures": fixtures}


def save_odds(league: str, payload: Dict[str, Any]) -> None:
    out_dir = Path("data") / "odds"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{league}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[{league}] Saved odds -> {path} (fixtures: {len(payload.get('fixtures', []))})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch odds from The Odds API")
    ap.add_argument("--leagues", nargs="+", required=True, help="League codes e.g. E0 D1 F1 I1 SP1")
    ap.add_argument("--regions", default=DEFAULT_REGIONS, help="Regions (Odds API) default=eu")
    args = ap.parse_args()

    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise SystemExit("THE_ODDS_API_KEY not set in environment; cannot fetch odds.")

    failed = False
    for lg in args.leagues:
        lg_code = lg.strip().upper()
        try:
            payload = fetch_for_league(lg_code, api_key=api_key, regions=args.regions)
            save_odds(lg_code, payload)
            if not payload.get("fixtures"):
                print(f"[{lg_code}] Warning: 0 fixtures returned from TOA (odds coverage empty).")
        except Exception as e:
            failed = True
            print(f"[{lg_code}] Failed to fetch odds: {e}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
