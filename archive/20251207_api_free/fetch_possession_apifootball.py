"""
Fetch possession percentages from API-Football and write a per-match CSV.

Requires env API_FOOTBALL_KEY. Uses league ID from config (LEAGUE_ID_API) unless overridden.

Outputs: data/processed/{LEAGUE}_possession.csv with columns:
  Date (YYYY-MM-DD), HomeTeam, AwayTeam, HomePoss, AwayPoss

Usage examples:
  python -m scripts.fetch_possession_apifootball --league E0 --season 2025 --out data/processed/E0_possession.csv
  python -m scripts.fetch_possession_apifootball --league E0 --season 2025 --dates 2025-08-01,2025-09-30
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple
import requests
import pandas as pd
from pathlib import Path
import config as cfg

# Load .env if present at repo root
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


API_URL = "https://v3.football.api-sports.io"


def _headers() -> Dict[str,str]:
    key = os.getenv('API_FOOTBALL_KEY')
    if not key:
        raise SystemExit('API_FOOTBALL_KEY not set in environment')
    return {"x-apisports-key": key}


def _league_id_for(league: str) -> int:
    if hasattr(cfg, 'LEAGUE_ID_API') and league == cfg.LEAGUE_CODE:
        return int(getattr(cfg, 'LEAGUE_ID_API'))
    # fallback map
    m = { 'E0': 39, 'D1': 78, 'F1': 61, 'SP1': 140, 'I1': 135 }
    return int(m.get(league, 39))


def fetch_fixtures(league_id: int, season: int, date_from: str|None=None, date_to: str|None=None) -> List[Dict]:
    params = { 'league': league_id, 'season': season }
    if date_from: params['from'] = date_from
    if date_to: params['to'] = date_to
    r = requests.get(f"{API_URL}/fixtures", headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get('response', [])
    return data


def fetch_stats_for_fixture(fixture_id: int) -> Dict[str, Dict[str, float]]:
    r = requests.get(f"{API_URL}/fixtures/statistics", headers=_headers(), params={'fixture': fixture_id}, timeout=30)
    r.raise_for_status()
    resp = r.json().get('response', [])
    out: Dict[str, Dict[str, float]] = {}
    for team_stats in resp:
        team = team_stats.get('team', {}).get('name')
        stats = team_stats.get('statistics', [])
        poss = None
        for s in stats:
            if str(s.get('type')).lower().strip() == 'ball possession':
                v = s.get('value')
                if isinstance(v, str) and v.endswith('%'):
                    try:
                        poss = float(v.strip('%'))
                    except Exception:
                        poss = None
                elif isinstance(v, (int,float)):
                    poss = float(v)
        if team and poss is not None:
            out[team] = {'poss': poss}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Fetch possession from API-Football')
    ap.add_argument('--league', default='E0')
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--dates', default=None, help='Optional date range: YYYY-MM-DD,YYYY-MM-DD')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    lid = _league_id_for(args.league)
    date_from = date_to = None
    if args.dates and ',' in args.dates:
        date_from, date_to = [s.strip() for s in args.dates.split(',',1)]
    fixtures = fetch_fixtures(lid, int(args.season), date_from, date_to)
    rows: List[Dict] = []
    for fx in fixtures:
        fid = fx.get('fixture',{}).get('id')
        dt = fx.get('fixture',{}).get('date')
        home_api = fx.get('teams',{}).get('home',{}).get('name')
        away_api = fx.get('teams',{}).get('away',{}).get('name')
        if not fid or not home_api or not away_api:
            continue
        try:
            stats = fetch_stats_for_fixture(int(fid))
        except Exception:
            continue
        hp = stats.get(home_api,{}).get('poss')
        ap = stats.get(away_api,{}).get('poss')
        if hp is None or ap is None:
            continue
        # Normalize team names to internal dataset names
        home = cfg.normalize_team_name(str(home_api))
        away = cfg.normalize_team_name(str(away_api))
        date_short = str(pd.to_datetime(dt, errors='coerce').date())
        rows.append({'Date': date_short, 'HomeTeam': home, 'AwayTeam': away, 'HomePoss': hp, 'AwayPoss': ap})

    if not rows:
        print('No possession rows fetched.')
        return
    out_df = pd.DataFrame(rows)
    out_path = args.out or str(Path('data')/'processed'/f"{args.league}_possession.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved possession -> {out_path} (rows={len(out_df)})")


if __name__ == '__main__':
    main()
