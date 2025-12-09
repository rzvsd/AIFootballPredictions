"""
Generate a weekly fixtures CSV for a league using API-Football.

Writes to data/fixtures/{LEAGUE}_weekly_fixtures.csv with columns: date,home,away

Usage:
  python -m scripts.gen_weekly_fixtures_from_api --league D1 --days 14 [--season 2025]

Env:
  API_FOOTBALL_KEY or API_FOOTBALL_ODDS_KEY
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Dict, List

import json
import requests

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import config


API_BASE = "https://v3.football.api-sports.io"
LEAGUE_IDS: Dict[str, int] = {
    'E0': 39,   # Premier League
    'D1': 78,   # Bundesliga
    'F1': 61,   # Ligue 1
    'I1': 135,  # Serie A
    'SP1': 140, # La Liga
}


def _headers() -> Dict[str, str]:
    key = (
        os.getenv('API_FOOTBALL_ODDS_KEY')
        or os.getenv('API_FOOTBALL_KEY')
        or os.getenv('API_FOOTBALL_DATA')  # legacy var accepted as fallback
    )
    if not key:
        raise RuntimeError('Missing API key: set API_FOOTBALL_KEY or API_FOOTBALL_ODDS_KEY')
    return {'x-apisports-key': key}


def _season_for_today() -> int:
    today = dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def _get_fixtures_by_window(league_id: int, dfrom: str, dto: str, season: int | None) -> List[Dict]:
    params = {
        'league': league_id,
        'season': season or _season_for_today(),
        'from': dfrom,
        'to': dto,
    }
    r = requests.get(f"{API_BASE}/fixtures", params=params, headers=_headers(), timeout=20)
    r.raise_for_status()
    data = r.json().get('response', [])
    out: List[Dict] = []
    for it in data:
        fx = it.get('fixture', {})
        teams = it.get('teams', {})
        out.append({
            'date': (fx.get('date') or '').replace('T', ' ').replace('Z', '')[:19],
            'home_api': (teams.get('home') or {}).get('name'),
            'away_api': (teams.get('away') or {}).get('name'),
        })
    return out


def _get_next_fixtures(league_id: int, n: int, season: int | None) -> List[Dict]:
    params = {'league': league_id, 'next': int(n)}
    # try with season first, then without
    data = []
    try:
        p = dict(params)
        p['season'] = season or _season_for_today()
        r = requests.get(f"{API_BASE}/fixtures", params=p, headers=_headers(), timeout=20)
        r.raise_for_status()
        data = r.json().get('response', [])
    except Exception:
        data = []
    if not data:
        try:
            r2 = requests.get(f"{API_BASE}/fixtures", params=params, headers=_headers(), timeout=20)
            r2.raise_for_status()
            data = r2.json().get('response', [])
        except Exception:
            data = []
    out: List[Dict] = []
    for it in data:
        fx = it.get('fixture', {})
        teams = it.get('teams', {})
        out.append({
            'date': (fx.get('date') or '').replace('T', ' ').replace('Z', '')[:19],
            'home_api': (teams.get('home') or {}).get('name'),
            'away_api': (teams.get('away') or {}).get('name'),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Create weekly fixtures CSV from API-Football')
    ap.add_argument('--league', required=True, help='League code, e.g., D1')
    ap.add_argument('--days', type=int, default=14, help='Days ahead to include (window)')
    ap.add_argument('--season', type=int, default=None, help='Season start year (e.g., 2025)')
    args = ap.parse_args()

    lg = args.league.upper().strip()
    league_id = LEAGUE_IDS.get(lg)
    if not league_id:
        print(f"Unknown league id for {lg}")
        sys.exit(1)

    today = dt.date.today()
    dfrom = today.isoformat()
    dto = (today + dt.timedelta(days=args.days)).isoformat()

    rows = []
    try:
        rows = _get_fixtures_by_window(league_id, dfrom, dto, args.season)
    except Exception as e:
        print(f"[warn] window fetch failed: {e}")
        rows = []
    if not rows:
        try:
            rows = _get_next_fixtures(league_id, max(args.days, 20), args.season)
            if rows:
                print(f"[info] Using 'next' fixtures fallback (N={len(rows)}).")
        except Exception as e:
            print(f"[warn] next fixtures fetch failed: {e}")
            rows = []

    # Map API names to internal names
    mapped = []
    for r in rows:
        h = config.normalize_team_name(str(r.get('home_api') or '').strip())
        a = config.normalize_team_name(str(r.get('away_api') or '').strip())
        d = str(r.get('date') or '').strip()
        if h and a and d:
            mapped.append({'date': d, 'home': h, 'away': a})

    out_dir = Path('data') / 'fixtures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{lg}_weekly_fixtures.csv"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('date,home,away\n')
        for m in mapped:
            f.write(f"{m['date']},{m['home']},{m['away']}\n")
    print(f"Saved fixtures CSV -> {out_path}  (rows: {len(mapped)})")


if __name__ == '__main__':
    main()
