"""
Generate weekly fixtures CSV using football-data.org API.

Writes to data/fixtures/{LEAGUE}_weekly_fixtures.csv with columns: date,home,away

Usage:
  python -m scripts.gen_weekly_fixtures_from_fd --league D1 --days 14

Env:
  API_FOOTBALL_DATA or FOOTBALL_DATA_API_KEY
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List

import requests

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import config

FD_IDS = {'E0': 2021, 'SP1': 2014, 'I1': 2019, 'D1': 2002, 'F1': 2015}


def main() -> None:
    ap = argparse.ArgumentParser(description='Create weekly fixtures CSV from football-data.org')
    ap.add_argument('--league', required=True)
    ap.add_argument('--days', type=int, default=14)
    args = ap.parse_args()

    lg = args.league.upper().strip()
    comp_id = FD_IDS.get(lg)
    if not comp_id:
        raise SystemExit(f'Unknown league: {lg}')

    api_fd = os.getenv('API_FOOTBALL_DATA') or os.getenv('FOOTBALL_DATA_API_KEY')
    if not api_fd:
        raise SystemExit('Missing football-data.org token in API_FOOTBALL_DATA or FOOTBALL_DATA_API_KEY')

    today = dt.date.today()
    dfrom = today.isoformat()
    dto = (today + dt.timedelta(days=args.days)).isoformat()

    headers = {'X-Auth-Token': api_fd}
    params = {'dateFrom': dfrom, 'dateTo': dto}
    r = requests.get(f'https://api.football-data.org/v4/competitions/{comp_id}/matches', headers=headers, params=params, timeout=20)
    if r.status_code != 200:
        raise SystemExit(f'HTTP {r.status_code}: {r.text[:200]}')

    recs: List[Dict[str, str]] = []
    for m in r.json().get('matches', []):
        try:
            d = (m.get('utcDate') or '').replace('T', ' ').replace('Z', '')
            h = config.normalize_team_name((m.get('homeTeam') or {}).get('name') or '')
            a = config.normalize_team_name((m.get('awayTeam') or {}).get('name') or '')
            if d and h and a:
                recs.append({'date': d, 'home': h, 'away': a})
        except Exception:
            continue

    out_dir = Path('data') / 'fixtures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{lg}_weekly_fixtures.csv'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('date,home,away\n')
        for r in recs:
            f.write(f"{r['date']},{r['home']},{r['away']}\n")
    print(f'Saved fixtures CSV -> {out_path}  (rows: {len(recs)})')


if __name__ == '__main__':
    main()
