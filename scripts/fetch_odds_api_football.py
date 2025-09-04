"""
Fetch real odds from API-Football and write local odds JSON used by bookmaker_api (ODDS_MODE=local).

Usage examples:
  # from fixtures CSV (recommended for name alignment)
  python -m scripts.fetch_odds_api_football --league E0 --fixtures-csv data/fixtures/E0_weekly_fixtures.csv

  # or from a date window (uses API fixtures)
  python -m scripts.fetch_odds_api_football --league E0 --days 7

Env:
  API_FOOTBALL_KEY (or API_FOOTBALL_ODDS_KEY)
  BOT_ODDS_DIR (default: data/odds)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import datetime as dt
from typing import Dict, List, Optional

import requests

# Load .env if present (project root)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import config


API_BASE = "https://v3.football.api-sports.io"

LEAGUE_IDS = {
    'E0': 39,   # Premier League
    'D1': 78,   # Bundesliga
    'F1': 61,   # Ligue 1
    'I1': 135,  # Serie A
    'SP1': 140, # La Liga
}


def _headers() -> Dict[str, str]:
    key = os.getenv('API_FOOTBALL_ODDS_KEY') or os.getenv('API_FOOTBALL_KEY')
    if not key:
        raise RuntimeError('Missing API key: set API_FOOTBALL_KEY or API_FOOTBALL_ODDS_KEY')
    return {'x-apisports-key': key}


def _season_for_today() -> int:
    today = dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def _get_fixtures_from_api(league_id: int, days: int) -> List[Dict]:
    today = dt.date.today()
    params = {
        'league': league_id,
        'season': _season_for_today(),
        'from': today.isoformat(),
        'to': (today + dt.timedelta(days=days)).isoformat(),
    }
    r = requests.get(f"{API_BASE}/fixtures", params=params, headers=_headers(), timeout=20)
    r.raise_for_status()
    data = r.json().get('response', [])
    fixtures = []
    for it in data:
        fx = it.get('fixture', {})
        teams = it.get('teams', {})
        fixtures.append({
            'fixture_id': fx.get('id'),
            'date': (fx.get('date') or '')[:10],
            'home_api': (teams.get('home') or {}).get('name'),
            'away_api': (teams.get('away') or {}).get('name'),
        })
    return fixtures


def _read_fixtures_csv(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get('date')
    home_col = cols.get('home') or cols.get('home_team')
    away_col = cols.get('away') or cols.get('away_team')
    res = []
    for _, r in df.iterrows():
        res.append({
            'date': str(r.get(date_col, ''))[:10] if date_col else '',
            'home_csv': str(r.get(home_col, '')).strip(),
            'away_csv': str(r.get(away_col, '')).strip(),
        })
    return res


def _map_names_to_api(fixtures_csv: List[Dict], league_id: int) -> List[Dict]:
    """Map CSV names to API fixtures by fuzzy matching team names/date."""
    import difflib
    # Pull next 10 days of fixtures to match
    api_fixtures = _get_fixtures_from_api(league_id, days=10)
    out = []
    for row in fixtures_csv:
        hcand = [fx for fx in api_fixtures]
        best = None
        best_score = 0.0
        for fx in hcand:
            score = 0.0
            # Date boost if matches
            if row['date'] and fx['date'] == row['date']:
                score += 0.2
            # Fuzzy for names
            home_csv = config.normalize_team_name(row['home_csv'])
            away_csv = config.normalize_team_name(row['away_csv'])
            score += difflib.SequenceMatcher(None, home_csv, fx['home_api'] or '').ratio()
            score += difflib.SequenceMatcher(None, away_csv, fx['away_api'] or '').ratio()
            if score > best_score:
                best_score = score
                best = fx
        if best and best_score >= 1.2:  # heuristic threshold
            out.append({
                'fixture_id': best['fixture_id'],
                'date': best['date'],
                'home': config.normalize_team_name(best['home_api'] or row['home_csv']),
                'away': config.normalize_team_name(best['away_api'] or row['away_csv']),
            })
        else:
            out.append({
                'fixture_id': None,
                'date': row.get('date', ''),
                'home': row['home_csv'],
                'away': row['away_csv'],
            })
    return out


def _fetch_odds_for_fixture(fixture_id: Optional[int]) -> Dict:
    if not fixture_id:
        return {}
    r = requests.get(f"{API_BASE}/odds", params={'fixture': fixture_id}, headers=_headers(), timeout=20)
    r.raise_for_status()
    resp = r.json().get('response', [])
    # Aggregate across bookmakers: choose first bookmaker available for simplicity
    out = {'1X2': {}, 'OU': {}, 'TG': {}}
    for item in resp:
        for book in item.get('bookmakers', []) or []:
            for bet in book.get('bets', []) or []:
                name = (bet.get('name') or '').lower()
                values = bet.get('values', []) or []
                if 'match winner' in name or '1x2' in name:
                    for v in values:
                        vn = (v.get('value') or '').lower()
                        odd = v.get('odd')
                        if vn in ('home', '1'):
                            out['1X2']['home'] = float(odd)
                        elif vn in ('draw', 'x'):
                            out['1X2']['draw'] = float(odd)
                        elif vn in ('away', '2'):
                            out['1X2']['away'] = float(odd)
                if 'over/under' in name or 'goals over/under' in name:
                    for v in values:
                        vn = str(v.get('value') or '')
                        odd = v.get('odd')
                        if vn == '2.5':
                            # API-Football returns separate Over/Under entries sometimes
                            label = (v.get('label') or '').lower()
                            if 'over' in label:
                                out['OU'].setdefault('2.5', {})['Over'] = float(odd)
                            elif 'under' in label:
                                out['OU'].setdefault('2.5', {})['Under'] = float(odd)
                if 'total goals' in name and 'interval' in name:
                    # Rare; if available, fill TG bands
                    for v in values:
                        vn = str(v.get('value') or '')
                        odd = v.get('odd')
                        out['TG'][vn] = float(odd)
            # Use just first bookmaker for simplicity
            break
        # Use just first item/fixture occurrence
        break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Fetch odds from API-Football and write data/odds/{LEAGUE}.json')
    ap.add_argument('--league', required=True, help='League code (E0,D1,F1,I1,SP1)')
    ap.add_argument('--fixtures-csv', default=None, help='Fixtures CSV to drive matching (preferred)')
    ap.add_argument('--days', type=int, default=7, help='If no CSV, how many days ahead to fetch')
    args = ap.parse_args()

    lg = args.league
    league_id = LEAGUE_IDS.get(lg)
    if not league_id:
        print(f"Unknown league id for {lg}")
        sys.exit(1)

    if args.fixtures_csv:
        csv_rows = _read_fixtures_csv(args.fixtures_csv)
        mapped = _map_names_to_api(csv_rows, league_id)
    else:
        api_fix = _get_fixtures_from_api(league_id, args.days)
        mapped = []
        for fx in api_fix:
            mapped.append({
                'fixture_id': fx['fixture_id'],
                'date': fx['date'],
                'home': config.normalize_team_name(fx['home_api'] or ''),
                'away': config.normalize_team_name(fx['away_api'] or ''),
            })

    out_fixtures = []
    for m in mapped:
        odds = _fetch_odds_for_fixture(m.get('fixture_id')) if m.get('fixture_id') else {}
        out_fixtures.append({
            'date': m.get('date'),
            'home': m.get('home'),
            'away': m.get('away'),
            'markets': odds,
        })

    odds_dir = os.getenv('BOT_ODDS_DIR', os.path.join('data','odds'))
    os.makedirs(odds_dir, exist_ok=True)
    path = os.path.join(odds_dir, f"{lg}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'fixtures': out_fixtures}, f, indent=2)
    print(f"Saved odds -> {path}  (fixtures: {len(out_fixtures)})")


if __name__ == '__main__':
    main()
