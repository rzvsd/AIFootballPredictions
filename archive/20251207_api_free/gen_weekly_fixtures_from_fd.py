"""
Generate weekly fixtures CSV using football-data.org when available,
otherwise synthesize fixtures from processed data.

Usage:
    python -m scripts.gen_weekly_fixtures_from_fd --league E0 --days 7

Output:
    data/fixtures/{LEAGUE}_weekly_fixtures.csv with columns: date, home, away
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import config

FD_IDS = {'E0': 2021, 'SP1': 2014, 'I1': 2019, 'D1': 2002, 'F1': 2015}


def guess_weekend_dates(start: dt.date, count: int) -> List[dt.datetime]:
    slots: List[dt.datetime] = []
    day = start
    for i in range(count):
        hour = 13 if i % 2 == 0 else 16
        slots.append(dt.datetime.combine(day, dt.time(hour=hour, minute=30)))
        if (i + 1) % 4 == 0:
            day += dt.timedelta(days=1)
    return slots


def build_pairings(teams: List[str]) -> List[tuple[str, str]]:
    teams = sorted(teams)
    if len(teams) % 2 == 1:
        teams.append("BYE")
    mid = len(teams) // 2
    homes = teams[:mid]
    aways = teams[mid:]
    pairings = []
    for h, a in zip(homes, aways):
        if "BYE" in (h, a):
            continue
        pairings.append((h, a))
    return pairings


def fetch_from_api(league: str, days: int) -> List[Dict[str, str]]:
    token = os.getenv('API_FOOTBALL_DATA') or os.getenv('FOOTBALL_DATA_API_KEY')
    comp_id = FD_IDS.get(league)
    if not token or not comp_id or requests is None:
        return []
    today = dt.date.today()
    params = {
        'dateFrom': today.isoformat(),
        'dateTo': (today + dt.timedelta(days=days)).isoformat(),
    }
    headers = {'X-Auth-Token': token}
    try:
        resp = requests.get(
            f'https://api.football-data.org/v4/competitions/{comp_id}/matches',
            params=params,
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
    except Exception:
        return []
    fixtures = []
    for m in resp.json().get('matches', []):
        utc = (m.get('utcDate') or '').replace('T', ' ').replace('Z', '')
        home = config.normalize_team_name((m.get('homeTeam') or {}).get('name') or '')
        away = config.normalize_team_name((m.get('awayTeam') or {}).get('name') or '')
        if utc and home and away:
            fixtures.append({'date': utc, 'home': home, 'away': away})
    return fixtures


def fallback_from_processed(league: str, days: int) -> List[Dict[str, str]]:
    proc_path = Path('data') / 'processed' / f'{league}_merged_preprocessed.csv'
    if not proc_path.exists():
        raise SystemExit(f"Processed file missing: {proc_path}")
    df = pd.read_csv(proc_path)
    if 'Date' not in df.columns:
        raise SystemExit("Processed data lacks 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam']).sort_values('Date')
    today = dt.date.today()
    window_end = today + dt.timedelta(days=days)
    mask = (df['Date'].dt.date >= today) & (df['Date'].dt.date <= window_end)
    window = df[mask]
    if not window.empty:
        fixtures = []
        for _, row in window.iterrows():
            fixtures.append(
                {
                    'date': row['Date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'home': config.normalize_team_name(str(row['HomeTeam'])),
                    'away': config.normalize_team_name(str(row['AwayTeam'])),
                }
            )
        return fixtures
    teams = sorted(set(df['HomeTeam']) | set(df['AwayTeam']))
    pairings = build_pairings(teams)
    slots = guess_weekend_dates(today, len(pairings))
    fixtures = []
    for (home, away), kickoff in zip(pairings, slots):
        fixtures.append(
            {
                'date': kickoff.strftime('%Y-%m-%d %H:%M:%S'),
                'home': config.normalize_team_name(home),
                'away': config.normalize_team_name(away),
            }
        )
    return fixtures


def main() -> None:
    ap = argparse.ArgumentParser(description='Create weekly fixtures CSV with API fallback.')
    ap.add_argument('--league', required=True, help='League code (E0, D1, F1, I1, SP1)')
    ap.add_argument('--days', type=int, default=7, help='Look-ahead window (days)')
    args = ap.parse_args()

    league = args.league.upper().strip()
    fixtures = fetch_from_api(league, args.days)
    if not fixtures:
        print(f"[info] API unavailable or empty for {league}; using processed data fallback.")
        fixtures = fallback_from_processed(league, args.days)

    out_dir = Path('data') / 'fixtures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{league}_weekly_fixtures.csv'
    pd.DataFrame(fixtures).to_csv(out_path, index=False)
    print(f"Saved fixtures CSV -> {out_path} (rows: {len(fixtures)})")


if __name__ == '__main__':
    main()
