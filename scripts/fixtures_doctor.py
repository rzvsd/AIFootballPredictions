"""
Validate and normalize fixtures CSV files ("fixtures doctor").

Checks:
- Required columns: date, home/home_team, away/away_team
- Date parsing (YYYY-MM-DD preferred)
- Team-name normalization via config.normalize_team_name
- Unknown teams (suggest closest known names)
- Duplicates (by date+home+away after normalization)

Outputs:
- Cleaned CSV with normalized names (same columns)
- Console report of issues; optional JSON report via --report

Usage:
  python -m scripts.fixtures_doctor --league E0 \
    --input data/fixtures/E0_weekly_fixtures.csv \
    --output data/fixtures/E0_weekly_fixtures_clean.csv --report data/fixtures/E0_doctor.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import difflib
import pandas as pd

import config


def _known_teams_for_league(league: str) -> List[str]:
    # Prefer enhanced final features -> HomeTeam/AwayTeam unique
    enh = os.path.join('data', 'enhanced', f'{league}_final_features.csv')
    if os.path.exists(enh):
        try:
            df = pd.read_csv(enh)
            teams = sorted(set(df.get('HomeTeam', pd.Series(dtype=str)).astype(str)) |
                           set(df.get('AwayTeam', pd.Series(dtype=str)).astype(str)))
            if teams:
                return teams
        except Exception:
            pass
    # Fallback to processed
    proc = os.path.join('data', 'processed', f'{league}_merged_preprocessed.csv')
    if os.path.exists(proc):
        try:
            df = pd.read_csv(proc)
            teams = sorted(set(df.get('HomeTeam', pd.Series(dtype=str)).astype(str)) |
                           set(df.get('AwayTeam', pd.Series(dtype=str)).astype(str)))
            if teams:
                return teams
        except Exception:
            pass
    # Last resort: values from TEAM_NAME_MAP
    try:
        mapping_vals = set(config.TEAM_NAME_MAP.values())
        return sorted(list(mapping_vals))
    except Exception:
        return []


def _normalize_team(name: str) -> str:
    try:
        return config.normalize_team_name(str(name))
    except Exception:
        return str(name)


def _suggest(name: str, universe: List[str], n: int = 3) -> List[str]:
    try:
        return difflib.get_close_matches(str(name), universe, n=n, cutoff=0.5)
    except Exception:
        return []


def _read_fixtures(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get('date')
    home_col = cols.get('home') or cols.get('home_team')
    away_col = cols.get('away') or cols.get('away_team')
    if not (date_col and home_col and away_col):
        raise ValueError('Fixtures CSV must contain columns: date, home/home_team, away/away_team')
    # Normalize column names for processing
    out = pd.DataFrame({
        'date': df[date_col],
        'home': df[home_col],
        'away': df[away_col],
    })
    return out, {'date': date_col, 'home': home_col, 'away': away_col}


def process(league: str, in_csv: str, out_csv: str | None, report_json: str | None) -> int:
    known = _known_teams_for_league(league)
    df, orig_cols = _read_fixtures(in_csv)

    # Parse/standardize date to YYYY-MM-DD where possible
    try:
        dd = pd.to_datetime(df['date'], errors='coerce')
        df['date_std'] = dd.dt.strftime('%Y-%m-%d')
        df['date_std'] = df['date_std'].fillna(df['date'].astype(str))
    except Exception:
        df['date_std'] = df['date'].astype(str)

    # Normalize names
    df['home_norm'] = df['home'].astype(str).map(_normalize_team)
    df['away_norm'] = df['away'].astype(str).map(_normalize_team)

    issues: Dict[str, List[Dict[str, str]]] = {
        'unknown_teams': [], 'duplicates': [], 'bad_dates': []
    }

    # Unknowns
    for side in ('home_norm', 'away_norm'):
        mask = ~df[side].isin(known)
        for _, r in df[mask].iterrows():
            suggestions = _suggest(r[side], known)
            issues['unknown_teams'].append({
                'date': str(r['date_std']),
                'input': str(r['home'] if side == 'home_norm' else r['away']),
                'normalized': str(r[side]),
                'suggestions': ', '.join(suggestions) if suggestions else ''
            })

    # Bad dates
    try:
        bad_mask = pd.to_datetime(df['date_std'], errors='coerce').isna()
        for _, r in df[bad_mask].iterrows():
            issues['bad_dates'].append({'value': str(r['date'])})
    except Exception:
        pass

    # Duplicates
    dkey = df[['date_std', 'home_norm', 'away_norm']].astype(str)
    dup_mask = dkey.duplicated(keep=False)
    for _, r in df[dup_mask].sort_values(['date_std','home_norm','away_norm']).iterrows():
        issues['duplicates'].append({
            'date': str(r['date_std']), 'home': str(r['home_norm']), 'away': str(r['away_norm'])
        })

    # Build cleaned output
    cleaned = pd.DataFrame({
        orig_cols['date']: df['date_std'],
        orig_cols['home']: df['home_norm'],
        orig_cols['away']: df['away_norm'],
    })

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        cleaned.to_csv(out_csv, index=False)

    # Console report
    print(f"[fixtures-doctor] League={league} input={in_csv}")
    print(f"- Teams known: {len(known)}; Rows: {len(df)}; Cleaned saved: {bool(out_csv)}")
    print(f"- Unknown teams: {len(issues['unknown_teams'])}; Duplicates: {len(issues['duplicates'])}; Bad dates: {len(issues['bad_dates'])}")
    if issues['unknown_teams']:
        print("  Examples (first 5):")
        for it in issues['unknown_teams'][:5]:
            print(f"   {it['date']}: '{it['input']}' -> '{it['normalized']}'  sugg: {it['suggestions']}")

    if report_json:
        os.makedirs(os.path.dirname(report_json) or '.', exist_ok=True)
        with open(report_json, 'w', encoding='utf-8') as f:
            json.dump(issues, f, indent=2)
        print(f"- Report saved -> {report_json}")

    # Return non-zero if issues present (except benign date formatting)
    err_count = len(issues['unknown_teams']) + len(issues['duplicates'])
    return 0 if err_count == 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description='Validate and normalize fixtures CSVs (fixtures doctor)')
    ap.add_argument('--league', required=True, help='League code, e.g., E0')
    ap.add_argument('--input', required=True, help='Input fixtures CSV path')
    ap.add_argument('--output', default=None, help='Output cleaned CSV path')
    ap.add_argument('--report', default=None, help='Optional JSON report with findings')
    args = ap.parse_args()
    code = process(args.league, args.input, args.output, args.report)
    raise SystemExit(code)


if __name__ == '__main__':
    main()

