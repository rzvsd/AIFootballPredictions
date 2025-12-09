"""
Multi-league shopping-list report (target odds + real odds if available).

Example:
  python -m scripts.multi_league_report --leagues E0 D1 F1 I1 --select prob --export
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Set, Optional

import pandas as pd

import bet_fusion as fusion
import config as cfgmod


def _read_fixtures_filter(path: str) -> Set[Tuple[str, str, str]]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read fixtures CSV '{path}': {e}")
        return set()
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get('date')
    home_col = cols.get('home') or cols.get('home_team')
    away_col = cols.get('away') or cols.get('away_team')
    if not (date_col and home_col and away_col):
        print("Fixtures CSV must contain columns: date, home/home_team, away/away_team")
        return set()
    try:
        dd = pd.to_datetime(df[date_col], errors='coerce')
        df['_date'] = dd.dt.strftime('%Y-%m-%d')
    except Exception:
        df['_date'] = df[date_col].astype(str)
    df['_home'] = df[home_col].astype(str).str.strip().apply(cfgmod.normalize_team_name)
    df['_away'] = df[away_col].astype(str).str.strip().apply(cfgmod.normalize_team_name)
    return set(df[['_date','_home','_away']].itertuples(index=False, name=None))


def _select_pick(group: pd.DataFrame, market_name: str, mode: str) -> Optional[pd.Series]:
    g = group[group['market'] == market_name]
    if g.empty:
        return None
    if mode == 'ev' and 'EV' in g.columns:
        return g.sort_values(['EV','prob'], ascending=[False, False]).iloc[0]
    return g.sort_values(['prob','EV'], ascending=[False, False]).iloc[0]


def _pick_str(row: Optional[pd.Series]) -> str:
    if row is None:
        return '-'
    try:
        prob = f"{float(row.get('prob', 0.0)):.1%}"
    except Exception:
        prob = "-"
    fair = row.get('fair_odds')
    book = row.get('book_odds')
    src = str(row.get('price_source',''))
    ev_disp = ""
    if src == 'real':
        try:
            ev_disp = f"EV {float(row.get('EV', 0.0)):.2f}"
        except Exception:
            ev_disp = ""
    fair_str = f"fair {float(fair):.2f}" if fair is not None and pd.notna(fair) else "fair -"
    book_str = f"book {float(book):.2f}" if book is not None and pd.notna(book) else ""
    parts = [f"{row.get('market')} {row.get('outcome')}", prob, fair_str]
    if book_str:
        parts.append(f"{book_str} ({src})" if src else book_str)
    if ev_disp:
        parts.append(ev_disp)
    return " | ".join(parts)


def league_report(league: str, fixtures_csv: Optional[str], select: str, use_calibration: bool) -> pd.DataFrame:
    cfg = fusion.load_config()
    cfg['league'] = league
    if fixtures_csv:
        cfg['fixtures_csv'] = fixtures_csv
    if not use_calibration:
        cfg['use_calibration'] = False
    mb = fusion.generate_market_book(cfg)
    if mb.empty:
        return pd.DataFrame()

    # Optional filter by fixtures CSV (single round)
    if fixtures_csv:
        wanted = _read_fixtures_filter(fixtures_csv)
        if wanted:
            def _key(row):
                try:
                    d = pd.to_datetime(row['date'], errors='coerce')
                    ds = d.strftime('%Y-%m-%d') if d is not None and not pd.isna(d) else str(row['date'])
                except Exception:
                    ds = str(row['date'])
                return (ds, str(row['home']).strip(), str(row['away']).strip())
            mb = mb[mb.apply(lambda r: _key(r) in wanted, axis=1)].copy()
            if mb.empty:
                return pd.DataFrame()

    # Keep 1X2, OU 2.5, and selected TG intervals
    tg_allowed = {"0-3","1-3","2-4","2-5","3-6"}
    mb = mb[((mb['market'] == '1X2') | (mb['market'] == 'OU 2.5') |
             ((mb['market'] == 'TG Interval') & (mb['outcome'].astype(str).isin(tg_allowed))))].copy()
    if mb.empty:
        return pd.DataFrame()

    df_odds = fusion.attach_value_metrics(
        fusion._fill_odds_for_df(mb, league, with_odds=True),
        use_placeholders=False,
        league_code=league,
    )

    rows = []
    for (date, home, away), g in df_odds.groupby(['date','home','away']):
        p1 = _select_pick(g, '1X2', select)
        p2 = _select_pick(g, 'OU 2.5', select)
        p3 = _select_pick(g[(g['market']=='TG Interval')], 'TG Interval', select)
        rows.append({
            'Date': str(date),
            'Home': str(home),
            'Away': str(away),
            '1X2': _pick_str(p1),
            'OU 2.5': _pick_str(p2),
            'TG': _pick_str(p3),
        })

    out = pd.DataFrame(rows).sort_values(['Date','Home'])
    return out


def main():
    ap = argparse.ArgumentParser(description='Multi-league compact shopping-list report')
    ap.add_argument('--leagues', nargs='+', required=True, help='Leagues, e.g., E0 D1 F1 I1 SP1')
    ap.add_argument('--fixtures-csv', nargs='*', default=[], help='Pairs LEAGUE=path.csv')
    ap.add_argument('--select', choices=['prob','ev'], default='prob', help='Selection mode per market')
    ap.add_argument('--no-calibration', action='store_true', help='Disable calibrators (raw probabilities)')
    ap.add_argument('--export', action='store_true', help='Export per-league CSVs under reports/')
    args = ap.parse_args()

    # Build fixtures map
    fx_map: Dict[str, str] = {}
    for pair in args.fixtures_csv:
        if '=' in pair:
            lg, p = pair.split('=', 1)
            fx_map[lg.strip()] = p.strip()

    any_printed = False
    for lg in args.leagues:
        out = league_report(lg, fx_map.get(lg), select=args.select, use_calibration=(not args.no_calibration))
        print(f"\n{lg}")
        if out.empty:
            print('(no fixtures or markets)')
            continue
        any_printed = True
        try:
            from tabulate import tabulate
            print(tabulate(out, headers='keys', tablefmt='fancy_grid', showindex=False))
        except Exception:
            print(out.to_string(index=False))

        if args.export:
            Path('reports').mkdir(parents=True, exist_ok=True)
            ts = pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')
            path = Path('reports') / f'{lg}_{ts}_round_report.csv'
            out.to_csv(path, index=False)
            print(f'(exported -> {path})')

    if not any_printed:
        print('No output produced.')


if __name__ == '__main__':
    main()
