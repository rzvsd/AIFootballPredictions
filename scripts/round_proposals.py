"""
Shopping-list per-fixture proposals (1X2 / OU 2.5 / TG) with target (fair) and book odds.
This replaces the old EV-style round_proposals. For richer UI, use print_best_per_match.
"""

from __future__ import annotations

import argparse
from typing import Set, Tuple

import pandas as pd

import bet_fusion as fusion

TG_ALLOWED = {"0-3", "1-3", "2-4", "2-5", "3-6"}


def _read_fixtures_filter(path: str) -> Set[Tuple[str, str, str]]:
    """Read fixtures CSV and return set of (date, home, away) keys."""
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
        print("Fixtures CSV must contain columns: date, home, away")
        return set()
    try:
        dd = pd.to_datetime(df[date_col], errors='coerce')
        df['_date'] = dd.dt.strftime('%Y-%m-%d')
    except Exception:
        df['_date'] = df[date_col].astype(str)
    df['_home'] = df[home_col].astype(str).str.strip()
    df['_away'] = df[away_col].astype(str).str.strip()
    return set(df[['_date','_home','_away']].itertuples(index=False, name=None))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--league', required=True, help='League code (e.g., E0, D1, F1, I1, SP1)')
    ap.add_argument('--days', type=int, default=7, help='Days ahead for fixtures')
    ap.add_argument('--fixtures-csv', default=None, help='Optional fixtures CSV (date,home,away) to filter a round')
    ap.add_argument('--top-n', type=int, default=50, help='Max rows to print')
    args = ap.parse_args()

    cfg = fusion.load_config()
    cfg['league'] = args.league
    cfg['fixtures_days'] = args.days
    league = args.league

    mb = fusion.generate_market_book(cfg)
    if mb.empty:
        print('No fixtures available.')
        return

    if args.fixtures_csv:
        wanted = _read_fixtures_filter(args.fixtures_csv)
        if wanted:
            def _key(row):
                try:
                    d = pd.to_datetime(row['date'], errors='coerce')
                    ds = d.strftime('%Y-%m-%d') if pd.notna(d) else str(row['date'])
                except Exception:
                    ds = str(row['date'])
                return (ds, str(row['home']).strip(), str(row['away']).strip())
            mb = mb[mb.apply(lambda r: _key(r) in wanted, axis=1)].copy()
            if mb.empty:
                print('No fixtures matched the provided CSV filter.')
                return

    mb = mb[((mb['market'] == '1X2') | (mb['market'] == 'OU 2.5') |
            ((mb['market'] == 'TG Interval') & (mb['outcome'].astype(str).isin(TG_ALLOWED))))].copy()
    if mb.empty:
        print('No eligible markets (1X2/OU 2.5/TG).')
        return

    df_odds = fusion._fill_odds_for_df(mb, league, with_odds=True)
    df_val = fusion.attach_value_metrics(df_odds, use_placeholders=True, league_code=league)

    rows = []
    for (date, home, away), g in df_val.groupby(['date','home','away']):
        row = {'Date': str(date), 'Home': str(home), 'Away': str(away)}

        def pick(market_name: str):
            sub = g[g['market'] == market_name].sort_values(['prob'], ascending=False)
            if sub.empty:
                return None
            r = sub.iloc[0]
            return {
                'outcome': r['outcome'],
                'prob': float(r['prob']),
                'target': r.get('fair_odds'),
                'book': r.get('book_odds'),
                'src': r.get('price_source',''),
            }

        one = pick('1X2')
        ou = pick('OU 2.5')
        tg = pick('TG Interval')
        if one:
            row.update({
                '1X2_pick': f"1X2 {one['outcome']}",
                '1X2_prob': f"{one['prob']*100:.1f}%",
                '1X2_target': f"{one['target']:.2f}+" if one['target'] else '-',
                '1X2_book': f"{one['book']:.2f}" if one['book'] else '-',
                '1X2_src': one['src'],
            })
        if ou:
            row.update({
                'OU_pick': f"OU 2.5 {ou['outcome']}",
                'OU_prob': f"{ou['prob']*100:.1f}%",
                'OU_target': f"{ou['target']:.2f}+" if ou['target'] else '-',
                'OU_book': f"{ou['book']:.2f}" if ou['book'] else '-',
                'OU_src': ou['src'],
            })
        if tg:
            row.update({
                'TG_pick': f"TG {tg['outcome']}",
                'TG_prob': f"{tg['prob']*100:.1f}%",
                'TG_target': f"{tg['target']:.2f}+" if tg['target'] else '-',
                'TG_book': f"{tg['book']:.2f}" if tg['book'] else '-',
                'TG_src': tg['src'],
            })
        rows.append(row)

    out = pd.DataFrame(rows).sort_values('Date').head(args.top_n)
    if out.empty:
        print("No picks found.")
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
