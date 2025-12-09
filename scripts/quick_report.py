"""Shopping-list quick report (per-market picks with fair vs book odds)."""

from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd

import bet_fusion as fusion


def _ev_display(row: pd.Series) -> str:
    try:
        if str(row.get('price_source')) != 'real':
            return ''
        return f"{float(row.get('EV', 0.0)):.2f}"
    except Exception:
        return ''


def run_report(league: str, xg_source: str, micro_path: str | None = None, top_n: int = 10) -> pd.DataFrame:
    cfg = fusion.load_config()
    cfg['league'] = league
    cfg['xg_source'] = xg_source
    if micro_path:
        cfg['micro_agg_path'] = micro_path
    mb = fusion.generate_market_book(cfg)
    if mb.empty:
        return mb
    # keep 1X2 and OU 2.5 only
    mb = mb[(mb['market'] == '1X2') | (mb['market'] == 'OU 2.5')].copy()
    df = fusion._fill_odds_for_df(mb, league, with_odds=True)
    df = fusion.attach_value_metrics(
        df,
        use_placeholders=False,
        league_code=league,
    )
    if df.empty:
        return df
    df['prob'] = pd.to_numeric(df['prob'], errors='coerce')
    # select one pick per market per match (highest prob then EV)
    rows = []
    for (d,h,a), g in df.groupby(['date','home','away']):
        for market in ('1X2','OU 2.5'):
            gm = g[g['market'] == market]
            if gm.empty:
                continue
            r = gm.sort_values(['prob','EV'], ascending=[False, False]).iloc[0]
            rows.append({
                'date': d, 'home': h, 'away': a,
                'market': market, 'pick': r['outcome'],
                'prob': float(r['prob']),
                'fair_odds': float(r.get('fair_odds', float('nan'))),
                'book_odds': float(r.get('book_odds', float('nan'))) if pd.notna(r.get('book_odds')) else None,
                'price_source': r.get('price_source',''),
                'EV_real': _ev_display(r),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(['date','home','market']).head(top_n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--league', default='E0')
    ap.add_argument('--xg-source', choices=['macro','micro','blend'], required=True)
    ap.add_argument('--micro-path', default='data/enhanced/micro_agg.csv')
    ap.add_argument('--top', type=int, default=10)
    args = ap.parse_args()

    df = run_report(args.league, args.xg_source, micro_path=args.micro_path, top_n=args.top)
    if df.empty:
        print('No data.')
        return
    try:
        from tabulate import tabulate
        disp = df.copy()
        disp['prob'] = disp['prob'].map(lambda x: f"{x:.1%}")
        disp['fair_odds'] = disp['fair_odds'].map(lambda x: f"{x:.2f}")
        disp['book_odds'] = disp['book_odds'].map(lambda x: f"{x:.2f}" if pd.notna(x) else '')
        print(tabulate(
            disp[['date','home','away','market','pick','prob','fair_odds','book_odds','price_source','EV_real']],
            headers='keys', tablefmt='github', showindex=False))
    except Exception:
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()

