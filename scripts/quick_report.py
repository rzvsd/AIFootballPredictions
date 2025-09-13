from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd

import bet_fusion as fusion


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
    # attach odds/edge/EV with placeholders (for comparability)
    df = fusion.attach_value_metrics(fusion._fill_odds_for_df(mb, league, with_odds=False), use_placeholders=True)
    # select one pick per market per match (highest EV then prob)
    rows = []
    for (d,h,a), g in df.groupby(['date','home','away']):
        # pick best 1X2
        g1 = g[g['market'] == '1X2']
        if not g1.empty:
            r = g1.sort_values(['EV','prob'], ascending=[False, False]).iloc[0]
            rows.append({'date': d, 'home': h, 'away': a, 'market': '1X2', 'pick': r['outcome'], 'prob': float(r['prob']), 'EV': float(r['EV'])})
        g2 = g[g['market'] == 'OU 2.5']
        if not g2.empty:
            r = g2.sort_values(['EV','prob'], ascending=[False, False]).iloc[0]
            rows.append({'date': d, 'home': h, 'away': a, 'market': 'OU 2.5', 'pick': r['outcome'], 'prob': float(r['prob']), 'EV': float(r['EV'])})
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
        disp['EV'] = disp['EV'].map(lambda x: f"{x:.2f}")
        print(tabulate(disp, headers='keys', tablefmt='github', showindex=False))
    except Exception:
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()

