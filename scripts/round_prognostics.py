"""
Round prognostics (shopping-list style).

Per match, surfaces the strongest 1X2, OU 2.5, and TG Interval pick with
probability, fair odds, and book odds (if real).

Usage:
  python -m scripts.round_prognostics --league D1 [--window-days 2]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def _load_market_df(league: str) -> pd.DataFrame:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.append(root)
    import bet_fusion as bf  # type: ignore

    cfg = bf.load_config()
    cfg['league'] = league
    df = bf.generate_market_book(cfg)
    if df is None or df.empty:
        return pd.DataFrame()
    df = bf._fill_odds_for_df(df, league, with_odds=True)
    df = bf.attach_value_metrics(df, use_placeholders=False, league_code=league)
    return df


def _best_1x2(sub: pd.DataFrame) -> Tuple[str, float, str]:
    def gp(o):
        r = sub[(sub['market'].astype(str) == '1X2') & (sub['outcome'].astype(str) == o)]
        return float(r['prob'].iloc[0]) if not r.empty else np.nan, r.iloc[0] if not r.empty else None
    vals = {'1': gp('H'), 'X': gp('D'), '2': gp('A')}
    best = max(vals.items(), key=lambda kv: (-1.0 if np.isnan(kv[1][0]) else kv[1][0]))
    prob, row = best[1]
    return best[0], prob, row


def _best_ou(sub: pd.DataFrame):
    def gp(line: float, side: str):
        r = sub[(sub['market'].astype(str) == f'OU {line}') & (sub['outcome'].astype(str) == side)]
        return float(r['prob'].iloc[0]) if not r.empty else np.nan, r.iloc[0] if not r.empty else None
    pov, row_ov = gp(2.5, 'Over')
    pun, row_un = gp(2.5, 'Under')
    if not np.isnan(pov) and not np.isnan(pun):
        if pov >= pun:
            return f"over 2.5g", pov, row_ov
        return f"under 2.5g", pun, row_un
    ous = sub[sub['market'].astype(str).str.startswith('OU ')]
    if ous.empty:
        return ('n/a', np.nan, None)
    row = ous.loc[ous['prob'].astype(float).idxmax()]
    try:
        line = str(row['market']).split(' ', 1)[1]
    except Exception:
        line = str(row['market']).replace('OU ', '')
    return (f"{str(row['outcome']).lower()} {line}g", float(row['prob']), row)


def _best_tg(sub: pd.DataFrame):
    tgs = sub[sub['market'].astype(str) == 'TG Interval']
    if tgs.empty:
        return ('n/a', np.nan, None)
    row = tgs.loc[tgs['prob'].astype(float).idxmax()]
    return (f"{row['outcome']}g", float(row['prob']), row)


def _fmt_pick(label: str, prob: float, row: pd.Series) -> str:
    prob_str = '' if np.isnan(prob) else f"{prob*100:.0f}%"
    fair = ''
    book = ''
    ev = ''
    src = ''
    if row is not None:
        try:
            fair = f"fair {float(row.get('fair_odds', 0.0)):.2f}"
        except Exception:
            fair = ''
        try:
            if pd.notna(row.get('book_odds')):
                book = f"book {float(row.get('book_odds')):.2f}"
        except Exception:
            book = ''
        src = str(row.get('price_source',''))
        if src == 'real':
            try:
                ev = f"EV {float(row.get('EV', 0.0)):.2f}"
            except Exception:
                ev = ''
    parts = [f"{label} {prob_str}"]
    if fair:
        parts.append(fair)
    if book:
        parts.append(f"{book} ({src})" if src else book)
    if ev:
        parts.append(ev)
    return " | ".join([p for p in parts if p])


def main() -> None:
    ap = argparse.ArgumentParser(description='Compact round prognostics (shopping-list view)')
    ap.add_argument('--league', required=True, help='League code, e.g., D1')
    ap.add_argument('--window-days', type=int, default=2, help='Days after earliest fixture to include as the round window')
    args = ap.parse_args()

    df = _load_market_df(args.league)
    if df is None or df.empty:
        print('No fixtures available.')
        return
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    min_date = df['date'].min()
    sub = df[(df['date'] >= min_date) & (df['date'] <= (min_date + pd.Timedelta(days=args.window_days)))].copy()

    rows = []
    for (d, h, a), g in sub.groupby(['date', 'home', 'away'], dropna=False):
        k1, p1, r1 = _best_1x2(g)
        ou, po, ro = _best_ou(g)
        tg, pt, rt = _best_tg(g)
        rows.append((d, str(h), str(a), k1, p1, ou, po, tg, pt, r1, ro, rt))
    rows.sort(key=lambda r: r[0])

    for i, (d, h, a, k1, p1, ou, po, tg, pt, r1, ro, rt) in enumerate(rows, start=1):
        date_str = pd.to_datetime(d).strftime('%Y-%m-%d %H:%M')
        print(
            f"{date_str} | game {i}) {h} vs {a} : "
            f"{_fmt_pick(f'1x2 => {k1}', p1, r1)} | "
            f"{_fmt_pick('over/under => ' + ou, po, ro)} | "
            f"{_fmt_pick('goal interval :' + tg, pt, rt)}"
        )


if __name__ == '__main__':
    main()
