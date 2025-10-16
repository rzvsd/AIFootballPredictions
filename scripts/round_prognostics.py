"""
Round prognostics: compact per-match summary for the next round.

Example output:
  game 1) TeamA vs TeamB : 1x2 => 1 / 67% | over/under => under 2.5g 70% | goal interval : 1-3g / 78%

Usage:
  python -m scripts.round_prognostics --league D1 [--window-days 2]

Notes:
  - Uses the fusion engine to generate market probabilities.
  - "Next round" is approximated as the earliest fixture date in the book,
    plus a small window (`--window-days`, default 2).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def _load_market_df(league: str) -> pd.DataFrame:
    # Ensure project root import works
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.append(root)
    import bet_fusion as bf  # type: ignore

    cfg = bf.load_config()
    cfg['league'] = league
    df = bf.generate_market_book(cfg)
    return df


def _best_1x2(sub: pd.DataFrame) -> Tuple[str, float]:
    def gp(o):
        r = sub[(sub['market'].astype(str) == '1X2') & (sub['outcome'].astype(str) == o)]
        return float(r['prob'].iloc[0]) if not r.empty else np.nan
    vals = {'1': gp('H'), 'X': gp('D'), '2': gp('A')}
    key = max(vals, key=lambda k: (-1.0 if np.isnan(vals[k]) else vals[k]))
    return key, vals[key]


def _best_ou(sub: pd.DataFrame) -> Tuple[str, float]:
    # Prefer OU 2.5 first
    def gp(line: float, side: str) -> float:
        r = sub[(sub['market'].astype(str) == f'OU {line}') & (sub['outcome'].astype(str) == side)]
        return float(r['prob'].iloc[0]) if not r.empty else np.nan
    pov, pun = gp(2.5, 'Over'), gp(2.5, 'Under')
    if not np.isnan(pov) and not np.isnan(pun):
        return (f"over 2.5g", pov) if pov >= pun else (f"under 2.5g", pun)
    # Fallback to any OU line with highest probability
    ous = sub[sub['market'].astype(str).str.startswith('OU ')]
    if ous.empty:
        return ('n/a', np.nan)
    row = ous.loc[ous['prob'].astype(float).idxmax()]
    try:
        line = str(row['market']).split(' ', 1)[1]
    except Exception:
        line = str(row['market']).replace('OU ', '')
    return (f"{str(row['outcome']).lower()} {line}g", float(row['prob']))


def _best_tg(sub: pd.DataFrame) -> Tuple[str, float]:
    tgs = sub[sub['market'].astype(str) == 'TG Interval']
    if tgs.empty:
        return ('n/a', np.nan)
    row = tgs.loc[tgs['prob'].astype(float).idxmax()]
    return (f"{row['outcome']}g", float(row['prob']))


def _fmtp(x: float) -> str:
    return '' if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x*100:.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description='Compact round prognostics (1X2, OU 2.5, TG interval)')
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
        k1, p1 = _best_1x2(g)
        ou, po = _best_ou(g)
        tg, pt = _best_tg(g)
        rows.append((d, str(h), str(a), k1, p1, ou, po, tg, pt))
    rows.sort(key=lambda r: r[0])

    for i, (d, h, a, k1, p1, ou, po, tg, pt) in enumerate(rows, start=1):
        date_str = pd.to_datetime(d).strftime('%Y-%m-%d %H:%M')
        print(
            f"{date_str} | game {i}) {h} vs {a} : "
            f"1x2 => {k1} / {_fmtp(p1)} | "
            f"over/under => {ou} {_fmtp(po)} | "
            f"goal interval : {tg} / {_fmtp(pt)}"
        )


if __name__ == '__main__':
    main()

