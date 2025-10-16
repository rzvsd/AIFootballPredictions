from __future__ import annotations

import os
import sys
from typing import List, Tuple

import pandas as pd

# Ensure project root in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import bet_fusion as fusion  # type: ignore
from bet_fusion import attach_value_metrics  # type: ignore


def _fmt_percent(x) -> str:
    try:
        return f"{float(x)*100:5.2f}%"
    except Exception:
        return "   n/a"


def _fmt_float(x) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "n/a"


def _best_1x2(sub: pd.DataFrame, by: str = 'ev') -> pd.Series | None:
    s = sub[sub['market'].astype(str) == '1X2']
    if s.empty:
        return None
    key = 'EV' if by == 'ev' else 'prob'
    return s.loc[s[key].idxmax()]


def _best_ou(sub: pd.DataFrame, by: str = 'ev') -> pd.Series | None:
    # Prefer exactly 'OU 2.5' if present, else any OU line by EV
    ou25 = sub[sub['market'].astype(str).eq('OU 2.5')]
    if not ou25.empty:
        key = 'EV' if by == 'ev' else 'prob'
        return ou25.loc[ou25[key].idxmax()]
    ou_any = sub[sub['market'].astype(str).str.startswith('OU ')]
    if ou_any.empty:
        return None
    key = 'EV' if by == 'ev' else 'prob'
    return ou_any.loc[ou_any[key].idxmax()]


def _best_tg(sub: pd.DataFrame, by: str = 'ev') -> pd.Series | None:
    tg = sub[sub['market'].astype(str) == 'TG Interval']
    if tg.empty:
        return None
    key = 'EV' if by == 'ev' else 'prob'
    return tg.loc[tg[key].idxmax()]


def _print_rows(rows: List[pd.Series]) -> None:
    # header
    cols = ['date','home','away','market','outcome','prob','odds','edge','EV']
    widths = {'date':19,'home':16,'away':16,'market':12,'outcome':8,'prob':7,'odds':6,'edge':7,'EV':6}
    if rows:
        print("\nCombined Picks (TG, OU, 1X2 per game)")
        print("-" * 44)
        print(" ".join([f"{c.upper():<{widths[c]}}" for c in cols]))
    for r in rows:
        if r is None:
            continue
        vals = []
        for c in cols:
            v = r.get(c)
            s = str(v)
            if c in ('prob','edge'):
                s = _fmt_percent(v)
            elif c in ('odds','EV'):
                s = _fmt_float(v)
            vals.append(f"{s:<{widths[c]}}")
        print(" ".join(vals))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description='Combined best-per-match picks')
    ap.add_argument('--by', choices=['ev','prob'], default='ev', help='Select best by expected value or by probability')
    args = ap.parse_args()

    cfg = fusion.load_config()
    df = fusion.generate_market_book(cfg)
    if df is None or df.empty:
        print("No fixtures available.")
        return
    full = attach_value_metrics(df, use_placeholders=True)
    # ensure numeric for ranking
    for c in ('EV','prob','odds','edge'):
        if c in full.columns:
            full[c] = pd.to_numeric(full[c], errors='coerce')
    # group by fixture and pick 3 rows: TG, OU, 1X2
    out_rows: List[pd.Series] = []
    for (date, home, away), sub in full.groupby(['date','home','away'], dropna=False):
        sub_sorted = sub.sort_values(['EV','prob'], ascending=[False, False]).copy()
        t_tg = _best_tg(sub_sorted, by=args.by)
        t_ou = _best_ou(sub_sorted, by=args.by)
        t_1x2 = _best_1x2(sub_sorted, by=args.by)
        # append in fixed order: TG, OU, 1X2
        for t in (t_tg, t_ou, t_1x2):
            if t is not None:
                out_rows.append(t)
    # sort by date for display
    out_rows.sort(key=lambda s: str(s.get('date')))
    _print_rows(out_rows)


if __name__ == '__main__':
    main()

