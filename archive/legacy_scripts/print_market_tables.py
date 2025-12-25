"""Shopping-list market tables (top picks with fair vs book odds)."""

from __future__ import annotations

from typing import List

import pandas as pd

import bet_fusion as fusion
from bet_fusion import attach_value_metrics


def _fmt_percent(x: float) -> str:
    try:
        return f"{float(x)*100:5.2f}%"
    except Exception:
        return "   n/a"


def _fmt_float(x) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "n/a"


def _fmt_ev(row) -> str:
    try:
        if str(row.get('price_source')) != 'real':
            return ''
        return f"{float(row.get('EV', 0.0)):.2f}"
    except Exception:
        return ''


def _print_table(title: str, rows: pd.DataFrame, cols: List[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    widths = {
        'date': 19, 'home': 16, 'away': 16, 'market': 12, 'outcome': 8,
        'prob': 7, 'fair_odds': 9, 'book_odds': 9, 'price_source': 10, 'EV': 6,
    }
    header = " ".join([f"{c.upper():<{widths.get(c, len(c))}}" for c in cols])
    print(header)
    for _, r in rows.iterrows():
        vals = []
        for c in cols:
            v = r.get(c)
            if c == "prob":
                s = _fmt_percent(v)
            elif c in ("fair_odds","book_odds"):
                s = _fmt_float(v)
            elif c == "EV":
                s = _fmt_ev(r)
            else:
                s = str(v)
            vals.append(f"{s:<{widths.get(c, len(c))}}")
        print(" ".join(vals))


def main() -> None:
    cfg = fusion.load_config()
    league = cfg.get('league', 'E0')
    df = fusion.generate_market_book(cfg)
    if df is None or df.empty:
        print("No fixtures available.")
        return
    full = attach_value_metrics(
        fusion._fill_odds_for_df(df, league, with_odds=True),
        use_placeholders=False,
        league_code=league,
    )

    N = 8
    base_cols = ['date','home','away','market','outcome','prob','fair_odds','book_odds','price_source','EV']

    def _best_per_fixture(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        d = df.copy()
        for c in ('EV','prob','odds','edge','book_odds','fair_odds'):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors='coerce')
        grp = d.groupby(['date','home','away'], dropna=False)
        idx = grp['prob'].idxmax()
        return d.loc[idx].copy()

    # 1X2 (pick single best outcome per match)
    one = full[full['market']=='1X2'].copy()
    if not one.empty:
        one = _best_per_fixture(one)
        one = one.sort_values(['prob','EV'], ascending=[False, False]).head(N)
        _print_table("Top 1X2 Picks", one[base_cols], base_cols)

    # Over/Under (prefer 2.5)
    ou = full[full['market'].astype(str).eq('OU 2.5')].copy()
    if ou.empty:
        ou = full[full['market'].astype(str).str.startswith('OU ')].copy()
    if not ou.empty:
        ou = _best_per_fixture(ou)
        ou = ou.sort_values(['prob','EV'], ascending=[False, False]).head(N)
        _print_table("Top Over/Under Picks", ou[base_cols], base_cols)

    # Total Goals Intervals
    tg = full[full['market']=='TG Interval'].copy()
    if not tg.empty:
        tg = _best_per_fixture(tg)
        tg = tg.sort_values(['prob','EV'], ascending=[False, False]).head(N)
        _print_table("Top Total Goals Interval Picks", tg[base_cols], base_cols)


if __name__ == '__main__':
    main()
