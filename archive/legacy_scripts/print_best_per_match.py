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


def _fmt_percent(x: float) -> str:
    try:
        return f"{float(x)*100:5.1f}%"
    except Exception:
        return "  n/a"


def _fmt_float(x: float) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "n/a"


def _choose_best(sub: pd.DataFrame, market_filter) -> pd.Series | None:
    cand = sub[market_filter(sub)]
    if cand.empty:
        return None
    # Pick the highest probability row (shopping list, not EV)
    cand = cand.sort_values(["prob"], ascending=False)
    return cand.iloc[0]


def _render_line(label: str, row: pd.Series) -> str:
    prob = _fmt_percent(row.get("prob"))
    tgt = _fmt_float(row.get("fair_odds")) + "+"
    odds = row.get("odds")
    odds_s = _fmt_float(odds) if pd.notna(odds) else "n/a"
    outcome = str(row.get("outcome", "")).strip()
    return f"  {label:<8} {outcome:<6} | p={prob:<7} | target>={tgt:<6} | book={odds_s}"


def main() -> None:
    cfg = fusion.load_config()
    league = cfg.get("league", "E0")
    df = fusion.generate_market_book(cfg)
    if df is None or df.empty:
        print("No fixtures available.")
        return
    df_odds = fusion._fill_odds_for_df(df, league, with_odds=True)
    full = attach_value_metrics(df_odds, use_placeholders=True, league_code=league)
    for c in ("prob", "odds", "fair_odds"):
        if c in full.columns:
            full[c] = pd.to_numeric(full[c], errors="coerce")

    # Group by fixture and print the shopping list: 1X2, OU (pref 2.5), TG interval
    fixtures = []
    for (date, home, away), sub in full.groupby(["date", "home", "away"], dropna=False):
        best_1x2 = _choose_best(sub, lambda s: s["market"].astype(str).eq("1X2"))
        # Prefer OU 2.5 else any OU
        best_ou = _choose_best(sub, lambda s: s["market"].astype(str).eq("OU 2.5"))
        if best_ou is None:
            best_ou = _choose_best(sub, lambda s: s["market"].astype(str).str.startswith("OU "))
        best_tg = _choose_best(sub, lambda s: s["market"].astype(str).eq("TG Interval"))
        fixtures.append((str(date), str(home), str(away), best_1x2, best_ou, best_tg))

    fixtures.sort(key=lambda t: t[0])

    print("\nPer-match best picks (Target/Fair Odds)")
    print("---------------------------------------")
    for date, home, away, r1, r2, r3 in fixtures:
        print(f"* {home} vs {away}  ({date})")
        if r1 is not None:
            print(_render_line("1X2", r1))
        if r2 is not None:
            print(_render_line("OU", r2))
        if r3 is not None:
            print(_render_line("TG", r3))
        print()


if __name__ == "__main__":
    main()

