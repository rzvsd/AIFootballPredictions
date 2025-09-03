"""
Prints one recommended pick per fixture (from 1X2 or OU 2.5)
using tuned thresholds in bot_config.yaml. Uses placeholder odds
for offline runs.

Usage:
  python scripts/round_proposals.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

import bet_fusion as fusion


def base_market(m: str) -> str:
    return 'OU' if m.startswith('OU ') else m


def get_thresholds(cfg: Dict) -> Dict[str, Dict[str, float]]:
    defaults = {
        '1X2': {'min_prob': 0.55, 'min_edge': 0.03},
        'OU': {'min_prob': 0.58, 'min_edge': 0.02},
    }
    th = cfg.get('thresholds', {}) or {}
    for k in ('1X2','OU'):
        if k in th and isinstance(th[k], dict):
            for kk in ('min_prob','min_edge'):
                if kk in th[k]:
                    defaults[k][kk] = float(th[k][kk])
    return defaults


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Per-match proposal table (1X2/OU2.5)')
    ap.add_argument('--league', default=None, help='Override league code (e.g., E0, D1, F1, SP1, I1)')
    args = ap.parse_args()

    cfg = fusion.load_config()
    if args.league:
        cfg['league'] = args.league
    league = cfg.get('league','E0')

    # Build market book
    mb = fusion.generate_market_book(cfg)
    if mb.empty:
        print('No fixtures available.')
        return
    # Keep only 1X2 and OU 2.5
    mb = mb[(mb['market'] == '1X2') | (mb['market'] == 'OU 2.5')].copy()
    if mb.empty:
        print('No eligible markets (1X2/OU 2.5).')
        return

    # Fill/attach odds and value metrics (placeholder odds)
    df_odds = fusion.attach_value_metrics(
        fusion._fill_odds_for_df(mb, league, with_odds=False),
        use_placeholders=False,
    )

    # Build both markets per match (1X2 and OU 2.5), no threshold filter to always show both
    recs = []
    for (date, home, away), g in df_odds.groupby(['date','home','away']):
        # 1X2 best by EV
        g1 = g[g['market'] == '1X2']
        pick_1x2 = {'Pick_1X2': '-', 'Prob_1X2': '-', 'Odds_1X2': '-', 'Edge_1X2': '-', 'EV_1X2': '-'}
        if not g1.empty:
            r1 = g1.sort_values(['EV','prob'], ascending=[False, False]).iloc[0]
            pick_1x2 = {
                'Pick_1X2': f"1X2 {r1['outcome']}",
                'Prob_1X2': f"{float(r1['prob']):.1%}",
                'Odds_1X2': f"{float(r1['odds']):.2f}",
                'Edge_1X2': f"{float(r1['edge']):.1%}",
                'EV_1X2': f"{float(r1['EV']):.2f}",
            }
        # OU 2.5 best by EV
        g2 = g[g['market'] == 'OU 2.5']
        pick_ou = {'Pick_OU': '-', 'Prob_OU': '-', 'Odds_OU': '-', 'Edge_OU': '-', 'EV_OU': '-'}
        if not g2.empty:
            r2 = g2.sort_values(['EV','prob'], ascending=[False, False]).iloc[0]
            pick_ou = {
                'Pick_OU': f"OU 2.5 {r2['outcome']}",
                'Prob_OU': f"{float(r2['prob']):.1%}",
                'Odds_OU': f"{float(r2['odds']):.2f}",
                'Edge_OU': f"{float(r2['edge']):.1%}",
                'EV_OU': f"{float(r2['EV']):.2f}",
            }
        recs.append({
            'Date': str(date), 'Home': str(home), 'Away': str(away),
            **pick_1x2, **pick_ou,
        })

    out = pd.DataFrame(recs).sort_values(['Date','Home'])
    try:
        from tabulate import tabulate
        print(tabulate(out, headers='keys', tablefmt='github', showindex=False))
    except Exception:
        # Plain print if tabulate missing
        print(out.to_string(index=False))


if __name__ == '__main__':
    main()
