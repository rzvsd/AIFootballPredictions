"""
Print per-fixture proposals for 1X2 and OU 2.5 with optional full probabilities.
Defaults to value-focused picks; can also show full 1X2 (H/D/A) and OU (Over/Under) probabilities per match.

Usage examples:
  python scripts/round_proposals.py --league E0
  python scripts/round_proposals.py --league E0 --show-all-probs
  python scripts/round_proposals.py --league E0 --fixtures-csv data/fixtures/E0_round.csv --show-all-probs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Set

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

TG_ALLOWED = {"0-3", "1-3", "2-4", "2-5", "3-6"}


def _read_fixtures_filter(path: str) -> Set[Tuple[str, str, str]]:
    """Read a fixtures CSV and return a set of (date, home, away) to filter results.

    Accepts headers like date/home/away or Date/Home/Away. Dates are parsed to YYYY-MM-DD when possible.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read fixtures CSV '{path}': {e}")
        return set()
    # Normalize columns (accept both 'home' and 'home_team' variants)
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


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Per-match proposal table (1X2/OU2.5)')
    ap.add_argument('--league', default=None, help='Override league code (e.g., E0, D1, F1, SP1, I1)')
    ap.add_argument('--show-all-probs', action='store_true', help='Show full probabilities for 1X2 (H/D/A) and OU 2.5 (Over/Under)')
    ap.add_argument('--fixtures-csv', default=None, help='Optional fixtures CSV (date,home,away) to filter a single round')
    ap.add_argument('--select', choices=['prob','ev'], default='prob', help='Select pick by highest probability (prob) or highest EV (ev)')
    ap.add_argument('--no-calibration', action='store_true', help='Disable per-league calibrators (use raw Poisson probabilities)')
    args = ap.parse_args()

    cfg = fusion.load_config()
    if args.league:
        cfg['league'] = args.league
    if args.no_calibration:
        cfg['use_calibration'] = False
    league = cfg.get('league','E0')

    # Build market book
    mb = fusion.generate_market_book(cfg)
    if mb.empty:
        print('No fixtures available.')
        return
    # Optional: filter to a specific set of fixtures (single round)
    if args.fixtures_csv:
        wanted = _read_fixtures_filter(args.fixtures_csv)
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
                print('No fixtures matched the provided CSV filter.')
                return

    # Keep only 1X2, OU 2.5 and desired TG intervals
    mb = mb[((mb['market'] == '1X2') | (mb['market'] == 'OU 2.5') |
            ((mb['market'] == 'TG Interval') & (mb['outcome'].astype(str).isin(TG_ALLOWED))))].copy()
    if mb.empty:
        print('No eligible markets (1X2/OU 2.5).')
        return

    # Fill/attach odds and value metrics (placeholder odds)
    df_odds = fusion.attach_value_metrics(
        fusion._fill_odds_for_df(mb, league, with_odds=True),
        use_placeholders=False,
    )

    # Build both markets per match (1X2 and OU 2.5), no threshold filter to always show both
    recs = []
    for (date, home, away), g in df_odds.groupby(['date','home','away']):
        # 1X2 best selection
        g1 = g[g['market'] == '1X2']
        pick_1x2 = {'Pick_1X2': '-', 'Prob_1X2': '-', 'Odds_1X2': '-', 'Edge_1X2': '-', 'EV_1X2': '-'}
        if not g1.empty:
            if args.select == 'ev' and 'EV' in g1.columns:
                r1 = g1.sort_values(['EV','prob'], ascending=[False, False]).iloc[0]
            else:
                r1 = g1.sort_values(['prob','EV'], ascending=[False, False]).iloc[0]
            pick_1x2 = {
                'Pick_1X2': f"1X2 {r1['outcome']}",
                'Prob_1X2': f"{float(r1['prob']):.1%}",
                'Odds_1X2': f"{float(r1['odds']):.2f}",
                'Edge_1X2': f"{float(r1['edge']):.1%}",
                'EV_1X2': f"{float(r1['EV']):.2f}",
            }
        # OU 2.5 best selection
        g2 = g[g['market'] == 'OU 2.5']
        pick_ou = {'Pick_OU': '-', 'Prob_OU': '-', 'Odds_OU': '-', 'Edge_OU': '-', 'EV_OU': '-'}
        if not g2.empty:
            if args.select == 'ev' and 'EV' in g2.columns:
                r2 = g2.sort_values(['EV','prob'], ascending=[False, False]).iloc[0]
            else:
                r2 = g2.sort_values(['prob','EV'], ascending=[False, False]).iloc[0]
            pick_ou = {
                'Pick_OU': f"OU 2.5 {r2['outcome']}",
                'Prob_OU': f"{float(r2['prob']):.1%}",
                'Odds_OU': f"{float(r2['odds']):.2f}",
                'Edge_OU': f"{float(r2['edge']):.1%}",
                'EV_OU': f"{float(r2['EV']):.2f}",
            }
        # TG Interval best pick among allowed
        g3 = g[(g['market'] == 'TG Interval') & (g['outcome'].astype(str).isin(TG_ALLOWED))]
        pick_tg = {'Pick_TG': '-', 'Prob_TG': '-', 'Odds_TG': '-', 'Edge_TG': '-', 'EV_TG': '-'}
        if not g3.empty:
            if args.select == 'ev' and 'EV' in g3.columns:
                r3 = g3.sort_values(['EV','prob'], ascending=[False, False]).iloc[0]
            else:
                r3 = g3.sort_values(['prob','EV'], ascending=[False, False]).iloc[0]
            pick_tg = {
                'Pick_TG': f"TG {r3['outcome']}",
                'Prob_TG': f"{float(r3['prob']):.1%}",
                'Odds_TG': f"{float(r3['odds']):.2f}",
                'Edge_TG': f"{float(r3['edge']):.1%}",
                'EV_TG': f"{float(r3['EV']):.2f}",
            }

        row = {
            'Date': str(date), 'Home': str(home), 'Away': str(away),
            **pick_1x2, **pick_ou, **pick_tg,
        }
        if args.show_all_probs:
            # Collect full probabilities for 1X2
            if not g1.empty:
                try:
                    pH = float(g1.loc[g1['outcome']=='H','prob'].iloc[0]) if (g1['outcome']=='H').any() else None
                    pD = float(g1.loc[g1['outcome']=='D','prob'].iloc[0]) if (g1['outcome']=='D').any() else None
                    pA = float(g1.loc[g1['outcome']=='A','prob'].iloc[0]) if (g1['outcome']=='A').any() else None
                except Exception:
                    pH=pD=pA=None
                row.update({
                    'P(H)': f"{pH:.1%}" if pH is not None else '-',
                    'P(D)': f"{pD:.1%}" if pD is not None else '-',
                    'P(A)': f"{pA:.1%}" if pA is not None else '-',
                })
            else:
                row.update({'P(H)':'-','P(D)':'-','P(A)':'-'})
            # Collect full probabilities for OU 2.5
            if not g2.empty:
                try:
                    pOver = float(g2.loc[g2['outcome']=='Over','prob'].iloc[0]) if (g2['outcome']=='Over').any() else None
                    pUnder= float(g2.loc[g2['outcome']=='Under','prob'].iloc[0]) if (g2['outcome']=='Under').any() else None
                except Exception:
                    pOver=pUnder=None
                row.update({
                    'P(Over2.5)': f"{pOver:.1%}" if pOver is not None else '-',
                    'P(Under2.5)': f"{pUnder:.1%}" if pUnder is not None else '-',
                })
            else:
                row.update({'P(Over2.5)':'-','P(Under2.5)':'-'})
        recs.append(row)

    out = pd.DataFrame(recs).sort_values(['Date','Home'])
    # Compact view: select key columns for readability
    preferred_cols = [
        'Date','Home','Away',
        'Pick_1X2','Prob_1X2','Odds_1X2','Edge_1X2','EV_1X2',
        'Pick_OU','Prob_OU','Odds_OU','Edge_OU','EV_OU',
        'Pick_TG','Prob_TG','Odds_TG','Edge_TG','EV_TG'
    ]
    cols_exist = [c for c in preferred_cols if c in out.columns]
    try:
        from tabulate import tabulate
        print(tabulate(out[cols_exist], headers='keys', tablefmt='github', showindex=False))
    except Exception:
        # Plain print if tabulate missing
        print(out[cols_exist].to_string(index=False))


if __name__ == '__main__':
    main()
