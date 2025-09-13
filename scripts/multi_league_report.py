"""
Generate a compact multi-league report (per fixture: best 1X2 and OU 2.5)
with probabilities, odds, edge and EV.

Examples:
  python -m scripts.multi_league_report --leagues E0 D1 F1 I1 \
    --fixtures-csv E0=data/fixtures/E0_weekly_fixtures.csv D1=data/fixtures/D1_manual.csv \
    --select prob --export

If fixtures are not provided for a league, built-in fallbacks are used
via bet_fusion (weekly/manual CSV discovery).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Set, Optional
import pandas as pd

import bet_fusion as fusion
import config as cfgmod


def _read_fixtures_filter(path: str) -> Set[Tuple[str, str, str]]:
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
        print("Fixtures CSV must contain columns: date, home/home_team, away/away_team")
        return set()
    try:
        dd = pd.to_datetime(df[date_col], errors='coerce')
        df['_date'] = dd.dt.strftime('%Y-%m-%d')
    except Exception:
        df['_date'] = df[date_col].astype(str)
    # Normalize team names to internal canonical names to match market book
    df['_home'] = df[home_col].astype(str).str.strip().apply(cfgmod.normalize_team_name)
    df['_away'] = df[away_col].astype(str).str.strip().apply(cfgmod.normalize_team_name)
    return set(df[['_date','_home','_away']].itertuples(index=False, name=None))


def _select_pick(group: pd.DataFrame, market_name: str, mode: str) -> Optional[pd.Series]:
    g = group[group['market'] == market_name]
    if g.empty:
        return None
    if mode == 'ev' and 'EV' in g.columns:
        return g.sort_values(['EV','prob'], ascending=[False, False]).iloc[0]
    return g.sort_values(['prob','EV'], ascending=[False, False]).iloc[0]


def league_report(league: str, fixtures_csv: Optional[str], select: str, use_calibration: bool) -> pd.DataFrame:
    cfg = fusion.load_config()
    cfg['league'] = league
    if fixtures_csv:
        cfg['fixtures_csv'] = fixtures_csv
    if not use_calibration:
        cfg['use_calibration'] = False
    mb = fusion.generate_market_book(cfg)
    if mb.empty:
        return pd.DataFrame()

    # Optional filter by fixtures CSV (single round)
    if fixtures_csv:
        wanted = _read_fixtures_filter(fixtures_csv)
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
                return pd.DataFrame()

    # Keep 1X2, OU 2.5, and selected TG intervals
    tg_allowed = {"0-3","1-3","2-4","2-5","3-6"}
    mb = mb[((mb['market'] == '1X2') | (mb['market'] == 'OU 2.5') |
             ((mb['market'] == 'TG Interval') & (mb['outcome'].astype(str).isin(tg_allowed))))].copy()
    if mb.empty:
        return pd.DataFrame()

    # Fill odds from API/local and attach metrics
    df_odds = fusion.attach_value_metrics(
        fusion._fill_odds_for_df(mb, league, with_odds=True),
        use_placeholders=False,
    )

    rows = []
    for (date, home, away), g in df_odds.groupby(['date','home','away']):
        p1 = _select_pick(g, '1X2', select)
        p2 = _select_pick(g, 'OU 2.5', select)
        row = {'Date': str(date), 'Home': str(home), 'Away': str(away)}
        if p1 is not None:
            row.update({
                'Pick_1X2': f"1X2 {p1['outcome']}",
                'Prob_1X2': f"{float(p1['prob']):.1%}",
                'Odds_1X2': f"{float(p1['odds']):.2f}",
                'Edge_1X2': f"{float(p1['edge']):.1%}",
                'EV_1X2': f"{float(p1['EV']):.2f}",
            })
            try:
                if 'delta_p_imp' in g.columns:
                    d = float(g[(g['market']=='1X2') & (g['outcome']==p1['outcome'])]['delta_p_imp'].iloc[0])
                    row['Delta_1X2'] = f"{d:+.2%}"
            except Exception:
                pass
        else:
            row.update({'Pick_1X2':'-','Prob_1X2':'-','Odds_1X2':'-','Edge_1X2':'-','EV_1X2':'-'})
        if p2 is not None:
            row.update({
                'Pick_OU': f"OU 2.5 {p2['outcome']}",
                'Prob_OU': f"{float(p2['prob']):.1%}",
                'Odds_OU': f"{float(p2['odds']):.2f}",
                'Edge_OU': f"{float(p2['edge']):.1%}",
                'EV_OU': f"{float(p2['EV']):.2f}",
            })
            try:
                if 'delta_p_imp' in g.columns:
                    d = float(g[(g['market']=='OU 2.5') & (g['outcome']==p2['outcome'])]['delta_p_imp'].iloc[0])
                    row['Delta_OU'] = f"{d:+.2%}"
            except Exception:
                pass
        else:
            row.update({'Pick_OU':'-','Prob_OU':'-','Odds_OU':'-','Edge_OU':'-','EV_OU':'-'})
        # TG Interval best
        p3 = _select_pick(g[(g['market']=='TG Interval')], 'TG Interval', select)
        if p3 is not None:
            row.update({
                'Pick_TG': f"TG {p3['outcome']}",
                'Prob_TG': f"{float(p3['prob']):.1%}",
                'Odds_TG': f"{float(p3['odds']):.2f}",
                'Edge_TG': f"{float(p3['edge']):.1%}",
                'EV_TG': f"{float(p3['EV']):.2f}",
            })
        else:
            row.update({'Pick_TG':'-','Prob_TG':'-','Odds_TG':'-','Edge_TG':'-','EV_TG':'-'})
        # Higher-probability market among the selected picks
        try:
            p1v = float(p1['prob']) if p1 is not None else float('nan')
            p2v = float(p2['prob']) if p2 is not None else float('nan')
            if pd.notna(p1v) and pd.notna(p2v):
                row['Higher'] = f"1X2 ({p1v:.1%})" if p1v >= p2v else f"OU ({p2v:.1%})"
            elif pd.notna(p1v):
                row['Higher'] = f"1X2 ({p1v:.1%})"
            elif pd.notna(p2v):
                row['Higher'] = f"OU ({p2v:.1%})"
            else:
                row['Higher'] = '-'
        except Exception:
            row['Higher'] = '-'
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(['Date','Home'])
    # Attach canonical IDs for export consumers (if present in market book)
    try:
        mb_cols = {c.lower(): c for c in df_odds.columns}
        def _id_lookup(d,h,a):
            g = df_odds[(df_odds['date'].astype(str)==str(d)) & (df_odds['home'].astype(str)==str(h)) & (df_odds['away'].astype(str)==str(a))]
            if not g.empty:
                return int(g.iloc[0].get('home_id')), int(g.iloc[0].get('away_id'))
            return None, None
        out['HomeID'] = [ _id_lookup(r['Date'], r['Home'], r['Away'])[0] for _, r in out.iterrows() ]
        out['AwayID'] = [ _id_lookup(r['Date'], r['Home'], r['Away'])[1] for _, r in out.iterrows() ]
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description='Multi-league compact report (1X2 and OU 2.5)')
    ap.add_argument('--leagues', nargs='+', required=True, help='Leagues, e.g., E0 D1 F1 I1 SP1')
    ap.add_argument('--fixtures-csv', nargs='*', default=[], help='Pairs LEAGUE=path.csv')
    ap.add_argument('--select', choices=['prob','ev'], default='prob', help='Selection mode per market')
    ap.add_argument('--no-calibration', action='store_true', help='Disable calibrators (raw probabilities)')
    ap.add_argument('--export', action='store_true', help='Export per-league CSVs under reports/')
    ap.add_argument('--bands', action='store_true', help='Show TG uncertainty band (80% shortest interval)')
    args = ap.parse_args()

    # Build fixtures map
    fx_map: Dict[str, str] = {}
    for pair in args.fixtures_csv:
        if '=' in pair:
            lg, p = pair.split('=', 1)
            fx_map[lg.strip()] = p.strip()

    any_printed = False
    for lg in args.leagues:
        out = league_report(lg, fx_map.get(lg), select=args.select, use_calibration=(not args.no_calibration))
        print(f"\n{lg}")
        if out.empty:
            print('(no fixtures or markets)')
            continue
        any_printed = True
        # Optional TG bands via fusion
        if args.bands:
            import bet_fusion as fusion
            cfg = fusion.load_config()
            cfg['league'] = lg
            tgci = fusion.compute_tg_ci(cfg, level=0.8)
            def _ci(row):
                return tgci.get((str(row['Date']), str(row['Home']), str(row['Away']))) or '-'
            out['CI_TG80'] = out.apply(_ci, axis=1)
        # Build a cleaner, compact display table
        def _pack(row, kind: str) -> str:
            pick = row.get(f'Pick_{kind}')
            if not pick or pick == '-':
                return '-'
            prob = row.get(f'Prob_{kind}', '-')
            odds = row.get(f'Odds_{kind}', '-')
            ev = row.get(f'EV_{kind}', '-')
            # Format: "Pick | Prob @ Odds | EV x.xx"
            return f"{pick} | {prob} @ {odds} | EV {ev}"

        rows = []
        for _, r in out.iterrows():
            rows.append({
                'Date': str(r.get('Date')),
                'Home': str(r.get('Home')),
                'Away': str(r.get('Away')),
                '1X2': _pack(r, '1X2'),
                'OU 2.5': _pack(r, 'OU'),
                'TG': _pack(r, 'TG'),
                'Higher': str(r.get('Higher', '-')),
                # Optional odds-movement deltas for selected picks
                'Δ1X2': str(r.get('Delta_1X2','-')),
                'ΔOU': str(r.get('Delta_OU','-')),
                'CI_TG80': str(r.get('CI_TG80','-')) if args.bands else '-',
            })
        disp_cols = ['Date','Home','Away','1X2','OU 2.5','TG','Higher','Δ1X2','ΔOU'] + (['CI_TG80'] if args.bands else [])
        try:
            from tabulate import tabulate
            # Prefer a clear box-drawn style; fall back if unsupported
            print(tabulate(rows, headers=disp_cols, tablefmt='fancy_grid', showindex=False))
        except Exception:
            import pandas as pd
            print(pd.DataFrame(rows, columns=disp_cols).to_string(index=False))

        # Legend under the table (short one-liners)
        try:
            print("\nLegend:")
            print("- EV: expected value at displayed odds.")
            print("- Higher: market with higher probability between 1X2 and OU 2.5.")
            print("- TG: total-goals interval (e.g., 0-3 means 0 up to 3 goals).")
        except Exception:
            pass

        if args.export:
            Path('reports').mkdir(parents=True, exist_ok=True)
            ts = pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')
            path = Path('reports') / f'{lg}_{ts}_round_report.csv'
            out.to_csv(path, index=False)
            print(f'(exported -> {path})')

    if not any_printed:
        print('No output produced.')


if __name__ == '__main__':
    main()

