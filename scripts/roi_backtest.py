"""
ROI backtester using model probabilities and historical odds.

Sources:
 - Predictions: feature_store + bet_fusion (XGB μ, optional micro blend, NegBin/Poisson)
 - Odds (historical): football-data raw CSV (preferred), with fallback to placeholders

Per match, we compute 1X2 and OU 2.5 probabilities, attach odds from football-data,
compute EV for each outcome, and simulate staking (flat or Kelly).

Outputs:
 - CSV log with bet-by-bet results under reports/roi_*.csv
 - Summary printed: total bets, stake, profit, ROI, win rate

Usage:
  python -m scripts.roi_backtest --league E0 --start 2024-08-01 --end 2025-06-01 \
     --xg-source micro --dist negbin --k 4 --stake-policy flat --stake 1.0 --place both
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import feature_store
import bet_fusion as fusion


def _read_raw(league: str) -> pd.DataFrame:
    raw = Path('data')/'raw'/f'{league}_merged.csv'
    if not raw.exists():
        raise SystemExit(f'Raw file not found: {raw}')
    df = pd.read_csv(raw)
    # Parse dates (football-data usually dd/mm/YYYY)
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    except Exception:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


def _find_odds_fd(row: dict, raw_df: pd.DataFrame) -> dict:
    """Find historical odds for 1X2 and OU 2.5 in football-data for the given match row.
    row: {'date': Timestamp, 'home': str, 'away': str}
    Returns dict: {'1X2': {'H':odd,'D':odd,'A':odd}, 'OU': {'2.5': {'Over':odd,'Under':odd}}}
    Missing keys are filled later with placeholders.
    """
    out = {'1X2': {}, 'OU': {'2.5': {}}}
    d = row['date']; h = str(row['home']); a = str(row['away'])
    # filter by date and team names as in raw
    candidates = raw_df[(raw_df['Date'] == d) & (raw_df['HomeTeam'].astype(str) == h) & (raw_df['AwayTeam'].astype(str) == a)]
    if candidates.empty:
        return out
    r = candidates.iloc[0]
    # 1X2 priorities: Bet365, Pinnacle(SB), Max
    # Common columns: B365H,B365D,B365A / PSH,PSD,PSA / MaxH,MaxD,MaxA / WHH,WHD,WHA / BWH,BWD,BWA
    for triplet in [('B365H','B365D','B365A'), ('PSH','PSD','PSA'), ('MaxH','MaxD','MaxA'), ('WHH','WHD','WHA'), ('BWH','BWD','BWA')]:
        try:
            oh,od,oa = float(r.get(triplet[0])), float(r.get(triplet[1])), float(r.get(triplet[2]))
            if oh>0 and od>0 and oa>0:
                out['1X2'] = {'H': oh, 'D': od, 'A': oa}; break
        except Exception:
            pass
    # OU 2.5 priorities: Bet365, Pinnacle, Max
    for pair in [("B365>2.5","B365<2.5"), ("P>2.5","P<2.5"), ("Max>2.5","Max<2.5")]:
        try:
            o_over, o_under = float(r.get(pair[0])), float(r.get(pair[1]))
            if o_over>0 and o_under>0:
                out['OU']['2.5'] = {'Over': o_over, 'Under': o_under}; break
        except Exception:
            pass
    return out


def _select_bets(prob_1x2: Dict[str,float], prob_ou: Dict[str,float], odds: dict, place_both: bool, thresholds: Dict[str, Dict[str,float]]) -> List[dict]:
    """Return a list of bets to place for this match based on EV and thresholds.
    Each bet: {'market':str,'outcome':str,'prob':float,'odds':float,'ev':float}
    """
    bets: List[dict] = []
    # 1X2
    if prob_1x2:
        best_outcome = max(prob_1x2.items(), key=lambda kv: kv[1])[0]
        o = float((odds.get('1X2') or {}).get({'H':'home','D':'draw','A':'away'}[best_outcome], fusion._placeholder_odds('1X2', best_outcome)))
        p = float(prob_1x2[best_outcome])
        # implied normalized is not needed here; we use simple EV: p*O-1
        ev = p*o - 1.0
        th = thresholds.get('1X2', {'min_prob':0.55,'min_edge':0.03})
        # Edge approx: model prob - 1/O normalized to 3 outcomes => use p - (1/o)/sum? Simpler: accept EV>0 and p>=min_prob
        if (p >= th['min_prob']) and (ev > 0):
            bets.append({'market':'1X2','outcome':best_outcome,'prob':p,'odds':o,'ev':ev})
    # OU 2.5
    if prob_ou and ('Over' in prob_ou) and ('Under' in prob_ou):
        # pick the larger of Over/Under EV using available odds
        oline = odds.get('OU',{}).get('2.5',{})
        evs = {}
        for side in ('Over','Under'):
            o = float(oline.get(side, fusion._placeholder_odds('OU 2.5', side)))
            p = float(prob_ou[side])
            evs[side] = (p*o - 1.0, o, p)
        side = max(evs, key=lambda s: evs[s][0])
        ev, o, p = evs[side]
        th = thresholds.get('OU', {'min_prob':0.58,'min_edge':0.02})
        if (p >= th['min_prob']) and (ev > 0):
            bets.append({'market':'OU 2.5','outcome':side,'prob':p,'odds':o,'ev':ev})
    # If not placing both, keep only best EV
    if not place_both and bets:
        bets = [max(bets, key=lambda b: b['ev'])]
    return bets


def main() -> None:
    ap = argparse.ArgumentParser(description='ROI backtest using football-data odds (1X2, OU 2.5)')
    ap.add_argument('--league', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--xg-source', choices=['macro','micro','blend'], default='micro')
    ap.add_argument('--blend-weight', type=float, default=0.5)
    ap.add_argument('--dist', choices=['poisson','negbin'], default='negbin')
    ap.add_argument('--k', type=float, default=4.0)
    ap.add_argument('--stake-policy', choices=['flat','kelly'], default='flat')
    ap.add_argument('--stake', type=float, default=1.0, help='Flat stake per bet or initial bankroll for Kelly')
    ap.add_argument('--kelly-fraction', type=float, default=0.25)
    ap.add_argument('--place', choices=['best','both'], default='both')
    args = ap.parse_args()

    lg = args.league
    raw_df = _read_raw(lg)
    # Build matches list from raw
    d = raw_df[(raw_df['Date'] >= pd.to_datetime(args.start)) & (raw_df['Date'] <= pd.to_datetime(args.end))]
    d = d.dropna(subset=['HomeTeam','AwayTeam','Date']).sort_values('Date')
    matches = [{'date': r['Date'], 'home': str(r['HomeTeam']), 'away': str(r['AwayTeam']), 'FTHG': int(r.get('FTHG',-1)), 'FTAG': int(r.get('FTAG',-1))} for _, r in d.iterrows()]
    if not matches:
        print('No matches in window.')
        return

    # Staking state
    bankroll = float(args.stake) if args.stake_policy == 'kelly' else None
    k_frac = float(args.kelly_fraction)
    place_both = (args.place == 'both')
    thresholds = fusion._get_thresholds(fusion.load_config())

    # Logs
    log_rows: List[Dict] = []
    total_stake = 0.0
    total_profit = 0.0
    total_bets = 0
    wins = 0

    for m in matches:
        cutoff = (pd.to_datetime(m['date']) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        # Snapshot as-of day before
        snap = feature_store.build_snapshot(
            enhanced_csv=str(Path('data')/'enhanced'/f'{lg}_final_features.csv'),
            as_of=cutoff,
            half_life_matches=5,
            elo_k=20.0,
            elo_home_adv=60.0,
        )
        feat = fusion._feature_row_from_snapshot(snap, m['home'], m['away'])
        if feat is None:
            continue
        # Compute μ via XGB then blend with micro if configured
        import joblib, os
        from xgboost import XGBRegressor
        hj = os.path.join('advanced_models', f'{lg}_ultimate_xgb_home.json')
        aj = os.path.join('advanced_models', f'{lg}_ultimate_xgb_away.json')
        hp = os.path.join('advanced_models', f'{lg}_ultimate_xgb_home.pkl')
        ap = os.path.join('advanced_models', f'{lg}_ultimate_xgb_away.pkl')
        def _load(json_path, pkl_path):
            try:
                if os.path.exists(json_path):
                    m = XGBRegressor(); m.load_model(json_path); return m
            except Exception:
                pass
            return joblib.load(pkl_path)
        xH = _load(hj, hp); xA = _load(aj, ap)
        mu_h_macro = float(fusion._predict_xgb(xH, feat)); mu_a_macro = float(fusion._predict_xgb(xA, feat))
        mu_h, mu_a, _src = fusion._blend_mu({'xg_source': args.xg_source, 'xg_blend_weight': args.blend_weight}, m['home'], m['away'], mu_h_macro, mu_a_macro)
        # Score matrix
        if args.dist == 'negbin':
            P = fusion._score_matrix_negbin(mu_h, mu_a, k_h=float(args.k), k_a=float(args.k), max_goals=10)
        else:
            P = fusion._score_matrix(mu_h, mu_a, max_goals=10)
        markets = fusion._evaluate_all_markets(P, ou_lines=[2.5], intervals=[(0,3),(1,3),(2,4)])
        prob_1x2 = markets.get('1X2') or {}
        prob_ou = (markets.get('OU') or {}).get('2.5') or {}
        # Attach odds from football-data
        odds = _find_odds_fd(m, raw_df)
        # Select bets
        bets = _select_bets(prob_1x2, prob_ou, odds, place_both, thresholds)
        if not bets:
            continue
        # Simulate results
        res = 'H' if m['FTHG']>m['FTAG'] else ('A' if m['FTAG']>m['FTHG'] else 'D')
        tot = m['FTHG'] + m['FTAG']
        for b in bets:
            outcome_win = False
            if b['market'] == '1X2':
                outcome_win = (b['outcome'] == res)
            elif b['market'] == 'OU 2.5':
                if b['outcome'] == 'Over': outcome_win = (tot > 2)
                else: outcome_win = (tot <= 2)
            # stake sizing
            if args.stake_policy == 'kelly' and bankroll is not None:
                O = float(b['odds']); p = float(b['prob'])
                try:
                    f_star = (p*O - (1.0 - p)) / max(O - 1.0, 1e-9)
                except Exception:
                    f_star = 0.0
                f_star = max(0.0, min(1.0, f_star))
                stake = max(0.0, bankroll * k_frac * f_star)
            else:
                stake = float(args.stake)
            if stake <= 0:
                continue
            profit = (stake * (b['odds'] - 1.0)) if outcome_win else -stake
            total_bets += 1; total_stake += stake; total_profit += profit
            if outcome_win: wins += 1
            if args.stake_policy == 'kelly' and bankroll is not None:
                bankroll += profit
            log_rows.append({
                'date': str(m['date'].date()), 'home': m['home'], 'away': m['away'],
                'market': b['market'], 'outcome': b['outcome'], 'prob': b['prob'], 'odds': b['odds'], 'ev': b['ev'],
                'stake': stake, 'profit_loss': profit, 'result': 'win' if outcome_win else 'lose'
            })

    if total_stake <= 0:
        print('No bets placed under thresholds.')
        return
    roi = (total_profit / total_stake) * 100.0
    win_rate = (wins / total_bets * 100.0) if total_bets else 0.0
    print(f"Bets: {total_bets}  Stake: {total_stake:.2f}  Profit: {total_profit:.2f}  ROI: {roi:.2f}%  Win%: {win_rate:.2f}%")
    # write CSV log
    out_dir = Path('reports'); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"roi_{lg}_{args.start}_{args.end}_{args.xg_source}_{args.stake_policy}.csv"
    pd.DataFrame(log_rows).to_csv(out_csv, index=False)
    print(f"Saved ROI log -> {out_csv}")


if __name__ == '__main__':
    main()

