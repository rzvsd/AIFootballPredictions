"""
Backtest runner for the hybrid (DC + xG) engine.

Evaluates calibration (log loss) and profitability (ROI) for 1X2 and Over/Under
markets over a historical period using the enhanced data CSV as the fixture and
ground-truth source. Optionally performs a coarse grid search over fusion weight
and per-market thresholds.

Usage (examples):
  python backtest/runner.py --league E0 --start 2023-01-01 --end 2023-06-30
  python backtest/runner.py --league E0 --optimize --start 2022-08-01 --end 2023-05-31

Outputs:
  - Prints calibration metrics and ROI summary
  - When --optimize is used, proposes tuned fusion_weight_dc and per-market thresholds
  - Optionally writes updates back to bot_config.yaml (pass --write-config)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
import yaml

import config
import feature_store
import bet_fusion as fusion
import calibrators as calib


def read_enhanced(league: str) -> pd.DataFrame:
    enh = Path("data") / "enhanced" / f"{league}_final_features.csv"
    if enh.exists():
        df = pd.read_csv(enh)
    else:
        # Fallback to processed merged file for backtesting
        proc = Path("data") / "processed" / f"{league}_merged_preprocessed.csv"
        if not proc.exists():
            raise FileNotFoundError(f"Enhanced CSV not found: {enh} and processed CSV not found: {proc}")
        df = pd.read_csv(proc)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Ensure labels present
    if not set(['FTHG','FTAG']).issubset(df.columns):
        raw = Path("data") / "raw" / f"{league}_merged.csv"
        if raw.exists():
            raw_df = pd.read_csv(raw, parse_dates=['Date'], dayfirst=True)
            keys = [c for c in ['Date','HomeTeam','AwayTeam'] if c in df.columns and c in raw_df.columns]
            if keys:
                df = pd.merge(df, raw_df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']], on=keys, how='left', suffixes=('', '_raw'))
    return df


def iter_matches(df: pd.DataFrame, start: str | None, end: str | None) -> List[dict]:
    d = df.copy()
    if start:
        d = d[d["Date"] >= pd.to_datetime(start)]
    if end:
        d = d[d["Date"] <= pd.to_datetime(end)]
    d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])  # ensure labels
    d = d.sort_values("Date")
    rows = []
    for _, r in d.iterrows():
        date = r["Date"]
        home = str(r["HomeTeam"])  # already dataset-normalized names
        away = str(r["AwayTeam"])  # already dataset-normalized names
        fthg = int(r["FTHG"])
        ftag = int(r["FTAG"])
        res = 'H' if fthg > ftag else ('A' if fthg < ftag else 'D')
        tot = fthg + ftag
        # Odds proxies if present
        odds = {
            'AvgH': r.get('AvgH', np.nan), 'AvgD': r.get('AvgD', np.nan), 'AvgA': r.get('AvgA', np.nan),
            'Avg>2.5': r.get('Avg>2.5', np.nan), 'Avg<2.5': r.get('Avg<2.5', np.nan)
        }
        rows.append({
            'date': date,
            'home': home,
            'away': away,
            'result': res,
            'total_goals': tot,
            'odds': odds,
        })
    return rows


def as_of_before(date: pd.Timestamp) -> str:
    # Use the day before to avoid leakage from same-day row
    cutoff = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return cutoff


def _predict_one(home: str, away: str, snap: pd.DataFrame,
                 xgb_home, xgb_away, dc_model, w_dc: float, max_goals: int) -> Tuple[np.ndarray, Dict[str,float]]:
    feat = fusion._feature_row_from_snapshot(snap, home, away)
    if feat is None:
        return None, {}
    mu_h_xgb = float(xgb_home.predict(feat)[0])
    mu_a_xgb = float(xgb_away.predict(feat)[0])
    mu_h_dc = mu_a_dc = None
    if dc_model is not None and w_dc > 0:
        try:
            mu_h_dc = float(dc_model.predict(pd.DataFrame({'team':[home],'opponent':[away],'home':[1]})).values[0])
            mu_a_dc = float(dc_model.predict(pd.DataFrame({'team':[away],'opponent':[home],'home':[0]})).values[0])
        except Exception:
            mu_h_dc = mu_a_dc = None
    P = fusion._fused_matrix(mu_h_xgb, mu_a_xgb, mu_h_dc, mu_a_dc, w_dc=w_dc, max_goals=max_goals)
    mk = fusion._evaluate_all_markets(P, ou_lines=[2.5], intervals=[(0,3),(1,3),(2,4)])
    return P, mk  # include reduced markets for speed


def log_loss_1x2(p: Dict[str,float], true_res: str, eps: float = 1e-12) -> float:
    return -np.log(max(eps, float(p.get(true_res, 0.0))))


def log_loss_ou(pu: Dict[str,float], tot: int, line: float = 2.5, eps: float = 1e-12) -> float:
    true_out = 'Over' if tot > line else 'Under'
    return -np.log(max(eps, float(pu.get(true_out, 0.0))))


def roi_from_picks(picks: List[Tuple[bool, float]], stake: float = 1.0) -> float:
    # picks: list of (won, odds)
    pnl = 0.0
    for won, O in picks:
        pnl += (O - 1.0) * stake if won else -stake
    return pnl


def backtest(league: str, start: str | None, end: str | None,
             thresholds: Dict[str, Dict[str,float]],
             half_life: int = 5, elo_k: float = 20.0, elo_home_adv: float = 60.0,
             max_goals: int = 10) -> Dict[str, float]:
    df = read_enhanced(league)
    matches = iter_matches(df, start, end)
    # Models
    home_path = Path("advanced_models") / f"{league}_ultimate_xgb_home.pkl"
    away_path = Path("advanced_models") / f"{league}_ultimate_xgb_away.pkl"
    dc_path = Path("advanced_models") / f"{league}_dixon_coles_model.pkl"
    xgb_home = joblib.load(home_path)
    xgb_away = joblib.load(away_path)
    dc_model = None
    if dc_path.exists():
        try:
            dc_model = joblib.load(dc_path)
        except Exception:
            dc_model = None
    enh_csv = str(Path("data")/"enhanced"/f"{league}_final_features.csv")

    # Accumulators
    ll_1x2 = []
    ll_ou25 = []
    roi_picks_1x2: List[Tuple[bool,float]] = []
    roi_picks_ou: List[Tuple[bool,float]] = []

    # For calibration fitting
    pmat_1x2 = []
    ylab_1x2 = []
    pover = []
    y_over = []

    for m in matches:
        cutoff = as_of_before(m['date'])
        snap = feature_store.build_snapshot(enhanced_csv=enh_csv, as_of=cutoff,
                                            half_life_matches=half_life, elo_k=elo_k, elo_home_adv=elo_home_adv)
        P, mk = _predict_one(m['home'], m['away'], snap, xgb_home, xgb_away, dc_model=None, w_dc=0.0, max_goals=max_goals)
        if P is None:
            continue
        # Calibration
        p1x2 = mk['1X2']
        ll_1x2.append(log_loss_1x2(p1x2, m['result']))
        pmat_1x2.append([p1x2['H'], p1x2['D'], p1x2['A']])
        ylab_1x2.append(m['result'])
        pou = mk['OU'].get('2.5') or mk['OU'].get('2.5'.replace(',', '.'))
        if pou:
            ll_ou25.append(log_loss_ou(pou, m['total_goals'], 2.5))
            pover.append(pou['Over'])
            y_over.append(1 if m['total_goals'] > 2.5 else 0)

        # ROI selection
        # 1X2
        th1 = thresholds.get('1X2', {'min_prob':0.55, 'min_edge':0.03})
        if all(k in m['odds'] for k in ('AvgH','AvgD','AvgA')) and pou:
            # normalize implied
            try:
                O_H = float(m['odds']['AvgH']); O_D = float(m['odds']['AvgD']); O_A = float(m['odds']['AvgA'])
                p_imp = np.array([1.0/O_H, 1.0/O_D, 1.0/O_A], dtype=float)
                p_imp /= p_imp.sum()
                for lbl, pi, O in zip(['H','D','A'], p_imp, [O_H, O_D, O_A]):
                    p_mod = float(p1x2[lbl])
                    if (p_mod >= th1['min_prob']) and ((p_mod - float(pi)) >= th1['min_edge']):
                        won = (lbl == m['result'])
                        roi_picks_1x2.append((won, O))
            except Exception:
                pass
        # OU 2.5
        thou = thresholds.get('OU', {'min_prob':0.58, 'min_edge':0.02})
        if pou and (not np.isnan(m['odds'].get('Avg>2.5', np.nan))) and (not np.isnan(m['odds'].get('Avg<2.5', np.nan))):
            try:
                O_over = float(m['odds']['Avg>2.5']); O_under = float(m['odds']['Avg<2.5'])
                p_imp = np.array([1.0/O_over, 1.0/O_under], dtype=float); p_imp /= p_imp.sum()
                for lbl, pi, O in zip(['Over','Under'], p_imp, [O_over, O_under]):
                    p_mod = float(pou[lbl])
                    if (p_mod >= thou['min_prob']) and ((p_mod - float(pi)) >= thou['min_edge']):
                        won = ((m['total_goals'] > 2.5) if lbl == 'Over' else (m['total_goals'] <= 2.5))
                        roi_picks_ou.append((won, O))
            except Exception:
                pass

    res = {
        'n_matches': len(ll_1x2),
        'logloss_1x2': float(np.mean(ll_1x2)) if ll_1x2 else np.nan,
        'logloss_ou25': float(np.mean(ll_ou25)) if ll_ou25 else np.nan,
        'picks_1x2': len(roi_picks_1x2),
        'roi_1x2': roi_from_picks(roi_picks_1x2) if roi_picks_1x2 else 0.0,
        'picks_ou': len(roi_picks_ou),
        'roi_ou': roi_from_picks(roi_picks_ou) if roi_picks_ou else 0.0,
    }
    # Attach calibration payload for optional fitting
    res['_calib_payload'] = {
        'p_1x2': np.array(pmat_1x2) if pmat_1x2 else None,
        'y_1x2': np.array(ylab_1x2) if ylab_1x2 else None,
        'p_over': np.array(pover) if pover else None,
        'y_over': np.array(y_over) if y_over else None,
    }
    return res


def optimize(league: str, start: str | None, end: str | None) -> Dict[str, any]:
    # coarse grid
    w_grid = [0.0, 0.2, 0.3, 0.4, 0.5]
    prob_grid_1x2 = [0.52, 0.54, 0.56, 0.58]
    edge_grid_1x2 = [0.01, 0.02, 0.03, 0.04]
    prob_grid_ou = [0.56, 0.58, 0.60, 0.62]
    edge_grid_ou = [0.01, 0.02, 0.03]

    best_w = 0.0
    best_ll = np.inf
    best_elo = (20.0, 60.0)
    for k in [15.0, 20.0, 25.0]:
        for ha in [40.0, 60.0, 80.0]:
            m = backtest(league, start, end, thresholds={'1X2': {'min_prob':0.0,'min_edge':0.0}, 'OU': {'min_prob':0.0,'min_edge':0.0}},
                         half_life=5, elo_k=k, elo_home_adv=ha)
            ll = (m.get('logloss_1x2', np.inf) + m.get('logloss_ou25', np.inf))
            if ll < best_ll:
                best_ll = ll; best_elo = (k, ha)

    # Search thresholds for ROI given best_w
    best_th = None
    best_roi = -np.inf
    for p1 in prob_grid_1x2:
        for e1 in edge_grid_1x2:
            for p2 in prob_grid_ou:
                for e2 in edge_grid_ou:
                    th = {'1X2': {'min_prob':p1,'min_edge':e1}, 'OU': {'min_prob':p2,'min_edge':e2}}
                    m = backtest(league, start, end, thresholds=th, half_life=5, elo_k=best_elo[0], elo_home_adv=best_elo[1])
                    roi = m.get('roi_1x2', 0.0) + m.get('roi_ou', 0.0)
                    # require some picks to avoid overfitting
                    if (m.get('picks_1x2',0)+m.get('picks_ou',0)) < 50:
                        continue
                    if roi > best_roi:
                        best_roi = roi; best_th = (th, m)

    return {'w_dc': best_w, 'elo': {'k': best_elo[0], 'home_adv': best_elo[1]}, 'thresholds': best_th[0] if best_th else None, 'metrics': best_th[1] if best_th else None}


def write_config_updates(upd: Dict[str, any], league: str, path: Path) -> None:
    cfg = {}
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    if upd.get('w_dc') is not None:
        cfg['fusion_weight_dc'] = float(upd['w_dc'])
    if upd.get('thresholds'):
        th = cfg.get('thresholds', {}) or {}
        th.update({k: {kk: float(vv) for kk, vv in v.items()} for k, v in upd['thresholds'].items()})
        cfg['thresholds'] = th
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description='Backtest runner for DC+xG hybrid engine')
    ap.add_argument('--league', default=config.LEAGUE_CODE)
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--optimize', action='store_true')
    ap.add_argument('--fit-calibrators', action='store_true', help='Fit and save per-league calibrators for 1X2 and OU2.5')
    ap.add_argument('--write-config', action='store_true', help='Write tuned params to bot_config.yaml')
    args = ap.parse_args()

    print(f"[backtest] league={args.league} range={args.start}..{args.end}")
    if args.optimize:
        res = optimize(args.league, args.start, args.end)
        print(f"Tuned elo={res.get('elo')} thresholds={res.get('thresholds')}")
        if res.get('metrics'):
            print(f"Metrics at tuned thresholds: {res['metrics']}")
        if args.write_config:
            write_config_updates(res, args.league, Path('bot_config.yaml'))
            print("Updated bot_config.yaml with tuned parameters.")
    else:
        # Run a single pass with current config
        # Load thresholds and elo params from config
        cfg = {}
        if Path('bot_config.yaml').exists():
            cfg = yaml.safe_load(open('bot_config.yaml','r',encoding='utf-8')) or {}
        th = cfg.get('thresholds', {'1X2': {'min_prob':0.55,'min_edge':0.03}, 'OU': {'min_prob':0.58,'min_edge':0.02}})
        m = backtest(args.league, args.start, args.end, thresholds=th,
                     half_life=int(cfg.get('half_life_matches', 5)),
                     elo_k=float(cfg.get('elo_k', 20.0)), elo_home_adv=float(cfg.get('elo_home_adv', 60.0)))
        print(f"Metrics: {m}")
        if args.fit_calibrators:
            payload = m.get('_calib_payload', {})
            p1x2 = payload.get('p_1x2'); y1x2 = payload.get('y_1x2')
            p_over = payload.get('p_over'); y_over = payload.get('y_over')
            out_dir = Path('calibrators'); out_dir.mkdir(exist_ok=True)
            if p1x2 is not None and y1x2 is not None and len(p1x2)>50:
                cal1 = calib.multi_calibrate_1x2(p1x2, y1x2, method='isotonic')
                calib.save_calibrators(str(out_dir / f"{args.league}_1x2.pkl"), cal1)
                print("Saved 1X2 calibrators.")
            if p_over is not None and y_over is not None and len(p_over)>50:
                calou = calib.calibrate_binary(p_over, y_over, method='isotonic')
                calib.save_calibrators(str(out_dir / f"{args.league}_ou25.pkl"), {'ou25': calou})
                print("Saved OU 2.5 calibrator.")


if __name__ == '__main__':
    main()
