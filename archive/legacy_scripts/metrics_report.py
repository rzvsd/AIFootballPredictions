"""
Metrics report for acceptance checks: LogLoss, Brier, and ECE.

Computes metrics for 1X2 and OU 2.5 over a historical window using the
current fusion engine (XGB + Poisson) and optional per-league calibration.

Usage examples:
  python -m scripts.metrics_report --league E0 --start 2023-08-01 --end 2024-06-30
  python -m scripts.metrics_report --league E0 --start 2022-08-01 --end 2023-05-31 --no-calibration

Outputs:
  - Prints metrics table to console, and optionally writes JSON via --out-json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import feature_store
import bet_fusion as fusion
import joblib
from xgboost import XGBRegressor


def read_enhanced(league: str) -> pd.DataFrame:
    enh = Path('data') / 'enhanced' / f'{league}_final_features.csv'
    if enh.exists():
        df = pd.read_csv(enh)
    else:
        proc = Path('data') / 'processed' / f'{league}_merged_preprocessed.csv'
        df = pd.read_csv(proc)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Ensure labels; if missing, merge from raw
    if not set(['FTHG','FTAG']).issubset(df.columns) or df.get('FTHG') is None or df['FTHG'].isna().all():
        raw = Path('data') / 'raw' / f'{league}_merged.csv'
        if raw.exists():
            raw_df = pd.read_csv(raw, parse_dates=['Date'], dayfirst=True)
            keys = [c for c in ['Date','HomeTeam','AwayTeam'] if c in df.columns and c in raw_df.columns]
            if keys:
                df = pd.merge(df, raw_df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']], on=keys, how='left', suffixes=('', '_raw'))
    return df


def iter_matches(df: pd.DataFrame, start: str | None, end: str | None) -> List[dict]:
    d = df.copy()
    if start:
        d = d[d['Date'] >= pd.to_datetime(start)]
    if end:
        d = d[d['Date'] <= pd.to_datetime(end)]
    d = d.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']).sort_values('Date')
    rows = []
    for _, r in d.iterrows():
        fthg = int(r['FTHG']); ftag = int(r['FTAG'])
        rows.append({
            'date': r['Date'],
            'home': str(r['HomeTeam']),
            'away': str(r['AwayTeam']),
            'res': 'H' if fthg > ftag else ('A' if fthg < ftag else 'D'),
            'tot': fthg + ftag,
        })
    return rows


def predict_probs(
    home: str,
    away: str,
    snap: pd.DataFrame,
    league: str,
    use_calibration: bool,
    max_goals: int = 10,
    prob_model: str = 'xgb',
    blend_weight: float = 0.5,
    dist: str = 'poisson',
    k: float = 6.0,
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    feat = fusion._feature_row_from_snapshot(snap, home, away)
    if feat is None:
        return {}, {}, None
    # Load models lazily via fusion's generate_market_book path is heavier; emulate small path
    import joblib, os
    hj = os.path.join('advanced_models', f'{league}_ultimate_xgb_home.json')
    aj = os.path.join('advanced_models', f'{league}_ultimate_xgb_away.json')
    hp = os.path.join('advanced_models', f'{league}_ultimate_xgb_home.pkl')
    ap = os.path.join('advanced_models', f'{league}_ultimate_xgb_away.pkl')
    def _load(json_path, pkl_path):
        try:
            if os.path.exists(json_path):
                m = XGBRegressor()
                m.load_model(json_path)
                return m
        except Exception:
            pass
        return joblib.load(pkl_path)
    xgb_home = _load(hj, hp)
    xgb_away = _load(aj, ap)
    # Choose goal means
    if prob_model == 'ngb':
        try:
            ngb_home = joblib.load(os.path.join('advanced_models', f'{league}_ngb_poisson_home.pkl'))
            ngb_away = joblib.load(os.path.join('advanced_models', f'{league}_ngb_poisson_away.pkl'))
            mu_h = float(max(0.05, ngb_home.predict(feat)[0]))
            mu_a = float(max(0.05, ngb_away.predict(feat)[0]))
        except Exception:
            mu_h = float(fusion._predict_xgb(xgb_home, feat))
            mu_a = float(fusion._predict_xgb(xgb_away, feat))
    elif prob_model == 'blend':
        try:
            ngb_home = joblib.load(os.path.join('advanced_models', f'{league}_ngb_poisson_home.pkl'))
            ngb_away = joblib.load(os.path.join('advanced_models', f'{league}_ngb_poisson_away.pkl'))
            mu_h_x = float(fusion._predict_xgb(xgb_home, feat)); mu_a_x = float(fusion._predict_xgb(xgb_away, feat))
            mu_h_n = float(max(0.05, ngb_home.predict(feat)[0])); mu_a_n = float(max(0.05, ngb_away.predict(feat)[0]))
            w = float(blend_weight)
            mu_h = (1.0 - w) * mu_h_x + w * mu_h_n
            mu_a = (1.0 - w) * mu_a_x + w * mu_a_n
        except Exception:
            mu_h = float(fusion._predict_xgb(xgb_home, feat))
            mu_a = float(fusion._predict_xgb(xgb_away, feat))
    else:
        mu_h = float(fusion._predict_xgb(xgb_home, feat))
        mu_a = float(fusion._predict_xgb(xgb_away, feat))
    # Score matrix
    if dist == 'negbin':
        P = fusion._score_matrix_negbin(mu_h, mu_a, k_h=float(k), k_a=float(k), max_goals=max_goals)
    else:
        P = fusion._score_matrix(mu_h, mu_a, max_goals=max_goals)
    mk = fusion._evaluate_all_markets(P, ou_lines=[2.5], intervals=[(0,3),(1,3),(2,4)])
    p1x2 = mk['1X2']
    pou = mk['OU'].get('2.5') or {}

    if use_calibration:
        try:
            from calibrators import load_calibrators, apply_calibration_1x2, calibrate_binary
            import numpy as np
            cal1 = load_calibrators(Path('calibrators')/f'{league}_1x2.pkl')
            if cal1:
                arr = np.array([[p1x2['H'], p1x2['D'], p1x2['A']]], dtype=float)
                arr_c = apply_calibration_1x2(arr, cal1)[0]
                p1x2 = {'H': float(arr_c[0]), 'D': float(arr_c[1]), 'A': float(arr_c[2])}
            calou = load_calibrators(Path('calibrators')/f'{league}_ou25.pkl')
            if calou and 'ou25' in calou and pou:
                model = calou['ou25']
                def _pred_b(model, p):
                    try:
                        from sklearn.isotonic import IsotonicRegression
                        from sklearn.linear_model import LogisticRegression
                        if isinstance(model, IsotonicRegression):
                            return float(np.clip(model.predict([p])[0], 1e-6, 1-1e-6))
                        if isinstance(model, LogisticRegression):
                            return float(model.predict_proba([[p]])[0,1])
                    except Exception:
                        pass
                    return float(p)
                over = _pred_b(model, float(pou['Over'])) if 'Over' in pou else None
                under = 1.0 - over if over is not None else None
                if over is not None and under is not None:
                    pou = {'Over': over, 'Under': under}
        except Exception:
            pass

    return p1x2, pou, P


def logloss_multiclass(p: Dict[str, float], y: str, eps: float = 1e-12) -> float:
    return -np.log(max(eps, float(p.get(y, 0.0))))


def brier_multiclass(p: Dict[str, float], y: str, classes=('H','D','A')) -> float:
    s = 0.0
    for c in classes:
        yt = 1.0 if y == c else 0.0
        s += (float(p.get(c, 0.0)) - yt) ** 2
    return s


def ece_binary(ps: List[float], ys: List[int], bins: int = 10) -> float:
    ps = np.asarray(ps, float)
    ys = np.asarray(ys, int)
    if len(ps) == 0:
        return float('nan')
    bin_ids = np.clip((ps * bins).astype(int), 0, bins-1)
    ece = 0.0
    n = len(ps)
    for b in range(bins):
        mask = (bin_ids == b)
        if not mask.any():
            continue
        conf = float(ps[mask].mean())
        acc = float(ys[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def ece_multiclass(p_list: List[Dict[str,float]], y_list: List[str], classes=('H','D','A'), bins: int = 10) -> float:
    # Average per-class ECE
    eces = []
    for c in classes:
        ps = [float(p.get(c, 0.0)) for p in p_list]
        ys = [1 if y == c else 0 for y in y_list]
        eces.append(ece_binary(ps, ys, bins=bins))
    return float(np.nanmean(eces))


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute LogLoss, Brier, and ECE for 1X2 and OU 2.5')
    ap.add_argument('--league', required=True)
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--no-calibration', action='store_true')
    ap.add_argument('--out-json', default=None)
    # Overrides for model/dist experiments
    ap.add_argument('--prob-model', choices=['xgb','ngb','blend'], default='xgb')
    ap.add_argument('--blend-weight', type=float, default=0.5)
    ap.add_argument('--dist', choices=['poisson','negbin'], default='poisson')
    ap.add_argument('--k', type=float, default=6.0, help='NegBin dispersion if dist=negbin')
    args = ap.parse_args()

    league = args.league
    df = read_enhanced(league)
    matches = iter_matches(df, args.start, args.end)
    if not matches:
        print('No matches found for the given window.')
        raise SystemExit(1)

    # Accumulators
    ll1, br1, p_list, y_list = [], [], [], []
    ll2, br2, pov, yov = [], [], [], []
    crps_vals = []

    for m in matches:
        cutoff = (pd.to_datetime(m['date']) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        snap = feature_store.build_snapshot(
            enhanced_csv=str(Path('data')/'enhanced'/f'{league}_final_features.csv'),
            as_of=cutoff,
            half_life_matches=5,
            elo_k=20.0,
            elo_home_adv=60.0,
        )
        # Fallback if snapshot is empty or lacks 'team'
        if snap is None or len(snap) == 0 or ('team' not in snap.columns):
            snap = feature_store.build_snapshot(
                enhanced_csv=str(Path('data')/'enhanced'/f'{league}_final_features.csv'),
                as_of=None,
                half_life_matches=5,
                elo_k=20.0,
                elo_home_adv=60.0,
            )
        p1x2, pou, P = predict_probs(
            m['home'], m['away'], snap, league,
            use_calibration=(not args.no_calibration),
            prob_model=args.prob_model, blend_weight=args.blend_weight,
            dist=args.dist, k=args.k,
        )
        if not p1x2:
            continue
        # 1X2
        ll1.append(logloss_multiclass(p1x2, m['res']))
        br1.append(brier_multiclass(p1x2, m['res']))
        p_list.append(p1x2)
        y_list.append(m['res'])
        # OU 2.5
        if pou and ('Over' in pou) and ('Under' in pou):
            y_over = 1 if m['tot'] > 2.5 else 0
            p_over = float(pou['Over'])
            # log loss
            ll2.append(-(y_over*np.log(max(1e-12, p_over)) + (1-y_over)*np.log(max(1e-12, 1-p_over))))
            # brier
            br2.append((p_over - y_over) ** 2 + ((1-p_over) - (1-y_over)) ** 2)
            pov.append(p_over)
            yov.append(y_over)
        # CRPS on total goals from full distribution
        try:
            dist = fusion._goals_distribution(P)
            cdf = np.cumsum(dist)
            ytot = int(m['tot'])
            crps = 0.0
            for t in range(len(dist)):
                F = cdf[t]
                H = 1.0 if t >= ytot else 0.0
                crps += (F - H) ** 2
            crps_vals.append(crps)
        except Exception:
            pass

    # Summaries
    def summarize(name: str, ll: List[float], br: List[float], ece: float | None, n: int) -> Dict[str, float]:
        return {
            'n': n,
            'logloss': float(np.nanmean(ll)) if ll else float('nan'),
            'brier': float(np.nanmean(br)) if br else float('nan'),
            'ece': float(ece) if ece is not None else float('nan'),
        }

    ece1 = ece_multiclass(p_list, y_list, classes=('H','D','A'), bins=10) if p_list else float('nan')
    ece2 = ece_binary(pov, yov, bins=10) if pov else float('nan')
    s1 = summarize('1X2', ll1, br1, ece1, len(p_list))
    s2 = summarize('OU25', ll2, br2, ece2, len(pov))

    print(f"League {league}  window=({args.start}..{args.end})  calibrated={not args.no_calibration}")
    print(f"1X2  n={s1['n']}  LogLoss={s1['logloss']:.4f}  Brier={s1['brier']:.4f}  ECE={s1['ece']:.4f}")
    print(f"OU2.5 n={s2['n']}  LogLoss={s2['logloss']:.4f}  Brier={s2['brier']:.4f}  ECE={s2['ece']:.4f}")
    if crps_vals:
        print(f"CRPS_TG mean={float(np.nanmean(crps_vals)):.4f} (n={len(crps_vals)})")

    if args.out_json:
        # Include CRPS for total goals to support k optimization
        crps_mean = float(np.nanmean(crps_vals)) if crps_vals else float('nan')
        crps_n = int(len(crps_vals)) if crps_vals else 0
        out = {
            'league': league,
            'start': args.start,
            'end': args.end,
            'calibrated': (not args.no_calibration),
            'oneX2': s1,
            'ou25': s2,
            'crps_tg_mean': crps_mean,
            'crps_tg_n': crps_n,
            'prob_model': args.prob_model,
            'dist': args.dist,
            'k': float(args.k),
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f"Saved metrics -> {args.out_json}")


if __name__ == '__main__':
    main()
