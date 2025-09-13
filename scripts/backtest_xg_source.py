"""
Backtest xg_source strategies (macro, micro, blend) on a historical window using CRPS and LogLoss.

Usage:
  python -m scripts.backtest_xg_source --league E0 --start 2023-08-01 --end 2024-06-30 \
    --sources macro micro blend --blend-weight 0.5 --no-calibration

Outputs metrics per strategy (1X2 LogLoss/Brier/ECE, OU2.5 LogLoss/Brier/ECE, CRPS over total goals).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import feature_store
import bet_fusion as fusion


def read_enhanced(league: str) -> pd.DataFrame:
    enh = Path('data') / 'enhanced' / f'{league}_final_features.csv'
    if enh.exists():
        df = pd.read_csv(enh)
    else:
        df = pd.read_csv(Path('data') / 'processed' / f'{league}_merged_preprocessed.csv')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Ensure labels; if missing, try merge from raw
    if not set(['FTHG','FTAG']).issubset(df.columns) or df['FTHG'].isna().all():
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
    d = d.dropna(subset=['HomeTeam','AwayTeam','FTHG','FTAG']).sort_values('Date')
    out = []
    for _, r in d.iterrows():
        out.append({'date': r['Date'], 'home': str(r['HomeTeam']), 'away': str(r['AwayTeam']), 'tot': int(r['FTHG'])+int(r['FTAG']), 'res': 'H' if r['FTHG']>r['FTAG'] else ('A' if r['FTAG']>r['FTHG'] else 'D')})
    return out


def logloss_multiclass(p: Dict[str, float], y: str, eps: float = 1e-12) -> float:
    return -float(np.log(max(eps, float(p.get(y, 0.0)))))


def brier_multiclass(p: Dict[str, float], y: str, classes=('H','D','A')) -> float:
    s = 0.0
    for c in classes:
        yt = 1.0 if y == c else 0.0
        s += (float(p.get(c, 0.0)) - yt) ** 2
    return float(s)


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
    return float(ece)


def ece_multiclass(p_list: List[Dict[str,float]], y_list: List[str], classes=('H','D','A'), bins: int = 10) -> float:
    eces = []
    for c in classes:
        ps = [float(p.get(c, 0.0)) for p in p_list]
        ys = [1 if y == c else 0 for y in y_list]
        eces.append(ece_binary(ps, ys, bins=bins))
    return float(np.nanmean(eces))


def main() -> None:
    ap = argparse.ArgumentParser(description='Backtest xg_source: macro vs micro vs blend')
    ap.add_argument('--league', required=True)
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--sources', nargs='+', default=['macro','micro','blend'], choices=['macro','micro','blend'])
    ap.add_argument('--blend-weight', type=float, default=0.5)
    ap.add_argument('--no-calibration', action='store_true')
    ap.add_argument('--dist', choices=['poisson','negbin'], default='poisson')
    ap.add_argument('--k', type=float, default=6.0)
    args = ap.parse_args()

    lg = args.league
    df = read_enhanced(lg)
    rows = iter_matches(df, args.start, args.end)
    if not rows:
        print('No matches found for window.')
        raise SystemExit(1)

    # accumulators per source
    metrics = {src: {'ll1':[], 'br1':[], 'p1':[], 'y1':[], 'll2':[], 'br2':[], 'pov':[], 'yov':[], 'crps':[]} for src in args.sources}

    for m in rows:
        cutoff = (pd.to_datetime(m['date']) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        snap = feature_store.build_snapshot(
            enhanced_csv=str(Path('data')/'enhanced'/f'{lg}_final_features.csv'),
            as_of=cutoff,
            half_life_matches=5,
            elo_k=20.0,
            elo_home_adv=60.0,
        )
        if snap is None or len(snap) == 0 or ('team' not in snap.columns):
            snap = feature_store.build_snapshot(
                enhanced_csv=str(Path('data')/'enhanced'/f'{lg}_final_features.csv'),
                as_of=None,
                half_life_matches=5,
                elo_k=20.0,
                elo_home_adv=60.0,
            )
        feat = fusion._feature_row_from_snapshot(snap, m['home'], m['away'])
        if feat is None:
            continue
        # Load macro μ from XGB
        import joblib, os
        from xgboost import XGBRegressor
        hj = os.path.join('advanced_models', f'{lg}_ultimate_xgb_home.json')
        aj = os.path.join('advanced_models', f'{lg}_ultimate_xgb_away.json')
        hp = os.path.join('advanced_models', f'{lg}_ultimate_xgb_home.pkl')
        ap = os.path.join('advanced_models', f'{lg}_ultimate_xgb_away.pkl')
        def _load(json_path, pkl_path):
            try:
                if os.path.exists(json_path):
                    m = XGBRegressor()
                    m.load_model(json_path)
                    return m
            except Exception:
                pass
            return joblib.load(pkl_path)
        xH = _load(hj, hp)
        xA = _load(aj, ap)
        mu_h_macro = float(fusion._predict_xgb(xH, feat)); mu_a_macro = float(fusion._predict_xgb(xA, feat))

        for src in args.sources:
            cfg = {'league': lg, 'xg_source': src, 'xg_blend_weight': args.blend_weight}
            mu_h, mu_a, _ = fusion._blend_mu(cfg, str(m['home']), str(m['away']), mu_h_macro, mu_a_macro)
            # matrix & markets
            if args.dist == 'negbin':
                P = fusion._score_matrix_negbin(mu_h, mu_a, k_h=float(args.k), k_a=float(args.k), max_goals=10)
            else:
                P = fusion._score_matrix(mu_h, mu_a, max_goals=10)
            mk = fusion._evaluate_all_markets(P, ou_lines=[2.5], intervals=[(0,3),(1,3),(2,4)])
            p1 = mk['1X2']
            pou = mk['OU'].get('2.5') or {}
            # calibrators optional
            if not args.no_calibration:
                try:
                    from calibrators import load_calibrators, apply_calibration_1x2, calibrate_binary
                    arr = np.array([[p1['H'], p1['D'], p1['A']]], dtype=float)
                    cal1 = load_calibrators(Path('calibrators')/f'{lg}_1x2.pkl')
                    if cal1:
                        arr_c = apply_calibration_1x2(arr, cal1)[0]
                        p1 = {'H': float(arr_c[0]), 'D': float(arr_c[1]), 'A': float(arr_c[2])}
                    calou = load_calibrators(Path('calibrators')/f'{lg}_ou25.pkl')
                    if calou and 'ou25' in calou and pou:
                        model = calou['ou25']
                        from sklearn.isotonic import IsotonicRegression
                        from sklearn.linear_model import LogisticRegression
                        def _pred_b(model, p):
                            try:
                                if isinstance(model, IsotonicRegression):
                                    return float(np.clip(model.predict([p])[0], 1e-6, 1-1e-6))
                                if isinstance(model, LogisticRegression):
                                    return float(model.predict_proba([[p]])[0,1])
                            except Exception:
                                pass
                            return float(p)
                        if 'Over' in pou:
                            over = _pred_b(model, float(pou['Over']))
                            pou = {'Over': over, 'Under': 1.0 - over}
                except Exception:
                    pass
            # 1X2 metrics
            metrics[src]['ll1'].append(logloss_multiclass(p1, m['res']))
            metrics[src]['br1'].append(brier_multiclass(p1, m['res']))
            metrics[src]['p1'].append(p1)
            metrics[src]['y1'].append(m['res'])
            # OU 2.5 metrics
            if pou and ('Over' in pou) and ('Under' in pou):
                y_over = 1 if m['tot'] > 2.5 else 0
                p_over = float(pou['Over'])
                metrics[src]['ll2'].append(-(y_over*np.log(max(1e-12, p_over)) + (1-y_over)*np.log(max(1e-12, 1-p_over))))
                metrics[src]['br2'].append((p_over - y_over) ** 2 + ((1-p_over) - (1-y_over)) ** 2)
                metrics[src]['pov'].append(p_over)
                metrics[src]['yov'].append(y_over)
            # CRPS TG
            try:
                dist = fusion._goals_distribution(P)
                cdf = np.cumsum(dist)
                ytot = int(m['tot'])
                crps = 0.0
                for t in range(len(dist)):
                    F = cdf[t]
                    H = 1.0 if t >= ytot else 0.0
                    crps += (F - H) ** 2
                metrics[src]['crps'].append(float(crps))
            except Exception:
                pass

    # summarize
    def mean(x: List[float]) -> float:
        return float(np.nanmean(x)) if x else float('nan')
    def ece_mc(src: str) -> float:
        return ece_multiclass(metrics[src]['p1'], metrics[src]['y1']) if metrics[src]['p1'] else float('nan')
    def ece_bin(src: str) -> float:
        return ece_binary(metrics[src]['pov'], metrics[src]['yov']) if metrics[src]['pov'] else float('nan')

    print(f"League {lg} window=({args.start}..{args.end}) calibrated={not args.no_calibration} blend_w={args.blend_weight} dist={args.dist} k={args.k}")
    for src in args.sources:
        print(f"[{src}] 1X2: LogLoss={mean(metrics[src]['ll1']):.4f}  Brier={mean(metrics[src]['br1']):.4f}  ECE={ece_mc(src):.4f}  n={len(metrics[src]['ll1'])}")
        print(f"[{src}] OU2.5: LogLoss={mean(metrics[src]['ll2']):.4f}  Brier={mean(metrics[src]['br2']):.4f}  ECE={ece_bin(src):.4f}  n={len(metrics[src]['ll2'])}")
        print(f"[{src}] CRPS_TG mean={mean(metrics[src]['crps']):.4f}  n={len(metrics[src]['crps'])}")


if __name__ == '__main__':
    main()
