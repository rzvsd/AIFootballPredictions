import argparse
import os
import pandas as pd
import numpy as np
import joblib

import config
import feature_store
import bet_fusion as fusion
from calibrators import load_calibrators, apply_calibration_1x2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--league', default='SP1')
    ap.add_argument('--fixtures', default=os.path.join('data','fixtures','SP1_manual.csv'))
    args = ap.parse_args()

    league = args.league
    enh_path = os.path.join('data','enhanced', f'{league}_final_features.csv')
    snap = feature_store.build_snapshot(enhanced_csv=enh_path, as_of=None,
                                        half_life_matches=5, elo_k=15.0, elo_home_adv=40.0)

    home_path = os.path.join('advanced_models', f'{league}_ultimate_xgb_home.pkl')
    away_path = os.path.join('advanced_models', f'{league}_ultimate_xgb_away.pkl')
    xgb_home = joblib.load(home_path)
    xgb_away = joblib.load(away_path)

    df = pd.read_csv(args.fixtures)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get('date')
    home_col = cols.get('home') or cols.get('home_team')
    away_col = cols.get('away') or cols.get('away_team')

    # optional calibration
    cal_path = os.path.join('calibrators', f'{league}_1x2.pkl')
    cal = load_calibrators(cal_path) if os.path.exists(cal_path) else None

    rows = []
    for _, r in df.iterrows():
        home_api = str(r[home_col]).strip()
        away_api = str(r[away_col]).strip()
        home = config.normalize_team_name(home_api)
        away = config.normalize_team_name(away_api)
        feat = fusion._feature_row_from_snapshot(snap, home, away)
        if feat is None:
            rows.append({'home': home, 'away': away, 'mu_h': None, 'mu_a': None})
            continue
        mu_h = float(xgb_home.predict(feat)[0])
        mu_a = float(xgb_away.predict(feat)[0])
        # compute 1X2 from Poisson
        P = fusion._score_matrix(mu_h, mu_a, max_goals=10, trim_epsilon=0.0)
        x = fusion._oneXtwo_from_matrix(P)
        p = np.array([[x['p_H'], x['p_D'], x['p_A']]])
        if cal:
            p_cal = apply_calibration_1x2(p, cal)[0]
            rows.append({'home': home, 'away': away, 'mu_h': mu_h, 'mu_a': mu_a,
                         'pH': x['p_H'], 'pD': x['p_D'], 'pA': x['p_A'],
                         'cH': p_cal[0], 'cD': p_cal[1], 'cA': p_cal[2]})
        else:
            rows.append({'home': home, 'away': away, 'mu_h': mu_h, 'mu_a': mu_a,
                         'pH': x['p_H'], 'pD': x['p_D'], 'pA': x['p_A']})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if not out['mu_h'].isna().all():
        print('\nmu_h stats:', out['mu_h'].describe().to_string())
        print('\nmu_a stats:', out['mu_a'].describe().to_string())
        for col in ['pH','pD','pA']:
            if col in out.columns:
                print(f"\n{col} stats:", out[col].describe().to_string())


if __name__ == '__main__':
    main()
