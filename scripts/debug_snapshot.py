import argparse
import os
import pandas as pd
import feature_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--league', default='SP1')
    ap.add_argument('--enhanced', default=None, help='Path to enhanced final_features CSV')
    args = ap.parse_args()

    league = args.league
    enh = args.enhanced or os.path.join('data','enhanced', f'{league}_final_features.csv')
    try:
        snap = feature_store.build_snapshot(enhanced_csv=enh, as_of=None, half_life_matches=5, elo_k=15.0, elo_home_adv=40.0)
    except Exception as e:
        print(f'Failed to build snapshot: {e}')
        return

    cols = [
        'team', 'xg_home_EWMA','xga_home_EWMA','xg_away_EWMA','xga_away_EWMA',
        'ppg_home_EWMA','ppg_away_EWMA','elo'
    ]
    have = [c for c in cols if c in snap.columns]
    print('Snapshot rows:', len(snap))
    print(snap[have].head(10).to_string(index=False))
    print('\nDescribe:')
    print(snap[have[1:]].describe().to_string())


if __name__ == '__main__':
    main()

