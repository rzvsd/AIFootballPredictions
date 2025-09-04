import argparse
import pandas as pd
import bet_fusion as fusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--league', default='SP1')
    args = ap.parse_args()

    cfg = fusion.load_config()
    cfg['league'] = args.league
    mb = fusion.generate_market_book(cfg)
    if mb.empty:
        print('empty market book')
        return
    # show first 3 fixtures worth of 1X2 rows
    shown = 0
    for (date, home, away), g in mb.groupby(['date','home','away']):
        print('\n', date, home, 'vs', away)
        print(g[g['market']=='1X2'][['market','outcome','prob']].to_string(index=False))
        shown += 1
        if shown >= 5:
            break


if __name__ == '__main__':
    main()

