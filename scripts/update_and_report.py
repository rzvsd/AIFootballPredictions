"""
Orchestrate the weekly flow: acquire -> preprocess -> (optional train) -> calibrate -> report.

Examples:
  python -m scripts.update_and_report \
    --leagues E0 D1 F1 I1 SP1 \
    --season-codes 2425 \
    --fixtures-csv E0=data/fixtures/E0_weekly_fixtures.csv D1=data/fixtures/D1_manual.csv F1=data/fixtures/F1_manual.csv I1=data/fixtures/I1_manual.csv SP1=data/fixtures/SP1_manual.csv \
    --select prob --export

Skip steps if you already have fresh files:
  python -m scripts.update_and_report --leagues E0 D1 F1 I1 SP1 --season-codes 2425 --skip-download --skip-preprocess --export
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run(cmd: List[str], cwd: Path | None = None) -> int:
    print("$", " ".join(shlex.quote(c) for c in cmd))
    try:
        cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
        return cp.returncode
    except Exception as e:
        print(f"[error] Failed: {' '.join(cmd)} => {e}")
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly update and report pipeline")
    ap.add_argument('--leagues', nargs='+', required=True, help='Leagues, e.g., E0 D1 F1 I1 SP1')
    ap.add_argument('--season-codes', nargs='+', default=['2425'], help='Football-data.co.uk season codes (e.g., 2425)')
    ap.add_argument('--fixtures-csv', nargs='*', default=[], help='Pairs LEAGUE=path.csv for report filtering')
    ap.add_argument('--since', type=int, default=2021, help='Calibration start year (inclusive)')
    ap.add_argument('--method', choices=['isotonic','platt'], default='isotonic', help='Calibration method')
    ap.add_argument('--select', choices=['prob','ev'], default='prob', help='Selection mode for report')
    ap.add_argument('--export', action='store_true', help='Export per-league CSVs from the report')
    ap.add_argument('--fetch-odds', action='store_true', help='Fetch odds via API-Football into data/odds before report')
    ap.add_argument('--no-calibration', action='store_true', help='Report without applying calibrators')
    ap.add_argument('--train', action='store_true', help='Retrain XGB models per league before calibration')
    ap.add_argument('--skip-download', action='store_true', help='Skip data acquisition (assume raw already updated)')
    ap.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing (assume processed/enhanced ready)')
    args = ap.parse_args()

    exe = sys.executable or 'python'
    leagues = args.leagues

    # 1) Acquire raw
    if not args.skip_download:
        cmd = [exe, '-m', 'scripts.data_acquisition', '--leagues', *leagues, '--seasons', *args.season_codes, '--raw_data_output_dir', 'data/raw']
        if run(cmd) != 0:
            print('[warn] Acquisition step failed or partially completed. Continuing...')

    # 2) Preprocess processed CSVs (feature engineering, cleaning)
    if not args.skip_preprocess:
        cmd = [exe, '-m', 'scripts.data_preprocessing', '--raw_data_input_dir', 'data/raw', '--processed_data_output_dir', 'data/processed']
        if run(cmd) != 0:
            print('[warn] Preprocessing step failed. Continuing...')

    # 3) Retrain (optional)
    if args.train:
        for lg in leagues:
            cmd = [exe, 'xgb_trainer.py', '--league', lg]
            if run(cmd) != 0:
                print(f'[warn] Training failed for {lg}. Continuing...')

    # 4) Calibrate per league
    cmd = [exe, '-m', 'scripts.calibrate_league', '--league', *leagues, '--method', args.method, '--since', str(args.since)]
    if run(cmd) != 0:
        print('[warn] Calibration failed. Continuing to report...')

    # 5) Odds fetch (optional)
    if args.fetch_odds:
        for lg in leagues:
            fx = None
            for pair in args.fixtures_csv:
                if pair.startswith(lg + "="):
                    fx = pair.split('=', 1)[1]
                    break
            cmd = [exe, '-m', 'scripts.fetch_odds_api_football', '--league', lg]
            if fx:
                cmd.extend(['--fixtures-csv', fx])
            else:
                cmd.extend(['--days', '7'])
            _ = run(cmd)

    # 6) Report
    fx_pairs: List[str] = []
    for pair in args.fixtures_csv:
        if '=' in pair:
            fx_pairs.append(pair)
    rep_cmd = [exe, '-m', 'scripts.multi_league_report', '--leagues', *leagues, '--select', args.select]
    if args.no_calibration:
        rep_cmd.append('--no-calibration')
    if args.export:
        rep_cmd.append('--export')
    if fx_pairs:
        rep_cmd.extend(['--fixtures-csv', *fx_pairs])
    code = run(rep_cmd)
    sys.exit(code)


if __name__ == '__main__':
    main()
