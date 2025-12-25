"""
Grid search for Negative Binomial dispersion parameter k per league.

Runs scripts.metrics_report for a set of k values and writes per-k metric
JSONs plus a summary picking the best k by CRPS over total goals.

Usage:
  python -m scripts.optimize_k --leagues E0,D1,F1,SP1,I1 \
      --k-grid 4,5,6,8 \
      --start 2024-08-01 --end 2025-06-30 \
      --prob-model xgb \
      --out-dir reports/metrics
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_metrics(league: str, k: float, start: str | None, end: str | None, prob_model: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{league}_{prob_model}_negbin_k{int(k)}.json"
    cmd = [
        sys.executable,
        '-m', 'scripts.metrics_report',
        '--league', league,
        '--dist', 'negbin',
        '--k', str(k),
        '--prob-model', prob_model,
        '--out-json', str(out_json),
    ]
    if start:
        cmd += ['--start', start]
    if end:
        cmd += ['--end', end]
    # Calibrated by default; keep as-is for comparability unless explicitly disabled
    subprocess.run(cmd, check=True)
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser(description='Optimize Negative Binomial k per league using CRPS over total goals')
    ap.add_argument('--leagues', default='E0', help='Comma-separated leagues, e.g., E0,D1,F1,SP1,I1')
    ap.add_argument('--k-grid', default='4,5,6,8', help='Comma-separated k values to test')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--prob-model', choices=['xgb','ngb','blend'], default='xgb')
    ap.add_argument('--out-dir', default='reports/metrics')
    ap.add_argument('--summary-out', default='reports/metrics/k_optimization_summary.json')
    args = ap.parse_args()

    leagues = [s.strip() for s in args.leagues.split(',') if s.strip()]
    k_values = [float(s.strip()) for s in args.k_grid.split(',') if s.strip()]
    out_dir = Path(args.out_dir)
    summary_path = Path(args.summary_out)

    results = {}
    for lg in leagues:
        results[lg] = {'tested': [], 'best_k': None, 'best_crps': None}
        for k in k_values:
            try:
                out_json = run_metrics(lg, k, args.start, args.end, args.prob_model, out_dir)
                data = json.loads(Path(out_json).read_text(encoding='utf-8'))
                crps = float(data.get('crps_tg_mean', float('nan')))
                results[lg]['tested'].append({'k': k, 'crps_tg_mean': crps, 'path': str(out_json)})
                bk = results[lg]['best_k']
                bc = results[lg]['best_crps']
                if bc is None or (not (crps != crps)) and crps < bc:  # crps==crps filters NaN
                    results[lg]['best_k'] = k
                    results[lg]['best_crps'] = crps
                print(f"{lg} k={k}: CRPS_TG={crps}")
            except subprocess.CalledProcessError as e:
                print(f"Failed {lg} k={k}: {e}")
            except Exception as e:
                print(f"Error reading metrics for {lg} k={k}: {e}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"Saved summary -> {summary_path}")


if __name__ == '__main__':
    main()

