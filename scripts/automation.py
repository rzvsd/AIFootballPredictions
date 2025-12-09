"""
Automation helpers for nightly/weekly/monthly jobs (Phase 5).

Jobs:
- nightly: refresh Understat fixtures snapshot for upcoming week
- weekly: update & report (optionally export)
- monthly: health report (metrics) per league

Usage examples:
  python -m scripts.automation nightly --leagues E0 D1 F1 I1 SP1 --tag opening
  python -m scripts.automation weekly --leagues E0 D1 F1 I1 SP1 --export
  python -m scripts.automation monthly --leagues E0 D1 F1 I1 SP1 --start 2023-08-01 --end 2024-06-30
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run(cmd: List[str]) -> int:
    print('$', ' '.join(cmd))
    return subprocess.call(cmd)


def cmd_py(module: str, *args: str) -> List[str]:
    exe = sys.executable or 'python'
    return [exe, '-m', module, *args]


def job_nightly(leagues: List[str], tag: str) -> int:
    # Refresh fixtures and odds for all leagues (ignores tag; kept for CLI compatibility)
    fx_args = ['--leagues', *leagues, '--days', '7']
    run(cmd_py('scripts.fetch_fixtures_understat', *fx_args))
    odds_args = ['--leagues', *leagues, '--days', '7']
    return run(cmd_py('scripts.fetch_odds_fd_simple', *odds_args))


def job_weekly(leagues: List[str], export: bool) -> int:
    args = ['--leagues', *leagues, '--season-codes', '2425']
    if export:
        args.append('--export')
    return run(cmd_py('scripts.update_and_report', *args))


def job_monthly(leagues: List[str], start: str | None, end: str | None) -> int:
    Path('reports/metrics').mkdir(parents=True, exist_ok=True)
    rc = 0
    for lg in leagues:
        out = Path('reports/metrics')/f'{lg}_metrics.json'
        args = ['--league', lg]
        if start: args += ['--start', start]
        if end: args += ['--end', end]
        args += ['--out-json', str(out)]
        r = run(cmd_py('scripts.metrics_report', *args))
        rc = rc or r
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description='Automation entrypoint')
    sub = ap.add_subparsers(dest='job', required=True)

    sp1 = sub.add_parser('nightly')
    sp1.add_argument('--leagues', nargs='+', required=True)
    sp1.add_argument('--tag', choices=['opening','closing'], default='closing')

    sp2 = sub.add_parser('weekly')
    sp2.add_argument('--leagues', nargs='+', required=True)
    sp2.add_argument('--export', action='store_true')

    sp3 = sub.add_parser('monthly')
    sp3.add_argument('--leagues', nargs='+', required=True)
    sp3.add_argument('--start', default=None)
    sp3.add_argument('--end', default=None)

    args = ap.parse_args()
    if args.job == 'nightly':
        sys.exit(job_nightly(args.leagues, args.tag))
    if args.job == 'weekly':
        sys.exit(job_weekly(args.leagues, args.export))
    if args.job == 'monthly':
        sys.exit(job_monthly(args.leagues, args.start, args.end))


if __name__ == '__main__':
    main()
