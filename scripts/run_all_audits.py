#!/usr/bin/env python3
"""
Run all audits for multi-league verification.

Usage:
    python scripts/run_all_audits.py
    python scripts/run_all_audits.py --critical-only --as-of-date YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]


def _build_audits(*, as_of_date: str) -> List[Dict[str, Any]]:
    """Build audit command list."""
    return [
        {
            "name": "Detailed Audit Log",
            "cmd": [sys.executable, "scripts/audit_detailed.py"],
            "critical": True,
        },
        {
            "name": "Upcoming Feed Scope",
            "cmd": [
                sys.executable,
                "-m",
                "scripts.audit_upcoming_feed",
                "--as-of-date",
                str(as_of_date),
            ],
            "critical": True,
        },
        {
            "name": "Multi-League Coverage",
            "cmd": [sys.executable, "scripts/audit_multi_league.py"],
            "critical": False,
        },
        {
            "name": "Prediction Data Health",
            "cmd": [sys.executable, "scripts/audit_prediction_data_health.py"],
            "critical": True,
        },
        {
            "name": "Skip Reconciliation",
            "cmd": [sys.executable, "scripts/audit_skip_reconciliation.py"],
            "critical": True,
        },
        {
            "name": "Backtest Leakage",
            "cmd": [sys.executable, "-m", "scripts.audit_backtest"],
            "critical": False,
        },
        {
            "name": "Calculation Verification",
            "cmd": [sys.executable, "scripts/audit_calculations.py"],
            "critical": True,
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run audit suite")
    ap.add_argument(
        "--critical-only",
        action="store_true",
        help="Run only critical audits (used by mandatory pre-bet gate).",
    )
    ap.add_argument(
        "--as-of-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="As-of date (YYYY-MM-DD) forwarded to scope audit.",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Per-audit timeout in seconds.",
    )
    args = ap.parse_args()

    audits = _build_audits(as_of_date=str(args.as_of_date))
    if args.critical_only:
        audits = [a for a in audits if bool(a.get("critical", False))]

    mode = "CRITICAL-ONLY" if args.critical_only else "FULL"
    print("=" * 80)
    print("RUNNING ALL AUDITS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {mode}")
    print(f"As-Of Date: {args.as_of_date}")
    print("=" * 80)

    results = []

    for audit in audits:
        name = audit["name"]
        cmd = audit["cmd"]
        critical = bool(audit.get("critical", False))

        print(f"\n{'-' * 80}")
        print(f"{name}")
        print(f"{'-' * 80}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(_ROOT),
                capture_output=True,
                text=True,
                timeout=max(1, int(args.timeout_sec)),
            )

            if result.stdout:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 50:
                    print(f"... ({len(lines) - 50} lines truncated)")
                    print("\n".join(lines[-50:]))
                else:
                    print(result.stdout)

            if result.returncode == 0:
                results.append((name, "PASS", critical))
            else:
                error_snippet = result.stderr[:200] if result.stderr else "Unknown error"
                results.append((name, f"FAIL: {error_snippet}", critical))
                if result.stderr:
                    print(f"\nError: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            results.append((name, "TIMEOUT", critical))
        except Exception as e:
            results.append((name, f"ERROR: {str(e)[:50]}", critical))

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, status, _ in results if "PASS" in status)
    failed = sum(1 for _, status, _ in results if "FAIL" in status or "ERROR" in status or "TIMEOUT" in status)
    critical_failed = sum(1 for _, status, critical in results if critical and "PASS" not in status)

    for name, status, critical in results:
        marker = "[CRITICAL]" if critical else ""
        print(f"  {status[:20]:<22} {name} {marker}")

    print("=" * 80)
    print(f"Total: {passed}/{len(results)} passed | {failed} issues")

    if critical_failed > 0:
        print(f"\n{critical_failed} CRITICAL audit(s) failed")
        return 1
    if failed > 0:
        print(f"\n{failed} non-critical issue(s) found")
        return 0

    print("\nAll audits passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
