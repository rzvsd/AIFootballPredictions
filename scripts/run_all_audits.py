#!/usr/bin/env python3
"""
Run All Audits for Multi-League Verification
=============================================
Runs all available audit scripts and reports results.

Usage:
    python scripts/run_all_audits.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Audit scripts to run (in order)
AUDITS = [
    {
        "name": "Multi-League Coverage",
        "cmd": [sys.executable, "scripts/audit_multi_league.py"],
        "critical": True,
    },
    {
        "name": "Upcoming Feed Scope",
        "cmd": [sys.executable, "-m", "scripts.audit_upcoming_feed", "--as-of-date", datetime.now().strftime("%Y-%m-%d")],
        "critical": True,
    },
    {
        "name": "Picks Engine (Goals)",
        "cmd": [sys.executable, "-m", "scripts.audit_picks_goals"],
        "critical": True,
    },
    {
        "name": "Pressure Features",
        "cmd": [sys.executable, "-m", "scripts.audit_pressure", "--cutoff", datetime.now().strftime("%Y-%m-%d")],
        "critical": False,
    },
    {
        "name": "xG Proxy Features",
        "cmd": [sys.executable, "-m", "scripts.audit_xg"],
        "critical": False,
    },
    {
        "name": "Time Decay Features",
        "cmd": [sys.executable, "-m", "scripts.audit_decay"],
        "critical": False,
    },
    {
        "name": "H2H Features",
        "cmd": [sys.executable, "-m", "scripts.audit_h2h"],
        "critical": False,
    },
    {
        "name": "League Features",
        "cmd": [sys.executable, "-m", "scripts.audit_league_features"],
        "critical": False,
    },
    {
        "name": "Narrator",
        "cmd": [sys.executable, "-m", "scripts.audit_narrator"],
        "critical": False,
    },
]


def main():
    print("=" * 80)
    print("🔍 RUNNING ALL AUDITS")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results = []
    
    for audit in AUDITS:
        name = audit["name"]
        cmd = audit["cmd"]
        critical = audit["critical"]
        
        print(f"\n{'─' * 80}")
        print(f"📋 {name}")
        print(f"{'─' * 80}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(_ROOT),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per audit
            )
            
            # Print output
            if result.stdout:
                # Limit output to last 50 lines for brevity
                lines = result.stdout.strip().split("\n")
                if len(lines) > 50:
                    print(f"... ({len(lines) - 50} lines truncated)")
                    print("\n".join(lines[-50:]))
                else:
                    print(result.stdout)
            
            if result.returncode == 0:
                results.append((name, "✅ PASS", critical))
            else:
                error_snippet = result.stderr[:200] if result.stderr else "Unknown error"
                results.append((name, f"❌ FAIL: {error_snippet}", critical))
                if result.stderr:
                    print(f"\nError: {result.stderr[:500]}")
                    
        except subprocess.TimeoutExpired:
            results.append((name, "⏱️ TIMEOUT", critical))
        except Exception as e:
            results.append((name, f"💥 ERROR: {str(e)[:50]}", critical))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 AUDIT SUMMARY")
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
        print(f"\n⚠️  {critical_failed} CRITICAL audit(s) failed!")
        return 1
    elif failed > 0:
        print(f"\n⚠️  {failed} non-critical issue(s) found")
        return 0
    else:
        print("\n✅ All audits passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
