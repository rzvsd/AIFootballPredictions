#!/usr/bin/env python3
"""
Skip Reconciliation Audit.

Goal:
  Ensure skipped fixtures are fully accounted for and not silently drifting.
  This audit does not change prediction logic; it validates run integrity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


_ALLOWED_SKIP_REASONS = {
    "missing_snapshot",
    "insufficient_history",
    "missing_elo",
    "missing_core_features",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _is_predictions_output(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).replace("/", "\\").lower()
    return s.endswith("reports\\cgm_upcoming_predictions.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit skipped-fixture reconciliation")
    ap.add_argument("--predictions", default="reports/cgm_upcoming_predictions.csv", help="Predictions CSV")
    ap.add_argument("--run-log", default="reports/run_log.jsonl", help="JSONL run log")
    ap.add_argument("--max-skip-ratio", type=float, default=0.10, help="Max allowed skipped/expected ratio")
    args = ap.parse_args()

    print("=" * 80)
    print("SKIP RECONCILIATION AUDIT")
    print("=" * 80)

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"[FAIL] Missing predictions file: {pred_path}")
        return 1

    df = pd.read_csv(pred_path)
    produced_rows = int(len(df))

    expected_after_filters = None
    run_asof = None
    if produced_rows > 0:
        if "upcoming_rows_after_filters" in df.columns:
            vals = pd.to_numeric(df["upcoming_rows_after_filters"], errors="coerce").dropna().unique()
            if len(vals) == 1:
                expected_after_filters = int(vals[0])
        if "run_asof_datetime" in df.columns:
            vals = pd.to_datetime(df["run_asof_datetime"], errors="coerce").dropna().astype(str).unique()
            if len(vals) == 1:
                run_asof = str(vals[0])

    events = _read_jsonl(Path(args.run_log))
    pred_event = None
    for e in events:
        if e.get("task") == "predict_upcoming" and _is_predictions_output(e.get("output")):
            pred_event = e
    if pred_event is None:
        print("[FAIL] No predict_upcoming event found for reports/cgm_upcoming_predictions.csv")
        return 1

    logged_rows = int(pred_event.get("rows", 0) or 0)
    skipped_counts_raw = pred_event.get("skipped_counts", {})
    skipped_counts = skipped_counts_raw if isinstance(skipped_counts_raw, dict) else {}
    skipped_total = int(sum(int(v or 0) for v in skipped_counts.values()))

    # Fallback for expected rows: use latest UPCOMING_SCOPE with same run_asof if available.
    if expected_after_filters is None:
        for e in reversed(events):
            if e.get("event") != "UPCOMING_SCOPE":
                continue
            ev_asof = str(e.get("run_asof_datetime") or "")
            if run_asof and ev_asof and run_asof.startswith(ev_asof[:19]):
                expected_after_filters = int(e.get("upcoming_rows_after_filters", 0) or 0)
                break
        if expected_after_filters is None:
            # Last-resort fallback: at least reconcile against logged rows.
            expected_after_filters = logged_rows + skipped_total

    issues: list[str] = []

    if produced_rows != logged_rows:
        issues.append(f"predictions rows mismatch: csv={produced_rows} log={logged_rows}")

    reconciled = produced_rows + skipped_total
    if reconciled != int(expected_after_filters):
        issues.append(
            f"row reconciliation failed: produced({produced_rows}) + skipped({skipped_total}) != expected_after_filters({expected_after_filters})"
        )

    unknown_reasons = sorted([k for k in skipped_counts.keys() if k not in _ALLOWED_SKIP_REASONS])
    if unknown_reasons:
        issues.append(f"unknown skip reasons present: {unknown_reasons}")

    if int(expected_after_filters) > 0:
        skip_ratio = float(skipped_total / float(expected_after_filters))
    else:
        skip_ratio = 0.0

    print(f"produced_rows={produced_rows}")
    print(f"expected_after_filters={expected_after_filters}")
    print(f"skipped_total={skipped_total}")
    print(f"skip_ratio={skip_ratio:.3f} max={args.max_skip_ratio:.3f}")
    print(f"skip_reasons={skipped_counts}")

    if skip_ratio > float(args.max_skip_ratio):
        issues.append(f"skip ratio too high: {skip_ratio:.3f} > {args.max_skip_ratio:.3f}")

    if issues:
        print("[FAIL] Skip reconciliation failed")
        for msg in issues:
            print(f"  - {msg}")
        return 1

    print("[PASS] Skip reconciliation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
