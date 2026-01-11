"""
CGM-only orchestrator (one-command pipeline).

This script runs the CGM pipeline end-to-end using only local CSV exports under "CGM data/".

Legacy Understat/odds/bot code has been archived under: archive/legacy_non_cgm_engine/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any


ROOT = Path(__file__).resolve().parent


# Pipeline step tracking
_pipeline_steps: List[Dict[str, Any]] = []


def _run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd or ROOT), env=env or os.environ.copy())
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _run_step(step_name: str, cmd: List[str], *, cwd: Optional[Path] = None) -> bool:
    """Run a pipeline step with logging. Returns True on success, False on failure."""
    print(f"\n{'='*60}")
    print(f"[STEP] {step_name}")
    print(f"{'='*60}")
    start = time.time()
    step_info: Dict[str, Any] = {"step": step_name, "start": dt.datetime.now().isoformat()}
    try:
        _run(cmd, cwd=cwd)
        elapsed = time.time() - start
        step_info["status"] = "OK"
        step_info["elapsed_sec"] = round(elapsed, 2)
        print(f"[OK] {step_name} completed in {elapsed:.1f}s")
        _pipeline_steps.append(step_info)
        return True
    except SystemExit as e:
        elapsed = time.time() - start
        step_info["status"] = "FAILED"
        step_info["exit_code"] = e.code
        step_info["elapsed_sec"] = round(elapsed, 2)
        print(f"[FAILED] {step_name} failed with exit code {e.code}")
        _pipeline_steps.append(step_info)
        return False


def _print_pipeline_summary(reports_dir: Path) -> None:
    """Print a human-readable summary of the pipeline run."""
    print(f"\n{'='*60}")
    print("PIPELINE RUN SUMMARY")
    print(f"{'='*60}")
    ok_count = sum(1 for s in _pipeline_steps if s.get("status") == "OK")
    fail_count = sum(1 for s in _pipeline_steps if s.get("status") == "FAILED")
    total = len(_pipeline_steps)
    
    for s in _pipeline_steps:
        status = s.get("status", "?")
        elapsed = s.get("elapsed_sec", 0)
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {s['step']}: {status} ({elapsed:.1f}s)")
    
    print(f"\nTotal: {ok_count}/{total} steps OK")
    if fail_count > 0:
        print(f"FAILURES: {fail_count} step(s) failed - check output above for details")
    else:
        print("All steps completed successfully!")
    
    # Write summary to JSON
    summary_path = reports_dir / "pipeline_summary.json"
    summary = {
        "run_time": dt.datetime.now().isoformat(),
        "steps": _pipeline_steps,
        "ok_count": ok_count,
        "fail_count": fail_count,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")


def _utc_today_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def _latest_file_mtime(path: Path) -> Optional[tuple[float, Path]]:
    if not path.exists():
        return None
    if path.is_file():
        return (path.stat().st_mtime, path)
    latest: Optional[tuple[float, Path]] = None
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            mtime = item.stat().st_mtime
        except OSError:
            continue
        if latest is None or mtime > latest[0]:
            latest = (mtime, item)
    return latest


def _fmt_mtime_utc(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> None:
    ap = argparse.ArgumentParser(description="CGM-only pipeline runner")
    ap.add_argument("--data-dir", default="CGM data", help="CGM export directory")
    ap.add_argument("--enhanced-dir", default="data/enhanced", help="Enhanced output directory")
    ap.add_argument("--models-dir", default="models", help="Model output directory")
    ap.add_argument("--reports-dir", default="reports", help="Reports/log output directory")
    ap.add_argument("--max-date", default=None, help="Elo cutoff date (YYYY-MM-DD). Defaults to today UTC.")
    ap.add_argument("--model-variant", choices=["full", "no_odds"], default="full", help="Model variant (feature set)")
    ap.add_argument("--pick-engine", choices=["goals"], default="goals", help="Pick engine: goals-only (OU25+BTTS)")

    ap.add_argument("--rebuild-history", action="store_true", help="Force rebuild match history from CGM exports")
    ap.add_argument("--skip-train", action="store_true", help="Skip training (use existing models)")
    ap.add_argument("--predict-only", action="store_true", help="Only run prediction using existing artifacts/models")
    ap.add_argument(
        "--allow-stale-history",
        action="store_true",
        help="Allow predict-only runs even if CGM data is newer than the cached history",
    )
    args = ap.parse_args()

    data_dir = str(args.data_dir)
    enhanced_dir = Path(args.enhanced_dir)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    max_date = str(args.max_date or _utc_today_iso())
    model_variant = str(getattr(args, "model_variant", "full") or "full")
    pick_engine = str(getattr(args, "pick_engine", "full") or "full")

    reports_dir.mkdir(parents=True, exist_ok=True)

    history_csv = enhanced_dir / "cgm_match_history.csv"
    elo_csv = enhanced_dir / "cgm_match_history_with_elo.csv"
    elo_stats_csv = enhanced_dir / "cgm_match_history_with_elo_stats.csv"
    elo_stats_xg_csv = enhanced_dir / "cgm_match_history_with_elo_stats_xg.csv"
    franken_csv = enhanced_dir / "frankenstein_training.csv"
    stats_candidates = [
        Path(data_dir) / "multiple leagues and seasons" / "upcoming.csv",
        Path(data_dir) / "cgmbetdatabase.csv",
        Path(data_dir) / "cgmbetdatabase.xls",
        Path(data_dir) / "goals statistics.csv",
    ]
    stats_source = next((p for p in stats_candidates if p.exists()), stats_candidates[-1])

    if args.predict_only:
        history_for_predict = elo_stats_xg_csv if elo_stats_xg_csv.exists() else elo_stats_csv
        latest_data = _latest_file_mtime(Path(data_dir))
        if history_for_predict.exists() and latest_data and latest_data[0] > history_for_predict.stat().st_mtime:
            history_mtime = _fmt_mtime_utc(history_for_predict.stat().st_mtime)
            latest_name = latest_data[1].name
            latest_mtime = _fmt_mtime_utc(latest_data[0])
            msg = (
                "WARNING: CGM data is newer than cached history in predict-only mode.\n"
                f"  history: {history_for_predict} (mtime {history_mtime})\n"
                f"  newest data: {latest_name} (mtime {latest_mtime})\n"
                "Rerun without --predict-only or pass --rebuild-history. "
                "Use --allow-stale-history to proceed anyway."
            )
            if args.allow_stale_history:
                print(msg)
            else:
                print(msg)
                raise SystemExit(2)
        _run(
            [
                sys.executable,
                "-m",
                "cgm.predict_upcoming",
                "--history",
                str(history_for_predict),
                "--models-dir",
                str(models_dir),
                "--model-variant",
                model_variant,
                "--out",
                str(reports_dir / "cgm_upcoming_predictions.csv"),
                "--data-dir",
                data_dir,
                "--as-of-date",
                max_date,
                "--log-json",
                str(reports_dir / "run_log.jsonl"),
                "--trace-json",
                str(reports_dir / "elo_trace.jsonl"),
            ]
        )
        pick_module = "cgm.pick_engine" if pick_engine == "full" else "cgm.pick_engine_goals"
        _run(
            [
                sys.executable,
                "-m",
                pick_module,
                "--in",
                str(reports_dir / "cgm_upcoming_predictions.csv"),
                "--out",
                str(reports_dir / "picks.csv"),
                "--debug-out",
                str(reports_dir / "picks_debug.csv"),
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "cgm.narrator",
                "--in",
                str(reports_dir / "picks.csv"),
                "--out",
                str(reports_dir / "picks_explained.csv"),
                "--preview-out",
                str(reports_dir / "picks_explained_preview.txt"),
            ]
        )
        # Print pipeline summary for predict-only mode
        _print_pipeline_summary(reports_dir)
        return

    # 1) Build match history (missing, forced, or CGM data newer)
    history_reason = None
    latest_data = None
    if args.rebuild_history:
        history_reason = "--rebuild-history"
    elif not history_csv.exists():
        history_reason = "history missing"
    else:
        latest_data = _latest_file_mtime(Path(data_dir))
        if latest_data and latest_data[0] > history_csv.stat().st_mtime:
            history_reason = "data newer than history"

    if history_reason:
        if history_reason == "data newer than history" and latest_data:
            history_mtime = _fmt_mtime_utc(history_csv.stat().st_mtime)
            latest_name = latest_data[1].name
            latest_mtime = _fmt_mtime_utc(latest_data[0])
            print(
                f"[history] newest export: {latest_name} @ {latest_mtime}; "
                f"history: {history_mtime} -> rebuilding."
            )
        else:
            print(f"[history] rebuilding ({history_reason})")
        _run(
            [
                sys.executable,
                "-m",
                "cgm.build_match_history",
                "--data-dir",
                data_dir,
                "--out",
                str(history_csv),
                "--max-date",
                max_date,
            ]
        )
    else:
        history_mtime = _fmt_mtime_utc(history_csv.stat().st_mtime)
        if latest_data:
            latest_name = latest_data[1].name
            latest_mtime = _fmt_mtime_utc(latest_data[0])
            print(
                "[history] reusing cached history "
                f"(mtime {history_mtime}); newest data {latest_name} at {latest_mtime}. "
                "Use --rebuild-history if you exported new data."
            )
        else:
            print(
                f"[history] reusing cached history (mtime {history_mtime}). "
                "Use --rebuild-history if you exported new data."
            )

    # 2) Build league/team baselines (updates match history in-place)
    _run(
        [
            sys.executable,
            "-m",
            "cgm.build_baselines",
            "--data-dir",
            data_dir,
            "--match-history",
            str(history_csv),
            "--out-team-baselines",
            str(enhanced_dir / "team_baselines.csv"),
        ]
    )

    # 3) Recompute Elo with cutoff (writes canonical with_elo.csv)
    _run(
        [
            sys.executable,
            "-m",
            "scripts.calc_cgm_elo",
            "--history",
            str(history_csv),
            "--out",
            str(elo_csv),
            "--data-dir",
            data_dir,
            "--max-date",
            max_date,
            "--log-json",
            str(reports_dir / "run_log.jsonl"),
        ]
    )

    # 4) Backfill per-match stats needed by pressure features (writes with_elo_stats.csv)
    _run(
        [
            sys.executable,
            "-m",
            "cgm.backfill_match_stats",
            "--history",
            str(elo_csv),
            "--stats",
            str(stats_source),
            "--out",
            str(elo_stats_csv),
            "--data-dir",
            data_dir,
        ]
    )

    # 5) Milestone 3: build leakage-safe xG proxy history
    _run(
        [
            sys.executable,
            "-m",
            "cgm.build_xg_proxy",
            "--history",
            str(elo_stats_csv),
            "--out",
            str(elo_stats_xg_csv),
        ]
    )

    # 6) Build Frankenstein training matrix
    _run(
        [
            sys.executable,
            "-m",
            "cgm.build_frankenstein",
            "--data-dir",
            str(enhanced_dir),
            "--match-history",
            elo_stats_xg_csv.name,
            "--out",
            str(franken_csv),
            "--out-full",
            str(enhanced_dir / "frankenstein_training_full.csv"),
        ]
    )

    # 7) Train mu models (unless skipped)
    if not args.skip_train:
        _run(
            [
                sys.executable,
                "-m",
                "cgm.train_frankenstein_mu",
                "--data",
                str(franken_csv),
                "--out-dir",
                str(models_dir),
                "--variant",
                model_variant,
            ]
        )

    # 8) Predict upcoming + EV
    _run(
        [
            sys.executable,
            "-m",
            "cgm.predict_upcoming",
            "--history",
            str(elo_stats_xg_csv),
            "--models-dir",
            str(models_dir),
            "--model-variant",
            model_variant,
            "--out",
            str(reports_dir / "cgm_upcoming_predictions.csv"),
            "--data-dir",
            data_dir,
            "--as-of-date",
            max_date,
            "--log-json",
            str(reports_dir / "run_log.jsonl"),
            "--trace-json",
            str(reports_dir / "elo_trace.jsonl"),
        ]
    )

    # 9) Milestone 7: deterministic goals-only pick engine (O/U 2.5 + BTTS)
    pick_module = "cgm.pick_engine" if pick_engine == "full" else "cgm.pick_engine_goals"
    _run(
        [
            sys.executable,
            "-m",
            pick_module,
            "--in",
            str(reports_dir / "cgm_upcoming_predictions.csv"),
            "--out",
            str(reports_dir / "picks.csv"),
            "--debug-out",
            str(reports_dir / "picks_debug.csv"),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "cgm.narrator",
            "--in",
            str(reports_dir / "picks.csv"),
            "--out",
            str(reports_dir / "picks_explained.csv"),
            "--preview-out",
            str(reports_dir / "picks_explained_preview.txt"),
        ]
    )

    # Print pipeline summary
    _print_pipeline_summary(reports_dir)
if __name__ == "__main__":
    main()
