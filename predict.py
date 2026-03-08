"""
CGM orchestrator (one-command pipeline).

Default flow is API-Football -> normalized CSV bridge -> CGM pipeline.
Legacy local-CSV input remains available via `--data-source csv`.

Legacy Understat/odds/bot code has been archived under: archive/legacy_non_cgm_engine/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import config


ROOT = Path(__file__).resolve().parent
CSV_DATA_DIR_DEFAULT = "CGM data"
API_HISTORY_FILENAME = "history_fixtures.csv"
API_UPCOMING_FILENAME = "upcoming_fixtures.csv"
API_QUALITY_REPORT_FILENAME = "fixture_quality_report.json"
QUALITY_GATE_DEFAULT_MIN_ODDS_COVERAGE = 0.60
QUALITY_GATE_DEFAULT_MIN_STATS_COVERAGE = 0.60


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
        icon = "[OK]" if status == "OK" else "[X]"
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


def _int_env_or_default(env_key: str, default_value: int) -> int:
    raw = os.getenv(env_key)
    if raw is not None and str(raw).strip():
        try:
            return int(str(raw).strip())
        except ValueError:
            pass
    return int(default_value)


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


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_api_output_file(data_dir: Path, filename: str) -> Optional[Path]:
    return _first_existing(
        [
            data_dir / filename,
            data_dir / "multiple leagues and seasons" / filename,
        ]
    )


def _copy_if_needed(src: Path, dst: Path) -> None:
    src_resolved = src.resolve()
    dst_resolved = dst.resolve() if dst.exists() else dst
    if src_resolved == dst_resolved:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _merge_csv_files_binary(history_src: Path, upcoming_src: Path, merged_out: Path) -> None:
    history_lines = history_src.read_bytes().splitlines(keepends=True)
    upcoming_lines = upcoming_src.read_bytes().splitlines(keepends=True)
    if not history_lines:
        raise ValueError(f"API history fixture file is empty: {history_src}")
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_out, "wb") as f:
        f.writelines(history_lines)
        if upcoming_lines:
            history_header = history_lines[0].strip()
            upcoming_header = upcoming_lines[0].strip()
            start_idx = 1 if upcoming_header == history_header else 0
            f.writelines(upcoming_lines[start_idx:])


def _bridge_api_fixture_inputs(data_dir: Path) -> Dict[str, Path]:
    history_src = _resolve_api_output_file(data_dir, API_HISTORY_FILENAME)
    upcoming_src = _resolve_api_output_file(data_dir, API_UPCOMING_FILENAME)
    if history_src is None:
        raise FileNotFoundError(
            f"Missing API history fixtures: expected '{API_HISTORY_FILENAME}' under {data_dir}"
        )
    if upcoming_src is None:
        raise FileNotFoundError(
            f"Missing API upcoming fixtures: expected '{API_UPCOMING_FILENAME}' under {data_dir}"
        )

    multi_dir = data_dir / "multiple leagues and seasons"
    bridged_history = multi_dir / API_HISTORY_FILENAME
    bridged_upcoming = multi_dir / API_UPCOMING_FILENAME
    legacy_allrating = multi_dir / "allratingv.csv"
    legacy_upcoming = multi_dir / "upcoming.csv"

    _copy_if_needed(history_src, bridged_history)
    _copy_if_needed(upcoming_src, bridged_upcoming)
    _copy_if_needed(bridged_upcoming, legacy_upcoming)
    _merge_csv_files_binary(bridged_history, bridged_upcoming, legacy_allrating)

    quality_report = _resolve_api_output_file(data_dir, API_QUALITY_REPORT_FILENAME)
    if quality_report is None:
        quality_report = data_dir / API_QUALITY_REPORT_FILENAME

    print(f"[api-football] bridged history fixtures -> {legacy_allrating}")
    print(f"[api-football] bridged upcoming fixtures -> {legacy_upcoming}")

    return {
        "history_fixtures": bridged_history,
        "upcoming_fixtures": bridged_upcoming,
        "quality_report": quality_report,
    }


def _coerce_ratio(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num < 0:
        return None
    if num > 1.0 and num <= 100.0:
        num = num / 100.0
    return num


def _flatten_json_paths(payload: Any, *, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            out[path.lower()] = value
            out.update(_flatten_json_paths(value, prefix=path))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            out.update(_flatten_json_paths(value, prefix=path))
    return out


def _extract_ratio(
    flat_payload: Dict[str, Any], token_sets: List[tuple[str, ...]]
) -> tuple[Optional[float], Optional[str]]:
    for tokens in token_sets:
        for path, raw_value in flat_payload.items():
            if all(token in path for token in tokens):
                ratio = _coerce_ratio(raw_value)
                if ratio is not None:
                    return ratio, path
    return None, None


def _extract_bool_flag(payload: Any, candidate_keys: set[str]) -> Optional[bool]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_norm = str(key).strip().lower()
            if key_norm in candidate_keys:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    value_norm = value.strip().lower()
                    if value_norm in {"true", "yes", "ok", "pass", "passed"}:
                        return True
                    if value_norm in {"false", "no", "fail", "failed"}:
                        return False
            nested = _extract_bool_flag(value, candidate_keys)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _extract_bool_flag(item, candidate_keys)
            if nested is not None:
                return nested
    return None


def _evaluate_fixture_quality_gate(report_path: Path, *, require_odds: bool = True) -> tuple[bool, List[str]]:
    if not report_path.exists():
        return False, [f"Quality report missing: {report_path}"]

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        report = json.loads(report_path.read_text(encoding="latin1"))
    except Exception as exc:
        return False, [f"Failed to parse quality report {report_path}: {exc}"]

    explicit_gate = _extract_bool_flag(
        report,
        {
            "quality_gate_passed",
            "gate_passed",
            "quality_passed",
            "fixture_quality_passed",
            "quality_ok",
        },
    )
    if explicit_gate is False:
        explicit_reasons = report.get("quality_gate_reasons") if isinstance(report, dict) else None
        if isinstance(explicit_reasons, list):
            reasons = [str(r).strip() for r in explicit_reasons if str(r).strip()]
            if reasons:
                return False, reasons
        return False, [f"Quality report marks gate as failed: {report_path}"]

    flat = _flatten_json_paths(report)
    odds_cov, odds_cov_path = _extract_ratio(
        flat,
        [
            ("odds", "coverage"),
            ("coverage", "odds"),
            ("odds", "available"),
            ("odds", "complete"),
        ],
    )
    stats_cov, stats_cov_path = _extract_ratio(
        flat,
        [
            ("stats", "coverage"),
            ("coverage", "stats"),
            ("stat", "coverage"),
            ("coverage", "stat"),
        ],
    )
    min_odds, min_odds_path = _extract_ratio(
        flat,
        [
            ("min", "odds", "coverage"),
            ("odds", "coverage", "min"),
            ("threshold", "odds", "coverage"),
            ("odds", "coverage", "threshold"),
        ],
    )
    min_stats, min_stats_path = _extract_ratio(
        flat,
        [
            ("min", "stats", "coverage"),
            ("stats", "coverage", "min"),
            ("threshold", "stats", "coverage"),
            ("stats", "coverage", "threshold"),
            ("min", "stat", "coverage"),
            ("stat", "coverage", "min"),
            ("threshold", "stat", "coverage"),
            ("stat", "coverage", "threshold"),
        ],
    )

    min_odds = min_odds if min_odds is not None else QUALITY_GATE_DEFAULT_MIN_ODDS_COVERAGE
    min_stats = min_stats if min_stats is not None else QUALITY_GATE_DEFAULT_MIN_STATS_COVERAGE

    if (require_odds and odds_cov is None) or stats_cov is None:
        if explicit_gate is True:
            return True, [f"Gate passed by explicit quality flag in {report_path.name}"]
        missing = []
        if require_odds and odds_cov is None:
            missing.append("odds coverage")
        if stats_cov is None:
            missing.append("stats coverage")
        return False, [f"Could not evaluate quality gate: missing {', '.join(missing)} metrics in {report_path}"]

    reasons: List[str] = []
    if require_odds and odds_cov is not None and odds_cov < min_odds:
        reasons.append(
            f"odds coverage {odds_cov:.1%} < minimum {min_odds:.1%} "
            f"(metrics: {odds_cov_path or 'unknown'} vs {min_odds_path or 'default'})"
        )
    if stats_cov < min_stats:
        reasons.append(
            f"stats coverage {stats_cov:.1%} < minimum {min_stats:.1%} "
            f"(metrics: {stats_cov_path or 'unknown'} vs {min_stats_path or 'default'})"
        )
    if reasons:
        return False, reasons

    ok_messages = [f"stats coverage {stats_cov:.1%} (min {min_stats:.1%})"]
    if require_odds and odds_cov is not None:
        ok_messages.insert(0, f"odds coverage {odds_cov:.1%} (min {min_odds:.1%})")
    else:
        ok_messages.insert(0, "odds coverage check skipped (predictions-only mode)")
    return True, ok_messages


def _run_mandatory_pre_bet_gate(*, as_of_date: str, model_variant: str) -> None:
    """
    Mandatory gate before pick generation.
    Runs critical audits only and blocks picks if any critical audit fails.
    """
    print("[pre-bet-gate] running mandatory critical audits before pick generation...")
    gate_cmd = [
        sys.executable,
        "scripts/run_all_audits.py",
        "--critical-only",
        "--as-of-date",
        str(as_of_date),
        "--model-variant",
        str(model_variant),
    ]
    _run(gate_cmd)
    print("[pre-bet-gate] PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description="API-first CGM pipeline runner")
    ap.add_argument(
        "--data-source",
        choices=["api-football", "csv"],
        default="api-football",
        help="Input source: API-Football sync outputs or legacy CSV exports",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Data directory. Defaults to API_FOOTBALL_DEFAULT_DATA_DIR for --data-source api-football, "
            "or 'CGM data' for --data-source csv."
        ),
    )
    ap.add_argument("--enhanced-dir", default="data/enhanced", help="Enhanced output directory")
    ap.add_argument("--models-dir", default="models", help="Model output directory")
    ap.add_argument("--reports-dir", default="reports", help="Reports/log output directory")
    ap.add_argument("--max-date", default=None, help="Elo cutoff date (YYYY-MM-DD). Defaults to today UTC.")
    ap.add_argument(
        "--model-variant",
        choices=["full", "no_odds"],
        default=str(getattr(config, "PIPELINE_DEFAULT_MODEL_VARIANT", "no_odds")),
        help="Model variant (feature set)",
    )
    ap.add_argument("--pick-engine", choices=["goals"], default="goals", help="Pick engine: goals-only (OU25+BTTS)")
    ap.add_argument(
        "--emit-picks",
        action="store_true",
        default=bool(getattr(config, "PIPELINE_EMIT_PICKS_DEFAULT", False)),
        help="Run pick engine and narrator outputs (disabled by default).",
    )
    ap.add_argument(
        "--next-round-only",
        dest="next_round_only",
        action="store_true",
        default=bool(getattr(config, "LIVE_SCOPE_NEXT_ROUND_ONLY", True)),
        help="Keep only immediate next round fixtures.",
    )
    ap.add_argument(
        "--all-upcoming",
        dest="next_round_only",
        action="store_false",
        help="Disable next-round-only filter and keep all fixtures in scope.",
    )
    ap.add_argument(
        "--next-round-span-days",
        type=int,
        default=int(getattr(config, "LIVE_SCOPE_NEXT_ROUND_SPAN_DAYS", 3) or 3),
        help="When next-round-only is enabled, keep fixtures up to this many days after the earliest fixture.",
    )

    ap.add_argument("--rebuild-history", action="store_true", help="Force rebuild match history from CGM exports")
    ap.add_argument("--predict-only", action="store_true", help="Only run prediction using existing artifacts/models")
    ap.add_argument(
        "--allow-stale-history",
        action="store_true",
        help="Allow predict-only runs even if CGM data is newer than the cached history",
    )
    args = ap.parse_args()

    data_source = str(getattr(args, "data_source", "api-football") or "api-football")
    api_default_data_dir = str(getattr(config, "API_FOOTBALL_DEFAULT_DATA_DIR", "data/api_football"))
    data_dir = str(args.data_dir or (api_default_data_dir if data_source == "api-football" else CSV_DATA_DIR_DEFAULT))
    data_dir_path = Path(data_dir)
    enhanced_dir = Path(args.enhanced_dir)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    max_date = str(args.max_date or _utc_today_iso())
    model_variant = str(
        getattr(args, "model_variant", getattr(config, "PIPELINE_DEFAULT_MODEL_VARIANT", "no_odds"))
        or getattr(config, "PIPELINE_DEFAULT_MODEL_VARIANT", "no_odds")
    )
    pick_engine = str(getattr(args, "pick_engine", "goals") or "goals")

    reports_dir.mkdir(parents=True, exist_ok=True)

    api_context: Optional[Dict[str, Path]] = None
    if data_source == "api-football":
        sync_script = ROOT / "scripts" / "sync_api_football.py"
        if not sync_script.exists():
            raise SystemExit(f"API data source requested but sync script not found: {sync_script}")
        sync_env = os.environ.copy()
        sync_env.setdefault("API_FOOTBALL_DATA_DIR", data_dir)
        env_league_ids = os.getenv("API_FOOTBALL_LEAGUE_IDS")
        if env_league_ids and env_league_ids.strip():
            sync_league_ids = [s.strip() for s in env_league_ids.split(",") if s.strip()]
        else:
            sync_league_ids = getattr(config, "API_FOOTBALL_DEFAULT_LEAGUE_IDS", [])
        if isinstance(sync_league_ids, str):
            sync_league_ids = [s.strip() for s in sync_league_ids.split(",") if s.strip()]
        sync_league_ids_csv = ",".join(str(v).strip() for v in (sync_league_ids or []) if str(v).strip())
        sync_history_days = _int_env_or_default(
            "API_FOOTBALL_HISTORY_DAYS",
            int(getattr(config, "API_FOOTBALL_DEFAULT_HISTORY_DAYS", 365)),
        )
        sync_horizon_days = _int_env_or_default(
            "API_FOOTBALL_HORIZON_DAYS",
            int(getattr(config, "API_FOOTBALL_DEFAULT_HORIZON_DAYS", 14)),
        )
        sync_max_requests = _int_env_or_default(
            "API_FOOTBALL_MAX_REQUESTS",
            int(getattr(config, "API_FOOTBALL_MAX_REQUESTS_FREE", 100)),
        )
        sync_rate_per_min = _int_env_or_default(
            "API_FOOTBALL_RATE_PER_MINUTE",
            int(getattr(config, "API_FOOTBALL_RATE_PER_MIN_FREE", 10)),
        )
        sync_cmd: List[str] = [
            sys.executable,
            str(sync_script),
            "--data-dir",
            data_dir,
            "--history-days",
            str(sync_history_days),
            "--horizon-days",
            str(sync_horizon_days),
            "--max-requests",
            str(sync_max_requests),
            "--rate-per-minute",
            str(sync_rate_per_min),
        ]
        if bool(args.emit_picks):
            sync_cmd.append("--fetch-odds")
        else:
            sync_cmd.append("--no-fetch-odds")
        if sync_league_ids_csv:
            sync_cmd.extend(["--league-ids", sync_league_ids_csv])
        print(f"[api-football] syncing fixtures via {sync_script}")
        _run(sync_cmd, env=sync_env)

        try:
            api_context = _bridge_api_fixture_inputs(data_dir_path)
        except Exception as exc:
            raise SystemExit(f"[api-football] failed to bridge sync files: {exc}") from exc

        quality_ok, quality_messages = _evaluate_fixture_quality_gate(
            api_context["quality_report"],
            require_odds=bool(args.emit_picks),
        )
        if not quality_ok:
            print("[quality-gate] FAILED: coverage checks did not pass.")
            for msg in quality_messages:
                print(f"  - {msg}")
            raise SystemExit(3)
        print("[quality-gate] PASSED")
        for msg in quality_messages:
            print(f"  - {msg}")

    history_csv = enhanced_dir / "cgm_match_history.csv"
    elo_csv = enhanced_dir / "cgm_match_history_with_elo.csv"
    elo_stats_csv = enhanced_dir / "cgm_match_history_with_elo_stats.csv"
    elo_stats_xg_csv = enhanced_dir / "cgm_match_history_with_elo_stats_xg.csv"
    if api_context is not None:
        stats_source = api_context["history_fixtures"]
    else:
        stats_candidates = [
            data_dir_path / "multiple leagues and seasons" / "upcoming.csv",
            data_dir_path / "cgmbetdatabase.csv",
            data_dir_path / "cgmbetdatabase.xls",
            data_dir_path / "goals statistics.csv",
        ]
        stats_source = next((p for p in stats_candidates if p.exists()), stats_candidates[-1])

    if args.predict_only:
        history_for_predict = elo_stats_xg_csv if elo_stats_xg_csv.exists() else elo_stats_csv
        latest_data = _latest_file_mtime(data_dir_path)
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
        if bool(args.next_round_only):
            _pipeline_cmd_extra = ["--next-round-only"]
        else:
            _pipeline_cmd_extra = ["--all-upcoming"]
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
                "--next-round-span-days",
                str(max(0, int(args.next_round_span_days))),
                *_pipeline_cmd_extra,
                "--log-json",
                str(reports_dir / "run_log.jsonl"),
                "--trace-json",
                str(reports_dir / "elo_trace.jsonl"),
            ]
        )
        if args.emit_picks:
            _run_mandatory_pre_bet_gate(as_of_date=max_date, model_variant=model_variant)
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
        else:
            print("[predictions] emit-picks disabled; skipped pick engine and narrator.")
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
        latest_data = _latest_file_mtime(data_dir_path)
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

    # 6) Strict module engine has no separate training step.

    # 7) Predict upcoming (internal model probabilities)
    if bool(args.next_round_only):
        _pipeline_cmd_extra = ["--next-round-only"]
    else:
        _pipeline_cmd_extra = ["--all-upcoming"]
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
            "--next-round-span-days",
            str(max(0, int(args.next_round_span_days))),
            *_pipeline_cmd_extra,
            "--log-json",
            str(reports_dir / "run_log.jsonl"),
            "--trace-json",
            str(reports_dir / "elo_trace.jsonl"),
        ]
    )

    # 9) Optional picks/narrator (disabled by default for internal-model-only workflow)
    if args.emit_picks:
        _run_mandatory_pre_bet_gate(as_of_date=max_date, model_variant=model_variant)
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
    else:
        print("[predictions] emit-picks disabled; skipped pick engine and narrator.")

    # Print pipeline summary
    _print_pipeline_summary(reports_dir)
if __name__ == "__main__":
    main()
