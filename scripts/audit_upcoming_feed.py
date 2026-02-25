"""
Upcoming feed audit (OTM-style scope proof).

Reads:
  - raw upcoming feed (default: data/api_football/upcoming_fixtures.csv)
  - filtered predictions output (default: reports/cgm_upcoming_predictions.csv)

Prints:
  - raw row counts and date range
  - counts dropped by each deterministic scope filter
  - filtered output count + date range
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config
from cgm.league_registry import canonicalize_df_league_country, canonicalize_scope


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _to_naive_ts(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    try:
        return ts.tz_localize(None)
    except Exception:
        return pd.to_datetime(ts, errors="coerce")


def _parse_upcoming_datetime(row: pd.Series) -> pd.Timestamp:
    if "fixture_datetime_utc" in row:
        dt = _to_naive_ts(row.get("fixture_datetime_utc"))
        if not pd.isna(dt):
            return dt
    if "fixture_datetime" in row:
        dt = _to_naive_ts(row.get("fixture_datetime"))
        if not pd.isna(dt):
            return dt

    date_raw = row.get("datameci") if "datameci" in row else row.get("date")
    date_raw = "" if date_raw is None else str(date_raw)
    dt = pd.to_datetime(date_raw, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(date_raw, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return pd.NaT

    orameci = row.get("orameci") if "orameci" in row else row.get("time")
    hour = 0
    minute = 0
    try:
        t = int(float(orameci)) if orameci is not None and str(orameci) != "nan" else 0
        hour = max(0, min(23, t // 100))
        minute = max(0, min(59, t % 100))
    except Exception:
        hour = 0
        minute = 0

    return pd.to_datetime(dt.normalize() + pd.Timedelta(hours=hour, minutes=minute), errors="coerce")


def _resolve_asof(args: argparse.Namespace, pred_path: Path, log_path: Path) -> tuple[pd.Timestamp, str]:
    if args.as_of_date:
        ts = pd.to_datetime(args.as_of_date, errors="coerce")
        if not pd.isna(ts):
            d = ts.date()
            run_asof = pd.Timestamp(d).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            return _to_naive_ts(run_asof), "cli"

    if pred_path.exists():
        try:
            pred = pd.read_csv(pred_path)
            if not pred.empty and "run_asof_datetime" in pred.columns:
                uniq = pd.to_datetime(pred["run_asof_datetime"], errors="coerce").dropna().unique()
                if len(uniq) == 1:
                    return _to_naive_ts(uniq[0]), "predictions"
        except Exception:
            pass

    if log_path.exists():
        try:
            last = None
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("event") == "UPCOMING_SCOPE":
                    last = obj
            if last and last.get("run_asof_datetime"):
                ts = pd.to_datetime(last["run_asof_datetime"], errors="coerce")
                if not pd.isna(ts):
                    return _to_naive_ts(ts), "run_log"
        except Exception:
            pass

    # Fallback: today UTC (only when nothing else is available).
    d = pd.Timestamp.utcnow().date()
    run_asof = pd.Timestamp(d).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return _to_naive_ts(run_asof), "utc_today_fallback"


def main() -> None:
    ap = argparse.ArgumentParser(description="Upcoming feed audit (scope filtering proof)")
    ap.add_argument("--raw", default="data/api_football/upcoming_fixtures.csv", help="Raw upcoming feed CSV")
    ap.add_argument("--predictions", default="reports/cgm_upcoming_predictions.csv", help="Filtered predictions CSV")
    ap.add_argument("--log-jsonl", default="reports/run_log.jsonl", help="Run log JSONL (optional)")

    ap.add_argument("--as-of-date", default=None, help="As-of date (YYYY-MM-DD). If omitted, inferred.")
    ap.add_argument("--scope-country", default=getattr(config, "LIVE_SCOPE_COUNTRY", ""), help="Optional country filter (empty disables).")
    ap.add_argument("--scope-league", default=getattr(config, "LIVE_SCOPE_LEAGUE", ""), help="Optional league filter (empty disables).")
    ap.add_argument("--scope-season-start", default=getattr(config, "LIVE_SCOPE_SEASON_START", ""), help="Optional season window start (YYYY-MM-DD).")
    ap.add_argument("--scope-season-end", default=getattr(config, "LIVE_SCOPE_SEASON_END", ""), help="Optional season window end (YYYY-MM-DD).")
    ap.add_argument("--horizon-days", type=int, default=int(getattr(config, "LIVE_SCOPE_HORIZON_DAYS", 0) or 0), help="Optional horizon (days). 0 disables.")
    ap.add_argument("--next-round-only", dest="next_round_only", action="store_true", help="Keep only fixtures in the next round window.")
    ap.add_argument("--all-future", dest="next_round_only", action="store_false", help="Disable next-round-only filter.")
    ap.add_argument("--next-round-span-days", type=int, default=int(getattr(config, "LIVE_SCOPE_NEXT_ROUND_SPAN_DAYS", 3) or 3), help="Round window size in days when --next-round-only is active.")
    ap.set_defaults(next_round_only=bool(getattr(config, "LIVE_SCOPE_NEXT_ROUND_ONLY", True)))
    args = ap.parse_args()

    raw_path = Path(args.raw)
    pred_path = Path(args.predictions)
    log_path = Path(args.log_jsonl)

    if not raw_path.exists():
        raise SystemExit(f"[audit_upcoming_feed] raw feed not found: {raw_path}")

    run_asof_dt, asof_source = _resolve_asof(args, pred_path, log_path)

    scope_country = str(args.scope_country or "").strip() or None
    scope_league = str(args.scope_league or "").strip() or None
    scope_country, scope_league = canonicalize_scope(scope_country, scope_league)
    ss = pd.to_datetime(str(args.scope_season_start or ""), errors="coerce")
    se = pd.to_datetime(str(args.scope_season_end or ""), errors="coerce")
    scope_season_start = ss.normalize() if not pd.isna(ss) else None
    scope_season_end = se.normalize() if not pd.isna(se) else None
    horizon_days = int(args.horizon_days or 0)
    if horizon_days <= 0:
        horizon_days = 0
    next_round_only = bool(getattr(args, "next_round_only", False))
    next_round_span_days = int(getattr(args, "next_round_span_days", 3) or 3)
    if next_round_span_days < 0:
        next_round_span_days = 0

    _print_header("Raw upcoming feed")
    raw = pd.read_csv(raw_path, sep=None, engine="python")
    raw = canonicalize_df_league_country(raw, league_col="league", country_col="country")
    raw["_fixture_dt"] = raw.apply(_parse_upcoming_datetime, axis=1)
    print("rows_in:", len(raw), "parsed:", int(raw["_fixture_dt"].notna().sum()))
    if raw["_fixture_dt"].notna().any():
        print("date_range:", str(raw["_fixture_dt"].min()), "->", str(raw["_fixture_dt"].max()))
    if "league" in raw.columns:
        print("\nby league:")
        print(raw["league"].astype(str).value_counts().head(20).to_string())
    if "sezonul" in raw.columns:
        print("\nby sezonul:")
        print(raw["sezonul"].astype(str).value_counts().head(20).to_string())

    _print_header("Scope filters")
    print("run_asof_datetime:", str(run_asof_dt), f"(source={asof_source})")
    print("scope_country:", scope_country)
    print("scope_league:", scope_league)
    print("scope_season_start:", str(scope_season_start) if scope_season_start is not None else None)
    print("scope_season_end:", str(scope_season_end) if scope_season_end is not None else None)
    print("horizon_days:", horizon_days)
    print("next_round_only:", next_round_only)
    print("next_round_span_days:", next_round_span_days)

    scoped = raw[raw["_fixture_dt"].notna()].copy()
    print("after_parse:", len(scoped))

    scoped = scoped[scoped["_fixture_dt"] > run_asof_dt].copy()
    print("after_drop_past:", len(scoped))

    if scope_season_start is not None and scope_season_end is not None:
        scoped = scoped[scoped["_fixture_dt"].between(scope_season_start, scope_season_end, inclusive="left")].copy()
    print("after_window:", len(scoped))

    if scope_league and "league" in scoped.columns:
        scoped = scoped[scoped["league"].astype(str) == scope_league].copy()
    if scope_country and "country" in scoped.columns:
        scoped = scoped[scoped["country"].astype(str) == scope_country].copy()
    print("after_league_country:", len(scoped))

    if horizon_days > 0:
        scoped = scoped[scoped["_fixture_dt"] <= (run_asof_dt + pd.Timedelta(days=horizon_days))].copy()
    print("after_horizon:", len(scoped))

    # Dedupe fixtures (mirror predict_upcoming.py behavior).
    dedupe_cols = ["_fixture_dt"]
    if "league" in scoped.columns:
        dedupe_cols.append("league")
    if {"codechipa1", "codechipa2"}.issubset(scoped.columns):
        dedupe_cols.extend(["codechipa1", "codechipa2"])
    elif {"txtechipa1", "txtechipa2"}.issubset(scoped.columns):
        dedupe_cols.extend(["txtechipa1", "txtechipa2"])
    if len(dedupe_cols) > 1:
        score_cols = [c for c in ["cotaa", "cotae", "cotad", "cotao", "cotau", "gg", "ng"] if c in scoped.columns]
        if score_cols:
            scoped["_dupe_score"] = scoped[score_cols].notna().sum(axis=1)
            scoped = scoped.sort_values(dedupe_cols + ["_dupe_score"], ascending=[True] * len(dedupe_cols) + [False], kind="mergesort")
        scoped = scoped.drop_duplicates(subset=dedupe_cols, keep="first")
        scoped.drop(columns=["_dupe_score"], inplace=True, errors="ignore")
    print("after_dedupe:", len(scoped))

    if next_round_only and not scoped.empty:
        first_fixture_dt = pd.to_datetime(scoped["_fixture_dt"], errors="coerce").min()
        if not pd.isna(first_fixture_dt):
            round_end = first_fixture_dt + pd.Timedelta(days=int(next_round_span_days))
            scoped = scoped[scoped["_fixture_dt"] <= round_end].copy()
    print("after_next_round:", len(scoped))

    _print_header("Filtered predictions output")
    if not pred_path.exists():
        print("[warn] predictions not found:", pred_path)
        return
    pred = pd.read_csv(pred_path)
    if pred.empty:
        print("predictions: empty (0 rows)")
        return

    pred_dt = pd.to_datetime(pred["fixture_datetime"], errors="coerce")
    print("rows:", len(pred), "dt_range:", str(pred_dt.min()), "->", str(pred_dt.max()))
    if "run_asof_datetime" in pred.columns:
        uniq = pd.to_datetime(pred["run_asof_datetime"], errors="coerce").dropna().unique()
        if len(uniq) == 1:
            print("run_asof_datetime:", str(pd.to_datetime(uniq[0])))
    print("expected_rows_after_filters:", len(scoped))
    if len(pred) != len(scoped):
        print("[warn] predictions row count does not match scoped raw feed count")


if __name__ == "__main__":
    main()

