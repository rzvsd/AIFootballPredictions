"""
xG-proxy audits (coverage + no-leak tripwires).

Intended usage (repo root):
  python -m scripts.audit_xg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cgm.build_xg_proxy import LOG_PATH_DEFAULT
from cgm.xg_form import add_xg_form_features
from cgm.pressure_inputs import ensure_pressure_inputs


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        return out
    time = out.get("time", "").astype(str)
    dt = pd.to_datetime(out["date"].astype(str) + " " + time, errors="coerce")
    dt2 = pd.to_datetime(out["date"], errors="coerce")
    out["datetime"] = dt.fillna(dt2)
    return out


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    ap = argparse.ArgumentParser(description="xG-proxy audits (coverage + no-leak tripwires)")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv")
    ap.add_argument("--training", default="data/enhanced/frankenstein_training.csv")
    ap.add_argument("--window", type=int, default=10, help="xG rolling window used for shift checks")
    ap.add_argument("--sample-n", type=int, default=20, help="Random sample size for walk-forward leakage tripwire")
    ap.add_argument("--sample-seed", type=int, default=42, help="Seed for random sampling (reproducible audits)")
    ap.add_argument("--team", default="Liverpool", help="Team name for the pre-match rolling integrity timeline (home context)")
    ap.add_argument("--team-limit", type=int, default=12, help="Max rows to print in the team timeline")
    ap.add_argument("--country", default="England", help="League filter for coverage reality (default: England)")
    ap.add_argument("--league", default="Premier L", help="League filter for coverage reality (default: Premier L)")
    ap.add_argument("--season", default="2025-2026", help="Season filter for coverage reality (default: 2025-2026)")
    ap.add_argument("--log-json", default=str(LOG_PATH_DEFAULT), help="Optional run log to cross-check week train cutoffs")
    args = ap.parse_args()

    hist_path = Path(args.history)
    train_path = Path(args.training)

    hist = pd.read_csv(hist_path)
    hist = _ensure_datetime(hist)
    hist = ensure_pressure_inputs(hist)

    _print_header("Coverage (history)")
    for c in ["xg_proxy_H", "xg_proxy_A", "xg_usable"]:
        if c not in hist.columns:
            print("[warn] history missing:", c)
    if {"xg_proxy_H", "xg_proxy_A"}.issubset(hist.columns):
        usable = (pd.to_numeric(hist.get("xg_usable"), errors="coerce").fillna(0.0) > 0) if "xg_usable" in hist.columns else hist[["xg_proxy_H", "xg_proxy_A"]].notna().all(axis=1)
        cov = (
            usable.groupby([hist.get("country"), hist.get("league"), hist.get("season")], dropna=False)
            .mean()
            .sort_values(ascending=False)
        )
        print(cov.head(30))
        print("overall:", float(usable.mean()), f"({int(usable.sum())}/{len(hist)})")

    _print_header("Sanity distributions (xg_proxy)")
    if {"xg_proxy_H", "xg_proxy_A"}.issubset(hist.columns):
        xh = pd.to_numeric(hist["xg_proxy_H"], errors="coerce")
        xa = pd.to_numeric(hist["xg_proxy_A"], errors="coerce")
        both = xh.notna() & xa.notna()
        for name, s in [("xg_proxy_H", xh[both]), ("xg_proxy_A", xa[both])]:
            if s.empty:
                print(name, "[skip] no values")
            else:
                print(
                    name,
                    {
                        "min": float(s.min()),
                        "p50": float(s.median()),
                        "p95": float(s.quantile(0.95)),
                        "max": float(s.max()),
                        "neg_rate": float((s < 0).mean()),
                    },
                )

    _print_header("Leakage tripwire (walk-forward training cutoff)")
    hist0 = hist.copy()
    hist0["_week_start"] = (
        pd.to_datetime(hist0["datetime"], errors="coerce")
        - pd.to_timedelta(pd.to_datetime(hist0["datetime"], errors="coerce").dt.weekday, unit="D")
    ).dt.normalize()
    xg_ok = (pd.to_numeric(hist0.get("xg_usable"), errors="coerce").fillna(0.0) > 0) if "xg_usable" in hist0.columns else pd.Series(False, index=hist0.index)
    # Compute train_last_datetime per week deterministically from the history itself.
    week_meta = {}
    for ws in sorted(hist0["_week_start"].dropna().unique()):
        train_mask = (pd.to_datetime(hist0["datetime"], errors="coerce") < ws) & xg_ok
        train_last = pd.to_datetime(hist0.loc[train_mask, "datetime"], errors="coerce").max() if train_mask.any() else pd.NaT
        week_meta[str(pd.to_datetime(ws).date())] = {"train_last_datetime": None if pd.isna(train_last) else str(train_last)}

    # Cross-check with run log if present (should match the builder).
    log_path = Path(args.log_json)
    if log_path.exists():
        try:
            import json

            for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("event") == "XG_PROXY_WEEK":
                    ws = ev.get("week_start")
                    if ws and ws in week_meta:
                        week_meta[ws]["train_last_datetime_log"] = ev.get("train_last_datetime")
                        week_meta[ws]["ok_train_cutoff_log"] = ev.get("ok_train_cutoff")
        except Exception:
            pass

    # Sample N usable matches and assert their batch train_last_datetime < week_start(match).
    rng = np.random.default_rng(int(args.sample_seed))
    usable_rows = hist0[xg_ok & hist0["datetime"].notna()].copy()
    n = min(int(args.sample_n), len(usable_rows))
    if n > 0:
        sample = usable_rows.sample(n=n, random_state=int(args.sample_seed)).sort_values("datetime")
        out_rows = []
        ok_all = True
        for _, r in sample.iterrows():
            ws = str(pd.to_datetime(r["_week_start"]).date())
            train_last = week_meta.get(ws, {}).get("train_last_datetime")
            ok = True
            if train_last:
                ok = pd.to_datetime(train_last) < pd.to_datetime(r["_week_start"])
            out_rows.append(
                {
                    "match_dt": str(r["datetime"]),
                    "week_start": ws,
                    "train_last_datetime": train_last,
                    "ok_train_lt_week": bool(ok),
                    "home": r.get("home"),
                    "away": r.get("away"),
                }
            )
            ok_all = ok_all and bool(ok)
        print(f"sample_passed: {ok_all} ({sum(1 for r in out_rows if r['ok_train_lt_week'])}/{len(out_rows)}) seed={args.sample_seed}")
        print(pd.DataFrame(out_rows).to_string(index=False))
    else:
        print("[skip] no usable xG rows to sample")

    _print_header("Leakage tripwires (xG shift checks)")
    hist2 = add_xg_form_features(hist, window=int(args.window))

    def _shift_mismatch_rate(group_col: str, pre_col: str, post_col: str) -> tuple[int, int, float, float]:
        g = hist2.sort_values("datetime").copy()
        shifted = g.groupby(group_col, group_keys=False)[post_col].shift(1)
        mask = shifted.notna()
        diff = (pd.to_numeric(g[pre_col], errors="coerce") - pd.to_numeric(shifted, errors="coerce")).abs()
        diff = diff.where(mask)
        bad = diff > 1e-12
        return (int(bad.sum()), int(mask.sum()), float(bad.mean()), float(diff.max() if mask.any() else 0.0))

    checks = [
        ("home", "xg_for_form_H", "_xg_for_form_H_post"),
        ("home", "xg_against_form_H", "_xg_against_form_H_post"),
        ("home", "xg_diff_form_H", "_xg_diff_form_H_post"),
        ("home", "xg_shot_quality_form_H", "_xg_shot_quality_form_H_post"),
        ("home", "xg_finishing_luck_form_H", "_xg_finishing_luck_form_H_post"),
        ("home", "xg_n_H", "_xg_n_H_post"),
        ("home", "xg_stats_n_H", "_xg_stats_n_H_post"),
        ("away", "xg_for_form_A", "_xg_for_form_A_post"),
        ("away", "xg_against_form_A", "_xg_against_form_A_post"),
        ("away", "xg_diff_form_A", "_xg_diff_form_A_post"),
        ("away", "xg_shot_quality_form_A", "_xg_shot_quality_form_A_post"),
        ("away", "xg_finishing_luck_form_A", "_xg_finishing_luck_form_A_post"),
        ("away", "xg_n_A", "_xg_n_A_post"),
        ("away", "xg_stats_n_A", "_xg_stats_n_A_post"),
    ]
    for group_col, pre_col, post_col in checks:
        if pre_col in hist2.columns and post_col in hist2.columns:
            bad, n, rate, max_diff = _shift_mismatch_rate(group_col, pre_col, post_col)
            print(f"{group_col:>4}  {pre_col:>22} == shift({post_col:>24}): bad={bad}/{n} rate={rate:.6f} max_abs_diff={max_diff:.3g}")
        else:
            print(f"[skip] missing columns for {pre_col}/{post_col}")

    _print_header("Leakage tripwires (training matrix)")
    if train_path.exists():
        train = pd.read_csv(train_path)
        leaked = [c for c in train.columns if c.startswith("_xg_") or c in {"xg_proxy_H", "xg_proxy_A", "xg_usable"}]
        print("training_cols:", len(train.columns), "leaked_xg_cols:", len(leaked))
        if leaked:
            print("leaked:", leaked[:25], "..." if len(leaked) > 25 else "")
    else:
        print("[skip] training not found:", train_path)

    _print_header("Pre-match rolling integrity (team timeline)")
    team = str(args.team)
    team_rows = hist2[(hist2.get("home") == team) & (pd.to_numeric(hist2.get("xg_usable"), errors="coerce").fillna(0.0) > 0)].sort_values("datetime").copy()
    if team_rows.empty:
        print(f"[skip] no usable home-context xG rows found for team={team!r}")
    else:
        team_rows["expected_xg_for_form_H"] = pd.to_numeric(team_rows["xg_proxy_H"], errors="coerce").shift().rolling(int(args.window), min_periods=1).mean()
        team_rows["abs_diff"] = (pd.to_numeric(team_rows["xg_for_form_H"], errors="coerce") - pd.to_numeric(team_rows["expected_xg_for_form_H"], errors="coerce")).abs()
        bad = int((team_rows["abs_diff"] > 1e-12).sum())
        max_diff = float(team_rows["abs_diff"].max())
        print(f"team={team} home_rows={len(team_rows)} max_abs_diff={max_diff:.3g} bad_rows={bad}")
        cols = ["date", "home", "away", "xg_proxy_H", "xg_for_form_H", "expected_xg_for_form_H", "abs_diff"]
        print(team_rows[cols].head(int(args.team_limit)).to_string(index=False))

    _print_header("SAFE artifact cleanliness (banned columns)")
    if train_path.exists():
        train = pd.read_csv(train_path)
        banned = {
            # Raw per-match truth/current-match stats
            "shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A",
            "shots", "shots_on_target", "corners", "possession_home", "possession_away",
            "ft_home", "ft_away", "ht_home", "ht_away", "result", "validated",
            # xG proxy per-match (must not be in SAFE)
            "xg_proxy_H", "xg_proxy_A", "xg_usable",
        }
        found = sorted([c for c in train.columns if c in banned])
        print("found_banned_cols:", len(found))
        if found:
            print(found)
        print("targets_present:", all(c in train.columns for c in ("y_home", "y_away")))

    _print_header("Coverage reality (league+season)")
    sub = hist2.copy()
    if "country" in sub.columns:
        sub = sub[sub["country"] == str(args.country)]
    if "league" in sub.columns:
        sub = sub[sub["league"] == str(args.league)]
    if "season" in sub.columns:
        sub = sub[sub["season"] == str(args.season)]
    usable = pd.to_numeric(sub.get("xg_usable"), errors="coerce").fillna(0.0) > 0
    print("rows:", len(sub), "xg_usable_rate:", float(usable.mean()) if len(sub) else 0.0)
    if len(sub) and usable.any():
        for col in ["xg_stats_n_H", "xg_stats_n_A"]:
            s = pd.to_numeric(sub.loc[usable, col], errors="coerce").fillna(0.0)
            counts = s.value_counts().sort_index().to_dict()
            print(col, {"min": float(s.min()), "p50": float(s.median()), "p90": float(s.quantile(0.9)), "max": float(s.max()),
                       "pct<=1": float((s <= 1).mean()), "pct>=3": float((s >= 3).mean())})
            print("counts:", counts)


if __name__ == "__main__":
    main()
