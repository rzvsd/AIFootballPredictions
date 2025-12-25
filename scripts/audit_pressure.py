"""
Pressure Cooker audits (coverage + no-leak tripwires).

Intended usage (repo root):
  python -m scripts.audit_pressure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cgm.backfill_match_stats import STATS_COLS, _parse_date_only, _read_table
from cgm.pressure_form import add_pressure_form_features
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


def _dom_ratio(for_s: pd.Series, against_s: pd.Series) -> pd.Series:
    f = pd.to_numeric(for_s, errors="coerce")
    a = pd.to_numeric(against_s, errors="coerce")
    den = f + a
    out = f / den
    out = out.where(den > 0)
    return out.clip(0.0, 1.0)


def _poss_share(pos_s: pd.Series) -> pd.Series:
    v = pd.to_numeric(pos_s, errors="coerce")
    out = v.where(v <= 1.0, v / 100.0)
    return out.clip(0.0, 1.0)


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pressure Cooker audits (coverage + no-leak tripwires)")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv")
    ap.add_argument("--training", default="data/enhanced/frankenstein_training.csv")
    ap.add_argument("--training-full", default="data/enhanced/frankenstein_training_full.csv")
    ap.add_argument("--stats", default="CGM data/cgmbetdatabase.xls")
    ap.add_argument("--cutoff", default=None, help="YYYY-MM-DD for played/future split; defaults to history max date")
    ap.add_argument("--window", type=int, default=10, help="Pressure rolling window used for shift checks")
    args = ap.parse_args()

    hist_path = Path(args.history)
    train_path = Path(args.training)
    train_full_path = Path(args.training_full)
    stats_path = Path(args.stats)
    if not stats_path.exists():
        # Fallback to the older default if a richer stats export is not present.
        for cand in [
            Path("CGM data") / "cgmbetdatabase.csv",
            Path("CGM data") / "goals statistics.csv",
        ]:
            if cand.exists():
                stats_path = cand
                break

    hist = pd.read_csv(hist_path)
    hist = _ensure_datetime(hist)

    _print_header("Coverage (history)")
    missing_cols = [c for c in STATS_COLS if c not in hist.columns]
    if missing_cols:
        print("[warn] history is missing stat cols:", missing_cols)
    else:
        hist["_stats_all_present"] = hist[STATS_COLS].notna().all(axis=1)
        cov = (
            hist.groupby(["country", "league", "season"], dropna=False)["_stats_all_present"]
            .mean()
            .sort_values(ascending=False)
        )
        print(cov.head(30))
        print("overall:", float(hist["_stats_all_present"].mean()), f"({int(hist['_stats_all_present'].sum())}/{len(hist)})")

    _print_header("Raw stats source (goals statistics.csv)")
    raw = _read_table(stats_path)
    print("rows:", len(raw), "cols:", len(raw.columns))
    if "sezonul" in raw.columns:
        uniq = raw["sezonul"].dropna().unique().tolist()
        print("unique sezonul:", sorted(uniq)[:25])
    if "datameci" in raw.columns:
        raw["_date_only"] = _parse_date_only(raw["datameci"], label=stats_path.name)
        cutoff = pd.to_datetime(args.cutoff).date() if args.cutoff else hist["datetime"].dt.date.max()
        played = (raw["_date_only"].notna() & (raw["_date_only"] <= cutoff)).sum()
        future = (raw["_date_only"].notna() & (raw["_date_only"] > cutoff)).sum()
        unparsable = raw["_date_only"].isna().sum()
        valid_dates = raw["_date_only"].dropna()
        date_min = valid_dates.min() if not valid_dates.empty else None
        date_max = valid_dates.max() if not valid_dates.empty else None
        print("date_min:", date_min, "date_max:", date_max)
        print({"cutoff": str(cutoff), "played_so_far": int(played), "future": int(future), "unparsable": int(unparsable)})

    _print_header("ensure_pressure_inputs overwrite audit (history)")
    if all(c in hist.columns for c in STATS_COLS):
        before = hist[STATS_COLS].copy()
        after = ensure_pressure_inputs(hist)[STATS_COLS]
        changed = (before.fillna(-9999) != after.fillna(-9999)).any(axis=1)
        print("rows_changed:", int(changed.sum()), "/", len(hist), "rate:", float(changed.mean()))
    else:
        print("[skip] missing required stat cols in history")

    _print_header("Leakage tripwires (Pressure shift checks)")
    # Recompute pressure features and validate the pre-match vs post-match shift identity:
    # pre(t) == post(t-1) within the same team+venue stream.
    hist2 = ensure_pressure_inputs(hist)
    hist2 = add_pressure_form_features(hist2, window=int(args.window))

    def _shift_mismatch_rate(group_col: str, pre_col: str, post_col: str) -> tuple[int, int, float, float]:
        g = hist2.sort_values("datetime").copy()
        shifted = g.groupby(group_col, group_keys=False)[post_col].shift(1)
        mask = shifted.notna()
        diff = (pd.to_numeric(g[pre_col], errors="coerce") - pd.to_numeric(shifted, errors="coerce")).abs()
        diff = diff.where(mask)
        bad = diff > 1e-12
        return (int(bad.sum()), int(mask.sum()), float(bad.mean()), float(diff.max() if mask.any() else 0.0))

    checks = [
        ("home", "press_form_H", "_press_form_H_post"),
        ("away", "press_form_A", "_press_form_A_post"),
        ("home", "press_n_H", "_press_n_H_post"),
        ("away", "press_n_A", "_press_n_A_post"),
        ("home", "press_dom_shots_H", "_press_dom_shots_H_post"),
        ("away", "press_dom_shots_A", "_press_dom_shots_A_post"),
        ("home", "press_dom_sot_H", "_press_dom_sot_H_post"),
        ("away", "press_dom_sot_A", "_press_dom_sot_A_post"),
        ("home", "press_dom_corners_H", "_press_dom_corners_H_post"),
        ("away", "press_dom_corners_A", "_press_dom_corners_A_post"),
        ("home", "press_dom_pos_H", "_press_dom_pos_H_post"),
        ("away", "press_dom_pos_A", "_press_dom_pos_A_post"),
        ("home", "press_stats_n_H", "_press_stats_n_H_post"),
        ("away", "press_stats_n_A", "_press_stats_n_A_post"),
    ]
    for group_col, pre_col, post_col in checks:
        if pre_col in hist2.columns and post_col in hist2.columns:
            bad, n, rate, max_diff = _shift_mismatch_rate(group_col, pre_col, post_col)
            print(f"{group_col:>4}  {pre_col:>18} == shift({post_col:>20}): bad={bad}/{n} rate={rate:.6f} max_abs_diff={max_diff:.3g}")
        else:
            print(f"[skip] missing columns for {pre_col}/{post_col}")

    _print_header("Leakage tripwires (training matrix)")
    if train_path.exists():
        train = pd.read_csv(train_path)
        leaked = [c for c in train.columns if c.startswith("_press_")]
        print("training_cols:", len(train.columns), "leaked__press_cols:", len(leaked))
        if leaked:
            print("leaked:", leaked[:25], "..." if len(leaked) > 25 else "")
    else:
        print("[skip] training not found:", train_path)

    # "Fingerprint" checks (avoid false positives by excluding y==0)
    if train_full_path.exists():
        df = pd.read_csv(train_full_path)
        targets = [c for c in ("ft_home", "ft_away", "y_home", "y_away") if c in df.columns]
        feats = [c for c in ("press_form_H", "press_form_A", "div_team_H", "div_team_A", "div_diff") if c in df.columns]
        print("targets:", targets)
        print("feats:", feats)
        for t in targets:
            y = pd.to_numeric(df[t], errors="coerce")
            nz = y.notna() & (y != 0)
            for f in feats:
                x = pd.to_numeric(df[f], errors="coerce")
                eq = (x == y) & nz
                rate = float(eq.mean())
                if rate > 0:
                    print(f"[warn] equality_nonzero: {f} == {t} rate={rate:.6g} (count={int(eq.sum())})")
    else:
        print("[skip] training_full not found:", train_full_path)

    _print_header("Leakage tripwires (current-match raw stats reconstruction)")
    if all(c in hist2.columns for c in STATS_COLS) and "press_dom_shots_H" in hist2.columns:
        has = hist2[STATS_COLS].notna().all(axis=1)
        h = hist2.loc[has, :].copy()
        raw_dom_shots_h = _dom_ratio(h["shots_H"], h["shots_A"]).fillna(0.5)
        raw_dom_sot_h = _dom_ratio(h["sot_H"], h["sot_A"]).fillna(0.5)
        raw_dom_cor_h = _dom_ratio(h["corners_H"], h["corners_A"]).fillna(0.5)
        raw_pos_h = _poss_share(h["pos_H"]).fillna(0.5)

        def _close_rate(feature: str, raw: pd.Series) -> None:
            x = pd.to_numeric(h[feature], errors="coerce")
            # Exclude the neutral point (0.5) to avoid symmetric-match false positives.
            mask = raw.notna() & (raw.sub(0.5).abs() > 0.02)
            close = np.isclose(x[mask].to_numpy(), raw[mask].to_numpy(), atol=1e-12, rtol=0.0)
            print(f"{feature:>18} ~= current_raw  rate={float(close.mean() if close.size else 0.0):.6f} (n={int(close.size)})")

        _close_rate("press_dom_shots_H", raw_dom_shots_h)
        _close_rate("press_dom_sot_H", raw_dom_sot_h)
        _close_rate("press_dom_corners_H", raw_dom_cor_h)
        _close_rate("press_dom_pos_H", raw_pos_h)
    else:
        print("[skip] missing stat columns or pressure features in history")


if __name__ == "__main__":
    main()
