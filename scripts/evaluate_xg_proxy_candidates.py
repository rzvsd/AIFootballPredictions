#!/usr/bin/env python3
"""
Compare two xG proxy outputs (baseline vs candidate) on the same match set.

Metrics (overall + per league):
- side_rmse: RMSE on side goals (home+away rows)
- total_rmse: RMSE on total goals per match
- side_poisson_nll: mean Poisson NLL without constant term
- side_brier_goal: Brier on P(goal>=1) where p = 1-exp(-mu)
- side_logloss_goal: log-loss on P(goal>=1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["league", "date", "home", "away", "ft_home", "ft_away", "xg_proxy_H", "xg_proxy_A"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing required columns: {miss}")
    return df


def _align(base: pd.DataFrame, cand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["league", "date", "home", "away"]
    b = base.copy()
    c = cand.copy()
    for col in key:
        b[col] = b[col].astype(str)
        c[col] = c[col].astype(str)
    merged = b[key].merge(c[key], on=key, how="inner").drop_duplicates()
    b2 = merged.merge(b, on=key, how="left")
    c2 = merged.merge(c, on=key, how="left")
    return b2, c2


def _side_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = np.concatenate(
        [
            pd.to_numeric(df["xg_proxy_H"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["xg_proxy_A"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    y = np.concatenate(
        [
            pd.to_numeric(df["ft_home"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["ft_away"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    ok = np.isfinite(mu) & np.isfinite(y)
    mu = np.clip(mu[ok], 1e-6, None)
    y = np.clip(y[ok], 0.0, None)
    return mu, y


def _safe_logloss(p: np.ndarray, o: np.ndarray) -> float:
    p2 = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    o2 = np.asarray(o, dtype=float)
    return float(-(o2 * np.log(p2) + (1.0 - o2) * np.log(1.0 - p2)).mean())


def _metrics(df: pd.DataFrame) -> Dict[str, float]:
    mu_side, y_side = _side_arrays(df)
    if len(mu_side) == 0:
        return {
            "n_matches": 0.0,
            "n_sides": 0.0,
            "side_rmse": np.nan,
            "total_rmse": np.nan,
            "side_poisson_nll": np.nan,
            "side_brier_goal": np.nan,
            "side_logloss_goal": np.nan,
        }

    side_rmse = float(np.sqrt(np.mean((mu_side - y_side) ** 2)))
    side_poisson_nll = float(np.mean(mu_side - y_side * np.log(mu_side)))

    p_goal = 1.0 - np.exp(-mu_side)
    o_goal = (y_side > 0).astype(float)
    side_brier_goal = float(np.mean((p_goal - o_goal) ** 2))
    side_logloss_goal = _safe_logloss(p_goal, o_goal)

    mu_total = pd.to_numeric(df["xg_proxy_H"], errors="coerce") + pd.to_numeric(df["xg_proxy_A"], errors="coerce")
    y_total = pd.to_numeric(df["ft_home"], errors="coerce") + pd.to_numeric(df["ft_away"], errors="coerce")
    ok_total = mu_total.notna() & y_total.notna()
    total_rmse = float(np.sqrt(np.mean((mu_total[ok_total].to_numpy(dtype=float) - y_total[ok_total].to_numpy(dtype=float)) ** 2))) if ok_total.any() else np.nan

    return {
        "n_matches": float(len(df)),
        "n_sides": float(len(mu_side)),
        "side_rmse": side_rmse,
        "total_rmse": total_rmse,
        "side_poisson_nll": side_poisson_nll,
        "side_brier_goal": side_brier_goal,
        "side_logloss_goal": side_logloss_goal,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline vs candidate xG proxy outputs")
    ap.add_argument("--baseline", required=True, help="Baseline CSV with xg_proxy_H/A")
    ap.add_argument("--candidate", required=True, help="Candidate CSV with xg_proxy_H/A")
    ap.add_argument("--out-prefix", default="reports/xg_proxy_compare", help="Output prefix (without extension)")
    ap.add_argument("--min-league-matches", type=int, default=30, help="Min matches for per-league table")
    args = ap.parse_args()

    b = _load(Path(args.baseline))
    c = _load(Path(args.candidate))
    b2, c2 = _align(b, c)

    overall_baseline = _metrics(b2)
    overall_candidate = _metrics(c2)

    rows = []
    metric_keys = [k for k in overall_baseline.keys() if k not in {"n_matches", "n_sides"}]
    for m in metric_keys:
        bv = overall_baseline.get(m, np.nan)
        cv = overall_candidate.get(m, np.nan)
        delta = cv - bv if np.isfinite(bv) and np.isfinite(cv) else np.nan
        rows.append({"scope": "OVERALL", "metric": m, "baseline": bv, "candidate": cv, "delta_candidate_minus_baseline": delta})

    per_league_rows = []
    leagues = sorted(set(b2["league"].astype(str).tolist()) & set(c2["league"].astype(str).tolist()))
    for lg in leagues:
        gb = b2[b2["league"].astype(str) == lg].copy()
        gc = c2[c2["league"].astype(str) == lg].copy()
        if len(gb) < int(args.min_league_matches) or len(gc) < int(args.min_league_matches):
            continue
        mb = _metrics(gb)
        mc = _metrics(gc)
        for m in metric_keys:
            bv = mb.get(m, np.nan)
            cv = mc.get(m, np.nan)
            delta = cv - bv if np.isfinite(bv) and np.isfinite(cv) else np.nan
            per_league_rows.append(
                {
                    "scope": str(lg),
                    "metric": m,
                    "baseline": bv,
                    "candidate": cv,
                    "delta_candidate_minus_baseline": delta,
                    "n_matches": len(gb),
                }
            )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_prefix.with_suffix(".csv")
    out_json = out_prefix.with_suffix(".json")

    out_df = pd.DataFrame(rows + per_league_rows)
    out_df.to_csv(out_csv, index=False)

    payload = {
        "baseline": str(Path(args.baseline)),
        "candidate": str(Path(args.candidate)),
        "aligned_matches": int(len(b2)),
        "overall_baseline": overall_baseline,
        "overall_candidate": overall_candidate,
        "rows_written": int(len(out_df)),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[ok] wrote comparison -> {out_csv}")
    print(f"[ok] wrote summary -> {out_json}")
    print("[overall]", payload["overall_baseline"])
    print("[overall_candidate]", payload["overall_candidate"])


if __name__ == "__main__":
    main()

