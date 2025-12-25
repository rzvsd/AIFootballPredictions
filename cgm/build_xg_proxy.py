"""
Milestone 3: xG-proxy "Sniper" engine (leakage-safe).

Builds a per-match xG proxy from shots + shots on target, then writes an enriched
history artifact that downstream feature builders can use to compute pre-match
rolling xG form.

Key properties
- No "true xG" is assumed (no shot locations); this is a proxy model.
- Walk-forward integrity: for matches in ISO week W, the proxy model is trained
  only on matches strictly before W (no within-week or future leakage).
- Symmetry: the same model predicts goals-for for both home and away sides using
  the same feature schema and an `is_home` flag.

Input
- data/enhanced/cgm_match_history_with_elo_stats.csv

Output
- data/enhanced/cgm_match_history_with_elo_stats_xg.csv
  Adds:
    xg_proxy_H, xg_proxy_A, xg_usable
    shot_quality_H/A, finishing_luck_H/A
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cgm.pressure_inputs import ensure_pressure_inputs


FEATURE_COLS = ["shots_for", "sot_for", "shots_against", "sot_against", "is_home"]
LOG_PATH_DEFAULT = Path("reports/run_log.jsonl")


def log_json(event: dict, log_path: Path = LOG_PATH_DEFAULT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = event.copy()
    out.setdefault("ts", pd.Timestamp.utcnow().isoformat())
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        if out["datetime"].notna().any():
            return out
    dt = pd.to_datetime(out["date"] + " " + out.get("time", "").astype(str), errors="coerce")
    dt2 = pd.to_datetime(out["date"], errors="coerce")
    out["datetime"] = pd.to_datetime(dt.fillna(dt2), errors="coerce")
    return out


def _week_start(dt: pd.Series) -> pd.Series:
    # ISO week start (Monday), normalized to midnight.
    dtn = pd.to_datetime(dt, errors="coerce")
    return (dtn - pd.to_timedelta(dtn.dt.weekday, unit="D")).dt.normalize()


def _xg_usable_mask(df: pd.DataFrame) -> pd.Series:
    required = ["shots_H", "shots_A", "sot_H", "sot_A"]
    for c in required:
        if c not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
    x = df[required].apply(pd.to_numeric, errors="coerce")
    ok = x.notna().all(axis=1)
    # Defensive: disallow negative values.
    ok &= (x >= 0).all(axis=1)
    return ok


def _stack_sides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match rows -> side rows (two per match) for symmetric modeling.

    Columns:
      match_idx, is_home, goals_for, shots_for, sot_for, shots_against, sot_against
    """
    out_rows = []
    for idx, r in df.iterrows():
        out_rows.append(
            {
                "match_idx": idx,
                "is_home": 1.0,
                "goals_for": r.get("ft_home"),
                "shots_for": r.get("shots_H"),
                "sot_for": r.get("sot_H"),
                "shots_against": r.get("shots_A"),
                "sot_against": r.get("sot_A"),
            }
        )
        out_rows.append(
            {
                "match_idx": idx,
                "is_home": 0.0,
                "goals_for": r.get("ft_away"),
                "shots_for": r.get("shots_A"),
                "sot_for": r.get("sot_A"),
                "shots_against": r.get("shots_H"),
                "sot_against": r.get("sot_H"),
            }
        )
    out = pd.DataFrame(out_rows)
    for c in ["goals_for"] + FEATURE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_xg_proxy_history(
    history_path: str | Path = "data/enhanced/cgm_match_history_with_elo_stats.csv",
    out_path: str | Path = "data/enhanced/cgm_match_history_with_elo_stats_xg.csv",
    *,
    min_train_rows: int = 30,
    prior_mu: float = 1.25,
    log_json_path: str | Path = LOG_PATH_DEFAULT,
    verbose: bool = True,
) -> Path:
    hist_p = Path(history_path)
    if not hist_p.exists():
        raise FileNotFoundError(f"{hist_p} not found (run stats backfill first).")
    df = pd.read_csv(hist_p)
    df = _ensure_datetime(df)
    df = ensure_pressure_inputs(df)

    # Numeric coercion for required columns.
    for c in ["ft_home", "ft_away", "shots_H", "shots_A", "sot_H", "sot_A"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)

    xg_usable = _xg_usable_mask(df)
    df["xg_usable"] = xg_usable.astype(int)
    df["week_start"] = _week_start(df["datetime"])

    # Allocate outputs (NaN by default).
    df["xg_proxy_H"] = np.nan
    df["xg_proxy_A"] = np.nan

    # Walk-forward per week.
    prev_model: Pipeline | None = None
    fit_weeks = 0
    used_prev_weeks = 0
    used_prior_weeks = 0
    train_sizes: list[int] = []
    log_path = Path(log_json_path) if log_json_path else LOG_PATH_DEFAULT

    for ws in sorted(df["week_start"].dropna().unique()):
        batch_mask = df["week_start"] == ws
        train_mask = (df["datetime"] < ws) & (df["xg_usable"] == 1)

        train_df = df.loc[train_mask, :]
        batch_df = df.loc[batch_mask & (df["xg_usable"] == 1), :]

        # If no usable rows in the batch, nothing to predict.
        if batch_df.empty:
            continue

        stacked_train = _stack_sides(train_df) if not train_df.empty else pd.DataFrame(columns=["goals_for"] + FEATURE_COLS)
        stacked_train = stacked_train.dropna(subset=["goals_for"] + FEATURE_COLS, how="any")

        model: Pipeline | None = None
        used_mode = "prior"
        if len(stacked_train) >= int(min_train_rows):
            X_tr = stacked_train[FEATURE_COLS].to_numpy(dtype=float)
            y_tr = stacked_train["goals_for"].to_numpy(dtype=float)
            y_tr = np.clip(y_tr, 0.0, None)
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("poisson", PoissonRegressor(alpha=0.1, max_iter=2000)),
                ]
            )
            model.fit(X_tr, y_tr)
            prev_model = model
            fit_weeks += 1
            train_sizes.append(int(len(stacked_train)))
            used_mode = "fit"
        elif prev_model is not None:
            model = prev_model
            used_prev_weeks += 1
            used_mode = "prev"
        else:
            model = None
            used_prior_weeks += 1
            used_mode = "prior"

        stacked_batch = _stack_sides(batch_df)
        stacked_batch = stacked_batch.dropna(subset=FEATURE_COLS, how="any")
        if stacked_batch.empty:
            continue

        train_last_dt = None
        try:
            train_last_dt = pd.to_datetime(train_df["datetime"], errors="coerce").max() if not train_df.empty else None
        except Exception:
            train_last_dt = None
        ok_cutoff = None
        if train_last_dt is not None and not pd.isna(train_last_dt):
            ok_cutoff = bool(train_last_dt < pd.to_datetime(ws))

        # Audit log line (non-negotiable for leakage checks): training max datetime per batch/week.
        log_json(
            {
                "event": "XG_PROXY_WEEK",
                "week_start": str(pd.to_datetime(ws).date()) if ws is not None else None,
                "train_last_datetime": str(train_last_dt) if train_last_dt is not None and not pd.isna(train_last_dt) else None,
                "ok_train_cutoff": ok_cutoff,
                "used_mode": used_mode,
                "train_matches": int(len(train_df)),
                "train_side_rows": int(len(stacked_train)),
                "batch_matches": int(len(batch_df)),
                "batch_side_rows": int(len(stacked_batch)),
                "min_train_side_rows": int(min_train_rows),
            },
            log_path=log_path,
        )

        if model is None:
            # Prior fallback: use historical means if available, else fixed.
            mu_home = float(prior_mu)
            mu_away = float(prior_mu)
            if not stacked_train.empty:
                try:
                    mu_home = float(stacked_train.loc[stacked_train["is_home"] == 1.0, "goals_for"].mean())
                    mu_away = float(stacked_train.loc[stacked_train["is_home"] == 0.0, "goals_for"].mean())
                    if np.isnan(mu_home):
                        mu_home = float(prior_mu)
                    if np.isnan(mu_away):
                        mu_away = float(prior_mu)
                except Exception:
                    mu_home = float(prior_mu)
                    mu_away = float(prior_mu)
            preds = np.where(stacked_batch["is_home"].to_numpy(dtype=float) == 1.0, mu_home, mu_away)
        else:
            X_b = stacked_batch[FEATURE_COLS].to_numpy(dtype=float)
            preds = model.predict(X_b)
            preds = np.clip(preds, 0.0, None)

        # Assign back to match rows.
        for (match_idx, is_home), mu in zip(stacked_batch[["match_idx", "is_home"]].itertuples(index=False, name=None), preds.tolist()):
            if int(is_home) == 1:
                df.loc[int(match_idx), "xg_proxy_H"] = float(mu)
            else:
                df.loc[int(match_idx), "xg_proxy_A"] = float(mu)

    # Helpers (per-match; NOT safe as model features)
    def _safe_div(num_s: pd.Series, den_s: pd.Series) -> pd.Series:
        num = pd.to_numeric(num_s, errors="coerce")
        den = pd.to_numeric(den_s, errors="coerce")
        den2 = den.where(den > 0, 1.0)
        return (num / den2).replace([np.inf, -np.inf], np.nan)

    df["shot_quality_H"] = _safe_div(df["xg_proxy_H"], df["shots_H"])
    df["shot_quality_A"] = _safe_div(df["xg_proxy_A"], df["shots_A"])
    df["finishing_luck_H"] = pd.to_numeric(df["ft_home"], errors="coerce") - pd.to_numeric(df["xg_proxy_H"], errors="coerce")
    df["finishing_luck_A"] = pd.to_numeric(df["ft_away"], errors="coerce") - pd.to_numeric(df["xg_proxy_A"], errors="coerce")

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=["week_start"], errors="ignore")
    df.to_csv(out_p, index=False)

    if verbose:
        usable_rate = float(df["xg_usable"].mean()) if len(df) else 0.0
        xg_ok = df[["xg_proxy_H", "xg_proxy_A"]].notna().all(axis=1)
        xg_rate = float(xg_ok.mean()) if len(df) else 0.0
        print(f"[ok] wrote -> {out_p}")
        print(
            "[xg_proxy]",
            {
                "rows": int(len(df)),
                "xg_usable_rate": round(usable_rate, 4),
                "xg_proxy_both_sides_rate": round(xg_rate, 4),
                "fit_weeks": int(fit_weeks),
                "used_prev_weeks": int(used_prev_weeks),
                "used_prior_weeks": int(used_prior_weeks),
                "train_rows_min": int(min(train_sizes)) if train_sizes else 0,
                "train_rows_med": float(np.median(train_sizes)) if train_sizes else 0.0,
                "train_rows_max": int(max(train_sizes)) if train_sizes else 0,
            },
        )
        log_json(
            {
                "event": "XG_PROXY_SUMMARY",
                "history_in": str(hist_p),
                "history_out": str(out_p),
                "rows": int(len(df)),
                "xg_usable_rate": float(usable_rate),
                "xg_proxy_both_sides_rate": float(xg_rate),
                "fit_weeks": int(fit_weeks),
                "used_prev_weeks": int(used_prev_weeks),
                "used_prior_weeks": int(used_prior_weeks),
                "train_rows_min": int(min(train_sizes)) if train_sizes else 0,
                "train_rows_med": float(np.median(train_sizes)) if train_sizes else 0.0,
                "train_rows_max": int(max(train_sizes)) if train_sizes else 0,
            },
            log_path=log_path,
        )
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser(description="Build walk-forward xG proxy history from CGM stats")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats.csv", help="Input history+stats CSV")
    ap.add_argument("--out", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv", help="Output history+xg CSV")
    ap.add_argument("--min-train-rows", type=int, default=30, help="Min side-rows required to refit weekly model")
    ap.add_argument("--prior-mu", type=float, default=1.25, help="Fallback mean goals per side when no model exists")
    ap.add_argument("--log-json", default=str(LOG_PATH_DEFAULT), help="Append audit lines to this JSONL log")
    args = ap.parse_args()
    build_xg_proxy_history(
        history_path=args.history,
        out_path=args.out,
        min_train_rows=int(args.min_train_rows),
        prior_mu=float(args.prior_mu),
        log_json_path=args.log_json,
        verbose=True,
    )


if __name__ == "__main__":
    main()
