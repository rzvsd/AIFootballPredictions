"""
Milestone 3: xG-proxy engine (leakage-safe, walk-forward).

Builds a per-match xG proxy from aggregate match stats and writes an enriched
history artifact for downstream rolling xG form features.

Feature sets
- v1: legacy baseline (shots/sot attack+defense + home flag)
- v2: enhanced (adds corners, possession, shot-share and Elo-opponent factor,
      plus optional league calibration)

Key properties
- No true shot-level xG (no location/body-part/event context).
- Walk-forward integrity by ISO week (train strictly before prediction week).
- Symmetric modeling for home/away via side-stacking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cgm.pressure_inputs import ensure_pressure_inputs

try:
    import config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore


FEATURE_COLS_V1 = ["shots_for", "sot_for", "shots_against", "sot_against", "is_home"]
FEATURE_COLS_V2 = [
    "shots_for",
    "sot_for",
    "shots_against",
    "sot_against",
    "corners_for",
    "corners_against",
    "poss_for",
    "poss_against",
    "sot_share_for",
    "sot_share_against",
    "opp_elo_factor",
    "is_home",
]

DEFAULT_FILL_VALUES = {
    "shots_for": 10.0,
    "sot_for": 3.5,
    "shots_against": 10.0,
    "sot_against": 3.5,
    "corners_for": 4.5,
    "corners_against": 4.5,
    "poss_for": 50.0,
    "poss_against": 50.0,
    "sot_share_for": 0.33,
    "sot_share_against": 0.33,
    "opp_elo_factor": 1.0,
    "is_home": 0.5,
}

LOG_PATH_DEFAULT = Path("reports/run_log.jsonl")


def _cfg(name: str, default):
    if config is None:
        return default
    return getattr(config, name, default)


def _safe_float(v, default: float) -> float:
    try:
        out = float(v)
        if np.isfinite(out):
            return float(out)
    except Exception:
        pass
    return float(default)


def log_json(event: dict, log_path: Path = LOG_PATH_DEFAULT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = event.copy()
    out.setdefault("ts", pd.Timestamp.utcnow().isoformat())
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True).dt.tz_convert(None)
        if out["datetime"].notna().any():
            return out
    dt = pd.to_datetime(out["date"] + " " + out.get("time", "").astype(str), errors="coerce", utc=True).dt.tz_convert(None)
    dt2 = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None)
    out["datetime"] = pd.to_datetime(dt.fillna(dt2), errors="coerce")
    return out


def _week_start(dt: pd.Series) -> pd.Series:
    dtn = pd.to_datetime(dt, errors="coerce")
    return (dtn - pd.to_timedelta(dtn.dt.weekday, unit="D")).dt.normalize()


def _xg_usable_mask(df: pd.DataFrame) -> pd.Series:
    required = ["shots_H", "shots_A", "sot_H", "sot_A"]
    for c in required:
        if c not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
    x = df[required].apply(pd.to_numeric, errors="coerce")
    ok = x.notna().all(axis=1)
    ok &= (x >= 0).all(axis=1)
    return ok


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    return np.where(d > 0, n / d, np.nan)


def _elo_cols(df: pd.DataFrame) -> tuple[str, str]:
    if "elo_home_calc" in df.columns and "elo_away_calc" in df.columns:
        return ("elo_home_calc", "elo_away_calc")
    return ("elo_home", "elo_away")


def _stack_sides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match rows -> side rows (two per match) for symmetric modeling.

    Columns include core features + league/team metadata used for calibration.
    """
    elo_h_col, elo_a_col = _elo_cols(df)
    rows = []
    for idx, r in df.iterrows():
        league = str(r.get("league") or "")
        home = str(r.get("home") or "")
        away = str(r.get("away") or "")

        shots_h = r.get("shots_H")
        shots_a = r.get("shots_A")
        sot_h = r.get("sot_H")
        sot_a = r.get("sot_A")
        cor_h = r.get("corners_H")
        cor_a = r.get("corners_A")
        pos_h = r.get("pos_H")
        pos_a = r.get("pos_A")

        elo_h = r.get(elo_h_col)
        elo_a = r.get(elo_a_col)

        rows.append(
            {
                "match_idx": idx,
                "league": league,
                "team": home,
                "opp_team": away,
                "is_home": 1.0,
                "goals_for": r.get("ft_home"),
                "shots_for": shots_h,
                "sot_for": sot_h,
                "shots_against": shots_a,
                "sot_against": sot_a,
                "corners_for": cor_h,
                "corners_against": cor_a,
                "poss_for": pos_h,
                "poss_against": pos_a,
                "sot_share_for": np.nan,
                "sot_share_against": np.nan,
                "opp_elo": elo_a,
                "opp_elo_factor": np.nan,
            }
        )
        rows.append(
            {
                "match_idx": idx,
                "league": league,
                "team": away,
                "opp_team": home,
                "is_home": 0.0,
                "goals_for": r.get("ft_away"),
                "shots_for": shots_a,
                "sot_for": sot_a,
                "shots_against": shots_h,
                "sot_against": sot_h,
                "corners_for": cor_a,
                "corners_against": cor_h,
                "poss_for": pos_a,
                "poss_against": pos_h,
                "sot_share_for": np.nan,
                "sot_share_against": np.nan,
                "opp_elo": elo_h,
                "opp_elo_factor": np.nan,
            }
        )

    out = pd.DataFrame(rows)
    num_cols = [
        "goals_for",
        "shots_for",
        "sot_for",
        "shots_against",
        "sot_against",
        "corners_for",
        "corners_against",
        "poss_for",
        "poss_against",
        "is_home",
        "opp_elo",
    ]
    _coerce_numeric(out, num_cols)

    out["sot_share_for"] = _safe_div(out["sot_for"], out["shots_for"])
    out["sot_share_against"] = _safe_div(out["sot_against"], out["shots_against"])
    return out


def _league_elo_reference(train_df: pd.DataFrame) -> Tuple[dict[str, float], float]:
    elo_h_col, elo_a_col = _elo_cols(train_df)
    if elo_h_col not in train_df.columns or elo_a_col not in train_df.columns:
        return {}, 1500.0

    tmp_h = pd.DataFrame({"league": train_df.get("league", ""), "elo": pd.to_numeric(train_df[elo_h_col], errors="coerce")})
    tmp_a = pd.DataFrame({"league": train_df.get("league", ""), "elo": pd.to_numeric(train_df[elo_a_col], errors="coerce")})
    tmp = pd.concat([tmp_h, tmp_a], ignore_index=True)
    tmp = tmp.dropna(subset=["elo"])
    if tmp.empty:
        return {}, 1500.0

    global_ref = float(tmp["elo"].mean()) if tmp["elo"].notna().any() else 1500.0
    league_ref = (
        tmp.groupby("league", dropna=False)["elo"]
        .mean()
        .dropna()
        .to_dict()
    )
    return {str(k): float(v) for k, v in league_ref.items()}, float(global_ref)


def _attach_elo_factor(stacked: pd.DataFrame, league_ref: dict[str, float], global_ref: float, clip_min: float, clip_max: float) -> pd.DataFrame:
    out = stacked.copy()
    ref_vals = out["league"].astype(str).map(league_ref).fillna(float(global_ref))
    opp_elo = pd.to_numeric(out.get("opp_elo"), errors="coerce")
    out["opp_elo_factor"] = np.where(ref_vals > 0, opp_elo / ref_vals, 1.0)
    out["opp_elo_factor"] = pd.to_numeric(out["opp_elo_factor"], errors="coerce").clip(clip_min, clip_max)
    return out


def _prepare_features(stacked: pd.DataFrame, feature_cols: list[str], fill_values: dict[str, float] | None = None) -> tuple[np.ndarray, dict[str, float]]:
    X = stacked.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)

    if fill_values is None:
        fill_values = {}
        for c in feature_cols:
            med = pd.to_numeric(X[c], errors="coerce").median()
            if pd.notna(med):
                fill_values[c] = float(med)
            else:
                fill_values[c] = float(DEFAULT_FILL_VALUES.get(c, 0.0))

    for c in feature_cols:
        X[c] = X[c].fillna(float(fill_values.get(c, DEFAULT_FILL_VALUES.get(c, 0.0))))

    return X[feature_cols].to_numpy(dtype=float), fill_values


def _build_league_calibration(
    stacked_train: pd.DataFrame,
    pred_train: np.ndarray,
    *,
    enabled: bool,
    min_rows: int,
    prior_strength: float,
    clip_min: float,
    clip_max: float,
) -> dict[str, float]:
    if not enabled:
        return {}

    tmp = stacked_train[["league", "goals_for"]].copy()
    tmp["pred"] = pd.to_numeric(pred_train, errors="coerce")
    tmp["goals_for"] = pd.to_numeric(tmp["goals_for"], errors="coerce")
    tmp = tmp.dropna(subset=["goals_for", "pred"])
    if tmp.empty:
        return {}

    out: dict[str, float] = {}
    for league, g in tmp.groupby("league", dropna=False):
        n = int(len(g))
        if n < int(min_rows):
            continue
        pred_mean = float(g["pred"].mean())
        obs_mean = float(g["goals_for"].mean())
        if pred_mean <= 1e-9:
            continue
        raw_mult = obs_mean / pred_mean
        w = n / (n + float(prior_strength))
        mult = 1.0 + w * (raw_mult - 1.0)
        mult = float(np.clip(mult, clip_min, clip_max))
        out[str(league)] = mult
    return out


def build_xg_proxy_history(
    history_path: str | Path = "data/enhanced/cgm_match_history_with_elo_stats.csv",
    out_path: str | Path = "data/enhanced/cgm_match_history_with_elo_stats_xg.csv",
    *,
    min_train_rows: int = 30,
    prior_mu: float = 1.25,
    feature_set: str = "v2",
    league_calibration_enabled: bool = True,
    league_calibration_min_rows: int = 80,
    league_calibration_prior_strength: float = 80.0,
    league_calibration_clip_min: float = 0.85,
    league_calibration_clip_max: float = 1.15,
    elo_factor_clip_min: float = 0.80,
    elo_factor_clip_max: float = 1.20,
    log_json_path: str | Path = LOG_PATH_DEFAULT,
    verbose: bool = True,
) -> Path:
    hist_p = Path(history_path)
    if not hist_p.exists():
        raise FileNotFoundError(f"{hist_p} not found (run stats backfill first).")

    feature_set = str(feature_set or "v2").strip().lower()
    feature_cols = FEATURE_COLS_V2 if feature_set == "v2" else FEATURE_COLS_V1

    df = pd.read_csv(hist_p)
    df = _ensure_datetime(df)
    df = ensure_pressure_inputs(df)

    _coerce_numeric(df, [
        "ft_home", "ft_away", "shots_H", "shots_A", "sot_H", "sot_A",
        "corners_H", "corners_A", "pos_H", "pos_A", "elo_home", "elo_away", "elo_home_calc", "elo_away_calc",
    ])

    df = df.sort_values("datetime").reset_index(drop=True)

    xg_usable = _xg_usable_mask(df)
    df["xg_usable"] = xg_usable.astype(int)
    df["week_start"] = _week_start(df["datetime"])

    df["xg_proxy_H"] = np.nan
    df["xg_proxy_A"] = np.nan

    prev_model: Pipeline | None = None
    fit_weeks = 0
    used_prev_weeks = 0
    used_prior_weeks = 0
    train_sizes: list[int] = []

    cal_mult_usage: list[float] = []
    log_path = Path(log_json_path) if log_json_path else LOG_PATH_DEFAULT

    for ws in sorted(df["week_start"].dropna().unique()):
        batch_mask = df["week_start"] == ws
        train_mask = (df["datetime"] < ws) & (df["xg_usable"] == 1)

        train_df = df.loc[train_mask, :]
        batch_df = df.loc[batch_mask & (df["xg_usable"] == 1), :]
        if batch_df.empty:
            continue

        stacked_train = _stack_sides(train_df) if not train_df.empty else pd.DataFrame(columns=["goals_for"] + feature_cols)
        stacked_train = stacked_train.dropna(subset=["goals_for"], how="any")

        league_elo_ref, global_elo_ref = _league_elo_reference(train_df)
        if not stacked_train.empty and "opp_elo_factor" in feature_cols:
            stacked_train = _attach_elo_factor(
                stacked_train,
                league_elo_ref,
                global_elo_ref,
                clip_min=float(elo_factor_clip_min),
                clip_max=float(elo_factor_clip_max),
            )

        model: Pipeline | None = None
        used_mode = "prior"
        fill_values: dict[str, float] = {}
        league_cal_map: dict[str, float] = {}

        if len(stacked_train) >= int(min_train_rows):
            X_tr, fill_values = _prepare_features(stacked_train, feature_cols)
            y_tr = pd.to_numeric(stacked_train["goals_for"], errors="coerce").to_numpy(dtype=float)
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

            if feature_set == "v2":
                pred_train = np.clip(model.predict(X_tr), 0.0, None)
                league_cal_map = _build_league_calibration(
                    stacked_train,
                    pred_train,
                    enabled=bool(league_calibration_enabled),
                    min_rows=int(league_calibration_min_rows),
                    prior_strength=float(league_calibration_prior_strength),
                    clip_min=float(league_calibration_clip_min),
                    clip_max=float(league_calibration_clip_max),
                )
        elif prev_model is not None:
            model = prev_model
            used_prev_weeks += 1
            used_mode = "prev"
        else:
            model = None
            used_prior_weeks += 1
            used_mode = "prior"

        stacked_batch = _stack_sides(batch_df)
        if "opp_elo_factor" in feature_cols:
            stacked_batch = _attach_elo_factor(
                stacked_batch,
                league_elo_ref,
                global_elo_ref,
                clip_min=float(elo_factor_clip_min),
                clip_max=float(elo_factor_clip_max),
            )

        train_last_dt = None
        try:
            train_last_dt = pd.to_datetime(train_df["datetime"], errors="coerce").max() if not train_df.empty else None
        except Exception:
            train_last_dt = None
        ok_cutoff = None
        if train_last_dt is not None and not pd.isna(train_last_dt):
            ok_cutoff = bool(train_last_dt < pd.to_datetime(ws))

        log_json(
            {
                "event": "XG_PROXY_WEEK",
                "week_start": str(pd.to_datetime(ws).date()) if ws is not None else None,
                "train_last_datetime": str(train_last_dt) if train_last_dt is not None and not pd.isna(train_last_dt) else None,
                "ok_train_cutoff": ok_cutoff,
                "used_mode": used_mode,
                "feature_set": feature_set,
                "train_matches": int(len(train_df)),
                "train_side_rows": int(len(stacked_train)),
                "batch_matches": int(len(batch_df)),
                "batch_side_rows": int(len(stacked_batch)),
                "min_train_side_rows": int(min_train_rows),
                "league_calibration_count": int(len(league_cal_map)),
            },
            log_path=log_path,
        )

        if model is None:
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
            X_b, _ = _prepare_features(stacked_batch, feature_cols, fill_values=fill_values)
            preds = np.clip(model.predict(X_b), 0.0, None)

        if feature_set == "v2" and len(preds) > 0:
            if league_cal_map:
                leagues = stacked_batch["league"].astype(str).tolist()
                mults = np.array([float(league_cal_map.get(lg, 1.0)) for lg in leagues], dtype=float)
                cal_mult_usage.extend(mults.tolist())
                preds = preds * mults

        preds = np.clip(preds, 0.0, None)

        for (match_idx, is_home), mu in zip(stacked_batch[["match_idx", "is_home"]].itertuples(index=False, name=None), preds.tolist()):
            if int(is_home) == 1:
                df.loc[int(match_idx), "xg_proxy_H"] = float(mu)
            else:
                df.loc[int(match_idx), "xg_proxy_A"] = float(mu)

    # Helpers (per-match; NOT safe as model features)
    def _safe_div_series(num_s: pd.Series, den_s: pd.Series) -> pd.Series:
        num = pd.to_numeric(num_s, errors="coerce")
        den = pd.to_numeric(den_s, errors="coerce")
        den2 = den.where(den > 0, 1.0)
        return (num / den2).replace([np.inf, -np.inf], np.nan)

    df["shot_quality_H"] = _safe_div_series(df["xg_proxy_H"], df["shots_H"])
    df["shot_quality_A"] = _safe_div_series(df["xg_proxy_A"], df["shots_A"])
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

        summary = {
            "rows": int(len(df)),
            "feature_set": feature_set,
            "xg_usable_rate": round(usable_rate, 4),
            "xg_proxy_both_sides_rate": round(xg_rate, 4),
            "fit_weeks": int(fit_weeks),
            "used_prev_weeks": int(used_prev_weeks),
            "used_prior_weeks": int(used_prior_weeks),
            "train_rows_min": int(min(train_sizes)) if train_sizes else 0,
            "train_rows_med": float(np.median(train_sizes)) if train_sizes else 0.0,
            "train_rows_max": int(max(train_sizes)) if train_sizes else 0,
            "league_cal_mult_med": float(np.median(cal_mult_usage)) if cal_mult_usage else 1.0,
        }
        print(f"[ok] wrote -> {out_p}")
        print("[xg_proxy]", summary)
        log_json(
            {
                "event": "XG_PROXY_SUMMARY",
                "history_in": str(hist_p),
                "history_out": str(out_p),
                **{k: (float(v) if isinstance(v, np.floating) else v) for k, v in summary.items()},
            },
            log_path=log_path,
        )

    return out_p


def main() -> None:
    default_feature_set = str(_cfg("XG_PROXY_FEATURE_SET_DEFAULT", "v2"))
    default_lg_cal = bool(_cfg("XG_PROXY_LEAGUE_CALIBRATION_ENABLED", True))
    default_lg_min_rows = int(_safe_float(_cfg("XG_PROXY_LEAGUE_CALIBRATION_MIN_ROWS", 80), 80))
    default_lg_prior = float(_safe_float(_cfg("XG_PROXY_LEAGUE_CALIBRATION_PRIOR_STRENGTH", 80.0), 80.0))
    default_lg_clip_min = float(_safe_float(_cfg("XG_PROXY_LEAGUE_CALIBRATION_CLIP_MIN", 0.85), 0.85))
    default_lg_clip_max = float(_safe_float(_cfg("XG_PROXY_LEAGUE_CALIBRATION_CLIP_MAX", 1.15), 1.15))
    default_elo_clip_min = float(_safe_float(_cfg("XG_PROXY_ELO_FACTOR_CLIP_MIN", 0.80), 0.80))
    default_elo_clip_max = float(_safe_float(_cfg("XG_PROXY_ELO_FACTOR_CLIP_MAX", 1.20), 1.20))

    ap = argparse.ArgumentParser(description="Build walk-forward xG proxy history from CGM stats")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats.csv", help="Input history+stats CSV")
    ap.add_argument("--out", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv", help="Output history+xg CSV")
    ap.add_argument("--min-train-rows", type=int, default=30, help="Min side-rows required to refit weekly model")
    ap.add_argument("--prior-mu", type=float, default=1.25, help="Fallback mean goals per side when no model exists")
    ap.add_argument("--feature-set", choices=["v1", "v2"], default=default_feature_set, help="xG proxy feature set")
    ap.add_argument("--league-calibration", dest="league_calibration", action="store_true", default=default_lg_cal, help="Enable league-level xG output calibration")
    ap.add_argument("--no-league-calibration", dest="league_calibration", action="store_false", help="Disable league-level xG output calibration")
    ap.add_argument("--league-cal-min-rows", type=int, default=default_lg_min_rows, help="Min train side-rows per league for calibration")
    ap.add_argument("--league-cal-prior", type=float, default=default_lg_prior, help="Shrink prior strength for league calibration")
    ap.add_argument("--league-cal-clip-min", type=float, default=default_lg_clip_min, help="Min league calibration multiplier")
    ap.add_argument("--league-cal-clip-max", type=float, default=default_lg_clip_max, help="Max league calibration multiplier")
    ap.add_argument("--elo-factor-clip-min", type=float, default=default_elo_clip_min, help="Min clip for opponent Elo factor")
    ap.add_argument("--elo-factor-clip-max", type=float, default=default_elo_clip_max, help="Max clip for opponent Elo factor")
    ap.add_argument("--log-json", default=str(LOG_PATH_DEFAULT), help="Append audit lines to this JSONL log")
    args = ap.parse_args()

    build_xg_proxy_history(
        history_path=args.history,
        out_path=args.out,
        min_train_rows=int(args.min_train_rows),
        prior_mu=float(args.prior_mu),
        feature_set=str(args.feature_set),
        league_calibration_enabled=bool(args.league_calibration),
        league_calibration_min_rows=int(args.league_cal_min_rows),
        league_calibration_prior_strength=float(args.league_cal_prior),
        league_calibration_clip_min=float(args.league_cal_clip_min),
        league_calibration_clip_max=float(args.league_cal_clip_max),
        elo_factor_clip_min=float(args.elo_factor_clip_min),
        elo_factor_clip_max=float(args.elo_factor_clip_max),
        log_json_path=args.log_json,
        verbose=True,
    )


if __name__ == "__main__":
    main()
