"""
Milestone 5: Live Inference & Value Scan using CGM data.

Uses only CGM CSVs to build opponent-aware Frankenstein features, predict mu_home/mu_away,
convert to Poisson probabilities, and compute EV vs book odds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import hashlib
import logging
import random

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
try:
    import config
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config

try:
    from cgm.team_registry import build_team_registry, normalize_team  # type: ignore
except ImportError:  # pragma: no cover - local fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from team_registry import build_team_registry, normalize_team  # type: ignore
from cgm.elo_similarity import prepare_histories, kernel_similarity
from cgm.pressure_inputs import ensure_pressure_inputs
from cgm.pressure_form import add_pressure_form_features
from cgm.xg_form import add_xg_form_features
from cgm.goal_timing import build_team_timing_profiles, compute_match_timing
from cgm.league_features import get_league_features_for_fixture
from cgm.h2h_features import get_h2h_features_for_fixture


LOG_PATH_DEFAULT = Path("reports/run_log.jsonl")
TRACE_PATH_DEFAULT = Path("reports/elo_trace.jsonl")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def log_json(event: dict, log_path: Path = LOG_PATH_DEFAULT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = event.copy()
    out.setdefault("ts", pd.Timestamp.utcnow().isoformat())
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


def _load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        dt = pd.to_datetime(df["date"] + " " + df.get("time", "").astype(str), errors="coerce")
        dt2 = pd.to_datetime(df["date"], errors="coerce")
        df["datetime"] = dt.fillna(dt2)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values(["datetime", "home", "away"], kind="mergesort")
    # Milestone 2: parse/split combined CGM stats before numeric coercion
    df = ensure_pressure_inputs(df)
    for col in df.columns:
        if col in ["home", "away", "date", "time", "result", "datetime"]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _stable_sort_history(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["datetime", "home", "away"] if c in df.columns]
    if not cols:
        return df
    return df.sort_values(cols, kind="mergesort")


def _rolling_stats(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = df.copy()

    def _roll_home(g):
        g = g.sort_values(["datetime", "away"], kind="mergesort").copy()
        for w in windows:
            g[f"H_gf_L{w}"] = g["ft_home"].rolling(w, min_periods=1).mean()
            g[f"H_ga_L{w}"] = g["ft_away"].rolling(w, min_periods=1).mean()
            g[f"H_shots_for_L{w}"] = g["shots_H"].rolling(w, min_periods=1).mean()
            g[f"H_sot_for_L{w}"] = g["sot_H"].rolling(w, min_periods=1).mean()
            g[f"H_cor_for_L{w}"] = g["corners_H"].rolling(w, min_periods=1).mean()
            g[f"H_poss_for_L{w}"] = g["pos_H"].rolling(w, min_periods=1).mean()
        return g

    df = df.groupby("home", group_keys=False).apply(_roll_home)

    def _roll_away(g):
        g = g.sort_values(["datetime", "home"], kind="mergesort").copy()
        for w in windows:
            g[f"A_gf_L{w}"] = g["ft_away"].rolling(w, min_periods=1).mean()
            g[f"A_ga_L{w}"] = g["ft_home"].rolling(w, min_periods=1).mean()
            g[f"A_shots_for_L{w}"] = g["shots_A"].rolling(w, min_periods=1).mean()
            g[f"A_sot_for_L{w}"] = g["sot_A"].rolling(w, min_periods=1).mean()
            g[f"A_cor_for_L{w}"] = g["corners_A"].rolling(w, min_periods=1).mean()
            g[f"A_poss_for_L{w}"] = g["pos_A"].rolling(w, min_periods=1).mean()
        return g

    df = df.groupby("away", group_keys=False).apply(_roll_away)
    return df


def _add_franken_features(df: pd.DataFrame, windows: List[int], alpha: float = 0.5, beta: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        for prefix in ["H", "A"]:
            shots = df.get(f"{prefix}_shots_for_L{w}")
            sot = df.get(f"{prefix}_sot_for_L{w}")
            gf = df.get(f"{prefix}_gf_L{w}")
            if shots is not None and sot is not None:
                df[f"{prefix}_shot_quality_L{w}"] = (sot + alpha) / (shots + beta)
            if sot is not None and gf is not None:
                df[f"{prefix}_finish_rate_L{w}"] = (gf + alpha) / (sot + beta)

    def _ratio(num, den):
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f == 0 or np.isnan(den_f) or np.isnan(num_f):
                return np.nan
            return num_f / den_f
        except Exception as e:
            _logger.warning(f"_ratio error: {e}")
            return np.nan

    df["Attack_H"] = df.apply(lambda r: _ratio(r.get("H_gf_L5"), r.get("lg_avg_gf_home")), axis=1)
    df["Defense_H"] = df.apply(lambda r: _ratio(r.get("H_ga_L5"), r.get("lg_avg_gf_away")), axis=1)
    df["Attack_A"] = df.apply(lambda r: _ratio(r.get("A_gf_L5"), r.get("lg_avg_gf_away")), axis=1)
    df["Defense_A"] = df.apply(lambda r: _ratio(r.get("A_ga_L5"), r.get("lg_avg_gf_home")), axis=1)
    df["Expected_Destruction_H"] = df["Attack_H"] * df["Defense_A"]
    df["Expected_Destruction_A"] = df["Attack_A"] * df["Defense_H"]
    return df


def _latest_side_snapshots(df_roll: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return latest home-context and away-context rows per team."""
    ordered = df_roll.sort_values(["datetime", "home", "away"], kind="mergesort")
    latest_home = ordered.groupby("home", as_index=False).tail(1).set_index("home")
    latest_away = ordered.groupby("away", as_index=False).tail(1).set_index("away")
    return latest_home, latest_away


def _band_from_diff(diff: float, thresh: float = 150.0) -> str:
    if diff >= thresh:
        return "BULLY"
    if diff <= -thresh:
        return "DOG"
    return "PEER"


def _precompute_band_stats(df: pd.DataFrame) -> dict:
    """Precompute per-team averages vs each band using historical matches (venue-aware)."""
    out = {}
    for side in ["home", "away"]:
        band_col = "Band_H" if side == "home" else "Band_A"
        gf_col = "ft_home" if side == "home" else "ft_away"
        ga_col = "ft_away" if side == "home" else "ft_home"
        for team, g in df.groupby(side):
            band_map = {}
            for band, gband in g.groupby(band_col):
                band_map[band] = {"gf": gband[gf_col].mean(), "ga": gband[ga_col].mean()}
            out.setdefault(team, {}).setdefault(side, {}).update(band_map)
    return out


def _implied_probs_1x2(odds_home: float | None, odds_draw: float | None, odds_away: float | None) -> dict:
    vals = []
    for o in (odds_home, odds_draw, odds_away):
        try:
            o_f = float(o)
            vals.append(1.0 / o_f if o_f > 0 else np.nan)
        except Exception:
            vals.append(np.nan)
    if all(np.isnan(v) for v in vals):
        return {"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan}
    total = np.nansum(vals)
    if total == 0 or np.isnan(total):
        return {"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan}
    return {"p_home": vals[0] / total if not np.isnan(vals[0]) else np.nan,
            "p_draw": vals[1] / total if not np.isnan(vals[1]) else np.nan,
            "p_away": vals[2] / total if not np.isnan(vals[2]) else np.nan}


def _implied_probs_two_way(odds_over: float | None, odds_under: float | None) -> dict:
    vals = []
    for o in (odds_over, odds_under):
        try:
            o_f = float(o)
            vals.append(1.0 / o_f if o_f > 0 else np.nan)
        except Exception:
            vals.append(np.nan)
    if all(np.isnan(v) for v in vals):
        return {"p_over": np.nan, "p_under": np.nan}
    total = np.nansum(vals)
    if total == 0 or np.isnan(total):
        return {"p_over": np.nan, "p_under": np.nan}
    return {"p_over": vals[0] / total if not np.isnan(vals[0]) else np.nan,
            "p_under": vals[1] / total if not np.isnan(vals[1]) else np.nan}


def _poisson_probs(mu_h: float, mu_a: float) -> dict:
    max_goals = 10
    ph = [poisson.pmf(i, mu_h) for i in range(max_goals + 1)]
    pa = [poisson.pmf(i, mu_a) for i in range(max_goals + 1)]
    P = np.outer(ph, pa)
    home = np.tril(P, -1).sum()
    draw = np.diag(P).sum()
    away = np.triu(P, 1).sum()
    ou25_over = P[np.add.outer(range(max_goals + 1), range(max_goals + 1)) > 2].sum()
    ou25_under = 1.0 - ou25_over
    return {"p_home": float(home), "p_draw": float(draw), "p_away": float(away),
            "p_over25": float(ou25_over), "p_under25": float(ou25_under)}


def _parse_upcoming_datetime(datameci: object, orameci: object) -> pd.Timestamp:
    date_raw = "" if datameci is None else str(datameci)
    dt = pd.to_datetime(date_raw, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(date_raw, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return pd.NaT

    hour = 0
    minute = 0
    try:
        t = int(float(orameci)) if orameci is not None and str(orameci) != "nan" else 0
        hour = max(0, min(23, t // 100))
        minute = max(0, min(59, t % 100))
    except Exception:
        hour = 0
        minute = 0

    return pd.to_datetime(dt.normalize() + pd.Timedelta(hours=hour, minutes=minute))


def _log_run(meta: dict, out_path: Path) -> None:
    log_dir = Path("reports")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run_log.jsonl"
    meta_out = meta.copy()
    meta_out["output"] = str(out_path)
    meta_out["ts"] = pd.Timestamp.utcnow().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta_out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict upcoming fixtures using Frankenstein models")
    ap.add_argument("--data-dir", default="CGM data", help="CGM data directory containing allratingv.csv/upcoming.csv")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv", help="Match history with features")
    ap.add_argument("--models-dir", default="models", help="Directory with frankenstein_mu_home/away.pkl")
    ap.add_argument("--model-variant", choices=["full", "no_odds"], default="full", help="Model variant (feature set)")
    ap.add_argument("--out", default="reports/cgm_upcoming_predictions.csv", help="Output predictions CSV")
    ap.add_argument("--windows", nargs="+", type=int, default=[5, 10], help="Rolling windows")
    ap.add_argument("--min-matches", type=int, default=8, help="Minimum matches required per team to use prediction")

    # Live scope (Milestone 4+). Defaults come from config.py; CLI overrides are deterministic.
    ap.add_argument("--as-of-date", default=None, help="Run as-of date (YYYY-MM-DD). Used to filter past fixtures strictly.")
    ap.add_argument("--scope-country", default=getattr(config, "LIVE_SCOPE_COUNTRY", ""), help="Optional country filter (empty disables).")
    ap.add_argument("--scope-league", default=getattr(config, "LIVE_SCOPE_LEAGUE", ""), help="Optional league filter (empty disables).")
    ap.add_argument("--scope-season-start", default=getattr(config, "LIVE_SCOPE_SEASON_START", ""), help="Optional season window start (YYYY-MM-DD).")
    ap.add_argument("--scope-season-end", default=getattr(config, "LIVE_SCOPE_SEASON_END", ""), help="Optional season window end (YYYY-MM-DD).")
    ap.add_argument("--horizon-days", type=int, default=int(getattr(config, "LIVE_SCOPE_HORIZON_DAYS", 0) or 0), help="Optional horizon (days). 0 disables.")

    ap.add_argument("--log-level", default="INFO", help="Logging level (INFO/DEBUG/WARNING)")
    ap.add_argument("--log-json", default=str(LOG_PATH_DEFAULT), help="Path to JSONL run log")
    ap.add_argument("--log-sample-rate", type=float, default=0.0, help="Probability to log per-fixture trace")
    ap.add_argument("--log-max-fixtures", type=int, default=25, help="Max fixtures to trace-log per run")
    ap.add_argument("--trace-json", default=str(TRACE_PATH_DEFAULT), help="Optional fixture-level trace JSONL")
    ap.add_argument("--trace-topk", type=int, default=0, help="(reserved) top-k contributors per kernel (not used yet)")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("predict_upcoming")

    reg = build_team_registry(args.data_dir)

    hist = _load_history(Path(args.history))
    hist_hash = file_hash(Path(args.history)) if Path(args.history).exists() else None
    logger.info("[ELO][DATA] history=%s rows=%d hash=%s date_range=[%s,%s]", args.history, len(hist), hist_hash,
                hist["datetime"].min(), hist["datetime"].max())
    log_json(
        {
            "event": "ELO_DATA",
            "history": str(Path(args.history)),
            "history_hash": hist_hash,
            "rows": len(hist),
            "date_min": str(hist["datetime"].min()),
            "date_max": str(hist["datetime"].max()),
        },
        log_path=Path(args.log_json),
    )

    # ------------------------------------------------------------------
    # Live scope (single source of truth)
    # ------------------------------------------------------------------
    scope_country = str(getattr(args, "scope_country", "") or "").strip() or None
    scope_league = str(getattr(args, "scope_league", "") or "").strip() or None

    _ss = pd.to_datetime(getattr(args, "scope_season_start", "") or "", errors="coerce")
    scope_season_start = _ss.normalize() if not pd.isna(_ss) else None
    _se = pd.to_datetime(getattr(args, "scope_season_end", "") or "", errors="coerce")
    scope_season_end = _se.normalize() if not pd.isna(_se) else None

    horizon_days = int(getattr(args, "horizon_days", 0) or 0)
    if horizon_days <= 0:
        horizon_days = 0

    # Derive as-of from CLI or (deterministically) from the history max date.
    as_of_date = None
    if getattr(args, "as_of_date", None):
        ts = pd.to_datetime(args.as_of_date, errors="coerce")
        if not pd.isna(ts):
            as_of_date = ts.date()
    if as_of_date is None:
        hist_max = pd.to_datetime(hist["datetime"], errors="coerce").max()
        if not pd.isna(hist_max):
            as_of_date = pd.Timestamp(hist_max).date()
    if as_of_date is None:
        as_of_date = pd.Timestamp.utcnow().date()

    # Strict "no same-day" cutoff: fixtures must be strictly after this timestamp.
    run_asof_datetime = (
        pd.Timestamp(as_of_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    )
    # History cutoff: prevent leakage by excluding rows after the run cutoff.
    hist_dt = pd.to_datetime(hist["datetime"], errors="coerce")
    future_mask = hist_dt.notna() & (hist_dt > run_asof_datetime)
    dropped_future = int(future_mask.sum())
    if dropped_future:
        hist = hist.loc[~future_mask].copy()
    na_hist = int(hist_dt.isna().sum())
    logger.info(
        "[HISTORY] cutoff=%s dropped_future=%d kept=%d na_datetime=%d",
        str(run_asof_datetime),
        dropped_future,
        len(hist),
        na_hist,
    )
    log_json(
        {
            "event": "HISTORY_CUTOFF",
            "cutoff": str(run_asof_datetime),
            "rows_before": int(len(hist) + dropped_future),
            "rows_after": int(len(hist)),
            "dropped_future": int(dropped_future),
            "na_datetime": int(na_hist),
        },
        log_path=Path(args.log_json),
    )
    hist = _stable_sort_history(hist)
    hist = _rolling_stats(hist, args.windows)
    hist = _stable_sort_history(hist)
    hist = _add_franken_features(hist, args.windows)
    hist = _stable_sort_history(hist)
    # Milestone 2: Pressure Cooker features (needed if models were trained with them)
    hist = ensure_pressure_inputs(hist)
    hist = _stable_sort_history(hist)
    hist = add_pressure_form_features(hist, window=10)
    hist = _stable_sort_history(hist)
    # Milestone 3: xG-proxy form (needed if models were trained with xg_* / div_px_* features)
    hist = add_xg_form_features(hist, window=10)
    hist = _stable_sort_history(hist)
    band_stats = _precompute_band_stats(hist)
    # Prepare Elo histories for similarity features (kernel)
    elo_histories = prepare_histories(hist)
    sigma_map = getattr(config, "ELO_SIM_SIGMA_PER_LEAGUE", {})
    default_sigma = 60.0  # wider for stability
    latest_home, latest_away = _latest_side_snapshots(hist)
    counts = hist["home"].value_counts().add(hist["away"].value_counts(), fill_value=0)

    # Pressure gating: mark fixtures as usable only when both teams have at least one match
    # with complete split stats in their relevant venue contexts.
    stat_cols = ["shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A"]
    pressure_usable_home_ctx: dict[str, bool] = {}
    pressure_usable_away_ctx: dict[str, bool] = {}
    try:
        if all(c in hist.columns for c in stat_cols):
            stats_all_present = hist[stat_cols].notna().all(axis=1)
            pressure_usable_home_ctx = stats_all_present.groupby(hist["home"]).any().to_dict()
            pressure_usable_away_ctx = stats_all_present.groupby(hist["away"]).any().to_dict()
    except Exception:
        pressure_usable_home_ctx = {}
        pressure_usable_away_ctx = {}

    # xG gating: usable only when at least one past match had shots+SOT (both sides) so xg_proxy exists.
    xg_usable_home_ctx: dict[str, bool] = {}
    xg_usable_away_ctx: dict[str, bool] = {}
    try:
        if "xg_usable" in hist.columns:
            xg_ok = pd.to_numeric(hist["xg_usable"], errors="coerce").fillna(0.0) > 0
        else:
            xg_ok = hist[["xg_proxy_H", "xg_proxy_A"]].notna().all(axis=1) if {"xg_proxy_H", "xg_proxy_A"}.issubset(hist.columns) else pd.Series(False, index=hist.index)
        xg_usable_home_ctx = xg_ok.groupby(hist["home"]).any().to_dict()
        xg_usable_away_ctx = xg_ok.groupby(hist["away"]).any().to_dict()
    except Exception:
        xg_usable_home_ctx = {}
        xg_usable_away_ctx = {}

    league_avg_cols = ["lg_avg_gf_home", "lg_avg_gf_away", "lg_avg_sot_home", "lg_avg_sot_away",
                       "lg_avg_cor_home", "lg_avg_cor_away", "lg_avg_poss_home", "lg_avg_poss_away"]
    league_meta = (hist.sort_values(["datetime", "home", "away"], kind="mergesort").groupby("league").tail(1)
                   [["league", "home_bonus_elo"] + league_avg_cols].set_index("league").to_dict("index"))

    # Prefer recalculated Elo columns if present
    elo_home_col = "elo_home_calc" if "elo_home_calc" in hist.columns else "elo_home"
    elo_away_col = "elo_away_calc" if "elo_away_calc" in hist.columns else "elo_away"

    # Infer the effective home advantage used in the history's EloDiff so inference matches training.
    try:
        if "EloDiff" in hist.columns and elo_home_col in hist.columns and elo_away_col in hist.columns:
            adv_series = hist["EloDiff"] - (hist[elo_home_col] - hist[elo_away_col])
            adv_series = (
                pd.to_numeric(adv_series, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            home_adv_used = float(adv_series.median()) if not adv_series.empty else 65.0
        else:
            home_adv_used = 65.0
    except Exception:
        home_adv_used = 65.0

    latest_elos: dict[str, float] = {}
    latest_team_meta: dict[str, dict] = {}
    latest_press_home: dict[str, float] = {}
    latest_press_away: dict[str, float] = {}
    latest_press_n_home: dict[str, float] = {}
    latest_press_n_away: dict[str, float] = {}
    latest_press_stats_n_home: dict[str, float] = {}
    latest_press_stats_n_away: dict[str, float] = {}
    latest_press_dom_home: dict[str, dict[str, float]] = {}
    latest_press_dom_away: dict[str, dict[str, float]] = {}
    # Milestone 9: Decay features
    latest_press_decay_home: dict[str, float] = {}
    latest_press_decay_away: dict[str, float] = {}
    latest_xg_for_decay_home: dict[str, float] = {}
    latest_xg_against_decay_home: dict[str, float] = {}
    latest_xg_for_decay_away: dict[str, float] = {}
    latest_xg_against_decay_away: dict[str, float] = {}

    latest_xg_for_home: dict[str, float] = {}
    latest_xg_against_home: dict[str, float] = {}
    latest_xg_diff_home: dict[str, float] = {}
    latest_xg_shot_quality_home: dict[str, float] = {}
    latest_xg_finishing_luck_home: dict[str, float] = {}
    latest_xg_n_home: dict[str, float] = {}
    latest_xg_stats_n_home: dict[str, float] = {}

    latest_xg_for_away: dict[str, float] = {}
    latest_xg_against_away: dict[str, float] = {}
    latest_xg_diff_away: dict[str, float] = {}
    latest_xg_shot_quality_away: dict[str, float] = {}
    latest_xg_finishing_luck_away: dict[str, float] = {}
    latest_xg_n_away: dict[str, float] = {}
    latest_xg_stats_n_away: dict[str, float] = {}
    for _, row in hist.sort_values(["datetime", "home", "away"], kind="mergesort").iterrows():
        for team, col_name in [(row["home"], elo_home_col), (row["away"], elo_away_col)]:
            try:
                val = float(row[col_name])
                if not np.isnan(val) and val > 0:
                    latest_elos[team] = val
            except Exception as e:
                logger.debug(f"Elo extract error for {team}: {e}")
        for team in [row["home"], row["away"]]:
            latest_team_meta[team] = {"season": row.get("season"), "country": row.get("country"), "league": row.get("league")}
        # Pressure snapshots (post-match rolling states)
        try:
            latest_press_home[str(row["home"])] = float(row.get("_press_form_H_post", 0.5))
            latest_press_n_home[str(row["home"])] = float(row.get("_press_n_H_post", 0.0))
            latest_press_stats_n_home[str(row["home"])] = float(row.get("_press_stats_n_H_post", 0.0))
            latest_press_dom_home[str(row["home"])] = {
                "shots": float(row.get("_press_dom_shots_H_post", 0.5)),
                "sot": float(row.get("_press_dom_sot_H_post", 0.5)),
                "corners": float(row.get("_press_dom_corners_H_post", 0.5)),
                "pos": float(row.get("_press_dom_pos_H_post", 0.5)),
            }
            # Milestone 9: Decay features (use pre-match column which uses .shift() internally)
            if "press_form_H_decay" in row.index:
                latest_press_decay_home[str(row["home"])] = float(row.get("press_form_H_decay", 0.5))
        except Exception as e:
            logger.debug(f"Pressure home stats error for {row['home']}: {e}")
        try:
            latest_press_away[str(row["away"])] = float(row.get("_press_form_A_post", 0.5))
            latest_press_n_away[str(row["away"])] = float(row.get("_press_n_A_post", 0.0))
            latest_press_stats_n_away[str(row["away"])] = float(row.get("_press_stats_n_A_post", 0.0))
            latest_press_dom_away[str(row["away"])] = {
                "shots": float(row.get("_press_dom_shots_A_post", 0.5)),
                "sot": float(row.get("_press_dom_sot_A_post", 0.5)),
                "corners": float(row.get("_press_dom_corners_A_post", 0.5)),
                "pos": float(row.get("_press_dom_pos_A_post", 0.5)),
            }
            # Milestone 9: Decay features
            if "press_form_A_decay" in row.index:
                latest_press_decay_away[str(row["away"])] = float(row.get("press_form_A_decay", 0.5))
        except Exception as e:
            logger.debug(f"Pressure away stats error for {row['away']}: {e}")

        # xG snapshots (post-match rolling states)
        try:
            latest_xg_for_home[str(row["home"])] = float(row.get("_xg_for_form_H_post", 0.0))
            latest_xg_against_home[str(row["home"])] = float(row.get("_xg_against_form_H_post", 0.0))
            latest_xg_diff_home[str(row["home"])] = float(row.get("_xg_diff_form_H_post", 0.0))
            latest_xg_shot_quality_home[str(row["home"])] = float(row.get("_xg_shot_quality_form_H_post", 0.0))
            latest_xg_finishing_luck_home[str(row["home"])] = float(row.get("_xg_finishing_luck_form_H_post", 0.0))
            latest_xg_n_home[str(row["home"])] = float(row.get("_xg_n_H_post", 0.0))
            latest_xg_stats_n_home[str(row["home"])] = float(row.get("_xg_stats_n_H_post", 0.0))
            # Milestone 9: xG decay features
            if "xg_for_form_H_decay" in row.index:
                latest_xg_for_decay_home[str(row["home"])] = float(row.get("xg_for_form_H_decay", 0.0))
            if "xg_against_form_H_decay" in row.index:
                latest_xg_against_decay_home[str(row["home"])] = float(row.get("xg_against_form_H_decay", 0.0))
        except Exception as e:
            logger.debug(f"xG home stats error for {row['home']}: {e}")
        try:
            latest_xg_for_away[str(row["away"])] = float(row.get("_xg_for_form_A_post", 0.0))
            latest_xg_against_away[str(row["away"])] = float(row.get("_xg_against_form_A_post", 0.0))
            latest_xg_diff_away[str(row["away"])] = float(row.get("_xg_diff_form_A_post", 0.0))
            latest_xg_shot_quality_away[str(row["away"])] = float(row.get("_xg_shot_quality_form_A_post", 0.0))
            latest_xg_finishing_luck_away[str(row["away"])] = float(row.get("_xg_finishing_luck_form_A_post", 0.0))
            latest_xg_n_away[str(row["away"])] = float(row.get("_xg_n_A_post", 0.0))
            latest_xg_stats_n_away[str(row["away"])] = float(row.get("_xg_stats_n_A_post", 0.0))
            # Milestone 9: xG decay features
            if "xg_for_form_A_decay" in row.index:
                latest_xg_for_decay_away[str(row["away"])] = float(row.get("xg_for_form_A_decay", 0.0))
            if "xg_against_form_A_decay" in row.index:
                latest_xg_against_decay_away[str(row["away"])] = float(row.get("xg_against_form_A_decay", 0.0))
        except Exception as e:
            logger.debug(f"xG away stats error for {row['away']}: {e}")

    # PATH DETERMINATION for upcoming fixtures
    # Use allratingv.csv for future fixtures; optionally enrich with gg/ng from upcoming.csv.
    base_dir = Path(args.data_dir)
    multi_league_dir = base_dir / "multiple leagues and seasons"

    odds_source = None
    if multi_league_dir.exists():
        allrating_csv = multi_league_dir / "allratingv.csv"
        upcoming_csv = multi_league_dir / "upcoming.csv"
        if allrating_csv.exists():
            primary_source = allrating_csv
            logger.info(f"[SOURCE] Using primary source: {primary_source}")
            if upcoming_csv.exists():
                odds_source = upcoming_csv
                logger.info(f"[SOURCE] Will enrich odds from: {odds_source}")
        elif upcoming_csv.exists():
            primary_source = upcoming_csv
            logger.info(f"[SOURCE] Using fallback source: {primary_source}")
        else:
            primary_source = base_dir / "multiple seasons.csv"
            logger.info(f"[SOURCE] Using legacy fallback source: {primary_source}")
    else:
        # Legacy single-league fallback
        primary_source = base_dir / "multiple seasons.csv"
        logger.info(f"[SOURCE] Using fallback source: {primary_source}")
    
    # Load upcoming data directly from the primary source (contains future fixtures)
    if primary_source.exists():
        try:
            up = pd.read_csv(primary_source, encoding="latin1", low_memory=False)
            logger.info(f"[SOURCE] Loaded {len(up)} total rows from {primary_source.name}")
            
            # Parse dates to identify future fixtures
            up["_fixture_dt"] = up.apply(
                lambda r: _parse_upcoming_datetime(r.get("datameci"), r.get("orameci")),
                axis=1
            )
            
            # Filter to only future fixtures (after run_asof_datetime)
            total_rows = len(up)
            up = up[up["_fixture_dt"] > run_asof_datetime].copy()
            logger.info(f"[SOURCE] Filtered to {len(up)} future fixtures (from {total_rows} total, after {run_asof_datetime.date()})")
            
        except Exception as e:
            logger.error(f"[SOURCE] Failed to load {primary_source}: {e}")
            up = pd.DataFrame()
    else:
        logger.warning(f"[SOURCE] Primary source not found: {primary_source}")
        up = pd.DataFrame()

    # Build fixture_datetime for scope filtering
    scope_counts: dict[str, object] = {"upcoming_rows_in": int(len(up))}
    scope_counts["upcoming_rows_parsed"] = int(up["_fixture_dt"].notna().sum())
    up = up[up["_fixture_dt"].notna()].copy()
    up["fixture_datetime"] = up["_fixture_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Optional enrichment: pull gg/ng (BTTS odds) from upcoming.csv for the same fixtures.
    if odds_source is not None and odds_source.exists():
        try:
            up_odds = pd.read_csv(odds_source, encoding="latin1", low_memory=False)
            up_odds["_fixture_dt"] = up_odds.apply(
                lambda r: _parse_upcoming_datetime(r.get("datameci"), r.get("orameci")),
                axis=1,
            )
            up_odds = up_odds[up_odds["_fixture_dt"].notna()].copy()
            up_odds = up_odds[up_odds["_fixture_dt"] > run_asof_datetime].copy()

            def _norm_code_col(df: pd.DataFrame, col: str) -> None:
                if col in df.columns:
                    s = df[col].astype(str).str.strip()
                    s = s.str.replace(r"\.0$", "", regex=True)
                    df[col] = s

            for c in ("codechipa1", "codechipa2"):
                _norm_code_col(up, c)
                _norm_code_col(up_odds, c)

            up["_date_only"] = up["_fixture_dt"].dt.date
            up_odds["_date_only"] = up_odds["_fixture_dt"].dt.date

            if {"codechipa1", "codechipa2"}.issubset(up.columns) and {"codechipa1", "codechipa2"}.issubset(up_odds.columns):
                merge_keys = ["_date_only", "codechipa1", "codechipa2"]
            else:
                merge_keys = ["_date_only", "txtechipa1", "txtechipa2"]

            cols = [c for c in ["gg", "ng", "cotao", "cotau", "cotaa", "cotae", "cotad"] if c in up_odds.columns]
            if cols:
                up = up.merge(
                    up_odds[merge_keys + cols],
                    how="left",
                    on=merge_keys,
                    suffixes=("", "_odds"),
                )
                for c in cols:
                    if f"{c}_odds" in up.columns:
                        if c in up.columns:
                            up[c] = up[c].combine_first(up[f"{c}_odds"])
                        else:
                            up[c] = up[f"{c}_odds"]
                        up.drop(columns=[f"{c}_odds"], inplace=True, errors="ignore")
            up.drop(columns=["_date_only"], inplace=True, errors="ignore")
        except Exception as e:
            logger.warning(f"[SOURCE] Failed to enrich odds from {odds_source}: {e}")

    # Apply deterministic live scope filters with step-by-step counts.
    up = up[up["_fixture_dt"] > run_asof_datetime].copy()
    scope_counts["upcoming_rows_after_past"] = int(len(up))

    if scope_season_start is not None and scope_season_end is not None:
        up = up[up["_fixture_dt"].between(scope_season_start, scope_season_end, inclusive="left")].copy()
    scope_counts["upcoming_rows_after_window"] = int(len(up))

    if scope_league:
        up = up[up["league"].astype(str) == scope_league].copy()
    if scope_country and "country" in up.columns:
        up = up[up["country"].astype(str) == scope_country].copy()
    scope_counts["upcoming_rows_after_league"] = int(len(up))

    if horizon_days > 0:
        horizon_end = run_asof_datetime + pd.Timedelta(days=int(horizon_days))
        up = up[up["_fixture_dt"] <= horizon_end].copy()
    scope_counts["upcoming_rows_after_horizon"] = int(len(up))
    # Dedupe fixtures (prefer rows with more odds info when duplicates exist).
    dedupe_cols = ["_fixture_dt"]
    if "league" in up.columns:
        dedupe_cols.append("league")
    if {"codechipa1", "codechipa2"}.issubset(up.columns):
        dedupe_cols.extend(["codechipa1", "codechipa2"])
    elif {"txtechipa1", "txtechipa2"}.issubset(up.columns):
        dedupe_cols.extend(["txtechipa1", "txtechipa2"])
    if len(dedupe_cols) > 1:
        score_cols = [c for c in ["cotaa", "cotae", "cotad", "cotao", "cotau", "gg", "ng"] if c in up.columns]
        if score_cols:
            up["_dupe_score"] = up[score_cols].notna().sum(axis=1)
            up = up.sort_values(dedupe_cols + ["_dupe_score"], ascending=[True] * len(dedupe_cols) + [False], kind="mergesort")
        before = len(up)
        up = up.drop_duplicates(subset=dedupe_cols, keep="first")
        scope_counts["upcoming_rows_after_dedupe"] = int(len(up))
        if len(up) != before:
            logger.info("[SCOPE] deduped %d duplicate fixtures", int(before - len(up)))
        up.drop(columns=["_dupe_score"], inplace=True, errors="ignore")
    else:
        scope_counts["upcoming_rows_after_dedupe"] = int(len(up))
    scope_counts["upcoming_rows_after_filters"] = int(len(up))

    logger.info(
        "[SCOPE] asof=%s run_asof_datetime=%s rows_in=%s parsed=%s after_past=%s after_window=%s after_league=%s after_horizon=%s after_dedupe=%s",
        str(as_of_date),
        str(run_asof_datetime),
        scope_counts.get("upcoming_rows_in"),
        scope_counts.get("upcoming_rows_parsed"),
        scope_counts.get("upcoming_rows_after_past"),
        scope_counts.get("upcoming_rows_after_window"),
        scope_counts.get("upcoming_rows_after_league"),
        scope_counts.get("upcoming_rows_after_horizon"),
        scope_counts.get("upcoming_rows_after_dedupe"),
    )
    log_json(
        {
            "event": "UPCOMING_SCOPE",
            "as_of_date": str(as_of_date),
            "run_asof_datetime": str(run_asof_datetime),
            "scope_country": scope_country,
            "scope_league": scope_league,
            "scope_season_start": str(scope_season_start) if scope_season_start is not None else None,
            "scope_season_end": str(scope_season_end) if scope_season_end is not None else None,
            "horizon_days": int(horizon_days),
            **scope_counts,
        },
        log_path=Path(args.log_json),
    )

    # ------------------------------------------------------------------
    # Milestone 7.2: Goal timing profiles (minute-goals from AGS export)
    # ------------------------------------------------------------------
    timing_profiles = {}
    timing_meta = None
    ags_path = Path(args.data_dir) / "AGS.CSV"
    if ags_path.exists():
        def _norm_team(v: object) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            s = str(v).strip()
            if not s or s.lower() == "nan":
                return ""
            return normalize_team(s, reg)

        try:
            timing_profiles, timing_meta = build_team_timing_profiles(
                ags_path,
                normalize=_norm_team,
                run_asof_datetime=run_asof_datetime,
                scope_league=scope_league,
            )
            logger.info(
                "[TIMING] source=%s rows_in=%d rows_used=%d goals=%d teams=%d prior_early=%.3f prior_late=%.3f",
                str(ags_path),
                getattr(timing_meta, "rows_in", 0),
                getattr(timing_meta, "rows_used", 0),
                getattr(timing_meta, "goals_total", 0),
                len(timing_profiles),
                float(getattr(timing_meta, "prior_early_share", 0.0) or 0.0),
                float(getattr(timing_meta, "prior_late_share", 0.0) or 0.0),
            )
            log_json(
                {
                    "event": "TIMING_META",
                    "source": str(ags_path),
                    "rows_in": int(getattr(timing_meta, "rows_in", 0) or 0),
                    "rows_used": int(getattr(timing_meta, "rows_used", 0) or 0),
                    "goals_total": int(getattr(timing_meta, "goals_total", 0) or 0),
                    "teams": int(len(timing_profiles)),
                    "prior_early_share": float(getattr(timing_meta, "prior_early_share", np.nan)),
                    "prior_late_share": float(getattr(timing_meta, "prior_late_share", np.nan)),
                },
                log_path=Path(args.log_json),
            )
        except Exception as e:
            logger.warning("[TIMING] failed to load %s (%s); timing features disabled", str(ags_path), str(e))
            timing_profiles = {}
            timing_meta = None
    else:
        logger.info("[TIMING] AGS.CSV not found; timing features disabled")

    model_variant = str(getattr(args, "model_variant", "full") or "full").strip() or "full"
    suffix = "" if model_variant == "full" else f"_{model_variant}"
    model_h_path = Path(args.models_dir) / f"frankenstein_mu_home{suffix}.pkl"
    model_a_path = Path(args.models_dir) / f"frankenstein_mu_away{suffix}.pkl"
    if not model_h_path.exists() or not model_a_path.exists():
        raise FileNotFoundError(
            f"Missing model files for variant='{model_variant}': {model_h_path} / {model_a_path}. "
            f"Train them via: python -m cgm.train_frankenstein_mu --variant {model_variant}"
        )
    model_h_hash = file_hash(model_h_path)
    model_a_hash = file_hash(model_a_path)
    model_h = joblib.load(model_h_path)
    model_a = joblib.load(model_a_path)

    # Milestone 13: Load Calibration Models
    calib_models = {}
    for name in ["home", "away", "over", "btts"]:
        cp = Path(args.models_dir) / "calibration" / f"calib_{name}.pkl"
        if cp.exists():
            try:
                calib_models[name] = joblib.load(cp)
                logger.info(f"[CALIB] Loaded {name} calibrator from {cp}")
            except Exception as e:
                logger.warning(f"[CALIB] Failed to load {name} calibrator: {e}")


    # Feature contract audit
    feat_cols_h = getattr(model_h, "feature_names_in_", None)
    if feat_cols_h is None or len(feat_cols_h) == 0:
        feat_cols_h = model_h.get_booster().feature_names
    feat_cols_a = getattr(model_a, "feature_names_in_", None)
    if feat_cols_a is None or len(feat_cols_a) == 0:
        feat_cols_a = model_a.get_booster().feature_names
    feat_cols_h = list(feat_cols_h)
    feat_cols_a = list(feat_cols_a)
    common_feats = sorted(set(feat_cols_h) | set(feat_cols_a))

    # Poison-pill: inference must never require raw post-match truth/current-match stats.
    banned_exact = {
        "ft_home", "ft_away", "ht_home", "ht_away", "result", "validated",
        "shots", "shots_on_target", "corners", "possession_home", "possession_away",
        "shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A",
        # Milestone 2/3 internal helpers and current-match derived xG proxy columns.
        "xg_proxy_H", "xg_proxy_A", "xg_usable",
        "shot_quality_H", "shot_quality_A", "finishing_luck_H", "finishing_luck_A",
    }
    banned_prefixes = ("_press_", "_xg_")
    banned_in_model = sorted([c for c in common_feats if (c in banned_exact) or c.startswith(banned_prefixes)])
    if banned_in_model:
        raise AssertionError(f"Banned features in model contract: {banned_in_model}")
    logger.info("[ELO][FEATS] model features home=%d away=%d union=%d", len(feat_cols_h), len(feat_cols_a), len(common_feats))
    log_json(
        {
            "event": "ELO_FEATS",
            "model_home": str(model_h_path),
            "model_home_hash": model_h_hash,
            "model_away": str(model_a_path),
            "model_away_hash": model_a_hash,
            "model_variant": model_variant,
            "feat_home": len(feat_cols_h),
            "feat_away": len(feat_cols_a),
            "feat_union": len(common_feats),
        },
        log_path=Path(args.log_json),
    )

    preds = []
    skipped = []
    neff_h_list = []
    neff_a_list = []
    trace_count = 0
    trace_path = Path(args.trace_json)
    for _, r in up.iterrows():
        home_raw = r["txtechipa1"]
        away_raw = r["txtechipa2"]
        home = normalize_team(home_raw, reg)
        away = normalize_team(away_raw, reg)

        if home not in latest_home.index or away not in latest_away.index:
            # Log which team is unseen for debugging
            if home not in latest_home.index:
                logger.warning("[UNSEEN] home team '%s' (raw: '%s') not in history - using defaults", home, home_raw)
            if away not in latest_away.index:
                logger.warning("[UNSEEN] away team '%s' (raw: '%s') not in history - using defaults", away, away_raw)
            skipped.append((home, away, "missing_snapshot"))
            continue
        if counts.get(home, 0) < args.min_matches or counts.get(away, 0) < args.min_matches:
            # Log low-history teams
            if counts.get(home, 0) < args.min_matches:
                logger.warning("[LOW_HISTORY] home '%s' has only %d matches (min=%d)", home, counts.get(home, 0), args.min_matches)
            if counts.get(away, 0) < args.min_matches:
                logger.warning("[LOW_HISTORY] away '%s' has only %d matches (min=%d)", away, counts.get(away, 0), args.min_matches)
            skipped.append((home, away, "insufficient_history"))
            continue

        meta_home = latest_team_meta.get(home, {})
        meta_away = latest_team_meta.get(away, {})
        league = r.get("league") or meta_home.get("league") or meta_away.get("league")
        country = r.get("country") or meta_home.get("country") or meta_away.get("country")
        season = r.get("sezonul") or meta_home.get("season") or meta_away.get("season")

        home_bonus = league_meta.get(league, {}).get("home_bonus_elo", 0.0)
        try:
            home_bonus_f = float(home_bonus) if home_bonus is not None else 0.0
        except Exception as e:
            logger.debug(f"Home bonus error for {league}: {e}")
            home_bonus_f = 0.0

        elo_home = latest_elos.get(home)
        elo_away = latest_elos.get(away)
        if elo_home is None or elo_away is None:
            skipped.append((home, away, "missing_elo"))
            continue
        elo_diff = elo_home - elo_away
        elo_diff_bonus = elo_diff + home_adv_used
        band_h = _band_from_diff(elo_diff_bonus)
        band_a = _band_from_diff(-elo_diff_bonus)

        snap_h = latest_home.loc[home]
        snap_a = latest_away.loc[away]

        # Milestone 2: Pressure Cooker snapshot + divergence (league-standardized z-scores)
        press_form_h = float(latest_press_home.get(home, 0.5))
        press_form_a = float(latest_press_away.get(away, 0.5))
        press_n_h = float(latest_press_n_home.get(home, 0.0))
        press_n_a = float(latest_press_n_away.get(away, 0.0))
        press_stats_n_h = float(latest_press_stats_n_home.get(home, 0.0))
        press_stats_n_a = float(latest_press_stats_n_away.get(away, 0.0))
        dom_h = latest_press_dom_home.get(home, {})
        dom_a = latest_press_dom_away.get(away, {})
        pressure_usable = int(bool(pressure_usable_home_ctx.get(home, False)) and bool(pressure_usable_away_ctx.get(away, False)))

        league_teams = [t for t, m in latest_team_meta.items() if m.get("league") == league] if league else []
        if not league_teams:
            league_teams = list(latest_team_meta.keys())
        elo_vals = np.array([latest_elos.get(t, np.nan) for t in league_teams], dtype=float)
        ph_vals = np.array([latest_press_home.get(t, np.nan) for t in league_teams], dtype=float)
        pa_vals = np.array([latest_press_away.get(t, np.nan) for t in league_teams], dtype=float)
        elo_mean, elo_std = float(np.nanmean(elo_vals)), float(np.nanstd(elo_vals, ddof=0))
        ph_mean, ph_std = float(np.nanmean(ph_vals)), float(np.nanstd(ph_vals, ddof=0))
        pa_mean, pa_std = float(np.nanmean(pa_vals)), float(np.nanstd(pa_vals, ddof=0))

        press_z_h = 0.0 if ph_std <= 0 or np.isnan(ph_std) else float((press_form_h - ph_mean) / ph_std)
        press_z_a = 0.0 if pa_std <= 0 or np.isnan(pa_std) else float((press_form_a - pa_mean) / pa_std)
        elo_z_h = 0.0 if elo_std <= 0 or np.isnan(elo_std) else float((elo_home - elo_mean) / elo_std)
        elo_z_a = 0.0 if elo_std <= 0 or np.isnan(elo_std) else float((elo_away - elo_mean) / elo_std)
        div_team_h = press_z_h - elo_z_h
        div_team_a = press_z_a - elo_z_a
        div_diff = div_team_h - div_team_a

        # Milestone 3: xG-proxy Sniper snapshot + Pressure-vs-xG disagreement
        xg_for_h = float(latest_xg_for_home.get(home, 0.0))
        xg_against_h = float(latest_xg_against_home.get(home, 0.0))
        xg_diff_h = float(latest_xg_diff_home.get(home, 0.0))
        xg_shot_quality_h = float(latest_xg_shot_quality_home.get(home, 0.0))
        xg_finishing_luck_h = float(latest_xg_finishing_luck_home.get(home, 0.0))
        xg_n_h = float(latest_xg_n_home.get(home, 0.0))
        xg_stats_n_h = float(latest_xg_stats_n_home.get(home, 0.0))

        xg_for_a = float(latest_xg_for_away.get(away, 0.0))
        xg_against_a = float(latest_xg_against_away.get(away, 0.0))
        xg_diff_a = float(latest_xg_diff_away.get(away, 0.0))
        xg_shot_quality_a = float(latest_xg_shot_quality_away.get(away, 0.0))
        xg_finishing_luck_a = float(latest_xg_finishing_luck_away.get(away, 0.0))
        xg_n_a = float(latest_xg_n_away.get(away, 0.0))
        xg_stats_n_a = float(latest_xg_stats_n_away.get(away, 0.0))

        xg_usable = int(bool(xg_usable_home_ctx.get(home, False)) and bool(xg_usable_away_ctx.get(away, False)))

        # League-standardize xG diff using only teams with any xG evidence in the relevant context.
        xh_vals = np.array(
            [latest_xg_diff_home.get(t, np.nan) for t in league_teams if float(latest_xg_stats_n_home.get(t, 0.0)) > 0.0],
            dtype=float,
        )
        xa_vals = np.array(
            [latest_xg_diff_away.get(t, np.nan) for t in league_teams if float(latest_xg_stats_n_away.get(t, 0.0)) > 0.0],
            dtype=float,
        )
        xh_mean = float(np.nanmean(xh_vals)) if xh_vals.size else float("nan")
        xh_std = float(np.nanstd(xh_vals, ddof=0)) if xh_vals.size else float("nan")
        xa_mean = float(np.nanmean(xa_vals)) if xa_vals.size else float("nan")
        xa_std = float(np.nanstd(xa_vals, ddof=0)) if xa_vals.size else float("nan")

        xg_z_h = 0.0
        if xg_stats_n_h > 0.0 and xh_std > 0 and not np.isnan(xh_std):
            xg_z_h = float((xg_diff_h - xh_mean) / xh_std) if not np.isnan(xh_mean) else 0.0
        xg_z_a = 0.0
        if xg_stats_n_a > 0.0 and xa_std > 0 and not np.isnan(xa_std):
            xg_z_a = float((xg_diff_a - xa_mean) / xa_std) if not np.isnan(xa_mean) else 0.0

        div_px_team_h = press_z_h - xg_z_h
        div_px_team_a = press_z_a - xg_z_a
        div_px_diff = div_px_team_h - div_px_team_a

        sterile_h = int((xg_stats_n_h > 0.0) and (press_z_h >= 1.0) and (xg_z_h <= -1.0))
        sterile_a = int((xg_stats_n_a > 0.0) and (press_z_a >= 1.0) and (xg_z_a <= -1.0))
        assassin_h = int((xg_stats_n_h > 0.0) and (press_z_h <= -1.0) and (xg_z_h >= 1.0))
        assassin_a = int((xg_stats_n_a > 0.0) and (press_z_a <= -1.0) and (xg_z_a >= 1.0))

        odds_home = r.get("cotaa")
        odds_draw = r.get("cotae")
        odds_away = r.get("cotad")
        odds_over = r.get("cotao")
        odds_under = r.get("cotau")
        imp_1x2 = _implied_probs_1x2(odds_home, odds_draw, odds_away)
        imp_ou = _implied_probs_two_way(odds_over, odds_under)
        if model_variant == "no_odds":
            odds_home = odds_draw = odds_away = np.nan
            odds_over = odds_under = np.nan
            imp_1x2 = {"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan}
            imp_ou = {"p_over": np.nan, "p_under": np.nan}

        # Elo similarity features (kernel-based, venue-aware) with cutoff at fixture datetime
        fixture_dt = pd.to_datetime(r.get("_fixture_dt"), errors="coerce")
        as_of = fixture_dt if not pd.isna(fixture_dt) else pd.to_datetime(r.get("datameci"), errors="coerce")
        sigma = float(sigma_map.get(str(league), default_sigma))
        h_hist = elo_histories.get(home, {}).get("home", pd.DataFrame())
        a_hist = elo_histories.get(away, {}).get("away", pd.DataFrame())
        gf_h, ga_h, w_h, n_h = kernel_similarity(h_hist, elo_away, sigma, as_of=as_of, inclusive=False)
        gf_a, ga_a, w_a, n_a = kernel_similarity(a_hist, elo_home, sigma, as_of=as_of, inclusive=False)

        feat_cols = feat_cols_h  # assume same feature set for home/away models

        band_home_stats = band_stats.get(home, {}).get("home", {}).get(band_h, {})
        band_away_stats = band_stats.get(away, {}).get("away", {}).get(band_a, {})

        feats: dict[str, float | str | None] = {
            "season": season,
            "country": country,
            "league": league,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_diff,
            "home_bonus_elo": home_bonus_f,
            "EloDiff": elo_diff_bonus,
            "Band_H": band_h,
            "Band_A": band_a,
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "odds_over": odds_over,
            "odds_under": odds_under,
            # Training data stores these as percentages (0..100). Keep inference on the same scale.
            "p_home": (100.0 * imp_1x2["p_home"]) if imp_1x2["p_home"] is not None and not np.isnan(imp_1x2["p_home"]) else np.nan,
            "p_draw": (100.0 * imp_1x2["p_draw"]) if imp_1x2["p_draw"] is not None and not np.isnan(imp_1x2["p_draw"]) else np.nan,
            "p_away": (100.0 * imp_1x2["p_away"]) if imp_1x2["p_away"] is not None and not np.isnan(imp_1x2["p_away"]) else np.nan,
            "fair_home": (1 / imp_1x2["p_home"]) if imp_1x2["p_home"] and not np.isnan(imp_1x2["p_home"]) else np.nan,
            "fair_draw": (1 / imp_1x2["p_draw"]) if imp_1x2["p_draw"] and not np.isnan(imp_1x2["p_draw"]) else np.nan,
            "fair_away": (1 / imp_1x2["p_away"]) if imp_1x2["p_away"] and not np.isnan(imp_1x2["p_away"]) else np.nan,
            "p_over": (100.0 * imp_ou["p_over"]) if imp_ou["p_over"] is not None and not np.isnan(imp_ou["p_over"]) else np.nan,
            "p_under": (100.0 * imp_ou["p_under"]) if imp_ou["p_under"] is not None and not np.isnan(imp_ou["p_under"]) else np.nan,
            "fair_over": (1 / imp_ou["p_over"]) if imp_ou["p_over"] and not np.isnan(imp_ou["p_over"]) else np.nan,
            "fair_under": (1 / imp_ou["p_under"]) if imp_ou["p_under"] and not np.isnan(imp_ou["p_under"]) else np.nan,
            "form_home": pd.to_numeric(r.get("formah"), errors="coerce"),
            "form_away": pd.to_numeric(r.get("formaa"), errors="coerce"),
            "H_gf_vs_band": band_home_stats.get("gf", np.nan),
            "H_ga_vs_band": band_home_stats.get("ga", np.nan),
            "A_gf_vs_band": band_away_stats.get("gf", np.nan),
            "A_ga_vs_band": band_away_stats.get("ga", np.nan),
            "GFvsSim_H": gf_h,
            "GAvsSim_H": ga_h,
            "GFvsSim_A": gf_a,
            "GAvsSim_A": ga_a,
            "wsum_sim_H": w_h,
            "wsum_sim_A": w_a,
            "neff_sim_H": n_h,
            "neff_sim_A": n_a,
            # Milestone 2: Pressure Cooker
            "press_form_H": press_form_h,
            "press_form_A": press_form_a,
            "press_n_H": press_n_h,
            "press_n_A": press_n_a,
            "press_stats_n_H": press_stats_n_h,
            "press_stats_n_A": press_stats_n_a,
            "press_dom_shots_H": float(dom_h.get("shots", 0.5)),
            "press_dom_sot_H": float(dom_h.get("sot", 0.5)),
            "press_dom_corners_H": float(dom_h.get("corners", 0.5)),
            "press_dom_pos_H": float(dom_h.get("pos", 0.5)),
            "press_dom_shots_A": float(dom_a.get("shots", 0.5)),
            "press_dom_sot_A": float(dom_a.get("sot", 0.5)),
            "press_dom_corners_A": float(dom_a.get("corners", 0.5)),
            "press_dom_pos_A": float(dom_a.get("pos", 0.5)),
            "press_z_H": press_z_h,
            "press_z_A": press_z_a,
            "elo_z_H": elo_z_h,
            "elo_z_A": elo_z_a,
            "div_team_H": div_team_h,
            "div_team_A": div_team_a,
            "div_diff": div_diff,
            "pressure_usable": pressure_usable,
            # Milestone 3: xG-proxy Sniper
            "xg_for_form_H": xg_for_h,
            "xg_against_form_H": xg_against_h,
            "xg_diff_form_H": xg_diff_h,
            "xg_shot_quality_form_H": xg_shot_quality_h,
            "xg_finishing_luck_form_H": xg_finishing_luck_h,
            "xg_n_H": xg_n_h,
            "xg_stats_n_H": xg_stats_n_h,
            "xg_for_form_A": xg_for_a,
            "xg_against_form_A": xg_against_a,
            "xg_diff_form_A": xg_diff_a,
            "xg_shot_quality_form_A": xg_shot_quality_a,
            "xg_finishing_luck_form_A": xg_finishing_luck_a,
            "xg_n_A": xg_n_a,
            "xg_stats_n_A": xg_stats_n_a,
            "xg_z_H": xg_z_h,
            "xg_z_A": xg_z_a,
            "div_px_team_H": div_px_team_h,
            "div_px_team_A": div_px_team_a,
            "div_px_diff": div_px_diff,
            "sterile_H": sterile_h,
            "sterile_A": sterile_a,
            "assassin_H": assassin_h,
            "assassin_A": assassin_a,
            # Milestone 9: Decay features
            "press_form_H_decay": float(latest_press_decay_home.get(home, 0.5)),
            "press_form_A_decay": float(latest_press_decay_away.get(away, 0.5)),
            "xg_for_form_H_decay": float(latest_xg_for_decay_home.get(home, 0.0)),
            "xg_against_form_H_decay": float(latest_xg_against_decay_home.get(home, 0.0)),
            "xg_for_form_A_decay": float(latest_xg_for_decay_away.get(away, 0.0)),
            "xg_against_form_A_decay": float(latest_xg_against_decay_away.get(away, 0.0)),
        }

        # Milestone 11: League-Specific Features (scoring patterns per competition)
        league_features_enabled = getattr(config, "LEAGUE_FEATURES_ENABLED", True)
        if league_features_enabled:
            lg_feats = get_league_features_for_fixture(
                hist, country=str(country or ""), league=str(league or ""), as_of=as_of
            )
            feats.update(lg_feats)

        # Milestone 10: Head-to-Head Features (direct matchup patterns)
        h2h_enabled = getattr(config, "H2H_ENABLED", True)
        if h2h_enabled:
            h2h_feats = get_h2h_features_for_fixture(
                hist, home=home, away=away, as_of_datetime=as_of
            )
            feats.update(h2h_feats)


        for w in args.windows:
            feats[f"H_gf_L{w}"] = snap_h.get(f"H_gf_L{w}", np.nan)
            feats[f"H_ga_L{w}"] = snap_h.get(f"H_ga_L{w}", np.nan)
            feats[f"H_sot_for_L{w}"] = snap_h.get(f"H_sot_for_L{w}", np.nan)
            feats[f"H_cor_for_L{w}"] = snap_h.get(f"H_cor_for_L{w}", np.nan)
            feats[f"H_poss_for_L{w}"] = snap_h.get(f"H_poss_for_L{w}", np.nan)

            feats[f"A_gf_L{w}"] = snap_a.get(f"A_gf_L{w}", np.nan)
            feats[f"A_ga_L{w}"] = snap_a.get(f"A_ga_L{w}", np.nan)
            feats[f"A_sot_for_L{w}"] = snap_a.get(f"A_sot_for_L{w}", np.nan)
            feats[f"A_cor_for_L{w}"] = snap_a.get(f"A_cor_for_L{w}", np.nan)
            feats[f"A_poss_for_L{w}"] = snap_a.get(f"A_poss_for_L{w}", np.nan)

            feats[f"H_shot_quality_L{w}"] = snap_h.get(f"H_shot_quality_L{w}", np.nan)
            feats[f"H_finish_rate_L{w}"] = snap_h.get(f"H_finish_rate_L{w}", np.nan)
            feats[f"A_shot_quality_L{w}"] = snap_a.get(f"A_shot_quality_L{w}", np.nan)
            feats[f"A_finish_rate_L{w}"] = snap_a.get(f"A_finish_rate_L{w}", np.nan)

        feats["Attack_H"] = snap_h.get("Attack_H", np.nan)
        feats["Defense_H"] = snap_h.get("Defense_H", np.nan)
        feats["Attack_A"] = snap_a.get("Attack_A", np.nan)
        feats["Defense_A"] = snap_a.get("Defense_A", np.nan)
        feats["Expected_Destruction_H"] = snap_h.get("Expected_Destruction_H", np.nan)
        feats["Expected_Destruction_A"] = snap_a.get("Expected_Destruction_A", np.nan)

        lg_defaults = league_meta.get(league, {})
        for c in league_avg_cols:
            feats[c] = snap_h.get(c, snap_a.get(c, lg_defaults.get(c, np.nan)))

        feat_cols = list(feat_cols)
        row_dict = {c: feats.get(c, np.nan) for c in feat_cols}
        missing = [c for c in feat_cols if c not in feats]
        if missing:
            logger.error("[ELO][FEATS] missing features for %s vs %s: %s", home, away, missing)
            raise AssertionError(f"Missing features for {home} vs {away}: {missing}")
        X_row = pd.DataFrame([row_dict])
        X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        required_feats = [f"H_gf_L{args.windows[0]}", f"A_gf_L{args.windows[0]}"]
        if any(pd.isna(row_dict.get(f)) for f in required_feats):
            skipped.append((home, away, "missing_core_features"))
            continue

        mu_h = float(model_h.predict(X_row)[0])
        mu_a = float(model_a.predict(X_row)[0])
        probs = _poisson_probs(mu_h, mu_a)
        neff_h_list.append(n_h)
        neff_a_list.append(n_a)

        # Optional trace logging (sampled)
        do_trace = (args.log_sample_rate > 0 and random.random() < args.log_sample_rate and trace_count < args.log_max_fixtures)
        if do_trace:
            trace_count += 1
            log_json(
                {
                    "event": "ELO_PRED_TRACE",
                    "fixture": {"date": r.get("datameci"), "league": league, "home": home, "away": away},
                    "elo": {
                        "home": elo_home,
                        "away": elo_away,
                        "home_bonus": home_bonus_f,
                        "elo_diff": elo_diff,
                        "elo_diff_bonus": elo_diff_bonus,
                        "band_h": band_h,
                        "band_a": band_a,
                    },
                    "kernel": {
                        "sigma": sigma,
                        "as_of": str(as_of),
                        "inclusive": False,
                        "GFvsSim_H": gf_h,
                        "GAvsSim_H": ga_h,
                        "GFvsSim_A": gf_a,
                        "GAvsSim_A": ga_a,
                        "wsum_sim_H": w_h,
                        "wsum_sim_A": w_a,
                        "neff_sim_H": n_h,
                        "neff_sim_A": n_a,
                    },
                    "mu": {"home": mu_h, "away": mu_a},
                    "probs": probs,
                },
                log_path=trace_path,
            )

        # Milestone 13: Apply Probability Calibration
        # We modify the probabilities IN-PLACE before EV calculation
        if calib_models:
             try:
                 # 1. Home/Away (normalize afterward to sum to 1.0 - p_draw)
                 # Actually, strict calibration means we trust the calibrator.
                 # But we must respect the draw probabilty? 
                 # Isotonic maps p -> p_calib.
                 if "home" in calib_models:
                     probs["p_home"] = float(calib_models["home"].predict([probs["p_home"]])[0])
                 if "away" in calib_models:
                     probs["p_away"] = float(calib_models["away"].predict([probs["p_away"]])[0])
                 # Re-normalize 1X2 so they sum to 1?
                 # Or just recalibrate draw? We don't have draw calibrator.
                 # Let's adjust Draw to be 1 - H - A (if < 0, normalize)
                 total_prob = probs["p_home"] + probs["p_away"] + probs["p_draw"]
                 if total_prob != 1.0 and total_prob > 0:
                     scale = 1.0 / total_prob
                     probs["p_home"] *= scale
                     probs["p_away"] *= scale
                     probs["p_draw"] *= scale

                 # 2. Over 2.5
                 if "over" in calib_models:
                     probs["p_over25"] = float(calib_models["over"].predict([probs["p_over25"]])[0])
                     probs["p_under25"] = 1.0 - probs["p_over25"]

             except Exception as e:
                 logger.debug(f"[CALIB] Error applying calibration: {e}")

        def ev(prob, odds):
            try:
                return prob * float(odds) - 1.0
            except Exception as e:
                logger.debug(f"EV calc error: prob={prob}, odds={odds}, err={e}")
                return np.nan

        ev_over = ev(probs["p_over25"], odds_over)
        ev_under = ev(probs["p_under25"], odds_under)

        # BTTS (Yes/No) derived from mu under Poisson independence.
        # P(BTTS=Yes) = (1 - e^-mu_home) * (1 - e^-mu_away)
        try:
            p_btts_yes = float((1.0 - np.exp(-mu_h)) * (1.0 - np.exp(-mu_a)))
            p_btts_yes = float(np.clip(p_btts_yes, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"BTTS calc error for mu_h={mu_h}, mu_a={mu_a}: {e}")
            p_btts_yes = np.nan
        
        # Calibration for BTTS
        if "btts" in calib_models and np.isfinite(p_btts_yes):
            try:
                p_btts_yes = float(calib_models["btts"].predict([p_btts_yes])[0])
            except Exception:
                pass
                
        p_btts_no = (1.0 - p_btts_yes) if np.isfinite(p_btts_yes) else np.nan

        odds_btts_yes = r.get("gg")
        odds_btts_no = r.get("ng")
        ev_btts_yes = ev(p_btts_yes, odds_btts_yes)
        ev_btts_no = ev(p_btts_no, odds_btts_no)

        # Milestone 7.2: timing flags used for risk gating (no timing markets exported).
        timing = compute_match_timing(
            home=home,
            away=away,
            mu_home=mu_h,
            mu_away=mu_a,
            profiles=timing_profiles,
            meta=timing_meta,
        )
        timing_payload = {
            "timing_usable": int(timing.get("timing_usable", 0) or 0),
            "timing_home_usable": int(timing.get("timing_home_usable", 0) or 0),
            "timing_away_usable": int(timing.get("timing_away_usable", 0) or 0),
            "timing_home_matches": int(timing.get("timing_home_matches", 0) or 0),
            "timing_away_matches": int(timing.get("timing_away_matches", 0) or 0),
            "timing_home_goals_scored": int(timing.get("timing_home_goals_scored", 0) or 0),
            "timing_home_goals_conceded": int(timing.get("timing_home_goals_conceded", 0) or 0),
            "timing_away_goals_scored": int(timing.get("timing_away_goals_scored", 0) or 0),
            "timing_away_goals_conceded": int(timing.get("timing_away_goals_conceded", 0) or 0),
            "timing_early_share": timing.get("timing_early_share", np.nan),
            "timing_late_share": timing.get("timing_late_share", np.nan),
            "slow_start_flag": int(timing.get("slow_start_flag", 0) or 0),
            "late_goal_flag": int(timing.get("late_goal_flag", 0) or 0),
        }

        preds.append(
            {
                "fixture_datetime": fixture_dt.isoformat() if not pd.isna(fixture_dt) else str(r.get("datameci")),
                "date": r.get("datameci"),
                "league": league,
                "country": country,
                "season": season,
                "home": home,
                "away": away,
                "model_variant": model_variant,
                # Scope metadata (copied into every row; deterministic audit trail)
                "run_asof_datetime": run_asof_datetime.isoformat(),
                "scope_country": scope_country or "",
                "scope_league": scope_league or "",
                "scope_season_start": scope_season_start.date().isoformat() if scope_season_start is not None else "",
                "scope_season_end": scope_season_end.date().isoformat() if scope_season_end is not None else "",
                "horizon_days": int(horizon_days),
                "upcoming_rows_in": int(scope_counts.get("upcoming_rows_in", 0) or 0),
                "upcoming_rows_parsed": int(scope_counts.get("upcoming_rows_parsed", 0) or 0),
                "upcoming_rows_after_past": int(scope_counts.get("upcoming_rows_after_past", 0) or 0),
                "upcoming_rows_after_window": int(scope_counts.get("upcoming_rows_after_window", 0) or 0),
                "upcoming_rows_after_league": int(scope_counts.get("upcoming_rows_after_league", 0) or 0),
                "upcoming_rows_after_horizon": int(scope_counts.get("upcoming_rows_after_horizon", 0) or 0),
                "upcoming_rows_after_dedupe": int(scope_counts.get("upcoming_rows_after_dedupe", 0) or 0),
                "upcoming_rows_after_filters": int(scope_counts.get("upcoming_rows_after_filters", 0) or 0),
                "mu_home": mu_h,
                "mu_away": mu_a,
                "mu_total": mu_h + mu_a,
                "p_over25": probs["p_over25"],
                "p_under25": probs["p_under25"],
                "p_over_2_5": probs["p_over25"],
                "p_under_2_5": probs["p_under25"],
                "odds_over25": odds_over,
                "odds_under25": odds_under,
                "odds_over_2_5": odds_over,
                "odds_under_2_5": odds_under,
                "EV_over25": ev_over,
                "EV_under25": ev_under,
                "p_btts_yes": p_btts_yes,
                "p_btts_no": p_btts_no,
                "odds_btts_yes": odds_btts_yes,
                "odds_btts_no": odds_btts_no,
                "EV_btts_yes": ev_btts_yes,
                "EV_btts_no": ev_btts_no,
                # Milestone 7.2 timing flags (risk gating only)
                **timing_payload,
                # Reliability/evidence (required by Milestone 4 pick engine)
                "neff_sim_H": n_h,
                "neff_sim_A": n_a,
                "press_stats_n_H": press_stats_n_h,
                "press_stats_n_A": press_stats_n_a,
                "xg_stats_n_H": xg_stats_n_h,
                "xg_stats_n_A": xg_stats_n_a,
                # Risk flags (Milestone 4 gates)
                "sterile_flag": int(bool(sterile_h or sterile_a)),
                "assassin_flag": int(bool(assassin_h or assassin_a)),
                # Directional flags (debuggable; not required)
                "sterile_H": sterile_h,
                "sterile_A": sterile_a,
                "assassin_H": assassin_h,
                "assassin_A": assassin_a,
            }
        )

    out_df = pd.DataFrame(preds)
    # Ensure a stable schema even when there are zero fixtures after scope filtering.
    if out_df.empty and len(out_df.columns) == 0:
        out_df = pd.DataFrame(
            columns=[
                "fixture_datetime",
                "date",
                "league",
                "country",
                "season",
                "home",
                "away",
                "model_variant",
                "run_asof_datetime",
                "scope_country",
                "scope_league",
                "scope_season_start",
                "scope_season_end",
                "horizon_days",
                "upcoming_rows_in",
                "upcoming_rows_parsed",
                "upcoming_rows_after_past",
                "upcoming_rows_after_window",
                "upcoming_rows_after_league",
                "upcoming_rows_after_horizon",
                "upcoming_rows_after_dedupe",
                "upcoming_rows_after_filters",
                "mu_home",
                "mu_away",
                "mu_total",
                "p_over25",
                "p_under25",
                "p_over_2_5",
                "p_under_2_5",
                "odds_over25",
                "odds_under25",
                "odds_over_2_5",
                "odds_under_2_5",
                "EV_over25",
                "EV_under25",
                "p_btts_yes",
                "p_btts_no",
                "odds_btts_yes",
                "odds_btts_no",
                "EV_btts_yes",
                "EV_btts_no",
                # Milestone 7.2 timing flags (risk gating only)
                "timing_usable",
                "timing_home_usable",
                "timing_away_usable",
                "timing_home_matches",
                "timing_away_matches",
                "timing_home_goals_scored",
                "timing_home_goals_conceded",
                "timing_away_goals_scored",
                "timing_away_goals_conceded",
                "timing_early_share",
                "timing_late_share",
                "slow_start_flag",
                "late_goal_flag",
                "neff_sim_H",
                "neff_sim_A",
                "press_stats_n_H",
                "press_stats_n_A",
                "xg_stats_n_H",
                "xg_stats_n_A",
                "sterile_flag",
                "assassin_flag",
                "sterile_H",
                "sterile_A",
                "assassin_H",
                "assassin_A",
            ]
        )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("[ELO][PRED] wrote upcoming predictions -> %s (rows=%d)", out_path, len(out_df))

    skipped_reasons = pd.Series([s[2] for s in skipped]).value_counts().to_dict() if skipped else {}
    mu_total = out_df["mu_home"] + out_df["mu_away"] if not out_df.empty else pd.Series(dtype=float)

    _log_run({"task": "predict_upcoming", "history": str(Path(args.history).resolve()),
              "upcoming": str(primary_source.resolve()),
              "model_variant": model_variant,
              "models": [str(model_h_path), str(model_a_path)],
              "windows": args.windows, "rows": len(out_df), "skipped": skipped[:10],
              "skipped_counts": skipped_reasons,
              "mu_total_min": float(mu_total.min()) if not mu_total.empty else None,
              "mu_total_max": float(mu_total.max()) if not mu_total.empty else None}, out_path)
    # Structured JSON log summary
    def _quantiles(vals: list[float]):
        s = pd.Series(vals) if vals else pd.Series(dtype=float)
        return {
            "min": float(s.min()) if not s.empty else None,
            "p10": float(s.quantile(0.10)) if not s.empty else None,
            "median": float(s.median()) if not s.empty else None,
            "p90": float(s.quantile(0.90)) if not s.empty else None,
            "max": float(s.max()) if not s.empty else None,
            "pct_lt_5": float((s < 5).mean()) if not s.empty else None,
        }

    log_json(
        {
            "event": "ELO_PRED_SUMMARY",
            "history": str(Path(args.history)),
            "history_hash": hist_hash,
            "upcoming": str(primary_source),
            "upcoming_hash": file_hash(primary_source) if primary_source.exists() else None,
            "models": {
                "home": str(model_h_path),
                "home_hash": model_h_hash,
                "away": str(model_a_path),
                "away_hash": model_a_hash,
            },
            "config": {
                "sigma_map": sigma_map,
                "default_sigma": default_sigma,
                "min_matches": args.min_matches,
                "windows": args.windows,
            },
            "rows": len(out_df),
            "skipped_counts": skipped_reasons,
            "mu_total_min": float(mu_total.min()) if not mu_total.empty else None,
            "mu_total_max": float(mu_total.max()) if not mu_total.empty else None,
            "neff_sim_H": _quantiles(neff_h_list),
            "neff_sim_A": _quantiles(neff_a_list),
        },
        log_path=Path(args.log_json),
    )


if __name__ == "__main__":
    main()
