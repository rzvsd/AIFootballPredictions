"""
Milestone 5: Live Inference & Value Scan.

Uses local normalized fixture/stats/odds data to build opponent-aware features,
predict mu_home/mu_away, and compute EV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import hashlib
import logging
import random

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
from cgm.strict_mu_engine import compute_strict_weighted_mu


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


def _load_history(path: Path, registry: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        dt = pd.to_datetime(df["date"] + " " + df.get("time", "").astype(str), errors="coerce")
        dt2 = pd.to_datetime(df["date"], errors="coerce")
        df["datetime"] = dt.fillna(dt2)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values(["datetime", "home", "away"], kind="mergesort")
    # Prefer code-based canonical names when available to avoid encoding drift.
    if registry and isinstance(registry, dict) and registry.get("code_to_name"):
        code_to_name = registry.get("code_to_name", {})
        code_home = df.get("code_home")
        if code_home is None and "codechipa1" in df.columns:
            code_home = df["codechipa1"]
        code_away = df.get("code_away")
        if code_away is None and "codechipa2" in df.columns:
            code_away = df["codechipa2"]

        def _map_code(series):
            if series is None:
                return None
            s = series.astype(str).str.strip()
            s = s.str.replace(r"\.0$", "", regex=True)
            s = s.replace({"nan": "", "None": "", "": ""})
            return s.map(code_to_name).replace({"": np.nan})

        home_from_code = _map_code(code_home)
        away_from_code = _map_code(code_away)
        if home_from_code is not None:
            df["home"] = home_from_code.combine_first(df["home"])
        if away_from_code is not None:
            df["away"] = away_from_code.combine_first(df["away"])
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
            if num is None or den is None:
                return np.nan
            num_f = float(num)
            den_f = float(den)
            if den_f == 0 or (not np.isfinite(den_f)) or (not np.isfinite(num_f)):
                return np.nan
            return num_f / den_f
        except Exception:
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


def _league_threshold(
    league_name: str,
    *,
    default_value: float,
    overrides: dict | None,
) -> float:
    """Resolve per-league threshold with bounded numeric fallback."""
    try:
        if isinstance(overrides, dict):
            raw = overrides.get(str(league_name))
            if raw is not None:
                v = float(raw)
                if np.isfinite(v):
                    return float(np.clip(v, 0.0, 1.0))
    except Exception:
        pass
    return float(np.clip(float(default_value), 0.0, 1.0))


def _league_scalar(
    league_name: str,
    *,
    default_value: float,
    overrides: dict | None,
    min_value: float,
    max_value: float,
) -> float:
    """Resolve per-league scalar with bounded numeric fallback."""
    try:
        if isinstance(overrides, dict):
            raw = overrides.get(str(league_name))
            if raw is not None:
                v = float(raw)
                if np.isfinite(v):
                    return float(np.clip(v, min_value, max_value))
    except Exception:
        pass
    return float(np.clip(float(default_value), min_value, max_value))


def _normalize_matrix(P: np.ndarray) -> np.ndarray:
    total = float(np.nansum(P))
    if not np.isfinite(total) or total <= 0:
        n = int(P.shape[0]) if P.ndim == 2 else 1
        return np.full((n, n), 1.0 / max(1, n * n), dtype=float)
    return P / total


def _max_goals_for_mu(mu_h: float, mu_a: float, floor: int = 10, cap: int = 14) -> int:
    mu_total = max(0.01, float(mu_h) + float(mu_a))
    dynamic = int(np.ceil(mu_total + 6.0 * np.sqrt(mu_total + 1.0)))
    return int(np.clip(max(floor, dynamic), floor, cap))


def _bivar_poisson_matrix(mu_h: float, mu_a: float, dep_strength: float, max_goals: int) -> np.ndarray:
    mu_h = max(0.01, float(mu_h))
    mu_a = max(0.01, float(mu_a))
    n = int(max_goals)
    dep_strength = float(np.clip(dep_strength, 0.0, 0.45))
    lam_c = float(dep_strength * min(mu_h, mu_a))
    lam_c = float(np.clip(lam_c, 0.0, max(0.0, min(mu_h, mu_a) - 1e-9)))
    lam_h = max(1e-9, mu_h - lam_c)
    lam_a = max(1e-9, mu_a - lam_c)

    pu = np.array([poisson.pmf(i, lam_h) for i in range(n + 1)], dtype=float)
    pv = np.array([poisson.pmf(i, lam_a) for i in range(n + 1)], dtype=float)
    pw = np.array([poisson.pmf(i, lam_c) for i in range(n + 1)], dtype=float)

    P = np.zeros((n + 1, n + 1), dtype=float)
    for w in range(n + 1):
        pw_w = float(pw[w])
        if pw_w <= 0:
            continue
        vmax = n - w
        pvw = pv[: vmax + 1]
        for u in range(n - w + 1):
            x = u + w
            pu_u = float(pu[u])
            if pu_u <= 0:
                continue
            P[x, w : vmax + w + 1] += pw_w * pu_u * pvw
    return _normalize_matrix(P)


def _dispersion_mixture_matrix(mu_h: float, mu_a: float, dep_strength: float, dispersion_alpha: float, max_goals: int) -> np.ndarray:
    # Symmetric two-point mean mixture preserving E[mu], inflating variance:
    # Var ~= mu + alpha * mu^2 via d = sqrt(alpha).
    alpha = float(np.clip(dispersion_alpha, 0.0, 0.60))
    if alpha <= 1e-12:
        return _bivar_poisson_matrix(mu_h, mu_a, dep_strength, max_goals)
    d = float(np.clip(np.sqrt(alpha), 0.0, 0.75))
    mu_h_low = max(0.01, float(mu_h) * (1.0 - d))
    mu_h_high = max(0.01, float(mu_h) * (1.0 + d))
    mu_a_low = max(0.01, float(mu_a) * (1.0 - d))
    mu_a_high = max(0.01, float(mu_a) * (1.0 + d))
    combos = [
        (mu_h_low, mu_a_low, 0.25),
        (mu_h_low, mu_a_high, 0.25),
        (mu_h_high, mu_a_low, 0.25),
        (mu_h_high, mu_a_high, 0.25),
    ]
    P = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for mh, ma, w in combos:
        P += float(w) * _bivar_poisson_matrix(mh, ma, dep_strength, max_goals)
    return _normalize_matrix(P)


def _apply_dixon_coles_low_score(P: np.ndarray, mu_h: float, mu_a: float, rho: float) -> np.ndarray:
    # Dixon-Coles style low-score adjustment on (0,0), (0,1), (1,0), (1,1).
    rho = float(np.clip(rho, -0.30, 0.30))
    if abs(rho) < 1e-12 or P.shape[0] < 2 or P.shape[1] < 2:
        return _normalize_matrix(P)
    T = np.ones_like(P, dtype=float)
    T[0, 0] = 1.0 - float(mu_h * mu_a * rho)
    T[0, 1] = 1.0 + float(mu_h * rho)
    T[1, 0] = 1.0 + float(mu_a * rho)
    T[1, 1] = 1.0 - float(rho)
    T = np.clip(T, 0.05, 5.0)
    return _normalize_matrix(P * T)


def _poisson_probs(
    mu_h: float,
    mu_a: float,
    *,
    max_goals: int | None = None,
    dep_strength: float = 0.0,
    dispersion_alpha: float = 0.0,
    dc_rho: float = 0.0,
) -> dict:
    n = _max_goals_for_mu(mu_h, mu_a) if max_goals is None else int(max(6, max_goals))
    P = _dispersion_mixture_matrix(
        mu_h=mu_h,
        mu_a=mu_a,
        dep_strength=dep_strength,
        dispersion_alpha=dispersion_alpha,
        max_goals=n,
    )
    P = _apply_dixon_coles_low_score(P, mu_h=mu_h, mu_a=mu_a, rho=dc_rho)
    home = np.tril(P, -1).sum()
    draw = np.diag(P).sum()
    away = np.triu(P, 1).sum()
    goal_grid = np.add.outer(range(n + 1), range(n + 1))
    ou25_over = P[goal_grid > 2].sum()
    ou25_under = 1.0 - ou25_over
    p_btts_yes = float(P[1:, 1:].sum())
    p_btts_no = float(1.0 - p_btts_yes)
    return {"p_home": float(home), "p_draw": float(draw), "p_away": float(away),
            "p_over25": float(ou25_over), "p_under25": float(ou25_under),
            "p_btts_yes": p_btts_yes, "p_btts_no": p_btts_no}


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
    ap = argparse.ArgumentParser(description="Predict upcoming fixtures using strict module mu engine")
    ap.add_argument("--data-dir", default="data/api_football", help="Data directory containing normalized fixture files")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv", help="Match history with features")
    ap.add_argument("--models-dir", default="models", help="Compatibility argument (ignored by strict mu engine)")
    ap.add_argument(
        "--model-variant",
        choices=["full", "no_odds", "no_lgavg"],
        default=str(getattr(config, "PIPELINE_DEFAULT_MODEL_VARIANT", "no_odds")),
        help="Prediction profile (odds usage toggles only; strict mu engine remains active)",
    )
    ap.add_argument("--out", default="reports/cgm_upcoming_predictions.csv", help="Output predictions CSV")
    ap.add_argument("--windows", nargs="+", type=int, default=[5, 10], help="Rolling windows")
    ap.add_argument(
        "--min-matches",
        type=int,
        default=int(getattr(config, "PIPELINE_MIN_MATCHES", 3) or 3),
        help="Minimum matches required per team to use prediction",
    )

    # Live scope (Milestone 4+). Defaults come from config.py; CLI overrides are deterministic.
    ap.add_argument("--as-of-date", default=None, help="Run as-of date (YYYY-MM-DD). Used to filter past fixtures strictly.")
    ap.add_argument("--scope-country", default=getattr(config, "LIVE_SCOPE_COUNTRY", ""), help="Optional country filter (empty disables).")
    ap.add_argument("--scope-league", default=getattr(config, "LIVE_SCOPE_LEAGUE", ""), help="Optional league filter (empty disables).")
    ap.add_argument("--scope-season-start", default=getattr(config, "LIVE_SCOPE_SEASON_START", ""), help="Optional season window start (YYYY-MM-DD).")
    ap.add_argument("--scope-season-end", default=getattr(config, "LIVE_SCOPE_SEASON_END", ""), help="Optional season window end (YYYY-MM-DD).")
    ap.add_argument("--horizon-days", type=int, default=int(getattr(config, "LIVE_SCOPE_HORIZON_DAYS", 0) or 0), help="Optional horizon (days). 0 disables.")
    ap.add_argument(
        "--next-round-only",
        dest="next_round_only",
        action="store_true",
        help="Keep only the immediate next round window (earliest fixture date + span).",
    )
    ap.add_argument(
        "--all-upcoming",
        dest="next_round_only",
        action="store_false",
        help="Disable next-round-only filtering and keep all fixtures in scope.",
    )
    ap.add_argument(
        "--next-round-span-days",
        type=int,
        default=int(getattr(config, "LIVE_SCOPE_NEXT_ROUND_SPAN_DAYS", 3) or 3),
        help="Number of days after earliest fixture date to keep when --next-round-only is active.",
    )

    ap.add_argument("--log-level", default="INFO", help="Logging level (INFO/DEBUG/WARNING)")
    ap.add_argument("--log-json", default=str(LOG_PATH_DEFAULT), help="Path to JSONL run log")
    ap.add_argument("--log-sample-rate", type=float, default=0.0, help="Probability to log per-fixture trace")
    ap.add_argument("--log-max-fixtures", type=int, default=25, help="Max fixtures to trace-log per run")
    ap.add_argument("--trace-json", default=str(TRACE_PATH_DEFAULT), help="Optional fixture-level trace JSONL")
    ap.add_argument("--trace-topk", type=int, default=0, help="(reserved) top-k contributors per kernel (not used yet)")
    ap.set_defaults(next_round_only=bool(getattr(config, "LIVE_SCOPE_NEXT_ROUND_ONLY", True)))
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("predict_upcoming")

    reg = build_team_registry(args.data_dir)

    hist = _load_history(Path(args.history), registry=reg)
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
    next_round_only = bool(getattr(args, "next_round_only", False))
    next_round_span_days = int(getattr(args, "next_round_span_days", 3) or 3)
    if next_round_span_days < 0:
        next_round_span_days = 0

    # BTTS can still use league-specific classification thresholds.
    btts_yes_threshold_default = float(getattr(config, "BTTS_YES_THRESHOLD_DEFAULT", 0.50))
    btts_yes_threshold_by_league = getattr(config, "BTTS_YES_THRESHOLD_BY_LEAGUE", {})
    mu_goal_multiplier_default = float(getattr(config, "MU_GOAL_MULTIPLIER_DEFAULT", 1.0))
    mu_goal_multiplier_by_league = getattr(config, "MU_GOAL_MULTIPLIER_BY_LEAGUE", {})
    strict_module_mu_enabled = bool(getattr(config, "STRICT_MODULE_MU_ENABLED", False))
    strict_module_weights = getattr(config, "STRICT_MODULE_WEIGHTS", {})
    strict_default_anchor_home = float(getattr(config, "STRICT_MODULE_DEFAULT_ANCHOR_HOME", 1.35))
    strict_default_anchor_away = float(getattr(config, "STRICT_MODULE_DEFAULT_ANCHOR_AWAY", 1.15))
    strict_goals_clip_min = float(getattr(config, "STRICT_MODULE_GOALS_CLIP_MIN", 0.20))
    strict_goals_clip_max = float(getattr(config, "STRICT_MODULE_GOALS_CLIP_MAX", 3.50))
    strict_pressure_share_clip_min = float(getattr(config, "STRICT_MODULE_PRESSURE_SHARE_CLIP_MIN", 0.25))
    strict_pressure_share_clip_max = float(getattr(config, "STRICT_MODULE_PRESSURE_SHARE_CLIP_MAX", 0.75))
    strict_pressure_total_clip_min = float(getattr(config, "STRICT_MODULE_PRESSURE_TOTAL_CLIP_MIN", 0.85))
    strict_pressure_total_clip_max = float(getattr(config, "STRICT_MODULE_PRESSURE_TOTAL_CLIP_MAX", 1.15))
    strict_pressure_total_strength = float(getattr(config, "STRICT_MODULE_PRESSURE_TOTAL_STRENGTH", 0.70))
    # Poisson V2 controls (incremental enhancement layer).
    poisson_v2_enabled = bool(getattr(config, "POISSON_V2_ENABLED", True))
    poisson_v2_max_goals = int(getattr(config, "POISSON_V2_MAX_GOALS", 12) or 12)
    poisson_v2_disp_alpha_default = float(getattr(config, "POISSON_V2_DISPERSION_ALPHA_DEFAULT", 0.06))
    poisson_v2_disp_alpha_by_league = getattr(config, "POISSON_V2_DISPERSION_ALPHA_BY_LEAGUE", {})
    poisson_v2_dep_strength_default = float(getattr(config, "POISSON_V2_DEP_STRENGTH_DEFAULT", 0.08))
    poisson_v2_dep_strength_by_league = getattr(config, "POISSON_V2_DEP_STRENGTH_BY_LEAGUE", {})
    poisson_v2_dc_rho_default = float(getattr(config, "POISSON_V2_DC_RHO_DEFAULT", -0.04))
    poisson_v2_dc_rho_by_league = getattr(config, "POISSON_V2_DC_RHO_BY_LEAGUE", {})

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

    # Infer/resolve effective home advantage from history (league-specific when available).
    home_adv_used = 65.0
    home_adv_by_league: dict[str, float] = {}
    try:
        if "elo_hfa_used" in hist.columns:
            hfa_series = (
                pd.to_numeric(hist["elo_hfa_used"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            home_adv_used = float(hfa_series.dropna().median()) if hfa_series.notna().any() else 65.0
            if "league" in hist.columns:
                hfa_df = pd.DataFrame({"league": hist["league"], "hfa": hfa_series}).dropna(subset=["league", "hfa"])
                if not hfa_df.empty:
                    home_adv_by_league = (
                        hfa_df.groupby("league", dropna=False)["hfa"]
                        .median()
                        .astype(float)
                        .to_dict()
                    )
        elif "EloDiff" in hist.columns and elo_home_col in hist.columns and elo_away_col in hist.columns:
            adv_series = (
                pd.to_numeric(hist["EloDiff"] - (hist[elo_home_col] - hist[elo_away_col]), errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            home_adv_used = float(adv_series.dropna().median()) if adv_series.notna().any() else 65.0
            if "league" in hist.columns:
                adv_df = pd.DataFrame({"league": hist["league"], "hfa": adv_series}).dropna(subset=["league", "hfa"])
                if not adv_df.empty:
                    home_adv_by_league = (
                        adv_df.groupby("league", dropna=False)["hfa"]
                        .median()
                        .astype(float)
                        .to_dict()
                    )
    except Exception:
        home_adv_used = 65.0
        home_adv_by_league = {}

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
                "goal_attempts": float(row.get("_press_dom_goal_attempts_H_post", 0.5)),
                "shots_off": float(row.get("_press_dom_shots_off_H_post", 0.5)),
                "blocked_shots": float(row.get("_press_dom_blocked_shots_H_post", 0.5)),
                "attacks": float(row.get("_press_dom_attacks_H_post", 0.5)),
                "dangerous_attacks": float(row.get("_press_dom_dangerous_attacks_H_post", 0.5)),
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
                "goal_attempts": float(row.get("_press_dom_goal_attempts_A_post", 0.5)),
                "shots_off": float(row.get("_press_dom_shots_off_A_post", 0.5)),
                "blocked_shots": float(row.get("_press_dom_blocked_shots_A_post", 0.5)),
                "attacks": float(row.get("_press_dom_attacks_A_post", 0.5)),
                "dangerous_attacks": float(row.get("_press_dom_dangerous_attacks_A_post", 0.5)),
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
    # Prefer API-normalized files, then legacy CSV exports.
    base_dir = Path(args.data_dir)
    multi_league_dir = base_dir / "multiple leagues and seasons"

    odds_source = None
    api_upcoming = base_dir / "upcoming_fixtures.csv"
    if api_upcoming.exists():
        primary_source = api_upcoming
        odds_source = api_upcoming
        logger.info(f"[SOURCE] Using API upcoming source: {primary_source}")
    elif multi_league_dir.exists():
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
    
    def _read_csv_smart(path: Path) -> pd.DataFrame:
        """
        Read CSV with robust encoding fallback to avoid mojibake team names.
        Priority: UTF-8 -> UTF-8-SIG -> latin1.
        """
        last_err: Exception | None = None
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except Exception as e:
                last_err = e
                continue
        raise last_err if last_err is not None else RuntimeError(f"Failed reading {path}")

    # Load upcoming data directly from the primary source (contains future fixtures)
    if primary_source.exists():
        try:
            up = _read_csv_smart(primary_source)
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

    def _normalize_upcoming_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        alias_map = {
            "home_name": "txtechipa1",
            "away_name": "txtechipa2",
            "home": "txtechipa1",
            "away": "txtechipa2",
            "home_id": "codechipa1",
            "away_id": "codechipa2",
            "date": "datameci",
            "odds_over25": "cotao",
            "odds_under25": "cotau",
            "odds_btts_yes": "gg",
            "odds_btts_no": "ng",
            "odds_home": "cotaa",
            "odds_draw": "cotae",
            "odds_away": "cotad",
        }
        for src, dst in alias_map.items():
            if dst not in out.columns and src in out.columns:
                out[dst] = out[src]
        if "orameci" not in out.columns:
            time_col = None
            if "kickoff_time" in out.columns:
                time_col = out["kickoff_time"].astype(str)
            elif "time" in out.columns:
                time_col = out["time"].astype(str)
            elif "kickoff_utc" in out.columns:
                time_col = pd.to_datetime(out["kickoff_utc"], errors="coerce").dt.strftime("%H:%M")
            if time_col is not None:
                hhmm = time_col.str.extract(r"(?P<h>\d{1,2}):(?P<m>\d{2})")
                out["orameci"] = pd.to_numeric(hhmm["h"], errors="coerce").fillna(0).astype(int) * 100 + pd.to_numeric(
                    hhmm["m"], errors="coerce"
                ).fillna(0).astype(int)
        return out

    up = _normalize_upcoming_columns(up)

    # Build fixture_datetime for scope filtering
    scope_counts: dict[str, object] = {"upcoming_rows_in": int(len(up))}
    scope_counts["upcoming_rows_parsed"] = int(up["_fixture_dt"].notna().sum())
    up = up[up["_fixture_dt"].notna()].copy()
    up["fixture_datetime"] = up["_fixture_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Optional enrichment: pull gg/ng (BTTS odds) from upcoming.csv for the same fixtures.
    if odds_source is not None and odds_source.exists():
        try:
            up_odds = _read_csv_smart(odds_source)
            up_odds = _normalize_upcoming_columns(up_odds)
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

    if next_round_only and not up.empty:
        first_fixture_dt = pd.to_datetime(up["_fixture_dt"], errors="coerce").min()
        if not pd.isna(first_fixture_dt):
            round_end = first_fixture_dt + pd.Timedelta(days=int(next_round_span_days))
            up = up[up["_fixture_dt"] <= round_end].copy()
    scope_counts["upcoming_rows_after_next_round"] = int(len(up))
    scope_counts["upcoming_rows_after_filters"] = int(len(up))

    logger.info(
        "[SCOPE] asof=%s run_asof_datetime=%s rows_in=%s parsed=%s after_past=%s after_window=%s after_league=%s after_horizon=%s after_dedupe=%s after_next_round=%s",
        str(as_of_date),
        str(run_asof_datetime),
        scope_counts.get("upcoming_rows_in"),
        scope_counts.get("upcoming_rows_parsed"),
        scope_counts.get("upcoming_rows_after_past"),
        scope_counts.get("upcoming_rows_after_window"),
        scope_counts.get("upcoming_rows_after_league"),
        scope_counts.get("upcoming_rows_after_horizon"),
        scope_counts.get("upcoming_rows_after_dedupe"),
        scope_counts.get("upcoming_rows_after_next_round"),
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
            "next_round_only": bool(next_round_only),
            "next_round_span_days": int(next_round_span_days),
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
    mu_engine_mode = "strict_module_weights_v1"
    if not strict_module_mu_enabled:
        raise RuntimeError(
            "STRICT_MODULE_MU_ENABLED must be True. Legacy trained-model mu engines are archived and disabled."
        )
    logger.info("[MU_ENGINE] strict module weights enabled")
    log_json(
        {
            "event": "STRICT_MU_ENGINE",
            "enabled": True,
            "mode": mu_engine_mode,
            "weights": strict_module_weights,
            "model_variant": model_variant,
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
        home_raw = r.get("txtechipa1", r.get("home_name", r.get("home", "")))
        away_raw = r.get("txtechipa2", r.get("away_name", r.get("away", "")))
        home = normalize_team(home_raw, reg)
        away = normalize_team(away_raw, reg)

        if home not in latest_home.index or away not in latest_away.index:
            # Fallback: try resolving by team codes when names fail.
            if home not in latest_home.index:
                code_home = r.get("codechipa1", r.get("code_home", r.get("home_id")))
                if code_home is not None and str(code_home).strip():
                    home_alt = normalize_team(str(code_home), reg)
                    if home_alt in latest_home.index:
                        home = home_alt
            if away not in latest_away.index:
                code_away = r.get("codechipa2", r.get("code_away", r.get("away_id")))
                if code_away is not None and str(code_away).strip():
                    away_alt = normalize_team(str(code_away), reg)
                    if away_alt in latest_away.index:
                        away = away_alt

        if home not in latest_home.index or away not in latest_away.index:
            # Log which team is unseen for debugging
            if home not in latest_home.index:
                logger.warning("[UNSEEN] home team '%s' (raw: '%s') not in history - using defaults", home, home_raw)
            if away not in latest_away.index:
                logger.warning("[UNSEEN] away team '%s' (raw: '%s') not in history - using defaults", away, away_raw)
            skipped.append((home, away, "missing_snapshot"))
            continue
        team_history_low = int(counts.get(home, 0) < args.min_matches or counts.get(away, 0) < args.min_matches)
        if (not strict_module_mu_enabled) and team_history_low:
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
        if (not strict_module_mu_enabled) and (elo_home is None or elo_away is None):
            skipped.append((home, away, "missing_elo"))
            continue
        elo_home = float(elo_home) if elo_home is not None else np.nan
        elo_away = float(elo_away) if elo_away is not None else np.nan
        elo_diff = elo_home - elo_away if np.isfinite(elo_home) and np.isfinite(elo_away) else np.nan
        home_adv_fixture = float(home_adv_by_league.get(league, home_adv_used))
        elo_diff_bonus = elo_diff + home_adv_fixture if np.isfinite(elo_diff) else np.nan
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
        elo_z_h = 0.0 if (elo_std <= 0 or np.isnan(elo_std) or not np.isfinite(elo_home)) else float((elo_home - elo_mean) / elo_std)
        elo_z_a = 0.0 if (elo_std <= 0 or np.isnan(elo_std) or not np.isfinite(elo_away)) else float((elo_away - elo_mean) / elo_std)
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

        odds_home = pd.to_numeric(r.get("cotaa", r.get("odds_home")), errors="coerce")
        odds_draw = pd.to_numeric(r.get("cotae", r.get("odds_draw")), errors="coerce")
        odds_away = pd.to_numeric(r.get("cotad", r.get("odds_away")), errors="coerce")
        odds_over = pd.to_numeric(r.get("cotao", r.get("odds_over25")), errors="coerce")
        odds_under = pd.to_numeric(r.get("cotau", r.get("odds_under25")), errors="coerce")
        odds_btts_yes = pd.to_numeric(r.get("gg", r.get("odds_btts_yes")), errors="coerce")
        odds_btts_no = pd.to_numeric(r.get("ng", r.get("odds_btts_no")), errors="coerce")
        imp_1x2 = _implied_probs_1x2(odds_home, odds_draw, odds_away)
        imp_ou = _implied_probs_two_way(odds_over, odds_under)
        if model_variant == "no_odds":
            odds_home = odds_draw = odds_away = np.nan
            odds_over = odds_under = np.nan
            odds_btts_yes = odds_btts_no = np.nan
            imp_1x2 = {"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan}
            imp_ou = {"p_over": np.nan, "p_under": np.nan}

        # Elo similarity features (kernel-based, venue-aware) with cutoff at fixture datetime
        fixture_dt = pd.to_datetime(r.get("_fixture_dt"), errors="coerce")
        as_of = fixture_dt if not pd.isna(fixture_dt) else pd.to_datetime(
            r.get("datameci", r.get("date", r.get("fixture_date"))), errors="coerce"
        )
        sigma = float(sigma_map.get(str(league), default_sigma))
        h_hist = elo_histories.get(home, {}).get("home", pd.DataFrame())
        a_hist = elo_histories.get(away, {}).get("away", pd.DataFrame())
        if np.isfinite(elo_away):
            gf_h, ga_h, w_h, n_h = kernel_similarity(h_hist, elo_away, sigma, as_of=as_of, inclusive=False)
        else:
            gf_h, ga_h, w_h, n_h = (np.nan, np.nan, 0.0, 0.0)
        if np.isfinite(elo_home):
            gf_a, ga_a, w_a, n_a = kernel_similarity(a_hist, elo_home, sigma, as_of=as_of, inclusive=False)
        else:
            gf_a, ga_a, w_a, n_a = (np.nan, np.nan, 0.0, 0.0)

        lg_defaults = league_meta.get(league, {})
        strict_mu_result = compute_strict_weighted_mu(
            weights=strict_module_weights if isinstance(strict_module_weights, dict) else {},
            lg_avg_gf_home=snap_h.get("lg_avg_gf_home", snap_a.get("lg_avg_gf_home", lg_defaults.get("lg_avg_gf_home", np.nan))),
            lg_avg_gf_away=snap_h.get("lg_avg_gf_away", snap_a.get("lg_avg_gf_away", lg_defaults.get("lg_avg_gf_away", np.nan))),
            default_anchor_home=strict_default_anchor_home,
            default_anchor_away=strict_default_anchor_away,
            gf_home_vs_sim=gf_h,
            ga_home_vs_sim=ga_h,
            gf_away_vs_sim=gf_a,
            ga_away_vs_sim=ga_a,
            xg_for_home=xg_for_h,
            xg_against_home=xg_against_h,
            xg_for_away=xg_for_a,
            xg_against_away=xg_against_a,
            xg_usable=bool(xg_usable),
            press_form_home=press_form_h,
            press_form_away=press_form_a,
            dom_home=dom_h,
            dom_away=dom_a,
            pressure_usable=bool(pressure_usable),
            goals_clip_min=strict_goals_clip_min,
            goals_clip_max=strict_goals_clip_max,
            pressure_share_clip_min=strict_pressure_share_clip_min,
            pressure_share_clip_max=strict_pressure_share_clip_max,
            pressure_total_clip_min=strict_pressure_total_clip_min,
            pressure_total_clip_max=strict_pressure_total_clip_max,
            pressure_total_strength=strict_pressure_total_strength,
        )
        mu_h_raw = float(strict_mu_result.mu_home_raw)
        mu_a_raw = float(strict_mu_result.mu_away_raw)
        mu_mult = _league_scalar(
            str(league),
            default_value=mu_goal_multiplier_default,
            overrides=mu_goal_multiplier_by_league if isinstance(mu_goal_multiplier_by_league, dict) else {},
            min_value=0.10,
            max_value=5.00,
        )
        # Apply league-level drift correction before Poisson conversion.
        mu_h = max(0.01, float(mu_h_raw * mu_mult))
        mu_a = max(0.01, float(mu_a_raw * mu_mult))
        pois_disp_alpha = _league_scalar(
            str(league),
            default_value=poisson_v2_disp_alpha_default,
            overrides=poisson_v2_disp_alpha_by_league if isinstance(poisson_v2_disp_alpha_by_league, dict) else {},
            min_value=0.0,
            max_value=0.60,
        )
        pois_dep_strength = _league_scalar(
            str(league),
            default_value=poisson_v2_dep_strength_default,
            overrides=poisson_v2_dep_strength_by_league if isinstance(poisson_v2_dep_strength_by_league, dict) else {},
            min_value=0.0,
            max_value=0.45,
        )
        pois_dc_rho = _league_scalar(
            str(league),
            default_value=poisson_v2_dc_rho_default,
            overrides=poisson_v2_dc_rho_by_league if isinstance(poisson_v2_dc_rho_by_league, dict) else {},
            min_value=-0.30,
            max_value=0.30,
        )
        if poisson_v2_enabled:
            probs = _poisson_probs(
                mu_h,
                mu_a,
                max_goals=poisson_v2_max_goals,
                dep_strength=pois_dep_strength,
                dispersion_alpha=pois_disp_alpha,
                dc_rho=pois_dc_rho,
            )
        else:
            probs = _poisson_probs(mu_h, mu_a, max_goals=poisson_v2_max_goals)
        neff_h_list.append(n_h)
        neff_a_list.append(n_a)

        # Optional trace logging (sampled)
        do_trace = (args.log_sample_rate > 0 and random.random() < args.log_sample_rate and trace_count < args.log_max_fixtures)
        if do_trace:
            trace_count += 1
            log_json(
                {
                    "event": "ELO_PRED_TRACE",
                    "fixture": {"date": r.get("datameci", r.get("date")), "league": league, "home": home, "away": away},
                    "elo": {
                        "home": elo_home,
                        "away": elo_away,
                        "hfa_used": home_adv_fixture,
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
                    "mu": {
                        "engine_mode": mu_engine_mode,
                        "home_raw": mu_h_raw,
                        "away_raw": mu_a_raw,
                        "multiplier": mu_mult,
                        "home": mu_h,
                        "away": mu_a,
                        "anchor_home": float(strict_mu_result.anchor_home) if strict_mu_result is not None else np.nan,
                        "anchor_away": float(strict_mu_result.anchor_away) if strict_mu_result is not None else np.nan,
                        "elo_home": float(strict_mu_result.elo_home) if strict_mu_result is not None else np.nan,
                        "elo_away": float(strict_mu_result.elo_away) if strict_mu_result is not None else np.nan,
                        "xg_home": float(strict_mu_result.xg_home) if strict_mu_result is not None else np.nan,
                        "xg_away": float(strict_mu_result.xg_away) if strict_mu_result is not None else np.nan,
                        "pressure_home": float(strict_mu_result.pressure_home) if strict_mu_result is not None else np.nan,
                        "pressure_away": float(strict_mu_result.pressure_away) if strict_mu_result is not None else np.nan,
                        "neutralized_modules": list(strict_mu_result.neutralized_modules) if strict_mu_result is not None else [],
                    },
                    "probs": probs,
                },
                log_path=trace_path,
            )

        def ev(prob, odds):
            try:
                return prob * float(odds) - 1.0
            except Exception as e:
                logger.debug(f"EV calc error: prob={prob}, odds={odds}, err={e}")
                return np.nan

        ev_over = ev(probs["p_over25"], odds_over)
        ev_under = ev(probs["p_under25"], odds_under)

        # BTTS from same joint score engine (Poisson V2), not independence shortcut.
        p_btts_yes = float(probs.get("p_btts_yes", np.nan))
        p_btts_no = float(probs.get("p_btts_no", np.nan)) if np.isfinite(p_btts_yes) else np.nan

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

        # Row-level data quality flags for easy non-technical review in output CSV.
        quality_flags: List[str] = []
        if pressure_usable <= 0:
            quality_flags.append("pressure_unusable")
        if xg_usable <= 0:
            quality_flags.append("xg_unusable")
        if team_history_low:
            quality_flags.append("team_history_low")
        if strict_mu_result is not None:
            for module_name in strict_mu_result.neutralized_modules:
                quality_flags.append(f"{module_name}_neutralized")

        if (press_stats_n_h <= 0.0) or (press_stats_n_a <= 0.0):
            quality_flags.append("press_stats_missing")
        elif min(press_stats_n_h, press_stats_n_a) < 3.0:
            quality_flags.append("press_stats_low")

        if (xg_stats_n_h <= 0.0) or (xg_stats_n_a <= 0.0):
            quality_flags.append("xg_stats_missing")
        elif min(xg_stats_n_h, xg_stats_n_a) < 3.0:
            quality_flags.append("xg_stats_low")

        if (n_h < float(args.min_matches)) or (n_a < float(args.min_matches)):
            quality_flags.append("elo_evidence_low")

        if not (np.isfinite(odds_over) and np.isfinite(odds_under)):
            quality_flags.append("odds_ou_missing")
        if not (np.isfinite(odds_btts_yes) and np.isfinite(odds_btts_no)):
            quality_flags.append("odds_btts_missing")

        critical_flags = {
            "pressure_unusable",
            "xg_unusable",
            "press_stats_missing",
            "xg_stats_missing",
            "elo_evidence_low",
        }
        quality_critical = int(any(flag in critical_flags for flag in quality_flags))
        if quality_critical:
            quality_status = "BAD"
        elif quality_flags:
            quality_status = "WARN"
        else:
            quality_status = "OK"

        preds.append(
            {
                "fixture_datetime": fixture_dt.isoformat() if not pd.isna(fixture_dt) else str(
                    r.get("datameci", r.get("date"))
                ),
                "date": r.get("datameci", r.get("date")),
                "league": league,
                "country": country,
                "season": season,
                "home": home,
                "away": away,
                "model_variant": model_variant,
                "mu_engine_mode": mu_engine_mode,
                # Scope metadata (copied into every row; deterministic audit trail)
                "run_asof_datetime": run_asof_datetime.isoformat(),
                "scope_country": scope_country or "",
                "scope_league": scope_league or "",
                "scope_season_start": scope_season_start.date().isoformat() if scope_season_start is not None else "",
                "scope_season_end": scope_season_end.date().isoformat() if scope_season_end is not None else "",
                "horizon_days": int(horizon_days),
                "next_round_only": int(next_round_only),
                "next_round_span_days": int(next_round_span_days),
                "upcoming_rows_in": int(scope_counts.get("upcoming_rows_in", 0) or 0),
                "upcoming_rows_parsed": int(scope_counts.get("upcoming_rows_parsed", 0) or 0),
                "upcoming_rows_after_past": int(scope_counts.get("upcoming_rows_after_past", 0) or 0),
                "upcoming_rows_after_window": int(scope_counts.get("upcoming_rows_after_window", 0) or 0),
                "upcoming_rows_after_league": int(scope_counts.get("upcoming_rows_after_league", 0) or 0),
                "upcoming_rows_after_horizon": int(scope_counts.get("upcoming_rows_after_horizon", 0) or 0),
                "upcoming_rows_after_dedupe": int(scope_counts.get("upcoming_rows_after_dedupe", 0) or 0),
                "upcoming_rows_after_next_round": int(scope_counts.get("upcoming_rows_after_next_round", 0) or 0),
                "upcoming_rows_after_filters": int(scope_counts.get("upcoming_rows_after_filters", 0) or 0),
                "mu_home": mu_h,
                "mu_away": mu_a,
                "mu_home_raw": mu_h_raw,
                "mu_away_raw": mu_a_raw,
                "mu_goal_multiplier": mu_mult,
                "mu_anchor_home": float(strict_mu_result.anchor_home) if strict_mu_result is not None else np.nan,
                "mu_anchor_away": float(strict_mu_result.anchor_away) if strict_mu_result is not None else np.nan,
                "mu_elo_home": float(strict_mu_result.elo_home) if strict_mu_result is not None else np.nan,
                "mu_elo_away": float(strict_mu_result.elo_away) if strict_mu_result is not None else np.nan,
                "mu_xg_home": float(strict_mu_result.xg_home) if strict_mu_result is not None else np.nan,
                "mu_xg_away": float(strict_mu_result.xg_away) if strict_mu_result is not None else np.nan,
                "mu_pressure_home": float(strict_mu_result.pressure_home) if strict_mu_result is not None else np.nan,
                "mu_pressure_away": float(strict_mu_result.pressure_away) if strict_mu_result is not None else np.nan,
                "mu_weight_anchor": float(strict_module_weights.get("league_anchor", np.nan)) if isinstance(strict_module_weights, dict) else np.nan,
                "mu_weight_elo": float(strict_module_weights.get("elo", np.nan)) if isinstance(strict_module_weights, dict) else np.nan,
                "mu_weight_xg": float(strict_module_weights.get("xg", np.nan)) if isinstance(strict_module_weights, dict) else np.nan,
                "mu_weight_pressure": float(strict_module_weights.get("pressure", np.nan)) if isinstance(strict_module_weights, dict) else np.nan,
                "poisson_v2_enabled": int(poisson_v2_enabled),
                "poisson_v2_disp_alpha": float(pois_disp_alpha),
                "poisson_v2_dep_strength": float(pois_dep_strength),
                "poisson_v2_dc_rho": float(pois_dc_rho),
                "mu_total": mu_h + mu_a,
                "elo_hfa_used": home_adv_fixture,
                "p_over25": probs["p_over25"],
                "p_under25": probs["p_under25"],
                "p_over_2_5": probs["p_over25"],
                "p_under_2_5": probs["p_under25"],
                # OU side selection is now pure probability comparison across all leagues.
                # This removes per-league threshold forcing that could flip labels
                # against the model's own p_over vs p_under signal.
                "pred_ou25": (
                    "OU25_OVER"
                    if np.isfinite(probs["p_over25"])
                    and (
                        (not np.isfinite(probs["p_under25"]))
                        or float(probs["p_over25"]) >= float(probs["p_under25"])
                    )
                    else "OU25_UNDER"
                ),
                "odds_over25": odds_over,
                "odds_under25": odds_under,
                "odds_over_2_5": odds_over,
                "odds_under_2_5": odds_under,
                "EV_over25": ev_over,
                "EV_under25": ev_under,
                "p_btts_yes": p_btts_yes,
                "p_btts_no": p_btts_no,
                "pred_btts": (
                    "BTTS_YES"
                    if np.isfinite(p_btts_yes)
                    and float(p_btts_yes)
                    >= _league_threshold(
                        str(league),
                        default_value=btts_yes_threshold_default,
                        overrides=btts_yes_threshold_by_league if isinstance(btts_yes_threshold_by_league, dict) else {},
                    )
                    else "BTTS_NO"
                ),
                "odds_btts_yes": odds_btts_yes,
                "odds_btts_no": odds_btts_no,
                "EV_btts_yes": ev_btts_yes,
                "EV_btts_no": ev_btts_no,
                # Data quality flags (explicitly surface weak/missing evidence rows).
                "quality_status": quality_status,
                "quality_critical": quality_critical,
                "quality_issue_count": int(len(quality_flags)),
                "quality_flags": ";".join(quality_flags),
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
                "mu_engine_mode",
                "run_asof_datetime",
                "scope_country",
                "scope_league",
                "scope_season_start",
                "scope_season_end",
                "horizon_days",
                "next_round_only",
                "next_round_span_days",
                "upcoming_rows_in",
                "upcoming_rows_parsed",
                "upcoming_rows_after_past",
                "upcoming_rows_after_window",
                "upcoming_rows_after_league",
                "upcoming_rows_after_horizon",
                "upcoming_rows_after_dedupe",
                "upcoming_rows_after_next_round",
                "upcoming_rows_after_filters",
                "mu_home",
                "mu_away",
                "mu_home_raw",
                "mu_away_raw",
                "mu_goal_multiplier",
                "mu_anchor_home",
                "mu_anchor_away",
                "mu_elo_home",
                "mu_elo_away",
                "mu_xg_home",
                "mu_xg_away",
                "mu_pressure_home",
                "mu_pressure_away",
                "mu_weight_anchor",
                "mu_weight_elo",
                "mu_weight_xg",
                "mu_weight_pressure",
                "poisson_v2_enabled",
                "poisson_v2_disp_alpha",
                "poisson_v2_dep_strength",
                "poisson_v2_dc_rho",
                "mu_total",
                "elo_hfa_used",
                "p_over25",
                "p_under25",
                "p_over_2_5",
                "p_under_2_5",
                "pred_ou25",
                "odds_over25",
                "odds_under25",
                "odds_over_2_5",
                "odds_under_2_5",
                "EV_over25",
                "EV_under25",
                "p_btts_yes",
                "p_btts_no",
                "pred_btts",
                "odds_btts_yes",
                "odds_btts_no",
                "EV_btts_yes",
                "EV_btts_no",
                "quality_status",
                "quality_critical",
                "quality_issue_count",
                "quality_flags",
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
              "mu_engine_mode": mu_engine_mode,
              "models": [],
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
            "models": {},
            "config": {
                "mu_engine_mode": mu_engine_mode,
                "strict_module_mu_enabled": bool(strict_module_mu_enabled),
                "strict_module_weights": strict_module_weights if isinstance(strict_module_weights, dict) else {},
                "sigma_map": sigma_map,
                "default_sigma": default_sigma,
                "min_matches": args.min_matches,
                "windows": args.windows,
                "poisson_v2_enabled": bool(poisson_v2_enabled),
                "poisson_v2_max_goals": int(poisson_v2_max_goals),
                "poisson_v2_disp_alpha_default": float(poisson_v2_disp_alpha_default),
                "poisson_v2_dep_strength_default": float(poisson_v2_dep_strength_default),
                "poisson_v2_dc_rho_default": float(poisson_v2_dc_rho_default),
            },
            "rows": len(out_df),
            "quality_status_counts": out_df["quality_status"].value_counts().to_dict() if ("quality_status" in out_df.columns and not out_df.empty) else {},
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
