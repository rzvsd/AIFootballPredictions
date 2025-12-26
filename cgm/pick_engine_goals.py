"""
Milestone 7.1: Goals-only Strategy Layer ("Pick Engine").

Turns model predictions into deterministic, explainable betting picks for:
  - Over/Under 2.5
  - BTTS (Yes/No) (only when odds are present)

Primary input:
  reports/cgm_upcoming_predictions.csv

Main output:
  reports/picks.csv

Optional debug output:
  reports/picks_debug.csv
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Import centralized constants from config.py
try:
    import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config


ALLOWED_MARKETS = {
    "OU25_OVER",
    "OU25_UNDER",
    "BTTS_YES",
    "BTTS_NO",
    # Milestone 7.2 timing markets (only when odds exist)
    "1H_OU05_OVER",
    "1H_OU05_UNDER",
    "2H_OU05_OVER",
    "2H_OU05_UNDER",
    "2H_OU15_OVER",
    "2H_OU15_UNDER",
    "GOAL_AFTER_75_YES",
    "GOAL_AFTER_75_NO",
}
MARKET_PRIORITY = [
    "OU25_OVER",
    "OU25_UNDER",
    "BTTS_YES",
    "BTTS_NO",
    "1H_OU05_UNDER",
    "1H_OU05_OVER",
    "2H_OU15_OVER",
    "2H_OU15_UNDER",
    "2H_OU05_OVER",
    "2H_OU05_UNDER",
    "GOAL_AFTER_75_YES",
    "GOAL_AFTER_75_NO",
]
MARKET_PRIORITY_RANK = {m: i for i, m in enumerate(MARKET_PRIORITY)}

# Gates - imported from config.py (uses stricter _GOALS variants)
ODDS_MIN = getattr(config, "ODDS_MIN_GOALS", 1.05)
MU_TOTAL_MIN = getattr(config, "MU_TOTAL_MIN", 1.6)
MU_TOTAL_MAX = getattr(config, "MU_TOTAL_MAX", 3.4)
NEFF_MIN = getattr(config, "NEFF_MIN_GOALS", 8.0)
PRESS_EVID_MIN = getattr(config, "PRESS_EVID_MIN_GOALS", 3.0)
XG_EVID_MIN = getattr(config, "XG_EVID_MIN_GOALS", 3.0)

# EV thresholds - imported from config.py
EV_MIN_OU25 = getattr(config, "EV_MIN_OU25", 0.04)
EV_MIN_BTTS = getattr(config, "EV_MIN_BTTS", 0.04)
EV_MIN_TIMING = getattr(config, "EV_MIN_TIMING", 0.05)
EV_MIN_STERILE_OVER = getattr(config, "EV_MIN_STERILE_OVER", 0.08)
EV_MIN_ASSASSIN_UNDER = getattr(config, "EV_MIN_ASSASSIN_UNDER", 0.08)
EV_MIN_LATE_HEAVY_UNDER = getattr(config, "EV_MIN_LATE_HEAVY_UNDER", 0.08)

# Milestone 13: League calibration
CALIBRATION_ENABLED = getattr(config, "CALIBRATION_ENABLED", True)
CALIBRATION_FILE = getattr(config, "CALIBRATION_FILE", "data/league_calibration.json")
CALIBRATION_MIN_SAMPLES = getattr(config, "CALIBRATION_MIN_SAMPLES", 50)

import json
import logging
_logger = logging.getLogger(__name__)

def _load_calibration() -> dict:
    """Load league calibration from JSON file."""
    try:
        import os
        cal_path = Path(__file__).resolve().parents[1] / CALIBRATION_FILE
        if cal_path.exists():
            with open(cal_path, "r") as f:
                cal = json.load(f)
            _logger.info(f"Loaded calibration for {len(cal)} leagues from {cal_path}")
            return cal
    except Exception as e:
        _logger.warning(f"Could not load calibration: {e}")
    return {}

_CALIBRATION = _load_calibration() if CALIBRATION_ENABLED else {}


def _get_calibrated_threshold(league: str, market_type: str) -> float:
    """
    Get the optimal threshold for a league/market from calibration.
    Returns 0.50 as default if no calibration exists.
    """
    if not CALIBRATION_ENABLED or league not in _CALIBRATION:
        return 0.50
    
    cal = _CALIBRATION[league]
    if cal.get("sample_size", 0) < CALIBRATION_MIN_SAMPLES:
        return 0.50
    
    if market_type == "OU25":
        return cal.get("ou25_optimal_threshold", 0.50)
    elif market_type == "BTTS":
        return cal.get("btts_optimal_threshold", 0.50)
    return 0.50


def file_sha256(path: Path) -> str:
    """SHA256 hash for file integrity (consistent with predict_upcoming.py)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _is_sane_odds(x: object) -> bool:
    v = _num(x)
    return bool(np.isfinite(v) and v > ODDS_MIN)


def _implied_prob(odds: float) -> float:
    if not np.isfinite(odds) or odds <= 0:
        return float("nan")
    return 1.0 / odds


def _ev(p_model: float, odds: float) -> float:
    if not (np.isfinite(p_model) and np.isfinite(odds)):
        return float("nan")
    return p_model * odds - 1.0


def _as_int_flag(x: object) -> int:
    v = _num(x)
    if not np.isfinite(v):
        return 0
    return int(v > 0)


def _require_columns(df: pd.DataFrame, required: Iterable[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"[pick_engine_goals] missing required columns in {context}: {missing}"
        raise SystemExit(msg)


def _p_under_2_5(mu_total: float) -> float:
    mu = float(mu_total)
    if not np.isfinite(mu) or mu < 0:
        return float("nan")
    # total goals ~ Poisson(mu_total)
    p0 = math.exp(-mu)
    p1 = p0 * mu
    p2 = p1 * mu / 2.0
    return float(np.clip(p0 + p1 + p2, 0.0, 1.0))


def _p_btts_yes(mu_home: float, mu_away: float) -> float:
    mh = float(mu_home)
    ma = float(mu_away)
    if not (np.isfinite(mh) and np.isfinite(ma)) or mh < 0 or ma < 0:
        return float("nan")
    # Independence assumption: P(H>=1 & A>=1) = (1-P(H=0))*(1-P(A=0))
    return float(np.clip((1.0 - math.exp(-mh)) * (1.0 - math.exp(-ma)), 0.0, 1.0))


@dataclass(frozen=True)
class Candidate:
    fixture_idx: int
    fixture_datetime: str
    league: str
    home: str
    away: str
    market: str
    odds: float
    p_model: float
    ev: float
    score: float
    ev_min_required: float
    reason_codes: tuple[str, ...]
    sterile_flag: int
    assassin_flag: int
    timing_usable: int
    slow_start_flag: int
    late_goal_flag: int
    timing_early_share: float
    timing_late_share: float
    mu_home: float
    mu_away: float
    mu_total: float
    neff_min: float
    press_n_min: float
    xg_n_min: float


def build_picks(df: pd.DataFrame, *, input_hash: str, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df0 = df.copy()

    # Normalize common alternate names (older exports).
    if "p_over_2_5" not in df0.columns and "p_over25" in df0.columns:
        df0["p_over_2_5"] = df0["p_over25"]
    if "p_under_2_5" not in df0.columns and "p_under25" in df0.columns:
        df0["p_under_2_5"] = df0["p_under25"]
    if "odds_over_2_5" not in df0.columns and "odds_over25" in df0.columns:
        df0["odds_over_2_5"] = df0["odds_over25"]
    if "odds_under_2_5" not in df0.columns and "odds_under25" in df0.columns:
        df0["odds_under_2_5"] = df0["odds_under25"]

    for c in [
        "mu_home",
        "mu_away",
        "mu_total",
        "p_over_2_5",
        "p_under_2_5",
        "odds_over_2_5",
        "odds_under_2_5",
        "neff_sim_H",
        "neff_sim_A",
        "press_stats_n_H",
        "press_stats_n_A",
        "xg_stats_n_H",
        "xg_stats_n_A",
        "sterile_flag",
        "assassin_flag",
        # Milestone 7.2 timing (minute-goals)
        "timing_usable",
        "slow_start_flag",
        "late_goal_flag",
        "timing_early_share",
        "timing_late_share",
        "p_1h_over_0_5",
        "p_1h_under_0_5",
        "p_2h_over_0_5",
        "p_2h_under_0_5",
        "p_2h_over_1_5",
        "p_2h_under_1_5",
        "p_goal_after_75_yes",
        "p_goal_after_75_no",
        "odds_1h_over_0_5",
        "odds_1h_under_0_5",
        "odds_2h_over_0_5",
        "odds_2h_under_0_5",
        "odds_2h_over_1_5",
        "odds_2h_under_1_5",
        "odds_goal_after_75_yes",
        "odds_goal_after_75_no",
        # Optional BTTS odds (if present in the predictions file)
        "odds_btts_yes",
        "odds_btts_no",
        "odds_gg",
        "odds_ng",
        "gg",
        "ng",
    ]:
        if c in df0.columns:
            df0[c] = pd.to_numeric(df0[c], errors="coerce")

    if "mu_total" not in df0.columns:
        df0["mu_total"] = df0["mu_home"] + df0["mu_away"]

    # Ensure O/U probabilities exist; compute from mu_total if absent.
    if "p_under_2_5" not in df0.columns:
        df0["p_under_2_5"] = np.nan
    if "p_over_2_5" not in df0.columns:
        df0["p_over_2_5"] = np.nan

    if df0["p_under_2_5"].isna().all() and df0["p_over_2_5"].isna().all():
        p_under = df0["mu_total"].apply(_p_under_2_5)
        df0["p_under_2_5"] = p_under
        df0["p_over_2_5"] = 1.0 - p_under
    else:
        if df0["p_over_2_5"].isna().all() and df0["p_under_2_5"].notna().any():
            df0["p_over_2_5"] = 1.0 - df0["p_under_2_5"]
        if df0["p_under_2_5"].isna().all() and df0["p_over_2_5"].notna().any():
            df0["p_under_2_5"] = 1.0 - df0["p_over_2_5"]

    # Deterministic sort for reproducibility
    dt = pd.to_datetime(df0.get("fixture_datetime"), errors="coerce")
    if dt.notna().any():
        df0["_fixture_dt"] = dt
    else:
        df0["_fixture_dt"] = pd.to_datetime(df0.get("date"), errors="coerce")
    df0 = df0.sort_values(["_fixture_dt", "league", "home", "away"], kind="mergesort").reset_index(drop=True)

    # Scope protection (same philosophy as Milestone 4).
    fixtures_in = int(len(df0))
    df0["_run_asof_dt"] = pd.to_datetime(df0.get("run_asof_datetime"), errors="coerce")
    asof_vals = df0["_run_asof_dt"].dropna().unique()
    if len(asof_vals) > 1:
        raise SystemExit(f"[pick_engine_goals] run_asof_datetime must be constant; found {len(asof_vals)} values")
    run_asof_dt = pd.to_datetime(asof_vals[0]) if len(asof_vals) == 1 else pd.NaT
    if fixtures_in and pd.isna(run_asof_dt):
        raise SystemExit("[pick_engine_goals] run_asof_datetime is required and must be parseable")

    scope_league_vals = pd.Series(df0.get("scope_league", pd.Series(dtype=object))).astype(str).str.strip()
    scope_league_clean = [v for v in scope_league_vals.dropna().unique().tolist() if v and v.lower() != "nan"]
    if len(set(scope_league_clean)) > 1:
        raise SystemExit(f"[pick_engine_goals] scope_league must be constant; found {sorted(set(scope_league_clean))}")
    scope_league = scope_league_clean[0] if scope_league_clean else None

    scope_country_vals = pd.Series(df0.get("scope_country", pd.Series(dtype=object))).astype(str).str.strip()
    scope_country_clean = [v for v in scope_country_vals.dropna().unique().tolist() if v and v.lower() != "nan"]
    if len(set(scope_country_clean)) > 1:
        raise SystemExit(f"[pick_engine_goals] scope_country must be constant; found {sorted(set(scope_country_clean))}")
    scope_country = scope_country_clean[0] if scope_country_clean else None

    ss_vals = pd.to_datetime(df0.get("scope_season_start"), errors="coerce").dropna().unique()
    se_vals = pd.to_datetime(df0.get("scope_season_end"), errors="coerce").dropna().unique()
    if len(ss_vals) > 1 or len(se_vals) > 1:
        raise SystemExit("[pick_engine_goals] scope_season_start/scope_season_end must be constant when present")
    scope_season_start = pd.to_datetime(ss_vals[0]).normalize() if len(ss_vals) == 1 else pd.NaT
    scope_season_end = pd.to_datetime(se_vals[0]).normalize() if len(se_vals) == 1 else pd.NaT

    horizon_vals = pd.to_numeric(df0.get("horizon_days"), errors="coerce").dropna().unique()
    if len(horizon_vals) > 1:
        raise SystemExit("[pick_engine_goals] horizon_days must be constant when present")
    horizon_days = int(horizon_vals[0]) if len(horizon_vals) == 1 and np.isfinite(horizon_vals[0]) else 0

    df0 = df0[df0["_fixture_dt"].notna()].copy()
    if not pd.isna(run_asof_dt):
        df0 = df0[df0["_fixture_dt"] > run_asof_dt].copy()
    if not pd.isna(scope_season_start) and not pd.isna(scope_season_end):
        df0 = df0[df0["_fixture_dt"].between(scope_season_start, scope_season_end, inclusive="left")].copy()
    if scope_league:
        df0 = df0[df0["league"].astype(str) == scope_league].copy()
    if scope_country and "country" in df0.columns:
        df0 = df0[df0["country"].astype(str) == scope_country].copy()
    if horizon_days > 0 and not pd.isna(run_asof_dt):
        df0 = df0[df0["_fixture_dt"] <= (run_asof_dt + pd.Timedelta(days=horizon_days))].copy()

    fixtures_in_scope = int(len(df0))
    df0 = df0.reset_index(drop=True)

    btts_supported = any(c in df0.columns for c in ("odds_btts_yes", "odds_btts_no", "odds_gg", "odds_ng", "gg", "ng"))

    def _any_sane_odds(col: str) -> bool:
        if col not in df0.columns:
            return False
        s = pd.to_numeric(df0[col], errors="coerce")
        return bool((s > ODDS_MIN).any())

    timing_1h_supported = _any_sane_odds("odds_1h_over_0_5") or _any_sane_odds("odds_1h_under_0_5")
    timing_2h05_supported = _any_sane_odds("odds_2h_over_0_5") or _any_sane_odds("odds_2h_under_0_5")
    timing_2h15_supported = _any_sane_odds("odds_2h_over_1_5") or _any_sane_odds("odds_2h_under_1_5")
    timing_after75_supported = _any_sane_odds("odds_goal_after_75_yes") or _any_sane_odds("odds_goal_after_75_no")

    candidates: list[Candidate] = []
    debug_rows: list[dict] = []

    for idx, r in df0.iterrows():
        fixture_datetime = str(r.get("fixture_datetime", ""))
        league = str(r.get("league", ""))
        home = str(r.get("home", ""))
        away = str(r.get("away", ""))

        run_asof_datetime_s = str(r.get("run_asof_datetime", ""))
        scope_season_start_s = str(r.get("scope_season_start", ""))
        scope_season_end_s = str(r.get("scope_season_end", ""))
        scope_league_s = str(r.get("scope_league", ""))
        scope_country_s = str(r.get("scope_country", ""))
        horizon_days_s = str(r.get("horizon_days", ""))

        mu_home = float(_num(r.get("mu_home")))
        mu_away = float(_num(r.get("mu_away")))
        mu_total = float(_num(r.get("mu_total")))

        p_over = float(_num(r.get("p_over_2_5")))
        p_under = float(_num(r.get("p_under_2_5")))

        neff_min = float(min(_num(r.get("neff_sim_H")), _num(r.get("neff_sim_A"))))
        press_n_min = float(min(_num(r.get("press_stats_n_H")), _num(r.get("press_stats_n_A"))))
        xg_n_min = float(min(_num(r.get("xg_stats_n_H")), _num(r.get("xg_stats_n_A"))))

        sterile_flag = _as_int_flag(r.get("sterile_flag"))
        assassin_flag = _as_int_flag(r.get("assassin_flag"))
        timing_usable = _as_int_flag(r.get("timing_usable"))
        slow_start_flag = _as_int_flag(r.get("slow_start_flag"))
        late_goal_flag = _as_int_flag(r.get("late_goal_flag"))
        timing_early_share = float(_num(r.get("timing_early_share")))
        timing_late_share = float(_num(r.get("timing_late_share")))

        base_fail: list[str] = []
        if not (np.isfinite(mu_total) and MU_TOTAL_MIN <= mu_total <= MU_TOTAL_MAX):
            base_fail.append("G2_FAIL_LOW_MU_TOTAL" if (np.isfinite(mu_total) and mu_total < MU_TOTAL_MIN) else "G2_FAIL_HIGH_MU_TOTAL")
        if not (np.isfinite(neff_min) and neff_min >= NEFF_MIN):
            base_fail.append("G3_FAIL_LOW_NEFF")
        if not (np.isfinite(press_n_min) and press_n_min >= PRESS_EVID_MIN):
            base_fail.append("G3_FAIL_LOW_PRESS_EVIDENCE")
        if not (np.isfinite(xg_n_min) and xg_n_min >= XG_EVID_MIN):
            base_fail.append("G3_FAIL_LOW_XG_EVIDENCE")

        def _score(ev_val: float) -> float:
            return float(
                ev_val
                + 0.01 * math.log(1.0 + max(neff_min, 0.0))
                + 0.005 * (max(press_n_min, 0.0) + max(xg_n_min, 0.0))
            )

        def _emit_candidate(*, market: str, odds: float, p_model: float, ev_min_req: float, odds_reason: str) -> None:
            reasons: list[str] = list(base_fail)
            timing_markets = {
                "1H_OU05_OVER",
                "1H_OU05_UNDER",
                "2H_OU05_OVER",
                "2H_OU05_UNDER",
                "2H_OU15_OVER",
                "2H_OU15_UNDER",
                "GOAL_AFTER_75_YES",
                "GOAL_AFTER_75_NO",
            }
            if market in timing_markets:
                if not timing_usable:
                    reasons.append("G7T_FAIL_NO_TIMING_EVIDENCE")
                if market in {"1H_OU05_UNDER"} and not slow_start_flag:
                    reasons.append("G7T_FAIL_NOT_SLOW_START")
                if market in {"1H_OU05_OVER"} and slow_start_flag:
                    reasons.append("G7T_FAIL_SLOW_START")
                if market in {"GOAL_AFTER_75_YES", "2H_OU05_OVER", "2H_OU15_OVER"} and not late_goal_flag:
                    reasons.append("G7T_FAIL_NOT_LATE_PROFILE")
                if market in {"GOAL_AFTER_75_NO", "2H_OU05_UNDER", "2H_OU15_UNDER"} and late_goal_flag:
                    reasons.append("G7T_FAIL_LATE_PROFILE")
            if not _is_sane_odds(odds):
                reasons.append(odds_reason)

            ev_val = _ev(p_model, odds) if not reasons else float("nan")
            if not reasons and not (np.isfinite(ev_val) and ev_val >= ev_min_req):
                reasons.append("G5_FAIL_LOW_EV")

            eligible = len(reasons) == 0
            score = _score(ev_val) if eligible else float("nan")
            if eligible:
                candidates.append(
                    Candidate(
                        fixture_idx=int(idx),
                        fixture_datetime=fixture_datetime,
                        league=league,
                        home=home,
                        away=away,
                        market=market,
                        odds=float(odds),
                        p_model=float(p_model),
                        ev=float(ev_val),
                        score=float(score),
                        ev_min_required=float(ev_min_req),
                        reason_codes=tuple(reasons),
                        sterile_flag=int(sterile_flag),
                        assassin_flag=int(assassin_flag),
                        timing_usable=int(timing_usable),
                        slow_start_flag=int(slow_start_flag),
                        late_goal_flag=int(late_goal_flag),
                        timing_early_share=float(timing_early_share),
                        timing_late_share=float(timing_late_share),
                        mu_home=float(mu_home),
                        mu_away=float(mu_away),
                        mu_total=float(mu_total),
                        neff_min=float(neff_min),
                        press_n_min=float(press_n_min),
                        xg_n_min=float(xg_n_min),
                    )
                )

            debug_rows.append(
                {
                    "fixture_datetime": fixture_datetime,
                    "league": league,
                    "home": home,
                    "away": away,
                    "run_asof_datetime": run_asof_datetime_s,
                    "scope_country": scope_country_s,
                    "scope_league": scope_league_s,
                    "scope_season_start": scope_season_start_s,
                    "scope_season_end": scope_season_end_s,
                    "horizon_days": int(_num(horizon_days_s)) if np.isfinite(_num(horizon_days_s)) else 0,
                    "market": market,
                    "odds": float(odds) if np.isfinite(_num(odds)) else np.nan,
                    "p_model": float(p_model) if np.isfinite(_num(p_model)) else np.nan,
                    "ev": float(ev_val) if np.isfinite(_num(ev_val)) else np.nan,
                    "ev_min_required": float(ev_min_req),
                    "score": float(score) if np.isfinite(_num(score)) else np.nan,
                    "eligible": bool(eligible),
                    "reason_codes": "|".join(reasons),
                    "mu_total": float(mu_total) if np.isfinite(_num(mu_total)) else np.nan,
                    "neff_min": float(neff_min) if np.isfinite(_num(neff_min)) else np.nan,
                    "press_n_min": float(press_n_min) if np.isfinite(_num(press_n_min)) else np.nan,
                    "xg_n_min": float(xg_n_min) if np.isfinite(_num(xg_n_min)) else np.nan,
                    "sterile_flag": int(sterile_flag),
                    "assassin_flag": int(assassin_flag),
                    "timing_usable": int(timing_usable),
                    "slow_start_flag": int(slow_start_flag),
                    "late_goal_flag": int(late_goal_flag),
                    "timing_early_share": float(timing_early_share) if np.isfinite(_num(timing_early_share)) else np.nan,
                    "timing_late_share": float(timing_late_share) if np.isfinite(_num(timing_late_share)) else np.nan,
                    "selected": False,
                }
            )

        # O/U 2.5 candidates (always evaluated)
        odds_over = float(_num(r.get("odds_over_2_5")))
        odds_under = float(_num(r.get("odds_under_2_5")))

        ev_min_over = EV_MIN_STERILE_OVER if sterile_flag else EV_MIN_OU25
        ev_min_under = float(EV_MIN_OU25)
        if assassin_flag:
            ev_min_under = float(max(ev_min_under, EV_MIN_ASSASSIN_UNDER))
        if timing_usable and late_goal_flag:
            ev_min_under = float(max(ev_min_under, EV_MIN_LATE_HEAVY_UNDER))

        _emit_candidate(
            market="OU25_OVER",
            odds=odds_over,
            p_model=p_over,
            ev_min_req=float(ev_min_over),
            odds_reason="G1_FAIL_MISSING_ODDS_OU25",
        )
        _emit_candidate(
            market="OU25_UNDER",
            odds=odds_under,
            p_model=p_under,
            ev_min_req=float(ev_min_under),
            odds_reason="G1_FAIL_MISSING_ODDS_OU25",
        )

        # BTTS candidates (only when odds exist in the input schema)
        if btts_supported:
            p_yes = _p_btts_yes(mu_home, mu_away)
            p_no = 1.0 - p_yes if np.isfinite(p_yes) else float("nan")

            odds_yes = float(
                _num(
                    r.get(
                        "odds_btts_yes",
                        r.get("odds_gg", r.get("gg")),
                    )
                )
            )
            odds_no = float(
                _num(
                    r.get(
                        "odds_btts_no",
                        r.get("odds_ng", r.get("ng")),
                    )
                )
            )

            _emit_candidate(
                market="BTTS_YES",
                odds=odds_yes,
                p_model=p_yes,
                ev_min_req=float(EV_MIN_BTTS),
                odds_reason="G1_FAIL_MISSING_ODDS_BTTS",
            )
            _emit_candidate(
                market="BTTS_NO",
                odds=odds_no,
                p_model=p_no,
                ev_min_req=float(EV_MIN_BTTS),
                odds_reason="G1_FAIL_MISSING_ODDS_BTTS",
            )

        # Milestone 7.2 timing markets (only if odds exist in the input)
        if timing_1h_supported:
            _emit_candidate(
                market="1H_OU05_OVER",
                odds=float(_num(r.get("odds_1h_over_0_5"))),
                p_model=float(_num(r.get("p_1h_over_0_5"))),
                ev_min_req=float(EV_MIN_TIMING),
                odds_reason="G1_FAIL_MISSING_ODDS_1H",
            )
            _emit_candidate(
                market="1H_OU05_UNDER",
                odds=float(_num(r.get("odds_1h_under_0_5"))),
                p_model=float(_num(r.get("p_1h_under_0_5"))),
                ev_min_req=float(EV_MIN_TIMING),
                odds_reason="G1_FAIL_MISSING_ODDS_1H",
            )

        if timing_2h05_supported:
            _emit_candidate(
                market="2H_OU05_OVER",
                odds=float(_num(r.get("odds_2h_over_0_5"))),
                p_model=float(_num(r.get("p_2h_over_0_5"))),
                ev_min_req=float(EV_MIN_TIMING),
                odds_reason="G1_FAIL_MISSING_ODDS_2H",
            )
            _emit_candidate(
                market="2H_OU05_UNDER",
                odds=float(_num(r.get("odds_2h_under_0_5"))),
                p_model=float(_num(r.get("p_2h_under_0_5"))),
                ev_min_req=float(EV_MIN_TIMING),
                odds_reason="G1_FAIL_MISSING_ODDS_2H",
            )

        if timing_2h15_supported:
            _emit_candidate(
                market="2H_OU15_OVER",
                odds=float(_num(r.get("odds_2h_over_1_5"))),
                p_model=float(_num(r.get("p_2h_over_1_5"))),
                ev_min_req=float(EV_MIN_TIMING),
                odds_reason="G1_FAIL_MISSING_ODDS_2H",
            )
            _emit_candidate(
                market="2H_OU15_UNDER",
                odds=float(_num(r.get("odds_2h_under_1_5"))),
                p_model=float(_num(r.get("p_2h_under_1_5"))),
                ev_min_req=float(EV_MIN_TIMING),
                odds_reason="G1_FAIL_MISSING_ODDS_2H",
            )

        if timing_after75_supported:
            _emit_candidate(
                market="GOAL_AFTER_75_YES",
                odds=float(_num(r.get("odds_goal_after_75_yes"))),
                p_model=float(_num(r.get("p_goal_after_75_yes"))),
                ev_min_req=float(EV_MIN_TIMING),
                odds_reason="G1_FAIL_MISSING_ODDS_AFTER75",
            )
            _emit_candidate(
                market="GOAL_AFTER_75_NO",
                odds=float(_num(r.get("odds_goal_after_75_no"))),
                p_model=float(_num(r.get("p_goal_after_75_no"))),
                ev_min_req=float(EV_MIN_TIMING),
                odds_reason="G1_FAIL_MISSING_ODDS_AFTER75",
            )

    # Select one best candidate per fixture
    selected: list[Candidate] = []
    if candidates:
        cand_df = pd.DataFrame([c.__dict__ for c in candidates])
        cand_df["market_priority"] = cand_df["market"].map(MARKET_PRIORITY_RANK).fillna(len(MARKET_PRIORITY)).astype(int)
        cand_df = cand_df.sort_values(
            ["fixture_idx", "score", "ev", "neff_min", "market_priority", "market"],
            ascending=[True, False, False, False, True, True],
            kind="mergesort",
        )
        top = cand_df.groupby("fixture_idx", as_index=False).head(1)
        candidate_fields = set(Candidate.__dataclass_fields__.keys())
        selected = [
            Candidate(**{k: v for k, v in row.items() if k in candidate_fields})
            for row in top.to_dict(orient="records")
        ]

    picks_rows: list[dict] = []
    for c in selected:
        rel = (
            min(c.neff_min / 10.0, 1.0)
            * min(c.press_n_min / 5.0, 1.0)
            * min(c.xg_n_min / 5.0, 1.0)
        )
        risk_penalty = 0.85 if (c.sterile_flag or c.assassin_flag) else 1.0
        confidence = float(c.ev * rel * risk_penalty)

        if confidence < 0.05:
            stake_units = 0.5
            stake_tier = "T1"
            stake_reason = "STAKE_T1_LOW_CONF"
        elif confidence < 0.09:
            stake_units = 1.0
            stake_tier = "T2"
            stake_reason = "STAKE_T2_MED_CONF"
        elif confidence < 0.13:
            stake_units = 1.5
            stake_tier = "T3"
            stake_reason = "STAKE_T3_HIGH_CONF"
        else:
            stake_units = 2.0
            stake_tier = "T4"
            stake_reason = "STAKE_T4_MAX_CONF"

        p_implied = float(_implied_prob(float(c.odds)))
        model_prob = float(c.p_model)
        value_margin = model_prob - p_implied
        risk_flags = "|".join(
            [
                name
                for name, cond in (
                    ("STERILE", c.sterile_flag),
                    ("ASSASSIN", c.assassin_flag),
                    ("SLOW_START", c.slow_start_flag),
                    ("LATE_GOALS", c.late_goal_flag),
                )
                if cond
            ]
        )

        picks_rows.append(
            {
                "fixture_datetime": c.fixture_datetime,
                "league": c.league,
                "home": c.home,
                "away": c.away,
                "run_asof_datetime": str(run_asof_dt) if not pd.isna(run_asof_dt) else "",
                "scope_country": scope_country or "",
                "scope_league": scope_league or "",
                "scope_season_start": scope_season_start.date().isoformat() if not pd.isna(scope_season_start) else "",
                "scope_season_end": scope_season_end.date().isoformat() if not pd.isna(scope_season_end) else "",
                "horizon_days": int(horizon_days),
                "market": c.market,
                "odds": float(c.odds),
                "p_model": float(c.p_model),
                "p_implied": p_implied,
                "model_prob": model_prob,
                "implied_prob": p_implied,
                "value_margin": float(value_margin) if np.isfinite(value_margin) else np.nan,
                "risk_flags": risk_flags,
                "ev": float(c.ev),
                "score": float(c.score),
                "mu_home": float(c.mu_home),
                "mu_away": float(c.mu_away),
                "mu_total": float(c.mu_total),
                "stake_units": float(stake_units),
                "stake_tier": stake_tier,
                "reason_codes": "|".join(("SELECT_TOP_SCORE", stake_reason)),
                "neff_min": float(c.neff_min),
                "press_n_min": float(c.press_n_min),
                "xg_n_min": float(c.xg_n_min),
                "sterile_flag": int(c.sterile_flag),
                "assassin_flag": int(c.assassin_flag),
                "timing_usable": int(c.timing_usable),
                "slow_start_flag": int(c.slow_start_flag),
                "late_goal_flag": int(c.late_goal_flag),
                "timing_early_share": float(c.timing_early_share) if np.isfinite(_num(c.timing_early_share)) else np.nan,
                "timing_late_share": float(c.timing_late_share) if np.isfinite(_num(c.timing_late_share)) else np.nan,
                "run_id": run_id,
                "input_hash": input_hash,
                "_confidence": confidence,
            }
        )

    picks_df = pd.DataFrame(picks_rows)
    if not picks_df.empty:
        picks_df = picks_df.sort_values(["fixture_datetime", "score"], ascending=[True, False], kind="mergesort")

    debug_df = pd.DataFrame(debug_rows)
    if not debug_df.empty:
        picked_key = {(r["fixture_datetime"], r["league"], r["home"], r["away"]): r["market"] for r in picks_rows}
        debug_df["selected"] = debug_df.apply(
            lambda r: picked_key.get((r["fixture_datetime"], r["league"], r["home"], r["away"])) == r["market"],
            axis=1,
        )
        debug_df = debug_df.sort_values(
            ["fixture_datetime", "league", "home", "away", "eligible", "score"],
            ascending=[True, True, True, True, False, False],
            kind="mergesort",
        )

    if debug_df.empty and len(debug_df.columns) == 0:
        debug_df = pd.DataFrame(
            columns=[
                "fixture_datetime",
                "league",
                "home",
                "away",
                "run_asof_datetime",
                "scope_country",
                "scope_league",
                "scope_season_start",
                "scope_season_end",
                "horizon_days",
                "market",
                "odds",
                "p_model",
                "ev",
                "ev_min_required",
                "score",
                "eligible",
                "reason_codes",
                "mu_total",
                "neff_min",
                "press_n_min",
                "xg_n_min",
                "sterile_flag",
                "assassin_flag",
                "timing_usable",
                "slow_start_flag",
                "late_goal_flag",
                "timing_early_share",
                "timing_late_share",
                "selected",
            ]
        )

    if not picks_df.empty:
        picks_df = picks_df.drop(columns=["_confidence"], errors="ignore")

    debug_df.attrs["scope"] = {"fixtures_in": fixtures_in, "fixtures_in_scope": fixtures_in_scope}

    if picks_df.empty and len(picks_df.columns) == 0:
        picks_df = pd.DataFrame(
            columns=[
                "fixture_datetime",
                "league",
                "home",
                "away",
                "run_asof_datetime",
                "scope_country",
                "scope_league",
                "scope_season_start",
                "scope_season_end",
                "horizon_days",
                "market",
                "odds",
                "p_model",
                "p_implied",
                "model_prob",
                "implied_prob",
                "value_margin",
                "risk_flags",
                "ev",
                "score",
                "mu_home",
                "mu_away",
                "mu_total",
                "stake_units",
                "stake_tier",
                "reason_codes",
                "neff_min",
                "press_n_min",
                "xg_n_min",
                "sterile_flag",
                "assassin_flag",
                "timing_usable",
                "slow_start_flag",
                "late_goal_flag",
                "timing_early_share",
                "timing_late_share",
                "run_id",
                "input_hash",
            ]
        )

    return picks_df.reset_index(drop=True), debug_df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Milestone 7.1: deterministic goals-only pick engine (OU2.5 + optional BTTS)")
    ap.add_argument("--in", dest="in_path", default="reports/cgm_upcoming_predictions.csv", help="Input predictions CSV")
    ap.add_argument("--out", dest="out_path", default="reports/picks.csv", help="Output picks CSV")
    ap.add_argument("--debug-out", dest="debug_out", default="reports/picks_debug.csv", help="Optional debug output CSV")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    debug_out = Path(args.debug_out) if args.debug_out else None

    if not in_path.exists():
        raise SystemExit(f"[pick_engine_goals] input not found: {in_path}")

    df = pd.read_csv(in_path)

    required = [
        "fixture_datetime",
        "league",
        "home",
        "away",
        "run_asof_datetime",
        "scope_country",
        "scope_league",
        "scope_season_start",
        "scope_season_end",
        "horizon_days",
        "mu_home",
        "mu_away",
        "odds_over_2_5",
        "odds_under_2_5",
        "neff_sim_H",
        "neff_sim_A",
        "press_stats_n_H",
        "press_stats_n_A",
        "xg_stats_n_H",
        "xg_stats_n_A",
        "sterile_flag",
        "assassin_flag",
    ]
    _require_columns(df, required, context=str(in_path))

    input_hash = file_sha256(in_path)
    run_id = f"PICKS_GOALS_{input_hash[:12]}"

    picks_df, debug_df = build_picks(df, input_hash=input_hash, run_id=run_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    picks_df.to_csv(out_path, index=False)

    if debug_out is not None:
        debug_out.parent.mkdir(parents=True, exist_ok=True)
        debug_df.to_csv(debug_out, index=False)

    scope = getattr(debug_df, "attrs", {}).get("scope", {})
    fixtures_in = int(scope.get("fixtures_in", len(df)))
    fixtures_in_scope = int(scope.get("fixtures_in_scope", len(df)))
    print(
        f"[pick_engine_goals] input_hash={input_hash} fixtures_in={fixtures_in} fixtures_in_scope={fixtures_in_scope} "
        f"picks={len(picks_df)} out={out_path}"
    )


if __name__ == "__main__":
    main()
