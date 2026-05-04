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
}
MARKET_PRIORITY = [
    "OU25_OVER",
    "OU25_UNDER",
    "BTTS_YES",
    "BTTS_NO",
]
MARKET_PRIORITY_RANK = {m: i for i, m in enumerate(MARKET_PRIORITY)}

# Gates - imported from config.py (uses stricter _GOALS variants)
ODDS_MIN = getattr(config, "ODDS_MIN_GOALS", 1.05)
MU_TOTAL_MIN = getattr(config, "MU_TOTAL_MIN", 1.6)
MU_TOTAL_MAX = getattr(config, "MU_TOTAL_MAX", 3.4)
NEFF_MIN = getattr(config, "NEFF_MIN_GOALS", 8.0)
PRESS_EVID_MIN = getattr(config, "PRESS_EVID_MIN_GOALS", 3.0)
XG_EVID_MIN = getattr(config, "XG_EVID_MIN_GOALS", 3.0)

# EV thresholds are retained only for backwards-compatible report columns.
# Current strategy treats EV as diagnostic metadata, not a gate or ranking input.
EV_PHANTOM_MODE = getattr(config, "PICK_ENGINE_EV_PHANTOM_MODE", True)
ODDS_PHANTOM_MODE = getattr(config, "PICK_ENGINE_ODDS_PHANTOM_MODE", True)
ROUND_QUOTA_ENABLED = getattr(config, "PICK_ENGINE_ROUND_QUOTA_ENABLED", True)
MAX_PICKS_PER_LEAGUE_ROUND = int(getattr(config, "PICK_ENGINE_MAX_PICKS_PER_LEAGUE_ROUND", 5) or 0)
MIN_TARGET_PICKS_PER_LEAGUE_ROUND = int(getattr(config, "PICK_ENGINE_MIN_TARGET_PICKS_PER_LEAGUE_ROUND", 4) or 0)
PRICED_ODDS_BONUS = float(getattr(config, "PICK_ENGINE_PRICED_ODDS_BONUS", 0.0) or 0.0)
MIN_MODEL_PROB = float(getattr(config, "PICK_ENGINE_MIN_MODEL_PROB", 0.50) or 0.0)
EVIDENCE_GATES_BY_LEAGUE = getattr(config, "PICK_ENGINE_EVIDENCE_GATES_BY_LEAGUE", {})
FALLBACK_EVIDENCE_GATE = getattr(config, "PICK_ENGINE_FALLBACK_EVIDENCE_GATE", {})
USE_GATE_PRIORITY_BY_LEAGUE = getattr(config, "PICK_ENGINE_USE_GATE_PRIORITY_BY_LEAGUE", {})
MARKET_BLOCKLIST_BY_LEAGUE = getattr(config, "PICK_ENGINE_MARKET_BLOCKLIST_BY_LEAGUE", {})
GATE_SCORE_ADJUSTMENTS_BY_LEAGUE = getattr(config, "PICK_ENGINE_GATE_SCORE_ADJUSTMENTS_BY_LEAGUE", {})
MARKET_SCORE_ADJUSTMENTS_BY_LEAGUE = getattr(config, "PICK_ENGINE_MARKET_SCORE_ADJUSTMENTS_BY_LEAGUE", {})
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

# Low-Scoring Scenario Detector (Under 2.5 Enhancement)
LOW_SCORING_ENABLED = getattr(config, "LOW_SCORING_ENABLED", True)
LOW_SCORING_MU_THRESHOLD = getattr(config, "LOW_SCORING_MU_THRESHOLD", 2.3)
LOW_SCORING_XG_FORM_THRESHOLD = getattr(config, "LOW_SCORING_XG_FORM_THRESHOLD", 1.0)
LOW_SCORING_UNDER_SCORE_BONUS = getattr(config, "LOW_SCORING_UNDER_SCORE_BONUS", 0.04)
LOW_SCORING_EV_THRESHOLD_REDUCTION = getattr(config, "LOW_SCORING_EV_THRESHOLD_REDUCTION", 0.02)

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


def _gate_value(gate: object, key: str, default: float) -> float:
    if not isinstance(gate, dict):
        return float(default)
    value = _num(gate.get(key, default))
    return float(value if np.isfinite(value) else default)


def _evidence_gate_for(league: str, *, fallback: bool = False) -> tuple[float, float, float]:
    if fallback:
        gate = FALLBACK_EVIDENCE_GATE
    elif isinstance(EVIDENCE_GATES_BY_LEAGUE, dict):
        gate = EVIDENCE_GATES_BY_LEAGUE.get(league, {})
    else:
        gate = {}
    return (
        _gate_value(gate, "neff_min", NEFF_MIN),
        _gate_value(gate, "press_min", PRESS_EVID_MIN),
        _gate_value(gate, "xg_min", XG_EVID_MIN),
    )


def _use_gate_priority_for(league: str) -> bool:
    if not isinstance(USE_GATE_PRIORITY_BY_LEAGUE, dict):
        return True
    return bool(USE_GATE_PRIORITY_BY_LEAGUE.get(league, True))


def _market_blocked_for_league(league: str, market: str) -> bool:
    if not isinstance(MARKET_BLOCKLIST_BY_LEAGUE, dict):
        return False
    blocked = MARKET_BLOCKLIST_BY_LEAGUE.get(league, [])
    if isinstance(blocked, str):
        blocked = [blocked]
    try:
        return market in set(blocked)
    except TypeError:
        return False


def _gate_score_adjustment(league: str, gate_tier: str) -> float:
    if not isinstance(GATE_SCORE_ADJUSTMENTS_BY_LEAGUE, dict):
        return 0.0
    league_adjustments = GATE_SCORE_ADJUSTMENTS_BY_LEAGUE.get(league, {})
    if not isinstance(league_adjustments, dict):
        return 0.0
    value = _num(league_adjustments.get(gate_tier, 0.0))
    return float(value if np.isfinite(value) else 0.0)


def _evidence_failures(
    *,
    neff_min: float,
    press_n_min: float,
    xg_n_min: float,
    neff_gate: float,
    press_gate: float,
    xg_gate: float,
) -> list[str]:
    failures: list[str] = []
    if not (np.isfinite(neff_min) and neff_min >= neff_gate):
        failures.append("G3_FAIL_LOW_NEFF")
    if not (np.isfinite(press_n_min) and press_n_min >= press_gate):
        failures.append("G3_FAIL_LOW_PRESS_EVIDENCE")
    if not (np.isfinite(xg_n_min) and xg_n_min >= xg_gate):
        failures.append("G3_FAIL_LOW_XG_EVIDENCE")
    return failures


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


def _is_low_scoring_scenario(
    mu_total: float,
    mu_home: float,
    mu_away: float,
    sterile_flag: int,
    p_under: float,
) -> tuple[bool, str]:
    """
    Detect low-scoring scenarios where Under 2.5 should be favored.
    
    Returns:
        (is_low_scoring, reason): Tuple of boolean flag and reason string.
    
    Criteria for low-scoring detection:
    1. mu_total is below threshold (expected goals suggest few goals)
    2. Both teams have low individual mu (neither is a high-scorer)
    3. Sterile flags present (teams struggle to create)
    4. P(Under) is meaningfully high (model believes in Under)
    """
    if not LOW_SCORING_ENABLED:
        return False, ""
    
    reasons = []
    
    # Check if mu_total is low
    if np.isfinite(mu_total) and mu_total < LOW_SCORING_MU_THRESHOLD:
        reasons.append(f"LOW_MU_TOTAL({mu_total:.2f}<{LOW_SCORING_MU_THRESHOLD})")
    
    # Check if both teams are low-scoring individually
    if np.isfinite(mu_home) and np.isfinite(mu_away):
        if mu_home < 1.2 and mu_away < 1.2:
            reasons.append(f"BOTH_LOW_MU(H={mu_home:.2f},A={mu_away:.2f})")
    
    # Sterile flag boosts confidence
    if sterile_flag:
        reasons.append("STERILE_FLAG")
    
    # High Under probability from model
    if np.isfinite(p_under) and p_under > 0.52:
        reasons.append(f"HIGH_P_UNDER({p_under:.2f})")
    
    # Need at least 2 reasons to trigger low-scoring bonus
    is_low_scoring = len(reasons) >= 2
    
    return is_low_scoring, "|".join(reasons) if reasons else ""


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
    gate_tier: str
    low_scoring_scenario: int  # 1 if low-scoring scenario detected, 0 otherwise


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

        calib_ou_thresh = None
        calib_btts_thresh = None
        if CALIBRATION_ENABLED:
            cal = _CALIBRATION.get(league, {})
            if cal.get("sample_size", 0) >= CALIBRATION_MIN_SAMPLES:
                if "ou25_optimal_threshold" in cal:
                    calib_ou_thresh = float(cal["ou25_optimal_threshold"])
                if "btts_optimal_threshold" in cal:
                    calib_btts_thresh = float(cal["btts_optimal_threshold"])

        p_over_eff = p_over
        if not np.isfinite(p_over_eff) and np.isfinite(p_under):
            p_over_eff = 1.0 - p_under
        p_btts_yes = float("nan")

        sterile_flag = _as_int_flag(r.get("sterile_flag"))
        assassin_flag = _as_int_flag(r.get("assassin_flag"))
        timing_usable = _as_int_flag(r.get("timing_usable"))
        slow_start_flag = _as_int_flag(r.get("slow_start_flag"))
        late_goal_flag = _as_int_flag(r.get("late_goal_flag"))
        timing_early_share = float(_num(r.get("timing_early_share")))
        timing_late_share = float(_num(r.get("timing_late_share")))

        hard_fail: list[str] = []
        if not (np.isfinite(mu_total) and MU_TOTAL_MIN <= mu_total <= MU_TOTAL_MAX):
            hard_fail.append("G2_FAIL_LOW_MU_TOTAL" if (np.isfinite(mu_total) and mu_total < MU_TOTAL_MIN) else "G2_FAIL_HIGH_MU_TOTAL")
        primary_neff_gate, primary_press_gate, primary_xg_gate = _evidence_gate_for(league)
        fallback_neff_gate, fallback_press_gate, fallback_xg_gate = _evidence_gate_for(league, fallback=True)
        primary_evidence_fail = _evidence_failures(
            neff_min=neff_min,
            press_n_min=press_n_min,
            xg_n_min=xg_n_min,
            neff_gate=primary_neff_gate,
            press_gate=primary_press_gate,
            xg_gate=primary_xg_gate,
        )
        fallback_evidence_fail = _evidence_failures(
            neff_min=neff_min,
            press_n_min=press_n_min,
            xg_n_min=xg_n_min,
            neff_gate=fallback_neff_gate,
            press_gate=fallback_press_gate,
            xg_gate=fallback_xg_gate,
        )
        if not primary_evidence_fail:
            gate_tier = "primary"
            base_fail = list(hard_fail)
        else:
            gate_tier = "fallback"
            base_fail = list(hard_fail) + list(fallback_evidence_fail)

        # Detect low-scoring scenario
        is_low_scoring, low_scoring_reason = _is_low_scoring_scenario(
            mu_total, mu_home, mu_away, sterile_flag, p_under
        )
        low_scoring_flag_int = 1 if is_low_scoring else 0

        def _score(p_model_val: float) -> float:
            prob = p_model_val if np.isfinite(p_model_val) else 0.5
            confidence_edge = max(prob - 0.50, 0.0) * 2.0
            return float(
                confidence_edge
                + 0.01 * math.log(1.0 + max(neff_min, 0.0))
                + 0.005 * (max(press_n_min, 0.0) + max(xg_n_min, 0.0))
            )

        def _priced_bonus(odds_val: float) -> float:
            return float(PRICED_ODDS_BONUS if _is_sane_odds(odds_val) else 0.0)

        def _market_adjustment(market_name: str) -> float:
            league_adjustments = MARKET_SCORE_ADJUSTMENTS_BY_LEAGUE.get(league, {})
            if not isinstance(league_adjustments, dict):
                return 0.0
            try:
                return float(league_adjustments.get(market_name, 0.0) or 0.0)
            except Exception:
                return 0.0

        def _emit_candidate(*, market: str, odds: float, p_model: float, ev_min_req: float, odds_reason: str) -> None:
            reasons: list[str] = list(base_fail)
            if _market_blocked_for_league(league, market):
                reasons.append("G7_FAIL_LEAGUE_MARKET_BLOCKED")

            # Apply low-scoring adjustments for Under 2.5
            effective_ev_min = 0.0 if EV_PHANTOM_MODE else ev_min_req
            score_bonus = 0.0
            
            if market == "OU25_UNDER" and is_low_scoring:
                if not EV_PHANTOM_MODE:
                    effective_ev_min = float(max(0.0, ev_min_req - LOW_SCORING_EV_THRESHOLD_REDUCTION))
                score_bonus = float(LOW_SCORING_UNDER_SCORE_BONUS)
            if market in {"OU25_OVER", "OU25_UNDER"} and calib_ou_thresh is not None:
                if not np.isfinite(p_over_eff):
                    reasons.append("G6_FAIL_CALIB_OU25")
                elif market == "OU25_OVER" and p_over_eff < calib_ou_thresh:
                    reasons.append("G6_FAIL_CALIB_OU25_OVER")
                elif market == "OU25_UNDER" and (1.0 - p_over_eff) < calib_ou_thresh: # Assuming symmetric threshold or check logic?
                    # If we use same threshold for Prob(Outcome)
                    pass 

            if market in {"BTTS_YES", "BTTS_NO"} and calib_btts_thresh is not None:
                if not np.isfinite(p_btts_yes):
                    reasons.append("G6_FAIL_CALIB_BTTS")
                elif market == "BTTS_YES" and p_btts_yes < calib_btts_thresh:
                    reasons.append("G6_FAIL_CALIB_BTTS_YES")
                elif market == "BTTS_NO" and p_btts_yes > calib_btts_thresh:
                    reasons.append("G6_FAIL_CALIB_BTTS_NO")
            if ODDS_PHANTOM_MODE:
                if np.isfinite(odds) and odds <= ODDS_MIN:
                    reasons.append("G1_FAIL_LOW_ODDS")
            elif not (np.isfinite(odds) and odds > 1.0):
                reasons.append(odds_reason)
            elif not _is_sane_odds(odds):
                reasons.append("G1_FAIL_LOW_ODDS")

            if not (np.isfinite(p_model) and p_model > MIN_MODEL_PROB):
                reasons.append("G5_FAIL_LOW_MODEL_PROB")
            
            # Skip calculation if already failed hard gates (optimization)
            # But let's calculate to see EV in debug
            
            ev_val = _ev(p_model, odds)
            
            if not EV_PHANTOM_MODE and not (np.isfinite(ev_val) and ev_val >= effective_ev_min):
                reasons.append(f"G4_FAIL_LOW_EV({ev_val:.3f}<{effective_ev_min:.3f})")

            # Add low scoring specific reasons if relevant
            if market == "OU25_UNDER" and is_low_scoring:
               # We don't append a fail reason, but we might want to log it in debug?
               # We can add a synthetic "PASS" reason or just check the flag later.
               pass

            eligible = len(reasons) == 0
            candidate_score = (
                _score(p_model)
                + score_bonus
                + _priced_bonus(odds)
                + _market_adjustment(market)
                + _gate_score_adjustment(league, gate_tier)
            )

            if eligible:
                candidates.append(
                    Candidate(
                        fixture_idx=idx,
                        fixture_datetime=fixture_datetime,
                        league=league,
                        home=home,
                        away=away,
                        market=market,
                        odds=odds,
                        p_model=p_model,
                        ev=ev_val,
                        score=candidate_score,
                        ev_min_required=effective_ev_min,
                        reason_codes=tuple(sorted(reasons)),
                        sterile_flag=sterile_flag,
                        assassin_flag=assassin_flag,
                        timing_usable=timing_usable,
                        slow_start_flag=slow_start_flag,
                        late_goal_flag=late_goal_flag,
                        timing_early_share=timing_early_share,
                        timing_late_share=timing_late_share,
                        mu_home=mu_home,
                        mu_away=mu_away,
                        mu_total=mu_total,
                        neff_min=neff_min,
                        press_n_min=press_n_min,
                        xg_n_min=xg_n_min,
                        gate_tier=gate_tier,
                        low_scoring_scenario=low_scoring_flag_int,
                    )
                )

            # Always add to debug rows
            debug_rows.append(
                {
                    "fixture_datetime": fixture_datetime,
                    "league": league,
                    "home": home,
                    "away": away,
                    "market": market,
                    "odds": odds,
                    "p_model": p_model,
                    "ev": ev_val,
                    "ev_min_req": effective_ev_min,
                    "score": candidate_score if eligible else float("nan"),
                    "eligible": eligible,
                    "reasons": "|".join(sorted(reasons)),
                    "mu_total": mu_total,
                    "gate_tier": gate_tier,
                    "primary_neff_gate": primary_neff_gate,
                    "primary_press_gate": primary_press_gate,
                    "primary_xg_gate": primary_xg_gate,
                    "fallback_neff_gate": fallback_neff_gate,
                    "fallback_press_gate": fallback_press_gate,
                    "fallback_xg_gate": fallback_xg_gate,
                    "is_low_scoring": low_scoring_flag_int,
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
            p_btts_yes = p_yes
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

    # Select one best candidate per fixture
    selected: list[Candidate] = []
    if candidates:
        cand_df = pd.DataFrame([c.__dict__ for c in candidates])
        cand_df["market_priority"] = cand_df["market"].map(MARKET_PRIORITY_RANK).fillna(len(MARKET_PRIORITY)).astype(int)
        cand_df["gate_rank"] = cand_df["gate_tier"].map({"primary": 0, "fallback": 1}).fillna(1).astype(int)
        fixture_groups: list[pd.DataFrame] = []
        for _league, group in cand_df.groupby("league", sort=True):
            if _use_gate_priority_for(str(_league)):
                group = group.sort_values(
                    ["fixture_idx", "gate_rank", "score", "p_model", "neff_min", "market_priority", "market"],
                    ascending=[True, True, False, False, False, True, True],
                    kind="mergesort",
                )
            else:
                group = group.sort_values(
                    ["fixture_idx", "score", "p_model", "gate_rank", "neff_min", "market_priority", "market"],
                    ascending=[True, False, False, True, False, True, True],
                    kind="mergesort",
                )
            fixture_groups.append(group.groupby("fixture_idx", as_index=False).head(1))
        top = pd.concat(fixture_groups, ignore_index=True) if fixture_groups else cand_df.iloc[0:0]
        if ROUND_QUOTA_ENABLED and MAX_PICKS_PER_LEAGUE_ROUND > 0 and not top.empty:
            selected_groups: list[pd.DataFrame] = []
            for _league, group in top.groupby("league", sort=True):
                if _use_gate_priority_for(str(_league)):
                    ordered = group.sort_values(
                        ["gate_rank", "score", "p_model", "neff_min", "market_priority", "market"],
                        ascending=[True, False, False, False, True, True],
                        kind="mergesort",
                    )
                    primary = ordered[ordered["gate_tier"] == "primary"]
                    if len(primary) >= MIN_TARGET_PICKS_PER_LEAGUE_ROUND:
                        chosen = primary.head(MAX_PICKS_PER_LEAGUE_ROUND)
                    else:
                        chosen = ordered.head(MAX_PICKS_PER_LEAGUE_ROUND)
                else:
                    ordered = group.sort_values(
                        ["score", "p_model", "gate_rank", "neff_min", "market_priority", "market"],
                        ascending=[False, False, True, False, True, True],
                        kind="mergesort",
                    )
                    chosen = ordered.head(MAX_PICKS_PER_LEAGUE_ROUND)
                selected_groups.append(chosen)
            top = pd.concat(selected_groups, ignore_index=True) if selected_groups else top.iloc[0:0]
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
        confidence_edge = max(0.0, float(c.p_model) - 0.50) if np.isfinite(c.p_model) else 0.0
        confidence = float(confidence_edge * rel * risk_penalty)

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

        reason_codes = ["SELECT_TOP_SCORE", "EV_PHANTOM", c.gate_tier.upper(), stake_reason]

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
                "reason_codes": "|".join(reason_codes),
                "neff_min": float(c.neff_min),
                "press_n_min": float(c.press_n_min),
                "xg_n_min": float(c.xg_n_min),
                "gate_tier": c.gate_tier,
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
                "gate_tier",
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
                "gate_tier",
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
