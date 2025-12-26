"""
Milestone 4: Strategy Layer ("Pick Engine").

Turns model predictions into deterministic, explainable betting picks for:
  - 1X2
  - Over/Under 2.5

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


ALLOWED_MARKETS = {"1X2_HOME", "1X2_DRAW", "1X2_AWAY", "OU25_OVER", "OU25_UNDER"}
MARKET_PRIORITY = ["OU25_OVER", "OU25_UNDER", "1X2_HOME", "1X2_AWAY", "1X2_DRAW"]
MARKET_PRIORITY_RANK = {m: i for i, m in enumerate(MARKET_PRIORITY)}

# Gates - imported from config.py (single source of truth)
ODDS_MIN = getattr(config, "ODDS_MIN_FULL", 1.01)
MU_TOTAL_MIN = getattr(config, "MU_TOTAL_MIN", 1.6)
MU_TOTAL_MAX = getattr(config, "MU_TOTAL_MAX", 3.4)
NEFF_MIN = getattr(config, "NEFF_MIN_FULL", 6.0)
PRESS_EVID_MIN = getattr(config, "PRESS_EVID_MIN_FULL", 2.0)
XG_EVID_MIN = getattr(config, "XG_EVID_MIN_FULL", 2.0)

# EV thresholds - imported from config.py
EV_MIN_1X2 = getattr(config, "EV_MIN_1X2", 0.05)
EV_MIN_OU25 = getattr(config, "EV_MIN_OU25", 0.04)
EV_MIN_STERILE_1X2 = getattr(config, "EV_MIN_STERILE_1X2", 0.07)
EV_MIN_ASSASSIN_ANY = getattr(config, "EV_MIN_ASSASSIN_ANY", 0.07)

# Assassin stricter reliability for O/U
ASSASSIN_NEFF_MIN_OU25 = getattr(config, "ASSASSIN_NEFF_MIN_OU25", 7.0)
ASSASSIN_PRESS_EVID_MIN_OU25 = getattr(config, "ASSASSIN_PRESS_EVID_MIN_OU25", 3.0)
ASSASSIN_XG_EVID_MIN_OU25 = getattr(config, "ASSASSIN_XG_EVID_MIN_OU25", 3.0)


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
        msg = f"[pick_engine] missing required columns in {context}: {missing}"
        raise SystemExit(msg)


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
    mu_home: float
    mu_away: float
    mu_total: float
    neff_min: float
    press_n_min: float
    xg_n_min: float


def build_picks(
    df: pd.DataFrame,
    *,
    input_hash: str,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      picks_df: one row per selected pick
      debug_df: one row per candidate (eligible or not)
    """
    df0 = df.copy()

    # Normalize numerics (deterministic)
    for c in [
        "mu_home",
        "mu_away",
        "mu_total",
        "p_home",
        "p_draw",
        "p_away",
        "p_over_2_5",
        "p_under_2_5",
        "odds_home",
        "odds_draw",
        "odds_away",
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
    ]:
        if c in df0.columns:
            df0[c] = pd.to_numeric(df0[c], errors="coerce")

    if "mu_total" not in df0.columns:
        df0["mu_total"] = df0["mu_home"] + df0["mu_away"]
    if "p_over_2_5" not in df0.columns and "p_under_2_5" in df0.columns:
        df0["p_over_2_5"] = 1.0 - df0["p_under_2_5"]
    if "p_under_2_5" not in df0.columns and "p_over_2_5" in df0.columns:
        df0["p_under_2_5"] = 1.0 - df0["p_over_2_5"]

    # Deterministic sort for reproducibility
    dt = pd.to_datetime(df0.get("fixture_datetime"), errors="coerce")
    if dt.notna().any():
        df0["_fixture_dt"] = dt
    else:
        df0["_fixture_dt"] = pd.to_datetime(df0.get("date"), errors="coerce")
    df0 = df0.sort_values(["_fixture_dt", "league", "home", "away"], kind="mergesort").reset_index(drop=True)

    # Scope protection: never emit picks for past/out-of-window fixtures even if upstream failed.
    fixtures_in = int(len(df0))
    df0["_run_asof_dt"] = pd.to_datetime(df0.get("run_asof_datetime"), errors="coerce")
    asof_vals = df0["_run_asof_dt"].dropna().unique()
    if len(asof_vals) > 1:
        raise SystemExit(f"[pick_engine] run_asof_datetime must be constant; found {len(asof_vals)} values")
    run_asof_dt = pd.to_datetime(asof_vals[0]) if len(asof_vals) == 1 else pd.NaT
    if fixtures_in and pd.isna(run_asof_dt):
        raise SystemExit("[pick_engine] run_asof_datetime is required and must be parseable")

    scope_league_vals = pd.Series(df0.get("scope_league", pd.Series(dtype=object))).astype(str).str.strip()
    scope_league_clean = [v for v in scope_league_vals.dropna().unique().tolist() if v and v.lower() != "nan"]
    if len(set(scope_league_clean)) > 1:
        raise SystemExit(f"[pick_engine] scope_league must be constant; found {sorted(set(scope_league_clean))}")
    scope_league = scope_league_clean[0] if scope_league_clean else None

    scope_country_vals = pd.Series(df0.get("scope_country", pd.Series(dtype=object))).astype(str).str.strip()
    scope_country_clean = [v for v in scope_country_vals.dropna().unique().tolist() if v and v.lower() != "nan"]
    if len(set(scope_country_clean)) > 1:
        raise SystemExit(f"[pick_engine] scope_country must be constant; found {sorted(set(scope_country_clean))}")
    scope_country = scope_country_clean[0] if scope_country_clean else None

    ss_vals = pd.to_datetime(df0.get("scope_season_start"), errors="coerce").dropna().unique()
    se_vals = pd.to_datetime(df0.get("scope_season_end"), errors="coerce").dropna().unique()
    if len(ss_vals) > 1 or len(se_vals) > 1:
        raise SystemExit("[pick_engine] scope_season_start/scope_season_end must be constant when present")
    scope_season_start = pd.to_datetime(ss_vals[0]).normalize() if len(ss_vals) == 1 else pd.NaT
    scope_season_end = pd.to_datetime(se_vals[0]).normalize() if len(se_vals) == 1 else pd.NaT

    horizon_vals = pd.to_numeric(df0.get("horizon_days"), errors="coerce").dropna().unique()
    if len(horizon_vals) > 1:
        raise SystemExit("[pick_engine] horizon_days must be constant when present")
    horizon_days = int(horizon_vals[0]) if len(horizon_vals) == 1 and np.isfinite(horizon_vals[0]) else 0

    # Apply scope filters.
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
        horizon_days_s = int(_num(r.get("horizon_days")) if np.isfinite(_num(r.get("horizon_days"))) else 0)

        mu_home = float(r.get("mu_home", np.nan))
        mu_away = float(r.get("mu_away", np.nan))
        mu_total = float(r.get("mu_total", np.nan))

        sterile_flag = _as_int_flag(r.get("sterile_flag"))
        assassin_flag = _as_int_flag(r.get("assassin_flag"))

        neff_min = float(np.nanmin([r.get("neff_sim_H", np.nan), r.get("neff_sim_A", np.nan)]))
        press_n_min = float(np.nanmin([r.get("press_stats_n_H", np.nan), r.get("press_stats_n_A", np.nan)]))
        xg_n_min = float(np.nanmin([r.get("xg_stats_n_H", np.nan), r.get("xg_stats_n_A", np.nan)]))

        g2_ok = bool(np.isfinite(mu_total) and MU_TOTAL_MIN <= mu_total <= MU_TOTAL_MAX)
        g3_ok = bool(
            np.isfinite(neff_min)
            and np.isfinite(press_n_min)
            and np.isfinite(xg_n_min)
            and neff_min >= NEFF_MIN
            and press_n_min >= PRESS_EVID_MIN
            and xg_n_min >= XG_EVID_MIN
        )
        fixture_block_reasons: list[str] = []
        if not g2_ok:
            if np.isfinite(mu_total) and mu_total < MU_TOTAL_MIN:
                fixture_block_reasons.append("G2_FAIL_LOW_MU_TOTAL")
            else:
                fixture_block_reasons.append("G2_FAIL_HIGH_MU_TOTAL")
        if not g3_ok:
            if not np.isfinite(neff_min) or neff_min < NEFF_MIN:
                fixture_block_reasons.append("G3_FAIL_LOW_NEFF")
            if not np.isfinite(press_n_min) or press_n_min < PRESS_EVID_MIN:
                fixture_block_reasons.append("G3_FAIL_LOW_PRESS_EVIDENCE")
            if not np.isfinite(xg_n_min) or xg_n_min < XG_EVID_MIN:
                fixture_block_reasons.append("G3_FAIL_LOW_XG_EVIDENCE")

        # Market-level odds gates
        odds_home = float(r.get("odds_home", np.nan))
        odds_draw = float(r.get("odds_draw", np.nan))
        odds_away = float(r.get("odds_away", np.nan))
        odds_over = float(r.get("odds_over_2_5", np.nan))
        odds_under = float(r.get("odds_under_2_5", np.nan))

        g1_1x2_ok = all(_is_sane_odds(o) for o in (odds_home, odds_draw, odds_away))
        g1_ou_ok = all(_is_sane_odds(o) for o in (odds_over, odds_under))

        # Candidate list (even if blocked, for debug determinism)
        cand_specs = [
            ("1X2_HOME", float(r.get("p_home", np.nan)), odds_home, "1X2"),
            ("1X2_DRAW", float(r.get("p_draw", np.nan)), odds_draw, "1X2"),
            ("1X2_AWAY", float(r.get("p_away", np.nan)), odds_away, "1X2"),
            ("OU25_OVER", float(r.get("p_over_2_5", np.nan)), odds_over, "OU25"),
            ("OU25_UNDER", float(r.get("p_under_2_5", np.nan)), odds_under, "OU25"),
        ]

        for market, p_model, odds, group in cand_specs:
            reasons: list[str] = []
            eligible = True

            if fixture_block_reasons:
                reasons.extend(fixture_block_reasons)
                eligible = False

            if group == "1X2" and not g1_1x2_ok:
                reasons.append("G1_FAIL_MISSING_ODDS_1X2")
                eligible = False
            if group == "OU25" and not g1_ou_ok:
                reasons.append("G1_FAIL_MISSING_ODDS_OU25")
                eligible = False

            # Risk gating (G4)
            if eligible and sterile_flag == 1 and market == "OU25_OVER":
                reasons.append("G4_BLOCK_OU_STERILE")
                eligible = False

            if eligible and assassin_flag == 1 and market.startswith("1X2"):
                # Deterministic variant: block all 1X2 unless EV is very high (handled here as G4 reason).
                ev_now = _ev(p_model, odds)
                if not np.isfinite(ev_now) or ev_now < EV_MIN_ASSASSIN_ANY:
                    reasons.append("G4_BLOCK_1X2_ASSASSIN")
                    eligible = False

            if eligible and assassin_flag == 1 and market.startswith("OU25"):
                # Spec: allow O/U, but require stricter reliability evidence.
                if not (
                    neff_min >= ASSASSIN_NEFF_MIN_OU25
                    and press_n_min >= ASSASSIN_PRESS_EVID_MIN_OU25
                    and xg_n_min >= ASSASSIN_XG_EVID_MIN_OU25
                ):
                    reasons.append("G4_BLOCK_OU_ASSASSIN")
                    eligible = False

            # EV gating (G5)
            ev_min_req = EV_MIN_1X2 if market.startswith("1X2") else EV_MIN_OU25
            if sterile_flag == 1 and market.startswith("1X2"):
                ev_min_req = max(ev_min_req, EV_MIN_STERILE_1X2)
            if assassin_flag == 1:
                ev_min_req = max(ev_min_req, EV_MIN_ASSASSIN_ANY)

            ev_val = _ev(p_model, odds)
            if eligible and (not np.isfinite(ev_val) or ev_val < ev_min_req):
                reasons.append("G5_FAIL_LOW_EV")
                eligible = False

            score = float("nan")
            if eligible:
                score = float(
                    ev_val
                    + 0.01 * math.log(1.0 + max(neff_min, 0.0))
                    + 0.005 * (max(press_n_min, 0.0) + max(xg_n_min, 0.0))
                )
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
                    "horizon_days": int(horizon_days_s),
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
                }
            )

    # Select one best candidate per fixture
    selected: list[Candidate] = []
    if candidates:
        cand_df = pd.DataFrame([c.__dict__ for c in candidates])
        # Deterministic tie-breaks (per fixture):
        # 1) higher score
        # 2) higher EV
        # 3) higher neff_min
        # 4) fixed market priority (avoids alphabetical weirdness)
        # 5) market name (last resort)
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
            [name for name, cond in (("STERILE", c.sterile_flag), ("ASSASSIN", c.assassin_flag)) if cond]
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
                "run_id": run_id,
                "input_hash": input_hash,
                "_confidence": confidence,
            }
        )

    picks_df = pd.DataFrame(picks_rows)
    if not picks_df.empty:
        picks_df = picks_df.sort_values(["fixture_datetime", "score"], ascending=[True, False], kind="mergesort")

    # Debug output: mark selected
    debug_df = pd.DataFrame(debug_rows)
    if not debug_df.empty:
        picked_key = {(r["fixture_datetime"], r["league"], r["home"], r["away"]): r["market"] for r in picks_rows}
        debug_df["selected"] = debug_df.apply(
            lambda r: picked_key.get((r["fixture_datetime"], r["league"], r["home"], r["away"])) == r["market"],
            axis=1,
        )
        debug_df = debug_df.sort_values(["fixture_datetime", "league", "home", "away", "eligible", "score"], ascending=[True, True, True, True, False, False], kind="mergesort")
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
                "selected",
            ]
        )

    # Drop internal helper cols from main output
    if not picks_df.empty:
        picks_df = picks_df.drop(columns=["_confidence"], errors="ignore")

    # Keep scope stats for console logging (no randomness).
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
                "run_id",
                "input_hash",
            ]
        )

    return picks_df.reset_index(drop=True), debug_df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Milestone 4: deterministic pick engine (1X2 + O/U 2.5)")
    ap.add_argument("--in", dest="in_path", default="reports/cgm_upcoming_predictions.csv", help="Input predictions CSV")
    ap.add_argument("--out", dest="out_path", default="reports/picks.csv", help="Output picks CSV")
    ap.add_argument("--debug-out", dest="debug_out", default="reports/picks_debug.csv", help="Optional debug output CSV")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    debug_out = Path(args.debug_out) if args.debug_out else None

    if not in_path.exists():
        raise SystemExit(f"[pick_engine] input not found: {in_path}")

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
        "p_home",
        "p_draw",
        "p_away",
        "p_over_2_5",
        "p_under_2_5",
        "odds_home",
        "odds_draw",
        "odds_away",
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
    run_id = f"PICKS_{input_hash[:12]}"

    picks_df, debug_df = build_picks(df, input_hash=input_hash, run_id=run_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    picks_df.to_csv(out_path, index=False)

    if debug_out is not None:
        debug_out.parent.mkdir(parents=True, exist_ok=True)
        debug_df.to_csv(debug_out, index=False)

    # Minimal console summary (deterministic)
    scope = getattr(debug_df, "attrs", {}).get("scope", {})
    fixtures_in = int(scope.get("fixtures_in", len(df)))
    fixtures_in_scope = int(scope.get("fixtures_in_scope", len(df)))
    print(
        f"[pick_engine] input_hash={input_hash} fixtures_in={fixtures_in} fixtures_in_scope={fixtures_in_scope} "
        f"picks={len(picks_df)} out={out_path}"
    )


if __name__ == "__main__":
    main()
