"""
Milestone 7.1 audits: Goals-only Pick Engine determinism + gate correctness.

Intended usage (repo root):
  python -m scripts.audit_picks_goals
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_MARKETS = {
    "OU25_OVER",
    "OU25_UNDER",
    "BTTS_YES",
    "BTTS_NO",
    "1H_OU05_OVER",
    "1H_OU05_UNDER",
    "2H_OU05_OVER",
    "2H_OU05_UNDER",
    "2H_OU15_OVER",
    "2H_OU15_UNDER",
    "GOAL_AFTER_75_YES",
    "GOAL_AFTER_75_NO",
}
ODDS_MIN = 1.05

# Gate thresholds (must match cgm.pick_engine_goals defaults)
MU_TOTAL_MIN = 1.6
MU_TOTAL_MAX = 3.4
NEFF_MIN = 8.0
PRESS_EVID_MIN = 3.0
XG_EVID_MIN = 3.0

EV_MIN_OU25 = 0.04
EV_MIN_BTTS = 0.04
EV_MIN_TIMING = 0.05
EV_MIN_STERILE_OVER = 0.08
EV_MIN_ASSASSIN_UNDER = 0.08
EV_MIN_LATE_HEAVY_UNDER = 0.08


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_ev_min(market: str, sterile: int, assassin: int, timing_usable: int, late_goal: int) -> float:
    if market == "OU25_OVER":
        return EV_MIN_STERILE_OVER if sterile else EV_MIN_OU25
    if market == "OU25_UNDER":
        ev_min = EV_MIN_ASSASSIN_UNDER if assassin else EV_MIN_OU25
        if timing_usable and late_goal:
            ev_min = max(ev_min, EV_MIN_LATE_HEAVY_UNDER)
        return float(ev_min)
    if market.startswith("BTTS_"):
        return EV_MIN_BTTS
    return EV_MIN_TIMING


def _stake_from_confidence(confidence: float) -> tuple[str, float]:
    if confidence < 0.05:
        return ("T1", 0.5)
    if confidence < 0.09:
        return ("T2", 1.0)
    if confidence < 0.13:
        return ("T3", 1.5)
    return ("T4", 2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Milestone 7.1: goals-only Pick Engine audits")
    ap.add_argument("--predictions", default="reports/cgm_upcoming_predictions.csv", help="Input predictions CSV")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"[audit_picks_goals] predictions not found: {pred_path}")

    _print_header("Reproducibility (run twice -> same hash)")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out1 = td_path / "picks1.csv"
        out2 = td_path / "picks2.csv"
        dbg1 = td_path / "picks_debug1.csv"
        dbg2 = td_path / "picks_debug2.csv"

        cmd = [sys.executable, "-m", "cgm.pick_engine_goals", "--in", str(pred_path), "--out"]
        subprocess.run(cmd + [str(out1), "--debug-out", str(dbg1)], check=True)
        subprocess.run(cmd + [str(out2), "--debug-out", str(dbg2)], check=True)

        h1 = _md5_file(out1)
        h2 = _md5_file(out2)
        print("hash1:", h1)
        print("hash2:", h2)
        print("reproducible:", bool(h1 == h2))
        if h1 != h2:
            raise SystemExit("[audit_picks_goals] non-deterministic output (hash mismatch)")

        picks = pd.read_csv(out1)

    _print_header("Schema + market sanity")
    required_cols = [
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
    missing = [c for c in required_cols if c not in picks.columns]
    print("rows:", len(picks), "cols:", len(picks.columns), "missing_required:", missing)
    if missing:
        raise SystemExit(f"[audit_picks_goals] missing required output columns: {missing}")
    if picks.empty:
        print("picks: empty (ok if gates filtered all fixtures or odds missing)")
        return

    _print_header("Derived columns consistency")
    model_prob = pd.to_numeric(picks["model_prob"], errors="coerce")
    p_model = pd.to_numeric(picks["p_model"], errors="coerce")
    implied_prob = pd.to_numeric(picks["implied_prob"], errors="coerce")
    p_implied = pd.to_numeric(picks["p_implied"], errors="coerce")
    value_margin = pd.to_numeric(picks["value_margin"], errors="coerce")

    if not np.allclose(model_prob.fillna(-1.0), p_model.fillna(-1.0), atol=1e-12):
        raise SystemExit("[audit_picks_goals] model_prob must match p_model")
    if not np.allclose(implied_prob.fillna(-1.0), p_implied.fillna(-1.0), atol=1e-12):
        raise SystemExit("[audit_picks_goals] implied_prob must match p_implied")

    expected_margin = model_prob - implied_prob
    if not np.allclose(
        value_margin.fillna(-999.0),
        expected_margin.fillna(-999.0),
        atol=1e-12,
    ):
        raise SystemExit("[audit_picks_goals] value_margin must equal model_prob - implied_prob")

    bad_market = ~picks["market"].isin(sorted(ALLOWED_MARKETS))
    print("bad_market_rows:", int(bad_market.sum()))
    if bad_market.any():
        raise SystemExit("[audit_picks_goals] picks.csv contains invalid market values")

    _print_header("Odds sanity (G1)")
    odds = pd.to_numeric(picks["odds"], errors="coerce")
    bad_odds = odds.isna() | ~(odds > ODDS_MIN)
    print("bad_odds_rows:", int(bad_odds.sum()))
    if bad_odds.any():
        raise SystemExit("[audit_picks_goals] found picks with missing/invalid odds")

    _print_header("mu sanity bounds (G2)")
    mu_total = pd.to_numeric(picks["mu_total"], errors="coerce")
    g2_bad = ~(mu_total.between(MU_TOTAL_MIN, MU_TOTAL_MAX))
    print("g2_bad_rows:", int(g2_bad.sum()))
    if g2_bad.any():
        raise SystemExit("[audit_picks_goals] found picks violating mu_total bounds")

    _print_header("Scope sanity (no past picks)")
    fixture_dt = pd.to_datetime(picks["fixture_datetime"], errors="coerce")
    asof_dt = pd.to_datetime(picks["run_asof_datetime"], errors="coerce")
    uniq_asof = asof_dt.dropna().unique()
    if len(uniq_asof) != 1:
        raise SystemExit(f"[audit_picks_goals] run_asof_datetime must be constant; found {len(uniq_asof)} values")
    run_asof = pd.to_datetime(uniq_asof[0])

    if fixture_dt.isna().any():
        raise SystemExit("[audit_picks_goals] unparseable fixture_datetime in picks.csv")

    past = fixture_dt <= run_asof
    print("past_pick_rows:", int(past.sum()))
    if past.any():
        bad = picks.loc[past, ["fixture_datetime", "home", "away", "market"]].head(10)
        print(bad.to_string(index=False))
        raise SystemExit("[audit_picks_goals] picks.csv contains past/same-day fixtures (must be strictly > run_asof_datetime)")

    ss_vals = pd.to_datetime(picks["scope_season_start"], errors="coerce").dropna().unique()
    se_vals = pd.to_datetime(picks["scope_season_end"], errors="coerce").dropna().unique()
    if len(ss_vals) == 1 and len(se_vals) == 1:
        ss = pd.to_datetime(ss_vals[0]).normalize()
        se = pd.to_datetime(se_vals[0]).normalize()
        oob = ~fixture_dt.between(ss, se, inclusive="left")
        print("season_window_oob_rows:", int(oob.sum()))
        if oob.any():
            raise SystemExit("[audit_picks_goals] picks.csv contains fixtures outside scope season window")

    hz_vals = pd.to_numeric(picks["horizon_days"], errors="coerce").dropna().unique()
    if len(hz_vals) == 1 and np.isfinite(hz_vals[0]) and float(hz_vals[0]) > 0:
        hz = int(hz_vals[0])
        horizon_end = run_asof + pd.Timedelta(days=hz)
        hz_bad = fixture_dt > horizon_end
        print("horizon_oob_rows:", int(hz_bad.sum()))
        if hz_bad.any():
            raise SystemExit("[audit_picks_goals] picks.csv contains fixtures beyond horizon_days")

    _print_header("Reliability minimums (G3)")
    neff = pd.to_numeric(picks["neff_min"], errors="coerce")
    press_n = pd.to_numeric(picks["press_n_min"], errors="coerce")
    xg_n = pd.to_numeric(picks["xg_n_min"], errors="coerce")
    g3_bad = (neff < NEFF_MIN) | (press_n < PRESS_EVID_MIN) | (xg_n < XG_EVID_MIN)
    print("g3_bad_rows:", int(g3_bad.sum()))
    if g3_bad.any():
        raise SystemExit("[audit_picks_goals] found picks violating minimum evidence thresholds")

    _print_header("EV thresholds (G5 + risk adjustments)")
    sterile = pd.to_numeric(picks["sterile_flag"], errors="coerce").fillna(0).astype(int)
    assassin = pd.to_numeric(picks["assassin_flag"], errors="coerce").fillna(0).astype(int)
    timing_usable = pd.to_numeric(picks["timing_usable"], errors="coerce").fillna(0).astype(int)
    late_goal = pd.to_numeric(picks["late_goal_flag"], errors="coerce").fillna(0).astype(int)
    market = picks["market"].astype(str)
    ev = pd.to_numeric(picks["ev"], errors="coerce")

    expected = pd.Series(
        [
            _expected_ev_min(m, int(s), int(a), int(tu), int(lg))
            for m, s, a, tu, lg in zip(
                market.tolist(),
                sterile.tolist(),
                assassin.tolist(),
                timing_usable.tolist(),
                late_goal.tolist(),
            )
        ],
        index=picks.index,
        dtype=float,
    )
    bad_ev = ev < expected
    print("bad_ev_rows:", int(bad_ev.sum()))
    if bad_ev.any():
        bad = picks.loc[bad_ev, ["fixture_datetime", "home", "away", "market", "ev"]].head(10)
        print(bad.to_string(index=False))
        raise SystemExit("[audit_picks_goals] found picks with EV below required threshold")

    _print_header("Timing gate integrity (Milestone 7.2)")
    slow_start = pd.to_numeric(picks["slow_start_flag"], errors="coerce").fillna(0).astype(int)
    timing_markets = picks["market"].isin(
        [
            "1H_OU05_OVER",
            "1H_OU05_UNDER",
            "2H_OU05_OVER",
            "2H_OU05_UNDER",
            "2H_OU15_OVER",
            "2H_OU15_UNDER",
            "GOAL_AFTER_75_YES",
            "GOAL_AFTER_75_NO",
        ]
    )
    print("timing_market_picks:", int(timing_markets.sum()))
    if timing_markets.any():
        bad_timing_usable = timing_markets & ~(timing_usable > 0)
        if bad_timing_usable.any():
            raise SystemExit("[audit_picks_goals] timing market pick without timing_usable==1")

        bad_slow = (
            (picks["market"].astype(str) == "1H_OU05_UNDER") & ~(slow_start > 0)
        ) | (
            (picks["market"].astype(str) == "1H_OU05_OVER") & (slow_start > 0)
        )
        if bad_slow.any():
            raise SystemExit("[audit_picks_goals] 1H timing pick violates slow_start_flag gate")

        late_req = picks["market"].isin(["GOAL_AFTER_75_YES", "2H_OU05_OVER", "2H_OU15_OVER"])
        late_block = picks["market"].isin(["GOAL_AFTER_75_NO", "2H_OU05_UNDER", "2H_OU15_UNDER"])
        if (late_req & ~(late_goal > 0)).any() or (late_block & (late_goal > 0)).any():
            raise SystemExit("[audit_picks_goals] timing pick violates late_goal_flag gate")

    _print_header("Stake tier mapping")
    rel = (
        np.minimum(neff / 10.0, 1.0)
        * np.minimum(press_n / 5.0, 1.0)
        * np.minimum(xg_n / 5.0, 1.0)
    )
    risk_penalty = np.where((sterile > 0) | (assassin > 0), 0.85, 1.0)
    confidence = (ev * rel * risk_penalty).astype(float)

    expected_tier_units = pd.DataFrame([_stake_from_confidence(float(c)) for c in confidence], columns=["tier", "units"])
    bad_tier = picks["stake_tier"].astype(str) != expected_tier_units["tier"].astype(str)
    bad_units = pd.to_numeric(picks["stake_units"], errors="coerce") != pd.to_numeric(expected_tier_units["units"], errors="coerce")
    print("bad_tier_rows:", int(bad_tier.sum()), "bad_units_rows:", int(bad_units.sum()))
    if bad_tier.any() or bad_units.any():
        raise SystemExit("[audit_picks_goals] stake_tier/stake_units mismatch with confidence mapping")

    print("\n[ok] audit passed")


if __name__ == "__main__":
    main()
