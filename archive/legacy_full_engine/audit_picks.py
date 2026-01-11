"""
Milestone 4 audits: Pick Engine determinism + gate correctness.

Intended usage (repo root):
  python -m scripts.audit_picks
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

# Import centralized constants from config.py
try:
    import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import config


ALLOWED_MARKETS = {"1X2_HOME", "1X2_DRAW", "1X2_AWAY", "OU25_OVER", "OU25_UNDER"}

# Gate thresholds - imported from config.py (single source of truth)
ODDS_MIN = getattr(config, "ODDS_MIN_FULL", 1.01)
MU_TOTAL_MIN = getattr(config, "MU_TOTAL_MIN", 1.6)
MU_TOTAL_MAX = getattr(config, "MU_TOTAL_MAX", 3.4)
NEFF_MIN = getattr(config, "NEFF_MIN_FULL", 6.0)
PRESS_EVID_MIN = getattr(config, "PRESS_EVID_MIN_FULL", 2.0)
XG_EVID_MIN = getattr(config, "XG_EVID_MIN_FULL", 2.0)

EV_MIN_1X2 = getattr(config, "EV_MIN_1X2", 0.05)
EV_MIN_OU25 = getattr(config, "EV_MIN_OU25", 0.04)
EV_MIN_STERILE_1X2 = getattr(config, "EV_MIN_STERILE_1X2", 0.07)
EV_MIN_ASSASSIN_ANY = getattr(config, "EV_MIN_ASSASSIN_ANY", 0.07)

ASSASSIN_NEFF_MIN_OU25 = getattr(config, "ASSASSIN_NEFF_MIN_OU25", 7.0)
ASSASSIN_PRESS_EVID_MIN_OU25 = getattr(config, "ASSASSIN_PRESS_EVID_MIN_OU25", 3.0)
ASSASSIN_XG_EVID_MIN_OU25 = getattr(config, "ASSASSIN_XG_EVID_MIN_OU25", 3.0)


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _sha256_file(path: Path) -> str:
    """SHA256 hash for file integrity (consistent with pick engines)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_ev_min(market: str, sterile: int, assassin: int) -> float:
    ev_min = EV_MIN_1X2 if market.startswith("1X2") else EV_MIN_OU25
    if sterile and market.startswith("1X2"):
        ev_min = max(ev_min, EV_MIN_STERILE_1X2)
    if assassin:
        ev_min = max(ev_min, EV_MIN_ASSASSIN_ANY)
    return ev_min


def _stake_from_confidence(confidence: float) -> tuple[str, float]:
    if confidence < 0.05:
        return ("T1", 0.5)
    if confidence < 0.09:
        return ("T2", 1.0)
    if confidence < 0.13:
        return ("T3", 1.5)
    return ("T4", 2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Milestone 4: Pick Engine audits")
    ap.add_argument("--predictions", default="reports/cgm_upcoming_predictions.csv", help="Input predictions CSV")
    ap.add_argument("--picks", default="reports/picks.csv", help="Output picks CSV path (optional; audit uses temp by default)")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"[audit_picks] predictions not found: {pred_path}")

    input_hash = _sha256_file(pred_path)

    _print_header("Reproducibility (run twice -> same hash)")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out1 = td_path / "picks1.csv"
        out2 = td_path / "picks2.csv"
        dbg1 = td_path / "picks_debug1.csv"
        dbg2 = td_path / "picks_debug2.csv"

        cmd = [sys.executable, "-m", "cgm.pick_engine", "--in", str(pred_path), "--out"]
        subprocess.run(cmd + [str(out1), "--debug-out", str(dbg1)], check=True)
        subprocess.run(cmd + [str(out2), "--debug-out", str(dbg2)], check=True)

        h1 = _sha256_file(out1)
        h2 = _sha256_file(out2)
        print("hash1:", h1)
        print("hash2:", h2)
        print("reproducible:", bool(h1 == h2))
        if h1 != h2:
            raise SystemExit("[audit_picks] non-deterministic output (hash mismatch)")

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
        "run_id",
        "input_hash",
    ]
    missing = [c for c in required_cols if c not in picks.columns]
    print("rows:", len(picks), "cols:", len(picks.columns), "missing_required:", missing)
    if missing:
        raise SystemExit(f"[audit_picks] missing required output columns: {missing}")
    if picks.empty:
        print("picks: empty (ok if gates filtered all fixtures)")
        return

    _print_header("Derived columns consistency")
    model_prob = pd.to_numeric(picks["model_prob"], errors="coerce")
    p_model = pd.to_numeric(picks["p_model"], errors="coerce")
    implied_prob = pd.to_numeric(picks["implied_prob"], errors="coerce")
    p_implied = pd.to_numeric(picks["p_implied"], errors="coerce")
    value_margin = pd.to_numeric(picks["value_margin"], errors="coerce")

    if not np.allclose(model_prob.fillna(-1.0), p_model.fillna(-1.0), atol=1e-12):
        raise SystemExit("[audit_picks] model_prob must match p_model")
    if not np.allclose(implied_prob.fillna(-1.0), p_implied.fillna(-1.0), atol=1e-12):
        raise SystemExit("[audit_picks] implied_prob must match p_implied")

    expected_margin = model_prob - implied_prob
    if not np.allclose(
        value_margin.fillna(-999.0),
        expected_margin.fillna(-999.0),
        atol=1e-12,
    ):
        raise SystemExit("[audit_picks] value_margin must equal model_prob - implied_prob")

    bad_market = ~picks["market"].isin(sorted(ALLOWED_MARKETS))
    print("bad_market_rows:", int(bad_market.sum()))
    if bad_market.any():
        raise SystemExit("[audit_picks] picks.csv contains invalid market values")

    _print_header("Odds sanity (G1)")
    odds = pd.to_numeric(picks["odds"], errors="coerce")
    bad_odds = odds.isna() | ~(odds > ODDS_MIN)
    print("bad_odds_rows:", int(bad_odds.sum()))
    if bad_odds.any():
        raise SystemExit("[audit_picks] found picks with missing/invalid odds")

    _print_header("mu sanity bounds (G2)")
    mu_total = pd.to_numeric(picks["mu_total"], errors="coerce")
    g2_bad = ~(mu_total.between(MU_TOTAL_MIN, MU_TOTAL_MAX))
    print("g2_bad_rows:", int(g2_bad.sum()))
    if g2_bad.any():
        raise SystemExit("[audit_picks] found picks violating mu_total bounds")

    _print_header("Scope sanity (no past picks)")
    fixture_dt = pd.to_datetime(picks["fixture_datetime"], errors="coerce")
    asof_dt = pd.to_datetime(picks["run_asof_datetime"], errors="coerce")
    uniq_asof = asof_dt.dropna().unique()
    if len(uniq_asof) != 1:
        raise SystemExit(f"[audit_picks] run_asof_datetime must be constant; found {len(uniq_asof)} values")
    run_asof = pd.to_datetime(uniq_asof[0])

    bad_dt = fixture_dt.isna()
    if bad_dt.any():
        raise SystemExit("[audit_picks] unparseable fixture_datetime in picks.csv")

    past = fixture_dt <= run_asof
    print("past_pick_rows:", int(past.sum()))
    if past.any():
        bad = picks.loc[past, ["fixture_datetime", "home", "away", "market"]].head(10)
        print(bad.to_string(index=False))
        raise SystemExit("[audit_picks] picks.csv contains past/same-day fixtures (must be strictly > run_asof_datetime)")

    # Window checks (if provided)
    ss_vals = pd.to_datetime(picks["scope_season_start"], errors="coerce").dropna().unique()
    se_vals = pd.to_datetime(picks["scope_season_end"], errors="coerce").dropna().unique()
    if len(ss_vals) == 1 and len(se_vals) == 1:
        ss = pd.to_datetime(ss_vals[0]).normalize()
        se = pd.to_datetime(se_vals[0]).normalize()
        oob = ~fixture_dt.between(ss, se, inclusive="left")
        print("season_window_oob_rows:", int(oob.sum()))
        if oob.any():
            bad = picks.loc[oob, ["fixture_datetime", "home", "away", "market"]].head(10)
            print(bad.to_string(index=False))
            raise SystemExit("[audit_picks] picks.csv contains fixtures outside scope season window")

    # League/country checks (if scope is non-empty)
    scope_league = picks["scope_league"].astype(str).str.strip().replace("nan", "")
    scope_league_vals = [v for v in scope_league.unique().tolist() if v]
    if len(set(scope_league_vals)) == 1:
        sl = scope_league_vals[0]
        bad_lg = picks["league"].astype(str) != sl
        print("scope_league_mismatch_rows:", int(bad_lg.sum()))
        if bad_lg.any():
            raise SystemExit("[audit_picks] picks.csv contains rows outside scope_league")

    scope_country = picks["scope_country"].astype(str).str.strip().replace("nan", "")
    scope_country_vals = [v for v in scope_country.unique().tolist() if v]
    if "country" in picks.columns and len(set(scope_country_vals)) == 1:
        sc = scope_country_vals[0]
        bad_ct = picks["country"].astype(str) != sc
        print("scope_country_mismatch_rows:", int(bad_ct.sum()))
        if bad_ct.any():
            raise SystemExit("[audit_picks] picks.csv contains rows outside scope_country")

    # Horizon check (if enabled)
    hz_vals = pd.to_numeric(picks["horizon_days"], errors="coerce").dropna().unique()
    if len(hz_vals) == 1 and np.isfinite(hz_vals[0]) and float(hz_vals[0]) > 0:
        hz = int(hz_vals[0])
        horizon_end = run_asof + pd.Timedelta(days=hz)
        hz_bad = fixture_dt > horizon_end
        print("horizon_oob_rows:", int(hz_bad.sum()))
        if hz_bad.any():
            raise SystemExit("[audit_picks] picks.csv contains fixtures beyond horizon_days")

    _print_header("Reliability minimums (G3)")
    neff = pd.to_numeric(picks["neff_min"], errors="coerce")
    press_n = pd.to_numeric(picks["press_n_min"], errors="coerce")
    xg_n = pd.to_numeric(picks["xg_n_min"], errors="coerce")
    g3_bad = (neff < NEFF_MIN) | (press_n < PRESS_EVID_MIN) | (xg_n < XG_EVID_MIN)
    print("g3_bad_rows:", int(g3_bad.sum()))
    if g3_bad.any():
        raise SystemExit("[audit_picks] found picks violating minimum evidence thresholds")

    _print_header("Risk blocks (G4)")
    sterile = pd.to_numeric(picks["sterile_flag"], errors="coerce").fillna(0).astype(int)
    assassin = pd.to_numeric(picks["assassin_flag"], errors="coerce").fillna(0).astype(int)

    sterile_over = (sterile == 1) & (picks["market"] == "OU25_OVER")
    print("sterile_over_block_violations:", int(sterile_over.sum()))
    if sterile_over.any():
        raise SystemExit("[audit_picks] found OU25_OVER pick while sterile_flag==1")

    assassin_ou = (assassin == 1) & picks["market"].str.startswith("OU25")
    assassin_ou_bad = assassin_ou & (
        (neff < ASSASSIN_NEFF_MIN_OU25)
        | (press_n < ASSASSIN_PRESS_EVID_MIN_OU25)
        | (xg_n < ASSASSIN_XG_EVID_MIN_OU25)
    )
    print("assassin_ou_strict_rel_violations:", int(assassin_ou_bad.sum()))
    if assassin_ou_bad.any():
        raise SystemExit("[audit_picks] found OU25 pick with assassin_flag==1 but without strict reliability")

    _print_header("EV thresholds (G5)")
    ev = pd.to_numeric(picks["ev"], errors="coerce")
    ev_min = picks.apply(
        lambda r: _expected_ev_min(str(r["market"]), int(r["sterile_flag"]), int(r["assassin_flag"])),
        axis=1,
    )
    ev_bad = ev.isna() | (ev < ev_min - 1e-12)
    print("ev_bad_rows:", int(ev_bad.sum()))
    if ev_bad.any():
        bad = picks.loc[ev_bad, ["fixture_datetime", "home", "away", "market", "ev", "sterile_flag", "assassin_flag"]].head(10)
        print(bad.to_string(index=False))
        raise SystemExit("[audit_picks] found picks below EV threshold")

    _print_header("Stake tier mapping")
    rel = (
        np.minimum(neff / 10.0, 1.0)
        * np.minimum(press_n / 5.0, 1.0)
        * np.minimum(xg_n / 5.0, 1.0)
    )
    risk_penalty = np.where((sterile == 1) | (assassin == 1), 0.85, 1.0)
    confidence = ev.to_numpy(dtype=float) * rel.to_numpy(dtype=float) * risk_penalty.astype(float)
    expected = [ _stake_from_confidence(float(c)) for c in confidence ]
    exp_tier = pd.Series([t for t, _ in expected], index=picks.index)
    exp_units = pd.Series([u for _, u in expected], index=picks.index)

    tier_bad = picks["stake_tier"].astype(str) != exp_tier.astype(str)
    units_bad = (pd.to_numeric(picks["stake_units"], errors="coerce") - exp_units).abs() > 1e-12
    print("tier_bad_rows:", int(tier_bad.sum()), "units_bad_rows:", int(units_bad.sum()))
    if tier_bad.any() or units_bad.any():
        bad = picks.loc[tier_bad | units_bad, ["fixture_datetime", "home", "away", "market", "ev", "neff_min", "press_n_min", "xg_n_min", "sterile_flag", "assassin_flag", "stake_tier", "stake_units"]].head(10)
        print(bad.to_string(index=False))
        raise SystemExit("[audit_picks] stake tier/units mapping mismatch")

    _print_header("Metadata sanity")
    print("input_hash_matches:", bool((picks["input_hash"] == input_hash).all()))
    if not (picks["input_hash"] == input_hash).all():
        raise SystemExit("[audit_picks] picks.csv input_hash does not match predictions md5")

    _print_header("OK")
    print("all audits passed")


if __name__ == "__main__":
    main()
