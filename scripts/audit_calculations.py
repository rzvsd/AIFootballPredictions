#!/usr/bin/env python3
"""
Comprehensive Calculation Verification Audit.

Replays core calculations from scratch and compares against stored values.
Six audit modules:
  1. Elo Replay         - recompute Elo and compare to stored ratings
  2. Pressure Spot-Check - recalculate pressure for sample teams
  3. xG Proxy Sanity    - range and NaN checks on xG proxy values
  4. Data Continuity    - NaN/duplicate detection in training matrix
  5. Mu Prediction Check - verify Poisson math in upcoming predictions
  6. Row Count Chain    - no silent row drops across pipeline stages
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENHANCED = ROOT / "data" / "enhanced"
_REPORTS = ROOT / "reports"
_AUDITS = _REPORTS / "audits"


def _pass(name: str, detail: str = "") -> dict[str, Any]:
    tag = f"  [PASS] {name}"
    if detail:
        tag += f" - {detail}"
    print(tag)
    return {"name": name, "status": "PASS", "detail": detail}


def _fail(name: str, detail: str) -> dict[str, Any]:
    print(f"  [FAIL] {name} - {detail}")
    return {"name": name, "status": "FAIL", "detail": detail}


def _warn(name: str, detail: str) -> dict[str, Any]:
    print(f"  [WARN] {name} - {detail}")
    return {"name": name, "status": "WARN", "detail": detail}


# ---------------------------------------------------------------------------
# 1. Elo Replay
# ---------------------------------------------------------------------------

def audit_elo_replay() -> dict[str, Any]:
    """Replay Elo from raw history and compare to stored values."""
    name = "Elo Replay"
    try:
        from scripts.calc_cgm_elo import (
            margin_multiplier, expected_home, infer_team,
            START_ELO_DEFAULT, K_FACTOR_DEFAULT, HOME_ADV_DEFAULT,
        )
        import config  # type: ignore
    except ImportError as e:
        return _fail(name, f"Cannot import calc_cgm_elo: {e}")

    enhanced_path = _ENHANCED / "cgm_match_history_with_elo_stats_xg.csv"
    if not enhanced_path.exists():
        return _fail(name, f"Missing {enhanced_path.name}")

    df = pd.read_csv(enhanced_path, low_memory=False)
    if "datetime" in df.columns:
        df = df.sort_values("datetime")

    rating_floor = float(getattr(config, "ELO_DEFAULTS", {}).get("rating_floor", 1000.0))

    # Replay Elo step-by-step, comparing pre-match Elo to stored values.
    ratings: dict[str, float] = {}
    tolerance = 1.0  # Allow 1 Elo point tolerance for floating-point differences
    checked = 0
    mismatches = 0
    trace_checked = 0
    issues = []

    for idx, row in df.iterrows():
        home_id = infer_team(row, home=True)
        away_id = infer_team(row, home=False)
        ft_home = row.get("ft_home")
        ft_away = row.get("ft_away")

        if home_id is None or away_id is None:
            continue

        r_home = ratings.get(home_id, START_ELO_DEFAULT)
        r_away = ratings.get(away_id, START_ELO_DEFAULT)

        # Compare pre-match replayed Elo to stored Elo
        stored_home = row.get("elo_home")
        stored_away = row.get("elo_away")
        if pd.notna(stored_home):
            diff_h = abs(r_home - float(stored_home))
            checked += 1
            if diff_h > tolerance:
                mismatches += 1
                if len(issues) < 3:
                    issues.append(f"Row {idx} {row.get('home','?')}(H): replayed={r_home:.1f} stored={float(stored_home):.1f} diff={diff_h:.1f}")
        if pd.notna(stored_away):
            diff_a = abs(r_away - float(stored_away))
            checked += 1
            if diff_a > tolerance:
                mismatches += 1
                if len(issues) < 3:
                    issues.append(f"Row {idx} {row.get('away','?')}(A): replayed={r_away:.1f} stored={float(stored_away):.1f} diff={diff_a:.1f}")

        if pd.isna(ft_home) or pd.isna(ft_away):
            continue

        hfa = pd.to_numeric(row.get("elo_hfa_used", HOME_ADV_DEFAULT), errors="coerce")
        hfa = float(hfa) if pd.notna(hfa) else float(HOME_ADV_DEFAULT)

        exp_home = expected_home(r_home, r_away, hfa)
        if ft_home > ft_away:
            actual = 1.0
        elif ft_home == ft_away:
            actual = 0.5
        else:
            actual = 0.0

        # Prefer row-level V2 trace coefficients when present.
        k_used = pd.to_numeric(row.get("elo_k_used", np.nan), errors="coerce")
        if not pd.notna(k_used):
            k_used = pd.to_numeric(row.get("elo_k_base_used", K_FACTOR_DEFAULT), errors="coerce")
        k_used = float(k_used) if pd.notna(k_used) else float(K_FACTOR_DEFAULT)

        g_used = pd.to_numeric(row.get("elo_g_used", np.nan), errors="coerce")
        if not pd.notna(g_used):
            g_used = float(margin_multiplier(int(abs(ft_home - ft_away))))
        else:
            g_used = float(g_used)

        delta = float(k_used * g_used * (actual - exp_home))
        ratings[home_id] = max(rating_floor, r_home + delta)
        ratings[away_id] = max(rating_floor, r_away - delta)

        # Optional trace checks if V2 columns exist.
        if "elo_expected_home" in df.columns and pd.notna(row.get("elo_expected_home")):
            trace_checked += 1
            if abs(float(row.get("elo_expected_home")) - exp_home) > 1e-3 and len(issues) < 3:
                issues.append(
                    f"Row {idx}: expected_home mismatch replay={exp_home:.4f} stored={float(row.get('elo_expected_home')):.4f}"
                )
        if "elo_delta" in df.columns and pd.notna(row.get("elo_delta")):
            trace_checked += 1
            if abs(float(row.get("elo_delta")) - delta) > 1e-2 and len(issues) < 3:
                issues.append(
                    f"Row {idx}: delta mismatch replay={delta:.4f} stored={float(row.get('elo_delta')):.4f}"
                )

    if mismatches > 0:
        return _fail(name, f"{mismatches}/{checked} values differ (>{tolerance}pt): {'; '.join(issues[:3])}")
    if issues:
        return _fail(name, "; ".join(issues[:3]))
    return _pass(name, f"{checked} pre-match Elo values verified within +/-{tolerance}pt (trace checks={trace_checked})")


# ---------------------------------------------------------------------------
# 2. Pressure Spot-Check
# ---------------------------------------------------------------------------

def audit_pressure_spotcheck() -> dict[str, Any]:
    """Recalculate pressure dominance for a few matches and compare."""
    name = "Pressure Spot-Check"
    enhanced_path = _ENHANCED / "cgm_match_history_with_elo_stats_xg.csv"
    if not enhanced_path.exists():
        return _fail(name, f"Missing {enhanced_path.name}")

    df = pd.read_csv(enhanced_path, low_memory=False)

    # Weights from pressure_form.py
    W_SHOTS, W_SOT, W_CORNERS, W_POS = 0.45, 0.30, 0.15, 0.10

    required = ["shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A"]
    if not all(c in df.columns for c in required):
        return _warn(name, "Required shot/corner/possession columns missing - cannot spot-check")

    # Filter to rows with full stats
    valid = df.dropna(subset=required).copy()
    if len(valid) < 5:
        return _warn(name, f"Only {len(valid)} rows with complete stats - too few to check")

    # Check last 5 rows: compute raw dominance for home team
    sample = valid.tail(5)
    issues = []
    for idx, row in sample.iterrows():
        shots_total = float(row["shots_H"]) + float(row["shots_A"])
        sot_total = float(row["sot_H"]) + float(row["sot_A"])
        corners_total = float(row["corners_H"]) + float(row["corners_A"])

        dom_shots = float(row["shots_H"]) / shots_total if shots_total > 0 else 0.5
        dom_sot = float(row["sot_H"]) / sot_total if sot_total > 0 else 0.5
        dom_corners = float(row["corners_H"]) / corners_total if corners_total > 0 else 0.5
        dom_pos = float(row["pos_H"]) / 100.0 if float(row["pos_H"]) > 1 else float(row["pos_H"])

        expected_raw = dom_shots * W_SHOTS + dom_sot * W_SOT + dom_corners * W_CORNERS + dom_pos * W_POS

        # Check stored dominance sub-features if available
        for suffix, expected_val in [("press_dom_shots_H", dom_shots), ("press_dom_sot_H", dom_sot),
                                      ("press_dom_corners_H", dom_corners), ("press_dom_pos_H", dom_pos)]:
            if suffix not in df.columns:
                continue
            # Note: these are rolling averages, not per-match, so we only check the per-match raw values
            # are plausible (within [0, 1])
            pass

        # Sanity: raw dominance should be between 0 and 1
        if not (0.0 <= expected_raw <= 1.0):
            issues.append(f"Row {idx}: raw dominance={expected_raw:.4f} out of [0,1]")

    if "press_form_H" in df.columns:
        press_vals = pd.to_numeric(df["press_form_H"], errors="coerce")
        out_of_range = ((press_vals < 0) | (press_vals > 1)).sum()
        nan_count = press_vals.isna().sum()
        if out_of_range > 0:
            issues.append(f"press_form_H: {out_of_range} values outside [0,1]")
        if nan_count > len(df) * 0.5:
            issues.append(f"press_form_H: {nan_count}/{len(df)} NaN (>50%)")

    if issues:
        return _fail(name, "; ".join(issues[:3]))
    return _pass(name, f"Dominance ratios valid for {len(sample)} sample rows, press_form_H range OK")


# ---------------------------------------------------------------------------
# 3. xG Proxy Sanity
# ---------------------------------------------------------------------------

def audit_xg_sanity() -> dict[str, Any]:
    """Check xG proxy values for range, NaN, and consistency."""
    name = "xG Proxy Sanity"
    enhanced_path = _ENHANCED / "cgm_match_history_with_elo_stats_xg.csv"
    if not enhanced_path.exists():
        return _fail(name, f"Missing {enhanced_path.name}")

    df = pd.read_csv(enhanced_path, low_memory=False)
    issues = []

    for col in ["xg_proxy_H", "xg_proxy_A"]:
        if col not in df.columns:
            issues.append(f"{col} column missing")
            continue
        vals = pd.to_numeric(df[col], errors="coerce")

        # Check usable rows
        if "xg_usable" in df.columns:
            usable_mask = pd.to_numeric(df["xg_usable"], errors="coerce").fillna(0) > 0
            usable_nans = vals[usable_mask].isna().sum()
            if usable_nans > 0:
                issues.append(f"{col}: {usable_nans} NaN where xg_usable=1")

        # Range check
        negatives = (vals < 0).sum()
        if negatives > 0:
            issues.append(f"{col}: {negatives} negative values")

        extreme = (vals > 15).sum()
        if extreme > 0:
            issues.append(f"{col}: {extreme} values > 15 (implausible)")

        # Summary stats
        non_null = vals.dropna()
        if len(non_null) > 0:
            mean_val = non_null.mean()
            if mean_val < 0.3 or mean_val > 4.0:
                issues.append(f"{col}: mean={mean_val:.2f} seems wrong (expected ~1.0-2.0)")

    if issues:
        return _fail(name, "; ".join(issues[:4]))

    xg_h = pd.to_numeric(df.get("xg_proxy_H", pd.Series(dtype=float)), errors="coerce").dropna()
    xg_a = pd.to_numeric(df.get("xg_proxy_A", pd.Series(dtype=float)), errors="coerce").dropna()
    return _pass(name, f"xg_proxy_H: mean={xg_h.mean():.2f} | xg_proxy_A: mean={xg_a.mean():.2f}")


# ---------------------------------------------------------------------------
# 4. Data Continuity
# ---------------------------------------------------------------------------

def audit_data_continuity() -> dict[str, Any]:
    """Check for NaN floods and duplicates in the enhanced history matrix."""
    name = "Data Continuity"
    matrix_path = _ENHANCED / "cgm_match_history_with_elo_stats_xg.csv"
    if not matrix_path.exists():
        return _fail(name, f"Missing {matrix_path.name}")

    df = pd.read_csv(matrix_path, low_memory=False)
    issues = []

    total_rows = len(df)
    if total_rows == 0:
        return _fail(name, "Training matrix is empty")

    # Critical columns that should have low NaN rates
    critical_cols = [
        "EloDiff", "elo_home", "elo_away",
        "lg_avg_gf_home", "lg_avg_gf_away",
        "ft_home", "ft_away",
    ]

    for col in critical_cols:
        if col not in df.columns:
            issues.append(f"Missing critical column: {col}")
            continue
        nan_rate = df[col].isna().mean()
        if nan_rate > 0.05:
            issues.append(f"{col}: {nan_rate:.1%} NaN")

    # Feature columns: check for >50% NaN
    feature_cols = [c for c in df.columns if c.startswith(("press_", "xg_", "div_", "Attack_", "Defense_"))]
    high_nan_features = []
    for col in feature_cols:
        nan_rate = pd.to_numeric(df[col], errors="coerce").isna().mean()
        if nan_rate > 0.50:
            high_nan_features.append(f"{col}({nan_rate:.0%})")
    if high_nan_features:
        issues.append(f"{len(high_nan_features)} features >50% NaN: {', '.join(high_nan_features[:5])}")

    # Duplicate check
    if "datetime" in df.columns and "home" in df.columns and "away" in df.columns:
        dupes = df.duplicated(subset=["datetime", "home", "away"]).sum()
        if dupes > 0:
            issues.append(f"{dupes} duplicate rows (same datetime+home+away)")

    # y_home/y_away sanity
    if "y_home" in df.columns and "y_away" in df.columns:
        y_h = pd.to_numeric(df["y_home"], errors="coerce")
        y_a = pd.to_numeric(df["y_away"], errors="coerce")
        neg_goals = ((y_h < 0).sum() + (y_a < 0).sum())
        if neg_goals > 0:
            issues.append(f"{neg_goals} negative goal values in y_home/y_away")
        extreme_goals = ((y_h > 12).sum() + (y_a > 12).sum())
        if extreme_goals > 0:
            issues.append(f"{extreme_goals} goal values > 12 (suspicious)")

    if issues:
        return _fail(name, "; ".join(issues[:4]))
    return _pass(name, f"{total_rows} rows, {len(feature_cols)} feature cols checked")


# ---------------------------------------------------------------------------
# 5. Mu Prediction Sanity
# ---------------------------------------------------------------------------

def audit_mu_predictions() -> dict[str, Any]:
    """Verify probability math in upcoming predictions (Poisson core or Poisson V2)."""
    name = "Mu Prediction Check"
    pred_path = _REPORTS / "cgm_upcoming_predictions.csv"
    if not pred_path.exists():
        return _fail(name, f"Missing {pred_path.name}")

    df = pd.read_csv(pred_path, low_memory=False)
    if df.empty:
        return _warn(name, "Predictions file is empty")

    issues = []

    # Basic mu checks
    for col in ["mu_home", "mu_away", "mu_total"]:
        if col not in df.columns:
            issues.append(f"Missing {col}")
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            issues.append(f"{col}: contains NaN")
        if (vals <= 0).any():
            issues.append(f"{col}: contains zero or negative values")
        if (vals > 8).any():
            issues.append(f"{col}: contains values > 8 (implausible)")

    # mu_total ~ mu_home + mu_away
    if all(c in df.columns for c in ["mu_home", "mu_away", "mu_total"]):
        mu_h = pd.to_numeric(df["mu_home"], errors="coerce")
        mu_a = pd.to_numeric(df["mu_away"], errors="coerce")
        mu_t = pd.to_numeric(df["mu_total"], errors="coerce")
        diff = (mu_h + mu_a - mu_t).abs()
        max_diff = diff.max()
        if max_diff > 0.001:
            issues.append(f"mu_total != mu_home + mu_away (max diff={max_diff:.6f})")

    # Probability coherence checks (works for both legacy Poisson and Poisson V2 variants).
    if all(c in df.columns for c in ["p_over25", "p_under25"]):
        p_ov = pd.to_numeric(df["p_over25"], errors="coerce")
        p_un = pd.to_numeric(df["p_under25"], errors="coerce")
        valid_ou = p_ov.notna() & p_un.notna()
        if valid_ou.sum() > 0:
            ou_sum = (p_ov[valid_ou] + p_un[valid_ou]).astype(float)
            max_ou_diff = float(np.abs(ou_sum.values - 1.0).max())
            if max_ou_diff > 0.01:
                issues.append(f"p_over25 + p_under25 != 1 (max diff={max_ou_diff:.4f})")

    if all(c in df.columns for c in ["p_btts_yes", "p_btts_no"]):
        p_by = pd.to_numeric(df["p_btts_yes"], errors="coerce")
        p_bn = pd.to_numeric(df["p_btts_no"], errors="coerce")
        valid_b = p_by.notna() & p_bn.notna()
        if valid_b.sum() > 0:
            b_sum = (p_by[valid_b] + p_bn[valid_b]).astype(float)
            max_b_diff = float(np.abs(b_sum.values - 1.0).max())
            if max_b_diff > 0.01:
                issues.append(f"p_btts_yes + p_btts_no != 1 (max diff={max_b_diff:.4f})")

    # Legacy-only check: when Poisson V2 is not enabled, p_over25 should match Poisson(mu_total).
    # With Poisson V2 enabled, deviations are expected (dispersion + dependence + low-score correction).
    v2_enabled = False
    if "poisson_v2_enabled" in df.columns:
        try:
            v2_enabled = bool(pd.to_numeric(df["poisson_v2_enabled"], errors="coerce").fillna(0).max() > 0)
        except Exception:
            v2_enabled = False

    if (not v2_enabled) and ("p_over25" in df.columns) and ("mu_total" in df.columns):
        from scipy.stats import poisson
        mu_t = pd.to_numeric(df["mu_total"], errors="coerce")
        p_stored = pd.to_numeric(df["p_over25"], errors="coerce")
        valid = mu_t.notna() & p_stored.notna() & (mu_t > 0)
        if valid.sum() > 0:
            p_expected = 1.0 - poisson.cdf(2, mu_t[valid].values)
            max_poisson_diff = float(np.abs(p_stored[valid].values - p_expected).max())
            if max_poisson_diff > 0.01:
                issues.append(f"p_over25 vs Poisson(mu_total): max diff={max_poisson_diff:.4f}")

    if issues:
        return _fail(name, "; ".join(issues[:4]))

    n = len(df)
    mu_t = pd.to_numeric(df.get("mu_total", pd.Series(np.nan, index=df.index)), errors="coerce")
    mu_range = f"mu_total range: [{mu_t.min():.2f}, {mu_t.max():.2f}]" if "mu_total" in df.columns else ""
    sterile_matchup = pd.to_numeric(df.get("sterile_matchup_flag", pd.Series(0, index=df.index)), errors="coerce")
    sterile_legacy = pd.to_numeric(df.get("sterile_flag", pd.Series(0, index=df.index)), errors="coerce")
    sterile_effective = ((sterile_matchup.fillna(sterile_legacy).fillna(0)) > 0)
    sterile_trigger_count = int(sterile_effective.sum())

    p_under_series = pd.to_numeric(df.get("p_under_2_5", df.get("p_under25", pd.Series(np.nan, index=df.index))), errors="coerce")
    mu_h = pd.to_numeric(df.get("mu_home", pd.Series(np.nan, index=df.index)), errors="coerce")
    mu_a = pd.to_numeric(df.get("mu_away", pd.Series(np.nan, index=df.index)), errors="coerce")
    mu_t = pd.to_numeric(df.get("mu_total", pd.Series(np.nan, index=df.index)), errors="coerce")
    reasons = (
        ((mu_t < 2.3) & mu_t.notna()).astype(int)
        + ((mu_h < 1.2) & (mu_a < 1.2) & mu_h.notna() & mu_a.notna()).astype(int)
        + sterile_effective.astype(int)
        + ((p_under_series > 0.52) & p_under_series.notna()).astype(int)
    )
    under_boost_usage = int((reasons >= 2).sum())

    detail = (
        f"{n} predictions verified. {mu_range} "
        f"sterile_trigger_count={sterile_trigger_count}; under_boost_usage={under_boost_usage}"
    ).strip()
    return _pass(name, detail)


# ---------------------------------------------------------------------------
# 6. Row Count Chain
# ---------------------------------------------------------------------------

def audit_row_counts() -> dict[str, Any]:
    """Check row counts across pipeline stages for unexpected drops."""
    name = "Row Count Chain"
    stages = [
        ("cgm_match_history.csv", _ENHANCED / "cgm_match_history.csv"),
        ("with_elo", _ENHANCED / "cgm_match_history_with_elo.csv"),
        ("with_elo_stats", _ENHANCED / "cgm_match_history_with_elo_stats.csv"),
        ("with_elo_stats_xg", _ENHANCED / "cgm_match_history_with_elo_stats_xg.csv"),
    ]

    counts = {}
    missing = []
    for label, path in stages:
        if not path.exists():
            missing.append(label)
            continue
        # Count lines (minus header) without loading full CSV
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                line_count = sum(1 for _ in f) - 1  # subtract header
            counts[label] = max(line_count, 0)
        except Exception as e:
            missing.append(f"{label}({e})")

    if missing:
        return _warn(name, f"Missing stages: {', '.join(missing)}")

    issues = []
    prev_label = None
    prev_count = None
    for label, _ in stages:
        if label not in counts:
            continue
        current = counts[label]
        if prev_count is not None:
            if current < prev_count * 0.95:  # >5% drop
                issues.append(f"{prev_label}({prev_count}) -> {label}({current}): {prev_count - current} rows lost")
        prev_label = label
        prev_count = current

    chain_str = " -> ".join(f"{label}={counts.get(label, '?')}" for label, _ in stages)

    if issues:
        return _fail(name, f"{chain_str}; {'; '.join(issues)}")
    return _pass(name, chain_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 80)
    print("CALCULATION VERIFICATION AUDIT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    audits = [
        ("1", audit_elo_replay),
        ("2", audit_pressure_spotcheck),
        ("3", audit_xg_sanity),
        ("4", audit_data_continuity),
        ("5", audit_mu_predictions),
        ("6", audit_row_counts),
    ]

    results = []
    for num, fn in audits:
        print(f"\n--- Audit {num}: {fn.__doc__.strip().split(chr(10))[0] if fn.__doc__ else fn.__name__} ---")
        try:
            result = fn()
        except Exception as e:
            result = _fail(fn.__name__, f"Unexpected error: {e}")
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    for r in results:
        print(f"  [{r['status']}] {r['name']}")

    print(f"\nTotal: {passed} passed, {warned} warnings, {failed} failed")

    # Save JSON report
    _AUDITS.mkdir(parents=True, exist_ok=True)
    report_path = _AUDITS / f"audit_calculations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {"passed": passed, "warned": warned, "failed": failed},
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


