"""
Frankenstein Features per Match (training matrix).

Transforms CGM match history into a feature matrix with rolling stats, Elo bands,
attack/defense indices, and interaction features for model training.

Inputs:
  - data/enhanced/cgm_match_history_with_elo_stats_xg.csv (history + Elo + gameplay stats + xG proxy)
  - data/enhanced/team_baselines.csv (team baselines)

Outputs:
  - data/enhanced/frankenstein_training.csv
    Columns: features + targets (y_home, y_away)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import config
from cgm.elo_similarity import prepare_histories, kernel_similarity
from cgm.pressure_inputs import ensure_pressure_inputs
from cgm.pressure_form import add_pressure_form_features
from cgm.pressure_divergence import add_pressure_divergence_features
from cgm.xg_form import add_xg_form_features
from cgm.pressure_xg_disagreement import add_pressure_xg_disagreement_features
from cgm.h2h_features import add_h2h_features
from cgm.league_features import add_league_features


def _load(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime" in df.columns:
        return df
    dt = pd.to_datetime(df["date"] + " " + df.get("time", "").astype(str), errors="coerce")
    # Fallback: try date only
    dt2 = pd.to_datetime(df["date"], errors="coerce")
    df["datetime"] = dt.fillna(dt2)
    return df


def _rolling_stats(group: pd.DataFrame, gf_col: str, ga_col: str, shots_col: str, sot_col: str, cor_col: str,
                   poss_col: str, prefix: str, windows: List[int]) -> pd.DataFrame:
    g = group.sort_values("datetime").copy()
    for w in windows:
        g[f"{prefix}_gf_L{w}"] = g[gf_col].shift().rolling(w, min_periods=1).mean()
        g[f"{prefix}_ga_L{w}"] = g[ga_col].shift().rolling(w, min_periods=1).mean()
        if shots_col in g:
            g[f"{prefix}_shots_for_L{w}"] = g[shots_col].astype(float).shift().rolling(w, min_periods=1).mean()
        if sot_col in g:
            g[f"{prefix}_sot_for_L{w}"] = g[sot_col].astype(float).shift().rolling(w, min_periods=1).mean()
        if cor_col in g:
            g[f"{prefix}_cor_for_L{w}"] = g[cor_col].astype(float).shift().rolling(w, min_periods=1).mean()
        if poss_col in g:
            g[f"{prefix}_poss_for_L{w}"] = g[poss_col].astype(float).shift().rolling(w, min_periods=1).mean()
    return g


def _apply_band_conditionals(df: pd.DataFrame, team_col: str, band_col: str, gf_col: str, ga_col: str,
                             prefix: str) -> pd.DataFrame:
    """Compute conditional GF/GA averages vs current band using past matches."""
    df = df.sort_values("datetime").copy()
    out_gf = []
    out_ga = []
    for _, row in df.iterrows():
        band = row[band_col]
        # past only
        past = df[df["datetime"] < row["datetime"]]
        if pd.notna(band):
            past_band = past[past[band_col] == band]
        else:
            past_band = past
        out_gf.append(past_band[gf_col].mean() if not past_band.empty else np.nan)
        out_ga.append(past_band[ga_col].mean() if not past_band.empty else np.nan)
    df[f"{prefix}_gf_vs_band"] = out_gf
    df[f"{prefix}_ga_vs_band"] = out_ga
    return df


def build_frankenstein(data_dir: str = "data/enhanced",
                       match_history_file: str = "cgm_match_history_with_elo_stats_xg.csv",
                       out_path: str = "data/enhanced/frankenstein_training.csv",
                       out_full_path: str | None = "data/enhanced/frankenstein_training_full.csv",
                       windows: List[int] | None = None,
                       alpha: float = 0.5,
                       beta: float = 1.0) -> Path:
    if windows is None:
        windows = [5, 10]
    mh_path = Path(data_dir) / match_history_file
    if not mh_path.exists():
        raise FileNotFoundError(f"{mh_path} not found (run Milestones 1 & 2 first).")

    df = _load(mh_path)
    df = _ensure_datetime(df)
    # Milestone 2: ensure split stats exist before numeric coercion (shots/corners can be "H-A" strings)
    df = ensure_pressure_inputs(df)
    # Coerce numeric columns
    num_cols = [
        "ft_home", "ft_away", "ht_home", "ht_away",
        # prefer calculated Elo columns
        "elo_home_calc", "elo_away_calc",
        "elo_diff",  # keep legacy diff if present
        "p_home", "p_draw", "p_away",
        "fair_home", "fair_draw", "fair_away",
        "odds_home", "odds_draw", "odds_away",
        "p_over", "p_under", "fair_over", "fair_under", "odds_over", "odds_under",
        "shots", "shots_on_target", "corners",
        "possession_home", "possession_away",
        # Pressure canonical inputs (split)
        "shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A",
        "lg_avg_gf_home", "lg_avg_gf_away",
        "lg_avg_shots_home", "lg_avg_shots_away",
        "lg_avg_sot_home", "lg_avg_sot_away",
        "lg_avg_cor_home", "lg_avg_cor_away",
        "lg_avg_poss_home", "lg_avg_poss_away",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_values("datetime", inplace=True)

    # Swap in calculated Elo as canonical (overwrite legacy cols for downstream)
    if "elo_home_calc" in df.columns and "elo_away_calc" in df.columns:
        df["elo_home"] = df["elo_home_calc"]
        df["elo_away"] = df["elo_away_calc"]
        df["elo_diff"] = df["elo_home"] - df["elo_away"]

    # Home context rolling
    home_grouped = df.groupby("home", group_keys=False).apply(
        _rolling_stats,
        gf_col="ft_home",
        ga_col="ft_away",
        shots_col="shots_H",
        sot_col="sot_H",
        cor_col="corners_H",
        poss_col="pos_H",
        prefix="H",
        windows=windows,
    )
    # Away context rolling
    away_grouped = home_grouped.groupby("away", group_keys=False).apply(
        _rolling_stats,
        gf_col="ft_away",
        ga_col="ft_home",
        shots_col="shots_A",
        sot_col="sot_A",
        cor_col="corners_A",
        poss_col="pos_A",
        prefix="A",
        windows=windows,
    )

    df_roll = away_grouped

    # Band conditionals
    df_roll = df_roll.groupby("home", group_keys=False).apply(
        _apply_band_conditionals,
        team_col="home",
        band_col="Band_H",
        gf_col="ft_home",
        ga_col="ft_away",
        prefix="H",
    )
    df_roll = df_roll.groupby("away", group_keys=False).apply(
        _apply_band_conditionals,
        team_col="away",
        band_col="Band_A",
        gf_col="ft_away",
        ga_col="ft_home",
        prefix="A",
    )

    # Efficiencies
    for w in windows:
        # home
        shots_for = df_roll.get(f"H_shots_for_L{w}")
        sot_for = df_roll.get(f"H_sot_for_L{w}")
        gf_for = df_roll.get(f"H_gf_L{w}")
        df_roll[f"H_shot_quality_L{w}"] = (sot_for + alpha) / (shots_for + beta) if shots_for is not None else np.nan
        df_roll[f"H_finish_rate_L{w}"] = (gf_for + alpha) / (sot_for + beta) if sot_for is not None else np.nan
        # away
        shots_for_a = df_roll.get(f"A_shots_for_L{w}")
        sot_for_a = df_roll.get(f"A_sot_for_L{w}")
        gf_for_a = df_roll.get(f"A_gf_L{w}")
        df_roll[f"A_shot_quality_L{w}"] = (sot_for_a + alpha) / (shots_for_a + beta) if shots_for_a is not None else np.nan
        df_roll[f"A_finish_rate_L{w}"] = (gf_for_a + alpha) / (sot_for_a + beta) if sot_for_a is not None else np.nan

    # Attack/defense indices vs league averages (using L5 as default)
    def _ratio(num, den):
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f == 0 or np.isnan(den_f) or np.isnan(num_f):
                return np.nan
            return num_f / den_f
        except Exception:
            return np.nan

    df_roll["Attack_H"] = df_roll.apply(
        lambda r: _ratio(r.get("H_gf_L5"), r.get("lg_avg_gf_home")), axis=1
    )
    df_roll["Defense_H"] = df_roll.apply(
        lambda r: _ratio(r.get("H_ga_L5"), r.get("lg_avg_gf_away")), axis=1
    )
    df_roll["Attack_A"] = df_roll.apply(
        lambda r: _ratio(r.get("A_gf_L5"), r.get("lg_avg_gf_away")), axis=1
    )
    df_roll["Defense_A"] = df_roll.apply(
        lambda r: _ratio(r.get("A_ga_L5"), r.get("lg_avg_gf_home")), axis=1
    )

    df_roll["Expected_Destruction_H"] = df_roll["Attack_H"] * df_roll["Defense_A"]
    df_roll["Expected_Destruction_A"] = df_roll["Attack_A"] * df_roll["Defense_H"]

    # Milestone 2: Pressure Cooker (venue-aware rolling dominance + divergence vs Elo)
    df_roll = ensure_pressure_inputs(df_roll)
    df_roll = add_pressure_form_features(df_roll, window=10)
    df_roll = add_pressure_divergence_features(df_roll, add_debug_parts=True)

    # Milestone 3: xG-proxy Sniper (venue-aware rolling xG form + Pressure-vs-xG disagreement)
    df_roll = add_xg_form_features(df_roll, window=10)
    df_roll = add_pressure_xg_disagreement_features(df_roll, add_debug_parts=True)

    # Milestone 10: Head-to-Head History (direct matchup patterns)
    h2h_enabled = getattr(config, "H2H_ENABLED", True)
    if h2h_enabled:
        df_roll = add_h2h_features(df_roll)

    # Milestone 11: League-Specific Features (scoring patterns per competition)
    league_features_enabled = getattr(config, "LEAGUE_FEATURES_ENABLED", True)
    if league_features_enabled:
        df_roll = add_league_features(df_roll)

    # Drop internal post-match helper columns (not features; contain current-match info)
    df_roll = df_roll.drop(
        columns=[c for c in df_roll.columns if c.startswith("_press_") or c.startswith("_xg_")],
        errors="ignore",
    )

    # Elo similarity features (kernel-based, venue-aware) with prefix histories to avoid leakage
    sigma_map = getattr(config, "ELO_SIM_SIGMA_PER_LEAGUE", {})
    default_sigma = 60.0  # slightly wider for stability
    # Maintain prefix histories per team/venue
    hist_lists: Dict[str, Dict[str, list]] = {}
    df_roll = df_roll.sort_values("datetime").reset_index(drop=True)

    gf_sim_h = []
    ga_sim_h = []
    gf_sim_a = []
    ga_sim_a = []
    wsum_h = []
    wsum_a = []
    neff_h = []
    neff_a = []

    for _, row in df_roll.iterrows():
        league = row.get("league")
        sigma = float(sigma_map.get(str(league), default_sigma))
        as_of = row.get("datetime")

        # Build DataFrames from prefix history lists
        h_hist_list = hist_lists.get(row["home"], {}).get("home", [])
        a_hist_list = hist_lists.get(row["away"], {}).get("away", [])
        h_hist_df = pd.DataFrame(h_hist_list)
        a_hist_df = pd.DataFrame(a_hist_list)

        gf_h, ga_h, w_h, n_h = kernel_similarity(h_hist_df, row.get("elo_away"), sigma, as_of=as_of)
        gf_a, ga_a, w_a, n_a = kernel_similarity(a_hist_df, row.get("elo_home"), sigma, as_of=as_of)

        gf_sim_h.append(gf_h)
        ga_sim_h.append(ga_h)
        gf_sim_a.append(gf_a)
        ga_sim_a.append(ga_a)
        wsum_h.append(w_h)
        wsum_a.append(w_a)
        neff_h.append(n_h)
        neff_a.append(n_a)

        # Append current match to histories for future rows
        hist_lists.setdefault(row["home"], {}).setdefault("home", []).append(
            {
                "datetime": row.get("datetime"),
                "opp_elo": row.get("elo_away"),
                "gf": row.get("ft_home"),
                "ga": row.get("ft_away"),
                "league": league,
            }
        )
        hist_lists.setdefault(row["away"], {}).setdefault("away", []).append(
            {
                "datetime": row.get("datetime"),
                "opp_elo": row.get("elo_home"),
                "gf": row.get("ft_away"),
                "ga": row.get("ft_home"),
                "league": league,
            }
        )

    df_roll["GFvsSim_H"] = gf_sim_h
    df_roll["GAvsSim_H"] = ga_sim_h
    df_roll["GFvsSim_A"] = gf_sim_a
    df_roll["GAvsSim_A"] = ga_sim_a
    df_roll["wsum_sim_H"] = wsum_h
    df_roll["wsum_sim_A"] = wsum_a
    df_roll["neff_sim_H"] = neff_h
    df_roll["neff_sim_A"] = neff_a

    # Drop calc/alias columns not meant for model features
    df_roll = df_roll.drop(
        columns=[
            "elo_home_calc",
            "elo_away_calc",
            "EloDiff_calc",
            "Band_H_calc",
            "Band_A_calc",
            "home_name_calc",
            "away_name_calc",
        ],
        errors="ignore",
    )

    # Targets
    df_roll["y_home"] = df_roll["ft_home"]
    df_roll["y_away"] = df_roll["ft_away"]

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Write a debug/full artifact (contains raw/truth columns; NOT safe for model training outside this repo).
    if out_full_path:
        out_full_p = Path(out_full_path)
        out_full_p.parent.mkdir(parents=True, exist_ok=True)
        df_roll.to_csv(out_full_p, index=False)

    # Write a SAFE artifact: numeric features only + targets (no raw post-match truth or raw current-match stats).
    banned_exact = {
        "ft_home", "ft_away", "ht_home", "ht_away", "result", "validated",
        "shots", "shots_on_target", "corners", "possession_home", "possession_away",
        "shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A",
        # Milestone 3: per-match xG proxy columns (current-match derived; unsafe as features)
        "xg_proxy_H", "xg_proxy_A", "xg_usable",
        "shot_quality_H", "shot_quality_A", "finishing_luck_H", "finishing_luck_A",
        # Per-match stat availability flags (post-match / not knowable pre-match).
        "pressure_usable",
        "_stats_src",
    }
    keep = [c for c in df_roll.select_dtypes(include=[np.number]).columns if c not in banned_exact]
    # Ensure targets exist
    for c in ["y_home", "y_away"]:
        if c not in keep and c in df_roll.columns:
            keep.append(c)
    safe = df_roll[keep].copy()
    safe.to_csv(out_p, index=False)

    msg = f"[ok] wrote frankenstein training -> {out_p} (rows={len(safe)}, cols={len(safe.columns)})"
    if out_full_path:
        msg += f" | debug full -> {out_full_path} (cols={len(df_roll.columns)})"
    print(msg)
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Frankenstein training matrix from CGM match history")
    ap.add_argument("--data-dir", default="data/enhanced", help="Directory containing enhanced data")
    ap.add_argument("--match-history", default="cgm_match_history_with_elo_stats.csv", help="Match history filename")
    ap.add_argument("--out", default="data/enhanced/frankenstein_training.csv", help="Output CSV path")
    ap.add_argument("--out-full", default="data/enhanced/frankenstein_training_full.csv",
                    help="Optional debug CSV (full, includes raw/truth cols). Set to empty to disable.")
    ap.add_argument("--windows", nargs="+", type=int, default=[5, 10], help="Rolling windows (games) for stats")
    ap.add_argument("--alpha", type=float, default=0.5, help="Smoothing numerator")
    ap.add_argument("--beta", type=float, default=1.0, help="Smoothing denominator")
    args = ap.parse_args()
    out_full = args.out_full if str(args.out_full).strip() else None
    build_frankenstein(args.data_dir, args.match_history, args.out, out_full, args.windows, args.alpha, args.beta)


if __name__ == "__main__":
    main()
