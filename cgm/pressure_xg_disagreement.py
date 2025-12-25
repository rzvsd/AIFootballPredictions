"""
Milestone 3: Pressure vs xG disagreement ("money zone").

Computes league-standardized z-scores at as-of time and produces:
  div_px_team_H = z(press_form_H) - z(xg_diff_form_H)
  div_px_team_A = z(press_form_A) - z(xg_diff_form_A)
  div_px_diff   = div_px_team_H - div_px_team_A

Also emits simple strategy flags:
  sterile_H / sterile_A  : pressure high, xg low
  assassin_H / assassin_A: pressure low, xg high

This module is leakage-safe: z-score states are advanced using *post-match*
rolling helper columns (`_press_form_*_post` and `_xg_diff_form_*_post`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _z(val: float, mean: float, std: float) -> float:
    try:
        if std <= 0 or np.isnan(std) or np.isnan(mean) or np.isnan(val):
            return 0.0
        return float((val - mean) / std)
    except Exception:
        return 0.0


def add_pressure_xg_disagreement_features(
    df: pd.DataFrame,
    *,
    league_col: str = "league",
    home_col: str = "home",
    away_col: str = "away",
    datetime_col: str = "datetime",
    z_thresh: float = 1.0,
    add_debug_parts: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    required = [
        "press_form_H",
        "press_form_A",
        "_press_form_H_post",
        "_press_form_A_post",
        "xg_diff_form_H",
        "xg_diff_form_A",
        "_xg_diff_form_H_post",
        "_xg_diff_form_A_post",
    ]
    for c in required:
        if c not in out.columns:
            raise ValueError(f"Missing required column for Pressure-vs-xG: {c}")

    group_cols = [league_col]
    if "country" in out.columns and league_col != "country":
        group_cols = ["country", league_col]
    out = out.sort_values(group_cols + [datetime_col])

    # Coerce numeric essentials.
    for c in [
        "press_form_H",
        "press_form_A",
        "_press_form_H_post",
        "_press_form_A_post",
        "xg_diff_form_H",
        "xg_diff_form_A",
        "_xg_diff_form_H_post",
        "_xg_diff_form_A_post",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Optional: reuse press_z_* if already computed by pressure_divergence, otherwise compute here.
    reuse_press_z = ("press_z_H" in out.columns) and ("press_z_A" in out.columns)
    if reuse_press_z:
        out["press_z_H"] = pd.to_numeric(out["press_z_H"], errors="coerce").fillna(0.0)
        out["press_z_A"] = pd.to_numeric(out["press_z_A"], errors="coerce").fillna(0.0)

    div_team_h: list[float] = []
    div_team_a: list[float] = []
    div_diff: list[float] = []
    xg_z_h: list[float] = []
    xg_z_a: list[float] = []
    sterile_h: list[int] = []
    sterile_a: list[int] = []
    assassin_h: list[int] = []
    assassin_a: list[int] = []
    press_z_h_list: list[float] = []
    press_z_a_list: list[float] = []

    for _, g in out.groupby(group_cols, sort=False):
        press_home_state: dict[str, float] = {}
        press_away_state: dict[str, float] = {}
        xg_home_state: dict[str, float] = {}
        xg_away_state: dict[str, float] = {}

        g_sorted = g.sort_values(datetime_col)
        for _, row in g_sorted.iterrows():
            home = str(row.get(home_col))
            away = str(row.get(away_col))

            # PRESS z-score (reuse if present, else compute from current league state).
            if reuse_press_z:
                pz_h = float(row.get("press_z_H", 0.0))
                pz_a = float(row.get("press_z_A", 0.0))
            else:
                ph_vals = np.array(list(press_home_state.values()), dtype=float) if press_home_state else np.array([], dtype=float)
                pa_vals = np.array(list(press_away_state.values()), dtype=float) if press_away_state else np.array([], dtype=float)
                ph_mean = float(np.nanmean(ph_vals)) if ph_vals.size else float("nan")
                ph_std = float(np.nanstd(ph_vals, ddof=0)) if ph_vals.size else float("nan")
                pa_mean = float(np.nanmean(pa_vals)) if pa_vals.size else float("nan")
                pa_std = float(np.nanstd(pa_vals, ddof=0)) if pa_vals.size else float("nan")
                pz_h = _z(float(row.get("press_form_H", np.nan)), ph_mean, ph_std)
                pz_a = _z(float(row.get("press_form_A", np.nan)), pa_mean, pa_std)

            # xG z-score (xg_diff form) from current league state.
            xh_vals = np.array(list(xg_home_state.values()), dtype=float) if xg_home_state else np.array([], dtype=float)
            xa_vals = np.array(list(xg_away_state.values()), dtype=float) if xg_away_state else np.array([], dtype=float)
            xh_mean = float(np.nanmean(xh_vals)) if xh_vals.size else float("nan")
            xh_std = float(np.nanstd(xh_vals, ddof=0)) if xh_vals.size else float("nan")
            xa_mean = float(np.nanmean(xa_vals)) if xa_vals.size else float("nan")
            xa_std = float(np.nanstd(xa_vals, ddof=0)) if xa_vals.size else float("nan")
            xz_h = _z(float(row.get("xg_diff_form_H", np.nan)), xh_mean, xh_std)
            xz_a = _z(float(row.get("xg_diff_form_A", np.nan)), xa_mean, xa_std)

            dth = pz_h - xz_h
            dta = pz_a - xz_a

            div_team_h.append(dth)
            div_team_a.append(dta)
            div_diff.append(dth - dta)
            xg_z_h.append(xz_h)
            xg_z_a.append(xz_a)
            press_z_h_list.append(pz_h)
            press_z_a_list.append(pz_a)

            sterile_h.append(int((pz_h >= z_thresh) and (xz_h <= -z_thresh)))
            sterile_a.append(int((pz_a >= z_thresh) and (xz_a <= -z_thresh)))
            assassin_h.append(int((pz_h <= -z_thresh) and (xz_h >= z_thresh)))
            assassin_a.append(int((pz_a <= -z_thresh) and (xz_a >= z_thresh)))

            # Advance states AFTER this match.
            try:
                press_home_state[home] = float(row.get("_press_form_H_post", 0.5))
            except Exception:
                press_home_state[home] = 0.5
            try:
                press_away_state[away] = float(row.get("_press_form_A_post", 0.5))
            except Exception:
                press_away_state[away] = 0.5

            try:
                x_post_h = float(row.get("_xg_diff_form_H_post"))
                if not np.isnan(x_post_h):
                    xg_home_state[home] = x_post_h
            except Exception:
                pass
            try:
                x_post_a = float(row.get("_xg_diff_form_A_post"))
                if not np.isnan(x_post_a):
                    xg_away_state[away] = x_post_a
            except Exception:
                pass

    out["div_px_team_H"] = div_team_h
    out["div_px_team_A"] = div_team_a
    out["div_px_diff"] = div_diff
    out["sterile_H"] = sterile_h
    out["sterile_A"] = sterile_a
    out["assassin_H"] = assassin_h
    out["assassin_A"] = assassin_a
    if add_debug_parts:
        out["xg_z_H"] = xg_z_h
        out["xg_z_A"] = xg_z_a
        if not reuse_press_z:
            out["press_z_H"] = press_z_h_list
            out["press_z_A"] = press_z_a_list

    return out

