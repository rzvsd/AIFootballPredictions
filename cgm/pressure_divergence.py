"""
Milestone 2: Pressure Cooker - divergence vs Elo (continuous).

Uses standardized scores (z) within league:
  div_team_H = z(press_form_H) - z(elo_team_H)
  div_team_A = z(press_form_A) - z(elo_team_A)
  div_diff   = div_team_H - div_team_A

Also provides optional debug parts:
  press_z_H/A, elo_z_H/A

Note: `press_form_*` must be leakage-safe (pre-match); internal `_press_form_*_post`
is used only to advance league state between matches and must not be used as a model feature.
"""

from __future__ import annotations

import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from types import ModuleType


def _z(val: float, mean: float, std: float) -> float:
    try:
        if std <= 0 or np.isnan(std) or np.isnan(mean) or np.isnan(val):
            return 0.0
        return float((val - mean) / std)
    except Exception:
        return 0.0


def _elo_cols(df: pd.DataFrame) -> tuple[str, str]:
    if "elo_home_calc" in df.columns and "elo_away_calc" in df.columns:
        return ("elo_home_calc", "elo_away_calc")
    return ("elo_home", "elo_away")


def _load_calc_cgm_elo_module() -> ModuleType | None:
    try:
        root = Path(__file__).resolve().parents[1]
        mod_path = root / "scripts" / "calc_cgm_elo.py"
        if not mod_path.exists():
            return None
        spec = importlib.util.spec_from_file_location("_calc_cgm_elo", str(mod_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    except Exception:
        return None


def add_pressure_divergence_features(
    df: pd.DataFrame,
    *,
    league_col: str = "league",
    home_col: str = "home",
    away_col: str = "away",
    datetime_col: str = "datetime",
    add_debug_parts: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    for col in ["press_form_H", "press_form_A", "_press_form_H_post", "_press_form_A_post"]:
        if col not in out.columns:
            raise ValueError(f"Missing required Pressure column: {col}")
    if league_col not in out.columns or datetime_col not in out.columns:
        raise ValueError("Missing required league/datetime columns")

    elo_h_col, elo_a_col = _elo_cols(out)
    if elo_h_col not in out.columns or elo_a_col not in out.columns:
        raise ValueError("Missing Elo columns required for divergence")

    out["press_form_H"] = pd.to_numeric(out["press_form_H"], errors="coerce").fillna(0.5)
    out["press_form_A"] = pd.to_numeric(out["press_form_A"], errors="coerce").fillna(0.5)
    out["_press_form_H_post"] = pd.to_numeric(out["_press_form_H_post"], errors="coerce").fillna(0.5)
    out["_press_form_A_post"] = pd.to_numeric(out["_press_form_A_post"], errors="coerce").fillna(0.5)
    out[elo_h_col] = pd.to_numeric(out[elo_h_col], errors="coerce")
    out[elo_a_col] = pd.to_numeric(out[elo_a_col], errors="coerce")

    group_cols = [league_col]
    if "country" in out.columns and league_col != "country":
        group_cols = ["country", league_col]
    out = out.sort_values(group_cols + [datetime_col])

    # Keep Elo-state updates consistent with the Elo rebuild used to generate the history.
    home_adv_global = 65.0
    try:
        if "EloDiff" in out.columns:
            adv_series = out["EloDiff"] - (out[elo_h_col] - out[elo_a_col])
            adv_series = (
                pd.to_numeric(adv_series, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not adv_series.empty:
                home_adv_global = float(adv_series.median())
    except Exception:
        home_adv_global = 65.0

    div_team_h: list[float] = []
    div_team_a: list[float] = []
    div_diff: list[float] = []
    press_z_h: list[float] = []
    press_z_a: list[float] = []
    elo_z_h: list[float] = []
    elo_z_a: list[float] = []

    for _, g in out.groupby(group_cols, sort=False):
        elo_state: dict[str, float] = {}
        press_home_state: dict[str, float] = {}
        press_away_state: dict[str, float] = {}

        # Prefer a per-league (or country+league) home-advantage estimate, with a global fallback.
        home_adv_used = home_adv_global
        try:
            if "EloDiff" in g.columns:
                adv_series = g["EloDiff"] - (g[elo_h_col] - g[elo_a_col])
                adv_series = (
                    pd.to_numeric(adv_series, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if not adv_series.empty:
                    home_adv_used = float(adv_series.median())
        except Exception:
            home_adv_used = home_adv_global

        calc_mod = _load_calc_cgm_elo_module()
        k_default = float(getattr(calc_mod, "K_FACTOR_DEFAULT", 20.0)) if calc_mod else 20.0
        expected_home = getattr(calc_mod, "expected_home", None) if calc_mod else None
        margin_multiplier = getattr(calc_mod, "margin_multiplier", None) if calc_mod else None

        g_sorted = g.sort_values(datetime_col)
        for _, row in g_sorted.iterrows():
            home = str(row.get(home_col))
            away = str(row.get(away_col))

            elo_vals = np.array(list(elo_state.values()), dtype=float) if elo_state else np.array([], dtype=float)
            ph_vals = np.array(list(press_home_state.values()), dtype=float) if press_home_state else np.array([], dtype=float)
            pa_vals = np.array(list(press_away_state.values()), dtype=float) if press_away_state else np.array([], dtype=float)

            elo_mean = float(np.nanmean(elo_vals)) if elo_vals.size else float(np.nan)
            elo_std = float(np.nanstd(elo_vals, ddof=0)) if elo_vals.size else float(np.nan)
            ph_mean = float(np.nanmean(ph_vals)) if ph_vals.size else float(np.nan)
            ph_std = float(np.nanstd(ph_vals, ddof=0)) if ph_vals.size else float(np.nan)
            pa_mean = float(np.nanmean(pa_vals)) if pa_vals.size else float(np.nan)
            pa_std = float(np.nanstd(pa_vals, ddof=0)) if pa_vals.size else float(np.nan)

            press_h = float(row["press_form_H"])
            press_a = float(row["press_form_A"])
            elo_h = float(row.get(elo_h_col)) if pd.notna(row.get(elo_h_col)) else float("nan")
            elo_a = float(row.get(elo_a_col)) if pd.notna(row.get(elo_a_col)) else float("nan")

            z_press_h = _z(press_h, ph_mean, ph_std)
            z_press_a = _z(press_a, pa_mean, pa_std)
            z_elo_h = _z(elo_h, elo_mean, elo_std)
            z_elo_a = _z(elo_a, elo_mean, elo_std)

            dth = z_press_h - z_elo_h
            dta = z_press_a - z_elo_a

            div_team_h.append(dth)
            div_team_a.append(dta)
            div_diff.append(dth - dta)
            press_z_h.append(z_press_h)
            press_z_a.append(z_press_a)
            elo_z_h.append(z_elo_h)
            elo_z_a.append(z_elo_a)

            # Update Elo state AFTER this match (Milestone 1 parameters, using pre-match Elo from row).
            fh = row.get("ft_home")
            fa = row.get("ft_away")
            try:
                fh_f = float(fh)
                fa_f = float(fa)
            except Exception:
                fh_f = np.nan
                fa_f = np.nan

            if not np.isnan(elo_h) and not np.isnan(elo_a) and not np.isnan(fh_f) and not np.isnan(fa_f):
                home_adv = home_adv_used
                if callable(expected_home):
                    exp_home = float(expected_home(elo_h, elo_a, home_adv))
                else:
                    diff = (elo_h + home_adv) - elo_a
                    exp_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
                if fh_f > fa_f:
                    actual = 1.0
                elif fh_f == fa_f:
                    actual = 0.5
                else:
                    actual = 0.0
                gd = int(abs(fh_f - fa_f))
                if callable(margin_multiplier):
                    mult = float(margin_multiplier(gd))
                else:
                    if gd <= 1:
                        mult = 1.0
                    elif gd == 2:
                        mult = 1.5
                    elif gd == 3:
                        mult = 1.75
                    else:
                        mult = 1.75 + (gd - 3) / 8.0
                delta = k_default * mult * (actual - exp_home)
                elo_state[home] = elo_h + delta
                elo_state[away] = elo_a - delta
            else:
                if not np.isnan(elo_h):
                    elo_state[home] = elo_h
                if not np.isnan(elo_a):
                    elo_state[away] = elo_a

            # Update Pressure states AFTER the match using post-match rolling values.
            press_home_state[home] = float(row.get("_press_form_H_post", 0.5))
            press_away_state[away] = float(row.get("_press_form_A_post", 0.5))

    out["div_team_H"] = div_team_h
    out["div_team_A"] = div_team_a
    out["div_diff"] = div_diff
    if add_debug_parts:
        out["press_z_H"] = press_z_h
        out["press_z_A"] = press_z_a
        out["elo_z_H"] = elo_z_h
        out["elo_z_A"] = elo_z_a

    return out
