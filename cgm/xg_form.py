"""
Milestone 3: xG-proxy form (venue-aware, leakage-safe).

Feature contract (v1)
- xg_for_form_H, xg_against_form_H, xg_diff_form_H
- xg_for_form_A, xg_against_form_A, xg_diff_form_A
- xg_shot_quality_form_H/A, xg_finishing_luck_form_H/A
- xg_n_H/A, xg_stats_n_H/A

Internal (helper) columns for inference snapshots / divergence state updates:
- _xg_*_post (post-match rolling states including the current match)

Notes
- `xg_proxy_*` is a per-match value derived from current-match stats, and must
  never be used directly as a feature for that same match.
- This module produces pre-match features via `.shift()` and provides separate
  post-match rolling states (suffix `_post`) for inference snapshots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_xg_form_features(
    df: pd.DataFrame,
    *,
    window: int = 10,
    datetime_col: str = "datetime",
    home_col: str = "home",
    away_col: str = "away",
) -> pd.DataFrame:
    out = df.copy()
    if datetime_col not in out.columns:
        raise ValueError(f"Missing datetime column: {datetime_col}")

    for c in ["xg_proxy_H", "xg_proxy_A"]:
        if c not in out.columns:
            raise ValueError(f"Missing required xG column: {c} (run build_xg_proxy first)")

    out["xg_proxy_H"] = pd.to_numeric(out["xg_proxy_H"], errors="coerce")
    out["xg_proxy_A"] = pd.to_numeric(out["xg_proxy_A"], errors="coerce")

    if "xg_usable" in out.columns:
        usable = pd.to_numeric(out["xg_usable"], errors="coerce").fillna(0.0) > 0
    else:
        usable = out["xg_proxy_H"].notna() & out["xg_proxy_A"].notna()

    out["_xg_stats_present_raw"] = usable.astype(float)

    out["_xg_for_H_raw"] = out["xg_proxy_H"]
    out["_xg_against_H_raw"] = out["xg_proxy_A"]
    out["_xg_diff_H_raw"] = out["_xg_for_H_raw"] - out["_xg_against_H_raw"]

    out["_xg_for_A_raw"] = out["xg_proxy_A"]
    out["_xg_against_A_raw"] = out["xg_proxy_H"]
    out["_xg_diff_A_raw"] = out["_xg_for_A_raw"] - out["_xg_against_A_raw"]

    # Per-match helpers (still post-match truth; used only for rolling pre/post).
    if "shot_quality_H" in out.columns and "shot_quality_A" in out.columns:
        out["_xg_shot_quality_H_raw"] = pd.to_numeric(out["shot_quality_H"], errors="coerce")
        out["_xg_shot_quality_A_raw"] = pd.to_numeric(out["shot_quality_A"], errors="coerce")
    else:
        shots_h = pd.to_numeric(out.get("shots_H"), errors="coerce")
        shots_a = pd.to_numeric(out.get("shots_A"), errors="coerce")
        out["_xg_shot_quality_H_raw"] = out["xg_proxy_H"] / shots_h.where(shots_h > 0, 1.0)
        out["_xg_shot_quality_A_raw"] = out["xg_proxy_A"] / shots_a.where(shots_a > 0, 1.0)

    if "finishing_luck_H" in out.columns and "finishing_luck_A" in out.columns:
        out["_xg_finishing_luck_H_raw"] = pd.to_numeric(out["finishing_luck_H"], errors="coerce")
        out["_xg_finishing_luck_A_raw"] = pd.to_numeric(out["finishing_luck_A"], errors="coerce")
    else:
        ft_h = pd.to_numeric(out.get("ft_home"), errors="coerce")
        ft_a = pd.to_numeric(out.get("ft_away"), errors="coerce")
        out["_xg_finishing_luck_H_raw"] = ft_h - out["xg_proxy_H"]
        out["_xg_finishing_luck_A_raw"] = ft_a - out["xg_proxy_A"]

    out = out.sort_values(datetime_col)

    def _roll_pre(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].shift().rolling(window, min_periods=1).mean()

    def _roll_post(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].rolling(window, min_periods=1).mean()

    def _roll_pre_count(g: pd.DataFrame, base_col: str) -> pd.Series:
        return g[base_col].shift().rolling(window, min_periods=1).count()

    def _roll_post_count(g: pd.DataFrame, base_col: str) -> pd.Series:
        return g[base_col].rolling(window, min_periods=1).count()

    def _roll_pre_sum(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].shift().rolling(window, min_periods=1).sum()

    def _roll_post_sum(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].rolling(window, min_periods=1).sum()

    # Home-context rolling
    out["xg_for_form_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_for_H_raw")
    ).fillna(0.0)
    out["xg_against_form_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_against_H_raw")
    ).fillna(0.0)
    out["xg_diff_form_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_diff_H_raw")
    ).fillna(0.0)
    out["xg_shot_quality_form_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_shot_quality_H_raw")
    ).fillna(0.0)
    out["xg_finishing_luck_form_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_finishing_luck_H_raw")
    ).fillna(0.0)
    out["xg_n_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre_count(g.sort_values(datetime_col), datetime_col)
    ).fillna(0.0)
    out["xg_stats_n_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre_sum(g.sort_values(datetime_col), "_xg_stats_present_raw")
    ).fillna(0.0)

    # Post-match home states
    out["_xg_for_form_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_for_H_raw")
    ).fillna(0.0)
    out["_xg_against_form_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_against_H_raw")
    ).fillna(0.0)
    out["_xg_diff_form_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_diff_H_raw")
    ).fillna(0.0)
    out["_xg_shot_quality_form_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_shot_quality_H_raw")
    ).fillna(0.0)
    out["_xg_finishing_luck_form_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_finishing_luck_H_raw")
    ).fillna(0.0)
    out["_xg_n_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post_count(g.sort_values(datetime_col), datetime_col)
    ).fillna(0.0)
    out["_xg_stats_n_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post_sum(g.sort_values(datetime_col), "_xg_stats_present_raw")
    ).fillna(0.0)

    # Away-context rolling
    out["xg_for_form_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_for_A_raw")
    ).fillna(0.0)
    out["xg_against_form_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_against_A_raw")
    ).fillna(0.0)
    out["xg_diff_form_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_diff_A_raw")
    ).fillna(0.0)
    out["xg_shot_quality_form_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_shot_quality_A_raw")
    ).fillna(0.0)
    out["xg_finishing_luck_form_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_xg_finishing_luck_A_raw")
    ).fillna(0.0)
    out["xg_n_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre_count(g.sort_values(datetime_col), datetime_col)
    ).fillna(0.0)
    out["xg_stats_n_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre_sum(g.sort_values(datetime_col), "_xg_stats_present_raw")
    ).fillna(0.0)

    # Post-match away states
    out["_xg_for_form_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_for_A_raw")
    ).fillna(0.0)
    out["_xg_against_form_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_against_A_raw")
    ).fillna(0.0)
    out["_xg_diff_form_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_diff_A_raw")
    ).fillna(0.0)
    out["_xg_shot_quality_form_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_shot_quality_A_raw")
    ).fillna(0.0)
    out["_xg_finishing_luck_form_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_xg_finishing_luck_A_raw")
    ).fillna(0.0)
    out["_xg_n_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post_count(g.sort_values(datetime_col), datetime_col)
    ).fillna(0.0)
    out["_xg_stats_n_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post_sum(g.sort_values(datetime_col), "_xg_stats_present_raw")
    ).fillna(0.0)

    # Drop per-match-only helper columns.
    out = out.drop(
        columns=[
            "_xg_stats_present_raw",
            "_xg_for_H_raw",
            "_xg_against_H_raw",
            "_xg_diff_H_raw",
            "_xg_for_A_raw",
            "_xg_against_A_raw",
            "_xg_diff_A_raw",
            "_xg_shot_quality_H_raw",
            "_xg_shot_quality_A_raw",
            "_xg_finishing_luck_H_raw",
            "_xg_finishing_luck_A_raw",
        ],
        errors="ignore",
    )

    return out

