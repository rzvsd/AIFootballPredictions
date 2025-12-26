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
        # Fix: When shots=0, shot quality is undefined (NaN), not xG/1
        out["_xg_shot_quality_H_raw"] = np.where(shots_h > 0, out["xg_proxy_H"] / shots_h, np.nan)
        out["_xg_shot_quality_A_raw"] = np.where(shots_a > 0, out["xg_proxy_A"] / shots_a, np.nan)

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

    # Milestone 9: Time decay weighted rolling (recent matches count more)
    try:
        import config
        decay_half_life = getattr(config, "DECAY_HALF_LIFE", 5)
        decay_enabled = getattr(config, "DECAY_ENABLED", True)
    except ImportError:
        decay_half_life = 5
        decay_enabled = True

    def _decay_weights(n: int, half_life: float) -> np.ndarray:
        """Generate exponential decay weights: most recent = 1.0, older = less."""
        ages = np.arange(n - 1, -1, -1)  # [n-1, n-2, ..., 1, 0] oldest to newest
        return np.exp(-0.693 * ages / half_life)

    def _roll_pre_decay(g: pd.DataFrame, value_col: str) -> pd.Series:
        """Exponentially weighted rolling mean (pre-match, excludes current)."""
        vals = g[value_col].shift().values
        result = np.full(len(vals), np.nan)
        for i in range(1, len(vals)):
            start_idx = max(0, i - window)
            window_vals = vals[start_idx:i]
            valid_mask = ~np.isnan(window_vals)
            if valid_mask.sum() == 0:
                continue
            w = _decay_weights(len(window_vals), decay_half_life)[valid_mask]
            v = window_vals[valid_mask]
            result[i] = np.sum(w * v) / np.sum(w)
        return pd.Series(result, index=g.index)

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

    # Milestone 9: Time decay xG features (home context)
    if decay_enabled:
        out["xg_for_form_H_decay"] = out.groupby(home_col, group_keys=False).apply(
            lambda g: _roll_pre_decay(g.sort_values(datetime_col), "_xg_for_H_raw")
        ).fillna(0.0)
        out["xg_against_form_H_decay"] = out.groupby(home_col, group_keys=False).apply(
            lambda g: _roll_pre_decay(g.sort_values(datetime_col), "_xg_against_H_raw")
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

    # Milestone 9: Time decay xG features (away context)
    if decay_enabled:
        out["xg_for_form_A_decay"] = out.groupby(away_col, group_keys=False).apply(
            lambda g: _roll_pre_decay(g.sort_values(datetime_col), "_xg_for_A_raw")
        ).fillna(0.0)
        out["xg_against_form_A_decay"] = out.groupby(away_col, group_keys=False).apply(
            lambda g: _roll_pre_decay(g.sort_values(datetime_col), "_xg_against_A_raw")
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

