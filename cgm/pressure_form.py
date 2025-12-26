"""
Milestone 2: Pressure Cooker - dominance + rolling PressureForm (venue-aware).

Feature contract (v1):
  press_form_H, press_form_A
  press_dom_shots_H/A, press_dom_sot_H/A, press_dom_corners_H/A, press_dom_pos_H/A
  press_n_H, press_n_A
  press_stats_n_H, press_stats_n_A

Internal (helper) columns (used by inference/divergence; must be dropped from training features):
  _press_form_H_post, _press_form_A_post
  _press_n_H_post, _press_n_A_post
  _press_dom_*_{H|A}_post
  _press_stats_n_H_post, _press_stats_n_A_post
"""

from __future__ import annotations

import numpy as np
import pandas as pd


W_SHOTS = 0.45
W_SOT = 0.30
W_CORNERS = 0.15
W_POS = 0.10


def _neutral_dom() -> float:
    return 0.5


import logging
_logger = logging.getLogger(__name__)


def _dom_ratio(for_v: float | None, against_v: float | None) -> float:
    """Calculate dominance ratio (for / (for + against)) with safety checks."""
    try:
        if for_v is None or against_v is None:
            return _neutral_dom()
        den = float(for_v) + float(against_v)
        if den <= 0 or np.isnan(den):
            return _neutral_dom()
        val = float(for_v) / den
        if np.isnan(val):
            return _neutral_dom()
        return float(np.clip(val, 0.0, 1.0))
    except Exception as e:
        # Log the exception instead of silently swallowing it
        _logger.warning(f"_dom_ratio fallback to neutral: for={for_v}, against={against_v}, error={e}")
        return _neutral_dom()


def _poss_share(pos_v: float | None) -> float:
    """Calculate possession share with safety checks."""
    try:
        if pos_v is None:
            return _neutral_dom()
        val = float(pos_v)
        if np.isnan(val):
            return _neutral_dom()
        if 0.0 <= val <= 1.0:
            return float(np.clip(val, 0.0, 1.0))
        if 0.0 <= val <= 100.0:
            return float(np.clip(val / 100.0, 0.0, 1.0))
        return _neutral_dom()
    except Exception as e:
        # Log the exception instead of silently swallowing it
        _logger.warning(f"_poss_share fallback to neutral: pos={pos_v}, error={e}")
        return _neutral_dom()


def add_pressure_form_features(
    df: pd.DataFrame,
    *,
    window: int = 10,
    datetime_col: str = "datetime",
    home_col: str = "home",
    away_col: str = "away",
) -> pd.DataFrame:
    """
    Adds Pressure Cooker rolling features to a match-level dataframe.

    Assumes the dataframe already has canonical split inputs:
      shots_H/shots_A, sot_H/sot_A, corners_H/corners_A, pos_H/pos_A
    """
    out = df.copy()
    if datetime_col not in out.columns:
        raise ValueError(f"Missing datetime column: {datetime_col}")

    shots_h = out.get("shots_H")
    shots_a = out.get("shots_A")
    sot_h = out.get("sot_H")
    sot_a = out.get("sot_A")
    cor_h = out.get("corners_H")
    cor_a = out.get("corners_A")
    pos_h = out.get("pos_H")
    pos_a = out.get("pos_A")

    def _dom_series(for_s: pd.Series | None, against_s: pd.Series | None) -> pd.Series:
        if for_s is None or against_s is None:
            return pd.Series([_neutral_dom()] * len(out), index=out.index, dtype=float)
        f = pd.to_numeric(for_s, errors="coerce").replace({np.nan: None})
        a = pd.to_numeric(against_s, errors="coerce").replace({np.nan: None})
        return pd.Series([_dom_ratio(fv, av) for fv, av in zip(f.tolist(), a.tolist())], index=out.index, dtype=float)

    # Keep per-match dominance as raw internal columns to avoid "rolling a rolling".
    out["_press_dom_shots_H_raw"] = _dom_series(shots_h, shots_a)
    out["_press_dom_shots_A_raw"] = _dom_series(shots_a, shots_h)
    out["_press_dom_sot_H_raw"] = _dom_series(sot_h, sot_a)
    out["_press_dom_sot_A_raw"] = _dom_series(sot_a, sot_h)
    out["_press_dom_corners_H_raw"] = _dom_series(cor_h, cor_a)
    out["_press_dom_corners_A_raw"] = _dom_series(cor_a, cor_h)

    def _poss_series(s: pd.Series | None) -> pd.Series:
        if s is None:
            return pd.Series([_neutral_dom()] * len(out), index=out.index, dtype=float)
        v = pd.to_numeric(s, errors="coerce").replace({np.nan: None})
        return pd.Series([_poss_share(x) for x in v.tolist()], index=out.index, dtype=float)

    out["_press_dom_pos_H_raw"] = _poss_series(pos_h)
    out["_press_dom_pos_A_raw"] = _poss_series(pos_a)
    if "pos_A" not in out.columns or out["pos_A"].isna().all():
        out["_press_dom_pos_A_raw"] = 1.0 - out["_press_dom_pos_H_raw"]

    out["_press_index_H"] = (
        W_SHOTS * out["_press_dom_shots_H_raw"]
        + W_SOT * out["_press_dom_sot_H_raw"]
        + W_CORNERS * out["_press_dom_corners_H_raw"]
        + W_POS * out["_press_dom_pos_H_raw"]
    )
    out["_press_index_A"] = (
        W_SHOTS * out["_press_dom_shots_A_raw"]
        + W_SOT * out["_press_dom_sot_A_raw"]
        + W_CORNERS * out["_press_dom_corners_A_raw"]
        + W_POS * out["_press_dom_pos_A_raw"]
    )

    out = out.sort_values(datetime_col)

    def _roll_pre(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].shift().rolling(window, min_periods=1).mean()

    def _roll_post(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].rolling(window, min_periods=1).mean()

    def _roll_pre_count(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].shift().rolling(window, min_periods=1).count().fillna(0).clip(0, window)

    def _roll_post_count(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].rolling(window, min_periods=1).count().fillna(0).clip(0, window)

    def _roll_pre_sum(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].shift().rolling(window, min_periods=1).sum().fillna(0).clip(0, window)

    def _roll_post_sum(g: pd.DataFrame, value_col: str) -> pd.Series:
        return g[value_col].rolling(window, min_periods=1).sum().fillna(0).clip(0, window)

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

    # Evidence signal: per-match availability of complete inputs.
    # Used ONLY to derive rolling counts; it is not leakage-safe by itself.
    if "pressure_usable" in out.columns:
        stats_present = pd.to_numeric(out["pressure_usable"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        stat_cols = ["shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A"]
        if all(c in out.columns for c in stat_cols):
            stats_present = out[stat_cols].notna().all(axis=1).astype(float)
        else:
            stats_present = pd.Series([0.0] * len(out), index=out.index, dtype=float)
    out["_press_stats_present_raw"] = stats_present

    # Home-context rolling (venue-aware)
    out["press_form_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_index_H")
    ).fillna(_neutral_dom())
    out["press_n_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre_count(g.sort_values(datetime_col), "_press_index_H")
    ).fillna(0.0)

    # Milestone 9: Time decay weighted rolling (if enabled)
    if decay_enabled:
        out["press_form_H_decay"] = out.groupby(home_col, group_keys=False).apply(
            lambda g: _roll_pre_decay(g.sort_values(datetime_col), "_press_index_H")
        ).fillna(_neutral_dom())

    out["press_dom_shots_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_dom_shots_H_raw")
    ).fillna(_neutral_dom())
    out["press_dom_sot_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_dom_sot_H_raw")
    ).fillna(_neutral_dom())
    out["press_dom_corners_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_dom_corners_H_raw")
    ).fillna(_neutral_dom())
    out["press_dom_pos_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_dom_pos_H_raw")
    ).fillna(_neutral_dom())

    # Post-match home states (including current match) for inference snapshots
    out["_press_form_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_index_H")
    ).fillna(_neutral_dom())
    out["_press_n_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post_count(g.sort_values(datetime_col), "_press_index_H")
    ).fillna(0.0)
    out["_press_dom_shots_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_dom_shots_H_raw")
    ).fillna(_neutral_dom())
    out["_press_dom_sot_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_dom_sot_H_raw")
    ).fillna(_neutral_dom())
    out["_press_dom_corners_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_dom_corners_H_raw")
    ).fillna(_neutral_dom())
    out["_press_dom_pos_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_dom_pos_H_raw")
    ).fillna(_neutral_dom())

    # Rolling evidence counts (how many matches in the window had complete stats).
    out["press_stats_n_H"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_pre_sum(g.sort_values(datetime_col), "_press_stats_present_raw")
    ).fillna(0.0)
    out["_press_stats_n_H_post"] = out.groupby(home_col, group_keys=False).apply(
        lambda g: _roll_post_sum(g.sort_values(datetime_col), "_press_stats_present_raw")
    ).fillna(0.0)

    # Away-context rolling (venue-aware)
    out["press_form_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_index_A")
    ).fillna(_neutral_dom())
    out["press_n_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre_count(g.sort_values(datetime_col), "_press_index_A")
    ).fillna(0.0)

    # Milestone 9: Time decay weighted rolling (away context)
    if decay_enabled:
        out["press_form_A_decay"] = out.groupby(away_col, group_keys=False).apply(
            lambda g: _roll_pre_decay(g.sort_values(datetime_col), "_press_index_A")
        ).fillna(_neutral_dom())

    out["press_dom_shots_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_dom_shots_A_raw")
    ).fillna(_neutral_dom())
    out["press_dom_sot_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_dom_sot_A_raw")
    ).fillna(_neutral_dom())
    out["press_dom_corners_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_dom_corners_A_raw")
    ).fillna(_neutral_dom())
    out["press_dom_pos_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre(g.sort_values(datetime_col), "_press_dom_pos_A_raw")
    ).fillna(_neutral_dom())

    # Post-match away states (including current match) for inference snapshots
    out["_press_form_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_index_A")
    ).fillna(_neutral_dom())
    out["_press_n_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post_count(g.sort_values(datetime_col), "_press_index_A")
    ).fillna(0.0)
    out["_press_dom_shots_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_dom_shots_A_raw")
    ).fillna(_neutral_dom())
    out["_press_dom_sot_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_dom_sot_A_raw")
    ).fillna(_neutral_dom())
    out["_press_dom_corners_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_dom_corners_A_raw")
    ).fillna(_neutral_dom())
    out["_press_dom_pos_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post(g.sort_values(datetime_col), "_press_dom_pos_A_raw")
    ).fillna(_neutral_dom())

    out["press_stats_n_A"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_pre_sum(g.sort_values(datetime_col), "_press_stats_present_raw")
    ).fillna(0.0)
    out["_press_stats_n_A_post"] = out.groupby(away_col, group_keys=False).apply(
        lambda g: _roll_post_sum(g.sort_values(datetime_col), "_press_stats_present_raw")
    ).fillna(0.0)

    # Remove per-match-only internal indices (leakage if kept)
    out = out.drop(
        columns=[
            "_press_index_H",
            "_press_index_A",
            "_press_dom_shots_H_raw",
            "_press_dom_shots_A_raw",
            "_press_dom_sot_H_raw",
            "_press_dom_sot_A_raw",
            "_press_dom_corners_H_raw",
            "_press_dom_corners_A_raw",
            "_press_dom_pos_H_raw",
            "_press_dom_pos_A_raw",
            "_press_stats_present_raw",
        ],
        errors="ignore",
    )

    return out
