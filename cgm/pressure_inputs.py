"""
Milestone 2: Pressure Cooker - input normalization.

Deterministic mapping from various CGM schemas (enhanced history / raw CGM / upcoming)
to a canonical split set of numeric columns used by Pressure features:

  shots_H, shots_A
  sot_H, sot_A
  corners_H, corners_A
  pos_H, pos_A
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _to_float(x) -> float | None:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return None
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def _split_ha(val) -> Tuple[float | None, float | None]:
    """
    Parse a "home-away" string like "12-2" (or "12 - 2") into (home, away).
    Returns (None, None) if parsing fails.
    """
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return (None, None)
        if isinstance(val, (int, float, np.number)):
            return (None, None)
        s = str(val).strip()
        if not s or s.lower() == "nan":
            return (None, None)
        if "-" not in s:
            return (None, None)
        a, b = s.split("-", 1)
        return (_to_float(a), _to_float(b))
    except Exception:
        return (None, None)


def _ensure_pair_from_split_cols(df: pd.DataFrame, home_col: str, away_col: str) -> tuple[pd.Series, pd.Series] | None:
    if home_col in df.columns and away_col in df.columns:
        h = pd.to_numeric(df[home_col], errors="coerce")
        a = pd.to_numeric(df[away_col], errors="coerce")
        return (h, a)
    return None


def _ensure_pair_from_combined_col(df: pd.DataFrame, combined_col: str) -> tuple[pd.Series, pd.Series] | None:
    if combined_col not in df.columns:
        return None
    pairs = df[combined_col].apply(_split_ha)
    h = pairs.apply(lambda t: t[0]).astype(float)
    a = pairs.apply(lambda t: t[1]).astype(float)
    return (pd.to_numeric(h, errors="coerce"), pd.to_numeric(a, errors="coerce"))


def ensure_pressure_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add canonical split stats columns to `df` (does not drop existing columns).

    Mapping priority (per stat):
      1) If canonical columns already exist, keep them (fill missing only).
      2) If split columns exist (e.g., suth/suta), use them.
      3) If combined "H-A" columns exist (e.g., shots from sut, ballp), parse them.
      4) Otherwise leave as NaN.
    """
    return ensure_pressure_inputs_fill(df, overwrite=False)


def ensure_pressure_inputs_fill(df: pd.DataFrame, *, overwrite: bool = False) -> pd.DataFrame:
    """
    Variant of `ensure_pressure_inputs` that can optionally overwrite canonical columns.

    Default behavior is safety-first: canonical columns are *only* filled where missing.
    Set `overwrite=True` to force a re-map from CGM source columns (useful for debugging).
    """
    out = df.copy()

    # Ensure canonical columns exist (keeps downstream code simpler).
    for c in ("shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A"):
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    def _apply_pair(home_col: str, away_col: str, pair: tuple[pd.Series, pd.Series] | None) -> None:
        if pair is None:
            return
        h, a = pair
        if overwrite:
            out[home_col] = h
            out[away_col] = a
        else:
            out[home_col] = out[home_col].combine_first(h)
            out[away_col] = out[away_col].combine_first(a)

    # Shots
    pair = (
        _ensure_pair_from_split_cols(out, "shots_home", "shots_away")
        or _ensure_pair_from_split_cols(out, "suth", "suta")
        or _ensure_pair_from_combined_col(out, "shots")
        or _ensure_pair_from_combined_col(out, "sut")
    )
    _apply_pair("shots_H", "shots_A", pair)

    # Shots on target
    pair = (
        _ensure_pair_from_split_cols(out, "sot_home", "sot_away")
        or _ensure_pair_from_split_cols(out, "sutht", "sutat")
        or _ensure_pair_from_combined_col(out, "shots_on_target")
        or _ensure_pair_from_combined_col(out, "sutt")
    )
    _apply_pair("sot_H", "sot_A", pair)

    # Corners
    pair = (
        _ensure_pair_from_split_cols(out, "corners_home", "corners_away")
        or _ensure_pair_from_split_cols(out, "corh", "cora")
        or _ensure_pair_from_combined_col(out, "corners")
        or _ensure_pair_from_combined_col(out, "cor")
    )
    _apply_pair("corners_H", "corners_A", pair)

    # Possession
    pair = (
        _ensure_pair_from_split_cols(out, "possession_home", "possession_away")
        or _ensure_pair_from_split_cols(out, "ballph", "ballpa")
        or _ensure_pair_from_combined_col(out, "ballp")
    )
    _apply_pair("pos_H", "pos_A", pair)

    return out
