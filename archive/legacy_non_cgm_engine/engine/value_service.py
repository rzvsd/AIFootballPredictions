"""
Value service: attach odds, price source, edge, EV to market probabilities.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import odds_service

_SYNTH_MARGIN_DEFAULT = float(os.getenv("BOT_SYNTH_MARGIN", "0.06"))
_SYNTH_MARGIN_BY_MARKET = {
    "1X2": 0.05,
    "DC": 0.05,
    "OU": 0.065,
    "TG Interval": 0.18,
}


def _market_base(m: str) -> str:
    return m.split()[0] if m.startswith("OU ") else m


def _market_is_exclusive(market: str) -> bool:
    base = _market_base(market)
    return base in ("1X2", "DC", "TG Interval") or market.startswith("OU ")


def _placeholder_odds(market: str, outcome: str) -> float:
    if market == "1X2":
        return 2.0
    if market == "DC":
        return 1.33
    if market.startswith("OU "):
        return 2.0
    if market == "TG Interval":
        return 3.0
    return 2.0


def _synth_odds_for_group(
    df: pd.DataFrame,
    market: str,
    margin_lookup: Dict[str, float],
    default_margin: float,
    force_all: bool = False,
) -> pd.DataFrame:
    probs = pd.to_numeric(df["prob"], errors="coerce")
    if not force_all:
        mask = df["odds"].isna()
    else:
        mask = pd.Series([True] * len(df), index=df.index)
    if not mask.any():
        df["_synthetic_odds"] = False
        return df
    base = _market_base(market)
    margin = margin_lookup.get(base, default_margin)
    if _market_is_exclusive(market):
        total = probs.sum()
        if total <= 0:
            fair_probs = pd.Series(1.0 / max(len(df), 1), index=df.index)
        else:
            fair_probs = probs / total
        adj_prob = (fair_probs * (1.0 + margin)).clip(lower=1e-6)
        new_odds = 1.0 / adj_prob
    else:
        adj_prob = (probs * (1.0 + margin)).clip(lower=1e-6)
        new_odds = 1.0 / adj_prob
    df.loc[mask, "odds"] = new_odds[mask]
    df["_synthetic_odds"] = mask
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce").clip(lower=1.01)
    return df


def attach_value_metrics(
    market_df: pd.DataFrame,
    use_placeholders: bool = True,
    synthesize_missing_odds: bool = True,
    synth_margin: float | None = None,
    league_code: str | None = None,
    synth_margin_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if market_df.empty:
        return market_df.copy()
    df = market_df.copy()
    synth_margin = _SYNTH_MARGIN_DEFAULT if synth_margin is None else float(synth_margin)
    margin_map = synth_margin_map or _SYNTH_MARGIN_BY_MARKET
    placeholder_filled = False
    try:
        df["fair_odds"] = 1.0 / pd.to_numeric(df["prob"], errors="coerce").clip(lower=1e-9)
    except Exception:
        df["fair_odds"] = None
    if "odds" not in df.columns and use_placeholders:
        df["odds"] = [_placeholder_odds(m, o) for m, o in zip(df["market"].astype(str), df["outcome"].astype(str))]
        placeholder_filled = True
    df["_synthetic_odds"] = False
    grouped = []
    for (date, home, away, market), sub in df.groupby(["date", "home", "away", "market"], dropna=False):
        sub2 = sub.copy()
        sub2["odds"] = pd.to_numeric(sub2["odds"], errors="coerce")
        if synthesize_missing_odds:
            sub2 = _synth_odds_for_group(sub2, market, margin_map, synth_margin, force_all=placeholder_filled)
        sub2["p_imp"] = 1.0 / sub2["odds"].astype(float)
        if _market_is_exclusive(market):
            s = sub2["p_imp"].sum()
            sub2["p_imp_norm"] = sub2["p_imp"] / s if s > 0 else sub2["p_imp"]
        else:
            sub2["p_imp_norm"] = sub2["p_imp"]
        sub2["edge"] = sub2["prob"] - sub2["p_imp_norm"]
        sub2["EV"] = sub2["prob"] * sub2["odds"] - 1.0
        grouped.append(sub2)
    out = pd.concat(grouped, ignore_index=True) if grouped else df
    try:
        out["book_odds"] = pd.to_numeric(out["odds"], errors="coerce")
    except Exception:
        out["book_odds"] = out.get("odds")
    out["price_source"] = np.where(out["_synthetic_odds"], "synth", "real")
    try:
        mask_synth = out["price_source"] != "real"
        if "EV" in out.columns:
            out.loc[mask_synth, "EV"] = np.nan
    except Exception:
        pass
    out.drop(columns=["_synthetic_odds"], errors="ignore", inplace=True)
    return out


def fill_and_value(markets_df: pd.DataFrame, league: str) -> pd.DataFrame:
    df_odds = odds_service.fill_odds_for_df(markets_df, league, with_odds=True)
    return attach_value_metrics(df_odds, use_placeholders=True, league_code=league)

