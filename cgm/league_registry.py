"""
League/Country canonicalization helpers.

Purpose:
  - Keep filters stable across providers/export variants.
  - Normalize legacy names (e.g., "Premier L", "Super Lig", "Liga 1")
    to API-football canonical names.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Tuple

import pandas as pd


LEAGUE_ALIASES = {
    # England
    "premier l": "Premier League",
    "premier league": "Premier League",
    "championship": "Championship",
    # Spain
    "primera": "La Liga",
    "la liga": "La Liga",
    # Germany
    "bundesliga": "Bundesliga",
    # Italy
    "serie a": "Serie A",
    "serie b": "Serie B",
    # France
    "ligue 1": "Ligue 1",
    "ligue 2": "Ligue 2",
    # Netherlands
    "eredivisie": "Eredivisie",
    # Portugal
    "primeira l": "Primeira Liga",
    "primeira liga": "Primeira Liga",
    # Turkey
    "super lig": "Süper Lig",
    "superlig": "Süper Lig",
    "sper lig": "Süper Lig",
    "süper lig": "Süper Lig",
    # Romania
    "liga 1": "Liga I",
    "liga i": "Liga I",
}


COUNTRY_ALIASES = {
    "england": "England",
    "spain": "Spain",
    "germany": "Germany",
    "italy": "Italy",
    "france": "France",
    "netherlands": "Netherlands",
    "holland": "Netherlands",
    "portugal": "Portugal",
    "turkey": "Turkey",
    "romania": "Romania",
}


def _fold(text: object) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canonicalize_league(value: object) -> str:
    s = str(value).strip() if value is not None else ""
    if not s or s.lower() == "nan":
        return ""
    key = _fold(s)
    return LEAGUE_ALIASES.get(key, s)


def canonicalize_country(value: object) -> str:
    s = str(value).strip() if value is not None else ""
    if not s or s.lower() == "nan":
        return ""
    key = _fold(s)
    return COUNTRY_ALIASES.get(key, s)


def canonicalize_scope(scope_country: str | None, scope_league: str | None) -> Tuple[str | None, str | None]:
    country = canonicalize_country(scope_country) if scope_country else None
    league = canonicalize_league(scope_league) if scope_league else None
    return country, league


def canonicalize_df_league_country(
    df: pd.DataFrame,
    *,
    league_col: str = "league",
    country_col: str = "country",
) -> pd.DataFrame:
    out = df.copy()
    if league_col in out.columns:
        out[league_col] = out[league_col].astype(str).apply(canonicalize_league)
    if country_col in out.columns:
        out[country_col] = out[country_col].astype(str).apply(canonicalize_country)
    return out

