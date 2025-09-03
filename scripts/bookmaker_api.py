"""Utility for fetching match odds from a bookmaker API.

This module provides a :func:`get_odds` function which retrieves the latest
odds for a specific football match.  Bookmaker team names are normalised to the
model's naming conventions by re-using the mapping defined in
:mod:`bet_fusion` and extending it with a few common variants.

The API base URL and authentication token are read from the environment
variables ``BOOKMAKER_API_URL`` and ``BOOKMAKER_API_KEY`` respectively.
"""

from __future__ import annotations

import os
import time
from typing import Dict

import requests

from config import TEAM_NAME_MAP as _TEAM_NAME_MAP

# Build a normalisation mapping to the model's canonical names by
# combining the project's TEAM_NAME_MAP with common bookmaker variants.
TEAM_NORMALIZE = {}
TEAM_NORMALIZE.update(_TEAM_NAME_MAP)  # API -> dataset names
TEAM_NORMALIZE.update({
    # Bookmaker short names -> dataset names
    "Man Utd": "Man United",
    "Man City": "Man City",
    "Nottm Forest": "Nott'm Forest",
    "Spurs": "Tottenham",
})

API_URL = os.getenv("BOOKMAKER_API_URL", "https://api.bookmaker.com/odds")
API_KEY = os.getenv("BOOKMAKER_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
MAX_RETRIES = 3
BACKOFF_SECONDS = 2


def _normalize_team(name: str) -> str:
    """Return the model's canonical team name for a bookmaker provided name."""

    if name is None:
        return name
    key = name.strip()
    return TEAM_NORMALIZE.get(key, key)


def get_odds(league: str, home_team: str, away_team: str) -> Dict[str, float]:
    """Fetch odds for a match.

    Parameters
    ----------
    league : str
        League identifier used by the bookmaker API (e.g. ``"E0"``).
    home_team : str
        Bookmaker's name for the home team.
    away_team : str
        Bookmaker's name for the away team.

    Returns
    -------
    dict
        Dictionary with odds for ``home``, ``draw`` and ``away`` outcomes. An
        empty dictionary is returned if the API call ultimately fails.
    """

    params = {
        "league": league,
        "home": _normalize_team(home_team),
        "away": _normalize_team(away_team),
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
            if response.status_code == 429:
                # Rate limit hit. Wait with exponential backoff and retry.
                time.sleep(BACKOFF_SECONDS * attempt)
                continue
            response.raise_for_status()
            data = response.json()
            # Expect keys "home", "draw" and "away"; fall back gracefully.
            odds = data.get("odds", data)
            return {
                "home": odds.get("home"),
                "draw": odds.get("draw"),
                "away": odds.get("away"),
            }
        except requests.RequestException as exc:  # API or network error
            last_error = exc
            time.sleep(BACKOFF_SECONDS * attempt)
    if last_error is not None:
        print(f"Failed to fetch odds after {MAX_RETRIES} attempts: {last_error}")
    return {}


def get_odds_ou(league: str, home_team: str, away_team: str, line: float) -> Dict[str, float]:
    """Fetch Over/Under odds for a specific line (e.g., 2.5).

    Returns a dict: {"Over": odd, "Under": odd} or empty dict on failure.
    """
    params = {
        "league": league,
        "home": _normalize_team(home_team),
        "away": _normalize_team(away_team),
        "market": "ou",
        "line": float(line),
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
            if response.status_code == 429:
                time.sleep(BACKOFF_SECONDS * attempt)
                continue
            response.raise_for_status()
            data = response.json()
            odds = data.get("odds", data)
            return {
                "Over": odds.get("Over") or odds.get("over") or odds.get("O"),
                "Under": odds.get("Under") or odds.get("under") or odds.get("U"),
            }
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(BACKOFF_SECONDS * attempt)
    if last_error is not None:
        print(f"Failed to fetch OU odds after {MAX_RETRIES} attempts: {last_error}")
    return {}


def get_odds_interval(league: str, home_team: str, away_team: str, a: int, b: int) -> float | None:
    """Fetch odds for Total Goals interval a-b inclusive. Returns float or None."""
    params = {
        "league": league,
        "home": _normalize_team(home_team),
        "away": _normalize_team(away_team),
        "market": "tg_interval",
        "from": int(a),
        "to": int(b),
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
            if response.status_code == 429:
                time.sleep(BACKOFF_SECONDS * attempt)
                continue
            response.raise_for_status()
            data = response.json()
            odds = data.get("odds", data)
            # Expect a single value for the interval
            return odds.get("odd") or odds.get("value") or odds.get("interval")
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(BACKOFF_SECONDS * attempt)
    if last_error is not None:
        print(f"Failed to fetch TG Interval odds after {MAX_RETRIES} attempts: {last_error}")
    return None


__all__ = ["get_odds", "get_odds_ou", "get_odds_interval", "TEAM_NORMALIZE"]
