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

# Load .env if present so env keys (e.g., BOOKMAKER_API_URL, BOT_ODDS_MODE) are available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

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

# Local odds support
ODDS_MODE = os.getenv("BOT_ODDS_MODE", "local").lower()  # 'local' or 'api'
ODDS_TAG = os.getenv("BOT_ODDS_TAG", "closing").lower()  # 'opening' or 'closing'
ODDS_DIR = os.getenv("BOT_ODDS_DIR", os.path.join("data", "odds"))


def _load_local_odds(league: str) -> dict:
    """Load local odds JSON for a league. Schema:
    {
      "fixtures": [
        {
          "date": "YYYY-MM-DD" (optional),
          "home": "Team",
          "away": "Team",
          "markets": {
            "1X2": {"home": 2.1, "draw": 3.4, "away": 3.6},
            "OU": {"2.5": {"Over": 1.95, "Under": 1.85}},
            "TG": {"0-3": 2.8, "1-3": 2.6, "2-4": 3.0, "2-5": 3.4, "3-6": 3.6}
          }
        }, ...
      ]
    }
    """
    path = os.path.join(ODDS_DIR, f"{league}.json")
    try:
        if os.path.exists(path):
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        return {}
    return {}


def _find_fixture_local(league: str, home: str, away: str) -> dict | None:
    data = _load_local_odds(league)
    fixtures = data.get("fixtures", [])
    home_n = _normalize_team(home)
    away_n = _normalize_team(away)
    for fx in fixtures:
        h = _normalize_team(str(fx.get("home", "")))
        a = _normalize_team(str(fx.get("away", "")))
        if h == home_n and a == away_n:
            return fx
    return None


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

    # Local odds first (if enabled)
    if ODDS_MODE == "local":
        fx = _find_fixture_local(league, home_team, away_team)
        if fx:
            mk = (fx.get("markets") or {}).get("1X2") or {}
            # Support nested opening/closing
            if any(k in mk for k in ("opening","closing")):
                op = (mk.get("opening") or {})
                cl = (mk.get("closing") or {})
                choose = cl if ODDS_TAG == 'closing' else op if ODDS_TAG == 'opening' else cl or op
                out = {"home": choose.get("home"), "draw": choose.get("draw"), "away": choose.get("away")}
                out["_open"] = {"home": op.get("home"), "draw": op.get("draw"), "away": op.get("away")}
                out["_close"] = {"home": cl.get("home"), "draw": cl.get("draw"), "away": cl.get("away")}
                return out
            return {"home": mk.get("home"), "draw": mk.get("draw"), "away": mk.get("away")}
        else:
            return {}

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
    # Local odds
    if ODDS_MODE == "local":
        fx = _find_fixture_local(league, home_team, away_team)
        if fx:
            ou = ((fx.get("markets") or {}).get("OU") or {}).get(f"{float(line):.1f}") or {}
            if any(k in ou for k in ("opening","closing")):
                op = (ou.get("opening") or {})
                cl = (ou.get("closing") or {})
                choose = cl if ODDS_TAG == 'closing' else op if ODDS_TAG == 'opening' else cl or op
                out = {"Over": choose.get("Over"), "Under": choose.get("Under")}
                out["_open"] = {"Over": op.get("Over"), "Under": op.get("Under")}
                out["_close"] = {"Over": cl.get("Over"), "Under": cl.get("Under")}
                return out
            return {"Over": ou.get("Over"), "Under": ou.get("Under")}
        else:
            return {}

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
    # Local odds
    if ODDS_MODE == "local":
        fx = _find_fixture_local(league, home_team, away_team)
        if fx:
            tg = ((fx.get("markets") or {}).get("TG") or {})
            val = tg.get(f"{a}-{b}")
            return float(val) if val is not None else None
        else:
            return None

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
