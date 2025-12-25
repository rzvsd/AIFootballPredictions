"""
Utility to compare Understat fixtures vs. local Bet365 odds (data/odds/{LEAGUE}.json)
and report any matches missing real prices.

Usage:
    python -m scripts.check_odds_alignment --league E0 --days 14
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

import config
from scripts.fetch_fixtures_understat import load_understat_fixtures


def _normalize_date_token(date_val) -> str:
    try:
        dt_obj = pd.to_datetime(date_val, errors="coerce")
        if pd.notna(dt_obj):
            return dt_obj.strftime("%Y-%m-%d")
    except Exception:
        pass
    text = str(date_val).strip()
    return text[:10] if len(text) >= 10 else text


def _fixture_key(date_val, home, away) -> Tuple[str, str, str]:
    date_str = _normalize_date_token(date_val)
    return (
        date_str,
        config.normalize_team_name(str(home or "")),
        config.normalize_team_name(str(away or "")),
    )


def _load_odds_keys(league: str) -> Set[Tuple[str, str, str]]:
    odds_path = Path("data") / "odds" / f"{league}.json"
    if not odds_path.exists():
        return set()
    try:
        data = json.loads(odds_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    keys: Set[Tuple[str, str, str]] = set()
    for fx in data.get("fixtures", []):
        keys.add(_fixture_key(fx.get("date"), fx.get("home"), fx.get("away")))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description="Check odds vs Understat fixtures")
    ap.add_argument("--league", required=True, help="League code (e.g., E0)")
    ap.add_argument("--days", type=int, default=14, help="Understat lookahead window (default: 14)")
    ap.add_argument("--season", type=int, default=None, help="Optional Understat season override")
    args = ap.parse_args()

    league = args.league.strip().upper()
    fixtures = load_understat_fixtures(league, days=args.days, season=args.season)
    if fixtures.empty:
        print(f"[warn] Understat returned 0 fixtures for {league}.")
        return
    fixtures["home_norm"] = fixtures["home"].map(config.normalize_team_name)
    fixtures["away_norm"] = fixtures["away"].map(config.normalize_team_name)
    fixtures["key"] = fixtures.apply(lambda r: _fixture_key(r["date"], r["home_norm"], r["away_norm"]), axis=1)

    odds_keys = _load_odds_keys(league)
    if not odds_keys:
        print(f"[warn] Odds file missing or empty: data/odds/{league}.json")

    missing = [tuple(k) for k in fixtures["key"] if k not in odds_keys]
    extra = [k for k in odds_keys if k not in set(fixtures["key"])]

    if missing:
        print(f"[warn] {len(missing)} Understat fixtures lack Bet365 odds:")
        for date, home, away in missing:
            print(f"  - {date} | {home} vs {away}")
    else:
        print("[ok] All Understat fixtures have matching odds entries.")

    if extra:
        print(f"[info] {len(extra)} odds entries do not match current Understat fixtures:")
        for date, home, away in extra:
            print(f"  - {date} | {home} vs {away}")


if __name__ == "__main__":
    main()
