"""
Fetch Understat league matches and per-match shots, save compact JSON files.

Usage examples:
  python -m scripts.fetch_understat_simple --league E0 --seasons 2024,2023

Notes:
- Requires network access and the `understat` package (aiohttp).
- Saves:
    data/understat/{LEAGUE}_{SEASON}_matches.json
    data/understat/{LEAGUE}_{SEASON}_shots.json  (flat list of shots with key fields)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List

import aiohttp
from understat import Understat  # type: ignore


LEAGUE_MAP = {
    "E0": "epl",
    "SP1": "la_liga",
    "D1": "bundesliga",
    "I1": "serie_a",
    "F1": "ligue_1",
}


async def fetch_for_league(league: str, seasons: List[int]) -> None:
    os.makedirs(os.path.join("data", "understat"), exist_ok=True)
    async with aiohttp.ClientSession() as session:
        us = Understat(session)
        league_key = LEAGUE_MAP.get(league, league)
        for season in seasons:
            # 1) Results first (finished matches). Fallback to fixtures if results are unavailable.
            try:
                matches = await us.get_league_results(league_key, season)
            except Exception:
                matches = []
            if not matches:
                try:
                    matches = await us.get_league_fixtures(league_key, season)
                except Exception:
                    matches = []
            m_path = os.path.join("data", "understat", f"{league}_{season}_matches.json")
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(matches, f)

            # 2) Shots per match (flatten)
            flat_shots: List[Dict[str, Any]] = []
            for m in matches:
                match_id = m.get("id") or m.get("match_id") or m.get("matchId")
                if not match_id:
                    continue
                try:
                    shots = await us.get_match_shots(int(match_id))
                except Exception:
                    continue
                # shots contains dict with keys 'h' and 'a', each a list of shot dicts
                for side_key in ("h", "a"):
                    for s in shots.get(side_key, []) or []:
                        flat_shots.append({
                            "match_id": int(match_id),
                            "h_team": s.get("h_team"),
                            "a_team": s.get("a_team"),
                            "team": s.get("team"),
                            "h_a": s.get("h_a"),
                            "minute": s.get("minute"),
                            "player": s.get("player"),
                            "shot_id": s.get("id"),
                            "result": s.get("result"),
                            "situation": s.get("situation"),
                            "shotType": s.get("shotType"),
                            "xG": s.get("xG"),
                            "X": s.get("X"),
                            "Y": s.get("Y"),
                            "fast_break": s.get("fast_break"),
                            "isKeyPass": s.get("isKeyPass"),
                            "date": s.get("date"),
                        })

            # 2b) Fallback: per-team shots if match_shots yielded nothing (site/API quirks)
            if not flat_shots:
                try:
                    teams = await us.get_league_teams(league_key, season)
                except Exception:
                    teams = []
                for t in teams:
                    team_id = t.get("id") or t.get("team_id")
                    if not team_id:
                        continue
                    try:
                        tshots = await us.get_team_shots(int(team_id), season)
                    except Exception:
                        continue
                    for s in tshots or []:
                        flat_shots.append({
                            "match_id": s.get("match_id"),
                            "h_team": s.get("h_team"),
                            "a_team": s.get("a_team"),
                            "team": s.get("team"),
                            "h_a": s.get("h_a"),
                            "minute": s.get("minute"),
                            "player": s.get("player"),
                            "shot_id": s.get("id"),
                            "result": s.get("result"),
                            "situation": s.get("situation"),
                            "shotType": s.get("shotType"),
                            "xG": s.get("xG"),
                            "X": s.get("X"),
                            "Y": s.get("Y"),
                            "fast_break": s.get("fast_break"),
                            "isKeyPass": s.get("isKeyPass"),
                            "date": s.get("date"),
                        })

            s_path = os.path.join("data", "understat", f"{league}_{season}_shots.json")
            with open(s_path, "w", encoding="utf-8") as f:
                json.dump(flat_shots, f)
            print(f"Saved matches -> {m_path}  shots -> {s_path}  (shots: {len(flat_shots)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Understat league matches and shots")
    ap.add_argument("--league", required=True, help="League code: E0,D1,F1,SP1,I1")
    ap.add_argument("--seasons", required=True, help="Comma-separated seasons (e.g., 2024,2023)")
    args = ap.parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    asyncio.run(fetch_for_league(args.league, seasons))


if __name__ == "__main__":
    main()
