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
from typing import Any, Dict, List, Tuple

import aiohttp
from understat import Understat  # type: ignore


LEAGUE_MAP = {
    "E0": "epl",
    "SP1": "la_liga",
    "D1": "bundesliga",
    "I1": "serie_a",
    "F1": "ligue_1",
}

MAX_CONCURRENT_REQUESTS = int(os.getenv("UNDERSTAT_MAX_CONCURRENCY", "8"))
USER_AGENT = os.getenv(
    "UNDERSTAT_USER_AGENT",
    "AIFootballPredictions/1.0 (+https://github.com/rzvsd/AIFootballPredictions)",
)


async def _fetch_match_shots(us: Understat, match_id: int, sem: asyncio.Semaphore) -> Tuple[int, Dict[str, Any] | None]:
    async with sem:
        try:
            data = await us.get_match_shots(int(match_id))
            return match_id, data
        except Exception:
            return match_id, None


async def _fetch_team_shots(us: Understat, team_id: int, season: int, sem: asyncio.Semaphore) -> Tuple[int, List[Dict[str, Any]]]:
    async with sem:
        try:
            shots = await us.get_team_shots(int(team_id), season)
            return team_id, shots or []
        except Exception:
            return team_id, []


async def fetch_for_league(league: str, seasons: List[int]) -> None:
    os.makedirs(os.path.join("data", "understat"), exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(headers=headers, raise_for_status=False) as session:
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
            sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            match_ids: List[int] = []
            for m in matches:
                match_id = m.get("id") or m.get("match_id") or m.get("matchId")
                if match_id is None:
                    continue
                try:
                    match_ids.append(int(match_id))
                except Exception:
                    continue

            if match_ids:
                tasks = [_fetch_match_shots(us, mid, sem) for mid in match_ids]
                results = await asyncio.gather(*tasks)
                for match_id, shots in results:
                    if not shots:
                        continue
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
                sem_fallback = asyncio.Semaphore(max(2, MAX_CONCURRENT_REQUESTS // 2))
                team_ids: List[int] = []
                for t in teams:
                    team_id = t.get("id") or t.get("team_id")
                    if team_id is None:
                        continue
                    try:
                        team_ids.append(int(team_id))
                    except Exception:
                        continue
                if team_ids:
                    tasks = [_fetch_team_shots(us, tid, season, sem_fallback) for tid in team_ids]
                    team_results = await asyncio.gather(*tasks)
                    for _, tshots in team_results:
                        for s in tshots:
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
    if os.name == "nt":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    asyncio.run(fetch_for_league(args.league, seasons))


if __name__ == "__main__":
    main()
