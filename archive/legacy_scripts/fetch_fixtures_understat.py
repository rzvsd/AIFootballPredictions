"""
Fetch upcoming fixtures directly from Understat (no API key required).
Writes to data/fixtures/{LEAGUE}_weekly_fixtures.csv

Usage:
  python -m scripts.fetch_fixtures_understat --leagues E0 D1 F1 I1 SP1
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Dict, List

import aiohttp
import pandas as pd
from understat import Understat  # type: ignore

import config

LEAGUE_MAP = {
    "E0": "epl",
    "SP1": "la_liga",
    "D1": "bundesliga",
    "I1": "serie_a",
    "F1": "ligue_1",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    )
}


def current_season_year() -> int:
    today = dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def _mock_fixtures_from_processed(league: str, horizon_days: int) -> List[Dict[str, str]]:
    """Fallback: synthesize a schedule using processed data team names."""
    proc_path = Path("data/processed") / f"{league}_merged_preprocessed.csv"
    if not proc_path.exists():
        print(f"[{league}] Mock fallback failed: {proc_path} missing.")
        return []
    try:
        df = pd.read_csv(proc_path)
    except Exception as exc:
        print(f"[{league}] Mock fallback failed to read processed CSV: {exc}")
        return []
    teams = sorted(set(df.get("HomeTeam").dropna().astype(str)).union(set(df.get("AwayTeam").dropna().astype(str))))
    teams = [config.normalize_team_name(t) for t in teams if t]
    teams = sorted(set(teams))
    if len(teams) < 2:
        print(f"[{league}] Mock fallback found insufficient teams ({len(teams)}).")
        return []
    # Simple round-robin pairing (first N unique pairs)
    pairs = []
    for i, h in enumerate(teams):
        for a in teams[i+1:]:
            pairs.append((h, a))
    if not pairs:
        return []
    # Start next Saturday at 15:00 local time and spread matches
    today = dt.date.today()
    days_to_sat = (5 - today.weekday()) % 7  # Saturday=5
    start_date = dt.datetime.combine(today + dt.timedelta(days=days_to_sat), dt.time(15, 0))
    out = []
    max_matches = min(len(pairs), horizon_days * 2)  # rough cap
    for idx, (h, a) in enumerate(pairs[:max_matches]):
        kickoff = start_date + dt.timedelta(days=idx // 5, hours=(idx % 5) * 2)
        out.append({
            "date": kickoff.strftime("%Y-%m-%d %H:%M:%S"),
            "home": h,
            "away": a,
        })
    print(f"[{league}] Mock fixtures synthesized from processed data: {len(out)} matches.")
    return out


async def fetch_league(league: str, horizon_days: int, season_override: int | None = None, quiet: bool = False) -> List[Dict[str, str]]:
    slug = LEAGUE_MAP.get(league)
    if not slug:
        if not quiet:
            print(f"[{league}] Unsupported league code.")
        return []
    season = season_override or current_season_year()
    now_utc = dt.datetime.now(dt.timezone.utc)
    window_end = now_utc + dt.timedelta(days=horizon_days)
    out: List[Dict[str, str]] = []
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        api = Understat(session)
        try:
            fixtures = await api.get_league_fixtures(slug, season)
        except Exception as exc:
            if not quiet:
                print(f"[{league}] Understat fetch failed: {exc} (season={season})")
            # Fallback: try previous season once (common Understat quirk around year change)
            try:
                alt_season = season - 1
                fixtures = await api.get_league_fixtures(slug, alt_season)
                if not quiet:
                    print(f"[{league}] Retried with season={alt_season}, got {len(fixtures or [])} fixtures")
            except Exception as exc2:
                if not quiet:
                    print(f"[{league}] Understat retry failed: {exc2}")
                return []
        for fx in fixtures or []:
            dt_str = fx.get("datetime") or fx.get("date")
            if not dt_str:
                continue
            kickoff_ts = pd.to_datetime(dt_str, errors="coerce", utc=True)
            if pd.isna(kickoff_ts):
                continue
            kickoff = kickoff_ts.to_pydatetime()
            if kickoff < now_utc or kickoff > window_end:
                continue
            home = config.normalize_team_name((fx.get("h") or {}).get("title") or "")
            away = config.normalize_team_name((fx.get("a") or {}).get("title") or "")
            if not home or not away:
                continue
            out.append(
                {
                    "date": kickoff.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
                    "home": home,
                    "away": away,
                }
            )
    out.sort(key=lambda r: r["date"])
    return out


def load_understat_fixtures(league: str, days: int = 14, season: int | None = None) -> pd.DataFrame:
    """Return a DataFrame of future fixtures for a league sourced from Understat."""
    league_norm = league.strip().upper()
    def _as_bool(val) -> bool:
        return str(val).strip().lower() in {"1", "true", "yes", "on"}
    quiet = _as_bool(os.getenv("BOT_UNDERSTAT_QUIET", "1"))
    if _as_bool(os.getenv("BOT_SKIP_UNDERSTAT")) or _as_bool(os.getenv("BOT_OFFLINE")) or _as_bool(os.getenv("DRY_RUN")):
        rows = _mock_fixtures_from_processed(league_norm, horizon_days=days)
        if not rows:
            return pd.DataFrame(columns=["date", "home", "away"])
        return pd.DataFrame(rows)
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Try the requested season; if it fails (e.g., future season not yet on Understat),
    # retry with previous/auto season before falling back to synthetic fixtures.
    def _run(season_override: int | None):
        return asyncio.run(fetch_league(league_norm, days, season_override=season_override, quiet=quiet))

    rows: List[Dict[str, str]] = []
    try:
        rows = _run(season)
    except Exception as e:
        prev = (season - 1) if season else None
        if prev is not None:
            try:
                rows = _run(prev)
                print(f"[{league_norm}] Understat retry succeeded with season={prev} (rows={len(rows)})")
            except Exception:
                rows = []
        if not rows and not quiet:
            print(f"[{league_norm}] Understat unavailable ({e}); using processed mock fixtures.")
    if not rows:
        # Fallback: synthesize from processed data
        rows = _mock_fixtures_from_processed(league_norm, horizon_days=days)
    if not rows:
        return pd.DataFrame(columns=["date", "home", "away"])
    return pd.DataFrame(rows)


async def main_async(leagues: List[str], horizon_days: int, season: int | None = None) -> None:
    tasks = [fetch_league(lg, horizon_days, season_override=season) for lg in leagues]
    results = await asyncio.gather(*tasks)
    fixtures_dir = Path("data") / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    for league, data in zip(leagues, results):
        if not data:
            print(f"[{league}] No upcoming fixtures found.")
            continue
        df = pd.DataFrame(data)
        out_path = fixtures_dir / f"{league}_weekly_fixtures.csv"
        df.to_csv(out_path, index=False)
        print(f"[{league}] Saved {len(df)} fixtures -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Understat fixtures")
    parser.add_argument("--leagues", nargs="+", required=True, help="Leagues e.g. E0 D1")
    parser.add_argument("--days", type=int, default=21, help="Lookahead window in days (default: 21)")
    parser.add_argument("--season", type=int, default=None, help="Optional Understat season start year override (e.g., 2025)")
    args = parser.parse_args()
    leagues = [lg.strip().upper() for lg in args.leagues]
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async(leagues, args.days, season=args.season))


if __name__ == "__main__":
    main()
