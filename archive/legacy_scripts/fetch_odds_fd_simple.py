"""
DEPRECATED: Legacy odds fetcher from football-data.co.uk (fixtures.csv).
Current pipeline uses scripts/fetch_odds_toa.py (The Odds API). Kept for reference only.

Usage (legacy):
    python -m scripts.fetch_odds_fd_simple --leagues E0 D1 F1 I1 SP1 --days 14 --tag closing
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import requests

import config

FD_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DEFAULT_LEAGUES = ["E0", "D1", "F1", "I1", "SP1"]


def _download_fixtures_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content = resp.content.decode("utf-8", errors="ignore")
    data = pd.read_csv(StringIO(content))
    return data


def _parse_datetime(date_str: str, time_str: Optional[str]) -> Optional[dt.datetime]:
    raw = (f"{date_str} {time_str}".strip() if time_str else date_str).strip()
    if not raw:
        return None
    try:
        return pd.to_datetime(raw, dayfirst=True, errors="coerce").to_pydatetime()
    except Exception:
        return None


def _to_float(val) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _build_markets(row: pd.Series, tag: str) -> Dict:
    markets: Dict[str, Dict] = {}
    b_h = _to_float(row.get("B365H"))
    b_d = _to_float(row.get("B365D"))
    b_a = _to_float(row.get("B365A"))
    if any(x is not None for x in (b_h, b_d, b_a)):
        markets.setdefault("1X2", {})[tag] = {
            "home": b_h,
            "draw": b_d,
            "away": b_a,
        }

    ou_lines: Dict[str, Dict[str, Optional[float]]] = {}
    for col, val in row.items():
        if not isinstance(col, str):
            continue
        col = col.strip()
        if col.startswith("B365>"):
            line = col.replace("B365>", "").strip()
            ou_lines.setdefault(line, {})["Over"] = _to_float(val)
        elif col.startswith("B365<"):
            line = col.replace("B365<", "").strip()
            ou_lines.setdefault(line, {})["Under"] = _to_float(val)
    if ou_lines:
        for line, kv in ou_lines.items():
            if kv:
                markets.setdefault("OU", {}).setdefault(line, {})[tag] = kv
    return markets


def _normalize_date_str(dt_obj: dt.datetime) -> str:
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def _process_league(df: pd.DataFrame, league: str, days: int, tag: str) -> List[Dict]:
    if "Div" not in df.columns:
        return []
    today = dt.datetime.now().date()
    horizon = today + dt.timedelta(days=max(days, 1))
    subset = df[df["Div"].astype(str).str.upper() == league]

    out: List[Dict] = []
    for _, row in subset.iterrows():
        date_raw = str(row.get("Date", "")).strip()
        time_raw = str(row.get("Time", "")).strip() if "Time" in row else ""
        kickoff = _parse_datetime(date_raw, time_raw)
        if kickoff is None:
            continue
        if kickoff.date() < today or kickoff.date() > horizon:
            continue
        home = config.normalize_team_name(str(row.get("HomeTeam", "")).strip())
        away = config.normalize_team_name(str(row.get("AwayTeam", "")).strip())
        if not home or not away:
            continue
        markets = _build_markets(row, tag)
        entry = {
            "date": _normalize_date_str(kickoff),
            "home": home,
            "away": away,
            "markets": markets,
        }
        out.append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Bet365 odds from football-data.co.uk fixtures.csv")
    ap.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES, help="League codes (e.g., E0 D1)")
    ap.add_argument("--url", default=FD_FIXTURES_URL, help="Fixtures CSV URL")
    ap.add_argument("--days", type=int, default=14, help="Include fixtures within the next N days (default: 14)")
    ap.add_argument("--tag", default="closing", help="Market tag to store odds under (default: closing)")
    args = ap.parse_args()

    leagues = [lg.strip().upper() for lg in args.leagues if lg.strip()]
    try:
        df = _download_fixtures_csv(args.url)
    except Exception as exc:
        raise SystemExit(f"Failed to download fixtures CSV: {exc}") from exc

    odds_dir = os.getenv("BOT_ODDS_DIR", os.path.join("data", "odds"))
    os.makedirs(odds_dir, exist_ok=True)

    for lg in leagues:
        entries = _process_league(df, lg, args.days, args.tag)
        if not entries:
            print(f"[warn] No upcoming fixtures with Bet365 odds for {lg} in next {args.days} days.")
            continue
        path = os.path.join(odds_dir, f"{lg}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"fixtures": entries}, f, indent=2)
        print(f"[ok] Saved odds -> {path} (fixtures: {len(entries)})")


if __name__ == "__main__":
    main()
