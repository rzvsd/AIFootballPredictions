"""
Odds service: lookup odds snapshots and attach prices to market rows.
"""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

import config

_ODDS_CACHE: Dict[str, tuple[float, str, dict]] = {}
_MISSING: Dict[str, set[tuple[str, str, str, str]]] = {}


def _normalize_fixture_key(date_val, home: str, away: str) -> tuple[str, str, str]:
    try:
        dt_obj = pd.to_datetime(date_val, errors="coerce")
        if pd.notna(dt_obj):
            date_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            date_str = str(date_val)
    except Exception:
        date_str = str(date_val)
    return (
        date_str,
        config.normalize_team_name(str(home or "")),
        config.normalize_team_name(str(away or "")),
    )


def _file_sig(path: str) -> tuple[float, str]:
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        return -1.0, ""
    try:
        with open(path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
    except Exception:
        digest = ""
    return mtime, digest


def load_odds_lookup(league: str) -> dict:
    path = os.path.join("data", "odds", f"{league}.json")
    mtime, sig = _file_sig(path)
    if os.getenv("BOT_RELOAD_ODDS") == "1":
        _ODDS_CACHE.pop(league, None)
    cached = _ODDS_CACHE.get(league)
    if cached and cached[0] == mtime and cached[1] == sig:
        return cached[2]
    lookup: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        _ODDS_CACHE[league] = (mtime, sig, lookup)
        return lookup
    for fx in data.get("fixtures", []):
        key = _normalize_fixture_key(fx.get("date"), fx.get("home"), fx.get("away"))
        lookup[key] = fx.get("markets") or {}
    _ODDS_CACHE[league] = (mtime, sig, lookup)
    return lookup


def _extract_market_price(markets: dict, market: str, outcome: str) -> Optional[float]:
    try:
        if market == "1X2":
            slot_map = {"H": "home", "D": "draw", "A": "away"}
            slot = slot_map.get(outcome)
            if not slot:
                return None
            main = markets.get("1X2") or {}
            val = None
            if isinstance(main, dict):
                if "main" in main and isinstance(main["main"], dict):
                    val = main["main"].get(slot)
                if val is None and "default" in main and isinstance(main["default"], dict):
                    val = main["default"].get(slot)
                if val is None:
                    for tag_data in main.values():
                        if isinstance(tag_data, dict):
                            val = tag_data.get(slot)
                            if val is not None:
                                break
            if val is not None and not pd.isna(val):
                return float(val)
        elif market.startswith("OU "):
            line = market.split(" ", 1)[1].strip()
            bucket = (markets.get("OU") or {}).get(line, {})
            if isinstance(bucket, dict):
                tag_data = None
                if "main" in bucket:
                    tag_data = bucket.get("main") or {}
                elif "default" in bucket:
                    tag_data = bucket.get("default") or {}
                else:
                    for v in bucket.values():
                        if isinstance(v, dict):
                            tag_data = v
                            break
                if tag_data:
                    val = tag_data.get(outcome)
                    if val is not None and not pd.isna(val):
                        return float(val)
        elif market == "DC":
            for tag_data in (markets.get("DC") or {}).values():
                val = tag_data.get(outcome)
                if val is not None and not pd.isna(val):
                    return float(val)
        elif market == "TG Interval":
            for tag_data in (markets.get("Intervals") or {}).values():
                val = tag_data.get(outcome)
                if val is not None and not pd.isna(val):
                    return float(val)
    except Exception:
        return None
    return None


def lookup_market_odds(lookup: dict, date_val, home, away, market: str, outcome: str) -> Optional[float]:
    if not lookup:
        return None
    candidates = []
    try:
        dt = pd.to_datetime(date_val, errors="coerce")
    except Exception:
        dt = None
    if pd.notna(dt):
        candidates = [dt, dt - pd.Timedelta(days=1), dt + pd.Timedelta(days=1)]
    else:
        candidates = [date_val]
    for cand in candidates:
        key = _normalize_fixture_key(cand, home, away)
        markets = lookup.get(key)
        if not markets:
            continue
        price = _extract_market_price(markets, market, outcome)
        if price is not None:
            return price
    return None


def fill_odds_for_df(df: pd.DataFrame, league: str, with_odds: bool = True) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    lookup = load_odds_lookup(league) if with_odds else {}
    out = df.copy()
    odds_vals = []
    for _, row in out.iterrows():
        price = None
        if lookup:
            price = lookup_market_odds(
                lookup,
                row.get("date"),
                row.get("home"),
                row.get("away"),
                str(row.get("market")),
                str(row.get("outcome")),
            )
        if price is None:
            if lookup:
                key = _normalize_fixture_key(row.get("date"), row.get("home"), row.get("away"))
                _MISSING.setdefault(league, set()).add(key + (str(row.get("market")),))
        odds_vals.append(price)
    out["odds"] = odds_vals
    return out


def flush_missing_odds_log(league: str) -> None:
    missing = _MISSING.pop(league, set())
    if not missing:
        return
    Path("reports").mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = Path("reports") / f"{league}_{ts}_missing_odds.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("date,home,away,market\n")
        for date, home, away, market in sorted(missing):
            f.write(f"{date},{home},{away},{market}\n")
    print(f"[warn] Missing odds for {len(missing)} entries -> {log_path}")

