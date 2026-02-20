#!/usr/bin/env python3
"""
Telegram picks sender (manual).

Sends picks only (OU2.5 + BTTS) grouped by Date -> League.
Source: reports/picks.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import requests

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

try:
    from telegram_config import (
        TELEGRAM_BOT_TOKEN,
        TELEGRAM_CHAT_ID,
        TELEGRAM_MESSAGE_MAX_CHARS,
        TELEGRAM_MIN_ODDS,
    )
except ImportError:
    print("ERROR: telegram_config.py not found.")
    sys.exit(1)


MARKET_LABEL = {
    "OU25_OVER": "Over 2.5",
    "OU25_UNDER": "Under 2.5",
    "BTTS_YES": "BTTS Yes",
    "BTTS_NO": "BTTS No",
}


def get_updates(token: str) -> dict:
    """Get recent updates to find chat ID."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    resp = requests.get(url, timeout=10)
    return resp.json()


def send_message(token: str, chat_id: str, text: str) -> bool:
    """Send a message via Telegram Bot API (plain text)."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
    }
    resp = requests.post(url, json=payload, timeout=10)
    result = resp.json()
    if not result.get("ok"):
        print(f"ERROR: {result.get('description', 'Unknown error')}")
        return False
    return True


def _split_messages(lines: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = current + [line]
        if len("\n".join(candidate)) > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current = candidate
    if current:
        chunks.append("\n".join(current))
    return chunks


def _format_picks_message(df: pd.DataFrame, *, min_odds: float) -> list[str]:
    if df.empty:
        return ["No picks available."]

    odds = pd.to_numeric(df.get("odds"), errors="coerce")
    df = df[odds >= float(min_odds)].copy()
    if df.empty:
        return [f"No picks with odds >= {min_odds:.2f}."]

    df["fixture_dt"] = pd.to_datetime(df["fixture_datetime"], errors="coerce")
    df = df[df["fixture_dt"].notna()].copy()
    df["date"] = df["fixture_dt"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["date", "league", "fixture_dt", "home", "away"], kind="mergesort")

    lines: list[str] = []
    today = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"PICKS ({today})")
    lines.append(f"Min odds: {min_odds:.2f}")
    lines.append("")

    for date_val in df["date"].unique():
        lines.append(date_val)
        day_df = df[df["date"] == date_val]
        for league in day_df["league"].unique():
            lines.append(f"  {league}")
            lg_df = day_df[day_df["league"] == league]
            for _, row in lg_df.iterrows():
                market = MARKET_LABEL.get(str(row.get("market", "")), str(row.get("market", "")))
                lines.append(f"    {row.get('home', '')} vs {row.get('away', '')} | {market}")
        lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Send picks to Telegram (manual)")
    parser.add_argument("--league", help="Filter by league")
    parser.add_argument("--get-chat-id", action="store_true", help="Show your chat ID")
    parser.add_argument("--test", action="store_true", help="Send test message")
    parser.add_argument("--dry-run", action="store_true", help="Print message without sending")
    parser.add_argument("--picks", default="reports/picks.csv", help="Picks CSV path")
    args = parser.parse_args()

    # Get chat ID mode
    if args.get_chat_id:
        if not TELEGRAM_BOT_TOKEN:
            print("ERROR: TELEGRAM_BOT_TOKEN is empty.")
            return 1
        print("Checking for chat ID...")
        updates = get_updates(TELEGRAM_BOT_TOKEN)
        if not updates.get("ok"):
            print(f"ERROR: {updates.get('description', 'Unknown error')}")
            return 1
        results = updates.get("result", [])
        if not results:
            print("No messages found. Send a message to your bot first.")
            return 1
        chat_id = results[-1]["message"]["chat"]["id"]
        print(f"Your Chat ID: {chat_id}")
        print('Add this to telegram_config.py:')
        print(f'TELEGRAM_CHAT_ID = "{chat_id}"')
        return 0

    if not args.dry_run and not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN is empty.")
        return 1
    if not TELEGRAM_CHAT_ID and not args.dry_run:
        print("ERROR: TELEGRAM_CHAT_ID is empty.")
        return 1

    if args.test:
        message = "Test message from AI Football Predictions."
        if args.dry_run:
            print(message)
            return 0
        return 0 if send_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, message) else 1

    picks_path = _ROOT / args.picks
    if not picks_path.exists():
        print(f"ERROR: Picks file not found: {picks_path}")
        return 1

    df = pd.read_csv(picks_path)
    if args.league:
        df = df[df["league"] == args.league]

    lines = _format_picks_message(df, min_odds=float(TELEGRAM_MIN_ODDS))
    messages = _split_messages(lines, int(TELEGRAM_MESSAGE_MAX_CHARS))

    for msg in messages:
        if args.dry_run:
            print(msg)
            print("-" * 60)
        else:
            if not send_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg):
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
