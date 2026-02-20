#!/usr/bin/env python3
"""
Weekly Telegram report: picks for Monday->Sunday (all leagues), grouped by Date->League.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from telegram_config import (  # noqa: E402
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_MESSAGE_MAX_CHARS,
    TELEGRAM_MIN_ODDS,
    TELEGRAM_TZ,
    TELEGRAM_WEEKLY_HORIZON_DAYS,
)


MARKET_LABEL = {
    "OU25_OVER": "Over 2.5",
    "OU25_UNDER": "Under 2.5",
    "BTTS_YES": "BTTS Yes",
    "BTTS_NO": "BTTS No",
}


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _send_message(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("description", "Telegram send failed"))
    return True


def _split_messages(body_lines: list[str], header_lines: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = list(header_lines)
    for line in body_lines:
        candidate = current + [line]
        if len("\n".join(candidate)) > max_chars and len(current) > len(header_lines):
            chunks.append("\n".join(current))
            current = list(header_lines) + [line]
        else:
            current = candidate
    if current:
        chunks.append("\n".join(current))
    return chunks


def _run_weekly_snapshot(
    *,
    week_start: date,
    history: Path,
    models_dir: Path,
    data_dir: Path,
    out_dir: Path,
    horizon_days: int,
    model_variant: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "cgm_upcoming_predictions.csv"
    picks_path = out_dir / "picks.csv"

    as_of_date = (week_start - timedelta(days=1)).isoformat()

    cmd_pred = [
        sys.executable,
        "-m",
        "cgm.predict_upcoming",
        "--history",
        str(history),
        "--models-dir",
        str(models_dir),
        "--model-variant",
        model_variant,
        "--out",
        str(pred_path),
        "--data-dir",
        str(data_dir),
        "--as-of-date",
        as_of_date,
        "--horizon-days",
        str(horizon_days),
        "--log-level",
        "WARNING",
    ]
    res = subprocess.run(cmd_pred, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"predict_upcoming failed:\nSTDERR:\n{res.stderr}")

    cmd_picks = [
        sys.executable,
        "-m",
        "cgm.pick_engine_goals",
        "--in",
        str(pred_path),
        "--out",
        str(picks_path),
    ]
    res = subprocess.run(cmd_picks, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pick_engine_goals failed:\nSTDERR:\n{res.stderr}")

    return pred_path, picks_path


def _format_weekly_lines(picks: pd.DataFrame, *, min_odds: float) -> list[str]:
    if picks.empty:
        return ["No picks available for this week."]

    odds = pd.to_numeric(picks.get("odds"), errors="coerce")
    picks = picks[odds >= float(min_odds)].copy()
    if picks.empty:
        return [f"No picks with odds >= {min_odds:.2f}."]

    picks["fixture_dt"] = pd.to_datetime(picks["fixture_datetime"], errors="coerce")
    picks = picks[picks["fixture_dt"].notna()].copy()
    picks["date"] = picks["fixture_dt"].dt.strftime("%Y-%m-%d")
    picks = picks.sort_values(["date", "league", "fixture_dt", "home", "away"], kind="mergesort")

    lines: list[str] = []
    for date_val in picks["date"].unique():
        lines.append(date_val)
        day_df = picks[picks["date"] == date_val]
        for league in day_df["league"].unique():
            lines.append(f"  {league}")
            lg_df = day_df[day_df["league"] == league]
            for _, row in lg_df.iterrows():
                market = MARKET_LABEL.get(str(row.get("market", "")), str(row.get("market", "")))
                lines.append(f"    {row.get('home', '')} vs {row.get('away', '')} | {market}")
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Send weekly picks report to Telegram")
    ap.add_argument("--week-start", help="Override week start (YYYY-MM-DD)")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--data-dir", default="CGM data")
    ap.add_argument("--model-variant", choices=["full", "no_odds"], default="full")
    ap.add_argument("--dry-run", action="store_true", help="Print messages without sending")
    args = ap.parse_args()

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
        return 1

    tz = ZoneInfo(TELEGRAM_TZ)
    today_local = datetime.now(tz).date()
    week_start = _week_start(today_local)
    if args.week_start:
        week_start = datetime.strptime(args.week_start, "%Y-%m-%d").date()

    week_end = week_start + timedelta(days=6)
    snap_dir = _ROOT / "reports" / "weekly_snapshots" / week_start.isoformat()

    _run_weekly_snapshot(
        week_start=week_start,
        history=_ROOT / args.history,
        models_dir=_ROOT / args.models_dir,
        data_dir=_ROOT / args.data_dir,
        out_dir=snap_dir,
        horizon_days=int(TELEGRAM_WEEKLY_HORIZON_DAYS),
        model_variant=str(args.model_variant),
    )

    picks_path = snap_dir / "picks.csv"
    picks = pd.read_csv(picks_path)

    header_lines = [
        f"WEEKLY PICKS {week_start} to {week_end}",
        f"Min odds: {TELEGRAM_MIN_ODDS:.2f}",
        "",
    ]
    body_lines = _format_weekly_lines(picks, min_odds=float(TELEGRAM_MIN_ODDS))
    messages = _split_messages(body_lines, header_lines, int(TELEGRAM_MESSAGE_MAX_CHARS))

    for msg in messages:
        if args.dry_run:
            print(msg)
            print("-" * 60)
        else:
            _send_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
