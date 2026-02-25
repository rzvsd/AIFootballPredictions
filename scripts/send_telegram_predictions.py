#!/usr/bin/env python3
"""
Send next-round predictions to Telegram (one message per league).

Source:
  reports/cgm_upcoming_predictions.csv

Message format per league:
  DATE | HOME vs AWAY
  OU: OVER/UNDER (confidence%) EV +/-x.x% | BTTS: YES/NO (confidence%) EV +/-x.x%
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Iterable, List

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from telegram_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_MESSAGE_MAX_CHARS
except ImportError as exc:
    raise SystemExit(f"telegram_config import failed: {exc}")


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _split_messages(lines: List[str], max_chars: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    for line in lines:
        candidate = current + [line]
        if len("\n\n".join(candidate)) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [line]
        else:
            current = candidate
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _send(token: str, chat_id: str, text: str) -> tuple[bool, str]:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=20)
    except Exception as exc:
        return False, f"request failed: {exc}"

    try:
        body = resp.json()
    except Exception:
        return False, f"non-json response status={resp.status_code}"

    if not body.get("ok"):
        return False, str(body.get("description", "unknown telegram error"))
    return True, ""


def _format_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")

    p_over_col = _pick_col(out, ["p_over25", "p_over_2_5"])
    p_btts_col = _pick_col(out, ["p_btts_yes"])
    ev_over_col = _pick_col(out, ["EV_over25"])
    ev_under_col = _pick_col(out, ["EV_under25"])
    ev_btts_yes_col = _pick_col(out, ["EV_btts_yes"])
    ev_btts_no_col = _pick_col(out, ["EV_btts_no"])

    required_map = {
        "p_over25/p_over_2_5": p_over_col,
        "p_btts_yes": p_btts_col,
        "EV_over25": ev_over_col,
        "EV_under25": ev_under_col,
        "EV_btts_yes": ev_btts_yes_col,
        "EV_btts_no": ev_btts_no_col,
    }
    missing = [name for name, col in required_map.items() if col is None]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    for c in [p_over_col, p_btts_col, ev_over_col, ev_under_col, ev_btts_yes_col, ev_btts_no_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["ou_lbl"] = out.get("pred_ou25", "").map({"OU25_OVER": "OVER", "OU25_UNDER": "UNDER"}).fillna(out.get("pred_ou25", ""))
    out["btts_lbl"] = out.get("pred_btts", "").map({"BTTS_YES": "YES", "BTTS_NO": "NO"}).fillna(out.get("pred_btts", ""))

    out["ou_conf_pct"] = np.where(out["ou_lbl"].eq("OVER"), out[p_over_col] * 100.0, (1.0 - out[p_over_col]) * 100.0)
    out["btts_conf_pct"] = np.where(out["btts_lbl"].eq("YES"), out[p_btts_col] * 100.0, (1.0 - out[p_btts_col]) * 100.0)
    out["ou_ev_pct"] = np.where(out["ou_lbl"].eq("OVER"), out[ev_over_col] * 100.0, out[ev_under_col] * 100.0)
    out["btts_ev_pct"] = np.where(out["btts_lbl"].eq("YES"), out[ev_btts_yes_col] * 100.0, out[ev_btts_no_col] * 100.0)

    return out.sort_values(["league", "date", "home", "away"], kind="mergesort")


def _render_league_message(df_league: pd.DataFrame, league: str) -> List[str]:
    header = [f"{league} - Next Round ({len(df_league)} games)"]
    body: List[str] = []
    for _, r in df_league.iterrows():
        ou_ev = "N/A" if pd.isna(r["ou_ev_pct"]) else f"{r['ou_ev_pct']:+.1f}%"
        bt_ev = "N/A" if pd.isna(r["btts_ev_pct"]) else f"{r['btts_ev_pct']:+.1f}%"
        body.append(
            f"{r['date']} | {r['home']} vs {r['away']}\n"
            f"OU: {r['ou_lbl']} ({r['ou_conf_pct']:.0f}%) EV {ou_ev} | "
            f"BTTS: {r['btts_lbl']} ({r['btts_conf_pct']:.0f}%) EV {bt_ev}"
        )
    return header + body


def main() -> int:
    parser = argparse.ArgumentParser(description="Send next-round predictions to Telegram (one message per league)")
    parser.add_argument("--predictions", default="reports/cgm_upcoming_predictions.csv", help="Predictions CSV path")
    parser.add_argument("--league", action="append", default=[], help="Optional league filter (repeatable)")
    parser.add_argument("--max-chars", type=int, default=int(TELEGRAM_MESSAGE_MAX_CHARS or 3500), help="Telegram message max chars")
    parser.add_argument("--dry-run", action="store_true", help="Print messages instead of sending")
    parser.add_argument("--sleep-sec", type=float, default=0.35, help="Delay between sends")
    parser.add_argument("--token", default="", help="Override Telegram bot token")
    parser.add_argument("--chat-id", default="", help="Override Telegram chat id")
    args = parser.parse_args()

    token = args.token.strip() or TELEGRAM_BOT_TOKEN
    chat_id = args.chat_id.strip() or str(TELEGRAM_CHAT_ID)

    if not args.dry_run and (not token or not chat_id):
        print("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing. Configure .env or pass --token --chat-id.")
        return 1

    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path
    if not pred_path.exists():
        print(f"ERROR: predictions file not found: {pred_path}")
        return 1

    try:
        df = pd.read_csv(pred_path)
    except Exception as exc:
        print(f"ERROR: failed to read predictions: {exc}")
        return 1

    try:
        df = _format_predictions(df)
    except Exception as exc:
        print(f"ERROR: invalid predictions schema: {exc}")
        return 1

    if args.league:
        allowed = set(args.league)
        df = df[df["league"].isin(allowed)].copy()

    if df.empty:
        print("No predictions to send after filters.")
        return 0

    sent = 0
    failed = 0

    for league, df_league in df.groupby("league", sort=True):
        blocks = _render_league_message(df_league, str(league))
        chunks = _split_messages(blocks, max_chars=max(500, int(args.max_chars)))
        for idx, msg in enumerate(chunks, start=1):
            if args.dry_run:
                if len(chunks) > 1:
                    print(f"[DRY] {league} part {idx}/{len(chunks)}")
                else:
                    print(f"[DRY] {league}")
                print(msg)
                print("-" * 60)
                sent += 1
            else:
                ok, err = _send(token, chat_id, msg)
                if ok:
                    print(f"SENT: {league} part {idx}/{len(chunks)}")
                    sent += 1
                else:
                    print(f"FAILED: {league} part {idx}/{len(chunks)} -> {err}")
                    failed += 1
                time.sleep(max(0.0, float(args.sleep_sec)))

    print(f"DONE sent={sent} failed={failed}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
