"""
Settle logged bets once the real match results are known.

Usage:
    python -m scripts.settle_bets [--log data/bets_log.csv]
                                  [--bankroll data/bankroll.json]
                                  [--results-dir data/processed]
                                  [--dry-run]

The script inspects data/bets_log.csv, matches every unsettled bet against the
processed match results (data/processed/{league}_merged_preprocessed.csv), and
then:
  • marks bets as WIN/LOSS/PUSH along with the actual score,
  • credits bankroll.json with payouts for wins (stake * odds) and pushes
    (stake refund),
  • writes the updated log back to disk.

This keeps bankroll.json in sync with the real-world outcomes and replaces the
previous one-way bankroll deduction behaviour.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

import config

BASE_COLS = [
    "date",
    "league",
    "home_team",
    "away_team",
    "market",
    "selection",
    "odds",
    "stake",
    "model_prob",
    "expected_value",
]

EXTRA_COLS = [
    "result",
    "actual_score",
    "payout",
    "net_profit",
    "settled_at",
    "notes",
]


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _load_results(league: str, results_dir: Path) -> pd.DataFrame:
    path = results_dir / f"{league}_merged_preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed file missing for league {league}: {path}")
    df = pd.read_csv(path)
    if "Date" not in df.columns or "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        raise ValueError(f"{path} does not contain Date/HomeTeam/AwayTeam columns")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["HomeTeamNorm"] = df["HomeTeam"].astype(str).map(config.normalize_team_name)
    df["AwayTeamNorm"] = df["AwayTeam"].astype(str).map(config.normalize_team_name)
    key = (
        df["Date"].dt.date.astype(str)
        + "|"
        + df["HomeTeamNorm"]
        + "|"
        + df["AwayTeamNorm"]
    )
    df["_lookup_key"] = key
    return df


def _match_row(row: pd.Series, cache: Dict[str, pd.DataFrame], results_dir: Path) -> Optional[pd.Series]:
    league = str(row.get("league", "")).strip().upper()
    if not league:
        return None
    if league not in cache:
        cache[league] = _load_results(league, results_dir)
    df = cache[league]
    date_val = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.isna(date_val):
        return None
    home = config.normalize_team_name(str(row.get("home_team", "")))
    away = config.normalize_team_name(str(row.get("away_team", "")))
    key = f"{date_val.date()}|{home}|{away}"
    matches = df[df["_lookup_key"] == key]
    if matches.empty:
        return None
    # In case multiple entries exist (rare double headers), take the first chronologically.
    return matches.sort_values("Date").iloc[0]


def _eval_1x2(selection: str, home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        actual = "H"
    elif home_goals < away_goals:
        actual = "A"
    else:
        actual = "D"
    sel = selection.strip().upper()
    if sel in {"1", "H"}:
        sel = "H"
    elif sel in {"2", "A"}:
        sel = "A"
    elif sel in {"X", "D"}:
        sel = "D"
    return "WIN" if sel == actual else "LOSS"


def _eval_dc(selection: str, home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        actual = "1"
    elif home_goals < away_goals:
        actual = "2"
    else:
        actual = "X"
    sel = selection.strip().upper()
    return "WIN" if actual in sel else "LOSS"


def _eval_ou(market: str, selection: str, total_goals: float) -> str:
    try:
        _, line_str = market.split(" ", 1)
        line = float(line_str.strip())
    except ValueError:
        return "UNKNOWN"
    sel = selection.strip().upper()
    if total_goals > line:
        actual = "OVER"
    elif total_goals < line:
        actual = "UNDER"
    else:
        return "PUSH"
    return "WIN" if sel.startswith(actual[:1]) else "LOSS"


def _eval_interval(selection: str, total_goals: float) -> str:
    sel = selection.strip()
    if not sel:
        return "UNKNOWN"
    if "+" in sel:
        try:
            low = float(sel.replace("+", "").strip())
        except ValueError:
            return "UNKNOWN"
        return "WIN" if total_goals >= low else "LOSS"
    if "-" in sel:
        try:
            low_str, high_str = sel.split("-", 1)
            low = float(low_str.strip())
            high = float(high_str.strip())
        except ValueError:
            return "UNKNOWN"
        return "WIN" if low <= total_goals <= high else "LOSS"
    # Exact total
    try:
        exact = float(sel)
    except ValueError:
        return "UNKNOWN"
    if total_goals == exact:
        return "WIN"
    return "LOSS"


def _settle_market(row: pd.Series, match: pd.Series) -> Tuple[str, str]:
    """Return (result, note)."""
    if match is None:
        return "PENDING", "No result available"
    fthg = _safe_float(match.get("FTHG"), -1)
    ftag = _safe_float(match.get("FTAG"), -1)
    if fthg < 0 or ftag < 0:
        return "PENDING", "Goals missing"
    total = fthg + ftag
    market = str(row.get("market", "")).strip()
    selection = str(row.get("selection", "")).strip()
    if not market:
        return "PENDING", "Market missing"
    if market == "1X2":
        return _eval_1x2(selection, fthg, ftag), ""
    if market == "DC":
        return _eval_dc(selection, fthg, ftag), ""
    if market.startswith("OU "):
        return _eval_ou(market, selection, total), ""
    if market == "TG Interval":
        return _eval_interval(selection, total), ""
    return "PENDING", f"Unsupported market '{market}'"


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = ""
    for col in EXTRA_COLS:
        if col not in df.columns:
            df[col] = "" if col not in {"payout", "net_profit"} else 0.0
    ordered = BASE_COLS + [c for c in EXTRA_COLS if c in df.columns]
    return df[ordered]


def _credit_bankroll(bankroll_path: Path, amount: float) -> float:
    bankroll_path.parent.mkdir(parents=True, exist_ok=True)
    if bankroll_path.exists():
        with open(bankroll_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    else:
        data = {}
    balance = float(data.get("bankroll", 1000.0))
    balance += amount
    with open(bankroll_path, "w", encoding="utf-8") as f:
        json.dump({"bankroll": balance}, f)
    return balance


def settle_bets(log_path: Path, bankroll_path: Path, results_dir: Path, dry_run: bool = False) -> None:
    if not log_path.exists():
        print(f"No bet log found at {log_path}. Nothing to settle.")
        return
    df = pd.read_csv(log_path)
    df = _ensure_columns(df)
    cache: Dict[str, pd.DataFrame] = {}
    settled_now = 0
    bankroll_credit = 0.0
    utc_now = (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    for idx, row in df.iterrows():
        current_result = str(row.get("result", "")).upper()
        if current_result in {"WIN", "LOSS", "PUSH"}:
            continue  # already settled in a previous run
        match = _match_row(row, cache, results_dir)
        result, note = _settle_market(row, match)
        if result not in {"WIN", "LOSS", "PUSH"}:
            df.at[idx, "notes"] = note
            continue
        settled_now += 1
        payout = 0.0
        stake = _safe_float(row.get("stake"), 0.0)
        odds = _safe_float(row.get("odds"), 0.0)
        if result == "WIN":
            payout = stake * odds
        elif result == "PUSH":
            payout = stake
        net = payout - stake
        bankroll_credit += payout

        score = ""
        if match is not None:
            score = f"{int(_safe_float(match.get('FTHG')))}-{int(_safe_float(match.get('FTAG')))}"
        df.at[idx, "result"] = result
        df.at[idx, "actual_score"] = score
        df.at[idx, "payout"] = round(payout, 2)
        df.at[idx, "net_profit"] = round(net, 2)
        df.at[idx, "settled_at"] = utc_now
        df.at[idx, "notes"] = note

    if settled_now == 0:
        print("No bets settled this run.")
        return

    if dry_run:
        print(f"[Dry-Run] Would credit bankroll with {bankroll_credit:.2f} for {settled_now} bets.")
        return

    df.to_csv(log_path, index=False)
    new_balance = _credit_bankroll(bankroll_path, bankroll_credit)
    print(f"Settled {settled_now} bet(s). Credited {bankroll_credit:.2f} to bankroll.")
    print(f"Updated bankroll balance: {new_balance:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Settle logged bets once match results are known.")
    parser.add_argument("--log", default="data/bets_log.csv", help="Path to bets log CSV.")
    parser.add_argument("--bankroll", default="data/bankroll.json", help="Path to bankroll JSON.")
    parser.add_argument(
        "--results-dir",
        default="data/processed",
        help="Directory containing *_merged_preprocessed.csv files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes; just report planned actions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settle_bets(
        log_path=Path(args.log),
        bankroll_path=Path(args.bankroll),
        results_dir=Path(args.results_dir),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
