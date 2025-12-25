from __future__ import annotations

import os
import csv
import json
from pathlib import Path


class BankrollManager:
    """Simple bankroll tracker and logger for bets."""

    def __init__(self, bankroll_file: str = "data/bankroll.json", log_file: str = "data/bets_log.csv"):
        self.bankroll_path = Path(bankroll_file)
        self.log_path = Path(log_file)
        self.bankroll = self._load_bankroll()
        print(f"Bankroll initialized at: {self.bankroll:.2f}")

    def _load_bankroll(self) -> float:
        self.bankroll_path.parent.mkdir(parents=True, exist_ok=True)
        if self.bankroll_path.exists():
            with open(self.bankroll_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("bankroll", 1000.0))  # Default to 1000 if not set
        return 1000.0

    def _save_bankroll(self) -> None:
        with open(self.bankroll_path, "w", encoding="utf-8") as f:
            json.dump({"bankroll": self.bankroll}, f)

    def _has_overdue_unsettled(self) -> bool:
        """Return True if there are unsettled bets older than the guard window."""
        try:
            if not self.log_path.exists():
                return False
            import pandas as pd  # local import

            df = pd.read_csv(self.log_path)
            if df.empty:
                return False
            if "result" not in df.columns:
                return False
            unsettled = df[df["result"].isna() | (df["result"].astype(str).str.strip() == "")]
            if unsettled.empty:
                return False
            days_guard = float(os.getenv("BOT_UNSETTLED_GUARD_DAYS", "2"))
            if days_guard <= 0:
                return True
            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_guard)
            if "date" not in unsettled.columns:
                return False
            unsettled["date_dt"] = pd.to_datetime(unsettled["date"], errors="coerce")
            overdue = unsettled[unsettled["date_dt"] < cutoff]
            return not overdue.empty
        except Exception:
            # Fail-safe: if we cannot determine, block to avoid drain
            return True

    def log_bet(
        self,
        date: str,
        league: str,
        home: str,
        away: str,
        market: str,
        selection: str,
        odds: float,
        stake: float,
        prob: float,
        ev: float,
    ):
        """Deduct stake and log the placed bet."""
        if self._has_overdue_unsettled():
            print(
                "Warning: Unsettled bets older than guard window; skipping new bet to avoid bankroll drift. Run settle_bets.py."
            )
            return
        if stake > self.bankroll:
            print(f"Warning: Insufficient bankroll ({self.bankroll:.2f}) for stake ({stake:.2f}). Skipping bet.")
            return
        self.bankroll -= stake
        self._save_bankroll()
        self._append_log(date, league, home, away, market, selection, odds, stake, prob, ev)
        print(f"Bet Logged. New bankroll: {self.bankroll:.2f}")

    def _append_log(self, date, league, home, away, market, selection, odds, stake, prob, ev):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists()
        cols = [
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
            "result",
            "actual_score",
            "payout",
            "net_profit",
            "settled_at",
            "notes",
        ]
        if file_exists:
            try:
                import pandas as pd  # local import

                df = pd.read_csv(self.log_path)
                missing = [c for c in cols if c not in df.columns]
                if missing:
                    for c in missing:
                        df[c] = "" if c not in {"payout", "net_profit"} else 0.0
                    df.to_csv(self.log_path, index=False)
            except Exception:
                pass
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(cols)
            writer.writerow(
                [
                    date,
                    league,
                    home,
                    away,
                    market,
                    selection,
                    odds,
                    stake,
                    f"{prob:.4f}",
                    f"{ev:.4f}",
                    "",
                    "",
                    0.0,
                    0.0,
                    "",
                    "",
                ]
            )
