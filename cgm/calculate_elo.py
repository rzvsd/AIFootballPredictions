"""
Compute football Elo ratings from CGM match history.

Inputs (minimal):
- date/time (used only for chronological ordering)
- code_home, code_away (team IDs; falls back to names if codes missing)
- ft_home, ft_away (full-time goals)

Outputs:
- JSON mapping team_id -> latest Elo at data/enhanced/current_elo.json (default)

Constants (can be overridden by CLI):
- START_ELO = 1500
- K_FACTOR = 20
- HOME_ADV = 65
- Goal-diff multiplier per World Football Elo style.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


START_ELO_DEFAULT = 1500.0
K_FACTOR_DEFAULT = 20.0
HOME_ADV_DEFAULT = 65.0


def margin_multiplier(goal_diff: int) -> float:
    """World Football Elo style margin multiplier."""
    gd = abs(int(goal_diff))
    if gd == 0:
        return 1.0
    if gd == 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8.0  # diminishing returns


def expected_home(r_home: float, r_away: float, home_adv: float) -> float:
    """Logistic expectation with home advantage (ratings in Elo points)."""
    diff = (r_home + home_adv) - r_away
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def infer_id(row: pd.Series, home: bool) -> str | None:
    code = row["code_home"] if home else row["code_away"]
    name = row["home"] if home else row["away"]
    if pd.notna(code):
        return str(code)
    if pd.notna(name):
        return str(name)
    return None


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        dt = pd.to_datetime(df["date"] + " " + df.get("time", "").astype(str), errors="coerce")
        dt2 = pd.to_datetime(df["date"], errors="coerce")
        df["datetime"] = dt.fillna(dt2)
    df = df.sort_values("datetime")
    return df


def calculate_elo(df: pd.DataFrame, start_elo: float, k_factor: float, home_adv: float) -> Dict[str, float]:
    ratings: Dict[str, float] = {}
    for _, row in df.iterrows():
        fh = row.get("ft_home")
        fa = row.get("ft_away")
        if pd.isna(fh) or pd.isna(fa):
            continue
        home_id = infer_id(row, home=True)
        away_id = infer_id(row, home=False)
        if home_id is None or away_id is None:
            continue
        r_home = ratings.get(home_id, start_elo)
        r_away = ratings.get(away_id, start_elo)

        exp_home = expected_home(r_home, r_away, home_adv)
        if fh > fa:
            actual = 1.0
        elif fh == fa:
            actual = 0.5
        else:
            actual = 0.0

        gd = int(abs(fh - fa))
        mult = margin_multiplier(gd)
        delta = k_factor * mult * (actual - exp_home)

        ratings[home_id] = r_home + delta
        ratings[away_id] = r_away - delta
    return ratings


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Elo ratings from CGM match history")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history.csv", help="Path to match history CSV")
    ap.add_argument("--out", default="data/enhanced/current_elo.json", help="Output JSON path")
    ap.add_argument("--start-elo", type=float, default=START_ELO_DEFAULT, help="Starting Elo for new teams")
    ap.add_argument("--k-factor", type=float, default=K_FACTOR_DEFAULT, help="K factor")
    ap.add_argument("--home-adv", type=float, default=HOME_ADV_DEFAULT, help="Home advantage (points)")
    args = ap.parse_args()

    hist = load_history(Path(args.history))
    ratings = calculate_elo(hist, start_elo=args.start_elo, k_factor=args.k_factor, home_adv=args.home_adv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)
    print(f"[ok] wrote Elo ratings -> {out_path} (teams={len(ratings)})")


if __name__ == "__main__":
    main()
