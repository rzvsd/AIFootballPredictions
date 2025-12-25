"""
Milestone 8: Narrator audit
- determinism (run twice -> same hash)
- required columns in explained output
- no raw bot keywords leak into narrative
- narrative contains key human elements (odds, model vs implied, stake tier/units)
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


EXPLAINED_COLS = {
    "pick_text",
    "title",
    "narrative",
    "confidence_label",
    "numbers_plain",
}

FORBIDDEN_SNIPPETS = [
    "ev=",
    "neff",
    "press_n",
    "xg_n",
    "sterile_flag",
    "assassin_flag",
    "p_model",
    "p_implied",
]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit narrator output for determinism and readability")
    ap.add_argument("--picks", default="reports/picks.csv", help="Input picks.csv")
    args = ap.parse_args()

    picks_path = Path(args.picks)
    if not picks_path.exists():
        raise SystemExit(f"[audit_narrator] picks not found: {picks_path}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        exp1 = td_path / "explained1.csv"
        exp2 = td_path / "explained2.csv"

        cmd = [sys.executable, "-m", "cgm.narrator", "--in", str(picks_path), "--out"]
        subprocess.run(cmd + [str(exp1)], check=True)
        subprocess.run(cmd + [str(exp2)], check=True)

        h1 = _md5(exp1)
        h2 = _md5(exp2)
        print("hash1:", h1)
        print("hash2:", h2)
        print("reproducible:", bool(h1 == h2))
        if h1 != h2:
            raise SystemExit("[audit_narrator] non-deterministic output")

        df = pd.read_csv(exp1)
        missing = [c for c in EXPLAINED_COLS if c not in df.columns]
        print("rows:", len(df), "cols:", len(df.columns), "missing_explained_cols:", missing)
        if missing:
            raise SystemExit(f"[audit_narrator] explained output missing columns: {missing}")

        if df.empty:
            print("explained: empty (ok if no picks)")
            return

        if df["narrative"].isna().any() or (df["narrative"].astype(str).str.strip() == "").any():
            raise SystemExit("[audit_narrator] missing narrative text")

        # Forbidden raw tokens must not appear
        bad_forbidden = df["narrative"].astype(str).str.contains("|".join(FORBIDDEN_SNIPPETS), case=False)
        print("forbidden_hits:", int(bad_forbidden.sum()))
        if bad_forbidden.any():
            bad = df.loc[bad_forbidden, ["fixture_datetime", "home", "away", "market", "narrative"]].head(5)
            print(bad.to_string(index=False))
            raise SystemExit("[audit_narrator] forbidden raw tokens leaked into narrative")

        # Basic presence checks
        has_odds = df["narrative"].str.contains("@", regex=False).mean()
        has_pct = df["narrative"].str.contains("%").mean()
        has_stake = df["narrative"].str.contains("Stake", case=False).mean() + df["narrative"].str.contains("size", case=False).mean()
        print("contains_odds_ratio:", has_odds, "contains_percent:", has_pct, "contains_stake_words:", has_stake)
        if has_pct == 0:
            raise SystemExit("[audit_narrator] narrative appears to lack model vs implied percentages")


if __name__ == "__main__":
    main()

