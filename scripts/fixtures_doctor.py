"""
Validate fixtures CSV inputs before fetching odds or generating reports.

Checks per LEAGUE=path:
- Path exists and is a file (not a directory)
- Columns: date, home/home_team, away/away_team present
- Dates parse correctly; flags invalid rows
- Team names: normalize via config; warn on names not in known canon set; suggest fuzzy matches
- Duplicates: same (date, home, away)

Exit code: 0 on success (only warnings allowed), 1 on fatal issues.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import difflib
import pandas as pd

import config


def _colmap(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = {c.strip().lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("matchdate") or cols.get("kickoff") or cols.get("ko")
    home_col = cols.get("home") or cols.get("home_team") or cols.get("home team") or cols.get("hometeam")
    away_col = cols.get("away") or cols.get("away_team") or cols.get("away team") or cols.get("awayteam")
    return date_col, home_col, away_col


def _known_team_set() -> set:
    # Build a set from mapping keys and values to increase coverage
    s = set()
    try:
        s.update(config.TEAM_NAME_MAP.keys())
        s.update(config.TEAM_NAME_MAP.values())
    except Exception:
        pass
    return {str(x) for x in s}


def validate_pair(league: str, path: str) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True
    if not path or not isinstance(path, str):
        return False, [f"[{league}] Missing path."]
    if not os.path.exists(path):
        return False, [f"[{league}] Fixtures CSV not found: {path}"]
    if not os.path.isfile(path):
        return False, [f"[{league}] Path is not a file (got directory?): {path}"]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, [f"[{league}] Could not read CSV '{path}': {e}"]

    date_col, home_col, away_col = _colmap(df)
    missing = [name for name, c in [("date", date_col), ("home", home_col), ("away", away_col)] if not c]
    if missing:
        return False, [f"[{league}] Missing columns: {', '.join(missing)} in {path}"]

    # Date parsing
    dd = pd.to_datetime(df[date_col], errors="coerce")
    bad_dates = dd.isna().sum()
    if bad_dates > 0:
        ok = False
        msgs.append(f"[{league}] Invalid date values: {bad_dates} row(s) in {path}")

    # Duplicates
    key_df = pd.DataFrame({
        "_d": dd.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "_h": df[home_col].astype(str).str.strip(),
        "_a": df[away_col].astype(str).str.strip(),
    })
    dups = key_df.duplicated().sum()
    if dups > 0:
        ok = False
        msgs.append(f"[{league}] Duplicate fixtures found: {dups} row(s) in {path}")

    # Team names sanity
    known = _known_team_set()
    warn_unknown = []
    for col_name in (home_col, away_col):
        for name in df[col_name].astype(str).str.strip().unique():
            if not name:
                continue
            nrm = config.normalize_team_name(name)
            if (nrm not in known) and (name not in known):
                # Suggest closest known
                cand = difflib.get_close_matches(name, list(known), n=1, cutoff=0.8)
                if cand:
                    warn_unknown.append(f"'{name}' -> maybe '{cand[0]}'")
                else:
                    warn_unknown.append(f"'{name}' (no close match)")
    if warn_unknown:
        msgs.append(f"[{league}] Unknown team names (check normalization): " + ", ".join(sorted(set(warn_unknown))[:10]))

    return ok, msgs


def parse_pairs(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Invalid fixtures-csv entry (expected LEAGUE=path): {p}")
        lg, path = p.split("=", 1)
        out[lg.strip()] = path.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate fixtures CSV inputs per league")
    ap.add_argument('--fixtures-csv', nargs='*', default=[], help='Pairs LEAGUE=path.csv')
    args = ap.parse_args()

    if not args.fixtures_csv:
        print("[fixtures-doctor] No fixtures provided (nothing to validate).")
        sys.exit(0)

    try:
        mapping = parse_pairs(args.fixtures_csv)
    except Exception as e:
        print(f"[fixtures-doctor] {e}")
        sys.exit(1)

    fatal = False
    for lg, path in mapping.items():
        ok, msgs = validate_pair(lg, path)
        for m in msgs:
            print(m)
        if not ok:
            fatal = True
        else:
            print(f"[{lg}] OK: {path}")

    if fatal:
        print("[fixtures-doctor] Validation failed. Please fix the above issues.")
        sys.exit(1)
    else:
        print("[fixtures-doctor] All fixtures validated.")
        sys.exit(0)


if __name__ == '__main__':
    main()

