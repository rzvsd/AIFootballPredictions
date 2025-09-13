"""
Ingest Understat shots JSON files and produce a flat CSV with engineered features.

Usage:
  python -m scripts.shots_ingest_understat --inputs data/understat/*_shots.json \
      --out data/shots/understat_shots.csv

If --inputs is omitted, the script will look for data/understat/*_shots.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd

from xg_shot_model.features import (
    shot_distance_m,
    shot_open_angle_deg,
    is_header,
)


KEEP_COLS = [
    "match_id", "date", "h_team", "a_team", "team", "h_a", "minute",
    "player", "shot_id", "result", "situation", "shotType", "xG",
    "X", "Y", "fast_break", "isKeyPass",
]


def read_shots(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows.extend(data)
        except Exception:
            continue
    return rows


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize numeric coords
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    df["X"] = df["X"].apply(to_float)
    df["Y"] = df["Y"].apply(to_float)
    # Distance & angle
    df["dist_m"] = df.apply(lambda r: shot_distance_m(r["X"], r["Y"]) if pd.notna(r["X"]) and pd.notna(r["Y"]) else None, axis=1)
    df["angle_deg"] = df.apply(lambda r: shot_open_angle_deg(r["X"], r["Y"]) if pd.notna(r["X"]) and pd.notna(r["Y"]) else None, axis=1)
    # Header flag
    df["is_header"] = df.apply(lambda r: is_header(str(r.get("shotType")), None), axis=1)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Understat shots and write engineered CSV")
    ap.add_argument("--inputs", nargs="*", default=[], help="Glob(s) for *_shots.json files")
    ap.add_argument("--out", default=os.path.join("data", "shots", "understat_shots.csv"))
    args = ap.parse_args()

    paths: List[str] = []
    if args.inputs:
        for g in args.inputs:
            paths.extend(glob.glob(g))
    else:
        paths = glob.glob(os.path.join("data", "understat", "*_shots.json"))
    if not paths:
        print("No *_shots.json files found. Run scripts.fetch_understat_simple first.")
        return

    shots = read_shots(paths)
    if not shots:
        print("No shots in inputs.")
        return
    df = pd.DataFrame(shots)
    # Keep consistent set of columns
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[KEEP_COLS].copy()
    df = compute_features(df)
    # Ensure output dir
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    df.sort_values(["match_id", "minute", "team"], inplace=True)
    df.to_csv(args.out, index=False)
    print(f"Saved shots CSV -> {args.out}  (rows: {len(df)})")


if __name__ == "__main__":
    main()

