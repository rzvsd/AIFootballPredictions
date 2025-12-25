"""
Absences MVP importer: build a simple team availability index CSV per league.

Input: a CSV with columns: team, availability_index [, date]
 - availability_index in [0,1], where 1.0 = full squad, 0.5 = many absences

Output: data/absences/{LEAGUE}_availability.csv merged/updated by team (keeps latest by date)

Usage:
  python -m scripts.absences_import --league E0 --input my_absences.csv
"""

from __future__ import annotations

import argparse
import os
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description='Import absences availability index per league')
    ap.add_argument('--league', required=True)
    ap.add_argument('--input', required=True, help='CSV with columns: team, availability_index [, date]')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if 'team' not in df.columns or 'availability_index' not in df.columns:
        raise SystemExit('Input must contain columns: team, availability_index')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    out_dir = os.path.join('data','absences')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.league}_availability.csv")

    if os.path.exists(out_path):
        cur = pd.read_csv(out_path)
        if 'date' in cur.columns:
            cur['date'] = pd.to_datetime(cur['date'], errors='coerce')
        merged = pd.concat([cur, df], ignore_index=True)
        # keep latest per (team, date)
        if 'date' in merged.columns:
            merged = merged.sort_values(['team','date']).groupby('team', as_index=False).tail(1)
        else:
            merged = merged.groupby('team', as_index=False).tail(1)
    else:
        merged = df
    merged.to_csv(out_path, index=False)
    print(f"Saved availability -> {out_path}  (rows={len(merged)})")


if __name__ == '__main__':
    main()

