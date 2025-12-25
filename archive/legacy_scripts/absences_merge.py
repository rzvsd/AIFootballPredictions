"""
Merge multiple absences sources into a single per-team availability file with optional positional weighting.

Inputs: one or more CSVs via --input, each may contain:
  - team, availability_index [, date]
  - or positional: team, availability_gk, availability_def, availability_mid, availability_fwd [, date]

Output: data/absences/{LEAGUE}_availability.csv

Usage:
  python -m scripts.absences_merge --league E0 --input src1.csv src2.csv --out data/absences/E0_availability.csv
"""

from __future__ import annotations

import argparse
import os
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description='Merge absences sources for a league')
    ap.add_argument('--league', required=True)
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    frames = []
    for p in args.input:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        frames.append(df)
    if not frames:
        raise SystemExit('No valid input files found.')
    merged = pd.concat(frames, ignore_index=True)
    # Reduce by team (keep latest date if present), average across sources for overlapping cols
    # If date exists, sort by date then groupby team tail(1)
    cols_pos = ['availability_gk','availability_def','availability_mid','availability_fwd']
    has_pos = set(cols_pos).issubset(merged.columns)
    if 'date' in merged.columns:
        merged = merged.sort_values(['team','date'])
        last = merged.groupby('team', as_index=False).tail(1)
    else:
        last = merged.copy()
    if has_pos:
        out = last[['team'] + cols_pos].copy()
        for c in cols_pos:
            out[c] = pd.to_numeric(out[c], errors='coerce').clip(0.0, 1.0)
    else:
        out = last[['team','availability_index']].copy()
        out['availability_index'] = pd.to_numeric(out['availability_index'], errors='coerce').clip(0.0, 1.0)

    odir = os.path.join('data','absences'); os.makedirs(odir, exist_ok=True)
    out_path = args.out or os.path.join(odir, f"{args.league}_availability.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved merged absences -> {out_path}  (rows={len(out)})")


if __name__ == '__main__':
    main()

