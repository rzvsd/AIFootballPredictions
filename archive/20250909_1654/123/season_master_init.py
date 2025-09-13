# season_master_init.py
# Build or update data/fixtures_master/{LEAGUE}_{SEASON}.csv from weekly CSV(s).
# Usage:
#   python season_master_init.py --league E0 --season 2025 --merge data/fixtures/E0_2025-08-15_to_2025-08-18.csv
#   python season_master_init.py --league E0 --season 2025 --merge data/fixtures/*.csv

import os, glob, argparse
import pandas as pd

MASTER = os.path.join("data","fixtures_master","{league}_{season}.csv")

def load_csv(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    alias = {"kickoff_utc":"date","kickoff":"date","home":"home_team","away":"away_team"}
    df.rename(columns={k:v for k,v in alias.items() if k in df.columns}, inplace=True)
    need = {"date","home_team","away_team"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing headers {need}")
    for c in need: df[c] = df[c].astype(str).str.strip()
    return df[["date","home_team","away_team"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--season", required=True)
    ap.add_argument("--merge", nargs="+", required=True, help="One or more weekly CSVs")
    args = ap.parse_args()

    os.makedirs(os.path.join("data","fixtures_master"), exist_ok=True)
    master_path = MASTER.format(league=args.league, season=args.season)

    frames=[]
    for pattern in args.merge:
        for p in glob.glob(pattern):
            try:
                frames.append(load_csv(p))
                print(f"[add] {p}")
            except Exception as e:
                print(f"[skip] {p}: {e}")

    if not frames:
        print("[warn] no valid weekly files matched.")
        return

    merged = pd.concat(frames, ignore_index=True).drop_duplicates().sort_values("date")
    if os.path.exists(master_path):
        base = load_csv(master_path)
        merged = pd.concat([base, merged], ignore_index=True).drop_duplicates().sort_values("date")

    merged.to_csv(master_path, index=False)
    print(f"[master] wrote {master_path} (rows={len(merged)})")

if __name__ == "__main__":
    main()
