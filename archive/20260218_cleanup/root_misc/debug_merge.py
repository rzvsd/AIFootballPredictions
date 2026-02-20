
import pandas as pd
import datetime as dt
from pathlib import Path

def _parse_upcoming_datetime(d_str, t_str):
    try:
        s = f"{d_str} {t_str}".strip()
        return pd.to_datetime(s, dayfirst=True, errors='coerce')
    except:
        return pd.NaT

def main():
    up_path = "CGM data/multiple leagues and seasons/upcoming.csv"
    odds_path = "CGM data/multiple leagues and seasons/allratingv.csv"

    print(f"Loading {up_path}...")
    up = pd.read_csv(up_path, encoding="latin1", low_memory=False)
    print(f"Loading {odds_path}...")
    up_odds = pd.read_csv(odds_path, encoding="latin1", low_memory=False)

    print("Parsing dates...")
    up["_fixture_dt"] = up.apply(lambda r: _parse_upcoming_datetime(r.get("datameci"), r.get("orameci")), axis=1)
    up_odds["_fixture_dt"] = up_odds.apply(lambda r: _parse_upcoming_datetime(r.get("datameci"), r.get("orameci")), axis=1)

    cutoff = pd.to_datetime("2026-01-11")
    up = up[up["_fixture_dt"] > cutoff].copy()
    up_odds = up_odds[up_odds["_fixture_dt"] > cutoff].copy()

    print(f"Future rows: UP={len(up)}, ODDS={len(up_odds)}")

    up["_date_only"] = up["_fixture_dt"].dt.date
    up_odds["_date_only"] = up_odds["_fixture_dt"].dt.date

    # Normalize teams (simplified)
    for df in [up, up_odds]:
        for c in ["txtechipa1", "txtechipa2"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

    merge_keys = ["_date_only", "txtechipa1", "txtechipa2"]
    if "codechipa1" in up.columns and "codechipa1" in up_odds.columns:
         print("Using code keys (codechipa1 present in both)")
         merge_keys = ["_date_only", "codechipa1", "codechipa2"]
    else:
         print(f"Using name keys (codes missing: UP={ 'codechipa1' in up.columns}, ODDS={ 'codechipa1' in up_odds.columns})")

    # Sample rows for inspection before merge
    print("UP Sample Keys:")
    print(up[merge_keys].head(2).to_string())
    print("ODDS Sample Keys:")
    print(up_odds[merge_keys].head(2).to_string())

    print(f"Merging on {merge_keys}...")
    merged = up.merge(up_odds, on=merge_keys, how="inner", suffixes=("", "_odds"))
    print(f"Merged Inner rows: {len(merged)}")
    
    if len(merged) == 0:
        print("Merge failed. Sample keys:")
        print("UP sample:", up[merge_keys].head(2).to_dict(orient="records"))
        print("ODDS sample:", up_odds[merge_keys].head(2).to_dict(orient="records"))

    # Check for cotao
    if len(merged) > 0:
        if "cotao" in merged.columns:
             print("cotao from UP present")
        if "cotao_odds" in merged.columns:
             print("cotao_odds from ODDS present:", merged["cotao_odds"].head().tolist())
        elif "cotao" in up_odds.columns:
             print("cotao in ODDS but not merged?!")

if __name__ == "__main__":
    main()
