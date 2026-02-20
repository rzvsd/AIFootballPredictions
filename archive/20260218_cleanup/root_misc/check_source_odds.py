import pandas as pd
from pathlib import Path

def parse_dt(d):
    try:
        return pd.to_datetime(d, dayfirst=True, errors="coerce")
    except:
        return pd.NaT

def main():
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", encoding="latin1", low_memory=False)
    
    allr["_dt"] = allr["datameci"].apply(parse_dt)
    cutoff = pd.to_datetime("2026-01-11")
    future = allr[allr["_dt"] > cutoff].copy()
    
    print(f"Total future fixtures in allratingv.csv: {len(future)}")
    
    # Check odds presence in source
    if "cotao" in future.columns:
        has_odds = future[future["cotao"].notna() & (future["cotao"] > 0)]
        no_odds = future[future["cotao"].isna() | (future["cotao"] == 0)]
        print(f"With O/U odds (cotao > 0): {len(has_odds)}")
        print(f"Without O/U odds (cotao = 0 or NaN): {len(no_odds)}")
        
        # Check by league
        if "league" in future.columns:
            print("\n=== BY LEAGUE - ALL FUTURE ===")
            print(future["league"].value_counts().head(15).to_string())
            
            print("\n=== BY LEAGUE - WITH ODDS ===")
            print(has_odds["league"].value_counts().head(15).to_string())
            
            print("\n=== BY LEAGUE - WITHOUT ODDS ===")
            print(no_odds["league"].value_counts().head(15).to_string())
    else:
        print("cotao column not found")

if __name__ == "__main__":
    main()
