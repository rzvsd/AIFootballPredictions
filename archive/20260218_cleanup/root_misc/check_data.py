import pandas as pd
from datetime import datetime

def main():
    up = pd.read_csv("CGM data/multiple leagues and seasons/upcoming.csv", encoding="latin1", low_memory=False)
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", encoding="latin1", low_memory=False)
    
    print("=== UPCOMING.CSV ===")
    print(f"Total rows: {len(up)}")
    odds_cols = [c for c in up.columns if "cota" in c.lower() or c in ["gg","ng"]]
    print(f"Columns with odds: {odds_cols}")
    print(f"Date range: {up['datameci'].min()} to {up['datameci'].max()}")
    
    print()
    print("=== ALLRATINGV.CSV ===") 
    print(f"Total rows: {len(allr)}")
    odds_cols2 = [c for c in allr.columns if "cota" in c.lower() or c in ["gg","ng"]]
    print(f"Columns with odds: {odds_cols2}")
    print(f"Date range: {allr['datameci'].min()} to {allr['datameci'].max()}")
    
    # Parse dates for future check
    def parse_date(d):
        try:
            return pd.to_datetime(d, dayfirst=True, errors="coerce")
        except:
            return pd.NaT
    
    allr["_dt"] = allr["datameci"].apply(parse_date)
    cutoff = pd.to_datetime("2026-01-11")
    future_allr = allr[allr["_dt"] > cutoff]
    print(f"\nFuture rows in allratingv.csv: {len(future_allr)}")
    
    up["_dt"] = up["datameci"].apply(parse_date)
    future_up = up[up["_dt"] > cutoff]
    print(f"Future rows in upcoming.csv: {len(future_up)}")
    
    # Check odds values in future allratingv
    if len(future_allr) > 0 and "cotao" in future_allr.columns:
        print(f"\nFuture allratingv cotao stats:")
        print(f"  Non-null: {future_allr['cotao'].notna().sum()}")
        print(f"  Non-zero: {(future_allr['cotao'] > 0).sum()}")
        print(f"  Sample values: {future_allr['cotao'].head(5).tolist()}")
    
    if "gg" in odds_cols2:
        print(f"  gg (BTTS Yes) non-null: {future_allr['gg'].notna().sum()}")
    else:
        print("  gg column NOT in allratingv.csv")
        
    if "gg" in odds_cols:
        print(f"\nUpcoming gg (BTTS Yes) stats in future rows: {future_up['gg'].notna().sum()}")
    
    # Show sample of future fixture with odds
    if len(future_allr) > 0:
        print("\n=== SAMPLE FUTURE FIXTURE FROM allratingv.csv ===")
        sample = future_allr.head(1)
        for c in ["datameci", "orameci", "txtechipa1", "txtechipa2", "codechipa1", "codechipa2", "cotaa", "cotae", "cotad", "cotao", "cotau", "gg", "ng"]:
            if c in sample.columns:
                print(f"  {c}: {sample[c].values[0]}")
            else:
                print(f"  {c}: NOT PRESENT")

if __name__ == "__main__":
    main()
