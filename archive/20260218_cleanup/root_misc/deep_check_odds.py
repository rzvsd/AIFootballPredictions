import pandas as pd
from pathlib import Path

def main():
    """Deep investigation of allratingv.csv odds data"""
    
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", 
                       encoding="latin1", low_memory=False)
    
    print(f"Total rows in allratingv.csv: {len(allr)}")
    print(f"Columns: {list(allr.columns)}")
    
    # Check cotao column type and values
    if "cotao" in allr.columns:
        print(f"\n=== COTAO COLUMN ANALYSIS ===")
        print(f"dtype: {allr['cotao'].dtype}")
        print(f"Total: {len(allr)}")
        print(f"Non-null: {allr['cotao'].notna().sum()}")
        print(f"Unique values (sample): {allr['cotao'].dropna().unique()[:20].tolist()}")
        
        # Try numeric conversion
        numeric_cotao = pd.to_numeric(allr["cotao"], errors="coerce")
        print(f"After numeric conversion - non-null: {numeric_cotao.notna().sum()}")
        print(f"After numeric conversion - > 0: {(numeric_cotao > 0).sum()}")
        print(f"After numeric conversion - == 0: {(numeric_cotao == 0).sum()}")
        
        # Check for date filtering issue
        print("\n=== CHECKING FUTURE FIXTURES ===")
        
        # Parse dates
        def parse_dt(d):
            try:
                return pd.to_datetime(d, dayfirst=True, errors="coerce")
            except:
                return pd.NaT
        
        allr["_dt"] = allr["datameci"].apply(parse_dt)
        cutoff = pd.to_datetime("2026-01-11")
        future = allr[allr["_dt"] > cutoff].copy()
        
        print(f"Future fixtures (after 2026-01-11): {len(future)}")
        
        # Check cotao in future
        future_cotao = pd.to_numeric(future["cotao"], errors="coerce")
        print(f"Future with cotao > 0: {(future_cotao > 0).sum()}")
        print(f"Future with cotao == 0: {(future_cotao == 0).sum()}")
        print(f"Future with cotao NaN: {future_cotao.isna().sum()}")
        
        # Sample of future fixtures
        print("\n=== SAMPLE FUTURE FIXTURES (raw values) ===")
        sample_cols = ["datameci", "orameci", "txtechipa1", "txtechipa2", "cotao", "cotau"]
        sample_cols = [c for c in sample_cols if c in future.columns]
        
        # Get some rows that SHOULD have odds
        sample = future[sample_cols].head(10)
        print(sample.to_string())
        
        # Check if cotao has non-numeric characters
        print("\n=== CHECKING FOR NON-NUMERIC VALUES IN COTAO ===")
        str_cotao = future["cotao"].astype(str)
        non_numeric = str_cotao[~str_cotao.str.match(r'^[\d\.]+$', na=False) & str_cotao.notna() & (str_cotao != 'nan')]
        print(f"Non-numeric cotao values (sample): {non_numeric.head(10).tolist()}")
        
        # Check date range of future fixtures
        print(f"\n=== DATE RANGE OF FUTURE ===")
        print(f"Min date: {future['_dt'].min()}")
        print(f"Max date: {future['_dt'].max()}")
        
        # Check fixtures by specific date e.g. Jan 12
        jan12 = future[future["_dt"].dt.date == pd.to_datetime("2026-01-12").date()]
        print(f"\nFixtures on Jan 12: {len(jan12)}")
        if len(jan12) > 0:
            jan12_cotao = pd.to_numeric(jan12["cotao"], errors="coerce")
            print(f"  With cotao > 0: {(jan12_cotao > 0).sum()}")
            print(f"  Sample:")
            print(jan12[sample_cols].head(5).to_string())
    else:
        print("cotao column not found!")
        print(f"Available columns containing 'cota': {[c for c in allr.columns if 'cota' in c.lower()]}")

if __name__ == "__main__":
    main()
