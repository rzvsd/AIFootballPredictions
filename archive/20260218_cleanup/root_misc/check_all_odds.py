import pandas as pd

def main():
    """Check all cota* columns in future fixtures"""
    
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", 
                       encoding="latin1", low_memory=False)
    
    # Get all cota columns
    cota_cols = [c for c in allr.columns if "cota" in c.lower()]
    print(f"All cota columns: {cota_cols}")
    
    # Parse dates
    allr["_dt"] = pd.to_datetime(allr["datameci"], dayfirst=False, errors="coerce")
    cutoff = pd.to_datetime("2026-01-11")
    future = allr[allr["_dt"] > cutoff].copy()
    
    print(f"\nFuture fixtures: {len(future)}")
    
    # Check each cota column
    print("\n=== ODDS COLUMNS IN FUTURE FIXTURES ===")
    for col in cota_cols:
        future[f"_{col}_num"] = pd.to_numeric(future[col], errors="coerce")
        valid = (future[f"_{col}_num"] > 0).sum()
        zero = (future[f"_{col}_num"] == 0).sum()
        na = future[f"_{col}_num"].isna().sum()
        print(f"{col}: valid={valid}, zero={zero}, nan={na}")
    
    # Sample raw data for a fixture WITH odds
    print("\n=== SAMPLE FIXTURE WITH ODDS ===")
    valid_rows = future[pd.to_numeric(future["cotao"], errors="coerce") > 0]
    if len(valid_rows) > 0:
        sample = valid_rows.iloc[0]
        for col in ["datameci", "orameci", "txtechipa1", "txtechipa2"] + cota_cols:
            print(f"  {col}: {sample.get(col, 'N/A')}")
    
    # Sample raw data for a fixture WITHOUT odds
    print("\n=== SAMPLE FIXTURE WITHOUT ODDS ===")
    zero_rows = future[pd.to_numeric(future["cotao"], errors="coerce") == 0]
    if len(zero_rows) > 0:
        sample = zero_rows.iloc[0]
        for col in ["datameci", "orameci", "txtechipa1", "txtechipa2"] + cota_cols:
            print(f"  {col}: {sample.get(col, 'N/A')}")
    
    # Check dates of fixtures WITH vs WITHOUT odds
    print("\n=== DATE DISTRIBUTION ===")
    valid_rows["_date"] = valid_rows["_dt"].dt.date
    zero_rows["_date"] = zero_rows["_dt"].dt.date  
    
    print("Fixtures WITH odds by date:")
    print(valid_rows["_date"].value_counts().sort_index().head(15).to_string())
    
    print("\nFixtures WITHOUT odds - date range:")
    print(f"  Min: {zero_rows['_dt'].min()}")
    print(f"  Max: {zero_rows['_dt'].max()}")

if __name__ == "__main__":
    main()
