import pandas as pd

def main():
    """Find fixtures between Jan 12-17 and check their odds"""
    
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", 
                       encoding="latin1", low_memory=False, dtype=str)
    
    # The date format appears to be MM/DD/YYYY (US format)
    # Find all fixtures for dates 1/12/2026 through 1/17/2026
    target_dates = ["1/12/2026", "1/13/2026", "1/14/2026", "1/15/2026", "1/16/2026", "1/17/2026"]
    
    for date in target_dates:
        matches = allr[allr["datameci"].astype(str).str.strip() == date]
        if len(matches) > 0:
            cotao_vals = matches["cotao"].astype(str).str.strip()
            valid = cotao_vals[(cotao_vals != "") & (cotao_vals != "0") & (cotao_vals != "nan")]
            print(f"\n{date}: {len(matches)} fixtures, {len(valid)} with valid cotao")
            print(matches[["txtechipa1", "txtechipa2", "cotao", "cotau"]].head(5).to_string())
    
    # Also check what the actual valid cotao dates are
    print("\n=== DATES WITH VALID COTAO (> 0) ===")
    allr["_cotao_valid"] = (allr["cotao"].astype(str).str.strip() != "") & \
                           (allr["cotao"].astype(str).str.strip() != "0") & \
                           (allr["cotao"].astype(str).str.strip() != "nan")
    valid_rows = allr[allr["_cotao_valid"]]
    
    # Get unique dates with valid cotao that look like 2026
    dates_2026 = valid_rows[valid_rows["datameci"].astype(str).str.contains("2026")]
    unique_dates = dates_2026["datameci"].unique()
    print(f"Unique 2026 dates with valid cotao: {len(unique_dates)}")
    print(f"Sample dates: {sorted(unique_dates)[:20]}")
    
    # Count by date
    if len(dates_2026) > 0:
        print("\nCount by date (2026 with valid cotao):")
        print(dates_2026["datameci"].value_counts().head(20).to_string())

if __name__ == "__main__":
    main()
