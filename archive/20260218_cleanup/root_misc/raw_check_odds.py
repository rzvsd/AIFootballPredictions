import pandas as pd

def main():
    """Check raw allratingv.csv without any processing"""
    
    # Read as raw strings first
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", 
                       encoding="latin1", low_memory=False, dtype=str)
    
    print(f"Total rows: {len(allr)}")
    
    # Check cotao raw values
    cotao_vals = allr["cotao"].dropna()
    cotao_str = cotao_vals.astype(str).str.strip()
    non_empty = cotao_str[cotao_str != ""]
    non_zero = non_empty[non_empty != "0"]
    
    print(f"\ncotao - Non-empty: {len(non_empty)}")
    print(f"cotao - Non-zero: {len(non_zero)}")
    
    # Check dates that look like Jan 2026
    jan_check = allr[allr["datameci"].astype(str).str.contains("/1/2026|1/202[0-9]", na=False, regex=True)]
    print(f"\nRows with patterns like Jan 2026: {len(jan_check)}")
    
    # Get unique dates for fixtures after Jan 10
    dates = allr["datameci"].astype(str).unique()
    jan_dates = [d for d in dates if "2026" in str(d) and any(x in str(d) for x in ["/1/2026", "1/2026"])]
    print(f"January 2026 dates found: {sorted(jan_dates)[:20]}")
    
    # Direct check: fixtures on 12/1/2026 (European format DD/MM/YYYY)
    jan12_eu = allr[allr["datameci"].astype(str).str.strip() == "12/1/2026"]
    jan13_eu = allr[allr["datameci"].astype(str).str.strip() == "13/1/2026"]
    jan14_eu = allr[allr["datameci"].astype(str).str.strip() == "14/1/2026"]
    
    # Also check US format
    jan12_us = allr[allr["datameci"].astype(str).str.strip() == "1/12/2026"]
    jan13_us = allr[allr["datameci"].astype(str).str.strip() == "1/13/2026"]
    
    print(f"\nFixtures by date:")
    print(f"  12/1/2026 (EU): {len(jan12_eu)}")
    print(f"  13/1/2026 (EU): {len(jan13_eu)}")  
    print(f"  14/1/2026 (EU): {len(jan14_eu)}")
    print(f"  1/12/2026 (US): {len(jan12_us)}")
    print(f"  1/13/2026 (US): {len(jan13_us)}")
    
    # Check Jan 13 fixtures and their cotao values
    if len(jan13_eu) > 0:
        print(f"\n=== JAN 13 FIXTURES ===")
        sample_cols = ["datameci", "orameci", "txtechipa1", "txtechipa2", "cotao", "cotau"]
        print(jan13_eu[sample_cols].head(10).to_string())
        
        # Count how many have valid odds
        jan13_cotao = jan13_eu["cotao"].astype(str).str.strip()
        valid = jan13_cotao[(jan13_cotao != "") & (jan13_cotao != "0") & (jan13_cotao != "nan")]
        print(f"\nJan 13 with valid cotao: {len(valid)} / {len(jan13_eu)}")

if __name__ == "__main__":
    main()
