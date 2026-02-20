import pandas as pd

def main():
    """Prove date parsing is the issue"""
    
    # Example dates from allratingv.csv
    test_dates = ["1/2/2026", "1/12/2026", "1/13/2026", "12/1/2026"]
    
    print("=== DATE PARSING COMPARISON ===")
    for d in test_dates:
        parsed_dayfirst = pd.to_datetime(d, dayfirst=True, errors="coerce")
        parsed_monthfirst = pd.to_datetime(d, dayfirst=False, errors="coerce")
        print(f"'{d}' -> dayfirst=True: {parsed_dayfirst.date()} | dayfirst=False: {parsed_monthfirst.date()}")
    
    # Load actual data
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", 
                       encoding="latin1", low_memory=False)
    
    # Parse with both methods
    allr["_dt_dayfirst"] = pd.to_datetime(allr["datameci"], dayfirst=True, errors="coerce")
    allr["_dt_monthfirst"] = pd.to_datetime(allr["datameci"], dayfirst=False, errors="coerce")
    
    cutoff = pd.to_datetime("2026-01-11")
    
    # Compare future counts
    future_dayfirst = allr[allr["_dt_dayfirst"] > cutoff]
    future_monthfirst = allr[allr["_dt_monthfirst"] > cutoff]
    
    print(f"\n=== FUTURE FIXTURE COUNTS ===")
    print(f"dayfirst=True (WRONG): {len(future_dayfirst)}")
    print(f"dayfirst=False (CORRECT): {len(future_monthfirst)}")
    
    # Check odds in correctly-parsed future
    future_monthfirst["_cotao"] = pd.to_numeric(future_monthfirst["cotao"], errors="coerce")
    valid_odds = future_monthfirst[future_monthfirst["_cotao"] > 0]
    
    print(f"\nWith CORRECT date parsing:")
    print(f"  Total future fixtures: {len(future_monthfirst)}")
    print(f"  With valid cotao: {len(valid_odds)}")
    print(f"  Percentage with odds: {100*len(valid_odds)/len(future_monthfirst):.1f}%")
    
    # Show sample
    print("\n=== SAMPLE WITH CORRECT PARSING ===")
    sample_cols = ["datameci", "_dt_monthfirst", "txtechipa1", "txtechipa2", "cotao"]
    print(future_monthfirst[sample_cols].head(10).to_string())

if __name__ == "__main__":
    main()
