import pandas as pd

# Exact replica of the function from predict_upcoming.py
def _parse_upcoming_datetime(datameci, orameci):
    date_raw = "" if datameci is None else str(datameci)
    dt = pd.to_datetime(date_raw, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(date_raw, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return pd.NaT

    hour = 0
    minute = 0
    try:
        t = int(float(orameci)) if orameci is not None and str(orameci) != "nan" else 0
        hour = max(0, min(23, t // 100))
        minute = max(0, min(59, t % 100))
    except Exception:
        hour = 0
        minute = 0

    return dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


def main():
    """Test the exact parsing function"""
    test_cases = [
        ("1/2/2026", "1500"),   # Should be Jan 2nd
        ("1/12/2026", "2000"),  # Should be Jan 12th  
        ("1/13/2026", "1430"),  # Should be Jan 13th
        ("12/1/2025", "1800"),  # Should be Dec 1st
    ]
    
    print("=== TESTING _parse_upcoming_datetime ===")
    for datameci, orameci in test_cases:
        result = _parse_upcoming_datetime(datameci, orameci)
        print(f"'{datameci}' + '{orameci}' -> {result}")
    
    # Now test on actual data
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", 
                       encoding="latin1", low_memory=False)
    
    # Apply the function
    allr["_fixture_dt"] = allr.apply(
        lambda r: _parse_upcoming_datetime(r.get("datameci"), r.get("orameci")), 
        axis=1
    )
    
    cutoff = pd.to_datetime("2026-01-11T23:59:59")
    future = allr[allr["_fixture_dt"] > cutoff]
    
    print(f"\nFuture fixtures found: {len(future)}")
    
    # Check odds
    future["_cotao"] = pd.to_numeric(future["cotao"], errors="coerce")
    valid = future[future["_cotao"] > 0]
    
    print(f"With valid cotao: {len(valid)}")
    print(f"Percentage with odds: {100*len(valid)/len(future):.1f}%" if len(future) > 0 else "N/A")
    
    # Show a few samples
    print("\n=== SAMPLE FIXTURES ===")
    sample_cols = ["datameci", "orameci", "_fixture_dt", "txtechipa1", "txtechipa2", "cotao"]
    print(future[future["_cotao"] > 0][sample_cols].head(10).to_string())

if __name__ == "__main__":
    main()
