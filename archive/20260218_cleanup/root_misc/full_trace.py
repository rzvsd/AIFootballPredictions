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
    """Full trace - what should the pipeline see?"""
    
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", 
                       encoding="latin1", low_memory=False)
    
    # Apply the function
    allr["_fixture_dt"] = allr.apply(
        lambda r: _parse_upcoming_datetime(r.get("datameci"), r.get("orameci")), 
        axis=1
    )
    
    cutoff = pd.to_datetime("2026-01-11T23:59:59")
    future = allr[allr["_fixture_dt"] > cutoff].copy()
    
    print(f"Future fixtures: {len(future)}")
    
    # Check odds
    future["_cotao"] = pd.to_numeric(future["cotao"], errors="coerce")
    valid = future[future["_cotao"] > 0]
    zero = future[future["_cotao"] == 0]
    na = future[future["_cotao"].isna()]
    
    print(f"\nOdds breakdown:")
    print(f"  Valid (cotao > 0): {len(valid)}")
    print(f"  Zero (cotao = 0): {len(zero)}")
    print(f"  NaN (cotao = NaN): {len(na)}")
    
    # Where are the zero odds fixtures coming from?
    print("\n=== ZERO ODDS - BY DATE ===")
    zero["_date"] = zero["_fixture_dt"].dt.date
    print(zero["_date"].value_counts().head(10).to_string())
    
    print("\n=== VALID ODDS - BY DATE ===")
    valid["_date"] = valid["_fixture_dt"].dt.date
    print(valid["_date"].value_counts().head(10).to_string())
    
    # Compare raw cotao values
    print("\n=== SAMPLE ZERO ODDS (raw cotao values) ===")
    print(zero[["datameci", "txtechipa1", "txtechipa2", "cotao", "_cotao"]].head(10).to_string())
    
    print("\n=== SAMPLE VALID ODDS (raw cotao values) ===")
    print(valid[["datameci", "txtechipa1", "txtechipa2", "cotao", "_cotao"]].head(10).to_string())

if __name__ == "__main__":
    main()
