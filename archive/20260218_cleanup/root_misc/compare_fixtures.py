import pandas as pd

def main():
    """Compare what predict_upcoming sees vs what's in source"""
    
    # Load source with date parsing like pipeline does
    allr = pd.read_csv("CGM data/multiple leagues and seasons/allratingv.csv", 
                       encoding="latin1", low_memory=False)
    
    print(f"Source total rows: {len(allr)}")
    
    # Parse dates like the pipeline does
    def parse_dt(d_str, t_str=""):
        try:
            s = f"{d_str} {t_str}".strip()
            return pd.to_datetime(s, dayfirst=True, errors='coerce')
        except:
            return pd.NaT
    
    allr["_fixture_dt"] = allr.apply(lambda r: parse_dt(r.get("datameci"), r.get("orameci")), axis=1)
    
    # Filter like pipeline does
    cutoff = pd.to_datetime("2026-01-11T23:59:59")
    future = allr[allr["_fixture_dt"] > cutoff].copy()
    
    print(f"Future fixtures (after cutoff): {len(future)}")
    print(f"Date range: {future['_fixture_dt'].min()} to {future['_fixture_dt'].max()}")
    
    # Check cotao in future
    future["_cotao_num"] = pd.to_numeric(future["cotao"], errors="coerce")
    valid_odds = future[future["_cotao_num"] > 0]
    zero_odds = future[(future["_cotao_num"] == 0) | (future["_cotao_num"].isna())]
    
    print(f"\nFuture with cotao > 0: {len(valid_odds)}")
    print(f"Future with cotao = 0 or NaN: {len(zero_odds)}")
    
    # Check which leagues have missing odds
    print("\n=== ZERO ODDS - BY LEAGUE ===")
    print(zero_odds["league"].value_counts().to_string())
    
    print("\n=== VALID ODDS - BY LEAGUE ===")
    print(valid_odds["league"].value_counts().to_string())
    
    # Sample zero odds fixture
    print("\n=== SAMPLE ZERO ODDS FIXTURE ===")
    sample_cols = ["datameci", "orameci", "_fixture_dt", "league", "txtechipa1", "txtechipa2", "cotao", "cotau"]
    print(zero_odds[sample_cols].head(5).to_string())
    
    # Sample valid odds fixture
    print("\n=== SAMPLE VALID ODDS FIXTURE ===")
    print(valid_odds[sample_cols].head(5).to_string())
    
    # Check the date parsing - is dayfirst=True causing issues?
    print("\n=== DATE PARSING CHECK ===")
    # Raw date for a fixture
    sample_raw = allr[["datameci", "orameci", "_fixture_dt"]].dropna().head(5)
    print(sample_raw.to_string())
    
    # Try parsing without dayfirst
    allr["_fixture_dt_us"] = allr.apply(
        lambda r: pd.to_datetime(f"{r.get('datameci')} {r.get('orameci', '')}".strip(), 
                                  dayfirst=False, errors='coerce'), axis=1)
    
    print(f"\nWith dayfirst=True, future count: {len(future)}")
    future_us = allr[allr["_fixture_dt_us"] > cutoff]
    print(f"With dayfirst=False, future count: {len(future_us)}")

if __name__ == "__main__":
    main()
