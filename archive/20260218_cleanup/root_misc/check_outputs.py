import pandas as pd
from pathlib import Path

def main():
    # Check picks debug
    debug_path = Path("reports/picks_debug.csv")
    if debug_path.exists():
        df = pd.read_csv(debug_path)
        print("=== PICKS DEBUG ===")
        print(f"Total rows: {len(df)}")
        if "market" in df.columns:
            print(f"Markets: {df['market'].value_counts().to_dict()}")
        if "reasons" in df.columns:
            print("Top failure reasons:")
            print(df["reasons"].value_counts().head(10).to_string())
        if "odds" in df.columns:
            print(f"\nOdds stats:")
            print(f"  Total: {len(df)}")
            print(f"  Non-zero: {len(df[df['odds'] > 0])}")
            print(f"  Zero: {len(df[df['odds'] == 0])}")
            print(f"  NaN: {df['odds'].isna().sum()}")
            print(f"  Sample: {df['odds'].head(5).tolist()}")
    else:
        print("picks_debug.csv does not exist")
    
    # Check predictions  
    pred_path = Path("reports/cgm_upcoming_predictions.csv")
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        print("\n=== PREDICTIONS ===")
        print(f"Total rows: {len(df)}")
        for c in ["odds_over_2_5", "odds_under_2_5", "odds_over25", "odds_under25"]:
            if c in df.columns:
                nonzero = len(df[df[c] > 0])
                zero = len(df[df[c] == 0])
                na = df[c].isna().sum()
                sample = df[c].head(3).tolist()
                print(f"  {c}: nonzero={nonzero}, zero={zero}, nan={na}, sample={sample}")
            else:
                print(f"  {c}: NOT PRESENT")
    else:
        print("\ncgm_upcoming_predictions.csv does not exist")
        
    # Check picks
    picks_path = Path("reports/picks.csv")
    if picks_path.exists():
        df = pd.read_csv(picks_path)
        print("\n=== PICKS ===")
        print(f"Total picks: {len(df)}")
        if "market" in df.columns:
            print(f"By market: {df['market'].value_counts().to_dict()}")
    else:
        print("\npicks.csv does not exist")

if __name__ == "__main__":
    main()
