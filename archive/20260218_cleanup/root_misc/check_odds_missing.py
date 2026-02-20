import pandas as pd
from pathlib import Path

def main():
    pred_path = Path("reports/cgm_upcoming_predictions.csv")
    if not pred_path.exists():
        print("Predictions file not found")
        return
    
    df = pd.read_csv(pred_path)
    print(f"Total fixtures: {len(df)}")
    
    # Split by odds presence
    odds_col = "odds_over_2_5"
    if odds_col not in df.columns:
        print(f"Column {odds_col} not found")
        return
    
    has_odds = df[df[odds_col] > 0]
    no_odds = df[df[odds_col] == 0]
    
    print(f"\nFixtures WITH odds: {len(has_odds)}")
    print(f"Fixtures WITHOUT odds (zero): {len(no_odds)}")
    
    # Check leagues
    if "league" in df.columns:
        print("\n=== FIXTURES WITHOUT ODDS - by league ===")
        print(no_odds["league"].value_counts().to_string())
        
        print("\n=== FIXTURES WITH ODDS - by league ===")  
        print(has_odds["league"].value_counts().to_string())
    
    # Sample fixtures without odds
    if len(no_odds) > 0:
        print("\n=== SAMPLE FIXTURES WITHOUT ODDS ===")
        sample_cols = ["fixture_datetime", "league", "home", "away", "odds_over_2_5", "odds_under_2_5"]
        sample_cols = [c for c in sample_cols if c in no_odds.columns]
        print(no_odds[sample_cols].head(5).to_string())
    
    # Sample fixtures with odds  
    if len(has_odds) > 0:
        print("\n=== SAMPLE FIXTURES WITH ODDS ===")
        sample_cols = ["fixture_datetime", "league", "home", "away", "odds_over_2_5", "odds_under_2_5"]
        sample_cols = [c for c in sample_cols if c in has_odds.columns]
        print(has_odds[sample_cols].head(5).to_string())

if __name__ == "__main__":
    main()
