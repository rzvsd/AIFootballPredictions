# feature_enhancer.py
# Build enhanced training data by merging Understat xG into your processed dataset.
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# ---------- Config / team name normalization ----------

LEAGUE_KEYS = {
    "E0": "epl",
}

# --- THE CORRECTED AND COMPLETE TEAM MAP ---
# Maps the Understat names (keys) to the football-data.co.uk names (values)
TEAM_MAP_E0: Dict[str, str] = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "Sheffield United": "Sheffield United",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds"
    # Note: We don't need to map names that are already identical (e.g., "Arsenal")
}

# ---------- Understat async fetch ----------

async def fetch_understat_league(season: int, league_code: str = "E0") -> pd.DataFrame:
    """
    Fetch historical matches with xG from Understat for one season+league.
    """
    import aiohttp
    from understat import Understat

    league_key = LEAGUE_KEYS.get(league_code, "epl")

    async with aiohttp.ClientSession() as s:
        u = Understat(s)
        matches: List[Dict[str, Any]] = await u.get_league_results(league_key, int(season))

        rows = []
        for m in matches:
            dt = (m.get("datetime") or "")[:10]
            hxg_raw = (m.get("xG") or {}).get("h")
            axg_raw = (m.get("xG") or {}).get("a")
            try:
                hxg = float(hxg_raw) if hxg_raw is not None else None
            except Exception:
                hxg = None
            try:
                axg = float(axg_raw) if axg_raw is not None else None
            except Exception:
                axg = None

            rows.append({
                "Date": dt,
                "HomeTeam": (m.get("h") or {}).get("title"),
                "AwayTeam": (m.get("a") or {}).get("title"),
                "Home_xG": hxg,
                "Away_xG": axg,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        return df

# ---------- Enhancer (merge + light features) ----------

def normalize_team_names(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """Apply league-specific team name normalization."""
    if league == "E0":
        # We are replacing the Understat names with the base data names
        df["HomeTeam"] = df["HomeTeam"].replace(TEAM_MAP_E0)
        df["AwayTeam"] = df["AwayTeam"].replace(TEAM_MAP_E0)
    return df

def merge_xg(base_path: Path, out_path: Path, xg_df: pd.DataFrame, league: str) -> Path:
    """Merge xG into the base data file and write enhanced CSV."""
    if not base_path.exists():
        raise FileNotFoundError(f"Base file not found: {base_path}")

    # Use the RAW data as the base, not the preprocessed one
    base = pd.read_csv(base_path, parse_dates=["Date"], dayfirst=True)
    
    # Normalize names on the xG side to match the base data
    xg_df = normalize_team_names(xg_df.copy(), league=league)

    # Merge on Date + teams
    merged = base.merge(xg_df, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    miss = merged["Home_xG"].isna().sum()
    print(f"Merged rows: {len(merged)} | Missing xG rows after merge: {miss}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote enhanced data to → {out_path.resolve()}")
    return out_path

# ---------- CLI / Main ----------

async def main():
    ap = argparse.ArgumentParser(description="Enhance raw data with Understat xG.")
    ap.add_argument("--league", default="E0", help="League code (default: E0 for EPL)")
    ap.add_argument("--seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024], help="Seasons to fetch")
    
    # We will use the RAW data as input and create a new ENHANCED file
    ap.add_argument("--in_dir", dest="in_dir", default="data/raw", help="Path to raw data directory")
    ap.add_argument("--out_dir", dest="out_dir", default="data/enhanced", help="Output enhanced CSV directory")
    
    args = ap.parse_args()

    league = args.league
    base_path = Path(args.in_dir) / f"{league}_merged.csv"
    out_path = Path(args.out_dir) / f"{league}_enhanced_with_xg.csv"

    print(f"Loading raw data for {league} from {base_path}...")
    if not base_path.exists():
        raise FileNotFoundError(f"Base file not found: {base_path}")

    all_parts: List[pd.DataFrame] = []
    for year in args.seasons:
        print(f"Fetching Understat xG data for {year} season...")
        season_xg_df = await fetch_understat_league(year, league_code=league)
        if season_xg_df.empty:
            print(f"WARNING: No xG rows returned for season {year}. Continuing.")
        else:
            all_parts.append(season_xg_df)

    if not all_parts:
        raise RuntimeError("No xG data fetched. Check network or season values.")

    xg_df = pd.concat(all_parts, ignore_index=True).drop_duplicates()

    merge_xg(base_path=base_path, out_path=out_path, xg_df=xg_df, league=league)

if __name__ == "__main__":
    asyncio.run(main())