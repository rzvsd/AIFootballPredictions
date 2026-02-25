import sys
from pathlib import Path
import os # Added os import

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set provider explicitly to fix auth issue
os.environ["API_FOOTBALL_PROVIDER"] = "api-sports"

import pandas as pd
import logging
from providers.api_football import APIFootballClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_historical")

# Configuration
LEAGUE_IDS = [39, 40, 140, 78, 135, 136, 61, 62, 88, 94, 203]
SEASONS = [2022, 2023, 2024]
DATA_DIR = Path("data/api_football")
OUTPUT_FILE = DATA_DIR / "historical_fixtures_22_24.csv"

def fetch_historical():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize client (will pick up API_FOOTBALL_KEY from .env or env vars)
    try:
        client = APIFootballClient(rate_per_minute=10, max_requests_per_day=100)
    except Exception as e:
        logger.error(f"Failed to initialize API client: {e}")
        return

    all_fixtures = []
    
    # 1. Fetch Fixture Lists (High-level)
    # This costs 1 request per league/season.
    for season in SEASONS:
        for league_id in LEAGUE_IDS:
            logger.info(f"Fetching fixture list for league {league_id}, season {season}...")
            try:
                # Use a specific cache key for historical seasons
                fname = f"fixtures_L{league_id}_S{season}.json"
                cache_key = f"historical/{fname}"
                
                fixtures = client.fixtures(
                    league_id=league_id,
                    season=season,
                    status=["FT", "AET", "PEN"],
                    cache_key=cache_key
                )
                
                for f in fixtures:
                    fix = f.get('fixture', {})
                    teams = f.get('teams', {})
                    goals = f.get('goals', {})
                    league = f.get('league', {})
                    
                    all_fixtures.append({
                        'fixture_id': fix.get('id'),
                        'date': fix.get('date', '')[:10],
                        'league_id': league.get('id'),
                        'league': league.get('name'),
                        'season': league.get('season'),
                        'home_id': teams.get('home', {}).get('id'),
                        'away_id': teams.get('away', {}).get('id'),
                        'home': teams.get('home', {}).get('name'),
                        'away': teams.get('away', {}).get('name'),
                        'score_home': goals.get('home'),
                        'score_away': goals.get('away'),
                        'status': fix.get('status', {}).get('short')
                    })
                logger.info(f"  Found {len(fixtures)} fixtures.")
            except Exception as e:
                logger.warning(f"  Error fetching fixtures for league {league_id}: {e}")
                if "budget exhausted" in str(e).lower():
                    logger.error("Budget exhausted during fixture list fetch.")
                    break
    
    if not all_fixtures:
        logger.warning("No fixtures found.")
        return

    df = pd.DataFrame(all_fixtures)
    logger.info(f"Total historical fixtures gathered: {len(df)}")

    # 2. Fetch Match Statistics (Incremental)
    # This costs 1 request PER match. We will only do as many as the budget allows.
    logger.info("Starting match statistics hydration (1 request per match)...")
    
    # We'll save progress frequently
    hydration_count = 0
    
    for idx, row in df.iterrows():
        # Check if we already have stats columns. If not, add them.
        stat_cols = ['shots_home', 'shots_away', 'shots_on_target_home', 'shots_on_target_away', 'corners_home', 'corners_away', 'possession_home', 'possession_away']
        for col in stat_cols:
            if col not in df.columns:
                df[col] = pd.NA

        # Skip if already has stats
        if not pd.isna(row.get('shots_home')):
            continue
            
        fixture_id = row['fixture_id']
        if not fixture_id:
            continue
            
        try:
            cache_key = f"stats/{fixture_id}.json"
            # Attempt to fetch (will use cache if exists, otherwise hit API)
            stats_payload = client.fixture_statistics(fixture_id, cache_key=cache_key)
            
            # Map stats payload to row
            # API returns a list of 2 dicts (home team and away team)
            stats_map = {}
            for entry in stats_payload:
                team_type = 'home' if entry.get('team', {}).get('id') == row['home_id'] else 'away'
                raw_stats = {s.get('type').lower().replace(' ', '_'): s.get('value') for s in (entry.get('statistics') or [])}
                
                stats_map[f'shots_{team_type}'] = raw_stats.get('total_shots')
                stats_map[f'shots_on_target_{team_type}'] = raw_stats.get('shots_on_target')
                stats_map[f'corners_{team_type}'] = raw_stats.get('corner_kicks')
                stats_map[f'possession_{team_type}'] = raw_stats.get('ball_possession')

            # Update DF
            for k, v in stats_map.items():
                df.at[idx, k] = v
            
            hydration_count += 1
            if hydration_count % 20 == 0:
                logger.info(f"Hydrated {hydration_count} matches...")
                df.to_csv(OUTPUT_FILE, index=False) # Checkpoint
                
        except Exception as e:
            if "budget exhausted" in str(e).lower():
                logger.warning(f"Daily budget exhausted at match {idx}. Stopping.")
                break
            logger.warning(f"Error fetching stats for fixture {fixture_id}: {e}")

    # Final Save
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Historical fetch complete. Total hydrated this session: {hydration_count}")
    logger.info(f"Final file saved to {OUTPUT_FILE}")
    
    status = client.budget_status()
    logger.info(f"Final Budget Status: {status}")

if __name__ == "__main__":
    fetch_historical()
