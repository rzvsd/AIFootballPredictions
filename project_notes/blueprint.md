Project Blueprint (Current - API-Football Only)

Purpose
- Predict goals markets only:
  - Over/Under 2.5
  - BTTS (Both Teams To Score)
- Uses internal Elo and internal xG proxy features.
- Uses API odds to compute EV and rank picks.

Data Source
- API-Football only
  - Orchestrator: `predict.py --data-source api-football` (default)
  - Sync script: `scripts/sync_api_football.py`
  - Normalized files:
    - `data/api_football/history_fixtures.csv`
    - `data/api_football/upcoming_fixtures.csv`
    - `data/api_football/fixture_quality_report.json`
  - Note: these files are created on each run; after cleanup they may not exist until the next sync.

Core Pipeline (high-level)
1. Optional API sync
2. Build match history
3. Build baselines
4. Recompute Elo
5. Backfill match stats
6. Build xG proxy
7. Build training matrix
8. Train mu models (unless `--skip-train`)
9. Predict upcoming fixtures
10. Compute EV-based picks (OU2.5 / BTTS)

Key Modules
- `scripts/sync_api_football.py`: fetch fixtures and stats (free-tier aware)
- `providers/api_football.py`: API client with caching, retry, rate and daily limits
- `cgm/build_match_history.py`: normalizes fixture history
- `scripts/calc_cgm_elo.py`: computes Elo timeline
- `cgm/build_xg_proxy.py`: computes leakage-safe xG proxy
- `cgm/predict_upcoming.py`: computes probabilities

Main Outputs
- Predictions: `reports/cgm_upcoming_predictions.csv`
- EV picks: `reports/picks.csv`
- Business summary: `reports/business_report.txt`
- Recent results summary: `reports/business_report_recent_results.csv`
- Upcoming summary: `reports/business_report_upcoming_summary.csv`
- Run summary: `reports/pipeline_summary.json`
