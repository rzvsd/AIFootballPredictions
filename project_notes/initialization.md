Initialization and Quick Start (Current)

Requirements
- Python environment with dependencies installed
- API key for API-Football in environment variable `API_FOOTBALL_KEY`

Install
- `pip install -r requirements.txt`

Recommended Run (API-first)
1. Set API key (PowerShell):
   - `$env:API_FOOTBALL_KEY="your_key_here"`
2. Run full pipeline:
   - `python predict.py --max-date YYYY-MM-DD`
   - This run creates `data/api_football/*.csv`, `data/api_football/fixture_quality_report.json`, and EV picks in `reports/picks.csv`.

Fast Daily Run
- `python predict.py --predict-only`

Business Report (non-technical)
- `python scripts/generate_business_report.py --rounds 5 --upcoming-limit 20`
- Outputs:
  - `reports/business_report.txt`
  - `reports/business_report_recent_results.csv`
  - `reports/business_report_upcoming_summary.csv`

Useful API Tuning (optional env vars)
- `API_FOOTBALL_LEAGUE_IDS` (example: `39`)
- `API_FOOTBALL_HISTORY_DAYS` (example: `365`)
- `API_FOOTBALL_HORIZON_DAYS` (example: `14`)
- `API_FOOTBALL_MAX_REQUESTS` (free tier default is low)
- `API_FOOTBALL_RATE_PER_MINUTE`
