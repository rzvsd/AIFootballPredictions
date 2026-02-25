Initialization and Quick Start (Current)

Branch model
- Branch roles and promotion flow are documented in:
  - `project_notes/branches.md`
- LATAM isolated workflow (Argentina + Brazil) is documented in:
  - `project_notes/latam_argentina_brazil_runbook.md`
  - One-command wrapper: `scripts/run_latam_pipeline.ps1`

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

Large-Window Rebuild (important for correct league averages)
- Command (PowerShell):
- `$env:API_FOOTBALL_HISTORY_DAYS="365"; $env:API_FOOTBALL_LEAGUE_IDS="39,40,140,78,135,136,61,62,88,94,203,283"; $env:API_FOOTBALL_MAX_REQUESTS="7500"; $env:API_FOOTBALL_RATE_PER_MINUTE="120"; python predict.py --rebuild-history --skip-train --max-date YYYY-MM-DD`
- What it does:
  - Re-syncs API history with a wider lookback window.
  - Rebuilds `data/enhanced/cgm_match_history.csv` and all downstream enhanced artifacts from fresh data.
  - Recomputes league baselines (home/away goal anchors), Elo, stats backfill, and xG proxy inputs.
- Why it is important:
  - League averages (`lg_avg_gf_home`, `lg_avg_gf_away`) are computed from rebuilt history.
  - If history is too short, league anchors can be biased and OU/BTTS quality drops.
  - This command keeps anchor math stable and season-representative.

Fast Daily Run
- `python predict.py --predict-only`

Telegram send (one message per league)
- Command:
  - `python scripts/send_telegram_predictions.py`
- What it does:
  - Reads `reports/cgm_upcoming_predictions.csv`.
  - Sends one Telegram message per league with OU/BTTS prediction, confidence, and EV.
  - Auto-splits a league message if it exceeds max chars.
- Dry run (no send):
  - `python scripts/send_telegram_predictions.py --dry-run`

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

League tuning routine (batch workflow)
1. Rebuild history/baselines for the active batch scope:
   - `$env:API_FOOTBALL_HISTORY_DAYS="365"; $env:API_FOOTBALL_LEAGUE_IDS="39,78,135,61,140,283"; $env:API_FOOTBALL_MAX_REQUESTS="7500"; $env:API_FOOTBALL_RATE_PER_MINUTE="120"; python predict.py --rebuild-history --skip-train --max-date YYYY-MM-DD`
2. Build tuning windows per league with `scripts.run_backtest`.
3. Scan thresholds with:
   - `python scripts/scan_thresholds.py ... --out reports/tuning_batchX_summary.csv`
4. Apply conservative overrides in `config.py`.
5. Validate on last round before promoting.

Important naming note
- Romania top league is stored as `Liga I` (Roman numeral), not `Liga 1`.
