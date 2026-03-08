Initialization and Quick Start (Current)

Current references
- Implemented engine ledger:
  - `project_notes/changes_2026_03_07_engine_ledger.md`
- Future full Poisson revamp (DEV only):
  - `project_notes/dev_poisson_full_revamp_plan.md`

Strict mu engine status (important)
- Active default:
  - `config.STRICT_MODULE_MU_ENABLED = True`
- Mu is calculated only from:
  - league anchor
  - Elo module
  - xG module
  - pressure module
- Fixed weights:
  - league anchor `40%`
  - Elo `10%`
  - xG `25%`
  - pressure `25%`
- Missing module behavior:
  - prediction row still generates
  - missing module is neutralized back to league anchor
  - warning is surfaced in prediction output
- Legacy trained-model mu path:
  - archived under `archive/legacy_engine/`
  - no active fallback remains in pipeline scripts

Branch model
- Branch roles and promotion flow are documented in:
  - `project_notes/branches.md`
- LATAM isolated workflow (Argentina + Brazil) is documented in:
  - `project_notes/latam_argentina_brazil_runbook.md`
  - One-command wrapper: `scripts/run_latam_pipeline.ps1`
- Future Monte Carlo rollout milestones are documented in:
  - `project_notes/monte_carlo_future_milestones.md`

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
   - If picks are enabled, a mandatory pre-bet gate runs before pick generation.

API sync quality-gate behavior (important)
- Default sync mode is soft-quality:
  - output files are still written even if quality thresholds are below target.
  - warnings are logged.
- If you need hard-stop behavior:
  - run sync with `--strict-quality-gate`.
- Hard safety stays active regardless of mode:
  - if API returns zero fixtures, sync fails.

Mandatory pre-bet gate
- Trigger:
  - Any `predict.py` run where picks are enabled (`--emit-picks` flow).
- Gate command (executed automatically by `predict.py`):
  - `python scripts/run_all_audits.py --critical-only --as-of-date YYYY-MM-DD`
- Behavior:
  - Critical audit fail -> pick generation blocked (no fresh `reports/picks.csv`).
  - Critical audit pass -> pick engine and narrator run normally.
- Scope:
  - Gate is enforcement only; it does not alter model strategy or thresholds.

Large-Window Rebuild (important for correct league averages)
- Command (PowerShell):
- `$env:API_FOOTBALL_HISTORY_DAYS="365"; $env:API_FOOTBALL_LEAGUE_IDS="39,40,140,78,135,136,61,62,88,94,203,283"; $env:API_FOOTBALL_MAX_REQUESTS="7500"; $env:API_FOOTBALL_RATE_PER_MINUTE="120"; python predict.py --rebuild-history --max-date YYYY-MM-DD`
- What it does:
  - Re-syncs API history with a wider lookback window.
  - Rebuilds `data/enhanced/cgm_match_history.csv` and all downstream enhanced artifacts from fresh data.
  - Recomputes league baselines (home/away goal anchors), Elo V2, stats backfill, and xG proxy inputs.
  - Pulls extra match-stat fields from API-Football when available (used by Pressure V2 enhancements).
  - Elo V2 includes league-specific K/HFA and writes per-row trace fields (`elo_*_used`, `elo_delta`).
- Why it is important:
  - League averages (`lg_avg_gf_home`, `lg_avg_gf_away`) are computed from rebuilt history.
  - If history is too short, league anchors can be biased and OU/BTTS quality drops.
  - This command keeps anchor math stable and season-representative.

Fast Daily Run
- `python predict.py --predict-only`
- In predict-only mode, if picks are enabled, the same mandatory pre-bet gate runs before pick generation.

How to identify weak fixtures directly in output
- Prediction CSV now includes:
  - `quality_status`, `quality_critical`, `quality_issue_count`, `quality_flags`
- File:
  - `reports/cgm_upcoming_predictions.csv`
- Global ELO evidence threshold for this flag:
  - `--min-matches` default is now `3` (from `config.PIPELINE_MIN_MATCHES`).
  - Applies to all leagues/teams unless you override CLI manually.

Telegram send (one message per league)
- Command:
  - `python scripts/send_telegram_predictions.py`
- What it does:
  - Reads `reports/cgm_upcoming_predictions.csv`.
  - Sends one Telegram message per league with OU/BTTS prediction, confidence, and EV.
  - Default filter is value-bets only:
    - `EV > 0`
    - `confidence >= 58%`
  - Auto-splits a league message if it exceeds max chars.
- Dry run (no send):
  - `python scripts/send_telegram_predictions.py --dry-run`
- Optional override (send everything, no value filter):
  - `python scripts/send_telegram_predictions.py --send-all`

One-command full run (sync + rebuild + predict + Telegram)
- Command:
  - `powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline_and_send_telegram.ps1`
- What it does:
  - Syncs API-Football data for all configured Europe leagues (includes `2. Bundesliga`).
  - Rebuilds history/baselines/Elo/stats/xG.
  - Trains `full` mu models.
  - Generates predictions (`reports/cgm_upcoming_predictions.csv`).
  - Sends one Telegram message per league.
  - Runs isolated LATAM pipeline (Argentina + Brazil) and sends those league messages too.
- Useful options:
  - `-AllUpcoming` to send all fixtures in horizon (not only next round).
  - `-HorizonDays 14` (or `21`) to widen fixture window for league coverage.
  - `-DryRunTelegram` to preview messages without sending.
  - `-NoFetchOdds` to skip odds fetch during sync.
  - `-SkipLatam` to skip Argentina/Brazil isolated run.

Example for maximum league coverage in one run:
- `powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline_and_send_telegram.ps1 -AllUpcoming -HorizonDays 21`

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
   - `$env:API_FOOTBALL_HISTORY_DAYS="365"; $env:API_FOOTBALL_LEAGUE_IDS="39,78,135,61,140,283"; $env:API_FOOTBALL_MAX_REQUESTS="7500"; $env:API_FOOTBALL_RATE_PER_MINUTE="120"; python predict.py --rebuild-history --max-date YYYY-MM-DD`
2. Build tuning windows per league with `scripts.run_backtest`.
3. Scan thresholds with:
   - `python scripts/scan_thresholds.py ... --out reports/tuning_batchX_summary.csv`
4. Apply conservative overrides in `config.py`.
5. Validate on last round before promoting.

Important naming note
- Romania top league is stored as `Liga I` (Roman numeral), not `Liga 1`.

Elo legacy archive
- Old Elo V1 code is archived for reference only:
  - `archive/legacy_elo_v1/calc_cgm_elo_v1.py`
  - `archive/legacy_elo_v1/calculate_elo_v1.py`
- Simple non-technical ELO V2 explanation:
  - `project_notes/elo_v2_simple.md`

xG enhancement quick test (v1 vs v2)
1. Build legacy xG baseline:
   - `python -m cgm.build_xg_proxy --history data/enhanced/cgm_match_history_with_elo_stats.csv --out data/enhanced/_xg_v1.csv --feature-set v1 --no-league-calibration`
2. Build enhanced xG candidate:
   - `python -m cgm.build_xg_proxy --history data/enhanced/cgm_match_history_with_elo_stats.csv --out data/enhanced/_xg_v2.csv --feature-set v2 --league-calibration`
3. Compare metrics:
   - `python scripts/evaluate_xg_proxy_candidates.py --baseline data/enhanced/_xg_v1.csv --candidate data/enhanced/_xg_v2.csv --out-prefix reports/xg_proxy_compare`
4. Promotion rule:
   - Keep default `XG_PROXY_FEATURE_SET_DEFAULT="v1"` until v2 improves quality metrics and downstream backtest impact.

