LATAM Runbook (Argentina + Brazil, Isolated)

Goal
- Run Argentina + Brazil with the same pipeline quality (history, league averages, training, predictions, backtest),
  without changing or breaking the current Europe-focused setup.

Safety Rule (Important)
- Keep LATAM isolated in dedicated folders:
  - `data/api_football_latam_s2025`
  - `data/api_football_latam_s2026`
  - `data/api_football_latam`
  - `data/enhanced_latam`
  - `models_latam`
  - `reports_latam`
- Do not overwrite:
  - `data/api_football`
  - `data/enhanced`
  - `models`
  - `reports`

League IDs
- Argentina top league: `128` (`Liga Profesional Argentina`)
- Brazil top league: `71` (`Serie A`)

Why we split by season
- South America needs `season=2026` for current upcoming fixtures.
- `season=2025` still provides useful historical depth.
- We merge both into one isolated LATAM dataset before feature engineering.

Step 0 - Prerequisites
1. API key set:
   - PowerShell: `$env:API_FOOTBALL_KEY="your_key_here"`
2. Dependencies installed:
   - `pip install -r requirements.txt`

Step 1 - Sync season 2025 history snapshot
- Command:
  - `python scripts/sync_api_football.py --data-dir data/api_football_latam_s2025 --league-ids 71,128 --season 2025 --history-days 365 --horizon-days 7 --max-requests 7500 --rate-per-minute 120 --no-fetch-odds`
- What this does:
  - Pulls historical fixtures/stats for both leagues into an isolated season folder.
  - Odds are skipped here to save calls (historical base layer).

Step 2 - Sync season 2026 current snapshot (with upcoming + odds)
- Command:
  - `python scripts/sync_api_football.py --data-dir data/api_football_latam_s2026 --league-ids 71,128 --season 2026 --history-days 365 --horizon-days 7 --max-requests 7500 --rate-per-minute 120 --fetch-odds`
- What this does:
  - Pulls current-season history and upcoming fixtures for next-round prediction.
  - Includes odds for EV calculations.

Step 3 - Build merged LATAM dataset
- Command:
  - `python scripts/build_latam_dataset.py`
- What this does:
  - Merges `s2025` + `s2026` history into `data/api_football_latam/history_fixtures.csv`.
  - Uses `s2026` upcoming as `data/api_football_latam/upcoming_fixtures.csv`.
  - Creates bridge files under:
    - `data/api_football_latam/multiple leagues and seasons/`
  - Renames Brazil league label from `Serie A` to `Serie A Brazil` in this isolated dataset.
    - Reason: avoid collisions with Italy `Serie A` thresholds/backtests.

Step 4 - Build enhanced history + league averages (isolated)
1. Build match history:
   - `python -m cgm.build_match_history --data-dir data/api_football_latam --out data/enhanced_latam/cgm_match_history.csv --max-date YYYY-MM-DD`
2. Build baselines:
   - `python -m cgm.build_baselines --data-dir data/api_football_latam --match-history data/enhanced_latam/cgm_match_history.csv --out-team-baselines data/enhanced_latam/team_baselines.csv`
- What this does:
  - Computes league average anchors and team baseline features from merged LATAM history.

Step 5 - Build Elo + stats + xG proxy (isolated)
1. Elo:
   - `python -m scripts.calc_cgm_elo --history data/enhanced_latam/cgm_match_history.csv --out data/enhanced_latam/cgm_match_history_with_elo.csv --data-dir data/api_football_latam --max-date YYYY-MM-DD --log-json reports_latam/run_log.jsonl`
2. Stats backfill:
   - `python -m cgm.backfill_match_stats --history data/enhanced_latam/cgm_match_history_with_elo.csv --stats "data/api_football_latam/multiple leagues and seasons/history_fixtures.csv" --out data/enhanced_latam/cgm_match_history_with_elo_stats.csv --data-dir data/api_football_latam`
3. xG proxy:
   - `python -m cgm.build_xg_proxy --history data/enhanced_latam/cgm_match_history_with_elo_stats.csv --out data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv`

Step 6 - Strict mu engine note (isolated)
- No separate model training step exists in current production path.
- Mu is computed at prediction time from:
  - league anchor
  - Elo module
  - xG module
  - pressure module

Step 7 - Next-round predictions (isolated)
1. Predict:
   - `python -m cgm.predict_upcoming --history data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv --models-dir models_latam --model-variant full --out reports_latam/cgm_upcoming_predictions.csv --data-dir data/api_football_latam --as-of-date YYYY-MM-DD --next-round-only --next-round-span-days 3 --log-json reports_latam/run_log.jsonl --trace-json reports_latam/elo_trace.jsonl`
2. Picks (optional):
   - `python -m cgm.pick_engine_goals --in reports_latam/cgm_upcoming_predictions.csv --out reports_latam/picks.csv --debug-out reports_latam/picks_debug.csv`
- Note:
  - Some fixtures can be filtered out by minimum-history guards (`min_matches`) in prediction step.
  - Brazil-only rebalance is applied automatically in this step (no extra command):
    - Config key: `MU_GOAL_MULTIPLIER_BY_LEAGUE["Serie A Brazil"] = 1.70`
    - Scope: only league label `Serie A Brazil`
    - Argentina is not touched.

Brazil-only rebalance (current strategy)
- Why:
  - In the isolated Brazil backtest window, raw model `mu_total` was systematically too low.
  - Window used: `2025-11-01..2026-02-19` (`102` matches).
- What changed:
  - Added pre-Poisson Brazil multiplier:
    - `MU_GOAL_MULTIPLIER_DEFAULT = 1.00`
    - `MU_GOAL_MULTIPLIER_BY_LEAGUE = {"Serie A Brazil": 1.70}`
  - Implementation point: `cgm/predict_upcoming.py` (applied to `mu_home` and `mu_away` before probability/EV).
  - Audit fields now exported:
    - `mu_home_raw`, `mu_away_raw`, `mu_goal_multiplier`.
- Operational impact:
  - No command change for normal runs.
  - To disable temporarily: set `MU_GOAL_MULTIPLIER_BY_LEAGUE["Serie A Brazil"] = 1.00`.

Brazil rebalance validation commands
1. Broad window check:
   - `python -m scripts.run_backtest --league "Serie A Brazil" --season "2025-2026" --start-date "2025-11-01" --history data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv --models-dir models_latam --out reports_latam/backtest_brazil_tuning_window_rebalanced.csv`
2. Last round check:
   - `python -m scripts.run_backtest --league "Serie A Brazil" --season "2025-2026" --start-date "2026-02-11" --history data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv --models-dir models_latam --out reports_latam/backtest_lastround_brazil_rebalanced.csv`

Step 8 - Previous round vs reality (isolated backtest)
- Argentina last round (example window):
  - `python -m scripts.run_backtest --league "Liga Profesional Argentina" --season "2025-2026" --start-date "2026-02-20" --history data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv --models-dir models_latam --out reports_latam/backtest_lastround_argentina.csv`
- Brazil last round (example window):
  - `python -m scripts.run_backtest --league "Serie A Brazil" --season "2025-2026" --start-date "2026-02-11" --history data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv --models-dir models_latam --out reports_latam/backtest_lastround_brazil.csv`

Backtest note
- `scripts/run_backtest.py` is date-window based (`--start-date`), not round-id based.
- Use league-specific date windows for the target "last round" in each competition.

Operational checklist (quick)
1. Sync `s2025` and `s2026`.
2. Run `build_latam_dataset.py`.
3. Build enhanced LATAM artifacts and train to `models_latam`.
4. Predict to `reports_latam/cgm_upcoming_predictions.csv`.
5. Run last-round backtest files in `reports_latam/`.

One-command wrapper (PowerShell)
- Command:
  - `powershell -ExecutionPolicy Bypass -File scripts/run_latam_pipeline.ps1 -MaxDate YYYY-MM-DD`
- Optional:
  - `-SkipPicks` to skip pick generation.
  - `-HistoryDays`, `-HorizonDays`, `-MaxRequests`, `-RatePerMinute` to tune API batch behavior.
- What it runs:
  - End-to-end isolated LATAM flow (sync -> merge -> features -> train -> predict -> picks).

What is safe to tune later
- Threshold tuning for LATAM should be kept separate from Europe until validated.
- If added, keep LATAM thresholds clearly isolated (or explicitly scoped) to avoid cross-league side effects.

