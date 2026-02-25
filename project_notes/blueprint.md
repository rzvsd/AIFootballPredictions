Project Blueprint (Current - API-Football Only)

Purpose
- Predict goals markets only:
  - Over/Under 2.5
  - BTTS (Both Teams To Score)
- Uses internal Elo and internal xG proxy features.
- Uses API odds to compute EV and rank picks.
- Uses team-centric baseline anchors (home/away + similar-ELO context) with league fallback.
- Current strategy on `test`:
  - League average anchor is the baseline.
  - Team-vs-team/similar-ELO evidence blends on top of that baseline.

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
7. Predict upcoming fixtures (calibrated Poisson formula; no XGBoost dependency)
8. Compute EV-based picks (OU2.5 / BTTS)

Brazil-specific rebalance (isolated LATAM only)
- Active only for league label `Serie A Brazil` (Argentina untouched).
- Mechanism:
  - Apply league-specific multiplier to model outputs before Poisson:
    - `MU_GOAL_MULTIPLIER_BY_LEAGUE["Serie A Brazil"] = 1.70`
    - `MU_GOAL_MULTIPLIER_DEFAULT = 1.00`
  - Implemented in `cgm/predict_upcoming.py`.
- Auditability:
  - Prediction exports include `mu_home_raw`, `mu_away_raw`, `mu_goal_multiplier`.
- Validation snapshot (window `2025-11-01..2026-02-19`, 102 matches):
  - OU accuracy: `46.1% -> 61.8%`
  - BTTS accuracy: `52.0% -> 66.7%`
  - Predicted OVER rate: `2.0% -> 37.3%` (actual over rate `55.9%`)

Key Modules
- `scripts/sync_api_football.py`: fetch fixtures and stats (free-tier aware)
- `providers/api_football.py`: API client with caching, retry, rate and daily limits
- `cgm/build_match_history.py`: normalizes fixture history
- `scripts/calc_cgm_elo.py`: computes Elo timeline
- `cgm/build_xg_proxy.py`: computes leakage-safe xG proxy
- `cgm/predict_upcoming.py`: computes team-centric mu + probabilities
- `cgm/pick_engine_goals.py`: applies OU/BTTS decision gates and EV ranking

Main Outputs
- Predictions: `reports/cgm_upcoming_predictions.csv`
- EV picks: `reports/picks.csv`
- Business summary: `reports/business_report.txt`
- Recent results summary: `reports/business_report_recent_results.csv`
- Upcoming summary: `reports/business_report_upcoming_summary.csv`
- Run summary: `reports/pipeline_summary.json`

Sterile Override + Team-Centric Anchor (Current)
- Trigger rule:
  - Use count-based sterile trigger only.
  - Trigger at `4/10` (moderate) and `6/10` (strong).
  - Require minimum `8` valid sterile signals before trigger can apply.
- Anchor behavior:
  - Base league anchor can be overridden downward for sterile fixtures.
  - Team-centric anchor then blends in home/away similar-ELO goal rates when evidence is sufficient.
  - Fallback stays league-safe when evidence is weak.
- Boost cap:
  - Under boost remains capped at `+0.05` after Poisson probability calculation.

OU Decision Paths (Current)
- Two mirrored scores are emitted per fixture:
  - `over_score`: attack strength + defensive leak + pressure.
  - `under_score`: attack weakness + defensive strength + low pressure + sterile support.
- Pick engine applies OU direction checks:
  - minimum score per side
  - minimum margin (`over_score - under_score` or inverse)
- EV and reliability gates still apply after path checks.

Bundesliga Tuning (Current)
- Cleaning applied:
  - Backtest temp fixtures now preserve team codes (`codechipa1` / `codechipa2`) to avoid name/encoding drops.
  - Upcoming CSV ingest uses UTF-8-first fallback (then latin1) to avoid mojibake team names.
- Decision thresholds (league-specific):
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["Bundesliga"] = 0.30`
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["Bundesliga"] = 0.28`
- Tuning window used:
  - `reports/backtest_bundesliga_tuning_window.csv` (start `2025-11-01`, 133 matches with results).

Threshold Scan Snapshots (Current)
- Evidence files generated:
  - `reports/backtest_premier_tuning_window.csv` (start `2025-11-01`, 181 matches with results)
  - `reports/backtest_serie_a_tuning_window.csv` (start `2025-11-01`, 170 matches with results)
  - `reports/tuning_threshold_scan_summary.csv` (raw accuracy scan)
  - `reports/tuning_threshold_scan_balanced.csv` (balanced-accuracy scan)
- Current interpretation:
  - Bundesliga thresholds remain validated (`OU=0.30`, `BTTS=0.28`).
  - EPL and Serie A show threshold sensitivity, especially on BTTS.
  - Conservative updates applied:
    - `BTTS_YES_THRESHOLD_BY_LEAGUE["Premier League"] = 0.31`
    - `OU25_OVER_THRESHOLD_BY_LEAGUE["Serie A"] = 0.54`
  - Not applied intentionally:
    - No Serie A BTTS override yet (kept at default due mixed short-window behavior).

Batch 1 Tuning (France/Spain/Romania)
- Rebuild scope used before tuning:
  - League IDs: `39,78,135,61,140,283` (keeps EPL/Bundesliga/Serie A plus batch-1 leagues).
- Batch-1 evidence files:
  - `reports/backtest_ligue1_tuning_window.csv` (117 matches with results)
  - `reports/backtest_laliga_tuning_window.csv` (148 matches with results)
  - `reports/backtest_ligai_tuning_window.csv` (111 matches with results)
  - `reports/tuning_batch1_summary.csv`
  - `reports/tuning_batch1_applied_summary.csv`
- Applied threshold overrides:
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["Ligue 1"] = 0.52`
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["La Liga"] = 0.46`
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["Liga I"] = 0.26` (recent-form pilot)
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["Ligue 1"] = 0.51`
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["Liga I"] = 0.33` (recent-form pilot)
- Kept default intentionally:
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["La Liga"]` remains default (0.50), because the scan-only improvement required an extreme threshold (`0.13`) that is likely unstable.

Batch 2 Tuning (England 2 / Holland)
- Rebuild scope used:
  - League IDs: `39,40,61,78,88,135,140,283`
- Batch-2 evidence files:
  - `reports/backtest_championship_tuning_window.csv` (250 matches with results)
  - `reports/backtest_eredivisie_tuning_window.csv` (125 matches with results)
  - `reports/tuning_batch_next_champ_eredivisie_summary.csv`
  - `reports/tuning_batch2_applied_summary.csv`
- Applied threshold overrides:
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["Championship"] = 0.32`
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["Championship"] = 0.30`
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["Eredivisie"] = 0.23`
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["Eredivisie"] = 0.25`

Batch 4 Tuning (Germany 2 / Turkey 1)
- Coverage status:
  - League id `79` (2. Bundesliga) sync now succeeds in current environment.
  - League id `203` (Süper Lig) sync succeeds in current environment.
- Batch-4 evidence files:
  - `reports/backtest_2bundesliga_tuning_window.csv` (115 matches with results)
  - `reports/backtest_super_lig_tuning_window.csv` (116 matches with results)
  - `reports/tuning_batch_2bund_superlig_summary.csv`
  - `reports/tuning_batch_2bund_superlig_balanced_summary.csv`
- Applied threshold overrides:
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["2. Bundesliga"] = 0.33`
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["2. Bundesliga"] = 0.29`
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["Süper Lig"] = 0.41`
- Kept default intentionally:
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["Süper Lig"]` remains default (`0.50`) due weak/unstable gain in balanced scan.

Batch 3 Tuning (Italy 2 / France 2 / Portugal 1)
- Rebuild scope used:
  - League IDs: `39,40,61,62,78,88,94,135,136,140,283`
- Batch-3 evidence files:
  - `reports/backtest_serieb_tuning_window.csv` (161 matches with results)
  - `reports/backtest_ligue2_tuning_window.csv` (103 matches with results)
  - `reports/backtest_primeira_tuning_window.csv` (125 matches with results)
  - `reports/tuning_batch_serieb_ligue2_primeira_summary.csv`
  - `reports/tuning_batch3_applied_summary.csv`
- Applied threshold overrides:
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["Serie B"] = 0.42`
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["Ligue 2"] = 0.32`
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["Ligue 2"] = 0.38`
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["Primeira Liga"] = 0.36`
- Kept default intentionally:
  - `OU25` for `Serie B` remains default (`0.50`) because window accuracy did not improve under a conservative-rate rule.
  - `BTTS` for `Primeira Liga` remains default (`0.50`) because tuned alternatives gave no stable gain.

History Rebuild Rule (for stable anchors)
- Run a larger-window rebuild regularly (or after data gaps) so league averages stay representative:
  - `$env:API_FOOTBALL_HISTORY_DAYS="365"; $env:API_FOOTBALL_LEAGUE_IDS="39,40,140,78,135,136,61,62,88,94,203,283"; $env:API_FOOTBALL_MAX_REQUESTS="7500"; $env:API_FOOTBALL_RATE_PER_MINUTE="120"; python predict.py --rebuild-history --skip-train --max-date YYYY-MM-DD`
