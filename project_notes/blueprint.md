Project Blueprint (Current - API-Football Only)

Purpose
- Predict goals markets only:
  - Over/Under 2.5
  - BTTS (Both Teams To Score)
- Uses internal Elo and internal xG proxy features.
- Uses API odds to compute EV and rank picks.
- Uses strict explicit mu engine:
  - league anchor baseline
  - Elo module
  - xG module
  - pressure module
  - Poisson V2 converts mu into probabilities

Current cycle change references (must-read)
- Implemented engine ledger:
  - `project_notes/changes_2026_03_07_engine_ledger.md`
- Future full Poisson revamp plan for `dev`:
  - `project_notes/dev_poisson_full_revamp_plan.md`

Strict mu engine status
- Current default is ON (`config.STRICT_MODULE_MU_ENABLED = True`).
- Mu weights are fixed:
  - league anchor `40%`
  - Elo `10%`
  - xG `25%`
  - pressure `25%`
- Missing module behavior:
  - row still predicts
  - missing module contribution falls back to league anchor
  - warning is exported in `quality_flags`
- Legacy trained-model challenger code is archived under `archive/legacy_engine/`.

Data Source
- API-Football only
  - Orchestrator: `predict.py --data-source api-football` (default)
  - Sync script: `scripts/sync_api_football.py`
  - Normalized files:
    - `data/api_football/history_fixtures.csv`
    - `data/api_football/upcoming_fixtures.csv`
    - `data/api_football/fixture_quality_report.json`
  - Note: these files are created on each run; after cleanup they may not exist until the next sync.

Sync warning behavior (important)
- `scripts/sync_api_football.py` now supports soft and strict quality gates.
- Default is soft mode:
  - writes outputs,
  - logs warnings when quality thresholds are not met.
- Strict mode:
  - add `--strict-quality-gate` to fail on quality gate violations.
- Hard stop always active:
  - if total loaded fixtures is zero, sync exits with error.

Core Pipeline (high-level)
1. Optional API sync
2. Build match history
3. Build baselines
4. Recompute Elo
5. Backfill match stats
6. Build xG proxy
7. Predict upcoming fixtures using explicit 4-module weighted mu + Poisson V2
8. Mandatory pre-bet gate (critical audits only)
9. Compute EV-based picks (OU2.5 / BTTS)

Pressure V2 (current)
- API sync/backfill now ingests additional per-match stats when available:
  - shots off goal, blocked shots, goal attempts
  - attacks, dangerous attacks
  - plus discipline/flow stats (fouls, offsides, cards, etc.) for future modules
- Pressure form remains leakage-safe and venue-aware, with dynamic weighting:
  - core components always used: shots, shots on target, corners, possession
  - optional components are used only when present (no forced fallback)
  - if optional stats are missing for a fixture/league, model falls back to core pressure only

Prediction output quality visibility (current)
- `reports/cgm_upcoming_predictions.csv` now contains:
  - `quality_status` (`OK`/`WARN`/`BAD`)
  - `quality_critical`
  - `quality_issue_count`
  - `quality_flags`
- Global ELO evidence threshold used for `elo_evidence_low`:
  - minimum effective similar-match evidence default is `3` for all leagues/teams
  - configurable via `config.PIPELINE_MIN_MATCHES` and CLI `--min-matches`.
- This is the primary per-fixture view to identify weak/missing evidence rows.

Poisson V2 (current)
- Active probability engine is Poisson V2 enhanced (incremental upgrade), not plain independent Poisson.
- Includes:
  - dispersion layer,
  - low-score correction,
  - light dependence between home/away goal counts.
- Runtime parameters are exported per prediction row for transparency:
  - `poisson_v2_enabled`, `poisson_v2_disp_alpha`, `poisson_v2_dep_strength`, `poisson_v2_dc_rho`.

Pre-bet gate enforcement
- Implemented in: `predict.py` before any pick engine call.
- Gate command:
  - `python scripts/run_all_audits.py --critical-only --as-of-date YYYY-MM-DD`
- Gate outcome:
  - Fail: stop run before `reports/picks.csv` write.
  - Pass: continue to pick engine + narrator.
- Strategy impact:
  - No change to feature engineering, model scoring, thresholds, or EV ranking.
  - Only blocks unsafe pick publication when integrity checks fail.

Prediction transparency (current)
- Prediction exports now include module-level mu columns:
  - `mu_anchor_home`, `mu_anchor_away`
  - `mu_elo_home`, `mu_elo_away`
  - `mu_xg_home`, `mu_xg_away`
  - `mu_pressure_home`, `mu_pressure_away`
- Engine mode column:
  - `mu_engine_mode`

Key Modules
- `scripts/sync_api_football.py`: fetch fixtures and stats (free-tier aware)
- `providers/api_football.py`: API client with caching, retry, rate and daily limits
- `cgm/build_match_history.py`: normalizes fixture history
- `scripts/calc_cgm_elo.py`: computes Elo timeline (Elo V2)
- `cgm/build_xg_proxy.py`: computes leakage-safe xG proxy
- `cgm/predict_upcoming.py`: computes team-centric mu + probabilities
- `cgm/pick_engine_goals.py`: applies OU/BTTS decision gates and EV ranking

xG Proxy V2 (implemented, safe rollout)
- Upgrade is additive (no rewrite of full engine).
- `cgm/build_xg_proxy.py` now supports:
  - `--feature-set v1` (legacy baseline)
  - `--feature-set v2` (enhanced features: corners/possession/shot-share/opponent-Elo factor)
  - league-level calibration (optional)
- Safety default remains `v1` in `config.py` until `v2` beats baseline in evaluation.
- Comparison utility:
  - `scripts/evaluate_xg_proxy_candidates.py`
  - outputs side-by-side metrics (RMSE, Poisson NLL, Brier, log-loss).

Elo V2 (active)
- Elo now uses league-aware parameters and per-match trace fields:
  - League-specific `k_factor` and `home_adv` (from `config.py`).
  - Match-type multipliers (league/cup/playoff/friendly).
  - New-team multiplier for low-match teams.
  - Upset multiplier when result strongly beats expectation.
  - Capped goal-difference multiplier.
- Trace columns written in history:
  - `elo_hfa_used`, `elo_k_base_used`, `elo_k_matchtype_mult`, `elo_k_newteam_mult`,
  - `elo_k_upset_mult`, `elo_k_used`, `elo_g_used`, `elo_expected_home`, `elo_actual_home`, `elo_delta`.
- Inference now reads league-specific HFA from history trace (`elo_hfa_used`) instead of a single global constant.
- Legacy Elo V1 files archived at:
  - `archive/legacy_elo_v1/calc_cgm_elo_v1.py`
  - `archive/legacy_elo_v1/calculate_elo_v1.py`
- Non-technical explainer:
  - `project_notes/elo_v2_simple.md`

Main Outputs
- Predictions: `reports/cgm_upcoming_predictions.csv`
- EV picks: `reports/picks.csv`
- Business summary: `reports/business_report.txt`
- Recent results summary: `reports/business_report_recent_results.csv`
- Upcoming summary: `reports/business_report_upcoming_summary.csv`
- Run summary: `reports/pipeline_summary.json`

Future roadmap (Monte Carlo)
- Planned milestones are documented in:
  - `project_notes/monte_carlo_future_milestones.md`
- Scope note:
  - Monte Carlo roadmap is currently future-only and not active in production.
  - First target phase is a pre-match Monte Carlo overlay for OU2.5 and BTTS, with confidence intervals.

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
  - League id `203` (SÃ¼per Lig) sync succeeds in current environment.
- Batch-4 evidence files:
  - `reports/backtest_2bundesliga_tuning_window.csv` (115 matches with results)
  - `reports/backtest_super_lig_tuning_window.csv` (116 matches with results)
  - `reports/tuning_batch_2bund_superlig_summary.csv`
  - `reports/tuning_batch_2bund_superlig_balanced_summary.csv`
- Applied threshold overrides:
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["2. Bundesliga"] = 0.33`
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["2. Bundesliga"] = 0.29`
  - `OU25_OVER_THRESHOLD_BY_LEAGUE["SÃ¼per Lig"] = 0.41`
- Kept default intentionally:
  - `BTTS_YES_THRESHOLD_BY_LEAGUE["SÃ¼per Lig"]` remains default (`0.50`) due weak/unstable gain in balanced scan.

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
  - `$env:API_FOOTBALL_HISTORY_DAYS="365"; $env:API_FOOTBALL_LEAGUE_IDS="39,40,140,78,135,136,61,62,88,94,203,283"; $env:API_FOOTBALL_MAX_REQUESTS="7500"; $env:API_FOOTBALL_RATE_PER_MINUTE="120"; python predict.py --rebuild-history --max-date YYYY-MM-DD`

