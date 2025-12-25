Run log (human-readable, ELO-focused)

2025-12-19 (UTC)
- Full rebuild: match_history -> baselines -> Elo (cutoff today UTC) -> backfill stats -> Frankenstein -> train -> predict.
  - `python predict.py --rebuild-history`
- Pressure stats audit:
  - Source file `CGM data/goals statistics.csv`: 385 rows, sezonul=`26.0`, date range 2025-08-15..2026-05-24.
  - Date parsing ambiguity (important): `datameci` is `MM/DD/YYYY` with 134 rows that are ambiguous-but-parseable (e.g. `10/3/2025`). `pd.to_datetime(..., dayfirst=True/False)` does not detect this because pandas infers `%m/%d/%Y` for the whole Series (ignores `dayfirst` and warns). Backfill now parses with explicit formats and logs `[date_parse]` counts + chosen format.
  - Of those 385 rows: 160 are <= today UTC (played so far) and 220 are future fixtures (not in history).
  - Safety: `cgm/backfill_match_stats.py` now parses `datameci` deterministically and filters stats rows to the history cutoff before joining (prevents future stats leakage).
  - Added `pressure_usable` flag (1 only when all split stats are present) into `data/enhanced/cgm_match_history_with_elo_stats.csv` and as a model feature.
  - Coverage in `data/enhanced/cgm_match_history_with_elo_stats.csv`:
    - England / Premier L / 2025-2026: stats coverage = 1.00
    - All older seasons: stats coverage = 0.00
    - Overall coverage across the full 2018..2025 history: ~0.0567
  - Practical impact: `press_form_H/A` is neutral (0.5) for ~95% of training rows; Pressure is only informative for the current season that has per-match stats.
- Outputs refreshed:
  - `data/enhanced/cgm_match_history.csv` (2820 rows; future rows filtered)
  - `data/enhanced/cgm_match_history_with_elo.csv` (Elo zeros eliminated; cutoff applied)
  - `data/enhanced/cgm_match_history_with_elo_stats.csv` (backfill matched_rows=160)
  - `data/enhanced/frankenstein_training.csv` (2820 rows)
  - `models/frankenstein_mu_home.pkl`, `models/frankenstein_mu_away.pkl` retrained
  - `reports/cgm_upcoming_predictions.csv` regenerated (530 rows)

2025-12-20 (UTC)
- Pressure reliability: `press_n_H/A` counts matches in window (neutral fallback makes it look “healthy” even when stats are missing). Added `press_stats_n_H` / `press_stats_n_A` to count how many prior matches in the Pressure window actually had complete stats.
  - Implemented in `cgm/pressure_form.py` (rolling sums over `pressure_usable` / stats-present indicator; adds `_press_stats_n_*_post` for inference snapshots).
  - Wired into inference feature vectors in `cgm/predict_upcoming.py`.
- Divergence Elo consistency: `cgm/pressure_divergence.py` now infers home advantage per `(country, league)` (with global fallback) instead of a single global median.
- Baselines sanity: `cgm/build_baselines.py` computes `home_bonus_elo` as a heuristic feature, but `EloDiff`/bands intentionally use the fixed Elo home-advantage baseline (65) to match the Elo rebuild and the “infer HA from EloDiff” logic.
- Team registry bugfix: `cgm/team_registry.py` previously cross-paired unrelated code/name columns (esp. in `goal statistics 2.csv`), producing corrupted aliases (e.g., `Crystal Palace -> Arsenal`) and incorrect match history.
  - Fix: only collect aligned `(code, name)` pairs from known columns, normalize codes (strip `.0`), rebuild `name_to_code` safely.
- Pipeline rerun: `python predict.py --rebuild-history --max-date 2025-12-19`
  - `data/enhanced/frankenstein_training.csv`: cols 109 (added `press_stats_n_*`)
  - models: feature contract union 98
  - match history sanity: no `home==away` rows; no code/name mismatch rows
- Pressure input audit: `cgm/pressure_inputs.ensure_pressure_inputs()` did not change any canonical split columns in `data/enhanced/cgm_match_history_with_elo_stats.csv` (0/2820 rows changed).
- Pressure no-leak audits:
  - Added `scripts/audit_pressure.py` to quickly check coverage + “no-leak” tripwires.
  - Shift identity checks (pre == previous post) are clean for all Pressure features tested (0 mismatches).
  - Training matrix contains no `_press_*` helper columns (drop confirmed).
  - Current-match raw-stat reconstruction check: 0 exact matches for shots/SOT/corners dominance; 1/147 exact match for possession dominance (expected coincidence, not leakage).
- Pressure input mapping hardening:
  - `cgm/pressure_inputs.ensure_pressure_inputs()` now fills canonical columns only where missing (no overwrites); optional `ensure_pressure_inputs_fill(..., overwrite=True)` kept for debugging.
- Full rebuild (post-audits): `python predict.py --rebuild-history --max-date 2025-12-19`
  - `data/enhanced/frankenstein_training.csv`: rows 2820, cols 109
  - models retrained; metrics RMSE_home=1.253, RMSE_away=1.145
  - `reports/cgm_upcoming_predictions.csv`: 530 rows

2025-12-12 (UTC)
- Pipeline run: recomputed Elo -> rebuilt Frankenstein -> retrained mu models -> predicted upcoming.
- Commands executed (repo root):
  - `python -m scripts.calc_cgm_elo --history data/enhanced/cgm_match_history.csv --out data/enhanced/cgm_match_history_with_elo.csv --max-date 2025-12-12`
  - `python -m cgm.backfill_match_stats --history data/enhanced/cgm_match_history_with_elo.csv --out data/enhanced/cgm_match_history_with_elo_stats.csv`
  - `python -m cgm.build_frankenstein --data-dir data/enhanced --match-history cgm_match_history_with_elo_stats.csv --out data/enhanced/frankenstein_training.csv`
  - `python -m cgm.train_frankenstein_mu --data data/enhanced/frankenstein_training.csv --out-dir models`
  - `python -m cgm.predict_upcoming --history data/enhanced/cgm_match_history_with_elo_stats.csv --models-dir models --out reports/cgm_upcoming_predictions.csv`
- Elo state: zeros eliminated; merge artefacts dropped (`*_x/_y`); canonical `elo_home/away/diff` and `_calc` aliases retained. Cutoff applied (no future leakage).
- Feature alignment: calc/name aliases dropped before training; similarity features use Gaussian kernel (sigma default 60) with wsum/neff; train/inference feature sets now identical (81 model features; training CSV has 107 columns including targets/meta).
- Model metrics: RMSE_home ≈ 1.265, RMSE_away ≈ 1.123 (Poisson objective, time-based split). Model hashes: home SHA256 3869c4162cea728f29e7c920eaf5c381fef83d42e4a10d59ac6afc29392b4633; away SHA256 a6a54c31485f32f172327109ddbe469d93d3e47044b2233def8382486ae763b3.
- Predictions: 530 upcoming rows written to `reports/cgm_upcoming_predictions.csv`; no missing-feature warnings. GroupBy FutureWarnings remain (pandas notice only).
- Watchouts: keep running calc_cgm_elo with a max-date before building features to avoid future leakage; monitor neff for sparse opponents if sigma is tightened.

2025-12-20 (UTC) - Milestone 3 (xG proxy "Sniper")
- Full rebuild + retrain (as-of cutoff): `python predict.py --rebuild-history --max-date 2025-12-19`
  - Stats source: `CGM data/cgmbetdatabase.xls` (Excel; requires `xlrd==2.0.1`); future rows safely dropped in backfill.
  - New artifact: `data/enhanced/cgm_match_history_with_elo_stats_xg.csv` (adds `xg_proxy_H/A`, `xg_usable`, helpers).
  - Training: `data/enhanced/frankenstein_training.csv` rows 2820, cols 131; models feature union 120; metrics RMSE_home=1.260, RMSE_away=1.139.
  - Inference: `cgm/predict_upcoming.py` now injects xG rolling form + Pressure-vs-xG disagreement features (div_px/sterile/assassin) so train/inference stay aligned.
  - Safety: `cgm/build_match_history.py` now takes `--max-date` so as-of runs can't accidentally include same-day matches in baselines/league averages.
- Audits:
  - `python -m scripts.audit_pressure --cutoff 2025-12-19` clean (shift identity checks pass; no `_press_*` in training).
  - `python -m scripts.audit_xg` clean (xg shift identity checks pass; no `_xg_*`/`xg_proxy_*` in training).

2025-12-20 (UTC) - Milestone 4 (Pick Engine)
- Added deterministic pick selection layer (1X2 + O/U 2.5):
  - `cgm/pick_engine.py` consumes `reports/cgm_upcoming_predictions.csv` and writes `reports/picks.csv` + `reports/picks_debug.csv` (candidate-level gate/score debug).
  - Hard gates: odds sanity, mu_total bounds, minimum evidence (`neff_sim_*`, `press_stats_n_*`, `xg_stats_n_*`), sterile/assassin risk rules, minimum EV thresholds, single best pick per fixture.
  - Tie-breaks: if candidates tie, prefers higher EV, higher `neff_min`, then a fixed market priority order (OU25_OVER, OU25_UNDER, 1X2_HOME, 1X2_AWAY, 1X2_DRAW).
- Inference export upgrade:
  - `cgm/predict_upcoming.py` now includes pick-engine-required columns (fixture_datetime, league, reliability, risk flags) in `reports/cgm_upcoming_predictions.csv`.
- Pipeline wiring:
  - `predict.py` now runs the pick engine as the last step (also in `--predict-only` mode).
- Validation run:
  - `python predict.py --predict-only` -> predictions (530 rows) + picks (378 rows) written.
  - `python -m scripts.audit_picks` clean (deterministic hash match; no gate violations; stake tiers consistent).

2025-12-21 (UTC) - Live scope filter (no past predictions)
- Added deterministic live scope to stop "schedule dump" files from producing retro predictions/picks:
  - Defaults in `config.py` (EPL 2025-26 window + horizon).
  - `cgm/predict_upcoming.py` now filters fixtures strictly after `run_asof_datetime` and logs `UPCOMING_SCOPE` counts.
  - `cgm/pick_engine.py` re-applies scope internally and requires `run_asof_datetime` + `scope_*` columns.
- Test run (using the same `CGM data/upcoming - Copy.CSV`, which contains only played matches up to 2025-12-08):
  - `python predict.py --max-date 2025-12-19 --predict-only` -> predictions rows=0, picks rows=0 (correct; no future fixtures in feed).
  - `python -m scripts.audit_upcoming_feed --as-of-date 2025-12-19` confirms all 530 raw rows were dropped by the "drop past" filter.
- Bugfix (inference odds/probability scaling):
  - Training uses `p_home/p_draw/p_away/p_over/p_under` on a 0..100 (%) scale (from CGM `homeprob/drawprob/...`).
  - Inference previously used 0..1 implied probabilities from odds, which is out-of-scale and can distort predictions.
  - Fixed: `cgm/predict_upcoming.py` now feeds implied `p_*` values as percentages (0..100) to match training.
  - Note: if your upcoming export has odds = 0.0, both the model and pick engine behave as if odds are missing (predictions become less reliable; picks are correctly blocked by odds sanity gates).
- Added odds-free internal model option:
  - `cgm/train_frankenstein_mu.py --variant no_odds` trains models without any market features (odds + CGM `p_*`).
  - `cgm/predict_upcoming.py --model-variant no_odds` predicts using those models (mu/probabilities independent of odds; EV/picks still require odds).
  - `scripts/audit_no_odds.py` verifies mu/probabilities are invariant to changes in the upcoming odds columns.
- Milestone 7.1 (goals-only picks):
  - Added `cgm/pick_engine_goals.py` (OU2.5 + optional BTTS; no 1X2 picks) and `scripts/audit_picks_goals.py`.
  - `predict.py` now supports `--pick-engine goals` to run the goals-only pick engine.
  - `cgm/predict_upcoming.py` now exports BTTS reporting fields (`p_btts_*`, `odds_btts_*`, `EV_btts_*`) when the CGM upcoming file contains `gg/ng` odds.
