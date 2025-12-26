# Bugs Fixed (CGM-only)

This is a running list of bugs found and fixed in the CGM-only engine. Each item is brief and points to the root cause + the fix location.

## 2025-12-21

1) Inference `p_*` scale mismatch (0..1 vs 0..100)
   - Symptom: implausible outcomes and “home/under bias”, especially when odds were missing/zero in the upcoming feed.
   - Root cause: training data stores `p_home/p_draw/p_away/p_over/p_under` as percentages (0..100) from CGM (`homeprob/drawprob/...`), but inference was feeding implied probabilities as fractions (0..1).
   - Fix: scale implied probabilities to 0..100 inside `cgm/predict_upcoming.py` so inference matches the training feature scale.

2) Retro predictions/picks caused by “schedule dump” upcoming file
   - Symptom: `reports/cgm_upcoming_predictions.csv` and `reports/picks.csv` included past fixtures (older season rows).
   - Root cause: `CGM data/upcoming - Copy.CSV` can contain a full season schedule/results, and the pipeline previously predicted every row.
   - Fix: add deterministic “live scope” filtering in `cgm/predict_upcoming.py` (strict `fixture_datetime > run_asof_datetime`, plus league/country + season-window + optional horizon). Enforce scope again in `cgm/pick_engine.py`. Added `scripts/audit_upcoming_feed.py` + scope assertions in `scripts/audit_picks.py`.

3) Pick engine determinism / tie ambiguity
   - Symptom: tie scenarios could select a different market depending on incidental ordering; hard to debug selection without a score.
   - Root cause: no explicit market priority tie-break and no exported selection score.
   - Fix: export `score` to `reports/picks.csv`, and add deterministic tie-break order (score -> EV -> neff -> fixed market priority list) in `cgm/pick_engine.py`.

4) Corrupted team registry mappings (wrong team codes)
   - Symptom: incorrect team aliasing (e.g., “Crystal Palace -> Arsenal”), which can break deterministic joins and degrade training/inference.
   - Root cause: registry builder was effectively cross-pairing unrelated code/name columns in some CGM exports.
   - Fix: `cgm/team_registry.py` now only collects aligned `(code, name)` pairs from known column pairs and normalizes codes (e.g., strips `.0`).

5) Pressure reliability signal was misleading (`press_n_*` counted matches, not evidence)
   - Symptom: `press_n_H/press_n_A` looked “healthy” even when match stats were missing, because missing stats fall back to neutral dominance (0.5).
   - Root cause: rolling counts were based on `_press_index_*` which always exists.
   - Fix: add `press_stats_n_H/press_stats_n_A` (rolling usable-stat counts) in `cgm/pressure_form.py` and export/use them in `cgm/predict_upcoming.py` + `cgm/pick_engine.py` quality gates.

6) Stats backfill: ambiguous date parsing + future rows (future-leakage risk)
   - Symptom: low join rate in older seasons; presence of future-dated rows in CGM stats exports.
   - Root cause: ambiguous date strings (parseable both ways) + stats sources that include future fixtures; without strict handling this can silently reduce joins or leak future stats.
   - Fix: `cgm/backfill_match_stats.py` parses dates deterministically and drops future stats rows relative to the history cutoff before joining; adds `pressure_usable` flag for downstream gating/audits.

7) Pressure input normalization overwrite hazard
   - Symptom: canonical split columns could be overwritten if multiple source schemas are present.
   - Root cause: normalization logic can have multiple mapping candidates (split vs packed vs alternate names).
   - Fix: `cgm/pressure_inputs.ensure_pressure_inputs()` is fill-only by default (no overwrites); an explicit `overwrite=True` path exists for debugging.

8) Goal-timing (AGS) league naming mismatch (timing profiles empty)
   - Symptom: Milestone 7.2 timing profiles showed `rows_used=0` even though `CGM data/AGS.CSV` had matches.
   - Root cause: AGS export encodes league as `England-Premier L` while live scope uses `Premier L`.
   - Fix: normalize AGS league values by stripping the `Country-` prefix before filtering in `cgm/goal_timing.py`.

## 2025-12-25

9) Train/inference skew in sterile/assassin flag computation
   - Symptom: sterile_flag and assassin_flag could differ between training and live prediction for the same fixture, causing model-strategy misalignment.
   - Root cause: inference code (`predict_upcoming.py` lines 778-781) gates flags on evidence (`xg_stats_n > 0`), but training code (`pressure_xg_disagreement.py` lines 140-143) did not, so flags were set even when xG z-scores had no data backing.
   - Fix: add evidence gating (`xg_stats_n > 0`) to the training code in `cgm/pressure_xg_disagreement.py` to match inference logic.
   - Impact: after rebuild, expect fewer sterile/assassin flags (only confident ones remain).

10) Scattered threshold constants (maintenance hazard)
    - Symptom: tuning thresholds required editing 4+ files (`pick_engine.py`, `pick_engine_goals.py`, `audit_picks.py`, `audit_picks_goals.py`), risking inconsistencies.
    - Root cause: each module defined its own local constants for NEFF_MIN, ODDS_MIN, EV thresholds, etc.
    - Fix: centralize all pick engine constants in `config.py` with clear naming (`*_FULL` for 1X2+OU engine, `*_GOALS` for goals-only engine). All modules now import from config.
    - Impact: change thresholds in one place; all modules stay in sync.

11) Inconsistent hash algorithm (MD5 vs SHA256)
    - Symptom: pick engines used MD5 while `predict_upcoming.py` used SHA256 for file integrity, causing confusion in logs.
    - Root cause: different modules were written at different times with different hash choices.
    - Fix: standardize all file hashing on SHA256 (`file_sha256()` function) for consistency with `predict_upcoming.py`.
    - Files updated: `cgm/pick_engine.py`, `cgm/pick_engine_goals.py`, `scripts/audit_picks.py`, `scripts/audit_picks_goals.py`.

12) NaN propagation in Elo similarity blending
    - Symptom: when history is empty or has all-NaN values, blending would produce NaN output passed to the model.
    - Root cause: `gf.mean()` returns NaN when series is empty; blending with NaN propagates garbage.
    - Fix: guard against NaN in `elo_similarity.py` by checking if base_gf/base_ga are NaN before blending; skip blending if so.

13) Division by zero in shot quality calculation
    - Symptom: when shots=0, shot quality was computed as xG/1 instead of NaN, producing misleading values.
    - Root cause: `.where(shots > 0, 1.0)` replaced 0 with 1 instead of NaN.
    - Fix: use `np.where(shots > 0, xg/shots, np.nan)` in `xg_form.py` so zero-shot cases return NaN.

14) Silent exception swallowing in pressure form
    - Symptom: bad data was silently replaced with neutral (0.5) without any logging, hiding data bugs.
    - Root cause: bare `except Exception: return 0.5` patterns.
    - Fix: add logging in `pressure_form.py` (`_dom_ratio`, `_poss_share`) before returning neutral fallback.

15) Hardcoded magic numbers scattered across modules
    - Symptom: tuning elo_similarity, pressure_form, or goal_timing required editing those files directly.
    - Root cause: constants defined locally in each module.
    - Fix: add `ELO_SIM_MIN_EFF`, `PRESSURE_W_*`, `TIMING_*` constants to `config.py`.

16) Team name normalization type safety
    - Symptom: if `normalize_team()` received None or non-string, downstream joins could fail.
    - Root cause: function assumed string input.
    - Fix: add type checks and logging in `team_registry.py` for non-string inputs.

17) Poisson log-likelihood silently clipping bad predictions
    - Symptom: model predicting mu=0 was silently clipped to 1e-6 without warning.
    - Root cause: no logging when predictions are pathologically bad.
    - Fix: add warning log in `train_frankenstein_mu.py` when mu < 0.01 before clipping.

18) Dead import (`import random`)
    - Symptom: unused import at top of `predict_upcoming.py`.
    - Root cause: leftover from development.
    - Fix: removed the import.

19) Unseen teams silently skipped without warning
    - Symptom: if a team in the upcoming feed was never seen in history, it was silently skipped with no logging.
    - Root cause: no informative logging when teams are filtered out.
    - Fix: added `[UNSEEN]` and `[LOW_HISTORY]` warning logs in `predict_upcoming.py` to identify fixtures being skipped due to missing team data.

20) H2H features missing at inference time (training/inference mismatch)
    - Symptom: model trained on H2H features (`h2h_matches`, `h2h_home_win_rate`, `h2h_goals_avg`, etc.) but predictions provided zeros for these columns.
    - Root cause: `cgm/h2h_features.py` has `get_h2h_features_for_fixture()` but it was never called in `predict_upcoming.py`.
    - Fix: added import for `get_h2h_features_for_fixture` and call it in the per-fixture loop after league features.
    - Impact: H2H features now properly populated at inference time, eliminating train/predict feature drift.

## Already Addressed (Review Confirmed)
- Model feature validation: already implemented via banned_exact and banned_prefixes checks in predict_upcoming.py
- Date parsing logging: already implemented in backfill_match_stats.py with detailed format inference logging
- Pandas deprecation: cosmetic warning only, group_keys=False already used correctly

## 2025-12-26

21) Elo index alignment bug (critical - affected all teams)
    - Symptom: stored Elo values in CSV didn't match calculated values. For Man Utd, 34 out of 283 matches had wrong Elo (e.g., Nov 01 showed 1586.5 instead of correct 1558.3).
    - Root cause: `compute_elo_series()` in `scripts/calc_cgm_elo.py` returned pandas Series without indices:
      ```python
      return pd.Series(elo_home_list), pd.Series(elo_away_list)  # BUG!
      ```
      When assigned to a dataframe sorted by datetime (with non-sequential original indices like 2743, 2754...), pandas aligned by position instead of index, placing Elo values in wrong rows.
    - Fix: pass the dataframe's index to the returned Series:
      ```python
      return pd.Series(elo_home_list, index=df.index), pd.Series(elo_away_list, index=df.index)
      ```
    - Verification: after fix, 0 mismatches across all 30 teams in the dataset.
    - Impact: all previous runs had corrupted Elo data; pipeline must be rebuilt with `--rebuild-history` to fix.

