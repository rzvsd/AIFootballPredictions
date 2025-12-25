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
