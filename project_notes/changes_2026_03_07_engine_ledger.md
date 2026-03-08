Engine Change Ledger (as of 2026-03-07)

Purpose
- Single clear record of implemented engine changes in current cycle.
- Includes warnings/edge behaviors that must be understood before running production predictions.

Scope covered in this ledger
- Strict explicit mu engine (active default).
- ELO revamp (V2).
- xG proxy revamp path and active behavior.
- Pressure V2 revamp with extended API stats.
- API sync quality-gate behavior (soft warnings vs strict fail).
- Prediction output quality flags (row-level BAD/WARN/OK).
- Poisson next step plan (enhancement path, not full replacement yet).
- Legacy model-path archival and cleanup.

0) Strict explicit mu engine (implemented, active default)
- What changed:
  - Mu no longer depends on hidden XGBoost/NLM feature buckets in the active path.
  - Active runtime path computes mu only from:
    - league anchor
    - Elo module
    - xG module
    - pressure module
  - Fixed weights:
    - league anchor `40%`
    - Elo `10%`
    - xG `25%`
    - pressure `25%`
  - Missing-module behavior:
    - prediction row still generates
    - missing module is neutralized to league anchor
    - warning is surfaced in `quality_flags`
  - Wrapper pipelines no longer include legacy Frankenstein/NLM train steps.
- Why:
  - Remove hidden channels (`team_form`, `other`, odds/h2h leakage into mu path).
  - Enforce a professional, auditable module contract.
- Files:
  - `cgm/strict_mu_engine.py`
  - `cgm/predict_upcoming.py`
  - `predict.py`
  - `scripts/run_full_pipeline_and_send_telegram.ps1`
  - `scripts/run_latam_pipeline.ps1`
  - `config.py`

0.1) Legacy model-path archival (implemented)
- What changed:
  - Archived legacy files:
    - `archive/legacy_engine/cgm/build_frankenstein.py`
    - `archive/legacy_engine/cgm/train_frankenstein_mu.py`
    - `archive/legacy_engine/cgm/league_weighting_nlm.py`
    - `archive/legacy_engine/cgm/train_calibration.py`
    - `archive/legacy_engine/scripts/audit_no_odds.py`
    - `archive/legacy_engine/scripts/audit_training_leakage_guard.py`
  - Removed active script references and fallback branches to those files.
- Why:
  - Keep one production path only (strict 4-module mu) and prevent accidental fallback.

1) ELO V2 (implemented, active)
- What changed:
  - League-aware ELO controls (league-specific home advantage and K behavior).
  - Better match-context handling (upset scaling, new-team handling, capped goal-diff effects).
  - Per-row ELO trace fields are written for auditability.
- Why:
  - Improve realism across leagues and reduce one-size-fits-all ELO drift.
- Files:
  - `scripts/calc_cgm_elo.py`
  - `cgm/calculate_elo.py`
  - `project_notes/elo_v2_simple.md`

2) xG proxy updates (implemented path; conservative activation)
- What changed:
  - xG pipeline supports enhanced feature mode (v2 candidate path) and strict anti-leak handling.
  - Team over-smoothing logic that hid team identity was removed from active path.
  - Current default remains conservative for stability unless a challenger proves better.
- Why:
  - Keep model responsive to real team identity (attack/defense profile), especially early season.
- Files:
  - `cgm/build_xg_proxy.py`
  - `scripts/evaluate_xg_proxy_candidates.py`
  - `config.py`

3) Pressure V2 (implemented, active)
- What changed:
  - Extended API stats ingestion added (when provider returns them):
    - shots off goal, blocked shots, goal attempts
    - attacks, dangerous attacks
    - plus additional flow/discipline fields for future modules
  - Pressure form upgraded to dynamic weighted composition:
    - core stats always active (shots, shots on target, corners, possession)
    - optional stats included only when present
    - no fake data fill-in when optional stats are missing
- Why:
  - Add richer match-control signal without breaking leagues that do not expose full stats.
- Files:
  - `scripts/sync_api_football.py`
  - `cgm/pressure_inputs.py`
  - `cgm/backfill_match_stats.py`
  - `cgm/pressure_form.py`
  - `cgm/predict_upcoming.py`
  - `config.py`

4) Sync quality-gate behavior (implemented)
- What changed:
  - Added `--strict-quality-gate` mode in API sync.
  - Default mode is soft-quality:
    - writes outputs
    - logs warnings if quality thresholds are missed
    - does not stop entire pipeline for partial data quality
  - Hard safety retained:
    - if total fixtures loaded is zero, sync fails with non-zero exit.
    - strict mode still fails on any quality-gate failure.
- Why:
  - Prevent "nothing generated" runs due to partial coverage while still blocking true outage/empty-data states.
- Files:
  - `scripts/sync_api_football.py`

Warning discussed and now documented
- Soft mode can continue with warnings when coverage is below threshold.
- This is intentional for operational continuity.
- If you want hard-stop behavior, run sync with `--strict-quality-gate`.
- True outage guard remains hard fail when fixtures loaded = 0.

5) Prediction table data-quality flags (implemented, active)
- What changed:
  - `reports/cgm_upcoming_predictions.csv` now includes row-level quality diagnostics:
    - `quality_status` (OK/WARN/BAD)
    - `quality_critical` (0/1)
    - `quality_issue_count`
    - `quality_flags` (semicolon-separated reasons)
- Current flag logic (high-level):
  - BAD: critical evidence gaps (example: missing core pressure/xG evidence, low ELO evidence).
  - WARN: non-critical issues (example: odds missing).
  - OK: no quality issues detected by current rules.
- Why:
  - Make weak fixtures visible directly in the prediction table, not only in logs.
- Current threshold update:
  - Global ELO evidence floor used by `elo_evidence_low` was reduced from `8` to `3`.
  - Implemented as `config.PIPELINE_MIN_MATCHES` (default), consumed by `cgm.predict_upcoming` (`--min-matches`).
- File:
  - `cgm/predict_upcoming.py`

6) Training leakage guard update for new raw stats (implemented)
- What changed:
  - Expanded feature bans/leak filters to prevent newly added raw fixture stats from slipping into training as leakage-prone direct inputs.
- Why:
  - Keep training/inference integrity after schema extension.
- File:
  - `archive/legacy_engine/cgm/train_frankenstein_mu.py`

7) Runtime behavior to remember
- Optional stats missing in some leagues/matches is normal with this provider.
- Engine behavior in that case:
  - Pressure V2 falls back to core pressure components.
  - Pipeline still runs (unless strict gate or zero-fixture hard fail is triggered).
- Where to inspect issues:
  - Sync quality report: `data/api_football/fixture_quality_report.json`
  - Pipeline run log: `reports/run_log.jsonl`
  - Per-fixture status: `reports/cgm_upcoming_predictions.csv` quality columns.

8) Poisson V2 (implemented, active)
- What changed:
  - Upgraded probability engine now applies:
    - dispersion mixture layer (overdispersion control),
    - low-score correction (Dixon-Coles style),
    - light home-away dependence coupling.
  - BTTS probability now comes from the same joint score engine (not independence shortcut).
  - Prediction output now exposes Poisson V2 runtime params per row:
    - `poisson_v2_enabled`, `poisson_v2_disp_alpha`, `poisson_v2_dep_strength`, `poisson_v2_dc_rho`.
- Config controls added:
  - `POISSON_V2_ENABLED`
  - `POISSON_V2_MAX_GOALS`
  - `POISSON_V2_DISPERSION_ALPHA_DEFAULT` / `_BY_LEAGUE`
  - `POISSON_V2_DEP_STRENGTH_DEFAULT` / `_BY_LEAGUE`
  - `POISSON_V2_DC_RHO_DEFAULT` / `_BY_LEAGUE`
- Validation note:
  - `scripts/audit_calculations.py` updated to validate Poisson V2 probability coherence.
- Files:
  - `cgm/predict_upcoming.py`
  - `config.py`
  - `scripts/audit_calculations.py`

9) Full Poisson revamp (future, DEV only)
- Full replacement research/implementation remains a future `dev` track.
- Reference:
  - `project_notes/dev_poisson_full_revamp_plan.md`

10) League-specific NLM/stacker (archived)
- What changed:
  - Added a league-aware correction layer trained on top of base mu outputs:
    - global combiner + league residual models with shrinkage toward global.
  - Training now writes optional combiner files:
    - `frankenstein_mu_home*_nlm.pkl`
    - `frankenstein_mu_away*_nlm.pkl`
  - Prediction can load these files when enabled.
- Safety status:
  - Active strict mu engine bypasses this path entirely.
  - No runtime toggle remains in active config.
  - Historical holdout check was weaker than strict base at time of archival.
- Files:
  - `archive/legacy_engine/cgm/league_weighting_nlm.py`
  - `archive/legacy_engine/cgm/train_frankenstein_mu.py`


