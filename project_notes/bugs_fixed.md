Bugs Fixed (Current Snapshot)

Date: 2026-02-18

1. API data-dir mismatch between orchestrator and sync script
- Issue: `predict.py` passed API data dir through env, but sync default path could diverge.
- Fix: sync now accepts env/config defaults and explicit CLI wiring from orchestrator.
- Files: `predict.py`, `scripts/sync_api_football.py`.

2. API free-tier guardrails not centralized in runtime flow
- Issue: request/rate limits could drift from config.
- Fix: orchestrator now passes free-tier defaults (or env overrides) directly to sync step.
- Files: `predict.py`, `config.py`.

3. Report output collision for league-specific runs
- Issue: custom report runs could overwrite default report CSV exports in `reports\`.
- Fix: report CSV filenames now follow output stem.
- File: `scripts/generate_business_report.py`.

Date: 2026-02-25

4. Audit scripts could fail for non-strategy reasons (encoding / empty artifacts)
- Issue: audit failures were possible from console encoding and empty-file edge cases, not from model/data correctness.
- Fix:
  - ASCII-safe output and robust empty-file handling added.
  - Multi-league audit now reads UTF-8 first and canonicalizes league names.
  - Upcoming scope audit now mirrors production filter chain (including dedupe + next-round).
- Files:
  - `scripts/audit_backtest.py`
  - `scripts/audit_calculations.py`
  - `scripts/audit_multi_league.py`
  - `scripts/audit_upcoming_feed.py`

5. Missing hard correctness checks for leakage/data health/skip reconciliation
- Issue: core audits existed but did not explicitly fail on training-leak signatures, weak live evidence coverage, or skip-accounting drift.
- Fix: new audits added and integrated into audit runner.
- Files:
  - `scripts/audit_training_leakage_guard.py`
  - `scripts/audit_prediction_data_health.py`
  - `scripts/audit_skip_reconciliation.py`
  - `scripts/run_all_audits.py`

6. Picks could be generated without a mandatory critical-audit gate
- Issue: pick emission could proceed even if critical integrity checks were not enforced immediately pre-pick.
- Fix: `predict.py` now runs mandatory critical audits before any pick engine call.
- Files:
  - `predict.py`
  - `scripts/run_all_audits.py`
