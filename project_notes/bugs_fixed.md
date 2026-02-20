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
