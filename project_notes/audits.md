Audits and Validation (Current)

Core checks
- Detailed audit log:
  - `python scripts/audit_detailed.py`
  - Produces a non-technical log in `reports/audits/` with Elo/xG/Pressure coverage.
  - Also saves JSON summary and per-league CSV coverage.

- Upcoming feed scope:
  - `python -m scripts.audit_upcoming_feed --as-of-date YYYY-MM-DD`
  - Validates:
    - Upcoming fixtures are future-only.
    - Scope matches the configured next-round window.

- No-odds invariance:
  - `python -m scripts.audit_no_odds --as-of-date YYYY-MM-DD`
  - Verifies probabilities are stable when odds are not used.
  - Validates:
    - Model output does not change when odds inputs change.

- Multi-league coverage snapshot:
  - `python scripts/audit_multi_league.py`
  - Validates:
    - History and upcoming data include the expected leagues.

Scope note
- These audits cover the core live pipeline (API sync, upcoming scope, no-odds output stability).
- Additional niche audits were archived during cleanup and are not part of the current default runbook.

Backtest checks
- Generate backtest:
  - `python -m scripts.run_backtest --league "Premier L" --season "2025-2026" --start-date "2025-09-01"`
  - Validates:
    - End-to-end training + prediction on historical fixtures for a league/season.
- Optional leakage sanity:
  - `python scripts/audit_backtest.py`
  - Validates:
    - Obvious leakage in backtest pipeline (sanity checks).

Business-facing report
- `python scripts/generate_business_report.py --rounds 5 --upcoming-limit 20`
- This is the easiest final validation for non-technical review.
  - Validates:
    - Report renders without errors and includes recent results + next round predictions.
