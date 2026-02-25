Branch Guide (Current)

Last updated: 2026-02-24

Current branch topology
- `main`: stable baseline branch.
- `test`: integration/validation branch.

Current lineage (at branch pointers)
- `main` -> commit `05430b3`
- `test` -> commit `05430b3`

How each branch is built

1) `main`
- Intent: stable reference branch for known-good pipeline state.
- Built from: validated changes promoted from `test`.
- Promotion flow (non-destructive):
  - `git checkout main`
  - `git merge --ff-only test`
  - `git push origin main`

2) `test`
- Intent: staging branch for integration checks and pre-release verification.
- Built from: `main` plus feature/testing updates.
- Typical update flow:
  - `git checkout test`
  - implement and validate selected changes
  - run audits/backtests
  - push when validated

Branch strategy by branch

1) `main` strategy (Release/Stable)
- Goal: stable, business-safe pipeline only.
- Rules:
  - no experimental logic
  - only validated changes promoted from `test`
  - keep commands and outputs predictable for daily use

2) `test` strategy (Integration/QA)
- Goal: validate release candidates before promotion to `main`.
- Rules:
  - run full audits and no-leak backtests
  - confirm league coverage, team normalization, and report outputs
  - keep baseline strategy: league anchor + team/similar-ELO blend
  - reject changes with regressions on key metrics

Branch usage policy
- Do all ongoing changes in `test` unless a temporary feature branch is explicitly created.
- Use `test` for verification and release-candidate checks.
- Keep `main` clean and promotion-only.
