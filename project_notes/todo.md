TODO (Current Priorities)

High Priority
- [ ] Keep baseline strategy stable on `test`
  - League average anchor first, then team/similar-ELO blend.
  - Avoid team-only anchor mode unless explicitly re-opened.
  - Track `mu_total` distribution by league after each rebuild.

- [ ] Run scheduled large-window history rebuilds
  - Use `API_FOOTBALL_HISTORY_DAYS=365` for anchor stability.
  - Keep multi-league scope ids current (`39,40,140,78,135,136,61,62,88,94,203,283`).
  - Rebuild whenever history coverage is short or league averages look compressed.

- [ ] Tune OU decision-path thresholds per league
  - Tune `OVER_PATH_MIN_SCORE`, `UNDER_PATH_MIN_SCORE`, and margin threshold.
  - Keep one global default; add league overrides only where justified by backtest.
  - Compare lift/regression separately for OVER and UNDER.
  - Bundesliga thresholds tuned from evidence window (`OU=0.30`, `BTTS=0.28`); monitor next rounds for stability.
  - EPL/Serie A evidence files are available (`backtest_*_tuning_window.csv` + `tuning_threshold_scan_*.csv`); currently applied conservative overrides:
    - EPL BTTS yes threshold `0.31`
    - Serie A OU over threshold `0.54`
  - Batch-1 applied:
    - Ligue 1: `OU=0.52`, `BTTS=0.51`
    - La Liga: `OU=0.46` (BTTS kept default)
    - Liga I: `OU=0.26`, `BTTS=0.33` (recent-form pilot)
  - Batch-4 applied:
    - 2. Bundesliga: `OU=0.33`, `BTTS=0.29`
    - SÃ¼per Lig: `OU=0.41` (BTTS kept default)
  - Keep monitoring before adding more league overrides.

- [ ] Backtest validation gate for sterile override v1 (active mode)
  - Compare baseline vs sterile-enabled configuration on OU2.5 and BTTS.
  - Confirm capped behavior under boost mode (`+0.05` max sterile uplift).
  - Keep active rollout only if no material regression is observed.

- [ ] Tune sterile thresholds per league after first backtest pass
  - Keep base trigger levels (`4/10`, `6/10`, min valid `8`) as defaults.
  - Adjust only where league-specific evidence justifies it.
  - Record tuned values in config and notes.

- [ ] Ensure recent-round history coverage (last 8-10 rounds)
  - Increase history window if needed.
  - Verify sync pulls and stores recent results correctly.

- [ ] Validate stat-field coverage for xG proxy
  - Confirm shots, shots on goal, goals, corners are present for target leagues.

- [ ] Poisson V2 league tuning and challenger validation
  - Poisson V2 core is now active.
  - Next step: tune per-league Poisson V2 params (`dispersion`, `dependence`, `dc_rho`) with time-split validation.
  - Promotion rule remains champion-vs-challenger by log-loss/Brier/RPS + league no-regression checks.
  - Reference:
    - `project_notes/changes_2026_03_07_engine_ledger.md`

- [ ] Monitor strict module weights by league (no legacy stacker path)
  - Validate per-league stability on rolling backtests.
  - Adjust only explicit config thresholds/Poisson V2 params when evidence is strong.

- [ ] Improve round-scoped upcoming selection
  - Ensure next-round filtering matches league schedules.

Medium Priority
- [ ] Add weekly automation for reports and predictions export.
- [ ] Add optional Telegram delivery for business report summary.
- [ ] Monte Carlo roadmap (future rollout; not active yet)
  - Follow `project_notes/monte_carlo_future_milestones.md`
  - Start with Milestone 1 only (pre-match replay + confidence intervals)
  - Keep current analytic pipeline as fallback during rollout

- [ ] DEV full Poisson revamp research/implementation track
  - Work only in `dev` branch.
  - Use full plan reference:
    - `project_notes/dev_poisson_full_revamp_plan.md`
  - Promotion allowed only via champion-vs-challenger validation.

Low Priority
- [ ] UI polish and mobile-first read view for reports.
- [ ] Optional cloud scheduler deployment.

