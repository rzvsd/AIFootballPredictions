Poisson Full Revamp Plan (DEV branch reference)

Status
- This is a future DEV plan.
- Not active in production until champion-vs-challenger validation is passed.

Goal
- Replace simple independent Poisson assumptions with a stronger probability engine that:
  - handles overdispersion,
  - improves low-score realism,
  - captures limited score dependence,
  - improves calibration for OU2.5 and BTTS across leagues.

Important principle
- Do not replace production blindly.
- Use controlled promotion:
  - current engine = Champion
  - full revamp = Challenger
  - promote only on objective holdout wins.

Architecture options (ordered by risk/benefit)

Option A (recommended first): Poisson V2 Enhanced (incremental)
- Keep existing mu generation path.
- Add:
  1) dispersion correction (negative-binomial style),
  2) low-score correction (Dixon-Coles style),
  3) light dependence adjustment.
- Pros:
  - lowest migration risk,
  - keeps interpretability and speed,
  - easiest rollback.
- Cons:
  - not as flexible as full Bayesian/coupled models.

Option B: Full bivariate count model
- Joint home/away model with explicit covariance term.
- Pros:
  - cleaner joint score math.
- Cons:
  - harder fitting, higher maintenance cost.

Option C: Hierarchical Bayesian count model
- Team/league random effects + dispersion + dependence.
- Pros:
  - strongest statistical framework.
- Cons:
  - highest complexity/latency, production overhead.

Recommended implementation path

Phase 0 - Baseline freeze
- Freeze current champion outputs and metrics snapshot:
  - log-loss, Brier, RPS, OU/BTTS accuracy, calibration bins.
- Save frozen artifacts for repeatability.

Phase 1 - Dispersion module
- Add dispersion parameter per league (and optionally season).
- Start with constrained range and shrink toward league prior when sample is small.
- Success criteria:
  - lower log-loss or Brier in at least 60 percent of target leagues,
  - no major degradation in top leagues.

Phase 2 - Low-score correction
- Add Dixon-Coles style adjustment for 0-0, 1-0, 0-1, 1-1 outcomes.
- Calibrate by league (global default + league override).
- Success criteria:
  - improved calibration in low-total goal bins,
  - improved OU2.5 around edge probabilities (0.45-0.60).

Phase 3 - Dependence adjustment
- Add light dependence term for home-away goal interaction.
- Keep constrained to avoid unstable tail effects.
- Success criteria:
  - improved BTTS calibration and log-loss,
  - no distortion of total-goals distribution.

Phase 4 - Calibration and reliability
- Per-league calibration layers:
  - probability calibration curves,
  - dispersion sanity checks,
  - EV realism checks against available odds.
- Add reliability tags to predictions.

Phase 5 - Promotion gate
- Challenger promoted only if:
  - improves weighted aggregate log-loss and Brier,
  - passes league-level no-regression guardrails in top leagues,
  - preserves operational latency budget.

Data requirements and constraints
- Required:
  - robust history window by league (ideally full season minimum),
  - stable team-name normalization,
  - complete core stats (shots, SOT, corners, possession, goals),
  - odds where EV evaluation is needed.
- Optional:
  - attacks/dangerous attacks/etc. to support pressure-enhanced features.
- Known constraint:
  - provider may not return all optional stats for every league/match.

Validation design (must use time split)
- Train/test split by time (no random split).
- Evaluate on:
  - log-loss,
  - Brier,
  - RPS,
  - OU/BTTS hit rate,
  - calibration plots (especially around market decision thresholds).
- Additional guardrails:
  - monitor rate of extreme probabilities,
  - monitor predicted total-goals distribution by league.

Operational guardrails
- Keep full rollback path to champion engine.
- Maintain deterministic outputs for same inputs.
- Preserve auditability:
  - prediction traces,
  - configuration snapshot,
  - model hash/version logging.

Branch strategy
- Implement full revamp only in `dev` branch.
- Never merge direct to `main`.
- Promotion route:
  - `dev` -> `test` after validation
  - `test` -> `main` only after approval + no-regression checks.

Deliverables checklist for DEV revamp
- [ ] design doc with formulas and constraints
- [ ] feature/schema update doc
- [ ] model training script updates
- [ ] inference integration updates
- [ ] calibration script updates
- [ ] backtest report by league
- [ ] promotion/no-promotion recommendation

Recommended immediate next action
- Execute Option A (Poisson V2 Enhanced) first as challenger.
- Only start Option B/C if Option A cannot deliver target calibration improvements.


