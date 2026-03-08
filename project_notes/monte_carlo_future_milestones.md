Monte Carlo Future Milestones (Planned, Not Active)

Status
- This is a future roadmap only.
- Current production logic remains unchanged until milestones are explicitly implemented and validated.

Why add Monte Carlo
- Today, the bot gives a probability from one model pass.
- Monte Carlo adds "virtual replay" of the same match many times.
- This gives:
  - probability from replay counts,
  - confidence range (how stable that probability is),
  - better risk visibility before placing picks.

Core formulas to carry from the PDFs
- Event probability estimator:
  - `p_hat = (1/N) * sum(1{event_i})`
- Standard error:
  - `SE = sqrt(p_hat * (1 - p_hat) / N)`
- 95% confidence interval:
  - `CI95 = p_hat +/- 1.96 * SE`
- Rare-event estimator (importance sampling, later phase):
  - `p_hat_IS = (1/N) * sum(1{X_i in A} * w(X_i))`

Milestone 1 - Pre-match Monte Carlo Replay (Phase 1)
Goal
- Add Monte Carlo as an optional overlay for OU2.5 and BTTS, using existing team strengths (`mu_home`, `mu_away`).

What users will see
- New fields in predictions:
  - `p_over25_mc`, `p_under25_mc`, `p_btts_yes_mc`, `p_btts_no_mc`
  - `*_mc_se`, `*_mc_ci_low`, `*_mc_ci_high`
- Current analytic probabilities remain available side-by-side.

Acceptance gate
- Runtime remains practical for daily use.
- MC vs analytic differences are explainable on sampled rounds.
- No strategy change by default unless explicitly enabled.

Milestone 2 - Confidence-Aware Decision Gate (Phase 2)
Goal
- Use confidence width to avoid acting on noisy simulations.

What changes
- Add a confidence rule for picks:
  - if CI width is too large, lower trust or skip pick.
- EV display shows both value and confidence stability.

Acceptance gate
- Fewer unstable picks, no large drop in hit rate.
- Clear "why skipped" explanation in debug output.

Milestone 3 - Rare-Event Stability (Phase 3, Optional)
Goal
- Improve estimates for rare tail markets where plain simulation needs too many runs.

Method from PDF
- Importance sampling with weighted estimator (`p_hat_IS`).

Scope
- Optional, only for specific rare-event modules.
- Not required for baseline OU/BTTS rollout.

Milestone 4 - Multi-Pick Correlation Risk (Phase 4)
Goal
- If multiple picks are played together, estimate portfolio risk more realistically.

Method from PDF
- Copula-based dependence (start simple; increase sophistication only if needed).

What users will see
- Portfolio-level stress view:
  - chance many picks fail together,
  - not just per-match standalone probabilities.

Milestone 5 - Live Updating (Phase 5, Future)
Goal
- If live/in-play feed is added, update probabilities as new evidence arrives.

Method from PDF
- Sequential Monte Carlo (particle filter style update).

Note
- Not needed for current pre-match workflow.

Milestone 6 - Agent-Based Simulation (Research Only)
Goal
- Explore market-noise behavior, not core match prediction.

Method from PDF
- Agent-based simulation (informed/noise/market-maker style dynamics).

Note
- Keep as research track; do not mix into core OU/BTTS engine without strong evidence.

Rollout policy
1. Implement one milestone at a time.
2. Backtest before/after for each milestone.
3. Keep feature flags so rollback is immediate.
4. Promote to production only after stable results.


