League-Specific Weighting Plan (NLM/Stacker, Non-Technical First)

Purpose
- Keep one core engine.
- Let each league use a slightly different mix of signals.
- Avoid fragile manual tuning by using data-driven weighting with safety controls.

What this means in simple words
- Today the bot mixes many inputs (ELO, xG, pressure, league context, form).
- Different leagues behave differently.
- Instead of one fixed mix for all leagues, we teach a combiner to learn a better mix per league.
- If a league has little data, we do not trust a custom mix fully.
- In low-data leagues, weights stay close to a safe global default.

Core idea
- Global model = safe default mix.
- League-specific model = adjustment around global mix, not full override.
- Small league sample -> small adjustment.
- Large league sample -> larger adjustment allowed.

Why this is safer than pure ROI tuning
- Pure "maximize past ROI" can overfit and collapse next week.
- This plan uses:
  - probability quality first (calibration/log-loss/Brier),
  - then ROI quality,
  - then stability controls (drawdown/variance).

What we will learn per league
- How much to trust each signal group:
  - ELO
  - xG
  - pressure
  - league season context
  - residual form/context features

High-level architecture
1) Base signals per fixture
- Export normalized module signals for OU and BTTS.

2) Global combiner
- Learns a stable global weighting profile.

3) League adjustments (NLM/stacker layer)
- Learns league-specific deltas from global profile.
- Uses shrinkage toward global when sample is small.

4) Final probabilities
- Uses adjusted mix to produce OU/BTTS probabilities.

5) Pick layer
- Keeps current EV/confidence logic and quality flags.

Phased rollout plan

Phase 0: Freeze baseline
- Lock current behavior and backtest outputs.
- Produce reproducible baseline reports.

Phase 1: Attribution and diagnostics
- Measure current group influence and failure slices.
- Confirm where imbalance really hurts performance.

Phase 2: Build challengers
- C1: Evidence-damped ELO influence.
- C2: Remove duplicated ELO channels.
- C3: Group-regularized combiner.
- C4: Reliability-aware post-combine blend.

Phase 3: Low-evidence policy
- BAD rows do not go in primary picks.
- WARN rows can be published with penalty/lower trust tier.

Phase 4: League shrinkage governance
- League adjustment only if sample is sufficient.
- Bound weekly parameter movement.

Phase 5: Promote winner
- Promote only after non-regression gates pass.
- Keep rollback triggers documented and active.

Validation protocol (must follow)
- Strict walk-forward only (no leakage):
  - train/features up to t-1,
  - predict t..t+window.
- Compare challenger vs baseline on identical fixture sets.
- Report by league and aggregate.

Metrics required for decision
- Probability quality:
  - log-loss
  - Brier
  - calibration bins
- Decision quality:
  - OU/BTTS hit rate
  - class-wise hit rates (OVER/UNDER, YES/NO)
- EV quality:
  - ROI
  - profit per 100 bets
  - EV-decile monotonicity
- Stability:
  - variance
  - max drawdown
  - bet-count stability
  - OK/WARN/BAD mix drift

Target ranges (guidance, not hard lock)
- ELO: 7% to 12%
- xG: 15% to 25%
- Pressure: 15% to 25%
- League season averages: 5% to 10%
- Anchor ratios: 4% to 8%

Important note
- These are governance bands.
- They are not rigid fixed weights per run.

Expected artifacts each cycle
- `reports/rebalance_baseline_scorecard.csv`
- `reports/rebalance_module_attribution.csv`
- `reports/rebalance_error_slices.csv`
- `reports/rebalance_challenger_comparison.csv`
- `reports/rebalance_quality_tier_performance.csv`
- `reports/rebalance_league_shrinkage_table.csv`
- `reports/rebalance_league_drift.csv`
- `reports/rebalance_go_no_go_summary.md`
- `reports/rebalance_rollback_triggers.md`

What needs to be installed
- Already likely present:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
- Recommended for optimization:
  - `optuna`
    - install: `pip install optuna`
- Optional advanced Bayesian layer (heavier/slower):
  - `pymc`
  - `arviz`
    - install: `pip install pymc arviz`

Operational constraints
- Use one fixed data snapshot per challenger batch.
- Keep odds snapshot consistent across baseline/challengers.
- Keep branch isolation (`dev` for challenger work, promote later).

Go/No-Go promotion criteria
- Leakage/audit gates pass.
- Probability metrics non-inferior in top leagues.
- Aggregate EV quality improves or remains stable with lower risk.
- No instability spike in drawdown/variance.
- BAD exposure is reduced or better controlled.

Rollback triggers
- Two consecutive cycles with:
  - clear ROI deterioration,
  - calibration drift breach,
  - drawdown breach,
  - quality-status drift beyond limits.

Timeline estimate
- Full program: ~11 to 14 working days.

Current status
- Plan approved for documentation stage.
- No implementation changes applied from this plan yet.


