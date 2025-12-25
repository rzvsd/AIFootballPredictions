Project blueprint (current)

Data sources (CGM only)
- Folder: `CGM data/` provides historical matches, stats, league averages, and rich upcoming fixtures with odds.
- Outputs live under `data/enhanced/` (history, baselines, training matrix) and `reports/` (predictions, logs).

Pipelines
- `predict.py`: one-command orchestrator (runs the full CGM pipeline end-to-end).
- `cgm/build_match_history.py`: merge CGM season tables, normalize teams, drop future rows.
- `cgm/build_baselines.py`: derive team/league baselines, update history with league averages and bands.
- `scripts/calc_cgm_elo.py`: recompute Elo time series from results (start 1500, K=20, HFA=65, margin multiplier, date cutoff), overwrite legacy Elo columns in history, drop merge artefacts (`*_x`/`*_y`).
- `cgm/backfill_match_stats.py`: join per-match gameplay stats (shots/SOT/corners/possession from `cgmbetdatabase.*` if present, else `goals statistics.csv`) into `cgm_match_history_with_elo.csv` and produce `cgm_match_history_with_elo_stats.csv` for Pressure features (coverage depends on available CGM exports).
- `cgm/build_xg_proxy.py`: build a leakage-safe xG proxy (walk-forward weekly) from shots+SOT and write `cgm_match_history_with_elo_stats_xg.csv`.
- `cgm/build_frankenstein.py`: default input `cgm_match_history_with_elo_stats_xg.csv`; rolling stats (L5/L10) home/away, band conditionals, efficiencies, attack/defense indices, Pressure + xG-proxy form + disagreement features, Elo similarity (GFvsSim/GAvsSim with kernel/neff, prefix histories to avoid leakage), plus wsum/neff indicators; drops calc/name aliases before training; targets set.
- `cgm/train_frankenstein_mu.py`: trains XGB mu_home/mu_away models; drops leakage cols; saves to `models/`.
  - Variants:
    - `full` (default): includes market features (`odds_*`, CGM `p_*`).
    - `no_odds`: excludes market features so mu/probabilities are driven only by internal signals (Elo/similarity + form + Pressure + xG-proxy).

Inference
- `cgm/predict_upcoming.py`: loads latest history, rolling snapshots, band stats; builds feature vectors for upcoming fixtures; predicts mu_home/mu_away; converts to Poisson 1X2/OU2.5; computes EV vs odds; writes `reports/cgm_upcoming_predictions.csv`; logs to `reports/run_log.jsonl`.
  - Supports `--model-variant full|no_odds` to select which model to load.
- Uses recalculated Elo (prefers elo_home_calc/elo_away_calc), league home_bonus_elo, fixed band threshold 150 (can tune), date cutoff enforced upstream, and the same Elo similarity features (GFvsSim/GAvsSim + wsum/neff).
- Also computes Pressure and xG-proxy rolling snapshots from history to keep the train/inference feature contract aligned.
- Live scope: `cgm/predict_upcoming.py` filters the "upcoming" feed deterministically (no past fixtures; date-window + league/country filters; optional horizon). Scope metadata is written into every prediction row (`run_asof_datetime`, `scope_*`, counts) and also logged as `UPCOMING_SCOPE` in `reports/run_log.jsonl`.
- `cgm/pick_engine.py`: deterministic strategy layer that converts `reports/cgm_upcoming_predictions.csv` into actionable picks for 1X2 and O/U 2.5 with quality gates, stake tiers, and reason codes; writes `reports/picks.csv` (+ `reports/picks_debug.csv`).
- `cgm/goal_timing.py`: parses `CGM data/AGS.CSV` goal-minute lists into per-team timing profiles, and computes match-level timing probabilities (does not change mu).
- `cgm/pick_engine_goals.py`: goals-only pick engine (Milestone 7) that ignores 1X2 and outputs only goal markets:
  - O/U 2.5 + BTTS (Milestone 7.1)
  - Optional timing markets (Milestone 7.2) when odds exist in the predictions input:
    - 1H O/U 0.5
    - 2H O/U 0.5 and O/U 1.5
    - Goal after 75' (Yes/No)
  - Writes `reports/picks.csv` (+ `reports/picks_debug.csv`) with the same schema as the full pick engine, plus narrator-friendly columns (`model_prob`, `implied_prob`, `value_margin`, `risk_flags`).
- `cgm/narrator.py`: turns picks into human-readable explanations (Milestone 8) and writes `reports/picks_explained.csv` (+ preview txt). Deterministic templates only; no new modeling.
- Safety: `cgm/pick_engine.py` also enforces the same live scope (fixture_datetime must be strictly > run_asof_datetime and within the configured window) so it cannot accidentally emit picks for past seasons if the upstream feed is a schedule dump.
  - `cgm/pick_engine_goals.py` applies the same scope protections.

Models/strategy
- Frankenstein mu models (home/away) power Poisson probability layer; EV computed directly vs market odds from CGM upcoming CSV.

Audits
- `python -m scripts.audit_pressure --cutoff YYYY-MM-DD`: Pressure coverage + no-leak tripwires (date parsing summary, shift checks, training scan for `_press_*`, raw-stat reconstruction checks).
- `python -m scripts.audit_xg`: xG proxy coverage + no-leak tripwires (shift checks, training scan for `_xg_*` / `xg_proxy_*`).
- `python -m scripts.audit_picks`: pick-engine audits (determinism, gate correctness, EV thresholds, stake-tier mapping).
- `python -m scripts.audit_picks_goals`: goals-only pick engine audits (determinism + gate correctness).
- `python -m scripts.audit_upcoming_feed --as-of-date YYYY-MM-DD`: scope filtering proof (raw feed date ranges + counts dropped by each filter step).
- `python -m scripts.audit_no_odds --upcoming PATH --as-of-date YYYY-MM-DD`: verifies `no_odds` mu/probabilities are invariant to odds changes in the upcoming feed.

Open considerations
- Fine-tune band thresholds and sigma per league; add EV/mu safety filters; run historical backtests. Ensure Elo recompute always runs with a cutoff to avoid future leakage.
