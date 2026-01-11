Project blueprint (current)

Data sources (CGM only)
- **Primary data files**:
  - `CGM data/multiple leagues and seasons/allratingv.csv` = historical matches + future fixtures list (predictions).
  - `CGM data/multiple leagues and seasons/upcoming.csv` = supplemental odds/stats (BTTS odds `gg/ng`).
- **Future detection**: Fixtures are identified as "future" by date (`datameci > today`), not by scores.
- Folder: `CGM data/` for additional files (stats, timing, upcoming exports).
- Outputs live under `data/enhanced/` (history, baselines, training matrix) and `reports/` (predictions, logs).

Supported Leagues (12)
- England: Premier L, Championship
- Italy: Serie A, Serie B
- Spain: Primera
- Germany: Bundesliga
- France: Ligue 1, Ligue 2
- Portugal: Primeira L
- Netherlands: Eredivisie
- Turkey: Super Lig
- Romania: Liga 1

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
- `cgm/predict_upcoming.py`: loads latest history, rolling snapshots, band stats; reads upcoming fixtures from `allratingv.csv` (future rows) and enriches gg/ng from `upcoming.csv` when present; predicts mu_home/mu_away; converts to Poisson O/U 2.5 + BTTS; computes EV vs odds; writes `reports/cgm_upcoming_predictions.csv`; logs to `reports/run_log.jsonl`.
  - Supports `--model-variant full|no_odds` to select which model to load.
- Uses recalculated Elo (prefers elo_home_calc/elo_away_calc), league home_bonus_elo, fixed band threshold 150 (can tune), date cutoff enforced upstream, and the same Elo similarity features (GFvsSim/GAvsSim + wsum/neff).
- Also computes Pressure and xG-proxy rolling snapshots from history to keep the train/inference feature contract aligned.
- Live scope: `cgm/predict_upcoming.py` filters the "upcoming" feed deterministically (no past fixtures; date-window + league/country filters; optional horizon). Scope metadata is written into every prediction row (`run_asof_datetime`, `scope_*`, counts) and also logged as `UPCOMING_SCOPE` in `reports/run_log.jsonl`.
- `archive/legacy_full_engine/pick_engine.py`: legacy full strategy layer (1X2 + O/U 2.5); archived and not used in the goals-only pipeline.
- `cgm/goal_timing.py`: parses `CGM data/AGS.CSV` goal-minute lists into per-team timing profiles, and computes match-level timing probabilities (does not change mu).
- `cgm/pick_engine_goals.py`: goals-only pick engine (Milestone 7) that outputs only O/U 2.5 + BTTS (GG/NG).
  - Writes `reports/picks.csv` (+ `reports/picks_debug.csv`) with the same schema as the full pick engine, plus narrator-friendly columns (`model_prob`, `implied_prob`, `value_margin`, `risk_flags`).
- `cgm/narrator.py`: turns picks into human-readable explanations (Milestone 8) and writes `reports/picks_explained.csv` (+ preview txt). Deterministic templates only; no new modeling.
- Safety: `cgm/pick_engine_goals.py` enforces the live scope (fixture_datetime must be strictly > run_asof_datetime and within the configured window) so it cannot accidentally emit picks for past seasons if the upstream feed is a schedule dump.

Backtest (Milestone 12)
- `scripts/run_backtest.py`: "time-travel" backtest orchestrator that simulates predictions on historical dates.
  - For each test date: creates filtered history (matches before test date) + blind upcoming file (no results).
  - Runs `cgm/predict_upcoming.py` with `--as-of-date` to ensure leakage-safe predictions.
  - Aggregates predictions, merges with actual results, outputs `reports/backtest_*.csv`.
- Decay feature fix: `cgm/predict_upcoming.py` updated to extract Milestone 9 decay features (`press_form_*_decay`, `xg_*_form_*_decay`) for inference alignment.

Streamlit UI
- `ui/streamlit_app.py`: web dashboard for predictions, picks, and backtest visualization.
  - Tabs: Predictions, Picks (Value Bets), Statistics, **Backtest Results**.
  - Backtest tab: summary metrics (O/U 2.5 accuracy, GG/NG accuracy), per-round picker, match cards with ✅/❌ icons.

Models/strategy
- Frankenstein mu models (home/away) power Poisson probability layer; EV computed directly vs market odds from CGM upcoming CSV.

Configuration (single source of truth)
- `config.py`: contains all tuneable constants for the CGM pipeline:
  - `ELO_SIM_SIGMA_PER_LEAGUE`: Gaussian kernel width for Elo-similarity features.
  - `LIVE_SCOPE_*`: country, league, season dates, horizon for live prediction filtering.
  - Pick engine gates (used by `pick_engine_goals.py` and goals-only audits):
    - `ODDS_MIN_FULL`, `ODDS_MIN_GOALS`: minimum odds to consider (1.01 vs 1.05).
    - `MU_TOTAL_MIN`, `MU_TOTAL_MAX`: expected goals bounds for pick eligibility.
    - `NEFF_MIN_*`, `PRESS_EVID_MIN_*`, `XG_EVID_MIN_*`: evidence thresholds (stricter for goals-only).
    - `EV_MIN_*`: minimum expected value thresholds per market type.
    - Risk-adjusted thresholds: `EV_MIN_STERILE_*`, `EV_MIN_ASSASSIN_*`, `EV_MIN_LATE_HEAVY_*`.
    - Assassin stricter evidence: `ASSASSIN_NEFF_MIN_OU25`, `ASSASSIN_PRESS_EVID_MIN_OU25`, `ASSASSIN_XG_EVID_MIN_OU25`.
  - To tune thresholds, edit `config.py` once; all modules import from it.

Audits
- `python scripts/run_all_audits.py`: runs all 10 audit scripts (recommended for full validation).
- `python scripts/audit_multi_league.py`: multi-league coverage verification (history, source, predictions).
- `python -m scripts.audit_pressure --cutoff YYYY-MM-DD`: Pressure coverage + no-leak tripwires (date parsing summary, shift checks, training scan for `_press_*`, raw-stat reconstruction checks).
- `python -m scripts.audit_xg`: xG proxy coverage + no-leak tripwires (shift checks, training scan for `_xg_*` / `xg_proxy_*`).
- `python -m scripts.audit_decay`: time decay feature validation (Milestone 9).
- `python -m scripts.audit_h2h`: head-to-head feature validation (Milestone 10).
- `python -m scripts.audit_league_features`: league-specific feature validation (Milestone 11).
- `python -m scripts.audit_picks_goals`: goals-only pick engine audits (determinism + gate correctness).
- `python -m scripts.audit_upcoming_feed --as-of-date YYYY-MM-DD`: scope filtering proof (raw feed date ranges + counts dropped by each filter step).
- `python -m scripts.audit_no_odds --upcoming PATH --as-of-date YYYY-MM-DD`: verifies `no_odds` mu/probabilities are invariant to odds changes in the upcoming feed (use allratingv.csv).
- `python -m scripts.audit_narrator`: narrator output validation.

Prediction Reports
- `python scripts/generate_predictions_report.py`: formatted predictions table (O/U, BTTS, EV).

Open considerations
- Fine-tune band thresholds and sigma per league; add EV/mu safety filters. Ensure Elo recompute always runs with a cutoff to avoid future leakage.
- Extend backtest to more leagues and longer date ranges; track cumulative profit/loss over time.
