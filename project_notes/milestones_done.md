Milestone 1 - Elo Rebuild, Similarity Bands, and Wiring (Dec 2025)

Goal
- Replace legacy/zero-prone Elo with a recalculated football Elo time series, enforce date cutoff (no future leakage), and add opponent-strength (kernel) features consumed end-to-end.

What was done
- New Elo engine: `scripts/calc_cgm_elo.py` recomputes Elo from results only (start 1500, K=20, home advantage 65, World-Football margin multiplier), with `--max-date` cutoff (default today UTC) and overwrites legacy Elo columns.
- Clean history: `data/enhanced/cgm_match_history_with_elo.csv` (cutoff 2025-12-11, no zeros); drops merge artefacts (`*_x`/`*_y` lg_avg columns), keeps canonical Elo and `_calc` aliases for compatibility.
- Wiring updates:
  - `cgm/build_frankenstein.py` defaults to the clean history, overwrites legacy Elo in memory, adds kernel-based opponent-similarity features `GFvsSim_* / GAvsSim_*` (Gaussian, sigma per league, default 60) using `cgm/elo_similarity.py` with as-of cutoff and effective sample size blending, and drops calc/name columns from the training matrix to align train/inference.
  - `cgm/predict_upcoming.py` uses the same similarity logic at inference (no leakage), prefers clean Elo.
- Pipeline rerun with cutoff: Elo  Frankenstein features (2810 rows, 107 cols after dropping legacy/duplicate columns)  train  predict_upcoming regenerated.

Key artifacts
- `data/enhanced/cgm_match_history_with_elo.csv` (clean Elo, cutoff applied, compatibility aliases).
- `data/enhanced/frankenstein_training.csv` (with GFvsSim_* + wsum/neff, aligned feature set).
- `models/frankenstein_mu_home.pkl`, `models/frankenstein_mu_away.pkl` retrained on clean Elo + similarity features (Poisson objective, time split).
- `reports/cgm_upcoming_predictions.csv` regenerated with updated models/history.

Milestone 1.1 - Elo Logging & Traceability (Dec 2025)

Goal
- Make Elo runs auditable in seconds: capture inputs/hashes, config/cutoff, feature contract, neff reliability, skips, and sampled per-fixture traces.

What was done
- `scripts/calc_cgm_elo.py`: added structured JSONL logging (history hash/rows/date range, Elo config K/HA/start/band/cutoff, Elo min/med/max, max per-team delta), configurable `--log-level` and `--log-json` (default `reports/run_log.jsonl`).
- `cgm/predict_upcoming.py`: added structured run summary (history/upcoming/model hashes, sigma map, min-matches, neff quantiles, skipped counts, mu_total min/max), hard-asserts on missing model features; optional sampled per-fixture traces (`--log-sample-rate`, `--trace-json`, `--log-max-fixtures`) logging Elo context, bands, kernel outputs (GFvsSim/GAvsSim + wsum/neff), mu/probs.
- Outputs: `reports/run_log.jsonl` for summaries; `reports/elo_trace.jsonl` for sampled fixture traces.
- How to read the logs (Elo-only):
  - `reports/run_log.jsonl`: look for `ELO_DATA` (input file/hash/date range/cutoff), `ELO_ELO` (Elo config + min/med/max + max delta), `ELO_FEATS` (model hashes + feature counts), `ELO_PRED_SUMMARY` (sigma map, min-matches, skipped counts, neff quantiles, mu_total min/max).
  - `reports/elo_trace.jsonl`: per-fixture samples with Elo context (home/away, bonus, bands), kernel outputs (GFvsSim/GAvsSim + wsum/neff), and mu/probs for trace-sampled fixtures.
  - Missing features now raise immediately (assert), so a run cannot silently proceed with mismatched contracts.

Milestone 1.2 - Leakage Fixes + Inference Alignment (Dec 2025)

What was wrong
- Match history future-row filtering could fail on newer pandas because `pd.Timestamp.utcnow()` became tz-aware while parsed CSV datetimes were tz-naive; this could leave future fixtures inside the history (leakage risk).
- `cgm/predict_upcoming.py` band-conditional features were not venue-aware: away-band stats could overwrite home-band stats for the same team.
- Home advantage used for `EloDiff` / bands was inconsistent between training history and inference (some paths used `home_bonus_elo`, others a constant).

What was changed
- `cgm/build_match_history.py`: normalize UTC cutoff to tz-naive before filtering future rows (prevents compare/type errors).
- `scripts/calc_cgm_elo.py`: hardened team-id inference and score coercion (prevents missing-code teams collapsing into a `"nan"` team).
- `cgm/predict_upcoming.py`: store band stats separately for home/away contexts; infer the effective home-advantage from the history’s `EloDiff` and use it for upcoming `EloDiff`/bands so inference matches training.

Milestone 2 - Pressure Cooker Strategy (Dec 2025)

Goal
- Add a gameplay-volume lens (shots/SOT primary; corners/possession secondary) that measures field tilt and compares it against team-level Elo (results strength) via league-standardized divergence.

What was done
- New Pressure Cooker modules (single source of truth):
  - `cgm/pressure_inputs.py`: deterministic mapping/parsing from CGM schemas to split inputs (`shots_H/A`, `sot_H/A`, `corners_H/A`, `pos_H/A`), including parsing combined "H-A" strings (e.g., `sut`, `sutt`, `cor`, `ballp`).
  - `cgm/pressure_form.py`: per-match dominance components with neutral fallback (0.5) when data is missing/zero; venue-aware rolling L10 (home-only / away-only) producing `press_form_H/A`, component means, match-count samples (`press_n_*`), and real-stats evidence counts (`press_stats_n_*`).
  - `cgm/pressure_divergence.py`: continuous divergence using league z-scores at as-of time: `div_team_* = z(press_form_*) - z(elo_team_*)`, plus fixture `div_diff`.
- Pipeline wiring:
  - `cgm/build_frankenstein.py`: uses split stats (H/A) for rolling shot/corner/possession features; adds Pressure Cooker features into the training matrix; drops internal `_press_*` post-match helper columns from the exported CSV.
  - `cgm/predict_upcoming.py`: computes the same Pressure features from history and injects them into the per-fixture feature vector (including component dominance), keeping train/inference contracts aligned.
- Match stats backfill (data coverage support):
  - `cgm/backfill_match_stats.py`: joins per-match stats from `CGM data/goals statistics.csv` into `data/enhanced/cgm_match_history_with_elo.csv` and writes `data/enhanced/cgm_match_history_with_elo_stats.csv`.
  - Current coverage is limited by the raw CGM export (example: `goals statistics.csv` has 385 rows, yielding ~5% coverage in a 2810-row history even with a perfect join).
- Training contract update:
  - `cgm/train_frankenstein_mu.py`: keeps `press_*` and `div_*` features (does not drop them via leakage-token filtering).

Feature contract (minimal)
- `press_form_H`, `press_form_A`, `press_n_H`, `press_n_A`, `press_stats_n_H`, `press_stats_n_A`, `div_team_H`, `div_team_A`, `div_diff`

Milestone 2.1 - Baselines Reliability + Elo Consistency (Dec 2025)

What was wrong
- League averages were brittle: `leageue statistics.csv` variants can be team-level (or missing season keys), leading to `lg_avg_*` being all-NaN and breaking attack/defense indices.
- `cgm/pressure_divergence.py` advanced Elo state using `home_bonus_elo`, which can drift from the Elo rebuild assumptions.
- `CGM data/goals statistics.csv` can include future fixtures with populated `sut/sutt/cor/ballp`; without strict cutoff + deterministic date parsing, there is a risk of match-stats leakage or incorrect joins.
- Team registry normalization could be corrupted by cross-pairing unrelated code/name columns, causing wrong team names in history (e.g., `Crystal Palace -> Arsenal`) and even `home==away` artefacts.

What was changed
- `cgm/build_baselines.py`: derive `lg_avg_*` directly from match history (league/season; optionally country/league/season), merge back correctly, and keep `EloDiff`/bands aligned to the Elo rebuild home-advantage baseline.
- `cgm/pressure_divergence.py`: infer the effective home-advantage from `EloDiff` per `(country, league)` (with global fallback) and use it for Elo-state updates (keeps divergence aligned with rebuilt Elo without cross-league drift).
- `cgm/backfill_match_stats.py`: deterministically parse `datameci` (explicit formats; logs ambiguous slash dates + chosen preference), filter stats rows to the history cutoff, dedupe collisions by completeness, and guard joins with FT-score validation to prevent accidental mis-matches/leakage.
- `cgm/team_registry.py`: only collect aligned `(code, name)` pairs from known CGM columns and normalize codes (strip `.0`) to prevent alias collisions and incorrect team normalization.

Milestone 2.2 - Pressure Audits + Input Hardening (Dec 2025)

Goal
- Make Pressure features auditable (shift/leak checks) and make input normalization safer for future multi-source merges.

What was wrong
- `ensure_pressure_inputs()` could be perceived as “overwrite-prone” if multiple sources ever populate canonical columns (even though it was safe in current artifacts).
- Naive “feature equals target” tripwires produced false positives when the target is often 0 and the feature frequently takes 0/neutral values.

What was changed
- `cgm/pressure_inputs.py`: canonical stat columns are now filled only where missing (no overwrites); `ensure_pressure_inputs_fill(..., overwrite=True)` is available for debugging.
- `scripts/audit_pressure.py`: added coverage and no-leak tripwires (shift identity checks, `_press_*` training scan, and raw-stat reconstruction checks).

Milestone 3 - xG-proxy "Sniper" Engine (Dec 2025)

Goal
- Add a chance-danger/quality lens (xG proxy) to complement Pressure's volume/field-tilt lens, and expose Pressure-vs-xG disagreement ("sterile dominance" vs "assassin counters").

What was done
- New xG proxy builder (walk-forward, leakage-safe):
  - `cgm/build_xg_proxy.py`: builds `xg_proxy_H` / `xg_proxy_A` from shots + SOT (plus against stats + `is_home`) using weekly walk-forward training (train strictly before each ISO week; predict inside the week).
  - Output: `data/enhanced/cgm_match_history_with_elo_stats_xg.csv` (adds `xg_proxy_*`, `xg_usable`, plus per-match helpers `shot_quality_*`, `finishing_luck_*`).
- Venue-aware rolling xG form:
  - `cgm/xg_form.py`: computes leakage-safe pre-match rolling xG form (L10) for home/away contexts: `xg_for_form_*`, `xg_against_form_*`, `xg_diff_form_*`, plus `xg_shot_quality_form_*`, `xg_finishing_luck_form_*`, and evidence counts `xg_stats_n_*`.
- Pressure vs xG disagreement:
  - `cgm/pressure_xg_disagreement.py`: league z-score disagreement features `div_px_team_*` / `div_px_diff` and simple flags `sterile_*` / `assassin_*`.
- Pipeline wiring:
  - `predict.py`: runs `cgm.build_xg_proxy` after stats backfill, then trains/predicts using the xG-enriched history.
  - `cgm/build_frankenstein.py`: reads `cgm_match_history_with_elo_stats_xg.csv` by default; adds xG form + Pressure-vs-xG disagreement features; explicitly bans per-match `xg_proxy_*` / `xg_usable` and `pressure_usable` from the SAFE training matrix.
  - `cgm/predict_upcoming.py`: computes xG rolling snapshots from history (post-match states) and injects `xg_*`, `xg_z_*`, `div_px_*`, `sterile_*`, `assassin_*` into the per-fixture feature vector so inference matches training.
- Audits:
  - `scripts/audit_xg.py`: xG coverage, sanity distributions, shift identity checks, and training-matrix leakage scans for `_xg_*` / `xg_proxy_*`.

Feature contract (minimal)
- `xg_for_form_H`, `xg_against_form_H`, `xg_diff_form_H`, `xg_stats_n_H`
- `xg_for_form_A`, `xg_against_form_A`, `xg_diff_form_A`, `xg_stats_n_A`
- `xg_z_H`, `xg_z_A`, `div_px_team_H`, `div_px_team_A`, `div_px_diff`, `sterile_H`, `sterile_A`, `assassin_H`, `assassin_A`

Milestone 4 - Strategy Layer ("Pick Engine") (Dec 2025)

Goal
- Convert model predictions into deterministic, explainable betting picks for two markets only:
  - 1X2 (Home/Draw/Away)
  - Over/Under 2.5
- Enforce hard quality gates, select at most one pick per fixture (or none), assign stake tiers, and write stable pick artifacts.

What was done
- New module: `cgm/pick_engine.py`
  - Reads `reports/cgm_upcoming_predictions.csv` and fails fast if required columns are missing.
  - Applies deterministic gates:
    - G1 odds sanity per market (requires complete 1X2 odds and complete OU2.5 odds).
    - G2 `mu_total` bounds.
    - G3 evidence minimums from `neff_sim_*`, `press_stats_n_*`, `xg_stats_n_*`.
    - G4 sterile/assassin risk gating.
    - G5 minimum EV thresholds (with stricter thresholds under risk flags).
  - Ranks eligible candidates by a reliability-aware score, selects the top candidate per fixture, assigns stake tier from a confidence formula, and outputs:
    - `reports/picks.csv` (one row per pick)
    - `reports/picks_debug.csv` (candidate-level gating + score debug)
- Deterministic tie-breaks: when candidates are tied, selection prefers higher EV, higher `neff_min`, then a fixed market priority order (`OU25_OVER`, `OU25_UNDER`, `1X2_HOME`, `1X2_AWAY`, `1X2_DRAW`) to avoid alphabetical drift.
- Inference output upgrade: `cgm/predict_upcoming.py`
  - Now writes pick-engine-required columns into `reports/cgm_upcoming_predictions.csv` (fixture_datetime, league, reliability evidence, sterile/assassin flags, and OU2.5 canonical naming).
- Pipeline wiring: `predict.py`
  - Runs the pick engine as the final step (also in `--predict-only` mode).
- Audits: `scripts/audit_picks.py`
  - Checks determinism (two runs -> identical hash), allowed markets only, odds sanity, mu bounds, reliability gates, sterile/assassin blocks, EV thresholds, stake-tier mapping, and input_hash integrity.

Key artifacts
- `reports/cgm_upcoming_predictions.csv` (now includes reliability + risk fields required for pick selection).
- `reports/picks.csv` (final deterministic picks).
- `reports/picks_debug.csv` (debuggable per-candidate gate results).

Milestone 4.1 - Live Scope Filtering (Dec 2025)

Goal
- Prevent accidental "retro predictions" / future-informed backtests by ensuring the live pipeline only emits predictions/picks for fixtures strictly after the run cutoff date, within the configured league+window (EPL 2025-26 for now).

What was done
- Single-source-of-truth defaults added in `config.py`:
  - `LIVE_SCOPE_COUNTRY`, `LIVE_SCOPE_LEAGUE`, `LIVE_SCOPE_SEASON_START`, `LIVE_SCOPE_SEASON_END`, `LIVE_SCOPE_HORIZON_DAYS`
- `cgm/predict_upcoming.py` now:
  - Parses `fixture_datetime` from `datameci` + `orameci`.
  - Applies deterministic scope filters (drop past strictly > `run_asof_datetime`, then season window, then league/country, then optional horizon) and logs counts at each step.
  - Writes scope metadata into every prediction row (`run_asof_datetime`, `scope_*`, counts) and logs `UPCOMING_SCOPE` into `reports/run_log.jsonl`.
- `cgm/pick_engine.py` now:
  - Requires `run_asof_datetime` + `scope_*` columns in its input contract.
  - Re-applies the same scope filter internally, so it cannot output picks for past fixtures even if upstream export is a schedule dump.
- New audit:
  - `scripts/audit_upcoming_feed.py` prints raw feed date ranges + counts dropped by each scope filter step.

Milestone 4.2 - No-odds Internal Model Variant (Dec 2025)

Goal
- Support a strict “internal-only” prediction mode where mu/probabilities do not use market inputs (odds or CGM market probabilities) as model features.
- Keep EV/picks as a separate layer that can still use odds when available.

What was done
- `cgm/train_frankenstein_mu.py`
  - Added `--variant full|no_odds`.
  - `no_odds` drops market features (`odds_*`, `fair_*`, `p_*`) from the training matrix before fitting.
  - Writes models to `models/frankenstein_mu_home_no_odds.pkl` and `models/frankenstein_mu_away_no_odds.pkl`.
- `cgm/predict_upcoming.py`
  - Added `--model-variant full|no_odds` to load the corresponding model files.
  - Writes `model_variant` into `reports/cgm_upcoming_predictions.csv` for auditability.
- `predict.py`
  - Added `--model-variant` and wires it into both training and inference.
- New audit:
  - `scripts/audit_no_odds.py` proves mu/probabilities are invariant to changes in the upcoming feed odds (and asserts the model contract contains no market features).

Milestone 7.1 - Goals-only Pick Engine (OU2.5 + BTTS) (Dec 2025)

Goal
- Provide a goals-market-only strategy layer that ignores 1X2 completely and outputs only:
  - O/U 2.5 (always)
  - BTTS Yes/No (only if BTTS odds exist; otherwise skipped deterministically)

What was done
- New module: `cgm/pick_engine_goals.py`
  - Reads `reports/cgm_upcoming_predictions.csv` and produces `reports/picks.csv` + `reports/picks_debug.csv` using goals-only markets.
  - Deterministic gates:
    - Odds sanity (odds > 1.05)
    - `mu_total` bounds (same defaults as before)
    - Evidence minimums (stricter than Milestone 4): `neff_min>=8`, `press_n_min>=3`, `xg_n_min>=3`
    - Risk-aware EV floors:
      - sterile -> Over 2.5 requires EV >= 0.08
      - assassin -> Under 2.5 requires EV >= 0.08
  - Selection: ranks eligible candidates by score (EV + small reliability bonus) and picks at most one per fixture.
  - Same output schema as Milestone 4 pick engine so downstream tooling stays compatible, plus narrator-friendly columns (`model_prob`, `implied_prob`, `value_margin`, `risk_flags`).
- Pipeline wiring: `predict.py`
  - Added `--pick-engine full|goals` to select between Milestone 4 and Milestone 7.1 pick engines.
- Inference export upgrade: `cgm/predict_upcoming.py`
  - Adds BTTS reporting fields:
    - `p_btts_yes`, `p_btts_no` (derived from mu)
    - `odds_btts_yes`, `odds_btts_no` (from CGM `gg/ng` columns when present)
    - `EV_btts_yes`, `EV_btts_no`
- New audit: `scripts/audit_picks_goals.py`
  - Checks determinism, allowed markets, odds sanity, mu bounds, evidence thresholds, risk-adjusted EV rules, and stake-tier mapping.

Milestone 7.2 - Minute-goals Timing Integration (Dec 2025)

Goal
- Use the goal-minute export to compute timing profiles (early vs late goal tendencies) and use them to:
  - add optional timing markets (only when odds exist)
  - add timing-aware risk gating (late-goal profile makes full-match unders riskier)
- Important: this does NOT change `mu_home` / `mu_away` (it is a strategy-layer bias only).

What was done
- New module: `cgm/goal_timing.py`
  - Parses goal-minute lists (from `CGM data/AGS.CSV`) and builds per-team timing profiles.
  - Computes match-level timing probabilities from `mu_home/mu_away` + timing profiles:
    - 1H O/U 0.5
    - 2H O/U 0.5 and O/U 1.5
    - Goal after 75' (Yes/No)
- `cgm/predict_upcoming.py`
  - Loads `CGM data/AGS.CSV` (when present) and exports timing probabilities + timing flags into `reports/cgm_upcoming_predictions.csv`.
- `cgm/pick_engine_goals.py`
  - Adds timing markets (only if their odds columns exist and are sane in the predictions input).
  - Adds timing risk gating: when `timing_usable=1` and `late_goal_flag=1`, full-match `OU25_UNDER` requires a higher EV floor.
  - Exposes timing flags + shares into `reports/picks.csv` (`timing_usable`, `slow_start_flag`, `late_goal_flag`, `timing_early_share`, `timing_late_share`).
- `scripts/audit_picks_goals.py`
  - Extended to allow timing markets, validate timing gate integrity, and enforce the late-goal EV adjustment for full-match unders.

Milestone 8 - Narrator Layer (Human Explanations) (Dec 2025)

Goal
- Turn machine-friendly picks into human-readable explanations without adding new modeling logic.

What was done
- New module: `cgm/narrator.py`
  - Loads `reports/picks.csv` and produces `reports/picks_explained.csv` (+ optional `picks_explained_preview.txt`).
  - Validates required columns, then adds narrator fields: `pick_text`, `title`, `narrative`, `confidence_label`, `numbers_plain`.
  - Deterministic templates translate model vs implied percentages, expected goals, evidence labels, and risk flags into plain language (no raw “EV=”, “neff_min”, etc).
- New audit: `scripts/audit_narrator.py`
  - Checks determinism (hash match on double run), presence of narrator columns, ensures no forbidden raw tokens leak into narratives, and that narratives contain key human elements.

Milestone 9 - Time Decay Weighting (Dec 2025)

Goal
- Make recent matches count more than older ones in form calculations, so teams on hot/cold streaks are captured faster.

What was done
- Config: added `DECAY_HALF_LIFE = 5` and `DECAY_ENABLED = True` to `config.py`.
- Pressure form: added exponential decay weighted rolling to `cgm/pressure_form.py`:
  - New features: `press_form_H_decay`, `press_form_A_decay`
  - Formula: `weight = exp(-0.693 * match_age / half_life)`
- xG form: added exponential decay weighted rolling to `cgm/xg_form.py`:
  - New features: `xg_for_form_H_decay`, `xg_against_form_H_decay`, `xg_for_form_A_decay`, `xg_against_form_A_decay`
- Decay can be toggled off with `DECAY_ENABLED = False` in config.

Key artifacts
- Updated `config.py` with decay settings.
- Updated `cgm/pressure_form.py` with `_roll_pre_decay()` function and decay features.
- Updated `cgm/xg_form.py` with `_roll_pre_decay()` function and decay features.

Milestone 10 - Head-to-Head History (Dec 2025)

Goal
- Track direct matchup patterns between specific teams to capture rivalry effects and psychological factors.

What was done
- Config: added `H2H_MIN_MATCHES = 3`, `H2H_MAX_LOOKBACK_YEARS = 5`, `H2H_ENABLED = True` to `config.py`.
- New module: `cgm/h2h_features.py`:
  - `add_h2h_features()`: computes H2H stats for all rows in training data
  - `get_h2h_features_for_fixture()`: computes H2H stats at inference time
  - Leakage-safe: only uses matches strictly before current match datetime
  - Handles reversed matchups (A vs B treated same as B vs A, with perspective flip)
- New features: `h2h_matches`, `h2h_home_win_rate`, `h2h_goals_avg`, `h2h_btts_rate`, `h2h_over25_rate`, `h2h_usable`
- Integration: `cgm/build_frankenstein.py` now calls `add_h2h_features()` after xG features

Key artifacts
- `cgm/h2h_features.py` (new module)
- Updated `config.py` with H2H settings
- Updated `cgm/build_frankenstein.py` with H2H integration

Milestone 11 - League-Specific Features (Dec 2025)

Goal
- Capture league-specific scoring patterns (EPL = high scoring, Serie A = defensive) so the model can adjust predictions based on competition characteristics.

What was done
- Config: added `LEAGUE_FEATURES_ENABLED = True`, `LEAGUE_MIN_MATCHES = 50`, `LEAGUE_PROFILE_WINDOW = 100` to `config.py`.
- New module: `cgm/league_features.py`:
  - `add_league_features()`: computes rolling league profile stats for all rows in training data
  - `get_league_features_for_fixture()`: computes league profile at inference time
  - Leakage-safe: only uses matches strictly before current match datetime
  - Rolling window ensures recent league patterns are weighted appropriately
- New features: `lg_goals_per_match`, `lg_home_win_rate`, `lg_btts_rate`, `lg_over25_rate`, `lg_home_advantage`, `lg_defensive_idx`, `lg_profile_usable`
- Integration: `cgm/build_frankenstein.py` now calls `add_league_features()` after H2H features
- Inference: `cgm/predict_upcoming.py` now calls `get_league_features_for_fixture()` for each upcoming fixture

Key artifacts
- `cgm/league_features.py` (new module)
- Updated `config.py` with league feature settings
- Updated `cgm/build_frankenstein.py` with league features integration
- Updated `cgm/predict_upcoming.py` with inference-time league lookup

Milestone 12 - Backtest System & Streamlit UI Enhancement (Dec 2025)

Goal
- Enable historical performance validation by running "time-travel" backtests that simulate predictions as if made on past dates (no future leakage).
- Add a dedicated Streamlit tab to visualize backtest results per round, focusing on GG/NG and Over/Under 2.5 markets.

What Was Done

1. Backtest Script (`scripts/run_backtest.py`):
   - New orchestrator that loops through historical match dates.
   - For each date:
     - Creates a **filtered history CSV** (matches strictly before test date).
     - Creates a **blind upcoming CSV** (no result columns) for that day's matches.
     - Runs `cgm/predict_upcoming.py` with `--as-of-date` to ensure no future leakage.
   - Aggregates all predictions and merges with actual results from history.
   - Outputs: `reports/backtest_epl_2025.csv` (or similar).

2. Bug Fix in `cgm/predict_upcoming.py`:
   - **Issue**: Model crashed with `Missing features: ['press_form_H_decay', ...]`.
   - **Root Cause**: Milestone 9 decay features were computed in history but not extracted for inference.
   - **Fix**: Added 6 decay snapshot dictionaries (`latest_press_decay_home/away`, `latest_xg_for/against_decay_home/away`) and corresponding extraction logic in the row loop (~line 450-500), plus 6 new entries in the `feats` dict (~line 930).

3. Streamlit UI Enhancement (`ui/streamlit_app.py`):
   - Added `load_backtest()` function to read backtest CSV files.
   - Added 4th tab: **"🔙 Backtest Results"**.
   - Tab features:
     - **Summary metrics**: Matches analyzed, O/U 2.5 accuracy, GG/NG accuracy, actual Over 2.5 rate.
     - **Per-round picker**: Dropdown to select specific match day.
     - **Match cards**: Show final score, bot's O/U 2.5 prediction vs actual (✅/❌), bot's GG/NG prediction vs actual (✅/❌).
     - **Raw data expander**: View underlying backtest CSV.

Key Artifacts
- `scripts/run_backtest.py` (new script)
- Updated `cgm/predict_upcoming.py` with decay feature fixes
- Updated `ui/streamlit_app.py` with Backtest Results tab
- `reports/backtest_epl_2025.csv` (example output)

Usage
```bash
# Run backtest for EPL 2025-2026 season
python -m scripts.run_backtest --league "Premier L" --season "2025-2026" --start-date "2025-09-01"

# Launch Streamlit to view results
python -m streamlit run ui/streamlit_app.py
```

Milestone 13 - League-Specific Probability Calibration (Dec 2025)

Goal
- Automatically detect and correct league-specific prediction bias so the bot's probabilities better match real-world outcomes.

What Was Done

1. Calibration Script (`scripts/calibrate_league.py`):
   - Analyzes backtest results to find optimal decision thresholds per league.
   - Calculates bias (how much model over/underestimates).
   - Outputs JSON file: `data/league_calibration.json`.

2. Example Calibration Results (EPL 2025-2026):
   - Default O/U 2.5 accuracy: 47.7%
   - Optimal O/U 2.5 threshold: 35% → accuracy: 60.4%
   - Bias: +17.7% (model underestimates Over probability)

3. Config Updates (`config.py`):
   - Added `CALIBRATION_ENABLED = True`
   - Added `CALIBRATION_FILE = "data/league_calibration.json"`
   - Added `CALIBRATION_MIN_SAMPLES = 50`

4. Pick Engine Integration (`cgm/pick_engine_goals.py`):
   - Added `_load_calibration()` to load JSON at startup.
   - Added `_get_calibrated_threshold()` for league-specific thresholds.

5. Streamlit UI Enhancement:
   - Added experiment sliders for threshold tuning in Backtest Results tab.
   - Real-time accuracy feedback as user adjusts thresholds.

Key Artifacts
- `scripts/calibrate_league.py` (new script)
- `data/league_calibration.json` (generated calibration data)
- Updated `config.py` with calibration settings
- Updated `cgm/pick_engine_goals.py` with calibration loading

Usage
```bash
# Run calibration for EPL
python -m scripts.calibrate_league --input reports/backtest_epl_2025.csv --league "Premier L"

# View calibration file
cat data/league_calibration.json
```

