# AIFootballPredictions (CGM-only)

This repository now focuses exclusively on the **CGM** pipeline (local CSV exports). It ingests CGM CSVs, builds leakage-safe features (Elo, Pressure, xG-proxy), trains Poisson mu models, predicts upcoming fixtures, selects picks deterministically, and (Milestone 8) verbalizes each pick for humans.

Legacy Understat/odds/bot code has been archived to `archive/legacy_non_cgm_engine/`.

## What the bot does (high level)

1) Build history: merge CGM season tables, drop future rows, recompute Elo with a strict cutoff (no leakage).
2) Engineer features: rolling form (L5/L10), Elo similarity kernels, Pressure (dominance) form, xG-proxy Sniper form, Pressure-vs-xG disagreement, attack/defense indices.
3) Train models: XGBoost Poisson `mu_home` / `mu_away` (variants: `full`, `no_odds`).
4) Predict upcoming: apply live scope filters (no past fixtures; league/date window), compute mu/probabilities/EV for O/U 2.5 + BTTS, add timing risk flags if AGS goal-minute data is present.
5) Select picks: deterministic goals-only pick engine (O/U 2.5 + BTTS) with quality gates, risk flags, stake tiers.
6) Verbalize (Milestone 8): narrator turns each pick into a readable paragraph (`picks_explained.csv` + preview text).

## Required Input Files (CGM data/)

**Primary data sources** (Milestone 15 - Multi-League):
- `multiple leagues and seasons/allratingv.csv` = historical matches + future fixtures list (used for predictions).
- `multiple leagues and seasons/upcoming.csv` = supplemental odds/stats (BTTS odds `gg/ng`, per-match stats when available).

**Optional files** (for enhanced features):
- `AGS.CSV` = goal-minute lists (enables timing flags for Milestone 7.2).
- `goals statistics.csv` = per-match stats backup.

**Supported Leagues (12)**:
- England: Premier L, Championship
- Italy: Serie A, Serie B  
- Spain: Primera
- Germany: Bundesliga
- France: Ligue 1, Ligue 2
- Portugal: Primeira L
- Netherlands: Eredivisie
- Turkey: Super Lig
- Romania: Liga 1

## Install

```
pip install -r requirements.txt
```

## Run (One Command)

Full refresh (history -> baselines -> Elo -> stats backfill -> features -> train -> predict).
History rebuilds automatically when any file in `CGM data/` is newer than
`data/enhanced/cgm_match_history.csv`; use `--rebuild-history` to force.

```
python predict.py --max-date YYYY-MM-DD
```

Default picks are goals-only: O/U 2.5 + BTTS (GG/NG).
Legacy 1X2 pick engine and audits are archived under `archive/legacy_full_engine/`.

Goals-only picks (Milestone 7: O/U 2.5 + BTTS only):

```
python predict.py --max-date YYYY-MM-DD --pick-engine goals
```

Odds-free internal model (mu/probabilities do NOT use odds/market probabilities as features; EV/picks still require odds in the upcoming feed):

```
python predict.py --max-date YYYY-MM-DD --model-variant no_odds
```

Predict-only (uses existing `data/enhanced/*` artifacts + `models/*`; still runs the pick engine and writes `reports/picks.csv` and `reports/picks_debug.csv`. If CGM exports are newer than the cached history, predict-only exits unless you pass `--allow-stale-history`):

```
python predict.py --predict-only
```

Live scope note: the pipeline filters the upcoming feed to fixtures strictly after `--max-date` (no retro predictions). If your CGM export contains only played matches, you will correctly get 0 predictions/picks.
Narrator runs automatically after picks and writes `reports/picks_explained.csv` and `reports/picks_explained_preview.txt`.

## Outputs

- Match history (canonical):
  - `data/enhanced/cgm_match_history.csv`
  - `data/enhanced/cgm_match_history_with_elo.csv`
  - `data/enhanced/cgm_match_history_with_elo_stats.csv`
  - `data/enhanced/cgm_match_history_with_elo_stats_xg.csv`
- Training matrix:
  - `data/enhanced/frankenstein_training.csv`
- Models:
  - `models/frankenstein_mu_home.pkl`
  - `models/frankenstein_mu_away.pkl`
  - `models/frankenstein_mu_home_no_odds.pkl` (optional)
  - `models/frankenstein_mu_away_no_odds.pkl` (optional)
- Predictions:
  - `reports/cgm_upcoming_predictions.csv`
    - includes BTTS probabilities/odds/EV and timing risk flags (no timing markets) when `AGS.CSV` is present
- Picks:
  - `reports/picks.csv`
  - `reports/picks_debug.csv` (candidate-level gate/score debug)
- Narrator (Milestone 8):
  - `reports/picks_explained.csv` (picks + human-readable fields)
  - `reports/picks_explained_preview.txt` (quick text preview)
- Logs (JSONL):
  - `reports/run_log.jsonl`
  - `reports/elo_trace.jsonl` (optional sampled traces; see `cgm/predict_upcoming.py`)

## Pick Engine (Goals-only)

Run standalone (consumes `reports/cgm_upcoming_predictions.csv`):

```
python -m cgm.pick_engine_goals --in reports/cgm_upcoming_predictions.csv --out reports/picks.csv
python -m scripts.audit_picks_goals --predictions reports/cgm_upcoming_predictions.csv
```

Output note: `reports/picks.csv` includes a `score` column (EV + small reliability nudges) used to deterministically select the single best market per fixture, plus narrator-friendly columns (`model_prob`, `implied_prob`, `value_margin`, `risk_flags`).

Narrator (Milestone 8: human-readable explanations)

```
python -m cgm.narrator --in reports/picks.csv --out reports/picks_explained.csv
python -m scripts.audit_narrator
```

Audit no-odds invariance (mu/probs unchanged if odds change):

```
python -m scripts.audit_no_odds --upcoming "CGM data/multiple leagues and seasons/allratingv.csv" --as-of-date YYYY-MM-DD
```

Upcoming feed scope audit (shows how many rows get dropped by each filter step):

```
python -m scripts.audit_upcoming_feed --as-of-date YYYY-MM-DD
```

Run ALL audits (10 scripts, comprehensive validation):

```
python scripts/run_all_audits.py
```

Multi-league coverage audit:

```
python scripts/audit_multi_league.py
```

## Prediction Report

Generate formatted prediction tables (O/U, BTTS, EV):

```
python scripts/generate_predictions_report.py
python scripts/generate_predictions_report.py --league "Premier L"
```

Output: `reports/predictions_report.md` and `reports/predictions_report.txt`

## Telegram Automation (Weekly)

Weekly picks snapshot (Monday report, picks only, grouped by date/league):

```
python scripts/telegram_weekly.py
```

Notes:
- Weekly snapshots are stored under `reports/weekly_snapshots/YYYY-MM-DD/`.
- Filters: only picks with odds >= `TELEGRAM_MIN_ODDS` are reported (odds are not shown in the message).

## Backtest (Milestone 12)

Run historical backtests to validate model performance on past matches. The system uses "time-travel" simulation where predictions are generated as if made on each historical date (no future leakage).

```bash
# Run backtest for EPL 2025-2026 season
python -m scripts.run_backtest --league "Premier L" --season "2025-2026" --start-date "2025-09-01"

# Output: reports/backtest_epl_2025.csv
```

Options:
- `--league`: League name as stored in history (e.g., "Premier L")
- `--season`: Season format (e.g., "2025-2026")
- `--start-date`: Start date for backtest range (YYYY-MM-DD)
- `--history`: Path to history CSV (default: `data/enhanced/cgm_match_history.csv`)
- `--out`: Output file path

## Streamlit UI

Launch the web dashboard to visualize predictions, picks, and backtest results:

```bash
python -m streamlit run ui/streamlit_app.py
```

Tabs:
- **📊 Predictions**: Upcoming match predictions with expected goals and probabilities
- **🎯 Picks**: Value bets from the pick engine with EV and stake recommendations
- **📈 Statistics**: Summary metrics across all predictions
- **🔙 Backtest Results**: Historical performance review focusing on GG/NG and O/U 2.5 markets

## CGM Column Legend (DO NOT DELETE)

CGM exports use multiple screens/tables, and column names can vary slightly between them. This legend captures the common meanings and the most frequent variants seen in your exports.

### Match identity

- `sezonul` = season number (ex: `26` for 2025-26 in your case).
- `datameci` = match date.
- `etapa` = round / matchweek.
- `txtechipa1` = home team name.
- `txtechipa2` = away team name.
- `place1` / `place2` = standings place (rank) for home / away.
- `etat1` / `etat2` = team state / status used by the tool.
- `codech1` / `codech2` = team code (behaves like a stable team id in exports; pipeline uses it as a deterministic join key when present).
  - Common variants: `codechipa1` / `codechipa2`.

### Score / result

- `scor1` / `scor2` = final score home / away.
- `scorp1` / `scorp2` = half-time score home / away.
- `result` = match/market result depending on the export table (in match exports it's effectively the match result).

### Goal timing / events

- `goals` = goals scored in minutes (timing list).
- `mgolh` / `mgola` = minutes of goals for home / away team.
- `goalsh` / `goalsa` / `goalsha` (seen in some exports) = same concept with a slightly different formatting:
  - `goalsh`: minute list for home goals (example: `37,49,88,90`).
  - `goalsa`: minute list for away goals.
  - `goalsha`: combined list with team markers (format varies by export).

### Odds (goals lines)

- `cotao` / `cotau` = odds for Over 2.5 / Under 2.5.
- `cotaa` / `cotae` / `cotad` = 1X2 odds (legacy full engine only; goals-only pipeline ignores them).
- Some exports also include extra totals lines using a consistent naming pattern:
  - `cotao0` / `cotau0` (inferred) = Over/Under 0.5
  - `cotao1` / `cotau1` (inferred) = Over/Under 1.5
  - `cotao` / `cotau` = Over/Under 2.5 (documented)
  - `cotao3` / `cotau3` (inferred) = Over/Under 3.5
  - `cotao4` / `cotau4` (inferred) = Over/Under 4.5

### Both teams to score market (BTTS)

- `gg` / `ng` = odds for Both teams scored / At least one team did NOT score (BTTS Yes / BTTS No).

### PredictZ

- `ggp` / `ngp` = % of games that ended GG / NG.

### Match stats (shots / corners / possession / fouls / cards)

CGM exports can provide either split home/away columns, or packed columns that combine both teams as `H-A`.

Split columns (home/away):
- `ballph` / `ballpa` = possession home / away.
- `suth` / `suta` = shots home / away.
- `sutth` / `sutta` = shots on target home / away.
- `corh` / `cora` = corners home / away.
- `fouh` / `foua` = fouls home / away.
- `ych` / `yca` = yellow cards home / away.
- `rateh` / `ratea` = rating/grade home / away (tool-specific).

Packed `H-A` columns (common in `goals statistics.csv`-style exports):
- `ballp` = possession as a single string like `61-39` (inferred: `ballph-ballpa`).
- `sut` = shots as home-away (inferred: `suth-suta`).
- `sutt` = shots on target as home-away (inferred: `sutth-sutta`).
- `cor` = corners as home-away (inferred: `corh-cora`).

### Filters / market analytics

- `market` = market name.
- `winp` = win percent.
- `winnr` = number of wins.
- `coef` = coefficient / odds (generic).
- `filtru` = filter.

### Elo-related columns

- `elohome` / `eloaway` = Elo coefficients for home/away team.
- `elohomeo` / `eloawayo` = Elo old / previous Elo.
- `elodiff` = typically home-away Elo difference, but confirm the meaning per export screen/table.

### UI table abbreviations (common grid headings)

- `ShH` / `ShA` = shots home / away.
- `StH` / `StA` = shots on target home / away.
- `CorH` / `CorA` = corners home / away.
- `FouH` / `FouA` = fouls home / away.
- `YcH` / `YcA` = yellow cards home / away.
- `BpH` / `BpA` = ball possession home / away.
