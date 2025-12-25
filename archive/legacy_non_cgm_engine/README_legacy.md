<div align="center">


Repository: https://github.com/rzvsd/AIFootballPredictions (branches: main, test)
Hybrid XGBoost + Probabilistic engine for 1X2, OU 2.5 and Total‑Goals intervals, with calibration, odds integration and compact round reports.
Current highlights:
- Micro xG (ShotXG) enrichment (possession/corners/xG‑per‑possession/xG‑from‑corners EWMAs)
- JSON model format for XGBoost (no pickle version warnings)
- One‑click weekly scripts (native + Docker)
- API integrations: odds (API‑Football), optional possession (fixture statistics)
- Backtests (CRPS/LogLoss) and a simple ROI backtester (1X2 + OU 2.5)

</div>

## Latest Updates

- **Understat Fetcher:** `scripts/fetch_understat_simple.py` now runs concurrent match requests (configurable semaphore), adds a User-Agent, and enforces the Windows selector event loop to prevent hangs.
- **Understat Fixture Sync:** `predict.py`/`bet_fusion.py` call the Understat fixture helper directly (no API-Football or manual CSV fallbacks). The CLI `scripts/fetch_fixtures_understat.py` remains available for manual snapshots.
- **Odds Simplification:** Value metrics now use Bet365 prices when available (via `fetch_odds_fd_simple`); if the feed is empty the engine falls back to deterministic placeholders (2.00/1.90/3.00) so runs remain reproducible.
- **Backdoor Odds:** Added `scripts/fetch_odds_fd_simple.py`, a lightweight downloader for football-data.co.uk's live fixtures feed (Bet365 odds). `predict.py` now runs it automatically so bet_fusion, reports, and dashboards see real prices whenever the CSV is available.
- **Model Freshness Guard:** `predict.py` now requires both home and away XGB files (JSON/PKL) before skipping a retrain.
- **Absences/Micro Aggregates:** `xgb_trainer.py` scopes absences by league and `bet_fusion.py` filters micro aggregates by `BOT_SNAPSHOT_AS_OF` (no leakage across leagues or future dates).
- **Team Normalization:** Added UTF‑8/ASCII Bundesliga variants (e.g., Bayern München, Köln, Mönchengladbach).
- **Understat Dependencies:** Installed `understat`, `aiohttp`, and helpers so Understat ingestion works without manual setup.
- **Thresholds:** Updated default betting thresholds and config entries to allow more 1X2/OU/DC candidates while retaining minimum edges.
- **Reporting:** Added a “match report” table format (see below) outlining the recommended way to present picks per game.

## Bot Blueprint (Capabilities)

1. **Data Acquisition**
   - Downloads multi-season CSVs from football-data.co.uk via `scripts/data_acquisition.py`.
   - Processes raw data into `data/processed/{LEAGUE}_merged_preprocessed.csv` with feature selection & clustering.
2. **Micro Aggregates**
   - Fetches Understat shots & matches concurrently, builds per-shot aggregates (`scripts.shots_ingest_understat`, `scripts.build_micro_aggregates`).
   - Persists league-specific micro aggregates (`data/enhanced/{LEAGUE}_micro_agg.csv`).
3. **Model Training**
   - `xgb_trainer.py` builds directional EWMA features, Elo, micro-enrichment, absences, and trains XGB regressors (home/away goals) per league.
   - Saves models in JSON and PKL; predict.py ensures freshness based on retrain window.
4. **Fixtures (Understat-Only)**
   - `scripts/fetch_fixtures_understat.py` (and helper functions imported by `predict.py`/`bet_fusion.py`) pull the upcoming schedule straight from Understat using browser headers and normalized team names.
   - If Understat ever returns an empty window, the engine simply reports “no fixtures”; there are no fragile API fallbacks or manual CSV templates to maintain.
5. **Bet365 Odds**
   - `scripts/fetch_odds_fd_simple.py` downloads football-data.co.uk’s `fixtures.csv`, filters it for the supported leagues, and saves Bet365 prices to `data/odds/{LEAGUE}.json`.
   - `bet_fusion.py` reads those JSON files automatically (per league) and falls back to placeholder odds only when a market isn’t listed.
6. **Prediction / Betting Engine**
   - `bet_fusion.py` loads per-league models, blends macro/micro xG, builds Poisson/NegBin score matrices, evaluates 1X2/OU/TG markets, applies calibrators, and ranks bets by EV.
   - Supports configurable thresholds, staking policies, odds fetching (local or API), and optional bankroll logging.
7. **Orchestration**
   - `predict.py` hooks everything: data refresh, micro rebuild, absences seed, conditional retrain, and predictions via `bet_fusion.py` (which now fetches fixtures from Understat on demand and uses deterministic placeholder odds).
7. **Reporting & Exports**
   - Picks saved per run (`reports/{LEAGUE}_{timestamp}_picks.csv`).
   - Dashboard (`dashboard/app.py`) visualizes stats when data/models exist.

## Recommended Output Table

When sharing picks per fixture, use a consolidated table with the following columns:

| Date/Time           | Teams                         | TG Interval                                           | 1X2                                              | Double Chance                    | Over/Under                                          |
|---------------------|-------------------------------|------------------------------------------------------|--------------------------------------------------|----------------------------------|----------------------------------------------------|
| 2025-12-06 12:30:00 | Aston Villa FC vs Arsenal FC | TG Interval 0-3 (p=79.7%, odds=3.00, edge=46.36%, EV=1.39) | – (threshold not met)                            | – (threshold not met)            | OU 3.5 Under (p=79.7%, odds=1.90, edge=29.70%, EV=0.51) |
| 2025-12-06 15:00:00 | Bournemouth vs Chelsea FC    | TG Interval 2-5 (p=63.3%, odds=3.00, edge=29.95%, EV=0.90) | –                                               | –                                | OU 1.5 Over (p=82.1%, odds=1.90, edge=32.09%, EV=0.56)   |
| 2025-12-15 20:00:00 | Manchester United vs Bournemouth | TG Interval 2-5 (p=63.6%, odds=3.00, edge=30.28%, EV=0.91) | 1X2 A (p=52.3%, odds=2.00, edge=18.92%, EV=0.05) | –                                | OU 1.5 Over (p=77.6%, odds=1.90, edge=27.61%, EV=0.47)   |

Notes:
- Use “–” when a market doesn’t pass the configured thresholds (probability or edge).
- Sort rows chronologically; include EV, edge, and implied probability for each showcased pick.


## One-Command Orchestrator

- Single league: `python predict.py --league E0`
- Multiple leagues: `python predict.py --leagues E0 D1 F1 I1 SP1`
- What it does (per league): refresh data (raw->processed), rebuild micro aggregates, seed absences if missing, (re)train if stale, and print top picks + per-match summary (fixtures pulled from Understat automatically).
- Requirements: Python 3.11+ and `pip install -r requirements.txt`. No external API keys needed.
- Min‑odds filter: set `BOT_MIN_ODDS=1.6` (default 1.60) to exclude very mici cote din ranking.

- Compact round table in one go:

  `python predict.py --league D1 --days 21 --compact`

## CGM / Elo + Pressure Pipeline (current)

- Recompute Elo from CGM results only (start 1500, K=20, home adv 65, margin multiplier), with date cutoff to avoid future leakage:  
  `python -m scripts.calc_cgm_elo --max-date YYYY-MM-DD` (defaults to today UTC).  
  Output: `data/enhanced/cgm_match_history_with_elo.csv` (clean Elo; canonical + `_calc` aliases).
- Backfill per-match gameplay stats (shots/SOT/corners/possession) from `CGM data/goals statistics.csv` into the canonical history:  
  `python -m cgm.backfill_match_stats`  
  Output: `data/enhanced/cgm_match_history_with_elo_stats.csv` (coverage depends on what seasons/leagues exist in your CGM exports).
- Build training features (rolling stats, bands, Elo similarity kernel, plus Milestone 2 Pressure Cooker features):  
  `python -m cgm.build_frankenstein --match-history cgm_match_history_with_elo_stats.csv`
- Train models:  
  `python -m cgm.train_frankenstein_mu`
- Predict upcoming (uses the same Elo + similarity logic):  
  `python cgm/predict_upcoming.py`

Notes:
- Similarity kernel uses league-specific sigma (`config.ELO_SIM_SIGMA_PER_LEAGUE`, default 60), with effective sample size blending.
- Pressure Cooker (Milestone 2) adds venue-aware rolling dominance (L10 home-only / away-only) and divergence vs team-level Elo using league z-scores:
  - Core: `press_form_H`, `press_form_A`, `press_n_H`, `press_n_A`
  - Divergence: `div_team_H`, `div_team_A`, `div_diff` (and optional debug `press_z_*`, `elo_z_*`)
- Always set a cutoff on Elo recompute to prevent future fixtures from leaking into ratings.

  This refreshes data, trains if needed, populates fixtures/odds (with fallback), prints picks, and shows the compact per-match prognostics.

## CGM CSV Legend (offline data source)

### multiple seasons.csv (historical matches, multi-season)
- Identification: country, league, datameci (date), orameci (time), etapa (round), txtechipa1/txtechipa2 (home/away names), place1/place2 (table positions), alert, verif.
- O/U (main line, usually 2.5): underprob, underoddsc (“fair” under), cotau (book under); overprob, overoddsc (“fair” over), cotao (book over).
- 1X2: homeprob, homeoddsc (“fair” home), cotaa (book home); drawprob, drawoddsc, cotae (book draw); awayprob, awayoddsc, cotad (book away).
- Result: score (text), validated (flag), result (1/X/2), scor1/scor2 (FT goals), scorp1/scorp2 (HT goals).
- Elo/strength: elohomeo, eloawayo, elodiff.
- Goal/performance aggregates: totalgh/totalga/totaldiff, scoredh/scoreda/scodiff, receivedh/receiveda/recdiff.
- Goal profile: ggh/gga, ngh/nga, gh0–gh4 (home goals freq), ga0–ga4 (away goals freq), goalminh/goalmina (goal-minute profile).
- Form: formah, formaa.
- Codes: codechipa1, codechipa2 (team IDs).

### goals statistics.csv (season match stats)
- Identification/result: sezonul, datameci, etapa, txtechipa1/txtechipa2, scor1/scor2, scorp1/scorp2, result, validated.
- Match stats: ballp (possession “home-away”), sut (shots), sutt (shots on target), cor (corners).
- Odds: cotaa/cotae/cotad (1X2), cotao/cotau (OU).
- Ratings/form: elohomeo/eloawayo, elohomedif/eloawaydif (delta), formah/formaa.
- Codes: codechipa1, codechipa2.

### goal statistics 2.csv (team standings & season aggregates)
- Overall: loc (rank), echipa (team), rating, mj (played), mc/me/mp (wins/draws/losses), gm/gp (scored/conceded), pct (points), codechipa.
- Home table: loch, echipah, mjh/mch/meh/mph, gmh/gph, pcth, codechipah.
- Away table: loca, echipaa, mja/mca/mea/mpa, gma/gpa, pcta, codechipaa.

### leageue statistics.csv (league-level snapshot)
- Identification: no (idx), country, league, rating, nextgame, sezonul, games.
- Outcome frequencies: home_val, draw_val, away_val, over_val, under_val, gg_val (BTTS yes), ng_val (BTTS no).
- Goal/score distribution: game1st/std1st, sco1st/stdsco, con1st/stdcon.
- League averages per match: gsco_val, grec_val, suth_val/suta_val (shots), sutht_val/sutat_val (shots on target), corh_val/cora_val (corners), foulsh_val/foulsa_val, ycardh_val/ycarda_val, ballph_val/ballpa_val (possession).

### upcoming - Copy.CSV (upcoming fixtures with CGM stats/odds)
- Identification: country, league, alert, verif, sezonul, datameci (date), orameci (time), etapa (round).
- Teams/positions: txtechipa1/txtechipa2, place1t/place1a (home positions), place2t/place2d (away positions).
- Custom flags: customh, customa.
- Home strength indices: home_val, avghome, home_val_2/3/4/5.
- Away strength indices: away_val, avgaway, away_val_2/3/4/5.
- Odds: cotaa/cotae/cotad (1X2); cotao0/cotau0, cotao1/cotau1, cotao/cotau (main OU), cotao3/cotau3, cotao4/cotau4 (alt OU); gg/ng (BTTS if present).
- Result (if played): result, scor1/scor2, scorp1/scorp2.
- Form/H2H: formah, formaa, formadif, h2h.
- Recent stats: suth/suta (shots), sutht/sutat (shots on target), corh/cora (corners), foulsh/foulsa, yellowh/yellowa, ballph/ballpa, goalminh/goalmina.
- Streak/summary: swinh/sdrawh/slosth, sscoreh/sreceiveh, soverh/sunderh (home); swina/sdrawa/slosta, sscorea/sreceivea, sovera/sundera (away).
- Misc: jucatori1/jucatori2 (player notes), wdlh1–wdlh5 and wdla1–wdla5 (recent W/D/L sequences).


- All leagues examples:

  `python predict.py --league E0 --days 21 --compact`
  `python predict.py --league D1 --days 21 --compact`
  `python predict.py --league F1 --days 21 --compact`
  `python predict.py --league I1 --days 21 --compact`
  `python predict.py --league SP1 --days 21 --compact`

- Multi-league (prints a section per league):

  `python predict.py --leagues E0 D1 F1 I1 SP1 --days 21 --compact`
### Fallbacks and Fixtures (New)

- If the primary odds/fixtures provider (API-Football) returns an empty window, the orchestrator now:
  1) Generates a weekly fixtures CSV via football-data.org: `data/fixtures/{LEAGUE}_weekly_fixtures.csv`
  2) Refetches odds using that CSV to align team names and dates

  This removes manual steps when international breaks or plan limits cause empty API-Football responses.

- Team name normalization expanded for Bundesliga (D1) to cover umlaut/variant names (e.g., "FC Bayern München", "1. FC Köln", "Borussia Mönchengladbach").

### Full Round Table
## Project Status Summary

- One-command orchestrator: `predict.py` refreshes data, builds micro aggregates, retrains if stale, and prints picks + per-match tables. Add `--compact` to print concise prognostics per game.
- Compact views:
  - Next round: `python -m scripts.print_best_per_match --by prob` or `predict.py --compact`.
  - Historical evaluations currently require manual score joins (the former API-based results mode has been deprecated).
- Fixtures & odds:
  - Fixtures are sourced exclusively from Understat (async fetch with browser headers).
  - Odds are deterministic placeholders (2.00/1.90/3.00) until a real feed is wired in again.
- Models & features:
  - XGBoost goal models (JSON preferred) + NegBin/Poisson score matrix.
  - ShotXG micro aggregates (Understat optional); proceeds with existing shots if Understat package is unavailable.
  - Per-league calibrators supported; defaults are safe if missing.
- Team names: normalization map expanded (Bundesliga umlauts/variants). Additional variants can be added as needed.
- Known caveats:
  - Placeholder odds mean edge/EV magnitudes are illustrative only; integrate a bookmaker feed if you need real staking decisions.
  - Console may show encoding artifacts for some team names; normalization still works for modeling.

- To print a compact table for the next round (all games; 1X2, OU 2.5, and Total-Goals intervals):

```
python -m scripts.print_best_per_match --by prob   # or --by ev
```

- The orchestrator already prints picks and a per-match summary; the command above is handy for exporting/printing the full next-round view.

## Table of Contents

1. Project Overview
2. Directory Structure
3. Setup & Installation
4. Weekly Pipeline
5. Core Commands
6. Fixtures & Odds
7. Betting Bot
8. Reporting (Dashboard)
9. Supported Leagues


10. Contributing
11. License
12. Disclaimer

## Project Overview

AIFootballPredictions builds per-league expected goals models (XGBoost regressors for home/away goals), converts them to a Poisson score matrix, and derives market probabilities for:

- 1X2 (home/draw/away)
- Over/Under 2.5
- Total Goals intervals (e.g., 0–3, 2–5)

Per-league probability calibrators (isotonic/Platt) correct biases. A compact report prints tables, and an optional betting bot ranks value picks and logs bankroll.

## Current Status

- Engine split: `bet_fusion.py` delegates to services:
  - `engine/predictor_service.py` (fixtures, snapshots, score matrices)
  - `engine/market_service.py` (probabilities for 1X2/DC/OU/TG)
  - `engine/odds_service.py` (load odds JSON, fuzzy lookup, missing log)
  - `engine/value_service.py` (attach odds, price_source, edge/EV; EV blank for synth)
- Modes:
  - `mode=sim` (default): allows placeholders/synthetic odds for completeness.
  - `mode=live`: real odds only; no placeholders/synth; rows without odds are dropped; TG intervals excluded unless feed supports them.
- Fixtures: Understat primary; if Understat fails, fixtures are synthesized from processed data to keep runs alive.
- Odds: The-Odds-API via `scripts/fetch_odds_toa.py`; missing odds log per league; synthetic odds only in sim mode.
- Models: XGBoost saved as JSON (PKL fallback); goals retained in preprocessing so Elo/Form move correctly.
- Dashboard: Reads reports/engine; shows price_source (real vs synth); EV blank when odds are synthetic.
- Layout: `engine/`, `strategies/`, `data_pipeline/`, `models/`, `risk/`, `ui/`, root orchestrator `bet_fusion.py`.

### Data shapes (shared language)

- MarketProbDF (from `market_service.build_market_book`):
  `date, home, away, market, outcome, prob` (optionally league).
- ValuedMarketDF (after odds + EV):
  MarketProbDF plus `odds, fair_odds, book_odds, edge, EV, price_source`.
- PicksDF (strategy output):
  At least `date, league, home, away, market, outcome, prob, odds, edge, EV, strategy`.

### Strategy conventions
- Strategies in `strategies/` return PicksDF and set a `strategy` column.
- Market string format for logging: `"1X2 H"`, `"OU 2.5 Over"`, `"TG Interval 0-3"`, `"DC 1X"`.
- Bankroll log columns stay: `date, league, home_team, away_team, market, selection, odds, stake, model_prob, expected_value`.

## Quick Start (Essentials)

- Install deps: `pip install -r requirements.txt`
- Build micro aggregates (from Understat shots):
  - `python -m scripts.build_micro_aggregates --league E0 --shots data/shots/understat_shots.csv --out data/enhanced/micro_agg.csv`
- Fetch odds via The-Odds-API (set `THE_ODDS_API_KEY` in env or .env):
  - `python -m scripts.fetch_odds_toa --leagues E0 D1 F1 I1 SP1`
- Train: `python xgb_trainer.py --league E0`
- Picks (uses Bet365 odds when available): `python bet_fusion.py --top 20`
- Combined table (TG, OU, 1X2 per match): `python scripts/print_best_per_match.py`
- Backtest metrics: `python -m scripts.backtest_xg_source --league E0 --start 2024-08-01 --end 2025-06-01 --sources micro macro blend --dist negbin --k 4`
- ROI (flat 1u): `python -m scripts.roi_backtest --league E0 --start 2024-08-01 --end 2025-06-01 --xg-source micro --dist negbin --k 4 --stake-policy flat --stake 1.0 --place both`

## How to Use the Bot (Daily Ops)

- **Refresh historical data (past rounds/years):** `python -m scripts.data_acquisition --leagues E0 D1 F1 I1 SP1 --seasons 2526 2425 2324 --raw_data_output_dir data/raw` followed by `python -m scripts.data_preprocessing --raw_data_input_dir data/raw --processed_data_output_dir data/processed`. This re-downloads football-data.co.uk CSVs for every listed season so the next training run has the full history.
- **Train/retrain league models:** `python xgb_trainer.py --league E0` (repeat per league). The trainer consumes the processed files created in the previous step, so running it after refreshing data retrains the models on all historical seasons you just ingested.
- **Update odds + catch team-name mismatches:** `python -m scripts.fetch_odds_fd_simple --leagues E0 D1 F1 I1 SP1 --days 14` to refresh Bet365 prices, then `python -m scripts.check_odds_alignment --league E0 --days 14` (swap the league code as needed). If the script reports warnings, add the missing variants to `config.TEAM_NAME_MAP` and rerun it until everything lines up.
- **Generate predictions / reports:** `python predict.py --leagues E0 D1 F1 I1 SP1 --days 7 --compact`. The orchestrator pulls Understat fixtures, rebuilds micro aggregates if needed, ensures odds exist (calling the command above when necessary), loads the latest models, and prints/saves the tables described in the Recommended Output section.

## Fixture Behavior

- No external APIs required. Fixtures are fetched solely from Understat and normalized through `config.TEAM_NAME_MAP`.
- `scripts/fetch_fixtures_understat.py --leagues E0 D1 F1 I1 SP1 --days 14` mirrors what the orchestrator does internally if you need a manual snapshot (CSV output optional).
- Odds are synthetic placeholders (2.00/1.90/3.00) until a trusted bookmaker feed is integrated again.

## Bluebook (Quick Facts)

- Engine: XGB μ → NegBin/Poisson score matrix → markets (1X2/OU/TG) → calibration.
- xg_source: micro by default (ShotXG aggregates). No H2H used.
- Enrichment: possession/corners/xG‑per‑possession/xG‑from‑corners (EWMAs) when available.
- Models: JSON preferred (stable across versions); PKL is fallback.
- One‑click: `scripts/weekly_refresh.ps1` or `scripts/weekly_refresh_docker.ps1`.
- Backtests: CRPS/LogLoss (backtest_xg_source.py), ROI (roi_backtest.py; 1X2 + OU 2.5 using football‑data odds).

## Directory Structure (Clean)

```
AIFootballPredictions/
  advanced_models/            # XGB models per league (*.pkl)
  calibrators/                # Per-league calibrators (*.pkl)
  conda/                      # Conda environments (optional)
  data/
    raw/                      # Merged raw CSVs per league
    processed/                # Preprocessed training data
    enhanced/                 # Enhanced features (if present)
    fixtures/                 # Fixture CSVs (weekly or manual)
    picks/                    # Saved picks (optional)
    store/                    # Team stats snapshots (*.parquet)
    odds/                     # Odds cache (optional)
  dashboard/                  # Streamlit app
  scripts/                    # Python scripts (training, odds, reports, bot)
```

## Setup & Installation

1) Python 3.11 or 3.12 recommended (3.13 OK for most flows).
2) Install dependencies:

```
pip install -r requirements.txt
```

## Weekly Pipeline

- Ensure processed data exists in `data/processed/{LEAGUE}_merged_preprocessed.csv`.
- Ensure models exist in `advanced_models/` (trainer writes JSON + PKL fallback).
- Use one‑click scripts to refresh data, micro aggregates, models, and picks.

Native (PowerShell):

```
pwsh -File scripts/weekly_refresh.ps1 -League E0 -Seasons "2526 2425 2324" -UnderstatSeasons "2025,2024" -TryInstallUnderstat
```

Docker (PowerShell):

```
pwsh -File scripts/weekly_refresh_docker.ps1 -League E0 -Seasons "2526 2425 2324" -UnderstatSeasons "2025,2024" -Build -TryInstallUnderstat
```

## Core Commands

- Betting bot (manual fixtures fallback):

```
python scripts/betting_bot.py --league E0
```

- One-click predictor with manual fixtures:

Legacy one-click helpers were kept for reference, but the recommended path is `predict.py` (it handles fixture fetches + placeholder odds without extra flags).


<!-- Note: legacy Poisson runner archived (run_predictions.py). Use one_click or report tools for the full engine. -->

## Fixtures & Odds

- Fixtures fetched from Understat; if Understat returns empty, a mock schedule is synthesized from processed data (team names already normalized).
- Team stats snapshots are built from `data/processed/{LEAGUE}_merged_preprocessed.csv`; enhanced features used when present.
- Odds come from The-Odds-API via `scripts/fetch_odds_toa.py`; missing odds are logged (`reports/*_missing_odds.log`).
- Live mode odds flow: markets → `fill_odds_for_df` (odds only, no placeholders) → `attach_value_metrics(use_placeholders=False, synthesize_missing_odds=False)` → drop rows with missing odds. Run `python -m scripts.check_odds_alignment --league E0 --days 14` to catch team-name mismatches.

## Betting Bot

Prereqs: trained models present in `advanced_models/` and processed data in `data/processed/`.
Legal: ensure compliance in your jurisdiction. Use responsibly.

## Reporting (Dashboard)

Run the Streamlit app:

```
streamlit run dashboard/app.py
```

### If the dashboard shows no data or fails to load

- Verify dependencies are installed: `pip install -r requirements.txt` (includes `streamlit`).
- Run from the project root: `streamlit run dashboard/app.py` (the app assumes `config.py` and models are importable from CWD).
- Ensure models and snapshots exist for the league you want to view:
  - Quick path: `python predict.py --league E0 --days 7` (builds processed data, trains models if stale, seeds absences, and prints Understat-based picks).
- Fixture sources and seasons (important!):
  - Understat uses the season start year (e.g., `--season 2025` for 2025‑26). You can also set `BOT_UNDERSTAT_SEASON`.
  - If Understat returns no fixtures (off-season), the dashboard will simply report “No fixtures available”.
- Network checklist:
  - Ensure outbound HTTPS access to understat.com (the Python package scrapes JSON endpoints).
- Port conflicts: change the port if needed: `streamlit run dashboard/app.py --server.port 8502`.
- If bet logs are empty, bankroll/P&L tables will be empty by design. Enable logging in `bot_config.yaml` (`log_bets: true`) or via env `BOT_LOG_BETS=1`.

### Understat fixture automation

- Manual snapshot: `python -m scripts.fetch_fixtures_understat --leagues E0 D1 F1 I1 SP1 --days 14`
- Environment knobs added to make runs deterministic:
  - `--season` in `predict.py` (or `BOT_UNDERSTAT_SEASON`) — season start year.
  - `BOT_FIXTURES_DAYS` — window length used whenever fixtures are fetched.
- Name normalization matters for cross‑provider fixtures. The `TEAM_NAME_MAP` includes common variants (e.g., `FC Internazionale Milano` → `Inter`). If new team names appear, add them there.

## Overdispersion (Negative Binomial)

When modeling goals, a Negative Binomial distribution can better capture overdispersion than a pure Poisson.
The dispersion parameter `k` acts as a stability regulator:

- Smaller `k` → more variance in total goals (more extreme scorelines).
- Larger `k` → tighter distributions around the xG means (more stable scorelines).

Tuning `k` per league yields a more realistic “chaos level” for each competition. See `scripts/optimize_k.py` for a simple CRPS-based grid search.

## xG Sources (Codenames)

- MacroXG (aka FormXG): team-level xG means (μ_home/μ_away) via XGBoost from directional form (EWMAs), Elo, opponent context (Elo bands + kernel similarity), stabilizers. Converted to markets via Poisson/NegBin.
- ShotXG (aka MicroXG): per-shot xG model (XGBoost Classifier) trained on shot “DNA” (distance, angle, header, phase, etc.), then aggregated as EWMA per team (home/away) and injected into the final engine.

### Stage 1 — Shot “DNA” Ingestion

1) Fetch Understat leagues and per-match shots:
```
python -m scripts.fetch_understat_simple --league E0 --seasons 2024,2023
```
Outputs JSON files under `data/understat/`.

2) Ingest and engineer per-shot features to CSV:
```
python -m scripts.shots_ingest_understat --inputs data/understat/*_shots.json --out data/shots/understat_shots.csv
```
CSV contains key fields plus engineered `dist_m`, `angle_deg`, `is_header`.

### Stage 2 — Train ShotXG (per-shot model)

Train an XGBoost classifier for goal probability per shot and fit an isotonic calibrator:

```
python -m xg_shot_model.train_gbm \
  --shots data/shots/understat_shots.csv \
  --out models/shotxg_xgb.pkl \
  --calib models/shotxg_iso.pkl \
  --report reports/shotxg_metrics.json
```

### Stage 3 — Build Micro→Macro Aggregates

Aggregate per-team EWMA features (home/away) from per-shot probabilities:

```
python -m scripts.build_micro_aggregates \
  --shots data/shots/understat_shots.csv \
  --model models/shotxg_xgb.pkl \
  --calib models/shotxg_iso.pkl \
  --out data/enhanced/micro_agg.csv
```

Columns include per‑match and EWMA: xg_for, xg_against, finishing_efficiency (G - xG), goalkeeping_efficiency (xGA - GA), plus side (H/A).

Additional enrichment (per side, EWMA; overlaid when available):
- possession_for_EWMA, possession_against_EWMA
- corners_total_for_EWMA, corners_total_against_EWMA
- xg_for_per_poss_EWMA, xg_against_per_poss_EWMA
- xg_from_corners_for_EWMA, xg_from_corners_against_EWMA

### Reference Report Snapshots (Do Not Modify)

Frozen snapshots to serve as long-term reference for table outputs produced by the engine after Stage 4 (xG source integration). Keep these files unchanged.

- MicroXG (ShotXG only)
  - Env: `BOT_ODDS_MODE=local`
  - Command: `python -m scripts.quick_report --league E0 --xg-source micro --top 10`
  - Snapshot: `reports/snapshots/E0_quick_report_micro.md`

- BlendXG (MacroXG + ShotXG, w=0.5)
  - Env: `BOT_ODDS_MODE=local`
  - Command: `python -m scripts.quick_report --league E0 --xg-source blend --top 10`
  - Snapshot: `reports/snapshots/E0_quick_report_blend.md`

Notes:
- Quick report uses placeholder odds (local mode) to compute EV; market probabilities and picks derive from the engine with `xg_source` as specified.
- For full multi-league tables (1X2, OU 2.5, TG intervals, calibrators, odds), use `scripts/multi_league_report.py`.

### Production Default

- Default strategy: `xg_source: micro` (ShotXG aggregates) in `bot_config.yaml`.
- Rationale: empirical backtest shows lower CRPS and LogLoss vs MacroXG and BlendXG on E0 (see `scripts/backtest_xg_source.py`).
- NegBin (k) and per‑league calibrators remain active above the chosen μ-source.


## Supported Leagues

- Premier League: E0
- Serie A: I1
- Ligue 1: F1
- La Liga: SP1
- Bundesliga: D1

## Contributing

Fork and open PRs. For large changes, open an issue first.

## License

BSD-3-Clause (see LICENSE).

## Disclaimer

This project is for educational purposes. Predictions are uncertain and should not be the sole basis for financial decisions.
## Operational Guide (Local Venv)

- Create and activate venv (Windows PowerShell):
  - `py -3.11 -m venv .venv311`
  - `Set-ExecutionPolicy -Scope Process Bypass; .\\.venv311\\Scripts\\Activate.ps1`
- Install dependencies: `pip install -r requirements.txt`
- Build micro aggregates (uses shots + optional possession/corners if present):
  - `python -m scripts.build_micro_aggregates --league E0 --shots data/shots/understat_shots.csv --out data/enhanced/micro_agg.csv`
- Train per-league models: `python xgb_trainer.py --league E0`
- Fetch Bet365 odds (optional but recommended): `python -m scripts.fetch_odds_fd_simple --leagues E0 D1 F1 I1 SP1 --days 14`
- Generate picks: `python bet_fusion.py --top 20`
- Dashboard:
  - Start: `streamlit run dashboard/app.py`
  - Open: http://127.0.0.1:8501
  - Stop: close the Streamlit window or terminate the `streamlit` process

### Weekly Refresh (One-Click)

- `powershell -ExecutionPolicy Bypass -File scripts/weekly_refresh.ps1 -League E0`
- Notes:
  - Script is PowerShell-compatible (uses try/catch; no `||` tokens).
  - Understat package is optional; if unavailable, the script proceeds with existing shots.

### Dashboard Notes

- Upcoming Predictions currently use placeholder odds (no bookmaker feed required). Integrate a real feed later by extending `bet_fusion.py`.
- Bets logging (optional): set `BOT_LOG_BETS=true` before running `bet_fusion.py` to write `data/bets_log.csv` and track bankroll in `data/bankroll.json`.

### Troubleshooting

- Matplotlib missing: ensure `pip install -r requirements.txt` ran in the venv. If a later `pip install` upgraded `numpy` and caused conflicts, pin back with `pip install --force-reinstall --no-deps numpy==1.23.5` and `pip install --no-deps matplotlib==3.8.4`.
- ModuleNotFoundError: `config` in Streamlit: run `streamlit` from the repo root (so `config.py` is importable). The app also adds the project root to `sys.path` automatically.






### Elo pipeline (clean CGM-only)
- Recompute Elo from results only: `python -m scripts.calc_cgm_elo --max-date YYYY-MM-DD` (defaults to today UTC). Uses start 1500, K=20, home advantage 65, World Football Elo margin multiplier.
- Output: `data/enhanced/cgm_match_history_with_elo.csv` with canonical Elo columns (`elo_home/elo_away/elo_diff`); drops merge artefacts (`*_x`/`*_y` lg_avg columns), keeps `_calc` aliases for compatibility; future rows beyond max-date are excluded.
- Downstream: `cgm/build_frankenstein.py --match-history cgm_match_history_with_elo.csv`, then `cgm/train_frankenstein_mu`, then `cgm/predict_upcoming.py` all consume the clean Elo.
- Tip: always pass a max-date to avoid future leakage; rerun the sequence after new played matches arrive.
