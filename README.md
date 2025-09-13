<div align="center">

# AI Football Predictions

Hybrid XGBoost + Probabilistic engine for 1X2, OU 2.5 and Total‑Goals intervals, with calibration, odds integration and compact round reports.
Current highlights:
- Micro xG (ShotXG) enrichment (possession/corners/xG‑per‑possession/xG‑from‑corners EWMAs)
- JSON model format for XGBoost (no pickle version warnings)
- One‑click weekly scripts (native + Docker)
- API integrations: odds (API‑Football), optional possession (fixture statistics)
- Backtests (CRPS/LogLoss) and a simple ROI backtester (1X2 + OU 2.5)

</div>

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

- Models: XGBoost saved as JSON (no pickle warnings); PKL kept as fallback.
- Micro‑xG Enrichment: possession, corners, xG/possession, xG from corners (EWMAs) when available.
- APIs:
  - Odds: API‑Football (works for upcoming fixtures).
  - Possession (fixture statistics): API‑Football; requires plan access to target seasons (season = start year).
- Backtesting: CRPS/LogLoss (scripts/backtest_xg_source.py) and ROI (scripts/roi_backtest.py, 1X2 + OU 2.5).
- No H2H: model uses team home/away form (EWMAs), Elo (bands/similarity), micro aggregates, corners, absences.

## Quick Start (Essentials)

- Install deps: `pip install -r requirements.txt`
- Odds (API‑Football): `python -m scripts.fetch_odds_api_football --league E0 --days 7`
- Possession (if plan allows):
  - `python -m scripts.fetch_possession_apifootball --league E0 --season 2024 --dates 2024-08-01,2025-05-31 --out data/processed/E0_possession.csv`
  - `python -m scripts.build_micro_aggregates --league E0 --shots data/shots/understat_shots.csv --out data/enhanced/micro_agg.csv`
- Train: `python xgb_trainer.py --league E0`
- Picks (with odds): `python bet_fusion.py --with-odds --top 20`
- Combined table (TG, OU, 1X2 per match): `python scripts/print_best_per_match.py`
- Backtest metrics: `python -m scripts.backtest_xg_source --league E0 --start 2024-08-01 --end 2025-06-01 --sources micro macro blend --dist negbin --k 4`
- ROI (flat 1u): `python -m scripts.roi_backtest --league E0 --start 2024-08-01 --end 2025-06-01 --xg-source micro --dist negbin --k 4 --stake-policy flat --stake 1.0 --place both`

## API Notes

- Keys: set in `.env` or env vars: `API_FOOTBALL_KEY` (and optionally `API_FOOTBALL_ODDS_KEY`).
- Season = start year (e.g., 2024 for 2024–2025).
- Fixture statistics (possession) return only for supported plans and completed matches.

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

```
python one_click_predictor.py --league E0 --fixtures-csv data/fixtures/E0_manual.csv
```

- One-click predictor with API fixtures (set key):

```
$env:API_FOOTBALL_KEY = "your_key"
python one_click_predictor.py --league E0
```

- Odds fetch (API‑Football; next 7 days):

```
python -m scripts.fetch_odds_api_football --league E0 --days 7
```

- Possession fetch (API‑Football fixture statistics; plan‑gated; season uses start‑year):

```
python -m scripts.fetch_possession_apifootball --league E0 --season 2024 --dates 2024-08-01,2025-05-31 --out data/processed/E0_possession.csv
python -m scripts.build_micro_aggregates --shots data/shots/understat_shots.csv --league E0 --out data/enhanced/micro_agg.csv
```

<!-- Note: legacy Poisson runner archived (run_predictions.py). Use one_click or report tools for the full engine. -->

## Fixtures & Odds

- If no weekly fixtures exist, a manual template is created at `data/fixtures/{LEAGUE}_manual.csv`.
- Team stats snapshots are built from `data/processed/{LEAGUE}_merged_preprocessed.csv` if enhanced features are missing.
- Odds are optional; placeholder odds are used for demos unless configured.

## Betting Bot

Prereqs: trained models present in `advanced_models/` and processed data in `data/processed/`.
Legal: ensure compliance in your jurisdiction. Use responsibly.

## Reporting (Dashboard)

Run the Streamlit app:

```
streamlit run dashboard/app.py
```

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
3) Optional API keys (.env at repo root or environment variables)
- API_FOOTBALL_KEY=your_key    (used by possession and odds; odds fetcher also reads API_FOOTBALL_ODDS_KEY)
- API_FOOTBALL_ODDS_KEY=your_key

## Operational Guide (Local Venv)

- Create and activate venv (Windows PowerShell):
  - `py -3.11 -m venv .venv311`
  - `Set-ExecutionPolicy -Scope Process Bypass; .\\.venv311\\Scripts\\Activate.ps1`
- Install dependencies: `pip install -r requirements.txt`
- Set API key (or use `.env`): `$env:API_FOOTBALL_KEY = "your_key"`
- Fetch odds (real odds saved to `data/odds/{LEAGUE}.json`):
  - `python -m scripts.fetch_odds_api_football --league E0 --days 7`
- Build micro aggregates (uses shots + optional possession/corners if present):
  - `python -m scripts.build_micro_aggregates --league E0 --shots data/shots/understat_shots.csv --out data/enhanced/micro_agg.csv`
- Train per-league models: `python xgb_trainer.py --league E0`
- Generate picks with odds: `python bet_fusion.py --with-odds --top 20`
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

- Upcoming Predictions now use real odds when available (BOT_ODDS_MODE=local reads `data/odds/{LEAGUE}.json`).
- Bets logging (optional): set `BOT_LOG_BETS=true` before running `bet_fusion.py` to write `data/bets_log.csv` and track bankroll in `data/bankroll.json`.

### Troubleshooting

- Matplotlib missing: ensure `pip install -r requirements.txt` ran in the venv. If a later `pip install` upgraded `numpy` and caused conflicts, pin back with `pip install --force-reinstall --no-deps numpy==1.23.5` and `pip install --no-deps matplotlib==3.8.4`.
- ModuleNotFoundError: `config` in Streamlit: run `streamlit` from the repo root (so `config.py` is importable). The app also adds the project root to `sys.path` automatically.
