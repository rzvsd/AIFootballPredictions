# Repository Guidelines

## Project Structure & Modules
- Root: model training, prediction, and betting utilities in Python.
- `scripts/`: pipeline steps (acquisition, preprocessing, training, calibration, reporting, bot).
- `advanced_models/`: trained XGB models per league (`{LEAGUE}_ultimate_xgb_{home,away}.pkl`).
- `data/`: `raw/`, `processed/`, `enhanced/`, `odds/`, `store/`, runtime outputs (`bankroll.json`, `bets_log.csv`).
- `reports/`: generated round/picks CSVs.
- `config.py`, `bot_config.yaml`: league, features, and runtime configuration.

## Build, Test & Dev Commands
- Acquire raw: `python -m scripts.data_acquisition --leagues E0 I1 SP1 F1 D1 --seasons 2425 2324`.
- Preprocess: `python -m scripts.data_preprocessing --raw_data_input_dir data/raw --processed_data_output_dir data/processed`.
- Train XGB per league: `python xgb_trainer.py --league E0`.
- Calibrate: `python -m scripts.calibrate_league --league E0 D1 F1 I1 SP1`.
- Weekly orchestration: `python -m scripts.update_and_report --leagues E0 D1 F1 I1 SP1 --season-codes 2425 --export` (add `--fetch-odds` for live odds).
- Report only: `python -m scripts.multi_league_report --leagues E0 D1 --export`.

## Coding Style & Naming
- Python 3.10+, 4‑space indent, type hints where practical.
- Functions/vars: `snake_case`; classes: `CapWords`; files: `lower_snake.py`.
- Keep functions small, pure where possible (e.g., helpers in `bet_fusion.py`).
- Prefer explicit paths and config over globals; reuse `config.normalize_team_name`.

## Testing Guidelines
- Add unit tests under `tests/` for: Poisson helpers, market assembly, name normalization, snapshot builder.
- Keep tests deterministic (seed models/matrices). Target >80% coverage on core helpers.
- Quick local check: run `scripts.multi_league_report` on a tiny fixtures CSV.

## Commit & PR Guidelines
- Commits: imperative mood, scoped prefix when useful (e.g., `feat(fusion): add OU line parsing`).
- PRs: clear goal, steps taken, inputs/outputs touched; include sample command and before/after snippet or CSV row.
- Link issues where applicable; attach screenshots for table outputs.

## Security & Config Tips
- API keys: set `API_FOOTBALL_KEY` (and/or `API_FOOTBALL_ODDS_KEY`) in environment; never commit secrets.
- Odds source: default is local (`data/odds/{LEAGUE}.json`). Use `scripts.fetch_odds_api_football` to refresh.
- Fixtures: place weekly CSVs in `data/fixtures/{LEAGUE}_weekly_fixtures.csv` or use generated `*_manual.csv`.
