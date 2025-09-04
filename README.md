
**Ligue 1**:
- ⚽ **Monaco** 🆚 **Reims**: Over 2.5 Goals! 🔥 (73.58% chance)
- ⚽ **St Etienne** 🆚 **Nice**: Under 2.5 Goals (93.4% chance)
- ⚽ **Lens** 🆚 **Le Havre**: Over 2.5 Goals! 🔥 (65.85% chance)
- ⚽ **Paris SG** 🆚 **Lille**: Over 2.5 Goals! 🔥 (66.92% chance)
- ⚽ **Lyon** 🆚 **Brest**: Over 2.5 Goals! 🔥 (92.07% chance)
- ⚽ **Montpellier** 🆚 **Rennes**: Over 2.5 Goals! 🔥 (76.95% chance)
- ⚽ **Auxerre** 🆚 **Strasbourg**: Under 2.5 Goals (70.88% chance)
- ⚽ **Angers** 🆚 **Toulouse**: Over 2.5 Goals! 🔥 (78.66% chance)
- ⚽ **Marseille** 🆚 **Nantes**: Under 2.5 Goals (56.8% chance)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Data Acquisition](#data-acquisition)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [Upcoming Matches Acquisition](#upcoming-matches-acquisition)
    - [Set up the API_KEY](#set-up-the-api_key)
8. [Making Predictions](#making-predictions)
9. [Betting Bot](#betting-bot)
10. [Supported Leagues](#supported-leagues)
11. [Contributing](#contributing)
12. [License](#license)
13. [Disclaimer](#disclaimer)

## Project Overview

AIFootballPredictions aims to create a predictive model to forecast whether a football match will exceed 2.5 goals. The project is divided into four main stages:

1. **Data Acquisition**: Download and merge historical football match data from multiple European leagues.
2. **Data Preprocessing**: Process the raw data to engineer features, handle missing values, and select the most relevant features.
3. **Model Training**: Train several machine learning models, perform hyperparameter tuning, and combine the best models into a voting classifier to make predictions.
4. **Making Predictions**: Use the trained models to predict outcomes for upcoming matches and generate a formatted message for sharing.

## Directory Structure

The project is organized into the following directories:

```
└─── `AIFootballPredictions`
    ├─── `conda`: all the conda environemnts
    ├─── `data`: the folder for the data
    │       ├─── `processed`
    │       └─── `raw`
    ├─── `models`: the folder with the saved and trained models
    ├─── `notebooks`: all the notebooks if any
    └─── `scripts`: all the python scripts
            ├─── `data_acquisition.py`
            ├─── `data_preprocessing.py`
@@ -191,50 +192,78 @@ In order to properly execute the `acquire_next_matches.py` script it is first ne
   - Write down the following line, replacing `your_personal_key` with your actual API key:
     - `API_FOOTBALL_DATA=your_personal_key`

4. **Save and Exit:**
   - Press the `Esc` key to exit insert mode.
   - Then, type `:wq!` and press `Enter` to save the changes and exit the editor.

5. **Verify the Variable:**
   - To check if the variable has been properly set, run the following command from the terminal:
     - `cat ~/.env`
   - You should see the `API_FOOTBALL_DATA` variable listed with your API key.

## Directory Structure (Clean)

The project is organized into the following directories:

```
AIFootballPredictions/
  conda/                 Conda environments
  data/
    raw/                 Raw league CSVs
    processed/           Preprocessed training data
    enhanced/            Enhanced features (e.g., *_final_features.csv)
    fixtures/            Fixture CSVs (weekly or manual)
    picks/               Saved picks from one-click predictor
    store/               Team stats snapshots (*.parquet)
  advanced_models/       Trained model files (*.pkl)
  models/                Voting/legacy models (*.pkl)
  notebooks/             Jupyter notebooks
  scripts/               All Python scripts
```

## Upcoming Matches Acquisition

### Set up the API_KEY

Set your API-Football key as an environment variable so tools can pull fixtures:

- Preferred: set `API_FOOTBALL_KEY` (the one-click predictor also checks `API_FOOTBALL_ODDS_KEY`).
- Optionally add a `.env` file in the repo root and include a line like:
  - `API_FOOTBALL_KEY=your_personal_key_here`

On Windows (PowerShell):
- Temporary for current shell: `$env:API_FOOTBALL_KEY = "your_personal_key_here"`
- To persist, add it to your user env vars via System Properties.

Once set, you can run tools that fetch fixtures, or fill the manual CSV fallback created under `data/fixtures/{LEAGUE}_manual.csv`.

## Making Predictions

To predict the outcomes for upcoming matches and generate a formatted message for sharing, run the `make_predictions.py` script:

```bash
python scripts/make_predictions.py \
  --input_leagues_models_dir models \
  --input_data_predict_dir data/processed \
  --final_predictions_out_file data/final_predictions.txt \
  --next_matches data/next_matches.json
```
This script will:

- Load the pre-trained models and the processed data.
- Make predictions for upcoming matches based on the next matches data.
- Format the predictions into a readable `.txt` message and save it to the specified output file.

## Betting Bot

The optional betting bot can place wagers based on the model's predictions.

### Prerequisites

1. A working Python 3 environment with this project's dependencies installed.
2. Trained models and recent match data.
3. Access to a betting platform that allows automated wagers.

### Legal Considerations

Always verify that automated betting is legal in your jurisdiction and permitted by your betting platform. The authors do not encourage gambling where it is prohibited.

### Setup Steps

1. Review the predictions generated in the previous step.
2. Configure any required credentials or API keys for your betting platform.
3. Run the bot for a specific league:

```bash
python scripts/betting_bot.py --league E0
```

### Fixtures Input

- If `data/fixtures/{LEAGUE}_weekly_fixtures.csv` is not present, the tools will create a manual template at `data/fixtures/{LEAGUE}_manual.csv` with columns `date,home,away`. Fill upcoming fixtures (use model-known names like "Man City") and re-run.
- Team statistics are summarized automatically from `data/enhanced/{LEAGUE}_final_features.csv` into a per-team snapshot.

### Risk Warning

Betting involves financial risk. Use the bot responsibly and comply with all local regulations.

## Supported Leagues

For the moment, the team name mapping has been done manually. The predictions currently support the following leagues:

- *Premier League*: **E0**
- *Serie A*: **I1**
- *Ligue 1*: **F1**
- *La Liga (Primera Division)*: **SP1**
- *Bundesliga*: **D1**

For this reason be carful when executing the [data acquisition](#data-acquisition) step. 

## Contributing

If you want to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [BSD-3-Clause License](LICENSE) - see the `LICENSE` file for details.

## Disclaimer

This project is intended for educational and informational purposes only. While the AIFootballPredictions system aims to provide accurate predictions for football matches, it is important to understand that predictions are inherently uncertain and should not be used as the sole basis for any decision-making, including betting or financial investments.

The predictions generated by this system can be used as an additional tool during the decision-making process. However, they should be considered alongside other factors and sources of information.
<div align="center">

# AI Football Predictions

Hybrid Poisson + XGB engine for 1X2, OU 2.5 and Total‑Goals intervals, with calibration, odds integration and compact round reports.

</div>

## Table of Contents

1. Project Overview
2. Directory Structure
3. Setup & Installation
4. Weekly Pipeline (one command)
5. Core Commands
6. Fixtures & Odds
7. Reporting (tables + legend)
8. Betting Bot
9. Supported Leagues
10. Contributing & Guidelines
11. License & Disclaimer

## Project Overview

This project builds per‑league expected goals models (XGBoost regressors for home/away goals), converts them to a Poisson score matrix, and derives market probabilities for:

- 1X2 (home/draw/away)
- Over/Under 2.5
- Total‑Goals intervals (e.g., 0‑3, 2‑5)

Per‑league probability calibrators (isotonic/Platt) correct biases. A compact report prints fancy‑bordered tables, and an optional betting bot ranks value picks and logs bankroll.

## Directory Structure

```
AIFootballPredictions/
  advanced_models/            # XGB models per league (*.pkl)
  calibrators/                # Per-league calibrators for 1X2 and OU 2.5 (*.pkl)
  conda/                      # Conda environment file(s)
  data/
    raw/                      # Merged raw CSVs per league
    processed/                # Preprocessed CSVs per league
    enhanced/                 # Enhanced features (optional)
    fixtures/                 # Weekly or manual fixtures CSVs
    odds/                     # Local odds JSON (used in local mode)
    store/                    # Team stats snapshots (parquet)
  reports/                    # Round reports / picks CSVs
  scripts/                    # Pipeline + utilities (see Core Commands)
  bot_config.yaml             # Runtime config for markets/thresholds
  config.py                   # League IDs, feature lists, name mapping
```

## Setup & Installation

- Python 3.10 recommended (see `conda/aifootball_predictions.yaml`).
- Quick install:
  - `pip install -r requirements.txt`
  - or: `conda env create -f conda/aifootball_predictions.yaml`
- Environment (PowerShell examples):
  - `$env:API_FOOTBALL_KEY = 'your_api_key'` (or `API_FOOTBALL_ODDS_KEY`)
  - Optional: `$env:BOT_ODDS_MODE = 'local'` and keep JSONs under `data/odds/`

## Weekly Pipeline (one command)

Update data, preprocess, (optionally train), calibrate, fetch odds, and export compact round reports:

```
python -m scripts.update_and_report \
  --leagues E0 D1 F1 I1 SP1 \
  --season-codes 2425 \
  --fixtures-csv E0=data/fixtures/E0_weekly_fixtures.csv D1=data/fixtures/D1_manual.csv F1=data/fixtures/F1_manual.csv I1=data/fixtures/I1_manual.csv SP1=data/fixtures/SP1_manual.csv \
  --select prob --export --train --fetch-odds
```

## Core Commands

- Acquire raw: `python -m scripts.data_acquisition --leagues E0 I1 SP1 F1 D1 --seasons 2425 2324`
- Preprocess: `python -m scripts.data_preprocessing --raw_data_input_dir data/raw --processed_data_output_dir data/processed`
- Train XGB (per league): `python xgb_trainer.py --league E0`
- Calibrate: `python -m scripts.calibrate_league --league E0 D1 F1 I1 SP1`
- Fetch odds (API‑Football): `python -m scripts.fetch_odds_api_football --league E0 --fixtures-csv data/fixtures/E0_weekly_fixtures.csv`
- Report (print + export): `python -m scripts.multi_league_report --leagues E0 D1 F1 I1 SP1 --select prob --export`
- Betting bot (print/log picks): `python scripts/betting_bot.py --league E0`

## Fixtures & Odds

- Fixtures: prefer `data/fixtures/{LEAGUE}_weekly_fixtures.csv`. If missing, a manual template `.../{LEAGUE}_manual.csv` is created (`date,home,away`). Team names are normalized via `config.TEAM_NAME_MAP`.
- Odds: by default we use local JSON under `data/odds/{LEAGUE}.json` (see `scripts/bookmaker_api.py`). Use `--fetch-odds` to query API‑Football and refresh local odds.

## Reporting (tables + legend)

Printed reports use the fancy‑border grid and compact market cells:

- Cell format: `Pick | Prob @ Odds | EV`
- Legend (printed under each table):
  - `EV`: Expected value of the bet at shown odds
  - `Higher`: Higher probability market between 1X2 and OU 2.5
  - `TG`: Total‑Goals interval (e.g., 0‑3 means total goals 0 to 3)

## Betting Bot

The optional bot ranks value picks and logs bankroll to `data/bankroll.json` and bets to `data/bets_log.csv`. Configure thresholds, staking (`flat`/`kelly`) and logging in `bot_config.yaml`.

Legal note: verify automation is allowed in your jurisdiction/platform.

## Supported Leagues

- Premier League: `E0`
- Serie A: `I1`
- Ligue 1: `F1`
- La Liga: `SP1`
- Bundesliga: `D1`

## Contributing & Guidelines

See `AGENTS.md` for contributor guidelines (structure, commands, style, tests, PR checklist). PRs should include example commands and before/after snippets or CSV rows.

## License

BSD‑3‑Clause License (see `LICENSE`).

## Disclaimer

This project is for educational/informational purposes. Predictions are uncertain; do not use as sole basis for betting or financial decisions. Use responsibly and comply with local regulations.
