# Football Predictions Bot - Quick Start

This bot predicts only:
- Over/Under 2.5 goals
- BTTS (both teams to score)

It uses API-Football as the default data source, then runs the internal CGM model pipeline.

## 1) Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your API key:

```bash
# PowerShell
$env:API_FOOTBALL_KEY="your_key_here"
```

## 2) Run

Full pipeline (recommended):

```bash
python predict.py --max-date YYYY-MM-DD
```

Predictions only, without pick/narrator files:

```bash
python predict.py --max-date YYYY-MM-DD --no-emit-picks
```

Predict-only (reuse existing trained models/history):

```bash
python predict.py --predict-only
```

Legacy CSV mode (if you do not want API sync):

```bash
python predict.py --data-source csv --max-date YYYY-MM-DD
```

## 3) Output files

Main outputs are in `reports/`:
- `cgm_upcoming_predictions.csv`
- `picks.csv`
- `picks_debug.csv`
- `picks_explained.csv`
- `picks_status.json`

Backtest the current round-based pick strategy:

```bash
python scripts/evaluate_pick_strategy_rounds.py --out-dir reports/round_strategy_eval
```

Business-friendly summary (last rounds + upcoming + backtest):

```bash
python scripts/generate_business_report.py --rounds 5 --upcoming-limit 20
```

Creates:
- `reports/business_report.txt`
- `reports/business_report_recent_results.csv`
- `reports/business_report_upcoming_summary.csv`

## 4) Important notes

- Free tier is limited to about 100 requests/day. Keep league scope small (default is league `39`).
- If quality coverage is too low (odds/stats), the pre-bet gate can stop live pick generation by design. Inside the pick engine, EV/missing odds are diagnostic in model-signal mode; real odds below the configured floor are still rejected.
- If there are no upcoming fixtures in range, zero picks is a valid result.
