Full-Season Threshold Tuning (OU + BTTS)

Purpose
- Tune league-specific decision thresholds for both markets:
  - Over/Under 2.5 (OU)
  - BTTS (Both Teams To Score)
- Use no-leak backtests (model only sees data available before each fixture date).

How It Is Generated
1. Build no-leak backtests from season start for each league.
   - Engine call per fixture date:
     - `python -m scripts.run_backtest --league "<LEAGUE>" --season 2025-2026 --start-date <SEASON_START> --history <HISTORY> --models-dir <MODELS_DIR> --model-variant full --out <BACKTEST_FILE>`
2. Scan threshold candidates per market.
   - Objective: maximize balanced accuracy (not raw YES count).
   - Candidate grid: `0.10 -> 0.90`, step `0.01`.
3. Apply conservative guardrails before writing config:
   - prefer thresholds with reasonable YES rate (avoid always-YES / always-NO behavior),
   - require stable sample size (small samples stay default/provisional).
4. Write thresholds in `config.py`:
   - `OU25_OVER_THRESHOLD_BY_LEAGUE`
   - `BTTS_YES_THRESHOLD_BY_LEAGUE`

Evidence Files (currently generated)
- Full-season no-leak:
  - `reports/backtest_premier_fullseason.csv`
  - `reports/backtest_championship_fullseason.csv`
  - `reports/backtest_bundesliga_fullseason.csv`
  - `reports/backtest_2bundesliga_fullseason.csv`
  - `reports/backtest_serie_a_fullseason.csv`
  - `reports/backtest_serieb_fullseason.csv`
  - `reports/backtest_laliga_fullseason.csv`
  - `reports/backtest_ligue1_fullseason.csv`
  - `reports/backtest_ligue2_fullseason.csv`
  - `reports/backtest_eredivisie_fullseason.csv`
  - `reports_latam/backtest_brazil_fullseason.csv`
- Summary from these files:
  - `reports/tuning_fullseason_available_summary.csv`
  - `reports/tuning_fullseason_available_guarded_summary.csv`

Current Applied Thresholds (config)
- Premier League: OU `0.49`, BTTS `0.53`
- Championship: OU `0.47`, BTTS `0.52`
- Bundesliga: OU `0.61`, BTTS `0.61`
- 2. Bundesliga: OU `0.53`, BTTS `0.55`
- Serie A: OU `0.44`, BTTS `0.48`
- Serie B: OU `0.46`, BTTS `0.50`
- La Liga: OU `0.49`, BTTS `0.52`
- Ligue 1: OU `0.53`, BTTS `0.54`
- Ligue 2: OU `0.45`, BTTS `0.51`
- Eredivisie: OU `0.61`, BTTS `0.63`
- Serie A Brazil: OU `0.79`, BTTS `0.50`

Provisional Thresholds (pending full-season no-leak completion for these leagues)
- Primeira Liga: OU `0.53`, BTTS `0.48` (tuning-window evidence)
- Liga I: OU `0.41`, BTTS `0.40` (tuning-window evidence)
- Süper Lig: OU `0.24`, BTTS `0.50` (tuning-window evidence)
- Liga Profesional Argentina: OU `0.50`, BTTS `0.50` (insufficient stable sample in current file)

Important
- This process tunes decision cutoffs only. Core model math remains unchanged.
- If a league lacks enough stable evidence, keep conservative/default values until more backtest rows are available.

Operational Recommendation
- Re-run full-season threshold tuning weekly after results ingestion.
- Re-run immediately after major data-source changes (API mapping, normalization, odds feed changes).
