Audits and Validation (Current)

Core checks
- Detailed audit log:
  - `python scripts/audit_detailed.py`
  - Produces a non-technical log in `reports/audits/` with Elo/xG/Pressure coverage.
  - Also includes team-centric anchor and OU-path fields when present (`team_anchor_*`, `over_score`, `under_score`).
  - Also saves JSON summary and per-league CSV coverage.

- Upcoming feed scope:
  - `python -m scripts.audit_upcoming_feed --as-of-date YYYY-MM-DD`
  - Validates:
    - Upcoming fixtures are future-only.
    - Scope matches the configured next-round window.

- No-odds invariance:
  - `python -m scripts.audit_no_odds --as-of-date YYYY-MM-DD`
  - Verifies probabilities are stable when odds are not used.
  - Validates:
    - Model output does not change when odds inputs change.

- Multi-league coverage snapshot:
  - `python scripts/audit_multi_league.py`
  - Validates:
    - History and upcoming data include the expected leagues.

Scope note
- These audits cover the core live pipeline (API sync, upcoming scope, no-odds output stability).
- Additional niche audits were archived during cleanup and are not part of the current default runbook.

Data freshness and anchor integrity
- Recommended rebuild command when league averages look compressed or rounds are missing:
  - `$env:API_FOOTBALL_HISTORY_DAYS="365"; $env:API_FOOTBALL_LEAGUE_IDS="39,40,140,78,135,136,61,62,88,94,203,283"; $env:API_FOOTBALL_MAX_REQUESTS="7500"; $env:API_FOOTBALL_RATE_PER_MINUTE="120"; python predict.py --rebuild-history --skip-train --max-date YYYY-MM-DD`
- Why this matters:
  - League anchors (`lg_avg_gf_home`, `lg_avg_gf_away`) are rebuilt from history and directly affect OU/BTTS probabilities.

Backtest checks
- Generate backtest:
  - `python -m scripts.run_backtest --league "Premier League" --season "2025-2026" --start-date "2025-09-01"`
  - Validates:
    - End-to-end training + prediction on historical fixtures for a league/season.
- Optional leakage sanity:
  - `python scripts/audit_backtest.py`
  - Validates:
    - Obvious leakage in backtest pipeline (sanity checks).

Bundesliga tuning audit recipe
- Build tuning window:
  - `python -m scripts.run_backtest --league "Bundesliga" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_bundesliga_tuning_window.csv"`
- Purpose:
  - Re-estimate league-specific OU/BTTS decision thresholds from a larger evidence window.
  - Confirm last-round coverage is full (9/9 fixtures for Bundesliga Round 22 in current data snapshot).

EPL and Serie A threshold-scan recipe
- Build EPL window:
  - `python -m scripts.run_backtest --league "Premier League" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_premier_tuning_window.csv"`
- Build Serie A window:
  - `python -m scripts.run_backtest --league "Serie A" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_serie_a_tuning_window.csv"`
- Analyze thresholds:
  - run local threshold scan and export:
    - `reports/tuning_threshold_scan_summary.csv`
    - `reports/tuning_threshold_scan_balanced.csv`
- Purpose:
  - Quantify per-league threshold sensitivity before changing config defaults.

Batch Tuning Runbook (repeatable)
- 1) Rebuild history + averages before tuning batch:
  - `$env:API_FOOTBALL_HISTORY_DAYS="365"; $env:API_FOOTBALL_LEAGUE_IDS="39,78,135,61,140,283"; $env:API_FOOTBALL_MAX_REQUESTS="7500"; $env:API_FOOTBALL_RATE_PER_MINUTE="120"; python predict.py --rebuild-history --skip-train --max-date YYYY-MM-DD`
- 2) Build tuning windows (example batch-1):
  - `python -m scripts.run_backtest --league "Ligue 1" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_ligue1_tuning_window.csv"`
  - `python -m scripts.run_backtest --league "La Liga" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_laliga_tuning_window.csv"`
  - `python -m scripts.run_backtest --league "Liga I" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_ligai_tuning_window.csv"`
- 3) Scan thresholds:
  - `python scripts/scan_thresholds.py --input "Ligue 1=reports/backtest_ligue1_tuning_window.csv" --input "La Liga=reports/backtest_laliga_tuning_window.csv" --input "Liga I=reports/backtest_ligai_tuning_window.csv" --out reports/tuning_batch1_summary.csv`
- 4) Apply conservative config overrides and validate last round:
  - `python -m scripts.run_backtest --league "Ligue 1" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_ligue_1.csv"`
  - `python -m scripts.run_backtest --league "La Liga" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_la_liga.csv"`
  - `python -m scripts.run_backtest --league "Liga I" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_liga_i.csv"`

Batch-2 commands (Championship + Eredivisie)
- Build windows:
  - `python -m scripts.run_backtest --league "Championship" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_championship_tuning_window.csv"`
  - `python -m scripts.run_backtest --league "Eredivisie" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_eredivisie_tuning_window.csv"`
- Scan:
  - `python scripts/scan_thresholds.py --input "Championship=reports/backtest_championship_tuning_window.csv" --input "Eredivisie=reports/backtest_eredivisie_tuning_window.csv" --out reports/tuning_batch_next_champ_eredivisie_summary.csv`
- Validate last round:
  - `python -m scripts.run_backtest --league "Championship" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_championship.csv"`
  - `python -m scripts.run_backtest --league "Eredivisie" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_eredivisie.csv"`

2. Bundesliga coverage check
- Probe command:
  - `python scripts/sync_api_football.py --data-dir data/api_football_probe79 --league-ids 79 --history-days 365 --horizon-days 7 --max-requests 7500 --rate-per-minute 120 --fetch-odds`
- Current result:
  - league `79` sync is now healthy in this environment (history + upcoming rows present).

Batch-3 commands (Serie B + Ligue 2 + Primeira Liga)
- Build windows:
  - `python -m scripts.run_backtest --league "Serie B" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_serieb_tuning_window.csv"`
  - `python -m scripts.run_backtest --league "Ligue 2" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_ligue2_tuning_window.csv"`
  - `python -m scripts.run_backtest --league "Primeira Liga" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_primeira_tuning_window.csv"`
- Scan:
  - `python scripts/scan_thresholds.py --input "Serie B=reports/backtest_serieb_tuning_window.csv" --input "Ligue 2=reports/backtest_ligue2_tuning_window.csv" --input "Primeira Liga=reports/backtest_primeira_tuning_window.csv" --out reports/tuning_batch_serieb_ligue2_primeira_summary.csv`
- Validate last round:
  - `python -m scripts.run_backtest --league "Serie B" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_serie_b.csv"`
  - `python -m scripts.run_backtest --league "Ligue 2" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_ligue_2.csv"`
  - `python -m scripts.run_backtest --league "Primeira Liga" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_primeira_liga.csv"`

Batch-4 commands (2. Bundesliga + Süper Lig)
- Build windows:
  - `python -m scripts.run_backtest --league "2. Bundesliga" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_2bundesliga_tuning_window.csv"`
  - `python -m scripts.run_backtest --league "Süper Lig" --season "2025-2026" --start-date "2025-11-01" --out "reports/backtest_super_lig_tuning_window.csv"`
- Scan:
  - `python scripts/scan_thresholds.py --input "2. Bundesliga=reports/backtest_2bundesliga_tuning_window.csv" --input "Süper Lig=reports/backtest_super_lig_tuning_window.csv" --out reports/tuning_batch_2bund_superlig_summary.csv`
  - `python scripts/scan_thresholds.py --input "2. Bundesliga=reports/backtest_2bundesliga_tuning_window.csv" --input "Süper Lig=reports/backtest_super_lig_tuning_window.csv" --min-yes-rate 0.35 --max-yes-rate 0.65 --out reports/tuning_batch_2bund_superlig_balanced_summary.csv`
- Validate last round:
  - `python -m scripts.run_backtest --league "2. Bundesliga" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_2bundesliga_after.csv"`
  - `python -m scripts.run_backtest --league "Süper Lig" --season "2025-2026" --start-date "2026-02-20" --out "reports/backtest_lastround_super_lig_after.csv"`

Recommended cadence
- Full rebuild (365-day window): once per week, and always before starting a new tuning batch.
- Threshold tuning batch: every 2-4 weeks, or sooner if 2 consecutive rounds show clear directional drift (too many OVER or too many BTTS_NO).
- Last-round validation: after every threshold change.

Sterile Override v1 Validation Gate
- Current mode:
  - Active: sterile counts can adjust Under 2.5 probability with cap controls.
- Team-centric + OU-path companion checks:
  - Confirm `team_anchor_rel` and `team_anchor_blend` rise only with sufficient similar-ELO evidence.
  - Confirm OU picks pass directional path checks (`over_score`/`under_score` + margin) before EV ranking.
- Activation rules under test:
  - Count-based triggers: `4/10` and `6/10`.
  - Minimum valid sterile signals required: `8`.
  - Under boost mode, sterile uplift is capped at `+0.05`.
- Validation gate for keeping active mode:
  - Backtest must show no material regression versus baseline on target markets (OU2.5, BTTS).
  - Audit evidence must confirm cap behavior and correct trigger firing.
  - If gate fails, reduce thresholds/boost or disable by config.

Business-facing report
- `python scripts/generate_business_report.py --rounds 5 --upcoming-limit 20`
- This is the easiest final validation for non-technical review.
  - Validates:
    - Report renders without errors and includes recent results + next round predictions.

LATAM isolated validation (Argentina + Brazil)
- Use the dedicated runbook:
  - `project_notes/latam_argentina_brazil_runbook.md`
- Key isolation checks:
  - Train to `models_latam` (not `models`).
  - Predict to `reports_latam` (not `reports`).
  - Backtest with explicit model folder:
    - `python -m scripts.run_backtest --league "Liga Profesional Argentina" --season "2025-2026" --start-date "YYYY-MM-DD" --history data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv --models-dir models_latam --out reports_latam/backtest_argentina.csv`
    - `python -m scripts.run_backtest --league "Serie A Brazil" --season "2025-2026" --start-date "YYYY-MM-DD" --history data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv --models-dir models_latam --out reports_latam/backtest_brazil.csv`
