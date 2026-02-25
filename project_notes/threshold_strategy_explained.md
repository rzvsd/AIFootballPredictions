Threshold Strategy (Easy Read)

Purpose
- The bot predicts two markets for every match:
  - Over/Under 2.5 goals (OU)
  - BTTS (Both Teams To Score)
- Same core model is used for all leagues.
- Only decision cutoffs (thresholds) can differ by league.

What a threshold means
- The model outputs probabilities:
  - `P(OVER 2.5)`
  - `P(BTTS YES)`
- We convert probability to final pick with a rule:
  - If probability >= threshold -> pick YES side (`OVER` or `BTTS_YES`)
  - Else -> pick NO side (`UNDER` or `BTTS_NO`)

Default vs Tuned
- Default threshold = `0.50`:
  - `P(OVER) >= 0.50` -> OVER, else UNDER
  - `P(BTTS_YES) >= 0.50` -> BTTS_YES, else BTTS_NO
- Tuned threshold:
  - League-specific value (example `0.31`, `0.54`) learned from backtest evidence.

Why some leagues are OU-only tuned or BTTS-only tuned
- OU and BTTS are tuned independently.
- We only change a threshold when evidence is stable for that market.
- If OU improves clearly but BTTS does not, we tune OU only and keep BTTS default.
- If BTTS improves clearly but OU does not, we tune BTTS only and keep OU default.
- This is intentional risk control, not a missing configuration.

Important clarification
- If a league is “BTTS-only tuned”, the bot still predicts OU as usual.
- If a league is “OU-only tuned”, the bot still predicts BTTS as usual.
- “Only tuned” means “only that threshold changed”, not “only that market is predicted”.

Lower vs Higher threshold (plain language)
- Lower threshold (ex: `0.31`) = easier to trigger YES side:
  - More `OVER` picks
  - More `BTTS_YES` picks
- Higher threshold (ex: `0.54`) = harder to trigger YES side:
  - More `UNDER` picks
  - More `BTTS_NO` picks

Example
- Suppose `P(BTTS_YES) = 0.42`
  - threshold `0.50` -> BTTS_NO
  - threshold `0.31` -> BTTS_YES

How we decide to tune
1. Rebuild history and league averages first (fresh data).
2. Run a tuning-window backtest for that league.
3. Scan threshold candidates.
4. Apply conservative values only (avoid extreme/overfit cutoffs).
5. Validate on last round after change.

Why not tune everything at once
- Different leagues behave differently.
- Some league/market pairs show unstable threshold signals.
- Batch tuning avoids introducing bad cutoffs globally.

Current policy
- Keep defaults where evidence is weak or unstable.
- Tune only where evidence is consistent.
- Recheck each tuned league regularly and adjust if drift appears.

Before Each Round (Operational Routine)

Do this every round
1. Generate fresh predictions and picks:
   - `python predict.py --predict-only --max-date YYYY-MM-DD`
2. Review output files:
   - `reports/cgm_upcoming_predictions.csv`
   - `reports/picks.csv`

Do this weekly (or after obvious data gaps)
1. Rebuild history and league averages:
   - `$env:API_FOOTBALL_HISTORY_DAYS="365"; $env:API_FOOTBALL_LEAGUE_IDS="..."; $env:API_FOOTBALL_MAX_REQUESTS="7500"; $env:API_FOOTBALL_RATE_PER_MINUTE="120"; python predict.py --rebuild-history --skip-train --max-date YYYY-MM-DD`

Do NOT retune thresholds every round by default
- Retune only if one of these is true:
  - 2 consecutive weak rounds for the same league/market.
  - Clear direction bias (example: almost all picks are UNDER/BTTS_NO) for 2 rounds.
  - Scheduled review window (every 2-4 weeks).

Retune workflow (only when triggered)
1. Build backtest window for target league.
2. Run threshold scan with `scripts/scan_thresholds.py`.
3. Apply conservative threshold change in `config.py`.
4. Validate on last round before keeping it.
