"""
CGM-only configuration.

The legacy (non-CGM) config has been archived to:
  archive/legacy_non_cgm_engine/config_legacy.py
"""

# ---------------------------------------------------------------------------
# API-Football Defaults
# ---------------------------------------------------------------------------
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_DEFAULT_DATA_DIR = "data/api_football"
API_FOOTBALL_RATE_PER_MIN_FREE = 10
API_FOOTBALL_MAX_REQUESTS_FREE = 100
API_FOOTBALL_DEFAULT_HISTORY_DAYS = 365
API_FOOTBALL_DEFAULT_HORIZON_DAYS = 7
API_FOOTBALL_DEFAULT_LEAGUE_IDS = [39]  # EPL on free tier; extend gradually as cache grows
API_FOOTBALL_FETCH_ODDS_DEFAULT = True

# Gaussian kernel width (sigma) per league for Elo-similarity features.
# Used by: cgm/build_frankenstein.py and cgm/predict_upcoming.py
ELO_SIM_SIGMA_PER_LEAGUE = {
    "E0": 50.0,
    "D1": 40.0,
    "F1": 45.0,
    "SP1": 55.0,
    "I1": 45.0,
}

# Scope of leagues to process
LIVE_SCOPE = {
    "England": ["Premier L", "Championship"],
    "Spain": ["Primera"],
    "Germany": ["Bundesliga"],
    "Italy": ["Serie A", "Serie B"],
    "France": ["Ligue 1", "Ligue 2"],
    "Netherlands": ["Eredivisie"],
    "Portugal": ["Primeira L"],
    "Turkey": ["Super Lig"],
    "Romania": ["Liga 1"],
}

# ---------------------------------------------------------------------------
# Live scope (single source of truth)
#
# These defaults define the "live" prediction window for Milestone 4+.
# They can be overridden via CLI args in `cgm.predict_upcoming` / `cgm.pick_engine_goals`.
# ---------------------------------------------------------------------------

# CGM league naming is not guaranteed to match bookmaker codes; use the literal values
# found in your exports (example: "Premier L").
LIVE_SCOPE_COUNTRY = ""
LIVE_SCOPE_LEAGUE = ""

# Date-window is the most robust season selector (avoid season-label quirks).
LIVE_SCOPE_SEASON_START = "2025-08-01"
LIVE_SCOPE_SEASON_END = "2026-07-01"

# Optional horizon (days) for how far ahead to consider fixtures. Use None/0 to disable.
LIVE_SCOPE_HORIZON_DAYS = 7

# Keep only the immediate next round window (first upcoming fixture date + span days).
LIVE_SCOPE_NEXT_ROUND_ONLY = True
LIVE_SCOPE_NEXT_ROUND_SPAN_DAYS = 3

# Internal defaults: odds-aware flow with EV-based picks enabled.
PIPELINE_DEFAULT_MODEL_VARIANT = "full"
PIPELINE_EMIT_PICKS_DEFAULT = True

# ---------------------------------------------------------------------------
# Pick Engine Gates (single source of truth)
#
# These thresholds are used by the goals-only engine.
# Legacy full-engine values are retained for reference (archived).
# ---------------------------------------------------------------------------

# Odds sanity: reject odds below this value (too close to 1.0 = no edge)
ODDS_MIN_FULL = 1.01        # Legacy full engine (1X2 + O/U)
ODDS_MIN_GOALS = 1.05       # Used by pick_engine_goals.py (goals-only)

# mu_total (expected total goals) must be in this range for a pick
MU_TOTAL_MIN = 1.6
MU_TOTAL_MAX = 3.4

# Evidence requirements (how much history we need before trusting a pick)
NEFF_MIN_FULL = 6.0         # Effective sample size for full engine
NEFF_MIN_GOALS = 8.0        # Stricter for goals-only engine

PRESS_EVID_MIN_FULL = 2.0   # Pressure evidence minimum for full engine
PRESS_EVID_MIN_GOALS = 3.0  # Stricter for goals-only engine

XG_EVID_MIN_FULL = 2.0      # xG evidence minimum for full engine
XG_EVID_MIN_GOALS = 3.0     # Stricter for goals-only engine

# EV (Expected Value) thresholds
EV_MIN_1X2 = 0.05           # Legacy full engine: minimum EV for 1X2 picks
EV_MIN_OU25 = 0.04          # Minimum EV for Over/Under 2.5
EV_MIN_BTTS = 0.04          # Minimum EV for BTTS
EV_MIN_TIMING = 0.05        # Minimum EV for timing markets
EV_MIN_STERILE_1X2 = 0.07   # Legacy full engine: higher EV for 1X2 with sterile flag
EV_MIN_STERILE_OVER = 0.08  # Higher EV required for Over with sterile flag
EV_MIN_ASSASSIN_ANY = 0.07  # Higher EV required when assassin flag is set
EV_MIN_ASSASSIN_UNDER = 0.08
EV_MIN_LATE_HEAVY_UNDER = 0.08

# Assassin-specific stricter requirements for O/U markets
ASSASSIN_NEFF_MIN_OU25 = 7.0
ASSASSIN_PRESS_EVID_MIN_OU25 = 3.0
ASSASSIN_XG_EVID_MIN_OU25 = 3.0

# ---------------------------------------------------------------------------
# Elo Similarity Settings
# ---------------------------------------------------------------------------
ELO_SIM_MIN_EFF = 5.0           # Minimum effective sample size for blending

# ---------------------------------------------------------------------------
# Pressure Form Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
PRESSURE_W_SHOTS = 0.45
PRESSURE_W_SOT = 0.30
PRESSURE_W_CORNERS = 0.15
PRESSURE_W_POSSESSION = 0.10

# ---------------------------------------------------------------------------
# Goal Timing Settings
# ---------------------------------------------------------------------------
TIMING_MIN_PROFILE_MATCHES = 10   # Minimum matches for usable timing profile
TIMING_MIN_PROFILE_GOALS = 8      # Minimum goals for usable timing profile
TIMING_EARLY_SHARE_MARGIN = 0.05  # Margin for slow start flag
TIMING_LATE_SHARE_MARGIN = 0.07   # Margin for late goal flag

# ---------------------------------------------------------------------------
# Milestone 9: Time Decay Weighting
# ---------------------------------------------------------------------------
# Recent matches count more than older ones in rolling form calculations.
# Half-life = 5 means: match 5 ago has 50% weight, match 10 ago has 25% weight.
DECAY_HALF_LIFE = 5               # Decay half-life in number of matches
DECAY_ENABLED = True              # Toggle time decay on/off

# ---------------------------------------------------------------------------
# Milestone 10: Head-to-Head History
# ---------------------------------------------------------------------------
H2H_MIN_MATCHES = 3               # Minimum H2H meetings to mark as "usable"
H2H_MAX_LOOKBACK_YEARS = 5        # Don't look back more than 5 years for H2H
H2H_ENABLED = True                # Toggle H2H features on/off

# ---------------------------------------------------------------------------
# Milestone 11: League-Specific Features
# ---------------------------------------------------------------------------
# Add league profile features that capture scoring patterns per competition.
# Works with EPL-only data now; activates fully with multi-league data.
LEAGUE_FEATURES_ENABLED = True    # Toggle league features on/off
LEAGUE_MIN_MATCHES = 50           # Minimum matches to compute league profile
LEAGUE_PROFILE_WINDOW = 100       # Rolling window for league stats (matches)

# ---------------------------------------------------------------------------
# Milestone 13: League-Specific Probability Calibration
# ---------------------------------------------------------------------------
# Automatically adjust decision thresholds based on backtest results per league.
# Run: python -m scripts.calibrate_league --input reports/backtest_*.csv
CALIBRATION_ENABLED = False
CALIBRATION_FILE = "data/league_calibration.json"
CALIBRATION_MIN_SAMPLES = 50      # Don't trust calibration with fewer samples

# ---------------------------------------------------------------------------
# League-Specific Decision Thresholds (light-touch post-probability tuning)
# ---------------------------------------------------------------------------
# Default classification thresholds
OU25_OVER_THRESHOLD_DEFAULT = 0.50
BTTS_YES_THRESHOLD_DEFAULT = 0.50

# Per-league overrides (canonical league names from data/API)
# Purpose: fix persistent directional bias without changing model training.
OU25_OVER_THRESHOLD_BY_LEAGUE = {
    "Bundesliga": 0.42,  # updated after 3-league threshold rescan (less over-bias)
    "Premier League": 0.39,  # updated after 3-league threshold rescan (less under-bias)
    "Serie A": 0.50,     # updated after 3-league threshold rescan (more balanced O/U split)
    "Ligue 1": 0.52,     # batch-1 tuning window 2025-11-01..2026-02-22 (117 matches)
    "La Liga": 0.46,     # batch-1 tuning window 2025-11-01..2026-02-23 (148 matches)
    "Liga I": 0.26,      # temporary recent-form pilot (2 rounds): adapt to Feb scoring shift
    "Championship": 0.32,  # batch-2 tuning window 2025-11-01..2026-02-22 (250 matches)
    "Eredivisie": 0.23,    # batch-2 tuning window 2025-11-01..2026-02-22 (125 matches)
    "Ligue 2": 0.32,       # batch-3 tuning window 2025-11-01..2026-02-23 (103 matches)
    "Primeira Liga": 0.36, # batch-3 tuning window 2025-11-01..2026-02-23 (125 matches)
    "2. Bundesliga": 0.33, # batch-4 balanced tuning window 2025-11-01..2026-02-22 (115 matches)
    "Süper Lig": 0.41,     # batch-4 balanced tuning window 2025-11-01..2026-02-23 (116 matches)
}
BTTS_YES_THRESHOLD_BY_LEAGUE = {
    "Bundesliga": 0.28,  # tuned on 2025-11-01..2026-02-22 backtest window (133 matches)
    "Premier League": 0.31,  # reduces BTTS-NO bias from 2025-11-01..2026-02-23 window (181 matches)
    "Ligue 1": 0.51,     # batch-1 tuning window 2025-11-01..2026-02-22 (117 matches)
    "Liga I": 0.33,      # temporary recent-form pilot (2 rounds): adapt to Feb scoring shift
    "Championship": 0.30,  # batch-2 tuning window 2025-11-01..2026-02-22 (250 matches)
    "Eredivisie": 0.25,    # batch-2 tuning window 2025-11-01..2026-02-22 (125 matches)
    "Serie B": 0.42,       # batch-3 tuning window 2025-11-01..2026-02-22 (161 matches)
    "Ligue 2": 0.38,       # batch-3 tuning window 2025-11-01..2026-02-23 (103 matches)
    "2. Bundesliga": 0.29, # batch-4 balanced tuning window 2025-11-01..2026-02-22 (115 matches)
}

# ---------------------------------------------------------------------------
# League-Specific Mu Rebalance (pre-Poisson)
# ---------------------------------------------------------------------------
# Use only where backtests show systematic goal underestimation.
# Multiplies both mu_home and mu_away before probabilities/EV.
MU_GOAL_MULTIPLIER_DEFAULT = 1.00
MU_GOAL_MULTIPLIER_BY_LEAGUE = {
    # Brazil-only rebalance from isolated LATAM backtest window:
    # reports_latam/backtest_brazil_tuning_window.csv (2025-11-01..2026-02-19, 102 matches)
    # Baseline mu_total mean ~1.49 vs actual goals mean ~2.90.
    # 1.70 selected as conservative correction (keeps some under edge, restores over paths).
    "Serie A Brazil": 1.70,
}

# ---------------------------------------------------------------------------
# Low-Scoring Scenario Detector (Under 2.5 Enhancement)
# ---------------------------------------------------------------------------
# Identifies fixtures where both teams have sterile attacking profiles,
# and gives Under bets a scoring bonus to compete with Over in pick selection.
LOW_SCORING_ENABLED = True            # Toggle low-scoring detector on/off
LOW_SCORING_MU_THRESHOLD = 2.3        # mu_total below this triggers detection
LOW_SCORING_XG_FORM_THRESHOLD = 1.0   # xG form below this indicates sterile attack
LOW_SCORING_UNDER_SCORE_BONUS = 0.04  # Score bonus for Under in low-scoring scenarios
LOW_SCORING_EV_THRESHOLD_REDUCTION = 0.02  # Reduce EV threshold for Under by this amount
