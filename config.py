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

# ---------------------------------------------------------------------------
# Elo V2 (league-aware, weighted, traceable)
# ---------------------------------------------------------------------------
ELO_V2_ENABLED = True
ELO_DEFAULTS = {
    "start_elo": 1500.0,
    "k_factor": 20.0,
    "home_adv": 65.0,
    "band_thresh": 150.0,
    "margin_cap": 2.75,
    "upset_expected_low": 0.35,
    "upset_expected_high": 0.65,
    "upset_multiplier": 1.20,
    "new_team_games": 12,
    "new_team_k_multiplier": 1.25,
}

# Per-league Elo controls. Values can be tuned later with walk-forward validation.
ELO_LEAGUE_PARAMS = {
    "Premier League": {"k_factor": 21.0, "home_adv": 58.0},
    "Championship": {"k_factor": 22.0, "home_adv": 60.0},
    "Bundesliga": {"k_factor": 21.0, "home_adv": 50.0},
    "2. Bundesliga": {"k_factor": 22.0, "home_adv": 54.0},
    "Serie A": {"k_factor": 20.0, "home_adv": 52.0},
    "Serie B": {"k_factor": 21.0, "home_adv": 56.0},
    "La Liga": {"k_factor": 20.0, "home_adv": 50.0},
    "Ligue 1": {"k_factor": 20.0, "home_adv": 52.0},
    "Ligue 2": {"k_factor": 21.0, "home_adv": 57.0},
    "Eredivisie": {"k_factor": 21.0, "home_adv": 48.0},
    "Primeira Liga": {"k_factor": 20.0, "home_adv": 52.0},
    "Liga I": {"k_factor": 21.0, "home_adv": 58.0},
    "S\u00fcper Lig": {"k_factor": 21.0, "home_adv": 56.0},
    "S?per Lig": {"k_factor": 21.0, "home_adv": 56.0},
    "Serie A Brazil": {"k_factor": 21.0, "home_adv": 52.0},
    "Liga Profesional Argentina": {"k_factor": 21.0, "home_adv": 56.0},
}

# Match-type weighting for Elo K. Match type is inferred from columns/name when absent.
ELO_MATCHTYPE_MULTIPLIERS = {
    "league": 1.00,
    "cup": 1.12,
    "playoff": 1.15,
    "friendly": 0.75,
    "unknown": 1.00,
}

# Gaussian kernel width (sigma) per league for Elo-similarity features.
# Used by: cgm/predict_upcoming.py
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
# Global minimum effective similar-match evidence required by predict_upcoming
# before a fixture is flagged with `elo_evidence_low`.
PIPELINE_MIN_MATCHES = 3

# ---------------------------------------------------------------------------
# Strict Mu Engine (explicit 4-module weighted engine)
# ---------------------------------------------------------------------------
# mu is built only from:
#   1) league anchor,
#   2) Elo module,
#   3) xG module,
#   4) pressure module.
STRICT_MODULE_MU_ENABLED = True
STRICT_MODULE_WEIGHTS = {
    "league_anchor": 0.40,
    "elo": 0.10,
    "xg": 0.25,
    "pressure": 0.25,
}
STRICT_MODULE_DEFAULT_ANCHOR_HOME = 1.35
STRICT_MODULE_DEFAULT_ANCHOR_AWAY = 1.15
STRICT_MODULE_GOALS_CLIP_MIN = 0.20
STRICT_MODULE_GOALS_CLIP_MAX = 3.50
STRICT_MODULE_PRESSURE_SHARE_CLIP_MIN = 0.25
STRICT_MODULE_PRESSURE_SHARE_CLIP_MAX = 0.75
STRICT_MODULE_PRESSURE_TOTAL_CLIP_MIN = 0.85
STRICT_MODULE_PRESSURE_TOTAL_CLIP_MAX = 1.15
STRICT_MODULE_PRESSURE_TOTAL_STRENGTH = 0.70

# ---------------------------------------------------------------------------
# Pick Engine Gates (single source of truth)
#
# These thresholds are used by the goals-only engine.
# Legacy full-engine values are retained for reference (archived).
# ---------------------------------------------------------------------------

# Odds sanity. Odds are diagnostic in the current model-signal strategy, so
# missing odds do not block a pick; when odds are present, very low prices are
# still treated as unusable for priced-bet reporting.
ODDS_MIN_FULL = 1.01        # Legacy full engine (1X2 + O/U)
ODDS_MIN_GOALS = 1.70       # Used by pick_engine_goals.py (goals-only)
PICK_ENGINE_ODDS_PHANTOM_MODE = True
PICK_ENGINE_EV_PHANTOM_MODE = True
PICK_ENGINE_ROUND_QUOTA_ENABLED = True
PICK_ENGINE_MAX_PICKS_PER_LEAGUE_ROUND = 5
PICK_ENGINE_MIN_TARGET_PICKS_PER_LEAGUE_ROUND = 4
PICK_ENGINE_PRICED_ODDS_BONUS = 0.08
PICK_ENGINE_MIN_MODEL_PROB = 0.52
PICK_ENGINE_EVIDENCE_GATES_BY_LEAGUE = {
    # Primary gates prefer mature model evidence. The engine can still fall
    # back below these when a round would otherwise miss the pick-volume target.
    "Premier League": {"neff_min": 4.0, "press_min": 2.0, "xg_min": 2.0},
    "Serie A": {"neff_min": 3.0, "press_min": 1.0, "xg_min": 1.0},
    "Ligue 1": {"neff_min": 4.0, "press_min": 2.0, "xg_min": 2.0},
}
PICK_ENGINE_FALLBACK_EVIDENCE_GATE = {
    "neff_min": 1.0,
    "press_min": 1.0,
    "xg_min": 1.0,
}
PICK_ENGINE_USE_GATE_PRIORITY_BY_LEAGUE = {
    # Ligue 1 backtests showed the primary evidence gate ranking below the
    # fallback pool; use raw model score for round ranking there.
    "Ligue 1": False,
}
PICK_ENGINE_MARKET_BLOCKLIST_BY_LEAGUE = {
    # Tiny but consistently bad in the current Ligue 1 reconstructed season.
    "Ligue 1": ["BTTS_NO"],
}
PICK_ENGINE_GATE_SCORE_ADJUSTMENTS_BY_LEAGUE = {
    # Ligue 1 fallback candidates outperformed primary-gate candidates in the
    # reconstructed 2025-26 sample, especially BTTS_YES and OU25_OVER.
    "Ligue 1": {"fallback": 0.06},
}
PICK_ENGINE_MARKET_SCORE_ADJUSTMENTS_BY_LEAGUE = {
    # Round-eval v1/v2 showed EPL over calls underperforming versus BTTS/Under.
    "Premier League": {"OU25_OVER": -0.18},
    # Serie A sweep showed BTTS_NO improves the Under-heavy selection.
    "Serie A": {"BTTS_NO": 0.18},
    # Round-eval v1-v4 showed Ligue 1 unders and broad BTTS calls underperforming.
    "Ligue 1": {"OU25_UNDER": -0.22, "BTTS_YES": -0.08},
}

# mu_total (expected total goals) must be in this range for a pick
MU_TOTAL_MIN = 1.6
MU_TOTAL_MAX = 3.4

# Evidence requirements (how much history we need before trusting a pick)
NEFF_MIN_FULL = 6.0         # Effective sample size for full engine
NEFF_MIN_GOALS = 1.0        # Model-signal mode: allow early-round coverage

PRESS_EVID_MIN_FULL = 2.0   # Pressure evidence minimum for full engine
PRESS_EVID_MIN_GOALS = 1.0  # Model-signal mode: allow early-round coverage

XG_EVID_MIN_FULL = 2.0      # xG evidence minimum for full engine
XG_EVID_MIN_GOALS = 1.0     # Model-signal mode: allow early-round coverage

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
# Pressure Form Weights
# Core weights are always used.
# Optional V2 weights are only used when those stats exist, and the pressure
# index is renormalized on available components (no forced fallback).
# ---------------------------------------------------------------------------
PRESSURE_W_SHOTS = 0.45
PRESSURE_W_SOT = 0.30
PRESSURE_W_CORNERS = 0.15
PRESSURE_W_POSSESSION = 0.10
PRESSURE_W_GOAL_ATTEMPTS = 0.12
PRESSURE_W_SHOTS_OFF = 0.06
PRESSURE_W_BLOCKED_SHOTS = 0.05
PRESSURE_W_ATTACKS = 0.05
PRESSURE_W_DANGEROUS_ATTACKS = 0.10

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
# xG Proxy V2 (additive enhancement over legacy xG proxy)
# ---------------------------------------------------------------------------
# Feature set: "v1" (legacy) or "v2" (enhanced).
# Safety default stays v1; promote v2 only after league-by-league validation.
XG_PROXY_FEATURE_SET_DEFAULT = "v1"

# League-level post-model calibration on xG outputs (walk-forward, train-only).
XG_PROXY_LEAGUE_CALIBRATION_ENABLED = False
XG_PROXY_LEAGUE_CALIBRATION_MIN_ROWS = 80
XG_PROXY_LEAGUE_CALIBRATION_PRIOR_STRENGTH = 80.0
XG_PROXY_LEAGUE_CALIBRATION_CLIP_MIN = 0.85
XG_PROXY_LEAGUE_CALIBRATION_CLIP_MAX = 1.15

# Opponent Elo factor guardrails.
XG_PROXY_ELO_FACTOR_CLIP_MIN = 0.80
XG_PROXY_ELO_FACTOR_CLIP_MAX = 1.20

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
    # Full-season no-leak tuning (2025-2026 season-to-date) with conservative yes-rate guardrails.
    "Premier League": 0.49,
    "Championship": 0.47,
    "Bundesliga": 0.61,
    "2. Bundesliga": 0.53,
    "Serie A": 0.44,
    "Serie B": 0.46,
    "La Liga": 0.49,
    "Ligue 1": 0.53,
    "Ligue 2": 0.45,
    "Eredivisie": 0.61,
    # Provisional (latest stable evidence pending full-season no-leak completion for these leagues):
    "Primeira Liga": 0.53,
    "Liga I": 0.41,
    "S\u00fcper Lig": 0.24,
    "S?per Lig": 0.24,
    "Serie A Brazil": 0.79,
    "Liga Profesional Argentina": 0.50,
}
BTTS_YES_THRESHOLD_BY_LEAGUE = {
    # Full-season no-leak tuning (2025-2026 season-to-date) with conservative yes-rate guardrails.
    "Premier League": 0.53,
    "Championship": 0.52,
    "Bundesliga": 0.61,
    "2. Bundesliga": 0.55,
    "Serie A": 0.48,
    "Serie B": 0.50,
    "La Liga": 0.52,
    "Ligue 1": 0.54,
    "Ligue 2": 0.51,
    "Eredivisie": 0.63,
    # Provisional (latest stable evidence pending full-season no-leak completion for these leagues):
    "Primeira Liga": 0.48,
    "Liga I": 0.40,
    "S\u00fcper Lig": 0.50,
    "S?per Lig": 0.50,
    "Serie A Brazil": 0.50,
    "Liga Profesional Argentina": 0.50,
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
# Poisson V2 (incremental enhancement over independent Poisson)
# ---------------------------------------------------------------------------
# This layer adds:
# 1) dispersion mixture (overdispersion control),
# 2) light home-away score dependence,
# 3) low-score correction (Dixon-Coles style rho).
# Keep conservative defaults and tune per league with time-split validation.
POISSON_V2_ENABLED = True
POISSON_V2_MAX_GOALS = 12

POISSON_V2_DISPERSION_ALPHA_DEFAULT = 0.06
POISSON_V2_DISPERSION_ALPHA_BY_LEAGUE = {
    # Add league overrides only after challenger validation.
}

POISSON_V2_DEP_STRENGTH_DEFAULT = 0.08
POISSON_V2_DEP_STRENGTH_BY_LEAGUE = {
    # Add league overrides only after challenger validation.
}

# Negative values increase 0-0 / 1-1 and reduce 1-0 / 0-1.
POISSON_V2_DC_RHO_DEFAULT = -0.04
POISSON_V2_DC_RHO_BY_LEAGUE = {
    # Add league overrides only after challenger validation.
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
