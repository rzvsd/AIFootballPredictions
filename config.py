"""
CGM-only configuration.

The legacy (non-CGM) config has been archived to:
  archive/legacy_non_cgm_engine/config_legacy.py
"""

# Gaussian kernel width (sigma) per league for Elo-similarity features.
# Used by: cgm/build_frankenstein.py and cgm/predict_upcoming.py
ELO_SIM_SIGMA_PER_LEAGUE = {
    "E0": 50.0,
    "D1": 40.0,
    "F1": 45.0,
    "SP1": 55.0,
    "I1": 45.0,
}

# ---------------------------------------------------------------------------
# Live scope (single source of truth)
#
# These defaults define the "live" prediction window for Milestone 4+.
# They can be overridden via CLI args in `cgm.predict_upcoming` / `cgm.pick_engine`.
# ---------------------------------------------------------------------------

# CGM league naming is not guaranteed to match bookmaker codes; use the literal values
# found in your exports (example: "Premier L").
LIVE_SCOPE_COUNTRY = "England"
LIVE_SCOPE_LEAGUE = "Premier L"

# Date-window is the most robust season selector (avoid season-label quirks).
LIVE_SCOPE_SEASON_START = "2025-08-01"
LIVE_SCOPE_SEASON_END = "2026-07-01"

# Optional horizon (days) for how far ahead to consider fixtures. Use None/0 to disable.
LIVE_SCOPE_HORIZON_DAYS = 14
