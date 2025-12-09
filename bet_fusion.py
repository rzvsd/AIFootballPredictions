"""
bet_fusion.py â€” Fusion engine and CLI

Exports generate_predictions(config) to compute 1X2 probabilities by fusing
champion XGB (home/away expected goals) with Dixon-Coles. Also includes a
simple CLI that logs bets via BankrollManager (odds placeholders when offline).
"""
import os
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import csv
import json

import pandas as pd
import joblib
from xgboost import XGBRegressor
import numpy as np
from scipy.stats import poisson
import math
import difflib

# Import our custom modules
from scripts.fetch_fixtures_understat import load_understat_fixtures
import xgb_trainer
import config
import feature_store
from engine import predictor_service, market_service, odds_service, value_service
from strategies import load_strategy

_ODDS_LOOKUP_CACHE: dict[str, tuple[float, str, dict]] = {}
_MISSING_ODDS: dict[str, set[tuple[str, str, str, str]]] = {}
_SYNTH_MARGIN_DEFAULT = float(os.getenv("BOT_SYNTH_MARGIN", "0.06"))
_SYNTH_MARGIN_BY_MARKET = {
    '1X2': 0.05,
    'DC': 0.05,
    'OU': 0.065,
    'TG Interval': 0.18,
}


def _normalize_fixture_key(date_val, home: str, away: str) -> tuple[str, str, str]:
    try:
        dt_obj = pd.to_datetime(date_val, errors="coerce")
        if pd.notna(dt_obj):
            date_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            date_str = str(date_val)
    except Exception:
        date_str = str(date_val)
    return (
        date_str,
        config.normalize_team_name(str(home or "")),
        config.normalize_team_name(str(away or "")),
    )


def _file_sig(path: str) -> tuple[float, str]:
    """Return (mtime, md5 hexdigest) for cache validation."""
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        return -1.0, ""
    try:
        with open(path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
    except Exception:
        digest = ""
    return mtime, digest


def _load_odds_lookup(league: str) -> dict:
    path = os.path.join("data", "odds", f"{league}.json")
    mtime, sig = _file_sig(path)
    if os.getenv("BOT_RELOAD_ODDS") == "1":
        _ODDS_LOOKUP_CACHE.pop(league, None)
    cached = _ODDS_LOOKUP_CACHE.get(league)
    if cached and cached[0] == mtime and cached[1] == sig:
        return cached[2]
    lookup: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        _ODDS_LOOKUP_CACHE[league] = (mtime, sig, lookup)
        return lookup
    for fx in data.get("fixtures", []):
        key = _normalize_fixture_key(fx.get("date"), fx.get("home"), fx.get("away"))
        lookup[key] = fx.get("markets") or {}
    _ODDS_LOOKUP_CACHE[league] = (mtime, sig, lookup)
    return lookup


def _lookup_market_odds(lookup: dict, date_val, home, away, market: str, outcome: str) -> Optional[float]:
    if not lookup:
        return None
    candidates = []
    try:
        dt = pd.to_datetime(date_val, errors="coerce")
    except Exception:
        dt = None
    if pd.notna(dt):
        candidates = [dt, dt - pd.Timedelta(days=1), dt + pd.Timedelta(days=1)]
    else:
        candidates = [date_val]
    for cand in candidates:
        key = _normalize_fixture_key(cand, home, away)
        markets = lookup.get(key)
        if not markets:
            continue
        price = _extract_market_price(markets, market, outcome)
        if price is not None:
            return price
    return None


def _extract_market_price(markets: dict, market: str, outcome: str) -> Optional[float]:
    try:
        if market == "1X2":
            slot_map = {"H": "home", "D": "draw", "A": "away"}
            slot = slot_map.get(outcome)
            if not slot:
                return None
            # Prefer explicit keys if provided
            main = markets.get("1X2") or {}
            val = None
            if isinstance(main, dict):
                if "main" in main and isinstance(main["main"], dict):
                    val = main["main"].get(slot)
                if val is None and "default" in main and isinstance(main["default"], dict):
                    val = main["default"].get(slot)
                if val is None:
                    # fallback: first tag_data dict
                    for tag_data in main.values():
                        if isinstance(tag_data, dict):
                            val = tag_data.get(slot)
                            if val is not None:
                                break
            if val is not None and not pd.isna(val):
                return float(val)
        elif market.startswith("OU "):
            line = market.split(" ", 1)[1].strip()
            bucket = (markets.get("OU") or {}).get(line, {})
            if isinstance(bucket, dict):
                tag_data = None
                if "main" in bucket:
                    tag_data = bucket.get("main") or {}
                elif "default" in bucket:
                    tag_data = bucket.get("default") or {}
                else:
                    # fallback: first tag_data
                    for v in bucket.values():
                        if isinstance(v, dict):
                            tag_data = v
                            break
                if tag_data:
                    val = tag_data.get(outcome)
                    if val is not None and not pd.isna(val):
                        return float(val)
        elif market == "DC":
            for tag_data in (markets.get("DC") or {}).values():
                val = tag_data.get(outcome)
                if val is not None and not pd.isna(val):
                    return float(val)
        elif market == "TG Interval":
            for tag_data in (markets.get("Intervals") or {}).values():
                val = tag_data.get(outcome)
                if val is not None and not pd.isna(val):
                    return float(val)
    except Exception:
        return None
    return None

# --- CONFIGURATION LOADER ---
CONFIG_PATH = Path(__file__).parent / "bot_config.yaml"

def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load bot configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    # Apply environment overrides if necessary (good for servers)
    config["stake_size"] = float(os.getenv("BOT_STAKE_SIZE", config.get("stake_size", 10.0)))
    config["probability_threshold"] = float(os.getenv("BOT_PROB_THRESHOLD", config.get("probability_threshold", 0.6)))
    config["edge_requirement"] = float(os.getenv("BOT_EDGE_REQUIREMENT", config.get("edge_requirement", 0.05)))
    # Logging switch
    lb = os.getenv("BOT_LOG_BETS")
    if lb is not None:
        config["log_bets"] = str(lb).strip().lower() in ("1","true","yes","on")
    else:
        config["log_bets"] = bool(config.get("log_bets", True))
    # Stake policy
    sp = os.getenv("BOT_STAKE_POLICY") or config.get("stake_policy", "flat")
    config["stake_policy"] = str(sp).strip().lower()
    try:
        config["kelly_fraction"] = float(os.getenv("BOT_KELLY_FRACTION", config.get("kelly_fraction", 0.25)))
    except Exception:
        config["kelly_fraction"] = 0.25
    # Min odds filter (drop very low prices)
    try:
        config["min_odds"] = float(os.getenv("BOT_MIN_ODDS", config.get("min_odds", 1.6)))
    except Exception:
        config["min_odds"] = 1.6

    # League override: allow environment to set the active league
    try:
        league_env = os.getenv("BOT_LEAGUE")
        if league_env:
            config["league"] = str(league_env).strip().upper()
        else:
            # normalize whatever is in YAML
            config["league"] = str(config.get("league", "E0")).strip().upper()
    except Exception:
        config["league"] = str(config.get("league", "E0")).strip().upper()
    # Mode: sim (default) or live (real odds only)
    mode_env = os.getenv("BOT_MODE")
    if mode_env:
        config["mode"] = str(mode_env).strip().lower()
    else:
        config["mode"] = str(config.get("mode", "sim")).strip().lower()
    return config

# --- BANKROLL & LOGGING (Your new class) ---
class BankrollManager:
    """Simple bankroll tracker and logger for bets."""
    def __init__(self, bankroll_file: str = "data/bankroll.json", log_file: str = "data/bets_log.csv"):
        self.bankroll_path = Path(bankroll_file)
        self.log_path = Path(log_file)
        self.bankroll = self._load_bankroll()
        print(f"Bankroll initialized at: {self.bankroll:.2f}")

    def _load_bankroll(self) -> float:
        self.bankroll_path.parent.mkdir(parents=True, exist_ok=True)
        if self.bankroll_path.exists():
            with open(self.bankroll_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("bankroll", 1000.0)) # Default to 1000 if not set
        return 1000.0

    def _save_bankroll(self) -> None:
        with open(self.bankroll_path, "w", encoding="utf-8") as f:
            json.dump({"bankroll": self.bankroll}, f)

    def _has_overdue_unsettled(self) -> bool:
        """Return True if there are unsettled bets older than the guard window."""
        try:
            if not self.log_path.exists():
                return False
            import pandas as pd  # local import
            df = pd.read_csv(self.log_path)
            if df.empty:
                return False
            unsettled = df[df['result'].isna() | (df['result'].astype(str).str.strip() == "")]
            if unsettled.empty:
                return False
            days_guard = float(os.getenv("BOT_UNSETTLED_GUARD_DAYS", "2"))
            if days_guard <= 0:
                return True
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days_guard)
            unsettled['date_dt'] = pd.to_datetime(unsettled['date'], errors='coerce')
            overdue = unsettled[unsettled['date_dt'] < cutoff]
            return not overdue.empty
        except Exception:
            # Fail-safe: if we cannot determine, block to avoid drain
            return True

    def log_bet(
        self,
        date: str,
        league: str,
        home: str,
        away: str,
        market: str,
        selection: str,
        odds: float,
        stake: float,
        prob: float,
        ev: float,
    ):
        """Deduct stake and log the placed bet."""
        if self._has_overdue_unsettled():
            print("Warning: Unsettled bets older than guard window; skipping new bet to avoid bankroll drift. Run settle_bets.py.")
            return
        if stake > self.bankroll:
            print(f"Warning: Insufficient bankroll ({self.bankroll:.2f}) for stake ({stake:.2f}). Skipping bet.")
            return
        self.bankroll -= stake
        self._save_bankroll()
        self._append_log(date, league, home, away, market, selection, odds, stake, prob, ev)
        print(f"Bet Logged. New bankroll: {self.bankroll:.2f}")

    def _append_log(self, date, league, home, away, market, selection, odds, stake, prob, ev):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "date",
                        "league",
                        "home_team",
                        "away_team",
                        "market",
                        "selection",
                        "odds",
                        "stake",
                        "model_prob",
                        "expected_value",
                    ]
                )
            writer.writerow([date, league, home, away, market, selection, odds, stake, f"{prob:.4f}", f"{ev:.4f}"])

# --- Helper functions used by the fusion engine ---
def _score_matrix(mu_h: float, mu_a: float, max_goals: int = 10, trim_epsilon: float = 0.0) -> np.ndarray:
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)
    ph = np.exp(-mu_h) * (mu_h ** hg) / np.array([math.factorial(int(i)) for i in hg])
    pa = np.exp(-mu_a) * (mu_a ** ag) / np.array([math.factorial(int(i)) for i in ag])
    P = np.outer(ph, pa)
    s = P.sum()
    if s > 0:
        P /= s
    if trim_epsilon and trim_epsilon > 0:
        P = np.where(P < trim_epsilon, 0.0, P)
        s2 = P.sum()
        if s2 > 0:
            P /= s2
    return P
def _score_matrix_negbin(mu_h: float, mu_a: float, k_h: float, k_a: float, max_goals: int = 10) -> np.ndarray:
    """Negative Binomial goal matrix with dispersion k (r)."""
    def nb_pmf_vec(mu: float, k: float, n: int) -> np.ndarray:
        mu = max(1e-9, float(mu)); k = max(1e-6, float(k))
        p = k / (k + mu)
        xs = np.arange(0, n+1, dtype=float)
        from math import lgamma
        logC = [lgamma(x + k) - lgamma(k) - lgamma(x + 1.0) for x in xs]
        logpmf = np.array(logC) + xs * np.log(max(1e-12, 1.0 - p)) + k * np.log(max(1e-12, p))
        pmf = np.exp(logpmf - np.max(logpmf))
        pmf = pmf / pmf.sum()
        return pmf
    ph = nb_pmf_vec(mu_h, k_h, max_goals)
    pa = nb_pmf_vec(mu_a, k_a, max_goals)
    P = np.outer(ph, pa)
    P = P / P.sum()
    return P

def _oneXtwo_from_matrix(P: np.ndarray) -> Dict[str, float]:
    home = float(np.tril(P, -1).sum())
    draw = float(np.diag(P).sum())
    away = float(np.triu(P, 1).sum())
    return {"p_H": home, "p_D": draw, "p_A": away}

# ---------------------------
# Calibration guards
# ---------------------------

def _model_predict_binary(model, p: np.ndarray) -> np.ndarray:
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
    except Exception:
        IsotonicRegression = object  # type: ignore
        LogisticRegression = object  # type: ignore
    if hasattr(model, 'predict') and model.__class__.__name__ == 'IsotonicRegression':
        return np.clip(model.predict(p), 1e-6, 1-1e-6)
    if hasattr(model, 'predict_proba') and model.__class__.__name__ == 'LogisticRegression':
        return model.predict_proba(p.reshape(-1,1))[:,1]
    # Fallback: identity
    return p


def _is_degenerate_calibrator_1x2(cal: Dict[str, object]) -> bool:
    """Return True if the 1X2 calibrator maps inputs to near-constants (degenerate)."""
    try:
        if not cal or not all(k in cal for k in ('H','D','A')):
            return True
        grid = np.linspace(0.01, 0.99, 25).astype(float)
        stds = []
        for k in ('H','D','A'):
            preds = _model_predict_binary(cal[k], grid.reshape(-1,1).ravel())
            stds.append(float(np.nanstd(preds)))
        # If all three vary less than threshold, consider degenerate
        return all(s < 1e-2 for s in stds)
    except Exception:
        return True


def _is_degenerate_calibrator_binary(model: object) -> bool:
    """Return True if a single binary calibrator (isotonic/platt) is near-constant."""
    try:
        grid = np.linspace(0.01, 0.99, 25).astype(float)
        preds = _model_predict_binary(model, grid.reshape(-1,1).ravel())
        return float(np.nanstd(preds)) < 1e-2
    except Exception:
        return True

# ---------------------------
# Market evaluation helpers
# ---------------------------

DEFAULT_OU_LINES = [1.5, 2.5, 3.5]
# Total-goals interval markets we evaluate by default.
# You can override via cfg['intervals'] (list of [low, high] pairs).
DEFAULT_INTERVALS = [(0, 3), (2, 4), (3, 5), (3, 6), (1, 2)]


def _goals_distribution(P: np.ndarray) -> np.ndarray:
    """Return distribution over total goals t by summing P[i,j] where i+j=t."""
    n = P.shape[0]  # assumes square matrix (0..max_goals)
    max_total = (n - 1) * 2
    dist = np.zeros(max_total + 1, dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i + j] += P[i, j]
    # numerical safety
    s = dist.sum()
    if s > 0:
        dist = dist / s
    return dist


def _eval_1x2(P: np.ndarray) -> Dict[str, float]:
    x = _oneXtwo_from_matrix(P)
    return {"H": x["p_H"], "D": x["p_D"], "A": x["p_A"]}


def _eval_double_chance(p1x2: Dict[str, float]) -> Dict[str, float]:
    return {
        "1X": p1x2["H"] + p1x2["D"],
        "X2": p1x2["D"] + p1x2["A"],
        "12": p1x2["H"] + p1x2["A"],
    }


def _eval_over_under(P: np.ndarray, lines: list[float]) -> Dict[str, Dict[str, float]]:
    dist = _goals_distribution(P)
    out: Dict[str, Dict[str, float]] = {}
    for line in lines:
        try:
            base = int(float(line))  # for 2.5 -> 2
        except Exception:
            continue
        under = float(dist[: base + 1].sum())  # totals 0..base
        over = float(dist[base + 1 :].sum())   # totals base+1 .. max
        out[f"{float(line):.1f}"] = {"Under": under, "Over": over}
    return out


def _eval_intervals(P: np.ndarray, intervals: list[tuple[int, int]]) -> Dict[str, float]:
    dist = _goals_distribution(P)
    out: Dict[str, float] = {}
    max_t = len(dist) - 1
    for a, b in intervals:
        a2 = max(0, int(a))
        b2 = min(max_t, int(b))
        if a2 > b2:
            prob = 0.0
        else:
            prob = float(dist[a2 : b2 + 1].sum())
        out[f"{a2}-{b2}"] = prob
    return out


def _evaluate_all_markets(P: np.ndarray,
                          ou_lines: Optional[list[float]] = None,
                          intervals: Optional[list[tuple[int, int]]] = None) -> Dict[str, Any]:
    p1x2 = _eval_1x2(P)
    dc = _eval_double_chance(p1x2)
    ou = _eval_over_under(P, ou_lines or DEFAULT_OU_LINES)
    ints = _eval_intervals(P, intervals or DEFAULT_INTERVALS)
    return {"1X2": p1x2, "DC": dc, "OU": ou, "Intervals": ints}

def _fused_matrix(mu_h_xgb: float, mu_a_xgb: float,
                  mu_h_dc: Optional[float], mu_a_dc: Optional[float],
                  w_dc: float, max_goals: int = 10) -> np.ndarray:
    """Combine XGB and DC score matrices with weight w_dc. Fallback to pure XGB if DC is unavailable or w_dc==0."""
    if mu_h_dc is None or mu_a_dc is None or w_dc <= 0.0:
        return _score_matrix(mu_h_xgb, mu_a_xgb, max_goals=max_goals)
    P_dc = _score_matrix(mu_h_dc, mu_a_dc, max_goals=max_goals)
    P_xgb = _score_matrix(mu_h_xgb, mu_a_xgb, max_goals=max_goals)
    P = w_dc * P_dc + (1.0 - w_dc) * P_xgb
    P /= P.sum()
    return P
def _score_from_cfg(mu_h: float, mu_a: float, cfg: Dict[str, Any], league: str, max_goals: int) -> np.ndarray:
    od = (cfg.get('overdispersion') or {})
    method = str(od.get('method','poisson')).lower()
    if method == 'negbin':
        k = float(od.get('k', 6.0))
        return _score_matrix_negbin(mu_h, mu_a, k_h=k, k_a=k, max_goals=max_goals)
    return _score_matrix(mu_h, mu_a, max_goals=max_goals, trim_epsilon=float(cfg.get('tail_epsilon', 0.0)))

def _feature_row_from_snapshot(stats: pd.DataFrame, home: str, away: str) -> Optional[pd.DataFrame]:
    """Build feature row in the order config.ULTIMATE_FEATURES expects using a team snapshot."""
    hs = stats.loc[stats["team"] == home]
    as_ = stats.loc[stats["team"] == away]
    if hs.empty or as_.empty:
        # Fallback: synthesize baseline rows using league medians so we don't drop fixtures
        try:
            med = stats.median(numeric_only=True)
            def make_side(prefix: str) -> pd.DataFrame:
                row = {
                    'xg_home_EWMA': float(med.get('xg_home_EWMA', med.get('xg_L5', 1.4))),
                    'xga_home_EWMA': float(med.get('xga_home_EWMA', med.get('xga_L5', 1.2))),
                    'xg_away_EWMA': float(med.get('xg_away_EWMA', med.get('xg_L5', 1.2))),
                    'xga_away_EWMA': float(med.get('xga_away_EWMA', med.get('xga_L5', 1.2))),
                    'ppg_home_EWMA': float(med.get('ppg_home_EWMA', med.get('gpg_L10', 1.5))),
                    'ppg_away_EWMA': float(med.get('ppg_away_EWMA', med.get('gpg_L10', 1.3))),
                    'corners_L10': float(med.get('corners_L10', 0.0)),
                    'corners_allowed_L10': float(med.get('corners_allowed_L10', 0.0)),
                    'elo': float(stats['elo'].mean()) if 'elo' in stats.columns else 1500.0,
                    # Optional band/sim features
                    'GFvsMid_H': float(med.get('GFvsMid_H', med.get('xg_L5', 1.3))),
                    'GAvsMid_H': float(med.get('GAvsMid_H', med.get('xga_L5', 1.2))),
                    'GFvsHigh_H': float(med.get('GFvsHigh_H', med.get('xg_L5', 1.1))),
                    'GAvsHigh_H': float(med.get('GAvsHigh_H', med.get('xga_L5', 1.3))),
                    'GFvsMid_A': float(med.get('GFvsMid_A', med.get('xg_L5', 1.1))),
                    'GAvsMid_A': float(med.get('GAvsMid_A', med.get('xga_L5', 1.2))),
                    'GFvsHigh_A': float(med.get('GFvsHigh_A', med.get('xg_L5', 1.0))),
                    'GAvsHigh_A': float(med.get('GAvsHigh_A', med.get('xga_L5', 1.4))),
                }
                return pd.DataFrame([row])
            if hs.empty:
                hs = make_side('H')
            if as_.empty:
                as_ = make_side('A')
        except Exception:
            return None
    feat_vals = []
    mapping = {
        # Prefer EWMA home/away splits; fallback to prior columns
        'ShotConv_H':            'xg_home_EWMA',
        'ShotConv_A':            'xg_away_EWMA',
        'ShotConvRec_H':         'xga_home_EWMA',
        'ShotConvRec_A':         'xga_away_EWMA',
        'PointsPerGame_H':       'ppg_home_EWMA',
        'PointsPerGame_A':       'ppg_away_EWMA',
        'CleanSheetStreak_H':    None,
        'CleanSheetStreak_A':    None,
        'xGDiff_H':              None,  # compute from ShotConv - ShotConvRec
        'xGDiff_A':              None,
        'CornersConv_H':         'corners_L10',
        'CornersConv_A':         'corners_L10',
        'CornersConvRec_H':      'corners_allowed_L10',
        'CornersConvRec_A':      'corners_allowed_L10',
        'NumMatches_H':          None,
        'NumMatches_A':          None,
        'Elo_H':                 'elo',
        'Elo_A':                 'elo',
        'EloDiff':               None,
        # Elo-band derived features
        'GFvsMid_H':             'GFvsMid_H',
        'GAvsMid_H':             'GAvsMid_H',
        'GFvsHigh_H':            'GFvsHigh_H',
        'GAvsHigh_H':            'GAvsHigh_H',
        'GFvsMid_A':             'GFvsMid_A',
        'GAvsMid_A':             'GAvsMid_A',
        'GFvsHigh_A':            'GFvsHigh_A',
        'GAvsHigh_A':            'GAvsHigh_A',
        # New: possession, corners totals, xG per possession, xG from corners
        'Possession_H':          'possession_home_EWMA',
        'Possession_A':          'possession_away_EWMA',
        'PossessionRec_H':       'possession_against_home_EWMA',
        'PossessionRec_A':       'possession_against_away_EWMA',
        'CornersFor_H':          'corners_for_home_EWMA',
        'CornersFor_A':          'corners_for_away_EWMA',
        'CornersAgainst_H':      'corners_against_home_EWMA',
        'CornersAgainst_A':      'corners_against_away_EWMA',
        'xGpp_H':                'xg_pp_home_EWMA',
        'xGpp_A':                'xg_pp_away_EWMA',
        'xGppRec_H':             'xg_pp_against_home_EWMA',
        'xGppRec_A':             'xg_pp_against_away_EWMA',
        'xGCorners_H':           'xg_from_corners_home_EWMA',
        'xGCorners_A':           'xg_from_corners_away_EWMA',
        'xGCornersRec_H':        'xg_from_corners_against_home_EWMA',
        'xGCornersRec_A':        'xg_from_corners_against_away_EWMA',
    }
    for col in config.ULTIMATE_FEATURES:
        key = mapping.get(col)
        if col == 'AvailabilityDiff':
            eh = float(hs.iloc[0].get('availability_index', 1.0))
            ea = float(as_.iloc[0].get('availability_index', 1.0))
            feat_vals.append(eh - ea)
            continue
        if col == 'Availability_H':
            feat_vals.append(float(hs.iloc[0].get('availability_index', 1.0)))
            continue
        if col == 'Availability_A':
            feat_vals.append(float(as_.iloc[0].get('availability_index', 1.0)))
            continue
        if col == 'xGDiff_H':
            val = (float(hs.iloc[0].get('xg_home_EWMA', np.nan)) - float(hs.iloc[0].get('xga_home_EWMA', np.nan)))
            feat_vals.append(val)
        elif col == 'xGDiff_A':
            val = (float(as_.iloc[0].get('xg_away_EWMA', np.nan)) - float(as_.iloc[0].get('xga_away_EWMA', np.nan)))
            feat_vals.append(val)
        elif col == 'NumMatches_H':
            feat_vals.append(20.0)
        elif col == 'NumMatches_A':
            feat_vals.append(20.0)
        elif col == 'EloDiff':
            eh = float(hs.iloc[0].get('elo', np.nan))
            ea = float(as_.iloc[0].get('elo', np.nan))
            feat_vals.append(eh - ea)
        elif col.endswith("_H"):
            if key and key in hs.columns:
                feat_vals.append(float(hs.iloc[0][key]))
            else:
                # Fallback legacy columns
                legacy = {
                    'ShotConv_H': 'xg_L5', 'ShotConvRec_H': 'xga_L5', 'PointsPerGame_H': 'gpg_L10',
                    'CornersConv_H': 'corners_L10', 'CornersConvRec_H': 'corners_allowed_L10', 'Elo_H': 'elo',
                    'GFvsMid_H': 'xg_L5', 'GAvsMid_H': 'xga_L5', 'GFvsHigh_H': 'xg_L5', 'GAvsHigh_H': 'xga_L5'
                }
                lk = legacy.get(col)
                feat_vals.append(float(hs.iloc[0].get(lk, 0.0)))
        elif col.endswith("_A"):
            if key and key in as_.columns:
                feat_vals.append(float(as_.iloc[0][key]))
            else:
                legacy = {
                    'ShotConv_A': 'xg_L5', 'ShotConvRec_A': 'xga_L5', 'PointsPerGame_A': 'gpg_L10',
                    'CornersConv_A': 'corners_L10', 'CornersConvRec_A': 'corners_allowed_L10', 'Elo_A': 'elo',
                    'GFvsMid_A': 'xg_L5', 'GAvsMid_A': 'xga_L5', 'GFvsHigh_A': 'xg_L5', 'GAvsHigh_A': 'xga_L5'
                }
                lk = legacy.get(col)
                feat_vals.append(float(as_.iloc[0].get(lk, 0.0)))
        else:
            # Unexpected columns
            feat_vals.append(0.0)
    return pd.DataFrame([feat_vals], columns=config.ULTIMATE_FEATURES)


def _read_any_fixture_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    def pick(aliases):
        for a in aliases:
            k = a.strip().lower()
            if k in cols:
                return cols[k]
        return None
    date_col = pick(["date", "matchdate", "kickoff", "ko"])
    home_col = pick(["home", "hometeam", "home_team", "home team", "home_team_api"])
    away_col = pick(["away", "awayteam", "away_team", "away team", "away_team_api"])
    if not (date_col and home_col and away_col):
        raise ValueError(f"Fixtures CSV {path} must contain date/home/away columns. Found: {list(df.columns)}")
    return pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors='coerce').dt.strftime("%Y-%m-%d %H:%M:%S"),
        "home_team_api": df[home_col].astype(str).str.strip(),
        "away_team_api": df[away_col].astype(str).str.strip(),
    })


def _fixtures_with_fallbacks(league_code: str, preferred_csv: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    """Backward-compatible shim kept for legacy callers (prefers Understat)."""
    if preferred_csv and os.path.exists(preferred_csv):
        try:
            return _read_any_fixture_csv(preferred_csv), f"from CSV: {preferred_csv}"
        except Exception:
            pass
    df = _load_understat_fixture_frame(league_code, {})
    return df, "understat"


def _load_understat_fixture_frame(league_code: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Fetch upcoming fixtures directly from Understat based on the desired horizon."""
    return predictor_service._load_understat_fixture_frame(league_code, cfg)


def _fill_odds_for_df(df: pd.DataFrame, league: str, with_odds: bool = True) -> pd.DataFrame:
    out = odds_service.fill_odds_for_df(df, league, with_odds=with_odds)
    if "odds" in out.columns:
        # Fill placeholders for missing to keep behavior
        out["odds"] = out["odds"].fillna(
            pd.Series(
                [_placeholder_odds(str(m), str(o)) for m, o in zip(out["market"].astype(str), out["outcome"].astype(str))],
                index=out.index,
            )
        )
    return out


def _flush_missing_odds_log(league: str) -> None:
    try:
        odds_service.flush_missing_odds_log(league)
    except Exception:
        pass

# --- Fusion engine ---
def generate_predictions(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return fused 1X2 probabilities for fixtures listed in data/fixtures.

    Expects models in advanced_models and enhanced features CSV present.
    Returns columns: date (if available), home, away, p_H, p_D, p_A
    """
    league = cfg.get('league', 'E0')
    max_goals = int(cfg.get('max_goals', config.MAX_GOALS_PER_LEAGUE.get(league, 10)))

    # Models (XGB only; DC removed)
    home_path = os.path.join("advanced_models", f"{league}_ultimate_xgb_home.pkl")
    away_path = os.path.join("advanced_models", f"{league}_ultimate_xgb_away.pkl")
    xgb_home = joblib.load(home_path)
    xgb_away = joblib.load(away_path)

    # Build per-team Elo-similarity histories from processed data (for dynamic features)
    try:
        proc_path = os.path.join('data','processed', f'{league}_merged_preprocessed.csv')
        df_pro = pd.read_csv(proc_path)
        d_elos = xgb_trainer._compute_ewma_elo_prematch(
            df_pro, half_life_matches=int(cfg.get('half_life_matches', 5)),
            elo_k=float(cfg.get('elo_k', 20.0)), elo_home_adv=float(cfg.get('elo_home_adv', 60.0)),
            elo_similarity_sigma=float(cfg.get('elo_similarity_sigma', config.ELO_SIM_SIGMA_PER_LEAGUE.get(league, 50.0)))
        )
        hist_home: dict[str, list[tuple[float,float,float]]] = {}
        hist_away: dict[str, list[tuple[float,float,float]]] = {}
        for _, rr in d_elos.iterrows():
            try:
                home_t = str(rr['HomeTeam']); away_t = str(rr['AwayTeam'])
                eh = float(rr['Elo_H']); ea = float(rr['Elo_A'])
                fthg = float(rr.get('FTHG', pd.NA)) if 'FTHG' in rr else None
                ftag = float(rr.get('FTAG', pd.NA)) if 'FTAG' in rr else None
                if fthg is None or ftag is None or pd.isna(fthg) or pd.isna(ftag):
                    continue
                hist_home.setdefault(home_t, []).append((ea, fthg, ftag))
                hist_away.setdefault(away_t, []).append((eh, ftag, fthg))
            except Exception:
                continue
    except Exception:
        hist_home = {}
        hist_away = {}

    fixtures = _load_understat_fixture_frame(league, cfg)
    if fixtures.empty:
        print(f"[warn] No upcoming fixtures available for {league}.")
        return pd.DataFrame(columns=["date", "home", "away", "p_H", "p_D", "p_A"])

    # Team snapshot (built from enhanced final features)
    enh_path = os.path.join("data", "enhanced", f"{league}_final_features.csv")
    # Optional cutoff for snapshot to avoid post-round leakage; if not set, use earliest fixture date
    as_of_env = os.getenv('BOT_SNAPSHOT_AS_OF') or os.getenv('BOT_FIXTURES_FROM')
    earliest_fixture = None
    try:
        if 'date' in fixtures.columns:
            earliest_fixture = pd.to_datetime(fixtures['date'], errors='coerce').min()
    except Exception:
        earliest_fixture = None
    try:
        if as_of_env:
            as_of_val = str(pd.to_datetime(as_of_env, errors='coerce'))
        elif pd.notna(earliest_fixture):
            as_of_val = (earliest_fixture - pd.Timedelta(seconds=1)).isoformat()
        else:
            as_of_val = None
    except Exception:
        as_of_val = as_of_env or None

    snap = feature_store.build_snapshot(
        enhanced_csv=enh_path,
        as_of=as_of_val,
        half_life_matches=int(cfg.get('half_life_matches', 5)),
        elo_k=float(cfg.get('elo_k', 20.0)),
        elo_home_adv=float(cfg.get('elo_home_adv', 60.0)),
        micro_agg_path=(cfg.get('micro_agg_path') or os.path.join('data','enhanced', f'{league}_micro_agg.csv'))
    )

    out = []
    for _, row in fixtures.iterrows():
        home_api = str(row.get("home_team", row.get("home_team_api", ""))).strip()
        away_api = str(row.get("away_team", row.get("away_team_api", ""))).strip()
        date_val = row.get("date")
        # Normalize to dataset names
        home = config.normalize_team_name(home_api)
        away = config.normalize_team_name(away_api)

        feat_row = _feature_row_from_snapshot(snap, home, away)
        if feat_row is None:
            continue
        # Inject Elo-similarity dynamic features if history available
        try:
            import math
            sigma = float(cfg.get('elo_similarity_sigma', config.ELO_SIM_SIGMA_PER_LEAGUE.get(league, 50.0)))
            def kernel_mean(hist, center):
                if not hist:
                    return (0.0, 0.0)
                inv2s2 = 1.0/(2.0*max(sigma,1e-6)*max(sigma,1e-6))
                wsum=0.0; gf_sum=0.0; ga_sum=0.0
                for opp_elo, gf, ga in hist:
                    w = math.exp(- (opp_elo-center)**2 * inv2s2)
                    wsum += w; gf_sum += w*float(gf); ga_sum += w*float(ga)
                if wsum<=0: return (0.0,0.0)
                return (gf_sum/wsum, ga_sum/wsum)
            # current pre-match Elo from features
            eh_cur = float(feat_row['Elo_H'].iloc[0]) if 'Elo_H' in feat_row.columns else None
            ea_cur = float(feat_row['Elo_A'].iloc[0]) if 'Elo_A' in feat_row.columns else None
            if ea_cur is not None:
                gf_h, ga_h = kernel_mean(hist_home.get(home), ea_cur)
                feat_row['GFvsSim_H'] = [gf_h]
                feat_row['GAvsSim_H'] = [ga_h]
            if eh_cur is not None:
                gf_a, ga_a = kernel_mean(hist_away.get(away), eh_cur)
                feat_row['GFvsSim_A'] = [gf_a]
                feat_row['GAvsSim_A'] = [ga_a]
        except Exception:
            # fallback zeros if something goes wrong
            for k in ('GFvsSim_H','GAvsSim_H','GFvsSim_A','GAvsSim_A'):
                if k not in feat_row.columns:
                    feat_row[k] = [0.0]
        mu_h_xgb = _predict_xgb(xgb_home, feat_row)
        mu_a_xgb = _predict_xgb(xgb_away, feat_row)
        # Optional per-league priors: blend XGB means with league baselines
        priors = (cfg.get('league_priors') or {}).get(league)
        if priors:
            try:
                w = float(priors.get('weight', 0.0))
                mh = priors.get('mu_home', None)
                ma = priors.get('mu_away', None)
                if w > 0.0 and mh is not None and ma is not None:
                    mu_h_xgb = (1.0 - w) * mu_h_xgb + w * float(mh)
                    mu_a_xgb = (1.0 - w) * mu_a_xgb + w * float(ma)
            except Exception:
                pass
        # Optionally switch/blend with ShotXG aggregates (MicroXG)
        mu_h_final, mu_a_final, _src = _blend_mu(cfg, home, away, mu_h_xgb, mu_a_xgb)
        P = _score_from_cfg(mu_h_final, mu_a_final, cfg, league, max_goals)

        probs = _oneXtwo_from_matrix(P)
        out.append({"date": date_val, "home": home, "away": away, **probs})

    return pd.DataFrame(out)


def generate_market_book(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return a flattened market book with probabilities for multiple markets per match.

    Columns: date, home, away, market, outcome, prob
    Markets include 1X2, Double Chance, Over/Under (configurable lines), and total-goal intervals.
    """
    preds = predictor_service.get_predictions(cfg)
    return market_service.build_market_book(preds, cfg)


# ---------------------------
# Phase 3: Odds, margin removal, Edge/EV
# ---------------------------

def _placeholder_odds(market: str, outcome: str) -> float:
    """Return a simple placeholder decimal odd for offline testing.

    These are not meant to be realistic; they enable EV/edge computation offline.
    """
    if market == "1X2":
        return 2.0
    if market == "DC":
        return 1.33
    if market.startswith("OU "):
        return 1.90
    if market == "TG Interval":
        return 3.00
    return 2.0


def _normalize_overround(group: pd.DataFrame) -> pd.DataFrame:
    """Given a market group with 'odds', compute normalized implied probabilities.

    Adds columns: p_imp, p_imp_norm
    """
    g = group.copy()
    g["p_imp"] = 1.0 / g["odds"].astype(float)
    s = g["p_imp"].sum()
    if s > 0:
        g["p_imp_norm"] = g["p_imp"] / s
    else:
        g["p_imp_norm"] = g["p_imp"]
    return g


def _market_is_exclusive(market: str) -> bool:
    """Return True if outcomes are mutually exclusive and cover the market.

    We treat '1X2' and 'OU <line>' as exclusive; 'DC' and 'TG Interval' are overlapping, so we do not normalize them.
    """
    m = str(market)
    return (m == '1X2') or m.startswith('OU ')


def _synth_odds_for_group(sub: pd.DataFrame, market: str, margin_lookup: Dict[str, float], default_margin: float, force_all: bool) -> pd.DataFrame:
    """Derive bookmaker-like odds from model probabilities when real odds are missing."""
    df = sub.copy()
    mask = df["odds"].isna() | force_all
    if not mask.any():
        df["_synthetic_odds"] = False
        return df
    probs = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0)
    base = _market_base(market)
    margin = margin_lookup.get(base, default_margin)
    if _market_is_exclusive(market):
        total = probs.sum()
        if total <= 0:
            fair_probs = pd.Series(1.0 / max(len(df), 1), index=df.index)
        else:
            fair_probs = probs / total
        adj_prob = (fair_probs * (1.0 + margin)).clip(lower=1e-6)
        new_odds = 1.0 / adj_prob
    else:
        adj_prob = (probs * (1.0 + margin)).clip(lower=1e-6)
        new_odds = 1.0 / adj_prob
    df.loc[mask, "odds"] = new_odds[mask]
    df["_synthetic_odds"] = mask
    # Avoid absurd payouts if prob is tiny
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce").clip(lower=1.01)
    return df


def attach_value_metrics(
    market_df: pd.DataFrame,
    use_placeholders: bool = True,
    synthesize_missing_odds: bool = True,
    synth_margin: float | None = None,
    league_code: str | None = None,
    synth_margin_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Attach odds, normalized implied probability, edge and EV to market probabilities."""
    return value_service.attach_value_metrics(
        market_df,
        use_placeholders=use_placeholders,
        synthesize_missing_odds=synthesize_missing_odds,
        synth_margin=synth_margin,
        league_code=league_code,
        synth_margin_map=synth_margin_map,
    )


def _log_odds_status(df: pd.DataFrame, fixtures: pd.DataFrame) -> None:
    """Print a concise odds coverage warning for CLI runs."""
    try:
        total = len(df)
        real = int((df.get("price_source") == "real").sum()) if "price_source" in df.columns else 0
        synth = total - real
        min_date = pd.to_datetime(fixtures["date"], errors="coerce").min() if not fixtures.empty else None
        sim_flag = False
        if pd.notna(min_date):
            days_ahead = (min_date - pd.Timestamp.utcnow()).total_seconds() / 86400.0
            sim_flag = days_ahead > 3
        print(f"[odds] markets={total} real={real} synth={synth} (sim_mode={sim_flag})")
        if total > 0 and (real / total) < 0.2:
            print("[warn] Odds coverage low (<20% real). EV hidden for synthetic prices.")
        if sim_flag and real == 0:
            print("[warn] Fixtures appear far in the future; real odds unlikely. Treat this run as simulation.")
    except Exception:
        pass


# --- Thresholds, odds filling, and CLI ---
def _market_base(m: str) -> str:
    return m.split()[0] if m.startswith("OU ") else m


def compute_tg_ci(cfg: Dict[str, Any], level: float = 0.8) -> Dict[tuple, str]:
    league = cfg.get('league','E0')
    max_goals = int(cfg.get('max_goals', config.MAX_GOALS_PER_LEAGUE.get(league, 10)))
    home_json = os.path.join("advanced_models", f"{league}_ultimate_xgb_home.json")
    away_json = os.path.join("advanced_models", f"{league}_ultimate_xgb_away.json")
    home_pkl = os.path.join("advanced_models", f"{league}_ultimate_xgb_home.pkl")
    away_pkl = os.path.join("advanced_models", f"{league}_ultimate_xgb_away.pkl")
    def _load_model(json_path, pkl_path):
        try:
            if os.path.exists(json_path):
                m = XGBRegressor()
                m.load_model(json_path)
                return m
        except Exception:
            pass
        return joblib.load(pkl_path)
    xgb_home = _load_model(home_json, home_pkl)
    xgb_away = _load_model(away_json, away_pkl)
    enh_path = os.path.join("data", "enhanced", f"{league}_final_features.csv")
    # Optional cutoff for snapshot to avoid post-round leakage
    as_of_env = os.getenv('BOT_SNAPSHOT_AS_OF') or os.getenv('BOT_FIXTURES_FROM')
    try:
        as_of_val = str(pd.to_datetime(as_of_env, errors='coerce')) if as_of_env else None
    except Exception:
        as_of_val = as_of_env or None
    snap = feature_store.build_snapshot(
        enhanced_csv=enh_path,
        as_of=as_of_val,
        half_life_matches=int(cfg.get('half_life_matches', 5)),
        elo_k=float(cfg.get('elo_k', 20.0)),
        elo_home_adv=float(cfg.get('elo_home_adv', 60.0)),
        micro_agg_path=(cfg.get('micro_agg_path') or os.path.join('data','enhanced', f'{league}_micro_agg.csv'))
    )
    fixtures = _load_understat_fixture_frame(league, cfg)
    if fixtures.empty:
        return pd.DataFrame(columns=["date", "home", "away", "market", "outcome", "prob"])
    out = {}
    for _, r in fixtures.iterrows():
        home_api = str(r.get("home_team", r.get("home_team_api", ""))).strip()
        away_api = str(r.get("away_team", r.get("away_team_api", ""))).strip()
        date_val = r.get("date")
        home = config.normalize_team_name(home_api)
        away = config.normalize_team_name(away_api)
        feat_row = _feature_row_from_snapshot(snap, home, away)
        if feat_row is None:
            continue
        mu_h = _predict_xgb(xgb_home, feat_row)
        mu_a = _predict_xgb(xgb_away, feat_row)
        P = _score_from_cfg(mu_h, mu_a, cfg, league, max_goals)
        dist = _goals_distribution(P)
        n = len(dist)
        csum = np.cumsum(dist)
        best = (0, n-1, 1e9)
        for a in range(n):
            for b in range(a, n):
                mass = csum[b] - (csum[a-1] if a>0 else 0.0)
                if mass >= level:
                    if (b-a) < best[2]:
                        best = (a,b,b-a)
                    break
        a,b,_ = best
        out[(str(date_val), home, away)] = f"{a}-{b}"
    return out



def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Hybrid DC+xG market engine with value metrics")
    ap.add_argument('--top', type=int, default=20, help='Show top N picks by EV')
    args = ap.parse_args()

    cfg = load_config()
    mode = str(cfg.get('mode', 'sim')).lower()
    print("\n--- Hybrid Engine Started ---")
    print(cfg)
    print(f"MODE: {mode.upper()} (sim = placeholders allowed, live = real odds only)")

    market_df = generate_market_book(cfg)
    if market_df.empty:
        print("No fixtures available.")
        return

    league = cfg.get('league', 'E0')
    if mode == 'live':
        # Live mode: focus on bettable markets only (exclude TG intervals unless feed supports them)
        market_df = market_df[~market_df['market'].eq('TG Interval')].copy()
        df_with_odds = odds_service.fill_odds_for_df(market_df, league, with_odds=True)
        odds_service.flush_missing_odds_log(league)
        df_val = value_service.attach_value_metrics(
            df_with_odds,
            use_placeholders=False,
            synthesize_missing_odds=False,
            league_code=league,
        )
        df_val = df_val[pd.notna(df_val.get('odds'))].copy()
    else:
        df_with_odds = _fill_odds_for_df(market_df, league, with_odds=True)
    df_val = attach_value_metrics(df_with_odds, use_placeholders=True, league_code=league)
        _flush_missing_odds_log(league)
    _log_odds_status(df_val, market_df)
    # Filter out very low odds (e.g., < 1.60)
    try:
        min_odds = float(cfg.get('min_odds', 1.6))
    except Exception:
        min_odds = 1.6
    if 'odds' in df_val.columns:
        df_val = df_val[pd.to_numeric(df_val['odds'], errors='coerce') >= float(min_odds)].copy()

    # Strategy selection
    context = {
        "league": league,
        "mode": mode,
        "strategies": cfg.get("strategies", {}),
    }
    strat_cfg = cfg.get("strategies", {}) or {}
    candidates = []
    for name, opts in strat_cfg.items():
        if not isinstance(opts, dict) or not opts.get("enabled", True):
            continue
        path = opts.get("path") or f"strategies.{name}:generate_candidates"
        try:
            fn = load_strategy(path)
            cands = fn(context, df_val)
            candidates.extend(cands or [])
        except Exception as e:
            print(f"[warn] strategy {name} failed: {e}")
            continue
    picks = pd.DataFrame(candidates)
    if picks.empty:
        print("No value picks under current strategies.")
        return

    # ensure EV column present
    if "EV" not in picks.columns and "disp_ev" in picks.columns:
        picks["EV"] = picks["disp_ev"]
    picks.sort_values(['EV','prob'], ascending=[False, False], inplace=True)
    show = picks.head(args.top)

    # Export full ranked picks to CSV (top) and full unfiltered snapshot
    try:
        ts = pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')
        rep_dir = Path('reports'); rep_dir.mkdir(parents=True, exist_ok=True)
        out_csv = rep_dir / f"{cfg.get('league','E0')}_{ts}_picks.csv"
        full_csv = rep_dir / f"{cfg.get('league','E0')}_{ts}_all_markets.csv"
        cols = ['date','home','away','market','outcome','prob','fair_odds','book_odds','price_source','edge','EV']
        picks.to_csv(out_csv, index=False, columns=[c for c in cols if c in picks.columns])
        # Save full snapshot (all markets)
        df_val.to_csv(full_csv, index=False, columns=[c for c in cols if c in df_val.columns])
        print(f"Saved picks CSV -> {out_csv}")
        print(f"Saved full markets CSV -> {full_csv}")
    except Exception as e:
        print(f"Could not save picks CSV: {e}")

    print("\n--- Top Picks (by EV) ---")
    for _, r in show.iterrows():
        print(f"{r['date']}  {r['home']} vs {r['away']} | {r['market']} {r['outcome']}  p={float(r['prob']):.2%}  odds={float(r['odds']):.2f}  edge={float(r['edge']):.2%}  EV={float(r['EV']):.2f}")

    # Portfolio risk control (correlation-aware throttling)
    def _compute_cluster_key(row) -> tuple:
        # cluster by date (day) + market base by default
        d = str(row.get('date'))[:10]
        m = _market_base(str(row.get('market','')))
        return (d, m)
    portfolio_cfg = cfg.get('portfolio', {}) or {}
    throttle = bool(portfolio_cfg.get('throttle', True))
    min_mult = float(portfolio_cfg.get('min_mult', 0.4))
    mode = str(portfolio_cfg.get('throttle_mode','sqrt')).lower()  # 'sqrt' or 'linear'
    # Precompute cluster sizes
    clusters = {}
    for _, r in show.iterrows():
        k = _compute_cluster_key(r)
        clusters[k] = clusters.get(k, 0) + 1

    # Log bets (optional)
    if cfg.get('log_bets', True):
        bm = BankrollManager()
        flat_stake = float(cfg.get('stake_size', 10.0))
        policy = cfg.get('stake_policy','flat')
        k_frac = float(cfg.get('kelly_fraction', 0.25))
        for _, r in show.iterrows():
            ev = float(r['EV'])
            if ev <= 0:
                continue
            p = float(r['prob']); O = float(r['odds'])
            if policy == 'kelly':
                # Kelly fraction per bet
                try:
                    f_star = (p*O - (1.0 - p)) / max(O - 1.0, 1e-9)
                except Exception:
                    f_star = 0.0
                f_star = max(0.0, min(1.0, f_star))
                stake = max(0.0, min(bm.bankroll, bm.bankroll * k_frac * f_star))
            else:
                stake = min(flat_stake, bm.bankroll)
            # Apply portfolio throttle multiplier based on cluster size
            if throttle:
                k = _compute_cluster_key(r)
                n = max(1, clusters.get(k, 1))
                if mode == 'linear':
                    mult = max(min_mult, 1.0 / float(n))
                else:
                    mult = max(min_mult, 1.0 / float(n)**0.5)
                stake *= mult
            if stake <= 0:
                continue
            bm.log_bet(
                date=str(r['date']),
                league=cfg.get('league','E0'),
                home=str(r['home']),
                away=str(r['away']),
                market=str(r['market']),
                selection=str(r['outcome']),
                odds=O,
                stake=stake,
                prob=p,
                ev=ev
            )
    else:
        print("Print-only mode: set log_bets=true to record stakes.")

    _flush_missing_odds_log(league)


# Note: keep __main__ at the end of file so that all helpers are defined before CLI runs
def _predict_xgb(model, feat_df: pd.DataFrame) -> float:
    """Predict with XGB model using model's known feature names if available to avoid name mismatch."""
    try:
        names = getattr(model, 'feature_names_in_', None)
        if names is None:
            try:
                names = model.get_booster().feature_names
            except Exception:
                names = None
        if names is not None:
            # ensure order and subset strictly to expected names
            cols = [c for c in names if c in feat_df.columns]
            X = feat_df[cols].values
        else:
            X = feat_df.values
        y = model.predict(X)
        return float(y[0] if hasattr(y, '__len__') else y)
    except Exception:
        # Fallback: try slicing first n features if model exposes n_features_in_
        try:
            n = int(getattr(model, 'n_features_in_', 0))
            if n > 0:
                X = feat_df.iloc[:, :n].values
                y = model.predict(X)
                return float(y[0] if hasattr(y, '__len__') else y)
        except Exception:
            pass
        y = model.predict(feat_df.values)
        return float(y[0] if hasattr(y, '__len__') else y)


# --- Probabilistic model integration (Phase 3) ---
_PROB_MODEL_CACHE: Dict[str, dict] = {}
_MICRO_AGG_CACHE: Dict[tuple[str, str], tuple[float, str, dict]] = {}

def _load_prob_models(league: str) -> dict:
    if league in _PROB_MODEL_CACHE:
        return _PROB_MODEL_CACHE[league]
    models = {}
    # NGBoost Poisson models
    try:
        hp = Path('advanced_models') / f'{league}_ngb_poisson_home.pkl'
        ap = Path('advanced_models') / f'{league}_ngb_poisson_away.pkl'
        if hp.exists() and ap.exists():
            models['ngb_poisson_home'] = joblib.load(hp)
            models['ngb_poisson_away'] = joblib.load(ap)
    except Exception:
        pass
    _PROB_MODEL_CACHE[league] = models
    return models


def _load_micro_map(path: str, as_of: str | None = None) -> dict:
    """Load micro aggregates CSV and return mapping (team_std, side) -> xg_for_EWMA.

    Team names are normalized via config.normalize_team_name to match engine names.
    The cache respects file mtime so long-running dashboards pick up refreshed data.
    """
    if not path:
        return {}
    cache_key = (path, str(as_of or ""))
    mtime, sig = _file_sig(path)
    if os.getenv("BOT_RELOAD_MICRO") == "1":
        _MICRO_AGG_CACHE.pop(cache_key, None)
    cached = _MICRO_AGG_CACHE.get(cache_key)
    if cached and cached[0] == mtime and cached[1] == sig:
        return cached[2]
    try:
        import pandas as pd  # local import
        if not os.path.exists(path):
            _MICRO_AGG_CACHE[cache_key] = (mtime, sig, {})
            return _MICRO_AGG_CACHE[cache_key][2]
        df = pd.read_csv(path)
        # Normalize
        if 'team' not in df.columns or 'side' not in df.columns:
            _MICRO_AGG_CACHE[cache_key] = (mtime, sig, {})
            return _MICRO_AGG_CACHE[cache_key][2]
        try:
            df['date'] = pd.to_datetime(df.get('date'), errors='coerce')
        except Exception:
            pass
        if as_of and 'date' in df.columns:
            try:
                cutoff = pd.to_datetime(as_of)
                df = df[df['date'] <= cutoff]
            except Exception:
                pass
        df['team'] = df['team'].astype(str).map(config.normalize_team_name)
        # Keep latest per (team, side)
        df = df.sort_values(['team','side','date'])
        last = df.groupby(['team','side'], as_index=False).tail(1)
        m = {}
        for _, r in last.iterrows():
            try:
                key = (str(r['team']), str(r['side']).upper()[:1])
                val = float(r.get('xg_for_EWMA', r.get('xg_for', 0.0)) or 0.0)
                if val <= 0:
                    # fallback to per-match xg_for if EWMA missing/nonpositive
                    val = float(r.get('xg_for', 0.0) or 0.0)
                m[key] = max(0.05, val)
            except Exception:
                continue
        _MICRO_AGG_CACHE[cache_key] = (mtime, sig, m)
        return m
    except Exception:
        _MICRO_AGG_CACHE[cache_key] = (mtime, sig, {})
        return _MICRO_AGG_CACHE[cache_key][2]


def _blend_mu(cfg: Dict[str, Any], home: str, away: str, mu_h_macro: float, mu_a_macro: float) -> tuple[float, float, str]:
    src = str(cfg.get('xg_source', 'macro')).lower()
    if src not in ('macro','micro','blend'):
        src = 'macro'
    micro_path = cfg.get('micro_agg_path', os.path.join('data','enhanced','micro_agg.csv'))
    as_of_env = os.getenv('BOT_SNAPSHOT_AS_OF') or os.getenv('BOT_FIXTURES_FROM')
    micro = _load_micro_map(micro_path, as_of=as_of_env) if src in ('micro','blend') else {}
    mu_h_micro = micro.get((home, 'H'))
    mu_a_micro = micro.get((away, 'A'))
    if src == 'micro':
        if mu_h_micro is not None and mu_a_micro is not None:
            return float(mu_h_micro), float(mu_a_micro), 'micro'
        return mu_h_macro, mu_a_macro, 'macro'
    if src == 'blend':
        try:
            w = float(cfg.get('xg_blend_weight', 0.5))
        except Exception:
            w = 0.5
        if mu_h_micro is not None:
            mu_h_macro = (1.0 - w) * mu_h_macro + w * float(mu_h_micro)
        if mu_a_micro is not None:
            mu_a_macro = (1.0 - w) * mu_a_macro + w * float(mu_a_micro)
        return mu_h_macro, mu_a_macro, 'blend'
    return mu_h_macro, mu_a_macro, 'macro'

def _predict_goals_mu(cfg: Dict[str, Any], league: str, feat_row: pd.DataFrame,
                      xgb_home_model, xgb_away_model) -> tuple[float, float, str]:
    pm = (cfg.get('prob_model') or {})
    enabled = bool(pm.get('enabled', False))
    kind = str(pm.get('kind','')).lower()
    if enabled and kind in ('ngboost_poisson','ngb_poisson','ngb_pois'):
        models = _load_prob_models(league)
        mH = models.get('ngb_poisson_home'); mA = models.get('ngb_poisson_away')
        if mH is not None and mA is not None:
            try:
                # NGBoost predict returns expectation under Poisson dist by default via .predict
                mu_h = float(mH.predict(feat_row)[0])
                mu_a = float(mA.predict(feat_row)[0])
                # Guard against non-positive
                mu_h = max(0.05, mu_h); mu_a = max(0.05, mu_a)
                return mu_h, mu_a, 'ngb_poisson'
            except Exception:
                pass
    # Fallback to XGB
    return _predict_xgb(xgb_home_model, feat_row), _predict_xgb(xgb_away_model, feat_row), 'xgb'
 
if __name__ == "__main__":
    main()
