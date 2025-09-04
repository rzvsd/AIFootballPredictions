"""
bet_fusion.py — Fusion engine and CLI

Exports generate_predictions(config) to compute 1X2 probabilities by fusing
champion XGB (home/away expected goals) with Dixon-Coles. Also includes a
simple CLI that fetches odds and logs bets via BankrollManager.
"""
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import csv
import json

import pandas as pd
import requests
from requests.exceptions import RequestException
import joblib
import numpy as np
from scipy.stats import poisson
import math
import difflib

# Import our custom modules
from scripts import bookmaker_api
import xgb_trainer
import config
import feature_store

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

    def log_bet(self, date: str, league: str, home: str, away: str, market: str, odds: float, stake: float, prob: float, ev: float):
        """Deduct stake and log the placed bet."""
        if stake > self.bankroll:
            print(f"Warning: Insufficient bankroll ({self.bankroll:.2f}) for stake ({stake:.2f}). Skipping bet.")
            return
        self.bankroll -= stake
        self._save_bankroll()
        self._append_log(date, league, home, away, market, odds, stake, prob, ev)
        print(f"Bet Logged. New bankroll: {self.bankroll:.2f}")

    def _append_log(self, date, league, home, away, market, odds, stake, prob, ev):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["date", "league", "home_team", "away_team", "market", "odds", "stake", "model_prob", "expected_value"])
            writer.writerow([date, league, home, away, market, odds, stake, f"{prob:.4f}", f"{ev:.4f}"])

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
DEFAULT_INTERVALS = [(0, 3), (1, 3), (2, 4), (2, 5), (3, 5), (3, 6)]


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

def _feature_row_from_snapshot(stats: pd.DataFrame, home: str, away: str) -> Optional[pd.DataFrame]:
    """Build feature row in the order config.ULTIMATE_FEATURES expects using a team snapshot."""
    hs = stats.loc[stats["team"] == home]
    as_ = stats.loc[stats["team"] == away]
    if hs.empty or as_.empty:
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
    }
    for col in config.ULTIMATE_FEATURES:
        key = mapping.get(col)
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
    tried = []
    if preferred_csv and os.path.exists(preferred_csv):
        try:
            return _read_any_fixture_csv(preferred_csv), f"from CSV: {preferred_csv}"
        except Exception:
            tried.append(preferred_csv)
    weekly = os.path.join("data", "fixtures", f"{league_code}_weekly_fixtures.csv")
    if os.path.exists(weekly):
        try:
            return _read_any_fixture_csv(weekly), f"from CSV: {weekly}"
        except Exception:
            tried.append(weekly)
    manual = os.path.join("data", "fixtures", f"{league_code}_manual.csv")
    if os.path.exists(manual):
        try:
            return _read_any_fixture_csv(manual), f"from CSV: {manual}"
        except Exception as e:
            print(f"[fallback] Failed to read manual fixtures {manual}: {e}")
    else:
        os.makedirs(os.path.dirname(manual), exist_ok=True)
        with open(manual, "w", encoding="utf-8") as f:
            f.write("date,home,away\n")
            f.write("2025-08-30 17:00,Man City,Everton\n")
            f.write("2025-08-31 19:30,Man United,Arsenal\n")
    print(f"[fixtures] No weekly fixtures found. Using/created manual template: {manual}")
    return pd.DataFrame(columns=["date","home_team_api","away_team_api"]), "manual"

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

    # Team snapshot (built from enhanced final features)
    enh_path = os.path.join("data", "enhanced", f"{league}_final_features.csv")
    snap = feature_store.build_snapshot(
        enhanced_csv=enh_path,
        as_of=None,
        half_life_matches=int(cfg.get('half_life_matches', 5)),
        elo_k=float(cfg.get('elo_k', 20.0)),
        elo_home_adv=float(cfg.get('elo_home_adv', 60.0))
    )
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

    # Fixtures with fallbacks
    fixtures, source = _fixtures_with_fallbacks(league_code=league)

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
        P = _score_matrix(mu_h_xgb, mu_a_xgb, max_goals=max_goals, trim_epsilon=float(cfg.get('tail_epsilon', 0.0)))

        probs = _oneXtwo_from_matrix(P)
        out.append({"date": date_val, "home": home, "away": away, **probs})

    return pd.DataFrame(out)


def generate_market_book(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return a flattened market book with probabilities for multiple markets per match.

    Columns: date, home, away, market, outcome, prob
    Markets include 1X2, Double Chance, Over/Under (configurable lines), and total-goal intervals.
    """
    league = cfg.get('league', 'E0')
    use_calibration = bool(cfg.get('use_calibration', True))
    max_goals = int(cfg.get('max_goals', config.MAX_GOALS_PER_LEAGUE.get(league, 10)))

    # Parse market config
    ou_lines = cfg.get('ou_lines', DEFAULT_OU_LINES)
    try:
        ou_lines = [float(x) for x in ou_lines]
    except Exception:
        ou_lines = DEFAULT_OU_LINES
    intervals_cfg = cfg.get('intervals', DEFAULT_INTERVALS)
    intervals: list[tuple[int, int]] = []
    for iv in intervals_cfg:
        try:
            if isinstance(iv, (list, tuple)) and len(iv) == 2:
                intervals.append((int(iv[0]), int(iv[1])))
        except Exception:
            continue
    if not intervals:
        intervals = DEFAULT_INTERVALS

    # Models (XGB only)
    home_path = os.path.join("advanced_models", f"{league}_ultimate_xgb_home.pkl")
    away_path = os.path.join("advanced_models", f"{league}_ultimate_xgb_away.pkl")
    xgb_home = joblib.load(home_path)
    xgb_away = joblib.load(away_path)

    enh_path = os.path.join("data", "enhanced", f"{league}_final_features.csv")
    snap = feature_store.build_snapshot(
        enhanced_csv=enh_path,
        as_of=None,
        half_life_matches=int(cfg.get('half_life_matches', 5)),
        elo_k=float(cfg.get('elo_k', 20.0)),
        elo_home_adv=float(cfg.get('elo_home_adv', 60.0))
    )
    fixtures, source = _fixtures_with_fallbacks(league_code=league)

    rows = []
    for _, r in fixtures.iterrows():
        home_api = str(r.get("home_team", r.get("home_team_api", ""))).strip()
        away_api = str(r.get("away_team", r.get("away_team_api", ""))).strip()
        date_val = r.get("date")
        home = config.normalize_team_name(home_api)
        away = config.normalize_team_name(away_api)

        feat_row = _feature_row_from_snapshot(snap, home, away)
        if feat_row is None:
            continue
        mu_h_xgb = _predict_xgb(xgb_home, feat_row)
        mu_a_xgb = _predict_xgb(xgb_away, feat_row)
        P = _score_matrix(mu_h_xgb, mu_a_xgb, max_goals=max_goals, trim_epsilon=float(cfg.get('tail_epsilon', 0.0)))

        markets = _evaluate_all_markets(P, ou_lines=ou_lines, intervals=intervals)
        # Apply calibration if available
        try:
            from calibrators import load_calibrators, apply_calibration_1x2
            cal1_path = os.path.join('calibrators', f"{league}_1x2.pkl")
            cal1 = load_calibrators(cal1_path)
        except Exception:
            cal1 = None
        if use_calibration and cal1 and not _is_degenerate_calibrator_1x2(cal1):
            arr = np.array([[markets['1X2']['H'], markets['1X2']['D'], markets['1X2']['A']]])
            arr_cal = apply_calibration_1x2(arr, cal1)[0]
            markets['1X2'] = {'H': float(arr_cal[0]), 'D': float(arr_cal[1]), 'A': float(arr_cal[2])}
        elif cal1:
            # Skip degenerate calibrator
            pass
        # Optional shrinkage toward neutral for stability
        try:
            shrink_1x2 = float(cfg.get('calibration_shrink', {}).get('oneX2', 0.0))
        except Exception:
            shrink_1x2 = 0.0
        if shrink_1x2 > 0:
            prior = 1.0/3.0
            markets['1X2'] = {
                'H': (1.0 - shrink_1x2) * markets['1X2']['H'] + shrink_1x2 * prior,
                'D': (1.0 - shrink_1x2) * markets['1X2']['D'] + shrink_1x2 * prior,
                'A': (1.0 - shrink_1x2) * markets['1X2']['A'] + shrink_1x2 * prior,
            }
        try:
            calou_path = os.path.join('calibrators', f"{league}_ou25.pkl")
            calou = load_calibrators(calou_path)
        except Exception:
            calou = None
        if use_calibration and calou and 'ou25' in calou and '2.5' in markets['OU'] and not _is_degenerate_calibrator_binary(calou.get('ou25')):
            model = calou['ou25']
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
            over_p = np.array([markets['OU']['2.5']['Over']])
            if isinstance(model, IsotonicRegression):
                cal_over = float(np.clip(model.predict(over_p), 1e-6, 1-1e-6))
            elif isinstance(model, LogisticRegression):
                cal_over = float(model.predict_proba(over_p.reshape(-1,1))[:,1][0])
            else:
                cal_over = float(over_p[0])
            markets['OU']['2.5'] = {'Over': cal_over, 'Under': float(max(0.0, 1.0 - cal_over))}
        elif calou:
            # Skip degenerate OU calibrator
            pass
        # Optional shrinkage toward 50/50 for OU 2.5
        try:
            shrink_ou = float(cfg.get('calibration_shrink', {}).get('ou25', 0.0))
        except Exception:
            shrink_ou = 0.0
        if shrink_ou > 0 and '2.5' in markets['OU']:
            pO = markets['OU']['2.5']['Over']
            pU = markets['OU']['2.5']['Under']
            pO = (1.0 - shrink_ou) * pO + shrink_ou * 0.5
            pU = 1.0 - pO
            markets['OU']['2.5'] = {'Over': float(pO), 'Under': float(pU)}
        # Flatten to rows
        # 1X2
        for k, v in markets["1X2"].items():
            rows.append({"date": date_val, "home": home, "away": away, "market": "1X2", "outcome": k, "prob": v})
        # Double Chance
        for k, v in markets["DC"].items():
            rows.append({"date": date_val, "home": home, "away": away, "market": "DC", "outcome": k, "prob": v})
        # Over/Under
        for line, kv in markets["OU"].items():
            rows.append({"date": date_val, "home": home, "away": away, "market": f"OU {line}", "outcome": "Under", "prob": kv["Under"]})
            rows.append({"date": date_val, "home": home, "away": away, "market": f"OU {line}", "outcome": "Over", "prob": kv["Over"]})
        # Intervals
        for iv, pv in markets["Intervals"].items():
            rows.append({"date": date_val, "home": home, "away": away, "market": "TG Interval", "outcome": iv, "prob": pv})

    return pd.DataFrame(rows)


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


def attach_value_metrics(market_df: pd.DataFrame, use_placeholders: bool = True) -> pd.DataFrame:
    """Attach odds, normalized implied probability, edge and EV to market probabilities.

    Expects columns: date, home, away, market, outcome, prob
    Returns a new DataFrame with added columns: odds, p_imp, p_imp_norm, edge, EV
    """
    if market_df.empty:
        return market_df.copy()
    df = market_df.copy()
    if use_placeholders:
        df["odds"] = [
            _placeholder_odds(m, o) for m, o in zip(df["market"].astype(str), df["outcome"].astype(str))
        ]
    # Group by market per match for normalization
    grouped = []
    for (date, home, away, market), sub in df.groupby(["date", "home", "away", "market"], dropna=False):
        sub2 = sub.copy()
        # implied prob from odds
        sub2["p_imp"] = 1.0 / sub2["odds"].astype(float)
        if _market_is_exclusive(market):
            s = sub2["p_imp"].sum()
            sub2["p_imp_norm"] = sub2["p_imp"] / s if s > 0 else sub2["p_imp"]
        else:
            # For overlapping outcomes, keep p_imp_norm equal to p_imp (no overround normalization)
            sub2["p_imp_norm"] = sub2["p_imp"]
        # Edge and EV
        sub2["edge"] = sub2["prob"] - sub2["p_imp_norm"]
        sub2["EV"] = sub2["prob"] * sub2["odds"] - 1.0
        grouped.append(sub2)
    out = pd.concat(grouped, ignore_index=True) if grouped else df
    return out


# --- Thresholds, odds filling, and CLI ---
def _market_base(m: str) -> str:
    return m.split()[0] if m.startswith("OU ") else m


def _get_thresholds(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    defaults = {
        '1X2': {'min_prob': 0.55, 'min_edge': 0.03},
        'DC': {'min_prob': 0.75, 'min_edge': 0.02},
        'OU': {'min_prob': 0.58, 'min_edge': 0.02},
        'TG Interval': {'min_prob': 0.30, 'min_edge': 0.05},
    }
    user = cfg.get('thresholds', {}) or {}
    for k, v in user.items():
        if k in defaults and isinstance(v, dict):
            for kk in ('min_prob','min_edge'):
                if kk in v:
                    defaults[k][kk] = float(v[kk])
    return defaults


def _fill_odds_for_df(df: pd.DataFrame, league: str, with_odds: bool) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out['odds'] = None
    for (date, home, away), sub_idx in out.groupby(['date','home','away']).groups.items():
        idx = list(sub_idx)
        # Pre-fetch 1X2 when --with-odds
        if with_odds:
            try:
                odds1x2 = bookmaker_api.get_odds(league, home, away)
            except Exception:
                odds1x2 = {}
        else:
            odds1x2 = {}
        for i in idx:
            m = str(out.at[i,'market'])
            o = str(out.at[i,'outcome'])
            if m == '1X2' and with_odds and odds1x2:
                if o == 'H' and odds1x2.get('home'):
                    out.at[i,'odds'] = float(odds1x2['home'])
                elif o == 'D' and odds1x2.get('draw'):
                    out.at[i,'odds'] = float(odds1x2['draw'])
                elif o == 'A' and odds1x2.get('away'):
                    out.at[i,'odds'] = float(odds1x2['away'])
            elif m.startswith('OU ') and with_odds:
                try:
                    line = float(m.split(' ',1)[1])
                    ou_odds = bookmaker_api.get_odds_ou(league, home, away, line)
                except Exception:
                    ou_odds = {}
                if o in ('Over','Under') and ou_odds.get(o):
                    out.at[i,'odds'] = float(ou_odds[o])
            elif m == 'TG Interval' and with_odds:
                try:
                    a_str, b_str = o.split('-')
                    odd_val = bookmaker_api.get_odds_interval(league, home, away, int(a_str), int(b_str))
                    if odd_val:
                        out.at[i,'odds'] = float(odd_val)
                except Exception:
                    pass
            if out.at[i,'odds'] is None:
                out.at[i,'odds'] = _placeholder_odds(m, o)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Hybrid DC+xG market engine with value metrics")
    ap.add_argument('--with-odds', action='store_true', help='Fetch 1X2 odds from bookmaker API (network needed)')
    ap.add_argument('--top', type=int, default=20, help='Show top N picks by EV')
    args = ap.parse_args()

    cfg = load_config()
    print("\n--- Hybrid Engine Started ---")
    print(cfg)

    market_df = generate_market_book(cfg)
    if market_df.empty:
        print("No fixtures available.")
        return

    df_with_odds = _fill_odds_for_df(market_df, cfg.get('league','E0'), with_odds=args.with_odds)
    df_val = attach_value_metrics(df_with_odds, use_placeholders=False)

    thresholds = _get_thresholds(cfg)
    def pass_threshold(row):
        base = _market_base(str(row['market']))
        th = thresholds.get(base, thresholds.get('1X2'))
        return (float(row['prob']) >= th['min_prob']) and (float(row['edge']) >= th['min_edge'])

    picks = df_val[df_val.apply(pass_threshold, axis=1)].copy()
    if picks.empty:
        print("No value picks under current thresholds.")
        return

    picks.sort_values(['EV','prob'], ascending=[False, False], inplace=True)
    show = picks.head(args.top)

    # Export full ranked picks to CSV
    try:
        ts = pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')
        rep_dir = Path('reports'); rep_dir.mkdir(parents=True, exist_ok=True)
        out_csv = rep_dir / f"{cfg.get('league','E0')}_{ts}_picks.csv"
        cols = ['date','home','away','market','outcome','prob','odds','edge','EV']
        picks.to_csv(out_csv, index=False, columns=[c for c in cols if c in picks.columns])
        print(f"Saved picks CSV -> {out_csv}")
    except Exception as e:
        print(f"Could not save picks CSV: {e}")

    print("\n--- Top Picks (by EV) ---")
    for _, r in show.iterrows():
        print(f"{r['date']}  {r['home']} vs {r['away']} | {r['market']} {r['outcome']}  p={float(r['prob']):.2%}  odds={float(r['odds']):.2f}  edge={float(r['edge']):.2%}  EV={float(r['EV']):.2f}")

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
            if stake <= 0:
                continue
            bm.log_bet(
                date=str(r['date']),
                league=cfg.get('league','E0'),
                home=str(r['home']),
                away=str(r['away']),
                market=f"{r['market']} {r['outcome']}",
                odds=O,
                stake=stake,
                prob=p,
                ev=ev
            )
    else:
        print("Print-only mode: set log_bets=true to record stakes.")


if __name__ == "__main__":
    main()
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
