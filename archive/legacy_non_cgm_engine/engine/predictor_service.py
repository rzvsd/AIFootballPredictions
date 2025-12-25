"""
Predictor service: assemble fixtures, snapshots, and score matrices.

Returns a list of per-fixture predictions with score matrices (P) that other
services can turn into markets/odds/EV.
"""
from __future__ import annotations

import os
import math
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from xgboost import XGBRegressor

import config
from data_pipeline import feature_store
from models import xgb_trainer
from engine import odds_service
from scripts.fetch_fixtures_understat import load_understat_fixtures


DEFAULT_OU_LINES = [0.5, 1.5, 2.5, 3.5]
DEFAULT_INTERVALS = [(0, 3), (1, 3), (2, 4), (2, 5), (3, 6)]


def _load_understat_fixture_frame(league_code: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    try:
        cfg_days = int(cfg.get("fixtures_days", 10))
    except Exception:
        cfg_days = 10
    try:
        env_days = int(os.getenv("BOT_FIXTURES_DAYS", cfg_days))
    except Exception:
        env_days = cfg_days
    days = env_days if env_days > 0 else cfg_days
    season_cfg = cfg.get("understat_season")
    try:
        season_env = int(os.getenv("BOT_UNDERSTAT_SEASON", "0"))
    except Exception:
        season_env = 0
    season_val = season_env if season_env > 0 else (season_cfg if isinstance(season_cfg, int) else None)
    try:
        df = load_understat_fixtures(league_code, days=days, season=season_val)
    except Exception as e:
        print(f"[{league_code}] Understat fetch failed: {e} (season={season_val})")
        # Retry without forcing a future season (or try previous season) to avoid NoneType.group crashes
        fallback_season = (season_val - 1) if season_val else None
        try:
            df = load_understat_fixtures(league_code, days=days, season=fallback_season)
            print(f"[{league_code}] Understat retry succeeded with season={fallback_season or 'auto'} (rows={len(df)})")
        except Exception as e2:
            print(f"[{league_code}] Understat retry failed: {e2}")
            df = pd.DataFrame(columns=["date", "home_team_api", "away_team_api"])
    if df.empty:
        return pd.DataFrame(columns=["date", "home_team_api", "away_team_api"])
    if "home" in df.columns:
        df = df.rename(columns={"home": "home_team_api", "away": "away_team_api"})
    wanted_cols = ["date", "home_team_api", "away_team_api"]
    for col in wanted_cols:
        if col not in df.columns:
            df[col] = None
    return df[wanted_cols]


def _fixtures_from_odds(league_code: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Build fixtures frame directly from odds JSON (names aligned with odds feed)."""
    path = Path("data") / "odds" / f"{league_code}.json"
    # If a CGM-derived odds file is present, use it transparently (same path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame(columns=["date", "home_team_api", "away_team_api"])
    fixtures = data.get("fixtures") or []
    if not fixtures:
        return pd.DataFrame(columns=["date", "home_team_api", "away_team_api"])
    df = pd.DataFrame(
        [
            {
                "date": fx.get("date"),
                "home_team_api": fx.get("home"),
                "away_team_api": fx.get("away"),
            }
    for fx in fixtures
        ]
    )
    # Horizon filter
    try:
        days_cfg = int(cfg.get("fixtures_days", 10))
    except Exception:
        days_cfg = 10
    try:
        env_days = int(os.getenv("BOT_FIXTURES_DAYS", days_cfg))
    except Exception:
        env_days = days_cfg
    days = env_days if env_days > 0 else days_cfg
    if "date" in df.columns:
        try:
            df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
            now = pd.Timestamp.utcnow()
            horizon = now + pd.Timedelta(days=days)
            df = df[(df["date_dt"] >= now) & (df["date_dt"] <= horizon)]
            df = df.drop(columns=["date_dt"])
        except Exception:
            pass
    df = df.drop_duplicates(subset=["date", "home_team_api", "away_team_api"])
    wanted_cols = ["date", "home_team_api", "away_team_api"]
    for col in wanted_cols:
        if col not in df.columns:
            df[col] = None
    return df[wanted_cols]


def _load_model(json_path: str, pkl_path: str):
    try:
        if os.path.exists(json_path):
            m = XGBRegressor()
            m.load_model(json_path)
            return m
    except Exception:
        pass
    return joblib.load(pkl_path)


def _predict_xgb(model, feat_df: pd.DataFrame) -> float:
    """Predict with XGB model using model feature names if available to avoid NaNs from column mismatch."""
    try:
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            try:
                names = model.get_booster().feature_names
            except Exception:
                names = None
        if names is not None:
            cols = [c for c in names if c in feat_df.columns]
            X = feat_df[cols].values
        else:
            X = feat_df.values
        y = model.predict(X)
        return float(y[0] if hasattr(y, "__len__") else y)
    except Exception:
        try:
            n = int(getattr(model, "n_features_in_", 0))
            if n > 0:
                X = feat_df.iloc[:, :n].values
                y = model.predict(X)
                return float(y[0] if hasattr(y, "__len__") else y)
        except Exception:
            pass
        try:
            y = model.predict(feat_df.values)
            return float(y[0] if hasattr(y, "__len__") else y)
        except Exception:
            return float("nan")


def _feature_row_from_snapshot(stats: pd.DataFrame, home: str, away: str, expected_features: list[str] | None = None) -> Optional[pd.DataFrame]:
    hs = stats.loc[stats["team"] == home]
    as_ = stats.loc[stats["team"] == away]
    med = stats.median(numeric_only=True) if not stats.empty else {}

    def _pick(df: pd.DataFrame, names: list[str], default_med_keys: list[str], hard_default: float) -> float:
        """Safely pick the first available column value; fall back to medians/hard default."""
        for n in names:
            if n in df.columns:
                try:
                    val = df[n].iloc[0]
                    if pd.notna(val):
                        return float(val)
                except Exception:
                    continue
        for mk in default_med_keys:
            try:
                if mk in med and pd.notna(med[mk]):
                    return float(med[mk])
            except Exception:
                continue
        return float(hard_default)

    if hs.empty or as_.empty:
        try:
            def make_side(prefix: str) -> pd.DataFrame:
                row = {
                    "xg_home_EWMA": float(med.get("xg_home_EWMA", med.get("xg_L5", 1.4))),
                    "xga_home_EWMA": float(med.get("xga_home_EWMA", med.get("xga_L5", 1.2))),
                    "xg_away_EWMA": float(med.get("xg_away_EWMA", med.get("xg_L5", 1.2))),
                    "xga_away_EWMA": float(med.get("xga_away_EWMA", med.get("xga_L5", 1.2))),
                    "ppg_home_EWMA": float(med.get("ppg_home_EWMA", med.get("gpg_L10", 1.5))),
                    "ppg_away_EWMA": float(med.get("ppg_away_EWMA", med.get("gpg_L10", 1.3))),
                    "corners_L10": float(med.get("corners_L10", 0.0)),
                    "corners_allowed_L10": float(med.get("corners_allowed_L10", 0.0)),
                    "elo": float(stats["elo"].mean()) if "elo" in stats.columns else 1500.0,
                    "GFvsMid_H": float(med.get("GFvsMid_H", med.get("xg_L5", 1.3))),
                    "GAvsMid_H": float(med.get("GAvsMid_H", med.get("xga_L5", 1.2))),
                    "GFvsHigh_H": float(med.get("GFvsHigh_H", med.get("xg_L5", 1.1))),
                    "GAvsHigh_H": float(med.get("GAvsHigh_H", med.get("xga_L5", 1.3))),
                    "GFvsMid_A": float(med.get("GFvsMid_A", med.get("xg_L5", 1.1))),
                    "GAvsMid_A": float(med.get("GAvsMid_A", med.get("xga_L5", 1.3))),
                    "GFvsHigh_A": float(med.get("GFvsHigh_A", med.get("xg_L5", 1.2))),
                    "GAvsHigh_A": float(med.get("GAvsHigh_A", med.get("xga_L5", 1.4))),
                    "Elo_H": float(med.get("elo", 1500.0)),
                    "Elo_A": float(med.get("elo", 1500.0)),
                }
                return pd.DataFrame({k: [v] for k, v in row.items()})

            hs = make_side("H")
            as_ = make_side("A")
        except Exception:
            return None
    hs = hs.reset_index(drop=True)
    as_ = as_.reset_index(drop=True)
    row = {
        "xg_home_EWMA": _pick(hs, ["xg_home_EWMA", "xg_L5", "xg_home"], ["xg_home_EWMA", "xg_L5"], 1.4),
        "xga_home_EWMA": _pick(hs, ["xga_home_EWMA", "xga_L5", "xga_home"], ["xga_home_EWMA", "xga_L5"], 1.2),
        "xg_away_EWMA": _pick(as_, ["xg_away_EWMA", "xg_L5", "xg_away"], ["xg_away_EWMA", "xg_L5"], 1.2),
        "xga_away_EWMA": _pick(as_, ["xga_away_EWMA", "xga_L5", "xga_away"], ["xga_away_EWMA", "xga_L5"], 1.2),
        "ppg_home_EWMA": _pick(hs, ["ppg_home_EWMA", "gpg_L10", "ppg_home"], ["ppg_home_EWMA", "gpg_L10"], 1.5),
        "ppg_away_EWMA": _pick(as_, ["ppg_away_EWMA", "gpg_L10", "ppg_away"], ["ppg_away_EWMA", "gpg_L10"], 1.3),
        "corners_L10_H": _pick(hs, ["corners_L10", "corners_for_L10"], ["corners_L10"], 0.0),
        "corners_L10_A": _pick(as_, ["corners_L10", "corners_for_L10"], ["corners_L10"], 0.0),
        "corners_allowed_L10_H": _pick(hs, ["corners_allowed_L10", "corners_against_L10"], ["corners_allowed_L10"], 0.0),
        "corners_allowed_L10_A": _pick(as_, ["corners_allowed_L10", "corners_against_L10"], ["corners_allowed_L10"], 0.0),
        "GFvsMid_H": _pick(hs, ["GFvsMid_H", "xg_L5"], ["GFvsMid_H", "xg_L5"], 1.3),
        "GAvsMid_H": _pick(hs, ["GAvsMid_H", "xga_L5"], ["GAvsMid_H", "xga_L5"], 1.2),
        "GFvsHigh_H": _pick(hs, ["GFvsHigh_H", "xg_L5"], ["GFvsHigh_H", "xg_L5"], 1.1),
        "GAvsHigh_H": _pick(hs, ["GAvsHigh_H", "xga_L5"], ["GAvsHigh_H", "xga_L5"], 1.3),
        "GFvsMid_A": _pick(as_, ["GFvsMid_A", "xg_L5"], ["GFvsMid_A", "xg_L5"], 1.1),
        "GAvsMid_A": _pick(as_, ["GAvsMid_A", "xga_L5"], ["GAvsMid_A", "xga_L5"], 1.3),
        "GFvsHigh_A": _pick(as_, ["GFvsHigh_A", "xg_L5"], ["GFvsHigh_A", "xg_L5"], 1.2),
        "GAvsHigh_A": _pick(as_, ["GAvsHigh_A", "xga_L5"], ["GAvsHigh_A", "xga_L5"], 1.4),
        "Elo_H": _pick(hs, ["elo"], ["elo"], 1500.0),
        "Elo_A": _pick(as_, ["elo"], ["elo"], 1500.0),
    }
    feat_df = pd.DataFrame({k: [v] for k, v in row.items()})

    if expected_features:
        # Fill missing expected features with medians (if present) or zeros to avoid NaNs in XGB input
        med_map = med.to_dict() if hasattr(med, "to_dict") else {}
        for f in expected_features:
            if f not in feat_df.columns:
                val = med_map.get(f, 0.0)
                if pd.isna(val):
                    val = 0.0
                feat_df[f] = [val]
        feat_df = feat_df.reindex(columns=expected_features, fill_value=0.0)
    return feat_df
def _score_matrix(mu_h: float, mu_a: float, max_goals: int = 10, trim_epsilon: float = 0.0) -> np.ndarray:
    lam_h = max(0.05, mu_h)
    lam_a = max(0.05, mu_a)
    P = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            P[i, j] = (math.exp(-lam_h) * lam_h**i / math.factorial(i)) * (
                math.exp(-lam_a) * lam_a**j / math.factorial(j)
            )
    if trim_epsilon > 0:
        P = np.clip(P, trim_epsilon, None)
    P = P / P.sum()
    return P


def _score_matrix_negbin(mu_h: float, mu_a: float, k_h: float = 6.0, k_a: float = 6.0, max_goals: int = 10) -> np.ndarray:
    import scipy.special as sc

    def nb_pmf(x, mu, k):
        p = k / (k + mu)
        return math.exp(sc.gammaln(k + x) - sc.gammaln(k) - sc.gammaln(x + 1) + k * math.log(p) + x * math.log(1 - p))

    lam_h = max(0.05, mu_h)
    lam_a = max(0.05, mu_a)
    P = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            P[i, j] = nb_pmf(i, lam_h, k_h) * nb_pmf(j, lam_a, k_a)
    P = P / P.sum()
    return P


def _score_from_cfg(mu_h: float, mu_a: float, cfg: Dict[str, Any], max_goals: int, league: str) -> np.ndarray:
    od = (cfg.get("overdispersion") or {})
    method = str(od.get("method", "poisson")).lower()
    if method == "negbin":
        k = float(od.get("k", 6.0))
        return _score_matrix_negbin(mu_h, mu_a, k_h=k, k_a=k, max_goals=max_goals)
    return _score_matrix(mu_h, mu_a, max_goals=max_goals, trim_epsilon=float(cfg.get("tail_epsilon", 0.0)))


def _blend_mu(cfg: Dict[str, Any], home: str, away: str, mu_h_macro: float, mu_a_macro: float, snap: pd.DataFrame) -> tuple[float, float, str]:
    src = str(cfg.get("xg_source", "macro")).lower()
    if src not in ("macro", "micro", "blend"):
        src = "macro"
    micro_path = cfg.get("micro_agg_path", os.path.join("data", "enhanced", "micro_agg.csv"))
    micro = {}
    if src in ("micro", "blend") and micro_path:
        try:
            df = pd.read_csv(micro_path)
            df["team"] = df["team"].astype(str).map(config.normalize_team_name)
            df["side"] = df["side"].astype(str)
            df = df.sort_values(["team", "side", "date"])
            df = df.groupby(["team", "side"], as_index=False).tail(1)
            micro = {(r["team"], str(r["side"]).upper()[:1]): float(r.get("xg_for_EWMA", r.get("xg_for", 0.0)) or 0.0) for _, r in df.iterrows()}
        except Exception:
            micro = {}
    mu_h_micro = micro.get((home, "H"))
    mu_a_micro = micro.get((away, "A"))
    if src == "micro":
        if mu_h_micro is not None and mu_a_micro is not None:
            return float(mu_h_micro), float(mu_a_micro), "micro"
        return mu_h_macro, mu_a_macro, "macro"
    if src == "blend":
        try:
            w = float(cfg.get("xg_blend_weight", 0.5))
        except Exception:
            w = 0.5
        if mu_h_micro is not None:
            mu_h_macro = (1.0 - w) * mu_h_macro + w * float(mu_h_micro)
        if mu_a_micro is not None:
            mu_a_macro = (1.0 - w) * mu_a_macro + w * float(mu_a_micro)
        return mu_h_macro, mu_a_macro, "blend"
    return mu_h_macro, mu_a_macro, "macro"


def _predict_goals_mu(cfg: Dict[str, Any], league: str, feat_row: pd.DataFrame, xgb_home_model, xgb_away_model) -> tuple[float, float, str]:
    pm = (cfg.get("prob_model") or {})
    enabled = bool(pm.get("enabled", False))
    kind = str(pm.get("kind", "")).lower()
    if enabled and kind in ("ngboost_poisson", "ngb_poisson", "ngb_pois"):
        return _predict_xgb(xgb_home_model, feat_row), _predict_xgb(xgb_away_model, feat_row), "xgb"
    return _predict_xgb(xgb_home_model, feat_row), _predict_xgb(xgb_away_model, feat_row), "xgb"


def get_predictions(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of fixture predictions with score matrices."""
    league = cfg.get("league", "E0")
    max_goals = int(cfg.get("max_goals", config.MAX_GOALS_PER_LEAGUE.get(league, 10)))

    # Fixture source selection: in LIVE mode prefer odds-based fixtures to ensure odds alignment.
    mode = str(cfg.get("mode", os.getenv("BOT_MODE", "sim"))).lower()
    fixture_source = str(cfg.get("fixture_source", "")).lower()
    if fixture_source == "odds" or mode == "live":
        fixtures = _fixtures_from_odds(league, cfg)
    else:
        fixtures = _load_understat_fixture_frame(league, cfg)
    if fixtures.empty:
        return []

    # Snapshot cutoff
    enh_path = os.path.join("data", "enhanced", f"{league}_final_features.csv")
    as_of_env = os.getenv("BOT_SNAPSHOT_AS_OF") or os.getenv("BOT_FIXTURES_FROM")
    earliest_fixture = None
    try:
        if "date" in fixtures.columns:
            earliest_fixture = pd.to_datetime(fixtures["date"], errors="coerce").min()
    except Exception:
        earliest_fixture = None
    try:
        if as_of_env:
            as_of_val = str(pd.to_datetime(as_of_env, errors="coerce"))
        elif pd.notna(earliest_fixture):
            as_of_val = (earliest_fixture - pd.Timedelta(seconds=1)).isoformat()
        else:
            as_of_val = None
    except Exception:
        as_of_val = as_of_env or None

    # Fallback to processed if enhanced snapshot missing
    snap_path = enh_path if os.path.exists(enh_path) else os.path.join("data", "processed", f"{league}_merged_preprocessed.csv")
    snap = feature_store.build_snapshot(
        enhanced_csv=snap_path,
        as_of=as_of_val,
        half_life_matches=int(cfg.get("half_life_matches", 5)),
        elo_k=float(cfg.get("elo_k", 20.0)),
        elo_home_adv=float(cfg.get("elo_home_adv", 60.0)),
        micro_agg_path=(cfg.get("micro_agg_path") or os.path.join("data", "enhanced", f"{league}_micro_agg.csv")),
    )

    # Models
    home_json = os.path.join("advanced_models", f"{league}_ultimate_xgb_home.json")
    away_json = os.path.join("advanced_models", f"{league}_ultimate_xgb_away.json")
    home_pkl = os.path.join("advanced_models", f"{league}_ultimate_xgb_home.pkl")
    away_pkl = os.path.join("advanced_models", f"{league}_ultimate_xgb_away.pkl")
    xgb_home = _load_model(home_json, home_pkl)
    xgb_away = _load_model(away_json, away_pkl)
    # Expected feature names (intersection of both models for safety)
    def _feat_names(model):
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            try:
                names = model.get_booster().feature_names
            except Exception:
                names = []
        try:
            return list(names)
        except Exception:
            try:
                return list(names.tolist())
            except Exception:
                return []
    feats_home = _feat_names(xgb_home)
    feats_away = _feat_names(xgb_away)
    expected_feats = list(dict.fromkeys([*feats_home, *feats_away]))  # preserve order, union

    preds: List[Dict[str, Any]] = []
    for _, r in fixtures.iterrows():
        home_api = str(r.get("home_team_api", r.get("home", ""))).strip()
        away_api = str(r.get("away_team_api", r.get("away", ""))).strip()
        date_val = r.get("date")
        home = config.normalize_team_name(home_api)
        away = config.normalize_team_name(away_api)

        feat_row = _feature_row_from_snapshot(snap, home, away, expected_features=expected_feats if expected_feats else None)
        if feat_row is None:
            continue
        mu_h, mu_a, _ = _predict_goals_mu(cfg, league, feat_row, xgb_home, xgb_away)
        if not np.isfinite(mu_h) or not np.isfinite(mu_a):
            # If either side is NaN, skip fixture rather than emitting degenerate probabilities.
            continue
        mu_h_final, mu_a_final, _ = _blend_mu(cfg, home, away, mu_h, mu_a, snap)
        P = _score_from_cfg(mu_h_final, mu_a_final, cfg, max_goals=max_goals, league=league)
        preds.append({"date": date_val, "home": home, "away": away, "P": P, "mu_h": mu_h_final, "mu_a": mu_a_final})
    return preds

