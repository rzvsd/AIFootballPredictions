"""
Recompute Elo time series from match history (ignores any Elo stored in the CSV).

Inputs:
- data/enhanced/cgm_match_history.csv (or --history)
  Required columns: date, home, away, ft_home, ft_away, code_home/code_away (optional but preferred)

Outputs:
- data/enhanced/cgm_match_history_with_elo.csv (or --out)
  Adds/overwrites: elo_home_calc, elo_away_calc, EloDiff_calc, Band_H_calc, Band_A_calc

Elo V2:
- league-aware K/home-advantage from config.ELO_LEAGUE_PARAMS
- match-type K multipliers from config.ELO_MATCHTYPE_MULTIPLIERS
- upset and new-team K multipliers
- capped goal-difference multiplier
- per-row trace fields (elo_*_used)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore

try:
    from cgm.team_registry import build_team_registry
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cgm"))
    from team_registry import build_team_registry  # type: ignore


START_ELO_DEFAULT = 1500.0
K_FACTOR_DEFAULT = 20.0
HOME_ADV_DEFAULT = 65.0
BAND_THRESH_DEFAULT = 150.0
LOG_PATH_DEFAULT = Path("reports/run_log.jsonl")


def _cfg_map(name: str, default: dict | None = None) -> dict:
    if default is None:
        default = {}
    if config is None:
        return dict(default)
    val = getattr(config, name, default)
    if isinstance(val, dict):
        return dict(val)
    return dict(default)


def _cfg_bool(name: str, default: bool) -> bool:
    if config is None:
        return bool(default)
    return bool(getattr(config, name, default))


def _safe_float(value, fallback: float) -> float:
    try:
        v = float(value)
        if np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return float(fallback)


def margin_multiplier(goal_diff: int, cap: float | None = None) -> float:
    """Football-style margin multiplier with optional cap."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        mult = 1.0
    elif gd == 2:
        mult = 1.5
    else:
        mult = (11.0 + float(gd)) / 8.0
    if cap is not None:
        mult = min(float(mult), float(cap))
    return float(mult)


def expected_home(r_home: float, r_away: float, home_adv: float) -> float:
    diff = (r_home + home_adv) - r_away
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _clean_team_id(val) -> str | None:
    """Normalize team identifiers (codes/names) and treat empty/'nan' as missing."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    try:
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                return None
            if float(val).is_integer():
                return str(int(val))
            return str(float(val))
    except Exception:
        pass
    s = str(val).strip()
    if not s:
        return None
    sl = s.lower()
    if sl in {"nan", "none", "null"}:
        return None
    if sl.endswith(".0") and sl[:-2].isdigit():
        s = s[:-2]
    return s


def infer_team(row: pd.Series, home: bool) -> str | None:
    code = row.get("code_home") if home else row.get("code_away")
    name = row.get("home") if home else row.get("away")
    code_id = _clean_team_id(code)
    if code_id:
        return code_id
    name_id = _clean_team_id(name)
    if name_id:
        return name_id
    return None


def _norm_key(val: object) -> str:
    return str(val or "").strip().lower()


def _infer_match_type(row: pd.Series) -> str:
    candidates = [
        row.get("match_type"),
        row.get("competition_type"),
        row.get("fixture_type"),
        row.get("stage"),
    ]
    for raw in candidates:
        s = str(raw or "").strip().lower()
        if not s:
            continue
        if "friend" in s or "amical" in s:
            return "friendly"
        if "playoff" in s or "play-off" in s or "promotion" in s or "relegation" in s:
            return "playoff"
        if "cup" in s or "copa" in s or "coupe" in s or "pokal" in s or "cupa" in s:
            return "cup"
        if "league" in s or "liga" in s or "division" in s:
            return "league"
        return "unknown"

    lg = str(row.get("league") or "").strip().lower()
    if any(tok in lg for tok in ["cup", "copa", "coupe", "pokal", "cupa"]):
        return "cup"
    return "league" if lg else "unknown"


def load_history(path: Path, max_date: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt_existing = (
        pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(None)
        if "datetime" in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    dt_from_parts = pd.Series(pd.NaT, index=df.index)
    if "date" in df.columns:
        dt_combo = pd.to_datetime(
            df["date"].astype(str) + " " + df.get("time", "").astype(str), errors="coerce", utc=True
        ).dt.tz_convert(None)
        dt_date_only = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        dt_from_parts = dt_combo.fillna(dt_date_only)
    df["datetime"] = dt_existing.fillna(dt_from_parts)
    df = df.sort_values("datetime")
    if max_date:
        cutoff = pd.to_datetime(max_date, utc=True).tz_convert(None)
        df = df[df["datetime"] <= cutoff]
    return df


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def log_json(event: dict, log_path: Path = LOG_PATH_DEFAULT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event_out = event.copy()
    event_out.setdefault("ts", pd.Timestamp.utcnow().isoformat())
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_out) + "\n")


def compute_elo_series(
    df: pd.DataFrame,
    start_elo: float,
    k_factor: float,
    home_adv: float,
) -> pd.DataFrame:
    defaults = _cfg_map("ELO_DEFAULTS", {})
    league_params = _cfg_map("ELO_LEAGUE_PARAMS", {})
    matchtype_mults = _cfg_map("ELO_MATCHTYPE_MULTIPLIERS", {})
    elo_v2_enabled = _cfg_bool("ELO_V2_ENABLED", True)

    margin_cap = _safe_float(defaults.get("margin_cap", 2.75), 2.75)
    upset_expected_low = _safe_float(defaults.get("upset_expected_low", 0.35), 0.35)
    upset_expected_high = _safe_float(defaults.get("upset_expected_high", 0.65), 0.65)
    upset_multiplier = _safe_float(defaults.get("upset_multiplier", 1.20), 1.20)
    new_team_games = int(_safe_float(defaults.get("new_team_games", 12), 12.0))
    new_team_k_multiplier = _safe_float(defaults.get("new_team_k_multiplier", 1.25), 1.25)
    rating_floor = _safe_float(defaults.get("rating_floor", 1000.0), 1000.0)

    # CLI keeps final control over base defaults.
    start_elo = _safe_float(start_elo, _safe_float(defaults.get("start_elo", start_elo), start_elo))
    k_factor = _safe_float(k_factor, _safe_float(defaults.get("k_factor", k_factor), k_factor))
    home_adv = _safe_float(home_adv, _safe_float(defaults.get("home_adv", home_adv), home_adv))

    league_params_ci = {_norm_key(k): v for k, v in league_params.items() if isinstance(v, dict)}
    mt_ci = {_norm_key(k): _safe_float(v, 1.0) for k, v in matchtype_mults.items()}

    ratings: Dict[str, float] = {}
    team_games: Dict[str, int] = {}

    elo_home_list = []
    elo_away_list = []
    elo_hfa_used = []
    elo_k_base_used = []
    elo_k_matchtype_mult = []
    elo_k_newteam_mult = []
    elo_k_upset_mult = []
    elo_k_used = []
    elo_g_used = []
    elo_expected_home = []
    elo_actual_home = []
    elo_delta = []
    elo_match_type = []

    for _, row in df.iterrows():
        home_id = infer_team(row, home=True)
        away_id = infer_team(row, home=False)

        fh_raw = row.get("ft_home")
        fa_raw = row.get("ft_away")
        try:
            fh = float(fh_raw)
        except Exception:
            fh = np.nan
        try:
            fa = float(fa_raw)
        except Exception:
            fa = np.nan

        r_home = ratings.get(home_id, start_elo) if home_id else start_elo
        r_away = ratings.get(away_id, start_elo) if away_id else start_elo

        elo_home_list.append(float(r_home))
        elo_away_list.append(float(r_away))

        league_name = str(row.get("league") or "").strip()
        league_cfg = league_params_ci.get(_norm_key(league_name), {}) if elo_v2_enabled else {}
        hfa_used = _safe_float(league_cfg.get("home_adv", home_adv), home_adv)
        k_base_used = _safe_float(league_cfg.get("k_factor", k_factor), k_factor)

        match_type = _infer_match_type(row)
        match_mult = _safe_float(mt_ci.get(_norm_key(match_type), mt_ci.get("unknown", 1.0)), 1.0)

        games_home = int(team_games.get(home_id, 0)) if home_id else 0
        games_away = int(team_games.get(away_id, 0)) if away_id else 0
        min_games = min(games_home, games_away)
        new_mult = float(new_team_k_multiplier) if min_games < new_team_games else 1.0

        # Pre-calc defaults for trace columns.
        exp_home = np.nan
        actual = np.nan
        g_mult = np.nan
        upset_mult = 1.0
        k_used = np.nan
        delta = np.nan

        if not (pd.isna(fh) or pd.isna(fa) or home_id is None or away_id is None):
            exp_home = expected_home(r_home, r_away, hfa_used)
            if fh > fa:
                actual = 1.0
            elif fh == fa:
                actual = 0.5
            else:
                actual = 0.0

            if elo_v2_enabled:
                if (actual == 1.0 and exp_home < upset_expected_low) or (actual == 0.0 and exp_home > upset_expected_high):
                    upset_mult = float(upset_multiplier)

            gd = int(abs(fh - fa))
            g_mult = margin_multiplier(gd, cap=margin_cap if elo_v2_enabled else None)

            k_used = float(k_base_used * match_mult * new_mult * upset_mult)
            delta = float(k_used * g_mult * (actual - exp_home))

            ratings[home_id] = max(rating_floor, float(r_home + delta))
            ratings[away_id] = max(rating_floor, float(r_away - delta))
            team_games[home_id] = games_home + 1
            team_games[away_id] = games_away + 1

        elo_hfa_used.append(float(hfa_used))
        elo_k_base_used.append(float(k_base_used))
        elo_k_matchtype_mult.append(float(match_mult))
        elo_k_newteam_mult.append(float(new_mult))
        elo_k_upset_mult.append(float(upset_mult))
        elo_k_used.append(float(k_used) if np.isfinite(k_used) else np.nan)
        elo_g_used.append(float(g_mult) if np.isfinite(g_mult) else np.nan)
        elo_expected_home.append(float(exp_home) if np.isfinite(exp_home) else np.nan)
        elo_actual_home.append(float(actual) if np.isfinite(actual) else np.nan)
        elo_delta.append(float(delta) if np.isfinite(delta) else np.nan)
        elo_match_type.append(str(match_type))

    return pd.DataFrame(
        {
            "elo_home_calc": pd.Series(elo_home_list, index=df.index),
            "elo_away_calc": pd.Series(elo_away_list, index=df.index),
            "elo_hfa_used": pd.Series(elo_hfa_used, index=df.index),
            "elo_k_base_used": pd.Series(elo_k_base_used, index=df.index),
            "elo_k_matchtype_mult": pd.Series(elo_k_matchtype_mult, index=df.index),
            "elo_k_newteam_mult": pd.Series(elo_k_newteam_mult, index=df.index),
            "elo_k_upset_mult": pd.Series(elo_k_upset_mult, index=df.index),
            "elo_k_used": pd.Series(elo_k_used, index=df.index),
            "elo_g_used": pd.Series(elo_g_used, index=df.index),
            "elo_expected_home": pd.Series(elo_expected_home, index=df.index),
            "elo_actual_home": pd.Series(elo_actual_home, index=df.index),
            "elo_delta": pd.Series(elo_delta, index=df.index),
            "elo_match_type": pd.Series(elo_match_type, index=df.index),
        }
    )


def band_from_diff(diff: float, thresh: float) -> str:
    if diff >= thresh:
        return "BULLY"
    if diff <= -thresh:
        return "DOG"
    return "PEER"


def main() -> None:
    cfg_defaults = _cfg_map("ELO_DEFAULTS", {})
    cfg_start = _safe_float(cfg_defaults.get("start_elo", START_ELO_DEFAULT), START_ELO_DEFAULT)
    cfg_k = _safe_float(cfg_defaults.get("k_factor", K_FACTOR_DEFAULT), K_FACTOR_DEFAULT)
    cfg_hfa = _safe_float(cfg_defaults.get("home_adv", HOME_ADV_DEFAULT), HOME_ADV_DEFAULT)
    cfg_band = _safe_float(cfg_defaults.get("band_thresh", BAND_THRESH_DEFAULT), BAND_THRESH_DEFAULT)

    ap = argparse.ArgumentParser(description="Recompute Elo time series from match history")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history.csv", help="Input history CSV")
    ap.add_argument("--out", default="data/enhanced/cgm_match_history_with_elo.csv", help="Output CSV path (cleaned)")
    ap.add_argument("--start-elo", type=float, default=cfg_start, help="Starting Elo for new teams")
    ap.add_argument("--k-factor", type=float, default=cfg_k, help="Base K factor")
    ap.add_argument("--home-adv", type=float, default=cfg_hfa, help="Base home advantage points")
    ap.add_argument("--band-thresh", type=float, default=cfg_band, help="Band threshold for BULLY/PEER/DOG")
    ap.add_argument("--data-dir", default="data/api_football", help="Data directory for team registry (id/name mapping)")
    ap.add_argument(
        "--max-date",
        default=None,
        help="Optional ISO date (YYYY-MM-DD); rows after this date are excluded from Elo updates. "
             "If omitted, defaults to today in UTC.",
    )
    ap.add_argument("--log-level", default="INFO", help="Logging level (INFO/DEBUG/WARNING)")
    ap.add_argument("--log-json", default=str(LOG_PATH_DEFAULT), help="Path to JSONL run log")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("calc_cgm_elo")

    max_date = args.max_date or pd.Timestamp.utcnow().normalize().date().isoformat()
    hist_path = Path(args.history)
    hist = load_history(hist_path, max_date=max_date)
    hist_hash = file_hash(hist_path) if hist_path.exists() else None
    logger.info("[ELO][DATA] history=%s rows=%d hash=%s date_range=[%s, %s] cutoff=%s",
                hist_path, len(hist), hist_hash, hist["datetime"].min(), hist["datetime"].max(), max_date)
    log_json(
        {
            "event": "ELO_DATA",
            "history": str(hist_path),
            "history_hash": hist_hash,
            "rows": len(hist),
            "date_min": str(hist["datetime"].min()),
            "date_max": str(hist["datetime"].max()),
            "cutoff": max_date,
        },
        log_path=Path(args.log_json),
    )

    elo_df = compute_elo_series(
        hist,
        start_elo=args.start_elo,
        k_factor=args.k_factor,
        home_adv=args.home_adv,
    )

    hist_out = hist.copy()
    for col in elo_df.columns:
        hist_out[col] = elo_df[col]

    # EloDiff now uses row-level HFA (league-aware in Elo V2).
    hfa_series = pd.to_numeric(hist_out.get("elo_hfa_used", args.home_adv), errors="coerce").fillna(float(args.home_adv))
    hist_out["EloDiff_calc"] = (hist_out["elo_home_calc"] + hfa_series) - hist_out["elo_away_calc"]
    hist_out["Band_H_calc"] = hist_out["EloDiff_calc"].apply(lambda d: band_from_diff(d, args.band_thresh))
    hist_out["Band_A_calc"] = hist_out["EloDiff_calc"].apply(lambda d: band_from_diff(-d, args.band_thresh))

    # Overwrite legacy Elo columns to avoid downstream accidental use.
    hist_out["elo_home"] = hist_out["elo_home_calc"]
    hist_out["elo_away"] = hist_out["elo_away_calc"]
    hist_out["elo_diff"] = hist_out["elo_home"] - hist_out["elo_away"]
    hist_out["EloDiff"] = hist_out["EloDiff_calc"]
    hist_out["Band_H"] = hist_out["Band_H_calc"]
    hist_out["Band_A"] = hist_out["Band_A_calc"]

    # Optional readability: add canonical names via team registry (code -> name)
    try:
        reg = build_team_registry(args.data_dir)
        code_to_name = reg.get("code_to_name", {})
    except Exception:
        code_to_name = {}

    def map_name(row, home: bool) -> str:
        code = row.get("code_home") if home else row.get("code_away")
        fallback = row.get("home") if home else row.get("away")
        code_key = _clean_team_id(code)
        if code_key and code_key in code_to_name:
            return code_to_name[code_key]
        try:
            if pd.notna(fallback):
                return str(fallback)
        except Exception:
            pass
        return str(fallback)

    hist_out["home_name_calc"] = hist_out.apply(lambda r: map_name(r, True), axis=1)
    hist_out["away_name_calc"] = hist_out.apply(lambda r: map_name(r, False), axis=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build cleaned dataframe: drop merge artefacts (_x/_y) while keeping canonical columns
    drop_cols = []
    for col in list(hist_out.columns):
        if col.endswith("_x") or col.endswith("_y"):
            base = col[:-2]
            if base in hist_out.columns:
                drop_cols.append(col)
    cleaned = hist_out.drop(columns=drop_cols)

    cleaned.to_csv(out_path, index=False)

    zero_home = int((hist_out["elo_home_calc"] == 0).sum())
    zero_away = int((hist_out["elo_away_calc"] == 0).sum())
    print(f"[ok] wrote Elo series -> {out_path} (rows={len(hist_out)}, zeros_home={zero_home}, zeros_away={zero_away})")
    logger.info(
        "[ELO][ELO] written=%s rows=%d zeros_home=%d zeros_away=%d min/med/max home=%s/%s/%s away=%s/%s/%s",
        out_path,
        len(hist_out),
        zero_home,
        zero_away,
        hist_out["elo_home_calc"].min(),
        hist_out["elo_home_calc"].median(),
        hist_out["elo_home_calc"].max(),
        hist_out["elo_away_calc"].min(),
        hist_out["elo_away_calc"].median(),
        hist_out["elo_away_calc"].max(),
    )

    home_delta = (
        hist_out.sort_values("datetime").groupby("home")["elo_home_calc"].diff().abs().max()
    )
    away_delta = (
        hist_out.sort_values("datetime").groupby("away")["elo_away_calc"].diff().abs().max()
    )
    log_json(
        {
            "event": "ELO_ELO",
            "out": str(out_path),
            "rows": len(hist_out),
            "zeros_home": zero_home,
            "zeros_away": zero_away,
            "elo_home_min": float(hist_out["elo_home_calc"].min()),
            "elo_home_med": float(hist_out["elo_home_calc"].median()),
            "elo_home_max": float(hist_out["elo_home_calc"].max()),
            "elo_away_min": float(hist_out["elo_away_calc"].min()),
            "elo_away_med": float(hist_out["elo_away_calc"].median()),
            "elo_away_max": float(hist_out["elo_away_calc"].max()),
            "max_abs_delta_home": float(home_delta) if pd.notna(home_delta) else None,
            "max_abs_delta_away": float(away_delta) if pd.notna(away_delta) else None,
            "config": {
                "start_elo": args.start_elo,
                "k_factor": args.k_factor,
                "home_adv": args.home_adv,
                "band_thresh": args.band_thresh,
                "cutoff": max_date,
                "elo_v2_enabled": _cfg_bool("ELO_V2_ENABLED", True),
                "elo_defaults": _cfg_map("ELO_DEFAULTS", {}),
                "elo_league_params": _cfg_map("ELO_LEAGUE_PARAMS", {}),
                "elo_matchtype_multipliers": _cfg_map("ELO_MATCHTYPE_MULTIPLIERS", {}),
            },
        },
        log_path=Path(args.log_json),
    )


if __name__ == "__main__":
    main()
