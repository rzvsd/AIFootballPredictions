"""
Recompute Elo time series from CGM match history (ignores any Elo stored in the CSV).

Inputs:
- data/enhanced/cgm_match_history.csv (or --history)
  Required columns: date, home, away, ft_home, ft_away, code_home/code_away (optional but preferred)

Outputs:
- data/enhanced/cgm_match_history_with_elo.csv (or --out)
  Adds/overwrites: elo_home_calc, elo_away_calc, EloDiff_calc, Band_H_calc, Band_A_calc

Constants (override via CLI):
- START_ELO = 1500
- K_FACTOR = 20
- HOME_ADV = 65
- BAND_THRESH = 150 (for BULLY/PEER/DOG labels)

Margin multiplier: World Football Elo style.
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
    from cgm.team_registry import build_team_registry
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cgm"))
    from team_registry import build_team_registry  # type: ignore


START_ELO_DEFAULT = 1500.0
K_FACTOR_DEFAULT = 20.0
HOME_ADV_DEFAULT = 65.0
BAND_THRESH_DEFAULT = 150.0
LOG_PATH_DEFAULT = Path("reports/run_log.jsonl")


def margin_multiplier(goal_diff: int) -> float:
    gd = abs(int(goal_diff))
    if gd == 0:
        return 1.0
    if gd == 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8.0


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


def load_history(path: Path, max_date: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        dt = pd.to_datetime(df["date"] + " " + df.get("time", "").astype(str), errors="coerce")
        dt2 = pd.to_datetime(df["date"], errors="coerce")
        df["datetime"] = dt.fillna(dt2)
    df = df.sort_values("datetime")
    if max_date:
        cutoff = pd.to_datetime(max_date)
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
) -> Tuple[pd.Series, pd.Series]:
    ratings: Dict[str, float] = {}
    elo_home_list = []
    elo_away_list = []

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

        # Default to starting Elo if unknown or missing
        r_home = ratings.get(home_id, start_elo) if home_id else start_elo
        r_away = ratings.get(away_id, start_elo) if away_id else start_elo

        elo_home_list.append(r_home)
        elo_away_list.append(r_away)

        # If result missing, don't update ratings
        if pd.isna(fh) or pd.isna(fa) or home_id is None or away_id is None:
            continue

        exp_home = expected_home(r_home, r_away, home_adv)
        if fh > fa:
            actual = 1.0
        elif fh == fa:
            actual = 0.5
        else:
            actual = 0.0

        gd = int(abs(fh - fa))
        mult = margin_multiplier(gd)
        delta = k_factor * mult * (actual - exp_home)

        ratings[home_id] = r_home + delta
        ratings[away_id] = r_away - delta

    return pd.Series(elo_home_list), pd.Series(elo_away_list)


def band_from_diff(diff: float, thresh: float) -> str:
    if diff >= thresh:
        return "BULLY"
    if diff <= -thresh:
        return "DOG"
    return "PEER"


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute Elo time series from CGM match history")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history.csv", help="Input history CSV")
    ap.add_argument("--out", default="data/enhanced/cgm_match_history_with_elo.csv", help="Output CSV path (cleaned)")
    ap.add_argument("--start-elo", type=float, default=START_ELO_DEFAULT, help="Starting Elo for new teams")
    ap.add_argument("--k-factor", type=float, default=K_FACTOR_DEFAULT, help="K factor")
    ap.add_argument("--home-adv", type=float, default=HOME_ADV_DEFAULT, help="Home advantage points")
    ap.add_argument("--band-thresh", type=float, default=BAND_THRESH_DEFAULT, help="Band threshold for BULLY/PEER/DOG")
    ap.add_argument("--data-dir", default="CGM data", help="CGM data directory for team registry (code->name)")
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
    elo_home, elo_away = compute_elo_series(
        hist,
        start_elo=args.start_elo,
        k_factor=args.k_factor,
        home_adv=args.home_adv,
    )

    hist_out = hist.copy()
    hist_out["elo_home_calc"] = elo_home
    hist_out["elo_away_calc"] = elo_away
    hist_out["EloDiff_calc"] = (hist_out["elo_home_calc"] + args.home_adv) - hist_out["elo_away_calc"]
    hist_out["Band_H_calc"] = hist_out["EloDiff_calc"].apply(lambda d: band_from_diff(d, args.band_thresh))
    hist_out["Band_A_calc"] = hist_out["EloDiff_calc"].apply(lambda d: band_from_diff(-d, args.band_thresh))

    # Overwrite legacy Elo columns to avoid downstream accidental use
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

    # Approx max Elo delta per team (home/away contexts)
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
                "margin_multiplier": "world_football",
            },
        },
        log_path=Path(args.log_json),
    )


if __name__ == "__main__":
    main()
