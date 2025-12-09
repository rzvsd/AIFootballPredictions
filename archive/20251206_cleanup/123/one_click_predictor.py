
#!/usr/bin/env python3
"""
one_click_predictor.py  — v2 (diagnostics + fallbacks)

Run end-to-end and produce 1X2 picks from your champion XGB models.

Usage examples:
  python one_click_predictor.py --league E0 --days 7 --min-prob 0.60
  python one_click_predictor.py --league E0 --fixtures-csv data/fixtures/E0_week.csv

Key improvements vs v1:
- Loud diagnostics (season, date range, API connectivity, fixture count).
- Fallbacks if the API returns nothing:
    1) --fixtures-csv path (any CSV with date/home/away columns)
    2) data/fixtures/{league}_weekly_fixtures.csv (if present)
    3) data/fixtures/{league}_manual.csv (create-once template if absent)
"""

import os
import sys
import argparse
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Iterable, Tuple
import pandas as pd
import numpy as np
import joblib

from scipy.stats import poisson

# Local modules
import config
import feature_store
import xgb_trainer

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import requests
import difflib

# ---------------------------
# Utils
# ---------------------------

def ensure_dirs():
    for p in ["data/store", "data/fixtures", "data/picks", "advanced_models"]:
        os.makedirs(p, exist_ok=True)

def season_from_date(d: date) -> int:
    # EPL season flips in August
    return d.year if d.month >= 8 else d.year - 1

def fuzzy_normalize(name: str, known: set, mapping: Dict[str,str]) -> str:
    if not name:
        return name
    mapped = mapping.get(name, name)
    if mapped in known:  # direct hit
        return mapped
    # cheap cleanups
    clean = mapped.replace("FC ", "").replace(" FC", "").replace(" Utd", " United").strip()
    if clean in known:
        return clean
    # fuzzy
    cand = difflib.get_close_matches(clean, list(known), n=1, cutoff=0.75)
    return cand[0] if cand else mapped

def poisson_1x2(home_mu: float, away_mu: float, max_goals: int = 10) -> Dict[str, float]:
    home_probs = [poisson.pmf(i, home_mu) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_mu) for i in range(max_goals + 1)]
    M = np.outer(np.array(home_probs), np.array(away_probs))
    M = M / M.sum()
    pH = float(np.tril(M, -1).sum())
    pD = float(np.diag(M).sum())
    pA = float(np.triu(M, 1).sum())
    return {"P(H)": pH, "P(D)": pD, "P(A)": pA}

def build_snapshot_if_needed(league: str, as_of: Optional[str] = None, verbose=True) -> str:
    as_of = as_of or date.today().isoformat()
    enh_path = os.path.join("data", "enhanced", f"{league}_final_features.csv")
    if not os.path.exists(enh_path):
        raise FileNotFoundError(f"Expected enhanced CSV at {enh_path}. Build it first (feature_enhancer.py -> feature_builder.py).")
    snap = feature_store.build_snapshot(enhanced_csv=enh_path, as_of=as_of)
    out_dir = os.path.join("data", "store")
    os.makedirs(out_dir, exist_ok=True)
    latest_path = os.path.join(out_dir, f"{league}_latest_team_stats.parquet")
    dated_path  = os.path.join(out_dir, f"{league}_snapshot_{as_of}.parquet")
    snap.to_parquet(latest_path, index=False)
    snap.to_parquet(dated_path, index=False)
    if verbose:
        print(f"[feature_store] snapshot for {league} as_of {as_of}: {len(snap)} teams -> {latest_path}")
    return latest_path

def train_models_if_missing(league: str, verbose=True):
    home_p = os.path.join("advanced_models", f"{league}_ultimate_xgb_home.pkl")
    away_p = os.path.join("advanced_models", f"{league}_ultimate_xgb_away.pkl")
    if os.path.exists(home_p) and os.path.exists(away_p):
        if verbose:
            print("[trainer] champion models found.")
        return
    if verbose:
        print("[trainer] training champion models...")
    # Fix: call the existing per-league trainer
    xgb_trainer.train_xgb_models_for_league(league)
    if verbose:
        print("[trainer] done.")

def dbg(msg: str):
    print(f"[debug] {msg}")

def fetch_fixtures_next_days(league_id: int, days: int, api_key: Optional[str]) -> pd.DataFrame:
    """Pull fixtures for next N days. Return empty df if failed/none."""
    today = date.today()
    to = today + timedelta(days=days)
    params = {
        'league': league_id,
        'season': season_from_date(today),
        'from': today.isoformat(),
        'to': to.isoformat()
    }
    dbg(f"season={params['season']} range={params['from']}..{params['to']} league_id={league_id}")
    if not api_key:
        dbg("No API key found in env (API_FOOTBALL_KEY or API_FOOTBALL_ODDS_KEY). Skipping API fetch.")
        return pd.DataFrame(columns=["date","home_team_api","away_team_api"])

    headers = {'x-apisports-key': api_key}
    url = "https://v3.football.api-sports.io/fixtures"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        dbg(f"GET {url} -> status={r.status_code}")
        if r.status_code != 200:
            dbg(f"API response text: {r.text[:300]}")
        r.raise_for_status()
        payload = r.json().get("response", [])
        dbg(f"API returned {len(payload)} fixtures.")
    except Exception as e:
        dbg(f"API error: {e}")
        return pd.DataFrame(columns=["date","home_team_api","away_team_api"])

    rows=[]
    for m in payload:
        try:
            rows.append({
                "date": m["fixture"]["date"],
                "home_team_api": m["teams"]["home"]["name"],
                "away_team_api": m["teams"]["away"]["name"],
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

# ---------- CSV fallbacks ----------


def read_any_fixture_csv(path: str) -> pd.DataFrame:
    """Read a user-provided fixtures CSV with flexible column names."""
    df = pd.read_csv(path)
    # map lowercase->original for robust matching
    cols = {c.strip().lower(): c for c in df.columns}

    def pick_alias(aliases):
        for a in aliases:
            k = a.strip().lower()
            if k in cols:
                return cols[k]
        return None

    date_col = pick_alias(["date", "matchdate", "kickoff", "ko"])
    home_col = pick_alias(["home", "hometeam", "home_team", "home team", "home_team_api"])
    away_col = pick_alias(["away", "awayteam", "away_team", "away team", "away_team_api"])
    if not (date_col and home_col and away_col):
        raise ValueError(f"CSV {path} must contain date/home/away (flex names ok). Found columns: {list(df.columns)}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d %H:%M:%S"),
        "home_team_api": df[home_col].astype(str).str.strip(),
        "away_team_api": df[away_col].astype(str).str.strip(),
    })
    return out
def fixtures_fallbacks(league_code: str, preferred_csv: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    tried = []
    # 1) Explicit --fixtures-csv
    if preferred_csv:
        if os.path.exists(preferred_csv):
            try:
                df = read_any_fixture_csv(preferred_csv)
                return df, f"from CSV: {preferred_csv}"
            except Exception as e:
                print(f"[fallback] Failed to read {preferred_csv}: {e}")
        tried.append(preferred_csv)

    # 2) Default weekly path
    weekly = os.path.join("data","fixtures", f"{league_code}_weekly_fixtures.csv")
    if os.path.exists(weekly):
        try:
            df = read_any_fixture_csv(weekly)
            return df, f"from CSV: {weekly}"
        except Exception as e:
            print(f"[fallback] Failed to read {weekly}: {e}")
        tried.append(weekly)

    # 3) Manual template
    manual = os.path.join("data","fixtures", f"{league_code}_manual.csv")
    if not os.path.exists(manual):
        os.makedirs(os.path.dirname(manual), exist_ok=True)
        # create a template with a couple of sample rows
        with open(manual, "w", encoding="utf-8") as f:
            f.write("date,home,away\n")
            f.write("2025-08-30 17:00,Man City,Everton\n")
            f.write("2025-08-31 19:30,Man United,Arsenal\n")
    tried.append(manual)
    print(f"[fallback] No API fixtures. Please fill: {manual} then re-run, or pass --fixtures-csv PATH")
    return pd.DataFrame(columns=["date","home_team_api","away_team_api"]), None

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default=config.LEAGUE_CODE, help="League code, e.g. E0")
    ap.add_argument("--days",   type=int, default=7, help="Days ahead to fetch via API")
    ap.add_argument("--min-prob", type=float, default=0.60, help="Minimum probability to output picks")
    ap.add_argument("--fixtures-csv", default=None, help="Optional fixtures CSV (date,home,away) if API fails")
    args = ap.parse_args()

    ensure_dirs()

    # Snapshot
    snap_path = build_snapshot_if_needed(args.league, as_of=date.today().isoformat())

    # Models
    train_models_if_missing(args.league)
    stats = pd.read_parquet(snap_path)
    home_model = joblib.load(config.HOME_MODEL_PATH)
    away_model = joblib.load(config.AWAY_MODEL_PATH)
    known_teams = set(stats["team"].dropna().astype(str))

    # Diagnostics
    api_key = os.getenv("API_FOOTBALL_ODDS_KEY") or os.getenv("API_FOOTBALL_KEY")
    dbg(f"env key present: {'yes' if api_key else 'no'}")
    dbg(f"config.LEAGUE_ID_API={getattr(config, 'LEAGUE_ID_API', None)}  config.LEAGUE_CODE={config.LEAGUE_CODE}")
    dbg(f"snapshot teams (first 10): {sorted(list(known_teams))[:10]} ... total={len(known_teams)}")

    # Try API
    fixtures = fetch_fixtures_next_days(getattr(config, "LEAGUE_ID_API", 39), args.days, api_key)

    source = "API-Football"
    if fixtures.empty:
        # Try fallbacks
        df2, src = fixtures_fallbacks(args.league, args.fixtures_csv)
        if src:
            fixtures = df2
            source = src
        else:
            print("No fixtures available to analyze. (API returned none; manual CSV template created.)")
            return

    print(f"[fixtures] Loaded {len(fixtures)} fixtures ({source}).")

    # Normalize names
    fixtures["home_team"] = fixtures["home_team_api"].apply(lambda s: fuzzy_normalize(s, known_teams, getattr(config, "TEAM_NAME_MAP", {})))
    fixtures["away_team"] = fixtures["away_team_api"].apply(lambda s: fuzzy_normalize(s, known_teams, getattr(config, "TEAM_NAME_MAP", {})))

    # Predict
    preds = []
    for _, fx in fixtures.iterrows():
        home = fx["home_team"]
        away = fx["away_team"]
        hs = stats.loc[stats["team"]==home]
        as_ = stats.loc[stats["team"]==away]
        if hs.empty or as_.empty:
            print(f"[skip] Missing stats for {home} or {away}. (Check team mapping or snapshot coverage.)")
            continue

        # Construct feature row in the order the model expects
        feat_vals = []
        for col in config.ULTIMATE_FEATURES:
            # Snapshot columns are rolling aggregates; map common ones here.
            # Adjust these keys if your snapshot uses slightly different names.
            mapping = {
                'ShotConv_H':            'xg_L5',
                'ShotConv_A':            'xg_L5',
                'ShotConvRec_H':         'xga_L5',
                'ShotConvRec_A':         'xga_L5',
                'PointsPerGame_H':       'gpg_L10',
                'PointsPerGame_A':       'gpg_L10',
                'CleanSheetStreak_H':    None,
                'CleanSheetStreak_A':    None,
                'xGDiff_H':              'xgdiff_L10',
                'xGDiff_A':              'xgdiff_L10',
                'CornersConv_H':         'corners_L10',
                'CornersConv_A':         'corners_L10',
                'CornersConvRec_H':      'corners_allowed_L10',
                'CornersConvRec_A':      'corners_allowed_L10',
                'NumMatches_H':          None,
                'NumMatches_A':          None,
            }
            key = mapping.get(col)
            if col.endswith("_H"):
                feat_vals.append(float(hs.iloc[0][key]) if key else 20.0 if 'NumMatches' in col else 0.0)
            elif col.endswith("_A"):
                feat_vals.append(float(as_.iloc[0][key]) if key else 20.0 if 'NumMatches' in col else 0.0)
            else:
                raise ValueError(f"Unexpected feature name: {col}")

        feature_row = pd.DataFrame([feat_vals], columns=config.ULTIMATE_FEATURES)
        mu_h = float(home_model.predict(feature_row)[0])
        mu_a = float(away_model.predict(feature_row)[0])
        probs = poisson_1x2(mu_h, mu_a)

        preds.append({
            "Date": pd.to_datetime(fx["date"]).strftime("%Y-%m-%d %H:%M"),
            "Home": fx["home_team_api"],
            "Away": fx["away_team_api"],
            "home_norm": home,
            "away_norm": away,
            "xG_H": round(mu_h, 3),
            "xG_A": round(mu_a, 3),
            "P(H)": round(probs["P(H)"], 4),
            "P(D)": round(probs["P(D)"], 4),
            "P(A)": round(probs["P(A)"], 4),
        })

    if not preds:
        print("No predictions produced. Likely team-name mismatch or snapshot missing promoted teams.")
        print("Tip: open data/fixtures/{league}_manual.csv, paste fixtures using model names (e.g., 'Man City'), re-run.")
        return

    out_df = pd.DataFrame(preds).sort_values("Date")
    mask = (out_df["P(H)"] >= args.min_prob) | (out_df["P(D)"] >= args.min_prob) | (out_df["P(A)"] >= args.min_prob)
    picks = out_df[mask].copy()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    all_path  = os.path.join("data", "picks", f"{args.league}_all_{ts}.csv")
    picks_path= os.path.join("data", "picks", f"{args.league}_picks_{ts}.csv")
    os.makedirs(os.path.dirname(all_path), exist_ok=True)
    out_df.to_csv(all_path, index=False)
    picks.to_csv(picks_path, index=False)

    print("\n=== SUMMARY ===")
    print(f"Analyzed {len(out_df)} fixtures | High-conv picks: {len(picks)} (min_prob={args.min_prob:.2f})")
    print(f"Saved all predictions -> {all_path}")
    print(f"Saved picks           -> {picks_path}")

    cols = ["Date","Home","Away","xG_H","xG_A","P(H)","P(D)","P(A)"]
    try:
        from tabulate import tabulate
        print("\nPicks:")
        print(tabulate(picks[cols], headers="keys", tablefmt="github", showindex=False))
    except Exception:
        print("\nPicks:")
        print(picks[cols].to_string(index=False))

if __name__ == "__main__":
    main()
