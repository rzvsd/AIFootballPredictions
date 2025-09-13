# weekly_predictor.py
# Odds-free weekly picks (Tue→Mon or custom window). Robust fixtures with local-CSV option.
# Adds:
#   --summary : prints training data coverage (rows + min/max Date)
#   --source  : 'local' skips APIs and only uses your CSV; 'auto' tries APIs then falls back to CSV
#
# Default probability threshold = 0.30 (ease of testing). Raise later for production.
#
# Usage examples:
#   python weekly_predictor.py --from 2025-08-30 --to 2025-09-01 --min-conf 0.30 --source local --summary
#   python weekly_predictor.py --min-conf 0.70 --summary
#
# Output: prints table + saves picks_week_<from>_to_<to>.csv

import os
import argparse
from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple, Iterable, Optional

import requests
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from math import factorial

load_dotenv()

# -------- League maps --------
AF_LEAGUE_IDS: Dict[str, int] = {
    "E0": 39,   # Premier League
    "D1": 78,   # Bundesliga
}
FD_COMP_CODES: Dict[str, str] = {
    "E0": "PL",
    "D1": "BL1",
}

# -------- Models / features --------
DC_MODEL_PATH = os.path.join("advanced_models", "{league}_dixon_coles_model.pkl")
HYB_HOME_PATH = os.path.join("advanced_models", "{league}_xgb_home_model.pkl")
HYB_AWAY_PATH = os.path.join("advanced_models", "{league}_xgb_away_model.pkl")
FEATURES_PATH = os.path.join("data", "enhanced", "{league}_final_features.csv")

# -------- Team-name normalization (API/local -> training names) --------
TEAM_MAP_API_TO_BASE = {
    # Official → base
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton and Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "AFC Bournemouth": "Bournemouth",
    "Sheffield Utd": "Sheffield United",
    # Common short forms / nicknames → base
    "Spurs": "Tottenham",
    "Man Utd": "Man United",
    "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle",
    "West Ham Utd": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton": "Wolves",
    "Nottingham": "Nott'm Forest",
    "Nott Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
}
def norm_team(x: str) -> str:
    return TEAM_MAP_API_TO_BASE.get(x, x)

# -------- Env helpers --------
def api_football_key() -> str:
    for k in ("API_FOOTBALL_KEY", "API_FOOTBALL_ODDS_KEY", "API_FOOTBALL_DATA"):
        v = os.getenv(k)
        if v: return v
    return ""

def football_data_key() -> str:
    return os.getenv("FOOTBALL_DATA_KEY", "")

# -------- Date helpers --------
def next_tue_to_mon() -> Tuple[str, str]:
    today = date.today()
    days_to_tue = (1 - today.weekday()) % 7
    start = today + timedelta(days=days_to_tue or 7)  # always NEXT Tuesday
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()

def window_or_default(dfrom: Optional[str], dto: Optional[str]) -> Tuple[str, str]:
    return (dfrom, dto) if (dfrom and dto) else next_tue_to_mon()

def season_start_year(dfrom: str) -> int:
    y, m, _ = map(int, dfrom.split("-"))
    return y if m >= 7 else y - 1

def daterange(d1: str, d2: str) -> Iterable[str]:
    a = datetime.fromisoformat(d1).date()
    b = datetime.fromisoformat(d2).date()
    for n in range((b - a).days + 1):
        yield (a + timedelta(days=n)).isoformat()

# -------- HTTP paging --------
def collect_all_pages(session: requests.Session, url: str, headers: dict, params: dict) -> List[dict]:
    out, page = [], 1
    while True:
        p = dict(params, page=page)
        r = session.get(url, headers=headers, params=p, timeout=25)
        r.raise_for_status()
        j = r.json()
        out.extend(j.get("response", []))
        pg = j.get("paging", {})
        if not pg or pg.get("current", 1) >= pg.get("total", 1):
            break
        page += 1
    return out

# -------- Fixtures: API-Football (multi-strategy) --------
def fetch_fixtures_api_football(api_key: str, league_id: int, dfrom: str, dto: str) -> List[dict]:
    headers = {"x-apisports-key": api_key}
    s = requests.Session()
    url = "https://v3.football.api-sports.io/fixtures"
    season = season_start_year(dfrom)

    def pack(items):
        fx = []
        for m in items:
            try:
                fx.append({
                    "date": m["fixture"]["date"],
                    "home_team": norm_team(m["teams"]["home"]["name"]),
                    "away_team": norm_team(m["teams"]["away"]["name"]),
                })
            except Exception:
                pass
        return fx

    # A) season + window + statuses
    total = []
    for st in ("NS", "TBD"):
        try:
            rows = collect_all_pages(s, url, headers, {
                "league": league_id, "season": season,
                "status": st, "from": dfrom, "to": dto, "timezone": "UTC"
            })
            total.extend(pack(rows))
        except Exception as e:
            print(f"[debug] AF A/{st} error: {e}")
    if total:
        df = pd.DataFrame(total).drop_duplicates(["date", "home_team", "away_team"])
        print(f"[debug] AF A count: {len(df)} (season={season})")
        return df.to_dict("records")

    # B) no season + window + statuses
    total = []
    for st in ("NS", "TBD"):
        try:
            rows = collect_all_pages(s, url, headers, {
                "league": league_id, "status": st,
                "from": dfrom, "to": dto, "timezone": "UTC"
            })
            total.extend(pack(rows))
        except Exception as e:
            print(f"[debug] AF B/{st} error: {e}")
    if total:
        df = pd.DataFrame(total).drop_duplicates(["date", "home_team", "away_team"])
        print(f"[debug] AF B count: {len(df)}")
        return df.to_dict("records")

    # C) per-day NS/TBD
    total = []
    for day in daterange(dfrom, dto):
        for st in ("NS", "TBD"):
            try:
                rows = collect_all_pages(s, url, headers, {
                    "league": league_id, "date": day, "status": st, "timezone": "UTC"
                })
                total.extend(pack(rows))
            except Exception as e:
                print(f"[debug] AF C/{day}/{st} error: {e}")
    if total:
        df = pd.DataFrame(total).drop_duplicates(["date", "home_team", "away_team"])
        print(f"[debug] AF C count: {len(df)}")
        return df.to_dict("records")

    # D) window no status
    try:
        rows = collect_all_pages(s, url, headers, {
            "league": league_id, "from": dfrom, "to": dto, "timezone": "UTC"
        })
        total = pack(rows)
        if total:
            df = pd.DataFrame(total).drop_duplicates(["date", "home_team", "away_team"])
            print(f"[debug] AF D count: {len(df)}")
            return df.to_dict("records")
    except Exception as e:
        print(f"[debug] AF D error: {e}")

    print(f"[debug] AF total=0 for league={league_id}, window={dfrom}->{dto}, season={season}")
    return []

# -------- Fixtures: football-data.org (fallback) --------
def fetch_fixtures_footballdata(fd_key: str, comp_code: str, dfrom: str, dto: str) -> List[dict]:
    url = "https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": fd_key}
    params = {"competitions": comp_code, "status": "SCHEDULED", "dateFrom": dfrom, "dateTo": dto}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=25)
        r.raise_for_status()
        j = r.json()
        out = [{
            "date": m["utcDate"],
            "home_team": norm_team(m["homeTeam"]["name"]),
            "away_team": norm_team(m["awayTeam"]["name"]),
        } for m in j.get("matches", [])]
        print(f"[debug] FD count: {len(out)}")
        return out
    except Exception as e:
        print(f"[debug] FD error: {e}")
        return []

# -------- Local CSV fallback (cleans rows) --------
def fetch_fixtures_local(league: str, dfrom: str, dto: str) -> List[dict]:
    """
    Load data/fixtures/{LEAGUE}_{from}_to_{to}.csv with columns:
      date,home_team,away_team
    Cleans: trims, drops header-like rows, validates ISO date.
    """
    os.makedirs("data/fixtures", exist_ok=True)
    path = os.path.join("data", "fixtures", f"{league}_{dfrom}_to_{dto}.csv")
    if not os.path.exists(path):
        # write a template to help the first run
        pd.DataFrame(columns=["date", "home_team", "away_team"]).to_csv(path, index=False)
        print(f"[info] No API fixtures. Created template → {path}. Fill it and re-run.")
        return []

    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        print(f"[debug] Local read error: {e}")
        return []

    # normalize headers to lower
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"date", "home_team", "away_team"}
    if not need.issubset(set(df.columns)):
        print(f"[debug] Local file missing columns. Need: {need}")
        return []

    # strip whitespace
    for c in ["date", "home_team", "away_team"]:
        df[c] = df[c].astype(str).str.strip()

    # drop header-like rows accidentally saved as data
    mask_bad = (
        df["home_team"].str.lower().eq("home_team") |
        df["away_team"].str.lower().eq("away_team") |
        df["date"].str.lower().eq("date")
    )
    df = df[~mask_bad]

    # validate ISO datetimes (keep only parseable)
    def is_iso(ts: str) -> bool:
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return True
        except Exception:
            return False

    df = df[df["date"].apply(is_iso)]

    # apply normalization map
    df["home_team"] = df["home_team"].map(norm_team)
    df["away_team"] = df["away_team"].map(norm_team)

    rows = df[["date", "home_team", "away_team"]].dropna()
    print(f"[debug] Local fixtures cleaned: {len(rows)} from {path}")
    return rows.to_dict("records")

# -------- Model utils (fixed Poisson PMF: no scipy) --------
def score_matrix_from_mus(mu_h: float, mu_a: float, max_goals: int = 10) -> np.ndarray:
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)

    def pois_pmf(k_arr, lam):
        k_arr = np.asarray(k_arr, dtype=float)
        fact = np.array([factorial(int(k)) for k in k_arr], dtype=float)
        return np.exp(-lam) * (lam ** k_arr) / fact

    ph = pois_pmf(hg, mu_h)
    pa = pois_pmf(ag, mu_a)
    P = np.outer(ph, pa)
    P /= P.sum()
    return P

def probs_1x2(P: np.ndarray) -> Dict[str, float]:
    return {"home": float(np.tril(P, -1).sum()),
            "draw": float(np.diag(P).sum()),
            "away": float(np.triu(P, 1).sum())}

def probs_ou25(P: np.ndarray) -> Dict[str, float]:
    grid = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
    over = float(P[grid >= 3].sum())
    return {"over25": over, "under25": 1.0 - over}

def dc_matrix(dc_model, home: str, away: str) -> Optional[np.ndarray]:
    """
    Works for statsmodels formula GLMs saved in training.
    Ensures we pass a DataFrame with the *exact* categorical levels used at train time.
    """
    try:
        import pandas as pd
        d = pd.DataFrame({"team": [home], "opponent": [away], "home": [1]})
        mu_h = float(dc_model.predict(d).values[0])
        d = pd.DataFrame({"team": [away], "opponent": [home], "home": [0]})
        mu_a = float(dc_model.predict(d).values[0])
        return score_matrix_from_mus(mu_h, mu_a)
    except Exception as e:
        print(f"[debug] DC matrix failed for {home} vs {away}: {e}")
        return None

def load_hybrid(league: str):
    hp = HYB_HOME_PATH.format(league=league)
    ap = HYB_AWAY_PATH.format(league=league)
    if not (os.path.exists(hp) and os.path.exists(ap)):
        return None, None, None
    try:
        home = joblib.load(hp)
        away = joblib.load(ap)
    except Exception as e:
        print(f"[{league}] Hybrid load failed: {e}")
        return None, None, None
    feats = None
    fpath = FEATURES_PATH.format(league=league)
    if os.path.exists(fpath):
        try:
            feats = pd.read_csv(fpath)
        except Exception:
            feats = None
    return home, away, feats

def hyb_matrix(h_model, a_model, feats_df: Optional[pd.DataFrame], home: str, away: str) -> Optional[np.ndarray]:
    if (h_model is None) or (a_model is None) or (feats_df is None):
        return None
    try:
        h = feats_df[feats_df["HomeTeam"] == home].tail(1).iloc[0]
        a = feats_df[feats_df["AwayTeam"] == away].tail(1).iloc[0]
        vals = {
            "HomeAvgGoalsScored_Last5": h.get("HomeAvgGoalsScored_Last5", np.nan),
            "HomeAvgGoalsConceded_Last5": h.get("HomeAvgGoalsConceded_Last5", np.nan),
            "AwayAvgGoalsScored_Last5": a.get("AwayAvgGoalsScored_Last5", np.nan),
            "AwayAvgGoalsConceded_Last5": a.get("AwayAvgGoalsConceded_Last5", np.nan),
            "HomeAvgShots_Last5": h.get("HomeAvgShots_Last5", np.nan),
            "AwayAvgShots_Last5": a.get("AwayAvgShots_Last5", np.nan),
            "HomeAvgShotsOnTarget_Last5": h.get("HomeAvgShotsOnTarget_Last5", np.nan),
            "AwayAvgShotsOnTarget_Last5": a.get("AwayAvgShotsOnTarget_Last5", np.nan),
            "HomeAvgCorners_Last5": h.get("HomeAvgCorners_Last5", np.nan),
            "AwayAvgCorners_Last5": a.get("AwayAvgCorners_Last5", np.nan),
            "EloDifference": h.get("HomeElo", np.nan) - a.get("AwayElo", np.nan),
        }
        X = pd.DataFrame([vals])
        mu_h = float(h_model.predict(X)[0])
        mu_a = float(a_model.predict(X)[0])
        return score_matrix_from_mus(mu_h, mu_a)
    except Exception:
        return None

# -------- Training summary helper --------
def print_training_summary(league: str):
    feat_path = FEATURES_PATH.format(league=league)
    if not os.path.exists(feat_path):
        print(f"[{league}] no features file found for summary: {feat_path}")
        return
    try:
        # Be tolerant of 'Date'/'date'
        usecols = None
        cols = pd.read_csv(feat_path, nrows=0).columns.str.lower().tolist()
        if "date" in cols:
            usecols = ["Date"] if "Date" in pd.read_csv(feat_path, nrows=0).columns else ["date"]
        dfh = pd.read_csv(feat_path, usecols=usecols) if usecols else pd.read_csv(feat_path)
        col = "Date" if "Date" in dfh.columns else ("date" if "date" in dfh.columns else None)
        if col:
            print(f"[{league}] trained on rows={len(dfh)}, span={dfh[col].min()} → {dfh[col].max()}")
        else:
            print(f"[{league}] summary: Date column not found in {feat_path}")
    except Exception as e:
        print(f"[{league}] summary read failed: {e}")

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Weekly odds-free picks (>= probability threshold).")
    ap.add_argument("--leagues", nargs="+", default=["E0"], help="League codes (e.g., E0 D1)")
    ap.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (default: next Tuesday)")
    ap.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (default: following Monday)")
    ap.add_argument("--min-conf", type=float, default=0.30, help="Minimum probability (default 0.30 = 30%)")
    ap.add_argument("--markets", default="1X2,OU25", help="Comma list: 1X2, OU25")
    ap.add_argument("--require-agreement", action="store_true",
                    help="If both DC and Hybrid available, require both >= threshold AND same side.")
    ap.add_argument("--summary", action="store_true",
                    help="Print training data coverage for each league")
    ap.add_argument("--source", choices=["auto", "local"], default="auto",
                    help="auto = APIs then local fallback; local = only CSV (skip APIs)")
    args = ap.parse_args()

    dfrom, dto = window_or_default(args.date_from, args.date_to)
    markets = {m.strip().upper() for m in args.markets.split(",") if m.strip()}
    min_conf = float(args.min_conf)

    af_key = api_football_key()
    fd_key = football_data_key()
    print(f"--- Weekly Predictor ---\nWindow: {dfrom} → {dto}\nMarkets: {', '.join(sorted(markets))}\nThreshold: {min_conf:.0%}")
    print(f"[keys] API-Football: {'set' if af_key else 'missing'} | football-data.org: {'set' if fd_key else 'missing'}\n")

    picks = []

    for lg in args.leagues:
        print(f"[{lg}] Starting…")
        if args.summary:
            print_training_summary(lg)

        league_id = AF_LEAGUE_IDS.get(lg)
        comp_code = FD_COMP_CODES.get(lg)

        # models
        dc_model = None
        dcp = DC_MODEL_PATH.format(league=lg)
        if os.path.exists(dcp):
            try:
                dc_model = joblib.load(dcp); print(f"[{lg}] DC model loaded.")
            except Exception as e:
                print(f"[{lg}] DC load failed: {e}")

        hyh, hya, hyf = load_hybrid(lg)
        print(f"[{lg}] Hybrid: {'loaded' if (hyh and hya and hyf is not None) else 'not used'}.")

        if dc_model is None and not (hyh and hya and hyf is not None):
            print(f"[{lg}] No usable models; skipping.")
            continue

        # fixtures
        fixtures: List[dict] = []
        if args.source == "local":
            fixtures = fetch_fixtures_local(lg, dfrom, dto)
        else:
            if af_key and league_id:
                fixtures = fetch_fixtures_api_football(af_key, league_id, dfrom, dto)
            if not fixtures and fd_key and comp_code:
                print(f"[{lg}] API-Football returned 0 fixtures. Trying football-data.org…")
                fixtures = fetch_fixtures_footballdata(fd_key, comp_code, dfrom, dto)
            if not fixtures:
                print(f"[{lg}] APIs returned 0 fixtures. Falling back to local CSV.")
                fixtures = fetch_fixtures_local(lg, dfrom, dto)

        if not fixtures:
            print(f"[{lg}] No fixtures available (API + local).")
            continue

        print(f"[{lg}] Fixtures in window: {len(fixtures)}")

        for fx in fixtures:
            h, a, kickoff = fx["home_team"], fx["away_team"], fx["date"]

            P_dc = dc_matrix(dc_model, h, a) if dc_model else None
            P_hy = hyb_matrix(hyh, hya, hyf, h, a) if (hyh and hya and hyf is not None) else None
            if P_dc is None and P_hy is None:
                continue

            # 1X2
            if "1X2" in markets:
                probs_dc = probs_1x2(P_dc) if P_dc is not None else None
                probs_hy = probs_1x2(P_hy) if P_hy is not None else None
                ref = probs_dc or probs_hy
                side = max(ref, key=ref.get)
                p_dc = probs_dc.get(side) if probs_dc else None
                p_hy = probs_hy.get(side) if probs_hy else None
                pass_dc = (p_dc is None) or (p_dc >= min_conf)
                pass_hy = (p_hy is None) or (p_hy >= min_conf)
                agree = True
                if args.require_agreement and (probs_dc is not None) and (probs_hy is not None):
                    agree = (max(probs_dc, key=probs_dc.get) == max(probs_hy, key=probs_hy.get))
                if pass_dc and pass_hy and agree:
                    picks.append({
                        "league": lg, "kickoff_utc": kickoff, "home": h, "away": a,
                        "market": "1X2", "pick": {"home": h, "draw": "Draw", "away": a}[side],
                        "side": side.upper(),
                        "prob_dc": round(p_dc, 4) if p_dc is not None else None,
                        "prob_hy": round(p_hy, 4) if p_hy is not None else None,
                    })

            # OU2.5
            if "OU25" in markets:
                def ou(P):
                    grid = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
                    over = float(P[grid >= 3].sum())
                    return {"over25": over, "under25": 1.0 - over}
                ou_dc = ou(P_dc) if P_dc is not None else None
                ou_hy = ou(P_hy) if P_hy is not None else None
                ref = ou_dc or ou_hy
                side = "over25" if ref["over25"] >= ref["under25"] else "under25"
                p_dc = ou_dc.get(side) if ou_dc else None
                p_hy = ou_hy.get(side) if ou_hy else None
                pass_dc = (p_dc is None) or (p_dc >= min_conf)
                pass_hy = (p_hy is None) or (p_hy >= min_conf)
                agree = True
                if args.require_agreement and (ou_dc is not None) and (ou_hy is not None):
                    sd = "over25" if ou_dc["over25"] >= ou_dc["under25"] else "under25"
                    sh = "over25" if ou_hy["over25"] >= ou_hy["under25"] else "under25"
                    agree = (sd == sh)
                if pass_dc and pass_hy and agree:
                    picks.append({
                        "league": lg, "kickoff_utc": kickoff, "home": h, "away": a,
                        "market": "OU2.5", "pick": "Over 2.5" if side == "over25" else "Under 2.5",
                        "side": side.upper(),
                        "prob_dc": round(p_dc, 4) if p_dc is not None else None,
                        "prob_hy": round(p_hy, 4) if p_hy is not None else None,
                    })

    print("\n--- WEEKLY PICKS ---")
    if not picks:
        print("No games met the probability threshold.")
        return
    df = pd.DataFrame(picks).sort_values(["league", "kickoff_utc", "market"], kind="stable")
    print(df.to_string(index=False))
    out = f"picks_week_{dfrom}_to_{dto}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()
