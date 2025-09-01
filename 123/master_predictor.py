# master_predictor.py  (weekly 1X2 picks; odds >= 1.70; prob >= 0.70)
import os
import argparse
from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple, Iterable

import requests
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv

# ---------- load .env ----------
load_dotenv()

# ---------- config ----------
LEAGUES_TO_PREDICT: Dict[str, int] = {
    "E0": 39,  # Premier League
    "D1": 78,  # Bundesliga
}

MIN_ODDS = 1.70
MIN_CONF = 0.70  # 70%

TEAM_MAP_API_TO_BASE = {
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
}

DC_MODEL_PATH = os.path.join("advanced_models", "{league}_dixon_coles_model.pkl")
HYB_HOME_PATH = os.path.join("advanced_models", "{league}_xgb_home_model.pkl")
HYB_AWAY_PATH = os.path.join("advanced_models", "{league}_xgb_away_model.pkl")
FEATURES_PATH = os.path.join("data", "enhanced", "{league}_final_features.csv")


# ---------- helpers ----------
def get_api_key() -> str:
    for key in ("API_FOOTBALL_KEY", "API_FOOTBALL_ODDS_KEY", "API_FOOTBALL_DATA"):
        val = os.getenv(key)
        if val:
            return val
    return ""


def normalize_team(name: str) -> str:
    return TEAM_MAP_API_TO_BASE.get(name, name)


def next_7day_window() -> Tuple[str, str]:
    start = date.today()
    end = start + timedelta(days=7)
    return start.isoformat(), end.isoformat()


def season_for_window(date_from: str) -> int:
    """API-Football season is the start year of the season (e.g., 2024 for 2024/25)."""
    y, m, _ = map(int, date_from.split("-"))
    return y if m >= 7 else y - 1


def daterange(d1: str, d2: str) -> Iterable[str]:
    a = datetime.fromisoformat(d1).date()
    b = datetime.fromisoformat(d2).date()
    for n in range((b - a).days + 1):
        yield (a + timedelta(days=n)).isoformat()


def collect_all_pages(session: requests.Session, url: str, headers: dict, params: dict) -> List[dict]:
    out = []
    page = 1
    while True:
        p = dict(params, page=page)
        r = session.get(url, headers=headers, params=p, timeout=25)
        r.raise_for_status()
        j = r.json()
        out.extend(j.get("response", []))
        paging = j.get("paging", {})
        if not paging or paging.get("current", 1) >= paging.get("total", 1):
            break
        page += 1
    return out


# ---------- API-FOOTBALL fetch (fixtures + odds) ----------
def fetch_fixtures_and_matchwinner_odds(api_key: str, league_id: int, date_from: str, date_to: str) -> List[dict]:
    """
    Returns [{date, home_team, away_team, odds_h, odds_d, odds_a}, ...]
    Strategy:
      1) fixtures (status=NS & TBD) within window
      2) odds per-day (market=1) → pick best across bookmakers
      3) fallback: per-fixture odds if per-day returns nothing
    """
    headers = {"x-apisports-key": api_key}
    s = requests.Session()
    season = season_for_window(date_from)

    # 1) fixtures
    fx_url = "https://v3.football.api-sports.io/fixtures"
    statuses = ["NS", "TBD"]  # not started + date TBD
    fx_map = {}

    for status in statuses:
        fx_params = {
            "league": league_id,
            "season": season,
            "status": status,
            "from": date_from,
            "to": date_to,
            "timezone": "UTC",
        }
        try:
            fixtures = collect_all_pages(s, fx_url, headers, fx_params)
            for m in fixtures:
                fid = m["fixture"]["id"]
                fx_map[fid] = {
                    "date": m["fixture"]["date"],
                    "home_team": normalize_team(m["teams"]["home"]["name"]),
                    "away_team": normalize_team(m["teams"]["away"]["name"]),
                }
        except Exception as e:
            print(f"[debug] fixtures error ({status}): {e}")

    if not fx_map:
        print("[debug] fixtures=0 in window")
        return []

    # 2) per-day odds (market=1 = 1X2)
    odds_url = "https://v3.football.api-sports.io/odds"
    daily_odds = []
    for day in daterange(date_from, date_to):
        odds_params = {"league": league_id, "season": season, "date": day, "bet": 1}
        try:
            daily = collect_all_pages(s, odds_url, headers, odds_params)
            daily_odds.extend(daily)
        except Exception as e:
            print(f"[debug] odds error (date {day}): {e}")

    # fallback: per-fixture odds if daily came back empty
    if not daily_odds:
        for fid in fx_map.keys():
            try:
                r = s.get(odds_url, headers=headers, params={"fixture": fid, "bet": 1}, timeout=15)
                r.raise_for_status()
                daily_odds.extend(r.json().get("response", []))
            except Exception:
                continue

    if not daily_odds:
        print(f"[debug] odds=0 for league_id={league_id}, season={season}, window={date_from}->{date_to}")
        return []

    # 3) best prices per fixture across bookmakers
    best = {}
    for item in daily_odds:
        fid = item["fixture"]["id"]
        for bk in item.get("bookmakers", []):
            for bet in bk.get("bets", []):
                if bet.get("id") != 1:
                    continue
                cur = best.get(fid, {"h": 0.0, "d": 0.0, "a": 0.0})
                for v in bet.get("values", []):
                    nm, price = v.get("value"), v.get("odd")
                    if not nm or not price:
                        continue
                    key = {"Home": "h", "1": "h", "Draw": "d", "X": "d", "Away": "a", "2": "a"}.get(nm)
                    if key:
                        try:
                            val = float(price)
                            if val > cur[key]:
                                cur[key] = val
                        except Exception:
                            pass
                best[fid] = cur

    # 4) merge fixtures + odds
    out = []
    for fid, meta in fx_map.items():
        o = best.get(fid)
        if not o:
            continue
        if o["h"] and o["d"] and o["a"]:
            out.append({
                "date": meta["date"],
                "home_team": meta["home_team"],
                "away_team": meta["away_team"],
                "odds_h": o["h"],
                "odds_d": o["d"],
                "odds_a": o["a"],
            })
    if not out:
        print(f"[debug] fixtures_with_odds=0 (fixtures={len(fx_map)}, raw_odds_items={len(daily_odds)})")
    return out


# ---------- models ----------
def score_matrix_from_mus(mu_h: float, mu_a: float, max_goals: int = 10) -> np.ndarray:
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)
    ph = np.exp(-mu_h) * (mu_h ** hg) / np.vectorize(np.math.factorial)(hg)
    pa = np.exp(-mu_a) * (mu_a ** ag) / np.vectorize(np.math.factorial)(ag)
    P = np.outer(ph, pa)
    P /= P.sum()
    return P


def probs_1x2_from_matrix(P: np.ndarray) -> Dict[str, float]:
    home = np.tril(P, -1).sum()
    draw = np.diag(P).sum()
    away = np.triu(P, 1).sum()
    return {"home": float(home), "draw": float(draw), "away": float(away)}


def dc_expected_goals(model, home_team: str, away_team: str) -> Tuple[float, float]:
    import pandas as pd
    mu_h = model.predict(pd.DataFrame({"team": [home_team], "opponent": [away_team], "home": [1]})).values[0]
    mu_a = model.predict(pd.DataFrame({"team": [away_team], "opponent": [home_team], "home": [0]})).values[0]
    return float(mu_h), float(mu_a)


def build_hybrid_features_if_available(league: str):
    path = FEATURES_PATH.format(league=league)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def hyb_expected_goals(home_model, away_model, features_row) -> Tuple[float, float]:
    mu_h = float(home_model.predict(features_row)[0])
    mu_a = float(away_model.predict(features_row)[0])
    return mu_h, mu_a


def get_features_for_match(features_df: pd.DataFrame, home_team: str, away_team: str):
    try:
        h = features_df[features_df["HomeTeam"] == home_team].tail(1).iloc[0]
        a = features_df[features_df["AwayTeam"] == away_team].tail(1).iloc[0]
    except Exception:
        return None
    elo_diff = h.get("HomeElo", np.nan) - a.get("AwayElo", np.nan)
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
        "EloDifference": elo_diff,
    }
    return pd.DataFrame([vals])


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Weekly snapshot 1X2 picks (odds >= 1.70, prob >= 0.70).")
    ap.add_argument("--leagues", nargs="+", default=["E0"], help="League codes (e.g., E0 D1)")
    ap.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (default: +7 days)")
    ap.add_argument("--min-odds", type=float, default=MIN_ODDS)
    ap.add_argument("--min-conf", type=float, default=MIN_CONF)
    args = ap.parse_args()

    date_from, date_to = (args.date_from, args.date_to)
    if not date_from or not date_to:
        date_from, date_to = next_7day_window()

    api_key = get_api_key()
    if not api_key:
        print("No API key found. Provide API_FOOTBALL_KEY / API_FOOTBALL_ODDS_KEY / API_FOOTBALL_DATA.")
        return

    print("--- Running Master Prediction Pipeline ---")
    print(f"Window: {date_from} → {date_to}")
    results = []

    for lg in args.leagues:
        league_id = LEAGUES_TO_PREDICT.get(lg)
        if league_id is None:
            print(f"[skip] Unknown league code: {lg}")
            continue

        # Load models
        dc_model = None
        dc_path = DC_MODEL_PATH.format(league=lg)
        if os.path.exists(dc_path):
            try:
                dc_model = joblib.load(dc_path)
                print(f"[{lg}] loaded DC model.")
            except Exception as e:
                print(f"[{lg}] failed to load DC model: {e}")

        hyb_home = hyb_away = None
        hyb_features_df = None
        hp = HYB_HOME_PATH.format(league=lg)
        apath = HYB_AWAY_PATH.format(league=lg)
        if os.path.exists(hp) and os.path.exists(apath):
            try:
                hyb_home = joblib.load(hp)
                hyb_away = joblib.load(apath)
                hyb_features_df = build_hybrid_features_if_available(lg)
                if hyb_features_df is None:
                    print(f"[{lg}] hybrid models present but features file missing; will skip hybrid scoring.")
                else:
                    print(f"[{lg}] loaded Hybrid models + features.")
            except Exception as e:
                print(f"[{lg}] failed to load Hybrid models: {e}")
                hyb_home = hyb_away = None

        if not dc_model and not (hyb_home and hyb_away):
            print(f"[{lg}] no usable models; skipping.")
            continue

        # Fetch fixtures + odds
        fx = fetch_fixtures_and_matchwinner_odds(api_key, league_id, date_from, date_to)
        if not fx:
            print(f"[{lg}] No upcoming fixtures with odds.")
            continue

        print(f"[{lg}] fixtures_with_odds: {len(fx)}")

        # Score & filter
        for m in fx:
            h = m["home_team"]
            a = m["away_team"]
            odds = {"home": float(m["odds_h"]), "draw": float(m["odds_d"]), "away": float(m["odds_a"])}

            if max(odds.values()) < args.min_odds:
                continue

            # DC
            dc_probs = None
            if dc_model:
                try:
                    mu_h, mu_a = dc_expected_goals(dc_model, h, a)
                    P = score_matrix_from_mus(mu_h, mu_a)
                    dc_probs = probs_1x2_from_matrix(P)
                except Exception:
                    dc_probs = None

            # Hybrid (optional)
            hyb_probs = None
            if hyb_home and hyb_away and hyb_features_df is not None:
                feats = get_features_for_match(hyb_features_df, h, a)
                if feats is not None:
                    try:
                        mu_h, mu_a = hyb_expected_goals(hyb_home, hyb_away, feats)
                        P = score_matrix_from_mus(mu_h, mu_a)
                        hyb_probs = probs_1x2_from_matrix(P)
                    except Exception:
                        hyb_probs = None

            ref_probs = dc_probs or hyb_probs
            if not ref_probs:
                continue

            pick_side = max(ref_probs, key=ref_probs.get)  # "home"|"draw"|"away"
            pick_prob_dc = dc_probs[pick_side] if dc_probs else None
            pick_prob_hy = hyb_probs[pick_side] if hyb_probs else None
            pick_odds = odds[pick_side]

            if pick_odds < args.min_odds:
                continue

            pass_dc = (pick_prob_dc is None) or (pick_prob_dc >= args.min_conf)
            pass_hy = (pick_prob_hy is None) or (pick_prob_hy >= args.min_conf)
            if (pick_prob_dc is not None) and (pick_prob_hy is not None):
                if not (pass_dc and pass_hy):
                    continue
            else:
                if not (pass_dc and pass_hy):
                    continue

            results.append({
                "league": lg,
                "kickoff_utc": m["date"],
                "home": h,
                "away": a,
                "side": pick_side.upper(),
                "pick": {"home": h, "draw": "Draw", "away": a}[pick_side],
                "odds": round(pick_odds, 3),
                "dc_prob": round(pick_prob_dc, 4) if pick_prob_dc is not None else None,
                "hyb_prob": round(pick_prob_hy, 4) if pick_prob_hy is not None else None,
            })

    print("\n--- MASTER PREDICTION REPORT ---")
    if not results:
        print("No picks met the filters (odds >= 1.70 and prob >= 70%).")
        return

    df = pd.DataFrame(results)
    df = df[["league", "kickoff_utc", "home", "away", "side", "pick", "odds", "dc_prob", "hyb_prob"]]
    df = df.sort_values(["league", "kickoff_utc"])
    pd.set_option("display.max_rows", 300)
    print(df.to_string(index=False))

    out_name = f"picks_week_{date_from}_to_{date_to}.csv"
    df.to_csv(out_name, index=False)
    print(f"\nSaved → {out_name}")


if __name__ == "__main__":
    main()
