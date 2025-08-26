# bet_fusion.py
# Final-bet fusion engine: blends model probabilities (DC and optional Hybrid) with form/xG/corners/ELO signals.
# Offline-first: uses local fixtures CSV (same format you've been using).
#
# Usage examples:
#   python bet_fusion.py --league E0 --from 2025-08-30 --to 2025-09-01 --source local --min-prob 0.55 --min-conf 0.65 --top-k 10
#   python bet_fusion.py --league E0 --source local --markets 1X2,DC,OU25,BTTS,INTERVALS,TEAM_GOALS,HCP --min-prob 0.60 --min-conf 0.70
#
# Output: console table + CSV "fusion_picks_<league>_<from>_to_<to>.csv"

import os
import re
import argparse
from math import factorial
from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

# ---------- config ----------
ENHANCED_PATH = os.path.join("data", "enhanced", "{league}_final_features.csv")
FIXTURES_PATH = os.path.join("data", "fixtures", "{league}_{dfrom}_to_{dto}.csv")

DC_MODEL = os.path.join("advanced_models", "{league}_dixon_coles_model.pkl")
HYB_HOME = os.path.join("advanced_models", "{league}_xgb_home_model.pkl")
HYB_AWAY = os.path.join("advanced_models", "{league}_xgb_away_model.pkl")

REPORT_CSV = "fusion_picks_{league}_{dfrom}_to_{dto}.csv"

TEAM_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton", "West Ham United": "West Ham",
    "Leeds United": "Leeds", "Leicester City": "Leicester",
    "AFC Bournemouth": "Bournemouth", "Sheffield Utd": "Sheffield United",
    # Short forms
    "Spurs": "Tottenham", "Man Utd": "Man United", "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle", "West Ham Utd": "West Ham",
    "Wolverhampton": "Wolves", "Brighton & Hove Albion": "Brighton",
    "Nottingham": "Nott'm Forest", "Nott Forest": "Nott'm Forest", "Nottm Forest": "Nott'm Forest",
}
def norm_team(x: str) -> str: return TEAM_MAP.get(x, x)

# ---------- weights (tweak here) ----------
WEIGHTS = {
    # blend between DC and Hybrid matrices
    "w_dc_matrix": 0.70,        # 0..1; 1.0 = pure DC
    # how much of final confidence is model prob vs signal bias
    "w_model_prob": 0.70,       # 0..1; 1.0 = pure model probability
    "w_signal_bias": 0.30,
    # form/tempo/elo components used to build per-market bias (scaled 0..1 via sigmoid)
    "form_w": 0.55,
    "tempo_w": 0.25,
    "elo_w": 0.20,
    # logistic slopes (controls how sharply differences change bias)
    "slope_strength": 1.6,
    "slope_tempo": 1.2,
    "slope_elo": 1.2,
}

# which markets to output by default
DEFAULT_MARKETS = ["1X2", "DC", "OU15", "OU25", "OU35", "BTTS", "INTERVALS", "TEAM_GOALS", "HCP"]

# ---------- helpers ----------
def next_tue_to_mon() -> Tuple[str, str]:
    today = date.today()
    days_to_tue = (1 - today.weekday()) % 7
    start = today + timedelta(days=days_to_tue or 7)
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cl = cand.lower()
        if cl in cols: return cols[cl]
    return None

def safe_div(a, b): 
    a = float(a) if pd.notna(a) else 0.0
    b = float(b) if pd.notna(b) else 0.0
    return a / b if b != 0.0 else 0.0

def sigmoid(x): 
    # stable sigmoid
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))

# ---------- data loaders ----------
def load_enhanced_long(league: str) -> pd.DataFrame:
    path = ENHANCED_PATH.format(league=league)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Enhanced data not found: {path}")
    base = pd.read_csv(path)

    col_date = pick_col(base, ["Date","date","MatchDate"])
    col_ht   = pick_col(base, ["HomeTeam","hometeam","home_team"])
    col_at   = pick_col(base, ["AwayTeam","awayteam","away_team"])

    col_fthg = pick_col(base, ["FTHG","HomeGoals","home_goals"])
    col_ftag = pick_col(base, ["FTAG","AwayGoals","away_goals"])

    col_hs   = pick_col(base, ["HS","HomeShots","home_shots"])
    col_as   = pick_col(base, ["AS","AwayShots","away_shots"])
    col_hst  = pick_col(base, ["HST","HomeShotsOnTarget","home_sot"])
    col_ast  = pick_col(base, ["AST","AwayShotsOnTarget","away_sot"])
    col_hc   = pick_col(base, ["HC","HomeCorners","home_corners"])
    col_ac   = pick_col(base, ["AC","AwayCorners","away_corners"])

    col_hxg  = pick_col(base, ["HxG","Home_xG","xG_Home","home_xg","HomexG"])
    col_axg  = pick_col(base, ["AxG","Away_xG","xG_Away","away_xg","AwayxG"])

    col_home_elo = pick_col(base, ["HomeElo","home_elo"])
    col_away_elo = pick_col(base, ["AwayElo","away_elo"])

    req = [col_date, col_ht, col_at, col_fthg, col_ftag]
    if any(c is None for c in req):
        raise ValueError("Enhanced CSV missing core cols: need Date, HomeTeam, AwayTeam, FTHG, FTAG (or equivalents).")

    rows = []
    for _, r in base.iterrows():
        dt = r[col_date]
        ht = norm_team(str(r[col_ht])); at = norm_team(str(r[col_at]))
        fthg = r[col_fthg]; ftag = r[col_ftag]

        hs  = r[col_hs]  if col_hs  else np.nan
        hst = r[col_hst] if col_hst else np.nan
        hc  = r[col_hc]  if col_hc  else np.nan
        as_ = r[col_as]  if col_as  else np.nan
        ast = r[col_ast] if col_ast else np.nan
        ac  = r[col_ac]  if col_ac  else np.nan

        hxg = r[col_hxg] if col_hxg else np.nan
        axg = r[col_axg] if col_axg else np.nan

        helo = r[col_home_elo] if col_home_elo else np.nan
        aelo = r[col_away_elo] if col_away_elo else np.nan

        # home row
        rows.append({"Date": dt,"team": ht,"opponent": at,"home":1,
                     "gf": fthg,"ga": ftag,
                     "shots_for": hs,"shots_against": as_,
                     "sot_for": hst,"sot_against": ast,
                     "corners_for": hc,"corners_against": ac,
                     "xg_for": hxg,"xg_against": axg,
                     "team_elo": helo,"opp_elo": aelo})
        # away row
        rows.append({"Date": dt,"team": at,"opponent": ht,"home":0,
                     "gf": ftag,"ga": fthg,
                     "shots_for": as_,"shots_against": hs,
                     "sot_for": ast,"sot_against": hst,
                     "corners_for": ac,"corners_against": hc,
                     "xg_for": axg,"xg_against": hxg,
                     "team_elo": aelo,"opp_elo": helo})
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["gf","ga","shots_for","shots_against","sot_for","sot_against","corners_for","corners_against","xg_for","xg_against","team_elo","opp_elo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["team","Date"]).reset_index(drop=True)

def load_fixtures(league: str, dfrom: str, dto: str) -> List[Dict[str,str]]:
    path = FIXTURES_PATH.format(league=league, dfrom=dfrom, dto=dto)
    if not os.path.exists(path):
        # create template
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(columns=["date","home_team","away_team"]).to_csv(path, index=False)
        print(f"[info] Created template: {path}. Fill it and re-run.")
        return []
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["date","home_team","away_team"]:
        if c not in df.columns: raise ValueError(f"Fixtures CSV missing: {c}")
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["date"].str.lower()!="date") & (df["home_team"].str.lower()!="home_team")]
    df["home_team"] = df["home_team"].map(norm_team)
    df["away_team"] = df["away_team"].map(norm_team)
    return df[["date","home_team","away_team"]].to_dict("records")

def load_dc(league: str):
    path = DC_MODEL.format(league=league)
    if not os.path.exists(path): raise FileNotFoundError(f"DC model not found: {path}")
    return joblib.load(path)

def load_hybrid(league: str):
    hp = HYB_HOME.format(league=league)
    ap = HYB_AWAY.format(league=league)
    if not (os.path.exists(hp) and os.path.exists(ap)): return None, None, None
    try:
        home = joblib.load(hp); away = joblib.load(ap)
    except Exception:
        return None, None, None
    feats_path = ENHANCED_PATH.format(league=league)
    feats = pd.read_csv(feats_path) if os.path.exists(feats_path) else None
    return home, away, feats

# ---------- models to distributions ----------
def pois_vec(k_arr, lam):
    k_arr = np.asarray(k_arr, dtype=float)
    fact = np.array([factorial(int(k)) for k in k_arr], dtype=float)
    return np.exp(-lam) * (lam ** k_arr) / fact

def score_matrix_from_mus(mu_h: float, mu_a: float, max_goals: int = 10) -> np.ndarray:
    hg = np.arange(0, max_goals+1); ag = np.arange(0, max_goals+1)
    ph = pois_vec(hg, mu_h); pa = pois_vec(ag, mu_a)
    P = np.outer(ph, pa); P /= P.sum()
    return P

def dc_matrix(dc_model, home: str, away: str, max_goals=10) -> Optional[np.ndarray]:
    try:
        import pandas as pd
        mu_h = float(dc_model.predict(pd.DataFrame({"team":[home], "opponent":[away], "home":[1]})).values[0])
        mu_a = float(dc_model.predict(pd.DataFrame({"team":[away], "opponent":[home], "home":[0]})).values[0])
        return score_matrix_from_mus(mu_h, mu_a, max_goals)
    except Exception as e:
        print(f"[debug] DC matrix failed for {home} vs {away}: {e}")
        return None

def hybrid_matrix(h_model, a_model, feats_df: Optional[pd.DataFrame], home: str, away: str, max_goals=10) -> Optional[np.ndarray]:
    if (h_model is None) or (a_model is None) or (feats_df is None): return None
    try:
        h = feats_df[feats_df["HomeTeam"]==home].tail(1).iloc[0]
        a = feats_df[feats_df["AwayTeam"]==away].tail(1).iloc[0]
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
        mu_h = float(h_model.predict(X)[0]); mu_a = float(a_model.predict(X)[0])
        return score_matrix_from_mus(mu_h, mu_a, max_goals)
    except Exception:
        return None

# ---------- derive market probabilities from matrix ----------
def probs_1x2(P: np.ndarray) -> Dict[str,float]:
    return {"home": float(np.tril(P,-1).sum()),
            "draw": float(np.diag(P).sum()),
            "away": float(np.triu(P, 1).sum())}

def prob_double_chance(p1x2: Dict[str,float]) -> Dict[str,float]:
    return {
        "1X": p1x2["home"] + p1x2["draw"],
        "12": p1x2["home"] + p1x2["away"],
        "X2": p1x2["draw"] + p1x2["away"],
    }

def prob_ou(P: np.ndarray, line: float) -> Dict[str,float]:
    grid = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
    # strict over (e.g., over 2.5) → totals >= ceil(line+0.00001)
    over = float(P[grid >= int(np.floor(line)+1)].sum()) if line%1==0.5 else float(P[grid > line].sum())
    # handle common lines explicitly
    if abs(line - 1.5) < 1e-6: over = float(P[grid >= 2].sum())
    if abs(line - 2.5) < 1e-6: over = float(P[grid >= 3].sum())
    if abs(line - 3.5) < 1e-6: over = float(P[grid >= 4].sum())
    return {"over": over, "under": 1.0 - over}

def prob_btts(P: np.ndarray) -> Dict[str,float]:
    # both teams score => i>=1 and j>=1
    i = np.arange(P.shape[0])[:,None]; j = np.arange(P.shape[1])[None,:]
    btts = float(P[(i>=1) & (j>=1)].sum())
    return {"btts_yes": btts, "btts_no": 1.0 - btts}

def prob_goals_intervals(P: np.ndarray) -> Dict[str,float]:
    grid = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
    p01 = float(P[grid <= 1].sum())
    p23 = float(P[(grid==2) | (grid==3)].sum())
    p4p = 1.0 - p01 - p23
    return {"G_0_1": p01, "G_2_3": p23, "G_4plus": p4p}

def prob_team_goals_bands(P: np.ndarray) -> Dict[str, float]:
    """
    Home 0–1 vs 2+, Away 0–1 vs 2+, from a score matrix P[home_goals, away_goals].
    Rows = home goals, columns = away goals.
    """
    n_h, n_a = P.shape
    rows_01 = (np.arange(n_h) <= 1)   # rows where home goals are 0 or 1
    cols_01 = (np.arange(n_a) <= 1)   # cols where away goals are 0 or 1

    h_01 = float(P[rows_01, :].sum())
    h_2p = float(P[~rows_01, :].sum())
    a_01 = float(P[:, cols_01].sum())
    a_2p = float(P[:, ~cols_01].sum())

    # sanity: these each partition their space
    # assert abs((h_01 + h_2p) - 1.0) < 1e-6
    # assert abs((a_01 + a_2p) - 1.0) < 1e-6

    return {"H_0_1": h_01, "H_2plus": h_2p, "A_0_1": a_01, "A_2plus": a_2p}

def prob_handicap(P: np.ndarray, line: float, side: str = "home") -> float:
    """
    Asian handicap (no half-wins/quarter-lines here): 
    Returns P( team_goals + line > opp_goals ).
    - side='home': uses (i + line) > j
    - side='away': uses (j + line) > i
    """
    i = np.arange(P.shape[0])[:, None]   # home goals
    j = np.arange(P.shape[1])[None, :]   # away goals
    if side == "home":
        mask = (i + line) > j
    else:
        mask = (j + line) > i
    return float(P[mask].sum())

# ---------- signals from form / xg / corners / elo ----------
def rolling(df_team: pd.DataFrame, n: int):
    d = df_team.tail(n)
    g = max(len(d), 1)
    return {
        "gpg": d["gf"].sum()/g, "gapg": d["ga"].sum()/g,
        "shots_pg": d["shots_for"].sum()/g, "sot_pg": d["sot_for"].sum()/g,
        "shots_allowed_pg": d["shots_against"].sum()/g, "sot_allowed_pg": d["sot_against"].sum()/g,
        "corners_pg": d["corners_for"].sum()/g, "corners_allowed_pg": d["corners_against"].sum()/g,
        "xg_pg": d["xg_for"].sum()/g if "xg_for" in d else np.nan,
        "xga_pg": d["xg_against"].sum()/g if "xg_against" in d else np.nan,
    }

def team_strength(df_long: pd.DataFrame, team: str, n: int = 10) -> Dict[str,float]:
    d = df_long[df_long["team"]==team]
    r = rolling(d, n)
    # dominance features
    xg_diff = (r["xg_pg"] - r["xga_pg"]) if pd.notna(r["xg_pg"]) and pd.notna(r["xga_pg"]) else 0.0
    goal_diff = r["gpg"] - r["gapg"]
    sot_dom = r["sot_pg"] - r["sot_allowed_pg"]
    shots_dom = r["shots_pg"] - r["shots_allowed_pg"]
    corner_dom = r["corners_pg"] - r["corners_allowed_pg"]

    # normalize by soft scales (keeps things in sensible ranges)
    def nz(v, s): return v / s if s != 0 else v
    features = {
        "xg_diff": xg_diff,                           # scale ~ [-2, +2]
        "goal_diff": nz(goal_diff, 2.0),              # [-2, +2] -> [-1, +1]
        "sot_dom": nz(sot_dom, 4.0),                  # roughly [-6, +6] -> [-1.5, +1.5]
        "shots_dom": nz(shots_dom, 10.0),             # [-20,+20] -> [-2,+2]
        "corner_dom": nz(corner_dom, 4.0),            # [-8,+8] -> [-2,+2]
    }
    # weighted sum → strength score
    strength = (0.40*features["xg_diff"] + 0.25*features["goal_diff"] +
                0.20*features["sot_dom"] + 0.10*features["shots_dom"] +
                0.05*features["corner_dom"])
    # tempo proxy (for OU markets): how frantic is the game likely?
    tempo = (abs(r["shots_pg"]) + abs(r["shots_allowed_pg"]) +
             0.7*(abs(r["sot_pg"]) + abs(r["sot_allowed_pg"])) +
             0.5*(abs(r["corners_pg"]) + abs(r["corners_allowed_pg"]))) / 40.0  # soft scale
    return {"strength": float(strength), "tempo": float(tempo)}

def recent_elo(df_long: pd.DataFrame, team: str) -> float:
    d = df_long[(df_long["team"]==team) & pd.notna(df_long["team_elo"])]
    return float(d["team_elo"].tail(1).values[0]) if len(d)>0 else np.nan

def signal_bias(home_sig: Dict[str,float], away_sig: Dict[str,float], elo_h: float, elo_a: float) -> Dict[str,float]:
    # convert diffs to 0..1 biases via sigmoids
    s_strength = WEIGHTS["slope_strength"]
    s_tempo = WEIGHTS["slope_tempo"]
    s_elo = WEIGHTS["slope_elo"]

    # Home-side bias for 1X2: >0 favors home, <0 favors away
    strength_diff = home_sig["strength"] - away_sig["strength"]
    tempo_sum = home_sig["tempo"] + away_sig["tempo"]
    elo_diff = (elo_h - elo_a) if pd.notna(elo_h) and pd.notna(elo_a) else 0.0

    b_strength = sigmoid(s_strength * strength_diff)  # 0..1
    b_tempo = sigmoid(s_tempo * (tempo_sum - 1.0))    # >1 → more likely over; center around 1.0
    b_elo_home = sigmoid(s_elo * (elo_diff / 200.0))  # 200 elo ≈ 0.73 bias

    return {"b_strength": b_strength, "b_tempo": b_tempo, "b_elo_home": b_elo_home}

# ---------- fusion ----------
def fuse_prob_with_bias(model_prob: float, bias_score: float) -> float:
    w_model = WEIGHTS["w_model_prob"]; w_bias = WEIGHTS["w_signal_bias"]
    return float(np.clip(w_model*model_prob + w_bias*bias_score, 0.0, 1.0))

# ---------- main pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Final bet fusion: model probs + weighted signals → picks")
    ap.add_argument("--league", required=True, help="League code, e.g., E0")
    ap.add_argument("--from", dest="dfrom", default=None, help="YYYY-MM-DD (default next Tue)")
    ap.add_argument("--to", dest="dto", default=None, help="YYYY-MM-DD (default following Mon)")
    ap.add_argument("--source", choices=["local"], default="local")
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS),
                    help="Comma list of markets: 1X2,DC,OU15,OU25,OU35,BTTS,INTERVALS,TEAM_GOALS,HCP")
    ap.add_argument("--min-prob", type=float, default=0.55, help="Minimum raw probability for a side (e.g., 0.55)")
    ap.add_argument("--min-conf", type=float, default=0.65, help="Minimum fused confidence (0..1)")
    ap.add_argument("--top-k", type=int, default=12, help="Max picks to output")
    ap.add_argument("--w-dc", type=float, default=WEIGHTS["w_dc_matrix"], help="Weight of DC matrix vs Hybrid (0..1)")
    args = ap.parse_args()

    dfrom = args.dfrom; dto = args.dto
    if not (dfrom and dto): dfrom, dto = next_tue_to_mon()
    markets = {m.strip().upper() for m in args.markets.split(",") if m.strip()}

    print(f"--- Fusion Engine ---\nLeague: {args.league} | Window: {dfrom} → {dto}\nMarkets: {', '.join(sorted(markets))}")
    print(f"Thresholds: min_prob={args.min_prob:.2f}  min_conf={args.min_conf:.2f}  top_k={args.top_k}")

    # Load data/models
    df_long = load_enhanced_long(args.league)
    dc = load_dc(args.league)
    hy_h, hy_a, feats = load_hybrid(args.league)

    fixtures = load_fixtures(args.league, dfrom, dto)
    if not fixtures:
        print("[warn] no fixtures in local CSV. Add rows and re-run.")
        return

    rows = []
    for fx in fixtures:
        kickoff = fx["date"]; home = fx["home_team"]; away = fx["away_team"]

        P_dc = dc_matrix(dc, home, away)
        P_hy = hybrid_matrix(hy_h, hy_a, feats, home, away) if (hy_h and hy_a and feats is not None) else None

        P = None
        if P_dc is not None and P_hy is not None:
            w = float(np.clip(args.w_dc, 0.0, 1.0))
            P = w*P_dc + (1.0 - w)*P_hy; P /= P.sum()
        elif P_dc is not None:
            P = P_dc
        elif P_hy is not None:
            P = P_hy
        else:
            print(f"[skip] No matrix for {home} vs {away}")
            continue

        # Derive baseline model probabilities
        p1x2 = probs_1x2(P)
        p_dc = prob_double_chance(p1x2)
        p_ou15 = prob_ou(P, 1.5)
        p_ou25 = prob_ou(P, 2.5)
        p_ou35 = prob_ou(P, 3.5)
        p_btts = prob_btts(P)
        p_intervals = prob_goals_intervals(P)
        p_team_bands = prob_team_goals_bands(P)

        # Signals
        hs = team_strength(df_long, home, n=10)
        as_ = team_strength(df_long, away, n=10)
        elo_h = recent_elo(df_long, home); elo_a = recent_elo(df_long, away)
        sb = signal_bias(hs, as_, elo_h, elo_a)

        # Compose a single bias for 1X2 home/away:
        bias_home = (WEIGHTS["form_w"]*sb["b_strength"] + WEIGHTS["tempo_w"]*sb["b_tempo"] + WEIGHTS["elo_w"]*sb["b_elo_home"])
        bias_away = 1.0 - bias_home  # symmetry as a first pass

        # Compose a tempo bias for OU:
        bias_over = sb["b_tempo"]; bias_under = 1.0 - bias_over

        # Build candidate picks
        def add_pick(market, side, base_prob, fused_conf):
            if (base_prob >= args.min_prob) and (fused_conf >= args.min_conf):
                rows.append({
                    "league": args.league, "kickoff_utc": kickoff, "home": home, "away": away,
                    "market": market, "side": side, "model_prob": round(base_prob,4), "conf": round(fused_conf,4),
                    "note": ""
                })

        # 1X2
        if "1X2" in markets:
            home_conf = fuse_prob_with_bias(p1x2["home"], bias_home)
            away_conf = fuse_prob_with_bias(p1x2["away"], bias_away)
            draw_conf = fuse_prob_with_bias(p1x2["draw"], 0.5)  # neutrality for draw
            add_pick("1X2", f"{home}", p1x2["home"], home_conf)
            add_pick("1X2", "Draw", p1x2["draw"], draw_conf)
            add_pick("1X2", f"{away}", p1x2["away"], away_conf)

        # Double chance
        if "DC" in markets:
            b_1x = fuse_prob_with_bias(p_dc["1X"], max(bias_home, 0.5))
            b_12 = fuse_prob_with_bias(p_dc["12"], 0.5)  # symmetric
            b_x2 = fuse_prob_with_bias(p_dc["X2"], max(bias_away, 0.5))
            add_pick("DC", "1X", p_dc["1X"], b_1x)
            add_pick("DC", "12", p_dc["12"], b_12)
            add_pick("DC", "X2", p_dc["X2"], b_x2)

        # OU lines
        if "OU15" in markets:
            add_pick("OU1.5", "Over", p_ou15["over"], fuse_prob_with_bias(p_ou15["over"], bias_over))
            add_pick("OU1.5", "Under", p_ou15["under"], fuse_prob_with_bias(p_ou15["under"], bias_under))
        if "OU25" in markets:
            add_pick("OU2.5", "Over", p_ou25["over"], fuse_prob_with_bias(p_ou25["over"], bias_over))
            add_pick("OU2.5", "Under", p_ou25["under"], fuse_prob_with_bias(p_ou25["under"], bias_under))
        if "OU35" in markets:
            add_pick("OU3.5", "Over", p_ou35["over"], fuse_prob_with_bias(p_ou35["over"], bias_over))
            add_pick("OU3.5", "Under", p_ou35["under"], fuse_prob_with_bias(p_ou35["under"], bias_under))

        # BTTS
        if "BTTS" in markets:
            add_pick("BTTS", "Yes", p_btts["btts_yes"], fuse_prob_with_bias(p_btts["btts_yes"], bias_over))
            add_pick("BTTS", "No", p_btts["btts_no"], fuse_prob_with_bias(p_btts["btts_no"], bias_under))

        # Total goals intervals
        if "INTERVALS" in markets:
            for lab, prob in p_intervals.items():
                # map interval bias: over-lean favors 2–3 and 4+, under favors 0–1
                bias = {"G_0_1": bias_under, "G_2_3": 0.5 + 0.3*(bias_over-0.5), "G_4plus": bias_over}[lab]
                add_pick("GOALS_INTERVAL", lab, prob, fuse_prob_with_bias(prob, bias))

        # Team goals bands
        if "TEAM_GOALS" in markets:
            tg = p_team_bands
            add_pick("HOME_GOALS", "0-1", tg["H_0_1"], fuse_prob_with_bias(tg["H_0_1"], 1.0 - bias_home))
            add_pick("HOME_GOALS", "2+",  tg["H_2plus"], fuse_prob_with_bias(tg["H_2plus"], bias_home))
            add_pick("AWAY_GOALS", "0-1", tg["A_0_1"], fuse_prob_with_bias(tg["A_0_1"], 1.0 - bias_away))
            add_pick("AWAY_GOALS", "2+",  tg["A_2plus"], fuse_prob_with_bias(tg["A_2plus"], bias_away))

        # Handicap (common lines)
        if "HCP" in markets:
            for line in [-1.0, -0.5, +0.5, +1.0]:
                p_home = prob_handicap(P, line, side="home")
                p_away = prob_handicap(P, line, side="away")
                add_pick(f"HCP({line:+.1f})", f"{home}", p_home, fuse_prob_with_bias(p_home, bias_home))
                add_pick(f"HCP({line:+.1f})", f"{away}", p_away, fuse_prob_with_bias(p_away, bias_away))

    if not rows:
        print("\n--- FUSION PICKS ---\nNo selections met thresholds.")
        return

    out = pd.DataFrame(rows).sort_values(["conf","model_prob"], ascending=[False, False])
    top = out.head(args.top_k).copy()

    print("\n--- FUSION PICKS (top {k}) ---".format(k=len(top)))
    with pd.option_context('display.max_rows', None, 'display.max_colwidth', 40):
        print(top[["league","kickoff_utc","home","away","market","side","model_prob","conf"]].to_string(index=False))

    path = REPORT_CSV.format(league=args.league, dfrom=dfrom, dto=dto)
    out.to_csv(path, index=False)
    print(f"\nSaved → {path}")

if __name__ == "__main__":
    main()
