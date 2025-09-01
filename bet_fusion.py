# bet_fusion.py (Final Corrected Version)
import os
import glob
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import poisson
from datetime import date

# --- CONFIGURATION ---
LEAGUE = "E0"
MIN_PROB = 0.55
TOP_K = 30
W_DC = 0.30
MAX_GOALS = 10

PATH_FEATURES_STRENGTH = os.path.join("data", "enhanced", f"{LEAGUE}_strength_adj.csv")
PATH_DC = os.path.join("advanced_models", f"{LEAGUE}_dixon_coles_model.pkl")
PATH_XGB_HOME = os.path.join("advanced_models", f"{LEAGUE}_ultimate_xgb_home.pkl")
PATH_XGB_AWAY = os.path.join("advanced_models", f"{LEAGUE}_ultimate_xgb_away.pkl")

TEAM_NORMALIZE = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton", "West Ham United": "West Ham",
    "AFC Bournemouth": "Bournemouth", "Sheffield Utd": "Sheffield United",
}
def norm_team(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip()
    return TEAM_NORMALIZE.get(s, s)

# --- Loaders and Helpers ---
def load_dc_model(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        for k in ("model", "fit", "dc_model"):
            if k in obj and obj[k] is not None:
                return obj[k]
        raise ValueError(f"DC bundle at {path} has no model inside.")
    return obj

def load_xgb(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

def latest_fixtures_csv(league: str) -> str:
    pat = os.path.join("data", "fixtures", f"{league}_*.csv")
    files = glob.glob(pat)
    if not files:
        raise FileNotFoundError(f"No fixtures CSV found at {pat}.")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def score_matrix_from_mus(mu_h: float, mu_a: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)
    def pois_pmf(k_arr, lam):
        k_arr = np.asarray(k_arr, dtype=float)
        fact = np.array([np.math.factorial(int(k)) for k in k_arr], dtype=float)
        return np.exp(-lam) * (lam ** k_arr) / fact
    ph = pois_pmf(hg, float(max(mu_h, 1e-6)))
    pa = pois_pmf(ag, float(max(mu_a, 1e-6)))
    P = np.outer(ph, pa)
    P /= P.sum()
    return P

def derive_all_market_probabilities(P: np.ndarray) -> dict:
    home = np.tril(P, -1).sum()
    draw = np.diag(P).sum()
    away = np.triu(P, 1).sum()
    grid = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
    over25 = float(P[grid >= 3].sum())
    prob_home_zero = P[0, :].sum()
    prob_away_zero = P[:, 0].sum()
    prob_zero_zero = P[0, 0]
    btts_no = prob_home_zero + prob_away_zero - prob_zero_zero
    btts = 1.0 - btts_no
    return {"p_H": home, "p_D": draw, "p_A": away, "p_O2.5": over25, "p_U2.5": 1.0 - over25, "p_BTTS_Y": btts, "p_BTTS_N": btts_no}

def get_dixon_coles_probs(model, home_team, away_team):
    try:
        design_info = model.model.data.design_info
        home_df = pd.DataFrame(columns=design_info.factor_names, index=[0]).fillna(0)
        home_df['team'] = home_team
        home_df['opponent'] = away_team
        home_df['home'] = 1
        away_df = pd.DataFrame(columns=design_info.factor_names, index=[0]).fillna(0)
        away_df['team'] = away_team
        away_df['opponent'] = home_team
        away_df['home'] = 0
        mu_h = float(model.predict(home_df).values[0])
        mu_a = float(model.predict(away_df).values[0])
        return mu_h, mu_a
    except Exception:
        return None, None

def get_features_for_match(features_df: pd.DataFrame, home_team: str, away_team: str):
    default_features = {
        'ShotConv': 0.1, 'ShotConvRec': 0.1, 'PointsPerGame': 1.3, 'CleanSheetStreak': 0,
        'xGDiff': 0.0, 'CornersConv': 0.05, 'CornersConvRec': 0.05, 'NumMatches': 5 
    }
    try:
        h_row = features_df[features_df["HomeTeam"] == home_team].tail(1).iloc[0]
        home_feats = {feat.replace('_H', ''): h_row.get(feat) for feat in features_df.columns if '_H' in feat}
    except IndexError:
        home_feats = default_features
    try:
        a_row = features_df[features_df["AwayTeam"] == away_team].tail(1).iloc[0]
        away_feats = {feat.replace('_A', ''): a_row.get(feat) for feat in features_df.columns if '_A' in feat}
    except IndexError:
        away_feats = default_features
    
    vals = {}
    for feat_base in default_features.keys():
        vals[f'{feat_base}_H'] = home_feats.get(feat_base, default_features[feat_base])
        vals[f'{feat_base}_A'] = away_feats.get(feat_base, default_features[feat_base])
        
    df = pd.DataFrame([vals])
    return df.astype(float)

# --- MAIN PREDICTOR ---
def main():
    print(f"--- Running Offline Fusion Predictor for League: {LEAGUE} ---")
    try:
        dc_model = load_dc_model(PATH_DC)
        xgb_home = load_xgb(PATH_XGB_HOME)
        xgb_away = load_xgb(PATH_XGB_AWAY)
        features_df = pd.read_csv(PATH_FEATURES_STRENGTH)
        fix_path = latest_fixtures_csv(LEAGUE)
        fx = pd.read_csv(fix_path)
        print("Successfully loaded all models, features, and fixtures.")
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. Details: {e}")
        return

    rows = []
    for _, r in fx.iterrows():
        home, away = norm_team(r["home_team"]), norm_team(r["away_team"])
        
        mu_h_dc, mu_a_dc = get_dixon_coles_probs(dc_model, home, away)
        
        match_features = get_features_for_match(features_df, home, away)
        if match_features is None:
            mu_h_hyb, mu_a_hyb = None, None
        else:
            mu_h_hyb = float(xgb_home.predict(match_features)[0])
            mu_a_hyb = float(xgb_away.predict(match_features)[0])

        if mu_h_dc is None and mu_h_hyb is None:
            print(f"[skip] No models could predict {home} vs {away}")
            continue
        elif mu_h_dc is None:
            P = score_matrix_from_mus(mu_h_hyb, mu_a_hyb)
            fused_note = "XGB-only"
        elif mu_h_hyb is None:
            P = score_matrix_from_mus(mu_h_dc, mu_a_dc)
            fused_note = "DC-only"
        else:
            P_dc = score_matrix_from_mus(mu_h_dc, mu_a_dc)
            P_hy = score_matrix_from_mus(mu_h_hyb, mu_a_hyb)
            P = W_DC * P_dc + (1.0 - W_DC) * P_hy
            fused_note = "DC+HYB"

        probs = derive_all_market_probabilities(P)
        
        picks = []
        if probs["p_H"] >= MIN_PROB:
            picks.append(("1X2","HOME", probs["p_H"]))
        if probs["p_A"] >= MIN_PROB:
            picks.append(("1X2","AWAY", probs["p_A"]))
        if probs["p_O2.5"] >= MIN_PROB:
            picks.append(("OU2.5","OVER", probs["p_O2.5"]))
        if probs["p_BTTS_Y"] >= MIN_PROB:
            picks.append(("BTTS","YES", probs["p_BTTS_Y"]))

        for mkt, side, p in picks:
            rows.append({"league": LEAGUE, "kickoff_utc": r["date"], "home": home, "away": away,
                         "market": mkt, "side": side, "prob": round(float(p), 4), "fusion": fused_note})

    if not rows:
        print("\nNo picks met the probability threshold.")
        return

    out = pd.DataFrame(rows).sort_values("prob", ascending=False)
    print(f"\n--- FUSION PICKS (top {min(TOP_K, len(out))}) ---")
    print(out.head(TOP_K).to_string(index=False))

if __name__ == "__main__":
    main()