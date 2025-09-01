# bet_fusion.py
# Offline fusion picks: Dixon–Coles + (optional) Hybrid (XGB) with safe fallbacks.
# - Uses the newest fixtures CSV in data/fixtures/{LEAGUE}_*.csv (shows which file).
# - Loads DC bundle safely (fitted results), and XGB models if present.
# - If a fixture’s teams were not in DC training, we try Hybrid-only (if features exist).
# - Prints clear counts for usable / skipped fixtures and saves picks CSV.

import os
import glob
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Set

# -------------------------------
# CONFIG
LEAGUE = "E0"
MIN_PROB = 0.30          # minimum market probability to print/save
TOP_K = 30               # max rows to print
W_DC = 0.30              # weight on DC when fusing with Hybrid
MAX_GOALS = 10           # grid size for score matrix

PATH_FEATURES = os.path.join("data", "enhanced", f"{LEAGUE}_final_features.csv")
PATH_FEATURES_STRENGTH = os.path.join("data", "enhanced", f"{LEAGUE}_strength_adj.csv")  # optional extra columns
PATH_DC = os.path.join("advanced_models", f"{LEAGUE}_dixon_coles_model.pkl")
PATH_XGB_HOME = os.path.join("advanced_models", f"{LEAGUE}_xgb_home.pkl")
PATH_XGB_AWAY = os.path.join("advanced_models", f"{LEAGUE}_xgb_away.pkl")

XGB_FEATURES = [
    'HomeAvgGoalsScored_Last5', 'HomeAvgGoalsConceded_Last5',
    'AwayAvgGoalsScored_Last5', 'AwayAvgGoalsConceded_Last5',
    'HomeAvgXgFor_Last5', 'HomeAvgXgAgainst_Last5',
    'AwayAvgXgFor_Last5', 'AwayAvgXgAgainst_Last5',
    'HomeAvgShots_Last5', 'AwayAvgShots_Last5',
    'HomeAvgShotsOnTarget_Last5', 'AwayAvgShotsOnTarget_Last5',
    'HomeAvgCorners_Last5', 'AwayAvgCorners_Last5',
    'EloDifference'
]

TEAM_NORMALIZE = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton",
    "Sheffield Utd": "Sheffield United",
    "AFC Bournemouth": "Bournemouth",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Nott Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
    "Man Utd": "Man United",
    "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle",
    "West Ham Utd": "West Ham",
    "Spurs": "Tottenham",
}
def norm_team(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip()
    return TEAM_NORMALIZE.get(s, s)

# -------------------------------
# Loaders

def load_dc_model(path: str):
    """Return a *fitted* statsmodels GLM results object from a saved bundle."""
    obj = joblib.load(path)

    def is_fitted_results(x):
        return hasattr(x, "predict") and hasattr(x, "model") and hasattr(x, "params")

    if isinstance(obj, dict):
        for k in ("fit", "result", "results", "res", "dc_model", "model"):
            x = obj.get(k)
            if x is not None and is_fitted_results(x):
                return x
        raise ValueError(
            f"DC bundle at {path} lacks fitted results "
            "(looked for: fit/result/results/res/dc_model/model)."
        )
    if is_fitted_results(obj):
        return obj
    raise ValueError(f"Object in {path} isn’t a fitted DC results object.")

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
        raise FileNotFoundError(f"No fixtures CSV found at {pat}. Create one in data/fixtures/")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

# -------------------------------
# Math helpers

def score_matrix_from_mus(mu_h: float, mu_a: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)

    def pois_pmf(k_arr, lam):
        k_arr = np.asarray(k_arr, dtype=float)
        fact = np.array([np.math.factorial(int(k)) for k in k_arr], dtype=float)
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            out = np.exp(-lam) * (lam ** k_arr) / fact
            out[np.isnan(out)] = 0.0
        return out

    ph = pois_pmf(hg, float(max(mu_h, 1e-6)))
    pa = pois_pmf(ag, float(max(mu_a, 1e-6)))
    P = np.outer(ph, pa)
    s = P.sum()
    return P / s if s > 0 else np.full((max_goals+1, max_goals+1), 1.0/((max_goals+1)**2))

def derive_all_market_probabilities(P: np.ndarray) -> dict:
    s = P.sum()
    if s <= 0 or not np.isfinite(s):
        return {"p_H": 1/3, "p_D": 1/3, "p_A": 1/3, "p_O2.5": 0.5, "p_U2.5": 0.5, "p_BTTS_Y": 0.5, "p_BTTS_N": 0.5}
    P = P / s

    home = float(np.tril(P, -1).sum())
    draw = float(np.diag(P).sum())
    away = float(np.triu(P, 1).sum())

    tot_goals = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
    over25 = float(P[tot_goals >= 3].sum())

    # BTTS via complement
    p_row0 = float(P[0, :].sum())
    p_col0 = float(P[:, 0].sum())
    btts = 1.0 - (p_row0 + p_col0 - P[0, 0])

    return {"p_H": home, "p_D": draw, "p_A": away,
            "p_O2.5": over25, "p_U2.5": 1.0 - over25,
            "p_BTTS_Y": btts, "p_BTTS_N": 1.0 - btts}

# -------------------------------
# DC helpers (robust categories)

def get_training_team_levels(dc_model) -> Optional[Set[str]]:
    """Union of team/opponent levels from model training frame."""
    try:
        frame = dc_model.model.data.frame
        if "team" in frame.columns and "opponent" in frame.columns:
            return set(str(x) for x in pd.unique(pd.concat([frame["team"], frame["opponent"]], ignore_index=True)))
    except Exception:
        pass
    return None

def dc_expected_goals(dc_model, home_team: str, away_team: str):
    levels = get_training_team_levels(dc_model)
    if not levels:
        # fallback direct predict (may fail on unseen levels)
        mu_h = float(dc_model.predict(pd.DataFrame({'team':[home_team],'opponent':[away_team],'home':[1]})).values[0])
        mu_a = float(dc_model.predict(pd.DataFrame({'team':[away_team],'opponent':[home_team],'home':[0]})).values[0])
        return mu_h, mu_a

    if home_team not in levels or away_team not in levels:
        raise ValueError(f"One of the teams ('{home_team}' or '{away_team}') was not in the training data.")

    def build(team, opp, home_flag):
        df = pd.DataFrame({"team":[team], "opponent":[opp], "home":[home_flag]})
        df["team"] = pd.Categorical(df["team"], categories=sorted(levels))
        df["opponent"] = pd.Categorical(df["opponent"], categories=sorted(levels))
        return df

    mu_h = float(dc_model.predict(build(home_team, away_team, 1)).values[0])
    mu_a = float(dc_model.predict(build(away_team, home_team, 0)).values[0])
    return mu_h, mu_a

# -------------------------------
# Hybrid helpers

def case_insensitive_map(df_cols):
    return {c.lower(): c for c in df_cols}

def extract_xgb_row(enh_df, home: str, away: str, cutoff_ts: pd.Timestamp) -> Optional[pd.DataFrame]:
    ci = case_insensitive_map(enh_df.columns)
    def col(name): return ci.get(name.lower())

    mask_home = ((enh_df.get(col("HomeTeam")) == home) | (enh_df.get(col("AwayTeam")) == home)) & (enh_df.get(col("Date")) < cutoff_ts)
    mask_away = ((enh_df.get(col("HomeTeam")) == away) | (enh_df.get(col("AwayTeam")) == away)) & (enh_df.get(col("Date")) < cutoff_ts)

    if mask_home.any():
        last_home = enh_df.loc[mask_home].sort_values(col("Date")).iloc[-1]
    else:
        return None
    if mask_away.any():
        last_away = enh_df.loc[mask_away].sort_values(col("Date")).iloc[-1]
    else:
        return None

    def pick(prefix, row):
        def g(base):
            for cand in [f"{prefix}{base}", f"{prefix}{base}".replace("Xg","XG")]:
                c = col(cand)
                if c in row.index: return row[c]
            return np.nan
        return {
            "AvgGoalsScored_Last5": g("AvgGoalsScored_Last5"),
            "AvgGoalsConceded_Last5": g("AvgGoalsConceded_Last5"),
            "AvgXgFor_Last5": g("AvgXgFor_Last5"),
            "AvgXgAgainst_Last5": g("AvgXgAgainst_Last5"),
            "AvgShots_Last5": g("AvgShots_Last5"),
            "AvgShotsOnTarget_Last5": g("AvgShotsOnTarget_Last5"),
            "AvgCorners_Last5": g("AvgCorners_Last5"),
        }

    home_feats = pick("Home" if (last_home.get(col("HomeTeam")) == home) else "Away", last_home)
    away_feats = pick("Home" if (last_away.get(col("HomeTeam")) == away) else "Away", last_away)

    data = {
        'HomeAvgGoalsScored_Last5': home_feats["AvgGoalsScored_Last5"],
        'HomeAvgGoalsConceded_Last5': home_feats["AvgGoalsConceded_Last5"],
        'AwayAvgGoalsScored_Last5': away_feats["AvgGoalsScored_Last5"],
        'AwayAvgGoalsConceded_Last5': away_feats["AvgGoalsConceded_Last5"],
        'HomeAvgXgFor_Last5': home_feats["AvgXgFor_Last5"],
        'HomeAvgXgAgainst_Last5': home_feats["AvgXgAgainst_Last5"],
        'AwayAvgXgFor_Last5': away_feats["AvgXgFor_Last5"],
        'AwayAvgXgAgainst_Last5': away_feats["AvgXgAgainst_Last5"],
        'HomeAvgShots_Last5': home_feats["AvgShots_Last5"],
        'AwayAvgShots_Last5': away_feats["AvgShots_Last5"],
        'HomeAvgShotsOnTarget_Last5': home_feats["AvgShotsOnTarget_Last5"],
        'AwayAvgShotsOnTarget_Last5': away_feats["AvgShotsOnTarget_Last5"],
        'HomeAvgCorners_Last5': home_feats["AvgCorners_Last5"],
        'AwayAvgCorners_Last5': away_feats["AvgCorners_Last5"],
        'EloDifference': float(last_home.get(col("EloDifference"))) if col("EloDifference") in last_home.index and pd.notna(last_home.get(col("EloDifference"))) else 0.0
    }

    X = pd.DataFrame([data])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X.isna().any().any():
        return None
    return X

# -------------------------------
def main():
    print(f"--- Running Offline Fusion Predictor for League: {LEAGUE} ---")

    # Load models
    dc_model = load_dc_model(PATH_DC)
    xgb_home = load_xgb(PATH_XGB_HOME)
    xgb_away = load_xgb(PATH_XGB_AWAY)

    # Load features
    if not os.path.exists(PATH_FEATURES):
        raise FileNotFoundError(f"Enhanced features not found: {PATH_FEATURES}")
    enh = pd.read_csv(PATH_FEATURES)

    # optional strength-adjusted merge
    if os.path.exists(PATH_FEATURES_STRENGTH):
        try:
            sa = pd.read_csv(PATH_FEATURES_STRENGTH)
            enh["Date"] = pd.to_datetime(enh.get("Date"), errors="coerce", utc=True)
            sa["Date"] = pd.to_datetime(sa.get("Date"), errors="coerce", utc=True)
            enh = enh.merge(sa, on=["Date","HomeTeam","AwayTeam"], how="left", suffixes=("","_sa"))
        except Exception:
            pass

    # Normalize names + parse dates
    enh["HomeTeam"] = enh["HomeTeam"].map(norm_team)
    enh["AwayTeam"] = enh["AwayTeam"].map(norm_team)
    enh["Date"] = pd.to_datetime(enh["Date"], errors="coerce", utc=True)

    teams_in_features = set(pd.unique(pd.concat([enh["HomeTeam"], enh["AwayTeam"]], ignore_index=True).dropna().astype(str)))

    # Load latest fixtures
    fix_path = latest_fixtures_csv(LEAGUE)
    print(f"[info] fixtures file: {fix_path}")
    fx = pd.read_csv(fix_path, dtype=str)
    need_cols = {"date","home_team","away_team"}
    if not need_cols.issubset({c.lower() for c in fx.columns}):
        raise SystemExit(f"Fixtures CSV missing required columns {need_cols}. Got: {list(fx.columns)}")
    fx_cols = {c.lower(): c for c in fx.columns}
    fx["date"] = pd.to_datetime(fx[fx_cols["date"]], errors="coerce", utc=True)
    fx["home_team"] = fx[fx_cols["home_team"]].map(norm_team)
    fx["away_team"] = fx[fx_cols["away_team"]].map(norm_team)
    fx = fx.dropna(subset=["date","home_team","away_team"]).sort_values("date")

    levels_dc = get_training_team_levels(dc_model) or set()
    if not isinstance(levels_dc, set):
        levels_dc = set(levels_dc)

    total = len(fx)
    rows = []
    skipped_dc_unknown = 0
    skipped_no_model = 0

    for _, r in fx.iterrows():
        dt = r["date"]; home = r["home_team"]; away = r["away_team"]

        mu_h_dc = mu_a_dc = None
        mu_h_hyb = mu_a_hyb = None
        used_dc = used_hyb = False

        # Try DC if both teams are in training levels
        if home in levels_dc and away in levels_dc:
            try:
                mu_h_dc, mu_a_dc = dc_expected_goals(dc_model, home, away)
                used_dc = True
            except Exception as e:
                # DC unavailable for this match
                pass
        else:
            skipped_dc_unknown += 1

        # Try Hybrid if both teams exist in features and models are present
        if xgb_home is not None and xgb_away is not None and home in teams_in_features and away in teams_in_features:
            X_row = extract_xgb_row(enh, home, away, dt)
            if X_row is not None:
                try:
                    X_row = X_row[XGB_FEATURES].astype(float)
                    mu_h_hyb = float(xgb_home.predict(X_row)[0])
                    mu_a_hyb = float(xgb_away.predict(X_row)[0])
                    used_hyb = True
                except Exception:
                    used_hyb = False

        if not used_dc and not used_hyb:
            skipped_no_model += 1
            continue

        # Build matrix
        if used_dc and used_hyb:
            P_dc = score_matrix_from_mus(mu_h_dc, mu_a_dc)
            P_hy = score_matrix_from_mus(mu_h_hyb, mu_a_hyb)
            P = W_DC * P_dc + (1.0 - W_DC) * P_hy
            pred_xg_h = 0.5 * (mu_h_dc + mu_h_hyb)
            pred_xg_a = 0.5 * (mu_a_dc + mu_a_hyb)
            fused_note = "DC+HYB"
        elif used_dc:
            P = score_matrix_from_mus(mu_h_dc, mu_a_dc)
            pred_xg_h, pred_xg_a = mu_h_dc, mu_a_dc
            fused_note = "DC-only"
        else:  # used_hyb
            P = score_matrix_from_mus(mu_h_hyb, mu_a_hyb)
            pred_xg_h, pred_xg_a = mu_h_hyb, mu_a_hyb
            fused_note = "HYB-only"

        probs = derive_all_market_probabilities(P)

        picks = []
        if probs["p_H"] >= MIN_PROB: picks.append(("1X2","HOME", probs["p_H"]))
        if probs["p_A"] >= MIN_PROB: picks.append(("1X2","AWAY", probs["p_A"]))
        if probs["p_O2.5"] >= MIN_PROB: picks.append(("OU2.5","OVER", probs["p_O2.5"]))
        if probs["p_BTTS_Y"] >= MIN_PROB: picks.append(("BTTS","YES", probs["p_BTTS_Y"]))

        for mkt, side, p in picks:
            rows.append({
                "league": LEAGUE,
                "kickoff_utc": dt.isoformat().replace("+00:00","Z"),
                "home": home,
                "away": away,
                "market": mkt,
                "side": side,
                "prob": round(float(p), 4),
                "Pred_xG_H": round(float(pred_xg_h), 3),
                "Pred_xG_A": round(float(pred_xg_a), 3),
                "fusion": fused_note
            })

    print("Successfully loaded all models, features, and fixtures.")
    print(f"[info] fixtures total: {total} | dc-unknown teams: {skipped_dc_unknown} | skipped (no usable model): {skipped_no_model}")

    if not rows:
        print("\nNo picks met the probability threshold.")
        return

    out = pd.DataFrame(rows).sort_values(["prob","kickoff_utc"], ascending=[False, True])
    print("\n--- FUSION PICKS (top {}) ---".format(min(TOP_K, len(out))))
    cols = ["league","kickoff_utc","home","away","market","side","prob","Pred_xG_H","Pred_xG_A","fusion"]
    print(out.head(TOP_K)[cols].to_string(index=False))

    base = os.path.splitext(os.path.basename(fix_path))[0]
    out_path = os.path.join("data","picks", f"{base}_picks.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
