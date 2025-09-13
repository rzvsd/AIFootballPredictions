# dixon_coles_trainer_recency.py
# Train recency-weighted Poisson GLMs (Dixon-Coles style) with exponential decay.
# Saves:
#   advanced_models/{LEAGUE}_dixon_coles_model.pkl  (latest)
#   advanced_models/snapshots/{LEAGUE}_dc_{AS_OF}.pkl
#   advanced_models/snapshots/{LEAGUE}_dc_{AS_OF}.json  (metadata)

import os, json, argparse
from datetime import datetime, date
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
import patsy

ENHANCED_PATH = os.path.join("data","enhanced","{league}_final_features.csv")
SNAP_DIR = os.path.join("advanced_models","snapshots")

TEAM_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham", "Leeds United": "Leeds",
    "Leicester City": "Leicester", "AFC Bournemouth": "Bournemouth",
    "Sheffield Utd": "Sheffield United",
}
def norm_team(x): return TEAM_MAP.get(x, x)

def pick(df,cands):
    m={c.lower():c for c in df.columns}
    for c in cands:
        if c.lower() in m: return m[c.lower()]
    return None

def build_long(df):
    cd=pick(df,["Date","date","MatchDate"])
    ch=pick(df,["HomeTeam","hometeam","home_team"])
    ca=pick(df,["AwayTeam","awayteam","away_team"])
    hg=pick(df,["FTHG","HomeGoals","home_goals"])
    ag=pick(df,["FTAG","AwayGoals","away_goals"])
    req=[cd,ch,ca,hg,ag]
    if any(r is None for r in req):
        raise ValueError("Missing core columns for DC training.")
    rows=[]
    for _,r in df.iterrows():
        dt=r[cd]; ht=norm_team(str(r[ch])); at=norm_team(str(r[ca]))
        h=r[hg]; a=r[ag]
        rows.append({"Date":dt,"team":ht,"opponent":at,"home":1,"goals":h})
        rows.append({"Date":dt,"team":at,"opponent":ht,"home":0,"goals":a})
    out=pd.DataFrame(rows)
    out["Date"]=pd.to_datetime(out["Date"],errors="coerce")
    out=out.dropna(subset=["Date"])
    return out.sort_values("Date").reset_index(drop=True)

def decay_weights(dates: pd.Series, as_of: str, tau_days: float):
    asof=pd.to_datetime(as_of)
    delta=(asof - dates).dt.days.clip(lower=0).astype(float)
    return np.exp(-delta / float(tau_days))

def fit_dc(df_long, as_of:str, tau_days: float):
    df=df_long[df_long["Date"]<=pd.to_datetime(as_of)].copy()
    if df.empty: raise ValueError("No matches available up to as_of date.")
    w=decay_weights(df["Date"], as_of, tau_days)
    # statsmodels formula: goals ~ home + team + opponent
    y, X = patsy.dmatrices("goals ~ home + C(team) + C(opponent)", df, return_type="dataframe")
    model = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=w)
    res = model.fit(maxiter=200, method="newton")
    meta = {
        "as_of": as_of,
        "tau_days": tau_days,
        "train_start": str(df["Date"].min().date()),
        "train_end": str(df["Date"].max().date()),
        "n_rows": int(len(df)),
        "teams": sorted(df["team"].unique().tolist()),
    }
    return res, meta

def save_snapshot(league, as_of, res, meta):
    os.makedirs("advanced_models", exist_ok=True)
    os.makedirs(SNAP_DIR, exist_ok=True)
    latest_path=os.path.join("advanced_models", f"{league}_dixon_coles_model.pkl")
    joblib.dump({"model":res, "meta":meta}, latest_path)

    snap_pkl=os.path.join(SNAP_DIR, f"{league}_dc_{as_of}.pkl")
    snap_json=os.path.join(SNAP_DIR, f"{league}_dc_{as_of}.json")
    joblib.dump({"model":res, "meta":meta}, snap_pkl)
    with open(snap_json,"w",encoding="utf-8") as f: json.dump(meta, f, indent=2)
    print(f"[dc] saved latest: {latest_path}")
    print(f"[dc] saved snapshot: {snap_pkl}")
    print(f"[dc] meta: {meta}")

def main():
    ap=argparse.ArgumentParser(description="Train recency-weighted DC and save snapshot.")
    ap.add_argument("--league", required=True)
    ap.add_argument("--as-of", default=date.today().isoformat())
    ap.add_argument("--tau-days", type=float, default=150.0, help="Decay half-life scale (days).")
    args=ap.parse_args()

    src=ENHANCED_PATH.format(league=args.league)
    if not os.path.exists(src): raise FileNotFoundError(src)
    raw=pd.read_csv(src, encoding="utf-8-sig")
    long=build_long(raw)
    res, meta=fit_dc(long, args.as_of, args.tau_days)
    save_snapshot(args.league, args.as_of, res, meta)

if __name__=="__main__":
    main()
