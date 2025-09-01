# feature_store.py
# Build small, versioned per-team feature snapshots from your enhanced CSV.
# Writes:
#   data/store/{LEAGUE}_latest_team_stats.parquet
#   data/store/{LEAGUE}_snapshot_{AS_OF}.parquet
#
# Usage:
#   python feature_store.py --league E0 --as-of 2025-08-18
#   python feature_store.py --league E0          (defaults to today)
#
# Notes:
# - Reads: data/enhanced/{LEAGUE}_final_features.csv
# - Computes rolling L5/L10/L20 for goals/shots/SOT/corners/xG and latest Elo.
# - Strictly uses matches with Date <= as_of (no lookahead leakage).

import os
import argparse
from datetime import datetime, date
import numpy as np
import pandas as pd

STORE_DIR = os.path.join("data", "store")
ENHANCED_PATH = os.path.join("data", "enhanced", "{league}_final_features.csv")

TEAM_MAP = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham", "Leeds United": "Leeds",
    "Leicester City": "Leicester", "AFC Bournemouth": "Bournemouth",
    "Sheffield Utd": "Sheffield United",
    "Spurs": "Tottenham", "Man Utd": "Man United", "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle", "West Ham Utd": "West Ham",
    "Wolverhampton": "Wolves", "Nottingham": "Nott'm Forest",
    "Nott Forest": "Nott'm Forest", "Nottm Forest": "Nott'm Forest",
}
def norm_team(x:str)->str: return TEAM_MAP.get(x, x)

def pick_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols: return cols[c.lower()]
    return None

def to_long(df):
    col_date = pick_col(df, ["Date","date","MatchDate"])
    col_ht   = pick_col(df, ["HomeTeam","hometeam","home_team"])
    col_at   = pick_col(df, ["AwayTeam","awayteam","away_team"])
    col_fthg = pick_col(df, ["FTHG","HomeGoals","home_goals"])
    col_ftag = pick_col(df, ["FTAG","AwayGoals","away_goals"])
    col_hs   = pick_col(df, ["HS","HomeShots","home_shots"])
    col_as   = pick_col(df, ["AS","AwayShots","away_shots"])
    col_hst  = pick_col(df, ["HST","HomeShotsOnTarget","home_sot"])
    col_ast  = pick_col(df, ["AST","AwayShotsOnTarget","away_sot"])
    col_hc   = pick_col(df, ["HC","HomeCorners","home_corners"])
    col_ac   = pick_col(df, ["AC","AwayCorners","away_corners"])
    col_hxg  = pick_col(df, ["HxG","Home_xG","xG_Home","home_xg","HomexG"])
    col_axg  = pick_col(df, ["AxG","Away_xG","xG_Away","away_xg","AwayxG"])
    col_home_elo = pick_col(df, ["HomeElo","home_elo"])
    col_away_elo = pick_col(df, ["AwayElo","away_elo"])

    req = [col_date,col_ht,col_at,col_fthg,col_ftag]
    if any(c is None for c in req):
        raise ValueError("Enhanced CSV missing required columns (Date,HomeTeam,AwayTeam,FTHG,FTAG).")

    rows=[]
    for _,r in df.iterrows():
        dt=r[col_date]; ht=norm_team(str(r[col_ht])); at=norm_team(str(r[col_at]))
        fthg=r[col_fthg]; ftag=r[col_ftag]
        hs  = r[col_hs]  if col_hs  else np.nan
        hst = r[col_hst] if col_hst else np.nan
        hc  = r[col_hc]  if col_hc  else np.nan
        as_ = r[col_as]  if col_as  else np.nan
        ast = r[col_ast] if col_ast else np.nan
        ac  = r[col_ac]  if col_ac  else np.nan
        hxg = r[col_hxg] if col_hxg else np.nan
        axg = r[col_axg] if col_axg else np.nan
        helo= r[col_home_elo] if col_home_elo else np.nan
        aelo= r[col_away_elo] if col_away_elo else np.nan

        rows.append({"Date":dt,"team":ht,"opponent":at,"home":1,
                     "gf":fthg,"ga":ftag,"shots_for":hs,"shots_against":as_,
                     "sot_for":hst,"sot_against":ast,"corners_for":hc,"corners_against":ac,
                     "xg_for":hxg,"xg_against":axg,"team_elo":helo,"opp_elo":aelo})
        rows.append({"Date":dt,"team":at,"opponent":ht,"home":0,
                     "gf":ftag,"ga":fthg,"shots_for":as_,"shots_against":hs,
                     "sot_for":ast,"sot_against":hst,"corners_for":ac,"corners_against":hc,
                     "xg_for":axg,"xg_against":hxg,"team_elo":aelo,"opp_elo":helo})
    out=pd.DataFrame(rows)
    out["Date"]=pd.to_datetime(out["Date"],errors="coerce")
    for c in ["gf","ga","shots_for","shots_against","sot_for","sot_against","corners_for","corners_against","xg_for","xg_against","team_elo","opp_elo"]:
        out[c]=pd.to_numeric(out[c],errors="coerce")
    return out.sort_values(["team","Date"]).reset_index(drop=True)

def roll_stats(df_team: pd.DataFrame, n:int):
    d=df_team.tail(n)
    g=max(len(d),1)
    def s(col): return float(d[col].sum())/g if col in d and g>0 else np.nan
    gpg=s("gf"); gapg=s("ga")
    shots=s("shots_for"); shots_a=s("shots_against")
    sot=s("sot_for"); sot_a=s("sot_against")
    corners=s("corners_for"); corners_a=s("corners_against")
    xg=s("xg_for"); xga=s("xg_against")
    return {
        f"gpg_L{n}":gpg, f"gapg_L{n}":gapg,
        f"shots_L{n}":shots, f"shots_allowed_L{n}":shots_a,
        f"sot_L{n}":sot, f"sot_allowed_L{n}":sot_a,
        f"corners_L{n}":corners, f"corners_allowed_L{n}":corners_a,
        f"xg_L{n}":xg, f"xga_L{n}":xga,
        f"xgdiff_L{n}": (xg - xga) if (not np.isnan(xg) and not np.isnan(xga)) else np.nan
    }

def build_snapshot(enhanced_csv:str, as_of:str)->pd.DataFrame:
    raw=pd.read_csv(enhanced_csv, encoding="utf-8-sig")
    long=to_long(raw)
    cutoff=pd.to_datetime(as_of)+pd.Timedelta(hours=23,minutes=59,seconds=59)
    long=long[long["Date"]<=cutoff]

    teams=sorted(long["team"].dropna().unique())
    rows=[]
    for t in teams:
        d=long[long["team"]==t]
        if d.empty: continue
        last_elo = float(d["team_elo"].dropna().tail(1).values[0]) if d["team_elo"].notna().any() else np.nan
        r5 = roll_stats(d,5); r10=roll_stats(d,10); r20=roll_stats(d,20)
        row={"as_of": as_of, "team": t, "elo": last_elo}
        row.update(r5); row.update(r10); row.update(r20)
        rows.append(row)
    snap=pd.DataFrame(rows)
    return snap

def main():
    ap=argparse.ArgumentParser(description="Build per-team feature snapshots (Feature Store lite).")
    ap.add_argument("--league", required=True, help="League code, e.g. E0")
    ap.add_argument("--as-of", default=date.today().isoformat(), help="YYYY-MM-DD (inclusive).")
    args=ap.parse_args()

    os.makedirs(STORE_DIR, exist_ok=True)
    src=ENHANCED_PATH.format(league=args.league)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Enhanced file missing: {src}")

    snap=build_snapshot(src, args.as_of)
    latest_path=os.path.join(STORE_DIR, f"{args.league}_latest_team_stats.parquet")
    snap.to_parquet(latest_path, index=False)

    snap_path=os.path.join(STORE_DIR, f"{args.league}_snapshot_{args.as_of}.parquet")
    snap.to_parquet(snap_path, index=False)

    print(f"[feature-store] wrote: {latest_path}")
    print(f"[feature-store] wrote: {snap_path}")
    print(f"[feature-store] teams: {snap['team'].nunique()}  as_of: {args.as_of}")

if __name__=="__main__":
    main()
