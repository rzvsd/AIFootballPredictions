# strength_enhancer.py
# Build strength-adjusted features using opponent-ELO proximity (Gaussian kernel)
# and recency (exponential time decay). Writes a merged CSV with new columns.
#
# Usage (from project root):
#   python strength_enhancer.py --league E0 \
#     --in data/enhanced/E0_final_features.csv \
#     --out data/enhanced/E0_strength_adj.csv \
#     --sigma 30 --tau 180
#
# If --in is omitted, it tries (in order):
#   data/enhanced/{LEAGUE}_final_features.csv
#   data/processed/{LEAGUE}.csv
#
# Required columns (case-insensitive, aliases supported below):
#   Date, HomeTeam, AwayTeam,
#   FTHG, FTAG (or HomeGoals/AwayGoals),
#   HomeShots, AwayShots, HomeShotsOnTarget, AwayShotsOnTarget,
#   HomeCorners, AwayCorners,
#   Home_xG, Away_xG,
#   HomeElo, AwayElo
#
# Outputs: original columns + the following per side (H/A):
#   ShotConv_H/A                 (goals / shots)
#   ShotConvRec_H/A             (conceded goals / shots allowed)
#   PointsPerGame_H/A
#   CleanSheetStreak_H/A        (consecutive past CS before match)
#   xGDiff_H/A                  (xG_for - xG_against)
#   CornersConv_H/A             (goals / corners)
#   CornersConvRec_H/A          (conceded / corners_against)
#   NumMatches_H/A              (# of past matches used with non-zero weight)
#
# Notes:
# - Uses only matches strictly BEFORE each match date (no leakage).
# - If ELO is missing, falls back to time decay only; if weights ~0, falls back to simple recent avg.
# - Adjust TEAM_NORMALIZE mapping if your names differ.

import os
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# -------------------------
# Team-name normalization (edit/extend if needed)
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
def norm_team(s):
    if pd.isna(s): return s
    s = str(s).strip()
    return TEAM_NORMALIZE.get(s, s)

# -------------------------
# Column alias map (case-sensitive to your CSV; we resolve case-insensitively)
ALIAS = {
    "date": ["Date", "date", "kickoff_utc", "kickoff"],
    "home": ["HomeTeam", "home_team", "home"],
    "away": ["AwayTeam", "away_team", "away"],

    "hg":   ["FTHG", "HomeGoals", "home_goals", "HG"],
    "ag":   ["FTAG", "AwayGoals", "away_goals", "AG"],

    "hs":   ["HomeShots", "HS", "home_shots"],
    "as":   ["AwayShots", "AS", "away_shots"],

    "hst":  ["HomeShotsOnTarget", "HST", "home_sot"],
    "ast":  ["AwayShotsOnTarget", "AST", "away_sot"],

    "hc":   ["HomeCorners", "HC", "corners_home", "home_corners"],
    "ac":   ["AwayCorners", "AC", "corners_away", "away_corners"],

    "hxg":  ["Home_xG", "xGHome", "xG_Home", "home_xg"],
    "axg":  ["Away_xG", "xGAway", "xG_Away", "away_xg"],

    "helo": ["HomeElo", "home_elo", "EloHome", "elo_home"],
    "aelo": ["AwayElo", "away_elo", "EloAway", "elo_away"],
}

def _resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    # case-insensitive lookup
    lower_map = {c.lower(): c for c in df.columns}
    resolved = {}
    for key, candidates in ALIAS.items():
        found = None
        for cand in candidates:
            lc = cand.lower()
            if lc in lower_map:
                found = lower_map[lc]
                break
        resolved[key] = found
    need_min = ["date", "home", "away", "hg", "ag"]
    missing = [k for k in need_min if resolved[k] is None]
    if missing:
        raise ValueError(f"Input is missing required core columns (any alias): {missing}\n"
                         f"Present columns: {list(df.columns)[:10]} ...")
    return resolved

# -------------------------
def elo_kernel(dist: float, sigma: float) -> float:
    return float(np.exp(-0.5 * (dist / sigma) ** 2))

def time_weight(age_days: float, tau_days: float) -> float:
    return float(np.exp(-age_days / tau_days))

def _safe_div(num, den):
    num = float(num) if pd.notna(num) else np.nan
    den = float(den) if pd.notna(den) else np.nan
    if den is None or np.isnan(den) or den <= 0:
        return np.nan
    return num / den

# -------------------------
def compute_strength_adjusted(
    df_in: pd.DataFrame,
    sigma: float = 30.0,
    tau_days: float = 180.0,
    min_matches: int = 3,
    max_hist: int = 200,
) -> pd.DataFrame:
    cols = _resolve_columns(df_in)
    df = df_in.copy()

    # Normalize team names early
    df[cols["home"]] = df[cols["home"]].map(norm_team)
    df[cols["away"]] = df[cols["away"]].map(norm_team)

    # Parse date and sort chronologically
    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce", utc=True)
    if df[cols["date"]].isna().any():
        raise ValueError("Some dates could not be parsed as UTC timestamps.")
    df = df.sort_values(cols["date"]).reset_index(drop=True)

    # Numeric conversions (optional columns)
    for k in ["hg","ag","hs","as","hst","ast","hc","ac","hxg","axg","helo","aelo"]:
        c = cols.get(k)
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build per-team history as list of dicts
    team_hist: Dict[str, List[Dict]] = {}

    out_rows = []
    N = len(df)
    for i, r in df.iterrows():
        dt  = r[cols["date"]]
        home = r[cols["home"]]; away = r[cols["away"]]

        # opponent ELO at *this* match time (center for kernel)
        helo = r[cols["helo"]] if cols["helo"] else np.nan
        aelo = r[cols["aelo"]] if cols["aelo"] else np.nan

        # container for this match features
        feats = {}

        for side, team, opp_center_elo, my_goals, opp_goals, my_shots, opp_shots, my_xg, opp_xg, my_corners, opp_corners in [
            ("H", home, aelo,
             r.get(cols["hg"]), r.get(cols["ag"]),
             r.get(cols["hs"]), r.get(cols["as"]),
             r.get(cols["hxg"]), r.get(cols["axg"]),
             r.get(cols["hc"]), r.get(cols["ac"])
            ),
            ("A", away, helo,
             r.get(cols["ag"]), r.get(cols["hg"]),
             r.get(cols["as"]), r.get(cols["hs"]),
             r.get(cols["axg"]), r.get(cols["hxg"]),
             r.get(cols["ac"]), r.get(cols["hc"])
            ),
        ]:
            hist = team_hist.get(team, [])
            if not hist:
                feats.update({
                    f"ShotConv_{side}": np.nan,
                    f"ShotConvRec_{side}": np.nan,
                    f"PointsPerGame_{side}": np.nan,
                    f"CleanSheetStreak_{side}": 0,
                    f"xGDiff_{side}": np.nan,
                    f"CornersConv_{side}": np.nan,
                    f"CornersConvRec_{side}": np.nan,
                    f"NumMatches_{side}": 0,
                })
            else:
                w_list = []
                vals = {
                    "gf": [], "ga": [], "sf": [], "sa": [],
                    "xgf": [], "xga": [], "cf": [], "ca": [],
                    "pts": [], "cs": []
                }
                used = 0
                for p in hist[-max_hist:]:
                    if p["date"] >= dt:  # only strictly earlier matches
                        continue
                    age_days = (dt - p["date"]).days
                    if pd.isna(opp_center_elo) or pd.isna(p["opp_elo"]):
                        w = time_weight(age_days, tau_days)
                    else:
                        w = elo_kernel(p["opp_elo"] - opp_center_elo, sigma) * time_weight(age_days, tau_days)
                    w_list.append(w)
                    vals["gf"].append(p["gf"]);  vals["ga"].append(p["ga"])
                    vals["sf"].append(p["sf"]);  vals["sa"].append(p["sa"])
                    vals["xgf"].append(p["xgf"]); vals["xga"].append(p["xga"])
                    vals["cf"].append(p["cf"]);  vals["ca"].append(p["ca"])
                    vals["pts"].append(p["pts"]); vals["cs"].append(1 if p["ga"] == 0 else 0)
                    used += 1

                w = np.array(w_list, dtype=float)
                if used == 0:
                    feats.update({
                        f"ShotConv_{side}": np.nan,
                        f"ShotConvRec_{side}": np.nan,
                        f"PointsPerGame_{side}": np.nan,
                        f"CleanSheetStreak_{side}": 0,
                        f"xGDiff_{side}": np.nan,
                        f"CornersConv_{side}": np.nan,
                        f"CornersConvRec_{side}": np.nan,
                        f"NumMatches_{side}": 0,
                    })
                elif np.nansum(w) <= 0 or np.count_nonzero(np.isfinite(w)) < min_matches:
                    # fallback: simple recent average on last min_matches
                    simple = hist[-min(len(hist), max(min_matches, 1)):]
                    gf = np.nanmean([p["gf"]  for p in simple])
                    ga = np.nanmean([p["ga"]  for p in simple])
                    sf = np.nanmean([p["sf"]  for p in simple])
                    sa = np.nanmean([p["sa"]  for p in simple])
                    xgf= np.nanmean([p["xgf"] for p in simple])
                    xga= np.nanmean([p["xga"] for p in simple])
                    cf = np.nanmean([p["cf"]  for p in simple])
                    ca = np.nanmean([p["ca"]  for p in simple])
                    pts= np.nanmean([p["pts"] for p in simple])
                    cs_streak = 0
                    for p in reversed(simple):
                        if p["date"] >= dt: continue
                        if p["ga"] == 0: cs_streak += 1
                        else: break
                    feats.update({
                        f"ShotConv_{side}": _safe_div(gf, sf),
                        f"ShotConvRec_{side}": _safe_div(ga, sa),
                        f"PointsPerGame_{side}": pts,
                        f"CleanSheetStreak_{side}": cs_streak,
                        f"xGDiff_{side}": (xgf - xga) if pd.notna(xgf) and pd.notna(xga) else np.nan,
                        f"CornersConv_{side}": _safe_div(gf, cf),
                        f"CornersConvRec_{side}": _safe_div(ga, ca),
                        f"NumMatches_{side}": int(len(simple)),
                    })
                else:
                    wsum = float(np.nansum(w))
                    def wmean(arr):
                        a = np.array(arr, dtype=float)
                        mask = np.isfinite(a) & np.isfinite(w)
                        if not mask.any(): return np.nan
                        return float(np.nansum(a[mask]*w[mask]) / np.nansum(w[mask]))
                    gf = wmean(vals["gf"]); ga = wmean(vals["ga"])
                    sf = wmean(vals["sf"]); sa = wmean(vals["sa"])
                    xgf= wmean(vals["xgf"]); xga= wmean(vals["xga"])
                    cf = wmean(vals["cf"]); ca = wmean(vals["ca"])
                    pts= wmean(vals["pts"])
                    # clean-sheet streak (unweighted, most recent backwards)
                    cs_streak = 0
                    for p in reversed(hist):
                        if p["date"] >= dt: continue
                        if p["ga"] == 0: cs_streak += 1
                        else: break
                    feats.update({
                        f"ShotConv_{side}": _safe_div(gf, sf),
                        f"ShotConvRec_{side}": _safe_div(ga, sa),
                        f"PointsPerGame_{side}": pts,
                        f"CleanSheetStreak_{side}": int(cs_streak),
                        f"xGDiff_{side}": (xgf - xga) if pd.notna(xgf) and pd.notna(xga) else np.nan,
                        f"CornersConv_{side}": _safe_div(gf, cf),
                        f"CornersConvRec_{side}": _safe_div(ga, ca),
                        f"NumMatches_{side}": int(np.count_nonzero(w > 0)),
                    })

        out_rows.append({
            "Date": dt,
            "HomeTeam": home,
            "AwayTeam": away,
            **feats
        })

        # append this match to team histories for future matches
        # build perspective records
        def _mk(val):
            return float(val) if pd.notna(val) else np.nan

        home_rec = {
            "date": dt,
            "opp_elo": _mk(aelo),
            "gf": _mk(r.get(cols["hg"])),
            "ga": _mk(r.get(cols["ag"])),
            "sf": _mk(r.get(cols["hs"])) if cols["hs"] else np.nan,
            "sa": _mk(r.get(cols["as"])) if cols["as"] else np.nan,
            "xgf": _mk(r.get(cols["hxg"])) if cols["hxg"] else np.nan,
            "xga": _mk(r.get(cols["axg"])) if cols["axg"] else np.nan,
            "cf": _mk(r.get(cols["hc"])) if cols["hc"] else np.nan,
            "ca": _mk(r.get(cols["ac"])) if cols["ac"] else np.nan,
            "pts": 3.0 if (pd.notna(r.get(cols["hg"])) and pd.notna(r.get(cols["ag"])) and r[cols["hg"]] > r[cols["ag"]])
                    else (1.0 if (pd.notna(r.get(cols["hg"])) and pd.notna(r.get(cols["ag"])) and r[cols["hg"]] == r[cols["ag"]]) else 0.0)
        }
        away_rec = {
            "date": dt,
            "opp_elo": _mk(helo),
            "gf": _mk(r.get(cols["ag"])),
            "ga": _mk(r.get(cols["hg"])),
            "sf": _mk(r.get(cols["as"])) if cols["as"] else np.nan,
            "sa": _mk(r.get(cols["hs"])) if cols["hs"] else np.nan,
            "xgf": _mk(r.get(cols["axg"])) if cols["axg"] else np.nan,
            "xga": _mk(r.get(cols["hxg"])) if cols["hxg"] else np.nan,
            "cf": _mk(r.get(cols["ac"])) if cols["ac"] else np.nan,
            "ca": _mk(r.get(cols["hc"])) if cols["hc"] else np.nan,
            "pts": 3.0 if (pd.notna(r.get(cols["ag"])) and pd.notna(r.get(cols["hg"])) and r[cols["ag"]] > r[cols["hg"]])
                    else (1.0 if (pd.notna(r.get(cols["ag"])) and pd.notna(r.get(cols["hg"])) and r[cols["ag"]] == r[cols["hg"]]) else 0.0)
        }

        team_hist.setdefault(home, []).append(home_rec)
        team_hist.setdefault(away, []).append(away_rec)

    out_df = pd.DataFrame(out_rows)
    # Merge back to original df by Date+teams
    merged = df.merge(out_df, left_on=[cols["date"], cols["home"], cols["away"]],
                      right_on=["Date", "HomeTeam", "AwayTeam"], how="left")

    return merged

# -------------------------
def guess_input_path(league: str) -> Optional[str]:
    candidates = [
        os.path.join("data", "enhanced", f"{league}_final_features.csv"),
        os.path.join("data", "processed", f"{league}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Strength-adjusted feature builder (ELO-neighborhood + time decay).")
    ap.add_argument("--league", required=True, help="League code, e.g., E0")
    ap.add_argument("--in", dest="in_path", default=None, help="Input CSV path")
    ap.add_argument("--out", dest="out_path", default=None, help="Output CSV path")
    ap.add_argument("--sigma", type=float, default=30.0, help="Gaussian sigma for ELO distance (points)")
    ap.add_argument("--tau", type=float, default=180.0, help="Time decay (days)")
    ap.add_argument("--min-matches", type=int, default=3, help="Fallback simple avg window if few weighted matches")
    ap.add_argument("--max-hist", type=int, default=200, help="Max past matches considered per team")
    args = ap.parse_args()

    in_path = args.in_path or guess_input_path(args.league)
    if not in_path or not os.path.exists(in_path):
        raise SystemExit(f"Input not found. Provide --in or create one of:\n  "
                         f"data/enhanced/{args.league}_final_features.csv\n  "
                         f"data/processed/{args.league}.csv")

    if not args.out_path:
        out_dir = os.path.join("data", "enhanced")
        os.makedirs(out_dir, exist_ok=True)
        args.out_path = os.path.join(out_dir, f"{args.league}_strength_adj.csv")

    print(f"[strength] reading: {in_path}")
    df_in = pd.read_csv(in_path)
    print(f"[strength] rows: {len(df_in)}  cols: {len(df_in.columns)}")

    df_out = compute_strength_adjusted(
        df_in,
        sigma=args.sigma,
        tau_days=args.tau,
        min_matches=args.min_matches,
        max_hist=args.max_hist,
    )

    # Basic summary
    new_cols = [c for c in df_out.columns if c.endswith(("_H","_A")) or c.startswith(("ShotConv_","ShotConvRec_","PointsPerGame_","CleanSheetStreak_","xGDiff_","CornersConv_","CornersConvRec_","NumMatches_"))]
    na_rate = df_out[new_cols].isna().mean().round(3).to_dict()
    print(f"[strength] created columns: {len(new_cols)}")
    print(f"[strength] sample NA rates (first 8): {dict(list(na_rate.items())[:8])}")

    df_out.to_csv(args.out_path, index=False)
    print(f"[strength] wrote: {args.out_path}")

if __name__ == "__main__":
    main()
