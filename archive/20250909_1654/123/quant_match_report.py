# quant_match_report.py
# Full quant pre-match analysis (offline-first).
#
# Inputs:
#   - Enhanced league data: data/enhanced/<LEAGUE>_final_features.csv
#   - Fixtures CSV: data/fixtures/<LEAGUE>_<from>_to_<to>.csv  (date,home_team,away_team)
#   - DC model: advanced_models/<LEAGUE>dixon_coles_model.pkl   (or <LEAGUE>_dixon_coles_model.pkl)
#
# Outputs:
#   - Console: detailed per-match report (home & away profiles, ELO context, DC probs)
#   - CSV: reports/match_reports_<LEAGUE>_<from>_to_<to>.csv
#   - Markdown: reports/<LEAGUE>_<kickoff>_<HOME>_vs_<AWAY>.md
#
# Usage:
#   python quant_match_report.py --league E0 --from 2025-08-30 --to 2025-09-01 --source local
#   (use --last-n 5 to change rolling window; default 5 and 10 are computed)

import os
import re
import argparse
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

# ---------- config ----------
ENHANCED_PATH = os.path.join("data", "enhanced", "{league}_final_features.csv")
DC_MODEL_CANDIDATES = [
    os.path.join("advanced_models", "{league}_dixon_coles_model.pkl"),
    os.path.join("advanced_models", "{league}dixon_coles_model.pkl"),
    os.path.join("advanced_models", "{league}_dc_model.pkl"),
]

REPORTS_DIR = "reports"

TEAM_MAP = {
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
    # Common short forms
    "Spurs": "Tottenham",
    "Man Utd": "Man United",
    "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle",
    "West Ham Utd": "West Ham",
    "Wolverhampton": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Nottingham": "Nott'm Forest",
    "Nott Forest": "Nott'm Forest",
    "Nottm Forest": "Nott'm Forest",
}
def norm_team(x: str) -> str:
    return TEAM_MAP.get(x, x)

# ---------- utils ----------
def next_tue_to_mon() -> Tuple[str, str]:
    today = date.today()
    days_to_tue = (1 - today.weekday()) % 7
    start = today + timedelta(days=days_to_tue or 7)
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cl = cand.lower()
        if cl in cols: 
            return cols[cl]
    return None

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0

# ---------- load enhanced data & build long form ----------
def load_enhanced_long(league: str) -> pd.DataFrame:
    path = ENHANCED_PATH.format(league=league)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Enhanced data not found: {path}")
    base = pd.read_csv(path)

    # Try to detect necessary columns (be tolerant)
    col_date = pick_col(base, ["Date", "date", "MatchDate"])
    col_ht   = pick_col(base, ["HomeTeam", "hometeam", "home_team"])
    col_at   = pick_col(base, ["AwayTeam", "awayteam", "away_team"])

    col_fthg = pick_col(base, ["FTHG", "HomeGoals", "home_goals"])
    col_ftag = pick_col(base, ["FTAG", "AwayGoals", "away_goals"])

    col_hs   = pick_col(base, ["HS", "HomeShots", "home_shots"])
    col_as   = pick_col(base, ["AS", "AwayShots", "away_shots"])
    col_hst  = pick_col(base, ["HST", "HomeShotsOnTarget", "home_sot"])
    col_ast  = pick_col(base, ["AST", "AwayShotsOnTarget", "away_sot"])
    col_hc   = pick_col(base, ["HC", "HomeCorners", "home_corners"])
    col_ac   = pick_col(base, ["AC", "AwayCorners", "away_corners"])

    # flexible xG names
    col_hxg  = pick_col(base, ["HxG","Home_xG","xG_Home","home_xg","HomexG"])
    col_axg  = pick_col(base, ["AxG","Away_xG","xG_Away","away_xg","AwayxG"])

    # ELO
    col_home_elo = pick_col(base, ["HomeElo","home_elo"])
    col_away_elo = pick_col(base, ["AwayElo","away_elo"])

    required = [col_date, col_ht, col_at, col_fthg, col_ftag]
    if any(c is None for c in required):
        raise ValueError("Missing core columns in enhanced CSV. Need Date, HomeTeam, AwayTeam, FTHG, FTAG (or equivalents).")

    # Build long form: one row per team per match
    rows = []
    for _, r in base.iterrows():
        dt = r[col_date]
        ht = norm_team(str(r[col_ht]))
        at = norm_team(str(r[col_at]))

        fthg = r[col_fthg]
        ftag = r[col_ftag]

        # HUD: pick with fallbacks
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

        # Home row
        rows.append({
            "Date": dt, "team": ht, "opponent": at, "home": 1,
            "gf": fthg, "ga": ftag,
            "shots_for": hs, "shots_against": as_,
            "sot_for": hst, "sot_against": ast,
            "corners_for": hc, "corners_against": ac,
            "xg_for": hxg, "xg_against": axg,
            "team_elo": helo, "opp_elo": aelo,
        })
        # Away row
        rows.append({
            "Date": dt, "team": at, "opponent": ht, "home": 0,
            "gf": ftag, "ga": fthg,
            "shots_for": as_, "shots_against": hs,
            "sot_for": ast, "sot_against": hst,
            "corners_for": ac, "corners_against": hc,
            "xg_for": axg, "xg_against": hxg,
            "team_elo": aelo, "opp_elo": helo,
        })

    df = pd.DataFrame(rows)
    # types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["gf","ga","shots_for","shots_against","sot_for","sot_against","corners_for","corners_against","xg_for","xg_against","team_elo","opp_elo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["team","Date"]).reset_index(drop=True)
    return df

# ---------- rolling metrics ----------
def rolling_metrics(df_team: pd.DataFrame, last_n: int) -> Dict[str,float]:
    d = df_team.tail(last_n)
    games = len(d)
    if games == 0:
        return {k: np.nan for k in [
            "gpg","gapg","shots_pg","sot_pg","conv","sot_conv",
            "shots_allowed_pg","sot_allowed_pg","concession_rate",
            "corners_pg","corners_allowed_pg",
            "xg_pg","xga_pg","xg_diff_pg"
        ]}
    gpg  = d["gf"].sum() / games
    gapg = d["ga"].sum() / games

    shots_pg = d["shots_for"].sum() / games
    sot_pg   = d["sot_for"].sum() / games
    conv     = safe_div(d["gf"].sum(), d["shots_for"].sum())  # goals per shot
    sot_conv = safe_div(d["gf"].sum(), d["sot_for"].sum())    # goals per SOT

    shots_allowed_pg = d["shots_against"].sum() / games
    sot_allowed_pg   = d["sot_against"].sum() / games
    concession_rate  = safe_div(d["ga"].sum(), d["shots_against"].sum())  # goals conceded per shot faced

    corners_pg = d["corners_for"].sum() / games
    corners_allowed_pg = d["corners_against"].sum() / games

    xg_pg  = d["xg_for"].sum() / games if "xg_for" in d else np.nan
    xga_pg = d["xg_against"].sum() / games if "xg_against" in d else np.nan
    xg_diff_pg = (xg_pg - xga_pg) if (not np.isnan(xg_pg) and not np.isnan(xga_pg)) else np.nan

    return {
        "gpg": gpg, "gapg": gapg,
        "shots_pg": shots_pg, "sot_pg": sot_pg,
        "conv": conv, "sot_conv": sot_conv,
        "shots_allowed_pg": shots_allowed_pg, "sot_allowed_pg": sot_allowed_pg,
        "concession_rate": concession_rate,
        "corners_pg": corners_pg, "corners_allowed_pg": corners_allowed_pg,
        "xg_pg": xg_pg, "xga_pg": xga_pg, "xg_diff_pg": xg_diff_pg,
    }

# ---------- ELO bucket performance ----------
def elo_buckets(df_team: pd.DataFrame, last_n: int = 20) -> Dict[str, float]:
    d = df_team.tail(last_n).copy()
    if d.empty or d["opp_elo"].isna().all():
        return {"elo_low_gd": np.nan, "elo_mid_gd": np.nan, "elo_high_gd": np.nan}

    # default absolute thresholds; if scale looks different, fallback to quantiles
    opp_elo = d["opp_elo"].dropna()
    if opp_elo.empty:
        return {"elo_low_gd": np.nan, "elo_mid_gd": np.nan, "elo_high_gd": np.nan}

    if (opp_elo.min() > 1200) and (opp_elo.max() < 2300):
        low_thr, high_thr = 1750, 1850
    else:
        low_thr = opp_elo.quantile(0.33)
        high_thr = opp_elo.quantile(0.67)

    d["gd"] = d["gf"] - d["ga"]
    low  = d[d["opp_elo"] <  low_thr]
    mid  = d[(d["opp_elo"] >= low_thr) & (d["opp_elo"] < high_thr)]
    high = d[d["opp_elo"] >= high_thr]

    def avg_gd(x): 
        return float(x["gd"].mean()) if len(x)>0 else np.nan

    return {
        "elo_low_gd":  avg_gd(low),
        "elo_mid_gd":  avg_gd(mid),
        "elo_high_gd": avg_gd(high),
    }

# ---------- Dixon–Coles ----------
def dc_score_matrix(dc_model, home_team: str, away_team: str, max_goals: int = 10) -> Optional[np.ndarray]:
    try:
        import pandas as pd
        d_h = pd.DataFrame({"team": [home_team], "opponent": [away_team], "home": [1]})
        mu_h = float(dc_model.predict(d_h).values[0])
        d_a = pd.DataFrame({"team": [away_team], "opponent": [home_team], "home": [0]})
        mu_a = float(dc_model.predict(d_a).values[0])

        k = np.arange(0, max_goals+1)
        def pois(lam):
            fact = np.array([np.math.factorial(int(x)) for x in k], dtype=float)
            return np.exp(-lam) * (lam ** k) / fact
        ph = pois(mu_h)
        pa = pois(mu_a)
        P = np.outer(ph, pa)
        P /= P.sum()
        return P
    except Exception as e:
        print(f"[debug] DC failed for {home_team} vs {away_team}: {e}")
        return None

def probs_1x2(P: np.ndarray) -> Dict[str,float]:
    return {"home": float(np.tril(P,-1).sum()),
            "draw": float(np.diag(P).sum()),
            "away": float(np.triu(P,1).sum())}

def probs_ou25(P: np.ndarray) -> Dict[str,float]:
    grid = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
    over = float(P[grid >= 3].sum())
    return {"over25": over, "under25": 1.0 - over}

# ---------- fixtures (local only for this analyzer) ----------
def load_fixtures_local(league: str, dfrom: str, dto: str) -> List[Dict[str,str]]:
    os.makedirs(os.path.join("data","fixtures"), exist_ok=True)
    path = os.path.join("data","fixtures", f"{league}_{dfrom}_to_{dto}.csv")
    if not os.path.exists(path):
        # create a template
        pd.DataFrame(columns=["date","home_team","away_team"]).to_csv(path, index=False)
        print(f"[info] Created template: {path}. Fill it and re-run.")
        return []
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["date","home_team","away_team"]:
        if c not in df.columns:
            raise ValueError(f"Fixtures CSV missing column: {c}")
        df[c] = df[c].astype(str).str.strip()
    # drop stray header rows
    df = df[(df["date"].str.lower()!="date") & (df["home_team"].str.lower()!="home_team")]
    # normalize names
    df["home_team"] = df["home_team"].map(norm_team)
    df["away_team"] = df["away_team"].map(norm_team)
    return df[["date","home_team","away_team"]].to_dict("records")

# ---------- main analysis ----------
def analyze_fixture(df_long: pd.DataFrame,
                    dc_model,
                    home: str,
                    away: str,
                    last_n: int = 5) -> Dict[str,object]:
    hlog = df_long[df_long["team"] == home]
    alog = df_long[df_long["team"] == away]

    # rolling windows (5 & 10 by default)
    h5 = rolling_metrics(hlog, last_n)
    h10 = rolling_metrics(hlog, 10)
    a5 = rolling_metrics(alog, last_n)
    a10 = rolling_metrics(alog, 10)

    # ELO context vs buckets (last 20)
    helo = elo_buckets(hlog, 20)
    aelo = elo_buckets(alog, 20)

    # DC
    P = dc_score_matrix(dc_model, home, away)
    probs = probs_1x2(P) if P is not None else {"home": np.nan, "draw": np.nan, "away": np.nan}
    ou = probs_ou25(P) if P is not None else {"over25": np.nan, "under25": np.nan}

    return {
        "home": home, "away": away,
        "dc_home": probs["home"], "dc_draw": probs["draw"], "dc_away": probs["away"],
        "dc_over25": ou["over25"], "dc_under25": ou["under25"],
        # Home last-5
        "H5_gpg": h5["gpg"], "H5_gapg": h5["gapg"],
        "H5_shots_pg": h5["shots_pg"], "H5_sot_pg": h5["sot_pg"],
        "H5_conv": h5["conv"], "H5_sot_conv": h5["sot_conv"],
        "H5_shots_allowed_pg": h5["shots_allowed_pg"], "H5_concession_rate": h5["concession_rate"],
        "H5_corners_pg": h5["corners_pg"], "H5_corners_allowed_pg": h5["corners_allowed_pg"],
        "H5_xg_pg": h5["xg_pg"], "H5_xga_pg": h5["xga_pg"], "H5_xgdiff_pg": h5["xg_diff_pg"],
        # Home last-10
        "H10_gpg": h10["gpg"], "H10_gapg": h10["gapg"],
        "H10_shots_pg": h10["shots_pg"], "H10_sot_pg": h10["sot_pg"],
        "H10_conv": h10["conv"], "H10_sot_conv": h10["sot_conv"],
        "H10_shots_allowed_pg": h10["shots_allowed_pg"], "H10_concession_rate": h10["concession_rate"],
        "H10_corners_pg": h10["corners_pg"], "H10_corners_allowed_pg": h10["corners_allowed_pg"],
        "H10_xg_pg": h10["xg_pg"], "H10_xga_pg": h10["xga_pg"], "H10_xgdiff_pg": h10["xg_diff_pg"],
        # Away last-5
        "A5_gpg": a5["gpg"], "A5_gapg": a5["gapg"],
        "A5_shots_pg": a5["shots_pg"], "A5_sot_pg": a5["sot_pg"],
        "A5_conv": a5["conv"], "A5_sot_conv": a5["sot_conv"],
        "A5_shots_allowed_pg": a5["shots_allowed_pg"], "A5_concession_rate": a5["concession_rate"],
        "A5_corners_pg": a5["corners_pg"], "A5_corners_allowed_pg": a5["corners_allowed_pg"],
        "A5_xg_pg": a5["xg_pg"], "A5_xga_pg": a5["xga_pg"], "A5_xgdiff_pg": a5["xg_diff_pg"],
        # Away last-10
        "A10_gpg": a10["gpg"], "A10_gapg": a10["gapg"],
        "A10_shots_pg": a10["shots_pg"], "A10_sot_pg": a10["sot_pg"],
        "A10_conv": a10["conv"], "A10_sot_conv": a10["sot_conv"],
        "A10_shots_allowed_pg": a10["shots_allowed_pg"], "A10_concession_rate": a10["concession_rate"],
        "A10_corners_pg": a10["corners_pg"], "A10_corners_allowed_pg": a10["corners_allowed_pg"],
        "A10_xg_pg": a10["xg_pg"], "A10_xga_pg": a10["xga_pg"], "A10_xgdiff_pg": a10["xg_diff_pg"],
        # ELO buckets
        "H_elo_low_gd": helo["elo_low_gd"], "H_elo_mid_gd": helo["elo_mid_gd"], "H_elo_high_gd": helo["elo_high_gd"],
        "A_elo_low_gd": aelo["elo_low_gd"], "A_elo_mid_gd": aelo["elo_mid_gd"], "A_elo_high_gd": aelo["elo_high_gd"],
    }

def pretty_print_match(rec: Dict[str,object], kickoff: str):
    h, a = rec["home"], rec["away"]
    print(f"\n=== {h} vs {a} ===  [{kickoff}]")
    print(f"DC 1X2:  HOME={rec['dc_home']:.3f}  DRAW={rec['dc_draw']:.3f}  AWAY={rec['dc_away']:.3f}")
    print(f"DC OU2.5:  OVER={rec['dc_over25']:.3f}  UNDER={rec['dc_under25']:.3f}")

    def line(tag, l5, l10): 
        print(f"{tag:18}  L5={l5:.2f}   L10={l10:.2f}")

    print("\n-- HOME form --")
    line("Goals per game", rec["H5_gpg"], rec["H10_gpg"])
    line("Goals conceded", rec["H5_gapg"], rec["H10_gapg"])
    line("Shots per game", rec["H5_shots_pg"], rec["H10_shots_pg"])
    line("SOT per game",   rec["H5_sot_pg"],  rec["H10_sot_pg"])
    line("Shot conv",      rec["H5_conv"],    rec["H10_conv"])
    line("SOT conv",       rec["H5_sot_conv"],rec["H10_sot_conv"])
    line("Shots allowed",  rec["H5_shots_allowed_pg"], rec["H10_shots_allowed_pg"])
    line("Concede/shot",   rec["H5_concession_rate"],  rec["H10_concession_rate"])
    line("Corners for",    rec["H5_corners_pg"], rec["H10_corners_pg"])
    line("Corners against",rec["H5_corners_allowed_pg"], rec["H10_corners_allowed_pg"])
    if not (np.isnan(rec["H5_xg_pg"]) and np.isnan(rec["H10_xg_pg"])):
        line("xG per game", rec["H5_xg_pg"], rec["H10_xg_pg"])
        line("xGA per game",rec["H5_xga_pg"], rec["H10_xga_pg"])
        line("xG diff",     rec["H5_xgdiff_pg"], rec["H10_xgdiff_pg"])

    print("\n-- AWAY form --")
    line("Goals per game", rec["A5_gpg"], rec["A10_gpg"])
    line("Goals conceded", rec["A5_gapg"], rec["A10_gapg"])
    line("Shots per game", rec["A5_shots_pg"], rec["A10_shots_pg"])
    line("SOT per game",   rec["A5_sot_pg"],  rec["A10_sot_pg"])
    line("Shot conv",      rec["A5_conv"],    rec["A10_conv"])
    line("SOT conv",       rec["A5_sot_conv"],rec["A10_sot_conv"])
    line("Shots allowed",  rec["A5_shots_allowed_pg"], rec["A10_shots_allowed_pg"])
    line("Concede/shot",   rec["A5_concession_rate"],  rec["A10_concession_rate"])
    line("Corners for",    rec["A5_corners_pg"], rec["A10_corners_pg"])
    line("Corners against",rec["A5_corners_allowed_pg"], rec["A10_corners_allowed_pg"])
    if not (np.isnan(rec["A5_xg_pg"]) and np.isnan(rec["A10_xg_pg"])):
        line("xG per game", rec["A5_xg_pg"], rec["A10_xg_pg"])
        line("xGA per game",rec["A5_xga_pg"], rec["A10_xga_pg"])
        line("xG diff",     rec["A5_xgdiff_pg"], rec["A10_xgdiff_pg"])

    print("\nELO context (avg goal diff last 20):")
    print(f"  HOME vs Low/Mid/High:  {rec['H_elo_low_gd']:.2f} / {rec['H_elo_mid_gd']:.2f} / {rec['H_elo_high_gd']:.2f}")
    print(f"  AWAY vs Low/Mid/High:  {rec['A_elo_low_gd']:.2f} / {rec['A_elo_mid_gd']:.2f} / {rec['A_elo_high_gd']:.2f}")

def write_markdown(rec: Dict[str,object], league: str, kickoff: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fname = f"{league}_{kickoff}_{sanitize_filename(rec['home'])}_vs_{sanitize_filename(rec['away'])}.md"
    path = os.path.join(REPORTS_DIR, fname)

    def fmt(x): 
        return "NA" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.3f}" if isinstance(x,float) else str(x)

    lines = []
    lines.append(f"# {rec['home']} vs {rec['away']}")
    lines.append(f"_Kickoff (UTC): {kickoff}_\n")
    lines.append("## Dixon–Coles")
    lines.append(f"- 1X2: **Home {fmt(rec['dc_home'])}**, **Draw {fmt(rec['dc_draw'])}**, **Away {fmt(rec['dc_away'])}**")
    lines.append(f"- OU2.5: **Over {fmt(rec['dc_over25'])}**, **Under {fmt(rec['dc_under25'])}**\n")

    def block(title, prefix):
        lines.append(f"## {title} (L5 / L10)")
        lines.append(f"- Goals per game: {fmt(rec[prefix+'5_gpg'])} / {fmt(rec[prefix+'10_gpg'])}")
        lines.append(f"- Goals conceded: {fmt(rec[prefix+'5_gapg'])} / {fmt(rec[prefix+'10_gapg'])}")
        lines.append(f"- Shots per game: {fmt(rec[prefix+'5_shots_pg'])} / {fmt(rec[prefix+'10_shots_pg'])}")
        lines.append(f"- SOT per game: {fmt(rec[prefix+'5_sot_pg'])} / {fmt(rec[prefix+'10_sot_pg'])}")
        lines.append(f"- Shot conversion: {fmt(rec[prefix+'5_conv'])} / {fmt(rec[prefix+'10_conv'])}")
        lines.append(f"- SOT conversion: {fmt(rec[prefix+'5_sot_conv'])} / {fmt(rec[prefix+'10_sot_conv'])}")
        lines.append(f"- Shots allowed: {fmt(rec[prefix+'5_shots_allowed_pg'])} / {fmt(rec[prefix+'10_shots_allowed_pg'])}")
        lines.append(f"- Concede per shot: {fmt(rec[prefix+'5_concession_rate'])} / {fmt(rec[prefix+'10_concession_rate'])}")
        lines.append(f"- Corners for: {fmt(rec[prefix+'5_corners_pg'])} / {fmt(rec[prefix+'10_corners_pg'])}")
        lines.append(f"- Corners against: {fmt(rec[prefix+'5_corners_allowed_pg'])} / {fmt(rec[prefix+'10_corners_allowed_pg'])}")
        if not (np.isnan(rec[prefix+'5_xg_pg']) and np.isnan(rec[prefix+'10_xg_pg'])):
            lines.append(f"- xG per game: {fmt(rec[prefix+'5_xg_pg'])} / {fmt(rec[prefix+'10_xg_pg'])}")
            lines.append(f"- xGA per game: {fmt(rec[prefix+'5_xga_pg'])} / {fmt(rec[prefix+'10_xga_pg'])}")
            lines.append(f"- xG diff: {fmt(rec[prefix+'5_xgdiff_pg'])} / {fmt(rec[prefix+'10_xgdiff_pg'])}")
        lines.append("")

    block("HOME form", "H")
    block("AWAY form", "A")

    lines.append("## ELO Context (avg goal diff last 20)")
    lines.append(f"- HOME vs Low / Mid / High: {fmt(rec['H_elo_low_gd'])} / {fmt(rec['H_elo_mid_gd'])} / {fmt(rec['H_elo_high_gd'])}")
    lines.append(f"- AWAY vs Low / Mid / High: {fmt(rec['A_elo_low_gd'])} / {fmt(rec['A_elo_mid_gd'])} / {fmt(rec['A_elo_high_gd'])}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def load_dc_model(league: str):
    for patt in DC_MODEL_CANDIDATES:
        path = patt.format(league=league)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                continue
    raise FileNotFoundError(f"DC model not found for league {league} in: {DC_MODEL_CANDIDATES}")

def main():
    p = argparse.ArgumentParser(description="Full quant pre-match analyzer (offline-first).")
    p.add_argument("--league", required=True, help="League code, e.g., E0")
    p.add_argument("--from", dest="dfrom", default=None, help="YYYY-MM-DD")
    p.add_argument("--to", dest="dto", default=None, help="YYYY-MM-DD")
    p.add_argument("--source", choices=["local"], default="local", help="Fixtures source (local only in this analyzer).")
    p.add_argument("--last-n", type=int, default=5, help="Rolling window size for L5 metrics (default 5).")
    args = p.parse_args()

    dfrom = args.dfrom
    dto = args.dto
    if not (dfrom and dto):
        dfrom, dto = next_tue_to_mon()

    print(f"--- Quant Analyzer ---\nLeague: {args.league} | Window: {dfrom} → {dto}\n")

    # Load data
    df_long = load_enhanced_long(args.league)
    print(f"[data] loaded long-form rows: {len(df_long)}  (teams: {df_long['team'].nunique()})")

    # Training date span (for your awareness)
    print(f"[data] span: {df_long['Date'].min().date()} → {df_long['Date'].max().date()}")

    # Fixtures
    fixtures = load_fixtures_local(args.league, dfrom, dto)
    if not fixtures:
        print("[warn] no fixtures found. Add rows to the CSV and re-run.")
        return
    print(f"[fixtures] count: {len(fixtures)}")

    # DC model
    dc_model = load_dc_model(args.league)
    print("[model] DC loaded.")

    # Analyze each fixture
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_rows = []
    for fx in fixtures:
        kickoff = fx["date"]
        home = norm_team(fx["home_team"])
        away = norm_team(fx["away_team"])

        rec = analyze_fixture(df_long, dc_model, home, away, last_n=args.last_n)
        pretty_print_match(rec, kickoff)
        md_path = write_markdown(rec, args.league, kickoff.replace(":","").replace("-",""))
        print(f"[saved] {md_path}")

        row = {"league": args.league, "kickoff_utc": kickoff, **rec}
        out_rows.append(row)

    # Save CSV
    out = pd.DataFrame(out_rows).sort_values(["kickoff_utc","home"])
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"match_reports_{args.league}_{dfrom}_to_{dto}.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[done] saved CSV → {out_path}")

if __name__ == "__main__":
    main()
