"""
Build Micro→Macro aggregates from per-shot CSV using ShotXG model.

Inputs:
  - shots CSV from Stage 1: data/shots/understat_shots.csv
  - ShotXG model: models/shotxg_xgb.pkl (+ optional calibrator models/shotxg_iso.pkl)

Outputs:
  - data/enhanced/micro_agg.csv: per-team, per-match aggregates + EWMA (home/away)

Usage:
  python -m scripts.build_micro_aggregates \
    --shots data/shots/understat_shots.csv \
    --model models/shotxg_xgb.pkl \
    --calib models/shotxg_iso.pkl \
    --out data/enhanced/micro_agg.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import config


def _as_int01(x) -> int:
    try:
        if isinstance(x, (int, float)):
            return int(x != 0)
        s = str(x).strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return 1
        if s in ("0", "false", "f", "no", "n", "none", "", "nan"):
            return 0
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def _build_X(shots: pd.DataFrame, features_ref: List[str] | None = None) -> pd.DataFrame:
    d = shots.copy()
    for col in ("dist_m", "angle_deg", "X", "Y", "minute"):
        if col not in d.columns:
            d[col] = np.nan
    d["dist_m"] = pd.to_numeric(d["dist_m"], errors="coerce")
    d["angle_deg"] = pd.to_numeric(d["angle_deg"], errors="coerce")
    d["X"] = pd.to_numeric(d["X"], errors="coerce")
    d["Y"] = pd.to_numeric(d["Y"], errors="coerce")
    d["minute"] = pd.to_numeric(d["minute"], errors="coerce")
    for col in ("is_header", "fast_break", "isKeyPass"):
        if col not in d.columns:
            d[col] = 0
        d[col] = d[col].apply(_as_int01).astype(int)
    for col in ("situation", "shotType", "h_a"):
        if col not in d.columns:
            d[col] = ""
        d[col] = d[col].astype(str).fillna("")

    base_cols = [
        "dist_m", "angle_deg", "X", "Y", "minute",
        "is_header", "fast_break", "isKeyPass",
    ]
    cat_cols = ["situation", "shotType", "h_a"]
    X_num = d[base_cols].astype(float).fillna(0.0)
    X_cat = pd.get_dummies(d[cat_cols], drop_first=False, dtype=int)
    X = pd.concat([X_num, X_cat], axis=1)
    if features_ref:
        # Align columns to training feature order, fill missing with 0
        for c in features_ref:
            if c not in X.columns:
                X[c] = 0
        X = X[features_ref]
    return X


def _ewma(series: pd.Series, half_life_matches: float) -> pd.Series:
    # Convert half-life in matches to exponential alpha
    alpha = 1 - 0.5 ** (1.0 / max(half_life_matches, 1.0))
    out = []
    v = None
    for x in series.astype(float).values:
        v = (1 - alpha) * (v if v is not None else x) + alpha * x
        out.append(v)
    return pd.Series(out, index=series.index)


def _load_processed_maps(league: str) -> Tuple[Dict[Tuple[str,str], Dict], Dict[Tuple[str,str], Dict]]:
    """Load processed league CSV and build maps for corners and possession.

    Returns two dicts keyed by (date_str, team_std):
      - corners_map: {'for': int, 'against': int}
      - poss_map: {'for': float, 'against': float}
    date_str is YYYY-MM-DD.
    """
    corners_map: Dict[Tuple[str,str], Dict] = {}
    poss_map: Dict[Tuple[str,str], Dict] = {}
    path_proc = os.path.join('data','processed', f'{league}_merged_preprocessed.csv')
    path_raw = os.path.join('data','raw', f'{league}_merged.csv')
    use_path = path_proc if os.path.exists(path_proc) else (path_raw if os.path.exists(path_raw) else None)
    if not use_path:
        return corners_map, poss_map
    try:
        d = pd.read_csv(use_path)
        if 'Date' not in d.columns:
            return corners_map, poss_map
        d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
        d = d.dropna(subset=['Date','HomeTeam','AwayTeam'])
        # Identify corner columns
        hc = 'HC' if 'HC' in d.columns else None
        ac = 'AC' if 'AC' in d.columns else None
        # Identify possession columns heuristically
        poss_h = None; poss_a = None
        for cand in ['HomePoss','PossessionH','HPoss','HP']:
            if cand in d.columns:
                poss_h = cand; break
        for cand in ['AwayPoss','PossessionA','APoss','AP']:
            if cand in d.columns:
                poss_a = cand; break
        d['date_d'] = d['Date'].dt.strftime('%Y-%m-%d')
        for _, r in d.iterrows():
            home = config.normalize_team_name(str(r['HomeTeam']))
            away = config.normalize_team_name(str(r['AwayTeam']))
            date_s = str(r['date_d'])
            # corners
            cf = int(r.get(hc, 0) or 0) if hc else 0
            ca = int(r.get(ac, 0) or 0) if ac else 0
            corners_map[(date_s, home)] = {'for': cf, 'against': ca}
            corners_map[(date_s, away)] = {'for': ca, 'against': cf}
            # possession
            pf = float(r.get(poss_h, np.nan)) if poss_h else np.nan
            pa = float(r.get(poss_a, np.nan)) if poss_a else np.nan
            if not np.isnan(pf):
                poss_map[(date_s, home)] = {'for': pf, 'against': (100.0 - pf) if np.isnan(pa) else pa}
            if not np.isnan(pa):
                poss_map[(date_s, away)] = {'for': pa, 'against': (100.0 - pa) if np.isnan(pf) else pf}
    except Exception:
        return corners_map, poss_map
    # Optional possession overlay from an external possession CSV
    try:
        pos_path = os.path.join('data','processed', f'{league}_possession.csv')
        if os.path.exists(pos_path):
            p = pd.read_csv(pos_path)
            # Expect Date, HomeTeam, AwayTeam, HomePoss, AwayPoss
            p['Date'] = pd.to_datetime(p['Date'], errors='coerce')
            p = p.dropna(subset=['Date','HomeTeam','AwayTeam'])
            p['date_d'] = p['Date'].dt.strftime('%Y-%m-%d')
            for _, r in p.iterrows():
                home = config.normalize_team_name(str(r['HomeTeam']))
                away = config.normalize_team_name(str(r['AwayTeam']))
                date_s = str(r['date_d'])
                hp = float(r.get('HomePoss', np.nan))
                ap = float(r.get('AwayPoss', np.nan))
                if not np.isnan(hp):
                    poss_map[(date_s, home)] = {'for': hp, 'against': ap if not np.isnan(ap) else (100.0 - hp)}
                if not np.isnan(ap):
                    poss_map[(date_s, away)] = {'for': ap, 'against': hp if not np.isnan(hp) else (100.0 - ap)}
    except Exception:
        pass
    return corners_map, poss_map


def _load_absences_index(league: str) -> Dict[Tuple[str,str], float]:
    """Load simple absences CSV and build availability index per (date, team).

    Expected CSV: data/absences/latest.csv with columns [date, league, team, player, status, weight]
    status in {'out','doubtful','in'}; weight in [0,1] (optional; default 1 for 'out').
    Index heuristic: availability = 1 - sum(weight for 'out'), clipped [0,1], per team up to the given date.
    """
    path = os.path.join('data','absences','latest.csv')
    idx: Dict[Tuple[str,str], float] = {}
    if not os.path.exists(path):
        return idx
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df.get('date'), errors='coerce')
        if 'league' in df.columns:
            df = df[df['league'].astype(str).str.upper() == str(league).upper()]
        df['team'] = df['team'].astype(str).map(config.normalize_team_name)
        # For each (date,team) compute index using rows up to that date
        for team, g in df.groupby('team'):
            g = g.sort_values('date')
            cum_out = 0.0
            for _, r in g.iterrows():
                status = str(r.get('status','')).lower().strip()
                w = r.get('weight'); w = float(w) if pd.notna(w) else 1.0
                if status == 'out':
                    cum_out += max(0.0, min(1.0, w))
                avail = max(0.0, 1.0 - cum_out)
                date_s = str(pd.to_datetime(r.get('date')).strftime('%Y-%m-%d'))
                idx[(date_s, team)] = avail
    except Exception:
        return idx
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-team micro aggregates from shots")
    ap.add_argument("--shots", required=True, help="CSV of shots (from shots_ingest_understat)")
    ap.add_argument("--model", default=os.path.join("models", "shotxg_xgb.pkl"))
    ap.add_argument("--calib", default=os.path.join("models", "shotxg_iso.pkl"))
    ap.add_argument("--out", default=os.path.join("data", "enhanced", "micro_agg.csv"))
    ap.add_argument("--league", default="E0")
    ap.add_argument("--half-life-matches", type=float, default=5.0)
    args = ap.parse_args()

    df = pd.read_csv(args.shots)
    # Predict per-shot goal prob
    clf = joblib.load(args.model)
    try:
        features_ref: List[str] | None = list(getattr(clf, "feature_names_in_", [])) or None
    except Exception:
        features_ref = None
    X = _build_X(df, features_ref=features_ref)
    p = clf.predict_proba(X)[:, 1]
    # Optional calibrator
    try:
        iso = joblib.load(args.calib)
        p = np.clip(iso.predict(p), 1e-9, 1 - 1e-9)
    except Exception:
        p = np.clip(p, 1e-9, 1 - 1e-9)
    df["p_goal"] = p

    # Normalize key fields
    df["result_goal"] = df.get("result").astype(str).str.lower().eq("goal").astype(int)
    # Ensure team names are strings
    for col in ("team", "h_team", "a_team"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Build per-match rows for both teams
    rows: List[Dict] = []
    # Determine dates
    try:
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    except Exception:
        pass
    for mid, g in df.groupby(["match_id"], dropna=False):
        # Resolve home/away names
        def first_nonempty(series: pd.Series) -> str | None:
            for v in series.astype(str).values:
                if isinstance(v, str) and v.strip():
                    return v
            return None
        h = first_nonempty(g.get("h_team") if "h_team" in g.columns else pd.Series(dtype=str))
        a = first_nonempty(g.get("a_team") if "a_team" in g.columns else pd.Series(dtype=str))
        # If missing, infer from h_a flag and team labels
        if not h or not a:
            teams_h = sorted(set(g.loc[g.get("h_a").astype(str).str.lower().str.startswith("h"), "team"].dropna().astype(str)))
            teams_a = sorted(set(g.loc[g.get("h_a").astype(str).str.lower().str.startswith("a"), "team"].dropna().astype(str)))
            h = h or (teams_h[0] if teams_h else None)
            a = a or (teams_a[0] if teams_a else None)
        if not h or not a:
            continue
        date_val = pd.to_datetime(g.get("date"), errors="coerce").dropna().min()
        # Split by h_a flag (robust to name mismatches)
        gh = g[g.get("h_a").astype(str).str.lower().str.startswith("h")]
        ga = g[g.get("h_a").astype(str).str.lower().str.startswith("a")]
        shots_h = int(len(gh)); shots_a = int(len(ga))
        xg_h = float(gh["p_goal"].sum()); xg_a = float(ga["p_goal"].sum())
        gl_h = int(gh["result_goal"].sum()); gl_a = int(ga["result_goal"].sum())
        # Corner conversion approximations
        def corner_conv(sub):
            ss = sub[sub.get("situation").astype(str).str.contains("corner", case=False, na=False)]
            xg_c = float(ss.get("p_goal", pd.Series(dtype=float)).sum())
            return int(len(ss)), int(ss.get("result_goal", pd.Series(dtype=int)).sum()), xg_c
        c_sh_h, c_go_h, cxg_h = corner_conv(gh)
        c_sh_a, c_go_a, cxg_a = corner_conv(ga)
        rows.append({
            "match_id": mid, "date": date_val, "team": h, "side": "H",
            "opponent": a,
            "xg_for": xg_h, "xg_against": xg_a,
            "goals_for": gl_h, "goals_against": gl_a,
            "shots_for": shots_h, "shots_against": shots_a,
            "corner_shots_for": c_sh_h, "corner_goals_for": c_go_h,
            "xg_from_corners_for": cxg_h,
        })
        rows.append({
            "match_id": mid, "date": date_val, "team": a, "side": "A",
            "opponent": h,
            "xg_for": xg_a, "xg_against": xg_h,
            "goals_for": gl_a, "goals_against": gl_h,
            "shots_for": shots_a, "shots_against": shots_h,
            "corner_shots_for": c_sh_a, "corner_goals_for": c_go_a,
            "xg_from_corners_for": cxg_a,
        })
    per_match = pd.DataFrame(rows)
    if per_match.empty:
        print("No per-match aggregates built.")
        return
    # Derived efficiencies
    per_match["finishing_efficiency"] = per_match["goals_for"] - per_match["xg_for"]
    per_match["goalkeeping_efficiency"] = per_match["xg_against"] - per_match["goals_against"]
    per_match.sort_values(["team", "date"], inplace=True)

    # EWMA per team and side
    out_rows: List[Dict] = []
    for (team, side), g in per_match.groupby(["team", "side"], dropna=False):
        g = g.sort_values("date")
        # Base EWMAs
        for col in ("xg_for", "xg_against", "finishing_efficiency", "goalkeeping_efficiency"):
            g[f"{col}_EWMA"] = _ewma(g[col].fillna(0.0), args.half_life_matches)
        # Possession and Corners totals EWMAs (if available)
        if "possession_for" in g.columns:
            g["possession_for_EWMA"] = _ewma(g["possession_for"].astype(float).fillna(0.0), args.half_life_matches)
        if "possession_against" in g.columns:
            g["possession_against_EWMA"] = _ewma(g["possession_against"].astype(float).fillna(0.0), args.half_life_matches)
        if "corners_total_for" in g.columns:
            g["corners_total_for_EWMA"] = _ewma(g["corners_total_for"].astype(float).fillna(0.0), args.half_life_matches)
        if "corners_total_against" in g.columns:
            g["corners_total_against_EWMA"] = _ewma(g["corners_total_against"].astype(float).fillna(0.0), args.half_life_matches)
        # xG per possession point (rates) EWMAs
        if "xg_for_per_poss" in g.columns:
            g["xg_for_per_poss_EWMA"] = _ewma(g["xg_for_per_poss"].astype(float).fillna(0.0), args.half_life_matches)
        if "xg_against_per_poss" in g.columns:
            g["xg_against_per_poss_EWMA"] = _ewma(g["xg_against_per_poss"].astype(float).fillna(0.0), args.half_life_matches)
        # xG from corners EWMAs (for and against; against added after opponent join)
        if "xg_from_corners_for" in g.columns:
            g["xg_from_corners_for_EWMA"] = _ewma(g["xg_from_corners_for"].astype(float).fillna(0.0), args.half_life_matches)
        out_rows.append(g)
    micro = pd.concat(out_rows, ignore_index=True)
    # Enrich with real corners/possession from processed (if available)
    try:
        micro['date_d'] = pd.to_datetime(micro.get('date'), errors='coerce').dt.strftime('%Y-%m-%d')
        micro['team_std'] = micro['team'].astype(str).map(config.normalize_team_name)
        c_map, p_map = _load_processed_maps(args.league)
        # corners totals
        micro['corners_total_for'] = [ (c_map.get((d,t)) or {}).get('for', np.nan) for d,t in micro[['date_d','team_std']].itertuples(index=False, name=None) ]
        micro['corners_total_against'] = [ (c_map.get((d,t)) or {}).get('against', np.nan) for d,t in micro[['date_d','team_std']].itertuples(index=False, name=None) ]
        # possession
        micro['possession_for'] = [ (p_map.get((d,t)) or {}).get('for', np.nan) for d,t in micro[['date_d','team_std']].itertuples(index=False, name=None) ]
        micro['possession_against'] = [ (p_map.get((d,t)) or {}).get('against', np.nan) for d,t in micro[['date_d','team_std']].itertuples(index=False, name=None) ]
        # corner goals against from opponent row within same match
        try:
            opp_corner_goals = micro.groupby('match_id')['corner_goals_for'].transform(lambda s: s.iloc[::-1].values if len(s)==2 else np.nan)
            micro['corner_goals_against'] = opp_corner_goals
        except Exception:
            micro['corner_goals_against'] = np.nan
        # mirror xG from corners for 'against'
        try:
            opp_cxg = micro.groupby('match_id')['xg_from_corners_for'].transform(lambda s: s.iloc[::-1].values if len(s)==2 else np.nan)
            micro['xg_from_corners_against'] = opp_cxg
        except Exception:
            micro['xg_from_corners_against'] = np.nan
        # EWMA for xG_from_corners_against
        try:
            out_rows2 = []
            for (team, side), g2 in micro.groupby(["team","side"], dropna=False):
                g2 = g2.sort_values('date')
                if 'xg_from_corners_against' in g2.columns:
                    g2['xg_from_corners_against_EWMA'] = _ewma(g2['xg_from_corners_against'].astype(float).fillna(0.0), args.half_life_matches)
                out_rows2.append(g2)
            micro = pd.concat(out_rows2, ignore_index=True)
        except Exception:
            pass
        # rates
        micro['corner_conversion_rate_for'] = micro['corner_goals_for'] / micro['corners_total_for'].replace(0, np.nan)
        micro['corner_conversion_rate_against'] = micro['corner_goals_against'] / micro['corners_total_against'].replace(0, np.nan)
        # possession-derived metrics
        micro['goals_for_per_poss'] = micro['goals_for'] / micro['possession_for'].replace(0, np.nan)
        micro['goals_against_per_poss'] = micro['goals_against'] / micro['possession_against'].replace(0, np.nan)
        micro['xg_for_per_poss'] = micro['xg_for'] / micro['possession_for'].replace(0, np.nan)
        micro['xg_against_per_poss'] = micro['xg_against'] / micro['possession_against'].replace(0, np.nan)
    except Exception:
        pass
    # Absences/availability index (simple heuristic)
    try:
        a_idx = _load_absences_index(args.league)
        micro['availability_index'] = [ a_idx.get((d,t)) for d,t in micro[['date_d','team_std']].itertuples(index=False, name=None) ]
    except Exception:
        pass

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    micro.to_csv(args.out, index=False)
    print(f"Saved micro aggregates -> {args.out}  (rows: {len(micro)})")


if __name__ == "__main__":
    main()
