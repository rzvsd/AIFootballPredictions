# xgb_trainer.py
import argparse
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
import config


def _build_fallback_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map processed dataset columns to ULTIMATE_FEATURES as a fallback training set.

    This uses last-5 goals proxies and season averages where needed.
    """
    X = pd.DataFrame(index=df.index)
    X['ShotConv_H'] = df.get('AvgLast5HomeGoalsScored')
    X['ShotConv_A'] = df.get('AvgLast5AwayGoalsScored')
    X['ShotConvRec_H'] = df.get('AvgLast5HomeGoalsConceded')
    X['ShotConvRec_A'] = df.get('AvgLast5AwayGoalsConceded')
    # Proxies for points per game: season average goals scored
    X['PointsPerGame_H'] = df.get('AvgHomeGoalsScored')
    X['PointsPerGame_A'] = df.get('AvgAwayGoalsScored')
    # Clean sheet streaks not available -> zeros
    X['CleanSheetStreak_H'] = 0.0
    X['CleanSheetStreak_A'] = 0.0
    # xG differential proxy from last-5 goals
    X['xGDiff_H'] = X['ShotConv_H'] - X['ShotConvRec_H']
    X['xGDiff_A'] = X['ShotConv_A'] - X['ShotConvRec_A']
    # Corners proxies not available -> zeros
    X['CornersConv_H'] = 0.0
    X['CornersConv_A'] = 0.0
    X['CornersConvRec_H'] = 0.0
    X['CornersConvRec_A'] = 0.0
    # Match-count stabilizers
    X['NumMatches_H'] = 20.0
    X['NumMatches_A'] = 20.0
    # Elo placeholders (will be filled later if computed)
    X['Elo_H'] = 1500.0
    X['Elo_A'] = 1500.0
    X['EloDiff'] = 0.0
    # Align to configured feature order, filling missing with 0.0
    return X.reindex(columns=config.ULTIMATE_FEATURES, fill_value=0.0)


def _compute_ewma_elo_prematch(df: pd.DataFrame, half_life_matches: int = 5,
                               elo_k: float = 20.0, elo_home_adv: float = 60.0,
                               elo_similarity_sigma: float = 50.0,
                               league_code: str | None = None) -> pd.DataFrame:
    """Compute per-match pre-game EWMAs (directional), Elo, and enriched micro features.

    Adds per-row pre-match columns including:
      - xg_home_EWMA, xga_home_EWMA, ppg_home_EWMA, xg_away_EWMA, xga_away_EWMA, ppg_away_EWMA
      - Elo_H, Elo_A, EloDiff
      - GF/GAvsMid/High (banded EWMAs)
      - GF/GAvsSim (kernel Elo similarity)
      - Possession_H/A, PossessionRec_H/A (EWMAs from processed possession if available)
      - CornersFor/Against_H/A (EWMAs from processed corners if available)
      - Availability_H/A (+ AvailabilityDiff) from data/absences/latest.csv if available
      - xGpp_H/A and xGppRec_H/A (xG per possession point, from micro aggregates if available)
      - xGCorners_H/A and xGCornersRec_H/A (xG from corners, from micro aggregates if available)
    """
    d = df.copy()
    if 'Date' in d.columns:
        try:
            d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
        except Exception:
            pass
    d = d.dropna(subset=['Date','HomeTeam','AwayTeam']).sort_values('Date')
    alpha = 1 - 0.5**(1/float(half_life_matches))
    # state per team
    state = {}
    elo = {}
    # pre-match lists
    cols = ['xg_home_EWMA','xga_home_EWMA','ppg_home_EWMA','xg_away_EWMA','xga_away_EWMA','ppg_away_EWMA','Elo_H','Elo_A','EloDiff',
            'GFvsMid_H','GAvsMid_H','GFvsHigh_H','GAvsHigh_H','GFvsMid_A','GAvsMid_A','GFvsHigh_A','GAvsHigh_A',
            'GFvsSim_H','GAvsSim_H','GFvsSim_A','GAvsSim_A',
            # Enrichment placeholders (pre-match)
            'Possession_H','Possession_A','PossessionRec_H','PossessionRec_A',
            'CornersFor_H','CornersFor_A','CornersAgainst_H','CornersAgainst_A',
            'Availability_H','Availability_A','AvailabilityDiff',
            'xGpp_H','xGpp_A','xGppRec_H','xGppRec_A',
            'xGCorners_H','xGCorners_A','xGCornersRec_H','xGCornersRec_A']
    store = {c: [] for c in cols}
    # Per-team histories for kernel similarity (separate home/away)
    hist_home: dict[str, list[tuple[float,float,float]]] = {}
    hist_away: dict[str, list[tuple[float,float,float]]] = {}
    # Identify possession/corner columns if present
    poss_h = None; poss_a = None
    for cand in ['HomePoss','PossessionH','HPoss','HP']:
        if cand in d.columns:
            poss_h = cand; break
    for cand in ['AwayPoss','PossessionA','APoss','AP']:
        if cand in d.columns:
            poss_a = cand; break
    hc = 'HC' if 'HC' in d.columns else None
    ac = 'AC' if 'AC' in d.columns else None
    # Load absences/availability simple index (optional)
    def _load_abs_idx(league_guess: str | None) -> dict:
        path = os.path.join('data','absences','latest.csv')
        idx = {}
        if not os.path.exists(path):
            return idx
        try:
            aa = pd.read_csv(path)
            aa['date'] = pd.to_datetime(aa.get('date'), errors='coerce')
            if league_guess and 'league' in aa.columns:
                aa = aa[aa['league'].astype(str).str.upper() == str(league_guess).upper()]
            aa['team'] = aa['team'].astype(str)
            for t, g in aa.groupby('team'):
                g = g.sort_values('date')
                cum_out = 0.0
                for _, rr in g.iterrows():
                    status = str(rr.get('status','')).lower().strip()
                    w = rr.get('weight'); w = float(w) if pd.notna(w) else 1.0
                    if status == 'out':
                        cum_out += max(0.0, min(1.0, w))
                    avail = max(0.0, 1.0 - cum_out)
                    idx[(str(pd.to_datetime(rr.get('date')).strftime('%Y-%m-%d')), t)] = avail
        except Exception:
            return {}
        return idx
    # Infer league code from processed filename if present
    abs_idx = _load_abs_idx(league_code)
    # Load micro aggregates maps (optional)
    micro_path = os.path.join('data','enhanced','micro_agg.csv')
    micro_xg_map = {}
    micro_cxg_map = {}
    try:
        if os.path.exists(micro_path):
            mm = pd.read_csv(micro_path)
            if 'date' in mm.columns:
                mm['date'] = pd.to_datetime(mm['date'], errors='coerce')
                mm['date_d'] = mm['date'].dt.strftime('%Y-%m-%d')
            else:
                mm['date_d'] = ''
            mm['team'] = mm['team'].astype(str)
            for _, rr in mm.iterrows():
                key = (str(rr.get('date_d','')), str(rr.get('team','')), str(rr.get('side',''))[:1].upper())
                if 'xg_for' in rr:
                    try:
                        micro_xg_map[key] = float(rr['xg_for'])
                    except Exception:
                        pass
                if 'xg_from_corners_for' in rr:
                    try:
                        micro_cxg_map[key] = float(rr['xg_from_corners_for'])
                    except Exception:
                        pass
    except Exception:
        pass

    for _, r in d.iterrows():
        home = str(r['HomeTeam']); away = str(r['AwayTeam'])
        # init state if needed
        if home not in state:
            state[home] = {'h_gf':1.2,'h_ga':1.2,'h_ppg':1.2,'a_gf':1.0,'a_ga':1.0,'a_ppg':1.0,
                           'h_gf_mid':1.2,'h_ga_mid':1.2,'h_gf_high':1.2,'h_ga_high':1.2,
                           'a_gf_mid':1.0,'a_ga_mid':1.0,'a_gf_high':1.0,'a_ga_high':1.0,
                           # enrichment directional states
                           'h_pos_for':50.0,'h_pos_against':50.0,'a_pos_for':50.0,'a_pos_against':50.0,
                           'h_corners_for':5.0,'h_corners_against':5.0,'a_corners_for':4.0,'a_corners_against':4.0,
                           'h_xgpp':0.02,'h_xgpp_rec':0.02,'a_xgpp':0.02,'a_xgpp_rec':0.02,
                           'h_xgcorners':0.10,'h_xgcorners_rec':0.10,'a_xgcorners':0.08,'a_xgcorners_rec':0.08}
        if away not in state:
            state[away] = {'h_gf':1.2,'h_ga':1.2,'h_ppg':1.2,'a_gf':1.0,'a_ga':1.0,'a_ppg':1.0,
                           'h_gf_mid':1.2,'h_ga_mid':1.2,'h_gf_high':1.2,'h_ga_high':1.2,
                           'a_gf_mid':1.0,'a_ga_mid':1.0,'a_gf_high':1.0,'a_ga_high':1.0,
                           'h_pos_for':50.0,'h_pos_against':50.0,'a_pos_for':50.0,'a_pos_against':50.0,
                           'h_corners_for':5.0,'h_corners_against':5.0,'a_corners_for':4.0,'a_corners_against':4.0,
                           'h_xgpp':0.02,'h_xgpp_rec':0.02,'a_xgpp':0.02,'a_xgpp_rec':0.02,
                           'h_xgcorners':0.10,'h_xgcorners_rec':0.10,'a_xgcorners':0.08,'a_xgcorners_rec':0.08}
        eh = float(elo.get(home, 1500.0)); ea = float(elo.get(away, 1500.0))
        # record pre-match
        sh = state[home]; sa = state[away]
        store['xg_home_EWMA'].append(sh['h_gf'])
        store['xga_home_EWMA'].append(sh['h_ga'])
        store['ppg_home_EWMA'].append(sh['h_ppg'])
        store['xg_away_EWMA'].append(sa['a_gf'])
        store['xga_away_EWMA'].append(sa['a_ga'])
        store['ppg_away_EWMA'].append(sa['a_ppg'])
        store['Elo_H'].append(eh)
        store['Elo_A'].append(ea)
        store['EloDiff'].append(eh - ea)
        # initialize banded stores with current pre-match EWMA (will update after match)
        store['GFvsMid_H'].append(state[home]['h_gf_mid'])
        store['GAvsMid_H'].append(state[home]['h_ga_mid'])
        store['GFvsHigh_H'].append(state[home]['h_gf_high'])
        store['GAvsHigh_H'].append(state[home]['h_ga_high'])
        store['GFvsMid_A'].append(state[away]['a_gf_mid'])
        store['GAvsMid_A'].append(state[away]['a_ga_mid'])
        store['GFvsHigh_A'].append(state[away]['a_gf_high'])
        store['GAvsHigh_A'].append(state[away]['a_ga_high'])
        # Enriched pre-match states
        # Possession and corners (pre)
        store['Possession_H'].append(sh.get('h_pos_for', 50.0))
        store['PossessionRec_H'].append(sh.get('h_pos_against', 50.0))
        store['Possession_A'].append(sa.get('a_pos_for', 50.0))
        store['PossessionRec_A'].append(sa.get('a_pos_against', 50.0))
        store['CornersFor_H'].append(sh.get('h_corners_for', 5.0))
        store['CornersAgainst_H'].append(sh.get('h_corners_against', 5.0))
        store['CornersFor_A'].append(sa.get('a_corners_for', 4.0))
        store['CornersAgainst_A'].append(sa.get('a_corners_against', 4.0))
        # Availability from absences index (by date, team) if present
        date_s = str(pd.to_datetime(r['Date']).strftime('%Y-%m-%d')) if 'Date' in r else ''
        avail_h = float(abs_idx.get((date_s, home), 1.0)) if abs_idx else 1.0
        avail_a = float(abs_idx.get((date_s, away), 1.0)) if abs_idx else 1.0
        store['Availability_H'].append(avail_h)
        store['Availability_A'].append(avail_a)
        store['AvailabilityDiff'].append(avail_h - avail_a)
        # xG per possession and from corners (pre)
        store['xGpp_H'].append(sh.get('h_xgpp', 0.02))
        store['xGppRec_H'].append(sh.get('h_xgpp_rec', 0.02))
        store['xGpp_A'].append(sa.get('a_xgpp', 0.02))
        store['xGppRec_A'].append(sa.get('a_xgpp_rec', 0.02))
        store['xGCorners_H'].append(sh.get('h_xgcorners', 0.10))
        store['xGCornersRec_H'].append(sh.get('h_xgcorners_rec', 0.10))
        store['xGCorners_A'].append(sa.get('a_xgcorners', 0.08))
        store['xGCornersRec_A'].append(sa.get('a_xgcorners_rec', 0.08))
        # kernel similarity features (based on opponent's pre-match Elo)
        import math
        def kernel_mean(hist: list[tuple[float,float,float]] | None, center: float, sigma: float) -> tuple[float,float]:
            if not hist:
                return (0.0, 0.0)
            wsum = 0.0; gf_sum = 0.0; ga_sum = 0.0
            inv2s2 = 1.0 / (2.0 * max(sigma, 1e-6) * max(sigma, 1e-6))
            for opp_elo, gf, ga in hist:
                w = math.exp(- (opp_elo - center)**2 * inv2s2)
                wsum += w; gf_sum += w * float(gf); ga_sum += w * float(ga)
            if wsum <= 0:
                return (0.0, 0.0)
            return (gf_sum/wsum, ga_sum/wsum)
        # home vs similar away Elo
        hhist = hist_home.get(home)
        k_gf_h, k_ga_h = kernel_mean(hhist, ea, float(elo_similarity_sigma))
        store['GFvsSim_H'].append(k_gf_h)
        store['GAvsSim_H'].append(k_ga_h)
        # away vs similar home Elo
        ahist = hist_away.get(away)
        k_gf_a, k_ga_a = kernel_mean(ahist, eh, float(elo_similarity_sigma))
        store['GFvsSim_A'].append(k_gf_a)
        store['GAvsSim_A'].append(k_ga_a)

        # update after match
        try:
            fthg = float(r['FTHG']); ftag = float(r['FTAG'])
        except Exception:
            continue
        sh['h_gf'] = (1-alpha)*sh['h_gf'] + alpha*fthg
        sh['h_ga'] = (1-alpha)*sh['h_ga'] + alpha*ftag
        sh['h_ppg'] = (1-alpha)*sh['h_ppg'] + alpha*(3.0 if fthg>ftag else (1.0 if fthg==ftag else 0.0))
        sa['a_gf'] = (1-alpha)*sa['a_gf'] + alpha*ftag
        sa['a_ga'] = (1-alpha)*sa['a_ga'] + alpha*fthg
        sa['a_ppg'] = (1-alpha)*sa['a_ppg'] + alpha*(3.0 if ftag>fthg else (1.0 if ftag==fthg else 0.0))
        exp_home = 1.0/(1.0 + 10**(-((eh + elo_home_adv) - ea)/400.0))
        S_home = 1.0 if fthg>ftag else (0.5 if fthg==ftag else 0.0)
        elo[home] = eh + elo_k*(S_home - exp_home)
        elo[away] = ea + elo_k*((1.0 - S_home) - (1.0 - exp_home))
        # Update banded EWMA based on opponent pre-match Elo
        if 1600.0 <= ea <= 1800.0:
            state[home]['h_gf_mid'] = (1-alpha)*state[home]['h_gf_mid'] + alpha*fthg
            state[home]['h_ga_mid'] = (1-alpha)*state[home]['h_ga_mid'] + alpha*ftag
        elif ea > 1800.0:
            state[home]['h_gf_high'] = (1-alpha)*state[home]['h_gf_high'] + alpha*fthg
            state[home]['h_ga_high'] = (1-alpha)*state[home]['h_ga_high'] + alpha*ftag
        if 1600.0 <= eh <= 1800.0:
            state[away]['a_gf_mid'] = (1-alpha)*state[away]['a_gf_mid'] + alpha*ftag
            state[away]['a_ga_mid'] = (1-alpha)*state[away]['a_ga_mid'] + alpha*fthg
        elif eh > 1800.0:
            state[away]['a_gf_high'] = (1-alpha)*state[away]['a_gf_high'] + alpha*ftag
            state[away]['a_ga_high'] = (1-alpha)*state[away]['a_ga_high'] + alpha*fthg
        # append to histories for kernel sim
        hist_home.setdefault(home, []).append((ea, fthg, ftag))
        hist_away.setdefault(away, []).append((eh, ftag, fthg))
        # Update enriched directional states from processed + micro maps (post)
        try:
            # possession
            ph = float(r.get(poss_h, np.nan)) if poss_h else np.nan
            pa = float(r.get(poss_a, np.nan)) if poss_a else np.nan
            if not np.isnan(ph):
                sh['h_pos_for'] = (1-alpha)*sh['h_pos_for'] + alpha*ph
                sa['a_pos_against'] = (1-alpha)*sa['a_pos_against'] + alpha*ph
            if not np.isnan(pa):
                sa['a_pos_for'] = (1-alpha)*sa['a_pos_for'] + alpha*pa
                sh['h_pos_against'] = (1-alpha)*sh['h_pos_against'] + alpha*pa
            # corners totals
            if hc:
                cf = float(r.get(hc, 0) or 0)
                sh['h_corners_for'] = (1-alpha)*sh['h_corners_for'] + alpha*cf
                sa['a_corners_against'] = (1-alpha)*sa['a_corners_against'] + alpha*cf
            if ac:
                ca = float(r.get(ac, 0) or 0)
                sa['a_corners_for'] = (1-alpha)*sa['a_corners_for'] + alpha*ca
                sh['h_corners_against'] = (1-alpha)*sh['h_corners_against'] + alpha*ca
            # micro-derived xG per possession and xG from corners
            if 'Date' in r:
                dstr = str(pd.to_datetime(r['Date']).strftime('%Y-%m-%d'))
            else:
                dstr = ''
            # home side
            xg_h = micro_xg_map.get((dstr, home, 'H'))
            if xg_h is not None and not np.isnan(ph) and ph > 0:
                xgpp_h = xg_h / float(ph)
                sh['h_xgpp'] = (1-alpha)*sh['h_xgpp'] + alpha*xgpp_h
                sa['a_xgpp_rec'] = (1-alpha)*sa['a_xgpp_rec'] + alpha*xgpp_h
            cxg_h = micro_cxg_map.get((dstr, home, 'H'))
            if cxg_h is not None:
                sh['h_xgcorners'] = (1-alpha)*sh['h_xgcorners'] + alpha*float(cxg_h)
                sa['a_xgcorners_rec'] = (1-alpha)*sa['a_xgcorners_rec'] + alpha*float(cxg_h)
            # away side
            xg_a = micro_xg_map.get((dstr, away, 'A'))
            if xg_a is not None and not np.isnan(pa) and pa > 0:
                xgpp_a = xg_a / float(pa)
                sa['a_xgpp'] = (1-alpha)*sa['a_xgpp'] + alpha*xgpp_a
                sh['h_xgpp_rec'] = (1-alpha)*sh['h_xgpp_rec'] + alpha*xgpp_a
            cxg_a = micro_cxg_map.get((dstr, away, 'A'))
            if cxg_a is not None:
                sa['a_xgcorners'] = (1-alpha)*sa['a_xgcorners'] + alpha*float(cxg_a)
                sh['h_xgcorners_rec'] = (1-alpha)*sh['h_xgcorners_rec'] + alpha*float(cxg_a)
        except Exception:
            pass
    # attach
    for c in cols:
        d[c] = store[c]
    return d


def train_xgb_models_for_league(league: str) -> None:
    # Prefer enhanced final_features; fallback to processed merged_preprocessed
    # Always use processed data to ensure labels and chronological order for Elo/EWMA
    proc_path = os.path.join('data', 'processed', f'{league}_merged_preprocessed.csv')
    print(f"Loading processed data: {proc_path}")
    df = pd.read_csv(proc_path)
    # Ensure labels exist; if missing, try raw merge
    y_home = df.get('FTHG'); y_away = df.get('FTAG')
    if y_home is None or (hasattr(y_home,'isna') and y_home.isna().all()) or y_away is None or (hasattr(y_away,'isna') and y_away.isna().all()):
        raw_path = os.path.join('data','raw', f'{league}_merged.csv')
        if os.path.exists(raw_path):
            raw = pd.read_csv(raw_path, parse_dates=['Date'], dayfirst=True)
            left = df.copy()
            if 'Date' in left.columns:
                try:
                    left['Date'] = pd.to_datetime(left['Date'], errors='coerce')
                except Exception:
                    pass
            keys = [c for c in ['Date','HomeTeam','AwayTeam'] if c in left.columns and c in raw.columns]
            if keys:
                df = pd.merge(left, raw[['Date','HomeTeam','AwayTeam','FTHG','FTAG']], on=keys, how='left', suffixes=('', '_raw'))
                y_home = df.get('FTHG')
                y_away = df.get('FTAG')
            else:
                print('Warning: could not align with raw file to get labels (missing keys).')
        else:
            print(f'Warning: raw file not found for labels: {raw_path}')
    # Compute per-match EWMA and Elo (with per-league Elo-sim sigma)
    sigma = float(getattr(config, 'ELO_SIM_SIGMA_PER_LEAGUE', {}).get(league, 50.0))
    d_feats = _compute_ewma_elo_prematch(
        df,
        half_life_matches=5,
        elo_k=20.0,
        elo_home_adv=60.0,
        elo_similarity_sigma=sigma,
        league_code=league,
    )
    # Build features consistent with inference mapping
    X = pd.DataFrame(index=d_feats.index)
    # Directional EWMAs
    X['ShotConv_H'] = d_feats['xg_home_EWMA']
    X['ShotConvRec_H'] = d_feats['xga_home_EWMA']
    X['PointsPerGame_H'] = d_feats['ppg_home_EWMA']
    X['ShotConv_A'] = d_feats['xg_away_EWMA']
    X['ShotConvRec_A'] = d_feats['xga_away_EWMA']
    X['PointsPerGame_A'] = d_feats['ppg_away_EWMA']
    # Enriched micro features if available
    for col in [
        'Possession_H','PossessionRec_H','CornersFor_H','CornersAgainst_H',
        'Availability_H','Availability_A','AvailabilityDiff',
        'xGpp_H','xGppRec_H','xGCorners_H','xGCornersRec_H',
        'Possession_A','PossessionRec_A','CornersFor_A','CornersAgainst_A',
        'xGpp_A','xGppRec_A','xGCorners_A','xGCornersRec_A']:
        if col in d_feats.columns:
            X[col] = d_feats[col]
        else:
            X[col] = 0.0
    # Derived diffs
    X['xGDiff_H'] = X['ShotConv_H'] - X['ShotConvRec_H']
    X['xGDiff_A'] = X['ShotConv_A'] - X['ShotConvRec_A']
    # Stabilizers / placeholders (directional)
    X['CleanSheetStreak_H'] = 0.0
    X['CleanSheetStreak_A'] = 0.0
    X['CornersConv_H'] = 0.0
    X['CornersConv_A'] = 0.0
    X['CornersConvRec_H'] = 0.0
    X['CornersConvRec_A'] = 0.0
    X['NumMatches_H'] = 20.0
    X['NumMatches_A'] = 20.0
    # Elo core
    X['Elo_H'] = d_feats['Elo_H']
    X['Elo_A'] = d_feats['Elo_A']
    X['EloDiff'] = d_feats['EloDiff']
    # Elo-band features
    for k in ['GFvsMid_H','GAvsMid_H','GFvsHigh_H','GAvsHigh_H','GFvsMid_A','GAvsMid_A','GFvsHigh_A','GAvsHigh_A']:
        X[k] = d_feats.get(k, 0.0)
    # Elo-similarity dynamic features
    for k in ['GFvsSim_H','GAvsSim_H','GFvsSim_A','GAvsSim_A']:
        X[k] = d_feats.get(k, 0.0)

    # Strict directional subsets
    home_cols = [
        'ShotConv_H','ShotConvRec_H','PointsPerGame_H','xGDiff_H',
        'CornersConv_H','CornersConvRec_H','NumMatches_H',
        'Elo_H','Elo_A','EloDiff',
        'GFvsMid_H','GAvsMid_H','GFvsHigh_H','GAvsHigh_H',
        'GFvsSim_H','GAvsSim_H',
        # Enriched
        'Possession_H','PossessionRec_H','CornersFor_H','CornersAgainst_H',
        'Availability_H','AvailabilityDiff','xGpp_H','xGppRec_H','xGCorners_H','xGCornersRec_H'
    ]
    away_cols = [
        'ShotConv_A','ShotConvRec_A','PointsPerGame_A','xGDiff_A',
        'CornersConv_A','CornersConvRec_A','NumMatches_A',
        'Elo_H','Elo_A','EloDiff',
        'GFvsMid_A','GAvsMid_A','GFvsHigh_A','GAvsHigh_A',
        'GFvsSim_A','GAvsSim_A',
        # Enriched
        'Possession_A','PossessionRec_A','CornersFor_A','CornersAgainst_A',
        'Availability_A','xGpp_A','xGppRec_A','xGCorners_A','xGCornersRec_A'
    ]
    # Ensure required columns exist
    for col in set(home_cols + away_cols):
        if col not in X.columns:
            X[col] = 0.0
    y_home = df.get('FTHG')
    y_away = df.get('FTAG')

    # Drop rows with missing labels (align safely)
    train_df = X.copy()
    train_df['FTHG'] = y_home.values if hasattr(y_home, 'values') else y_home
    train_df['FTAG'] = y_away.values if hasattr(y_away, 'values') else y_away
    train_df = train_df.dropna(subset=['FTHG','FTAG'])
    y_home = train_df['FTHG'].astype(float)
    y_away = train_df['FTAG'].astype(float)
    Xf = train_df.drop(columns=['FTHG','FTAG'], errors='ignore').astype(float).fillna(0.0)
    X_home = Xf[home_cols]
    X_away = Xf[away_cols]
    print(f"Training XGB for {league}: home={X_home.shape[1]} feats, away={X_away.shape[1]} feats, samples={len(X_home)}")
    home_model = XGBRegressor(objective='reg:squarederror', n_estimators=220, learning_rate=0.06, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42)
    away_model = XGBRegressor(objective='reg:squarederror', n_estimators=220, learning_rate=0.06, max_depth=5, subsample=0.9, colsample_bytree=0.9, random_state=42)
    home_model.fit(X_home, y_home)
    away_model.fit(X_away, y_away)

    out_dir = 'advanced_models'
    os.makedirs(out_dir, exist_ok=True)
    # Prefer JSON (native XGBoost) to avoid pickle compatibility warnings
    try:
        home_json = os.path.join(out_dir, f'{league}_ultimate_xgb_home.json')
        away_json = os.path.join(out_dir, f'{league}_ultimate_xgb_away.json')
        home_model.save_model(home_json)
        away_model.save_model(away_json)
    except Exception:
        pass
    # Also keep pickle for backward compatibility (optional)
    try:
        joblib.dump(home_model, os.path.join(out_dir, f'{league}_ultimate_xgb_home.pkl'))
        joblib.dump(away_model, os.path.join(out_dir, f'{league}_ultimate_xgb_away.pkl'))
    except Exception:
        pass
    print(f"Saved models for {league} in {out_dir} (.json preferred, .pkl fallback)")


def main():
    ap = argparse.ArgumentParser(description='Train XGB goal models per league (fallback features)')
    ap.add_argument('--league', required=True, help='League code, e.g., E0, D1, F1, SP1, I1')
    args = ap.parse_args()
    train_xgb_models_for_league(args.league)


if __name__ == '__main__':
    main()


