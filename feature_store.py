# feature_store.py
import pandas as pd
import os
import json
import numpy as np

def build_feature_store():
    LEAGUE = 'E0'
    data_path = os.path.join('data', 'enhanced', f'{LEAGUE}_strength_adj.csv')
    output_path = os.path.join('data', 'enhanced', f'{LEAGUE}_feature_store.json')

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    feature_store = {}

    print(f"Building feature store for {len(all_teams)} teams...")

    for team in all_teams:
        last_home_game = df[df['HomeTeam'] == team].sort_values('Date').tail(1)
        last_away_game = df[df['AwayTeam'] == team].sort_values('Date').tail(1)
        
        if last_home_game.empty and last_away_game.empty: continue

        if last_home_game.empty: latest_game, is_home = last_away_game.iloc[0], False
        elif last_away_game.empty: latest_game, is_home = last_home_game.iloc[0], True
        else:
            if last_home_game['Date'].iloc[0] > last_away_game['Date'].iloc[0]:
                latest_game, is_home = last_home_game.iloc[0], True
            else:
                latest_game, is_home = last_away_game.iloc[0], False

        team_features = {}
        prefix = 'H' if is_home else 'A'
        
        for col in df.columns:
            if col.endswith(f'_{prefix}'):
                feature_name = col[:-2]
                value = latest_game[col]
                if isinstance(value, np.integer): team_features[feature_name] = int(value)
                elif isinstance(value, np.floating): team_features[feature_name] = float(value)
                else: team_features[feature_name] = value
        
        feature_store[team] = team_features

    with open(output_path, 'w') as f:
        json.dump(feature_store, f, indent=4)
        
    print(f"Feature store successfully built and saved to {output_path}")

if __name__ == "__main__":
    build_feature_store()


# --- New: snapshot builder used by one_click_predictor.py ---
def build_snapshot(enhanced_csv: str, as_of: str | None = None, half_life_matches: int = 5,
                   elo_k: float = 20.0, elo_home_adv: float = 60.0) -> pd.DataFrame:
    """
    Build a per-team snapshot DataFrame from an enhanced CSV.

    Returns columns expected by one_click_predictor:
      - team
      - xg_L5
      - xga_L5
      - xgdiff_L10 (xg_L5 - xga_L5)
      - gpg_L10    (approx from last-5 goals)
      - corners_L10
      - corners_allowed_L10 (opponent's average corners)

    Notes:
      - Uses the latest match per team up to `as_of` if provided (YYYY-MM-DD).
      - Falls back to NaN where source fields are missing.
    """
    # Determine league code
    base = os.path.basename(enhanced_csv)
    league = base.split('_')[0] if '_' in base else base.split('.')[0]

    # Prefer enhanced file; if missing or lacks labels, use processed for time-series stats
    use_proc_for_ts = False
    if not os.path.exists(enhanced_csv):
        proc_path = os.path.join('data','processed', f'{league}_merged_preprocessed.csv')
        if not os.path.exists(proc_path):
            raise FileNotFoundError(f"Enhanced CSV not found: {enhanced_csv} and processed CSV not found: {proc_path}")
        df = pd.read_csv(proc_path)
        use_proc_for_ts = True
    else:
        df = pd.read_csv(enhanced_csv)
        if not set(['FTHG','FTAG']).issubset(df.columns):
            # load processed for time series labels
            proc_path = os.path.join('data','processed', f'{league}_merged_preprocessed.csv')
            if os.path.exists(proc_path):
                df = pd.read_csv(proc_path)
                use_proc_for_ts = True
    # If labels missing, merge from raw to ensure EWMA updates
    if not set(['FTHG','FTAG']).issubset(df.columns) or df['FTHG'].isna().all() if 'FTHG' in df.columns else True:
        raw_path = os.path.join('data','raw', f'{league}_merged.csv')
        if os.path.exists(raw_path):
            raw_df = pd.read_csv(raw_path, parse_dates=['Date'], dayfirst=True)
            if 'Date' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                except Exception:
                    pass
            keys = [c for c in ['Date','HomeTeam','AwayTeam'] if c in df.columns and c in raw_df.columns]
            if keys:
                df = pd.merge(df, raw_df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']], on=keys, how='left', suffixes=('', '_raw'))
    # Parse dates and filter as_of
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date','HomeTeam','AwayTeam'])
        if as_of:
            try:
                cutoff = pd.to_datetime(as_of)
                df = df[df['Date'] <= cutoff]
            except Exception:
                pass
    df = df.sort_values('Date')

    # Build EWMA (home/away) and Elo per team
    alpha = 1 - 0.5**(1/float(half_life_matches))
    state = {}
    elo = {}
    def st(team):
        if team not in state:
            state[team] = {
                'h_gf': 1.2, 'h_ga': 1.2, 'h_ppg': 1.2, 'h_n': 0,
                'a_gf': 1.0, 'a_ga': 1.0, 'a_ppg': 1.0, 'a_n': 0,
                # Elo-band splits
                'h_gf_mid': 1.2, 'h_ga_mid': 1.2, 'h_gf_high': 1.2, 'h_ga_high': 1.2,
                'a_gf_mid': 1.0, 'a_ga_mid': 1.0, 'a_gf_high': 1.0, 'a_ga_high': 1.0,
            }
        return state[team]
    def elo_get(team):
        return elo.get(team, 1500.0)
    teams = sorted(set(df['HomeTeam'].astype(str)) | set(df['AwayTeam'].astype(str)))
    # We will create the snapshot after scanning all matches
    for _, r in df.iterrows():
        home = str(r['HomeTeam']); away = str(r['AwayTeam'])
        fthg = float(r.get('FTHG', np.nan)); ftag = float(r.get('FTAG', np.nan))
        if np.isnan(fthg) or np.isnan(ftag):
            continue
        # record pre-match Elo for both teams
        eh = elo_get(home); ea = elo_get(away)
        # Update EWMA states (pre-match features would read previous state; here we just maintain running)
        sh = st(home); sa = st(away)
        # Update with match outcomes
        # home
        sh['h_gf'] = (1-alpha)*sh['h_gf'] + alpha*fthg
        sh['h_ga'] = (1-alpha)*sh['h_ga'] + alpha*ftag
        # ppg: 3/1/0 from FTR
        S_home = 3.0 if fthg>ftag else (1.0 if fthg==ftag else 0.0)
        sh['h_ppg'] = (1-alpha)*sh['h_ppg'] + alpha*S_home
        sh['h_n'] += 1
        # away
        sa['a_gf'] = (1-alpha)*sa['a_gf'] + alpha*ftag
        sa['a_ga'] = (1-alpha)*sa['a_ga'] + alpha*fthg
        S_away = 3.0 if ftag>fthg else (1.0 if ftag==fthg else 0.0)
        sa['a_ppg'] = (1-alpha)*sa['a_ppg'] + alpha*S_away
        sa['a_n'] += 1
        # Elo update
        exp_home = 1.0/(1.0 + 10**(-((eh + elo_home_adv) - ea)/400.0))
        S_home_bin = 1.0 if fthg>ftag else (0.5 if fthg==ftag else 0.0)
        elo[home] = eh + elo_k*(S_home_bin - exp_home)
        elo[away] = ea + elo_k*((1.0-S_home_bin) - (1.0-exp_home))

        # Update Elo-band EWMAs by opponent Elo
        # Home vs away's pre-match Elo (ea)
        if 1600.0 <= ea <= 1800.0:
            sh['h_gf_mid'] = (1-alpha)*sh['h_gf_mid'] + alpha*fthg
            sh['h_ga_mid'] = (1-alpha)*sh['h_ga_mid'] + alpha*ftag
        elif ea > 1800.0:
            sh['h_gf_high'] = (1-alpha)*sh['h_gf_high'] + alpha*fthg
            sh['h_ga_high'] = (1-alpha)*sh['h_ga_high'] + alpha*ftag
        # Away vs home pre-match Elo (eh)
        if 1600.0 <= eh <= 1800.0:
            sa['a_gf_mid'] = (1-alpha)*sa['a_gf_mid'] + alpha*ftag
            sa['a_ga_mid'] = (1-alpha)*sa['a_ga_mid'] + alpha*fthg
        elif eh > 1800.0:
            sa['a_gf_high'] = (1-alpha)*sa['a_gf_high'] + alpha*ftag
            sa['a_ga_high'] = (1-alpha)*sa['a_ga_high'] + alpha*fthg

    # Build output snapshot per team
    rows = []
    for team in teams:
        s = st(team)
        rows.append({
            'team': team,
            'xg_home_EWMA': float(s['h_gf']),
            'xga_home_EWMA': float(s['h_ga']),
            'xg_away_EWMA': float(s['a_gf']),
            'xga_away_EWMA': float(s['a_ga']),
            'ppg_home_EWMA': float(s['h_ppg']),
            'ppg_away_EWMA': float(s['a_ppg']),
            'elo': float(elo_get(team)),
            # Elo-band splits
            'GFvsMid_H': float(s['h_gf_mid']),
            'GAvsMid_H': float(s['h_ga_mid']),
            'GFvsHigh_H': float(s['h_gf_high']),
            'GAvsHigh_H': float(s['h_ga_high']),
            'GFvsMid_A': float(s['a_gf_mid']),
            'GAvsMid_A': float(s['a_ga_mid']),
            'GFvsHigh_A': float(s['a_gf_high']),
            'GAvsHigh_A': float(s['a_ga_high']),
            # legacy columns for backward compatibility
            'xg_L5': float((s['h_gf']+s['a_gf'])/2.0),
            'xga_L5': float((s['h_ga']+s['a_ga'])/2.0),
            'gpg_L10': float((s['h_ppg']+s['a_ppg'])/2.0),
            'corners_L10': np.nan,
            'corners_allowed_L10': np.nan,
        })
    snap = pd.DataFrame(rows)
    # Merge Absences/Availability MVP (optional)
    try:
        base = os.path.basename(enhanced_csv)
        league = base.split('_')[0] if '_' in base else base.split('.')[0]
        abs_path = os.path.join('data','absences', f'{league}_availability.csv')
        if os.path.exists(abs_path):
            av = pd.read_csv(abs_path)
            av['team'] = av['team'].astype(str)
            if 'date' in av.columns:
                av['date'] = pd.to_datetime(av['date'], errors='coerce')
                cutoff = pd.to_datetime(as_of) if as_of else None
                if cutoff is not None:
                    av = av[av['date'] <= cutoff]
            # keep latest per team
            av = av.sort_values('date' if 'date' in av.columns else 'team')
            # If positional columns present, compute weighted availability
            if set(['availability_gk','availability_def','availability_mid','availability_fwd']).issubset(av.columns):
                av_idx = av.groupby('team').tail(1)[['team','availability_gk','availability_def','availability_mid','availability_fwd']]
                snap = snap.merge(av_idx, on='team', how='left')
                # weights (can be overridden later to features)
                wgk, wdef, wmid, wfwd = 0.15, 0.35, 0.30, 0.20
                snap['availability_index'] = (
                    snap.get('availability_gk', 1.0).fillna(1.0) * wgk +
                    snap.get('availability_def', 1.0).fillna(1.0) * wdef +
                    snap.get('availability_mid', 1.0).fillna(1.0) * wmid +
                    snap.get('availability_fwd', 1.0).fillna(1.0) * wfwd
                )
            else:
                av_idx = av.groupby('team').tail(1)[['team','availability_index']]
                snap = snap.merge(av_idx, on='team', how='left')
                snap['availability_index'] = snap['availability_index'].fillna(1.0)
        else:
            snap['availability_index'] = 1.0
    except Exception:
        snap['availability_index'] = 1.0
    return snap
