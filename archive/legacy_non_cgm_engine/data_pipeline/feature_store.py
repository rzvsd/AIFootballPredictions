from __future__ import annotations

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, Any

import config

# (content same as previous feature_store.py)
def build_snapshot(
    enhanced_csv: str,
    as_of: str | None = None,
    half_life_matches: int = 5,
    elo_k: float = 20.0,
    elo_home_adv: float = 60.0,
    micro_agg_path: str | None = None,
) -> pd.DataFrame:
    if not os.path.exists(enhanced_csv):
        raise FileNotFoundError(f"{enhanced_csv} not found.")
    df = pd.read_csv(enhanced_csv)
    if as_of:
        try:
            cutoff = pd.to_datetime(as_of)
            df = df[pd.to_datetime(df['Date'], errors='coerce') <= cutoff]
        except Exception:
            pass
    df['HomeTeamStd'] = df['HomeTeam'].astype(str).map(config.normalize_team_name)
    df['AwayTeamStd'] = df['AwayTeam'].astype(str).map(config.normalize_team_name)
    df.rename(columns={'HomeTeamStd': 'team'}, inplace=True)
    # Compute EWMA goals and Elo
    df['elo'] = _compute_elo(df, elo_k=elo_k, elo_home_adv=elo_home_adv)
    # Aggregate per team latest row
    df.sort_values(['team', 'Date'], inplace=True)
    snap = df.groupby('team', as_index=False).tail(1)
    snap.reset_index(drop=True, inplace=True)
    return snap


def _compute_elo(df: pd.DataFrame, elo_k: float = 20.0, elo_home_adv: float = 60.0) -> pd.Series:
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    elo = {t: 1500.0 for t in teams}
    elo_list = []
    for _, row in df.sort_values('Date').iterrows():
        h = row['HomeTeam']; a = row['AwayTeam']
        e_h = 1.0 / (1 + 10 ** ((elo[a] + elo_home_adv - elo[h]) / 400))
        e_a = 1.0 - e_h
        res = row.get('FTHG', 0) - row.get('FTAG', 0)
        s_h = 1.0 if res > 0 else (0.5 if res == 0 else 0.0)
        s_a = 1.0 - s_h
        elo[h] += elo_k * (s_h - e_h)
        elo[a] += elo_k * (s_a - e_a)
        elo_list.append(elo[h])
    return pd.Series(elo_list)
