"""
team_registry.py — Canonical Team IDs per league

Provides a small registry that maps canonical (normalized) team names to
stable integer IDs per league. Built from enhanced/processed data the first
time and persisted under data/store/{LEAGUE}_team_registry.json.

Public API:
- ensure_registry(league: str) -> dict[name->id]
- get_team_id(league: str, name: str) -> int | None
- get_registry_df(league: str) -> pandas.DataFrame
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

import config


def _registry_path(league: str) -> str:
    return os.path.join('data', 'store', f'{league}_team_registry.json')


def _collect_teams(league: str) -> List[str]:
    enh = os.path.join('data', 'enhanced', f'{league}_final_features.csv')
    if os.path.exists(enh):
        try:
            df = pd.read_csv(enh)
            teams = sorted(set(df.get('HomeTeam', pd.Series(dtype=str)).astype(str)) |
                           set(df.get('AwayTeam', pd.Series(dtype=str)).astype(str)))
            if teams:
                return teams
        except Exception:
            pass
    proc = os.path.join('data', 'processed', f'{league}_merged_preprocessed.csv')
    if os.path.exists(proc):
        try:
            df = pd.read_csv(proc)
            teams = sorted(set(df.get('HomeTeam', pd.Series(dtype=str)).astype(str)) |
                           set(df.get('AwayTeam', pd.Series(dtype=str)).astype(str)))
            if teams:
                return teams
        except Exception:
            pass
    # Fallback: values from TEAM_NAME_MAP (may include other leagues)
    try:
        vals = sorted(set(config.TEAM_NAME_MAP.values()))
        return vals
    except Exception:
        return []


def ensure_registry(league: str) -> Dict[str, int]:
    path = _registry_path(league)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                if isinstance(obj, dict) and obj:
                    return {str(k): int(v) for k, v in obj.items()}
        except Exception:
            pass
    teams = _collect_teams(league)
    # Normalize names and assign deterministic IDs (sorted order)
    names = sorted(set(config.normalize_team_name(t) for t in teams))
    reg = {name: i + 1 for i, name in enumerate(names)}  # IDs start at 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(reg, f, indent=2)
    return reg


def get_team_id(league: str, name: str) -> int | None:
    reg = ensure_registry(league)
    return reg.get(config.normalize_team_name(name))


def get_registry_df(league: str) -> pd.DataFrame:
    reg = ensure_registry(league)
    return pd.DataFrame({'team': list(reg.keys()), 'team_id': list(reg.values())})


__all__ = ['ensure_registry', 'get_team_id', 'get_registry_df']

