from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class StrictMuResult:
    mu_home_raw: float
    mu_away_raw: float
    anchor_home: float
    anchor_away: float
    elo_home: float
    elo_away: float
    xg_home: float
    xg_away: float
    pressure_home: float
    pressure_away: float
    neutralized_modules: list[str]


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def _finite(values: Iterable[object]) -> list[float]:
    out: list[float] = []
    for value in values:
        f = _safe_float(value)
        if np.isfinite(f):
            out.append(float(f))
    return out


def _avg_or_default(values: Iterable[object], default: float) -> float:
    clean = _finite(values)
    if not clean:
        return float(default)
    return float(np.mean(clean))


def _normalize_weights(raw: dict[str, float] | None) -> dict[str, float]:
    weights = {
        "league_anchor": _safe_float((raw or {}).get("league_anchor"), 0.40),
        "elo": _safe_float((raw or {}).get("elo"), 0.10),
        "xg": _safe_float((raw or {}).get("xg"), 0.25),
        "pressure": _safe_float((raw or {}).get("pressure"), 0.25),
    }
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        raise ValueError("Strict mu weights must sum to a positive value")
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def _resolve_anchor(
    *,
    lg_avg_gf_home: object,
    lg_avg_gf_away: object,
    default_home: float,
    default_away: float,
    goals_clip_min: float,
    goals_clip_max: float,
) -> tuple[float, float]:
    home = _safe_float(lg_avg_gf_home, default_home)
    away = _safe_float(lg_avg_gf_away, default_away)
    return (
        _clip(home, goals_clip_min, goals_clip_max),
        _clip(away, goals_clip_min, goals_clip_max),
    )


def _elo_module(
    *,
    anchor_home: float,
    anchor_away: float,
    gf_home_vs_sim: object,
    ga_home_vs_sim: object,
    gf_away_vs_sim: object,
    ga_away_vs_sim: object,
    goals_clip_min: float,
    goals_clip_max: float,
) -> tuple[float, float, bool]:
    home = _avg_or_default([gf_home_vs_sim, ga_away_vs_sim], anchor_home)
    away = _avg_or_default([gf_away_vs_sim, ga_home_vs_sim], anchor_away)
    usable = bool(_finite([gf_home_vs_sim, ga_home_vs_sim, gf_away_vs_sim, ga_away_vs_sim]))
    return (
        _clip(home, goals_clip_min, goals_clip_max),
        _clip(away, goals_clip_min, goals_clip_max),
        usable,
    )


def _xg_module(
    *,
    anchor_home: float,
    anchor_away: float,
    xg_for_home: object,
    xg_against_home: object,
    xg_for_away: object,
    xg_against_away: object,
    usable: bool,
    goals_clip_min: float,
    goals_clip_max: float,
) -> tuple[float, float, bool]:
    if not usable:
        return anchor_home, anchor_away, False
    home = _avg_or_default([xg_for_home, xg_against_away], anchor_home)
    away = _avg_or_default([xg_for_away, xg_against_home], anchor_away)
    usable_now = bool(_finite([xg_for_home, xg_against_home, xg_for_away, xg_against_away]))
    return (
        _clip(home, goals_clip_min, goals_clip_max),
        _clip(away, goals_clip_min, goals_clip_max),
        usable_now,
    )


def _pressure_module(
    *,
    anchor_home: float,
    anchor_away: float,
    press_form_home: object,
    press_form_away: object,
    dom_home: dict[str, float] | None,
    dom_away: dict[str, float] | None,
    usable: bool,
    share_clip_min: float,
    share_clip_max: float,
    total_clip_min: float,
    total_clip_max: float,
    total_strength: float,
    goals_clip_min: float,
    goals_clip_max: float,
) -> tuple[float, float, bool]:
    if not usable:
        return anchor_home, anchor_away, False

    press_h = _safe_float(press_form_home, 0.5)
    press_a = _safe_float(press_form_away, 0.5)
    anchor_total = max(anchor_home + anchor_away, goals_clip_min * 2.0)
    anchor_share_home = anchor_home / anchor_total

    attack_h = _avg_or_default([press_h, 1.0 - press_a], 0.5)
    attack_a = _avg_or_default([press_a, 1.0 - press_h], 0.5)
    share_den = attack_h + attack_a
    if share_den <= 0:
        share_home = anchor_share_home
    else:
        share_home = attack_h / share_den
    share_home = _clip(share_home, share_clip_min, share_clip_max)

    dom_h_vals = _finite((dom_home or {}).values())
    dom_a_vals = _finite((dom_away or {}).values())
    tempo_h = _avg_or_default([press_h] + dom_h_vals, 0.5)
    tempo_a = _avg_or_default([press_a] + dom_a_vals, 0.5)
    tempo = _avg_or_default([tempo_h, tempo_a], 0.5)
    total_mult = _clip(
        1.0 + (float(total_strength) * (tempo - 0.5)),
        total_clip_min,
        total_clip_max,
    )
    module_total = anchor_total * total_mult
    home = module_total * share_home
    away = module_total * (1.0 - share_home)
    usable_now = bool(np.isfinite(press_h) and np.isfinite(press_a))
    return (
        _clip(home, goals_clip_min, goals_clip_max),
        _clip(away, goals_clip_min, goals_clip_max),
        usable_now,
    )


def compute_strict_weighted_mu(
    *,
    weights: dict[str, float] | None,
    lg_avg_gf_home: object,
    lg_avg_gf_away: object,
    default_anchor_home: float,
    default_anchor_away: float,
    gf_home_vs_sim: object,
    ga_home_vs_sim: object,
    gf_away_vs_sim: object,
    ga_away_vs_sim: object,
    xg_for_home: object,
    xg_against_home: object,
    xg_for_away: object,
    xg_against_away: object,
    xg_usable: bool,
    press_form_home: object,
    press_form_away: object,
    dom_home: dict[str, float] | None,
    dom_away: dict[str, float] | None,
    pressure_usable: bool,
    goals_clip_min: float,
    goals_clip_max: float,
    pressure_share_clip_min: float,
    pressure_share_clip_max: float,
    pressure_total_clip_min: float,
    pressure_total_clip_max: float,
    pressure_total_strength: float,
) -> StrictMuResult:
    w = _normalize_weights(weights)
    anchor_home, anchor_away = _resolve_anchor(
        lg_avg_gf_home=lg_avg_gf_home,
        lg_avg_gf_away=lg_avg_gf_away,
        default_home=default_anchor_home,
        default_away=default_anchor_away,
        goals_clip_min=goals_clip_min,
        goals_clip_max=goals_clip_max,
    )

    elo_home, elo_away, elo_usable = _elo_module(
        anchor_home=anchor_home,
        anchor_away=anchor_away,
        gf_home_vs_sim=gf_home_vs_sim,
        ga_home_vs_sim=ga_home_vs_sim,
        gf_away_vs_sim=gf_away_vs_sim,
        ga_away_vs_sim=ga_away_vs_sim,
        goals_clip_min=goals_clip_min,
        goals_clip_max=goals_clip_max,
    )
    xg_home, xg_away, xg_usable_now = _xg_module(
        anchor_home=anchor_home,
        anchor_away=anchor_away,
        xg_for_home=xg_for_home,
        xg_against_home=xg_against_home,
        xg_for_away=xg_for_away,
        xg_against_away=xg_against_away,
        usable=bool(xg_usable),
        goals_clip_min=goals_clip_min,
        goals_clip_max=goals_clip_max,
    )
    pressure_home, pressure_away, pressure_usable_now = _pressure_module(
        anchor_home=anchor_home,
        anchor_away=anchor_away,
        press_form_home=press_form_home,
        press_form_away=press_form_away,
        dom_home=dom_home,
        dom_away=dom_away,
        usable=bool(pressure_usable),
        share_clip_min=pressure_share_clip_min,
        share_clip_max=pressure_share_clip_max,
        total_clip_min=pressure_total_clip_min,
        total_clip_max=pressure_total_clip_max,
        total_strength=pressure_total_strength,
        goals_clip_min=goals_clip_min,
        goals_clip_max=goals_clip_max,
    )

    neutralized: list[str] = []
    if not elo_usable:
        neutralized.append("elo")
    if not xg_usable_now:
        neutralized.append("xg")
    if not pressure_usable_now:
        neutralized.append("pressure")

    mu_home_raw = (
        w["league_anchor"] * anchor_home
        + w["elo"] * elo_home
        + w["xg"] * xg_home
        + w["pressure"] * pressure_home
    )
    mu_away_raw = (
        w["league_anchor"] * anchor_away
        + w["elo"] * elo_away
        + w["xg"] * xg_away
        + w["pressure"] * pressure_away
    )

    return StrictMuResult(
        mu_home_raw=_clip(mu_home_raw, goals_clip_min, goals_clip_max),
        mu_away_raw=_clip(mu_away_raw, goals_clip_min, goals_clip_max),
        anchor_home=anchor_home,
        anchor_away=anchor_away,
        elo_home=elo_home,
        elo_away=elo_away,
        xg_home=xg_home,
        xg_away=xg_away,
        pressure_home=pressure_home,
        pressure_away=pressure_away,
        neutralized_modules=neutralized,
    )
