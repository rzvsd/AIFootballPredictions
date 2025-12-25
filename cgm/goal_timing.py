"""
Milestone 7.2: Goal timing profiles from CGM goal-minute exports.

This module turns per-match goal minute lists (e.g. from `CGM data/AGS.CSV`) into
team-level timing distributions and match-level timing probabilities.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


BIN_NAMES = ("0_15", "16_30", "31_45", "46_60", "61_75", "76_90")

MIN_PROFILE_MATCHES = 10
MIN_PROFILE_GOALS = 8

EARLY_SHARE_MARGIN = 0.05
LATE_SHARE_MARGIN = 0.07


NormalizeFn = Callable[[str], str]


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _bin_idx(minute: int) -> int:
    if minute <= 15:
        return 0
    if minute <= 30:
        return 1
    if minute <= 45:
        return 2
    if minute <= 60:
        return 3
    if minute <= 75:
        return 4
    return 5


def parse_goal_minutes(raw: object) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []

    out: list[int] = []
    for tok in re.split(r"[,\s;]+", s):
        tok = tok.strip()
        if not tok:
            continue
        tok = tok.lstrip("HA").strip()
        if "+" in tok:
            tok = tok.split("+", 1)[0].strip()
        try:
            m = int(tok)
        except Exception:
            continue
        if 0 <= m <= 130:
            out.append(m)
    return out


def _as_dist(counts: Iterable[int], total: int, prior: tuple[float, ...]) -> tuple[float, ...]:
    if total <= 0:
        return prior
    c = list(counts)
    if len(c) != len(prior):
        return prior
    return tuple(float(x) / float(total) for x in c)


@dataclass(frozen=True)
class TeamTimingProfile:
    team: str
    league: str | None
    matches: int
    goals_scored_total: int
    goals_conceded_total: int
    scored_dist: tuple[float, ...]
    conceded_dist: tuple[float, ...]
    early_goal_index: float
    late_goal_index: float
    early_concede_index: float
    late_concede_index: float
    usable: int


@dataclass(frozen=True)
class TimingMeta:
    source_path: str
    rows_in: int
    rows_used: int
    goals_total: int
    prior_scored_dist: tuple[float, ...]
    prior_conceded_dist: tuple[float, ...]
    prior_early_share: float
    prior_late_share: float


def build_team_timing_profiles(
    ags_path: Path,
    *,
    normalize: NormalizeFn,
    run_asof_datetime: pd.Timestamp,
    scope_league: str | None = None,
) -> tuple[dict[str, TeamTimingProfile], TimingMeta]:
    df = pd.read_csv(ags_path, sep=None, engine="python", encoding_errors="ignore")
    rows_in = int(len(df))

    # Expected columns in AGS export.
    date_col = "datamecic" if "datamecic" in df.columns else ("datameci" if "datameci" in df.columns else None)
    if date_col is None:
        raise ValueError(f"AGS file missing date column (expected datamecic/datameci): {ags_path}")
    if "txtechipa1" not in df.columns or "txtechipa2" not in df.columns:
        raise ValueError(f"AGS file missing team columns (txtechipa1/txtechipa2): {ags_path}")

    league_col = "league" if "league" in df.columns else None
    raw_dates = df[date_col].astype(str)
    dt = pd.to_datetime(raw_dates, errors="coerce", format="%d/%m/%y")
    dt2 = pd.to_datetime(raw_dates, errors="coerce", format="%d/%m/%Y")
    df["_dt"] = dt.fillna(dt2)
    df = df[df["_dt"].notna()].copy()
    df = df[df["_dt"] <= pd.to_datetime(run_asof_datetime)].copy()
    if scope_league and league_col:
        league_norm = df[league_col].astype(str).str.split("-", n=1).str[-1].str.strip()
        df = df[league_norm == str(scope_league)].copy()

    rows_used = int(len(df))

    matches: dict[str, int] = {}
    scored_counts: dict[str, list[int]] = {}
    conceded_counts: dict[str, list[int]] = {}
    goals_scored_total: dict[str, int] = {}
    goals_conceded_total: dict[str, int] = {}

    def _ensure(team: str) -> None:
        if team not in scored_counts:
            scored_counts[team] = [0] * len(BIN_NAMES)
            conceded_counts[team] = [0] * len(BIN_NAMES)
            goals_scored_total[team] = 0
            goals_conceded_total[team] = 0
            matches[team] = 0

    total_scored_bins = [0] * len(BIN_NAMES)
    total_conceded_bins = [0] * len(BIN_NAMES)
    goals_total = 0

    for _, r in df.iterrows():
        home = normalize(r.get("txtechipa1"))
        away = normalize(r.get("txtechipa2"))
        if not home or not away:
            continue

        _ensure(home)
        _ensure(away)
        matches[home] += 1
        matches[away] += 1

        mins_home = parse_goal_minutes(r.get("goalsh"))
        mins_away = parse_goal_minutes(r.get("goalsa"))

        for m in mins_home:
            b = _bin_idx(m)
            scored_counts[home][b] += 1
            conceded_counts[away][b] += 1
            goals_scored_total[home] += 1
            goals_conceded_total[away] += 1
            total_scored_bins[b] += 1
            total_conceded_bins[b] += 1
            goals_total += 1

        for m in mins_away:
            b = _bin_idx(m)
            scored_counts[away][b] += 1
            conceded_counts[home][b] += 1
            goals_scored_total[away] += 1
            goals_conceded_total[home] += 1
            total_scored_bins[b] += 1
            total_conceded_bins[b] += 1
            goals_total += 1

    if goals_total <= 0:
        prior = tuple([1.0 / len(BIN_NAMES)] * len(BIN_NAMES))
        meta = TimingMeta(
            source_path=str(ags_path),
            rows_in=rows_in,
            rows_used=rows_used,
            goals_total=0,
            prior_scored_dist=prior,
            prior_conceded_dist=prior,
            prior_early_share=float(prior[0] + prior[1]),
            prior_late_share=float(prior[-1]),
        )
        return {}, meta

    prior_scored = tuple(float(x) / float(goals_total) for x in total_scored_bins)
    prior_conceded = tuple(float(x) / float(goals_total) for x in total_conceded_bins)
    prior_early_share = float(prior_scored[0] + prior_scored[1])
    prior_late_share = float(prior_scored[-1])

    league_val = str(scope_league) if scope_league else (str(df[league_col].iloc[0]) if league_col and rows_used else None)
    profiles: dict[str, TeamTimingProfile] = {}
    for team in sorted(scored_counts.keys()):
        sc_total = int(goals_scored_total.get(team, 0) or 0)
        co_total = int(goals_conceded_total.get(team, 0) or 0)
        m = int(matches.get(team, 0) or 0)

        scored_dist = _as_dist(scored_counts[team], sc_total, prior_scored)
        conceded_dist = _as_dist(conceded_counts[team], co_total, prior_conceded)

        early_goal_index = float(scored_dist[0] + scored_dist[1])
        late_goal_index = float(scored_dist[-1])
        early_concede_index = float(conceded_dist[0] + conceded_dist[1])
        late_concede_index = float(conceded_dist[-1])

        usable = int((m >= MIN_PROFILE_MATCHES) and (sc_total >= MIN_PROFILE_GOALS) and (co_total >= MIN_PROFILE_GOALS))
        profiles[team] = TeamTimingProfile(
            team=team,
            league=league_val,
            matches=m,
            goals_scored_total=sc_total,
            goals_conceded_total=co_total,
            scored_dist=scored_dist,
            conceded_dist=conceded_dist,
            early_goal_index=early_goal_index,
            late_goal_index=late_goal_index,
            early_concede_index=early_concede_index,
            late_concede_index=late_concede_index,
            usable=usable,
        )

    meta = TimingMeta(
        source_path=str(ags_path),
        rows_in=rows_in,
        rows_used=rows_used,
        goals_total=goals_total,
        prior_scored_dist=prior_scored,
        prior_conceded_dist=prior_conceded,
        prior_early_share=prior_early_share,
        prior_late_share=prior_late_share,
    )
    return profiles, meta


def _avg_dist(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    if len(a) != len(b) or not a:
        return a
    return tuple(0.5 * (float(x) + float(y)) for x, y in zip(a, b))


def _p_ge_1(lmb: float) -> float:
    if not np.isfinite(lmb) or lmb < 0:
        return float("nan")
    return float(np.clip(1.0 - math.exp(-float(lmb)), 0.0, 1.0))


def _p_ge_2(lmb: float) -> float:
    if not np.isfinite(lmb) or lmb < 0:
        return float("nan")
    lam = float(lmb)
    return float(np.clip(1.0 - math.exp(-lam) * (1.0 + lam), 0.0, 1.0))


def compute_match_timing(
    *,
    home: str,
    away: str,
    mu_home: float,
    mu_away: float,
    profiles: dict[str, TeamTimingProfile],
    meta: TimingMeta | None,
) -> dict[str, float | int]:
    mh = float(mu_home)
    ma = float(mu_away)
    mu_total = mh + ma

    if meta is None:
        prior = tuple([1.0 / len(BIN_NAMES)] * len(BIN_NAMES))
        meta = TimingMeta(
            source_path="",
            rows_in=0,
            rows_used=0,
            goals_total=0,
            prior_scored_dist=prior,
            prior_conceded_dist=prior,
            prior_early_share=float(prior[0] + prior[1]),
            prior_late_share=float(prior[-1]),
        )

    ph = profiles.get(home)
    pa = profiles.get(away)
    home_ok = int(ph.usable) if ph is not None else 0
    away_ok = int(pa.usable) if pa is not None else 0
    timing_usable = int(bool(home_ok) and bool(away_ok))

    home_scored = ph.scored_dist if ph is not None else meta.prior_scored_dist
    home_conceded = ph.conceded_dist if ph is not None else meta.prior_conceded_dist
    away_scored = pa.scored_dist if pa is not None else meta.prior_scored_dist
    away_conceded = pa.conceded_dist if pa is not None else meta.prior_conceded_dist

    match_scoring_home = _avg_dist(home_scored, away_conceded)
    match_scoring_away = _avg_dist(away_scored, home_conceded)

    lam_bins = [0.0] * len(BIN_NAMES)
    for i in range(len(BIN_NAMES)):
        lam_bins[i] = (mh * float(match_scoring_home[i])) + (ma * float(match_scoring_away[i]))

    lam_early_0_30 = float(lam_bins[0] + lam_bins[1])
    lam_1h = float(lam_bins[0] + lam_bins[1] + lam_bins[2])
    lam_2h = float(lam_bins[3] + lam_bins[4] + lam_bins[5])
    lam_75p = float(lam_bins[5])

    timing_early_share = float(lam_early_0_30 / mu_total) if np.isfinite(mu_total) and mu_total > 0 else float("nan")
    timing_late_share = float(lam_75p / mu_total) if np.isfinite(mu_total) and mu_total > 0 else float("nan")

    slow_start_flag = int(
        np.isfinite(timing_early_share) and (timing_early_share < float(meta.prior_early_share - EARLY_SHARE_MARGIN))
    )
    late_goal_flag = int(
        np.isfinite(timing_late_share) and (timing_late_share > float(meta.prior_late_share + LATE_SHARE_MARGIN))
    )

    p_1h_over_0_5 = _p_ge_1(lam_1h)
    p_1h_under_0_5 = 1.0 - p_1h_over_0_5 if np.isfinite(p_1h_over_0_5) else float("nan")

    p_2h_over_0_5 = _p_ge_1(lam_2h)
    p_2h_under_0_5 = 1.0 - p_2h_over_0_5 if np.isfinite(p_2h_over_0_5) else float("nan")

    p_2h_over_1_5 = _p_ge_2(lam_2h)
    p_2h_under_1_5 = 1.0 - p_2h_over_1_5 if np.isfinite(p_2h_over_1_5) else float("nan")

    p_goal_after_75_yes = _p_ge_1(lam_75p)
    p_goal_after_75_no = 1.0 - p_goal_after_75_yes if np.isfinite(p_goal_after_75_yes) else float("nan")

    return {
        "timing_usable": int(timing_usable),
        "timing_home_usable": int(home_ok),
        "timing_away_usable": int(away_ok),
        "timing_home_matches": int(ph.matches) if ph is not None else 0,
        "timing_away_matches": int(pa.matches) if pa is not None else 0,
        "timing_home_goals_scored": int(ph.goals_scored_total) if ph is not None else 0,
        "timing_home_goals_conceded": int(ph.goals_conceded_total) if ph is not None else 0,
        "timing_away_goals_scored": int(pa.goals_scored_total) if pa is not None else 0,
        "timing_away_goals_conceded": int(pa.goals_conceded_total) if pa is not None else 0,
        "timing_early_share": timing_early_share,
        "timing_late_share": timing_late_share,
        "slow_start_flag": int(slow_start_flag),
        "late_goal_flag": int(late_goal_flag),
        "p_1h_over_0_5": float(p_1h_over_0_5),
        "p_1h_under_0_5": float(p_1h_under_0_5),
        "p_2h_over_0_5": float(p_2h_over_0_5),
        "p_2h_under_0_5": float(p_2h_under_0_5),
        "p_2h_over_1_5": float(p_2h_over_1_5),
        "p_2h_under_1_5": float(p_2h_under_1_5),
        "p_goal_after_75_yes": float(p_goal_after_75_yes),
        "p_goal_after_75_no": float(p_goal_after_75_no),
    }
