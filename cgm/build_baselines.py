"""
Milestone 2: Team & League Baselines + Elo Bands

Inputs:
  - data/enhanced/cgm_match_history.csv (from Milestone 1)
  - CGM data/goal statistics 2.csv (team season stats)
  - team_registry (for normalization)

Outputs:
  - data/enhanced/team_baselines.csv (per team-season summary with normalized attack/defense)
  - data/enhanced/cgm_match_history.csv (updated with EloDiff, Band_H, Band_A, league baselines)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from cgm.team_registry import build_team_registry, normalize_team  # type: ignore
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from team_registry import build_team_registry, normalize_team  # type: ignore

try:
    from cgm.pressure_inputs import ensure_pressure_inputs  # type: ignore
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pressure_inputs import ensure_pressure_inputs  # type: ignore


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _band_from_diff(diff: float, bully_thresh: float = 150.0) -> str:
    if diff >= bully_thresh:
        return "BULLY"
    if diff <= -bully_thresh:
        return "DOG"
    return "PEER"


HOME_ADV_DEFAULT = 65.0


def build_baselines(data_dir: str = "CGM data",
                    match_history_path: str = "data/enhanced/cgm_match_history.csv",
                    out_team_baselines: str = "data/enhanced/team_baselines.csv") -> None:
    reg = build_team_registry(data_dir)

    mh_path = Path(match_history_path)
    if not mh_path.exists():
        raise FileNotFoundError(f"{mh_path} not found (run Milestone 1 first).")
    mh = pd.read_csv(mh_path)

    # League averages (robust): derive from match history itself so we don't depend on
    # fragile/variant CGM exports for league-level snapshot tables.
    mh_avg = mh.copy()
    for col in ["ft_home", "ft_away"]:
        if col in mh_avg.columns:
            mh_avg[col] = pd.to_numeric(mh_avg[col], errors="coerce")
    mh_avg = ensure_pressure_inputs(mh_avg)
    for col in ["shots_H", "shots_A", "sot_H", "sot_A", "corners_H", "corners_A", "pos_H", "pos_A"]:
        if col in mh_avg.columns:
            mh_avg[col] = pd.to_numeric(mh_avg[col], errors="coerce")

    key_cols_fallback = ["league", "season"]
    use_country = "country" in mh_avg.columns and mh_avg["country"].notna().any()
    key_cols_primary = ["country", "league", "season"] if use_country else key_cols_fallback

    agg: dict[str, str] = {}
    if "ft_home" in mh_avg.columns:
        agg["ft_home"] = "mean"
    if "ft_away" in mh_avg.columns:
        agg["ft_away"] = "mean"
    if "shots_H" in mh_avg.columns:
        agg["shots_H"] = "mean"
    if "shots_A" in mh_avg.columns:
        agg["shots_A"] = "mean"
    if "sot_H" in mh_avg.columns:
        agg["sot_H"] = "mean"
    if "sot_A" in mh_avg.columns:
        agg["sot_A"] = "mean"
    if "corners_H" in mh_avg.columns:
        agg["corners_H"] = "mean"
    if "corners_A" in mh_avg.columns:
        agg["corners_A"] = "mean"
    if "pos_H" in mh_avg.columns:
        agg["pos_H"] = "mean"
    if "pos_A" in mh_avg.columns:
        agg["pos_A"] = "mean"

    league_lookup_primary = mh_avg.groupby(key_cols_primary, dropna=False).agg(agg).reset_index()
    league_lookup_primary = league_lookup_primary.rename(
        columns={
            "ft_home": "lg_avg_gf_home",
            "ft_away": "lg_avg_gf_away",
            "shots_H": "lg_avg_shots_home",
            "shots_A": "lg_avg_shots_away",
            "sot_H": "lg_avg_sot_home",
            "sot_A": "lg_avg_sot_away",
            "corners_H": "lg_avg_cor_home",
            "corners_A": "lg_avg_cor_away",
            "pos_H": "lg_avg_poss_home",
            "pos_A": "lg_avg_poss_away",
        }
    )
    league_lookup_fallback = None
    if use_country:
        league_lookup_fallback = mh_avg.groupby(key_cols_fallback, dropna=False).agg(agg).reset_index()
        league_lookup_fallback = league_lookup_fallback.rename(
            columns={
                "ft_home": "lg_avg_gf_home",
                "ft_away": "lg_avg_gf_away",
                "shots_H": "lg_avg_shots_home",
                "shots_A": "lg_avg_shots_away",
                "sot_H": "lg_avg_sot_home",
                "sot_A": "lg_avg_sot_away",
                "corners_H": "lg_avg_cor_home",
                "corners_A": "lg_avg_cor_away",
                "pos_H": "lg_avg_poss_home",
                "pos_A": "lg_avg_poss_away",
            }
        )

    # Build (team, season) -> league/country map from match history
    mh_map = (
        mh.groupby(["home", "season"])
        .agg({"league": lambda x: x.mode().iat[0] if not x.mode().empty else np.nan,
              "country": lambda x: x.mode().iat[0] if not x.mode().empty else np.nan})
        .reset_index()
        .rename(columns={"home": "team"})
    )

    # Load team season stats
    team_stats = _read_csv(Path(data_dir) / "goal statistics 2.csv")
    team_stats["team"] = team_stats.get("echipa", "").astype(str).apply(lambda x: normalize_team(x, reg))
    if "sezonul" in team_stats.columns:
        team_stats["season"] = team_stats["sezonul"].astype(str).str.replace(".0", "", regex=False)
    else:
        team_stats["season"] = np.nan
    team_stats["season"] = team_stats["season"].astype(str)
    mh_map["season"] = mh_map["season"].astype(str)
    # Fill league/country from match history mapping
    team_stats = team_stats.merge(mh_map, how="left", on=["team", "season"])
    if "league" not in team_stats.columns:
        team_stats["league"] = team_stats["league_y"] if "league_y" in team_stats.columns else np.nan
    if "country" not in team_stats.columns:
        team_stats["country"] = team_stats["country_y"] if "country_y" in team_stats.columns else np.nan
    team_stats.drop(columns=[c for c in ["league_y", "country_y"] if c in team_stats.columns], inplace=True)

    league_lookup_idx_primary = league_lookup_primary.set_index(key_cols_primary) if not league_lookup_primary.empty else None
    league_lookup_idx_fallback = None
    if league_lookup_fallback is not None and not league_lookup_fallback.empty:
        league_lookup_idx_fallback = league_lookup_fallback.set_index(key_cols_fallback)

    # Team baselines: compute attack/defense ratios vs league averages
    def _safe_ratio(num, den):
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f == 0 or np.isnan(den_f) or np.isnan(num_f):
                return np.nan
            return num_f / den_f
        except Exception:
            return np.nan

    team_baselines = []
    for _, r in team_stats.iterrows():
        league = r.get("league") if pd.notna(r.get("league")) else r.get("league_x") if "league_x" in r else None
        country = r.get("country") if pd.notna(r.get("country")) else r.get("country_x") if "country_x" in r else None
        season = r.get("season")
        lg_row = None
        if league_lookup_idx_primary is not None:
            try:
                key = (country, league, season) if use_country else (league, season)
                lg_row = league_lookup_idx_primary.loc[key]
            except Exception:
                lg_row = None
        if lg_row is None and league_lookup_idx_fallback is not None:
            try:
                lg_row = league_lookup_idx_fallback.loc[(league, season)]
            except Exception:
                lg_row = None
        if isinstance(lg_row, pd.DataFrame):
            lg_row = lg_row.iloc[0] if not lg_row.empty else None
        gf_h = r.get("gmh")
        ga_h = r.get("gph")
        gf_a = r.get("gma")
        ga_a = r.get("gpa")
        attack_h = _safe_ratio(gf_h, lg_row["lg_avg_gf_home"]) if lg_row is not None else np.nan
        attack_a = _safe_ratio(gf_a, lg_row["lg_avg_gf_away"]) if lg_row is not None else np.nan
        defense_h = _safe_ratio(ga_h, lg_row["lg_avg_gf_away"]) if lg_row is not None else np.nan
        defense_a = _safe_ratio(ga_a, lg_row["lg_avg_gf_home"]) if lg_row is not None else np.nan
        team_baselines.append({
            "team": r["team"],
            "season": season,
            "country": country,
            "league": league,
            "rating": r.get("rating"),
            "gmh": gf_h, "gph": ga_h, "gma": gf_a, "gpa": ga_a,
            "attack_home_vs_lg": attack_h,
            "attack_away_vs_lg": attack_a,
            "defense_home_vs_lg": defense_h,
            "defense_away_vs_lg": defense_a,
        })

    team_baselines_df = pd.DataFrame(team_baselines)
    out_team_path = Path(out_team_baselines)
    out_team_path.parent.mkdir(parents=True, exist_ok=True)
    team_baselines_df.to_csv(out_team_path, index=False)
    print(f"[ok] wrote team baselines -> {out_team_path} (rows={len(team_baselines_df)})")

    # Attach EloDiff and bands, plus league baselines to match history
    mh = mh.copy()
    for col in ["ft_home", "ft_away", "elo_home", "elo_away"]:
        if col in mh.columns:
            mh[col] = pd.to_numeric(mh[col], errors="coerce")
    # Home bonus heuristic: mean (ft_home - ft_away) per league/season -> convert to Elo-ish points.
    home_adv_primary = {}
    home_adv_fallback = {}
    if {"league", "season", "ft_home", "ft_away"}.issubset(mh.columns):
        home_adv_fallback = (
            mh.groupby(["league", "season"], group_keys=False)
            .apply(lambda g: (g["ft_home"] - g["ft_away"]).mean())
            .to_dict()
        )
    if use_country and {"country", "league", "season", "ft_home", "ft_away"}.issubset(mh.columns):
        home_adv_primary = (
            mh.groupby(["country", "league", "season"], group_keys=False)
            .apply(lambda g: (g["ft_home"] - g["ft_away"]).mean())
            .to_dict()
        )

    def _home_bonus(row):
        league = row.get("league")
        season = row.get("season")
        ga = None
        if use_country:
            ga = home_adv_primary.get((row.get("country"), league, season), None)
        if ga is None:
            ga = home_adv_fallback.get((league, season), 0.0)
        try:
            # Rough conversion: Elo bonus ≈ 400 * log10(1 + goal_advantage)
            ga_f = float(ga) if ga is not None else 0.0
            ga_f = float(np.clip(ga_f, -5.0, 5.0))
            bonus = 400.0 * float(np.sign(ga_f)) * float(np.log10(1.0 + abs(ga_f)))
            return float(np.clip(bonus, -200.0, 200.0))
        except Exception:
            return 0.0

    # Note: `home_bonus_elo` is a heuristic feature (derived from observed goal-diff).
    # We intentionally do NOT use it for `EloDiff` / bands because the rebuilt Elo series
    # uses a fixed home-advantage baseline (see `scripts/calc_cgm_elo.py --home-adv`),
    # and `EloDiff` must stay consistent with that to keep train/inference aligned.
    mh["home_bonus_elo"] = mh.apply(_home_bonus, axis=1)
    elo_home = mh["elo_home"].fillna(0.0) if "elo_home" in mh.columns else pd.Series(0.0, index=mh.index, dtype=float)
    elo_away = mh["elo_away"].fillna(0.0) if "elo_away" in mh.columns else pd.Series(0.0, index=mh.index, dtype=float)
    # `EloDiff` here is a *feature* baseline, not a learned/estimated league HFA.
    mh["EloDiff"] = (elo_home + HOME_ADV_DEFAULT) - elo_away
    mh["Band_H"] = mh["EloDiff"].apply(lambda x: _band_from_diff(x))
    mh["Band_A"] = (-mh["EloDiff"]).apply(lambda x: _band_from_diff(x))

    # Attach league averages
    lg_avg_cols = [c for c in league_lookup_primary.columns if c.startswith("lg_avg_")]
    mh = mh.drop(columns=[c for c in mh.columns if c.startswith("lg_avg_")], errors="ignore")
    mh = mh.merge(league_lookup_primary, how="left", on=key_cols_primary)
    if league_lookup_fallback is not None:
        mh_fb = mh.merge(league_lookup_fallback, how="left", on=key_cols_fallback, suffixes=("", "_fb"))
        for c in lg_avg_cols:
            fb = f"{c}_fb"
            if fb in mh_fb.columns:
                mh_fb[c] = mh_fb[c].fillna(mh_fb[fb])
        mh = mh_fb.drop(columns=[f"{c}_fb" for c in lg_avg_cols if f"{c}_fb" in mh_fb.columns], errors="ignore")

    # Write back updated match history
    mh.to_csv(mh_path, index=False)
    print(f"[ok] updated match history with baselines -> {mh_path} (rows={len(mh)}, cols={len(mh.columns)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build team/league baselines and update match history")
    ap.add_argument("--data-dir", default="CGM data", help="Directory containing CGM CSVs")
    ap.add_argument("--match-history", default="data/enhanced/cgm_match_history.csv", help="Path to match history CSV")
    ap.add_argument("--out-team-baselines", default="data/enhanced/team_baselines.csv", help="Output team baselines CSV")
    args = ap.parse_args()
    build_baselines(args.data_dir, args.match_history, args.out_team_baselines)


if __name__ == "__main__":
    main()
