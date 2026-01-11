"""
Build a unified CGM match history table (one row per played match).

Inputs (from "CGM data"):
  - multiple seasons.csv / multiple leagues and seasons/allratingv.csv (outcomes, Elo, CGM probs/odds)
  - goals statistics.csv or multiple leagues and seasons/upcoming.csv (shots, shots on target, corners, possession)
  - team_registry (normalize codes/names)

Output:
  data/enhanced/cgm_match_history.csv
    Columns: season, date, league, country, home, away, code_home, code_away,
             ft_home, ft_away, ht_home, ht_away, result, validated,
             elohome, eloaway, elodiff,
             homeprob, drawprob, awayprob, cotaa, cotae, cotad,
             overprob, underprob, cotao, cotau,
             sut, sutt, cor, ballph, ballpa, form indices (if present)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from cgm.team_registry import build_team_registry, normalize_team  # type: ignore
except ImportError:
    # Allow execution when run directly without package install
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from team_registry import build_team_registry, normalize_team  # type: ignore


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _normalize_df(df: pd.DataFrame, registry: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Add normalized names and code strings."""
    df = df.copy()
    def _col_series(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series([np.nan] * len(df), index=df.index)

    for col in ("codechipa1", "codechipa2"):
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            # Common in CGM exports: numeric-like codes serialized as "1002.0"
            s = s.str.replace(r"\.0$", "", regex=True)
            s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
            df[col] = s
    home_raw = _col_series("txtechipa1").where(lambda s: s.notna(), "")
    away_raw = _col_series("txtechipa2").where(lambda s: s.notna(), "")
    df["home"] = home_raw.astype(str).apply(lambda x: normalize_team(x, registry))
    df["away"] = away_raw.astype(str).apply(lambda x: normalize_team(x, registry))
    df["code_home"] = _col_series("codechipa1").astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    df["code_away"] = _col_series("codechipa2").astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return df


def _split_possession(val) -> Tuple[float | None, float | None]:
    try:
        s = str(val)
        if "-" in s:
            a, b = s.split("-", 1)
            return float(a), float(b)
    except Exception:
        return (None, None)
    return (None, None)


def _parse_date_only(s: pd.Series) -> pd.Series:
    """
    Deterministically parse dates from CGM exports.

    Mirrors cgm.backfill_match_stats._parse_date_only to keep joins consistent.
    """
    raw = s.astype(str).str.strip().replace({"nan": "", "None": "", "": ""})
    raw = raw.str.split().str[0]

    def _infer_ambiguous_slash_pref(vals: pd.Series, *, year_fmt: str) -> str:
        parts = vals.str.split("/", n=2, expand=True)
        a = pd.to_numeric(parts[0], errors="coerce")
        b = pd.to_numeric(parts[1], errors="coerce")
        mdy_def = int(((b > 12) & (a <= 12)).sum())
        dmy_def = int(((a > 12) & (b <= 12)).sum())
        if dmy_def > mdy_def:
            return f"%d/%m/{year_fmt}"
        return f"%m/%d/{year_fmt}"

    dt = pd.Series(pd.NaT, index=s.index)
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"]:
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        dt = dt.fillna(parsed)

    def _parse_slash(mask: pd.Series, *, year_fmt: str) -> None:
        if not mask.any():
            return
        vals = raw[mask]
        parts = vals.str.split("/", n=2, expand=True)
        a = pd.to_numeric(parts[0], errors="coerce")
        b = pd.to_numeric(parts[1], errors="coerce")
        mdy_unamb = (b > 12) & (a <= 12)
        dmy_unamb = (a > 12) & (b <= 12)
        amb_or_same = (a <= 12) & (b <= 12)
        amb_fmt = _infer_ambiguous_slash_pref(vals, year_fmt=year_fmt)
        fmt_mdy = f"%m/%d/{year_fmt}"
        fmt_dmy = f"%d/%m/{year_fmt}"
        parse_mdy = mdy_unamb | (amb_or_same & (amb_fmt == fmt_mdy))
        parse_dmy = dmy_unamb | (amb_or_same & (amb_fmt == fmt_dmy))
        if parse_mdy.any():
            dt.loc[vals.index[parse_mdy]] = pd.to_datetime(vals[parse_mdy], format=fmt_mdy, errors="coerce")
        if parse_dmy.any():
            dt.loc[vals.index[parse_dmy]] = pd.to_datetime(vals[parse_dmy], format=fmt_dmy, errors="coerce")

    _parse_slash(dt.isna() & raw.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False), year_fmt="%Y")
    _parse_slash(dt.isna() & raw.str.match(r"^\d{1,2}/\d{1,2}/\d{2}$", na=False), year_fmt="%y")

    for pat, fmt in [(r"^\d{1,2}-\d{1,2}-\d{4}$", "%d-%m-%Y"), (r"^\d{1,2}\.\d{1,2}\.\d{4}$", "%d.%m.%Y")]:
        mask = dt.isna() & raw.str.match(pat, na=False)
        if mask.any():
            dt.loc[mask] = pd.to_datetime(raw[mask], format=fmt, errors="coerce")

    return dt.dt.date


def build_match_history(
    data_dir: str = "CGM data",
    out_path: str = "data/enhanced/cgm_match_history.csv",
    *,
    max_date: str | None = None,
) -> Path:
    """
    Build the canonical CGM match history.

    max_date:
      Optional YYYY-MM-DD cutoff for the history (inclusive). When provided, the
      output is guaranteed to contain no rows after this date (prevents leakage
      when building features for a historical as-of run).
    """
    # Pandas 2.3+ returns tz-aware UTC timestamps; normalize to tz-naive to avoid
    # tz-aware vs tz-naive comparison errors when filtering parsed CSV datetimes.
    cutoff = pd.Timestamp.utcnow().tz_localize(None).normalize()
    if max_date:
        cutoff = pd.to_datetime(max_date, errors="coerce")
        cutoff = cutoff.tz_localize(None) if getattr(cutoff, "tzinfo", None) is not None else cutoff
        cutoff = cutoff.normalize() if not pd.isna(cutoff) else pd.Timestamp.utcnow().tz_localize(None).normalize()
    base = Path(data_dir)
    multi_league_dir = base / "multiple leagues and seasons"
    
    # DETERMINE PATHS
    # If the new multi-league folder exists, use it. Otherwise fallback to root.
    if multi_league_dir.exists():
        ms_path = multi_league_dir / "allratingv.csv"
        # For goals statistics, we need to aggregate all small CSVs in the folder
        # explicitly excluding the big combined files
        all_files = list(multi_league_dir.glob("*.csv"))
        stats_files = [
            f for f in all_files 
            if f.name.lower() not in ["allratingv.csv", "allags.csv", "allupcoming - copy.csv"]
            and ("upcoming" not in f.name.lower() or f.name.lower() == "upcoming.csv")
        ]
    else:
        # Legacy single-league fallback
        ms_path = base / "multiple seasons.csv"
        stats_files = [base / "goals statistics.csv"]

    if not ms_path.exists():
        raise FileNotFoundError(f"Required history file not found: {ms_path}")

    registry = build_team_registry(base)
    
    # 1. READ MATCH HISTORY (multiple seasons / allratingv.csv)
    logger.info(f"Loading match history from: {ms_path}")
    ms = _read_csv(ms_path)
    
    # Normalize before merge
    ms = _normalize_df(ms, registry)
    
    # 2. READ & AGGREGATE STATS (goals statistics / laliga.csv, etc)
    gs_list = []
    
    # Ensure we load the legacy stats file too if using multi-league (it often contains EPL stats)
    if multi_league_dir.exists():
        legacy_gs = base / "goals statistics.csv"
        if legacy_gs.exists():
            stats_files.append(legacy_gs)

    for sf in stats_files:
        if sf.exists():
            try:
                temp_df = _read_csv(sf)
                gs_list.append(temp_df)
            except Exception as e:
                logger.warning(f"Failed to read stats file {sf}: {e}")
    
    if gs_list:
        gs = pd.concat(gs_list, ignore_index=True)
    else:
        logger.warning("No statistics files found. Creating empty stats dataframe.")
        gs = pd.DataFrame()

    gs = _normalize_df(gs, registry)
    
    # Debug sizes
    logger.info(f"Match History (MS) Rows: {len(ms)}")
    logger.info(f"Goal Stats (GS) Rows: {len(gs)}")

    # Possession split
    if "ballp" in gs.columns:
        poss = gs["ballp"].apply(_split_possession)
        gs["ballph"] = poss.apply(lambda x: x[0])
        gs["ballpa"] = poss.apply(lambda x: x[1])

    # Merge on date + codes if possible, else date+names
    # Deterministic parsing to align with stats backfill.
    date_col = "datameci" if "datameci" in ms.columns else "date"
    ms["_date_only"] = _parse_date_only(ms[date_col])

    gs_date_col = "datameci" if "datameci" in gs.columns else "date"
    if not gs.empty:
        gs["_date_only"] = _parse_date_only(gs[gs_date_col])
    else:
        gs["_date_only"] = pd.Series(dtype="datetime64[ns]")
    
    # Try merging on codechipa if available (most reliable)
    if "codechipa1" in ms.columns and "codechipa1" in gs.columns:
        merge_keys = ["_date_only", "codechipa1", "codechipa2"]
    else:
        merge_keys = ["_date_only", "home", "away"]

    logger.info(f"Merging with keys: {merge_keys}")

    # Create temp merge frames to handle the date type explicitly
    merged = pd.merge(ms, gs, how="left", on=merge_keys, suffixes=("", "_gs"))
    
    logger.info(f"Merged Rows: {len(merged)}")
    
    # Drop temp date col
    if "_date_only" in merged.columns:
        merged = merged.drop(columns=["_date_only"])

    # Parse/clean season from 'sezonul' if present
    if "sezonul" in merged.columns:
        def _season_norm(x):
            try:
                s = str(x).strip()
                # Drop trailing '.0' from numeric-like strings
                if s.endswith(".0"):
                    s = s[:-2]
                if not s or s.lower() == "nan":
                    return np.nan
                if "-" in s:
                    return s
                if len(s) == 2 and s.isdigit():
                    y1 = int(s)
                    y_full = 2000 + y1 if y1 < 100 else y1
                    return f"{y_full}-{y_full+1}"
                if len(s) == 4 and s.isdigit():
                    y = int(s)
                    return f"{y}-{y+1}"
                return s
            except Exception:
                return np.nan
        merged["sezonul"] = merged["sezonul"].apply(_season_norm)

    # Deduplicate on date/home/away
    if {"datameci", "home", "away"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["datameci", "home", "away"])

    # Drop future matches (strict cutoff: max_date if provided, else today UTC)
    try:
        date_src = "datameci" if "datameci" in merged.columns else "date"
        parsed_dates = _parse_date_only(merged[date_src]) if date_src in merged.columns else pd.Series([pd.NaT] * len(merged))
        na_dates = int(pd.isna(parsed_dates).sum())
        if na_dates:
            logger.warning("Dropping %d rows with unparseable dates in match history.", na_dates)
        merged = merged[parsed_dates.notna()].copy()
        merged = merged[parsed_dates <= cutoff.date()].copy()
        merged["datetime"] = pd.to_datetime(merged[date_src], errors="coerce")
    except Exception as e:
        print(f"[warn] Could not filter future rows from CGM history: {e}")

    # Select/rename key columns
    out_cols = {
        "sezonul": "season",
        "datameci": "date",
        "orameci": "time",
        "country": "country",
        "league": "league",
        "home": "home",
        "away": "away",
        "code_home": "code_home",
        "code_away": "code_away",
        "scor1": "ft_home",
        "scor2": "ft_away",
        "scorp1": "ht_home",
        "scorp2": "ht_away",
        "result": "result",
        "validated": "validated",
        "elohomeo": "elo_home",
        "eloawayo": "elo_away",
        "elodiff": "elo_diff",
        "homeprob": "p_home",
        "drawprob": "p_draw",
        "awayprob": "p_away",
        "homeoddsc": "fair_home",
        "drawoddsc": "fair_draw",
        "awayoddsc": "fair_away",
        "cotaa": "odds_home",
        "cotae": "odds_draw",
        "cotad": "odds_away",
        "overprob": "p_over",
        "underprob": "p_under",
        "overoddsc": "fair_over",
        "underoddsc": "fair_under",
        "cotao": "odds_over",
        "cotau": "odds_under",
        "sut": "shots",
        "sutt": "shots_on_target",
        "cor": "corners",
        "ballph": "possession_home",
        "ballpa": "possession_away",
        "formah": "form_home",
        "formaa": "form_away",
    }

    df_out = pd.DataFrame()
    # Season fallback: carry season from cleaned sez, otherwise year from date if parseable
    if "sezonul" in merged.columns and "date" not in out_cols:
        # already handled below via out_cols mapping
        pass
    # Build outputs
    for src, dst in out_cols.items():
        if src in merged.columns:
            df_out[dst] = merged[src]
        else:
            df_out[dst] = np.nan
    # Fill missing season from date if possible
    if "season" in df_out.columns:
        def _season_from_date(val):
            try:
                dt = pd.to_datetime(val, errors="coerce", dayfirst=False)
                if pd.isna(dt):
                    dt = pd.to_datetime(val, errors="coerce", dayfirst=True)
                if pd.isna(dt):
                    return np.nan
                y = dt.year
                start = y if dt.month >= 8 else y - 1
                return f"{start}-{start+1}"
            except Exception:
                return np.nan
        mask = df_out["season"].isna()
        df_out.loc[mask, "season"] = df_out.loc[mask, "date"].apply(_season_from_date)
        # CGM exports sometimes encode season as a numeric end-year (e.g., "26" for 2025-2026).
        # If a parsed season disagrees with the date-derived season, prefer the date-derived value.
        try:
            expected = df_out["date"].apply(_season_from_date)
            bad = df_out["season"].notna() & expected.notna() & (df_out["season"].astype(str) != expected.astype(str))
            df_out.loc[bad, "season"] = expected.loc[bad]
        except Exception:
            pass

    # Drop any duplicate lg_avg_* suffix columns if present (keep base)
    lg_cols = [c for c in merged.columns if c.startswith("lg_avg_")]
    for col in lg_cols:
        if col.endswith("_gs") and col[:-3] in merged.columns:
            base = col[:-3]
            merged[base] = merged[base].fillna(merged[col])
            merged.drop(columns=[col], inplace=True, errors="ignore")

    # Ensure dirs and write
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_p, index=False)
    print(f"[ok] wrote match history -> {out_p} (rows={len(df_out)}, cols={len(df_out.columns)})")
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified CGM match history CSV")
    ap.add_argument("--data-dir", default="CGM data", help="Directory containing CGM CSVs")
    ap.add_argument("--out", default="data/enhanced/cgm_match_history.csv", help="Output CSV path")
    ap.add_argument("--max-date", default=None, help="Optional cutoff date (YYYY-MM-DD) to exclude future rows")
    args = ap.parse_args()
    build_match_history(args.data_dir, args.out, max_date=args.max_date)


if __name__ == "__main__":
    main()
