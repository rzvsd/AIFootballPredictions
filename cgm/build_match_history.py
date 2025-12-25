"""
Build a unified CGM match history table (one row per played match).

Inputs (from "CGM data"):
  - multiple seasons.csv   (outcomes, Elo, CGM probs/odds)
  - goals statistics.csv   (shots, shots on target, corners, possession)
  - team_registry          (normalize codes/names)

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


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _normalize_df(df: pd.DataFrame, registry: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Add normalized names and code strings."""
    df = df.copy()
    for col in ("codechipa1", "codechipa2"):
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            # Common in CGM exports: numeric-like codes serialized as "1002.0"
            s = s.str.replace(r"\.0$", "", regex=True)
            s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
            df[col] = s
    df["home"] = df.get("txtechipa1", "").astype(str).apply(lambda x: normalize_team(x, registry))
    df["away"] = df.get("txtechipa2", "").astype(str).apply(lambda x: normalize_team(x, registry))
    df["code_home"] = df.get("codechipa1", "").astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    df["code_away"] = df.get("codechipa2", "").astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
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
    ms_path = base / "multiple seasons.csv"
    gs_path = base / "goals statistics.csv"
    if not ms_path.exists() or not gs_path.exists():
        raise FileNotFoundError("Required CGM CSVs not found in 'CGM data'")

    registry = build_team_registry(base)

    # Base matches (multiple seasons)
    ms = _read_csv(ms_path)
    ms = _normalize_df(ms, registry)

    # Stats per match (goals statistics)
    gs = _read_csv(gs_path)
    gs = _normalize_df(gs, registry)
    # Possession split
    if "ballp" in gs.columns:
        poss = gs["ballp"].apply(_split_possession)
        gs["ballph"] = poss.apply(lambda x: x[0])
        gs["ballpa"] = poss.apply(lambda x: x[1])

    # Merge on season/date + codes if present, else names/date
    merge_keys = []
    if all(k in ms.columns for k in ["sezonul", "datameci", "codechipa1", "codechipa2"]):
        merge_keys = ["sezonul", "datameci", "codechipa1", "codechipa2"]
    elif all(k in ms.columns for k in ["datameci", "home", "away"]):
        merge_keys = ["datameci", "home", "away"]

    if not merge_keys:
        raise ValueError("Cannot determine merge keys for CGM match history.")

    merged = pd.merge(ms, gs, how="left", on=merge_keys, suffixes=("", "_gs"))

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
        merged["datetime"] = pd.to_datetime(merged["datameci"], errors="coerce")
        # Keep unparsable rows (NaT); filter only where datetime is known.
        merged = merged[merged["datetime"].isna() | (merged["datetime"] <= cutoff)]
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
