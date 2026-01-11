"""
Milestone 2 (support): Backfill per-match gameplay stats into the canonical CGM history.

Why:
  The Pressure Cooker features depend on per-match shots/SOT/corners/possession.
  If the enhanced history is missing those stats, the pressure features collapse to neutral 0.5.

Inputs:
  - data/enhanced/cgm_match_history_with_elo.csv (canonical history + Elo)
  - CGM data/goals statistics.csv or CGM data/multiple leagues and seasons/upcoming.csv
    (per-match stats; split columns or combined "H-A" strings: sut/sutt/cor/ballp)

Output:
  - data/enhanced/cgm_match_history_with_elo_stats.csv (history + split numeric stat columns)

Join keys (priority):
  1) date + code_home + code_away
  2) date + normalized home/away names (fallback)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cgm.pressure_inputs import ensure_pressure_inputs

try:
    from cgm.team_registry import build_team_registry, normalize_team  # type: ignore
except Exception:  # pragma: no cover - execution fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from team_registry import build_team_registry, normalize_team  # type: ignore


STATS_COLS = [
    "shots_H",
    "shots_A",
    "sot_H",
    "sot_A",
    "corners_H",
    "corners_A",
    "pos_H",
    "pos_A",
]


def _read_table(path: Path) -> pd.DataFrame:
    """
    Read CGM exports from either CSV or Excel (xls/xlsx).

    Note: `.xls` requires `xlrd>=2.0.1`.
    """
    suf = path.suffix.lower()
    if suf in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")


def _parse_date_only(s: pd.Series, *, label: str | None = None) -> pd.Series:
    """
    Deterministically parse dates from CGM exports.

    CGM files can mix formats; avoid pandas' non-deterministic per-element inference by:
      - parsing common explicit formats first
      - inferring month/day order for slash-formats (MM/DD vs DD/MM) from unambiguous rows
    """
    raw = s.astype(str).str.strip().replace({"nan": "", "None": "", "": ""})
    # If time is embedded, keep only the first token (date part).
    raw = raw.str.split().str[0]

    def _infer_ambiguous_slash_pref(vals: pd.Series, *, year_fmt: str) -> str:
        parts = vals.str.split("/", n=2, expand=True)
        a = pd.to_numeric(parts[0], errors="coerce")
        b = pd.to_numeric(parts[1], errors="coerce")
        # If b>12 then b cannot be month => MM/DD (month-first).
        mdy_def = int(((b > 12) & (a <= 12)).sum())
        # If a>12 then a cannot be month => DD/MM (day-first).
        dmy_def = int(((a > 12) & (b <= 12)).sum())
        if dmy_def > mdy_def:
            return f"%d/%m/{year_fmt}"
        return f"%m/%d/{year_fmt}"

    dt = pd.Series(pd.NaT, index=s.index)
    # ISO / year-first
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
        amb = (a <= 12) & (b <= 12) & (a != b)
        amb_or_same = (a <= 12) & (b <= 12)  # includes a==b (ambiguous but same result)
        bad = a.isna() | b.isna() | ((a > 12) & (b > 12))

        amb_fmt = _infer_ambiguous_slash_pref(vals, year_fmt=year_fmt)

        if label is not None:
            print(
                "[date_parse]",
                {
                    "label": str(label),
                    "year_fmt": str(year_fmt),
                    "slash_rows": int(len(vals)),
                    "mdy_def": int(mdy_unamb.sum()),
                    "dmy_def": int(dmy_unamb.sum()),
                    "ambiguous": int(amb.sum()),
                    "ambiguous_or_same": int(amb_or_same.sum()),
                    "invalid": int(bad.sum()),
                    "chosen_for_ambiguous": amb_fmt,
                },
            )

        fmt_mdy = f"%m/%d/{year_fmt}"
        fmt_dmy = f"%d/%m/{year_fmt}"
        parse_mdy = mdy_unamb | (amb_or_same & (amb_fmt == fmt_mdy))
        parse_dmy = dmy_unamb | (amb_or_same & (amb_fmt == fmt_dmy))

        if parse_mdy.any():
            dt.loc[vals.index[parse_mdy]] = pd.to_datetime(vals[parse_mdy], format=fmt_mdy, errors="coerce")
        if parse_dmy.any():
            dt.loc[vals.index[parse_dmy]] = pd.to_datetime(vals[parse_dmy], format=fmt_dmy, errors="coerce")

    # Slash dates (4-digit year)
    _parse_slash(dt.isna() & raw.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False), year_fmt="%Y")
    # Slash dates (2-digit year)
    _parse_slash(dt.isna() & raw.str.match(r"^\d{1,2}/\d{1,2}/\d{2}$", na=False), year_fmt="%y")

    # Dash and dot dates (prefer DMY; these are typically European in CGM exports)
    for pat, fmt in [(r"^\d{1,2}-\d{1,2}-\d{4}$", "%d-%m-%Y"), (r"^\d{1,2}\.\d{1,2}\.\d{4}$", "%d.%m.%Y")]:
        mask = dt.isna() & raw.str.match(pat, na=False)
        if mask.any():
            dt.loc[mask] = pd.to_datetime(raw[mask], format=fmt, errors="coerce")

    return dt.dt.date


def _norm_code(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    # Common in CGM exports: numeric-like codes serialized as "1002.0"
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return out


def _dedupe_stats(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if not keys:
        return df
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        # Keep the row with the most available stats (deterministic tie-breaker).
        stat_cols = [c for c in STATS_COLS if c in df.columns]
        if stat_cols:
            completeness = df[stat_cols].notna().sum(axis=1)
            df = df.assign(_completeness=completeness)
            df = df.sort_values(keys + ["_completeness"], ascending=[True] * len(keys) + [False])
            df = df.drop_duplicates(subset=keys, keep="first").drop(columns=["_completeness"], errors="ignore")
        else:
            df = df.sort_values(keys).drop_duplicates(subset=keys, keep="first")
    return df


def backfill_match_stats(
    *,
    history_path: Path,
    stats_path: Path,
    out_path: Path,
    data_dir: Path,
) -> Path:
    hist = pd.read_csv(history_path)
    if not stats_path.exists():
        # Optional stats file missing: emit history with empty stat columns.
        print(f"[warn] stats file not found: {stats_path}. Writing history with empty stats.")
        hist = ensure_pressure_inputs(hist)
        hist["pressure_usable"] = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(out_path, index=False)
        print(f"[ok] wrote -> {out_path} (rows={len(hist)})")
        return out_path
    stats = _read_table(stats_path)

    # Prepare keys in history
    if "date" not in hist.columns:
        raise ValueError("history missing 'date'")
    hist["_date_only"] = _parse_date_only(hist["date"])
    for c in ["ft_home", "ft_away"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
    if "code_home" in hist.columns:
        hist["code_home"] = _norm_code(hist["code_home"])
    if "code_away" in hist.columns:
        hist["code_away"] = _norm_code(hist["code_away"])
    hist["home"] = hist.get("home", "").astype(str).str.strip()
    hist["away"] = hist.get("away", "").astype(str).str.strip()
    hist_cutoff = hist["_date_only"].max()

    # Prepare keys in stats
    if "datameci" not in stats.columns:
        raise ValueError("stats missing 'datameci' (expected goals statistics.csv / CGM bet database schema)")
    stats["_date_only"] = _parse_date_only(stats["datameci"], label="stats.datameci")
    if "codechipa1" in stats.columns:
        stats["code_home"] = _norm_code(stats["codechipa1"])
    else:
        stats["code_home"] = np.nan
    if "codechipa2" in stats.columns:
        stats["code_away"] = _norm_code(stats["codechipa2"])
    else:
        stats["code_away"] = np.nan

    # Normalize names for fallback join
    reg = build_team_registry(str(data_dir))
    stats["home"] = stats.get("txtechipa1", "").astype(str).apply(lambda x: normalize_team(x, reg)).str.strip()
    stats["away"] = stats.get("txtechipa2", "").astype(str).apply(lambda x: normalize_team(x, reg)).str.strip()

    # Parse combined "H-A" stats to split numeric columns
    stats = ensure_pressure_inputs(stats)
    for c in STATS_COLS:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors="coerce")

    # Leakage safety: ignore stats rows without a parsed date, ignore rows after the history cutoff,
    # and ignore rows that contain no usable pressure inputs.
    n_stats_total = len(stats)
    n_unparsable = int(stats["_date_only"].isna().sum())
    stats = stats.dropna(subset=["_date_only"], how="any")

    n_future = 0
    if hist_cutoff is not None and str(hist_cutoff) != "nan":
        future_mask = stats["_date_only"] > hist_cutoff
        n_future = int(future_mask.sum())
        stats = stats[~future_mask]

    stats_any = stats[STATS_COLS].notna().any(axis=1)
    n_no_stats = int((~stats_any).sum())
    stats = stats[stats_any]

    extra_cols: list[str] = []
    if "scor1" in stats.columns and "scor2" in stats.columns:
        stats["ft_home_stats"] = pd.to_numeric(stats["scor1"], errors="coerce")
        stats["ft_away_stats"] = pd.to_numeric(stats["scor2"], errors="coerce")
        extra_cols = ["ft_home_stats", "ft_away_stats"]

    stats_code = stats[["_date_only", "code_home", "code_away"] + extra_cols + STATS_COLS].copy()
    stats_name = stats[["_date_only", "home", "away"] + extra_cols + STATS_COLS].copy()
    stats_code = stats_code.dropna(subset=["_date_only", "code_home", "code_away"], how="any")
    stats_name = stats_name.dropna(subset=["_date_only", "home", "away"], how="any")

    stats_code = _dedupe_stats(stats_code, ["_date_only", "code_home", "code_away"])
    stats_name = _dedupe_stats(stats_name, ["_date_only", "home", "away"])

    # Merge (priority 1): date + codes
    merged = hist.merge(
        stats_code,
        how="left",
        on=["_date_only", "code_home", "code_away"],
        suffixes=("", "_stats"),
    )
    merged["_stats_src"] = np.where(merged[STATS_COLS[0]].notna(), "code", "none")

    # Merge (priority 2): date + names for rows still missing
    need = merged[STATS_COLS[0]].isna()
    if need.any():
        m2 = merged.loc[need, :].merge(
            stats_name,
            how="left",
            on=["_date_only", "home", "away"],
            suffixes=("", "_stats2"),
        )
        for c in STATS_COLS:
            merged.loc[need, c] = m2[c].to_numpy()
        merged.loc[need & merged[STATS_COLS[0]].notna(), "_stats_src"] = "name"

    # Guardrail: if stats provide FT scores, require they match the history scores (prevents bad joins).
    if {"ft_home", "ft_away", "ft_home_stats", "ft_away_stats"}.issubset(merged.columns):
        matchable = (
            merged[STATS_COLS[0]].notna()
            & merged["ft_home"].notna()
            & merged["ft_away"].notna()
            & merged["ft_home_stats"].notna()
            & merged["ft_away_stats"].notna()
        )
        mismatch = matchable & (
            (merged["ft_home"].astype(float) != merged["ft_home_stats"].astype(float))
            | (merged["ft_away"].astype(float) != merged["ft_away_stats"].astype(float))
        )
        if mismatch.any():
            merged.loc[mismatch, STATS_COLS] = np.nan
            merged.loc[mismatch, "_stats_src"] = "none_score_mismatch"

    # Simple gating flag: 1 only when all pressure inputs are present for the row.
    usable = pd.Series(True, index=merged.index, dtype=bool)
    for c in STATS_COLS:
        if c not in merged.columns:
            usable &= False
        else:
            usable &= merged[c].notna()
    merged["pressure_usable"] = usable.astype(int)

    # Drop intermediate join-only columns
    merged = merged.drop(columns=["ft_home_stats", "ft_away_stats"], errors="ignore")
    merged = merged.drop(columns=["_date_only"], errors="ignore")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    # Simple summary for operator
    n = len(merged)
    got = int(merged[STATS_COLS[0]].notna().sum()) if n else 0
    by_src = merged["_stats_src"].value_counts(dropna=False).to_dict() if n else {}
    cov = {c: float(merged[c].notna().mean()) for c in STATS_COLS} if n else {}
    print(f"[ok] wrote -> {out_path}")
    print(f"[join] rows={n} matched_rows={got} matched_rate={got/max(n,1):.3f} by_src={by_src}")
    print("[coverage]", {k: round(v, 4) for k, v in cov.items()})
    print(
        "[stats_src]",
        {
            "rows_total": int(n_stats_total),
            "unparsable_dates": int(n_unparsable),
            "future_rows_dropped": int(n_future),
            "no_stats_rows_dropped": int(n_no_stats),
        },
    )

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill CGM per-match stats into canonical history")
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo.csv", help="Input history CSV")
    ap.add_argument("--stats", default="CGM data/goals statistics.csv", help="Goals statistics CSV (per-match stats)")
    ap.add_argument("--out", default="data/enhanced/cgm_match_history_with_elo_stats.csv", help="Output history+stats CSV")
    ap.add_argument("--data-dir", default="CGM data", help="CGM data directory for team registry normalization")
    args = ap.parse_args()

    backfill_match_stats(
        history_path=Path(args.history),
        stats_path=Path(args.stats),
        out_path=Path(args.out),
        data_dir=Path(args.data_dir),
    )


if __name__ == "__main__":
    main()
