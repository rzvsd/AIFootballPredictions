"""
CGM upcoming fixtures -> odds JSON + fixtures frame.

Reads "CGM data/upcoming - Copy.CSV" and emits a data/odds-like JSON that
the existing odds_service can read, plus an optional fixtures CSV.

Market mapping:
  - 1X2 odds from columns: cotaa (home), cotae (draw), cotad (away)
  - OU odds from columns: cotao (over 2.5), cotau (under 2.5)
  - Additional OU lines (if present): cotao0/cotau0, cotao1/cotau1, cotao3/cotau3, cotao4/cotau4
Team normalization uses cgm.team_registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from cgm.team_registry import build_team_registry, normalize_team


def _ou_bucket(row: pd.Series, over_col: str, under_col: str, line: str) -> Dict[str, float]:
    over = row.get(over_col)
    under = row.get(under_col)
    bucket = {}
    try:
        if pd.notna(over):
            bucket["Over"] = float(over)
    except Exception:
        pass
    try:
        if pd.notna(under):
            bucket["Under"] = float(under)
    except Exception:
        pass
    return {line: {"main": bucket}} if bucket else {}


def convert_upcoming_to_odds(data_dir: str = "CGM data", out_dir: str = "data/odds", league: str = "E0") -> Path:
    base = Path(data_dir)
    reg = build_team_registry(base)
    
    all_fixtures = []
    seen_keys = set()
    
    # Primary source: upcoming - Copy.CSV
    src = base / "upcoming - Copy.CSV"
    if src.exists():
        df = pd.read_csv(src, sep=None, engine="python")
        required_cols = ["datameci", "txtechipa1", "txtechipa2", "cotaa", "cotae", "cotad"]
        missing = [c for c in required_cols if c not in df.columns]
        if not missing:
            for _, r in df.iterrows():
                fixture = _extract_fixture(r, reg)
                if fixture:
                    key = (fixture["date"], fixture["home"], fixture["away"])
                    if key not in seen_keys:
                        all_fixtures.append(fixture)
                        seen_keys.add(key)
    
    # Fallback source: multiple seasons.csv (for unplayed fixtures with score 0-0)
    ms_src = base / "multiple seasons.csv"
    if ms_src.exists():
        try:
            ms_df = pd.read_csv(ms_src, sep=None, engine="python", encoding="latin1")
            # Filter to unplayed matches (score columns = 0 or NaN)
            if "scor1" in ms_df.columns and "scor2" in ms_df.columns:
                ms_df["scor1"] = pd.to_numeric(ms_df["scor1"], errors="coerce").fillna(0)
                ms_df["scor2"] = pd.to_numeric(ms_df["scor2"], errors="coerce").fillna(0)
                future = ms_df[(ms_df["scor1"] == 0) & (ms_df["scor2"] == 0)]
            else:
                future = ms_df
            
            required_cols = ["datameci", "txtechipa1", "txtechipa2", "cotaa", "cotae", "cotad"]
            if all(c in future.columns for c in required_cols):
                for _, r in future.iterrows():
                    fixture = _extract_fixture(r, reg)
                    if fixture:
                        key = (fixture["date"], fixture["home"], fixture["away"])
                        if key not in seen_keys:
                            all_fixtures.append(fixture)
                            seen_keys.add(key)
                            print(f"[fallback] Added fixture from multiple seasons: {fixture['home']} vs {fixture['away']} ({fixture['date']})")
        except Exception as e:
            print(f"[warn] Could not load fallback from multiple seasons.csv: {e}")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_path / f"{league}.json"
    out_data = {"fixtures": all_fixtures}
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    return out_path


def _extract_fixture(r: pd.Series, reg: dict) -> dict | None:
    """Extract fixture dict from a row with odds columns."""
    date_val = r.get("datameci")
    home_raw = r.get("txtechipa1")
    away_raw = r.get("txtechipa2")
    if pd.isna(home_raw) or pd.isna(away_raw):
        return None
    home = normalize_team(home_raw, reg)
    away = normalize_team(away_raw, reg)

    markets: Dict[str, Any] = {}
    # 1X2
    try:
        markets["1X2"] = {
            "main": {
                "home": float(r.get("cotaa")),
                "draw": float(r.get("cotae")),
                "away": float(r.get("cotad")),
            }
        }
    except Exception:
        pass
    # OU 2.5 primary
    ou = {}
    ou.update(_ou_bucket(r, "cotao", "cotau", "2.5"))
    # Optional other lines
    ou.update(_ou_bucket(r, "cotao0", "cotau0", "0.5"))
    ou.update(_ou_bucket(r, "cotao1", "cotau1", "1.5"))
    ou.update(_ou_bucket(r, "cotao3", "cotau3", "3.5"))
    ou.update(_ou_bucket(r, "cotao4", "cotau4", "4.5"))
    if ou:
        markets["OU"] = ou

    return {
        "date": date_val,
        "home": home,
        "away": away,
        "markets": markets,
    }


def convert_upcoming_to_fixtures_csv(data_dir: str = "CGM data", out_path: str = "data/fixtures/cgm_upcoming.csv") -> Path:
    base = Path(data_dir)
    src = base / "upcoming - Copy.CSV"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")
    df = pd.read_csv(src, sep=None, engine="python")
    reg = build_team_registry(base)
    out_rows = []
    for _, r in df.iterrows():
        date_val = r.get("datameci")
        home_raw = r.get("txtechipa1")
        away_raw = r.get("txtechipa2")
        if pd.isna(home_raw) or pd.isna(away_raw):
            continue
        home = normalize_team(home_raw, reg)
        away = normalize_team(away_raw, reg)
        out_rows.append({"date": date_val, "home": home, "away": away})
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_p, index=False)
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CGM upcoming CSV to odds JSON and fixtures CSV")
    ap.add_argument("--data-dir", default="CGM data", help="Directory containing CGM CSVs")
    ap.add_argument("--out-odds-dir", default="data/odds", help="Output directory for odds JSON")
    ap.add_argument("--league", default="E0", help="League code to use for odds file name")
    ap.add_argument("--fixtures-out", default=None, help="Optional fixtures CSV output path")
    args = ap.parse_args()

    odds_path = convert_upcoming_to_odds(args.data_dir, args.out_odds_dir, args.league)
    print(f"Wrote odds JSON -> {odds_path}")
    if args.fixtures_out:
        fix_path = convert_upcoming_to_fixtures_csv(args.data_dir, args.fixtures_out)
        print(f"Wrote fixtures CSV -> {fix_path}")


if __name__ == "__main__":
    main()
