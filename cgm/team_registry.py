"""
Team registry builder for CGM data files.

Reads the CSV files in "CGM data" and builds a normalized mapping from
team codes to canonical names (and a reverse alias map for name lookups).

Usage (library):
    from cgm.team_registry import build_team_registry, normalize_team
    registry = build_team_registry("CGM data")
    name = normalize_team("1020", registry)  # code to name
    code = registry["name_to_code"].get("Arsenal")

Usage (CLI):
    python -m cgm.team_registry --data-dir "CGM data" --out team_map.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    """Best-effort CSV reader with delimiter inference."""
    return pd.read_csv(path, sep=None, engine="python")


def _norm_code(code) -> str | None:
    try:
        if code is None or pd.isna(code):
            return None
        s = str(code).strip()
        if not s or s.lower() == "nan":
            return None
        if s.endswith(".0"):
            s = s[:-2]
        return s or None
    except Exception:
        return None


def _norm_name(name) -> str | None:
    try:
        if name is None or pd.isna(name):
            return None
        s = str(name).strip()
        if not s or s.lower() == "nan":
            return None
        return s
    except Exception:
        return None


def _collect_pairs(df: pd.DataFrame, code_col: str, name_col: str) -> Iterable[Tuple[str, str]]:
    """Yield aligned (code, name) pairs from a specific (code_col, name_col)."""
    if code_col not in df.columns or name_col not in df.columns:
        return []
    out: list[Tuple[str, str]] = []
    for code, name in zip(df[code_col], df[name_col]):
        code_s = _norm_code(code)
        name_s = _norm_name(name)
        if code_s and name_s:
            out.append((code_s, name_s))
    return out


def build_team_registry(data_dir: str | Path = "CGM data") -> Dict[str, Dict[str, str]]:
    """
    Build a registry:
      - code_to_name: code -> canonical name
      - name_to_code: normalized name -> primary code
      - aliases: code -> set of all seen aliases
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"{base} not found")

    files = [
        base / "goal statistics 2.csv",
        base / "goals statistics.csv",
        base / "leageue statistics.csv",
        base / "multiple seasons.csv",
        base / "upcoming - Copy.CSV",
    ]
    multi = base / "multiple leagues and seasons"
    if multi.exists():
        files.extend(
            [
                multi / "allratingv.csv",
                multi / "upcoming.csv",
            ]
        )

    # Accumulate name occurrences per code
    names_for_code: dict[str, Counter] = defaultdict(Counter)

    for path in files:
        if not path.exists():
            continue
        try:
            df = _read_csv(path)
        except Exception as e:
            _logger.warning(f"Error reading {path}: {e}")
            continue
        pairs: list[Tuple[str, str]] = []
        # Canonical: match exports with explicit team code + team name columns.
        pairs.extend(_collect_pairs(df, "codechipa1", "txtechipa1"))
        pairs.extend(_collect_pairs(df, "codechipa2", "txtechipa2"))
        # Season aggregates / tables (team code + team name)
        pairs.extend(_collect_pairs(df, "codechipa", "echipa"))
        pairs.extend(_collect_pairs(df, "codechipah", "echipah"))
        pairs.extend(_collect_pairs(df, "codechipaa", "echipaa"))

        for code, name in pairs:
            names_for_code[code][name] += 1

    code_to_name: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    for code, counter in names_for_code.items():
        if not counter:
            continue
        # Canonical: most common name; break ties by shortest length then lexicographic
        most_common = counter.most_common()
        top_freq = most_common[0][1]
        candidates = [n for n, freq in most_common if freq == top_freq]
        best = sorted(candidates, key=lambda x: (len(x), x))[0]
        code_to_name[code] = best
        aliases[code] = set(counter.keys())

    # Build name -> primary code map (use canonical names)
    name_to_code: dict[str, str] = {}
    for code, name in code_to_name.items():
        name_norm = name.strip()
        name_to_code[name_norm] = code
        # also add aliases
        for alias in aliases.get(code, []):
            alias_norm = alias.strip()
            if alias_norm and alias_norm not in name_to_code:
                name_to_code[alias_norm] = code

    return {"code_to_name": code_to_name, "name_to_code": name_to_code, "aliases": {k: sorted(v) for k, v in aliases.items()}}


import logging
_logger = logging.getLogger(__name__)


def normalize_team(value, registry: Dict[str, Dict[str, str]]) -> str:
    """
    Normalize a team code or name to the canonical name using the registry.
    If the value cannot be resolved, return the stripped original.
    
    Type-safe: handles None, non-string inputs gracefully with logging.
    """
    # Type safety: handle None and non-string inputs
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception as e:
            _logger.warning(f"normalize_team: cannot convert {type(value)} to str: {e}")
            return ""
    
    val = value.strip()
    if not val or val.lower() == "nan":
        return ""
    if val in registry.get("code_to_name", {}):
        return registry["code_to_name"][val]
    if val in registry.get("name_to_code", {}):
        code = registry["name_to_code"][val]
        return registry["code_to_name"].get(code, val)
    return val


def main() -> None:
    ap = argparse.ArgumentParser(description="Build team registry from CGM data CSVs")
    ap.add_argument("--data-dir", default="CGM data", help="Directory containing CGM CSV files")
    ap.add_argument("--out", default=None, help="Optional path to write JSON mapping")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    reg = build_team_registry(args.data_dir)
    _logger.info(f"codes: {len(reg['code_to_name'])}, aliases: {sum(len(v) for v in reg['aliases'].values())}")
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        _logger.info(f"Wrote registry -> {out_path}")


if __name__ == "__main__":
    main()
