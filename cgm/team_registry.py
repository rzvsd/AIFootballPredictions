"""
Team registry builder for local football data files.

Builds a normalized mapping from team IDs/codes to canonical names and a reverse
alias map for robust lookups.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import re
import unicodedata


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


def _simplify_name(name: str, *, loose: bool = False) -> str:
    """
    Normalize team names for fuzzy lookup.
    - Unicode fold to ASCII (e.g., München -> Munchen)
    - Lowercase, strip punctuation
    - Collapse whitespace
    - Optionally drop common leading tokens (loose mode)
    """
    if not name:
        return ""
    # Fold accents -> ASCII
    folded = unicodedata.normalize("NFKD", name)
    folded = folded.encode("ascii", "ignore").decode("ascii")
    # Lowercase and replace non-alnum with space
    lowered = folded.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not loose:
        return lowered

    tokens = lowered.split()
    if not tokens:
        return ""
    # Drop leading numeric tokens and common prefixes like "fc", "sc", etc.
    drop_prefixes = {"fc", "sc", "sv", "vfl", "vfb", "tsg", "fcv", "1", "1fc"}
    while tokens and (tokens[0].isdigit() or tokens[0] in drop_prefixes):
        tokens = tokens[1:]
    return " ".join(tokens).strip()


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


def build_team_registry(data_dir: str | Path = "data/api_football") -> Dict[str, Dict[str, str]]:
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
        base / "history_fixtures.csv",
        base / "upcoming_fixtures.csv",
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
        pairs.extend(_collect_pairs(df, "home_id", "home_name"))
        pairs.extend(_collect_pairs(df, "away_id", "away_name"))
        pairs.extend(_collect_pairs(df, "code_home", "home"))
        pairs.extend(_collect_pairs(df, "code_away", "away"))
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
    name_to_code_norm: dict[str, str] = {}
    name_to_code_loose: dict[str, str] = {}
    for code, name in code_to_name.items():
        name_norm = name.strip()
        name_to_code[name_norm] = code
        strict_key = _simplify_name(name_norm, loose=False)
        loose_key = _simplify_name(name_norm, loose=True)
        if strict_key and strict_key not in name_to_code_norm:
            name_to_code_norm[strict_key] = code
        if loose_key and loose_key not in name_to_code_loose:
            name_to_code_loose[loose_key] = code
        # also add aliases
        for alias in aliases.get(code, []):
            alias_norm = alias.strip()
            if alias_norm and alias_norm not in name_to_code:
                name_to_code[alias_norm] = code
            strict_alias = _simplify_name(alias_norm, loose=False)
            loose_alias = _simplify_name(alias_norm, loose=True)
            if strict_alias and strict_alias not in name_to_code_norm:
                name_to_code_norm[strict_alias] = code
            if loose_alias and loose_alias not in name_to_code_loose:
                name_to_code_loose[loose_alias] = code

    return {
        "code_to_name": code_to_name,
        "name_to_code": name_to_code,
        "name_to_code_norm": name_to_code_norm,
        "name_to_code_loose": name_to_code_loose,
        "aliases": {k: sorted(v) for k, v in aliases.items()},
    }


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
    # Fuzzy lookup (accent-folding + punctuation removal)
    strict_key = _simplify_name(val, loose=False)
    if strict_key and strict_key in registry.get("name_to_code_norm", {}):
        code = registry["name_to_code_norm"][strict_key]
        return registry["code_to_name"].get(code, val)
    # Loose lookup (drop common leading tokens like "FC", "1")
    loose_key = _simplify_name(val, loose=True)
    if loose_key and loose_key in registry.get("name_to_code_loose", {}):
        code = registry["name_to_code_loose"][loose_key]
        return registry["code_to_name"].get(code, val)
    return val


def main() -> None:
    ap = argparse.ArgumentParser(description="Build team registry from local football CSVs")
    ap.add_argument("--data-dir", default="data/api_football", help="Directory containing source CSV files")
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
