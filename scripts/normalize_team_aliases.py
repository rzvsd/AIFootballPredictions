"""
Audit and maintain manual team alias overrides across all leagues.

Usage examples:
  python -m scripts.normalize_team_aliases --data-dir data/api_football
  python -m scripts.normalize_team_aliases --data-dir data/api_football --apply --min-score 0.92
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
import sys

import pandas as pd

try:
    from cgm.team_registry import build_team_registry, normalize_team  # type: ignore
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from cgm.team_registry import build_team_registry, normalize_team  # type: ignore


SOURCE_FILES = [
    "history_fixtures.csv",
    "upcoming_fixtures.csv",
    "goal statistics 2.csv",
    "goals statistics.csv",
    "leageue statistics.csv",
    "multiple seasons.csv",
]

TEAM_COLS = [
    "txtechipa1",
    "txtechipa2",
    "home_name",
    "away_name",
    "home",
    "away",
    "echipa",
    "echipah",
    "echipaa",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _iter_team_names(df: pd.DataFrame, cols: Iterable[str]) -> Iterable[str]:
    for col in cols:
        if col not in df.columns:
            continue
        for v in df[col].dropna().astype(str):
            s = v.strip()
            if s and s.lower() != "nan":
                yield s


def _best_match(name: str, canonical_names: list[str]) -> tuple[str, float]:
    best_name = ""
    best_score = 0.0
    for cand in canonical_names:
        score = SequenceMatcher(None, name.lower(), cand.lower()).ratio()
        if score > best_score:
            best_score = score
            best_name = cand
    return best_name, float(best_score)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit/apply team alias normalization overrides")
    ap.add_argument("--data-dir", default="data/api_football")
    ap.add_argument("--overrides", default=None, help="Path to team_alias_overrides.json")
    ap.add_argument("--report-out", default=None, help="Path to write JSON report")
    ap.add_argument("--apply", action="store_true", help="Apply high-confidence suggestions into overrides")
    ap.add_argument("--min-score", type=float, default=0.92, help="Minimum similarity score for auto-apply")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    overrides_path = Path(args.overrides) if args.overrides else (data_dir / "team_alias_overrides.json")
    report_path = Path(args.report_out) if args.report_out else (data_dir / "team_alias_report.json")

    reg = build_team_registry(data_dir)
    canonical_names = sorted(set(reg.get("code_to_name", {}).values()))

    all_names: set[str] = set()
    for rel in SOURCE_FILES:
        p = data_dir / rel
        if not p.exists():
            continue
        try:
            df = _read_csv(p)
        except Exception:
            continue
        all_names.update(_iter_team_names(df, TEAM_COLS))

    unresolved = []
    for name in sorted(all_names):
        normalized = normalize_team(name, reg)
        known_name = name in reg.get("name_to_code", {})
        if known_name:
            continue
        if normalized != name and normalized in reg.get("name_to_code", {}):
            continue
        best_name, best_score = _best_match(name, canonical_names)
        unresolved.append(
            {
                "alias": name,
                "suggested_canonical": best_name,
                "score": round(best_score, 4),
            }
        )

    auto_applied = []
    if args.apply:
        overrides = {}
        if overrides_path.exists():
            try:
                overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
            except Exception:
                overrides = {}
        if not isinstance(overrides, dict):
            overrides = {}
        canonical_by_alias = overrides.get("canonical_by_alias", {})
        if not isinstance(canonical_by_alias, dict):
            canonical_by_alias = {}
        for row in unresolved:
            if row["score"] >= args.min_score and row["suggested_canonical"]:
                alias = row["alias"]
                canonical = row["suggested_canonical"]
                if alias not in canonical_by_alias:
                    canonical_by_alias[alias] = canonical
                    auto_applied.append({"alias": alias, "canonical": canonical, "score": row["score"]})
        overrides["canonical_by_alias"] = dict(sorted(canonical_by_alias.items()))
        overrides_path.parent.mkdir(parents=True, exist_ok=True)
        overrides_path.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "data_dir": str(data_dir),
        "total_unique_names_scanned": len(all_names),
        "canonical_team_count": len(canonical_names),
        "unresolved_count": len(unresolved),
        "auto_applied_count": len(auto_applied),
        "auto_apply_min_score": args.min_score,
        "unresolved": unresolved,
        "auto_applied": auto_applied,
        "overrides_path": str(overrides_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Scanned names: {len(all_names)}")
    print(f"Unresolved names: {len(unresolved)}")
    print(f"Auto-applied aliases: {len(auto_applied)}")
    print(f"Report: {report_path}")
    print(f"Overrides: {overrides_path}")


if __name__ == "__main__":
    main()
