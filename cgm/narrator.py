"""
Milestone 8: Narrator Layer (human-readable explanations for picks).

Consumes:
  reports/picks.csv (required)
  reports/picks_debug.csv (optional, reserved for future "why_not" explanations)

Produces:
  reports/picks_explained.csv (same rows as picks.csv, plus narrator columns)
  reports/picks_explained_preview.txt (optional quick preview)
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "fixture_datetime",
    "league",
    "home",
    "away",
    "market",
    "odds",
    "stake_tier",
    "stake_units",
    "p_model",
    "p_implied",
    "mu_home",
    "mu_away",
    "mu_total",
    "neff_min",
    "press_n_min",
    "xg_n_min",
    "sterile_flag",
    "assassin_flag",
]

NARRATOR_COLS = [
    "pick_text",
    "title",
    "narrative",
    "confidence_label",
    "numbers_plain",
]

MARKET_NAME_MAP = {
    "OU25_OVER": "Over 2.5 goals",
    "OU25_UNDER": "Under 2.5 goals",
    "1X2_HOME": "Home win",
    "1X2_DRAW": "Draw",
    "1X2_AWAY": "Away win",
    "BTTS_YES": "Both teams to score: Yes",
    "BTTS_NO": "Both teams to score: No",
    "1H_OU05_OVER": "1st half Over 0.5",
    "1H_OU05_UNDER": "1st half Under 0.5",
    "2H_OU05_OVER": "2nd half Over 0.5",
    "2H_OU05_UNDER": "2nd half Under 0.5",
    "2H_OU15_OVER": "2nd half Over 1.5",
    "2H_OU15_UNDER": "2nd half Under 1.5",
    "GOAL_AFTER_75_YES": "Goal after 75': Yes",
    "GOAL_AFTER_75_NO": "Goal after 75': No",
}


def _require_cols(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[narrator] missing required columns in {context}: {missing}")


def _safe_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _edge_strength(ev_val: float, p_model: float, p_implied: float) -> str:
    base = ev_val
    if not np.isfinite(base):
        diff = p_model - p_implied
        base = diff if np.isfinite(diff) else float("nan")
    if not np.isfinite(base):
        return "an almost fair price"
    if base < 0.02:
        return "an almost fair price"
    if base < 0.04:
        return "a small edge"
    if base < 0.08:
        return "a solid edge"
    return "a strong edge"


def _pct(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{int(round(v * 100))}%"


def _label_neff(v: float) -> str:
    if not np.isfinite(v):
        return "unknown comparable history"
    if v < 4:
        return "thin comparable history"
    if v < 7:
        return "some comparable history"
    if v < 10:
        return "solid comparable history"
    return "strong comparable history"


def _label_press(v: float) -> str:
    if not np.isfinite(v):
        return "unknown recent performance evidence"
    if v < 2:
        return "very limited recent performance evidence"
    if v < 4:
        return "some recent performance evidence"
    return "good recent performance evidence"


def _label_xg(v: float) -> str:
    if not np.isfinite(v):
        return "unknown chance-quality evidence"
    if v < 2:
        return "very limited chance-quality evidence"
    if v < 4:
        return "some chance-quality evidence"
    return "good chance-quality evidence"


def _confidence_label(stake_tier: str) -> str:
    tier = str(stake_tier).upper()
    if tier == "T1":
        return "Low"
    if tier == "T2":
        return "Medium"
    return "High"


def _market_label(market: str) -> str:
    return MARKET_NAME_MAP.get(market, market)


def _title(home: str, away: str, market: str) -> str:
    return f"{home} vs {away} — {_market_label(market)}"


def _pick_text(market: str, odds: float, stake_units: float, stake_tier: str) -> str:
    return f"Pick: {_market_label(market)} @ {odds:.2f} | Stake: {stake_units:g}u ({stake_tier})"


def _evidence_sentence(neff: float, press: float, xg: float) -> str:
    neff_s = _label_neff(neff)
    press_s = _label_press(press)
    xg_s = _label_xg(xg)
    return f"Confidence is supported by {neff_s}, plus {press_s} and {xg_s}."


def _risk_sentence(sterile: int, assassin: int) -> str:
    if sterile:
        return (
            "Style warning: one side can look dominant on the ball without creating enough real chances, "
            "so we avoid goal-chasing logic and keep sizing disciplined."
        )
    if assassin:
        return (
            "Style warning: counter-attacking profile — games like this can swing on a few big moments, "
            "so we demand a stronger edge and size carefully."
        )
    return "No major style warnings stand out."


def _edge_sentence(odds: float, p_model: float, p_implied: float, edge_strength: str) -> str:
    return (
        f"The odds imply roughly {_pct(p_implied)}, while our estimate is about {_pct(p_model)}, "
        f"which is {edge_strength} versus the market."
    )


def _numbers_plain(row: pd.Series) -> str:
    model_pct = _pct(row.get("p_model"))
    imp_pct = _pct(row.get("p_implied"))
    mu_total = _safe_float(row.get("mu_total"))
    evid = f"{_label_neff(row.get('neff_min'))} / {_label_press(row.get('press_n_min'))} / {_label_xg(row.get('xg_n_min'))}"
    flags = "none"
    if int(_safe_float(row.get("sterile_flag"))) or int(_safe_float(row.get("assassin_flag"))):
        parts = []
        if int(_safe_float(row.get("sterile_flag"))):
            parts.append("sterile")
        if int(_safe_float(row.get("assassin_flag"))):
            parts.append("assassin")
        flags = ", ".join(parts)
    return f"Model {model_pct} vs odds-implied {imp_pct} | Expected goals: {mu_total:.1f} | Evidence: {evid} | Flags: {flags}"


def _build_narrative(row: pd.Series) -> str:
    market = str(row.get("market", ""))
    odds = _safe_float(row.get("odds"))
    stake_units = _safe_float(row.get("stake_units"))
    stake_tier = str(row.get("stake_tier"))
    mu_home = _safe_float(row.get("mu_home"))
    mu_away = _safe_float(row.get("mu_away"))
    mu_total = _safe_float(row.get("mu_total"))
    p_model = _safe_float(row.get("p_model"))
    p_implied = _safe_float(row.get("p_implied"))
    ev_val = _safe_float(row.get("ev"))
    neff = _safe_float(row.get("neff_min"))
    press_n = _safe_float(row.get("press_n_min"))
    xg_n = _safe_float(row.get("xg_n_min"))
    sterile = int(_safe_float(row.get("sterile_flag")))
    assassin = int(_safe_float(row.get("assassin_flag")))

    edge_strength = _edge_strength(ev_val, p_model, p_implied)
    evidence_s = _evidence_sentence(neff, press_n, xg_n)
    risk_s = _risk_sentence(sterile, assassin)
    stake_s = f"That's why we size it as {stake_units:g}u ({stake_tier})."

    if market.startswith("OU25"):
        first = f"We're taking {_market_label(market)} here at {odds:.2f}."
        second = f"The model expects around {mu_total:.1f} total goals in this match."
        third = _edge_sentence(odds, p_model, p_implied, edge_strength)
        return " ".join([first, second, third, evidence_s, risk_s, stake_s])

    if market.startswith("1X2"):
        first = f"We're taking {_market_label(market)} at {odds:.2f}."
        second = (
            f"Our model gives this outcome about {_pct(p_model)}, with the odds implying around {_pct(p_implied)}."
        )
        third = f"That gap is {edge_strength}, which is enough for a play at this price."
        fourth = f"Goal expectation is {mu_home:.1f}–{mu_away:.1f} (about {mu_total:.1f} total), which fits the game script behind this pick."
        return " ".join([first, second, third, fourth, evidence_s, risk_s, stake_s])

    if market.startswith("BTTS"):
        first = f"We're taking {_market_label(market)} at {odds:.2f}."
        second = (
            f"Our model puts this around {_pct(p_model)}, versus an implied {_pct(p_implied)}, which is {edge_strength}."
        )
        third = f"The overall goal expectation is about {mu_total:.1f}, which suits this angle."
        return " ".join([first, second, third, evidence_s, risk_s, stake_s])

    # Timing/half markets: reuse totals phrasing
    first = f"We're taking {_market_label(market)} at {odds:.2f}."
    second = f"The full-match model expects around {mu_total:.1f} total goals."
    third = _edge_sentence(odds, p_model, p_implied, edge_strength)
    return " ".join([first, second, third, evidence_s, risk_s, stake_s])


def build_explained(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _require_cols(df, REQUIRED_COLS, context="picks.csv")

    explained_rows = []
    for _, r in df.iterrows():
        market = str(r["market"])
        odds = _safe_float(r.get("odds"))
        stake_units = _safe_float(r.get("stake_units"))
        stake_tier = str(r.get("stake_tier"))
        row_dict = r.to_dict()
        row_dict["pick_text"] = _pick_text(market, odds, stake_units, stake_tier)
        row_dict["title"] = _title(str(r.get("home")), str(r.get("away")), market)
        row_dict["narrative"] = _build_narrative(r)
        row_dict["confidence_label"] = _confidence_label(stake_tier)
        row_dict["numbers_plain"] = _numbers_plain(r)
        explained_rows.append(row_dict)

    explained = pd.DataFrame(explained_rows)
    if explained.empty:
        # Preserve columns even if empty
        for c in df.columns:
            if c not in explained.columns:
                explained[c] = pd.Series(dtype=df[c].dtype)
        for c in NARRATOR_COLS:
            if c not in explained.columns:
                explained[c] = pd.Series(dtype=object)
    return explained[df.columns.tolist() + [c for c in NARRATOR_COLS if c not in df.columns]]


def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_preview(df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    if df.empty:
        lines.append("No picks to explain.")
    else:
        for _, r in df.iterrows():
            lines.append(r.get("title", ""))
            lines.append(r.get("pick_text", ""))
            lines.append(r.get("narrative", ""))
            lines.append("")  # spacer
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Milestone 8: Narrator layer for picks")
    ap.add_argument("--in", dest="in_path", default="reports/picks.csv", help="Input picks CSV")
    ap.add_argument("--out", dest="out_path", default="reports/picks_explained.csv", help="Output explained CSV")
    ap.add_argument(
        "--preview-out",
        dest="preview_out",
        default="reports/picks_explained_preview.txt",
        help="Optional human-readable preview",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    preview_out = Path(args.preview_out) if args.preview_out else None

    if not in_path.exists():
        raise SystemExit(f"[narrator] input not found: {in_path}")

    picks = pd.read_csv(in_path)
    explained = build_explained(picks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    explained.to_csv(out_path, index=False)

    if preview_out is not None:
        _write_preview(explained, preview_out)

    print(f"[narrator] wrote {len(explained)} rows -> {out_path} (hash={_hash_file(out_path)})")


if __name__ == "__main__":
    main()

