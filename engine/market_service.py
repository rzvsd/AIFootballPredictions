"""
Market service: convert fixture score matrices into market probabilities.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Any, Dict, List

import pandas as pd

import config

DEFAULT_OU_LINES = [0.5, 1.5, 2.5, 3.5]
DEFAULT_INTERVALS = [(0, 3), (1, 3), (2, 4), (2, 5), (3, 6)]


def _eval_1x2(P: np.ndarray) -> Dict[str, float]:
    p_h = float(np.tril(P, -1).sum())
    p_a = float(np.triu(P, 1).sum())
    p_d = float(np.trace(P))
    return {"H": p_h, "D": p_d, "A": p_a}


def _eval_double_chance(p1x2: Dict[str, float]) -> Dict[str, float]:
    return {"1X": p1x2["H"] + p1x2["D"], "12": p1x2["H"] + p1x2["A"], "X2": p1x2["D"] + p1x2["A"]}


def _goals_distribution(P: np.ndarray) -> np.ndarray:
    max_goals = P.shape[0] - 1
    dist = np.zeros(max_goals * 2 + 1)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            dist[i + j] += P[i, j]
    return dist


def _eval_over_under(P: np.ndarray, lines: list[float]) -> Dict[str, Dict[str, float]]:
    dist = _goals_distribution(P)
    csum = np.cumsum(dist)
    out: Dict[str, Dict[str, float]] = {}
    for line in lines:
        threshold = float(line)
        under_prob = float(dist[: int(np.floor(threshold + 1e-9)) + 1].sum())
        over_prob = float(1.0 - under_prob)
        out[str(line)] = {"Over": over_prob, "Under": under_prob}
    return out


def _eval_intervals(P: np.ndarray, intervals: list[tuple[int, int]]) -> Dict[str, float]:
    dist = _goals_distribution(P)
    out: Dict[str, float] = {}
    max_t = len(dist) - 1
    for a, b in intervals:
        a2 = max(0, int(a))
        b2 = min(max_t, int(b))
        if a2 > b2:
            prob = 0.0
        else:
            prob = float(dist[a2 : b2 + 1].sum())
        out[f"{a2}-{b2}"] = prob
    return out


def _evaluate_all_markets(P: np.ndarray, ou_lines: list[float], intervals: list[tuple[int, int]]) -> Dict[str, Any]:
    p1x2 = _eval_1x2(P)
    dc = _eval_double_chance(p1x2)
    ou = _eval_over_under(P, ou_lines or DEFAULT_OU_LINES)
    ints = _eval_intervals(P, intervals or DEFAULT_INTERVALS)
    return {"1X2": p1x2, "DC": dc, "OU": ou, "Intervals": ints}


def build_market_book(predictions: List[Dict[str, Any]], cfg: Dict[str, Any]) -> pd.DataFrame:
    league = cfg.get("league", "E0")
    use_calibration = bool(cfg.get("use_calibration", True))
    ou_lines_cfg = cfg.get("ou_lines", DEFAULT_OU_LINES)
    try:
        ou_lines = [float(x) for x in ou_lines_cfg]
    except Exception:
        ou_lines = DEFAULT_OU_LINES
    intervals_cfg = cfg.get("intervals", DEFAULT_INTERVALS)
    intervals: list[tuple[int, int]] = []
    for iv in intervals_cfg:
        try:
            if isinstance(iv, (list, tuple)) and len(iv) == 2:
                intervals.append((int(iv[0]), int(iv[1])))
        except Exception:
            continue
    if not intervals:
        intervals = DEFAULT_INTERVALS

    rows = []
    for pred in predictions:
        P = pred.get("P")
        if P is None:
            continue
        markets = _evaluate_all_markets(P, ou_lines=ou_lines, intervals=intervals)
        # Calibration hook (reuse existing calibrators if present)
        try:
            from calibrators import load_calibrators, apply_calibration_1x2

            cal1_path = os.path.join("calibrators", f"{league}_1x2.pkl")
            cal1 = load_calibrators(cal1_path)
        except Exception:
            cal1 = None
        if use_calibration and cal1:
            try:
                arr = np.array([[markets["1X2"]["H"], markets["1X2"]["D"], markets["1X2"]["A"]]])
                arr_cal = apply_calibration_1x2(arr, cal1)[0]
                markets["1X2"] = {"H": float(arr_cal[0]), "D": float(arr_cal[1]), "A": float(arr_cal[2])}
            except Exception:
                pass

        # Optional shrinkage
        try:
            shrink_1x2 = float(cfg.get("calibration_shrink", {}).get("oneX2", 0.0))
        except Exception:
            shrink_1x2 = 0.0
        if shrink_1x2 > 0:
            prior = 1.0 / 3.0
            markets["1X2"] = {
                "H": (1.0 - shrink_1x2) * markets["1X2"]["H"] + shrink_1x2 * prior,
                "D": (1.0 - shrink_1x2) * markets["1X2"]["D"] + shrink_1x2 * prior,
                "A": (1.0 - shrink_1x2) * markets["1X2"]["A"] + shrink_1x2 * prior,
            }

        # Optional OU calibration/shrinkage
        try:
            calou_path = os.path.join("calibrators", f"{league}_ou25.pkl")
            from calibrators import load_calibrators

            calou = load_calibrators(calou_path)
        except Exception:
            calou = None
        if use_calibration and calou and "ou25" in calou and "2.5" in markets["OU"]:
            try:
                model = calou["ou25"]
                from sklearn.isotonic import IsotonicRegression
                from sklearn.linear_model import LogisticRegression

                over_p = np.array([markets["OU"]["2.5"]["Over"]])
                if isinstance(model, IsotonicRegression):
                    cal_over = float(np.clip(model.predict(over_p), 1e-6, 1 - 1e-6))
                elif isinstance(model, LogisticRegression):
                    cal_over = float(model.predict_proba(over_p.reshape(-1, 1))[:, 1][0])
                else:
                    cal_over = float(over_p[0])
                markets["OU"]["2.5"] = {"Over": cal_over, "Under": float(max(0.0, 1.0 - cal_over))}
            except Exception:
                pass
        try:
            shrink_ou = float(cfg.get("calibration_shrink", {}).get("ou25", 0.0))
        except Exception:
            shrink_ou = 0.0
        if shrink_ou > 0 and "2.5" in markets["OU"]:
            pO = markets["OU"]["2.5"]["Over"]
            pO = (1.0 - shrink_ou) * pO + shrink_ou * 0.5
            markets["OU"]["2.5"] = {"Over": float(pO), "Under": float(1.0 - pO)}

        date_val = pred.get("date")
        home = pred.get("home")
        away = pred.get("away")
        for k, v in markets["1X2"].items():
            rows.append({"date": date_val, "home": home, "away": away, "market": "1X2", "outcome": k, "prob": v})
        for k, v in markets["DC"].items():
            rows.append({"date": date_val, "home": home, "away": away, "market": "DC", "outcome": k, "prob": v})
        for line, kv in markets["OU"].items():
            rows.append({"date": date_val, "home": home, "away": away, "market": f"OU {line}", "outcome": "Under", "prob": kv["Under"]})
            rows.append({"date": date_val, "home": home, "away": away, "market": f"OU {line}", "outcome": "Over", "prob": kv["Over"]})
        for iv, pv in markets["Intervals"].items():
            rows.append({"date": date_val, "home": home, "away": away, "market": "TG Interval", "outcome": iv, "prob": pv})

    return pd.DataFrame(rows)

