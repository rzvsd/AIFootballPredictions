"""
Scan OU/BTTS decision thresholds on backtest CSVs.

Example:
  python scripts/scan_thresholds.py ^
    --input "Ligue 1=reports/backtest_ligue1_tuning_window.csv" ^
    --input "La Liga=reports/backtest_laliga_tuning_window.csv" ^
    --input "Liga I=reports/backtest_ligai_tuning_window.csv" ^
    --out reports/tuning_batch1_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_map(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Expected KEY=VALUE, got: {raw}")
        key, value = raw.split("=", 1)
        k = key.strip()
        v = value.strip()
        if not k or not v:
            raise ValueError(f"Invalid mapping: {raw}")
        out[k] = v
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = float((tp + tn) / len(y_true)) if len(y_true) else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bacc = 0.5 * (tpr + tnr)
    yes_rate = float(y_pred.mean()) if len(y_pred) else 0.0
    return acc, bacc, yes_rate


def _best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    t_step: float,
    min_yes_rate: float | None,
    max_yes_rate: float | None,
) -> tuple[float, float, float, float]:
    best_t = t_min
    best_acc = -1.0
    best_bacc = -1.0
    best_rate = 0.0

    t = t_min
    while t <= t_max + 1e-12:
        pred = (probs >= t).astype(int)
        acc, bacc, yes_rate = _metrics(labels, pred)
        if min_yes_rate is not None and yes_rate < min_yes_rate:
            t += t_step
            continue
        if max_yes_rate is not None and yes_rate > max_yes_rate:
            t += t_step
            continue
        if bacc > best_bacc:
            best_t = float(round(t, 2))
            best_acc = acc
            best_bacc = bacc
            best_rate = yes_rate
        t += t_step

    return best_t, best_acc, best_bacc, best_rate


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan per-league OU/BTTS thresholds from backtest CSVs")
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help='Input mapping "League=path/to/backtest.csv". Use multiple times.',
    )
    ap.add_argument(
        "--ou-current",
        action="append",
        default=[],
        help='Current OU threshold override mapping "League=0.46".',
    )
    ap.add_argument(
        "--btts-current",
        action="append",
        default=[],
        help='Current BTTS threshold override mapping "League=0.42".',
    )
    ap.add_argument("--default-ou", type=float, default=0.50)
    ap.add_argument("--default-btts", type=float, default=0.50)
    ap.add_argument("--t-min", type=float, default=0.10)
    ap.add_argument("--t-max", type=float, default=0.90)
    ap.add_argument("--t-step", type=float, default=0.01)
    ap.add_argument(
        "--min-yes-rate",
        type=float,
        default=None,
        help="Optional minimum predicted YES rate (0..1) to avoid extreme thresholds.",
    )
    ap.add_argument(
        "--max-yes-rate",
        type=float,
        default=None,
        help="Optional maximum predicted YES rate (0..1) to avoid extreme thresholds.",
    )
    ap.add_argument("--out", default="reports/tuning_threshold_scan.csv")
    args = ap.parse_args()

    input_map = _parse_map(args.input)
    ou_map = {k: float(v) for k, v in _parse_map(args.ou_current).items()} if args.ou_current else {}
    btts_map = {k: float(v) for k, v in _parse_map(args.btts_current).items()} if args.btts_current else {}

    rows: list[dict[str, float | str | int]] = []

    for league, path_raw in input_map.items():
        path = Path(path_raw)
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if not {"ft_home", "ft_away", "p_over25", "p_btts_yes"}.issubset(df.columns):
            continue
        df = df.dropna(subset=["ft_home", "ft_away"]).copy()
        if df.empty:
            continue

        y_ou = ((df["ft_home"] + df["ft_away"]) > 2).astype(int).to_numpy()
        y_btts = ((df["ft_home"] > 0) & (df["ft_away"] > 0)).astype(int).to_numpy()
        p_ou = pd.to_numeric(df["p_over25"], errors="coerce").fillna(0.5).to_numpy()
        p_btts = pd.to_numeric(df["p_btts_yes"], errors="coerce").fillna(0.5).to_numpy()

        cur_ou_t = float(ou_map.get(league, args.default_ou))
        cur_btts_t = float(btts_map.get(league, args.default_btts))

        cur_ou_acc, cur_ou_bacc, cur_ou_rate = _metrics(y_ou, (p_ou >= cur_ou_t).astype(int))
        cur_btts_acc, cur_btts_bacc, cur_btts_rate = _metrics(y_btts, (p_btts >= cur_btts_t).astype(int))

        best_ou_t, best_ou_acc, best_ou_bacc, best_ou_rate = _best_threshold(
            p_ou,
            y_ou,
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            t_step=float(args.t_step),
            min_yes_rate=args.min_yes_rate,
            max_yes_rate=args.max_yes_rate,
        )
        best_btts_t, best_btts_acc, best_btts_bacc, best_btts_rate = _best_threshold(
            p_btts,
            y_btts,
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            t_step=float(args.t_step),
            min_yes_rate=args.min_yes_rate,
            max_yes_rate=args.max_yes_rate,
        )

        rows.append(
            {
                "league": league,
                "matches": int(len(df)),
                "current_ou_thr": cur_ou_t,
                "current_ou_acc": cur_ou_acc,
                "current_ou_bacc": cur_ou_bacc,
                "current_ou_yes_rate": cur_ou_rate,
                "best_ou_thr_bacc": best_ou_t,
                "best_ou_acc": best_ou_acc,
                "best_ou_bacc": best_ou_bacc,
                "best_ou_yes_rate": best_ou_rate,
                "ou_bacc_gain": best_ou_bacc - cur_ou_bacc,
                "current_btts_thr": cur_btts_t,
                "current_btts_acc": cur_btts_acc,
                "current_btts_bacc": cur_btts_bacc,
                "current_btts_yes_rate": cur_btts_rate,
                "best_btts_thr_bacc": best_btts_t,
                "best_btts_acc": best_btts_acc,
                "best_btts_bacc": best_btts_bacc,
                "best_btts_yes_rate": best_btts_rate,
                "btts_bacc_gain": best_btts_bacc - cur_btts_bacc,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    if not out_df.empty:
        print(out_df.to_string(index=False))
    print(f"[ok] wrote -> {out_path}")


if __name__ == "__main__":
    main()
