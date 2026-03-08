"""
League-specific weighting combiner (NLM/stacker style) with shrinkage-to-global.

Design goals:
- Keep base model as primary signal producer (mu_home / mu_away).
- Learn a global correction from module-group signals.
- Learn league-specific residual corrections and shrink them toward global when data is sparse.
- Provide deterministic, serializable behavior for train/inference parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


GROUP_NAMES: List[str] = ["elo", "xg", "pressure", "league_anchor", "team_form"]


def _category_for_feature(name: str) -> str | None:
    n = str(name).lower()
    if any(t in n for t in ["elodiff", "elo_diff", "elo_hfa_used", "elo_home", "elo_away", "elo_", "band_", "neff_sim", "vs_band"]):
        return "elo"
    if "xg" in n:
        return "xg"
    if any(t in n for t in ["press_", "div_", "sterile", "assassin"]):
        return "pressure"
    if n.startswith("lg_avg_") or n in {"attack_h", "attack_a", "defense_h", "defense_a", "expected_destruction_h", "expected_destruction_a"}:
        return "league_anchor"
    if any(t in n for t in ["_l5", "_l10", "shot_quality", "finish_rate", "gfvssim", "gavssim", "form_"]):
        return "team_form"
    return None


def infer_group_columns(feature_cols: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {g: [] for g in GROUP_NAMES}
    for c in feature_cols:
        g = _category_for_feature(c)
        if g is not None:
            out[g].append(c)
    return out


def _safe_to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


@dataclass
class LeagueNLMCombiner:
    group_cols: Dict[str, List[str]]
    prior_strength: float = 120.0
    min_league_rows: int = 80
    alpha_global: float = 1.0
    alpha_league: float = 2.0
    min_mu: float = 0.01

    # Learned parameters
    group_mean_: Dict[str, float] = field(default_factory=dict)
    group_std_: Dict[str, float] = field(default_factory=dict)
    global_intercept_: float = 0.0
    global_coef_: np.ndarray | None = None
    league_params_: Dict[str, Dict[str, object]] = field(default_factory=dict)
    feature_order_: List[str] = field(default_factory=lambda: ["base_mu"] + [f"group_{g}" for g in GROUP_NAMES])

    def _group_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = _safe_to_numeric_df(X)
        scores: Dict[str, pd.Series] = {}
        for g in GROUP_NAMES:
            cols = [c for c in self.group_cols.get(g, []) if c in Xn.columns]
            if not cols:
                scores[g] = pd.Series(0.0, index=Xn.index, dtype=float)
                continue
            scores[g] = Xn[cols].mean(axis=1, skipna=True).fillna(0.0)
        return pd.DataFrame(scores, index=Xn.index)

    def _zscore_fit(self, S: pd.DataFrame) -> pd.DataFrame:
        Z = S.copy()
        for g in GROUP_NAMES:
            mu = float(pd.to_numeric(S[g], errors="coerce").mean()) if g in S.columns else 0.0
            sd = float(pd.to_numeric(S[g], errors="coerce").std(ddof=0)) if g in S.columns else 1.0
            if not np.isfinite(sd) or sd <= 1e-9:
                sd = 1.0
            self.group_mean_[g] = mu
            self.group_std_[g] = sd
            Z[g] = (pd.to_numeric(S[g], errors="coerce").fillna(0.0) - mu) / sd
        return Z

    def _zscore_apply(self, S: pd.DataFrame) -> pd.DataFrame:
        Z = S.copy()
        for g in GROUP_NAMES:
            mu = float(self.group_mean_.get(g, 0.0))
            sd = float(self.group_std_.get(g, 1.0))
            if not np.isfinite(sd) or sd <= 1e-9:
                sd = 1.0
            Z[g] = (pd.to_numeric(S[g], errors="coerce").fillna(0.0) - mu) / sd
        return Z

    def _make_design(self, base_mu: np.ndarray, Z_groups: pd.DataFrame) -> np.ndarray:
        b = np.asarray(base_mu, dtype=float).reshape(-1, 1)
        G = np.column_stack([pd.to_numeric(Z_groups[g], errors="coerce").fillna(0.0).to_numpy(dtype=float) for g in GROUP_NAMES])
        return np.column_stack([b, G])

    def fit(
        self,
        *,
        X: pd.DataFrame,
        y: np.ndarray,
        base_mu: np.ndarray,
        leagues: pd.Series,
    ) -> "LeagueNLMCombiner":
        yv = np.asarray(y, dtype=float)
        bm = np.asarray(base_mu, dtype=float)
        lg = leagues.astype(str).fillna("GLOBAL")

        S = self._group_scores(X)
        Zg = self._zscore_fit(S)
        Z = self._make_design(bm, Zg)

        gmod = Ridge(alpha=float(self.alpha_global), fit_intercept=True, random_state=42)
        gmod.fit(Z, yv)
        self.global_intercept_ = float(gmod.intercept_)
        self.global_coef_ = np.asarray(gmod.coef_, dtype=float)

        pred_g = self.global_intercept_ + Z @ self.global_coef_
        resid = yv - pred_g

        self.league_params_ = {}
        df_meta = pd.DataFrame({"league": lg, "resid": resid})
        for league_name, idx in df_meta.groupby("league").groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            n = int(len(idx_arr))
            if n < int(self.min_league_rows):
                continue
            Z_l = Z[idx_arr]
            r_l = resid[idx_arr]
            lmod = Ridge(alpha=float(self.alpha_league), fit_intercept=True, random_state=42)
            lmod.fit(Z_l, r_l)
            shrink = float(n / (n + float(self.prior_strength)))
            self.league_params_[str(league_name)] = {
                "n": n,
                "shrink": shrink,
                "intercept": float(lmod.intercept_),
                "coef": np.asarray(lmod.coef_, dtype=float),
            }
        return self

    def predict_batch(
        self,
        *,
        X: pd.DataFrame,
        base_mu: np.ndarray,
        leagues: pd.Series,
    ) -> np.ndarray:
        if self.global_coef_ is None:
            raise RuntimeError("LeagueNLMCombiner is not fitted")

        bm = np.asarray(base_mu, dtype=float)
        lg = leagues.astype(str).fillna("GLOBAL")
        S = self._group_scores(X)
        Zg = self._zscore_apply(S)
        Z = self._make_design(bm, Zg)

        pred = self.global_intercept_ + Z @ self.global_coef_
        pred = np.asarray(pred, dtype=float)

        # League residual adjustment with shrinkage.
        for league_name, params in self.league_params_.items():
            mask = lg == league_name
            if not bool(mask.any()):
                continue
            rows = np.where(mask.to_numpy())[0]
            coef = np.asarray(params["coef"], dtype=float)
            intercept = float(params["intercept"])
            shrink = float(params["shrink"])
            adj = intercept + Z[rows] @ coef
            pred[rows] += shrink * adj

        pred = np.clip(pred, float(self.min_mu), None)
        return pred

    def predict_one(self, *, X_row: pd.DataFrame, base_mu: float, league: str | None) -> float:
        leagues = pd.Series([str(league or "GLOBAL")])
        pred = self.predict_batch(X=X_row, base_mu=np.asarray([base_mu], dtype=float), leagues=leagues)
        return float(pred[0])

    def summary(self) -> Dict[str, object]:
        return {
            "groups": {k: len(v) for k, v in self.group_cols.items()},
            "league_models": len(self.league_params_),
            "prior_strength": float(self.prior_strength),
            "min_league_rows": int(self.min_league_rows),
            "alpha_global": float(self.alpha_global),
            "alpha_league": float(self.alpha_league),
        }

