from __future__ import annotations

import os
import joblib
import numpy as np
from typing import Dict, Callable
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(probs, labels)
    return ir


def fit_platt(probs: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(max_iter=1000)
    X = probs.reshape(-1, 1)
    lr.fit(X, labels)
    return lr


def multi_calibrate_1x2(p_1x2: np.ndarray, y: np.ndarray, method: str = 'isotonic') -> Dict[str, object]:
    # p_1x2 shape (n,3); y in {'H','D','A'}
    lbl_map = {'H': 0, 'D': 1, 'A': 2}
    y_idx = np.vectorize(lbl_map.get)(y)
    out: Dict[str, object] = {}
    for name, idx in [('H',0), ('D',1), ('A',2)]:
        binary = (y_idx == idx).astype(int)
        pr = p_1x2[:, idx]
        model = fit_isotonic(pr, binary) if method=='isotonic' else fit_platt(pr, binary)
        out[name] = model
    return out


def apply_calibration_1x2(p_1x2: np.ndarray, cal: Dict[str, object]) -> np.ndarray:
    def apply_model(model, p):
        if isinstance(model, IsotonicRegression):
            return np.clip(model.predict(p), 1e-6, 1-1e-6)
        elif isinstance(model, LogisticRegression):
            return model.predict_proba(p.reshape(-1,1))[:,1]
        else:
            return p
    pH = apply_model(cal['H'], p_1x2[:,0])
    pD = apply_model(cal['D'], p_1x2[:,1])
    pA = apply_model(cal['A'], p_1x2[:,2])
    s = (pH + pD + pA)
    s = np.where(s<=0, 1.0, s)
    return np.vstack([pH/s, pD/s, pA/s]).T


def calibrate_binary(p: np.ndarray, y: np.ndarray, method: str = 'isotonic') -> object:
    return fit_isotonic(p, y) if method=='isotonic' else fit_platt(p, y)


def save_calibrators(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_calibrators(path: str) -> Dict | None:
    if not os.path.exists(path):
        return None
    return joblib.load(path)
