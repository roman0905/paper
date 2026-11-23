"""Selective prediction: when should the detector abstain?

Every method gets the standard MaxProb confidence |p - 0.5|.  Methods that expose internal
reliability signals can in addition fit a small logistic "selector" on the validation split that
predicts whether the detector's own decision is correct:

* CLARA: branch disagreement |P_MIL - sigmoid(s_x)|, mean gate, strongest instance, bag size
* ARG / FactGuard: their two rationale gate values

The selector never sees test labels; it is fitted on validation predictions of the same run.
"""
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clara.metrics import selective_metrics


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def features(pred: Dict[str, np.ndarray], split: str) -> Optional[np.ndarray]:
    """Internal-signal features of one split, or None if the method exposes none."""
    p = pred[split + "/prob"].astype(np.float64)
    cols = [np.abs(p - 0.5)]
    if split + "/p_bag" in pred and split + "/g_mean" in pred:        # CLARA
        p_bag, s_x = pred[split + "/p_bag"].astype(np.float64), pred[split + "/s_x"].astype(np.float64)
        off = pred[split + "/off"]
        cols += [np.abs(p_bag - _sigmoid(s_x)), pred[split + "/g_mean"], pred[split + "/p_max"],
                 np.abs(p_bag - 0.5), np.log(off[1:] - off[:-1])]
    elif split + "/gate" in pred:                                     # ARG / FactGuard
        g = pred[split + "/gate"].astype(np.float64)
        cols += [g[:, 0], g[:, 1]]
    else:
        return None
    return np.stack(cols, axis=1)


def evaluate(pred: Dict[str, np.ndarray], test: str = "test") -> Dict[str, Dict[str, float]]:
    """Selective-prediction metrics of one run for MaxProb and, if available, the fitted selector."""
    y, p = pred[test + "/label"], pred[test + "/prob"]
    out = {"maxprob": selective_metrics(y, p, np.abs(p - 0.5))}
    fv, ft = features(pred, "val"), features(pred, test)
    if fv is not None:
        yv, pv = pred["val/label"], pred["val/prob"]
        correct = ((pv >= 0.5).astype(int) == yv).astype(int)
        if 0 < correct.sum() < len(correct):
            sel = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            sel.fit(fv, correct)
            out["selector"] = selective_metrics(y, p, sel.predict_proba(ft)[:, 1])
    return out
