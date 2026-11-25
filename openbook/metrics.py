"""Multi-class metrics for veracity prediction, and the paired significance test."""
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def classification_metrics(y_true: Sequence[int], prob: np.ndarray) -> Dict[str, float]:
    """Macro precision / recall / F1 (the metrics of CofCED and its successors: unweighted means of
    the per-class values), accuracy and, since the labels are ordinal, the mean absolute error in
    classes.  ``prob`` is [n, C]; the prediction is its argmax."""
    y = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    n_classes = prob.shape[1]
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(n_classes)), zero_division=0)
    mp, mr = float(p.mean()), float(r.mean())
    out = {"macf1": float(f.mean()), "macp": mp, "macr": mr,
           # several papers on these datasets report F1 = 2PR/(P+R) of the macro-averaged P and R
           "f1_h": 2.0 * mp * mr / (mp + mr) if mp + mr > 0 else 0.0,
           "acc": float((pred == y).mean()), "mae": float(np.abs(pred - y).mean())}
    for c in range(n_classes):
        out["f1_%d" % c] = float(f[c])
    conf = prob.max(1)
    out["conf"] = float(conf.mean())
    out["ece"] = ece(y, prob)
    return out


def ece(y_true: Sequence[int], prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the top-class confidence (equal-width bins)."""
    y = np.asarray(y_true).astype(int)
    conf, pred = prob.max(1), prob.argmax(1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def macro_f1_rows(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Macro-F1 of every row of ``pred`` against the matching row of ``y`` (both [R, n] ints)."""
    out = np.zeros(y.shape[0])
    for c in range(n_classes):
        yc, pc = (y == c), (pred == c)
        tp = (yc & pc).sum(1).astype(np.float64)
        denom = yc.sum(1) + pc.sum(1)
        out += np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
    return out / n_classes


def paired_bootstrap(y_true: Sequence[int], preds_a: Sequence[np.ndarray], preds_b: Sequence[np.ndarray],
                     n_classes: int, n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap over test items of the difference in seed-averaged macro-F1 (A minus B).

    ``preds_*`` hold one array of predicted classes per seed.  Returns the observed difference, a
    95% interval and a two-sided p-value (twice the smaller tail mass of the bootstrap
    distribution around zero)."""
    y = np.asarray(y_true).astype(int)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(y), size=(n_boot, len(y)))
    yb = y[idx]

    def mean_f1(preds, rows):
        return np.mean([macro_f1_rows(rows, np.asarray(p)[idx] if rows is yb else np.asarray(p)[None], n_classes)
                        for p in preds], axis=0)

    obs = float(mean_f1(preds_a, y[None])[0] - mean_f1(preds_b, y[None])[0])
    diff = mean_f1(preds_a, yb) - mean_f1(preds_b, yb)
    p = 2.0 * min(float((diff <= 0).mean()), float((diff >= 0).mean()))
    return {"diff": obs, "ci_lo": float(np.percentile(diff, 2.5)), "ci_hi": float(np.percentile(diff, 97.5)),
            "p": min(1.0, p)}
