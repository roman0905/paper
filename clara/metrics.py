"""Evaluation metrics: detection, calibration, selective prediction and significance tests."""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)


def detection_metrics(y_true: Sequence[int], prob: Sequence[float], thr: float = 0.5) -> Dict[str, float]:
    """Same definitions as the official ARG evaluation (label 1 = fake, macro-F1 is the model
    selection metric); ``prob`` is P(fake)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=np.float64)
    pred = (p >= thr).astype(int)
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")
    out["macf1"] = float(f1_score(y, pred, average="macro"))
    f1 = f1_score(y, pred, average=None, labels=[0, 1], zero_division=0)
    out["f1_real"], out["f1_fake"] = float(f1[0]), float(f1[1])
    out["acc"] = float(accuracy_score(y, pred))
    out["recall"] = float(recall_score(y, pred, average="macro", zero_division=0))
    out["precision"] = float(precision_score(y, pred, average="macro", zero_division=0))
    return out


def ece(y_true: Sequence[int], prob: Sequence[float], n_bins: int = 15) -> float:
    """Expected calibration error of the predicted class confidence (equal-width bins)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=np.float64)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    total, n = 0.0, len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def risk_coverage(correct: Sequence[float], confidence: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Risk (error rate) as a function of coverage when abstaining on the least confident items."""
    c = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-np.asarray(confidence, dtype=np.float64), kind="stable")
    c = c[order]
    n = len(c)
    cov = np.arange(1, n + 1) / n
    risk = 1.0 - np.cumsum(c) / np.arange(1, n + 1)
    return cov, risk


def aurc(correct: Sequence[float], confidence: Sequence[float]) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(correct, confidence)
    return float(risk.mean())


def eaurc(correct: Sequence[float], confidence: Sequence[float]) -> float:
    """Excess AURC over the optimal confidence ranking (Geifman et al., 2019)."""
    c = np.asarray(correct, dtype=np.float64)
    r = 1.0 - c.mean()
    opt = r + (1.0 - r) * np.log(1.0 - r) if 0.0 < r < 1.0 else 0.0
    return float(aurc(correct, confidence) - opt)


def selective_metrics(y_true: Sequence[int], prob: Sequence[float], confidence: Sequence[float],
                      coverages: Sequence[float] = (0.9, 0.8, 0.7)) -> Dict[str, float]:
    """AURC / E-AURC plus accuracy and macro-F1 on the retained items at fixed coverages."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(np.float64)
    out = {"aurc": aurc(correct, conf), "eaurc": eaurc(correct, conf)}
    order = np.argsort(-conf, kind="stable")
    for cv in coverages:
        k = max(1, int(round(cv * len(y))))
        keep = order[:k]
        tag = "{:d}".format(int(round(cv * 100)))
        out["acc@" + tag] = float(correct[keep].mean())
        out["macf1@" + tag] = float(f1_score(y[keep], pred[keep], average="macro", zero_division=0))
    return out


def paired_bootstrap(y_true: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float],
                     metric: str = "macf1", n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap test for ``metric(a) - metric(b)`` on the same test items.

    Returns the observed difference, a 95% CI and the one-sided p-value for H0: a is not better.
    """
    y = np.asarray(y_true).astype(int)
    pa, pb = np.asarray(prob_a, dtype=np.float64), np.asarray(prob_b, dtype=np.float64)
    rng = np.random.RandomState(seed)
    n = len(y)

    def m(yy, pp):
        pred = (pp >= 0.5).astype(int)
        if metric == "acc":
            return (pred == yy).mean()
        return f1_score(yy, pred, average="macro", zero_division=0)

    obs = m(y, pa) - m(y, pb)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs[i] = m(y[idx], pa[idx]) - m(y[idx], pb[idx])
    return {"diff": float(obs), "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)), "p": float((diffs <= 0).mean())}


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def best_threshold(y_true: Sequence[int], prob: Sequence[float]) -> float:
    """Threshold maximising macro-F1 (only ever fitted on validation data)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=np.float64)
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        v = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t


def localisation_metrics(scores: Sequence[Sequence[float]], gold: Sequence[int]) -> Dict[str, float]:
    """Hit@1 / MRR of the gold (tampered) unit among the per-item unit scores."""
    hits, rr, n = 0.0, 0.0, 0
    for s, g in zip(scores, gold):
        s = np.asarray(s, dtype=np.float64)
        if len(s) == 0 or g is None or g < 0 or g >= len(s):
            continue
        rank = 1 + int((s > s[g]).sum())
        hits += float(rank == 1)
        rr += 1.0 / rank
        n += 1
    return {"hit@1": hits / max(n, 1), "mrr": rr / max(n, 1), "n": float(n)}
