"""Inference procedures of the verifier and the quantities derived from them."""
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from openbook import views as V
from openbook.verifier import Verifier, class_probs

Cand = Tuple[int, int]


def single_pass(v: Verifier, examples: Sequence[Dict[str, Any]], n_classes: int, batch: int = 16) -> Dict[str, Any]:
    """Class probabilities of every query and the evidence scores of its sentences."""
    enc = v.encode(examples)
    prob, scores = v.predict(enc, batch)
    return {"prob": class_probs(prob, n_classes), "scores": scores, "n_ev": [e["n_ev"] for e in enc],
            "tokens": float(np.mean([len(e["ids"]) for e in enc])) if enc else 0.0}


def read_select_verify(v: Verifier, items: Sequence[Dict[str, Any]], ranking: Dict[str, V.Ranked], m: int, top: int,
                       n_classes: int, batch: int = 16) -> Dict[str, Any]:
    """Two-pass inference.  Pass 1 reads the best ``top`` first-stage candidates of a claim in
    queries of ``m`` sentences and scores every sentence with the evidence head; pass 2 verifies
    the claim against its ``m`` best-scored sentences.

    Returns the class probabilities of pass 2 (``prob``), the mean distribution of the pass-1
    queries (``prob_read``), the sentence scores and the selected candidates of every claim."""
    first = V.chunked(items, ranking, m, top)
    r1 = single_pass(v, first, n_classes, batch)
    scores: Dict[str, Dict[Cand, float]] = {}
    read: Dict[str, List[np.ndarray]] = {}
    for ex, p, s, n in zip(first, r1["prob"], r1["scores"], r1["n_ev"]):
        d = scores.setdefault(ex["id"], {})
        for c, x in zip(ex["cands"][:n], s):
            d[c] = float(x)
        read.setdefault(ex["id"], []).append(p)
    second = V.selected(items, scores, m)
    r2 = single_pass(v, second, n_classes, batch)
    return {"prob": r2["prob"], "prob_read": np.stack([np.mean(read[it["id"]], axis=0) for it in items]),
            "scores": scores, "selected": [ex["cands"] for ex in second],
            "tokens": r2["tokens"] + r1["tokens"] * len(first) / max(len(items), 1), "queries": 1.0 + len(first) / max(len(items), 1)}


def silver_precision(items: Sequence[Dict[str, Any]], chosen: Sequence[Sequence[Cand]]) -> float:
    """Share of silver evidence among the chosen sentences (claims without any choice skipped)."""
    vals = [np.mean([it["reports"][ri]["ev"][si] for ri, si in c]) for it, c in zip(items, chosen) if c]
    return float(np.mean(vals)) if vals else float("nan")


def score_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """AUC of a score separating two groups (probability that a positive outranks a negative)."""
    from sklearn.metrics import roc_auc_score

    if not len(pos) or not len(neg):
        return float("nan")
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg)))
