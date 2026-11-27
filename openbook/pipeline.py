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
            "cands": [ex["cands"][:e["n_ev"]] for ex, e in zip(examples, enc)],
            "tokens": float(np.mean([len(e["ids"]) for e in enc])) if enc else 0.0}


def read_ensemble(v: Verifier, items: Sequence[Dict[str, Any]], ranking: Dict[str, V.Ranked], m: int, top: int,
                  n_classes: int, batch: int = 16) -> np.ndarray:
    """Mean class distribution over the queries that together read the best ``top`` candidates
    (``m`` per query): what reading more evidence buys without any selection."""
    first = V.chunked(items, ranking, m, top)
    r1 = single_pass(v, first, n_classes, batch)
    read: Dict[str, List[np.ndarray]] = {}
    for ex, p in zip(first, r1["prob"]):
        read.setdefault(ex["id"], []).append(p)
    return np.stack([np.mean(read[it["id"]], axis=0) for it in items])


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


def score_maps(items: Sequence[Dict[str, Any]], result: Dict[str, Any]) -> List[Dict[Cand, float]]:
    """Per claim, the evidence score of every sentence the verifier read (from either procedure)."""
    if isinstance(result["scores"], dict):
        return [result["scores"].get(it["id"], {}) for it in items]
    return [{c: float(x) for c, x in zip(cands, s)} for cands, s in zip(result["cands"], result["scores"])]


def usability_scores(items: Sequence[Dict[str, Any]], result: Dict[str, Any], m: int) -> np.ndarray:
    """Claim-level evidence usability a(c): the mean of the ``m`` highest evidence scores among the
    sentences the verifier read for the claim (0 if it read none)."""
    return np.array([float(np.mean(sorted(s.values(), reverse=True)[:m])) if s else 0.0 for s in score_maps(items, result)])


def rouge_f(references: Sequence[str], hypotheses: Sequence[str]) -> Dict[str, float]:
    """Mean ROUGE-1 / ROUGE-2 / ROUGE-L F1 of generated texts against their references (stemmed)."""
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = [sc.score(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not rows:
        return {}
    return {k: float(np.mean([x[k].fmeasure for x in rows])) for k in ("rouge1", "rouge2", "rougeL")}


def _mean_auc(groups: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> float:
    """Mean over claims of the AUC of positives against negatives (claims lacking either skipped)."""
    vals = [score_auc(p, n) for p, n in groups if len(p) and len(n)]
    return float(np.mean(vals)) if vals else float("nan")


def evidence_diagnostics(items: Sequence[Dict[str, Any]], ranking: Dict[str, V.Ranked], m: int, own: Dict[str, Any],
                         perturbed: Dict[str, Tuple[Sequence[Dict[str, Any]], Dict[str, Any]]], select: bool) -> Dict[str, float]:
    """What the evidence head knows.

    ``auc_silver``            does it rank the silver evidence of a claim above its other sentences?
    ``silver_p_*``            share of silver evidence in the pool read, in the first-stage top-m and
                              (two-pass runs) among the selected sentences
    ``auc_own_vs_foreign``    half-swapped pools: are the claim's own sentences scored above the
                              sentences that belong to another claim?  (mean per-claim AUC)
    ``halfswap_foreign_sel``  two-pass runs: share of foreign sentences among those selected
    ``auc_usable``            can a claim-level summary of the scores (mean of the m best) tell a
                              claim with its own pool from a claim with a fully mismatched pool?
    """
    out: Dict[str, float] = {}
    maps = score_maps(items, own)
    silver = lambda it, c: it["reports"][c[0]]["ev"][c[1]]   # noqa: E731
    out["auc_silver"] = _mean_auc([([x for c, x in s.items() if silver(it, c)], [x for c, x in s.items() if not silver(it, c)])
                                   for it, s in zip(items, maps)])
    out["silver_p_pool"] = silver_precision(items, [list(s) for s in maps])
    out["silver_p_bm25"] = silver_precision(items, [[(a, b) for a, b, _ in ranking.get(it["id"], [])[:m]] for it in items])
    if select:
        out["silver_p_selected"] = silver_precision(items, own["selected"])
    items_h, res_h = perturbed["halfswap"]
    maps_h = score_maps(items_h, res_h)
    out["auc_own_vs_foreign"] = _mean_auc([([x for c, x in s.items() if not V.is_foreign(it, c)],
                                            [x for c, x in s.items() if V.is_foreign(it, c)]) for it, s in zip(items_h, maps_h)])
    if select:
        out["halfswap_foreign_sel"] = float(np.mean([np.mean([V.is_foreign(it, tuple(c)) for c in sel])
                                                     for it, sel in zip(items_h, res_h["selected"]) if sel]))
    own_top = usability_scores(items, own, m).tolist()
    out["usable_own_mean"] = float(np.mean(own_top))
    for kind, key in (("swap", "usable"), ("nearswap", "usable_near")):
        if kind in perturbed:
            other_top = usability_scores(perturbed[kind][0], perturbed[kind][1], m).tolist()
            out[key + "_swap_mean" if kind == "swap" else key + "_mean"] = float(np.mean(other_top))
            out["auc_" + key] = score_auc(own_top, other_top)
    return out


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
