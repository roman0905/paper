"""Verification examples: which evidence a claim is shown, under which condition.

An example is ``{"id", "claim", "evidence", "label", "kind", "cands", "tags"}``: ``evidence`` is a
list of (source host, sentence) pairs or ``None`` for a closed-book query, ``cands`` the matching
(report, sentence) addresses and ``tags`` their silver evidence labels (training signal only).
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Ranked = List[Tuple[int, int, float]]
Cand = Tuple[int, int]


def layout(chosen: Sequence[Sequence]) -> List[Cand]:
    """Order candidates for reading: reports by their best-placed sentence, the sentences of a
    report in their original order.  ``chosen`` is given best first."""
    first: Dict[int, int] = {}
    for pos, c in enumerate(chosen):
        first.setdefault(c[0], pos)
    return [(c[0], c[1]) for c in sorted(chosen, key=lambda c: (first[c[0]], c[1]))]


def example(item: Dict[str, Any], cands: Optional[Sequence[Cand]], kind: str = "orig",
            source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A query for ``item`` with the candidate sentences ``cands`` of ``source`` (default: the
    item itself; another claim for mismatched evidence).  Mismatched evidence has no positive tag."""
    src = item if source is None else source
    ex = {"id": item["id"], "claim": item["claim"], "label": item["label"], "kind": kind,
          "evidence": None, "cands": [], "tags": []}
    if cands is not None:
        ex["cands"] = list(cands)
        ex["evidence"] = [(src["reports"][ri]["host"], src["reports"][ri]["sents"][si]) for ri, si in cands]
        own = source is None or source is item
        ex["tags"] = [int(src["reports"][ri]["ev"][si]) if own else 0 for ri, si in cands]
    return ex


def standard(items: Sequence[Dict[str, Any]], ranking: Dict[str, Ranked], m: int) -> List[Dict[str, Any]]:
    """Every claim with its own top-``m`` evidence."""
    return [example(it, layout(ranking.get(it["id"], [])[:m])) for it in items]


def withhold_claim(examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evidence-only control: the same queries with the claim text removed.  If a verifier does as
    well without the claim, the evidence gives the verdict away (Hansen et al., 2021)."""
    from openbook.prompts import NO_CLAIM

    out = []
    for ex in examples:
        ex = dict(ex)
        ex["claim"] = NO_CLAIM
        out.append(ex)
    return out


def closed(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Closed-book queries: the claim alone."""
    return [example(it, None, kind="closed") for it in items]


def derangement(n: int, seed: int) -> np.ndarray:
    """A random permutation without fixed points (n > 1)."""
    rng = np.random.RandomState(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p


def swapped(items: Sequence[Dict[str, Any]], ranking: Dict[str, Ranked], m: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Every claim with the top-``m`` evidence of another claim of the same split (labels
    unchanged): what does the verifier do when retrieval returns material that is not about the
    claim?"""
    perm = derangement(len(items), seed)
    return [example(it, layout(ranking.get(items[int(perm[i])]["id"], [])[:m]), kind="swap", source=items[int(perm[i])])
            for i, it in enumerate(items)]


# ------------------------------------------------------------------------- perturbed evidence
# A perturbation maps (items, ranking) to (items', ranking') with the same ids, so that whatever
# inference procedure a verifier uses can be run unchanged on the perturbed evidence pool.

def with_foreign(item: Dict[str, Any], other: Dict[str, Any], ranked_own: Ranked, ranked_other: Ranked,
                 n_own: int, n_other: int) -> Tuple[Dict[str, Any], Ranked]:
    """A copy of ``item`` whose candidate pool holds its own best ``n_own`` sentences and the best
    ``n_other`` sentences of another claim, interleaved by rank.  The foreign reports are appended
    to the report list (so their report index is >= ``n_reports_own``) and carry no silver tag."""
    off = len(item["reports"])
    virtual = dict(item)
    virtual["reports"] = list(item["reports"]) + [dict(r, ev=[0] * len(r["ev"])) for r in other["reports"]]
    virtual["n_reports_own"] = off
    own, foreign = list(ranked_own[:n_own]), [(ri + off, si, s) for ri, si, s in ranked_other[:n_other]]
    ranked: Ranked = []
    for i in range(max(len(own), len(foreign))):
        ranked += own[i:i + 1] + foreign[i:i + 1]
    return virtual, ranked


def nearest_claims(items: Sequence[Dict[str, Any]], universe: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For every item, the claim of ``universe`` whose text is most similar to it (BM25 over the
    claim texts; the item itself and verbatim duplicates are excluded)."""
    from openbook.retrieval import bm25_scores, terms

    docs = [terms(u["claim"]) for u in universe]
    out = []
    for it in items:
        scores = bm25_scores(terms(it["claim"]), docs)
        for j, u in enumerate(universe):
            if u["id"] == it["id"] or u["claim"] == it["claim"]:
                scores[j] = -1.0
        out.append(universe[int(np.argmax(scores))])
    return out


def perturb(items: Sequence[Dict[str, Any]], ranking: Dict[str, Ranked], kind: str, m: int, top: int,
            seed: int = 0, universe: Optional[Sequence[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Ranked]]:
    """Test-time evidence perturbations (labels never change):

    ``swap``      the whole pool comes from a random other claim: retrieval returned material that
                  is not about the claim
    ``nearswap``  the whole pool comes from the most similar other claim of ``universe`` (default:
                  ``items``): material about a related but different claim
    ``mix``       half of the pool is the claim's own, half comes from a random other claim
    ``droptop``   the first-stage top-``m`` sentences are missing: weak retrieval
    """
    pool = max(top, m)
    if kind == "droptop":
        return list(items), {it["id"]: ranking.get(it["id"], [])[m:] for it in items}
    if kind not in ("swap", "mix", "nearswap"):
        raise ValueError(kind)
    if kind == "nearswap":
        others = nearest_claims(items, universe if universe is not None else items)
    else:
        perm = derangement(len(items), seed)
        others = [items[int(perm[i])] for i in range(len(items))]
    out_items, out_rank = [], {}
    for i, it in enumerate(items):
        other = others[i]
        n_own = 0 if kind in ("swap", "nearswap") else pool // 2
        v, r = with_foreign(it, other, ranking.get(it["id"], []), ranking.get(other["id"], []), n_own, pool - n_own)
        out_items.append(v)
        out_rank[it["id"]] = r
    return out_items, out_rank


def is_foreign(item: Dict[str, Any], cand: Cand) -> bool:
    return cand[0] >= item.get("n_reports_own", len(item["reports"]))


def sample_picks(n_ranked: int, m: int, top: int, rng: np.random.RandomState, bias: float = 8.0) -> List[int]:
    """``m`` rank positions out of the best ``top``, without replacement, with probability
    proportional to 1/(rank + bias): mostly well-ranked sentences, but every rank is seen."""
    n = min(n_ranked, top)
    if n <= m:
        return list(range(n))
    w = 1.0 / (np.arange(n) + bias)
    return sorted(int(i) for i in rng.choice(n, size=m, replace=False, p=w / w.sum()))


def training(items: Sequence[Dict[str, Any]], ranking: Dict[str, Ranked], m: int, rng: np.random.RandomState,
             top: int = 0, swap_rate: float = 0.0) -> List[Dict[str, Any]]:
    """Training examples of one epoch.

    ``top > m``: the evidence of a claim is re-sampled from its best ``top`` candidates in every
    epoch (otherwise the fixed top-``m``).  ``swap_rate``: probability that a claim is shown the
    evidence of another claim; it keeps its own label, and none of the sentences is tagged."""
    perm = derangement(len(items), int(rng.randint(1 << 30))) if swap_rate > 0 else None
    out = []
    for i, it in enumerate(items):
        swap = perm is not None and rng.rand() < swap_rate
        src = items[int(perm[i])] if swap else it
        r = ranking.get(src["id"], [])
        picks = sample_picks(len(r), m, top, rng) if top > m else list(range(min(m, len(r))))
        out.append(example(it, layout([r[p] for p in picks]), kind="swap" if swap else "orig", source=src))
    return out


def chunked(items: Sequence[Dict[str, Any]], ranking: Dict[str, Ranked], m: int, top: int) -> List[Dict[str, Any]]:
    """First-pass queries: the best ``top`` candidates of every claim, dealt out by rank into
    ceil(top/m) queries of at most ``m`` sentences (so that every query mixes ranks, as the sampled
    training queries do).  A claim without candidates gets one query with empty evidence."""
    out = []
    for it in items:
        r = ranking.get(it["id"], [])[:top]
        k = max(1, math.ceil(len(r) / m))
        for j in range(k):
            ex = example(it, layout(r[j::k]))
            ex["chunk"] = j
            out.append(ex)
    return out


def oracle(items: Sequence[Dict[str, Any]], ranking: Dict[str, Ranked], m: int, top: int) -> List[Dict[str, Any]]:
    """Upper-bound analysis, NOT a legitimate test condition: the ``m`` sentences of the pool are
    chosen with the silver evidence labels of the test claims (silver first, each group in
    first-stage order)."""
    out = []
    for it in items:
        r = ranking.get(it["id"], [])[:top]
        best = sorted(r, key=lambda c: -it["reports"][c[0]]["ev"][c[1]])[:m]       # stable: keeps the rank order
        out.append(example(it, layout(best), kind="oracle"))
    return out


def selected(items: Sequence[Dict[str, Any]], scores: Dict[str, Dict[Cand, float]], m: int) -> List[Dict[str, Any]]:
    """Second-pass queries: every claim with its ``m`` best-scored candidates."""
    out = []
    for it in items:
        s = scores.get(it["id"], {})
        best = sorted(s, key=lambda c: -s[c])[:m]
        out.append(example(it, layout(best)))
    return out
