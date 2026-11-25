"""Evidence candidates and first-stage (lexical) ranking.

A candidate is one sentence of one report of the claim, addressed as ``(report index, sentence
index)`` into the normalised item, so rankings stay valid whatever protocol filtered the pool.
"""
import functools
import json
import math
import os
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from nltk.stem import PorterStemmer

from openbook.data import reports_under

Cand = Tuple[int, int]

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an the of to in on at by for with from as and or but if than then so such that this these those it its
is are was were be been being am do does did have has had not no nor can could would should may might will shall must
he she they we you i his her their our your my me him them us who whom which what when where why how all any both each
more most other some own same too very s t just also into over under again further once here there about against
between through during before after above below up down out off""".split())
_stem = functools.lru_cache(maxsize=500000)(PorterStemmer().stem)


def terms(text: str) -> List[str]:
    """Lower-cased, stop-word free, stemmed word tokens.  The reports are released lemmatised while
    the claims are not, so both sides are stemmed before they are matched."""
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _key(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))


def pool(item: Dict[str, Any], protocol: str, min_words: int = 5) -> List[Cand]:
    """Distinct candidate sentences of a claim under a protocol (first occurrence of a duplicate
    is kept; fragments shorter than ``min_words`` words are dropped)."""
    allowed = {id(r) for r in reports_under(item, protocol)}
    seen, out = set(), []
    for ri, rep in enumerate(item["reports"]):
        if id(rep) not in allowed:
            continue
        for si, s in enumerate(rep["sents"]):
            k = _key(s)
            if k in seen or len(k.split()) < min_words:
                continue
            seen.add(k)
            out.append((ri, si))
    return out


def sentence(item: Dict[str, Any], c: Cand) -> str:
    return item["reports"][c[0]]["sents"][c[1]]


def is_silver(item: Dict[str, Any], c: Cand) -> int:
    return int(item["reports"][c[0]]["ev"][c[1]])


def bm25_scores(query: Sequence[str], docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 of every document for the query; document frequencies are those of ``docs``."""
    n = len(docs)
    if n == 0:
        return np.zeros(0)
    df: Dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    avg = sum(len(d) for d in docs) / n or 1.0
    q = [w for w in dict.fromkeys(query) if w in df]
    idf = {w: math.log(1.0 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in q}
    out = np.zeros(n)
    for i, d in enumerate(docs):
        if not d:
            continue
        tf: Dict[str, int] = {}
        for w in d:
            if w in idf:
                tf[w] = tf.get(w, 0) + 1
        norm = k1 * (1.0 - b + b * len(d) / avg)
        out[i] = sum(idf[w] * f * (k1 + 1.0) / (f + norm) for w, f in tf.items())
    return out


def bm25_ranking(item: Dict[str, Any], protocol: str) -> List[Tuple[int, int, float]]:
    """All candidates of a claim, best first, scored by BM25 against the claim."""
    cands = pool(item, protocol)
    scores = bm25_scores(terms(item["claim"]), [terms(sentence(item, c)) for c in cands])
    order = np.argsort(-scores, kind="stable")
    return [(cands[i][0], cands[i][1], float(scores[i])) for i in order]


# ------------------------------------------------------------------------------------- rankings
def ranking_path(root: str, dataset: str, protocol: str, ranker: str) -> str:
    return os.path.join(root, "outputs", "ob", "rank", "{}.{}.{}.json".format(dataset, protocol, ranker))


def save_ranking(path: str, ranking: Dict[str, List[Tuple[int, int, float]]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: [[a, b, round(s, 4)] for a, b, s in v] for k, v in ranking.items()}, f)


def load_ranking(path: str) -> Dict[str, List[Tuple[int, int, float]]]:
    with open(path, encoding="utf-8") as f:
        return {k: [(int(a), int(b), float(s)) for a, b, s in v] for k, v in json.load(f).items()}


# ---------------------------------------------------------------------------- ranking evaluation
def ranking_quality(items: Sequence[Dict[str, Any]], ranking: Dict[str, List[Tuple[int, int, float]]],
                    ks: Sequence[int] = (8, 16, 32)) -> Dict[str, float]:
    """How well a ranking recovers the silver evidence sentences (claims without any silver
    evidence in the ranked pool are skipped).  ``lift@k`` is precision@k over the base rate."""
    acc: Dict[str, List[float]] = {}
    for it in items:
        r = ranking.get(it["id"], [])
        rel = np.array([is_silver(it, (a, b)) for a, b, _ in r], dtype=float)
        if rel.sum() == 0:
            continue
        base = rel.mean()
        for k in ks:
            top = rel[:k]
            acc.setdefault("p@%d" % k, []).append(float(top.mean()))
            acc.setdefault("r@%d" % k, []).append(float(top.sum() / rel.sum()))
            acc.setdefault("lift@%d" % k, []).append(float(top.mean() / base))
        ranks = np.arange(1, len(rel) + 1)
        acc.setdefault("map", []).append(float(((np.cumsum(rel) / ranks) * rel).sum() / rel.sum()))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["n"] = float(len(acc.get("map", [])))
    return out
