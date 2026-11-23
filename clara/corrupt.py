"""Test-time corruption of LLM outputs (robustness to unfaithful analyses).

Two granularities:

``replace_llm_outputs``  (item level)  with probability ``rate`` *all* LLM-generated fields of an
    item are replaced by those of another, randomly chosen item of the same split.  Simulates an
    analysis that is about something else entirely.

``replace_units``  (unit level)  every unit of the LLM's textual analysis is replaced with
    probability ``rate`` by a unit of another item: a claim (with its arguments and score) for
    the claim-structured analysis, a sentence for the free-text rationales of ARG / FactGuard.
    Simulates partially hallucinated analyses.  The LLM's holistic verdict and its zero-shot
    score, which do not consist of such units, are left untouched.

The news text and the label never change, so a detector that knows which parts of the analysis
to ignore should degrade gracefully.
"""
import copy
import random
from typing import Any, Dict, List

from clara.text import split_sentences

LLM_FIELDS = ("clara", "td", "cs", "tc", "zs", "cot", "gpt")
TEXT_FIELDS = ("td", "cs")          # {"rationale": str, ...}
_JOIN = {"zh": "", "en": " "}


def replace_llm_outputs(items: List[Dict[str, Any]], rate: float, seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    n = len(items)
    out = []
    for i, it in enumerate(items):
        it = dict(it)
        it["corrupted"] = False
        if n > 1 and rng.random() < rate:
            j = rng.randrange(n - 1)
            j = j + 1 if j >= i else j
            for f in LLM_FIELDS:
                if f in items[j]:
                    it[f] = items[j][f]
            it["corrupted"] = True
        out.append(it)
    return out


def _other(rng: random.Random, n: int, i: int) -> int:
    j = rng.randrange(n - 1)
    return j + 1 if j >= i else j


def _mix_text(rng: random.Random, text: str, pool: List[List[str]], i: int, rate: float, lang: str):
    """Replace each sentence of ``text`` with probability ``rate`` by a random sentence of
    another item's text of the same kind.  Returns (new text, number of replaced sentences)."""
    sents = [s.strip() for s in split_sentences(text, lang, max_sents=64)]
    n_rep = 0
    for k in range(len(sents)):
        if rng.random() < rate:
            donor = pool[_other(rng, len(pool), i)]
            if donor:
                sents[k] = rng.choice(donor)
                n_rep += 1
    return _JOIN[lang].join(sents), n_rep


def replace_units(items: List[Dict[str, Any]], rate: float, lang: str, seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    n = len(items)
    pools = {f: [[s.strip() for s in split_sentences(it[f]["rationale"], lang, max_sents=64)] if f in it else []
                 for it in items] for f in TEXT_FIELDS}
    pools["tc"] = [[s.strip() for s in split_sentences(it["tc"], lang, max_sents=64)] if "tc" in it else []
                   for it in items]
    out = []
    for i, it in enumerate(items):
        it = dict(it)
        n_rep = 0
        if n > 1:
            if it.get("clara"):
                c = copy.copy(it["clara"])
                claims = []
                for cl in c["claims"]:
                    cl = dict(cl)
                    cl["corrupted"] = False
                    if rng.random() < rate:
                        donor = items[_other(rng, n, i)].get("clara")
                        if donor and donor["claims"]:
                            cl = dict(rng.choice(donor["claims"]))
                            cl["corrupted"] = True
                            n_rep += 1
                    claims.append(cl)
                c["claims"] = claims
                it["clara"] = c
            for f in TEXT_FIELDS:
                if f in it:
                    r = dict(it[f])
                    r["rationale"], k = _mix_text(rng, r["rationale"], pools[f], i, rate, lang)
                    it[f] = r
                    n_rep += k
            if "tc" in it:
                it["tc"], k = _mix_text(rng, it["tc"], pools["tc"], i, rate, lang)
                n_rep += k
        it["corrupted"] = n_rep > 0
        out.append(it)
    return out
