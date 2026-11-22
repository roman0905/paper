"""Test-time corruption of LLM outputs (rationale-noise robustness).

``replace_llm_outputs`` simulates irrelevant / hallucinated LLM analyses: with probability ``rate``
all LLM-generated fields of an item are replaced by those of another, randomly chosen item of the
same split. The news text and the label are untouched, so a detector that knows when to ignore the
LLM should degrade gracefully towards its text-only performance.
"""
import random
from typing import Any, Dict, List

LLM_FIELDS = ("clara", "td", "cs", "tc", "zs", "cot", "gpt")


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
