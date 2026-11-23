"""Dataset container and loading.

A dataset file (written by ``scripts/build_dataset.py``) is one JSON object::

    {"name": "ARG-CN", "lang": "zh", "llm": "qwen25", "items": [item, ...]}

with one item per news piece::

    id, split, label (1 = fake), time, content, sents, domain (optional),
    gpt:   {"td": {"rationale", "pred", "acc"}, "cs": {...}}      # released GPT-3.5 rationales
    td/cs: {"rationale", "pred"}                                  # same prompts, our LLM
    tc:    str                                                    # FactGuard topic-content
    zs:    {"pred", "p_fake"}; cot: {"rationale", "pred"}
    clara: {"claims": [...], "overall": {...}, "soft": bool} or None

``pred`` is 0 (real) / 1 (fake) / 2 (other), following ARG.
"""
import json
from typing import Any, Dict, List, Optional

LABEL = {"real": 0, "fake": 1, 0: 0, 1: 1}
PRED = {"real": 0, "fake": 1, "other": 2, 0: 0, 1: 1, 2: 2}
SPLITS = ("train", "val", "test")

OVERALL_CLAIM = {"zh": "这条消息整体的可信度", "en": "the overall credibility of this message"}
CLAIM_TYPE_ID = {"fact": 0, "number": 1, "quote": 2, "opinion": 3, "overall": 4}


class Dataset:
    def __init__(self, name: str, lang: str, items: List[Dict[str, Any]], llm: Optional[str] = None):
        self.name, self.lang, self.llm = name, lang, llm
        self.items = items
        self.by_split = {s: [it for it in items if it["split"] == s] for s in SPLITS}

    @classmethod
    def load(cls, path: str) -> "Dataset":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["name"], obj["lang"], obj["items"], obj.get("llm"))

    def split(self, name: str) -> List[Dict[str, Any]]:
        return self.by_split[name]

    def resplit(self, assign) -> "Dataset":
        """New dataset whose split of every item is ``assign(item)`` (None drops the item)."""
        items = []
        for it in self.items:
            s = assign(it)
            if s is not None:
                it = dict(it)
                it["split"] = s
                items.append(it)
        return Dataset(self.name, self.lang, items, self.llm)


def instances(item: Dict[str, Any], lang: str, max_claims: int = 6,
              v0_source: str = "zs") -> List[Dict[str, Any]]:
    """Instance list of one news item for CLARA: the holistic instance first, then the claims.

    Every instance has the post-argument suspicion score ``v``.  The holistic instance has in
    addition a pre-argument holistic score ``v0``: the LLM's plain zero-shot P(fake)
    (``v0_source="zs"``, a separate one-token query) or the first impression stated at the start
    of the structured call (``v0_source="fi"``); claims have ``v0 = v``.

    If the LLM output could not be parsed the item keeps only a neutral holistic instance, so the
    model falls back to its text-only branch for it.
    """
    c = item.get("clara")
    if not c:
        return [{"claim": OVERALL_CLAIM[lang], "support": "", "refute": "", "v": 0.5, "v0": 0.5,
                 "dist": None, "type": "overall", "sent": -1, "valid": False}]
    o = c["overall"]
    v0 = o.get("v0", o["v"])
    if v0_source == "zs":
        p_zs = (item.get("zs") or {}).get("p_fake")
        v0 = v0 if p_zs is None else p_zs
    out = [{"claim": OVERALL_CLAIM[lang], "support": o["support"], "refute": o["refute"],
            "v": float(o["v"]), "v0": float(v0), "dist": o.get("dist"),
            "type": "overall", "sent": -1, "valid": True}]
    for cl in c["claims"][:max_claims]:
        out.append({"claim": cl["claim"], "support": cl["support"], "refute": cl["refute"],
                    "v": float(cl["v"]), "v0": float(cl["v"]), "dist": cl.get("dist"),
                    "type": cl["type"], "sent": int(cl["sent"]) - 1, "valid": True,
                    "corrupted": bool(cl.get("corrupted", False))})
    return out
