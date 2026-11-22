"""Sentence splitting shared by the LLM prompts, the claim->sentence alignment and the student model.

The same splitter must be used everywhere: the LLM is shown the numbered sentences produced
here, and the sentence indices it returns are only meaningful with respect to this exact split.
"""
import re
from typing import List

MAX_SENTS = 12

_ZH_BOUNDARY = re.compile(r"(?<=[。！？!?；;…\n】])")
# a sentence ends with terminal punctuation, optionally followed by closing quotes/brackets
# (curly quotes may be separated by a space in pre-tokenised corpora such as GossipCop)
_EN_BOUNDARY = re.compile(
    r"[.!?]+(?:[\"'”’)\]]|\s+[”’])*\s+(?=(?:[\"'“‘(\[`]\s*)*[A-Z0-9])"
)
_EN_ABBR = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|No|Inc|Ltd|Co|Corp|Gov|Sen|Rep|Gen|Col|Lt|Sgt|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|a\.m|p\.m|U\.S|U\.K|e\.g|i\.e)\.|\b[A-Z]\.)$"
)


def _merge_short(parts: List[str], min_len: int) -> List[str]:
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and (len(p) < min_len or len(out[-1]) < min_len):
            out[-1] = out[-1] + p
        else:
            out.append(p)
    return out


def _cap(parts: List[str], max_sents: int, joiner: str) -> List[str]:
    """Greedily merge the shortest adjacent pair until at most ``max_sents`` sentences remain."""
    parts = list(parts)
    while len(parts) > max_sents:
        best, best_len = 0, None
        for i in range(len(parts) - 1):
            cur = len(parts[i]) + len(parts[i + 1])
            if best_len is None or cur < best_len:
                best, best_len = i, cur
        parts[best:best + 2] = [parts[best] + joiner + parts[best + 1]]
    return parts


def split_sentences(text: str, lang: str, max_sents: int = MAX_SENTS) -> List[str]:
    """Split ``text`` into at most ``max_sents`` sentences. ``lang`` is ``zh`` or ``en``.

    Always returns at least one sentence (the whole text) for non-empty input.
    """
    text = (text or "").strip()
    if not text:
        return [""]
    if lang == "zh":
        parts = _merge_short(_ZH_BOUNDARY.split(text), min_len=6)
        parts = _cap(parts, max_sents, "")
    else:
        raw, last = [], 0
        for m in _EN_BOUNDARY.finditer(text):
            raw.append(text[last:m.end()])
            last = m.end()
        raw.append(text[last:])
        merged: List[str] = []
        for p in raw:
            p = p.strip()
            if not p:
                continue
            if merged and _EN_ABBR.search(merged[-1]):
                merged[-1] = merged[-1] + " " + p
            else:
                merged.append(p)
        parts = []
        for p in merged:
            if parts and (len(p) < 20 or len(parts[-1]) < 20):
                parts[-1] = parts[-1] + " " + p
            else:
                parts.append(p)
        parts = _cap(parts, max_sents, " ")
    return parts if parts else [text]


def number_sentences(sents: List[str]) -> str:
    return "\n".join("[{}] {}".format(i + 1, s) for i, s in enumerate(sents))
