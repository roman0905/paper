"""Loading and normalising RAWFC / LIAR-RAW, and the evidence protocols.

Both datasets (Yang et al., COLING 2022) ship, for every claim, the raw reports that a search
engine returned for it, a gold fact-checking justification (``explain``) and sentence-level silver
evidence labels (``is_evidence``) that were derived from that justification.  Two things follow:

* ``explain`` and ``is_evidence`` are *privileged*: they may supervise training, they must never be
  part of a test-time input.
* some of the raw reports are themselves fact-checks or copies of the dataset.  Every report is
  therefore tagged (:func:`report_flags`) and the ``strict`` protocol drops the tagged ones.

A normalised item::

    {"id": str, "split": str, "claim": str, "label": int, "explain": str,
     "reports": [{"link": str, "host": str, "flags": [str], "sents": [str], "ev": [0/1]}]}

Labels are ordered from least to most truthful.
"""
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

SPLITS = ("train", "val", "test")

LABELS = {
    "RAWFC": ["false", "half", "true"],
    "LIAR-RAW": ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"],
}
#: how the labels are shown to an LLM / a reader
LABEL_TEXT = {
    "RAWFC": ["false", "half true", "true"],
    "LIAR-RAW": ["pants on fire", "false", "barely true", "half true", "mostly true", "true"],
}

# ------------------------------------------------------------------------------------ leak tagging
#: the fact-checker a dataset's gold labels come from
_SOURCE = re.compile(r"(^|\.)(snopes\.com|politifact\.com)$")
#: other dedicated fact-checking sites and aggregators of fact-checks
_CHECKER = re.compile(
    r"(^|\.)(factcheck\.org|truthorfiction\.com|leadstories\.com|checkyourfact\.com|fullfact\.org|"
    r"factcheck\.afp\.com|hoax-slayer\.(com|net)|mediabiasfactcheck\.com|rssing\.com|altnews\.in|"
    r"boomlive\.in|factcheck\.thedispatch\.com|verafiles\.org|africacheck\.org)$")
#: code / data hosting: copies of the benchmark itself
_DUMP = re.compile(
    r"(^|\.)(github\.com|githubusercontent\.com|kaggle\.com|huggingface\.co|paperswithcode\.com|"
    r"gitlab\.com|rpubs\.com|colab\.research\.google\.com|data\.world)$")
#: fact-check sections of general news outlets, recognised by their URL
_URL_FC = re.compile(r"fact-?check|factcheck|/fact_check|debunk|hoax|pinocchio|truth-o-meter", re.I)
#: reports whose text talks about fact-checking
_TEXT_FC = re.compile(
    r"\b(fact[- ]?check\w*|snopes|politifact|pants on fire|truth-o-meter|debunk\w*|pinocchio\w*)\b", re.I)

#: flags that the ``strict`` protocol removes
LEAK_FLAGS = ("source", "checker", "dump", "fc_url", "fc_text")
PROTOCOLS = ("all", "strict")


def host_of(link: str) -> str:
    try:
        h = urlparse(link if "//" in link else "//" + link).netloc.lower()
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


def report_flags(link: str, sents: Iterable[str]) -> List[str]:
    """Why a report may give the verdict away (empty list: no reason found)."""
    h, flags = host_of(link), []
    if _SOURCE.search(h):
        flags.append("source")
    if _CHECKER.search(h):
        flags.append("checker")
    if _DUMP.search(h):
        flags.append("dump")
    if not flags and _URL_FC.search(link or ""):
        flags.append("fc_url")
    if _TEXT_FC.search(" ".join(sents)):
        flags.append("fc_text")
    return flags


# ---------------------------------------------------------------------------------- normalisation
_WS = re.compile(r"\s+")
_PRE = re.compile(r"\s+([,.;:!?%)\]}]|'s\b|n't\b|'re\b|'ve\b|'ll\b|'d\b|'m\b)")
_POST = re.compile(r"([(\[{$#])\s+")


def clean_sentence(s: str) -> str:
    """The released reports are tokenised (and lemmatised) text; undo the spacing around
    punctuation and the PTB quotes so that the sentences cost fewer LLM tokens."""
    s = _WS.sub(" ", s.replace("``", '"').replace("''", '"')).strip()
    s = _PRE.sub(r"\1", s)
    s = _POST.sub(r"\1", s)
    return s


def normalise(raw: Dict[str, Any], dataset: str, split: str) -> Dict[str, Any]:
    names = LABELS[dataset]
    reports = []
    for rep in raw.get("reports") or []:
        link = rep.get("link") or ""
        sents, ev = [], []
        for t in rep.get("tokenized") or []:
            s = clean_sentence(t.get("sent") or "")
            if s:
                sents.append(s)
                ev.append(int(t.get("is_evidence") or 0))
        if sents:
            reports.append({"link": link, "host": host_of(link), "flags": report_flags(link, sents),
                            "sents": sents, "ev": ev})
    return {"id": str(raw["event_id"]).replace(".json", ""), "split": split,
            "claim": _WS.sub(" ", raw["claim"]).strip(), "label": names.index(raw["label"]),
            "explain": _WS.sub(" ", raw.get("explain") or "").strip(), "reports": reports}


def load_raw(root: str, split: str) -> List[Dict[str, Any]]:
    """RAWFC stores one file per claim, LIAR-RAW one file per split."""
    path = os.path.join(root, split + ".json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    d = os.path.join(root, split)
    out = []
    for fn in sorted(os.listdir(d)):
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def reports_under(item: Dict[str, Any], protocol: str) -> List[Dict[str, Any]]:
    """The reports of a claim that a protocol allows as evidence."""
    if protocol == "all":
        return item["reports"]
    if protocol == "strict":
        return [r for r in item["reports"] if not set(r["flags"]) & set(LEAK_FLAGS)]
    raise ValueError("unknown protocol " + protocol)


class Corpus:
    """All claims of one dataset."""

    def __init__(self, name: str, items: List[Dict[str, Any]]):
        self.name, self.items = name, items
        self.labels, self.label_text = LABELS[name], LABEL_TEXT[name]
        self.by_split: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLITS}
        for it in items:
            self.by_split.setdefault(it["split"], []).append(it)

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    def split(self, name: str) -> List[Dict[str, Any]]:
        return self.by_split[name]

    @classmethod
    def from_raw(cls, name: str, root: str, limit: Optional[int] = None) -> "Corpus":
        items = []
        for sp in SPLITS:
            raw = load_raw(root, sp)
            if limit:
                raw = raw[:limit]
            items += [normalise(r, name, sp) for r in raw]
        return cls(name, items)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "items": self.items}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Corpus":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["name"], d["items"])
