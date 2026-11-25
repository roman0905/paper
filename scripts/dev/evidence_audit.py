#!/usr/bin/env python
"""Audit of the raw reports shipped with RAWFC / LIAR-RAW: where do they come from, and how
often do they give the verdict away?

    python -I scripts/dev/evidence_audit.py data/RAWFC data/LIAR-RAW

Read-only; prints aggregate counts (no dataset text is written anywhere).
"""
import collections
import json
import os
import re
import sys
from urllib.parse import urlparse


def host(u):
    try:
        h = urlparse(u if "//" in u else "//" + u).netloc.lower()
    except ValueError:
        return "?"
    return h[4:] if h.startswith("www.") else h


def load(root, split):
    path = os.path.join(root, split + ".json")
    if os.path.isfile(path):                                  # LIAR-RAW: one file per split
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    d = os.path.join(root, split)                             # RAWFC: one file per claim
    out = []
    for fn in sorted(os.listdir(d)):
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            out.append(json.load(f))
    return out


SOURCE = re.compile(r"(^|\.)(snopes\.com|politifact\.com)$")
CHECKER = re.compile(r"(^|\.)(factcheck\.org|truthorfiction\.com|leadstories\.com|checkyourfact\.com|fullfact\.org|"
                     r"factcheck\.afp\.com|hoax-slayer\.(com|net)|mediabiasfactcheck\.com|rssing\.com|altnews\.in|boomlive\.in)$")
DUMP = re.compile(r"(^|\.)(github\.com|githubusercontent\.com|kaggle\.com|huggingface\.co|paperswithcode\.com|"
                  r"gitlab\.com|rpubs\.com|colab\.research\.google\.com|data\.world)$")
URL_FC = re.compile(r"fact-?check|factcheck|/fact_check|debunk|hoax|pinocchio|truth-o-meter", re.I)
TEXT_FC = re.compile(r"\b(fact[- ]?check\w*|snopes|politifact|pants on fire|truth-o-meter|debunk\w*|pinocchio\w*)\b", re.I)
VERDICT = re.compile(r"\b(false|true|misleading|hoax|fake|untrue|incorrect|no evidence|mostly true|half true|"
                     r"mostly false|barely true|pants on fire|mixture|unproven)\b", re.I)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def audit(root):
    name = os.path.basename(root.rstrip("/"))
    print("=" * 100)
    print(name)
    for split in ("train", "val", "test"):
        data = load(root, split)
        n = len(data)
        c = collections.Counter()
        per_claim = collections.Counter()
        hosts = collections.Counter()
        n_rep = n_sent = n_ev = 0
        ev_fc = sent_fc = 0
        for r in data:
            claim_n = norm(r["claim"])
            flags = set()
            for rep in r.get("reports") or []:
                n_rep += 1
                h = host(rep.get("link") or "")
                hosts[h] += 1
                link = rep.get("link") or ""
                sents = [t["sent"] for t in rep.get("tokenized") or []]
                text = " ".join(sents)
                kinds = []
                if SOURCE.search(h):
                    kinds.append("source")
                if CHECKER.search(h):
                    kinds.append("checker")
                if DUMP.search(h):
                    kinds.append("dump")
                if URL_FC.search(link) and not kinds:
                    kinds.append("fc_url")
                if TEXT_FC.search(text):
                    kinds.append("fc_text")
                if claim_n and len(claim_n) > 40 and claim_n in norm(text):
                    kinds.append("verbatim")
                for k in kinds:
                    c["reports/" + k] += 1
                    flags.add(k)
                if not kinds:
                    c["reports/clean"] += 1
                for t in rep.get("tokenized") or []:
                    n_sent += 1
                    e = int(t.get("is_evidence") or 0)
                    n_ev += e
                    if TEXT_FC.search(t["sent"]):
                        sent_fc += 1
                        ev_fc += e
            for k in flags:
                per_claim[k] += 1
            if flags & {"source", "checker", "dump", "fc_url"}:
                per_claim["any_site_level"] += 1
            if flags & {"source", "checker", "dump", "fc_url", "fc_text"}:
                per_claim["any_incl_text"] += 1
        print("\n[{}] claims {}  reports {}  sentences {}  silver evidence sentences {} ({:.1f}%)".format(
            split, n, n_rep, n_sent, n_ev, 100.0 * n_ev / max(n_sent, 1)))
        print("   reports by flag  :", {k.split("/")[1]: "{} ({:.1f}%)".format(v, 100.0 * v / n_rep) for k, v in sorted(c.items())})
        print("   claims with >=1  :", {k: "{} ({:.1f}%)".format(v, 100.0 * v / n) for k, v in sorted(per_claim.items())})
        print("   sentences mentioning fact-checking: {} ({:.2f}%); share among silver evidence {:.2f}%".format(
            sent_fc, 100.0 * sent_fc / max(n_sent, 1), 100.0 * ev_fc / max(n_ev, 1)))
        if split == "test":
            sus = [(h, k) for h, k in hosts.most_common(4000) if SOURCE.search(h) or CHECKER.search(h) or DUMP.search(h)]
            print("   flagged hosts    :", sus[:25])


if __name__ == "__main__":
    for root in sys.argv[1:]:
        audit(root)
