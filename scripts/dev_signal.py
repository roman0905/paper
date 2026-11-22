#!/usr/bin/env python
"""Prompt-development diagnostic: how much label signal do the raw LLM outputs carry?

Run only on samples drawn from the TRAIN split.  Usage: dev_signal.py OUT_DIR NAME DATA_JSON
"""
import collections
import json
import math
import statistics
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    out_dir, name, data = sys.argv[1:4]
    gold = {x["source_id"]: x for x in json.load(open(data, encoding="utf-8"))}
    lab = lambda i: {"real": 0, "fake": 1, 0: 0, 1: 1}[gold[i]["label"]]
    R = load("{}/{}.clara.jsonl".format(out_dir, name))
    ok = [r for r in R if r["parsed"]]
    print("== {}: n={} parsed={} soft={} attempt2={} finish={}".format(
        name, len(R), len(ok), sum(r["parsed"]["soft"] for r in ok),
        sum(r["attempt"] == 2 for r in R), dict(collections.Counter(r["finish"] for r in R))))
    print("   tokens: prompt mean {:.0f}, completion mean {:.0f} / max {}".format(
        statistics.mean(r["prompt_tokens"] for r in R), statistics.mean(r["completion_tokens"] for r in R),
        max(r["completion_tokens"] for r in R)))
    K = [len(r["parsed"]["claims"]) for r in ok]
    print("   K: mean {:.2f} dist {}".format(statistics.mean(K), sorted(collections.Counter(K).items())))
    print("   types:", dict(collections.Counter(c["type"] for r in ok for c in r["parsed"]["claims"])))
    y = np.array([lab(r["id"]) for r in ok])
    for yy in (0, 1):
        h = collections.Counter(c["suspicion"] for r in ok if lab(r["id"]) == yy for c in r["parsed"]["claims"])
        f = collections.Counter(r["parsed"]["overall"]["suspicion"] for r in ok if lab(r["id"]) == yy)
        print("   label={} claim suspicion {} | overall suspicion {}".format(yy, sorted(h.items()), sorted(f.items())))
    feats = {
        "max claim v": [max(c["v"] for c in r["parsed"]["claims"]) for r in ok],
        "mean claim v": [statistics.mean(c["v"] for c in r["parsed"]["claims"]) for r in ok],
        "first claim v": [r["parsed"]["claims"][0]["v"] for r in ok],
        "overall v": [r["parsed"]["overall"]["v"] for r in ok],
        "noisy-or(claims+overall)": [1 - math.prod(1 - 0.98 * c["v"] for c in r["parsed"]["claims"] + [r["parsed"]["overall"]]) for r in ok],
        "mean(all v)": [statistics.mean([c["v"] for c in r["parsed"]["claims"]] + [r["parsed"]["overall"]["v"]]) for r in ok],
        "0.5*max+0.5*overall": [0.5 * max(c["v"] for c in r["parsed"]["claims"]) + 0.5 * r["parsed"]["overall"]["v"] for r in ok],
    }
    if all("v0" in r["parsed"]["overall"] for r in ok):
        feats["first impression v0"] = [r["parsed"]["overall"]["v0"] for r in ok]
        feats["0.5*v0+0.5*overall"] = [0.5 * r["parsed"]["overall"]["v0"] + 0.5 * r["parsed"]["overall"]["v"] for r in ok]
        feats["(v0+overall+max claim)/3"] = [(r["parsed"]["overall"]["v0"] + r["parsed"]["overall"]["v"] + max(c["v"] for c in r["parsed"]["claims"])) / 3 for r in ok]
        sh = np.array([abs(r["parsed"]["overall"]["v0"] - r["parsed"]["overall"]["v"]) for r in ok])
        hard0 = np.array([(r["parsed"]["overall"]["v0"] >= 0.5) == bool(lab(r["id"])) for r in ok])
        print("   belief shift |v0-overall|: mean {:.3f}; first-impression correct when shift<median {:.3f} vs >=median {:.3f}".format(
            sh.mean(), hard0[sh < np.median(sh)].mean(), hard0[sh >= np.median(sh)].mean()))
    for k, v in feats.items():
        v = np.array(v)
        best = max(f1_score(y, (v >= t).astype(int), average="macro") for t in np.linspace(0.05, 0.95, 37))
        print("   AUC {:.3f}  macF1@0.5 {:.3f}  best-thr macF1 {:.3f}   <- {}".format(
            roc_auc_score(y, v), f1_score(y, (v >= 0.5).astype(int), average="macro"), best, k))
    # order effect: does writing 'refute' last push suspicion up?
    for o in (0, 1):
        vs = [c["v"] for r in ok if r["order"] == o for c in r["parsed"]["claims"]]
        print("   order={} ({} last): mean claim v {:.3f}".format(o, "refute" if o == 0 else "support", statistics.mean(vs)))
    for t in ("td", "cs", "zs", "cot"):
        try:
            P = load("{}/{}.{}.jsonl".format(out_dir, name, t))
        except IOError:
            continue
        yy = np.array([lab(p["id"]) for p in P])
        pr = np.array([1 if p["pred"] == "fake" else 0 for p in P])
        oth = sum(p["pred"] == "other" for p in P)
        soft = ""
        if all(p.get("p_fake") is not None for p in P):
            soft = " soft-AUC {:.3f}".format(roc_auc_score(yy, np.array([p["p_fake"] for p in P])))
        print("   {}: acc {:.3f} macF1 {:.3f} pred-fake-rate {:.2f} other {} length-cut {}{}".format(
            t, (pr == yy).mean(), f1_score(yy, pr, average="macro"), pr.mean(), oth,
            sum(p["finish"] == "length" for p in P), soft))


if __name__ == "__main__":
    main()
