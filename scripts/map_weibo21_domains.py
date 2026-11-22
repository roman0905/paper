#!/usr/bin/env python
"""Recover Weibo21 domain labels for ARG-CN items.

ARG-CN is Weibo21 re-split by time and ships without the nine domain labels of the original
dataset.  Given a local copy of Weibo21 that still has them (the pickles distributed with
M3FEND: train/val/test.pkl with columns ``content``, ``category``, ``label``), items are matched
on their whitespace-stripped, lower-cased text (exact match first, then a 40-character prefix).
Only matches whose veracity label agrees are kept.

    python -I scripts/map_weibo21_domains.py WEIBO21_PKL_DIR data/ARG-CN outputs/datasets/ARG-CN.domains.json
"""
import collections
import json
import os
import re
import sys


def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def main():
    src, arg, out = sys.argv[1:4]
    import pandas as pd

    full, pref = {}, {}
    for sp in ("train", "val", "test"):
        df = pd.read_pickle(os.path.join(src, sp + ".pkl"))
        for c, cat, lab in zip(df["content"], df["category"], df["label"]):
            k = norm(c)
            full[k] = (cat, int(lab))
            pref.setdefault(k[:40], (cat, int(lab)))

    domains, stats = {}, collections.Counter()
    for sp in ("train", "val", "test"):
        for x in json.load(open(os.path.join(arg, sp + ".json"), encoding="utf-8")):
            k = norm(x["content"])
            y = {"real": 0, "fake": 1}[x["label"]]
            m = full.get(k) or pref.get(k[:40])
            if m is None:
                stats[sp + "/unmatched"] += 1
            elif m[1] != y:
                stats[sp + "/label_conflict"] += 1
            elif m[0] == "无法确定":
                stats[sp + "/undetermined"] += 1
            else:
                domains[x["source_id"]] = m[0]
                stats[sp + "/matched"] += 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(domains, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(dict(stats))
    print(collections.Counter(domains.values()).most_common())


if __name__ == "__main__":
    main()
