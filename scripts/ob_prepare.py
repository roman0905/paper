#!/usr/bin/env python
"""Normalise RAWFC / LIAR-RAW and compute the first-stage (BM25) evidence rankings.

    python -I scripts/ob_prepare.py RAWFC data/RAWFC
    python -I scripts/ob_prepare.py LIAR-RAW data/LIAR-RAW

Writes outputs/ob/<dataset>.json (normalised corpus) and
outputs/ob/rank/<dataset>.<protocol>.bm25.json (top ``--keep`` candidates per claim), and prints
dataset statistics that the thesis quotes (also stored in results/ob/<dataset>.stats.json: counts
only, no text).
"""
import argparse
import collections
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from openbook import retrieval as R  # noqa: E402
from openbook.data import LEAK_FLAGS, PROTOCOLS, SPLITS, Corpus, reports_under  # noqa: E402


def stats(corpus):
    out = {}
    for sp in SPLITS:
        items = corpus.split(sp)
        n = len(items)
        flag_reports, flag_claims = collections.Counter(), collections.Counter()
        n_rep = n_sent = n_ev = 0
        leaky_claims = 0
        for it in items:
            flags = set()
            for rep in it["reports"]:
                n_rep += 1
                n_sent += len(rep["sents"])
                n_ev += sum(rep["ev"])
                for f in rep["flags"]:
                    flag_reports[f] += 1
                    flags.add(f)
            for f in flags:
                flag_claims[f] += 1
            leaky_claims += bool(flags & set(LEAK_FLAGS))
        row = {"claims": n, "labels": np.bincount([it["label"] for it in items], minlength=corpus.n_classes).tolist(),
               "reports": n_rep, "sentences": n_sent, "silver": n_ev,
               "claim_words": float(np.mean([len(it["claim"].split()) for it in items])),
               "explain_words": float(np.mean([len(it["explain"].split()) for it in items])),
               "flagged_reports": dict(flag_reports), "flagged_claims": dict(flag_claims),
               "claims_with_leak": leaky_claims}
        for p in PROTOCOLS:
            pools = [R.pool(it, p) for it in items]
            row["pool_" + p] = {"mean": float(np.mean([len(x) for x in pools])),
                                "empty": int(sum(len(x) == 0 for x in pools)),
                                "reports": float(np.mean([len(reports_under(it, p)) for it in items])),
                                "silver_rate": float(np.mean([R.is_silver(it, c) for it, x in zip(items, pools) for c in x]))}
        out[sp] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["RAWFC", "LIAR-RAW"])
    ap.add_argument("root")
    ap.add_argument("--keep", type=int, default=64, help="candidates kept per claim in a ranking file")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    corpus = Corpus.from_raw(args.dataset, args.root, limit=args.limit or None)
    path = os.path.join(ROOT, "outputs", "ob", args.dataset + ".json")
    corpus.save(path)
    print("wrote", path, "({} claims, {:.0f}s)".format(len(corpus.items), time.time() - t0), flush=True)

    st = stats(corpus)
    for sp, row in st.items():
        print(sp, json.dumps(row, ensure_ascii=False))
    quality = {}
    for p in PROTOCOLS:
        ranking = {it["id"]: R.bm25_ranking(it, p) for it in corpus.items}
        quality[p] = {sp: R.ranking_quality(corpus.split(sp), ranking) for sp in SPLITS}   # on the full pools
        R.save_ranking(R.ranking_path(ROOT, args.dataset, p, "bm25"), {k: v[:args.keep] for k, v in ranking.items()})
        print("bm25", p, {sp: {k: round(v, 3) for k, v in q.items()} for sp, q in quality[p].items()}, flush=True)
    if not args.limit:
        res = os.path.join(ROOT, "results", "ob", args.dataset + ".stats.json")
        os.makedirs(os.path.dirname(res), exist_ok=True)
        with open(res, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "splits": st, "bm25": quality}, f, indent=1, sort_keys=True)
        print("wrote", res)
    print("done in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
