#!/usr/bin/env python
"""Merge the raw ARG data with the offline LLM caches into one dataset file per (dataset, LLM).

usage: build_dataset.py --data data/ARG-CN --name ARG-CN --lang zh --llm-dir outputs/llm/qwen25 \
           --llm qwen25 --out outputs/datasets/ARG-CN.qwen25.json [--domains FILE.json]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clara.data import LABEL, PRED, SPLITS  # noqa: E402
from clara.text import split_sentences  # noqa: E402


EXTRA_FIELDS = ("orig_id", "tamper_sent", "orig_sentence", "new_sentence", "style", "rewritten")


def read_jsonl(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lang", required=True, choices=["zh", "en"])
    ap.add_argument("--llm-dir", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default=None, help="optional JSON {id: domain}")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = ap.parse_args()

    domains = {}
    if args.domains and os.path.exists(args.domains):
        domains = {int(k): v for k, v in json.load(open(args.domains, encoding="utf-8")).items()}

    items, stats = [], collections.Counter()
    for sp in args.splits:
        raw = json.load(open(os.path.join(args.data, sp + ".json"), encoding="utf-8"))
        cache = {t: read_jsonl(os.path.join(args.llm_dir, "{}.{}.{}.jsonl".format(args.name, sp, t)))
                 for t in ("clara", "td", "cs", "tc", "zs", "cot")}
        for d in raw:
            iid = d["source_id"]
            it = {
                "id": iid, "split": sp, "label": LABEL[d["label"]], "time": d.get("time"),
                "content": d["content"], "sents": split_sentences(d["content"], args.lang),
                "gpt": {
                    "td": {"rationale": d["td_rationale"], "pred": PRED[d["td_pred"]], "acc": int(d["td_acc"])},
                    "cs": {"rationale": d["cs_rationale"], "pred": PRED[d["cs_pred"]], "acc": int(d["cs_acc"])},
                },
            }
            if iid in domains:
                it["domain"] = domains[iid]
            for k in EXTRA_FIELDS:        # provenance of perturbed items
                if k in d:
                    it[k] = d[k]
            for t in ("td", "cs", "cot"):
                r = cache[t].get(iid)
                if r is not None:
                    it[t] = {"rationale": r["rationale"], "pred": PRED[r["pred"]]}
                    stats[t] += 1
                    stats[t + "_other"] += int(r["pred"] == "other")
            r = cache["tc"].get(iid)
            if r is not None:
                it["tc"] = r["raw"].strip()
                stats["tc"] += 1
            r = cache["zs"].get(iid)
            if r is not None:
                it["zs"] = {"pred": PRED[r["pred"]], "p_fake": r.get("p_fake")}
                stats["zs"] += 1
            r = cache["clara"].get(iid)
            if r is not None:
                stats["clara"] += 1
                it["clara"] = r["parsed"]
                if r["parsed"] is None:
                    stats["clara_failed"] += 1
                else:
                    stats["clara_soft"] += int(r["parsed"]["soft"])
                    it["clara"]["order"] = r["order"]
                it["cost"] = {"clara": [r["prompt_tokens"], r["completion_tokens"]]}
            items.append(it)
        stats["n_" + sp] = len(raw)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "lang": args.lang, "llm": args.llm, "items": items}, f, ensure_ascii=False)
    print(args.out, dict(stats))


if __name__ == "__main__":
    main()
