#!/usr/bin/env python
"""Build controlled perturbations of a test split.

style   fact-preserving style transfer (fake -> authoritative, real -> sensational):
          1. python scripts/llm_generate.py --tasks style --split test \
                 --spec ARG-CN.style=outputs/datasets/ARG-CN.qwen25.json:zh ...
          2. python scripts/make_perturb.py style-build --dataset ... --gen .../ARG-CN.style.style.jsonl \
                 --out outputs/perturb/ARG-CN-style/test.json
tamper  one minimal factual edit in one sentence of a real test item (falsehood position known):
          1. python scripts/make_perturb.py tamper-input --dataset ... --out outputs/perturb/ARG-CN-tamper/input.json
          2. python scripts/llm_generate.py --tasks tamper --spec ARG-CN.tamperin=.../input.json:zh ...
          3. python scripts/make_perturb.py tamper-build --input .../input.json --gen ... --out .../test.json

The written test.json has the raw ARG format, so the usual generation + build_dataset pipeline can
analyse the perturbed texts.  New items get fresh ids (``ID_OFFSET`` + original id).
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clara.data import Dataset  # noqa: E402
from clara.text import split_sentences  # noqa: E402

ID_OFFSET = {"style": 10_000_000, "tamper": 20_000_000}
MIN_SENT_CHARS = {"zh": 12, "en": 40}
JOIN = {"zh": "", "en": " "}


def read_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def raw_item(it, content, label, new_id, extra):
    """An item in the raw ARG format (released GPT rationales are kept only as placeholders)."""
    g = it["gpt"]
    names = {0: "real", 1: "fake", 2: "other"}
    d = {"content": content, "label": names[label], "time": it.get("time"), "source_id": new_id,
         "td_rationale": g["td"]["rationale"], "td_pred": names[g["td"]["pred"]], "td_acc": g["td"]["acc"],
         "cs_rationale": g["cs"]["rationale"], "cs_pred": names[g["cs"]["pred"]], "cs_acc": g["cs"]["acc"],
         "split": "test", "orig_id": it["id"]}
    d.update(extra)
    return d


def style_build(args):
    ds = Dataset.load(args.dataset)
    gen = read_jsonl(args.gen)
    out, stats = [], collections.Counter()
    for it in ds.split("test"):
        r = gen.get(it["id"])
        text = (r or {}).get("text", "").strip()
        ratio = len(text) / max(len(it["content"]), 1)
        ok = r is not None and r["finish"] == "stop" and 0.4 <= ratio <= 2.5
        stats["ok" if ok else "kept_original"] += 1
        stats["label{}_{}".format(it["label"], r["style"] if r else "na")] += 1
        out.append(raw_item(it, text if ok else it["content"], it["label"], ID_OFFSET["style"] + it["id"],
                            {"style": r["style"] if r else None, "rewritten": bool(ok)}))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(args.out, len(out), dict(stats))


def tamper_input(args):
    ds = Dataset.load(args.dataset)
    rng = random.Random(args.seed)
    out = []
    for it in ds.split("test"):
        if it["label"] != 0 or len(it["sents"]) < 2:
            continue
        cand = [j for j, s in enumerate(it["sents"]) if len(s.strip()) >= MIN_SENT_CHARS[ds.lang]]
        if not cand:
            continue
        d = dict(it)
        d["tamper_sent"] = rng.choice(cand)
        d["lang"] = ds.lang
        out.append(d)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(args.out, len(out), "of", sum(it["label"] == 0 for it in ds.split("test")), "real test items")


def tamper_build(args):
    items = json.load(open(args.input, encoding="utf-8"))
    gen = read_jsonl(args.gen)
    out, stats = [], collections.Counter()
    for it in items:
        lang, j = it["lang"], it["tamper_sent"]
        r = gen.get(it["id"])
        if r is None or r["finish"] != "stop":
            stats["no_generation"] += 1
            continue
        old, new = it["sents"][j].strip(), r["text"].strip()
        if not new or new == old or not (0.5 <= len(new) / len(old) <= 1.6):
            stats["bad_edit"] += 1
            continue
        sents = list(it["sents"])
        sents[j] = new
        content = JOIN[lang].join(s.strip() if lang == "en" else s for s in sents)
        again = split_sentences(content, lang)
        if len(again) != len(sents) or any(a.strip() != b.strip() for a, b in zip(again, sents)):
            stats["split_changed"] += 1
            continue
        stats["ok"] += 1
        out.append(raw_item(it, content, 1, ID_OFFSET["tamper"] + it["id"],
                            {"tamper_sent": j, "orig_sentence": old, "new_sentence": new}))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(args.out, len(out), dict(stats))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("style-build")
    a.add_argument("--dataset", required=True)
    a.add_argument("--gen", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("tamper-input")
    a.add_argument("--dataset", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("tamper-build")
    a.add_argument("--input", required=True)
    a.add_argument("--gen", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"style-build": style_build, "tamper-input": tamper_input, "tamper-build": tamper_build}[args.cmd](args)


if __name__ == "__main__":
    main()
