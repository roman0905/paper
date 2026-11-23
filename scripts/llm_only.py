#!/usr/bin/env python
"""Training-free LLM baselines computed from a dataset file (no GPU needed).

Hard judgments of the zero-shot / CoT / perspective prompts, and soft scores thresholded either
at 0.5 or at the threshold that maximises macro-F1 on the validation split.

    python scripts/llm_only.py outputs/datasets/ARG-CN.qwen25.json [--json results/llm_only/X.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from clara.data import Dataset  # noqa: E402
from clara.metrics import best_threshold, detection_metrics  # noqa: E402


def hard(items, get):
    """Hard predictions; an undecided LLM ("other") is counted as predicting real."""
    return np.array([1.0 if get(it) == 1 else 0.0 for it in items])


def soft_scores(it):
    c = it.get("clara")
    if not c:
        return None
    claims = [x["v"] for x in c["claims"]]
    o = c["overall"]
    return {
        "CLARA-prompt: first impression": o["v0"],
        "CLARA-prompt: overall verdict": o["v"],
        "CLARA-prompt: max claim suspicion": max(claims),
        "CLARA-prompt: mean claim suspicion": float(np.mean(claims)),
        "CLARA-prompt: noisy-OR of claims": 1 - float(np.prod([1 - 0.999 * v for v in claims])),
        "CLARA-prompt: mean(first, overall)": 0.5 * (o["v0"] + o["v"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ds = Dataset.load(args.dataset)
    val, test = ds.split("val"), ds.split("test")
    yv, yt = np.array([it["label"] for it in val]), np.array([it["label"] for it in test])
    rows = {}

    for name, get in [
        ("GPT-3.5 TD (released)", lambda it: it["gpt"]["td"]["pred"]),
        ("GPT-3.5 CS (released)", lambda it: it["gpt"]["cs"]["pred"]),
        ("zero-shot", lambda it: it["zs"]["pred"]),
        ("zero-shot CoT", lambda it: it["cot"]["pred"]),
        ("CoT, textual description (TD)", lambda it: it["td"]["pred"]),
        ("CoT, commonsense (CS)", lambda it: it["cs"]["pred"]),
    ]:
        try:
            rows[name] = {"test": detection_metrics(yt, hard(test, get)), "thr": 0.5, "kind": "hard",
                          "other_rate": float(np.mean([get(it) == 2 for it in test]))}
        except (KeyError, TypeError):
            continue

    softs = {}
    if all((it.get("zs") or {}).get("p_fake") is not None for it in val + test):
        softs["zero-shot, soft P(fake)"] = (np.array([it["zs"]["p_fake"] for it in val]),
                                            np.array([it["zs"]["p_fake"] for it in test]))
    sv, st = [soft_scores(it) for it in val], [soft_scores(it) for it in test]
    if all(x is not None for x in sv + st):
        for k in sv[0]:
            softs[k] = (np.array([x[k] for x in sv]), np.array([x[k] for x in st]))
    else:
        print("note: {} val / {} test items without a parsed CLARA analysis; soft CLARA rows skipped".format(
            sum(x is None for x in sv), sum(x is None for x in st)))
    for name, (pv, pt) in softs.items():
        thr = best_threshold(yv, pv)
        rows[name + " @0.5"] = {"test": detection_metrics(yt, pt, 0.5), "thr": 0.5, "kind": "soft"}
        rows[name + " @val-thr"] = {"test": detection_metrics(yt, pt, thr), "thr": thr, "kind": "soft"}

    print("| {} ({}) | macF1 | Acc | F1-real | F1-fake | AUC | thr |".format(ds.name, ds.llm))
    print("|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        m = r["test"]
        auc = "{:.2f}".format(100 * m["auc"]) if r["kind"] == "soft" else "-"
        print("| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} | {:.2f} |".format(
            name, 100 * m["macf1"], 100 * m["acc"], 100 * m["f1_real"], 100 * m["f1_fake"], auc, r["thr"]))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump({"dataset": ds.name, "llm": ds.llm, "rows": rows}, open(args.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
