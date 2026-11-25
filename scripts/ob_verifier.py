#!/usr/bin/env python
"""Train and evaluate one LLM verifier on RAWFC or LIAR-RAW.

    python scripts/ob_verifier.py --dataset RAWFC --closed --seed 1          # closed-book LoRA
    python scripts/ob_verifier.py --dataset RAWFC --epochs 0                 # zero-shot + evidence
    python scripts/ob_verifier.py --dataset RAWFC --seed 1                   # LoRA + BM25 evidence
    python scripts/ob_verifier.py --dataset RAWFC --seed 1 --name full \
        --ev-weight 0.5 --top 48 --swap-rate 0.15 --select                   # the full method

Evidence comes from a first-stage ranking (outputs/ob/rank/<dataset>.<protocol>.<ranker>.json).
The model is selected by validation macro-F1 after every epoch and then evaluated once on the
test split under several conditions:

    test            the inference procedure of the run
    test@bm25       (runs with --select) one pass over the first-stage top-m instead
    test@read       (runs with --select) mean of the first-pass queries
    test@<proto>    evidence pool of the other protocol
    test@closed     no evidence at all
    test@swap       the evidence of another claim

Writes results/ob/runs/<dataset>.<protocol>/<name>/s<seed>.json (metrics, configuration) and the
matching predictions under outputs/ob/runs/.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from clara.train import set_seed  # noqa: E402
from openbook import pipeline as P  # noqa: E402
from openbook import retrieval as R  # noqa: E402
from openbook import views as V  # noqa: E402
from openbook.data import PROTOCOLS, Corpus  # noqa: E402
from openbook.metrics import classification_metrics  # noqa: E402
from openbook.verifier import Verifier, fit  # noqa: E402


def run_name(args):
    if args.name:
        return args.name
    base = "zs" if args.epochs == 0 else "lora"
    return "{}-{}".format(base, "closed" if args.closed else "{}-m{}".format(args.ranker, args.m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["RAWFC", "LIAR-RAW"])
    ap.add_argument("--protocol", default="strict", choices=PROTOCOLS)
    ap.add_argument("--ranker", default="bm25")
    ap.add_argument("--m", type=int, default=16, help="evidence sentences per query")
    ap.add_argument("--closed", action="store_true", help="closed-book: train and test without evidence")
    # the components of the full method
    ap.add_argument("--ev-weight", type=float, default=0.0, help="weight of the evidence-tagging loss (0: no evidence head)")
    ap.add_argument("--top", type=int, default=0, help="size of the candidate pool: evidence is re-sampled from it in "
                                                        "training and read in full by --select (0: fixed top-m)")
    ap.add_argument("--swap-rate", type=float, default=0.0, help="share of training claims shown mismatched evidence")
    ap.add_argument("--select", action="store_true", help="two-pass inference: read the pool, verify on the selected evidence")
    # optimisation
    ap.add_argument("--model", default="/media/dell/DATA/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=4, help="0: no training (zero-shot read-out)")
    ap.add_argument("--micro-batch", type=int, default=None, help="default: 4 closed-book, 2 with evidence")
    ap.add_argument("--accum", type=int, default=None, help="default: micro-batch x accum = 16")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing (needs more memory)")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1100)
    ap.add_argument("--name", default=None)
    ap.add_argument("--train-limit", type=int, default=0, help="use only the first N training claims")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the size of every split, write nothing")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.select and not (args.ev_weight > 0 and args.top > args.m):
        ap.error("--select needs --ev-weight > 0 and --top > --m")

    if args.micro_batch is None:
        args.micro_batch = 4 if args.closed else 2
    if args.accum is None:
        args.accum = max(1, 16 // args.micro_batch)
    name = run_name(args)
    tag = "{}.{}".format(args.dataset, args.protocol)
    seed_tag = "s0" if args.epochs == 0 else "s{}".format(args.seed)
    res_path = os.path.join(ROOT, "results", "ob", "runs", tag, name, seed_tag + ".json")
    if os.path.exists(res_path) and not args.force and not args.limit:
        print("exists:", res_path)
        return

    corpus = Corpus.load(os.path.join(ROOT, "outputs", "ob", args.dataset + ".json"))
    rank = {p: R.load_ranking(R.ranking_path(ROOT, args.dataset, p, args.ranker)) for p in PROTOCOLS}
    splits = {s: corpus.split(s) for s in ("train", "val", "test")}
    if args.train_limit:
        splits["train"] = splits["train"][:args.train_limit]
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    C = corpus.n_classes
    other = [p for p in PROTOCOLS if p != args.protocol][0]
    y = {s: np.array([it["label"] for it in splits[s]]) for s in splits}

    set_seed(args.seed)
    t0 = time.time()
    v = Verifier(args.model, corpus.label_text, lora_rank=args.rank if args.epochs > 0 else 0, max_tokens=args.max_tokens,
                 grad_checkpoint=not args.closed and not args.no_grad_ckpt, evidence_head=args.ev_weight > 0)

    def infer(split, protocol=args.protocol):
        """The inference procedure of this run on a split."""
        if args.closed:
            return P.single_pass(v, V.closed(splits[split]), C, args.eval_batch)
        if args.select:
            return P.read_select_verify(v, splits[split], rank[protocol], args.m, args.top, C, args.eval_batch)
        return P.single_pass(v, V.standard(splits[split], rank[protocol], args.m), C, args.eval_batch)

    def val_score():
        return classification_metrics(y["val"], infer("val")["prob"])["macf1"]

    summary = {"best_epoch": -1, "epochs_run": 0, "history": [], "train_seconds": 0.0}
    if args.epochs > 0:
        rng = np.random.RandomState(1000 + args.seed)

        def train_fn(epoch):
            if args.closed:
                return V.closed(splits["train"])
            return V.training(splits["train"], rank[args.protocol], args.m, rng, top=args.top, swap_rate=args.swap_rate)

        summary = fit(v, train_fn, val_score, epochs=args.epochs, lr=args.lr, micro_batch=args.micro_batch,
                      accum=args.accum, seed=args.seed, lambda_ev=args.ev_weight, log=lambda s: print(s, flush=True))

    # ------------------------------------------------------------------------------ evaluation
    out, extra = {}, {}
    main_val, main_test = infer("val"), infer("test")
    out["val"], out["test"] = main_val["prob"], main_test["prob"]
    cost = {"tokens": main_test["tokens"], "queries": main_test.get("queries", 1.0)}
    if not args.closed:
        out["test@" + other] = infer("test", other)["prob"]
        out["test@closed"] = P.single_pass(v, V.closed(splits["test"]), C, args.eval_batch)["prob"]
        own = P.single_pass(v, V.standard(splits["test"], rank[args.protocol], args.m), C, args.eval_batch)
        swap = P.single_pass(v, V.swapped(splits["test"], rank[args.protocol], args.m, seed=0), C, args.eval_batch)
        out["test@swap"] = swap["prob"]
        if args.select:
            out["test@bm25"], out["test@read"] = own["prob"], main_test["prob_read"]
            out["test@mix"] = 0.5 * (main_test["prob"] + main_test["prob_read"])
            out["val@bm25"] = P.single_pass(v, V.standard(splits["val"], rank[args.protocol], args.m), C, args.eval_batch)["prob"]
            out["val@read"] = main_val["prob_read"]
            out["val@mix"] = 0.5 * (main_val["prob"] + main_val["prob_read"])
        if args.ev_weight > 0:
            # does the evidence head know which sentences belong to the claim, and which are useful?
            first = [it for it in splits["test"]]
            bm25_top = [[(a, b) for a, b, _ in rank[args.protocol].get(it["id"], [])[:args.m]] for it in first]
            extra["silver_p_bm25"] = P.silver_precision(first, bm25_top)
            if args.select:
                extra["silver_p_selected"] = P.silver_precision(first, main_test["selected"])
            own_s = np.concatenate([s for s in own["scores"]] or [np.zeros(0)])
            swap_s = np.concatenate([s for s in swap["scores"]] or [np.zeros(0)])
            extra["score_own_mean"], extra["score_swap_mean"] = float(own_s.mean()), float(swap_s.mean())
            extra["auc_own_vs_swap_sentence"] = P.score_auc(own_s, swap_s)
            extra["auc_own_vs_swap_claim"] = P.score_auc([s.max() for s in own["scores"] if len(s)],
                                                         [s.max() for s in swap["scores"] if len(s)])
            tags = np.array([t for ex, n in zip(V.standard(first, rank[args.protocol], args.m), own["n_ev"]) for t in ex["tags"][:n]])
            extra["auc_silver_sentence"] = P.score_auc(own_s[tags == 1], own_s[tags == 0])

    def split_of(k):
        return k.split("@")[0]

    metrics = {k: classification_metrics(y[split_of(k)], p) for k, p in out.items()}
    result = {
        "dataset": corpus.name, "protocol": args.protocol, "method": name, "name": name, "seed": args.seed,
        "encoder": os.path.basename(args.model.rstrip("/")),
        "config": {"ranker": args.ranker, "m": args.m, "closed": args.closed, "rank": args.rank, "lr": args.lr,
                   "epochs": args.epochs, "batch_size": args.micro_batch * args.accum, "max_tokens": args.max_tokens,
                   "ev_weight": args.ev_weight, "top": args.top, "swap_rate": args.swap_rate, "select": args.select,
                   "train_limit": args.train_limit},
        "summary": {k: summary[k] for k in ("best_epoch", "epochs_run", "train_seconds")}, "history": summary["history"],
        "metrics": metrics, "extra": extra, "cost": cost, "n_trainable": int(sum(p.numel() for p in v.params)),
        "n_items": {s: len(splits[s]) for s in splits},
        "seconds": time.time() - t0, "max_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
    }
    print(name, seed_tag, {k: round(m["macf1"], 4) for k, m in metrics.items()},
          {k: round(x, 3) for k, x in extra.items()}, "max mem {:.1f} GB".format(result["max_memory_gb"]), flush=True)
    if args.limit:
        return
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, sort_keys=True)
    pred_path = os.path.join(ROOT, "outputs", "ob", "runs", tag, name, seed_tag + ".npz")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    flat = {}
    for k, p in out.items():
        flat[k + "/prob"] = p.astype(np.float32)
        flat[k + "/label"] = y[split_of(k)]
        flat[k + "/id"] = np.array([it["id"] for it in splits[split_of(k)]])
    np.savez_compressed(pred_path, **flat)
    if args.select:
        with open(pred_path[:-4] + ".selected.json", "w", encoding="utf-8") as f:
            json.dump({"test": {it["id"]: c for it, c in zip(splits["test"], main_test["selected"])},
                       "val": {it["id"]: c for it, c in zip(splits["val"], main_val["selected"])}}, f)


if __name__ == "__main__":
    main()
