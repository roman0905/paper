#!/usr/bin/env python
"""Train and evaluate one LLM verifier on RAWFC or LIAR-RAW.

    python scripts/ob_verifier.py --dataset RAWFC --closed --seed 1          # closed-book LoRA
    python scripts/ob_verifier.py --dataset RAWFC --epochs 0                 # zero-shot + evidence
    python scripts/ob_verifier.py --dataset RAWFC --seed 1                   # LoRA + first-stage evidence
    python scripts/ob_verifier.py --dataset RAWFC --seed 1 --no-claim        # evidence-only control
    python scripts/ob_verifier.py --dataset RAWFC --seed 1 --name sieve \
        --ev-weight 0.5 --train-sample --swap-rate 0.15 --select             # the full method

Evidence comes from a first-stage ranking (outputs/ob/rank/<dataset>.<protocol>.<ranker>.json).
The model is selected by validation macro-F1 after every epoch and then evaluated once on the
test split.  ``test`` is the inference procedure of the run; the other conditions apply the same
procedure to a changed evidence pool, or change the procedure:

    test@<protocol>   evidence pool of the other protocol
    test@closed       no evidence at all
    test@swap         the whole pool comes from another claim
    test@halfswap     half of the pool comes from another claim
    test@droptop      the first-stage top-m sentences are missing
    test@oracle       upper bound, not a legitimate condition: evidence chosen with the test claims'
                      silver evidence labels
    test@read         mean over the queries that read the whole pool, no selection
    test@bm25         (runs with --select) one pass over the first-stage top-m
    test@avg          (runs with --select) mean of test and test@read
    xfer              (with --transfer) test split of the other dataset, labels mapped to 3 classes

Writes results/ob/runs/<dataset>.<protocol>/<name>/s<seed>.json (metrics, configuration); the
predictions and the trained adapter go to outputs/ob/runs/ (not tracked).
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

PERTURBATIONS = ("swap", "halfswap", "droptop")


def run_name(args):
    if args.name:
        return args.name
    base = "zs" if args.epochs == 0 else "lora"
    if args.no_claim:
        return "{}-noclaim-{}-m{}".format(base, args.ranker, args.m)
    return "{}-{}".format(base, "closed" if args.closed else "{}-m{}".format(args.ranker, args.m))


def to_three(prob, labels, n_classes):
    """Six PolitiFact classes -> three (pants-fire/false, barely-true/half-true, mostly-true/true)."""
    if n_classes == 6:
        return prob.reshape(len(prob), 3, 2).sum(-1), labels
    return prob, labels // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["RAWFC", "LIAR-RAW"])
    ap.add_argument("--protocol", default="strict", choices=PROTOCOLS)
    ap.add_argument("--ranker", default="bm25")
    ap.add_argument("--m", type=int, default=16, help="evidence sentences per query")
    ap.add_argument("--pool", type=int, default=48, help="first-stage candidates per claim that the method may read")
    ap.add_argument("--closed", action="store_true", help="closed-book: train and test without evidence")
    ap.add_argument("--no-claim", action="store_true", help="evidence-only control: the claim is withheld in training and test")
    # the components of the full method
    ap.add_argument("--ev-weight", type=float, default=0.0, help="weight of the evidence-tagging loss (0: no evidence head)")
    ap.add_argument("--train-sample", action="store_true", help="re-sample the training evidence from the pool in every epoch")
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
    ap.add_argument("--transfer", action="store_true", help="also evaluate on the test split of the other dataset")
    ap.add_argument("--perturb-n", type=int, default=0,
                    help="evaluate the perturbed-evidence conditions on a fixed random subset of N test claims (0: all)")
    ap.add_argument("--eval-only", action="store_true", help="load the saved adapter of this run and redo the evaluation")
    ap.add_argument("--train-limit", type=int, default=0, help="use only the first N training claims")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the size of every split, write nothing")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.select and not args.ev_weight > 0:
        ap.error("--select needs --ev-weight > 0")
    if args.no_claim and (args.closed or args.select):
        ap.error("--no-claim is a single-pass, evidence-only control")

    if args.micro_batch is None:
        args.micro_batch = 4 if args.closed else 2
    if args.accum is None:
        args.accum = max(1, 16 // args.micro_batch)
    name = run_name(args)
    tag = "{}.{}".format(args.dataset, args.protocol)
    seed_tag = "s0" if args.epochs == 0 else "s{}".format(args.seed)
    res_path = os.path.join(ROOT, "results", "ob", "runs", tag, name, seed_tag + ".json")
    pred_path = os.path.join(ROOT, "outputs", "ob", "runs", tag, name, seed_tag + ".npz")
    adapter_path = pred_path[:-4] + ".pt"
    if os.path.exists(res_path) and not (args.force or args.limit or args.eval_only):
        print("exists:", res_path)
        return
    if args.eval_only and not os.path.exists(adapter_path):
        sys.exit("no saved adapter: " + adapter_path)

    corpus = Corpus.load(os.path.join(ROOT, "outputs", "ob", args.dataset + ".json"))
    rank = {p: R.load_ranking(R.ranking_path(ROOT, args.dataset, p, args.ranker)) for p in PROTOCOLS}
    splits = {s: corpus.split(s) for s in ("train", "val", "test")}
    if args.train_limit:
        splits["train"] = splits["train"][:args.train_limit]
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    C = corpus.n_classes
    other = [p for p in PROTOCOLS if p != args.protocol][0]
    own_rank = rank[args.protocol]
    labels = lambda items: np.array([it["label"] for it in items])   # noqa: E731

    set_seed(args.seed)
    t0 = time.time()
    trained = args.epochs > 0
    v = Verifier(args.model, corpus.label_text, lora_rank=args.rank if trained else 0, max_tokens=args.max_tokens,
                 grad_checkpoint=not args.closed and not args.no_grad_ckpt, evidence_head=args.ev_weight > 0)

    def infer(items, ranking):
        """The inference procedure of this run for the claims ``items`` with the pools ``ranking``."""
        if args.closed:
            return P.single_pass(v, V.closed(items), C, args.eval_batch)
        if args.select:
            return P.read_select_verify(v, items, ranking, args.m, args.pool, C, args.eval_batch)
        ex = V.standard(items, ranking, args.m)
        return P.single_pass(v, V.withhold_claim(ex) if args.no_claim else ex, C, args.eval_batch)

    def val_score():
        return classification_metrics(labels(splits["val"]), infer(splits["val"], own_rank)["prob"])["macf1"]

    summary = {"best_epoch": -1, "epochs_run": 0, "history": [], "train_seconds": 0.0}
    if args.eval_only:
        v.load_state(torch.load(adapter_path, map_location="cpu"))
        with open(res_path, encoding="utf-8") as f:
            old = json.load(f)
        summary = dict(old["summary"], history=old["history"])
    elif trained:
        rng = np.random.RandomState(1000 + args.seed)

        def train_fn(epoch):
            if args.closed:
                return V.closed(splits["train"])
            ex = V.training(splits["train"], own_rank, args.m, rng, top=args.pool if args.train_sample else 0,
                            swap_rate=args.swap_rate)
            return V.withhold_claim(ex) if args.no_claim else ex

        summary = fit(v, train_fn, val_score, epochs=args.epochs, lr=args.lr, micro_batch=args.micro_batch,
                      accum=args.accum, seed=args.seed, lambda_ev=args.ev_weight, log=lambda s: print(s, flush=True))
        if not args.limit:
            os.makedirs(os.path.dirname(adapter_path), exist_ok=True)
            torch.save({k: (x.half() if x.is_floating_point() else x) for k, x in v.state().items()}, adapter_path)

    # ------------------------------------------------------------------------------ evaluation
    test, val = splits["test"], splits["val"]
    out, ys, ids, extra, selected = {}, {}, {}, {}, {}

    def put(key, prob, items):
        out[key], ys[key], ids[key] = prob, labels(items), np.array([it["id"] for it in items])

    main_val, main_test = infer(val, own_rank), infer(test, own_rank)
    put("val", main_val["prob"], val)
    put("test", main_test["prob"], test)
    cost = {"tokens": main_test["tokens"], "queries": main_test.get("queries", 1.0)}
    if not args.closed:
        put("test@" + other, infer(test, rank[other])["prob"], test)
    if not args.closed and not args.no_claim:
        put("test@closed", P.single_pass(v, V.closed(test), C, args.eval_batch)["prob"], test)
        if args.select:
            selected = {"test": main_test["selected"], "val": main_val["selected"]}
            put("test@read", main_test["prob_read"], test)
            put("val@read", main_val["prob_read"], val)
            put("test@avg", 0.5 * (main_test["prob"] + main_test["prob_read"]), test)
            put("val@avg", 0.5 * (main_val["prob"] + main_val["prob_read"]), val)
            put("test@bm25", P.single_pass(v, V.standard(test, own_rank, args.m), C, args.eval_batch)["prob"], test)
            put("val@bm25", P.single_pass(v, V.standard(val, own_rank, args.m), C, args.eval_batch)["prob"], val)
        else:
            put("test@read", P.read_ensemble(v, test, own_rank, args.m, args.pool, C, args.eval_batch), test)
            put("val@read", P.read_ensemble(v, val, own_rank, args.m, args.pool, C, args.eval_batch), val)
        put("test@oracle", P.single_pass(v, V.oracle(test, own_rank, args.m, args.pool), C, args.eval_batch)["prob"], test)
        sub = test
        if args.perturb_n and args.perturb_n < len(test):
            keep = sorted(np.random.RandomState(0).permutation(len(test))[:args.perturb_n])
            sub = [test[i] for i in keep]
        perturbed = {}
        for kind in PERTURBATIONS:
            items_p, rank_p = V.perturb(sub, own_rank, {"halfswap": "mix"}.get(kind, kind), args.m, args.pool, seed=0)
            res = infer(items_p, rank_p)
            put("test@" + kind, res["prob"], sub)
            perturbed[kind] = (items_p, res)
            pred = res["prob"].argmax(1)
            extra[kind + "_pred_share"] = (np.bincount(pred, minlength=C) / max(len(pred), 1)).tolist()
            extra[kind + "_conf"] = float(res["prob"].max(1).mean())
        extra["own_conf"] = float(main_test["prob"].max(1).mean())
        # under fully mismatched evidence: does the verdict follow the claim the evidence belongs to?
        source = labels(sub)[V.derangement(len(sub), 0)]
        extra["swap_acc_source_label"] = float((perturbed["swap"][1]["prob"].argmax(1) == source).mean())
        if args.ev_weight > 0:
            extra.update(P.evidence_diagnostics(test, own_rank, args.m, main_test, perturbed, args.select))
    if args.transfer:
        other_ds = "LIAR-RAW" if args.dataset == "RAWFC" else "RAWFC"
        corpus2 = Corpus.load(os.path.join(ROOT, "outputs", "ob", other_ds + ".json"))
        test2 = corpus2.split("test")[:args.limit or None]
        res = infer(test2, R.load_ranking(R.ranking_path(ROOT, other_ds, args.protocol, args.ranker)))
        out["xfer"], ys["xfer"] = to_three(res["prob"], labels(test2), C)
        ids["xfer"] = np.array([it["id"] for it in test2])

    metrics = {k: classification_metrics(ys[k], p) for k, p in out.items()}
    result = {
        "dataset": corpus.name, "protocol": args.protocol, "method": name, "name": name, "seed": args.seed,
        "encoder": os.path.basename(args.model.rstrip("/")),
        "config": {"ranker": args.ranker, "m": args.m, "pool": args.pool, "closed": args.closed, "no_claim": args.no_claim,
                   "rank": args.rank, "lr": args.lr, "epochs": args.epochs, "batch_size": args.micro_batch * args.accum,
                   "max_tokens": args.max_tokens, "ev_weight": args.ev_weight, "train_sample": args.train_sample,
                   "swap_rate": args.swap_rate, "select": args.select, "train_limit": args.train_limit,
                   "perturb_n": args.perturb_n,
                   "grad_checkpoint": not args.closed and not args.no_grad_ckpt},
        "summary": {k: summary[k] for k in ("best_epoch", "epochs_run", "train_seconds")}, "history": summary["history"],
        "metrics": metrics, "extra": extra, "cost": cost, "n_trainable": int(sum(p.numel() for p in v.params)),
        "n_items": {s: len(splits[s]) for s in splits}, "n_eval": {k: int(len(x)) for k, x in ys.items()},
        "seconds": time.time() - t0, "max_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
    }
    print(name, seed_tag, {k: round(m["macf1"], 4) for k, m in metrics.items()}, flush=True)
    print("   extra", {k: (round(x, 3) if isinstance(x, float) else [round(z, 2) for z in x]) for k, x in extra.items()},
          "max mem {:.1f} GB".format(result["max_memory_gb"]), flush=True)
    if args.limit:
        return
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    flat = {}
    for k, p in out.items():
        flat[k + "/prob"] = p.astype(np.float32)
        flat[k + "/label"] = ys[k]
        flat[k + "/id"] = ids[k]
    np.savez_compressed(pred_path, **flat)
    if selected:
        with open(pred_path[:-4] + ".selected.json", "w", encoding="utf-8") as f:
            json.dump({sp: {it["id"]: c for it, c in zip(splits[sp], sel)} for sp, sel in selected.items()}, f)


if __name__ == "__main__":
    main()
