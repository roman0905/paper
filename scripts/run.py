#!/usr/bin/env python
"""Run a sweep of experiments on one dataset, skipping runs whose result file already exists.

    python scripts/run.py --dataset outputs/datasets/ARG-CN.qwen25.json --sweep main --seeds 1 2 3 4 5
    python scripts/run.py --dataset ... --sweep ablation --lr 5e-5 --seeds 1 2 3 4 5

Results:      results/runs/<dataset-tag>/<experiment>/s<seed>.json
Predictions:  outputs/runs/<dataset-tag>/<experiment>/s<seed>.npz
"""
import argparse
import gc
import inspect
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from clara.data import Dataset  # noqa: E402
from clara.experiment import Session, write_result  # noqa: E402
from configs import sweeps  # noqa: E402


def perturb_evals(ds, exp):
    """Controlled test sets that exist for this dataset/LLM (see scripts/make_perturb.py)."""
    if (exp.get("model") or {}).get("rationale_source") == "gpt":
        return []                      # no GPT-3.5 rationales exist for perturbed texts
    out = []
    kinds = ["style", "tamper"] + (["occl"] if exp["method"] == "bert" else [])
    for kind in kinds:
        llm = "none" if kind == "occl" else ds.llm
        path = os.path.join(ROOT, "outputs", "datasets", "{}-{}.{}.json".format(ds.name, kind, llm))
        if os.path.exists(path):
            out.append({"name": kind, "path": path})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--lr", type=float, default=None, help="for sweeps that take a fixed learning rate")
    ap.add_argument("--only", nargs="*", default=None, help="run only experiments with these names")
    ap.add_argument("--tag", default=None, help="result sub-directory (default: dataset file name)")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-preds", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="recompute the frozen encoder layers every step instead of caching them")
    ap.add_argument("--amp", action="store_true", help="mixed-precision training of the top layers")
    ap.add_argument("--perturb", action="store_true", help="also evaluate on the style / tamper sets")
    ap.add_argument("--holdout", default=None,
                    help="leave-one-domain-out: train/val on the other domains, test on this one")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override model hyper-parameters of every experiment (JSON values)")
    args = ap.parse_args()

    ds = Dataset.load(args.dataset)
    tag = args.tag or os.path.basename(args.dataset)[:-len(".json")]
    if args.holdout:
        d = args.holdout
        ds = ds.resplit(lambda it: None if "domain" not in it else
                        (it["split"] if (it["split"] == "test") == (it["domain"] == d) else None))
        tag = "{}-holdout-{}".format(tag, d)
        print({k: len(v) for k, v in ds.by_split.items()}, flush=True)
    fn = getattr(sweeps, args.sweep)
    kw = {"lr": args.lr} if "lr" in inspect.signature(fn).parameters else {}
    if "lr" in kw and args.lr is None:
        ap.error("sweep '{}' needs --lr".format(args.sweep))
    exps = fn(ds.lang, **kw)
    if args.only:
        exps = [e for e in exps if e["name"] in args.only]
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except ValueError:
            overrides[k] = v
    for e in exps:
        if overrides:
            e["model"] = dict(e.get("model") or {}, **overrides)
        if args.amp:
            e["train"] = dict(e.get("train") or {}, amp=True)
            e["teacher_train"] = dict(e.get("teacher_train") or {}, amp=True)
        if args.perturb:
            e["evals"] = list(e.get("evals") or []) + perturb_evals(ds, e)

    todo = []
    for e in exps:
        for seed in args.seeds:
            path = os.path.join(ROOT, "results", "runs", tag, e["name"], "s{}.json".format(seed))
            if args.force or not os.path.exists(path):
                todo.append((e, seed, path))
    print("{}: {} experiments x {} seeds, {} runs to do".format(tag, len(exps), len(args.seeds), len(todo)), flush=True)

    session, group = None, None
    t_all = time.time()
    for i, (e, seed, path) in enumerate(sorted(todo, key=lambda t: t[0]["group"])):
        if e["group"] != group:
            session = None
            gc.collect()
            torch.cuda.empty_cache()
            session = Session(ds, encoder_path=args.encoder, log=lambda s: None, live=args.live)
            group = e["group"]
        pred = None if args.no_preds else os.path.join(ROOT, "outputs", "runs", tag, e["name"], "s{}.npz".format(seed))
        res = session.run(e["method"], seed, model=e.get("model"), train=e.get("train"),
                          teacher_model=e.get("teacher_model"), teacher_train=e.get("teacher_train"),
                          evals=e.get("evals"), pred_path=pred)
        res["name"] = e["name"]
        write_result(path, res)
        m = res["metrics"]
        print("[{}/{}] {:28s} s{}  val {:.4f}  test macF1 {:.4f} acc {:.4f} auc {:.4f}  ep {}  {:.0f}s  (cache {:.1f} GB, total {:.0f} min)".format(
            i + 1, len(todo), e["name"], seed, m["val"]["macf1"], m["test"]["macf1"], m["test"]["acc"],
            m["test"]["auc"], res["summary"]["best_epoch"], res["seconds"], session.store.memory_gb(),
            (time.time() - t_all) / 60), flush=True)


if __name__ == "__main__":
    main()
