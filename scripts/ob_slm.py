#!/usr/bin/env python
"""Small-model baseline: a fine-tuned encoder classifier that reads the claim (and its evidence).

    python scripts/ob_slm.py --dataset RAWFC --seed 1                 # claim + top-m BM25 evidence
    python scripts/ob_slm.py --dataset RAWFC --seed 1 --closed        # claim only

The encoder (RoBERTa-large by default) is fine-tuned in full; the input is the claim followed by
the evidence sentences in ranked order, cut at the encoder's length limit.  Model selection and
the result files follow scripts/ob_verifier.py.
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
import torch.nn.functional as F  # noqa: E402

from clara.train import set_seed  # noqa: E402
from openbook import retrieval as R  # noqa: E402
from openbook import views as V  # noqa: E402
from openbook.data import PROTOCOLS, Corpus  # noqa: E402
from openbook.metrics import classification_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["RAWFC", "LIAR-RAW"])
    ap.add_argument("--protocol", default="strict", choices=PROTOCOLS)
    ap.add_argument("--ranker", default="bm25")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--closed", action="store_true")
    ap.add_argument("--model", default="/media/dell/DATA/models/roberta-large")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    short = os.path.basename(args.model.rstrip("/"))
    name = args.name or "{}-{}".format(short, "closed" if args.closed else "{}-m{}".format(args.ranker, args.m))
    tag = "{}.{}".format(args.dataset, args.protocol)
    res_path = os.path.join(ROOT, "results", "ob", "runs", tag, name, "s{}.json".format(args.seed))
    if os.path.exists(res_path) and not args.force and not args.limit:
        print("exists:", res_path)
        return

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    corpus = Corpus.load(os.path.join(ROOT, "outputs", "ob", args.dataset + ".json"))
    rank = {p: R.load_ranking(R.ranking_path(ROOT, args.dataset, p, args.ranker)) for p in PROTOCOLS}
    splits = {s: corpus.split(s) for s in ("train", "val", "test")}
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    C = corpus.n_classes
    other = [p for p in PROTOCOLS if p != args.protocol][0]

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=C).cuda()

    def encode(examples):
        claims = [ex["claim"] for ex in examples]
        if examples and examples[0]["evidence"] is None:
            return tok(claims, truncation=True, max_length=args.max_len)
        ev = [" ".join(s for _, s in ex["evidence"]) or "no evidence" for ex in examples]
        return tok(claims, ev, truncation="only_second", max_length=args.max_len)

    def view(split, protocol=args.protocol):
        return V.closed(splits[split]) if args.closed else V.standard(splits[split], rank[protocol], args.m)

    conditions = {"val": view("val"), "test": view("test")}
    if not args.closed:
        conditions["test@" + other] = view("test", other)
        conditions["test@swap"] = V.swapped(splits["test"], rank[args.protocol], args.m, seed=0)
    enc = {k: encode(v) for k, v in conditions.items()}
    y = {k: np.array([ex["label"] for ex in v]) for k, v in conditions.items()}
    train = view("train")
    enc_train, y_train = encode(train), torch.tensor([ex["label"] for ex in train])

    def batch(e, idx):
        b = tok.pad({"input_ids": [e["input_ids"][i] for i in idx], "attention_mask": [e["attention_mask"][i] for i in idx]},
                    return_tensors="pt")
        return {k: v.cuda() for k, v in b.items()}

    @torch.no_grad()
    def predict(e):
        model.eval()
        n = len(e["input_ids"])
        order = np.argsort([len(x) for x in e["input_ids"]])
        prob = np.zeros((n, C), dtype=np.float32)
        for s in range(0, n, 64):
            idx = order[s:s + 64]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**batch(e, idx)).logits
            prob[idx] = torch.softmax(lg.float(), -1).cpu().numpy()
        return prob

    n = len(train)
    steps = int(np.ceil(n / args.batch)) * args.epochs
    warm = max(1, int(0.06 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / warm if st < warm else max(0.0, (steps - st) / max(1, steps - warm)))
    scaler = torch.cuda.amp.GradScaler()
    rng = np.random.RandomState(args.seed)
    best, best_state, best_epoch, history = -1.0, None, -1, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        run, nb = 0.0, 0
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(**batch(enc_train, idx)).logits
            loss = F.cross_entropy(lg.float(), y_train[idx].cuda())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
            nb += 1
        score = classification_metrics(y["val"], predict(enc["val"]))["macf1"]
        history.append({"epoch": epoch, "val": score, "loss": run / max(nb, 1)})
        print("epoch {} loss {:.4f} val {:.4f} ({:.0f}s)".format(epoch, run / max(nb, 1), score, time.time() - t0), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    probs = {k: predict(e) for k, e in enc.items()}
    metrics = {k: classification_metrics(y[k], p) for k, p in probs.items()}
    result = {
        "dataset": corpus.name, "protocol": args.protocol, "method": name, "name": name, "seed": args.seed, "encoder": short,
        "config": {"ranker": args.ranker, "m": args.m, "closed": args.closed, "lr": args.lr, "epochs": args.epochs,
                   "batch_size": args.batch, "max_len": args.max_len},
        "summary": {"best_epoch": best_epoch, "epochs_run": args.epochs, "train_seconds": train_seconds}, "history": history,
        "metrics": metrics, "extra": {}, "n_trainable": int(sum(p.numel() for p in model.parameters())),
        "n_items": {s: len(splits[s]) for s in splits}, "seconds": time.time() - t0,
    }
    print(name, "s{}".format(args.seed), {k: round(m["macf1"], 4) for k, m in metrics.items()}, flush=True)
    if args.limit:
        return
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, sort_keys=True)
    pred_path = os.path.join(ROOT, "outputs", "ob", "runs", tag, name, "s{}.npz".format(args.seed))
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    flat = {}
    for k, p in probs.items():
        flat[k + "/prob"] = p
        flat[k + "/label"] = y[k]
        flat[k + "/id"] = np.array([ex["id"] for ex in conditions[k]])
    np.savez_compressed(pred_path, **flat)


if __name__ == "__main__":
    main()
