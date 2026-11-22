#!/usr/bin/env python
"""LoRA fine-tuning of the 7B LLM as a direct fake news classifier (strong text-only baseline).

The model reads exactly the zero-shot prompt of the ``zs`` task and is trained with a two-way
cross-entropy over the next-token logits of "0" (fake) and "1" (real), so the only difference
from the zero-shot LLM baseline is the fine-tuning.  Model selection follows the protocol of all
other methods: validation macro-F1 at threshold 0.5, evaluated after every epoch.

    python scripts/lora_baseline.py --dataset outputs/datasets/ARG-CN.qwen25.json --seed 1 --lr 1e-4

Writes results/runs/<tag>/lora@lr<lr>/s<seed>.json and the matching predictions .npz.
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from clara.data import SPLITS, Dataset  # noqa: E402
from clara.experiment import write_result  # noqa: E402
from clara.llm import tasks as T  # noqa: E402
from clara.metrics import detection_metrics, ece  # noqa: E402
from clara.train import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="/media/dell/DATA/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=24)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--eval", action="append", default=[], help="NAME=DATASET_JSON (its test split)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the size of every split")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds = Dataset.load(args.dataset)
    tag = args.tag or os.path.basename(args.dataset)[:-len(".json")]
    name = "lora@lr{:g}".format(args.lr)
    res_path = os.path.join(ROOT, "results", "runs", tag, name, "s{}.json".format(args.seed))
    if os.path.exists(res_path) and not args.force:
        print("exists:", res_path)
        return

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    backbone = model.get_base_model()
    label_ids = [tok.encode("0", add_special_tokens=False), tok.encode("1", add_special_tokens=False)]
    assert all(len(x) == 1 for x in label_ids), label_ids
    head = backbone.lm_head.weight[[label_ids[0][0], label_ids[1][0]]].detach().float()   # [2, H]
    params = [p for p in model.parameters() if p.requires_grad]

    def render(content):
        b = T.build("zs", content[:args.max_chars], ds.lang, 0)
        msgs = [{"role": "system", "content": b["system"]}, {"role": "user", "content": b["user"]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    splits = {s: ds.split(s) for s in SPLITS}
    for spec in args.eval:
        n, path = spec.split("=", 1)
        splits[n] = Dataset.load(path).split("test")
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    enc = {}
    for s, items in splits.items():
        e = tok([render(it["content"]) for it in items], add_special_tokens=False)
        enc[s] = {"ids": e["input_ids"], "y": np.array([it["label"] for it in items])}
    print({s: (len(v["y"]), int(np.mean([len(x) for x in v["ids"]]))) for s, v in enc.items()}, flush=True)

    def batch_tensors(ids_list):
        L = max(len(x) for x in ids_list)
        ids = torch.full((len(ids_list), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(ids_list), L), dtype=torch.long)
        for i, x in enumerate(ids_list):
            ids[i, L - len(x):] = torch.tensor(x)
            att[i, L - len(x):] = 1
        return ids.cuda(), att.cuda()

    def logits2(ids_list):
        ids, att = batch_tensors(ids_list)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = backbone.model(input_ids=ids, attention_mask=att).last_hidden_state[:, -1, :]
        return h.float() @ head.T                      # column 0 = "0" (fake), column 1 = "1" (real)

    @torch.no_grad()
    def predict(split):
        model.eval()
        ids = enc[split]["ids"]
        order = np.argsort([len(x) for x in ids])
        prob = np.zeros(len(ids), dtype=np.float32)
        for s in range(0, len(order), args.eval_batch):
            idx = order[s:s + args.eval_batch]
            lg = logits2([ids[i] for i in idx])
            prob[idx] = torch.softmax(lg, -1)[:, 0].cpu().numpy()   # P(fake)
        return prob

    def evaluate(split, prob):
        m = detection_metrics(enc[split]["y"], prob)
        m["ece"] = ece(enc[split]["y"], prob)
        return m

    n = len(enc["train"]["y"])
    steps_per_epoch = math.ceil(n / (args.micro_batch * args.accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.06 * total_steps))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / warmup if st < warmup else max(0.0, (total_steps - st) / max(1, total_steps - warmup)))
    rng = np.random.RandomState(args.seed)
    y_train = torch.tensor(1 - enc["train"]["y"])          # class index: 0 = fake, 1 = real
    best, best_state, best_epoch, history = -1.0, None, -1, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        run_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for b, s in enumerate(range(0, n, args.micro_batch)):
            idx = perm[s:s + args.micro_batch]
            lg = logits2([enc["train"]["ids"][i] for i in idx])
            loss = F.cross_entropy(lg, y_train[idx].cuda())
            (loss / args.accum).backward()
            run_loss += float(loss)
            nb += 1
            if (b + 1) % args.accum == 0 or s + args.micro_batch >= n:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if nb % 200 == 0:
                print("  ep {} step {}/{} loss {:.4f} {:.0f}s".format(
                    epoch, nb, math.ceil(n / args.micro_batch), run_loss / nb, time.time() - t0), flush=True)
        val = evaluate("val", predict("val"))
        history.append({"epoch": epoch, "val_macf1": val["macf1"], "cls": run_loss / max(nb, 1)})
        print("epoch {} loss {:.4f} val macF1 {:.4f} acc {:.4f} ({:.0f}s)".format(
            epoch, run_loss / max(nb, 1), val["macf1"], val["acc"], time.time() - t0), flush=True)
        if val["macf1"] > best:
            best, best_epoch = val["macf1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    train_seconds = time.time() - t0
    with torch.no_grad():
        for k, v in model.named_parameters():
            if v.requires_grad:
                v.copy_(best_state[k].to(v.device))

    preds = {s: predict(s) for s in enc if s != "train"}
    result = {
        "dataset": ds.name, "llm": ds.llm, "method": "lora", "name": name, "seed": args.seed,
        "encoder": os.path.basename(args.model.rstrip("/")),
        "config": {"model": {"rank": args.rank, "alpha": 2 * args.rank, "targets": "attn+mlp"},
                   "train": {"lr": args.lr, "epochs": args.epochs, "batch_size": args.micro_batch * args.accum,
                             "seed": args.seed}},
        "summary": {"best_epoch": best_epoch, "epochs_run": args.epochs, "best_val_macf1": best,
                    "train_seconds": train_seconds},
        "history": history,
        "metrics": {s: evaluate(s, p) for s, p in preds.items()},
        "n_trainable": int(sum(p.numel() for p in params)),
        "n_items": {s: len(v["y"]) for s, v in enc.items()},
        "seconds": time.time() - t0,
    }
    write_result(res_path, result)
    flat = {}
    for s, p in preds.items():
        flat[s + "/prob"] = p
        flat[s + "/label"] = enc[s]["y"]
        flat[s + "/id"] = np.array([it["id"] for it in splits[s]])
    pred_path = os.path.join(ROOT, "outputs", "runs", tag, name, "s{}.npz".format(args.seed))
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    np.savez_compressed(pred_path, **flat)
    print(name, "seed", args.seed, {s: round(m["macf1"], 4) for s, m in result["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
