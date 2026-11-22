#!/usr/bin/env python
"""Offline batched LLM generation with vLLM (run inside the ``vllm`` conda env).

One process loads the model once and runs any number of (input file, task) jobs. Every job
writes an append-only JSONL cache keyed by item id, so interrupted runs resume where they stopped.

Example
-------
python scripts/llm_generate.py --model /media/dell/DATA/models/Qwen2.5-7B-Instruct \
    --out-dir outputs/llm/qwen25 --tasks clara td cs tc \
    --spec ARG-CN.train=data/ARG-CN/train.json:zh --spec ARG-CN.val=data/ARG-CN/val.json:zh
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clara.llm import tasks as T  # noqa: E402


def load_items(path):
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    if isinstance(data, dict) and "items" in data:      # a file written by build_dataset.py
        data = data["items"]
    items = []
    for i, d in enumerate(data):
        d = dict(d)
        d["id"] = d.get("source_id", d.get("id", i))
        items.append(d)
    return items


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    continue
    return ids


def to_tokens(output):
    """vLLM CompletionOutput -> [(decoded_token, {alt_decoded: logprob})]."""
    toks = []
    if not output.logprobs:
        return toks
    for tid, lp in zip(output.token_ids, output.logprobs):
        cur = lp.get(tid)
        dec = cur.decoded_token if cur is not None and cur.decoded_token is not None else ""
        alts = {}
        for v in lp.values():
            if v.decoded_token is None:
                continue
            # two token ids can decode to the same string; keep the total mass
            alts[v.decoded_token] = max(alts.get(v.decoded_token, -1e9), v.logprob)
        toks.append((dec, alts))
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", nargs="+", required=True, choices=sorted(T.TASKS))
    ap.add_argument("--spec", action="append", required=True,
                    help="NAME=PATH:LANG, e.g. ARG-CN.train=data/ARG-CN/train.json:zh")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items of every input")
    ap.add_argument("--split", default=None, help="keep only items of this split (dataset files)")
    ap.add_argument("--chunk", type=int, default=768)
    ap.add_argument("--max-claims", type=int, default=6)
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-content-chars", type=int, default=4000)
    ap.add_argument("--no-maxlen", action="store_true", help="drop maxLength from the JSON schema")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-think", action="store_true",
                    help="pass enable_thinking=False to the chat template (Qwen3)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    os.makedirs(args.out_dir, exist_ok=True)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_model_len, seed=args.seed, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    tmpl_kw = {"enable_thinking": False} if args.no_think else {}

    def render(b):
        msgs = [{"role": "system", "content": b["system"]}, {"role": "user", "content": b["user"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tmpl_kw)
        return text + b["prefix"]

    def sampling(task, b, max_tokens=None, temperature=None):
        mt, temp, need_lp = T.TASKS[task]
        kw = dict(temperature=temp if temperature is None else temperature,
                  max_tokens=max_tokens or mt, seed=args.seed)
        if need_lp:
            kw["logprobs"] = 12
        if b["schema"] is not None:
            kw["structured_outputs"] = StructuredOutputsParams(json=b["schema"])
        return SamplingParams(**kw)

    def run(task, batch_items, lang, **build_kw):
        if task in ("style", "tamper"):
            builds = [T.build(task, it["content"][:args.max_content_chars], lang, it["id"], item=it)
                      for it in batch_items]
        else:
            builds = [T.build(task, it["content"][:args.max_content_chars], lang, it["id"], **build_kw)
                      for it in batch_items]
        prompts = [render(b) for b in builds]
        sps = [sampling(task, b) for b in builds]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        return builds, outs

    for spec in args.spec:
        name, rest = spec.split("=", 1)
        path, lang = rest.rsplit(":", 1)
        items = load_items(path)
        if args.split:
            items = [it for it in items if it.get("split") == args.split]
        if args.limit:
            items = items[:args.limit]
        for task in args.tasks:
            out_path = os.path.join(args.out_dir, "{}.{}.jsonl".format(name, task))
            have = done_ids(out_path)
            todo = [it for it in items if it["id"] not in have]
            print("[{}] {}: {} items, {} cached, {} to run".format(
                task, name, len(items), len(have), len(todo)), flush=True)
            t0, n_out_tok, n_fail = time.time(), 0, 0
            with open(out_path, "a", encoding="utf-8") as fout:
                for s in range(0, len(todo), args.chunk):
                    batch = todo[s:s + args.chunk]
                    kw = {"max_claims": args.max_claims, "use_maxlen": not args.no_maxlen} \
                        if task == "clara" else {}
                    builds, outs = run(task, batch, lang, **kw)
                    records = []
                    retry = []
                    for it, b, o in zip(batch, builds, outs):
                        c = o.outputs[0]
                        rec = {"id": it["id"], "task": task, "raw": c.text,
                               "finish": c.finish_reason,
                               "prompt_tokens": len(o.prompt_token_ids),
                               "completion_tokens": len(c.token_ids), "attempt": 1}
                        n_out_tok += len(c.token_ids)
                        if task == "clara":
                            dists = T.suspicion_distributions(to_tokens(c))
                            rec["parsed"] = T.parse_clara(c.text, b["meta"], dists)
                            rec["order"] = b["meta"]["order"]
                            rec["n_sents"] = len(b["meta"]["sents"])
                            if rec["parsed"] is None:
                                retry.append(len(records))
                        elif task in ("td", "cs", "cot"):
                            rec["pred"] = T.parse_binary(c.text)
                            rec["rationale"] = T.clean_rationale(c.text, b["prefix"])
                        elif task in ("style", "tamper"):
                            rec["text"] = T.clean_generation(c.text)
                            rec.update(b["meta"])
                        elif task == "zs":
                            rec["pred"] = T.parse_binary(c.text)
                            rec["p_fake"] = T.binary_soft(to_tokens(c))
                        records.append(rec)
                    if retry:
                        # second and last attempt: fewer claims, no length caps, more room
                        sub = [batch[i] for i in retry]
                        b2 = [T.build("clara", it["content"][:args.max_content_chars], lang, it["id"],
                                      max_claims=3, use_maxlen=False) for it in sub]
                        sp2 = [sampling("clara", b, max_tokens=1600) for b in b2]
                        o2 = llm.generate([render(b) for b in b2], sp2, use_tqdm=False)
                        for i, b, o in zip(retry, b2, o2):
                            c = o.outputs[0]
                            dists = T.suspicion_distributions(to_tokens(c))
                            rec = records[i]
                            rec.update(raw=c.text, finish=c.finish_reason, attempt=2,
                                       completion_tokens=rec["completion_tokens"] + len(c.token_ids),
                                       prompt_tokens=rec["prompt_tokens"] + len(o.prompt_token_ids),
                                       parsed=T.parse_clara(c.text, b["meta"], dists),
                                       order=b["meta"]["order"])
                            n_out_tok += len(c.token_ids)
                            if rec["parsed"] is None:
                                n_fail += 1
                    for rec in records:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    el = time.time() - t0
                    print("  {}/{}  {:.0f}s  {:.0f} out-tok/s  fail={}".format(
                        min(s + args.chunk, len(todo)), len(todo), el, n_out_tok / max(el, 1e-6), n_fail),
                        flush=True)


if __name__ == "__main__":
    main()
