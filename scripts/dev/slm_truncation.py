#!/usr/bin/env python
"""How much of the evidence does the encoder baseline actually read?

RoBERTa-large takes at most 512 tokens.  For the test claims of both datasets (strict protocol,
first-stage top-16 sentences, the input format of scripts/ob_slm.py) this reports the mean input
length before truncation, the share of inputs that are truncated and the mean share of a claim's
evidence sentences that fit completely.  Tokenizer only, no model, no labels.

    python scripts/dev/slm_truncation.py --model /path/to/roberta-large   ->  results/ob/slm_truncation.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from openbook import retrieval as R  # noqa: E402
from openbook import views as V  # noqa: E402
from openbook.data import Corpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SLM_PATH", "roberta-large"))
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    out = {"max_len": args.max_len, "m": args.m, "datasets": {}}
    for D in ("RAWFC", "LIAR-RAW"):
        corpus = Corpus.load(os.path.join(ROOT, "outputs", "ob", D + ".json"))
        rank = R.load_ranking(R.ranking_path(ROOT, D, "strict", "bm25"))
        examples = V.standard(corpus.split("test"), rank, args.m)
        lengths, fit = [], []
        for ex in examples:
            ev = " ".join(s for _, s in ex["evidence"]) or "no evidence"
            lengths.append(len(tok(ex["claim"], ev, truncation=False)["input_ids"]))
            used = len(tok(ex["claim"])["input_ids"]) + 2          # <s> claim </s></s> evidence </s>
            k = 0
            for _, s in ex["evidence"]:
                used += len(tok(" " + s, add_special_tokens=False)["input_ids"])
                if used > args.max_len:
                    break
                k += 1
            if ex["evidence"]:
                fit.append(k / len(ex["evidence"]))
        lengths = np.array(lengths)
        out["datasets"][D] = {"claims": len(examples), "mean_tokens": float(lengths.mean()),
                              "truncated_share": float((lengths > args.max_len).mean()),
                              "sentences_fit_share": float(np.mean(fit))}
        print(D, out["datasets"][D])
    path = os.path.join(ROOT, "results", "ob", "slm_truncation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
