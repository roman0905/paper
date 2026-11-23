#!/usr/bin/env python
"""Parse the test metrics of runs of the OFFICIAL ARG code into results/official_arg.json.

The official code (github.com/ICTMCG/ARG) was run unchanged except for library compatibility
(pandas `_append`, unused nltk/jieba imports, a no-op tensorboardX stand-in) with its published
Chinese configuration (run_zh.sh) and the GPT-3.5 rationales shipped with the data; see
outputs/official_arg/run_check.sh.  Only aggregate test metrics are stored.

    python scripts/collect_official_arg.py outputs/official_arg
"""
import ast
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    src = sys.argv[1]
    runs = {}
    for path in sorted(glob.glob(os.path.join(src, "run_s*.log"))):
        seed = int(re.search(r"run_s(\d+)\.log", path).group(1))
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = [ln for ln in f if ln.startswith("test results:")]
        if not lines:
            continue
        d = ast.literal_eval(lines[-1].split("test results:", 1)[1].strip())
        runs[str(seed)] = {"macf1": d["metric"], "acc": d["acc"], "auc": d["auc"],
                           "f1_real": d["f1_real"], "f1_fake": d["f1_fake"]}
    out = {"dataset": "ARG-CN", "code": "https://github.com/ICTMCG/ARG", "config": "run_zh.sh (lr 2e-5, weights 2.2 / 1.8)",
           "rationales": "GPT-3.5 (released with the data)", "runs": runs}
    path = os.path.join(ROOT, "results", "official_arg.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(path, {k: round(v["macf1"], 4) for k, v in runs.items()})


if __name__ == "__main__":
    main()
