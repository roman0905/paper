#!/usr/bin/env python
"""Is the explanation target part of the final method?  A validation-only rule, fixed in code
before any run with the explanation target existed.

The verifier exists in two variants that differ only in the explanation target (runs ``sieve`` and
``sieve-j``).  On each dataset the validation macro-F1 of the selected checkpoints is averaged over
the seeds that both variants have; the explanation target is adopted iff the difference
(sieve-j minus sieve), averaged over the two datasets with weights proportional to the number of
validation claims, is positive.  Test results are never read.

A decision needs both datasets with at least ``MIN_SEEDS`` common seeds each; otherwise nothing is
written and the exit status is 2.  The decision is written once to results/ob/decision.json and
only read afterwards.

    python scripts/ob_decide.py            # decide if undecided; print the name of the final run
    python scripts/ob_decide.py --dry      # show the current evidence, write nothing
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from openbook import report as RP  # noqa: E402

PATH = os.path.join(ROOT, "results", "ob", "decision.json")
BASE, WITH = "sieve", "sieve-j"
DATASETS = ("RAWFC", "LIAR-RAW")
MIN_SEEDS = 3


def evidence():
    out = {"val_delta": {}, "val_macf1": {}, "seeds": {}, "n_val": {}}
    for D in DATASETS:
        runs = RP.load_runs(D + ".strict")
        common = sorted(set(runs.get(BASE, {})) & set(runs.get(WITH, {})))
        if not common:
            continue
        a = [runs[WITH][s]["metrics"]["val"]["macf1"] for s in common]
        b = [runs[BASE][s]["metrics"]["val"]["macf1"] for s in common]
        out["val_delta"][D] = float(np.mean(a) - np.mean(b))
        out["val_macf1"][D] = {WITH: float(np.mean(a)), BASE: float(np.mean(b))}
        out["seeds"][D] = common
        out["n_val"][D] = int(runs[BASE][common[0]]["n_items"]["val"])
    return out


def decide(ev):
    """None if the evidence is incomplete, else the decision record."""
    if any(len(ev["seeds"].get(D, [])) < MIN_SEEDS for D in DATASETS):
        return None
    w = np.array([ev["n_val"][D] for D in DATASETS], dtype=float)
    d = np.array([ev["val_delta"][D] for D in DATASETS])
    score = float((w * d).sum() / w.sum())
    return dict(ev, final=WITH if score > 0 else BASE, score=score,
                rule="adopt the explanation target iff the validation macro-F1 difference (sieve-j minus sieve; selected "
                     "checkpoints; seeds common to both), averaged over the two datasets with weights proportional to "
                     "their number of validation claims, is positive",
                decided_at=time.strftime("%Y-%m-%d %H:%M:%S"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="show the current evidence and what it would decide, write nothing")
    args = ap.parse_args()
    if os.path.exists(PATH) and not args.dry:
        with open(PATH, encoding="utf-8") as f:
            print(json.load(f)["final"])
        return
    ev = evidence()
    d = decide(ev)
    if args.dry:
        print(json.dumps(d if d is not None else dict(ev, final=None), indent=1))
        return
    if d is None:
        sys.stderr.write("{} no decision: need runs of '{}' and '{}' with >= {} common seeds on both datasets, have {}\n".format(
            time.strftime("%Y-%m-%d %H:%M:%S"), BASE, WITH, MIN_SEEDS, {D: ev["seeds"].get(D, []) for D in DATASETS}))
        sys.exit(2)
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, sort_keys=True)
    print(d["final"])


if __name__ == "__main__":
    main()
