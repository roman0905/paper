#!/usr/bin/env python
"""Print mean ± std tables of finished runs.

    python scripts/aggregate.py ARG-CN.qwen25 [--all] [--split test]

By default grid-searched experiments (``name@lr...``) are collapsed to the grid point selected by
validation macro-F1; ``--all`` lists every grid point.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clara.report import fmt, load_runs, select_by_val, summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    runs = load_runs(args.tag)
    if not runs:
        print("no runs for", args.tag)
        return
    names = sorted(runs) if args.all else sorted(select_by_val(runs).values())
    print("| experiment | n | val macF1 | {} macF1 | Acc | F1-real | F1-fake | AUC | ECE | ep | s/run |".format(args.split))
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        if any(args.split not in r["metrics"] for r in runs[n].values()):
            continue
        s = summarise(runs[n], args.split)
        print("| {} | {} | {} | **{}** | {} | {} | {} | {} | {} | {:.1f} | {:.0f} |".format(
            n, s["n"], fmt(s["val_macf1"], s["val_macf1_std"]), fmt(s["macf1"], s["macf1_std"]),
            fmt(s["acc"], s["acc_std"]), fmt(s["f1_real"], s["f1_real_std"]),
            fmt(s["f1_fake"], s["f1_fake_std"]), fmt(s["auc"], s["auc_std"]),
            fmt(s["ece"], s["ece_std"]), s["best_epoch"], s["seconds"]))


if __name__ == "__main__":
    main()
