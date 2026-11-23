#!/usr/bin/env python
"""Grid points selected by mean validation macro-F1.

    python scripts/selected_lr.py ARG-CN.qwen25 clara                 ->  5e-05
    python scripts/selected_lr.py ARG-CN.qwen25 --names bert clara    ->  bert@lr2e-05 clara@lr5e-05
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clara.report import load_runs, select_by_val  # noqa: E402


def main():
    tag, rest = sys.argv[1], sys.argv[2:]
    names = rest[1:] if rest[0] == "--names" else rest[:1]
    runs = {k: v for k, v in load_runs(tag).items() if "@lr" in k and k.split("@")[0] in names}
    chosen = select_by_val(runs)
    missing = [n for n in names if n not in chosen]
    if missing:
        sys.exit("no grid runs for {} in {}".format(missing, tag))
    if rest[0] == "--names":
        print(" ".join(chosen[n] for n in names))
    else:
        print(chosen[names[0]].split("@lr")[1])


if __name__ == "__main__":
    main()
