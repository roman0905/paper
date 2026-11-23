#!/usr/bin/env python
"""Print the learning rate of the grid point selected by mean validation macro-F1.

    python scripts/selected_lr.py ARG-CN.qwen25 clara     ->  5e-05
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clara.report import load_runs, select_by_val  # noqa: E402

tag, base = sys.argv[1], sys.argv[2]
runs = {k: v for k, v in load_runs(tag).items() if k.split("@")[0] == base and "@lr" in k}
if not runs:
    sys.exit("no grid runs for {} in {}".format(base, tag))
print(select_by_val(runs)[base].split("@lr")[1])
