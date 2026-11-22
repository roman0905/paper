"""Loading and summarising finished runs."""
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ("macf1", "acc", "f1_real", "f1_fake", "auc", "ece")


def load_runs(tag: str, root: str = ROOT) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """{experiment name: {seed: result}} for one dataset tag."""
    runs: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for path in glob.glob(os.path.join(root, "results", "runs", tag, "*", "s*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        runs[os.path.basename(os.path.dirname(path))][int(r["seed"])] = r
    return dict(runs)


def stat(values) -> Tuple[float, float]:
    v = np.asarray(list(values), dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def summarise(seeds: Dict[int, Dict[str, Any]], split: str = "test") -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(seeds), "seeds": sorted(seeds)}
    for m in METRICS:
        out[m], out[m + "_std"] = stat(r["metrics"][split][m] for r in seeds.values())
    out["val_macf1"], out["val_macf1_std"] = stat(r["metrics"]["val"]["macf1"] for r in seeds.values())
    out["best_epoch"] = float(np.mean([r["summary"]["best_epoch"] for r in seeds.values()]))
    out["seconds"] = float(np.mean([r["seconds"] for r in seeds.values()]))
    return out


def select_by_val(runs: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, str]:
    """For experiments run over a grid (``name@lr...``) pick the grid point with the best mean
    validation macro-F1.  Returns {base name: selected experiment name}."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for name in runs:
        groups[name.split("@")[0]].append(name)
    chosen = {}
    for base, names in groups.items():
        chosen[base] = max(names, key=lambda n: np.mean([r["metrics"]["val"]["macf1"] for r in runs[n].values()]))
    return chosen


def load_preds(tag: str, name: str, seed: int, root: str = ROOT) -> Optional[Dict[str, np.ndarray]]:
    path = os.path.join(root, "outputs", "runs", tag, name, "s{}.npz".format(seed))
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fmt(mean: float, std: float, scale: float = 100.0, digits: int = 2) -> str:
    return "{:.{d}f} ± {:.{d}f}".format(mean * scale, std * scale, d=digits)
