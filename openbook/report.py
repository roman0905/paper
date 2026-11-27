"""Reading the result files of the open-book runs and comparing methods."""
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from openbook.metrics import classification_metrics, paired_bootstrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Runs = Dict[str, Dict[int, Dict[str, Any]]]

METRICS = ("macf1", "macp", "macr", "f1_h", "acc", "mae", "ece", "conf")


def load_runs(tag: str) -> Runs:
    """results/ob/runs/<tag>/<name>/s<seed>.json -> {name: {seed: result}}"""
    out: Runs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "ob", "runs", tag, "*", "s*.json"))):
        name = os.path.basename(os.path.dirname(path))
        if not os.path.basename(path)[1:-5].isdigit():       # e.g. s1.reeval.json: a re-evaluation, kept apart
            continue
        seed = int(os.path.basename(path)[1:-5])
        with open(path, encoding="utf-8") as f:
            out.setdefault(name, {})[seed] = json.load(f)
    return out


def load_preds(tag: str, name: str, seed: int, reeval: bool = False) -> Optional[Dict[str, np.ndarray]]:
    """Stored predictions of a run (``reeval``: those of its re-evaluation from the saved adapter)."""
    path = os.path.join(ROOT, "outputs", "ob", "runs", tag, name, "s{}{}.npz".format(seed, ".reeval" if reeval else ""))
    if not os.path.exists(path):
        return None
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def ms(values: Sequence[float]) -> Dict[str, float]:
    v = [float(x) for x in values]
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def welch(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def summarise(seeds: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Mean / std over seeds of every metric of every evaluation condition of one method."""
    runs = list(seeds.values())
    out: Dict[str, Any] = {"seeds": sorted(seeds), "n": len(runs)}
    for cond in sorted(set.intersection(*[set(r["metrics"]) for r in runs])):
        out[cond] = {m: ms([r["metrics"][cond][m] for r in runs]) for m in METRICS if m in runs[0]["metrics"][cond]}
        n_cls = sum(1 for k in runs[0]["metrics"][cond] if k.startswith("f1_") and k[3:].isdigit())
        out[cond]["per_class"] = [float(np.mean([r["metrics"][cond]["f1_%d" % c] for r in runs])) for c in range(n_cls)]
    extra = [r.get("extra") or {} for r in runs]
    keys = set.intersection(*[set(e) for e in extra]) if extra else set()
    out["extra"] = {}
    for k in sorted(keys):
        if isinstance(extra[0][k], list):
            out["extra"][k] = np.mean([e[k] for e in extra], axis=0).tolist()
        else:
            out["extra"][k] = ms([e[k] for e in extra])
    out["best_epoch"] = [r["summary"]["best_epoch"] for r in runs]
    out["train_seconds"] = float(np.mean([r["summary"]["train_seconds"] for r in runs]))
    out["seconds"] = float(np.mean([r["seconds"] for r in runs]))
    out["n_trainable"] = int(runs[0].get("n_trainable", 0))
    cost = [r.get("cost") for r in runs if r.get("cost")]
    if cost:
        out["cost"] = {k: float(np.mean([c[k] for c in cost])) for k in cost[0]}
    if runs[0].get("max_memory_gb"):
        out["max_memory_gb"] = float(np.max([r["max_memory_gb"] for r in runs]))
    return out


def _ids(tag: str, name: str, runs: Dict[int, Dict[str, Any]], cond: str) -> Optional[List[str]]:
    """Item ids of a condition (identical for every seed of a method)."""
    for s in sorted(runs):
        p = load_preds(tag, name, s)
        if p is not None and cond + "/id" in p:
            return [str(x) for x in p[cond + "/id"]]
    return None


def _pick(p: Dict[str, np.ndarray], cond: str, keep: Optional[Sequence[str]]):
    """Labels and predicted classes of a condition, restricted to (and ordered by) ``keep``."""
    y, pred = p[cond + "/label"], p[cond + "/prob"].argmax(1)
    if keep is None:
        return y, pred
    pos = {str(x): k for k, x in enumerate(p[cond + "/id"])}
    idx = [pos[i] for i in keep]
    return y[idx], pred[idx]


def seed_scores(tag: str, name: str, runs: Dict[int, Dict[str, Any]], cond: str, n_classes: int,
                keep: Optional[Sequence[str]] = None) -> Dict[int, float]:
    """Macro-F1 of every seed; read from the result files, or recomputed from the stored
    predictions when the evaluation is restricted to the items ``keep``."""
    out = {}
    for s, r in runs.items():
        if keep is None:
            if cond in r["metrics"]:
                out[s] = r["metrics"][cond]["macf1"]
            continue
        p = load_preds(tag, name, s)
        if p is not None and cond + "/prob" in p:
            y, pred = _pick(p, cond, keep)
            out[s] = classification_metrics(y, np.eye(n_classes)[pred])["macf1"]
    return out


def compare(tag_a: str, runs_a: Dict[int, Dict[str, Any]], name_a: str, cond_a: str,
            tag_b: str, runs_b: Dict[int, Dict[str, Any]], name_b: str, cond_b: str,
            n_classes: int, keep: Optional[set] = None) -> Dict[str, Any]:
    """A minus B in macro-F1.

    Both sides are evaluated on the same items: the items common to the two conditions (some
    conditions are computed on a subset of the test claims), further restricted to ``keep``.
    Reports the Welch test over seeds (all seeds of either side) and the paired bootstrap over
    items (seeds common to both sides)."""
    ia, ib = _ids(tag_a, name_a, runs_a, cond_a), _ids(tag_b, name_b, runs_b, cond_b)
    if ia is None or ib is None:
        return {}
    setb = set(ib)
    common_items = [i for i in ia if i in setb and (keep is None or i in keep)]
    if not common_items:
        return {}
    restrict = None if (keep is None and len(common_items) == len(ia) == len(ib)) else common_items
    sa = seed_scores(tag_a, name_a, runs_a, cond_a, n_classes, restrict)
    sb = seed_scores(tag_b, name_b, runs_b, cond_b, n_classes, restrict)
    if not sa or not sb:
        return {}
    fa, fb = list(sa.values()), list(sb.values())
    out: Dict[str, Any] = {"a": float(np.mean(fa)), "b": float(np.mean(fb)), "diff_mean": float(np.mean(fa) - np.mean(fb)),
                           "welch_p": welch(fa, fb), "n_a": len(fa), "n_b": len(fb), "n_items": len(common_items)}
    seeds = sorted(set(sa) & set(sb))
    al = []
    for s in seeds:
        pa, pb = load_preds(tag_a, name_a, s), load_preds(tag_b, name_b, s)
        if pa is None or pb is None:
            continue
        ya, qa = _pick(pa, cond_a, common_items)
        yb, qb = _pick(pb, cond_b, common_items)
        assert np.array_equal(ya, yb), "labels of the two conditions disagree"
        al.append((ya, qa, qb))
    if al:
        y = al[0][0]
        out["bootstrap"] = paired_bootstrap(y, [x[1] for x in al], [x[2] for x in al], n_classes)
        out["acc_a"] = float(np.mean([(x[1] == y).mean() for x in al]))
        out["acc_b"] = float(np.mean([(x[2] == y).mean() for x in al]))
        out["seeds_paired"] = seeds
    return out


def significant(cmp: Dict[str, Any], level: float = 0.05) -> bool:
    """The thesis calls a difference significant only if both tests agree."""
    if not cmp or "bootstrap" not in cmp:
        return False
    w = cmp.get("welch_p", float("nan"))
    return cmp["bootstrap"]["p"] < level and w == w and w < level


def select_by_val(runs: Runs, prefix: str) -> Optional[str]:
    """Among the configurations whose name starts with ``prefix`` (a learning-rate grid), the one
    with the best mean validation macro-F1."""
    cands = {n: np.mean([r["metrics"]["val"]["macf1"] for r in s.values()]) for n, s in runs.items() if n.startswith(prefix)}
    return max(cands, key=cands.get) if cands else None
