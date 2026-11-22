"""Experiment sweeps.  Each function maps a language to a list of experiment specs

    {"name", "group", "method", "model", "train", "teacher_model", "teacher_train", "evals"}

``group`` names the set of cached sequences an experiment needs; ``scripts/run.py`` keeps one
frozen-prefix cache per group so that seeds and variants of the same group share it.
"""
from typing import Any, Dict, List

LR_GRID = [2e-5, 5e-5, 1e-4]


def _lr_name(name: str, lr: float) -> str:
    return "{}@lr{:g}".format(name, lr)


def _with_lr(base: Dict[str, Any], grid=LR_GRID) -> List[Dict[str, Any]]:
    out = []
    for lr in grid:
        e = dict(base)
        e["name"] = _lr_name(base["name"], lr)
        key = "teacher_train" if base["method"].endswith("_d") else "train"
        e[key] = dict(base.get(key, {}), lr=lr)
        out.append(e)
    return out


NOISE_EVALS = [{"name": "noise{:02d}".format(int(r * 100)), "transform": "replace", "rate": r, "seed": 0}
               for r in (0.2, 0.4, 0.6, 0.8, 1.0)]


def main(lang: str) -> List[Dict[str, Any]]:
    """Main comparison; every method is run over the same learning-rate grid and the grid point
    is later selected by validation macro-F1."""
    exps: List[Dict[str, Any]] = []
    exps += _with_lr({"name": "bert", "group": "trim", "method": "bert"})
    exps += _with_lr({"name": "clara", "group": "trim", "method": "clara", "evals": NOISE_EVALS})
    exps += _with_lr({"name": "arg_gpt", "group": "arg_gpt", "method": "arg",
                      "model": {"rationale_source": "gpt"}})
    exps += _with_lr({"name": "arg", "group": "arg", "method": "arg",
                      "model": {"rationale_source": "llm"}, "evals": NOISE_EVALS})
    exps += _with_lr({"name": "factguard", "group": "factguard", "method": "factguard", "evals": NOISE_EVALS})
    return exps


CLARA_ABLATIONS: Dict[str, Dict[str, Any]] = {
    # unit of analysis
    "no_claims": {"use_claims": False},
    "no_overall": {"use_overall": False},
    # arguments and scores
    "refute_only": {"stance": "refute"},
    "support_only": {"stance": "support"},
    "no_args": {"stance": "none"},
    "no_v": {"use_v": False},
    "no_v0": {"use_v0": False},
    # reliability gate
    "no_gate": {"gate": "none"},
    "item_gate": {"gate": "instance"},
    "no_gate_fake": {"gate_fake": False},
    # aggregation
    "agg_attention": {"agg": "attention"},
    "agg_max": {"agg": "max"},
    "agg_mean": {"agg": "mean"},
    "no_mil_loss": {"lambda_mil": 0.0},
    "no_global": {"use_global": False},
}


def ablation(lang: str, lr: float) -> List[Dict[str, Any]]:
    return [{"name": "clara-" + k, "group": "trim", "method": "clara", "model": v, "train": {"lr": lr}}
            for k, v in CLARA_ABLATIONS.items()]


def claims_k(lang: str, lr: float) -> List[Dict[str, Any]]:
    return [{"name": "clara-k{}".format(k), "group": "trim", "method": "clara",
             "model": {"max_claims": k}, "train": {"lr": lr}} for k in (1, 2, 3, 4)]


def sensitivity(lang: str, lr: float) -> List[Dict[str, Any]]:
    exps = []
    for lm in (0.0, 0.25, 1.0, 2.0):
        exps.append({"name": "clara-lmil{:g}".format(lm), "group": "trim", "method": "clara",
                     "model": {"lambda_mil": lm}, "train": {"lr": lr}})
    for lg in (0.0, 0.25, 1.0, 2.0):
        exps.append({"name": "clara-lgate{:g}".format(lg), "group": "trim", "method": "clara",
                     "model": {"lambda_gate": lg}, "train": {"lr": lr}})
    return exps
