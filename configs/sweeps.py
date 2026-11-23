"""Experiment sweeps.  Each function maps a language to a list of experiment specs

    {"name", "group", "method", "model", "train", "teacher_model", "teacher_train", "evals"}

``group`` names the set of cached sequences an experiment needs; ``scripts/run.py`` keeps one
frozen-prefix cache per group so that seeds and variants of the same group share it.
"""
from typing import Any, Dict, List

# the two learning rates used by the official ARG / FactGuard configurations (zh: 2e-5, en: 5e-5);
# every method is run with both and the better one on validation is kept
LR_GRID = [2e-5, 5e-5]


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


# item-level replacement of all LLM outputs, and unit-level replacement (claims / rationale sentences)
NOISE_EVALS = [{"name": "noise{:02d}".format(int(r * 100)), "transform": "replace", "rate": r, "seed": 0}
               for r in (0.2, 0.4, 0.6, 0.8, 1.0)]
UNIT_EVALS = [{"name": "unit{:02d}".format(int(r * 100)), "transform": "units", "rate": r, "seed": 0}
              for r in (0.25, 0.5, 1.0)]
ALL_NOISE = NOISE_EVALS + UNIT_EVALS


def main(lang: str) -> List[Dict[str, Any]]:
    """Main comparison; every method is run over the same learning-rate grid and the grid point
    is later selected by validation macro-F1."""
    exps: List[Dict[str, Any]] = []
    exps += _with_lr({"name": "bert", "group": "trim", "method": "bert"})
    exps += _with_lr({"name": "clara", "group": "trim", "method": "clara", "evals": ALL_NOISE})
    exps += _with_lr({"name": "arg_gpt", "group": "arg_gpt", "method": "arg",
                      "model": {"rationale_source": "gpt"}})
    exps += _with_lr({"name": "arg", "group": "arg", "method": "arg",
                      "model": {"rationale_source": "llm"}, "evals": ALL_NOISE})
    exps += _with_lr({"name": "factguard", "group": "factguard", "method": "factguard", "evals": ALL_NOISE})
    return exps


def controls(lang: str, lr: float) -> List[Dict[str, Any]]:
    """Control for CLARA's mismatch negatives: ARG trained with the same augmentation.
    ``lr`` is the (validation-selected) learning rate of ARG."""
    return [{"name": "arg_swap", "group": "arg", "method": "arg",
             "model": {"rationale_source": "llm", "swap_rate": 0.2}, "train": {"lr": lr}, "evals": ALL_NOISE}]


def main_zs(lang: str) -> List[Dict[str, Any]]:
    """Simplest way to use an LLM: BERT plus late fusion of the LLM's zero-shot soft judgment."""
    return _with_lr({"name": "bert_zs", "group": "trim", "method": "bert", "model": {"use_zs": True},
                     "evals": NOISE_EVALS})


def compare(lang: str, lr: float) -> List[Dict[str, Any]]:
    """The four trainable SLM-side methods at one fixed learning rate (cross-domain runs)."""
    t = {"lr": lr}
    return [
        {"name": "bert", "group": "trim", "method": "bert", "train": t},
        {"name": "clara", "group": "trim", "method": "clara", "train": t},
        {"name": "arg", "group": "arg", "method": "arg", "model": {"rationale_source": "llm"}, "train": t},
        {"name": "factguard", "group": "factguard", "method": "factguard", "train": t},
    ]


def distill(lang: str, lr: float) -> List[Dict[str, Any]]:
    """LLM-free students.  ``lr`` is the teacher's (selected) learning rate for CLARA; the other
    teachers use the learning rates published with their official code."""
    t = {"lr": lr}
    return [
        {"name": "arg_d", "group": "arg", "method": "arg_d", "teacher_model": {"rationale_source": "llm"}},
        {"name": "factguard_d", "group": "factguard", "method": "factguard_d"},
        {"name": "clara_d", "group": "trim", "method": "clara_d", "teacher_train": t},
        {"name": "clara_d-no_inst", "group": "trim", "method": "clara_d", "model": {"kd_inst": 0.0}, "teacher_train": t},
        {"name": "clara_d-no_teacher", "group": "trim", "method": "clara_d", "model": {"no_teacher": True}},
    ]


CLARA_ABLATIONS: Dict[str, Dict[str, Any]] = {
    # unit of analysis
    "no_claims": {"use_claims": False},
    "no_overall": {"use_overall": False},
    # arguments and scores
    "no_args": {"stance": "none"},
    "no_v": {"use_v": False},
    "fi_score": {"v0_source": "fi"},      # strictly single-call: in-prompt first impression
    # verdict gate
    "no_gate": {"gate": "none"},
    "item_gate": {"gate": "instance"},
    # relevance gate and mismatch negatives
    "no_relevance": {"relevance": False},
    # aggregation and structure
    "agg_attention": {"agg": "attention"},
    "no_global": {"use_global": False},
    "entangled": {"factorised": False},   # instance scores that also see the article representation
}


def ablation(lang: str, lr: float) -> List[Dict[str, Any]]:
    return [{"name": "clara-" + k, "group": "trim", "method": "clara", "model": v, "train": {"lr": lr},
             "evals": ALL_NOISE} for k, v in CLARA_ABLATIONS.items()]


def claims_k(lang: str, lr: float) -> List[Dict[str, Any]]:
    return [{"name": "clara-k{}".format(k), "group": "trim", "method": "clara",
             "model": {"max_claims": k}, "train": {"lr": lr}} for k in (1, 2, 3)]


def sensitivity(lang: str, lr: float) -> List[Dict[str, Any]]:
    """Two alternative values for each loss weight / rate that CLARA fixes a priori."""
    grid = {"swap": ("swap_rate", (0.1, 0.4)), "lrel": ("lambda_rel", (0.25, 1.0)),
            "lmil": ("lambda_mil", (0.0, 1.0)), "lgate": ("lambda_gate", (0.0, 1.0))}
    return [{"name": "clara-{}{:g}".format(tag, v), "group": "trim", "method": "clara",
             "model": {key: v}, "train": {"lr": lr}, "evals": NOISE_EVALS[-1:]}
            for tag, (key, values) in grid.items() for v in values]
