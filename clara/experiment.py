"""Experiment sessions: one dataset + one frozen-prefix cache, many runs."""
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from clara.corrupt import replace_llm_outputs
from clara.data import SPLITS, Dataset
from clara.encoder import FrozenPrefix
from clara.models.base import Method
from clara.models.baselines import ARG, FactGuard, FeatureDistill, TextOnly
from clara.models.clara import CLARA
from clara.train import evaluate, fit, n_trainable, predict, save_predictions, set_seed, to_device

ENCODER = {
    "zh": os.environ.get("CLARA_BERT_ZH", "/media/dell/DATA/models/chinese-bert-wwm-ext"),
    "en": os.environ.get("CLARA_BERT_EN", "/media/dell/DATA/models/bert-base-uncased"),
}

# Hyper-parameters published with the official code of each baseline (run_zh.sh / run_en.sh).
# model = method-specific, train = optimisation.
PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "bert": {"zh": {"train": {"lr": 2e-5}}, "en": {"train": {"lr": 5e-5}}},
    "arg": {
        "zh": {"model": {"w_use": 2.2, "w_pred": 1.8}, "train": {"lr": 2e-5}},
        "en": {"model": {"w_use": 1.5, "w_pred": 1.0}, "train": {"lr": 5e-5}},
    },
    "arg_d": {
        "zh": {"model": {"kd": 15.0}, "train": {"lr": 5e-4}},
        "en": {"model": {"kd": 1.0}, "train": {"lr": 5e-5}},
    },
    "factguard": {
        "zh": {"model": {"alpha": 0.4, "beta": 0.15912392468192324}, "train": {"lr": 2e-5, "optimizer": "adamw"}},
        "en": {"model": {"alpha": 0.5, "beta": 0.579938945331448}, "train": {"lr": 5e-5, "optimizer": "adamw"}},
    },
    "factguard_d": {
        "zh": {"model": {"kd": 8.0}, "train": {"lr": 5e-4, "optimizer": "adamw"}},
        "en": {"model": {"kd": 8.0}, "train": {"lr": 5e-5, "optimizer": "adamw"}},
    },
    "clara": {"zh": {"train": {"lr": 2e-5}}, "en": {"train": {"lr": 5e-5}}},
    "clara_d": {"zh": {"train": {"lr": 2e-5}}, "en": {"train": {"lr": 5e-5}}},
}
TEACHER = {"arg_d": "arg", "factguard_d": "factguard", "clara_d": "clara"}


def resolve(method: str, lang: str, model: Optional[Dict[str, Any]] = None,
            train: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    p = PRESETS[method][lang]
    return {"model": dict(p.get("model", {}), **(model or {})),
            "train": dict(p.get("train", {}), **(train or {}))}


class Session:
    def __init__(self, dataset: Dataset, encoder_path: Optional[str] = None, device: str = "cuda",
                 log=print, live: bool = False):
        self.ds = dataset
        self.device = torch.device(device)
        self.encoder_path = encoder_path or ENCODER[dataset.lang]
        self.store = FrozenPrefix(self.encoder_path, device=device, live=live)
        self.log = log

    # -- construction -------------------------------------------------------------------
    def _build(self, method: str, model_cfg: Dict[str, Any], teacher: Optional[Method] = None,
               targets: Optional[Dict[Any, Any]] = None) -> Method:
        if method == "bert":
            return TextOnly(model_cfg, self.store, self.ds.lang)
        if method == "arg":
            return ARG(model_cfg, self.store, self.ds.lang)
        if method == "factguard":
            return FactGuard(model_cfg, self.store, self.ds.lang)
        if method in ("arg_d", "factguard_d"):
            return FeatureDistill(model_cfg, self.store, self.ds.lang, teacher)
        if method == "clara":
            return CLARA(model_cfg, self.store, self.ds.lang)
        if method == "clara_d":
            from clara.models.student import CLARAStudent
            return CLARAStudent(model_cfg, self.store, self.ds.lang, teacher, targets)
        raise KeyError(method)

    def _eval_items(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "path" in spec:
            return Dataset.load(spec["path"]).split(spec.get("split", "test"))
        base = self.ds.split(spec.get("base", "test"))
        if spec["transform"] == "replace":
            return replace_llm_outputs(base, spec["rate"], spec.get("seed", 0))
        raise KeyError(spec["transform"])

    def _fit_one(self, method: Method, splits: Dict[str, List[Dict[str, Any]]], train_cfg: Dict[str, Any]):
        arrays = {k: method.prepare(v) for k, v in splits.items()}
        self.store.build()
        data = {k: to_device(a, self.device) for k, a in arrays.items()}
        method.to(self.device)
        summary = fit(method, data, train_cfg, self.log)
        return data, summary

    # -- one run ------------------------------------------------------------------------
    def run(self, method: str, seed: int, model: Optional[Dict[str, Any]] = None,
            train: Optional[Dict[str, Any]] = None, teacher_model: Optional[Dict[str, Any]] = None,
            teacher_train: Optional[Dict[str, Any]] = None, evals: Optional[List[Dict[str, Any]]] = None,
            pred_path: Optional[str] = None, keep: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        lang = self.ds.lang
        cfg = resolve(method, lang, model, train)
        cfg["train"]["seed"] = seed
        splits = {s: self.ds.split(s) for s in SPLITS}
        eval_specs = evals or []
        for spec in eval_specs:
            splits[spec["name"]] = self._eval_items(spec)

        result: Dict[str, Any] = {"dataset": self.ds.name, "llm": self.ds.llm, "method": method,
                                  "seed": seed, "encoder": os.path.basename(self.encoder_path)}
        set_seed(seed)
        teacher, targets = None, None
        if method in TEACHER and not cfg["model"].get("no_teacher"):
            tcfg = resolve(TEACHER[method], lang, teacher_model, teacher_train)
            tcfg["train"]["seed"] = seed
            teacher = self._build(TEACHER[method], tcfg["model"])
            tdata, tsum = self._fit_one(teacher, splits, tcfg["train"])
            result["teacher"] = {"config": tcfg, "summary": {k: v for k, v in tsum.items() if k != "history"},
                                 "metrics": {s: evaluate(predict(teacher, tdata[s])) for s in ("val", "test")}}
            self.log("  teacher test macF1 {:.4f}".format(result["teacher"]["metrics"]["test"]["macf1"]))
            if method == "clara_d":
                from clara.models.student import teacher_targets
                targets = teacher_targets(teacher, tdata["train"], splits["train"])
            del tdata
            set_seed(seed)

        m = self._build(method, cfg["model"], teacher, targets)
        data, summary = self._fit_one(m, splits, cfg["train"])
        preds = {s: predict(m, data[s]) for s in splits if s != "train"}
        result.update({
            "config": {"model": m.cfg, "train": cfg["train"]},
            "summary": {k: v for k, v in summary.items() if k != "history"},
            "history": summary["history"],
            "metrics": {s: evaluate(p) for s, p in preds.items()},
            "n_trainable": n_trainable(m),
            "n_items": {s: len(v) for s, v in splits.items()},
            "seconds": time.time() - t0,
        })
        if pred_path:
            ids = {s: np.array([it["id"] for it in splits[s]]) for s in preds}
            extra = {}
            for s in preds:
                if "off" in data[s]:
                    extra[s + "/off"] = data[s]["off"].cpu().numpy()
                flags = [it.get("corrupted") for it in splits[s]]
                if any(f is not None for f in flags):
                    extra[s + "/corrupted"] = np.array([bool(f) for f in flags])
            save_predictions(pred_path, preds, ids, extra)
        if keep:
            result["_method"], result["_data"], result["_preds"] = m, data, preds
        return result


def write_result(path: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
