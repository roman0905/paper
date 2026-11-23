"""Training / evaluation loop shared by every method (protocol of ARG):

Adam(W), batch size 64, at most 50 epochs, model selection by validation macro-F1 with early
stopping (patience 5), decision threshold 0.5, test once with the selected checkpoint.
"""
import copy
import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from clara.data import SPLITS, Dataset
from clara.encoder import FrozenPrefix
from clara.metrics import detection_metrics, ece
from clara.models.base import Method

TRAIN_DEFAULTS: Dict[str, Any] = dict(
    lr=2e-5, weight_decay=5e-5, optimizer="adam", batch_size=64, epochs=50, patience=5,
    eval_batch_size=256, seed=3759, amp=False,
    # Scalar parameters (fusion weights, temperatures, biases) get their own, larger step size:
    # with the encoder learning rate (2e-5 .. 5e-5) Adam moves a scalar by at most ~0.02 over a
    # whole run, i.e. such parameters would effectively stay at their initial values.
    lr_scalar=1e-2,
    # Optional separate learning rate for freshly initialised modules (everything that is not a
    # pretrained encoder layer).  None = same as the encoder, i.e. the protocol of ARG.
    lr_head=None,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_device(arrays: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v).to(device) for k, v in arrays.items()}


@torch.no_grad()
def predict(method: Method, data: Dict[str, torch.Tensor], batch_size: int = 256) -> Dict[str, np.ndarray]:
    """Run the model over a whole split, in item order."""
    method.eval()
    n = len(data["label"])
    item_out: Dict[str, List[torch.Tensor]] = {}
    inst_out: Dict[str, List[torch.Tensor]] = {}
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n), device=data["label"].device)
        out = method(data, idx)
        item = dict(method.item_outputs(out), prob=out["prob"])
        if "feature" in out:
            item["feature"] = out["feature"]
        for k, v in item.items():
            item_out.setdefault(k, []).append(v.float().cpu())
        for k, v in method.instance_outputs(out).items():
            inst_out.setdefault(k, []).append(v.cpu())
    res = {k: torch.cat(v).numpy() for k, v in item_out.items()}
    res.update({"inst_" + k: torch.cat(v).numpy() for k, v in inst_out.items()})
    res["label"] = data["label"].cpu().numpy()
    return res


def evaluate(pred: Dict[str, np.ndarray]) -> Dict[str, float]:
    m = detection_metrics(pred["label"], pred["prob"])
    m["ece"] = ece(pred["label"], pred["prob"])
    return m


def fit(method: Method, data: Dict[str, Dict[str, torch.Tensor]], cfg: Dict[str, Any],
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Train ``method`` on data["train"], select on data["val"]. Returns the training summary and
    leaves the best checkpoint loaded."""
    cfg = dict(TRAIN_DEFAULTS, **cfg)
    named = [(n, p) for n, p in method.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    scalars = [p for _, p in named if p.ndim == 0]
    encoder = [p for n, p in named if p.ndim > 0 and n.startswith("top")]       # pretrained layers
    heads = [p for n, p in named if p.ndim > 0 and not n.startswith("top")]
    groups = [{"params": encoder}]
    if heads:
        groups.append({"params": heads, "lr": cfg["lr_head"] or cfg["lr"]})
    if scalars:
        groups.append({"params": scalars, "lr": cfg["lr_scalar"], "weight_decay": 0.0})
    opt_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    opt = opt_cls(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    device = data["train"]["label"].device
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    method.amp = use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    n = len(data["train"]["label"])
    best, best_state, best_epoch, history = -1.0, None, -1, []
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        method.train()
        perm = torch.randperm(n, generator=gen).to(device)
        agg: Dict[str, float] = {}
        nb = 0
        for s in range(0, n, cfg["batch_size"]):
            idx = perm[s:s + cfg["batch_size"]]
            if len(idx) < 2:        # BatchNorm cannot handle a batch of one
                continue
            out = method(data["train"], idx)
            loss, logs = method.loss(out, data["train"], idx, epoch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            nb += 1
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
        val = evaluate(predict(method, data["val"], cfg["eval_batch_size"]))
        history.append({"epoch": epoch, "val_macf1": val["macf1"], "val_auc": val["auc"],
                        "seconds": time.time() - t0, **{k: v / max(nb, 1) for k, v in agg.items()}})
        log("  epoch {:2d}  {}  val macF1 {:.4f} acc {:.4f}".format(
            epoch, " ".join("{}={:.4f}".format(k, v / max(nb, 1)) for k, v in agg.items()), val["macf1"], val["acc"]))
        if val["macf1"] > best:
            best, best_epoch = val["macf1"], epoch
            best_state = {k: v.detach().clone() for k, v in method.state_dict().items()}
        elif epoch - best_epoch >= cfg["patience"]:
            break
    method.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "epochs_run": len(history), "best_val_macf1": best,
            "train_seconds": time.time() - t0, "history": history}


def n_trainable(method: Method) -> int:
    return int(sum(p.numel() for p in method.parameters() if p.requires_grad))


def save_predictions(path: str, preds: Dict[str, Dict[str, np.ndarray]],
                     ids: Dict[str, np.ndarray], extra: Optional[Dict[str, np.ndarray]] = None) -> None:
    flat = {}
    for split, d in preds.items():
        flat[split + "/id"] = ids[split]
        for k, v in d.items():
            if k == "feature":
                continue
            flat[split + "/" + k] = v
    if extra:
        flat.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **flat)
