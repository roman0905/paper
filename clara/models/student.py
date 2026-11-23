"""CLARA-D: an LLM-free student with sentence-level localisation.

The student sees only the news.  Its bag consists of a holistic instance plus the sentences of the
news, scored by the SLM branch of the teacher architecture (no arguments, no gate) from the text
of each sentence alone:

    p_j = sigmoid(b0 + MLP_s(c_j)),   P_MIL = 1 - prod_j (1 - p_j),
    y_hat = sigmoid(s_x + softplus(gamma) * [logit(P_MIL) - logit(P_null)] + b)

Besides the usual response distillation, the teacher's claim-level probabilities are aligned to
sentences through the sentence index j_k that the LLM attached to every claim:

    t_j = max_{k : j_k = j} p_k^T          (sentences without an aligned claim are not supervised)

    L = BCE(y_hat, y) + kd_out * BCE(y_hat, y_hat^T) + kd_inst * mean_j BCE(p_j, t_j)
        + kd_feat * MSE(h_x, h_x^T) + lambda_mil * BCE(P_MIL, y)

which is what lets a model that never calls an LLM at test time point at the suspicious sentence.
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from clara.data import CLAIM_TYPE_ID, OVERALL_CLAIM
from clara.models.base import Method
from clara.models.clara import CLARA
from clara.models.layers import MLP, MaskAttention, bce, logit

DEFAULTS: Dict[str, Any] = dict(
    max_len=170, sent_len=64, kd_out=1.0, kd_inst=1.0, kd_feat=1.0, lambda_mil=0.5,
    init_from_teacher=True, hidden=384, dropout=0.2, null_bias=-2.0,
)


@torch.no_grad()
def teacher_targets(teacher: CLARA, data: Dict[str, torch.Tensor], items: List[Dict[str, Any]],
                    batch_size: int = 256) -> Dict[Any, Dict[str, Any]]:
    """Per training item: teacher probability, news feature and sentence-aligned instance targets."""
    teacher.eval()
    out: Dict[Any, Dict[str, Any]] = {}
    n = len(items)
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n), device=data["label"].device)
        o = teacher(data, idx)
        p, item = o["p"].cpu().numpy(), o["item"].cpu().numpy()
        sent = data["sent"][o["inst"]].cpu().numpy()
        typ = data["type"][o["inst"]].cpu().numpy()
        valid = data["valid"][o["inst"]].cpu().numpy()
        prob, feat = o["prob"].cpu().numpy(), o["feature"].cpu().numpy()
        for b in range(len(idx)):
            sel = item == b
            tgt: Dict[int, float] = {}
            for pk, jk, tk, vk in zip(p[sel], sent[sel], typ[sel], valid[sel]):
                if vk < 0.5:
                    continue
                key = -1 if tk == CLAIM_TYPE_ID["overall"] else int(jk)
                tgt[key] = max(tgt.get(key, 0.0), float(pk))
            out[items[s + b]["id"]] = {"prob": float(prob[b]), "feat": feat[b], "inst": tgt}
    return out


class CLARAStudent(Method):
    def __init__(self, cfg, store, lang, teacher: Optional[CLARA] = None,
                 targets: Optional[Dict[Any, Dict[str, Any]]] = None):
        cfg = dict(DEFAULTS, **cfg)
        super().__init__(cfg, store, lang)
        d, h, dr = self.dim, cfg["hidden"], cfg["dropout"]
        object.__setattr__(self, "targets", targets or {})
        self.top_content = store.new_top()
        self.top_sent = store.new_top()
        self.pool_x = MaskAttention(d)
        self.pool_claim = MaskAttention(d)
        self.mlp_s = MLP(d, [h], dr)
        self.b0 = nn.Parameter(torch.tensor(float(cfg["null_bias"])))
        self.mlp_x = MLP(d, [h], dr)
        self.gamma = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))
        self.bias = nn.Parameter(torch.zeros(()))
        if teacher is not None and cfg["init_from_teacher"]:
            self.top_content.load_state_dict(teacher.top_content.state_dict())
            self.top_sent.load_state_dict(teacher.top_llm.state_dict())
            names = ["pool_x", "pool_claim", "mlp_x"] + (["mlp_s"] if teacher.cfg["factorised"] else [])
            for name in names:
                getattr(self, name).load_state_dict(getattr(teacher, name).state_dict())
            self.gamma.data.copy_(teacher.gamma.data)
            self.bias.data.copy_(teacher.bias.data)
            self.b0.data.copy_(teacher.b0.data)

    def prepare(self, items):
        cfg = self.cfg
        off, texts, sent = [0], [], []
        t_inst, w_inst, t_prob, w_item, feat = [], [], [], [], []
        for it in items:
            tg = self.targets.get(it["id"])
            units = [(-1, OVERALL_CLAIM[self.lang])] + list(enumerate(it["sents"]))
            for j, text in units:
                texts.append(text)
                sent.append(j)
                if tg is not None and j in tg["inst"]:
                    t_inst.append(tg["inst"][j])
                    w_inst.append(1.0)
                else:
                    t_inst.append(0.0)
                    w_inst.append(0.0)
            off.append(len(texts))
            t_prob.append(tg["prob"] if tg is not None else 0.5)
            w_item.append(1.0 if tg is not None else 0.0)
            feat.append(tg["feat"] if tg is not None else np.zeros(self.dim, dtype=np.float32))
        reg = self.store.register
        return {
            "label": np.array([it["label"] for it in items], dtype=np.int64),
            "content": reg([it["content"] for it in items], max_len=cfg["max_len"]),
            "off": np.array(off, dtype=np.int64),
            "unit": reg(texts, max_len=cfg["sent_len"]),
            "sent": np.array(sent, dtype=np.int64),
            "t_inst": np.array(t_inst, dtype=np.float32), "w_inst": np.array(w_inst, dtype=np.float32),
            "t_prob": np.array(t_prob, dtype=np.float32), "w_item": np.array(w_item, dtype=np.float32),
            "t_feat": np.stack(feat).astype(np.float32),
        }

    def forward(self, data, idx):
        B = len(idx)
        hx_seq, mx = self.encode(self.top_content, data["content"][idx])
        hx, _ = self.pool_x(hx_seq, mx)
        inst, item = CLARA._ragged(data["off"], idx)
        hs, ms = self.encode(self.top_sent, data["unit"][inst])
        c, _ = self.pool_claim(hs, ms)
        z = self.b0 + self.mlp_s(c).squeeze(-1)
        p = torch.sigmoid(z)
        sum_log1m = torch.zeros(B, device=z.device).index_add_(0, item, nn.functional.logsigmoid(-z)).clamp(max=-1e-6)
        p_bag = (-torch.expm1(sum_log1m)).clamp(1e-6, 1 - 1e-6)
        cnt = torch.zeros(B, device=z.device).index_add_(0, item, torch.ones_like(z))
        p_null = (-torch.expm1(cnt * torch.log1p(-torch.sigmoid(self.b0)))).clamp(1e-6, 1 - 1e-6)
        s_x = self.mlp_x(hx).squeeze(-1)
        evidence = logit(p_bag, 1e-6) - logit(p_null, 1e-6)
        prob = torch.sigmoid(s_x + nn.functional.softplus(self.gamma) * evidence + self.bias)
        return {"prob": prob, "p_bag": p_bag, "s_x": s_x, "feature": hx, "p": p, "inst": inst, "item": item}

    def loss(self, out, data, idx, epoch):
        cfg = self.cfg
        y = data["label"][idx].float()
        l_cls = bce(out["prob"], y)
        total, logs = l_cls, {"cls": float(l_cls)}
        if cfg["lambda_mil"] > 0:
            l_mil = bce(out["p_bag"], y)
            total = total + cfg["lambda_mil"] * l_mil
            logs["mil"] = float(l_mil)
        w_item = data["w_item"][idx]
        if cfg["kd_out"] > 0 and float(w_item.sum()) > 0:
            l_out = bce(out["prob"], data["t_prob"][idx], w_item)
            total = total + cfg["kd_out"] * l_out
            logs["kd_out"] = float(l_out)
        w_inst = data["w_inst"][out["inst"]]
        if cfg["kd_inst"] > 0 and float(w_inst.sum()) > 0:
            l_inst = bce(out["p"], data["t_inst"][out["inst"]], w_inst)
            total = total + cfg["kd_inst"] * l_inst
            logs["kd_inst"] = float(l_inst)
        if cfg["kd_feat"] > 0 and float(w_item.sum()) > 0:
            diff = ((out["feature"] - data["t_feat"][idx]) ** 2).mean(-1)
            l_feat = (diff * w_item).sum() / w_item.sum()
            total = total + cfg["kd_feat"] * l_feat
            logs["kd_feat"] = float(l_feat)
        return total, logs

    def item_outputs(self, out):
        B = len(out["prob"])
        p_max = torch.zeros(B, device=out["p"].device).scatter_reduce_(0, out["item"], out["p"], reduce="amax", include_self=False)
        return {"p_bag": out["p_bag"], "s_x": out["s_x"], "p_max": p_max}

    def instance_outputs(self, out):
        return {"p": out["p"], "inst": out["inst"]}
