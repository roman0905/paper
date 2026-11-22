"""Baselines trained under exactly the same protocol as CLARA.

* ``TextOnly``  - the SLM baseline of ARG: encoder + attention pooling + MLP.
* ``ARG``       - Hu et al., "Bad Actor, Good Advisor", AAAI 2024.
* ``FactGuard`` - Liu et al., "FactGuard", AAAI 2026.
* ``FeatureDistill`` - the rationale-free students ARG-D / FactGuard-D.

``ARG`` and ``FactGuard`` are re-implementations that follow the forward pass and the loss of the
official code (github.com/ICTMCG/ARG, github.com/ryliu68/FACTGUARD) line by line; places where
the official behaviour is unusual are kept and marked ``# official:``.  Their inputs are padded to
``max_len`` exactly as in the official data loaders, because both models average over all
positions, padding included.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clara.models.base import Method
from clara.models.layers import MLP, CrossAttention, MaskAttention, bce


class TextOnly(Method):
    def __init__(self, cfg, store, lang):
        super().__init__(cfg, store, lang)
        self.top = store.new_top()
        self.pool = MaskAttention(self.dim)
        self.mlp = MLP(self.dim, [384], 0.2)

    def prepare(self, items):
        return {
            "label": np.array([it["label"] for it in items], dtype=np.int64),
            "content": self.store.register([it["content"] for it in items], max_len=self.cfg.get("max_len", 170)),
        }

    def forward(self, data, idx):
        h, m = self.encode(self.top, data["content"][idx])
        feat, _ = self.pool(h, m)
        return {"prob": torch.sigmoid(self.mlp(feat).squeeze(-1)), "feature": feat}

    def loss(self, out, data, idx, epoch):
        l = bce(out["prob"], data["label"][idx].float())
        return l, {"cls": float(l)}


def _usefulness_head(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, 384), nn.ReLU(), nn.Linear(384, 1), nn.Sigmoid())


def _score_mapper(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(384, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(64, 1), nn.Sigmoid())


def _judgment_head(dim: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, 384), nn.ReLU(), nn.Linear(384, n_out))


class _TwoRationale(Method):
    """Shared skeleton of ARG and FactGuard: news + two LLM texts."""

    def __init__(self, cfg, store, lang, n_judge_2: int):
        super().__init__(cfg, store, lang)
        d = self.dim
        self.top_content = store.new_top()
        self.top_ftr = store.new_top()
        self.aggregator = MaskAttention(d)
        self.mlp = MLP(d, [384], 0.2)
        self.hard_mlp_2, self.hard_mlp_3 = _usefulness_head(d), _usefulness_head(d)
        self.score_mapper_2, self.score_mapper_3 = _score_mapper(d), _score_mapper(d)
        self.simple_attn_2, self.simple_attn_3 = MaskAttention(d), MaskAttention(d)
        self.simple_mlp_2, self.simple_mlp_3 = _judgment_head(d, n_judge_2), _judgment_head(d, 3)
        self.content_attention = MaskAttention(d)
        self.ca_content_2, self.ca_content_3 = CrossAttention(d), CrossAttention(d)
        self.ca_ftr_2, self.ca_ftr_3 = CrossAttention(d), CrossAttention(d)

    def _register(self, texts: List[str]) -> np.ndarray:
        return self.store.register(texts, max_len=self.cfg.get("max_len", 170), pad_full=True)

    def _fuse(self, attn_content, expert_2, expert_3, mf_2, mf_3) -> Dict[str, torch.Tensor]:
        w2, w3 = self.score_mapper_2(mf_2), self.score_mapper_3(mf_3)
        feats = torch.stack([attn_content, w2 * expert_2, w3 * expert_3], dim=1)
        final, _ = self.aggregator(feats)
        return {"prob": torch.sigmoid(self.mlp(final).squeeze(-1)), "feature": final,
                "content_feature": attn_content, "gate": torch.cat([w2, w3], dim=1),
                "hard_2": self.hard_mlp_2(mf_2).squeeze(-1), "hard_3": self.hard_mlp_3(mf_3).squeeze(-1)}

    def item_outputs(self, out):
        return {"gate": out["gate"]}


class ARG(_TwoRationale):
    """cfg: ``rationale_source`` in {"gpt", "llm"}, ``w_use``, ``w_pred``."""

    def __init__(self, cfg, store, lang):
        super().__init__(cfg, store, lang, n_judge_2=3)

    def prepare(self, items):
        src = self.cfg.get("rationale_source", "llm")
        td, cs = [], []
        for it in items:
            base = it["gpt"] if src == "gpt" else it
            for name, out in (("td", td), ("cs", cs)):
                r = base[name]
                acc = r["acc"] if "acc" in r else int(r["pred"] == it["label"])
                out.append((r["rationale"], r["pred"], acc))
        return {
            "label": np.array([it["label"] for it in items], dtype=np.int64),
            "content": self._register([it["content"] for it in items]),
            "r2": self._register([x[0] for x in td]), "r3": self._register([x[0] for x in cs]),
            "pred2": np.array([x[1] for x in td], dtype=np.int64),
            "pred3": np.array([x[1] for x in cs], dtype=np.int64),
            "acc2": np.array([x[2] for x in td], dtype=np.float32),
            "acc3": np.array([x[2] for x in cs], dtype=np.float32),
        }

    def forward(self, data, idx):
        hc, mc = self.encode(self.top_content, data["content"][idx])
        h2, m2 = self.encode(self.top_ftr, data["r2"][idx])
        h3, m3 = self.encode(self.top_ftr, data["r3"][idx])
        # official: plain mean over all max_len positions, padding included
        expert_2 = self.ca_content_2(h2, hc, mc)[0].mean(1)
        expert_3 = self.ca_content_3(h3, hc, mc)[0].mean(1)
        mf_2 = self.ca_ftr_2(hc, h2, m2)[0].mean(1)
        mf_3 = self.ca_ftr_3(hc, h3, m3)[0].mean(1)
        attn_content, _ = self.content_attention(hc, mc)
        out = self._fuse(attn_content, expert_2, expert_3, mf_2, mf_3)
        # official: the judgment predictor pools the rationale without a padding mask
        out["simple_2"] = self.simple_mlp_2(self.simple_attn_2(h2)[0])
        out["simple_3"] = self.simple_mlp_3(self.simple_attn_3(h3)[0])
        return out

    def loss(self, out, data, idx, epoch):
        l_cls = bce(out["prob"], data["label"][idx].float())
        l_use = bce(out["hard_2"], data["acc2"][idx]) + bce(out["hard_3"], data["acc3"][idx])
        l_pred = F.cross_entropy(out["simple_2"], data["pred2"][idx]) + \
            F.cross_entropy(out["simple_3"], data["pred3"][idx])
        l = l_cls + self.cfg["w_use"] * l_use / 2 + self.cfg["w_pred"] * l_pred / 2
        return l, {"cls": float(l_cls), "use": float(l_use), "pred": float(l_pred)}


class FactGuard(_TwoRationale):
    """News N, LLM topic-content C (slot 2) and commonsense rationale R (slot 3).

    cfg: ``alpha`` (usability weight), ``beta`` (text weight).
    """

    def __init__(self, cfg, store, lang):
        super().__init__(cfg, store, lang, n_judge_2=2)

    def prepare(self, items):
        y_llm = np.array([it["cs"]["pred"] for it in items], dtype=np.int64)
        return {
            "label": np.array([it["label"] for it in items], dtype=np.int64),
            "content": self._register([it["content"] for it in items]),
            "r2": self._register([it["tc"] for it in items]),
            "r3": self._register([it["cs"]["rationale"] for it in items]),
            "y_llm": y_llm,
            # an undecided LLM judgment ("other") gives an uninformative usability target
            "y_llm_soft": np.where(y_llm == 2, 0.5, y_llm).astype(np.float32),
        }

    def forward(self, data, idx):
        hn, mn = self.encode(self.top_content, data["content"][idx])
        hc, mcc = self.encode(self.top_ftr, data["r2"][idx])
        hr, mr = self.encode(self.top_ftr, data["r3"][idx])
        # official: topic-content replaces the news in every cross-attention, both expert
        # branches use (C, R), and the key masks below are the ones used by the official code
        expert_2 = self.ca_content_2(hr, hc, mn)[0].mean(1)
        expert_3 = self.ca_content_3(hr, hc, mn)[0].mean(1)
        mf_2 = self.ca_ftr_2(hc, hr, mcc)[0].mean(1)
        mf_3 = self.ca_ftr_3(hc, hr, mr)[0].mean(1)
        attn_content, _ = self.content_attention(hn, mn)
        out = self._fuse(attn_content, expert_2, expert_3, mf_2, mf_3)
        out["simple_2"] = self.simple_mlp_2(self.simple_attn_2(hc)[0])
        out["simple_3"] = self.simple_mlp_3(self.simple_attn_3(hr)[0])
        return out

    def loss(self, out, data, idx, epoch):
        y = data["label"][idx]
        l_cls = bce(out["prob"], y.float())
        l_use = bce(out["hard_2"], data["y_llm_soft"][idx]) + bce(out["hard_3"], torch.zeros_like(out["hard_3"]))
        l_text = F.cross_entropy(out["simple_2"], y) + F.cross_entropy(out["simple_3"], data["y_llm"][idx])
        l = l_cls + self.cfg["alpha"] * l_use / 2 + self.cfg["beta"] * l_text / 2
        return l, {"cls": float(l_cls), "use": float(l_use), "text": float(l_text)}


class FeatureDistill(Method):
    """ARG-D / FactGuard-D: a rationale-free student regressing the teacher's fused feature.

    Student = (teacher's news encoder, frozen) -> Transformer block -> attention pooling -> MLP,
    trained with ``BCE + kd * MSE(student feature, teacher feature)``.  Following the papers, the
    news encoder and the classifier are initialised from the trained teacher.
    """

    def __init__(self, cfg, store, lang, teacher: _TwoRationale):
        super().__init__(cfg, store, lang)
        from timm.models.vision_transformer import Block

        object.__setattr__(self, "teacher", teacher.eval())
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.top = store.new_top()
        self.top.load_state_dict(teacher.top_content.state_dict())
        for p in self.top.parameters():
            p.requires_grad_(False)
        self.block = Block(dim=self.dim, num_heads=4)
        self.pool = MaskAttention(self.dim)
        self.mlp = MLP(self.dim, [384], 0.2)
        self.mlp.load_state_dict(teacher.mlp.state_dict())

    def prepare(self, items):
        return self.teacher.prepare(items)

    def train(self, mode: bool = True):
        super().train(mode)
        self.top.eval()
        return self

    def forward(self, data, idx):
        with torch.no_grad():
            h, _ = self.encode(self.top, data["content"][idx])
        feat, _ = self.pool(self.block(h))
        return {"prob": torch.sigmoid(self.mlp(feat).squeeze(-1)), "feature": feat}

    def loss(self, out, data, idx, epoch):
        with torch.no_grad():
            t = self.teacher(data, idx)["feature"]
        l_cls = bce(out["prob"], data["label"][idx].float())
        l_kd = F.mse_loss(out["feature"], t)
        return l_cls + self.cfg["kd"] * l_kd, {"cls": float(l_cls), "kd": float(l_kd)}
