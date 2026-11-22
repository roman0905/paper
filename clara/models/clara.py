"""CLARA: Claim-Level Argument Reliability Adjudication.

Every news item x is a bag of instances: one holistic instance (the LLM's two-sided arguments and
verdict about the message as a whole) followed by up to K atomic claims, each with the LLM's
strongest supporting argument a+, strongest refuting argument a-, and a suspicion score v in [0,1].

For every instance k the adjudicator computes

    e_k      claim-in-context representation (claim text attending over the news tokens);
             it contains no LLM argument
    s_slm_k  = MLP_s([e_k ; h_x])                       the SLM's own evidence
    s_llm_k  = MLP_a([u-_k - u+_k ; u-_k * u+_k ; e_k]) + alpha * logit(v_k)     the LLM's evidence
    g_k      = sigmoid(MLP_g([e_k ; u+_k ; u-_k ; type ; score features]))     reliability gate
    p_k      = sigmoid(s_slm_k + g_k * s_llm_k)         P(instance k is problematic)

and aggregates with a noisy-OR, P_MIL = 1 - prod_k (1 - p_k), which is fused with a global text
branch:  y_hat = sigmoid(s_x + softplus(gamma) * logit(P_MIL) + b).

Training uses only article-level labels:

    L = BCE(y_hat, y) + lambda_mil * BCE(P_MIL, y) + lambda_gate * L_gate

L_gate supervises the gate with "did the LLM take the right side on this instance?".  For the
holistic instance this is observable.  For claims it follows from the MIL asymmetry: in real news
every claim is unproblematic, so the LLM was right iff it did not suspect the claim; in fake news
the latent claim state is replaced by its noisy-OR posterior q_k = p_k / P_MIL (stop-gradient,
enabled after a warm-up).
"""
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from clara.data import CLAIM_TYPE_ID, instances
from clara.models.base import Method
from clara.models.layers import MLP, CrossAttention, MaskAttention, bce, logit

DEFAULTS: Dict[str, Any] = dict(
    max_len=170, pair_len=128, claim_len=48, max_claims=6,
    use_claims=True, use_overall=True, stance="both", use_v=True, use_v0=True,
    gate="claim", gate_fake=True, agg="noisy_or", use_global=True,
    lambda_mil=0.5, lambda_gate=0.5, eta=0.5, warmup=2, delta=0.05,
    hidden=384, dropout=0.2, inst_bias=-2.0,
)

N_SCORE_FEATS = 5


def _entropy(dist) -> float:
    if not dist:
        return 0.0
    return float(-sum(p * math.log(p) for p in dist if p > 0) / math.log(5))


class CLARA(Method):
    def __init__(self, cfg, store, lang):
        cfg = dict(DEFAULTS, **cfg)
        super().__init__(cfg, store, lang)
        d, h, dr = self.dim, cfg["hidden"], cfg["dropout"]
        self.top_content = store.new_top()
        self.top_llm = store.new_top()
        self.pool_x = MaskAttention(d)
        self.pool_pair = MaskAttention(d)
        self.pool_claim = MaskAttention(d)
        self.ctx = CrossAttention(d, heads=4)
        self.ctx_norm = nn.LayerNorm(d)

        n_arg = {"both": 2 * d, "refute": d, "support": d, "none": 0}[cfg["stance"]]
        self.mlp_a = MLP(n_arg + d, [h], dr)
        self.mlp_s = MLP(2 * d, [h], dr)
        nn.init.constant_(self.mlp_s.net[-1].bias, cfg["inst_bias"])
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.alpha0 = nn.Parameter(torch.tensor(1.0))

        n_gate_arg = {"both": 2 * d, "refute": d, "support": d, "none": 0}[cfg["stance"]]
        self.type_emb = nn.Embedding(len(CLAIM_TYPE_ID), 16)
        self.mlp_g = MLP(d + n_gate_arg + 16 + N_SCORE_FEATS, [h], dr)

        if cfg["agg"] == "attention":
            self.agg_attn = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.mlp_x = MLP(d, [h], dr)
        self.gamma = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))   # softplus(gamma) = 1
        self.bias = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------ data
    def prepare(self, items: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        cfg, lang = self.cfg, self.lang
        off, claim, sup, ref = [0], [], [], []
        v, v0, ent, typ, sent, valid = [], [], [], [], [], []
        for it in items:
            inst = instances(it, lang, cfg["max_claims"])
            if not cfg["use_claims"]:
                inst = inst[:1]
            elif not cfg["use_overall"] and len(inst) > 1:
                inst = inst[1:]
            for x in inst:
                claim.append(x["claim"])
                sup.append((x["claim"], x["support"]))
                ref.append((x["claim"], x["refute"]))
                v.append(x["v"])
                v0.append(x["v0"])
                ent.append(_entropy(x["dist"]))
                typ.append(CLAIM_TYPE_ID.get(x["type"], 0))
                sent.append(x["sent"])
                valid.append(x["valid"])
            off.append(len(claim))
        reg = self.store.register
        return {
            "label": np.array([it["label"] for it in items], dtype=np.int64),
            "content": reg([it["content"] for it in items], max_len=cfg["max_len"]),
            "off": np.array(off, dtype=np.int64),
            "claim": reg(claim, max_len=cfg["claim_len"]),
            "sup": reg([a for a, _ in sup], [b for _, b in sup], max_len=cfg["pair_len"]),
            "ref": reg([a for a, _ in ref], [b for _, b in ref], max_len=cfg["pair_len"]),
            "v": np.array(v, dtype=np.float32), "v0": np.array(v0, dtype=np.float32),
            "ent": np.array(ent, dtype=np.float32),
            "type": np.array(typ, dtype=np.int64), "sent": np.array(sent, dtype=np.int64),
            "valid": np.array(valid, dtype=np.float32),
        }

    @staticmethod
    def _ragged(off: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flat instance indices of the batch and, for each, its position in the batch."""
        start = off[idx]
        cnt = off[idx + 1] - start
        item = torch.repeat_interleave(torch.arange(len(idx), device=idx.device), cnt)
        first = torch.cumsum(cnt, 0) - cnt
        inst = start[item] + (torch.arange(len(item), device=idx.device) - first[item])
        return inst, item

    # ------------------------------------------------------------------ model
    def forward(self, data, idx):
        cfg = self.cfg
        B = len(idx)
        hx_seq, mx = self.encode(self.top_content, data["content"][idx])
        hx, _ = self.pool_x(hx_seq, mx)

        inst, item = self._ragged(data["off"], idx)
        N = len(inst)
        v, v0, valid = data["v"][inst], data["v0"][inst], data["valid"][inst]
        is_overall = (data["type"][inst] == CLAIM_TYPE_ID["overall"]).float()

        # claim-in-context representation, free of any LLM argument
        hc, mc = self.encode(self.top_llm, data["claim"][inst])
        c, _ = self.pool_claim(hc, mc)
        ctx, _ = self.ctx(c[:, None, :], hx_seq[item], mx[item])
        e = self.ctx_norm(c + ctx.squeeze(1))

        # argument encodings
        args = []
        if cfg["stance"] != "none":
            hp, mp = self.encode(self.top_llm, torch.cat([data["sup"][inst], data["ref"][inst]]))
            u, _ = self.pool_pair(hp, mp)
            u = u * valid.repeat(2)[:, None]
            u_sup, u_ref = u[:N], u[N:]
            if cfg["stance"] == "both":
                args = [u_ref - u_sup, u_ref * u_sup]
                gate_args = [u_sup, u_ref]
            elif cfg["stance"] == "refute":
                args, gate_args = [u_ref], [u_ref]
            else:
                args, gate_args = [u_sup], [u_sup]
        else:
            gate_args = []

        s_llm = self.mlp_a(torch.cat(args + [e], dim=-1)).squeeze(-1)
        if cfg["use_v"]:
            s_llm = s_llm + self.alpha * logit(v, 0.02)
            if cfg["use_v0"]:
                s_llm = s_llm + self.alpha0 * logit(v0, 0.02) * is_overall
        s_llm = s_llm * valid
        s_slm = self.mlp_s(torch.cat([e, hx[item]], dim=-1)).squeeze(-1)

        if cfg["use_v"]:
            feats = torch.stack([v, (v - 0.5).abs(), v0, (v0 - v).abs(), data["ent"][inst]], dim=-1)
        else:
            feats = torch.zeros(N, N_SCORE_FEATS, device=v.device)
        g_in = torch.cat([e] + gate_args + [self.type_emb(data["type"][inst]), feats], dim=-1)
        g = torch.sigmoid(self.mlp_g(g_in).squeeze(-1))
        if cfg["gate"] == "none":
            g_used = torch.ones_like(g)
        elif cfg["gate"] == "instance":
            # one gate per news item (taken from its first instance), shared by all its instances
            first = torch.zeros(B, dtype=torch.long, device=idx.device)
            first.scatter_reduce_(0, item, torch.arange(N, device=idx.device), reduce="amin", include_self=False)
            g = g[first][item]
            g_used = g
        else:
            g_used = g

        z = s_slm + g_used * s_llm
        p = torch.sigmoid(z)

        # bag-level aggregation
        log1m = torch.nn.functional.logsigmoid(-z)
        sum_log1m = torch.zeros(B, device=z.device).index_add_(0, item, log1m).clamp(max=-1e-6)
        p_nor = -torch.expm1(sum_log1m)
        cnt = torch.zeros(B, device=z.device).index_add_(0, item, torch.ones_like(z))
        if cfg["agg"] == "noisy_or":
            p_bag = p_nor
        elif cfg["agg"] == "max":
            p_bag = torch.zeros(B, device=z.device).scatter_reduce_(0, item, p, reduce="amax", include_self=False)
        elif cfg["agg"] == "mean":
            p_bag = torch.zeros(B, device=z.device).index_add_(0, item, p) / cnt
        elif cfg["agg"] == "attention":
            a = self.agg_attn(e).squeeze(-1)
            a = a - torch.zeros(B, device=z.device).scatter_reduce_(0, item, a, reduce="amax", include_self=False)[item]
            w = torch.exp(a)
            w = w / torch.zeros(B, device=z.device).index_add_(0, item, w)[item]
            p_bag = torch.sigmoid(torch.zeros(B, device=z.device).index_add_(0, item, w * z))
        else:
            raise KeyError(cfg["agg"])
        p_bag = p_bag.clamp(1e-6, 1 - 1e-6)

        s_x = self.mlp_x(hx).squeeze(-1)
        if cfg["use_global"]:
            prob = torch.sigmoid(s_x + torch.nn.functional.softplus(self.gamma) * logit(p_bag, 1e-6) + self.bias)
        else:
            prob = p_bag
        return {"prob": prob, "p_bag": p_bag, "p_nor": p_nor, "s_x": s_x, "feature": hx,
                "p": p, "g": g, "s_llm": s_llm, "s_slm": s_slm, "inst": inst, "item": item, "cnt": cnt}

    # ------------------------------------------------------------------ loss
    def gate_targets(self, out, data, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Target "the LLM took the right side" and its weight, per instance."""
        cfg = self.cfg
        inst, item = out["inst"], out["item"]
        y = data["label"][idx][item].float()
        v, v0 = data["v"][inst], data["v0"][inst]
        is_overall = data["type"][inst] == CLAIM_TYPE_ID["overall"]
        score = torch.where(is_overall, 0.5 * (v + v0), v) if cfg["use_v0"] else v
        says_fake = (score > 0.5 + cfg["delta"]).float()
        says_real = (score < 0.5 - cfg["delta"]).float()
        # probability that the instance is problematic given the bag label
        q = (out["p"] / out["p_nor"][item].clamp_min(1e-6)).clamp(0, 1).detach()
        z = torch.where(is_overall, y, torch.where(y > 0.5, q, torch.zeros_like(y)))
        target = z * says_fake + (1 - z) * says_real
        weight = data["valid"][inst].clone()
        fake_claim = (~is_overall) & (y > 0.5)
        weight = torch.where(fake_claim, weight * cfg["eta"], weight)
        return target, weight

    def loss(self, out, data, idx, epoch):
        cfg = self.cfg
        y = data["label"][idx].float()
        l_cls = bce(out["prob"], y)
        logs = {"cls": float(l_cls)}
        total = l_cls
        if cfg["use_global"] and cfg["lambda_mil"] > 0:
            l_mil = bce(out["p_bag"], y)
            total = total + cfg["lambda_mil"] * l_mil
            logs["mil"] = float(l_mil)
        if cfg["gate"] != "none" and cfg["lambda_gate"] > 0:
            target, weight = self.gate_targets(out, data, idx)
            inst, item = out["inst"], out["item"]
            is_overall = data["type"][inst] == CLAIM_TYPE_ID["overall"]
            fake_claim = (~is_overall) & (data["label"][idx][item] > 0)
            if cfg["gate"] == "instance":
                # ARG-style: only the article-level usefulness of the holistic verdict is supervised
                weight = weight * is_overall.float()
            elif not cfg["gate_fake"] or epoch < cfg["warmup"]:
                weight = torch.where(fake_claim, torch.zeros_like(weight), weight)
            l_gate = bce(out["g"], target, weight)
            total = total + cfg["lambda_gate"] * l_gate
            logs["gate"] = float(l_gate)
        return total, logs

    def item_outputs(self, out):
        B = len(out["prob"])
        g_mean = torch.zeros(B, device=out["g"].device).index_add_(0, out["item"], out["g"]) / out["cnt"]
        p_max = torch.zeros(B, device=out["p"].device).scatter_reduce_(0, out["item"], out["p"], reduce="amax", include_self=False)
        return {"p_bag": out["p_bag"], "s_x": out["s_x"], "g_mean": g_mean, "p_max": p_max}

    def instance_outputs(self, out):
        return {"p": out["p"], "g": out["g"], "s_llm": out["s_llm"], "s_slm": out["s_slm"], "inst": out["inst"]}
