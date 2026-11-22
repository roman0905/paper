"""Shared building blocks."""
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAttention(nn.Module):
    """Additive attention pooling over a sequence: scores = w^T h, masked softmax, weighted sum."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.score(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask.bool(), float("-inf"))
        a = torch.softmax(s, dim=-1)
        return torch.einsum("bl,blh->bh", a, x), a


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float, out_dim: int = 1):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """Multi-head scaled dot-product attention with separate q/k/v projections and an output
    projection; ``mask`` marks valid key positions. Returns [B, Lq, H]."""

    def __init__(self, dim: int, heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Lq, _ = query.shape
        Lm = memory.shape[1]
        q = self.q(query).view(B, Lq, self.h, self.dk).transpose(1, 2)
        k = self.k(memory).view(B, Lm, self.h, self.dk).transpose(1, 2)
        v = self.v(memory).view(B, Lm, self.h, self.dk).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            s = s.masked_fill(~mask.bool()[:, None, None, :], float("-inf"))
        a = self.drop(F.softmax(s, dim=-1))
        out = torch.matmul(a, v).transpose(1, 2).contiguous().view(B, Lq, self.h * self.dk)
        return self.o(out), a


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


def logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def bce(prob: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None,
        eps: float = 1e-6) -> torch.Tensor:
    """Binary cross-entropy on probabilities with soft targets; mean (or weighted mean)."""
    prob = prob.clamp(eps, 1.0 - eps)
    loss = -(target * torch.log(prob) + (1.0 - target) * torch.log1p(-prob))
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum().clamp_min(eps)
