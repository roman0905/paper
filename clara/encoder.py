"""Frozen-prefix encoder cache.

All methods in this repository follow the protocol of ARG (Hu et al., AAAI 2024): the pretrained
encoder is frozen except for its last Transformer layer. The frozen part therefore maps every input
sequence to the same hidden states in every epoch, so we run it exactly once per run (eval mode,
fp16 autocast), keep the states of all sequences in one flat GPU tensor, and train only private
copies of the last layer on top of them. This is mathematically the frozen-lower-layers setting,
minus the dropout noise of the frozen layers, and makes a run ~10x cheaper.
"""
import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TopLayers(nn.Module):
    """Trainable copy of the last ``n`` encoder layers, applied to cached hidden states."""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = copy.deepcopy(layers)
        for p in self.layers.parameters():
            p.requires_grad_(True)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ext = (1.0 - mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=ext)[0]
        return hidden


class FrozenPrefix:
    def __init__(self, model_path: str, device: str = "cuda", n_top: int = 1):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModel.from_pretrained(model_path, add_pooling_layer=False)
        self.hidden_size = model.config.hidden_size
        self._top_template = copy.deepcopy(model.encoder.layer[-n_top:])
        model.encoder.layer = model.encoder.layer[:-n_top]
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._key2id: Dict[Tuple, int] = {}
        self._specs: List[Tuple] = []          # (text, pair, max_len, pad_full)
        self._n_built = 0
        self.states = torch.empty(0, self.hidden_size, dtype=torch.float16, device=self.device)
        self.offsets = torch.zeros(1, dtype=torch.long, device=self.device)
        self.attn_len = torch.empty(0, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ registration
    def register(self, texts: Sequence[str], pairs: Optional[Sequence[str]] = None,
                 max_len: int = 170, pad_full: bool = False) -> np.ndarray:
        """Register sequences (optionally as text pairs) and return their ids."""
        ids = np.empty(len(texts), dtype=np.int64)
        for i, t in enumerate(texts):
            key = (t, None if pairs is None else pairs[i], max_len, pad_full)
            j = self._key2id.get(key)
            if j is None:
                j = len(self._specs)
                self._key2id[key] = j
                self._specs.append(key)
            ids[i] = j
        return ids

    def new_top(self) -> TopLayers:
        return TopLayers(self._top_template)

    # ------------------------------------------------------------------ frozen forward
    @torch.no_grad()
    def build(self, tokens_per_batch: int = 60000) -> None:
        """Encode every sequence registered since the last call."""
        new = self._specs[self._n_built:]
        if not new:
            return
        groups: Dict[Tuple[int, bool, bool], List[int]] = {}
        for i, (_, pair, max_len, pad_full) in enumerate(new):
            groups.setdefault((max_len, pad_full, pair is not None), []).append(i)

        input_ids: List[Optional[List[int]]] = [None] * len(new)
        type_ids: List[Optional[List[int]]] = [None] * len(new)
        n_real = np.zeros(len(new), dtype=np.int64)
        has_types = "token_type_ids" in self.tokenizer.model_input_names
        for (max_len, pad_full, is_pair), idx in groups.items():
            a = [new[i][0] for i in idx]
            b = [new[i][1] for i in idx] if is_pair else None
            enc = self.tokenizer(a, b, truncation="longest_first" if is_pair else True,
                                 max_length=max_len, padding="max_length" if pad_full else False,
                                 return_token_type_ids=has_types)
            for k, i in enumerate(idx):
                input_ids[i] = enc["input_ids"][k]
                type_ids[i] = enc["token_type_ids"][k] if has_types else None
                n_real[i] = int(sum(enc["attention_mask"][k]))

        lengths = np.array([len(x) for x in input_ids], dtype=np.int64)
        order = np.argsort(-lengths, kind="stable")
        pad_id = self.tokenizer.pad_token_id
        chunks: List[Optional[torch.Tensor]] = [None] * len(new)
        s = 0
        while s < len(order):
            L = int(lengths[order[s]])
            bs = max(1, tokens_per_batch // max(L, 1))
            idx = order[s:s + bs]
            s += bs
            ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            tt = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                n = int(lengths[i])
                ids[r, :n] = torch.tensor(input_ids[i])
                att[r, :int(n_real[i])] = 1
                if has_types:
                    tt[r, :n] = torch.tensor(type_ids[i])
            kw = dict(input_ids=ids.to(self.device), attention_mask=att.to(self.device))
            if has_types:
                kw["token_type_ids"] = tt.to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                h = self.model(**kw).last_hidden_state
            h = h.to(torch.float16)
            for r, i in enumerate(idx):
                chunks[i] = h[r, :int(lengths[i])]
        self.states = torch.cat([self.states] + chunks, dim=0)
        new_off = self.offsets[-1] + torch.cumsum(torch.as_tensor(lengths, device=self.device), 0)
        self.offsets = torch.cat([self.offsets, new_off])
        self.attn_len = torch.cat([self.attn_len, torch.as_tensor(n_real, device=self.device)])
        self._n_built = len(self._specs)

    def release_model(self) -> None:
        """Free the frozen layers once everything needed has been encoded."""
        self.model = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ lookup
    def fetch(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ids: LongTensor [n] -> (hidden [n, L, H] float32, mask [n, L] bool)."""
        ids = ids.to(self.device)
        start = self.offsets[ids]
        length = self.offsets[ids + 1] - start
        L = int(length.max().item())
        pos = torch.arange(L, device=self.device)[None, :]
        index = (start[:, None] + pos).clamp_(max=self.states.shape[0] - 1)
        hidden = self.states[index].float()
        mask = pos < self.attn_len[ids][:, None]
        stored = pos < length[:, None]
        hidden = hidden * stored[:, :, None]
        return hidden, mask

    @property
    def n_sequences(self) -> int:
        return len(self._specs)

    def memory_gb(self) -> float:
        return self.states.numel() * 2 / 1e9
