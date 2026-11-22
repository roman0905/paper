"""Frozen-prefix encoder cache.

All methods in this repository follow the protocol of ARG (Hu et al., AAAI 2024): the pretrained
encoder is frozen except for its last Transformer layer. The frozen part therefore maps every input
sequence to the same hidden states in every epoch, so we run it exactly once per session (eval
mode, fp16 autocast), keep the states of all sequences in flat GPU tensors, and train only private
copies of the last layer on top of them.

The one difference from running the full encoder in every step is that the frozen layers see no
dropout noise during training.  ``live=True`` disables the cache and recomputes the frozen layers
on the fly in the module's current train/eval mode, which reproduces the official behaviour
exactly; it is used to verify that caching does not change the conclusions.
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
    def __init__(self, model_path: str, device: str = "cuda", n_top: int = 1, live: bool = False):
        self.device = torch.device(device)
        self.live = live
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModel.from_pretrained(model_path, add_pooling_layer=False)
        self.hidden_size = model.config.hidden_size
        self._top_template = copy.deepcopy(model.encoder.layer[-n_top:])
        model.encoder.layer = model.encoder.layer[:-n_top]
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.has_types = "token_type_ids" in self.tokenizer.model_input_names

        self._key2id: Dict[Tuple, int] = {}
        self._specs: List[Tuple] = []          # (text, pair, max_len, pad_full)
        self._n_built = 0
        self.blocks: List[torch.Tensor] = []   # hidden states (cached mode) or token ids (live mode)
        self.type_blocks: List[torch.Tensor] = []
        z = lambda: torch.empty(0, dtype=torch.long, device=self.device)
        self.block_of, self.start, self.length, self.attn_len = z(), z(), z(), z()

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

    def set_training(self, mode: bool) -> None:
        """Only relevant in live mode: put the frozen layers in train (dropout) or eval mode."""
        if self.live and self.model is not None:
            self.model.train(mode)

    # ------------------------------------------------------------------ frozen forward
    def _frozen(self, ids: torch.Tensor, att: torch.Tensor, tt: Optional[torch.Tensor]) -> torch.Tensor:
        kw = dict(input_ids=ids, attention_mask=att)
        if self.has_types:
            kw["token_type_ids"] = tt
        return self.model(**kw).last_hidden_state

    @torch.no_grad()
    def build(self, tokens_per_batch: int = 60000) -> None:
        """Tokenise (and, unless live, encode) every sequence registered since the last call."""
        new = self._specs[self._n_built:]
        if not new:
            return
        groups: Dict[Tuple[int, bool, bool], List[int]] = {}
        for i, (_, pair, max_len, pad_full) in enumerate(new):
            groups.setdefault((max_len, pad_full, pair is not None), []).append(i)

        input_ids: List[Optional[List[int]]] = [None] * len(new)
        type_ids: List[Optional[List[int]]] = [None] * len(new)
        n_real = np.zeros(len(new), dtype=np.int64)
        for (max_len, pad_full, is_pair), idx in groups.items():
            a = [new[i][0] for i in idx]
            b = [new[i][1] for i in idx] if is_pair else None
            enc = self.tokenizer(a, b, truncation="longest_first" if is_pair else True,
                                 max_length=max_len, padding="max_length" if pad_full else False,
                                 return_token_type_ids=self.has_types)
            for k, i in enumerate(idx):
                input_ids[i] = enc["input_ids"][k]
                type_ids[i] = enc["token_type_ids"][k] if self.has_types else None
                n_real[i] = int(sum(enc["attention_mask"][k]))

        lengths = np.array([len(x) for x in input_ids], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        total = int(lengths.sum())
        flat_ids = torch.tensor([t for x in input_ids for t in x], dtype=torch.long, device=self.device)
        flat_types = None
        if self.has_types:
            flat_types = torch.tensor([t for x in type_ids for t in x], dtype=torch.long, device=self.device)

        if self.live:
            block = flat_ids
        else:
            block = torch.empty(total, self.hidden_size, dtype=torch.float16, device=self.device)
            order = np.argsort(-lengths, kind="stable")
            pad_id = self.tokenizer.pad_token_id
            s = 0
            while s < len(order):
                L = int(lengths[order[s]])
                bs = max(1, tokens_per_batch // max(L, 1))
                idx = order[s:s + bs]
                s += bs
                st = torch.as_tensor(starts[idx], device=self.device)
                ln = torch.as_tensor(lengths[idx], device=self.device)
                nr = torch.as_tensor(n_real[idx], device=self.device)
                pos = torch.arange(L, device=self.device)[None, :]
                stored = pos < ln[:, None]
                index = (st[:, None] + pos).clamp_(max=total - 1)
                ids = torch.where(stored, flat_ids[index], torch.full_like(index, pad_id))
                tt = torch.where(stored, flat_types[index], torch.zeros_like(index)) if self.has_types else None
                att = (pos < nr[:, None]).long()
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                    h = self._frozen(ids, att, tt)
                block[index[stored]] = h.to(torch.float16)[stored]

        b = len(self.blocks)
        self.blocks.append(block)
        self.type_blocks.append(flat_types)
        dev = self.device
        self.block_of = torch.cat([self.block_of, torch.full((len(new),), b, dtype=torch.long, device=dev)])
        self.start = torch.cat([self.start, torch.as_tensor(starts, device=dev)])
        self.length = torch.cat([self.length, torch.as_tensor(lengths, device=dev)])
        self.attn_len = torch.cat([self.attn_len, torch.as_tensor(n_real, device=dev)])
        self._n_built = len(self._specs)

    # ------------------------------------------------------------------ lookup
    def fetch(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ids: LongTensor [n] -> (hidden [n, L, H] float32, mask [n, L] bool)."""
        ids = ids.to(self.device)
        blk, start, length = self.block_of[ids], self.start[ids], self.length[ids]
        L = int(length.max().item())
        pos = torch.arange(L, device=self.device)[None, :]
        stored = pos < length[:, None]
        mask = pos < self.attn_len[ids][:, None]
        index = start[:, None] + pos
        blocks = torch.unique(blk).tolist()

        def gather(source: List[torch.Tensor]) -> torch.Tensor:
            """Rows ``index`` of the per-block flat tensors (positions past a sequence are junk)."""
            if len(blocks) == 1:
                src = source[blocks[0]]
                return src[index.clamp(max=src.shape[0] - 1)]
            shape = (len(ids), L) + tuple(source[blocks[0]].shape[1:])
            out = torch.empty(shape, dtype=source[blocks[0]].dtype, device=self.device)
            for b in blocks:
                sel = (blk == b).nonzero(as_tuple=True)[0]
                out[sel] = source[b][index[sel].clamp(max=source[b].shape[0] - 1)]
            return out

        if not self.live:
            return gather(self.blocks).float() * stored[:, :, None], mask

        tok = torch.where(stored, gather(self.blocks), torch.full_like(index, self.tokenizer.pad_token_id))
        tt = None
        if self.has_types:
            tt = torch.where(stored, gather(self.type_blocks), torch.zeros_like(index))
        with torch.no_grad():
            hidden = self._frozen(tok, mask.long(), tt).float()
        return hidden * stored[:, :, None], mask

    @property
    def n_sequences(self) -> int:
        return len(self._specs)

    def memory_gb(self) -> float:
        return sum(b.numel() * b.element_size() for b in self.blocks) / 1e9
