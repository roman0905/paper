"""Common interface of all trainable methods.

A method owns (i) the conversion of dataset items into index/feature arrays, registering every text
it needs with the shared :class:`~clara.encoder.FrozenPrefix`, (ii) a forward pass over a batch of
item indices and (iii) its training loss.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from clara.encoder import FrozenPrefix


class Method(nn.Module):
    def __init__(self, cfg: Dict[str, Any], store: FrozenPrefix, lang: str):
        super().__init__()
        self.cfg, self.lang = cfg, lang
        # keep the store out of the module tree: it holds no trainable state
        object.__setattr__(self, "store", store)
        self.dim = store.hidden_size

    # -- data ---------------------------------------------------------------------------
    def prepare(self, items: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    # -- model --------------------------------------------------------------------------
    def forward(self, data: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def loss(self, out: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor], idx: torch.Tensor,
             epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def item_outputs(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Per-item tensors (first dim = batch) worth saving next to the prediction."""
        return {}

    def instance_outputs(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Per-instance tensors (flattened over the batch, item order preserved)."""
        return {}

    def encode(self, top: nn.Module, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, mask = self.store.fetch(ids)
        return top(hidden, mask), mask
