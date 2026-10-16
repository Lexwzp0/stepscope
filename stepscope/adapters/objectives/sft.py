"""Token-level NLL: c_k = 1 on target tokens."""
from __future__ import annotations

from ...core.registry import OBJECTIVES
from .base import SparseForce


@OBJECTIVES.register("sft")
class SFTObjective:
    name = "sft"

    def loss(self, model, batch):
        raise NotImplementedError

    def loss_by_source(self, model, batch) -> dict:
        raise NotImplementedError

    def force(self, model, batch) -> SparseForce:
        raise NotImplementedError
