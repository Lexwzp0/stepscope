"""chosen positive, rejected negative, times a model-dependent sigmoid factor (cf. TRL DPOTrainer)."""
from __future__ import annotations

from ...core.registry import OBJECTIVES
from .base import SparseForce


@OBJECTIVES.register("dpo")
class DPOObjective:
    name = "dpo"

    def loss(self, model, batch):
        raise NotImplementedError

    def loss_by_source(self, model, batch) -> dict:
        raise NotImplementedError

    def force(self, model, batch) -> SparseForce:
        raise NotImplementedError
