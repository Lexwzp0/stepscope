"""On-policy distillation: KL over the vocabulary on student samples."""
from __future__ import annotations

from ...core.registry import OBJECTIVES
from .base import SparseForce


@OBJECTIVES.register("opd")
class OPDObjective:
    name = "opd"

    def loss(self, model, batch):
        raise NotImplementedError

    def loss_by_source(self, model, batch) -> dict:
        raise NotImplementedError

    def force(self, model, batch) -> SparseForce:
        raise NotImplementedError
