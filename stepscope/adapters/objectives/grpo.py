"""Policy gradient with group-relative advantage, probability ratio and clipping (cf. TRL GRPOTrainer, verl compute_policy_loss_vanilla)."""
from __future__ import annotations

from ...core.registry import OBJECTIVES
from .base import SparseForce


@OBJECTIVES.register("grpo")
class GRPOObjective:
    name = "grpo"

    def loss(self, model, batch):
        raise NotImplementedError

    def loss_by_source(self, model, batch) -> dict:
        raise NotImplementedError

    def force(self, model, batch) -> SparseForce:
        raise NotImplementedError
