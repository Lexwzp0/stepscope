"""Open-MOPD: top-K student candidates per teacher, three weightings, PPO dual clip, domain and token weights (cf. ood_ld forces.py)."""
from __future__ import annotations

from ...core.registry import OBJECTIVES
from .base import SparseForce


@OBJECTIVES.register("mopd")
class MOPDObjective:
    name = "mopd"

    def loss(self, model, batch):
        raise NotImplementedError

    def loss_by_source(self, model, batch) -> dict:
        raise NotImplementedError

    def force(self, model, batch) -> SparseForce:
        raise NotImplementedError
