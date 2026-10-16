"""AdamW decomposition (cf. ood_ld update_decomp.py).

With D fixed at the value produced by the actual total gradient and c the real clip coefficient:
    Δθ = -lr*wd*θ  -  lr/(1-b1^t) * b1*m*D  -  lr/(1-b1^t) * (1-b1) * c * Σ_d g_d * D
The parts sum exactly to the real update; removing a source would also change D and c.
"""
from __future__ import annotations

from ...core.registry import OPTIMIZERS
from .base import Decomposition


@OPTIMIZERS.register("adamw")
class AdamWAdapter:
    name = "adamw"

    def reconstruct(self, pkg, grads_by_source):
        raise NotImplementedError

    def validate(self, reconstructed, pkg) -> bool:
        raise NotImplementedError

    def decompose(self, pkg, grads_by_source) -> Decomposition:
        raise NotImplementedError
