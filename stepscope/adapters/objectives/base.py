"""ObjectiveAdapter (blueprint §五). Required: replicated loss and loss split by source.
Optional: token-level force f = -dL/dz (LDCL plugin)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class SparseForce:
    """Per response position t: candidate ids v_{t,k} and coefficients c_{t,k}; f_t = sum_k c (e_v - p_t)/T."""
    ids: Any
    coef: Any
    temperature: float = 1.0


@runtime_checkable
class ObjectiveAdapter(Protocol):
    name: str

    def loss(self, model: Any, batch: Any) -> Any:
        """Required. Replicated real loss; must match the framework's loss in tests."""

    def loss_by_source(self, model: Any, batch: Any) -> dict:
        """Required. Loss terms split by source (teacher / domain / sample) for per-source gradients."""

    def force(self, model: Any, batch: Any) -> SparseForce:
        """Optional (LDCL). Token-level force on logits."""
