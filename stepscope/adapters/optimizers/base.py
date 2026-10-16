"""OptimizerAdapter (blueprint §五): decompose one real update into parts that sum to it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...core.model import UpdatePackage


@dataclass
class Decomposition:
    """Parts of Δθ keyed by source id (plus 'momentum', 'weight_decay'); ``residual`` is Δθ_measured − Σ parts."""
    parts: dict[str, Any] = field(default_factory=dict)
    residual: Any = None
    validated_against_measured: bool = False


@runtime_checkable
class OptimizerAdapter(Protocol):
    name: str

    def reconstruct(self, pkg: UpdatePackage, grads_by_source: dict) -> Any:
        """Rebuild Δθ from pre-update state, config and per-source gradients."""

    def validate(self, reconstructed: Any, pkg: UpdatePackage) -> bool:
        """Compare against the measured Δθ referenced by the package. Must pass before decompose() output is stored."""

    def decompose(self, pkg: UpdatePackage, grads_by_source: dict) -> Decomposition:
        """Conditional attribution with the real preconditioner and clip fixed (not counterfactual)."""
