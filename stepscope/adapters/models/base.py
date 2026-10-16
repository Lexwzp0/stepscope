"""ModelAdapter (blueprint §五). Required: logits, parameter groups, tokenizer.
Optional capabilities (only the LDCL plugin needs them): residual-branch inputs and readout matrix."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    name: str

    def forward(self, batch: Any) -> Any:
        """Required. Returns an object with ``.logits``."""

    def param_groups(self) -> dict[str, list]:
        """Required. Parameters grouped by layer / module name."""

    def residual_inputs(self, batch: Any) -> dict:
        """Optional (LDCL). Normalised inputs of each residual branch {layer: h_l} plus readout input hL."""

    def readout(self) -> Any:
        """Optional (LDCL). Readout matrix W of shape [V, d]."""


def supports(adapter: ModelAdapter, capability: str) -> bool:
    """Capability check used by plugins: 'residual_inputs' | 'readout'."""
    fn = getattr(adapter, capability, None)
    return callable(fn) and not getattr(fn, "__isabstractmethod__", False)
