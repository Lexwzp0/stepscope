"""ModelAdapter (blueprint §五). Required: logits, parameter groups, feature sites, tokenizer.

Optional capabilities are declared explicitly in ``capabilities``; ``supports()`` reads that declaration
and nothing else. An adapter that keeps an optional method as a placeholder simply does not declare it,
so no plugin ever calls it.

    hidden_states     residual stream after every block (representation analyses)
    readout           readout matrix W [V, d] (LDCL plugin)
    residual_inputs   normalised input of every residual branch plus the readout input (LDCL plugin)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

CAPABILITIES = frozenset({"hidden_states", "readout", "residual_inputs"})


@dataclass
class ResidualInputs:
    """States recorded in one forward pass, every tensor [B, T, d] except ``logits`` [B, T, V].

    ``branches`` maps a feature site (e.g. ``layers.0.attn_in``) to the normalised input of that residual
    branch; ``final`` is the input of the readout layer (output of the final norm)."""
    branches: dict[str, Any] = field(default_factory=dict)
    final: Any = None
    logits: Any = None


@runtime_checkable
class ModelAdapter(Protocol):
    name: str
    capabilities: frozenset[str]

    def forward(self, batch: Any) -> Any:
        """Required. Returns an object with ``.logits``."""

    def param_groups(self) -> dict[str, list]:
        """Required. Parameters grouped by layer / module name."""

    def feature_sites(self) -> dict[str, str]:
        """Required. Feature site -> module name: residual-branch inputs, final norm output, readout layer."""

    # Optional; called only when declared in ``capabilities``.
    def hidden_states(self, batch: Any) -> tuple:
        """``hidden_states``: residual stream entering each block plus the final output."""

    def residual_inputs(self, batch: Any) -> ResidualInputs:
        """``residual_inputs``: normalised inputs of each residual branch and the readout input."""

    def readout(self) -> Any:
        """``readout``: readout matrix W of shape [V, d]."""


def supports(adapter: Any, capability: str) -> bool:
    """True iff ``adapter`` declares ``capability``. Never inspects methods."""
    if capability not in CAPABILITIES:
        raise ValueError(f"unknown capability {capability!r}; known: {sorted(CAPABILITIES)}")
    return capability in frozenset(getattr(adapter, "capabilities", frozenset()))


def missing_capabilities(adapter: Any, required: Iterable[str]) -> list[str]:
    """Model capabilities in ``required`` that ``adapter`` does not declare (other kinds of requirement ignored)."""
    return sorted(c for c in required if c in CAPABILITIES and not supports(adapter, c))
