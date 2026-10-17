"""Analysis plugin protocol (blueprint §五/§六). Consumes standard inputs, writes standard tables.

``requires`` mixes input kinds (``INPUT_KINDS``) and model capabilities (``adapters.models.base.CAPABILITIES``).
``unmet_requirements`` says what is missing; ``run_analysis`` skips the analysis in that case and records the
reason in the run manifest instead of failing at runtime."""
from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from ..adapters.models.base import CAPABILITIES, missing_capabilities

INPUT_KINDS = frozenset({"interval", "update_package", "probes", "gradients", "effects"})


@runtime_checkable
class Analysis(Protocol):
    name: str
    version: str
    level: str            # exact | first_order | channel | counterfactual
    requires: set[str]    # input kinds plus model capabilities, e.g. {"update_package", "probes", "readout"}
    writes: set[str]      # table names

    def run(self, inputs: dict, probes: list, store: Any, run_id: str) -> None: ...


def unmet_requirements(analysis: Any, available_inputs: Iterable[str], adapter: Any = None) -> list[str]:
    """Human-readable reasons why ``analysis`` cannot run; empty when everything it requires is present."""
    req = set(getattr(analysis, "requires", ()))
    unknown = req - INPUT_KINDS - CAPABILITIES
    if unknown:
        raise ValueError(f"{analysis.name}: unknown requirement(s) {sorted(unknown)}")
    have = set(available_inputs)
    reasons = [f"input {k!r} not available" for k in sorted((req & INPUT_KINDS) - have)]
    caps = req & CAPABILITIES
    if caps:
        if adapter is None:
            reasons += [f"model capability {c!r} required but no model adapter given" for c in sorted(caps)]
        else:
            name = getattr(adapter, "model_type", "") or getattr(adapter, "name", "model")
            reasons += [f"model {name!r} does not declare capability {c!r}" for c in missing_capabilities(adapter, caps)]
    return reasons


def run_analysis(analysis: Any, inputs: dict, probes: list, store: Any, run_id: str,
                 available_inputs: Iterable[str], adapter: Any = None, context: str | None = None) -> bool:
    """Run ``analysis`` if its requirements are met; otherwise record the skip in the manifest. Returns ran?"""
    reasons = unmet_requirements(analysis, available_inputs, adapter)
    if reasons:
        store.record_analysis(run_id, analysis.name, analysis.version, status="skipped", reasons=reasons, context=context)
        return False
    analysis.run(inputs, probes, store, run_id)
    store.record_analysis(run_id, analysis.name, analysis.version, status="ran", context=context)
    return True
