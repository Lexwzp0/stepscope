"""Analysis plugin protocol (blueprint §五/§六). Consumes standard inputs, writes standard tables."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Analysis(Protocol):
    name: str
    version: str
    level: str            # exact | first_order | channel | counterfactual
    requires: set[str]    # subset of {"interval", "update_package", "probes", "gradients"}
    writes: set[str]      # table names

    def run(self, inputs: dict, probes: list, store: Any, run_id: str) -> None: ...
