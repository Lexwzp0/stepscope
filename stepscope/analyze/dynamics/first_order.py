"""∇Φ · Δθ. level=first_order. Interval: multi-step net approximation (coarse check). Update package: strict one-step expansion."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("dynamics.first_order")
class FirstOrder:
    name = "dynamics.first_order"
    version = "0.0.1"
    level = "first_order"
    requires = {"interval", "probes"}
    writes = {"effects"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
