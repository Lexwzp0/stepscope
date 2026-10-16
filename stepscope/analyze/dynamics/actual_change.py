"""Φ(θ_to) − Φ(θ_from). level=exact. interval or update package."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("dynamics.actual_change")
class ActualChange:
    name = "dynamics.actual_change"
    version = "0.0.1"
    level = "exact"
    requires = {"interval", "probes"}
    writes = {"effects"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
