"""Per-item trajectories and flips (correct-set turnover)."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("behavior.trajectory")
class Trajectory:
    name = "behavior.trajectory"
    version = "0.0.1"
    level = "exact"
    requires = {"interval", "probes"}
    writes = {"behavior"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
