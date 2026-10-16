"""pass@k unbiased estimate, item-level CIs, paired comparison."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("behavior.stats")
class Stats:
    name = "behavior.stats"
    version = "0.0.1"
    level = "exact"
    requires: set[str] = set()
    writes: set[str] = set()

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
