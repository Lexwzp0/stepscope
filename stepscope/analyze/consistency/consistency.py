"""Prediction vs actual: corr, sign agreement, two residuals (actual−pred, pred−Σparts). Decompositions only."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("consistency.consistency")
class Consistency:
    name = "consistency.consistency"
    version = "0.0.1"
    level = "exact"
    requires = {"effects", "contributions"}
    writes = {"consistency"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
