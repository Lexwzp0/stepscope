"""Counterfactual estimates via bergson (SOURCE, approx-unrolling) and dattri-llm (TracIn, IF, LESS). Needs training gradients/curvature, lr, steps, optimizer state. Never added to effects."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("interventions.data_intervention")
class DataIntervention:
    name = "interventions.data_intervention"
    version = "0.0.1"
    level = "counterfactual"
    requires = {"gradients"}
    writes = {"interventions"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
