"""Readout geometry comparison between checkpoints (cf. LDCL plasticity)."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("representation.readout_geometry")
class ReadoutGeometry:
    name = "representation.readout_geometry"
    version = "0.0.1"
    level = "exact"
    requires: set[str] = set()
    writes: set[str] = set()

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
