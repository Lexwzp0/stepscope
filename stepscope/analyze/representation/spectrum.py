"""RankMe, α-ReQ, MMD, adjacent-step update similarity, subspace overlap."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("representation.spectrum")
class Spectrum:
    name = "representation.spectrum"
    version = "0.0.1"
    level = "exact"
    requires: set[str] = set()
    writes: set[str] = set()

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
