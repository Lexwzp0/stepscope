"""Per-source parts of one AdamW step via OptimizerAdapter; stored only after validation against measured Δθ."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("dynamics.source_decomposition")
class SourceDecomposition:
    name = "dynamics.source_decomposition"
    version = "0.0.1"
    level = "first_order"
    requires = {"update_package", "gradients", "probes"}
    writes = {"contributions", "consistency"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
