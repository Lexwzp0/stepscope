"""LDCL CH1/CH2 aggregated per source (cf. ood_ld ldcl_channels.py). Applicability: SGD-direction updates; AdamW → qualitative, flagged."""
from __future__ import annotations

from ...core.registry import ANALYSES


@ANALYSES.register("dynamics.ldcl_channels")
class LdclChannels:
    name = "dynamics.ldcl_channels"
    version = "0.0.1"
    level = "channel"
    requires = {"update_package", "probes"}
    writes = {"channels"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        raise NotImplementedError
