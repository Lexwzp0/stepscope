"""∇Φ · Δθ per item and observable. level=first_order.

Interval: first-order approximation of a multi-step net update (coarse consistency check only).
Update package: strict one-step expansion. Registry wrapper around ``interval.first_order``."""
from __future__ import annotations

from ...core.registry import ANALYSES
from . import interval as iv


@ANALYSES.register("dynamics.first_order")
class FirstOrder:
    name = "dynamics.first_order"
    version = "0.1.0"
    level = "first_order"
    requires = {"interval", "probes"}
    writes = {"effects"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        enc, names, bs = inputs["enc"], inputs["names"], inputs.get("batch_size", 4)
        delta = inputs["delta"]
        phi_from, dphi, method = iv.first_order(inputs["model_from"], delta, enc, names, inputs.get("method", "auto"), bs)
        phi_to = inputs["phi_to"]
        rows = iv.effects_rows(enc, names, phi_from, phi_to, dphi, step_from=inputs["step_from"], step_to=inputs["step_to"],
                               probe_set=inputs["probe_set"], probe_version=inputs["probe_version"], method=method,
                               prov=inputs["prov"])
        store.write(run_id, "effects", rows, part=inputs.get("part", "0000"))
