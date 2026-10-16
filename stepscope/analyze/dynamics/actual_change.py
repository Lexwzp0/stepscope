"""Φ(θ_to) − Φ(θ_from) per item and observable. level=exact. Interval or update package.

Registry wrapper around ``interval.observe``; the chain-one pipeline calls the functions directly."""
from __future__ import annotations

from ...core.registry import ANALYSES
from . import interval as iv


@ANALYSES.register("dynamics.actual_change")
class ActualChange:
    name = "dynamics.actual_change"
    version = "0.1.0"
    level = "exact"
    requires = {"interval", "probes"}
    writes = {"effects", "behavior"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        enc, names, bs = inputs["enc"], inputs["names"], inputs.get("batch_size", 4)
        s_from = iv.observe(inputs["model_from"], enc, names, bs)
        s_to = iv.observe(inputs["model_to"], enc, names, bs)
        nan = float("nan")
        dphi = s_from["phi"] * 0 + nan
        rows = iv.effects_rows(enc, names, s_from["phi"], s_to["phi"], dphi, step_from=inputs["step_from"],
                               step_to=inputs["step_to"], probe_set=inputs["probe_set"],
                               probe_version=inputs["probe_version"], method="none", prov=inputs["prov"])
        store.write(run_id, "effects", rows, part=inputs.get("part", "0000"))
        for step, scored in ((inputs["step_from"], s_from), (inputs["step_to"], s_to)):
            store.write(run_id, "behavior", iv.behavior_rows(enc, scored, step=step, probe_set=inputs["probe_set"],
                                                              probe_version=inputs["probe_version"], prov=inputs["prov_behavior"]),
                        part=f"{inputs.get('part', '0000')}_{step}")
