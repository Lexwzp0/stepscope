"""First study: MOPD OOD spillover — which teacher helps or hurts which untrained domain, when, through what."""
from __future__ import annotations

from ...core.registry import STUDIES
from ..base import ComparisonProtocol, Study

OOD_SPILLOVER = Study(
    name="ood_spillover",
    probe_sets=[],  # to decide: MOPD's two OOD suites + general-capability suite
    analyses=[
        "behavior.trajectory", "dynamics.actual_change", "dynamics.first_order",
        "dynamics.source_decomposition", "consistency.consistency",
    ],
    protocol=ComparisonProtocol(
        baselines=["initial_student", "each_teacher", "single_teacher_opd"],
        controls={"budget": "matched"},
    ),
    report_template="source_x_probe_timeline",
)

STUDIES.register("ood_spillover")(OOD_SPILLOVER)
