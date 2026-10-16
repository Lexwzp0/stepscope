"""Study template (blueprint §一/§五): probes + analyses + comparison protocol + report template."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComparisonProtocol:
    """What is compared against what, under which budget control (e.g. MOPD vs initial, vs each teacher, vs single-teacher OPD)."""
    baselines: list[str] = field(default_factory=list)
    controls: dict = field(default_factory=dict)


@dataclass
class Study:
    name: str
    probe_sets: list[str]
    analyses: list[str]
    protocol: ComparisonProtocol
    report_template: str
