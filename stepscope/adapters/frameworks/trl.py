"""TRL trainer output directory (SFTTrainer / GRPOTrainer / DPOTrainer)."""
from __future__ import annotations

from ...core.registry import FRAMEWORKS


@FRAMEWORKS.register("trl")
class TRLReader:
    name = "trl"

    def read_run(self, path: str):
        raise NotImplementedError

    def intervals(self, run):
        raise NotImplementedError

    def update_packages(self, run):
        raise NotImplementedError
