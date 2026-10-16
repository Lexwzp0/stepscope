"""verl output directory: checkpoints + per-step source log. Note: several optimizer steps per global step (ppo_epochs × mini-batches)."""
from __future__ import annotations

from ...core.registry import FRAMEWORKS


@FRAMEWORKS.register("verl")
class VERLReader:
    name = "verl"

    def read_run(self, path: str):
        raise NotImplementedError

    def intervals(self, run):
        raise NotImplementedError

    def update_packages(self, run):
        raise NotImplementedError
