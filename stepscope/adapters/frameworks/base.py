"""FrameworkReader: turn a training output directory into a Run, intervals and update packages."""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ...core.model import IntervalRecord, Run, UpdatePackage


@runtime_checkable
class FrameworkReader(Protocol):
    name: str

    def read_run(self, path: str) -> Run: ...
    def intervals(self, run: Run) -> Iterable[IntervalRecord]: ...
    def update_packages(self, run: Run) -> Iterable[UpdatePackage]: ...
