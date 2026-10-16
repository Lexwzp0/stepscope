"""Assemble and persist UpdatePackage artefacts (skeleton)."""
from __future__ import annotations

from ..core.model import UpdatePackage


class PackageWriter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def write(self, pkg: UpdatePackage) -> str:
        raise NotImplementedError
