"""ProbeSet loading, versioning and frozen rendering (date, template, seed)."""
from __future__ import annotations

from ..core.model import ProbeSet


def load_probe_set(path: str) -> ProbeSet:
    raise NotImplementedError
