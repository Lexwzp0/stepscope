"""Registries for the three extension points: adapters, analyses, studies (blueprint §三 原则 8)."""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, object] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} already registered")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str):
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; known: {sorted(self._items)}") from None

    def names(self) -> list[str]:
        return sorted(self._items)


MODELS = Registry("model adapter")
OBJECTIVES = Registry("objective adapter")
OPTIMIZERS = Registry("optimizer adapter")
FRAMEWORKS = Registry("framework reader")
ANALYSES = Registry("analysis")
STUDIES = Registry("study")
