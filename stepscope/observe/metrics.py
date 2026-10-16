"""Behavioural metrics on generations (cf. ood_ld scorers.py): accuracy, refusal, truncation, failure category."""
from __future__ import annotations


def score_generation(text: str, item) -> dict:
    raise NotImplementedError
