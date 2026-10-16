"""Provenance keys that every table row carries."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def code_hash() -> str:
    """Git HEAD of this package's repository, or a content hash of the package when not in git."""
    here = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        h = hashlib.sha1()
        for p in sorted(here.rglob("*.py")):
            h.update(p.read_bytes())
        return h.hexdigest()[:12]


def provenance(run: str, analysis: str, version: str, level: str,
               cost_s: float = 0.0, cost_mem_gb: float = 0.0) -> dict:
    return {
        "run": run, "analysis": analysis, "analysis_version": version,
        "code_hash": code_hash(), "level": level, "cost_s": cost_s, "cost_mem_gb": cost_mem_gb,
    }
