"""Data model (blueprint §四). Plain dataclasses; no torch dependency.

Every record that reaches a table carries the provenance keys defined in
``provenance.py``: run, step or checkpoint, probe-set version, analysis name and
version, code hash, approximation level and cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Level = Literal["exact", "first_order", "channel", "counterfactual"]


@dataclass(frozen=True)
class Source:
    """A learning source. Hierarchical: teacher / domain / sample / token."""
    kind: Literal["teacher", "domain", "sample", "token", "momentum", "weight_decay"]
    id: str
    parent: Optional["Source"] = None


@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    step: int
    path: str
    param_hash: Optional[str] = None


@dataclass
class Run:
    """One training run: framework, config hash, code version, checkpoints, source log."""
    id: str
    framework: Literal["verl", "trl", "other"]
    config_hash: str
    code_version: str
    checkpoints: list[Checkpoint] = field(default_factory=list)
    source_log_path: Optional[str] = None  # per-step (sample -> domain/teacher/weight)
    meta: dict = field(default_factory=dict)  # framework-specific scan details (pending merges, roots)


@dataclass(frozen=True)
class IntervalRecord:
    """Two checkpoints plus the source log in between. Default input (§三 原则 1)."""
    run_id: str
    ckpt_from: Checkpoint
    ckpt_to: Checkpoint
    source_log_path: Optional[str] = None


@dataclass(frozen=True)
class UpdatePackage:
    """One real optimizer step (§四). Carries BOTH sides of the update.

    ``post_params_ref`` (post-update parameters, or the measured Δθ) is mandatory:
    a reconstructed update is validated against it before any decomposition is
    written to ``contributions``.
    """
    run_id: str
    optimizer_step: int
    pre_params_ref: str
    pre_optimizer_state_ref: str  # m, v, step counter (Adam-family)
    post_params_ref: str          # post-update params or measured Δθ
    microbatches_ref: str         # every micro-batch consumed by this step
    config: dict                  # lr, beta1, beta2, weight_decay, clip, loss normalisation, grad_accum
    gradients_ref: Optional[str] = None  # per-source / per-sample gradients captured by dattri-llm


@dataclass(frozen=True, eq=False)
class ProbeItem:
    """One multiple-choice probe item. ``options`` are already in display order; ``answer_idx`` indexes them."""
    id: str
    domain: str
    question: str
    options: tuple[str, ...]
    answer_idx: int
    meta: dict = field(default_factory=dict)


@dataclass
class ProbeSet:
    name: str
    version: str          # content hash of the probe file
    items: list[ProbeItem]
    source_path: str = ""
    frozen_date: str = ""
    template_hash: str = ""


@dataclass(frozen=True)
class Observable:
    """Differentiable quantity Φ_o(θ): enters first-order prediction and channel analyses."""
    name: str  # e.g. phi_fmt, phi_ans, ref_logprob, option_margin


@dataclass(frozen=True)
class Metric:
    """Behavioural metric (generation + scorer): actual change and validation only."""
    name: str  # e.g. accuracy, refusal_rate, truncation_rate, failure_category


@dataclass
class AnalysisRun:
    analysis: str
    version: str
    level: Level
    inputs: dict
    code_hash: str
    cost_s: float = 0.0
    cost_mem_gb: float = 0.0
    output_tables: dict[str, str] = field(default_factory=dict)
