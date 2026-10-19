"""Minimal training step (blueprint §十九 校验链 2): replicated loss + backward + one SGD or AdamW step,
producing a single-step UpdatePackage (§四) with the pre-update parameters, the optimizer state before the
step, the batch, the effective configuration and the measured Δθ = θ_after − θ_before.

Pure SGD (no momentum) by default, matching the LDCL validation protocol; AdamW for the real optimizer.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import torch

from ..adapters.objectives.base import TrainBatch


@dataclass
class OptimizerConfig:
    name: str = "sgd"                          # sgd | adamw
    lr: float = 1e-2
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float | None = None

    def build(self, params):
        if self.name == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=0.0, weight_decay=self.weight_decay)
        if self.name == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, betas=tuple(self.betas), eps=self.eps,
                                     weight_decay=self.weight_decay)
        raise ValueError(f"unknown optimizer {self.name!r} (sgd | adamw)")


@dataclass
class StepPackage:
    """In-memory single-step update package. ``config`` is what ``UpdatePackage.config`` records."""
    run_id: str
    optimizer_step: int
    pre_params: dict[str, torch.Tensor]           # θ before the step (CPU, float as stored)
    delta: dict[str, torch.Tensor]                # measured Δθ = θ_after − θ_before (CPU, float32)
    optimizer_state_pre: dict                     # optimizer.state_dict() before the step
    batch: TrainBatch
    objective: str
    objective_config: dict
    optimizer: OptimizerConfig
    loss: float
    loss_by_source: dict[str, float]
    grad_norm: float
    clip_coef: float
    pre_params_ref: str | None = None             # checkpoint the pre-update parameters came from, if any
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> dict:
        return {
            "optimizer": asdict(self.optimizer), "objective": self.objective, "objective_config": self.objective_config,
            "loss": self.loss, "loss_by_source": self.loss_by_source, "grad_norm": self.grad_norm,
            "clip_coef": self.clip_coef, "grad_accum": 1, "n_rows": len(self.batch),
            "n_supervised_tokens": int(self.batch.loss_mask.sum()), "source_kind": self.batch.source_kind,
            "sources": self.batch.source_names(),
        }

    def delta_norm(self) -> float:
        return float(sum(float(v.double().pow(2).sum()) for v in self.delta.values()) ** 0.5)


def _state_to_cpu(state: dict) -> dict:
    return copy.deepcopy({k: (v.detach().cpu() if torch.is_tensor(v) else _state_to_cpu(v) if isinstance(v, dict) else v)
                          for k, v in state.items()})


def _effective_optimizer_config(optim, requested: OptimizerConfig) -> OptimizerConfig:
    """Optimizer state wins over constructor settings; clipping remains a capture setting.

    ``build`` creates one parameter group, and ``load_state_dict`` requires the same group count.
    Keep an independent snapshot so later caller edits cannot change the recorded update.
    """
    group = optim.param_groups[0]
    effective = replace(requested, lr=float(group["lr"]), weight_decay=float(group["weight_decay"]))
    if requested.name == "adamw":
        effective = replace(effective, betas=tuple(float(v) for v in group["betas"]), eps=float(group["eps"]))
    return effective


def take_step(model, objective, batch: TrainBatch, opt: OptimizerConfig, *, run_id: str = "", step: int = 0,
              optimizer_state: dict | None = None, pre_params_ref: str | None = None) -> StepPackage:
    """Run one optimizer step in place. Loaded state overrides ``opt``'s optimizer settings;
    ``max_grad_norm`` still comes from ``opt`` because clipping is outside the optimizer.
    """
    names = [k for k, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    pre = {k: p.detach().to("cpu", copy=True) for k, p in zip(names, params)}
    was_training = model.training
    model.eval()                                   # no dropout: the step must be reproducible from the package
    for p in params:
        p.requires_grad_(True)
        p.grad = None
    optim = opt.build(params)
    if optimizer_state is not None:
        optim.load_state_dict(optimizer_state)
    effective_opt = _effective_optimizer_config(optim, opt)
    state_pre = _state_to_cpu(optim.state_dict())
    try:
        batch_dev = batch.to(params[0].device)
        terms = objective.terms_from_logits(model(**batch_dev.model_kwargs()).logits, batch_dev)
        loss = terms.sum()
        by_source = {s: float(v) for s, v in objective.by_source(terms.detach(), batch_dev).items()}
        loss.backward()
        grad_norm = float(torch.sqrt(sum(p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None)))
        clip_coef = 1.0
        if opt.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, opt.max_grad_norm)
            clip_coef = min(1.0, opt.max_grad_norm / (grad_norm + 1e-6))
        optim.step()
    finally:
        for p in params:
            p.grad = None
            p.requires_grad_(False)
        model.train(was_training)
    delta = {k: (p.detach().float().cpu() - pre[k].float()) for k, p in zip(names, params)}
    del optim
    return StepPackage(run_id=run_id, optimizer_step=step, pre_params=pre, delta=delta, optimizer_state_pre=state_pre,
                       batch=batch, objective=objective.name, objective_config=objective.config(), optimizer=effective_opt,
                       loss=float(loss.detach()), loss_by_source=by_source, grad_norm=grad_norm, clip_coef=clip_coef,
                       pre_params_ref=pre_params_ref)


@torch.no_grad()
def load_params(model, params: dict[str, torch.Tensor]) -> None:
    """Copy ``params`` into ``model`` in place (names from ``named_parameters()``)."""
    for k, p in model.named_parameters():
        p.copy_(params[k].to(device=p.device, dtype=p.dtype))
