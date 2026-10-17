"""Persist and load single-step update packages (blueprint §四 UpdatePackage).

Layout of a package directory::

    package.json               UpdatePackage fields, effective configuration, file hashes
    delta.safetensors          measured Δθ = θ_after − θ_before (post_params_ref)
    pre_params.safetensors     θ_before, unless it is a checkpoint referenced by pre_params_ref
    optimizer_state_pre.pt     optimizer state before the step (m, v, step counter for Adam)
    batch.pt                   every row the step consumed, with sources and objective inputs
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ..core.model import UpdatePackage
from .batch import load_batch, save_batch
from .train_step import OptimizerConfig, StepPackage


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _save_tensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(path))


def _load_tensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


class PackageWriter:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)

    def write(self, pkg: StepPackage, save_pre_params: bool | None = None) -> UpdatePackage:
        d = self.out_dir
        d.mkdir(parents=True, exist_ok=True)
        _save_tensors(pkg.delta, d / "delta.safetensors")
        save_pre = pkg.pre_params_ref is None if save_pre_params is None else save_pre_params
        if save_pre:
            _save_tensors(pkg.pre_params, d / "pre_params.safetensors")
        torch.save(pkg.optimizer_state_pre, d / "optimizer_state_pre.pt")
        save_batch(pkg.batch, d / "batch.pt")
        ref = UpdatePackage(
            run_id=pkg.run_id, optimizer_step=pkg.optimizer_step,
            pre_params_ref="pre_params.safetensors" if save_pre else str(pkg.pre_params_ref),
            pre_optimizer_state_ref="optimizer_state_pre.pt", post_params_ref="delta.safetensors",
            microbatches_ref="batch.pt", config=pkg.config,
        )
        files = [f for f in ("delta.safetensors", "pre_params.safetensors", "optimizer_state_pre.pt", "batch.pt")
                 if (d / f).exists()]
        doc = {**asdict(ref), "post_params_kind": "measured_delta", "delta_norm": pkg.delta_norm(),
               "sha1": {f: _sha1(d / f) for f in files}, "meta": pkg.meta}
        (d / "package.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return ref


def load_package(pkg_dir: str | Path, model=None) -> StepPackage:
    """Load a package directory. When the pre-update parameters are a referenced checkpoint, pass the model
    loaded from it as ``model`` (its parameters become ``pre_params``)."""
    d = Path(pkg_dir)
    doc = json.loads((d / "package.json").read_text(encoding="utf-8"))
    cfg = doc["config"]
    if (d / "pre_params.safetensors").exists():
        pre = _load_tensors(d / "pre_params.safetensors")
        pre_ref = None
    else:
        if model is None:
            raise ValueError(f"{d}: pre-update parameters live in {doc['pre_params_ref']}; pass the model loaded from it")
        pre = {k: p.detach().to("cpu", copy=True) for k, p in model.named_parameters()}
        pre_ref = doc["pre_params_ref"]
    opt = cfg["optimizer"]
    return StepPackage(
        run_id=doc["run_id"], optimizer_step=int(doc["optimizer_step"]), pre_params=pre,
        delta=_load_tensors(d / "delta.safetensors"),
        optimizer_state_pre=torch.load(d / "optimizer_state_pre.pt", map_location="cpu", weights_only=False),
        batch=load_batch(d / "batch.pt"), objective=cfg["objective"], objective_config=cfg["objective_config"],
        optimizer=OptimizerConfig(**{**opt, "betas": tuple(opt["betas"])}), loss=cfg["loss"],
        loss_by_source=cfg["loss_by_source"], grad_norm=cfg["grad_norm"], clip_coef=cfg["clip_coef"],
        pre_params_ref=pre_ref, meta=doc.get("meta", {}),
    )
