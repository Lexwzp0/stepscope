"""Chain one: two checkpoints → per-item actual change, first-order prediction, residual.

    ΔΦ_o = Φ_o(θ_to) − Φ_o(θ_from)            actual (exact)
    ΔΦ̂_o = ∇_θ Φ_o(θ_from)^T Δθ              first-order, Δθ = θ_to − θ_from

Both checkpoints and both first-order algorithms consume the same ``EncodedProbes`` (same tokens,
same option ids). Between consecutive checkpoints Δθ is the net effect of many optimizer steps, so the
first-order term is a first-order approximation of a multi-step net update: trajectories and a
coarse consistency check, never a per-source decomposition.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from ...observe.observables import phi_all
from ...observe.scoring import EncodedProbes, batches, forward_kwargs, last_slot_logits, score_encoded


@dataclass
class DeltaInfo:
    norm: float
    group_norms: dict[str, float]
    n_params: int


def delta_params(model_from, model_to, groups: dict[str, list[str]] | None = None, release: bool = False):
    """Δθ = θ_to − θ_from keyed by ``named_parameters()`` names (tied weights appear once).

    ``release=True`` frees each tensor of ``model_to`` as soon as its difference is taken, so the peak is two
    copies of the parameters (θ_from and θ_to shrinking while Δθ grows) instead of three; ``model_to`` is
    unusable afterwards. Names and shapes are checked before anything is released."""
    pf = dict(model_from.named_parameters())
    pt = dict(model_to.named_parameters())
    if pf.keys() != pt.keys():
        missing = sorted(set(pf) ^ set(pt))
        raise ValueError(f"parameter names differ between checkpoints: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    for k, a in pf.items():
        if a.shape != pt[k].shape:
            raise ValueError(f"shape mismatch for {k}: {tuple(a.shape)} vs {tuple(pt[k].shape)}")
    delta = {}
    for k, a in pf.items():
        b = pt[k]
        delta[k] = (b.detach().float() - a.detach().float()).to(a.device)
        if release:
            b.data = torch.empty(0, dtype=b.dtype, device=b.device)
    sq = {k: float(v.pow(2).sum()) for k, v in delta.items()}
    gnorms = {}
    if groups:
        for g, names in groups.items():
            gnorms[g] = math.sqrt(sum(sq.get(n, 0.0) for n in names))
    info = DeltaInfo(norm=math.sqrt(sum(sq.values())), group_norms=gnorms, n_params=sum(v.numel() for v in delta.values()))
    return delta, info


def observe(model, enc: EncodedProbes, names: Sequence[str], batch_size: int = 4) -> dict[str, torch.Tensor]:
    return score_encoded(model, enc, names, batch_size=batch_size)


def _tangents_for(model, delta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: delta[k].to(v.dtype).to(v.device) for k, v in model.named_parameters()}


def first_order_jvp(model, delta: dict[str, torch.Tensor], enc: EncodedProbes, names: Sequence[str],
                    batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-mode AD along Δθ. Returns (phi_from, dphi), each [len(names), N]. One forward per batch,
    answer-slot logits only."""
    from torch.func import functional_call, jvp

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    tangents = _tangents_for(model, delta)
    phis, dphis = [], []
    for sl, ids, mask, pos, opt_ids, correct in batches(enc, batch_size, next(model.parameters()).device):
        kw = forward_kwargs(ids, mask, pos)

        def f(p, _kw=kw, _opt=opt_ids, _cor=correct):
            try:
                out = functional_call(model, {**p, **buffers}, args=(), kwargs={**_kw, "logits_to_keep": 1})
            except TypeError:
                out = functional_call(model, {**p, **buffers}, args=(), kwargs=_kw)
            return phi_all(names, out.logits[:, -1, :], _opt, _cor)

        phi, dphi = jvp(f, (params,), (tangents,))
        phis.append(phi.detach().float().cpu())
        dphis.append(dphi.detach().float().cpu())
    return torch.cat(phis, dim=1), torch.cat(dphis, dim=1)


def first_order_backward(model, delta: dict[str, torch.Tensor], enc: EncodedProbes, names: Sequence[str],
                         batch_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse-mode: for each item and observable, backward then Σ_k <∂Φ/∂θ_k, Δθ_k>.
    Compatibility fallback and cross-check for the JVP path; memory ordering vs JVP is empirical."""
    params = [(k, v) for k, v in model.named_parameters()]
    for _, v in params:
        v.requires_grad_(True)
    try:
        phis, dphis = [], []
        for sl, ids, mask, pos, opt_ids, correct in batches(enc, batch_size, next(model.parameters()).device):
            logits = last_slot_logits(model, ids, mask, pos)
            phi = phi_all(names, logits, opt_ids, correct)          # [n_obs, B]
            n_obs, B = phi.shape
            d = torch.zeros(n_obs, B)
            for o in range(n_obs):
                for i in range(B):
                    grads = torch.autograd.grad(phi[o, i], [v for _, v in params], retain_graph=True, allow_unused=True)
                    acc = 0.0
                    for (k, v), g in zip(params, grads):
                        if g is not None:
                            acc += float((g.float() * delta[k]).sum())
                    d[o, i] = acc
            phis.append(phi.detach().float().cpu())
            dphis.append(d)
            del logits, phi
        return torch.cat(phis, dim=1), torch.cat(dphis, dim=1)
    finally:
        for _, v in params:
            v.requires_grad_(False)
            v.grad = None


def first_order(model, delta, enc, names, method: str = "auto", batch_size: int = 4):
    """Returns (phi_from, dphi, method_used)."""
    if method in ("jvp", "auto"):
        try:
            phi, dphi = first_order_jvp(model, delta, enc, names, batch_size)
            return phi, dphi, "jvp"
        except Exception as e:  # noqa: BLE001 - fall back only when asked to
            if method == "jvp":
                raise
            fallback_reason = f"{type(e).__name__}: {e}"
        phi, dphi = first_order_backward(model, delta, enc, names, batch_size)
        return phi, dphi, f"backward (jvp failed: {fallback_reason[:120]})"
    if method == "backward":
        phi, dphi = first_order_backward(model, delta, enc, names, batch_size)
        return phi, dphi, "backward"
    raise ValueError(f"unknown method {method!r}")


def effects_rows(enc: EncodedProbes, names: Sequence[str], phi_from: torch.Tensor, phi_to: torch.Tensor,
                 dphi: torch.Tensor, *, step_from: int, step_to: int, probe_set: str, probe_version: str,
                 method: str, prov: dict) -> list[dict]:
    rows = []
    for oi, name in enumerate(names):
        for i, item in enumerate(enc.item_ids):
            vf, vt, d = float(phi_from[oi, i]), float(phi_to[oi, i]), float(dphi[oi, i])
            rows.append(dict(step_from=step_from, step_to=step_to, probe_set=probe_set, probe_version=probe_version,
                             item=item, domain=enc.domains[i], observable=name,
                             value_from=vf, value_to=vt, actual=vt - vf, first_order=d, method=method, **prov))
    return rows


def behavior_rows(enc: EncodedProbes, scored: dict[str, torch.Tensor], *, step: int, probe_set: str,
                  probe_version: str, prov: dict, metrics=("p_correct", "argmax_in_set_correct")) -> list[dict]:
    rows = []
    for m in metrics:
        vals = scored[m]
        for i, item in enumerate(enc.item_ids):
            rows.append(dict(step=step, probe_set=probe_set, probe_version=probe_version, item=item,
                             domain=enc.domains[i], sample=0, metric=m, score=float(vals[i]), category="",
                             truncated=False, refused=False, wording="default", option_order="given", **prov))
    return rows


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
