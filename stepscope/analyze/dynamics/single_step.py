"""Single-step check chain on one update package (blueprint §十九 校验链 1, 2):

    CH1 + CH2   →   exact first order along the batch's own update   →   actual change after the step

All three refer to the same batch and the same update:

- actual:       Φ(θ + Δθ_measured) − Φ(θ), Δθ_measured from the package;
- first order:  JVP at θ with tangent Δθ_u. SGD: Δθ_u = −η (c ∇L_u + λ θ) rebuilt from the batch, the
                objective and the recorded clip coefficient c, and validated against the measured Δθ before use
                (fallback: the measured Δθ, recorded). AdamW: the measured Δθ;
- channels:     η c S_u(o) with S = CH1 + CH2 per source and site (SGD only; needs the adapter's ``readout`` and
                ``residual_inputs``). Diagnostic: η c S1 must equal the JVP along the readout part of Δθ_u.

Under SGD the first-order effect also splits exactly by source, Δθ_u = Σ_d −η c ∇L_d (+ weight decay), written
to ``contributions`` with target = first_order. Everything model- or loss-specific goes through the adapters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from ...adapters.models.base import missing_capabilities
from ...capture.train_step import StepPackage, load_params
from ...observe.scoring import EncodedProbes
from ..consistency.consistency import summarize
from . import interval as iv
from . import ldcl_channels as ch

VERSION = "0.1.0"
CHANNEL_CAPS = ("readout", "residual_inputs")


@dataclass
class ChainResult:
    phi_from: torch.Tensor
    phi_to: torch.Tensor
    first_order: torch.Tensor
    channels: torch.Tensor | None                    # [n_obs, N] Σ ch1 + ch2, or None when skipped
    channel_preds: dict | None
    contributions: dict[str, torch.Tensor]           # component -> [n_obs, N]
    scored_from: dict
    scored_to: dict
    diagnostics: dict = field(default_factory=dict)
    skipped: dict[str, list[str]] = field(default_factory=dict)
    costs: dict[str, float] = field(default_factory=dict)


def _grads(model, objective, batch, by_source: bool):
    """Yields (source, {name: ∂L_source/∂θ}) one source at a time (one forward, graph retained), or
    ("all", ∇L) when not splitting."""
    names = [k for k, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    for p in params:
        p.requires_grad_(True)
    try:
        terms = objective.terms_from_logits(model(**batch.model_kwargs()).logits, batch)
        parts = objective.by_source(terms, batch) if by_source else {"all": terms.sum()}
        keys = list(parts)
        for i, s in enumerate(keys):
            gs = torch.autograd.grad(parts[s], params, retain_graph=i < len(keys) - 1, allow_unused=True)
            yield s, {k: (g.detach().float() if g is not None else torch.zeros_like(p, dtype=torch.float32))
                      for k, p, g in zip(names, params, gs)}
    finally:
        for p in params:
            p.requires_grad_(False)
            p.grad = None


def _max_abs(d: dict[str, torch.Tensor]) -> float:
    return max((float(v.abs().max()) for v in d.values() if v.numel()), default=0.0)


def _norm(d: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values()))


def run_chain(adapter, objective, pkg: StepPackage, enc: EncodedProbes, names: Sequence[str], *, batch_size: int = 4,
              by_source: bool = True, channels: bool = True, recon_rtol: float = 1e-3) -> ChainResult:
    model = adapter.model
    dev = adapter.device
    opt = pkg.optimizer
    sgd = opt.name == "sgd"
    batch = pkg.batch.to(dev)
    eta = opt.lr * pkg.clip_coef
    costs, diag, skipped = {}, {}, {}

    load_params(model, pkg.pre_params)
    with iv.Timer() as t:
        s_from = iv.observe(model, enc, names, batch_size)
        with torch.no_grad():
            for k, p in model.named_parameters():
                p.add_(pkg.delta[k].to(device=p.device, dtype=p.dtype))
        s_to = iv.observe(model, enc, names, batch_size)
        load_params(model, pkg.pre_params)
    costs["observe"] = t.seconds

    contributions: dict[str, torch.Tensor] = {}
    measured = pkg.delta
    if sgd:
        with iv.Timer() as t:
            total = None
            split = by_source and len(batch.source_names()) > 0
            for src, g in _grads(model, objective, batch, split):
                for v in g.values():               # in place: one source gradient plus the running total
                    v.mul_(-eta)
                if split:
                    _, contributions[src] = iv.first_order_jvp(model, g, enc, names, batch_size)
                if total is None:
                    total = g
                else:
                    for k, v in g.items():
                        total[k].add_(v)
                    del g
            readout_tangent = None
            ro = adapter.feature_sites().get("readout", "") + ".weight"
            if ro in total:                         # a tied readout is listed under the input embedding: no check
                readout_tangent = {ro: total[ro].clone()}
            if opt.weight_decay:
                wd = {k: -opt.lr * opt.weight_decay * p.detach().float() for k, p in model.named_parameters()}
                if split:
                    _, contributions["weight_decay"] = iv.first_order_jvp(model, wd, enc, names, batch_size)
                for k, v in wd.items():
                    total[k].add_(v)
                del wd
        costs["gradients_and_sources"] = t.seconds
        err = max(float((total[k].cpu() - measured[k].float()).abs().max()) for k in total)
        scale = _max_abs(measured)
        # the measured Δθ carries the rounding of θ_after in the parameter dtype
        eps = max(torch.finfo(p.dtype).eps for p in model.parameters())
        atol = 4 * eps * _max_abs(pkg.pre_params)
        ok = err <= recon_rtol * scale + atol
        diag["reconstruction"] = {"max_abs_err": err, "max_abs_measured": scale, "rounding_atol": atol,
                                  "rel_err": err / scale if scale else float("nan"), "passed": ok}
        tangent = total if ok else {k: v.to(dev) for k, v in measured.items()}
        diag["tangent"] = "reconstructed_sgd" if ok else "measured (reconstruction failed)"
        if not ok:
            contributions = {}
            skipped["dynamics.sgd_source_decomposition"] = ["rebuilt SGD update does not match the measured Δθ"]
    else:
        tangent = {k: v.to(dev) for k, v in measured.items()}
        diag["tangent"] = "measured"
        readout_tangent = None
        skipped["dynamics.sgd_source_decomposition"] = [f"optimizer {opt.name!r}: per-source parts need the "
                                                        "optimizer adapter's decomposition"]
    diag["tangent_norm"] = _norm(tangent)
    diag["delta_norm"] = pkg.delta_norm()

    with iv.Timer() as t:
        phi_from2, dphi = iv.first_order_jvp(model, tangent, enc, names, batch_size)
    costs["first_order"] = t.seconds
    diag["phi_from_recompute_max_gap"] = float((phi_from2 - s_from["phi"]).abs().max())

    preds, ch_total = None, None
    reasons = []
    if not channels:
        reasons.append("channels not requested")
    if not sgd:
        reasons.append(f"optimizer {opt.name!r}: CH1/CH2 assume an SGD-direction update")
    elif diag["tangent"] != "reconstructed_sgd":
        reasons.append("rebuilt SGD update does not match the measured Δθ")
    name = getattr(adapter, "model_type", "") or adapter.name
    reasons += [f"model {name!r} does not declare capability {c!r}" for c in missing_capabilities(adapter, CHANNEL_CAPS)]
    if reasons:
        skipped[ch.LdclChannels.name] = reasons
    else:
        with iv.Timer() as t:
            preds = ch.channel_predictions(adapter, objective, batch, enc, names, eta, batch_size)
            ch_total = ch.channel_totals_tensor(preds, names)
        costs["channels"] = t.seconds
        if readout_tangent is not None:
            _, d_ro = iv.first_order_jvp(model, readout_tangent, enc, names, batch_size)
            s1 = torch.stack([sum(preds["ch1"][o].values()) for o in names])
            err = float((s1 - d_ro).abs().max())
            diag["readout_channel_check"] = {"max_abs_err": err, "max_abs_jvp": float(d_ro.abs().max()),
                                             "rel_err": err / max(float(d_ro.abs().max()), 1e-30),
                                             "readout_tied": bool(getattr(adapter, "readout_tied", False))}
            # CH2 against the exact first-order effect of everything except the readout (the part CH2 estimates)
            backbone, ch2 = dphi - d_ro, ch_total - s1
            diag["ch2_vs_backbone_first_order"] = {
                o: {**{k: summarize(backbone[i], ch2[i])[k] for k in ("corr", "sign_agree")},
                    "rms_ch1": float(s1[i].pow(2).mean().sqrt()), "rms_ch2": float(ch2[i].pow(2).mean().sqrt()),
                    "rms_backbone_first_order": float(backbone[i].pow(2).mean().sqrt())}
                for i, o in enumerate(names)}
        diag["kappa"] = "not computed"
    return ChainResult(phi_from=s_from["phi"], phi_to=s_to["phi"], first_order=dphi, channels=ch_total,
                       channel_preds=preds, contributions=contributions, scored_from=s_from, scored_to=s_to,
                       diagnostics=diag, skipped=skipped, costs=costs)


def contribution_rows(enc: EncodedProbes, names: Sequence[str], parts: dict[str, torch.Tensor], *, step_from: int,
                      step_to: int, probe_set: str, probe_version: str, source_kind: str, prov: dict,
                      decomp_id: str = "sgd_by_source") -> list[dict]:
    rows = []
    for comp, vals in parts.items():
        kind = "weight_decay" if comp == "weight_decay" else source_kind
        for oi, o in enumerate(names):
            for i, item in enumerate(enc.item_ids):
                rows.append(dict(step_from=step_from, step_to=step_to, probe_set=probe_set, probe_version=probe_version,
                                 item=item, observable=o, decomp_id=decomp_id, target="first_order", source_kind=kind,
                                 source=comp, contrib=float(vals[oi, i]), method="jvp", **prov))
    return rows
