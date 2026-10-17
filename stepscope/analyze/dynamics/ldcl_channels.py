"""LDCL forward-computable interaction channels, aggregated per source (LDCL eq. (5); blueprint §六, §十九).

For one (observed slot o, updating token u) pair:

    Δ(o,u) ≈ η (g_o·f_u)(hL_o·hL_u)  +  η (Wᵀg_o)·(Wᵀf_u) (Σ_l h̃l_o·h̃l_u + κ)
             \\________ CH1 ________/     \\______________ CH2 ______________/

g_o = ∂Φ_o/∂z at the observed slot; f_u = −∂L/∂z_u the update force (ObjectiveAdapter.force); W the readout
[V, d]; hL the readout input; h̃l the normalised input of residual branch l (attention and MLP inputs of each
block). Summing over the updating tokens u of a source d factorises:

    M_d     = Σ_u f_u hL_uᵀ           (V × d)  = minus the readout gradient of that source's loss
    N_{l,d} = Σ_u (Wᵀf_u) h̃l_uᵀ       (d × d)
    S1_d(o) = g_oᵀ M_d hL_o,    S2_{l,d}(o) = (Wᵀg_o)ᵀ N_{l,d} h̃l_o

so S_d(o) = S1_d(o) + Σ_l S2_{l,d}(o) needs one forward per side. η S1 equals the exact first-order effect of
the readout part of an SGD step when the readout is untied and has no bias; CH2 rests on the identity-path and
linear-block approximations (checked against the exact first order by the single-step chain). κ, the
embedding token-overlap term, is not computed. Applicability: SGD-direction updates; under AdamW the channel
values are qualitative only. Model access goes through the adapter's declared ``readout`` and
``residual_inputs`` capabilities, so the code is backbone independent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ...core.registry import ANALYSES
from ...observe.observables import observable_logit_grad
from ...observe.scoring import EncodedProbes, batches, forward_kwargs
from .forces import readout_projected_force

VERSION = "0.1.0"


@dataclass
class SourceMatrices:
    M: torch.Tensor                                   # [V, d]
    N: dict[str, torch.Tensor] = field(default_factory=dict)   # site -> [d, d]
    n_tokens: int = 0


# --- pairwise / dense forms (reference and tests) --------------------------------------------------------
@torch.no_grad()
def accumulate_source_matrices(forces: torch.Tensor, final_states: torch.Tensor, branch_states: list[torch.Tensor],
                               readout: torch.Tensor, chunk: int = 512):
    """M and N_l from n updating tokens with dense forces [n, V]; final_states [n, d]; branch_states list of [n, d]."""
    W = readout.float()
    M = torch.zeros(W.shape[0], W.shape[1], dtype=torch.float32, device=W.device)
    N = [torch.zeros(W.shape[1], W.shape[1], dtype=torch.float32, device=W.device) for _ in branch_states]
    for s in range(0, forces.shape[0], chunk):
        f = forces[s:s + chunk].float()
        M += f.T @ final_states[s:s + chunk].float()
        wf = f @ W
        for i, hl in enumerate(branch_states):
            N[i] += wf.T @ hl[s:s + chunk].float()
    return M, N


@torch.no_grad()
def channel_scores(g_obs: torch.Tensor, final_obs: torch.Tensor, branch_obs: list[torch.Tensor],
                   readout: torch.Tensor, M: torch.Tensor, N: list[torch.Tensor]):
    """S1 [m] and S2 [L, m] for m observed slots. g_obs [m, V]; final_obs [m, d]; branch_obs list of [m, d]."""
    W = readout.float()
    g = g_obs.float()
    s1 = ((g @ M) * final_obs.float()).sum(-1)
    wg = g @ W
    s2 = torch.stack([((wg @ Nl) * hl.float()).sum(-1) for Nl, hl in zip(N, branch_obs)]) if N else g.new_zeros(0, g.shape[0])
    return s1, s2


def pairwise_reference(g_obs, final_obs, branch_obs, forces, final_upd, branch_upd, readout):
    """Explicit O(m n) sum over token pairs; for tests only."""
    W = readout.float()
    gg = g_obs.float() @ forces.float().T
    hh = final_obs.float() @ final_upd.float().T
    s1 = (gg * hh).sum(-1)
    ww = (g_obs.float() @ W) @ (forces.float() @ W).T
    s2 = torch.stack([(ww * (ho.float() @ hu.float().T)).sum(-1) for ho, hu in zip(branch_obs, branch_upd)])
    return s1, s2


# --- sparse forms (used by the platform) -------------------------------------------------------------------
@torch.no_grad()
def source_matrices_sparse(coef: torch.Tensor, ids: torch.Tensor, logits: torch.Tensor, final: torch.Tensor,
                           branches: dict[str, torch.Tensor], readout: torch.Tensor, temperature: float = 1.0,
                           chunk: int = 256) -> SourceMatrices:
    """M and N_l from n updating tokens in sparse force form (coef, ids [n, K]) without densifying f:
    M = Σ_u Σ_k c_k e_{v_k} hL_uᵀ − Pᵀ (C ⊙ hL),  N_l = Σ_u (Wᵀf_u) h̃l_uᵀ. ``logits`` [n, V]; states [n, d]."""
    W = readout.float()
    V, d = W.shape
    M = torch.zeros(V, d, dtype=torch.float32, device=W.device)
    N = {s: torch.zeros(d, d, dtype=torch.float32, device=W.device) for s in branches}
    n = coef.shape[0]
    for s in range(0, n, chunk):
        c = coef[s:s + chunk].float()
        v = ids[s:s + chunk].long()
        hL = final[s:s + chunk].float()
        p = torch.softmax(logits[s:s + chunk].float() / temperature, dim=-1)
        for k in range(c.shape[1]):
            M.index_add_(0, v[:, k], c[:, k:k + 1] * hL / temperature)
        M -= p.T @ (c.sum(-1, keepdim=True) * hL) / temperature
        wf = readout_projected_force(c, v, p, W, temperature)
        for site, h in branches.items():
            N[site] += wf.T @ h[s:s + chunk].float()
    return SourceMatrices(M=M, N=N, n_tokens=int(n))


@torch.no_grad()
def update_side(adapter, objective, batch) -> dict[str, SourceMatrices]:
    """Per-source M_d, N_{l,d} at the current parameters from one forward over the training batch."""
    batch = batch.to(adapter.device)
    ri = adapter.residual_inputs(batch.model_kwargs())
    sf = objective.force_from_logits(ri.logits, batch)
    W = adapter.readout().detach()
    sup = batch.supervised_logit_positions()
    out = {}
    for src in batch.source_names():
        rows = torch.zeros_like(sup)
        rows[batch.rows_of(src)] = True
        sel = sup & rows
        out[src] = source_matrices_sparse(sf.coef[sel], sf.ids[sel], ri.logits[sel], ri.final[sel],
                                          {site: h[sel] for site, h in ri.branches.items()}, W, sf.temperature)
    return out


@torch.no_grad()
def channel_predictions(adapter, objective, batch, enc: EncodedProbes, names: Sequence[str], eta: float,
                        batch_size: int = 4) -> dict:
    """η-scaled channel predictions for every probe item: {"ch1": {obs: {src: [N]}}, "ch2": {obs: {src: {site: [N]}}}}."""
    mats = update_side(adapter, objective, batch)
    W = adapter.readout().detach().float()
    sites = list(next(iter(mats.values())).N) if mats else []
    ch1 = {o: {s: [] for s in mats} for o in names}
    ch2 = {o: {s: {site: [] for site in sites} for s in mats} for o in names}
    for sl, ids, mask, pos, opt_ids, correct in batches(enc, batch_size, adapter.device):
        kw = forward_kwargs(ids, mask, pos)
        try:
            ri = adapter.residual_inputs({**kw, "logits_to_keep": 1})
        except TypeError:
            ri = adapter.residual_inputs(kw)
        probs = torch.softmax(ri.logits[:, -1].float(), dim=-1)
        hL = ri.final[:, -1].float()
        hb = [ri.branches[site][:, -1].float() for site in sites]
        for o in names:
            g = observable_logit_grad(probs, opt_ids, correct, o)
            for src, m in mats.items():
                s1, s2 = channel_scores(g, hL, hb, W, m.M, [m.N[site] for site in sites])
                ch1[o][src].append((eta * s1).cpu())
                for i, site in enumerate(sites):
                    ch2[o][src][site].append((eta * s2[i]).cpu())
    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0)  # noqa: E731
    return {"ch1": {o: {s: cat(v) for s, v in d.items()} for o, d in ch1.items()},
            "ch2": {o: {s: {site: cat(v) for site, v in dd.items()} for s, dd in d.items()} for o, d in ch2.items()},
            "sites": sites, "n_tokens": {s: m.n_tokens for s, m in mats.items()}}


def channel_rows(enc: EncodedProbes, names: Sequence[str], preds: dict, *, step_from: int, step_to: int,
                 probe_set: str, probe_version: str, source_kind: str, applicability: str, prov: dict) -> list[dict]:
    rows = []
    for o in names:
        for src, s1 in preds["ch1"][o].items():
            for i, item in enumerate(enc.item_ids):
                base = dict(step_from=step_from, step_to=step_to, probe_set=probe_set, probe_version=probe_version,
                            item=item, observable=o, source_kind=source_kind, source=src, applicability=applicability, **prov)
                rows.append(dict(base, layer="readout", ch1=float(s1[i]), ch2=0.0))
                for site in preds["sites"]:
                    rows.append(dict(base, layer=site, ch1=0.0, ch2=float(preds["ch2"][o][src][site][i])))
    return rows


def channel_totals_tensor(preds: dict, names: Sequence[str]) -> torch.Tensor:
    """[len(names), N] Σ over sources and sites of ch1 + ch2."""
    out = []
    for o in names:
        tot = None
        for src, s1 in preds["ch1"][o].items():
            t = s1.clone()
            for site in preds["sites"]:
                t += preds["ch2"][o][src][site]
            tot = t if tot is None else tot + t
        out.append(tot)
    return torch.stack(out)


@ANALYSES.register("dynamics.ldcl_channels")
class LdclChannels:
    """Registry wrapper; the single-step chain (``single_step.py``) calls the functions directly."""
    name = "dynamics.ldcl_channels"
    version = VERSION
    level = "channel"
    requires = {"update_package", "probes", "readout", "residual_inputs"}
    writes = {"channels"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        preds = channel_predictions(inputs["adapter"], inputs["objective"], inputs["batch"], inputs["enc"],
                                    inputs["names"], inputs["eta"], inputs.get("batch_size", 4))
        rows = channel_rows(inputs["enc"], inputs["names"], preds, step_from=inputs["step_from"], step_to=inputs["step_to"],
                            probe_set=inputs["probe_set"], probe_version=inputs["probe_version"],
                            source_kind=inputs["batch"].source_kind, applicability=inputs.get("applicability", "sgd"),
                            prov=inputs["prov"])
        store.write(run_id, "channels", rows, part=inputs.get("part", "0000"))
