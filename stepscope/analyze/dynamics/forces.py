"""Output-side update force from the sparse form produced by an ObjectiveAdapter (blueprint §五).

    f_t = Σ_k c_{t,k} (e_{v_{t,k}} − p_t) / T

Densify only small slices; the readout projection and the energy are computed from the sparse form."""
from __future__ import annotations

import torch


def dense_force(coef: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """[..., K] coefficients and ids, [..., V] probabilities -> [..., V] force."""
    f = -coef.float().sum(dim=-1, keepdim=True) * probs.float()
    f = f.scatter_add(-1, ids.long(), coef.float())
    return f / temperature


def readout_projected_force(coef: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor, readout: torch.Tensor,
                            temperature: float = 1.0) -> torch.Tensor:
    """W^T f_t without materialising f: Σ_k c_k W[v_k] − (Σ_k c_k) W^T p_t  ->  [..., d]."""
    W = readout.float()
    sparse = torch.einsum("...k,...kd->...d", coef.float(), W[ids.long()])
    dense = coef.float().sum(-1, keepdim=True) * (probs.float() @ W)
    return (sparse - dense) / temperature


def token_energy(coef: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """||f_t||² from the sparse form (exact when the ids of one position are distinct): [...]."""
    c = coef.float()
    C = c.sum(-1)
    p_top = probs.float().gather(-1, ids.long())
    p_sq = probs.float().pow(2).sum(-1)
    return (c.pow(2).sum(-1) - 2 * C * (c * p_top).sum(-1) + C.pow(2) * p_sq) / temperature ** 2
