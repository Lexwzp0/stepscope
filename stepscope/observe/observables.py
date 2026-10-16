"""Differentiable observables on multiple-choice items (ported from ood_ld observables.py).

For an item o with option labels A_o and correct label a*, at the answer slot:

    phi_fmt(o) = log Σ_{a∈A_o} p(a | x_o)                       willingness to answer in the required form
    phi_ans(o) = log p(a* | x_o) − log Σ_{a≠a*} p(a | x_o)      correct vs wrong inside the option set

so that log p(correct) = phi_fmt + log p(correct | valid) and phi_ans = logit p(correct | valid).
All functions take full-vocabulary logits/log-probs at the answer slot so format mass is not
renormalised away. Everything is differentiable w.r.t. the logits.
"""
from __future__ import annotations

import torch

OBSERVABLES = ("phi_fmt", "phi_ans")


def option_logprobs(answer_slot_logits: torch.Tensor, option_ids, temperature: float = 1.0) -> torch.Tensor:
    """[N, V] logits at the answer slot -> [N, K] full-vocabulary log-probs of the K options.

    ``option_ids`` is either a 1-D sequence shared by all rows or an [N, K] tensor (per item)."""
    logp = torch.log_softmax(answer_slot_logits.float() / temperature, dim=-1)
    idx = torch.as_tensor(option_ids, device=logp.device, dtype=torch.long)
    if idx.dim() == 1:
        return logp.index_select(-1, idx)
    if idx.shape[0] != logp.shape[0]:
        raise ValueError(f"option_ids rows {idx.shape[0]} != logits rows {logp.shape[0]}")
    return logp.gather(-1, idx)


def phi_fmt(opt_logp: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(opt_logp, dim=-1)


def phi_ans(opt_logp: torch.Tensor, correct_idx: torch.Tensor) -> torch.Tensor:
    correct_idx = correct_idx.to(opt_logp.device).long()
    correct = opt_logp.gather(-1, correct_idx.unsqueeze(-1)).squeeze(-1)
    is_correct = torch.zeros_like(opt_logp, dtype=torch.bool).scatter_(-1, correct_idx.unsqueeze(-1), True)
    others = torch.logsumexp(opt_logp.masked_fill(is_correct, float("-inf")), dim=-1)
    return correct - others


def phi(name: str, opt_logp: torch.Tensor, correct_idx: torch.Tensor) -> torch.Tensor:
    if name == "phi_fmt":
        return phi_fmt(opt_logp)
    if name == "phi_ans":
        return phi_ans(opt_logp, correct_idx)
    raise ValueError(f"unknown observable {name!r}; known: {OBSERVABLES}")


def phi_from_optlogp(names, opt_logp: torch.Tensor, correct_idx: torch.Tensor) -> torch.Tensor:
    """[len(names), N] stack of observables from option log-probs. Differentiable."""
    return torch.stack([phi(n, opt_logp, correct_idx) for n in names], dim=0)


def phi_all(names, answer_slot_logits: torch.Tensor, option_ids, correct_idx: torch.Tensor) -> torch.Tensor:
    """[len(names), N] stack of observables computed from answer-slot logits. Differentiable."""
    return phi_from_optlogp(names, option_logprobs(answer_slot_logits, option_ids), correct_idx)


@torch.no_grad()
def decompose(opt_logp: torch.Tensor, correct_idx: torch.Tensor) -> dict[str, torch.Tensor]:
    """p(correct) = p(valid) · p(correct | valid), plus both observables and the in-set argmax."""
    f = phi_fmt(opt_logp)
    a = phi_ans(opt_logp, correct_idx)
    ci = correct_idx.to(opt_logp.device).long()
    log_p_correct = opt_logp.gather(-1, ci.unsqueeze(-1)).squeeze(-1)
    return {
        "phi_fmt": f,
        "phi_ans": a,
        "p_valid": f.exp(),
        "p_correct_given_valid": torch.sigmoid(a),
        "p_correct": log_p_correct.exp(),
        "argmax_in_set_correct": (opt_logp.argmax(-1) == ci).float(),
    }
