"""Open-MOPD loss (top_k_strategy=only_stu, adv_estimator=token_reward_direct), as a candidate objective.

The student's own top-K candidates v_{t,1..K} are supervised by the teacher's log-probs q on them:

    A_{t,k} = −(log p_s(v_k) − log q(v_k)) · w_{t,k},   w = softmax_k(log p_s)   (student_p)
                                                       w = softmax_k(log q)     (teacher_p)
                                                       w = 1/K                  (none)

times the per-row domain weight and per-token weight; PPO dual clip per candidate on the ratio
exp(log p − log p_old); summed over K; token mean inside each micro-batch; mean over micro-batches (verl
dp_actor: loss_scale_factor = 1 / gradient_accumulation). With ``refresh=True`` the advantage is rebuilt from
the current (detached) log-probs, as the patched trainer does.

Batch extras (aligned with input positions): ``topk_ids`` [B, T, K], ``teacher_logp`` [B, T, K]; optional
``old_logp`` [B, T, K] (default: the current log-probs, i.e. the on-policy first inner step), ``advantages``
[B, T, K] (``refresh=False``), ``seq_weight`` [B], ``token_weight`` [B, T].

Since every candidate enters only through log p(v_k), c_{t,k} = −∂L/∂log p(v_{t,k}) and the force is
f_t = Σ_k c_{t,k} (e_{v_k} − p_t) / T (blueprint §五). The replica is checked against verl's
``core_algos.compute_policy_loss_vanilla`` when verl is importable.
"""
from __future__ import annotations

import torch

from ...core.registry import OBJECTIVES
from .base import CandidateObjective, TrainBatch


def open_mopd_advantage(student_topk_logp: torch.Tensor, teacher_topk_logp: torch.Tensor,
                        weight_mode: str = "student_p") -> torch.Tensor:
    """Advantage [..., K]; detached inputs."""
    s = student_topk_logp.detach().float()
    t = teacher_topk_logp.detach().float()
    if weight_mode == "student_p":
        w = torch.softmax(s, dim=-1)
    elif weight_mode == "teacher_p":
        w = torch.softmax(t, dim=-1)
    elif weight_mode == "none":
        w = torch.full_like(s, 1.0 / s.shape[-1])
    else:
        raise ValueError(f"unknown weight_mode {weight_mode!r}")
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return -(s - t) * w


def dual_clip_pg(logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, clip_low: float, clip_high: float,
                 clip_c: float) -> torch.Tensor:
    """Per-candidate PPO dual-clip policy loss, summed over K: [..., K] -> [...]."""
    ratio = torch.exp(torch.clamp(logp - old_logp, min=-20.0, max=20.0))
    l1 = -adv * ratio
    l2 = -adv * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    clip1 = torch.maximum(l1, l2)
    clip2 = torch.minimum(-adv * clip_c, clip1)
    return torch.where(adv < 0, clip2, clip1).sum(dim=-1)


def micro_batch_token_weights(mask: torch.Tensor, micro_batch_size: int) -> torch.Tensor:
    """Weight of each token in the final loss: masked token mean per micro-batch of consecutive rows, then the
    mean over micro-batches."""
    mask = mask.float()
    B = mask.shape[0]
    starts = range(0, B, micro_batch_size)
    w = torch.zeros_like(mask)
    for s in starts:
        sl = slice(s, s + micro_batch_size)
        w[sl] = mask[sl] / mask[sl].sum().clamp(min=1e-8)
    return w / len(starts)


@OBJECTIVES.register("mopd")
class MOPDObjective(CandidateObjective):
    name = "mopd"

    def __init__(self, weight_mode: str = "student_p", clip_low: float = 0.2, clip_high: float = 0.28,
                 clip_c: float = 3.0, micro_batch_size: int | None = None, refresh: bool = True,
                 temperature: float = 1.0):
        self.weight_mode = weight_mode
        self.clip_low, self.clip_high, self.clip_c = clip_low, clip_high, clip_c
        self.micro_batch_size = micro_batch_size
        self.refresh = refresh
        self.temperature = temperature

    def config(self) -> dict:
        return dict(weight_mode=self.weight_mode, clip_low=self.clip_low, clip_high=self.clip_high, clip_c=self.clip_c,
                    micro_batch_size=self.micro_batch_size, refresh=self.refresh, temperature=self.temperature)

    def candidates(self, batch: TrainBatch) -> torch.Tensor:
        return batch.extras["topk_ids"]

    def token_terms(self, cand_logp: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
        ex = batch.extras
        teacher = self.shift(ex["teacher_logp"]).float()
        if self.refresh:
            adv = open_mopd_advantage(cand_logp, teacher, self.weight_mode)
        else:
            if "advantages" not in ex:
                raise ValueError("refresh=False needs batch.extras['advantages']")
            adv = self.shift(ex["advantages"]).float()
        if "seq_weight" in ex:
            adv = adv * ex["seq_weight"].float().view(-1, 1, 1)
        if "token_weight" in ex:
            adv = adv * self.shift(ex["token_weight"]).float().unsqueeze(-1)
        old = self.shift(ex["old_logp"]).float() if "old_logp" in ex else cand_logp.detach()
        pg = dual_clip_pg(cand_logp, old, adv, self.clip_low, self.clip_high, self.clip_c)
        mbs = self.micro_batch_size or len(batch)
        return pg * micro_batch_token_weights(self.shift(batch.loss_mask), mbs)
