"""Token-level cross-entropy (SFT). Candidate = the target token, so c_t = (normaliser of t) on supervised
positions and f_t = c_t (e_{y_t} − p_t).

``normalization``: ``token_mean`` (Σ NLL / number of supervised tokens in the batch; the Hugging Face and TRL
default), ``seq_mean`` (token mean inside each row, then mean over rows) or ``sum``."""
from __future__ import annotations

import torch

from ...core.registry import OBJECTIVES
from .base import CandidateObjective, TrainBatch


@OBJECTIVES.register("sft")
class SFTObjective(CandidateObjective):
    name = "sft"

    def __init__(self, normalization: str = "token_mean", temperature: float = 1.0):
        if normalization not in ("token_mean", "seq_mean", "sum"):
            raise ValueError(f"unknown normalization {normalization!r}")
        self.normalization = normalization
        self.temperature = temperature

    def config(self) -> dict:
        return {"normalization": self.normalization, "temperature": self.temperature}

    def candidates(self, batch: TrainBatch) -> torch.Tensor:
        return batch.input_ids.unsqueeze(-1)

    def token_terms(self, cand_logp: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
        mask = self.shift(batch.loss_mask).float()
        nll = -cand_logp[..., 0] * mask
        if self.normalization == "token_mean":
            return nll / mask.sum().clamp(min=1.0)
        if self.normalization == "seq_mean":
            return nll / mask.sum(-1, keepdim=True).clamp(min=1.0) / mask.shape[0]
        return nll
