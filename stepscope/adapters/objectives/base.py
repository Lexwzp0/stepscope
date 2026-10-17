"""ObjectiveAdapter (blueprint §五). Required: replicated loss and loss split by source.
Optional: token-level force f = -dL/dz (LDCL plugin).

Most post-training losses enter the logits only through the log-probabilities of a few candidate tokens per
position (SFT: the target; Open-MOPD: the student's top-K; GRPO / DPO: the sampled token). ``CandidateObjective``
uses that: a subclass names the candidates and writes the loss as per-token terms of their log-probs, and gets

    loss            = Σ_{b,t} terms[b, t]
    loss_by_source  = {source: Σ over that source's rows}          (sums exactly to the loss)
    force           c_{t,k} = -∂loss/∂log p(v_{t,k}) by autograd on the [B, T, K] candidate log-probs, so

                    f_t = -∂loss/∂z_t = Σ_k c_{t,k} (e_{v_{t,k}} - p_t) / temperature

with clip, ratio, weights and normalisation included. Position convention: batch tensors are indexed by input
position, a target at position t is predicted by the logits at t - 1; the force is indexed by logit position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class SparseForce:
    """Per logit position (b, t): candidate ids v_{t,k} and coefficients c_{t,k}; f_t = Σ_k c (e_v − p_t) / T.
    Coefficients are zero at unsupervised positions."""
    ids: Any            # [B, T, K] long
    coef: Any           # [B, T, K] float
    temperature: float = 1.0


@dataclass
class TrainBatch:
    """The micro-batches consumed by one optimizer step, right padded.

    ``loss_mask[b, t] = 1`` when token t of row b is supervised (predicted from the logits at t − 1);
    ``sources`` names each row's source (teacher / domain); ``extras`` holds objective inputs aligned with input
    positions ([B, T, ...]) or rows ([B])."""
    input_ids: Any
    attention_mask: Any
    loss_mask: Any
    sources: list[str]
    source_kind: str = "domain"
    extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def to(self, device) -> "TrainBatch":
        mv = lambda x: x.to(device) if torch.is_tensor(x) else x  # noqa: E731
        return TrainBatch(mv(self.input_ids), mv(self.attention_mask), mv(self.loss_mask), list(self.sources),
                          self.source_kind, {k: mv(v) for k, v in self.extras.items()})

    def model_kwargs(self) -> dict:
        return dict(input_ids=self.input_ids, attention_mask=self.attention_mask, use_cache=False)

    def source_names(self) -> list[str]:
        return sorted(set(self.sources))

    def rows_of(self, source: str) -> list[int]:
        return [i for i, s in enumerate(self.sources) if s == source]

    def supervised_logit_positions(self):
        """[B, T] bool: logit positions whose next token is supervised."""
        m = torch.zeros_like(self.loss_mask, dtype=torch.bool)
        m[:, :-1] = self.loss_mask[:, 1:].bool()
        return m


@runtime_checkable
class ObjectiveAdapter(Protocol):
    name: str

    def loss(self, model: Any, batch: Any) -> Any:
        """Required. Replicated real loss; must match the framework's loss in tests."""

    def loss_by_source(self, model: Any, batch: Any) -> dict:
        """Required. Loss terms split by source (teacher / domain / sample) for per-source gradients."""

    def force(self, model: Any, batch: Any) -> SparseForce:
        """Optional (LDCL). Token-level force on logits."""


class CandidateObjective:
    """Base for objectives whose loss depends on the logits only through candidate log-probs."""
    name = "base"
    temperature: float = 1.0

    def candidates(self, batch: TrainBatch) -> torch.Tensor:
        """[B, T, K] candidate ids aligned with input positions (meaningful where loss_mask = 1)."""
        raise NotImplementedError

    def token_terms(self, cand_logp: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
        """Per-target loss terms [B, T − 1] from shifted candidate log-probs [B, T − 1, K]; Σ terms = loss.
        Inputs are shifted: index j refers to target position j + 1. Use ``shift(batch, key)`` for extras."""
        raise NotImplementedError

    def config(self) -> dict:
        return {"temperature": self.temperature}

    # --- shared machinery --------------------------------------------------------------------------
    @staticmethod
    def shift(x: torch.Tensor) -> torch.Tensor:
        """Target-position tensor [B, T, ...] -> [B, T − 1, ...] aligned with logits[:, :-1]."""
        return x[:, 1:]

    def candidate_logp(self, logits: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
        """[B, T − 1, K] log-probs of the candidates of every target position (logits[:, :-1] predict [:, 1:])."""
        logp = torch.log_softmax(logits[:, :-1].float() / self.temperature, dim=-1)
        return logp.gather(-1, self.shift(self.candidates(batch)).long())

    def terms_from_logits(self, logits: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
        return self.token_terms(self.candidate_logp(logits, batch), batch)

    def loss_from_logits(self, logits: torch.Tensor, batch: TrainBatch) -> torch.Tensor:
        return self.terms_from_logits(logits, batch).sum()

    def by_source(self, terms: torch.Tensor, batch: TrainBatch) -> dict[str, torch.Tensor]:
        return {s: terms[batch.rows_of(s)].sum() for s in batch.source_names()}

    def force_from_logits(self, logits: torch.Tensor, batch: TrainBatch) -> SparseForce:
        lp = self.candidate_logp(logits.detach(), batch).detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.token_terms(lp, batch).sum(), lp)
        cand = self.candidates(batch).long()
        B, T, K = cand.shape
        ids = torch.zeros(B, T, K, dtype=torch.long, device=cand.device)
        coef = torch.zeros(B, T, K, dtype=torch.float32, device=cand.device)
        ids[:, :-1] = self.shift(cand)
        coef[:, :-1] = -grad.float()
        return SparseForce(ids=ids, coef=coef, temperature=self.temperature)

    @staticmethod
    def _logits(model, batch: TrainBatch) -> torch.Tensor:
        return model(**batch.model_kwargs()).logits

    def loss(self, model, batch: TrainBatch) -> torch.Tensor:
        return self.loss_from_logits(self._logits(model, batch), batch)

    def loss_by_source(self, model, batch: TrainBatch) -> dict[str, torch.Tensor]:
        return self.by_source(self.terms_from_logits(self._logits(model, batch), batch), batch)

    def force(self, model, batch: TrainBatch) -> SparseForce:
        with torch.no_grad():
            logits = self._logits(model, batch)
        return self.force_from_logits(logits, batch)
