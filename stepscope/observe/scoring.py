"""Encoding and answer-slot scoring (ported from ood_ld scoring.py, with two changes after review):

1. Answer labels are resolved on the FULL rendered prompt, item by item: appending a label must add
   exactly one token and leave the prompt's own tokens unchanged (BPE can otherwise merge the label
   with the preceding text). The resulting input ids, option ids and correct indices are frozen in an
   ``EncodedProbes`` and shared by both checkpoints and by the JVP / backward first-order paths, so
   every measured change refers to the same observable on the same tokens.
2. Only answer-slot logits are produced (``logits_to_keep=1`` when supported); per batch the full
   vocabulary is normalised once and only the K option log-probs and derived metrics are kept.

Left padding; position ids follow the attention mask so padded and unpadded runs agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..core.model import ProbeItem
from .observables import decompose, option_logprobs, phi_from_optlogp
from .probes import RenderConfig, render_all


@dataclass
class EncodedProbes:
    item_ids: list[str]
    domains: list[str]
    prompts: list[str]
    id_lists: list[list[int]]         # prompt tokens (answer slot = next token)
    option_ids: torch.Tensor          # [N, K] token id of each label in each prompt's context
    correct_idx: torch.Tensor         # [N]
    labels: tuple[str, ...]           # rendered label strings
    pad_id: int

    def __len__(self) -> int:
        return len(self.id_lists)


def pad_id_of(tokenizer) -> int:
    pid = getattr(tokenizer, "pad_token_id", None)
    if pid is None:
        pid = getattr(tokenizer, "eos_token_id", None)
    if pid is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    return int(pid)


def _ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def encode_probes(tokenizer, items: list[ProbeItem], cfg: RenderConfig) -> EncodedProbes:
    """Render, tokenize and resolve the answer token of every label on every full prompt."""
    prompts = render_all(items, cfg, tokenizer)
    k = len(items[0].options)
    labels = tuple(cfg.label_tokens(k))
    id_lists, opt_rows = [], []
    for it, prompt in zip(items, prompts):
        if len(it.options) != k:
            raise ValueError(f"item {it.id} has {len(it.options)} options; expected {k} (one option count per probe set)")
        base = _ids(tokenizer, prompt)
        row = []
        for lab in labels:
            full = _ids(tokenizer, prompt + lab)
            if full[: len(base)] != base or len(full) != len(base) + 1:
                raise ValueError(
                    f"item {it.id}: label {lab!r} is not a single clean token after the full prompt "
                    f"(prompt tokens {len(base)}, with label {len(full)}, prefix preserved="
                    f"{full[: len(base)] == base}). Adjust --answer-prefix / --label-format."
                )
            row.append(full[-1])
        if len(set(row)) != len(row):
            raise ValueError(f"item {it.id}: labels map to duplicate token ids {dict(zip(labels, row))}")
        id_lists.append(base)
        opt_rows.append(row)
    return EncodedProbes(
        item_ids=[it.id for it in items], domains=[it.domain for it in items], prompts=prompts,
        id_lists=id_lists, option_ids=torch.as_tensor(opt_rows, dtype=torch.long),
        correct_idx=torch.as_tensor([it.answer_idx for it in items], dtype=torch.long),
        labels=labels, pad_id=pad_id_of(tokenizer),
    )


def left_pad(id_lists: Sequence[Sequence[int]], pad_id: int, device=None):
    T = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), T), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, T - len(x):] = torch.as_tensor(list(x), dtype=torch.long)
        mask[i, T - len(x):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    if device is not None:
        ids, mask, pos = ids.to(device), mask.to(device), pos.to(device)
    return ids, mask, pos


def forward_kwargs(ids, mask, pos) -> dict:
    return dict(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)


def last_slot_logits(model, ids, mask, pos) -> torch.Tensor:
    """[B, V] logits at the last real token; asks the model for that position only when supported.
    Differentiable; makes no grad-mode assumptions, so it also serves the first-order paths."""
    kw = forward_kwargs(ids, mask, pos)
    try:
        out = model(**kw, logits_to_keep=1)
    except TypeError:
        out = model(**kw)
    return out.logits[:, -1, :]


def batches(enc: EncodedProbes, batch_size: int, device=None):
    for s in range(0, len(enc), batch_size):
        sl = slice(s, s + batch_size)
        ids, mask, pos = left_pad(enc.id_lists[sl], enc.pad_id, device)
        yield sl, ids, mask, pos, enc.option_ids[sl].to(ids.device), enc.correct_idx[sl].to(ids.device)


@torch.no_grad()
def score_encoded(model, enc: EncodedProbes, names: Sequence[str], batch_size: int = 4,
                  device=None) -> dict[str, torch.Tensor]:
    """Observables (``names``) and in-set metrics for every item at one checkpoint.
    Keeps only [N, K] option log-probs and per-item scalars; never the full-vocabulary logits."""
    device = device or next(model.parameters()).device
    phis, opts, metrics = [], [], []
    for sl, ids, mask, pos, opt_ids, correct in batches(enc, batch_size, device):
        logits = last_slot_logits(model, ids, mask, pos)
        opt = option_logprobs(logits, opt_ids).float().cpu()
        del logits
        correct = correct.cpu()
        phis.append(phi_from_optlogp(names, opt, correct))
        opts.append(opt)
        metrics.append(decompose(opt, correct))
    out = {"phi": torch.cat(phis, dim=1), "option_logp": torch.cat(opts)}
    for key in metrics[0]:
        out[key] = torch.cat([m[key] for m in metrics])
    out["n_prompt_tokens"] = torch.as_tensor([len(x) for x in enc.id_lists])
    return out
