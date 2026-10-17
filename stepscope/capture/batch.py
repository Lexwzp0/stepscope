"""Training batches for one optimizer step: build, read from a batch file, save, and add MOPD teacher inputs.

Batch file: JSON lines, one row (sequence) per line::

    {"source": "math", "prompt": "Q: 2+2=", "response": " 4"}                 text, tokenized by the checkpoint
    {"source": "code", "input_ids": [5, 9, 11, 3], "prompt_len": 2}            token ids; or "loss_mask": [...]

``source`` is the row's teacher or domain (default "all"); the response tokens are supervised. MOPD rows may
carry ``topk_ids`` and ``teacher_logp`` (and ``old_logp``), one K-list per input position; without them,
``mopd_extras`` fills them from the student's own top-K and the teachers' log-probs at θ.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from ..adapters.objectives.base import TrainBatch

_TOKEN_FIELDS = ("topk_ids", "teacher_logp", "old_logp", "advantages", "token_weight")


def make_batch(id_lists: Sequence[Sequence[int]], mask_lists: Sequence[Sequence[int]], sources: Sequence[str],
               pad_id: int, source_kind: str = "domain", extras: dict[str, list] | None = None) -> TrainBatch:
    """Right-padded batch. ``extras`` maps a field to per-row lists ([T_b] or [T_b][K]) or per-row scalars."""
    B, T = len(id_lists), max(len(x) for x in id_lists)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    lm = torch.zeros(B, T, dtype=torch.long)
    for i, (x, m) in enumerate(zip(id_lists, mask_lists)):
        if len(m) != len(x):
            raise ValueError(f"row {i}: loss mask length {len(m)} != {len(x)} tokens")
        ids[i, :len(x)] = torch.as_tensor(list(x))
        att[i, :len(x)] = 1
        lm[i, :len(x)] = torch.as_tensor(list(m))
    lm[:, 0] = 0
    ex = {}
    for key, rows in (extras or {}).items():
        if key == "seq_weight":
            ex[key] = torch.as_tensor(rows, dtype=torch.float32)
            continue
        first = torch.as_tensor(rows[0])
        dtype = torch.long if key == "topk_ids" else torch.float32
        out = torch.zeros((B, T) + tuple(first.shape[1:]), dtype=dtype)
        for i, r in enumerate(rows):
            r = torch.as_tensor(r, dtype=dtype)
            out[i, :r.shape[0]] = r
        ex[key] = out
    return TrainBatch(ids, att, lm, list(sources), source_kind, ex)


def load_batch_file(path: str | Path, tokenizer=None, source_kind: str = "domain") -> TrainBatch:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no rows")
    id_lists, masks, sources = [], [], []
    extras: dict[str, list] = {}
    for ln, r in enumerate(rows, 1):
        if "input_ids" in r:
            ids = [int(x) for x in r["input_ids"]]
            if "loss_mask" in r:
                mask = [int(x) for x in r["loss_mask"]]
            else:
                pl = int(r.get("prompt_len", 0))
                mask = [0] * pl + [1] * (len(ids) - pl)
        elif "prompt" in r and "response" in r:
            if tokenizer is None:
                raise ValueError(f"{path}:{ln}: text rows need a tokenizer")
            p = list(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"])
            a = list(tokenizer(r["response"], add_special_tokens=False)["input_ids"])
            ids, mask = p + a, [0] * len(p) + [1] * len(a)
        else:
            raise ValueError(f"{path}:{ln}: need input_ids (+ prompt_len or loss_mask) or prompt + response")
        id_lists.append(ids)
        masks.append(mask)
        sources.append(str(r.get("source", "all")))
        for key in _TOKEN_FIELDS + ("seq_weight",):
            if key in r:
                extras.setdefault(key, []).append(r[key])
    for key, vals in extras.items():
        if len(vals) != len(rows):
            raise ValueError(f"{path}: field {key!r} present on some rows only")
    pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    return make_batch(id_lists, masks, sources, pad if pad is not None else 0, source_kind, extras)


def save_batch(batch: TrainBatch, path: str | Path) -> Path:
    path = Path(path)
    torch.save({"input_ids": batch.input_ids.cpu(), "attention_mask": batch.attention_mask.cpu(),
                "loss_mask": batch.loss_mask.cpu(), "sources": list(batch.sources), "source_kind": batch.source_kind,
                "extras": {k: v.cpu() for k, v in batch.extras.items()}}, path)
    return path


def load_batch(path: str | Path) -> TrainBatch:
    d = torch.load(path, map_location="cpu", weights_only=True)
    return TrainBatch(d["input_ids"], d["attention_mask"], d["loss_mask"], list(d["sources"]), d["source_kind"],
                      dict(d["extras"]))


@torch.no_grad()
def mopd_extras(student, teachers, batch: TrainBatch, k: int, temperature: float = 1.0) -> TrainBatch:
    """Fill ``topk_ids`` (the student's own top-K at θ for every target position) and ``teacher_logp`` (the
    log-prob each row's teacher assigns to those candidates). ``teachers``: one model or {source: model}."""
    kw = batch.model_kwargs()
    s_logits = student(**kw).logits[:, :-1].float() / temperature
    B, T = batch.input_ids.shape
    top = torch.topk(s_logits, k, dim=-1).indices                      # [B, T-1, K]
    ids = torch.zeros(B, T, k, dtype=torch.long, device=top.device)
    ids[:, 1:] = top
    tl = torch.zeros(B, T, k, dtype=torch.float32, device=top.device)
    by_source = teachers if isinstance(teachers, dict) else {s: teachers for s in batch.source_names()}
    for s in batch.source_names():
        if s not in by_source:
            raise KeyError(f"no teacher for source {s!r}; have {sorted(by_source)}")
        rows = batch.rows_of(s)
        t_logits = by_source[s](**{key: v[rows] for key, v in kw.items() if key != "use_cache"}, use_cache=False).logits
        t_logp = torch.log_softmax(t_logits[:, :-1].float() / temperature, dim=-1)
        tl[rows, 1:] = t_logp.gather(-1, top[rows])
    extras = {**batch.extras, "topk_ids": ids, "teacher_logp": tl}
    return TrainBatch(batch.input_ids, batch.attention_mask, batch.loss_mask, list(batch.sources), batch.source_kind,
                      extras)
