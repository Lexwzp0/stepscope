"""Probe files, rendering and option-label resolution.

Probe file: JSON lines, one multiple-choice item per line::

    {"id": "gpqa_001#r0", "domain": "chemistry", "question": "...",
     "options": ["...", "...", "...", "..."], "answer_idx": 2}

``options`` are in display order (rotations are the caller's business, cf. ood_ld
``balanced_rotations``); ``answer_idx`` indexes them. Extra keys go to ``meta``.

Rendering is deterministic: instruction + question + "A. option" lines + answer prefix.
The option *labels* ("A", "B", ...) are rendered as the next token after the answer
prefix via ``label_format`` and must be single tokens in that context (checked by
``scoring.resolve_options``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..core.model import ProbeItem, ProbeSet

DEFAULT_INSTRUCTION = "Answer the following multiple choice question with the letter of the correct option."
DEFAULT_LABELS = ("A", "B", "C", "D", "E", "F", "G", "H")


@dataclass(frozen=True)
class RenderConfig:
    instruction: str = DEFAULT_INSTRUCTION
    labels: tuple[str, ...] = DEFAULT_LABELS
    answer_prefix: str = "\nAnswer:"     # appended after the options; scoring happens at the next token
    label_format: str = " {label}"       # how a label appears as the next token after answer_prefix
    chat_template: bool = False          # wrap the user text with the tokenizer's chat template
    system: str | None = None

    def label_tokens(self, k: int) -> list[str]:
        return [self.label_format.format(label=lab) for lab in self.labels[:k]]


def load_probe_file(path: str | Path, name: str | None = None) -> ProbeSet:
    path = Path(path)
    raw = path.read_bytes()
    version = hashlib.sha1(raw).hexdigest()[:12]
    items: list[ProbeItem] = []
    for ln, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        for key in ("id", "domain", "question", "options", "answer_idx"):
            if key not in d:
                raise ValueError(f"{path}:{ln}: missing field {key!r}")
        opts = tuple(str(o) for o in d["options"])
        ai = int(d["answer_idx"])
        if not 0 <= ai < len(opts):
            raise ValueError(f"{path}:{ln}: answer_idx {ai} out of range for {len(opts)} options")
        meta = {k: v for k, v in d.items() if k not in ("id", "domain", "question", "options", "answer_idx")}
        items.append(ProbeItem(str(d["id"]), str(d["domain"]), str(d["question"]), opts, ai, meta))
    if not items:
        raise ValueError(f"{path}: no items")
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate item ids")
    return ProbeSet(name=name or path.stem, version=version, items=items, source_path=str(path))


def format_mcq(item: ProbeItem, cfg: RenderConfig) -> str:
    if len(item.options) > len(cfg.labels):
        raise ValueError(f"item {item.id}: {len(item.options)} options but only {len(cfg.labels)} labels")
    lines = [cfg.instruction, "", item.question.strip(), ""]
    lines += [f"{lab}. {opt.strip()}" for lab, opt in zip(cfg.labels, item.options)]
    return "\n".join(lines)


def render_item(item: ProbeItem, cfg: RenderConfig, tokenizer=None) -> str:
    """Full prompt ending with the answer prefix (the next token is the answer slot)."""
    user = format_mcq(item, cfg)
    if cfg.chat_template:
        if tokenizer is None:
            raise ValueError("chat_template=True needs a tokenizer")
        msgs = ([{"role": "system", "content": cfg.system}] if cfg.system else []) + [{"role": "user", "content": user}]
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return rendered + cfg.answer_prefix.lstrip("\n")
    return user + cfg.answer_prefix


def render_all(items: list[ProbeItem], cfg: RenderConfig, tokenizer=None) -> list[str]:
    return [render_item(it, cfg, tokenizer) for it in items]
