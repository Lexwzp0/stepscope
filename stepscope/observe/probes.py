"""Probe files, rendering and option-label resolution.

Probe file: JSON lines, one multiple-choice item per line::

    {"id": "gpqa_001#r0", "domain": "chemistry", "question": "...",
     "options": ["...", "...", "...", "..."], "answer_idx": 2}

``options`` are in display order (rotations are the caller's business, cf. ood_ld
``balanced_rotations``); ``answer_idx`` indexes them. Extra keys go to ``meta``.

Rendering is deterministic: instruction + question + "A. option" lines + answer prefix.
The option *labels* ("A", "B", ...) are rendered as the next token after the answer
prefix via ``label_format`` and must be single tokens in that context (checked on the
full prompt by ``scoring.encode_probes``).

With ``chat_template=True`` the user text is wrapped by the tokenizer's chat template
(``enable_thinking`` passed through when the template accepts it, a frozen date shadowing
``strftime_now`` so prompts do not change daily), then ``no_think_suffix`` (e.g. an empty
<think></think> block) and ``answer_prefix`` are appended — the protocol frozen by ood_ld's
J06 ``template.json`` loads directly via ``RenderConfig.from_protocol``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from ..core.model import ProbeItem, ProbeSet

DEFAULT_INSTRUCTION = "Answer the following multiple-choice question."
DEFAULT_LABELS = ("A", "B", "C", "D", "E", "F", "G", "H")


def fixed_strftime(date: str):
    """Stand-in for a chat template's ``strftime_now`` that always formats ``date`` (YYYY-MM-DD)."""
    d = datetime.strptime(date, "%Y-%m-%d")
    return lambda fmt: d.strftime(fmt)


@dataclass(frozen=True)
class RenderConfig:
    instruction: str = DEFAULT_INSTRUCTION
    labels: tuple[str, ...] = DEFAULT_LABELS
    answer_prefix: str = "\nAnswer:"     # appended last; scoring happens at the next token
    label_format: str = " {label}"       # "{label}" or positional "{}" (ood_ld style)
    chat_template: bool = False          # wrap the user text with the tokenizer's chat template
    system: str | None = None
    enable_thinking: bool | None = None  # passed to apply_chat_template when not None
    no_think_suffix: str = ""            # appended after the generation prompt (before answer_prefix)
    fixed_date: str | None = None        # YYYY-MM-DD; shadows the template's strftime_now

    def label_tokens(self, k: int) -> list[str]:
        if k > len(self.labels):
            raise ValueError(f"{k} options but only {len(self.labels)} labels")
        return [self.label_format.format(lab, label=lab) for lab in self.labels[:k]]

    @classmethod
    def from_protocol(cls, path: str | Path, **overrides) -> "RenderConfig":
        """Load the direct-scoring protocol frozen by ood_ld ``check_template.py`` (template.json)."""
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        d = doc.get("direct", doc)
        kw = dict(chat_template=True)
        if "answer_prefix" in d: kw["answer_prefix"] = d["answer_prefix"]
        if "label_format" in d: kw["label_format"] = d["label_format"]
        if d.get("labels"): kw["labels"] = tuple(d["labels"])
        if "enable_thinking" in d: kw["enable_thinking"] = bool(d["enable_thinking"])
        if d.get("no_think_suffix"): kw["no_think_suffix"] = d["no_think_suffix"]
        if d.get("fixed_date"): kw["fixed_date"] = d["fixed_date"]
        if doc.get("instruction"): kw["instruction"] = doc["instruction"]
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)


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


def item_rank(item_id: str, salt: str = "stepscope_select") -> int:
    return int(hashlib.sha1(f"{salt}:{item_id}".encode()).hexdigest()[:8], 16)


def select_items(ps: ProbeSet, max_items: int | None) -> ProbeSet:
    """Deterministic subset by id hash (same subset on every machine); version records the selection."""
    if max_items is None or max_items >= len(ps.items):
        return ps
    chosen = sorted(ps.items, key=lambda it: item_rank(it.id))[:max_items]
    chosen.sort(key=lambda it: ps.items.index(it))
    return replace(ps, items=chosen, version=f"{ps.version}-n{max_items}")


def format_mcq(item: ProbeItem, cfg: RenderConfig) -> str:
    if len(item.options) > len(cfg.labels):
        raise ValueError(f"item {item.id}: {len(item.options)} options but only {len(cfg.labels)} labels")
    lines = [cfg.instruction, "", item.question.strip(), ""]
    lines += [f"{lab}. {opt.strip()}" for lab, opt in zip(cfg.labels, item.options)]
    return "\n".join(lines)


def render_item(item: ProbeItem, cfg: RenderConfig, tokenizer=None) -> str:
    """Full prompt ending with the answer prefix (the next token is the answer slot)."""
    user = format_mcq(item, cfg)
    if not cfg.chat_template:
        return user + cfg.answer_prefix
    if tokenizer is None:
        raise ValueError("chat_template=True needs a tokenizer")
    msgs = ([{"role": "system", "content": cfg.system}] if cfg.system else []) + [{"role": "user", "content": user}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    if cfg.fixed_date:
        kw["strftime_now"] = fixed_strftime(cfg.fixed_date)
    if cfg.enable_thinking is not None:
        try:
            rendered = tokenizer.apply_chat_template(msgs, enable_thinking=cfg.enable_thinking, **kw)
        except TypeError:
            rendered = tokenizer.apply_chat_template(msgs, **kw)
    else:
        rendered = tokenizer.apply_chat_template(msgs, **kw)
    return rendered + cfg.no_think_suffix + cfg.answer_prefix


def render_all(items: list[ProbeItem], cfg: RenderConfig, tokenizer=None) -> list[str]:
    return [render_item(it, cfg, tokenizer) for it in items]
