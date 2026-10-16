"""Hugging Face causal-LM adapter with the default forward-hook backend.

Required capabilities: ``forward`` (answer-slot logits via observe.scoring), ``param_groups``,
``named_parameters``. Optional capabilities for the LDCL plugin (``residual_inputs``, ``readout``)
are not implemented yet; ``supports()`` reports that."""
from __future__ import annotations

import re
from collections import OrderedDict

import torch

from ...core.registry import MODELS

_LAYER = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.")


@MODELS.register("hf")
class HFModelAdapter:
    name = "hf"

    def __init__(self, model, tokenizer, path: str = ""):
        self.model = model
        self.tokenizer = tokenizer
        self.path = path

    @classmethod
    def load(cls, path: str, dtype: str = "float32", device: str = "cpu",
             attn_implementation: str = "eager") -> "HFModelAdapter":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch_dtype, attn_implementation=attn_implementation)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        tok = AutoTokenizer.from_pretrained(path)
        return cls(model, tok, path)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def named_parameters(self) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict((k, v) for k, v in self.model.named_parameters())

    def param_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for k, _ in self.model.named_parameters():
            m = _LAYER.match(k)
            g = f"layers.{m.group(1)}" if m else k.rsplit(".", 1)[0]
            groups.setdefault(g, []).append(k)
        return groups

    def free(self) -> None:
        self.model = None

    # optional capabilities (LDCL plugin): not implemented in this adapter yet
    def residual_inputs(self, batch):
        raise NotImplementedError("residual-branch inputs are an optional capability; not implemented")

    def readout(self):
        raise NotImplementedError("readout matrix is an optional capability; not implemented")
