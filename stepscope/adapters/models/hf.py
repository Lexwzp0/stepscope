"""Hugging Face causal-LM adapter with the default forward-hook backend.

Required: ``forward``, ``param_groups``, ``feature_sites``, ``named_parameters``. Optional capabilities are
declared per architecture family from ``SITE_SPECS`` (keyed by ``config.model_type``):

    hidden_states     every HF causal LM (``output_hidden_states``)
    readout           the output embedding exists (``get_output_embeddings``)
    residual_inputs   the architecture has a site spec and every module it names exists

A new backbone family is one ``SiteSpec`` line (or ``sites=`` when constructing the adapter); analysis code
does not change.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass

import torch

from ...core.registry import MODELS
from .base import ResidualInputs

_LAYER = re.compile(r"^(?:model\.|transformer\.)?(?:language_model\.)?(?:layers|h)\.(\d+)\.")


@dataclass(frozen=True)
class SiteSpec:
    """Where the residual-branch inputs live. ``branches`` lists (site suffix, norm module inside a block)."""
    layers: str
    branches: tuple[tuple[str, str], ...]
    final_norm: str


LLAMA_LIKE = SiteSpec("model.layers", (("attn_in", "input_layernorm"), ("mlp_in", "post_attention_layernorm")), "model.norm")
GPT2_LIKE = SiteSpec("transformer.h", (("attn_in", "ln_1"), ("mlp_in", "ln_2")), "transformer.ln_f")
SITE_SPECS: dict[str, SiteSpec] = {
    "llama": LLAMA_LIKE, "mistral": LLAMA_LIKE, "qwen2": LLAMA_LIKE, "qwen3": LLAMA_LIKE, "smollm3": LLAMA_LIKE,
    "gpt2": GPT2_LIKE,
}


@MODELS.register("hf")
class HFModelAdapter:
    name = "hf"

    def __init__(self, model, tokenizer=None, path: str = "", sites: SiteSpec | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.path = path
        self.model_type = getattr(getattr(model, "config", None), "model_type", "")
        self.spec = sites or SITE_SPECS.get(self.model_type)
        self._sites = self._resolve_sites()
        self.capabilities = self._declare()

    @classmethod
    def load(cls, path: str, dtype: str = "float32", device: str = "cpu",
             attn_implementation: str = "eager", sites: SiteSpec | None = None) -> "HFModelAdapter":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch_dtype, attn_implementation=attn_implementation)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        tok = AutoTokenizer.from_pretrained(path)
        return cls(model, tok, path, sites=sites)

    # --- declaration -------------------------------------------------------------------------------
    def _readout_name(self) -> str | None:
        out = self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else None
        if out is None:
            return None
        for name, mod in self.model.named_modules():
            if mod is out:
                return name
        return None

    def _resolve_sites(self) -> "OrderedDict[str, str]":
        sites: OrderedDict[str, str] = OrderedDict()
        modules = dict(self.model.named_modules())
        if self.spec is not None and self.spec.layers in modules and self.spec.final_norm in modules:
            blocks = modules[self.spec.layers]
            ok = True
            for i in range(len(blocks)):
                for suffix, attr in self.spec.branches:
                    name = f"{self.spec.layers}.{i}.{attr}"
                    if name not in modules:
                        ok = False
                    sites[f"layers.{i}.{suffix}"] = name
            if ok:
                sites["final_norm"] = self.spec.final_norm
            else:
                sites.clear()
        ro = self._readout_name()
        if ro is not None:
            sites["readout"] = ro
        return sites

    def _declare(self) -> frozenset[str]:
        caps = {"hidden_states"}
        if "readout" in self._sites:
            caps.add("readout")
        if "final_norm" in self._sites and "readout" in self._sites:
            caps.add("residual_inputs")
        return frozenset(caps)

    # --- required ----------------------------------------------------------------------------------
    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, batch: dict):
        return self.model(**batch)

    def named_parameters(self) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict((k, v) for k, v in self.model.named_parameters())

    def param_groups(self) -> dict[str, list[str]]:
        prefix = re.compile(rf"^{re.escape(self.spec.layers)}\.(\d+)\.") if self.spec else _LAYER
        groups: dict[str, list[str]] = {}
        for k, _ in self.model.named_parameters():
            m = prefix.match(k) or _LAYER.match(k)
            g = f"layers.{m.group(1)}" if m else k.rsplit(".", 1)[0]
            groups.setdefault(g, []).append(k)
        return groups

    def feature_sites(self) -> dict[str, str]:
        return dict(self._sites)

    def branch_sites(self) -> list[str]:
        return [s for s in self._sites if s.startswith("layers.")]

    @property
    def readout_tied(self) -> bool:
        """True when the readout shares its tensor with the input embedding (an update of that tensor also
        moves the input path, which the readout channel does not see)."""
        out = self.model.get_output_embeddings()
        inp = self.model.get_input_embeddings()
        return out is not None and inp is not None and out.weight is inp.weight

    def free(self) -> None:
        self.model = None

    # --- optional (declared above) -----------------------------------------------------------------
    def hidden_states(self, batch: dict) -> tuple:
        with torch.no_grad():
            return self.model(**batch, output_hidden_states=True).hidden_states

    def readout(self) -> torch.Tensor:
        if "readout" not in self.capabilities:
            raise NotImplementedError(f"{self.model_type}: readout not declared")
        return self.model.get_output_embeddings().weight

    def residual_inputs(self, batch: dict) -> ResidualInputs:
        if "residual_inputs" not in self.capabilities:
            raise NotImplementedError(f"{self.model_type}: residual_inputs not declared (no site spec)")
        modules = dict(self.model.named_modules())
        records: dict[str, torch.Tensor] = {}
        handles = []

        def hook(site):
            def fn(_m, _inp, out):
                records[site] = (out[0] if isinstance(out, tuple) else out).detach()
            return fn

        wanted = self.branch_sites() + ["final_norm"]
        try:
            for site in wanted:
                handles.append(modules[self._sites[site]].register_forward_hook(hook(site)))
            with torch.no_grad():
                out = self.model(**batch)
        finally:
            for h in handles:
                h.remove()
        return ResidualInputs(branches=OrderedDict((s, records[s]) for s in self.branch_sites()),
                              final=records["final_norm"], logits=out.logits.detach())
