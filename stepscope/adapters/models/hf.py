"""Hugging Face model adapter with the default forward-hook backend (skeleton)."""
from __future__ import annotations

from ...core.registry import MODELS


@MODELS.register("hf")
class HFModelAdapter:
    name = "hf"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, batch):
        raise NotImplementedError

    def param_groups(self) -> dict[str, list]:
        raise NotImplementedError

    def residual_inputs(self, batch) -> dict:
        # Default backend: forward hooks on input_layernorm / post_attention_layernorm outputs
        # and the final norm (cf. ood_ld NormedStateRecorder). Optional nnsight backend later.
        raise NotImplementedError

    def readout(self):
        raise NotImplementedError
