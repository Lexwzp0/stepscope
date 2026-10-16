"""CaptureHooks (blueprint §五): the platform's thin framework layer around one real optimizer step.

Division of labour:
- dattri-llm's HookManager callbacks capture per-sample gradients and Adam moments.
  Its OptimizerStateCallback.on_step_end fires inside backward, BEFORE optimizer.step();
  the post-step moments need record_post(step).
- CaptureHooks aligns all micro-batches of one optimizer step, records references to the
  pre- and post-update parameters (or the measured Δθ) and writes the UpdatePackage.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CaptureHooks(Protocol):
    def on_optimizer_step_begin(self, step: int) -> None:
        """Record pre-update parameter / optimizer-state references; open micro-batch collection."""

    def on_microbatch(self, step: int, batch, meta: dict) -> None:
        """Attach batch metadata (teacher ids, domains, weights, logprobs) to the open step."""

    def on_optimizer_step_end(self, step: int) -> None:
        """Record post-update parameters or measured Δθ; close collection; emit an UpdatePackage."""
