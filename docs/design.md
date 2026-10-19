# Design and roadmap

[Back to overview](../README.md) · [Usage](usage.md)

Four axes are kept orthogonal: **backbones** (model adapters), **objectives** (per-token weights and
forces), **analyses** (plugins over fixed tables) and **phenomena** (study templates). Two kinds of input:

- **Interval record** — two checkpoints plus the source log of that interval (chain one; works with
  checkpoints alone).
- **Update package** — one real optimizer step: pre-update model and optimizer state, a reference to the
  post-update parameters (or the measured Δθ), every micro-batch consumed by that step, the effective
  training config. Required for per-source decomposition and single-step calibration.

Tables: `effects`, `contributions`, `channels`, `behavior`, `representation`, `consistency`,
`interventions` (`stepscope/core/schema.py`). `contributions` holds only decompositions: every row of one
decomposition shares a `decomp_id`, and its `target` names the column the parts must sum to (`first_order` for
the parts of an update, `actual` for exact path decompositions); `consistency` checks each one against that
column with the same rule. `interventions` holds counterfactual estimates (SOURCE, TracIn, influence
functions), which are never added to `effects`. Every row carries run, step, probe version, analysis
version, code hash, approximation level and cost.

```
stepscope/
  core/        data model, parquet store + manifest, provenance, registry
  adapters/    models (HF; declared capabilities, Llama-like and GPT-2 site specs), objectives (SFT, MOPD;
               GRPO/DPO/OPD stubs), optimizers (AdamW stub), frameworks (verl)
  capture/     training batches, minimal training step, update packages; CaptureHooks for frameworks (interface)
  observe/     probe files, rendering, answer-slot encoding, differentiable observables and their logit gradients
  analyze/     dynamics (interval, single step, forces, LDCL channels), consistency;
               behavior / representation / interventions stubs
  pipelines/   interval.py (chain one), step.py (single-step chain)
  report/      static HTML
  cli/         stepscope interval | step | ingest | report | probes
  testing/     offline tiny model + tokenizer used by tests and the demo
```

## Roadmap

1. Update-package capture inside training frameworks around one real optimizer step (gradients and optimizer
   moments via [dattri-llm](https://github.com/TRAIS-Lab/dattri-llm) callbacks); the implemented minimal training step
   already writes the same package.
2. AdamW per-source decomposition (weight decay, inherited momentum, each source) validated against the
   measured Δθ; teacher × capability contribution maps for multi-teacher distillation.
3. Trajectories: the starting-point channel prediction against real checkpoints, and a symmetric g / K
   crossing to tell a changed training signal from a changed geometry.
4. Behaviour layer: trajectories, flips, failure categories, counterbalanced wordings; lighteval bridge.
5. Representation layer: layer-wise probes, RankMe / α-ReQ, readout geometry.
6. Data-intervention estimates via [bergson](https://github.com/EleutherAI/bergson) / dattri-llm, kept apart from decompositions.

## Not reimplemented here

Per-sample gradient capture and data attribution (dattri-llm, bergson), model internals access beyond
plain hooks (nnsight), public benchmarks (lighteval), training (verl, TRL).

## References

- Ren & Sutherland, *Learning Dynamics of LLM Finetuning*, ICLR 2025. [arXiv:2407.10490](https://arxiv.org/abs/2407.10490)
- Ren, Deng, Hong, Lyle, Gal, *Learning Dynamics of Continual Learning*, 2026. [arXiv:2609.33620](https://arxiv.org/abs/2609.33620)

## License and provenance

[MIT](../LICENSE). Parts of `observe/` were first written for the author's own
distillation experiments and ported here.
