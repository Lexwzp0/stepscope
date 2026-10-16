# StepScope

Post-training analysis platform. The basic object of study is one real update:

    ΔΦ = Φ(θ + Δθ) − Φ(θ),      first-order prediction  ∇Φ · Δθ

for any backbone, any post-training objective (SFT, GRPO, DPO, OPD, MOPD) and any
optimizer. Everything else — per-source decomposition of an AdamW step, the LDCL
token-interaction channels, behaviour trajectories, representation metrics,
counterfactual data-intervention estimates — is an analysis plugin that writes
to fixed tables, with a consistency table that checks predictions against the
real update.

Status: skeleton only (2026-10-17). Interfaces, data model and table schemas are
defined; implementations are `NotImplementedError`. The design blueprint lives
in the parent workspace (`../设计.md`, Chinese); this README will be rewritten
from it when the first study runs end to end.

## Layout

```
stepscope/
  core/        data model, parquet store + manifest, provenance, registry
  adapters/    models (HF; optional nnsight), objectives, optimizers, frameworks (verl, TRL)
  capture/     update-package capture: dattri-llm callbacks + CaptureHooks (optimizer-step boundaries)
  observe/     probe sets, differentiable observables, behavioural metrics
  analyze/     behavior/, dynamics/, representation/, consistency/, interventions/
  studies/     phenomenon templates (ood_spillover first)
  report/      static HTML report
  cli/         stepscope ingest | capture | observe | analyze | report | compare
tests/         synthetic small-model tests; no GPU
examples/      public checkpoint sequences
```

## Two kinds of input

- **IntervalRecord** — two checkpoints plus the source log of that interval. Enough
  for trajectories, actual change and an interval first-order approximation.
  Default entry point: works with checkpoints alone.
- **UpdatePackage** — one real optimizer step: pre-update model and optimizer
  state, a reference to the post-update parameters (or the measured Δθ), every
  micro-batch consumed by that step, and the effective training config.
  Required for per-source decomposition and single-step calibration.

Consecutive checkpoints are usually many optimizer steps apart (verl runs
`ppo_epochs × mini-batches` steps per global step), so a checkpoint difference is
never treated as a single-step decomposition input.

## Tables

`effects`, `contributions`, `channels`, `behavior`, `representation`,
`consistency`, `interventions` — see `stepscope/core/schema.py`. `contributions`
holds only decompositions whose parts sum to the first-order prediction;
`interventions` holds counterfactual estimates (SOURCE, TracIn, influence
functions) which are never added to `effects`.

## Not reimplemented here

Per-sample gradient capture and data attribution (dattri-llm, bergson), model
internals access beyond plain hooks (nnsight), public benchmarks (lighteval),
training (verl, TRL).
