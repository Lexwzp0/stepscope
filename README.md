# StepScope

**A research toolkit for post-training dynamics.**

[![tests](https://github.com/Lexwzp0/stepscope/actions/workflows/tests.yml/badge.svg)](https://github.com/Lexwzp0/stepscope/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

StepScope treats **one real parameter update** as the basic object of study. For any backbone,
any post-training objective (SFT, GRPO, DPO, on-policy distillation, ...) and any optimizer it
measures what the update actually did to a set of behaviours, predicts that change to first order
along the measured update, and reports how well the prediction holds:

```
ΔΦ  = Φ(θ + Δθ) − Φ(θ)        actual change of a differentiable observable
ΔΦ̂ = ∇Φ(θ) · Δθ               first-order prediction along the real update
```

Everything else — per-source decomposition of an AdamW step, the LDCL token-interaction
channels, behaviour trajectories, representation metrics, counterfactual data-intervention
estimates — is an analysis plugin writing to fixed tables, with a consistency table that
checks every prediction against the real update.

> Status: **early research release (0.1.0)**. Chain one — *two checkpoints → per-item actual
> change, first-order prediction, residual, HTML report* — is implemented, numerically validated
> and tested offline. The rest of the design is interfaces and stubs; see *Roadmap*.

## Quick start (offline, CPU, under a minute)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your own torch
pip install -e ".[torch]"
python examples/tiny_demo.py --out demo_out
```

This builds a random tiny Llama, takes one real SGD step, saves both as HF checkpoints, writes an
8-item probe file and runs the analysis. Open `demo_out/store/runA__0_10/report.html`: section 2 is
the prediction-calibration scatter (actual vs first-order), section 1 the change by domain.

## Chain one on your own checkpoints

```bash
stepscope interval --from runs/global_step_100/actor/huggingface \
                   --to   runs/global_step_200/actor/huggingface \
                   --probes probes/gpqa_dev.jsonl --out store \
                   [--observables phi_fmt phi_ans] [--method auto|jvp|backward] [--batch-size 4] \
                   [--dtype float32] [--device cuda] [--max-items 200] \
                   [--protocol template.json | --chat-template --no-thinking --answer-prefix "\nAnswer:" --label-format " {label}"]
```

verl outputs: `stepscope ingest <default_local_dir> --out store --run-id R` scans
`global_step_*/actor/huggingface`, writes `store/R/run.json` (usable checkpoints; pending ones with the
exact `verl.model_merger` command), then `stepscope interval --run store/R/run.json --steps 100 200 ...`.

Outputs under `store/<run_id>/` (`run_id` defaults to `<training run>__<from>_<to>`):

| file | content |
|---|---|
| `effects/*.parquet` | one row per item × observable: `value_from`, `value_to`, `actual`, `first_order`, `method`, `domain` |
| `behavior/*.parquet` | per checkpoint × item: `p_correct`, `argmax_in_set_correct` |
| `consistency/*.parquet` | per observable: `n`, Pearson `corr`, `sign_agree`, residual mean / RMS, actual RMS |
| `interval_<a>_<b>.json` | inputs and hashes, steps, render config, option token ids, `‖Δθ‖` per layer group, timings, peak memory |
| `report_<a>_<b>.html` | change by domain, prediction-calibration scatter, items by \|actual\| (no external assets) |

Several intervals may share one `run_id`; each report shows only its own interval and probe version
(`report.html` / `interval.json` are copies of the latest; `stepscope report <run_dir> --interval 100_200` re-renders one).

**Probes** are multiple-choice items, one JSON object per line:
`{"id": ..., "domain": ..., "question": ..., "options": [...], "answer_idx": k}`.
`stepscope probes convert rows.jsonl --out probes.jsonl [--split dev] [--rotations all|first]` turns rows with
`source / item_id / rotation / split` fields into this format (option order and `answer_idx` untouched);
`--max-items N` takes a deterministic subset by id hash.

**Observables** at the answer slot after the answer prefix: `phi_fmt = log Σ_options p(label)`
(willingness to answer in the required form) and `phi_ans = log p(correct) − log Σ_wrong p(label)`
(correct vs wrong inside the option set). Labels are resolved on the full rendered prompt, item by item,
and must be a single clean token there; the resulting encoding is shared by both checkpoints and both
first-order algorithms, so every measured change refers to the same observable on the same tokens.
`--protocol template.json` loads a frozen direct-scoring protocol (chat template on, `enable_thinking=False`,
an empty think block as suffix, answer prefix, label format, a frozen date so prompts do not change daily).

**First-order prediction** uses forward-mode AD (`torch.func.jvp`) along Δθ with answer-slot-only
logits; `--method backward` is the reverse-mode cross-check. Between checkpoints Δθ is the net effect of
many optimizer steps, so this is a first-order approximation of a multi-step net update: trajectories and
a coarse consistency check, never a per-source decomposition. Low correlation on real intervals is a
result, not a failure.

## Validation

Tiny random Llama, offline, CPU (`pytest`, 25 tests):

| one real update | max error / max actual change | JVP vs backward | correlation |
|---|---:|---:|---:|
| SGD, lr 1e-2 | 0.8 % | 3e-10 | 0.99999 |
| AdamW, lr 1e-4 | 0.6 % | 3e-10 | 0.99997 |

Scaling the same Δθ by 1/2 and 1/4 shrinks the error by 0.250 and 0.261 — the second-order term.

## Design

Four axes are kept orthogonal: **backbones** (model adapters), **objectives** (per-token weights and
forces), **analyses** (plugins over fixed tables) and **phenomena** (study templates). Two kinds of input:

- **Interval record** — two checkpoints plus the source log of that interval (chain one; works with
  checkpoints alone).
- **Update package** — one real optimizer step: pre-update model and optimizer state, a reference to the
  post-update parameters (or the measured Δθ), every micro-batch consumed by that step, the effective
  training config. Required for per-source decomposition and single-step calibration.

Tables: `effects`, `contributions`, `channels`, `behavior`, `representation`, `consistency`,
`interventions` (`stepscope/core/schema.py`). `contributions` holds only decompositions whose parts sum
to the first-order prediction; `interventions` holds counterfactual estimates (SOURCE, TracIn, influence
functions), which are never added to `effects`. Every row carries run, step, probe version, analysis
version, code hash, approximation level and cost.

```
stepscope/
  core/        data model, parquet store + manifest, provenance, registry
  adapters/    models (HF), objectives (SFT/GRPO/DPO/OPD/MOPD stubs), optimizers (AdamW stub), frameworks (verl)
  capture/     update-package capture: CaptureHooks around one optimizer step (interface)
  observe/     probe files, rendering, answer-slot encoding, differentiable observables, conversion
  analyze/     dynamics (interval: actual + first-order), consistency; behavior / representation / interventions stubs
  pipelines/   interval.py (chain one)
  report/      static HTML
  cli/         stepscope interval | ingest | report | probes
  testing/     offline tiny model + tokenizer used by tests and the demo
```

## Roadmap

1. Update-package capture around one real optimizer step (gradients and optimizer moments via
   [dattri-llm](https://github.com/TRAIS-Lab/dattri-llm) callbacks; the platform records both sides of the step).
2. AdamW per-source decomposition (weight decay, inherited momentum, each source) validated against the
   measured Δθ; teacher × capability contribution maps for multi-teacher distillation.
3. LDCL interaction channels as a plugin with stated applicability (SGD-direction updates).
4. Behaviour layer: trajectories, flips, failure categories, counterbalanced wordings; lighteval bridge.
5. Representation layer: layer-wise probes, RankMe / α-ReQ, readout geometry.
6. Data-intervention estimates via [bergson](https://github.com/EleutherAI/bergson) / dattri-llm, kept apart from decompositions.

## Not reimplemented here

Per-sample gradient capture and data attribution (dattri-llm, bergson), model internals access beyond
plain hooks (nnsight), public benchmarks (lighteval), training (verl, TRL).

## References

- Ren & Sutherland, *Learning Dynamics of LLM Finetuning*, ICLR 2025. [arXiv:2407.10490](https://arxiv.org/abs/2407.10490)
- Ren, Deng, Hong, Lyle, Gal, *Learning Dynamics of Continual Learning*, 2026. [arXiv:2609.33620](https://arxiv.org/abs/2609.33620)

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT. Parts of `observe/` were first written for the author's own distillation experiments and ported here.
