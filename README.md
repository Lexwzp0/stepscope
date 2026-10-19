# StepScope

**See how model updates change behavior.** A research toolkit for post-training dynamics.

[![tests](https://github.com/Lexwzp0/stepscope/actions/workflows/tests.yml/badge.svg)](https://github.com/Lexwzp0/stepscope/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

StepScope treats **one real parameter update** as the basic object of study. On supported Hugging Face
causal language models, it measures changes in differentiable observables and checks a first-order
prediction against the actual change. Checkpoint-pair analysis uses the net parameter difference;
the single-step path currently implements SFT and Open-MOPD losses with SGD and AdamW:

```
ΔΦ  = Φ(θ + Δθ) − Φ(θ)        actual change of a differentiable observable
ΔΦ̂ = ∇Φ(θ) · Δθ               first-order prediction along the real update
```

SGD per-source decomposition and approximate LDCL token-interaction channels write to fixed result
tables, with explicit checks against exact first order and actual change. AdamW per-source decomposition,
trajectory studies, representation metrics and data-intervention estimates are planned extensions;
see *Roadmap*.

> Status: **early research release**. Two chains are implemented and tested offline: *two checkpoints →
> per-item actual change, first-order prediction, residual, report*, and *one real optimizer step → LDCL
> channels → exact first order → actual change*, on two tiny backbone families (Llama-like, GPT-2) and two
> objectives (SFT, Open-MOPD). The single-step chain has also been run on a real 3B model (below). The rest
> of the design is interfaces and stubs; see *Roadmap*.

## On a real model

One Open-MOPD update of [Open-MOPD-SmolLM3-3B-MixSFT](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT)
(six student responses scored by the math / code / IF teachers), measured on 80 MMLU-Redux probe rows
(20 questions × 4 answer rotations), FP32:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/real_case_calibration_dark.png">
  <img alt="First-order prediction vs actual change for SGD and AdamW at two step sizes" src="docs/img/real_case_calibration_light.png" width="640">
</picture>

- At a 10× smaller step, **max absolute error / max absolute actual change is at most 2.2 %** for
  each observable and optimizer. Both maxima are taken over all 80 rows; this is not a per-item relative
  error bound. The reduction is consistent with a second-order remainder.
- At AdamW lr `1.5e-6`, the first step with newly initialized optimizer moments has 25.7–26.9 % max error
  by that measure (correlation 0.96). This result does not establish the error of later training steps.
- The LDCL channels CH1 + CH2 do **not** track the exact first order on this model. Details, tables and
  the open question are in [*Validation*](#validation); reproduction in [`examples/real_case/`](examples/real_case/).

## Quick start (offline, CPU, under a minute)

```bash
git clone https://github.com/Lexwzp0/stepscope.git
cd stepscope
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your own torch
pip install -e ".[torch]"
python examples/tiny_demo.py --out demo_out
```

This builds a random tiny Llama, takes one real SGD step, saves both as HF checkpoints, writes an
8-item probe file and runs the analysis. Open `demo_out/store/runA__0_10/report.html`: section 2 is
the prediction-calibration scatter (actual vs first-order), section 1 the change by domain.

```bash
python examples/tiny_step_demo.py --out step_demo --arch llama --objective mopd --optimizer sgd
```

runs the single-step chain below on a tiny model, a 4-row batch from two sources and 64 probe items;
the report is `step_demo/store/<run_id>/report_step.html`. Other supported demo options are
`--arch gpt2`, `--objective sft` and `--optimizer adamw`.

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
| `interval_<tag>.json` | inputs and hashes, steps, render config, option token ids, `‖Δθ‖` per layer group, timings, peak memory |
| `report_<tag>.html` | change by domain, prediction-calibration scatter, items by \|actual\| (no external assets) |

`<tag>` is `<from>_<to>_<probe version>`, so several intervals and probe sets may share one `run_id`; each
report shows only its own interval and probe version (`report.html` / `interval.json` are copies of the latest;
`stepscope report <run_dir> --interval 100_200` re-renders one, or pass the full tag when that interval has
several probe versions). Δθ is taken tensor by tensor while θ_to is released, so the peak is two copies of the
parameters.

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

## Single-step chain: one real update, two approximation checks

```bash
stepscope step --checkpoint ckpt/ --batch batch.jsonl --probes probes.jsonl --out store \
               --objective sft|mopd [--objective-arg weight_mode=teacher_p micro_batch_size=2] \
               [--teacher math=ckpt_math/ code=ckpt_code/ --topk 4] \
               --optimizer sgd|adamw --lr 1e-2 [--weight-decay 0] [--max-grad-norm 1.0] [render options as above]
```

The batch file has one row per line: `{"source": "math", "prompt": "...", "response": "..."}` (or `input_ids` with
`prompt_len` / `loss_mask`); response tokens are supervised and `source` names the row's teacher or domain.
For MOPD the batch may carry `topk_ids` / `teacher_logp`, otherwise the student's own top-K at θ is scored by the
teacher of each source. StepScope replicates the loss, runs backward and one optimizer step, and writes an
**update package** (`package_<step>/`: measured Δθ, pre-update parameters or the checkpoint reference, optimizer
state before the step, the batch, the effective configuration). Then, on the same batch and the same update:

| level | prediction | reference |
|---|---|---|
| 1 | CH1 + CH2: LDCL token-interaction channels per source and residual-branch site (SGD only) | exact first order |
| 2 | exact first order: JVP at θ along the batch's own update (SGD: rebuilt from the batch and checked against the measured Δθ; AdamW: the measured Δθ) | actual change Φ(θ + Δθ) − Φ(θ) |

Under SGD the first order also splits exactly by source (`contributions`, `target = first_order`).
For an untied, bias-free readout, CH1 is checked against the readout part of the loss-gradient update;
CH2 is compared with the remaining loss-gradient first-order effect. Weight decay is excluded from these
channel references. The two independent readout/backbone checks are skipped for tied embeddings, with the
reason recorded. Channel agreement is measured, not assumed. Outputs: `effects`,
`channels`, `contributions`, `behavior`, `consistency`, `step_<tag>.json`, `report_step_<tag>.html`.

Model access goes through declared adapter capabilities (`readout`, `residual_inputs`); the loss, its split
by source and the token-level force f = −∂L/∂z come from the objective adapter. Switching backbone or objective
changes only the adapter and the configuration; analyses that need a capability a model does not declare are
skipped and the reason is written to the manifest.

## Validation

Chain one, tiny random Llama, offline, CPU (`pytest` runs everything here offline):

| one real update | max error / max actual change | JVP vs backward | correlation |
|---|---:|---:|---:|
| SGD, lr 1e-2 | 0.8 % | 3e-10 | 0.99999 |
| AdamW, lr 1e-4 | 0.6 % | 3e-10 | 0.99997 |

Scaling the same Δθ by 1/2 and 1/4 shrinks the error by 0.250 and 0.261 — the second-order term.

Single-step chain, `examples/tiny_step_demo.py` defaults (4-row batch from two sources, 64 probe items, random
initialisation):

| backbone · objective · optimizer | first order vs actual: max error / max change (Φ_fmt, Φ_ans) | readout channel vs JVP | Σ sources vs first order (RMS) |
|---|---:|---:|---:|
| Llama · SFT · SGD 3e-3 | 2.0 %, 1.6 % | 2e-7 | 8e-10 |
| Llama · MOPD · SGD 3e-3 | 4.3 %, 0.3 % | 3e-7 | 2e-10 |
| GPT-2 · SFT · SGD 3e-4 | 2.6 %, 0.8 % | 2e-7 | 2e-10 |
| GPT-2 · MOPD · SGD 3e-4 | 2.1 %, 0.3 % | 2e-7 | 2e-11 |
| Llama / GPT-2 · SFT, MOPD · AdamW | 0.3 – 1.3 % | – | – |

The channel implementation agrees with the official LDCL definitions pair by pair to 1e-7. On these random tiny
models CH2 is 10–100× smaller than the exact non-readout first-order effect (their residual stream entering the blocks has
RMS ≈ 0.02–0.03, and the identity path leaves out the final-norm Jacobian), so CH1 + CH2 tracks the exact first order
only partly: correlation 0.5–0.9 for Φ_ans; for Φ_fmt between −0.3 and 0.7, with the sign wrong on every item for
the Llama with SFT, where the exact non-readout part dominates and has the opposite sign to CH1. Whether the
approximation holds on trained models is what the chain measures there.

### Real model: one Open-MOPD step on SmolLM3-3B

Setup ([`examples/real_case/`](examples/real_case/)): pinned public MixSFT checkpoint and its three teachers;
six student responses (two per math / code / IF source, prompts ≤ 512 tokens, responses ≤ 128 tokens,
no-thinking mode), scored once by the teachers (top-4) and reused by every run; 80 MMLU-Redux dev rows
(5 questions from each of 4 subjects, 4 answer rotations each, so 20 independent questions); FP32, TF32
off, gradient clipping at 1.0, one RTX PRO 6000 per run. Each run starts from the same checkpoint and a new
optimizer; AdamW starts with empty moment state, betas `(0.9, 0.999)` and eps `1e-8`. The experimental code
baseline is [`76d027e`](https://github.com/Lexwzp0/stepscope/commit/76d027e68fe376c3612f6c14b47799f15815c59b).
This is a controlled single-step case, not a reproduction of Open-MOPD training and not a benchmark.

Paired values below are Φ_fmt / Φ_ans. Max error / max change is
`max(abs(first_order - actual)) / max(abs(actual))`; relative RMS error is
`RMS(first_order - actual) / RMS(actual)`, each computed over the 80 rows for one observable.

| optimizer, lr (weight decay) | first order vs actual: corr (Φ_fmt, Φ_ans) | max error / max change | relative RMS error | peak allocated |
|---|---:|---:|---:|---:|
| SGD, 1e-2 (0) | 0.979, 0.992 | 25.6 %, 18.5 % | 21.2 %, 12.9 % | 72.8 GiB |
| SGD, 1e-3 (0) | 0.99990, 0.99993 | **2.2 %, 1.5 %** | 2.4 %, 1.3 % | 72.8 GiB |
| AdamW, 1.5e-6 (0.01) | 0.963, 0.962 | 26.9 %, 25.7 % | 22.4 %, 27.1 % | 57.3 GiB |
| AdamW, 1.5e-7 (0.01) | 0.99978, 0.99964 | **2.2 %, 1.7 %** | 2.3 %, 2.8 % | 57.3 GiB |

Runs shared GPUs with other workloads, so elapsed times are not reported as performance measurements.
Peak allocated memory is per process for this configuration. The smaller AdamW run omitted the update package.
The SGD update rebuilt from the batch matches the measured Δθ within rounding, and the per-source parts
(math, code, IF) sum to the exact first order within 7e-7 (max absolute difference).

Change by domain for the AdamW lr `1.5e-6` step (mean over 5 questions × 4 rotations in each subject).
These are changes in the two observables, not changes in accuracy:

| subject | Φ_fmt actual | Φ_fmt first order | Φ_ans actual | Φ_ans first order |
|---|---:|---:|---:|---:|
| abstract algebra | −0.036 | −0.035 | −0.010 | −0.013 |
| college computer science | −0.078 | −0.073 | +0.008 | +0.014 |
| high-school physics | −0.087 | −0.078 | −0.026 | −0.022 |
| philosophy | −0.032 | −0.014 | +0.052 | +0.046 |

**CH1 + CH2 on this model.** Against the exact first order of the SGD step the channel sum has correlation
−0.72 (Φ_fmt) and −0.40 (Φ_ans), sign agreement 22.5 % and 35 %, and relative RMS error above 100 %. These
numbers stay the same at the 10× smaller step, so the mismatch is not a step-size effect. SmolLM3 ties its
input and output embeddings: the shared matrix's update also acts through the input path, which CH1 does
not include, and the independent readout-channel check is skipped for tied weights. Which part of the gap
comes from the tied path and which from the CH2 approximation is not isolated yet. AdamW has no source or
channel decomposition yet.

## Design

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
   moments via [dattri-llm](https://github.com/TRAIS-Lab/dattri-llm) callbacks); the minimal training step above
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

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT. Parts of `observe/` were first written for the author's own distillation experiments and ported here.
