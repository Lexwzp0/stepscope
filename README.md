# StepScope

Post-training analysis platform. The basic object of study is one real update:

    ΔΦ = Φ(θ + Δθ) − Φ(θ),      first-order prediction  ∇Φ · Δθ

for any backbone, any post-training objective (SFT, GRPO, DPO, OPD, MOPD) and any
optimizer. Everything else — per-source decomposition of an AdamW step, the LDCL
token-interaction channels, behaviour trajectories, representation metrics,
counterfactual data-intervention estimates — is an analysis plugin that writes
to fixed tables, with a consistency table that checks predictions against the
real update.

Status (2026-10-17): chain one works end to end; everything else is interface
and stubs. The design blueprint lives in the parent workspace (`../设计.md`,
Chinese).

## Chain one: two checkpoints → actual change, first-order prediction, residual, report

```bash
pip install -e ".[torch,dev]"
stepscope interval --from runs/global_step_100 --to runs/global_step_200 \
    --probes probes/gpqa_dev.jsonl --out store \
    [--observables phi_fmt phi_ans] [--method auto|jvp|backward] [--batch-size 4] \
    [--dtype float32] [--device cuda] [--answer-prefix "\nAnswer:"] [--label-format " {label}"] [--chat-template]
```

Outputs under `store/<run_id>/`:

| file | content |
|---|---|
| `effects/*.parquet` | one row per item × observable: `value_from`, `value_to`, `actual`, `first_order`, `method`, `domain` |
| `behavior/*.parquet` | per checkpoint × item: `p_correct`, `argmax_in_set_correct` |
| `consistency/*.parquet` | per observable: `n`, Pearson `corr`, `sign_agree`, residual mean / RMS, actual RMS |
| `interval.json` | inputs, hashes, steps, render config, option token ids, `‖Δθ‖` per layer group, timings |
| `report.html` | capability change by domain, prediction-calibration scatter, items by \|actual\| |

verl outputs: `stepscope ingest <default_local_dir> --out store [--run-id R]` scans `global_step_*/actor/huggingface`,
writes `store/R/run.json` (usable checkpoints, pending ones with the `verl.model_merger` command), then
`stepscope interval --run store/R/run.json --steps 100 200 --probes ... --out store`.

Run identity: the default `run_id` is `<training run>__<from>_<to>`, derived from
`.../<run>/global_step_<N>/actor/huggingface`, so verl exports (both ending in `huggingface`) never collide.
Several intervals may share one `run_id`: tables are stored per interval, and each
`report_<from>_<to>.html` / `interval_<from>_<to>.json` shows only its own interval and probe version
(`report.html` / `interval.json` are copies of the latest). `stepscope report <run_dir> --interval 0_10` re-renders one.

Probe file (JSON lines): `{"id": ..., "domain": ..., "question": ..., "options": [...], "answer_idx": k}`.
`stepscope probes convert <build_mcq_rows.jsonl> --out probes.jsonl [--split dev] [--rotations all|first]`
converts ood_ld `build_mcq.py` rows (`source` → `domain`; `item_id`, `rotation`, `split` kept; option order and
`answer_idx` untouched). `--max-items N` takes a deterministic subset by id hash.

Rendering protocol: `--protocol template.json` loads the direct-scoring protocol frozen by ood_ld
`check_template.py` (chat template on, `enable_thinking=False`, empty think block suffix, answer prefix,
label format, frozen date so the prompt does not change daily); the same fields are available as flags
(`--chat-template --no-thinking --no-think-suffix ... --answer-prefix ... --label-format " {label}" --fixed-date YYYY-MM-DD`).
`interval.json` records the render config, option token ids, `‖Δθ‖` per layer group, timings and peak memory.
Observables: `phi_fmt = log Σ_options p(label)` (answers in the required form) and
`phi_ans = log p(correct) − log Σ_wrong p(label)` (correct vs wrong inside the option set), both at the
answer slot after `--answer-prefix`. Labels are resolved on the full rendered prompt item by item and
must be a single clean token there; the resulting encoding is shared by both checkpoints and both
first-order algorithms.

First-order prediction uses forward-mode AD (`torch.func.jvp`) along Δθ with answer-slot-only logits;
`--method backward` is the reverse-mode cross-check. Between checkpoints Δθ is the net effect of many
optimizer steps, so this is a first-order approximation of a multi-step net update: trajectories and a
coarse consistency check, never a per-source decomposition. Low correlation on real intervals is a
result, not a failure.

Validation (tiny random Llama, offline, CPU; `pytest`): after one real SGD (lr 1e-2) or AdamW (lr 1e-4)
step the first-order error is a few percent of the largest actual change, JVP and backward agree to
1e-9, and the error falls roughly quadratically when the same Δθ is scaled by 1/2 and 1/4.

## Layout

```
stepscope/
  core/        data model, parquet store + manifest, provenance, registry
  adapters/    models (HF; optional nnsight), objectives, optimizers, frameworks (verl, TRL)
  capture/     update-package capture: dattri-llm callbacks + CaptureHooks (optimizer-step boundaries)
  observe/     probe files, rendering, answer-slot encoding, differentiable observables
  analyze/     behavior/, dynamics/ (interval: actual + first-order), representation/, consistency/, interventions/
  pipelines/   interval.py (chain one)
  studies/     phenomenon templates (ood_spillover first)
  report/      static HTML report (no external assets)
  cli/         stepscope interval | report | ingest | capture | observe | analyze | compare
tests/         synthetic tiny model + char tokenizer; no GPU, no downloads
```

## Two kinds of input

- **IntervalRecord** — two checkpoints plus the source log of that interval (chain one).
- **UpdatePackage** — one real optimizer step: pre-update model and optimizer state, a reference to the
  post-update parameters (or the measured Δθ), every micro-batch consumed by that step, and the
  effective training config. Required for per-source decomposition and single-step calibration
  (not implemented yet).

## Tables

`effects`, `contributions`, `channels`, `behavior`, `representation`, `consistency`, `interventions`
— see `stepscope/core/schema.py`. `contributions` holds only decompositions whose parts sum to the
first-order prediction; `interventions` holds counterfactual estimates (SOURCE, TracIn, influence
functions) which are never added to `effects`.

## Not reimplemented here

Per-sample gradient capture and data attribution (dattri-llm, bergson), model internals access beyond
plain hooks (nnsight), public benchmarks (lighteval), training (verl, TRL).
