# Usage

[Back to overview](../README.md) · [Examples](../examples/README.md)

## Compare your own checkpoints

```bash
stepscope interval --from checkpoints/before --to checkpoints/after \
  --probes probes.jsonl --out store --protocol template.json --device cuda
```

Use `stepscope interval --help` for observable selection, JVP/backward methods,
batch size, dtype and item limits. `--protocol` supplies your frozen rendering
settings; see the observable definitions below.

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

## Analyze one optimizer step

```bash
stepscope step --checkpoint checkpoints/before --batch batch.jsonl \
  --probes probes.jsonl --out store --objective sft \
  --optimizer sgd --lr 1e-3 --protocol template.json --device cuda
```

Use `stepscope step --help` for AdamW, weight decay, clipping, and MOPD teacher
options. The [real 3B example](../examples/real_case/README.md) prepares all inputs
and shows the complete MOPD workflow. These commands require your own local
checkpoints, probe file and rendering protocol.

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
