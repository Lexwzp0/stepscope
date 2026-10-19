# Reproduce the real 3B case

[Overview](../../README.md) · [Results and limitations](../../docs/validation.md)

This prepares a small public Open-MOPD case: MixSFT, its three domain teachers,
six student responses, and 80 MMLU-Redux probe rows. It does not train the full
Open-MOPD recipe. Everything downloaded or generated goes into `runs/real_case/`,
which is ignored by Git.

Use a Python environment with `pip install -e '.[torch]' huggingface-hub`.
The model architecture is `SmolLM3ForCausalLM`; use a Transformers version that
provides it. The reported results were produced with Transformers 4.57.1
and PyTorch 2.8.0 (CUDA 12.8); the adapter module layout is also checked with
Transformers 5.20.1. On an RTX PRO 6000, the PyTorch installation must also
support that GPU.

To match the reported Transformers environment, install `transformers==4.57.1`.
The measured peak allocated memory was 72.8 GiB for SGD and 57.3 GiB for AdamW;
SGD peak reserved memory was 76.7 GiB. These measurements used the short sequences
and batch settings below on a 96 GB GPU, and do not cover longer sequences or larger batches.

Prepare inputs while the GPU is stopped, or on another machine with a persistent
disk. These commands download about 27 GB of model weights plus a 0.94 GB prompt
file. Hugging Face's cache holds one copy; repeated preparation reuses it.

```bash
python examples/prepare_real_case.py --download-only --out runs/real_case
```

After preparation, generate the six short responses on the GPU:

```bash
python examples/prepare_real_case.py --generate --out runs/real_case --device cuda
```

Generation uses MixSFT in FP32, seed 20261019, temperature 0.6, top-p 0.95,
top-k 20, no-thinking mode, and at most 128 new tokens. The same already-rendered
prompt token IDs and generated response IDs are saved for analysis. Set
`--max-new-tokens 512` on the first generation call to use the longer cap.
Generation saves progress after each response and resumes with per-row seeds.
Changed selection or generation settings require a different output directory.

The resulting files are:

- `manifest.json`: exact public repositories, revisions, selected row IDs, input
  hashes, dependency versions, and cached local model paths.
- `protocol.json`: a frozen chat template hash, no-thinking setting, fixed date,
  and answer-label protocol validated on every probe.
- `probes.jsonl`: five valid dev questions from each of abstract algebra,
  college computer science, high-school physics, and philosophy. Each question
  appears in four balanced answer-position rotations. These are 20 independent
  questions, not 80 independent questions.
- `prompts.jsonl`: the first two prompts per math/code/IF domain whose rendered
  length is at most 512 tokens. Selection never uses model answers.
- `batch.jsonl`: six generated sequences, saved as `input_ids` and `prompt_len`.
  Use the same file for SGD and AdamW; truncation and EOS flags are retained.

Point `stepscope step` at `batch.jsonl`, `probes.jsonl`, and `protocol.json`.
Read `model_paths` from the manifest for the checkpoint and teacher arguments.
Use `--max-items 20 --batch-size 1` for the first calibration smoke. A 20-row
smoke may contain rotations of the same question; it measures runtime and basic
numerics, not benchmark performance. Keep all 80 rows for the small final case.

Run the first SGD smoke, then inspect its two calibration levels separately:

```bash
python examples/run_real_case.py --optimizer sgd --run-id smoke_sgd --max-items 20
```

The default SGD learning rate is `1e-2`, with zero decay. It is a starting
configuration, not a claimed first-order accuracy threshold. Preserve smoke
outputs if testing smaller steps. Select a step size from first-order-versus-actual
error and numerical resolution, never from CH1/CH2 agreement.

The smoke package saves the scored batch, so subsequent runs can reuse exactly
the same student candidates and teacher log-probabilities without loading the
teachers again. Both runs below start from the original pinned MixSFT weights:

```bash
python examples/run_real_case.py --optimizer sgd --run-id sgd_80 --max-items 80 \
  --scored-batch runs/real_case/results/smoke_sgd/package_0/batch.pt
python examples/run_real_case.py --optimizer adamw --run-id adamw_80 --max-items 80 \
  --scored-batch runs/real_case/results/smoke_sgd/package_0/batch.pt
```

AdamW defaults to learning rate `1.5e-6` and weight decay `0.01`. Each run starts
with a new optimizer and empty moment state; it does not restore moments from an
ongoing training run. Its first-order prediction uses the measured update; source and LDCL channel decompositions are
not available for AdamW. Each run writes its own report, summary, and update
package below `runs/real_case/results/`. `--no-package` avoids another large
update package for additional exploratory smokes.

The README figure adds the same two runs at a 10× smaller step, reusing the same
scored batch, and is drawn from the four result directories:

```bash
python examples/run_real_case.py --optimizer sgd --lr 1e-3 --run-id sgd_lr1e3_80 --max-items 80 \
  --scored-batch runs/real_case/results/smoke_sgd/package_0/batch.pt
python examples/run_real_case.py --optimizer adamw --lr 1.5e-7 --run-id adamw_lr1p5e7_80 --max-items 80 \
  --scored-batch runs/real_case/results/smoke_sgd/package_0/batch.pt --no-package
python examples/real_case/plot_calibration.py runs/real_case/results --out docs/img   # needs matplotlib, pandas
```

The plot script exports light and dark SVGs for the README, plus PNG copies.
Matplotlib renders the formulas directly; no LaTeX installation is needed.

The student uses tied input/output embeddings. The adapter supports its module
layout, but the independent readout-channel exactness check is skipped for tied
weights. Preserve that fact in the report.

All four model repositories are public, ungated, and Apache-2.0. MMLU-Redux 2.0
is public, ungated, and CC-BY-4.0. The original training mixture's card requires
following each upstream source's terms and does not specify a unified license.
Publish the scripts, pinned source references, row IDs, and aggregate results;
keep downloaded examples and generated token sequences in the ignored run
directory. GPQA is gated and is not needed for this first case.

Sources:

- [Open-MOPD data](https://huggingface.co/datasets/BytedTsinghua-SIA/Open-MOPD-Data/tree/9e897efe3257599d4300e2d5ee865a1cc714af87)
- [MMLU-Redux 2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0/tree/372ea425445d51e1ba1188c56e5e893f8138621f)
- [MixSFT](https://huggingface.co/BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT/tree/c9e7bad031667828656ead188d7e8ea162c048a4)

The probe item-level dev split and option rotations follow the existing OOD-LD
preparation convention. This small no-thinking batch is a reproducible update
calibration, not a reproduction of the original long-rollout training run.
