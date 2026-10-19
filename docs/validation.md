# Validation

[Back to overview](../README.md) · [Reproduce the 3B case](../examples/real_case/README.md)

## Real model: one Open-MOPD step on SmolLM3-3B

Setup ([reproduction instructions](../examples/real_case/README.md)): pinned public MixSFT checkpoint and its three teachers;
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

## Offline tiny-model checks

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
