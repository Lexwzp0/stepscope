# Examples

Offline, CPU, no downloads:

- `tiny_demo.py` — chain one: two checkpoints of a tiny random Llama one SGD step apart → actual change,
  first-order prediction, consistency, report.
- `tiny_step_demo.py` — single-step chain: a tiny random model (`--arch llama|gpt2`), a 4-row batch from two
  sources, 64 probe items, one SGD or AdamW step with the SFT or MOPD objective → update package, CH1 + CH2 →
  exact first order → actual change, report. Channel and source analyses are SGD-only.

Real, trained checkpoint inputs:

- [Open-MOPD 3B calibration](real_case/README.md): pinned MixSFT and three domain teachers,
  six generated sequences, and 20 independent MMLU-Redux questions with four answer-position rotations.
  `prepare_real_case.py` prepares the inputs; `run_real_case.py` executes one SGD or AdamW calibration.

[Usage guide](../docs/usage.md) · [Validation results](../docs/validation.md)
