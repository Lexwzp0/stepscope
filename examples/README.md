# Examples

Offline, CPU, no downloads:

- `tiny_demo.py` — chain one: two checkpoints of a tiny random Llama one SGD step apart → actual change,
  first-order prediction, consistency, report.
- `tiny_step_demo.py` — single-step chain: a tiny random model (`--arch llama|gpt2`), a 4-row batch from two
  sources, 64 probe items, one SGD or AdamW step with the SFT or MOPD objective → update package, CH1 + CH2 →
  exact first order → actual change, report.

Public checkpoint sequences for end-to-end runs. Candidates (not yet wired):

- Tülu 3.1 8B, DPO start → GRPO checkpoints (the lineage used by rlvr-safety-dynamics).
- MOPD checkpoints, subject to the open-source boundary.
