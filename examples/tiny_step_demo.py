"""Offline demo of the single-step check chain. No downloads, no GPU.

    python examples/tiny_step_demo.py --out step_demo [--arch llama|gpt2] [--objective sft|mopd] [--optimizer sgd|adamw]

Builds a random tiny model (and, for MOPD, two random teachers routed by source), writes a 4-row training batch
from two sources and a 64-item probe file, takes one real optimizer step and runs

    CH1 + CH2  →  exact first order along the batch's own update  →  actual change after the step

writing the tables, an update package and a static HTML report. Switching backbone or objective only changes
the adapter and the configuration; the analysis code is the same.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stepscope.capture.train_step import OptimizerConfig
from stepscope.pipelines.step import StepArgs, run_step
from stepscope.testing.tiny import BACKBONES, RENDER, save_checkpoint, write_batch_file, write_probe_file

# default step sizes inside the first-order regime of each random tiny model (GPT-2 at init is more curved)
LR = {("llama", "sgd"): 3e-3, ("gpt2", "sgd"): 3e-4, ("llama", "adamw"): 3e-5, ("gpt2", "adamw"): 1e-5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step_demo")
    ap.add_argument("--arch", choices=sorted(BACKBONES), default="llama")
    ap.add_argument("--objective", choices=["sft", "mopd"], default="sft")
    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    ap.add_argument("--lr", type=float, help="default: a step size inside the first-order regime of the tiny model")
    ap.add_argument("--extra-probes", type=int, default=56, help="synthetic items added to the 8 hand-written ones")
    args = ap.parse_args()
    out = Path(args.out)
    ck = save_checkpoint(BACKBONES[args.arch](0), out / "checkpoints" / f"tiny_{args.arch}")
    teachers = None
    if args.objective == "mopd":
        teachers = {"math": str(save_checkpoint(BACKBONES[args.arch](7), out / "checkpoints" / "teacher_math")),
                    "code": str(save_checkpoint(BACKBONES[args.arch](8), out / "checkpoints" / "teacher_code"))}
    lr = args.lr or LR[(args.arch, args.optimizer)]
    s = run_step(StepArgs(checkpoint=str(ck), batch=str(write_batch_file(out / "batch.jsonl")),
                          probes=str(write_probe_file(out / "probes.jsonl", args.extra_probes)), out=str(out / "store"),
                          objective=args.objective, teachers=teachers, source_kind="teacher" if teachers else "domain",
                          optimizer=OptimizerConfig(args.optimizer, lr=lr), render=RENDER))
    show = {k: s[k] for k in ("run_id", "model_type", "objective", "optimizer", "loss", "n_items", "cost_s_total")}
    show["check_chain"] = {k: {m: round(v[m], 4) for m in ("corr", "sign_agree")} for k, v in s["consistency"].items()}
    show["diagnostics"] = {k: s["diagnostics"][k] for k in ("tangent", "reconstruction", "readout_channel_check")
                           if k in s["diagnostics"]}
    print(json.dumps(show, indent=2, default=str))
    print(f"report: {s['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
