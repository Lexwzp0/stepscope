"""Offline 30-second demo of chain one. No downloads, no GPU.

    python examples/tiny_demo.py --out demo_out

Builds a random tiny Llama, takes one real SGD step on a random batch, saves both as HF checkpoints
laid out like a verl export, writes an 8-item probe file, then runs the interval analysis:
actual change vs first-order prediction per item, consistency, and a static HTML report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stepscope.pipelines.interval import IntervalArgs, run_interval
from stepscope.testing.tiny import RENDER, make_checkpoint_pair


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--lr", type=float, default=1e-2, help="size of the one SGD step between the two checkpoints")
    ap.add_argument("--method", choices=["auto", "jvp", "backward"], default="auto")
    args = ap.parse_args()
    out = Path(args.out)
    ck_from, ck_to, probes = make_checkpoint_pair(out / "checkpoints", lr=args.lr)
    summary = run_interval(IntervalArgs(ckpt_from=str(ck_from), ckpt_to=str(ck_to), probes=str(probes),
                                        out=str(out / "store"), method=args.method, render=RENDER))
    print(json.dumps({k: summary[k] for k in ("run_id", "n_items", "method", "delta_norm", "cost_s_total", "consistency")},
                     indent=2))
    print(f"report: {summary['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
