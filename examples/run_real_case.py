"""Run one calibration from the pinned inputs prepared by prepare_real_case.py.

The default is a 20-row smoke, not a benchmark. Use --max-items 80 for all
20 independent questions in four answer-position rotations. Preserve every
smoke result when choosing a step size; channel agreement is not a selection
criterion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stepscope.capture.train_step import OptimizerConfig
from stepscope.observe.probes import RenderConfig
from stepscope.pipelines.step import StepArgs, run_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path("runs/real_case"))
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], required=True)
    parser.add_argument("--lr", type=float, help="default: SGD 1e-2; AdamW 1.5e-6")
    parser.add_argument("--weight-decay", type=float, help="default: SGD 0; AdamW .01")
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scored-batch", type=Path,
                        help="reuse package_0/batch.pt from a run at the same pinned MixSFT checkpoint")
    parser.add_argument("--run-id", required=True, help="use a different ID for every optimizer/step size/probe subset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-package", action="store_true")
    args = parser.parse_args()
    root = args.case.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    batch = args.scored_batch.resolve() if args.scored_batch else root / "batch.jsonl"
    for path in (batch, root / "probes.jsonl", root / "protocol.json"):
        if not path.is_file():
            raise SystemExit(f"Missing {path}; finish input preparation first.")
    lr = args.lr if args.lr is not None else (1e-2 if args.optimizer == "sgd" else 1.5e-6)
    decay = args.weight_decay if args.weight_decay is not None else (0.0 if args.optimizer == "sgd" else .01)
    paths = manifest["model_paths"]
    summary = run_step(StepArgs(
        checkpoint=paths["MixSFT"], batch=str(batch), probes=str(root / "probes.jsonl"),
        out=str(root / "results"), run_id=args.run_id, objective="mopd", source_kind="teacher",
        teachers={"math": paths["RL-Math"], "code": paths["RL-Code"], "if": paths["RL-IF"]},
        topk=4, optimizer=OptimizerConfig(args.optimizer, lr=lr, weight_decay=decay, max_grad_norm=1.0),
        render=RenderConfig.from_protocol(root / "protocol.json"), dtype="float32", device=args.device,
        batch_size=args.batch_size, max_items=args.max_items, save_package=not args.no_package,
    ))
    evidence = {"models": manifest["models"], "selection": manifest["selection"],
                "input_manifest": str(root / "manifest.json"), "batch": str(batch),
                "scope": "standalone one-step calibration; not full Open-MOPD training reproduction",
                "summary": summary}
    (Path(summary["run_dir"]) / "real_case.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
