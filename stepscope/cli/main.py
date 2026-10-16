"""stepscope CLI (blueprint §九): ingest | capture | observe | analyze | report | compare | interval."""
from __future__ import annotations

import argparse
import json
import sys

COMMANDS = ("ingest", "capture", "observe", "analyze", "report", "compare", "interval")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stepscope")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="register a training output directory as a Run").add_argument("path")
    sub.add_parser("capture", help="attach capture hooks to a training run (optional)").add_argument("config")
    s = sub.add_parser("observe", help="run probe sets on every checkpoint"); s.add_argument("run"); s.add_argument("--probes", nargs="+", required=True)
    s = sub.add_parser("analyze", help="run a study or named analyses"); s.add_argument("run"); s.add_argument("--study"); s.add_argument("--analyses", nargs="*")
    s = sub.add_parser("report", help="re-render the static HTML report of a run directory"); s.add_argument("run_dir"); s.add_argument("--out")
    s = sub.add_parser("compare", help="compare runs on the same probes and criteria"); s.add_argument("runs", nargs="+")

    s = sub.add_parser("interval", help="two HF checkpoints + probe file -> actual change, first-order prediction, residual, report")
    s.add_argument("--from", dest="ckpt_from", required=True, help="HF checkpoint directory (θ_from)")
    s.add_argument("--to", dest="ckpt_to", required=True, help="HF checkpoint directory (θ_to)")
    s.add_argument("--probes", required=True, help="probe file (JSON lines)")
    s.add_argument("--out", required=True, help="store root; tables go to <out>/<run_id>/")
    s.add_argument("--run-id")
    s.add_argument("--step-from", type=int); s.add_argument("--step-to", type=int)
    s.add_argument("--observables", nargs="+", default=["phi_fmt", "phi_ans"])
    s.add_argument("--method", choices=["auto", "jvp", "backward"], default="auto")
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--dtype", default="float32"); s.add_argument("--device", default="cpu")
    s.add_argument("--instruction"); s.add_argument("--answer-prefix"); s.add_argument("--label-format")
    s.add_argument("--labels", nargs="+"); s.add_argument("--system")
    s.add_argument("--chat-template", action="store_true")
    s.add_argument("--report", help="report path (default <out>/<run_id>/report.html)")
    return p


def _interval(args) -> int:
    from ..observe.probes import RenderConfig
    from ..pipelines.interval import IntervalArgs, run_interval

    rc = {}
    if args.instruction is not None: rc["instruction"] = args.instruction
    if args.answer_prefix is not None: rc["answer_prefix"] = args.answer_prefix.replace("\\n", "\n")
    if args.label_format is not None: rc["label_format"] = args.label_format
    if args.labels: rc["labels"] = tuple(args.labels)
    if args.system is not None: rc["system"] = args.system
    if args.chat_template: rc["chat_template"] = True
    summary = run_interval(IntervalArgs(
        ckpt_from=args.ckpt_from, ckpt_to=args.ckpt_to, probes=args.probes, out=args.out, run_id=args.run_id,
        step_from=args.step_from, step_to=args.step_to, observables=tuple(args.observables), method=args.method,
        batch_size=args.batch_size, dtype=args.dtype, device=args.device, render=RenderConfig(**rc), report=args.report,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _report(args) -> int:
    from ..report.html import render
    out = args.out or str(__import__("pathlib").Path(args.run_dir) / "report.html")
    print(render(args.run_dir, out))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "interval":
        return _interval(args)
    if args.cmd == "report":
        return _report(args)
    print(f"stepscope {args.cmd}: not implemented yet (skeleton)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
