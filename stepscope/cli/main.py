"""stepscope CLI (blueprint §九): ingest | capture | observe | analyze | report | compare | interval."""
from __future__ import annotations

import argparse
import json
import sys

COMMANDS = ("ingest", "capture", "observe", "analyze", "report", "compare", "interval")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stepscope")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest", help="scan a verl output directory and write <out>/<run_id>/run.json")
    s.add_argument("path"); s.add_argument("--out", required=True); s.add_argument("--run-id"); s.add_argument("--framework", default="verl")
    sub.add_parser("capture", help="attach capture hooks to a training run (optional)").add_argument("config")
    s = sub.add_parser("observe", help="run probe sets on every checkpoint"); s.add_argument("run"); s.add_argument("--probes", nargs="+", required=True)
    s = sub.add_parser("analyze", help="run a study or named analyses"); s.add_argument("run"); s.add_argument("--study"); s.add_argument("--analyses", nargs="*")
    s = sub.add_parser("report", help="re-render the static HTML report of a run directory"); s.add_argument("run_dir"); s.add_argument("--out")
    s = sub.add_parser("compare", help="compare runs on the same probes and criteria"); s.add_argument("runs", nargs="+")

    s = sub.add_parser("interval", help="two HF checkpoints + probe file -> actual change, first-order prediction, residual, report")
    s.add_argument("--from", dest="ckpt_from", help="HF checkpoint directory (θ_from)")
    s.add_argument("--to", dest="ckpt_to", help="HF checkpoint directory (θ_to)")
    s.add_argument("--run", help="run.json written by `stepscope ingest`; use with --steps")
    s.add_argument("--steps", nargs=2, type=int, metavar=("FROM", "TO"), help="two global steps listed in run.json")
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

    run_id = args.run_id
    if args.run:
        if not args.steps:
            raise SystemExit("--run needs --steps FROM TO")
        run = json.loads(__import__("pathlib").Path(args.run).read_text(encoding="utf-8"))
        by_step = {int(c["step"]): c for c in run["checkpoints"]}
        missing = [s for s in args.steps if s not in by_step]
        if missing:
            pending = {int(p["step"]): p["merge_cmd"] for p in run.get("pending", [])}
            hints = "\n".join(f"  step {s}: {pending.get(s) or 'not found'}" for s in missing)
            raise SystemExit(f"steps without HF weights in {args.run}:\n{hints}")
        args.ckpt_from, args.ckpt_to = by_step[args.steps[0]]["path"], by_step[args.steps[1]]["path"]
        args.step_from, args.step_to = args.steps
        run_id = run_id or f"{run['run_id']}__{args.steps[0]}_{args.steps[1]}"
    elif not (args.ckpt_from and args.ckpt_to):
        raise SystemExit("give --from and --to, or --run with --steps")
    rc = {}
    if args.instruction is not None: rc["instruction"] = args.instruction
    if args.answer_prefix is not None: rc["answer_prefix"] = args.answer_prefix.replace("\\n", "\n")
    if args.label_format is not None: rc["label_format"] = args.label_format
    if args.labels: rc["labels"] = tuple(args.labels)
    if args.system is not None: rc["system"] = args.system
    if args.chat_template: rc["chat_template"] = True
    summary = run_interval(IntervalArgs(
        ckpt_from=args.ckpt_from, ckpt_to=args.ckpt_to, probes=args.probes, out=args.out, run_id=run_id,
        step_from=args.step_from, step_to=args.step_to, observables=tuple(args.observables), method=args.method,
        batch_size=args.batch_size, dtype=args.dtype, device=args.device, render=RenderConfig(**rc), report=args.report,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _ingest(args) -> int:
    from pathlib import Path

    if args.framework != "verl":
        raise SystemExit(f"framework {args.framework!r} not supported yet (verl only)")
    from ..adapters.frameworks.verl import run_from_verl

    run = run_from_verl(args.path, args.run_id)
    out_dir = Path(args.out) / run.id
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        **run.meta,
        "run_id": run.id, "framework": run.framework, "config_hash": run.config_hash,
        "checkpoints": [{"step": c.step, "path": c.path} for c in run.checkpoints],
    }
    (out_dir / "run.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    usable = [c.step for c in run.checkpoints]
    print(f"run {run.id}: {len(usable)} checkpoints with HF weights {usable}; {len(run.meta['pending'])} pending merge")
    for p in run.meta["pending"]:
        print(f"  step {p['step']}: {p['merge_cmd'] or 'no shards found'}")
    print(out_dir / "run.json")
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
    if args.cmd == "ingest":
        return _ingest(args)
    print(f"stepscope {args.cmd}: not implemented yet (skeleton)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
