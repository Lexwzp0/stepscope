"""stepscope CLI (blueprint §九): interval | step | ingest | report | probes | (observe | analyze | compare | capture: stubs)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMANDS = ("ingest", "capture", "observe", "analyze", "report", "compare", "interval", "step", "probes")


def _add_render_args(s) -> None:
    s.add_argument("--protocol", help="template.json frozen by ood_ld check_template.py (sets chat template, prefix, labels, thinking, date)")
    s.add_argument("--instruction"); s.add_argument("--answer-prefix"); s.add_argument("--label-format")
    s.add_argument("--labels", nargs="+"); s.add_argument("--system")
    s.add_argument("--chat-template", action="store_true")
    s.add_argument("--no-thinking", action="store_true", help="apply_chat_template(enable_thinking=False) when supported")
    s.add_argument("--no-think-suffix", help="text appended after the generation prompt, e.g. an empty think block")
    s.add_argument("--fixed-date", help="YYYY-MM-DD; shadows the chat template's strftime_now")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stepscope")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="scan a verl output directory and write <out>/<run_id>/run.json")
    s.add_argument("path"); s.add_argument("--out", required=True); s.add_argument("--run-id"); s.add_argument("--framework", default="verl")
    sub.add_parser("capture", help="attach capture hooks to a training run (optional)").add_argument("config")
    s = sub.add_parser("observe", help="run probe sets on every checkpoint"); s.add_argument("run"); s.add_argument("--probes", nargs="+", required=True)
    s = sub.add_parser("analyze", help="run a study or named analyses"); s.add_argument("run"); s.add_argument("--study"); s.add_argument("--analyses", nargs="*")
    s = sub.add_parser("report", help="re-render the static HTML report of a run directory")
    s.add_argument("run_dir"); s.add_argument("--out"); s.add_argument("--interval", help="<from>_<to>_<probe version>, or <from>_<to> when that interval has one probe version; default: latest")
    s = sub.add_parser("compare", help="compare runs on the same probes and criteria"); s.add_argument("runs", nargs="+")

    s = sub.add_parser("probes", help="probe file utilities")
    ps = s.add_subparsers(dest="probes_cmd", required=True)
    c = ps.add_parser("convert", help="ood_ld build_mcq.py rows -> probe file (source -> domain; order and answer_idx kept)")
    c.add_argument("src"); c.add_argument("--out", required=True)
    c.add_argument("--split", help="keep only rows with this split (e.g. dev)")
    c.add_argument("--rotations", choices=["all", "first"], default="all")

    s = sub.add_parser("interval", help="two HF checkpoints + probe file -> actual change, first-order prediction, residual, report")
    s.add_argument("--from", dest="ckpt_from", help="HF checkpoint directory (θ_from)")
    s.add_argument("--to", dest="ckpt_to", help="HF checkpoint directory (θ_to)")
    s.add_argument("--run", help="run.json written by `stepscope ingest`; use with --steps")
    s.add_argument("--steps", nargs=2, type=int, metavar=("FROM", "TO"), help="two global steps listed in run.json")
    s.add_argument("--probes", required=True, help="probe file (JSON lines)")
    s.add_argument("--out", required=True, help="store root; tables go to <out>/<run_id>/")
    s.add_argument("--run-id", help="default: <training run>__<from>_<to>")
    s.add_argument("--step-from", type=int); s.add_argument("--step-to", type=int)
    s.add_argument("--observables", nargs="+", default=["phi_fmt", "phi_ans"])
    s.add_argument("--method", choices=["auto", "jvp", "backward"], default="auto")
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--dtype", default="float32"); s.add_argument("--device", default="cpu")
    s.add_argument("--max-items", type=int, help="deterministic subset of the probe file (by id hash)")
    _add_render_args(s)
    s.add_argument("--report", help="report path (default <out>/<run_id>/report_<from>_<to>_<probe version>.html)")

    s = sub.add_parser("step", help="checkpoint + training batch + probe file -> one real optimizer step and the check "
                                    "chain CH1+CH2 -> exact first order -> actual change, with an update package and report")
    s.add_argument("--checkpoint", required=True, help="HF checkpoint directory (θ before the step)")
    s.add_argument("--batch", required=True, help="batch file (JSON lines: source + prompt/response or input_ids) or batch.pt")
    s.add_argument("--probes", required=True, help="probe file (JSON lines)")
    s.add_argument("--out", required=True, help="store root; tables go to <out>/<run_id>/")
    s.add_argument("--objective", default="sft", help="registered objective adapter: sft | mopd")
    s.add_argument("--objective-arg", nargs="+", default=[], metavar="KEY=VALUE",
                   help="objective configuration, e.g. weight_mode=teacher_p micro_batch_size=2")
    s.add_argument("--teacher", nargs="+", default=[], metavar="[SOURCE=]PATH",
                   help="mopd without top-K inputs in the batch file: teacher checkpoint per source (or one for all)")
    s.add_argument("--topk", type=int, default=4, help="student top-K candidates per position (mopd)")
    s.add_argument("--source-kind", default="domain", help="what a row's source is: domain | teacher | ...")
    s.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    s.add_argument("--lr", type=float, default=1e-2)
    s.add_argument("--weight-decay", type=float, default=0.0)
    s.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    s.add_argument("--eps", type=float, default=1e-8)
    s.add_argument("--max-grad-norm", type=float)
    s.add_argument("--run-id", help="default: <checkpoint name>__<objective>_<optimizer>")
    s.add_argument("--step", type=int, default=0, help="optimizer step number recorded for this update")
    s.add_argument("--observables", nargs="+", default=["phi_fmt", "phi_ans"])
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--dtype", default="float32"); s.add_argument("--device", default="cpu")
    s.add_argument("--max-items", type=int, help="deterministic subset of the probe file (by id hash)")
    s.add_argument("--no-by-source", action="store_true", help="skip the per-source split of the SGD first order")
    s.add_argument("--no-channels", action="store_true", help="skip CH1/CH2")
    s.add_argument("--no-package", action="store_true", help="do not write the update package")
    _add_render_args(s)
    s.add_argument("--report", help="report path (default <out>/<run_id>/report_step_<tag>.html)")
    return p


def _unescape(s: str | None) -> str | None:
    return None if s is None else s.replace("\\n", "\n").replace("\\t", "\t")


def _render_config(args):
    from ..observe.probes import RenderConfig

    ov = dict(instruction=args.instruction, answer_prefix=_unescape(args.answer_prefix), label_format=args.label_format,
              labels=tuple(args.labels) if args.labels else None, system=args.system,
              chat_template=True if args.chat_template else None, enable_thinking=False if args.no_thinking else None,
              no_think_suffix=_unescape(args.no_think_suffix), fixed_date=args.fixed_date)
    if args.protocol:
        return RenderConfig.from_protocol(args.protocol, **ov)
    return RenderConfig(**{k: v for k, v in ov.items() if v is not None})


def _kv(items: list[str], what: str) -> dict:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{what}: expected KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _step(args) -> int:
    from ..capture.train_step import OptimizerConfig
    from ..pipelines.step import StepArgs, run_step

    teachers = None
    if args.teacher:
        teachers = {}
        for t in args.teacher:
            src, _, path = t.rpartition("=") if "=" in t and not Path(t).exists() else ("*", "", t)
            teachers[src or "*"] = path
    opt = OptimizerConfig(name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas),
                          eps=args.eps, max_grad_norm=args.max_grad_norm)
    summary = run_step(StepArgs(
        checkpoint=args.checkpoint, batch=args.batch, probes=args.probes, out=args.out, objective=args.objective,
        objective_config=_kv(args.objective_arg, "--objective-arg"), optimizer=opt, teachers=teachers, topk=args.topk,
        source_kind=args.source_kind, run_id=args.run_id, step=args.step, observables=tuple(args.observables),
        batch_size=args.batch_size, dtype=args.dtype, device=args.device, render=_render_config(args),
        max_items=args.max_items, by_source=not args.no_by_source, channels=not args.no_channels,
        save_package=not args.no_package, report=args.report,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def _interval(args) -> int:
    from ..pipelines.interval import IntervalArgs, run_interval

    run_id = args.run_id
    if args.run:
        if not args.steps:
            raise SystemExit("--run needs --steps FROM TO")
        run = json.loads(Path(args.run).read_text(encoding="utf-8"))
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

    render = _render_config(args)
    summary = run_interval(IntervalArgs(
        ckpt_from=args.ckpt_from, ckpt_to=args.ckpt_to, probes=args.probes, out=args.out, run_id=run_id,
        step_from=args.step_from, step_to=args.step_to, observables=tuple(args.observables), method=args.method,
        batch_size=args.batch_size, dtype=args.dtype, device=args.device, render=render, report=args.report,
        max_items=args.max_items,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _ingest(args) -> int:
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
    out = args.out or str(Path(args.run_dir) / (f"report_{args.interval}.html" if args.interval else "report.html"))
    print(render(args.run_dir, out, args.interval))
    return 0


def _probes(args) -> int:
    from ..observe.convert import convert_file
    if args.probes_cmd == "convert":
        n = convert_file(args.src, args.out, split=args.split, rotations=args.rotations)
        print(f"wrote {n} probes to {args.out}")
        return 0
    raise SystemExit(f"unknown probes command {args.probes_cmd!r}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "interval":
        return _interval(args)
    if args.cmd == "step":
        return _step(args)
    if args.cmd == "report":
        return _report(args)
    if args.cmd == "ingest":
        return _ingest(args)
    if args.cmd == "probes":
        return _probes(args)
    print(f"stepscope {args.cmd}: not implemented yet (skeleton)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
