"""Single-step chain end to end (blueprint §十九 校验链 1–2, §二十 step two): a checkpoint, a training batch and
a probe file → one real optimizer step (update package) → effects / channels / contributions / behavior /
consistency tables, ``step_<tag>.json`` and ``report_step_<tag>.html``.

Everything backbone- or loss-specific goes through the model adapter (declared capabilities) and the objective
adapter (loss, loss_by_source, force); this module and the analyses never branch on either.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..adapters.models.hf import HFModelAdapter
from ..adapters.objectives import mopd as _mopd  # noqa: F401 - registers "mopd"
from ..adapters.objectives import sft as _sft  # noqa: F401 - registers "sft"
from ..analyze.consistency.consistency import VERSION as CONS_VERSION
from ..analyze.consistency.consistency import channel_consistency_rows, consistency_rows, decomposition_rows
from ..analyze.dynamics import interval as iv
from ..analyze.dynamics import ldcl_channels as ch
from ..analyze.dynamics import single_step as ss
from ..capture.batch import load_batch, load_batch_file, mopd_extras
from ..capture.package import PackageWriter
from ..capture.train_step import OptimizerConfig, take_step
from ..core.provenance import code_hash, provenance
from ..core.registry import OBJECTIVES
from ..core.store import Store
from ..observe.probes import RenderConfig, load_probe_file, select_items
from ..observe.scoring import encode_probes
from ..report.html import render_step_report
from .interval import _peak_memory_gb, _slug, describe_checkpoint, shard_tag

VERSION = "0.1.0"
DECOMP_ID = "sgd_by_source"


@dataclass
class StepArgs:
    checkpoint: str
    batch: str
    probes: str
    out: str
    objective: str = "sft"
    objective_config: dict = field(default_factory=dict)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    teachers: dict[str, str] | None = None      # source -> checkpoint ("*" = every source); MOPD without top-K inputs
    topk: int = 4
    source_kind: str = "domain"
    run_id: str | None = None
    step: int = 0
    observables: tuple[str, ...] = ("phi_fmt", "phi_ans")
    batch_size: int = 4
    dtype: str = "float32"
    device: str = "cpu"
    render: RenderConfig = field(default_factory=RenderConfig)
    max_items: int | None = None
    by_source: bool = True
    channels: bool = True
    save_package: bool = True
    report: str | None = None


def default_run_id(checkpoint: str, objective: str, optimizer: str) -> str:
    name, _ = describe_checkpoint(checkpoint)
    return f"{_slug(name)}__{objective}_{optimizer}"


def run_step(a: StepArgs) -> dict:
    t0 = time.perf_counter()
    if a.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    adapter = HFModelAdapter.load(a.checkpoint, a.dtype, a.device)
    path = Path(a.batch)
    batch = load_batch(path) if path.suffix == ".pt" else load_batch_file(path, adapter.tokenizer, a.source_kind)
    objective = OBJECTIVES.get(a.objective)(**a.objective_config)
    teachers_used = None
    if a.objective == "mopd" and "topk_ids" not in batch.extras:
        if not a.teachers:
            raise SystemExit("mopd needs topk_ids/teacher_logp in the batch file or --teacher checkpoints")
        loaded = {p: HFModelAdapter.load(p, a.dtype, a.device).model for p in sorted(set(a.teachers.values()))}
        by_src = ({s: loaded[a.teachers["*"]] for s in batch.source_names()} if "*" in a.teachers
                  else {s: loaded[p] for s, p in a.teachers.items()})
        batch = mopd_extras(adapter.model, by_src, batch.to(adapter.device), a.topk, objective.temperature).to("cpu")
        teachers_used = dict(a.teachers)
        del loaded, by_src
    run_id = a.run_id or default_run_id(a.checkpoint, a.objective, a.optimizer.name)
    meta = {"checkpoint": a.checkpoint, "batch_file": str(a.batch), "teachers": teachers_used, "topk": a.topk,
            "dtype": a.dtype, "device": a.device}
    summary = run_step_on(adapter, objective, batch, a.optimizer, probes_path=a.probes, out=a.out, run_id=run_id,
                          step=a.step, names=a.observables, batch_size=a.batch_size, render=a.render,
                          max_items=a.max_items, by_source=a.by_source, channels=a.channels,
                          save_package=a.save_package, report=a.report, pre_params_ref=a.checkpoint, meta=meta, t0=t0)
    return summary


def run_step_on(adapter, objective, batch, opt: OptimizerConfig, *, probes_path: str, out: str, run_id: str,
                step: int = 0, names=("phi_fmt", "phi_ans"), batch_size: int = 4, render: RenderConfig | None = None,
                max_items: int | None = None, by_source: bool = True, channels: bool = True, save_package: bool = True,
                report: str | None = None, pre_params_ref: str | None = None, meta: dict | None = None,
                t0: float | None = None) -> dict:
    """The chain on an already loaded adapter (tests and notebooks use this directly)."""
    t0 = t0 or time.perf_counter()
    names = tuple(names)
    render = render or RenderConfig()
    store = Store(out)
    run_dir = store.run_dir(run_id)
    probes = select_items(load_probe_file(probes_path), max_items)
    enc = encode_probes(adapter.tokenizer, probes.items, render)
    step_from, step_to = step, step + 1
    tag = shard_tag(step_from, step_to, probes.version)

    with iv.Timer() as t_step:
        pkg = take_step(adapter.model, objective, batch, opt, run_id=run_id, step=step, pre_params_ref=pre_params_ref)
    pkg.meta = {**(meta or {}), "model_type": getattr(adapter, "model_type", "")}
    pkg_dir = None
    if save_package:
        pkg_dir = run_dir / f"package_{step}"
        PackageWriter(pkg_dir).write(pkg)

    res = ss.run_chain(adapter, objective, pkg, enc, names, batch_size=batch_size, by_source=by_source, channels=channels)
    peak = _peak_memory_gb(str(adapter.device))
    mem = 0.0 if math.isnan(peak) else peak
    keys = dict(step_from=step_from, step_to=step_to, probe_set=probes.name, probe_version=probes.version)

    prov_eff = provenance(run_id, "dynamics.single_step", ss.VERSION, "first_order", res.costs.get("first_order", 0.0), mem)
    eff_rows = iv.effects_rows(enc, names, res.phi_from, res.phi_to, res.first_order, prov=prov_eff, method="jvp", **keys)
    store.write(run_id, "effects", eff_rows, part=tag)
    store.record_analysis(run_id, "dynamics.single_step", ss.VERSION, "ran", context=tag)
    prov_beh = provenance(run_id, "observe.score", VERSION, "exact", res.costs.get("observe", 0.0), mem)
    for s, scored in ((step_from, res.scored_from), (step_to, res.scored_to)):
        store.write(run_id, "behavior", iv.behavior_rows(enc, scored, step=s, probe_set=probes.name,
                                                          probe_version=probes.version, prov=prov_beh),
                    part=f"{s}_{probes.version}")

    ch_rows: list[dict] = []
    if res.channel_preds is not None:
        prov_ch = provenance(run_id, ch.LdclChannels.name, ch.VERSION, "channel", res.costs.get("channels", 0.0), mem)
        ch_rows = ch.channel_rows(enc, names, res.channel_preds, source_kind=batch.source_kind, applicability="sgd",
                                  prov=prov_ch, **keys)
        store.write(run_id, "channels", ch_rows, part=tag)
    co_rows: list[dict] = []
    if res.contributions:
        prov_co = provenance(run_id, "dynamics.sgd_source_decomposition", ss.VERSION, "first_order",
                             res.costs.get("gradients_and_sources", 0.0), mem)
        co_rows = ss.contribution_rows(enc, names, res.contributions, source_kind=batch.source_kind, prov=prov_co,
                                       decomp_id=DECOMP_ID, **keys)
        store.write(run_id, "contributions", co_rows, part=tag)
    for an, why in res.skipped.items():
        store.record_analysis(run_id, an, ss.VERSION, "skipped", reasons=why, context=tag)
    for an, rows in ((ch.LdclChannels.name, ch_rows), ("dynamics.sgd_source_decomposition", co_rows)):
        if rows:
            store.record_analysis(run_id, an, ss.VERSION, "ran", context=tag)

    prov_con = provenance(run_id, "consistency.consistency", CONS_VERSION, "exact")
    con_rows = consistency_rows(eff_rows, prov_con) + channel_consistency_rows(eff_rows, ch_rows, prov_con) \
        + decomposition_rows(eff_rows, co_rows, prov_con)
    store.write(run_id, "consistency", con_rows, part=tag)

    cost = time.perf_counter() - t0
    summary = {
        "run_id": run_id, "kind": "single_step", "step_from": step_from, "step_to": step_to, "tag": tag,
        "probe_set": probes.name, "probe_version": probes.version, "n_items": len(enc), "observables": list(names),
        "model_type": getattr(adapter, "model_type", ""), "capabilities": sorted(adapter.capabilities),
        "objective": objective.name, "objective_config": objective.config(), "optimizer": asdict(opt),
        "loss": pkg.loss, "loss_by_source": pkg.loss_by_source, "grad_norm": pkg.grad_norm, "clip_coef": pkg.clip_coef,
        "source_kind": batch.source_kind, "sources": batch.source_names(), "n_rows": len(batch),
        "n_supervised_tokens": int(batch.loss_mask.sum()), "method": "jvp",
        "diagnostics": res.diagnostics, "skipped": res.skipped,
        "consistency": {f"{r['prediction']}{'[' + r['decomp_id'] + ']' if r['decomp_id'] else ''}→{r['reference']}"
                        f":{r['observable']}": {k: r[k] for k in ("n", "corr", "sign_agree", "resid_rms", "actual_rms",
                                                                  "resid_target_minus_sum")} for r in con_rows},
        "package": str(pkg_dir) if pkg_dir else None, "code_hash": code_hash(),
        "cost_s_total": round(cost, 3), "cost_s_step": round(t_step.seconds, 3),
        "cost_s": {k: round(v, 3) for k, v in res.costs.items()},
        "peak_mem_gb": None if math.isnan(peak) else round(peak, 3), **(meta or {}),
    }
    doc = json.dumps({"summary": summary}, ensure_ascii=False, indent=2, default=str)
    (run_dir / f"step_{tag}.json").write_text(doc, encoding="utf-8")
    (run_dir / "step.json").write_text(doc, encoding="utf-8")
    report_path = Path(report) if report else run_dir / f"report_step_{tag}.html"
    render_step_report(store, run_id, summary, report_path)
    if not report:
        shutil.copyfile(report_path, run_dir / "report_step.html")
    summary["report"] = str(report_path)
    summary["run_dir"] = str(run_dir)
    return summary
