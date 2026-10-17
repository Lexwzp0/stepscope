"""Chain one end to end: two HF checkpoints + a probe file → effects / behavior / consistency tables,
interval.json and report.html (blueprint §十四).

Run identity: the default run_id names the training run and both global steps (derived from
``.../<run>/global_step_<N>/actor/huggingface``), so verl exports never collide on ``huggingface``.
Several intervals and probe sets may share one run_id: every table part, ``interval_<tag>.json`` and
``report_<tag>.html`` are named by ``shard_tag`` = ``<from>_<to>_<probe version>``, and the report only shows
its own interval and probe version.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..adapters.models.hf import HFModelAdapter
from ..analyze.consistency.consistency import VERSION as CONS_VERSION, consistency_rows
from ..analyze.dynamics import interval as iv
from ..core.provenance import code_hash, provenance
from ..core.store import Store
from ..observe.probes import RenderConfig, load_probe_file, select_items
from ..observe.scoring import encode_probes
from ..report.html import render_report

VERSION = "0.1.2"
_STEP_DIR = re.compile(r"^global_step_(\d+)$")
_STEP = re.compile(r"(?:global_step|step)[_\-]?(\d+)", re.I)


@dataclass
class IntervalArgs:
    ckpt_from: str
    ckpt_to: str
    probes: str
    out: str
    run_id: str | None = None
    step_from: int | None = None
    step_to: int | None = None
    observables: tuple[str, ...] = ("phi_fmt", "phi_ans")
    method: str = "auto"
    batch_size: int = 4
    dtype: str = "float32"
    device: str = "cpu"
    render: RenderConfig = field(default_factory=RenderConfig)
    report: str | None = None
    max_items: int | None = None


def shard_tag(step_from: int, step_to: int, probe_version: str) -> str:
    """Name of every per-interval artefact: table parts, interval_<tag>.json, report_<tag>.html."""
    return f"{step_from}_{step_to}_{probe_version}"


def describe_checkpoint(path: str, default_step: int | None = None) -> tuple[str, int | None]:
    """(training run name, global step) from a checkpoint path.

    ``.../<run>/global_step_<N>/actor/huggingface`` → (<run>, N); a directory named ``global_step_<N>``
    → (its parent's name, N); otherwise (directory name, step parsed from the name or default)."""
    p = Path(path)
    for q in (p, *p.parents):
        m = _STEP_DIR.match(q.name)
        if m:
            return q.parent.name or "run", int(m.group(1))
    name = p.parent.name if p.name == "huggingface" and p.parent.name else p.name
    m = _STEP.search(name)
    return name or "run", (int(m.group(1)) if m else default_step)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "-", s).strip("-") or "run"


def default_run_id(ckpt_from: str, ckpt_to: str, step_from: int, step_to: int) -> str:
    a, _ = describe_checkpoint(ckpt_from)
    b, _ = describe_checkpoint(ckpt_to)
    if a == b:
        return f"{_slug(a)}__{step_from}_{step_to}"
    return f"{_slug(a)}_{step_from}__{_slug(b)}_{step_to}"


def _dir_hash(path: str) -> str:
    p = Path(path)
    h = hashlib.sha1()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            st = f.stat()
            h.update(f"{f.relative_to(p)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def _peak_memory_gb(device: str) -> float:
    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
        import psutil  # optional

        mi = psutil.Process().memory_info()
        return (getattr(mi, "peak_wset", None) or mi.rss) / 1e9
    except Exception:  # noqa: BLE001
        return float("nan")


def run_interval(a: IntervalArgs) -> dict:
    t0 = time.perf_counter()
    if a.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _, sf = describe_checkpoint(a.ckpt_from, 0)
    _, st = describe_checkpoint(a.ckpt_to, None)
    step_from = a.step_from if a.step_from is not None else (sf if sf is not None else 0)
    step_to = a.step_to if a.step_to is not None else (st if st is not None else step_from + 1)
    run_id = a.run_id or default_run_id(a.ckpt_from, a.ckpt_to, step_from, step_to)
    names = tuple(a.observables)
    store = Store(a.out)
    run_dir = store.run_dir(run_id)

    probes = select_items(load_probe_file(a.probes), a.max_items)
    tag = shard_tag(step_from, step_to, probes.version)
    ad_from = HFModelAdapter.load(a.ckpt_from, a.dtype, a.device)
    enc = encode_probes(ad_from.tokenizer, probes.items, a.render)

    with iv.Timer() as t_obs_from:
        s_from = iv.observe(ad_from.model, enc, names, a.batch_size)

    ad_to = HFModelAdapter.load(a.ckpt_to, a.dtype, a.device)
    enc_to = encode_probes(ad_to.tokenizer, probes.items, a.render)
    if enc_to.id_lists != enc.id_lists or not torch.equal(enc_to.option_ids, enc.option_ids):
        raise ValueError("the two checkpoints tokenize the probes differently; both ends must share one encoding")
    with iv.Timer() as t_obs_to:
        s_to = iv.observe(ad_to.model, enc, names, a.batch_size)
    delta, dinfo = iv.delta_params(ad_from.model, ad_to.model, ad_from.param_groups(), release=True)
    ad_to.free(); del ad_to; gc.collect()

    with iv.Timer() as t_fo:
        phi_from2, dphi, method_used = iv.first_order(ad_from.model, delta, enc, names, a.method, a.batch_size)
    phi_gap = float((phi_from2 - s_from["phi"]).abs().max())

    cost = time.perf_counter() - t0
    peak = _peak_memory_gb(a.device)
    mem = 0.0 if math.isnan(peak) else peak
    prov_eff = provenance(run_id, "dynamics.interval", VERSION, "first_order", cost_s=t_fo.seconds, cost_mem_gb=mem)
    prov_beh = provenance(run_id, "observe.score", VERSION, "exact", cost_s=t_obs_from.seconds + t_obs_to.seconds, cost_mem_gb=mem)
    prov_con = provenance(run_id, "consistency.consistency", CONS_VERSION, "exact")

    eff_rows = iv.effects_rows(enc, names, s_from["phi"], s_to["phi"], dphi, step_from=step_from, step_to=step_to,
                               probe_set=probes.name, probe_version=probes.version, method=method_used, prov=prov_eff)
    store.write(run_id, "effects", eff_rows, part=tag)
    for step, scored in ((step_from, s_from), (step_to, s_to)):
        store.write(run_id, "behavior", iv.behavior_rows(enc, scored, step=step, probe_set=probes.name,
                                                          probe_version=probes.version, prov=prov_beh),
                    part=f"{step}_{probes.version}")
    con_rows = consistency_rows(eff_rows, prov_con)
    store.write(run_id, "consistency", con_rows, part=tag)

    summary = {
        "run_id": run_id, "step_from": step_from, "step_to": step_to, "tag": tag,
        "ckpt_from": a.ckpt_from, "ckpt_to": a.ckpt_to,
        "probe_set": probes.name, "probe_version": probes.version, "n_items": len(enc),
        "observables": list(names), "method": method_used, "batch_size": a.batch_size, "dtype": a.dtype,
        "device": a.device, "delta_norm": dinfo.norm, "n_params": dinfo.n_params,
        "phi_from_recompute_max_gap": phi_gap, "code_hash": code_hash(),
        "cost_s_total": round(cost, 3), "cost_s_observe": round(t_obs_from.seconds + t_obs_to.seconds, 3),
        "cost_s_first_order": round(t_fo.seconds, 3), "peak_mem_gb": None if math.isnan(peak) else round(peak, 3),
        "consistency": {r["observable"]: {k: r[k] for k in ("n", "corr", "sign_agree", "resid_rms", "actual_rms")} for r in con_rows},
    }
    meta = {
        "summary": summary,
        "inputs": {"ckpt_from_hash": _dir_hash(a.ckpt_from), "ckpt_to_hash": _dir_hash(a.ckpt_to),
                   "probe_file": str(a.probes), "probe_version": probes.version, "max_items": a.max_items},
        "render": asdict(a.render), "labels": list(enc.labels),
        "option_ids_unique": sorted({int(x) for x in enc.option_ids.flatten().tolist()}),
        "delta_group_norms": dinfo.group_norms,
    }
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
    (run_dir / f"interval_{tag}.json").write_text(meta_json, encoding="utf-8")
    (run_dir / "interval.json").write_text(meta_json, encoding="utf-8")
    report_path = Path(a.report) if a.report else run_dir / f"report_{tag}.html"
    render_report(store, run_id, summary, report_path)
    if not a.report:
        shutil.copyfile(report_path, run_dir / "report.html")
    summary["report"] = str(report_path)
    summary["run_dir"] = str(run_dir)
    return summary
