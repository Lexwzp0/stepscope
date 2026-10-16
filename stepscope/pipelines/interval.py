"""Chain one end to end: two HF checkpoints + a probe file → effects / behavior / consistency tables,
interval.json and report.html (blueprint §十四)."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..adapters.models.hf import HFModelAdapter
from ..analyze.consistency.consistency import VERSION as CONS_VERSION, consistency_rows
from ..analyze.dynamics import interval as iv
from ..core.provenance import code_hash, provenance
from ..core.store import Store
from ..observe.probes import RenderConfig, load_probe_file
from ..observe.scoring import encode_probes
from ..report.html import render_report

VERSION = "0.1.0"
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


def parse_step(path: str, default: int) -> int:
    m = _STEP.search(Path(path).name) or _STEP.search(str(path))
    return int(m.group(1)) if m else default


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "-", Path(s).name).strip("-") or "ckpt"


def _dir_hash(path: str) -> str:
    p = Path(path)
    h = hashlib.sha1()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            st = f.stat()
            h.update(f"{f.relative_to(p)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def run_interval(a: IntervalArgs) -> dict:
    t0 = time.perf_counter()
    step_from = a.step_from if a.step_from is not None else parse_step(a.ckpt_from, 0)
    step_to = a.step_to if a.step_to is not None else parse_step(a.ckpt_to, step_from + 1)
    run_id = a.run_id or f"{_slug(a.ckpt_from)}__{_slug(a.ckpt_to)}"
    names = tuple(a.observables)
    store = Store(a.out)
    run_dir = store.run_dir(run_id)

    probes = load_probe_file(a.probes)
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
    delta, dinfo = iv.delta_params(ad_from.model, ad_to.model, ad_from.param_groups())
    ad_to.free(); del ad_to; gc.collect()

    with iv.Timer() as t_fo:
        phi_from2, dphi, method_used = iv.first_order(ad_from.model, delta, enc, names, a.method, a.batch_size)
    phi_gap = float((phi_from2 - s_from["phi"]).abs().max())

    cost = time.perf_counter() - t0
    prov_eff = provenance(run_id, "dynamics.interval", VERSION, "first_order", cost_s=t_fo.seconds)
    prov_beh = provenance(run_id, "observe.score", VERSION, "exact", cost_s=t_obs_from.seconds + t_obs_to.seconds)
    prov_con = provenance(run_id, "consistency.consistency", CONS_VERSION, "exact")

    eff_rows = iv.effects_rows(enc, names, s_from["phi"], s_to["phi"], dphi, step_from=step_from, step_to=step_to,
                               probe_set=probes.name, probe_version=probes.version, method=method_used, prov=prov_eff)
    store.write(run_id, "effects", eff_rows, part=f"{step_from}_{step_to}")
    for step, scored in ((step_from, s_from), (step_to, s_to)):
        store.write(run_id, "behavior", iv.behavior_rows(enc, scored, step=step, probe_set=probes.name,
                                                          probe_version=probes.version, prov=prov_beh), part=f"{step}")
    con_rows = consistency_rows(eff_rows, prov_con)
    store.write(run_id, "consistency", con_rows, part=f"{step_from}_{step_to}")

    summary = {
        "run_id": run_id, "step_from": step_from, "step_to": step_to,
        "ckpt_from": a.ckpt_from, "ckpt_to": a.ckpt_to,
        "probe_set": probes.name, "probe_version": probes.version, "n_items": len(enc),
        "observables": list(names), "method": method_used, "batch_size": a.batch_size, "dtype": a.dtype,
        "delta_norm": dinfo.norm, "n_params": dinfo.n_params,
        "phi_from_recompute_max_gap": phi_gap, "code_hash": code_hash(),
        "cost_s_total": round(cost, 3), "cost_s_first_order": round(t_fo.seconds, 3),
        "consistency": {r["observable"]: {k: r[k] for k in ("n", "corr", "sign_agree", "resid_rms", "actual_rms")} for r in con_rows},
    }
    meta = {
        "summary": summary,
        "inputs": {"ckpt_from_hash": _dir_hash(a.ckpt_from), "ckpt_to_hash": _dir_hash(a.ckpt_to),
                   "probe_file": str(a.probes), "probe_version": probes.version},
        "render": asdict(a.render), "labels": list(enc.labels),
        "option_ids_unique": sorted({int(x) for x in enc.option_ids.flatten().tolist()}),
        "delta_group_norms": dinfo.group_norms,
    }
    (run_dir / "interval.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = Path(a.report) if a.report else run_dir / "report.html"
    render_report(store, run_id, summary, report_path)
    summary["report"] = str(report_path)
    summary["run_dir"] = str(run_dir)
    return summary
