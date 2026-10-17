"""Static HTML reports, no external assets; they read the result tables plus the run's json summary only.

``render_report``: chain one (blueprint §十四.5), one interval. ``render_step_report``: the single-step check
chain (§十九), one update. Rows are filtered by (step_from, step_to, probe_set, probe_version) taken from the
summary, so several intervals or steps stored under one run_id never mix. Both use the same components."""
from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path

from ..core.store import Store

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222;max-width:1200px}
h1{font-size:20px}h2{font-size:16px;margin-top:28px}h3{font-size:14px;margin-top:16px}table{border-collapse:collapse;font-size:13px}
th,td{border:1px solid #ddd;padding:4px 8px;text-align:right}th{background:#f4f4f4}td.l,th.l{text-align:left}
.meta td{text-align:left}.note{color:#666;font-size:12px}.row{display:flex;gap:24px;flex-wrap:wrap}
svg{background:#fff;border:1px solid #ddd}
"""


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def scatter_svg(xs, ys, labels, title: str, note: str, size: int = 360, pad: int = 40,
                x_label: str = "actual ΔΦ", y_label: str = "first-order ∇Φ·Δθ") -> str:
    pts = [(x, y, l) for x, y, l in zip(xs, ys, labels) if not (math.isnan(x) or math.isnan(y))]
    if not pts:
        return f"<svg width='{size}' height='{size}'><text x='10' y='20'>{html.escape(title)}: no data</text></svg>"
    vals = [v for x, y, _ in pts for v in (x, y)]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        lo, hi = lo - 1e-6, hi + 1e-6
    span = hi - lo
    lo, hi = lo - 0.05 * span, hi + 0.05 * span

    def sx(v):
        return pad + (v - lo) / (hi - lo) * (size - 2 * pad)

    def sy(v):
        return size - pad - (v - lo) / (hi - lo) * (size - 2 * pad)

    out = [f"<svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>",
           f"<text x='{pad}' y='18' font-size='13' font-weight='bold'>{html.escape(title)}</text>",
           f"<text x='{pad}' y='32' font-size='11' fill='#555'>{html.escape(note)}</text>",
           f"<line x1='{sx(lo)}' y1='{sy(lo)}' x2='{sx(hi)}' y2='{sy(hi)}' stroke='#999' stroke-dasharray='4 3'/>",
           f"<line x1='{pad}' y1='{size-pad}' x2='{size-pad}' y2='{size-pad}' stroke='#333'/>",
           f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{size-pad}' stroke='#333'/>",
           f"<text x='{size/2}' y='{size-8}' font-size='11' text-anchor='middle'>{html.escape(x_label)}</text>",
           f"<text x='12' y='{size/2}' font-size='11' transform='rotate(-90 12,{size/2})' text-anchor='middle'>{html.escape(y_label)}</text>",
           f"<text x='{pad}' y='{size-pad+14}' font-size='10'>{lo:.3g}</text>",
           f"<text x='{size-pad}' y='{size-pad+14}' font-size='10' text-anchor='end'>{hi:.3g}</text>"]
    for x, y, l in pts:
        out.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3' fill='#1f77b4' fill-opacity='0.7'>"
                   f"<title>{html.escape(l)}: {html.escape(x_label)}={x:.4g}, {html.escape(y_label)}={y:.4g}</title></circle>")
    out.append("</svg>")
    return "\n".join(out)


def _select(rows, **keys):
    return [r for r in rows if all(r.get(k) == v for k, v in keys.items() if v is not None)]


def render_report(store: Store, run_id: str, meta: dict, out_path: str | Path) -> Path:
    step_from, step_to = meta.get("step_from"), meta.get("step_to")
    probe_set, probe_version = meta.get("probe_set"), meta.get("probe_version")
    sel = dict(step_from=step_from, step_to=step_to, probe_set=probe_set, probe_version=probe_version)
    effects = _select(store.read(run_id, "effects").to_pylist(), **sel)
    cons = _select(store.read(run_id, "consistency").to_pylist(), **sel)
    behavior = [r for r in _select(store.read(run_id, "behavior").to_pylist(), probe_set=probe_set, probe_version=probe_version)
                if r["step"] in (step_from, step_to)]
    observables = sorted({r["observable"] for r in effects})

    # --- per-domain table
    acc = defaultdict(lambda: defaultdict(list))   # domain -> step -> scores
    for r in behavior:
        if r["metric"] == "argmax_in_set_correct":
            acc[r["domain"]][r["step"]].append(r["score"])
    eff = defaultdict(lambda: defaultdict(list))   # domain -> observable -> (actual, pred)
    for r in effects:
        eff[r["domain"]][r["observable"]].append((r["actual"], r["first_order"]))
    domains = sorted(set(acc) | set(eff))
    head = "<tr><th class='l'>domain</th><th>n</th><th>acc_from</th><th>acc_to</th><th>Δacc</th>"
    head += "".join(f"<th>mean actual {html.escape(o)}</th><th>mean first-order {html.escape(o)}</th>" for o in observables) + "</tr>"
    rows_html = []
    for d in domains:
        a0 = acc[d].get(step_from, []); a1 = acc[d].get(step_to, [])
        n = len(a0) or len(next(iter(eff[d].values()), []))
        m0 = sum(a0) / len(a0) if a0 else float("nan"); m1 = sum(a1) / len(a1) if a1 else float("nan")
        cells = f"<td class='l'>{html.escape(d)}</td><td>{n}</td><td>{_fmt(m0,3)}</td><td>{_fmt(m1,3)}</td><td>{_fmt(m1-m0,3)}</td>"
        for o in observables:
            pairs = eff[d].get(o, [])
            ma = sum(p[0] for p in pairs) / len(pairs) if pairs else float("nan")
            fin = [p[1] for p in pairs if not math.isnan(p[1])]
            mp = sum(fin) / len(fin) if fin else float("nan")
            cells += f"<td>{_fmt(ma)}</td><td>{_fmt(mp)}</td>"
        rows_html.append(f"<tr>{cells}</tr>")
    domain_table = f"<table>{head}{''.join(rows_html)}</table>"

    # --- calibration scatters
    cons_by_obs = {r["observable"]: r for r in cons}
    plots = []
    for o in observables:
        rs = [r for r in effects if r["observable"] == o]
        c = cons_by_obs.get(o, {})
        ratio = float("nan")
        if c.get("resid_rms") is not None and c.get("actual_rms"):
            ratio = c["resid_rms"] / c["actual_rms"]
        note = (f"n={c.get('n','–')}  corr={_fmt(c.get('corr'),3)}  sign agree={_fmt(c.get('sign_agree'),3)}  "
                f"resid RMS/actual RMS={_fmt(ratio,3)}")
        plots.append(scatter_svg([r["actual"] for r in rs], [r["first_order"] for r in rs], [r["item"] for r in rs],
                                 f"{o}: prediction calibration", note))
    plots_html = "<div class='row'>" + "".join(plots) + "</div>"

    # --- per-item table
    top = sorted(effects, key=lambda r: -abs(r["actual"]))[:100]
    item_rows = "".join(
        f"<tr><td class='l'>{html.escape(r['item'])}</td><td class='l'>{html.escape(r['domain'])}</td><td class='l'>{html.escape(r['observable'])}</td>"
        f"<td>{_fmt(r['value_from'])}</td><td>{_fmt(r['value_to'])}</td><td>{_fmt(r['actual'])}</td><td>{_fmt(r['first_order'])}</td>"
        f"<td>{_fmt(r['actual']-r['first_order']) if not math.isnan(r['first_order']) else '–'}</td></tr>" for r in top)
    item_table = ("<table><tr><th class='l'>item</th><th class='l'>domain</th><th class='l'>observable</th><th>Φ_from</th><th>Φ_to</th>"
                  "<th>actual</th><th>first-order</th><th>residual</th></tr>" + item_rows + "</table>")

    meta_rows = "".join(f"<tr><td class='l'><b>{html.escape(str(k))}</b></td><td>{html.escape(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v))}</td></tr>"
                        for k, v in meta.items())
    title = f"interval {step_from} → {step_to} · {probe_set} ({probe_version})"
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>StepScope · {html.escape(run_id)} · {html.escape(title)}</title><style>{CSS}</style></head>
<body><h1>StepScope · {html.escape(run_id)} · {html.escape(title)}</h1>
<table class='meta'>{meta_rows}</table>
<h2>1. Capability change by domain</h2>{domain_table}
<p class='note'>acc = mean of argmax-in-set correctness over the probe items of the domain at each checkpoint.</p>
<h2>2. Prediction calibration (first figure)</h2>{plots_html}
<p class='note'>Dashed line is y = x. Between checkpoints Δθ is the net effect of many optimizer steps; the first-order
term is a first-order approximation of that net update, used for trajectories and a coarse consistency check only.
No per-source attribution is implied. Low correlation or large residuals on real intervals are a result, not a failure.</p>
<h2>3. Items sorted by |actual| (top {len(top)} of {len(effects)} rows of this interval; full data in effects/*.parquet)</h2>{item_table}
</body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def find_interval_meta(run_dir: Path, interval: str | None) -> Path:
    """``interval`` is a full tag "<from>_<to>_<probe version>" or "<from>_<to>" when only one probe version
    exists for that interval; None picks interval.json (the latest run)."""
    if not interval:
        return run_dir / "interval.json"
    exact = run_dir / f"interval_{interval}.json"
    if exact.exists():
        return exact
    found = sorted(run_dir.glob(f"interval_{interval}_*.json"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(exact)
    tags = ", ".join(p.stem[len("interval_"):] for p in found)
    raise ValueError(f"interval {interval} has several probe versions in {run_dir}; pass one of: {tags}")


def render(run_dir: str, out_html: str, interval: str | None = None) -> Path:
    """Re-render from a run directory (see ``find_interval_meta`` for ``interval``)."""
    run_dir = Path(run_dir)
    meta_path = find_interval_meta(run_dir, interval)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return render_report(Store(run_dir.parent), run_dir.name, meta.get("summary", meta), out_html)


def _mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _matrix_table(rows_by, cols, title_col: str) -> str:
    head = f"<tr><th class='l'>{html.escape(title_col)}</th>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>"
    body = "".join(f"<tr><td class='l'>{html.escape(r)}</td>" + "".join(f"<td>{_fmt(_mean(rows_by[r].get(c, [])), 5)}</td>" for c in cols)
                   + "</tr>" for r in sorted(rows_by))
    return f"<table>{head}{body}</table>"


def render_step_report(store: Store, run_id: str, meta: dict, out_path: str | Path) -> Path:
    """One update: check-chain table, calibration scatters (first order vs actual; CH1 + CH2 vs first order),
    source × domain tables for the per-source first-order parts and the channel prediction, items."""
    sel = dict(step_from=meta["step_from"], step_to=meta["step_to"], probe_set=meta["probe_set"],
               probe_version=meta["probe_version"])
    effects = _select(store.read(run_id, "effects").to_pylist(), **sel)
    cons = _select(store.read(run_id, "consistency").to_pylist(), **sel)
    chans = _select(store.read(run_id, "channels").to_pylist(), **sel)
    contribs = _select(store.read(run_id, "contributions").to_pylist(), **sel)
    observables = sorted({r["observable"] for r in effects})
    domain_of = {r["item"]: r["domain"] for r in effects}
    ch_tot = defaultdict(float)
    for r in chans:
        ch_tot[(r["item"], r["observable"])] += r["ch1"] + r["ch2"]

    # --- 1. check chain
    chain_rows = []
    for r in sorted(cons, key=lambda r: (r["observable"], r["prediction"], r["reference"])):
        ratio = r["resid_rms"] / r["actual_rms"] if r.get("actual_rms") else float("nan")
        label = r["prediction"] + (f" [{r['decomp_id']}]" if r.get("decomp_id") else "")
        chain_rows.append(f"<tr><td class='l'>{html.escape(r['observable'])}</td><td class='l'>{html.escape(label)}</td>"
                          f"<td class='l'>{html.escape(r['reference'])}</td><td>{r['n']}</td><td>{_fmt(r['corr'], 4)}</td>"
                          f"<td>{_fmt(r['sign_agree'], 3)}</td><td>{_fmt(ratio, 3)}</td><td>{_fmt(r['resid_target_minus_sum'], 3)}</td></tr>")
    chain_table = ("<table><tr><th class='l'>observable</th><th class='l'>prediction</th><th class='l'>reference</th><th>n</th>"
                   "<th>corr</th><th>sign agree</th><th>resid RMS / ref RMS</th><th>target − Σ parts (RMS)</th></tr>"
                   + "".join(chain_rows) + "</table>")
    diag = meta.get("diagnostics", {})
    diag_html = "".join(f"<tr><td class='l'><b>{html.escape(k)}</b></td><td class='l'>{html.escape(json.dumps(v, ensure_ascii=False, default=str))}</td></tr>"
                        for k, v in diag.items())
    skipped = meta.get("skipped", {})
    skip_html = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape('; '.join(v))}</li>" for k, v in skipped.items())

    # --- 2. scatters
    cons_by = {(r["observable"], r["prediction"], r["reference"]): r for r in cons}
    plots = []
    for o in observables:
        rs = [r for r in effects if r["observable"] == o]
        c = cons_by.get((o, "first_order", "actual"), {})
        plots.append(scatter_svg([r["actual"] for r in rs], [r["first_order"] for r in rs], [r["item"] for r in rs],
                                 f"{o}: first order vs actual", f"corr={_fmt(c.get('corr'), 4)}  sign={_fmt(c.get('sign_agree'), 3)}"))
        if chans:
            c = cons_by.get((o, "ch1_ch2", "first_order"), {})
            plots.append(scatter_svg([r["first_order"] for r in rs], [ch_tot[(r["item"], o)] for r in rs], [r["item"] for r in rs],
                                     f"{o}: CH1+CH2 vs first order", f"corr={_fmt(c.get('corr'), 4)}  sign={_fmt(c.get('sign_agree'), 3)}",
                                     x_label="first-order ∇Φ·Δθ_u", y_label="CH1 + CH2"))
    plots_html = "<div class='row'>" + "".join(plots) + "</div>"

    # --- 3. source × domain
    domains = sorted(set(domain_of.values()))
    blocks = []
    for o in observables:
        if contribs:
            by = defaultdict(lambda: defaultdict(list))
            for r in contribs:
                if r["observable"] == o:
                    by[r["source"]][domain_of[r["item"]]].append(r["contrib"])
            blocks.append(f"<h3>{html.escape(o)}: first-order part of each source (mean over items)</h3>" + _matrix_table(by, domains, "source"))
        if chans:
            by = defaultdict(lambda: defaultdict(list))
            per = defaultdict(float)
            for r in chans:
                if r["observable"] == o:
                    per[(r["source"], r["item"])] += r["ch1"] + r["ch2"]
            for (src, item), v in per.items():
                by[src][domain_of[item]].append(v)
            blocks.append(f"<h3>{html.escape(o)}: CH1 + CH2 of each source (mean over items)</h3>" + _matrix_table(by, domains, "source"))
    source_html = "".join(blocks) or "<p class='note'>No per-source tables for this update (see skipped analyses).</p>"

    # --- 4. items
    top = sorted(effects, key=lambda r: -abs(r["actual"]))[:100]
    item_rows = "".join(
        f"<tr><td class='l'>{html.escape(r['item'])}</td><td class='l'>{html.escape(r['domain'])}</td><td class='l'>{html.escape(r['observable'])}</td>"
        f"<td>{_fmt(r['value_from'])}</td><td>{_fmt(r['actual'], 6)}</td><td>{_fmt(r['first_order'], 6)}</td>"
        f"<td>{_fmt(ch_tot.get((r['item'], r['observable'])), 6) if chans else '–'}</td></tr>" for r in top)
    item_table = ("<table><tr><th class='l'>item</th><th class='l'>domain</th><th class='l'>observable</th><th>Φ before</th>"
                  "<th>actual</th><th>first order</th><th>CH1 + CH2</th></tr>" + item_rows + "</table>")

    keep = ("run_id", "model_type", "capabilities", "objective", "objective_config", "optimizer", "loss", "loss_by_source",
            "grad_norm", "clip_coef", "sources", "n_rows", "n_supervised_tokens", "probe_set", "probe_version", "n_items",
            "package", "code_hash", "cost_s_total", "peak_mem_gb")
    meta_rows = "".join(f"<tr><td class='l'><b>{html.escape(k)}</b></td><td>{html.escape(json.dumps(meta[k], ensure_ascii=False, default=str) if isinstance(meta[k], (dict, list)) else str(meta[k]))}</td></tr>"
                        for k in keep if k in meta)
    title = f"step {meta['step_from']} → {meta['step_to']} · {meta['probe_set']} ({meta['probe_version']})"
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>StepScope · {html.escape(run_id)} · {html.escape(title)}</title><style>{CSS}</style></head>
<body><h1>StepScope · {html.escape(run_id)} · {html.escape(title)}</h1>
<table class='meta'>{meta_rows}</table>
<h2>1. Check chain: CH1 + CH2 → exact first order → actual</h2>{chain_table}
<p class='note'>All three refer to the same batch and the same update. First order: JVP at θ along the batch's own update
(SGD: rebuilt from the batch and validated against the measured Δθ; AdamW: the measured Δθ). CH1 + CH2: LDCL channels,
SGD direction only; κ (embedding token overlap) not computed. Per-source parts of the SGD first order sum to it exactly.</p>
<table class='meta'>{diag_html}</table>
{('<p><b>Skipped</b></p><ul>' + skip_html + '</ul>') if skip_html else ''}
<h2>2. Calibration</h2>{plots_html}
<h2>3. Sources × domains</h2>{source_html}
<h2>4. Items sorted by |actual| (top {len(top)} of {len(effects)} rows; full data in the parquet tables)</h2>{item_table}
<p class='note'>One step from θ: single-step, first-order statements about this update only. Channel values are
η-scaled predictions in the units of the observable; the readout channel is exact for an untied readout without bias.</p>
</body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
