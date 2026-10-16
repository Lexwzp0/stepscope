"""Static HTML report for chain one (blueprint §十四.5). Reads tables + interval.json only; no external assets."""
from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path

from ..core.store import Store

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222;max-width:1200px}
h1{font-size:20px}h2{font-size:16px;margin-top:28px}table{border-collapse:collapse;font-size:13px}
th,td{border:1px solid #ddd;padding:4px 8px;text-align:right}th{background:#f4f4f4}td.l,th.l{text-align:left}
.meta td{text-align:left}.note{color:#666;font-size:12px}.row{display:flex;gap:24px;flex-wrap:wrap}
svg{background:#fff;border:1px solid #ddd}
"""


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def scatter_svg(xs, ys, labels, title: str, note: str, size: int = 360, pad: int = 40) -> str:
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
           f"<text x='{size/2}' y='{size-8}' font-size='11' text-anchor='middle'>actual ΔΦ</text>",
           f"<text x='12' y='{size/2}' font-size='11' transform='rotate(-90 12,{size/2})' text-anchor='middle'>first-order ∇Φ·Δθ</text>",
           f"<text x='{pad}' y='{size-pad+14}' font-size='10'>{lo:.3g}</text>",
           f"<text x='{size-pad}' y='{size-pad+14}' font-size='10' text-anchor='end'>{hi:.3g}</text>"]
    for x, y, l in pts:
        out.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3' fill='#1f77b4' fill-opacity='0.7'>"
                   f"<title>{html.escape(l)}: actual={x:.4g}, first_order={y:.4g}</title></circle>")
    out.append("</svg>")
    return "\n".join(out)


def render_report(store: Store, run_id: str, meta: dict, out_path: str | Path) -> Path:
    effects = store.read(run_id, "effects").to_pylist()
    behavior = store.read(run_id, "behavior").to_pylist()
    cons = store.read(run_id, "consistency").to_pylist()
    observables = sorted({r["observable"] for r in effects})
    step_from = meta.get("step_from"); step_to = meta.get("step_to")

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
            mp = sum(p[1] for p in pairs if not math.isnan(p[1])) / max(1, sum(1 for p in pairs if not math.isnan(p[1]))) if pairs else float("nan")
            cells += f"<td>{_fmt(ma)}</td><td>{_fmt(mp)}</td>"
        rows_html.append(f"<tr>{cells}</tr>")
    domain_table = f"<table>{head}{''.join(rows_html)}</table>"

    # --- calibration scatters
    cons_by_obs = {r["observable"]: r for r in cons}
    plots = []
    for o in observables:
        rs = [r for r in effects if r["observable"] == o]
        c = cons_by_obs.get(o, {})
        note = (f"n={c.get('n','–')}  corr={_fmt(c.get('corr'),3)}  sign agree={_fmt(c.get('sign_agree'),3)}  "
                f"resid RMS/actual RMS={_fmt((c.get('resid_rms') or float('nan'))/(c.get('actual_rms') or float('nan')),3)}")
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

    meta_rows = "".join(f"<tr><td class='l'><b>{html.escape(str(k))}</b></td><td>{html.escape(str(v))}</td></tr>" for k, v in meta.items())
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>StepScope · {html.escape(run_id)}</title><style>{CSS}</style></head>
<body><h1>StepScope · interval report · {html.escape(run_id)}</h1>
<table class='meta'>{meta_rows}</table>
<h2>1. Capability change by domain</h2>{domain_table}
<p class='note'>acc = mean of argmax-in-set correctness over the probe items of the domain at each checkpoint.</p>
<h2>2. Prediction calibration (first figure)</h2>{plots_html}
<p class='note'>Dashed line is y = x. Between checkpoints Δθ is the net effect of many optimizer steps; the first-order
term is a first-order approximation of that net update, used for trajectories and a coarse consistency check only.
No per-source attribution is implied. Low correlation or large residuals on real intervals are a result, not a failure.</p>
<h2>3. Items sorted by |actual| (top {len(top)} of {len(effects)} rows; full data in effects/*.parquet)</h2>{item_table}
</body></html>"""
    out_path = Path(out_path)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def render(run_dir: str, out_html: str) -> Path:
    run_dir = Path(run_dir)
    meta_path = run_dir / "interval.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return render_report(Store(run_dir.parent), run_dir.name, meta.get("summary", meta), out_html)
