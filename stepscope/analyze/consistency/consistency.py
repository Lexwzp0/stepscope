"""Prediction vs reference, and decompositions vs their target (blueprint §四 Consistency, §七, §十四.4).

One row per (probe_set, probe_version, observable, step_from, step_to) and per prediction kind:

- ``first_order`` vs ``actual``: approximation error of the first-order prediction;
- ``ch1_ch2`` vs ``first_order_loss`` when a separate loss-only reference is available; otherwise vs
  ``first_order`` (chain level 1) and ``actual``. The channel prediction sums over sources and sites and
  excludes weight decay, so it is not directly compared to the full actual update when decay is present;
- ``decomposition_sum`` per ``decomp_id``: Σ parts of the ``contributions`` rows vs ``actual`` for the
  approximation error, and vs the decomposition's own ``target`` column (first_order or actual) for the
  alignment residual ``resid_target_minus_sum``. Every decomposition goes through the same rule.

Statistics: n, Pearson correlation, sign agreement (items with |reference| < eps excluded), mean residual,
residual RMS, reference RMS."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ...core.registry import ANALYSES

VERSION = "0.2.1"
_GROUP = ("probe_set", "probe_version", "observable", "step_from", "step_to")


def summarize(reference, pred, eps: float = 1e-8) -> dict:
    a = np.asarray(reference, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    n = int(a.size)
    corr = float("nan")
    if n >= 2 and a.std() > 0 and p.std() > 0:
        corr = float(np.corrcoef(a, p)[0, 1])
    m = np.abs(a) >= eps
    sign_agree = float(np.mean(np.sign(a[m]) == np.sign(p[m]))) if m.any() else float("nan")
    resid = a - p
    return dict(
        n=n, corr=corr, sign_agree=sign_agree,
        resid_actual_minus_pred=float(resid.mean()) if n else float("nan"),
        resid_rms=float(math.sqrt(np.mean(resid ** 2))) if n else float("nan"),
        actual_rms=float(math.sqrt(np.mean(a ** 2))) if n else float("nan"),
        resid_target_minus_sum=float("nan"),
    )


def _key(r: dict) -> tuple:
    return (r["probe_set"], r["probe_version"], r["observable"], int(r["step_from"]), int(r["step_to"]))


def _row(key: tuple, prediction: str, reference: str, stats: dict, prov: dict, decomp_id: str | None = None) -> dict:
    ps, pv, obs, sf, st = key
    return dict(step_from=sf, step_to=st, probe_set=ps, probe_version=pv, observable=obs, prediction=prediction,
                decomp_id=decomp_id, reference=reference, **stats, **prov)


def _finite(x) -> bool:
    return x is not None and not math.isnan(x)


def consistency_rows(effects_rows: list[dict], prov: dict, prediction: str = "first_order") -> list[dict]:
    """``first_order`` vs ``actual`` per group (items without a first-order value are left out)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in effects_rows:
        groups[_key(r)].append(r)
    out = []
    for key, rows in sorted(groups.items()):
        rows = [r for r in rows if _finite(r.get("first_order"))]
        s = summarize([r["actual"] for r in rows], [r["first_order"] for r in rows])
        out.append(_row(key, prediction, "actual", s, prov))
    return out


def channel_totals(channel_rows: list[dict]) -> dict[tuple, float]:
    """(group key, item) -> Σ over sources and sites of ch1 + ch2."""
    tot: dict[tuple, float] = defaultdict(float)
    for r in channel_rows:
        tot[(_key(r), r["item"])] += float(r["ch1"]) + float(r["ch2"])
    return dict(tot)


def channel_consistency_rows(effects_rows: list[dict], channel_rows: list[dict], prov: dict) -> list[dict]:
    """Compare channels with the loss-only reference when present; preserve legacy zero-decay comparisons."""
    tot = channel_totals(channel_rows)
    groups: dict[tuple, list[tuple[dict, float]]] = defaultdict(list)
    for r in effects_rows:
        k = (_key(r), r["item"])
        if k in tot:
            groups[_key(r)].append((r, tot[k]))
    out = []
    for key, pairs in sorted(groups.items()):
        reference = "first_order_loss" if any(_finite(r.get("first_order_loss")) for r, _ in pairs) else "first_order"
        fo = [(r, p) for r, p in pairs if _finite(r.get(reference))]
        out.append(_row(key, "ch1_ch2", reference,
                        summarize([r[reference] for r, _ in fo], [p for _, p in fo]), prov))
        if reference == "first_order":
            out.append(_row(key, "ch1_ch2", "actual",
                            summarize([r["actual"] for r, _ in pairs], [p for _, p in pairs]), prov))
    return out


def decomposition_rows(effects_rows: list[dict], contribution_rows: list[dict], prov: dict) -> list[dict]:
    """One row per decomposition and group: Σ parts vs actual, and RMS(target − Σ parts) with the target column
    named by the decomposition's rows (all rows of one decomposition must name the same target)."""
    eff = {(_key(r), r["item"]): r for r in effects_rows}
    sums: dict[tuple, float] = defaultdict(float)
    targets: dict[str, set] = defaultdict(set)
    for r in contribution_rows:
        sums[(r["decomp_id"], _key(r), r["item"])] += float(r["contrib"])
        targets[r["decomp_id"]].add(r["target"])
    for d, t in targets.items():
        if len(t) != 1 or not t <= {"first_order", "actual"}:
            raise ValueError(f"decomposition {d!r}: target must be one of first_order / actual, got {sorted(t)}")
    groups: dict[tuple, list[tuple[dict, float]]] = defaultdict(list)
    for (d, key, item), s in sums.items():
        e = eff.get((key, item))
        if e is None:
            raise ValueError(f"decomposition {d!r}: no effects row for {key} item {item!r}")
        groups[(d, key)].append((e, s))
    out = []
    for (d, key), pairs in sorted(groups.items()):
        target = next(iter(targets[d]))
        stats = summarize([e["actual"] for e, _ in pairs], [s for _, s in pairs])
        gaps = [e[target] - s for e, s in pairs if _finite(e.get(target))]
        stats["resid_target_minus_sum"] = float(math.sqrt(np.mean(np.square(gaps)))) if gaps else float("nan")
        out.append(_row(key, "decomposition_sum", "actual", stats, prov, decomp_id=d))
    return out


@ANALYSES.register("consistency.consistency")
class Consistency:
    name = "consistency.consistency"
    version = VERSION
    level = "exact"
    requires = {"effects"}            # channels and contributions are used when present
    writes = {"consistency"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        effects = store.read(run_id, "effects").to_pylist()
        prov = inputs["prov"]
        rows = consistency_rows(effects, prov, prediction=inputs.get("prediction", "first_order"))
        rows += channel_consistency_rows(effects, store.read(run_id, "channels").to_pylist(), prov)
        rows += decomposition_rows(effects, store.read(run_id, "contributions").to_pylist(), prov)
        store.write(run_id, "consistency", rows, part=inputs.get("part", "0000"))
