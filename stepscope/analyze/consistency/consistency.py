"""Prediction vs actual (blueprint §四 Consistency, §十四.4).

Per (probe_set, probe_version, observable, step_from, step_to): n, Pearson correlation, sign agreement
(items with |actual| < eps excluded), mean residual, residual RMS, actual RMS. Decompositions only:
``resid_pred_minus_sum`` stays NaN until a per-source decomposition exists for the same rows."""
from __future__ import annotations

import math

import numpy as np

from ...core.registry import ANALYSES

VERSION = "0.1.0"


def summarize(actual, pred, eps: float = 1e-8) -> dict:
    a = np.asarray(actual, dtype=np.float64)
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
        resid_pred_minus_sum=float("nan"),
    )


def consistency_rows(effects_rows: list[dict], prov: dict, prediction: str = "first_order") -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in effects_rows:
        key = (r["probe_set"], r["probe_version"], r["observable"], int(r["step_from"]), int(r["step_to"]))
        groups.setdefault(key, []).append(r)
    out = []
    for (ps, pv, obs, sf, st), rows in sorted(groups.items()):
        rows = [r for r in rows if r.get("first_order") is not None and not math.isnan(r["first_order"])]
        s = summarize([r["actual"] for r in rows], [r["first_order"] for r in rows])
        out.append(dict(step_from=sf, step_to=st, probe_set=ps, probe_version=pv, observable=obs,
                        prediction=prediction, **s, **prov))
    return out


@ANALYSES.register("consistency.consistency")
class Consistency:
    name = "consistency.consistency"
    version = VERSION
    level = "exact"
    requires = {"effects"}            # contributions are optional (fill resid_pred_minus_sum when present)
    writes = {"consistency"}

    def run(self, inputs: dict, probes: list, store, run_id: str) -> None:
        effects = store.read(run_id, "effects").to_pylist()
        rows = consistency_rows(effects, inputs["prov"], prediction=inputs.get("prediction", "first_order"))
        store.write(run_id, "consistency", rows, part=inputs.get("part", "0000"))
