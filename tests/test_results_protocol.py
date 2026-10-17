"""Result protocol (blueprint §七): decompositions are checked against the column their ``target`` names,
channel predictions against first_order and actual; representation rows may reference an artefact;
delta_params releases θ_to tensor by tensor."""
import copy
import math

import pytest
import torch

from stepscope.analyze.consistency.consistency import (channel_consistency_rows, consistency_rows,
                                                       decomposition_rows)
from stepscope.analyze.dynamics import interval as iv
from stepscope.core.provenance import provenance
from stepscope.core.store import Store
from stepscope.testing.tiny import build_tiny_model, one_sgd_step

PROV = provenance("r", "test", "0", "exact")
KEY = dict(step_from=0, step_to=1, probe_set="p", probe_version="v")


def _effects():
    rows = []
    for i, (a, f) in enumerate([(0.10, 0.09), (-0.20, -0.18), (0.05, 0.06), (0.30, 0.25)]):
        rows.append(dict(**KEY, item=f"i{i}", domain="d", observable="phi_ans", value_from=0.0, value_to=a,
                         actual=a, first_order=f, method="jvp", **PROV))
    return rows


def _parts(effects, decomp_id, target, shift=0.0):
    rows = []
    for e in effects:
        tot = e[target] + shift
        for src, frac in (("math", 0.7), ("code", 0.3)):
            rows.append(dict(**KEY, item=e["item"], observable=e["observable"], decomp_id=decomp_id, target=target,
                             source_kind="teacher", source=src, contrib=frac * tot, method="test", **PROV))
    return rows


def test_decomposition_checked_against_its_target():
    eff = _effects()
    contrib = _parts(eff, "by_source", "first_order") + _parts(eff, "by_layer", "actual") \
        + _parts(eff, "misaligned", "first_order", shift=0.01)
    rows = {r["decomp_id"]: r for r in decomposition_rows(eff, contrib, PROV)}
    assert set(rows) == {"by_source", "by_layer", "misaligned"}
    assert rows["by_source"]["resid_target_minus_sum"] < 1e-12
    assert rows["by_layer"]["resid_target_minus_sum"] < 1e-12
    assert abs(rows["misaligned"]["resid_target_minus_sum"] - 0.01) < 1e-12
    for r in rows.values():
        assert r["prediction"] == "decomposition_sum" and r["reference"] == "actual" and r["n"] == 4
    # Σ parts of the first-order decomposition vs actual = first-order approximation error
    fo = consistency_rows(eff, PROV)[0]
    assert math.isclose(rows["by_source"]["resid_rms"], fo["resid_rms"]) and fo["decomp_id"] is None
    assert rows["by_layer"]["resid_rms"] < 1e-12 and math.isnan(fo["resid_target_minus_sum"])


def test_one_decomposition_one_target():
    eff = _effects()
    bad = _parts(eff, "d", "first_order")
    bad[0]["target"] = "actual"
    with pytest.raises(ValueError, match="target"):
        decomposition_rows(eff, bad, PROV)


def test_channel_rows_compared_with_first_order_and_actual():
    eff = _effects()
    ch = []
    for e in eff:
        for layer, c1, c2 in (("readout", e["first_order"] * 0.5, 0.0), ("layers.0.attn_in", 0.0, e["first_order"] * 0.5)):
            ch.append(dict(**KEY, item=e["item"], observable="phi_ans", source_kind="domain", source="all",
                           layer=layer, ch1=c1, ch2=c2, applicability="sgd", **PROV))
    rows = {r["reference"]: r for r in channel_consistency_rows(eff, ch, PROV)}
    assert rows["first_order"]["prediction"] == "ch1_ch2" and rows["first_order"]["resid_rms"] < 1e-12
    assert math.isclose(rows["actual"]["corr"], consistency_rows(eff, PROV)[0]["corr"])


def test_tables_roundtrip_with_new_columns(tmp_path):
    st = Store(tmp_path)
    eff = _effects()
    st.write("r", "contributions", _parts(eff, "by_source", "first_order"), part="a")
    st.write("r", "consistency", consistency_rows(eff, PROV) + decomposition_rows(eff, _parts(eff, "x", "actual"), PROV), part="a")
    st.write("r", "representation", [
        dict(step=0, layer="layers.0", metric="rankme", probe_set="p", probe_version="v", value=3.2, artifact=None, **PROV),
        dict(step=0, layer="layers.0", metric="spectrum", probe_set="p", probe_version="v", value=float("nan"),
             artifact="artifacts/spectrum_0.npy#sha1=abc", **PROV)], part="a")
    assert {r["target"] for r in st.read("r", "contributions").to_pylist()} == {"first_order"}
    assert [r["decomp_id"] for r in st.read("r", "consistency").to_pylist()] == [None, "x"]
    assert [r["artifact"] for r in st.read("r", "representation").to_pylist()] == [None, "artifacts/spectrum_0.npy#sha1=abc"]


def test_delta_params_release_frees_theta_to():
    m0 = build_tiny_model(0)
    m1 = one_sgd_step(m0, 1e-2)
    ref, _ = iv.delta_params(m0, m1)
    m1b = copy.deepcopy(m1)
    delta, info = iv.delta_params(m0, m1b, release=True)
    assert all(torch.equal(delta[k], ref[k]) for k in ref) and info.norm > 0
    assert all(p.numel() == 0 for p in m1b.parameters())          # every θ_to tensor was released
    assert all(p.numel() > 0 for p in m0.parameters())


def test_delta_params_checks_before_releasing():
    m0 = build_tiny_model(0)
    m1 = copy.deepcopy(m0)
    with torch.no_grad():
        m1.lm_head.weight.data = torch.zeros(5, 5)
    with pytest.raises(ValueError, match="shape mismatch"):
        iv.delta_params(m0, m1, release=True)
    assert all(p.numel() > 0 for p in m1.parameters())             # nothing released on failure
