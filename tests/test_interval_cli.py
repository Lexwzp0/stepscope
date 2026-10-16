"""End-to-end: two saved tiny checkpoints + probe file -> tables, interval.json, report.html."""
import json
import math

import pytest

from stepscope.cli.main import main
from stepscope.core.store import Store
from tests.conftest import build_char_tokenizer, build_tiny_model, one_sgd_step, write_probe_file


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpts")
    m0 = build_tiny_model(0)
    m1 = one_sgd_step(m0, 1e-2)
    d0, d1 = root / "global_step_0", root / "global_step_10"
    for d, m in ((d0, m0), (d1, m1)):
        m.save_pretrained(str(d))
        build_char_tokenizer(d)
    probes = write_probe_file(root / "probes.jsonl")
    return d0, d1, probes, root


def _run(ckpts, extra):
    d0, d1, probes, root = ckpts
    out = root / "store"
    rc = main(["interval", "--from", str(d0), "--to", str(d1), "--probes", str(probes), "--out", str(out),
               "--answer-prefix", "\\nAnswer: ", "--label-format", "{label}", *extra])
    assert rc == 0
    return out


def test_interval_end_to_end(ckpts, capsys):
    out = _run(ckpts, ["--run-id", "e2e"])
    summary = json.loads(capsys.readouterr().out)
    st = Store(out)
    eff = st.read("e2e", "effects").to_pylist()
    beh = st.read("e2e", "behavior").to_pylist()
    con = st.read("e2e", "consistency").to_pylist()
    assert len(eff) == 8 * 2 and len(beh) == 8 * 2 * 2 and len(con) == 2
    assert {r["step_from"] for r in eff} == {0} and {r["step_to"] for r in eff} == {10}
    assert all(not math.isnan(r["first_order"]) for r in eff)
    assert all(r["corr"] > 0.99 for r in con), con
    assert summary["method"] == "jvp"
    assert summary["phi_from_recompute_max_gap"] < 1e-5
    run_dir = out / "e2e"
    assert (run_dir / "interval.json").exists() and (run_dir / "manifest.json").exists()
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html and "prediction calibration" in html and "math" in html and "science" in html


def test_backward_method_agrees(ckpts, capsys):
    out = _run(ckpts, ["--run-id", "bwd", "--method", "backward"])
    capsys.readouterr()
    st = Store(out)
    a = {(r["item"], r["observable"]): r["first_order"] for r in st.read("e2e", "effects").to_pylist()}
    b = {(r["item"], r["observable"]): r["first_order"] for r in st.read("bwd", "effects").to_pylist()}
    assert a.keys() == b.keys()
    assert max(abs(a[k] - b[k]) for k in a) < 1e-5
    assert {r["method"] for r in st.read("bwd", "effects").to_pylist()} == {"backward"}


def test_report_command_rerenders(ckpts, capsys):
    out = _run(ckpts, ["--run-id", "rr"])
    capsys.readouterr()
    rc = main(["report", str(out / "rr"), "--out", str(out / "rr" / "again.html")])
    assert rc == 0 and (out / "rr" / "again.html").exists()
