"""Regression (reviewed bug): default run_id must name the training run and both steps, and a report must
show only its own interval even when several intervals share one run_id."""
import json
import re

import pytest

from stepscope.cli.main import main
from stepscope.core.store import Store
from stepscope.pipelines.interval import default_run_id, describe_checkpoint
from tests.conftest import build_char_tokenizer, build_tiny_model, one_sgd_step, write_probe_file


def test_describe_checkpoint_and_default_run_id():
    assert describe_checkpoint("/x/runA/global_step_5/actor/huggingface") == ("runA", 5)
    assert describe_checkpoint("/x/runA/global_step_12") == ("runA", 12)
    assert describe_checkpoint("/models/MixSFT") == ("MixSFT", None)
    assert describe_checkpoint("/exports/ckpt_step_30/huggingface", 0) == ("ckpt_step_30", 30)
    assert default_run_id("/x/runA/global_step_5/actor/huggingface", "/x/runA/global_step_9/actor/huggingface", 5, 9) == "runA__5_9"
    assert default_run_id("/m/MixSFT", "/m/Final", 0, 1) == "MixSFT_0__Final_1"


@pytest.fixture(scope="module")
def trajectory(tmp_path_factory):
    root = tmp_path_factory.mktemp("traj") / "runA"
    m0 = build_tiny_model(0)
    m1 = one_sgd_step(m0, 1e-2, seed=1)
    m2 = one_sgd_step(m1, 1e-2, seed=2)
    paths = []
    for step, m in ((0, m0), (10, m1), (20, m2)):
        hf = root / f"global_step_{step}" / "actor" / "huggingface"
        m.save_pretrained(str(hf)); build_char_tokenizer(hf)
        paths.append(hf)
    return root, paths, write_probe_file(root / "probes.jsonl")


def _interval(paths, probes, out, a, b, extra=()):
    rc = main(["interval", "--from", str(paths[a]), "--to", str(paths[b]), "--probes", str(probes), "--out", str(out),
               "--answer-prefix", "\\nAnswer: ", "--label-format", "{label}", *extra])
    assert rc == 0


def test_default_run_id_from_verl_export_paths(trajectory, capsys):
    root, paths, probes = trajectory
    out = root / "store1"
    _interval(paths, probes, out, 0, 1)
    s = json.loads(capsys.readouterr().out)
    assert s["run_id"] == "runA__0_10" and (s["step_from"], s["step_to"]) == (0, 10)
    assert (out / "runA__0_10" / "report_0_10.html").exists() and (out / "runA__0_10" / "interval_0_10.json").exists()


def test_two_intervals_in_one_run_id_do_not_mix(trajectory, capsys):
    root, paths, probes = trajectory
    out = root / "store2"
    _interval(paths, probes, out, 0, 1, ["--run-id", "shared"]); capsys.readouterr()
    _interval(paths, probes, out, 1, 2, ["--run-id", "shared"]); s2 = json.loads(capsys.readouterr().out)
    st = Store(out)
    eff = st.read("shared", "effects").to_pylist()
    assert len(eff) == 2 * 16 and {(r["step_from"], r["step_to"]) for r in eff} == {(0, 10), (10, 20)}
    assert len(st.read("shared", "consistency").to_pylist()) == 4
    run_dir = out / "shared"
    latest = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "interval 10 → 20" in latest and "interval 0 → 10" not in latest
    assert len(re.findall(r"<circle ", latest)) == 16          # 8 items × 2 observables, this interval only
    assert "of 16 rows of this interval" in latest
    first = (run_dir / "report_0_10.html").read_text(encoding="utf-8")
    assert "interval 0 → 10" in first and len(re.findall(r"<circle ", first)) == 16
    assert s2["consistency"]["phi_ans"]["n"] == 8
    # re-render a specific interval from the run directory
    assert main(["report", str(run_dir), "--interval", "0_10", "--out", str(run_dir / "again.html")]) == 0
    assert "interval 0 → 10" in (run_dir / "again.html").read_text(encoding="utf-8")
