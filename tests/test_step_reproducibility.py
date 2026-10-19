"""Result isolation on reruns and reproducible answer-slot protocols."""
import copy
import json
import math
from dataclasses import asdict, replace

import pyarrow.parquet as pq
import pytest

from stepscope.adapters.models.hf import HFModelAdapter
from stepscope.adapters.objectives.sft import SFTObjective
from stepscope.analyze.consistency.consistency import summarize
from stepscope.capture.train_step import OptimizerConfig
from stepscope.core.store import Store
from stepscope.observe.probes import load_probe_file
from stepscope.observe.scoring import encode_probes
from stepscope.pipelines.step import _probe_protocol, run_step_on
from stepscope.report.html import render_step_report
from stepscope.testing.tiny import BACKBONES, RENDER, random_train_batch


def _run(tokenizer, probe_file, tmp_path, *, render=RENDER, optimizer="sgd", weight_decay=0.0, **kwargs):
    return run_step_on(
        HFModelAdapter(BACKBONES["llama"](0), tokenizer), SFTObjective(), random_train_batch(1),
        OptimizerConfig(optimizer, lr=1e-3, weight_decay=weight_decay),
        probes_path=str(probe_file), out=str(tmp_path), run_id="r",
        render=render, max_items=2, **kwargs,
    )


@pytest.mark.parametrize("rerun", [{"by_source": False, "channels": False}, {"optimizer": "adamw"}])
def test_rerun_clears_optional_tables_and_status(tokenizer, probe_file, tmp_path, rerun):
    first = _run(tokenizer, probe_file, tmp_path, save_package=False)
    store = Store(tmp_path)
    assert store.read("r", "channels").num_rows > 0
    assert store.read("r", "contributions").num_rows > 0

    latest = _run(tokenizer, probe_file, tmp_path, save_package=False, **rerun)
    assert latest["tag"] == first["tag"]
    manifest = store.manifest("r")
    for table, analysis in (("channels", "dynamics.ldcl_channels"),
                            ("contributions", "dynamics.sgd_source_decomposition")):
        assert store.read("r", table).num_rows == 0
        assert manifest["tables"][table][latest["tag"]]["rows"] == 0
        entry = manifest["analyses"][f"{analysis}@{latest['tag']}"]
        assert entry["status"] == "skipped" and entry["reasons"] == latest["skipped"][analysis]
    assert {r["prediction"] for r in store.read("r", "consistency").to_pylist()} == {"first_order"}
    report = (tmp_path / "r" / "report_step.html").read_text(encoding="utf-8")
    assert "CH1+CH2 vs first order" not in report
    assert "first-order part of each source" not in report
    assert "No per-source tables" in report

    restored = _run(tokenizer, probe_file, tmp_path, save_package=False)
    assert not restored["skipped"]
    assert all(v["status"] == "ran" for v in store.manifest("r")["analyses"].values())
    assert store.read("r", "channels").num_rows > 0


def test_changed_prefix_keeps_results_separate_and_protocol_reproducible(tokenizer, probe_file, tmp_path):
    first = _run(tokenizer, probe_file, tmp_path, by_source=False, channels=False)
    changed = replace(RENDER, answer_prefix="\nAnswer is: ")
    second = _run(tokenizer, probe_file, tmp_path, render=changed, by_source=False, channels=False)
    assert first["probe_file_version"] == second["probe_file_version"]
    assert first["probe_version"] != second["probe_version"] and first["tag"] != second["tag"]
    assert first["encoding_sha256"] != second["encoding_sha256"]
    assert first["protocol_sha256"] != second["protocol_sha256"]
    store = Store(tmp_path)
    rows = store.read("r", "effects").to_pylist()
    assert len(rows) == 8 and {r["probe_version"] for r in rows} == {first["probe_version"], second["probe_version"]}
    behavior = store.read("r", "behavior").to_pylist()
    assert len(behavior) == 2 * 2 * 2 * 2  # protocols × steps × items × behavior metrics
    assert {r["probe_version"] for r in behavior} == {first["probe_version"], second["probe_version"]}
    for summary in (first, second):
        doc = json.loads((tmp_path / "r" / f"step_{summary['tag']}.json").read_text(encoding="utf-8"))
        assert doc["render"] == doc["probe_protocol"]["render"] == doc["summary"]["render"]
        assert doc["probe_protocol"] == store.manifest("r")["probe_protocols"][summary["tag"]]
        report_path = tmp_path / "r" / f"again_{summary['tag']}.html"
        render_step_report(store, "r", doc["summary"], report_path)
        # Two observables × two items, with no rows from the other protocol.
        assert report_path.read_text(encoding="utf-8").count("<circle ") == 4
    pkg = json.loads((tmp_path / "r" / "package_0" / "package.json").read_text(encoding="utf-8"))
    assert pkg["meta"]["probe_protocol"]["protocol_sha256"] == second["protocol_sha256"]
    assert pkg["meta"]["probe_protocol"]["render"] == json.loads(json.dumps(asdict(changed)))
    assert (tmp_path / "r" / "report_step.html").exists()


def test_template_and_exact_encoding_both_affect_protocol_fingerprint(tokenizer, probe_file):
    tok = copy.deepcopy(tokenizer)
    cfg = replace(RENDER, chat_template=True, enable_thinking=False, fixed_date="2026-10-19")
    tok.chat_template = "{{ messages[0]['content'] }}"
    items = load_probe_file(probe_file).items[:2]
    enc = encode_probes(tok, items, cfg)
    first = _probe_protocol(tok, cfg, enc)
    assert first == _probe_protocol(tok, cfg, enc)

    # Even templates that currently render the same text have distinct provenance.
    tok.chat_template += "{# template revision #}"
    enc_same = encode_probes(tok, items, cfg)
    second = _probe_protocol(tok, cfg, enc_same)
    assert first["encoding_sha256"] == second["encoding_sha256"]
    assert first["chat_template_sha256"] != second["chat_template_sha256"]
    assert first["protocol_sha256"] != second["protocol_sha256"]

    changed_ids = copy.deepcopy(enc_same.id_lists)
    changed_ids[0][0] += 1
    third = _probe_protocol(tok, cfg, replace(enc_same, id_lists=changed_ids))
    assert second["chat_template_sha256"] == third["chat_template_sha256"]
    assert second["encoding_sha256"] != third["encoding_sha256"]
    assert second["protocol_sha256"] != third["protocol_sha256"]


@pytest.mark.parametrize("by_source", [True, False])
def test_public_channel_calibration_excludes_decay(tokenizer, probe_file, tmp_path, by_source):
    baseline = _run(tokenizer, probe_file, tmp_path, by_source=by_source, save_package=False)
    store = Store(tmp_path)
    loss_only = {(r["item"], r["observable"]): r["first_order"] for r in store.read("r", "effects").to_pylist()}
    assert baseline["channel_reference"] == "first_order"
    latest = _run(tokenizer, probe_file, tmp_path, by_source=by_source, weight_decay=0.4, save_package=False)
    effects = store.read("r", "effects").to_pylist()
    assert latest["channel_reference"] == "first_order_loss"
    assert all(math.isclose(r["first_order_loss"], loss_only[(r["item"], r["observable"])],
                            rel_tol=1e-5, abs_tol=1e-7) for r in effects)
    assert max(abs(r["first_order"] - r["first_order_loss"]) for r in effects) > 1e-5
    totals = {}
    for row in store.read("r", "channels").to_pylist():
        key = row["item"], row["observable"]
        totals[key] = totals.get(key, 0.0) + row["ch1"] + row["ch2"]
    consistency = store.read("r", "consistency").to_pylist()
    channel_checks = [r for r in consistency if r["prediction"] == "ch1_ch2"]
    assert {r["reference"] for r in channel_checks} == {"first_order_loss"}
    for check in channel_checks:
        rows = [r for r in effects if r["observable"] == check["observable"]]
        expected = summarize([r["first_order_loss"] for r in rows],
                             [totals[(r["item"], r["observable"])] for r in rows])
        assert math.isclose(check["resid_rms"], expected["resid_rms"], rel_tol=1e-10)
    assert {r["reference"] for r in consistency if r["prediction"] == "first_order"} == {"actual"}
    if by_source:
        assert all(r["resid_target_minus_sum"] < 1e-7 for r in consistency if r["prediction"] == "decomposition_sum")
    report = (tmp_path / "r" / "report_step.html").read_text(encoding="utf-8")
    assert "CH1+CH2 vs loss-only first order" in report and "weight decay excluded" in report
    assert "CH1+CH2 vs first order" not in report and "first order vs actual" in report


def test_legacy_and_current_effect_parts_can_be_read_and_rendered_together(tmp_path):
    store = Store(tmp_path)
    keys = dict(step_from=0, step_to=1, probe_set="p", probe_version="old")
    old = dict(**keys, item="i0", domain="math", observable="phi_ans", value_from=0.0, value_to=0.1,
               actual=0.1, first_order=0.09, method="jvp")
    path = store.write("r", "effects", [old], part="old")
    # Reproduce a pre-change parquet file, which physically lacks the added nullable column.
    pq.write_table(pq.read_table(path).drop(["first_order_loss"]), path)
    store.write("r", "effects", [{**old, "probe_version": "new", "first_order_loss": 0.08}], part="new")
    loaded = {r["probe_version"]: r for r in store.read("r", "effects").to_pylist()}
    assert loaded["old"]["first_order_loss"] is None and loaded["new"]["first_order_loss"] == 0.08
    report_path = tmp_path / "legacy.html"
    render_step_report(store, "r", {"run_id": "r", **keys}, report_path)
    report = report_path.read_text(encoding="utf-8")
    assert report.count("<circle ") == 1 and "loss-only first order" not in report
