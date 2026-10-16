"""verl layout scan, ingest, and `interval --run run.json --steps A B` on a fake verl tree."""
import json

import pytest

from stepscope.adapters.frameworks.verl import VERLReader, scan_verl_run
from stepscope.cli.main import main
from stepscope.core.store import Store
from tests.conftest import build_char_tokenizer, build_tiny_model, one_sgd_step, write_probe_file


@pytest.fixture(scope="module")
def verl_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("verl_run")
    m0 = build_tiny_model(0)
    m1 = one_sgd_step(m0, 1e-2)
    for step, m in ((10, m0), (20, m1)):
        hf = root / f"global_step_{step}" / "actor" / "huggingface"
        m.save_pretrained(str(hf))
        build_char_tokenizer(hf)
        (root / f"global_step_{step}" / "actor" / "model_world_size_2_rank_0.pt").write_bytes(b"x")
    # step 30: shards only, no HF weights -> pending merge
    a30 = root / "global_step_30" / "actor"
    (a30 / "huggingface").mkdir(parents=True)
    (a30 / "huggingface" / "config.json").write_text("{}")
    for r in range(2):
        (a30 / f"model_world_size_2_rank_{r}.pt").write_bytes(b"x")
    (root / "latest_checkpointed_iteration.txt").write_text("30")
    (root / "not_a_step").mkdir()
    probes = write_probe_file(root / "probes.jsonl")
    return root, probes


def test_scan_and_intervals(verl_tree):
    root, _ = verl_tree
    cks = scan_verl_run(root)
    assert [c.step for c in cks] == [10, 20, 30]
    assert [c.has_weights for c in cks] == [True, True, False]
    assert cks[2].merge_cmd and "verl.model_merger merge --backend fsdp" in cks[2].merge_cmd and cks[2].n_shards == 2
    run = VERLReader().read_run(str(root))
    assert [c.step for c in run.checkpoints] == [10, 20]
    assert run.meta["latest_checkpointed_iteration"] == "30" and [p["step"] for p in run.meta["pending"]] == [30]
    iv = list(VERLReader().intervals(run))
    assert len(iv) == 1 and (iv[0].ckpt_from.step, iv[0].ckpt_to.step) == (10, 20)
    assert list(VERLReader().update_packages(run)) == []


def test_ingest_then_interval_by_steps(verl_tree, capsys):
    root, probes = verl_tree
    store = root / "store"
    assert main(["ingest", str(root), "--out", str(store), "--run-id", "mopd"]) == 0
    out = capsys.readouterr().out
    assert "2 checkpoints with HF weights [10, 20]" in out and "1 pending merge" in out
    run_json = store / "mopd" / "run.json"
    doc = json.loads(run_json.read_text(encoding="utf-8"))
    assert [c["step"] for c in doc["checkpoints"]] == [10, 20] and doc["pending"][0]["step"] == 30

    rc = main(["interval", "--run", str(run_json), "--steps", "10", "20", "--probes", str(probes), "--out", str(store),
               "--answer-prefix", "\\nAnswer: ", "--label-format", "{label}"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["run_id"] == "mopd__10_20" and (summary["step_from"], summary["step_to"]) == (10, 20)
    eff = Store(store).read("mopd__10_20", "effects").to_pylist()
    assert len(eff) == 16 and {r["step_from"] for r in eff} == {10}

    with pytest.raises(SystemExit) as e:
        main(["interval", "--run", str(run_json), "--steps", "20", "30", "--probes", str(probes), "--out", str(store)])
    assert "step 30" in str(e.value) and "model_merger" in str(e.value)
