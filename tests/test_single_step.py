"""Single-step check chain (blueprint §十九 校验链 1–2) and the reuse acceptance of §二十: two backbones (tiny
Llama, tiny GPT-2) × two objectives (SFT, MOPD). Every combination runs the same analysis code; only the
adapter and the configuration below change."""
import json
import math

import pytest
import torch

from stepscope.adapters.models.hf import HFModelAdapter
from stepscope.adapters.objectives.mopd import MOPDObjective
from stepscope.adapters.objectives.sft import SFTObjective
from stepscope.analyze.dynamics.ldcl_channels import (accumulate_source_matrices, channel_scores, pairwise_reference,
                                                      source_matrices_sparse)
from stepscope.analyze.dynamics.forces import dense_force
from stepscope.analyze.dynamics.single_step import run_chain
from stepscope.capture.batch import mopd_extras
from stepscope.capture.package import PackageWriter, load_package
from stepscope.capture.train_step import OptimizerConfig, take_step
from stepscope.cli.main import main
from stepscope.core.store import Store
from stepscope.observe.observables import observable_logit_grad, phi_all
from stepscope.observe.probes import load_probe_file
from stepscope.observe.scoring import encode_probes
from stepscope.pipelines.step import run_step_on
from stepscope.testing.tiny import (BACKBONES, RENDER, random_train_batch, save_checkpoint, write_batch_file,
                                    write_probe_file)

NAMES = ("phi_fmt", "phi_ans")
# configuration per combination: the only thing that changes besides the adapter
LR = {("llama", "sgd"): 1e-2, ("gpt2", "sgd"): 1e-3, ("llama", "adamw"): 1e-4, ("gpt2", "adamw"): 1e-4}
OBJECTIVES = {"sft": lambda: SFTObjective(), "mopd": lambda: MOPDObjective(weight_mode="student_p")}


def _batch(arch, objective, model):
    b = random_train_batch(1)
    if objective == "mopd":
        b = mopd_extras(model, {"math": BACKBONES[arch](7), "code": BACKBONES[arch](8)}, b, k=4)
    return b


def _within_first_order(actual, pred, tol_rel=0.05, tol_abs=1e-6):
    err = (actual - pred).abs().max().item()
    scale = actual.abs().max().item()
    assert scale > 1e-5, "update too small to test anything"
    assert err <= tol_rel * scale + tol_abs, f"first-order error {err:.3e} vs max actual {scale:.3e}"


def _corr(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("objective", ["sft", "mopd"])
@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_chain_sgd(arch, objective, tokenizer, probe_file, tmp_path):
    model = BACKBONES[arch](0)
    ad = HFModelAdapter(model, tokenizer)
    s = run_step_on(ad, OBJECTIVES[objective](), _batch(arch, objective, model), OptimizerConfig("sgd", lr=LR[(arch, "sgd")]),
                    probes_path=str(probe_file), out=str(tmp_path), run_id="r", render=RENDER)
    d = s["diagnostics"]
    assert d["reconstruction"]["passed"] and d["tangent"] == "reconstructed_sgd"
    assert d["readout_channel_check"]["rel_err"] < 1e-4                 # η·CH1 = JVP along the readout part, exactly
    assert d["phi_from_recompute_max_gap"] < 1e-5 and not s["skipped"]
    st = Store(tmp_path)
    eff = st.read("r", "effects").to_pylist()
    actual = torch.tensor([[r["actual"] for r in eff if r["observable"] == o] for o in NAMES])
    fo = torch.tensor([[r["first_order"] for r in eff if r["observable"] == o] for o in NAMES])
    _within_first_order(actual, fo)                                     # chain level 2
    assert all(_corr(actual[i], fo[i]) > 0.99 for i in range(2))
    con = {(r["prediction"], r["reference"], r["observable"]): r for r in st.read("r", "consistency").to_pylist()}
    for o in NAMES:
        c = con[("ch1_ch2", "first_order", o)]                          # chain level 1: recorded, not tuned
        assert c["n"] == 8 and math.isfinite(c["corr"]) and 0.0 <= c["sign_agree"] <= 1.0
        dec = con[("decomposition_sum", "actual", o)]
        assert dec["decomp_id"] == "sgd_by_source" and dec["resid_target_minus_sum"] < 1e-8
    ch = st.read("r", "channels").to_pylist()
    assert len(ch) == 8 * 2 * 2 * (1 + 4) and {r["applicability"] for r in ch} == {"sgd"}  # items×obs×sources×(readout+4 sites)
    assert {r["source"] for r in st.read("r", "contributions").to_pylist()} == {"math", "code"}
    analyses = st.manifest("r")["analyses"]
    assert all(v["status"] == "ran" for v in analyses.values()) and len(analyses) == 3
    html = (tmp_path / "r" / "report_step.html").read_text(encoding="utf-8")
    assert "CH1+CH2 vs first order" in html and "first order vs actual" in html and "Sources × domains" in html


@pytest.mark.parametrize("objective", ["sft", "mopd"])
@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_chain_adamw_compares_first_order_with_actual_only(arch, objective, tokenizer, probe_file, tmp_path):
    model = BACKBONES[arch](0)
    ad = HFModelAdapter(model, tokenizer)
    s = run_step_on(ad, OBJECTIVES[objective](), _batch(arch, objective, model),
                    OptimizerConfig("adamw", lr=LR[(arch, "adamw")], weight_decay=0.01),
                    probes_path=str(probe_file), out=str(tmp_path), run_id="r", render=RENDER)
    assert s["diagnostics"]["tangent"] == "measured"
    st = Store(tmp_path)
    eff = st.read("r", "effects").to_pylist()
    _within_first_order(torch.tensor([r["actual"] for r in eff]), torch.tensor([r["first_order"] for r in eff]))
    assert {r["prediction"] for r in st.read("r", "consistency").to_pylist()} == {"first_order"}
    assert st.read("r", "channels").num_rows == 0 and st.read("r", "contributions").num_rows == 0
    analyses = st.manifest("r")["analyses"]
    skipped = {k.split("@")[0]: v for k, v in analyses.items() if v["status"] == "skipped"}
    assert "SGD-direction" in " ".join(skipped["dynamics.ldcl_channels"]["reasons"])
    assert "dynamics.sgd_source_decomposition" in skipped


def test_package_roundtrip_reproduces_the_chain(tokenizer, probe_file, tmp_path):
    model = BACKBONES["llama"](0)
    ad = HFModelAdapter(model, tokenizer)
    enc = encode_probes(tokenizer, load_probe_file(probe_file).items, RENDER)
    obj = MOPDObjective()
    pkg = take_step(model, obj, _batch("llama", "mopd", model), OptimizerConfig("sgd", lr=1e-2), run_id="r", step=3)
    ref = PackageWriter(tmp_path / "pkg").write(pkg)
    assert ref.post_params_ref == "delta.safetensors" and ref.config["objective"] == "mopd"
    doc = json.loads((tmp_path / "pkg" / "package.json").read_text(encoding="utf-8"))
    assert doc["post_params_kind"] == "measured_delta" and doc["optimizer_step"] == 3 and set(doc["sha1"]) >= {"batch.pt"}
    a = run_chain(ad, obj, pkg, enc, NAMES)
    loaded = load_package(tmp_path / "pkg")
    b = run_chain(ad, MOPDObjective(**loaded.objective_config), loaded, enc, NAMES)
    assert torch.allclose(a.first_order, b.first_order, atol=1e-9) and torch.allclose(a.channels, b.channels, atol=1e-9)
    assert torch.equal(a.phi_to, b.phi_to) and loaded.optimizer.name == "sgd" and loaded.batch.sources == pkg.batch.sources
    # the chain leaves the model at θ before the step
    assert all(torch.equal(p, pkg.pre_params[k]) for k, p in model.named_parameters())


def test_observable_logit_grad_matches_autograd():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(5, 40, generator=g)
    opt = torch.stack([torch.randperm(40, generator=g)[:4] for _ in range(5)])
    correct = torch.tensor([0, 1, 2, 3, 1])
    for i, name in enumerate(NAMES):
        z = logits.clone().requires_grad_(True)
        (ref,) = torch.autograd.grad(phi_all([name], z, opt, correct).sum(), z)
        assert torch.allclose(observable_logit_grad(torch.softmax(logits, -1), opt, correct, name), ref, atol=1e-6)


def test_aggregated_channels_equal_pairwise_sum():
    g = torch.Generator().manual_seed(0)
    V, d, L, m, n, K = 30, 8, 3, 4, 50, 3
    W = torch.randn(V, d, generator=g)
    logits = torch.randn(n, V, generator=g)
    ids = torch.stack([torch.randperm(V, generator=g)[:K] for _ in range(n)])
    coef = torch.randn(n, K, generator=g)
    hL_u = torch.randn(n, d, generator=g)
    hl_u = {f"s{i}": torch.randn(n, d, generator=g) for i in range(L)}
    g_o, hL_o = torch.randn(m, V, generator=g), torch.randn(m, d, generator=g)
    hl_o = [torch.randn(m, d, generator=g) for _ in range(L)]
    f = dense_force(coef, ids, torch.softmax(logits, -1))
    M, N = accumulate_source_matrices(f, hL_u, list(hl_u.values()), W, chunk=7)
    sm = source_matrices_sparse(coef, ids, logits, hL_u, hl_u, W, chunk=9)
    assert torch.allclose(sm.M, M, atol=1e-4) and all(torch.allclose(sm.N[k], N[i], atol=1e-3) for i, k in enumerate(hl_u))
    s1, s2 = channel_scores(g_o, hL_o, hl_o, W, sm.M, list(sm.N.values()))
    r1, r2 = pairwise_reference(g_o, hL_o, hl_o, f, hL_u, list(hl_u.values()), W)
    assert torch.allclose(s1, r1, rtol=1e-4, atol=1e-3) and torch.allclose(s2, r2, rtol=1e-4, atol=1e-3)


def test_M_is_minus_readout_gradient():
    g = torch.Generator().manual_seed(1)
    V, d, n = 20, 6, 15
    W = torch.randn(V, d, generator=g, requires_grad=True)
    h = torch.randn(n, d, generator=g)
    y = torch.randint(0, V, (n,), generator=g)
    z = h @ W.T
    (gW,) = torch.autograd.grad(torch.nn.functional.cross_entropy(z, y, reduction="sum"), W)
    sm = source_matrices_sparse(torch.ones(n, 1), y.unsqueeze(-1), z.detach(), h, {}, W.detach())
    assert torch.allclose(sm.M, -gW, atol=1e-5)


@pytest.fixture(scope="module")
def step_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("step")
    ck = save_checkpoint(BACKBONES["llama"](0), root / "student")
    t1 = save_checkpoint(BACKBONES["llama"](7), root / "teacher_math")
    t2 = save_checkpoint(BACKBONES["llama"](8), root / "teacher_code")
    return root, ck, t1, t2, write_batch_file(root / "batch.jsonl"), write_probe_file(root / "probes.jsonl")


def _cli(step_files, extra):
    root, ck, _, _, batch, probes = step_files
    return main(["step", "--checkpoint", str(ck), "--batch", str(batch), "--probes", str(probes), "--out",
                 str(root / "store"), "--answer-prefix", "\\nAnswer: ", "--label-format", "{label}", *extra])


def test_step_cli_sft(step_files, capsys):
    assert _cli(step_files, ["--objective", "sft", "--optimizer", "sgd", "--lr", "1e-2"]) == 0
    s = json.loads(capsys.readouterr().out)
    run_dir = step_files[0] / "store" / "student__sft_sgd"
    assert s["run_id"] == "student__sft_sgd" and s["sources"] == ["code", "math"]
    assert (run_dir / f"step_{s['tag']}.json").exists() and (run_dir / f"report_step_{s['tag']}.html").exists()
    pkg = json.loads((run_dir / "package_0" / "package.json").read_text(encoding="utf-8"))
    assert pkg["pre_params_ref"].endswith("student") and not (run_dir / "package_0" / "pre_params.safetensors").exists()
    assert s["consistency"]["first_order→actual:phi_ans"]["corr"] > 0.99


def test_step_cli_mopd_with_teachers(step_files, capsys):
    root, _, t1, t2, _, _ = step_files
    assert _cli(step_files, ["--objective", "mopd", "--objective-arg", "weight_mode=teacher_p", "micro_batch_size=2",
                             "--teacher", f"math={t1}", f"code={t2}", "--topk", "3", "--source-kind", "teacher"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["objective_config"]["weight_mode"] == "teacher_p" and s["objective_config"]["micro_batch_size"] == 2
    assert s["teachers"] == {"math": str(t1), "code": str(t2)} and s["source_kind"] == "teacher"
    co = Store(root / "store").read("student__mopd_sgd", "contributions").to_pylist()
    assert {r["source_kind"] for r in co} == {"teacher"} and {r["source"] for r in co} == {"math", "code"}
