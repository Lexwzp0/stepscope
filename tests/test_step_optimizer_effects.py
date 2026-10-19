"""Regression checks for restored optimizer settings and the loss-only CH2 reference."""
import copy
import math

import pytest
import torch

from stepscope.adapters.models.hf import HFModelAdapter
from stepscope.adapters.objectives.sft import SFTObjective
from stepscope.analyze.dynamics.interval import first_order_jvp
from stepscope.analyze.dynamics.single_step import run_chain
from stepscope.capture.package import PackageWriter, load_package
from stepscope.capture.train_step import OptimizerConfig, load_params, take_step
from stepscope.observe.probes import load_probe_file
from stepscope.observe.scoring import encode_probes
from stepscope.testing.tiny import BACKBONES, RENDER, random_train_batch

NAMES = ("phi_fmt", "phi_ans")


@pytest.mark.parametrize("name", ["sgd", "adamw"])
def test_restored_settings_are_recorded_and_replayed(name, tokenizer, probe_file, tmp_path):
    model = BACKBONES["llama"](0)
    state_config = OptimizerConfig(name, lr=2e-3, weight_decay=0.05, betas=(0.8, 0.95), eps=1e-6)
    optim = state_config.build(model.parameters())
    # Initialize real Adam moments, rather than testing only an empty state dictionary.
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.01)
    optim.step()
    optim.zero_grad(set_to_none=True)
    state = copy.deepcopy(optim.state_dict())
    requested = OptimizerConfig(name, lr=0.1, weight_decay=0.9, betas=(0.6, 0.7), eps=1e-3,
                                max_grad_norm=0.2)
    obj = SFTObjective()
    pkg = take_step(model, obj, random_train_batch(), requested, optimizer_state=state)

    assert pkg.optimizer.lr == state_config.lr
    assert pkg.optimizer.weight_decay == state_config.weight_decay
    assert pkg.optimizer.max_grad_norm == requested.max_grad_norm
    if name == "adamw":
        assert pkg.optimizer.betas == state_config.betas and pkg.optimizer.eps == state_config.eps
    assert requested.lr == 0.1 and requested.weight_decay == 0.9
    requested.lr = 0.5
    assert pkg.optimizer.lr == state_config.lr
    ref = PackageWriter(tmp_path / "package").write(pkg)
    assert ref.config["optimizer"]["lr"] == state["param_groups"][0]["lr"]
    loaded = load_package(tmp_path / "package")
    load_params(model, loaded.pre_params)
    replayed = take_step(model, obj, loaded.batch, loaded.optimizer, optimizer_state=loaded.optimizer_state_pre)
    assert all(torch.equal(replayed.delta[k], pkg.delta[k]) for k in pkg.delta)

    if name == "sgd":
        enc = encode_probes(tokenizer, load_probe_file(probe_file).items, RENDER)
        result = run_chain(HFModelAdapter(model, tokenizer), obj, loaded, enc, NAMES)
        assert result.diagnostics["reconstruction"]["passed"]
        assert result.diagnostics["readout_channel_check"]["rel_err"] < 1e-4


@pytest.mark.parametrize("by_source", [True, False])
def test_ch2_reference_excludes_decay_but_full_prediction_keeps_it(by_source, tokenizer, probe_file):
    model = BACKBONES["llama"](0)
    adapter = HFModelAdapter(model, tokenizer)
    obj, batch = SFTObjective(), random_train_batch()
    opt = OptimizerConfig("sgd", lr=1e-2, weight_decay=0.4)
    pkg = take_step(model, obj, batch, opt)
    enc = encode_probes(tokenizer, load_probe_file(probe_file).items, RENDER)
    result = run_chain(adapter, obj, pkg, enc, NAMES, by_source=by_source)
    assert result.diagnostics["reconstruction"]["passed"]

    # Independently form the loss-only backbone tangent directly from autograd.
    params = dict(model.named_parameters())
    for p in params.values():
        p.requires_grad_(True)
    loss = obj.terms_from_logits(model(**batch.model_kwargs()).logits, batch).sum()
    grads = torch.autograd.grad(loss, list(params.values()))
    loss_tangent = {k: -opt.lr * g.detach() for k, g in zip(params, grads)}
    for p in params.values():
        p.requires_grad_(False)
    ro = adapter.feature_sites()["readout"] + ".weight"
    _, expected_backbone = first_order_jvp(model, {k: v for k, v in loss_tangent.items() if k != ro}, enc, NAMES)
    _, loss_effect = first_order_jvp(model, loss_tangent, enc, NAMES)
    _, decay_effect = first_order_jvp(
        model, {k: -opt.lr * opt.weight_decay * p.detach() for k, p in params.items()}, enc, NAMES)
    assert float(decay_effect.abs().max()) > 1e-5
    assert torch.allclose(result.first_order, loss_effect + decay_effect, atol=1e-7, rtol=1e-5)
    assert torch.allclose(result.first_order_loss, loss_effect, atol=1e-7, rtol=1e-5)
    for i, observable in enumerate(NAMES):
        actual_rms = result.diagnostics["ch2_vs_backbone_first_order"][observable]["rms_backbone_first_order"]
        expected_rms = float(expected_backbone[i].square().mean().sqrt())
        assert math.isclose(actual_rms, expected_rms, rel_tol=1e-5, abs_tol=1e-7)
    if by_source:
        assert torch.allclose(result.contributions["weight_decay"], decay_effect, atol=1e-8)
        assert torch.allclose(sum(result.contributions.values()), result.first_order, atol=1e-7, rtol=1e-5)
    else:
        assert result.contributions == {}


def test_tied_readout_explains_missing_exact_diagnostics_without_skipping_channels(tokenizer, probe_file):
    model = BACKBONES["llama"](0)
    model.config.tie_word_embeddings = True
    model.tie_weights()
    adapter = HFModelAdapter(model, tokenizer)
    assert adapter.readout_tied
    obj = SFTObjective()
    pkg = take_step(model, obj, random_train_batch(), OptimizerConfig("sgd", lr=1e-3))
    enc = encode_probes(tokenizer, load_probe_file(probe_file).items, RENDER)
    result = run_chain(adapter, obj, pkg, enc, NAMES)

    assert result.diagnostics["readout_tied"] is True
    assert result.diagnostics["reconstruction"]["passed"]
    assert result.channels is not None and torch.isfinite(result.channels).all()
    assert set(result.contributions) == {"math", "code"}
    for name in ("readout_channel_check", "ch2_vs_backbone_first_order"):
        assert name not in result.diagnostics
        assert "tied input/readout weights" in " ".join(result.skipped[f"diagnostics.{name}"])
    assert "dynamics.ldcl_channels" not in result.skipped
