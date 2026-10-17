"""Objective adapters (blueprint §五): replicated losses, loss split by source, and the sparse force
f = −∂L/∂z checked against autograd through a real forward pass."""
import pytest
import torch

from stepscope.adapters.objectives.base import TrainBatch
from stepscope.adapters.objectives.mopd import MOPDObjective, dual_clip_pg, open_mopd_advantage
from stepscope.adapters.objectives.sft import SFTObjective
from stepscope.analyze.dynamics.forces import dense_force, readout_projected_force, token_energy
from stepscope.capture.batch import load_batch_file, mopd_extras
from stepscope.testing.tiny import BACKBONES, random_train_batch


def _mopd_batch(student, seed=1, off_policy=False, seq_weight=False):
    b = mopd_extras(student, {"math": BACKBONES["llama"](7), "code": BACKBONES["llama"](8)}, random_train_batch(seed), k=4)
    if off_policy:
        g = torch.Generator().manual_seed(seed)
        lp = torch.log_softmax(student(**b.model_kwargs()).logits.detach()[:, :-1], -1).gather(-1, b.extras["topk_ids"][:, 1:])
        old = torch.zeros_like(b.extras["teacher_logp"])
        old[:, 1:] = lp + 0.3 * torch.randn(lp.shape, generator=g)
        b.extras["old_logp"] = old
    if seq_weight:
        b.extras["seq_weight"] = torch.tensor([1.0, 2.0, 0.5, 1.5])
    return b


def _autograd_force(model, obj, batch):
    logits = model(**batch.model_kwargs()).logits.detach().requires_grad_(True)
    loss = obj.loss_from_logits(logits, batch)
    (g,) = torch.autograd.grad(loss, logits)
    return -g, logits.detach(), loss.detach()


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_sft_loss_matches_hf_loss(arch):
    model = BACKBONES[arch](0)
    b = random_train_batch()
    labels = b.input_ids.masked_fill(b.loss_mask == 0, -100)
    ref = model(**b.model_kwargs(), labels=labels).loss
    assert torch.allclose(SFTObjective().loss(model, b), ref, atol=1e-6)


CASES = [
    ("sft", lambda m: SFTObjective(), lambda m: random_train_batch()),
    ("sft_seq_mean", lambda m: SFTObjective("seq_mean"), lambda m: random_train_batch()),
    ("mopd_student_p", lambda m: MOPDObjective(), lambda m: _mopd_batch(m)),
    ("mopd_teacher_p_mb2", lambda m: MOPDObjective("teacher_p", micro_batch_size=2), lambda m: _mopd_batch(m, seq_weight=True)),
    ("mopd_none_offpolicy", lambda m: MOPDObjective("none", micro_batch_size=1), lambda m: _mopd_batch(m, off_policy=True)),
]


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_sparse_force_equals_autograd(arch, case):
    _, make_obj, make_batch = case
    model = BACKBONES[arch](0)
    obj, b = make_obj(model), make_batch(model)
    f_ref, logits, loss = _autograd_force(model, obj, b)
    sf = obj.force(model, b)
    f = dense_force(sf.coef, sf.ids, torch.softmax(logits / sf.temperature, -1), sf.temperature)
    assert loss.abs() > 0 and f_ref.abs().max() > 0
    assert torch.allclose(f, f_ref, atol=1e-6), (f - f_ref).abs().max()
    # no force at unsupervised logit positions
    assert torch.all(sf.coef[~b.supervised_logit_positions()] == 0)
    parts = obj.loss_by_source(model, b)
    assert set(parts) == {"math", "code"}
    assert torch.allclose(sum(parts.values()), obj.loss(model, b), atol=1e-6)


def test_mopd_clipped_candidates_have_zero_coefficient():
    model = BACKBONES["llama"](0)
    b = _mopd_batch(model)
    lp = torch.log_softmax(model(**b.model_kwargs()).logits[:, :-1], -1).gather(-1, b.extras["topk_ids"][:, 1:])
    old = torch.zeros_like(b.extras["teacher_logp"])
    old[:, 1:] = lp - 1.0                       # ratio = e > 1 + clip_high
    b.extras.update(old_logp=old, advantages=torch.ones_like(old))
    sf = MOPDObjective(refresh=False).force(model, b)
    assert torch.all(sf.coef == 0)


def test_mopd_dual_clip_branches():
    lp = torch.zeros(1, 3)
    adv = torch.tensor([[1.0, -1.0, -1.0]])
    old = torch.tensor([[0.0, 0.0, -3.0]])      # third: ratio e^3 > clip_c with negative advantage
    pg = dual_clip_pg(lp, old, adv, 0.2, 0.28, 3.0)
    assert torch.allclose(pg, torch.tensor([-1.0 + 1.0 + 3.0]))
    a = open_mopd_advantage(torch.log(torch.tensor([[0.5, 0.5]])), torch.log(torch.tensor([[0.25, 0.75]])), "none")
    assert torch.allclose(a, -0.5 * torch.log(torch.tensor([[2.0, 2 / 3]])))


def test_energy_and_readout_projection_from_sparse_form():
    model = BACKBONES["llama"](0)
    b = _mopd_batch(model)
    sf = MOPDObjective().force(model, b)
    p = torch.softmax(model(**b.model_kwargs()).logits.detach(), -1)
    f = dense_force(sf.coef, sf.ids, p)
    assert torch.allclose(token_energy(sf.coef, sf.ids, p), f.pow(2).sum(-1), atol=1e-7)
    W = model.lm_head.weight
    assert torch.allclose(readout_projected_force(sf.coef, sf.ids, p, W), f @ W, atol=1e-6)


def test_batch_file_text_and_ids(tmp_path, tokenizer):
    f = tmp_path / "batch.jsonl"
    f.write_text('{"source": "math", "prompt": "2+2=", "response": "4"}\n'
                 '{"source": "code", "input_ids": [5, 6, 7, 8, 9], "prompt_len": 3}\n', encoding="utf-8")
    b = load_batch_file(f, tokenizer)
    assert isinstance(b, TrainBatch) and b.sources == ["math", "code"] and b.input_ids.shape == (2, 5)
    assert b.loss_mask.tolist() == [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]
    assert b.attention_mask.tolist() == [[1] * 5, [1] * 5]


def test_matches_verl_policy_loss_when_available():
    core = pytest.importorskip("verl.trainer.ppo.core_algos")
    from omegaconf import OmegaConf

    g = torch.Generator().manual_seed(0)
    B, T, K = 3, 6, 4
    lp = torch.log_softmax(torch.randn(B, T, 30, generator=g), -1)[..., :K]
    old = lp + 0.1 * torch.randn(B, T, K, generator=g)
    adv = open_mopd_advantage(lp, lp + torch.randn(B, T, K, generator=g))
    mask = torch.ones(B, T)
    cfg = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.28, "clip_ratio_c": 3.0})
    ref, _ = core.compute_policy_loss_vanilla(old, lp, adv, mask, loss_agg_mode="token-mean", config=cfg)
    pg = dual_clip_pg(lp, old, adv, 0.2, 0.28, 3.0)
    assert torch.allclose(ref, (pg * mask).sum() / mask.sum(), atol=1e-5)
