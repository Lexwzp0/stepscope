"""Numerical validation of chain one on a tiny model with one real update (blueprint §十四.0 acceptance)."""
import copy

import pytest
import torch

from stepscope.analyze.dynamics import interval as iv
from stepscope.observe.probes import load_probe_file
from stepscope.observe.scoring import encode_probes
from tests.conftest import RENDER, one_adamw_step, one_sgd_step

NAMES = ("phi_fmt", "phi_ans")


def _setup(tokenizer, probe_file, tiny_model, update, lr):
    ps = load_probe_file(probe_file)
    enc = encode_probes(tokenizer, ps.items, RENDER)
    m_from = tiny_model
    m_to = update(m_from, lr)
    s_from = iv.observe(m_from, enc, NAMES)
    s_to = iv.observe(m_to, enc, NAMES)
    delta, info = iv.delta_params(m_from, m_to)
    assert info.norm > 0
    return enc, m_from, m_to, s_from, s_to, delta


def _check(actual, pred, tol_rel=0.05, tol_abs=1e-6):
    err = (actual - pred).abs().max().item()
    scale = actual.abs().max().item()
    assert scale > 1e-5, "update too small to test anything"
    assert err <= tol_rel * scale + tol_abs, f"first-order error {err:.3e} vs max actual {scale:.3e}"
    return err, scale


@pytest.mark.parametrize("update,lr", [(one_sgd_step, 1e-2), (one_adamw_step, 1e-4)])
def test_first_order_matches_one_real_update(tokenizer, probe_file, tiny_model, update, lr):
    enc, m_from, m_to, s_from, s_to, delta = _setup(tokenizer, probe_file, tiny_model, update, lr)
    actual = s_to["phi"] - s_from["phi"]
    phi_j, d_jvp = iv.first_order_jvp(m_from, delta, enc, NAMES)
    phi_b, d_bwd = iv.first_order_backward(m_from, delta, enc, NAMES)
    assert torch.allclose(phi_j, s_from["phi"], atol=1e-5)
    assert torch.allclose(phi_b, s_from["phi"], atol=1e-5)
    assert (d_jvp - d_bwd).abs().max().item() <= 1e-5 * max(1.0, d_jvp.abs().max().item())
    _check(actual, d_jvp)
    _check(actual, d_bwd)
    for o in range(len(NAMES)):   # per-observable correlation
        a, p = actual[o], d_jvp[o]
        corr = torch.corrcoef(torch.stack([a, p]))[0, 1].item()
        assert corr > 0.99


def test_zero_update_gives_zero(tokenizer, probe_file, tiny_model):
    ps = load_probe_file(probe_file)
    enc = encode_probes(tokenizer, ps.items, RENDER)
    m_to = copy.deepcopy(tiny_model)
    delta, info = iv.delta_params(tiny_model, m_to)
    assert info.norm == 0.0
    s = iv.observe(tiny_model, enc, NAMES)
    _, d = iv.first_order_jvp(tiny_model, delta, enc, NAMES)
    assert torch.all(d == 0) and torch.allclose(iv.observe(m_to, enc, NAMES)["phi"], s["phi"])


def test_error_shrinks_quadratically_along_fixed_delta(tokenizer, probe_file, tiny_model):
    """Scale the same Δθ by s ∈ {1, 1/2, 1/4}: the first-order prediction scales linearly,
    the error should fall roughly as s² (we require < 0.6× per halving)."""
    enc, m_from, m_to, s_from, _, delta = _setup(tokenizer, probe_file, tiny_model, one_sgd_step, 3e-2)
    _, d1 = iv.first_order_jvp(m_from, delta, enc, NAMES)
    errs = []
    for s in (1.0, 0.5, 0.25):
        m_s = copy.deepcopy(m_from)
        with torch.no_grad():
            for k, p in m_s.named_parameters():
                p += s * delta[k].to(p.dtype)
        actual_s = iv.observe(m_s, enc, NAMES)["phi"] - s_from["phi"]
        errs.append((actual_s - s * d1).abs().max().item())
    assert errs[0] > 0
    assert errs[1] < 0.6 * errs[0], errs
    assert errs[2] < 0.6 * errs[1], errs


def test_auto_method_reports_what_it_used(tokenizer, probe_file, tiny_model):
    enc, m_from, _, _, _, delta = _setup(tokenizer, probe_file, tiny_model, one_sgd_step, 1e-2)
    _, _, method = iv.first_order(m_from, delta, enc, NAMES, method="auto")
    assert method in ("jvp",) or method.startswith("backward")
