"""Explicit capability declarations (blueprint §五): supports() reads the declaration only; the HF adapter
declares what it implements per architecture family; analyses with unmet requirements are skipped and the
reason lands in the manifest."""
import pytest
import torch

from stepscope.adapters.models import hf
from stepscope.adapters.models.base import CAPABILITIES, missing_capabilities, supports
from stepscope.adapters.models.hf import HFModelAdapter
from stepscope.analyze.base import run_analysis, unmet_requirements
from stepscope.core.store import Store
from stepscope.testing.tiny import BACKBONES


class _Undeclared:
    """Has callable optional methods but declares nothing: supports() must say no."""
    name = "fake"
    capabilities = frozenset()

    def readout(self):
        return torch.zeros(3, 2)

    def residual_inputs(self, batch):
        raise NotImplementedError


def test_supports_reads_declaration_only():
    a = _Undeclared()
    assert not supports(a, "readout") and not supports(a, "residual_inputs")
    a.capabilities = frozenset({"readout"})
    assert supports(a, "readout") and not supports(a, "residual_inputs")
    assert missing_capabilities(a, {"readout", "residual_inputs", "probes"}) == ["residual_inputs"]
    with pytest.raises(ValueError):
        supports(a, "teleport")


def _batch(B=2, T=9):
    ids = torch.randint(3, 100, (B, T), generator=torch.Generator().manual_seed(0))
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)


@pytest.mark.parametrize("arch,norm_names", [
    ("llama", ("model.layers.{i}.input_layernorm", "model.layers.{i}.post_attention_layernorm", "model.norm")),
    ("gpt2", ("transformer.h.{i}.ln_1", "transformer.h.{i}.ln_2", "transformer.ln_f")),
])
def test_hf_adapter_declares_and_implements(arch, norm_names):
    model = BACKBONES[arch](0)
    ad = HFModelAdapter(model)
    assert ad.capabilities == CAPABILITIES
    sites = ad.feature_sites()
    modules = dict(model.named_modules())
    assert set(sites) == {"layers.0.attn_in", "layers.0.mlp_in", "layers.1.attn_in", "layers.1.mlp_in",
                          "final_norm", "readout"}
    assert all(name in modules for name in sites.values())
    assert sites["layers.1.attn_in"] == norm_names[0].format(i=1)
    assert sites["layers.0.mlp_in"] == norm_names[1].format(i=0) and sites["final_norm"] == norm_names[2]
    assert set(ad.param_groups()) >= {"layers.0", "layers.1"} and not ad.readout_tied

    batch = _batch()
    ri = ad.residual_inputs(batch)
    W = ad.readout()
    assert W.shape == (128, 32) and list(ri.branches) == ad.branch_sites()
    assert all(h.shape == (2, 9, 32) for h in ri.branches.values()) and ri.final.shape == (2, 9, 32)
    # the recorded readout input reproduces the logits exactly: z = W hL
    assert torch.allclose(ri.final @ W.T, ri.logits, atol=1e-5)
    # the first branch input is the first block's norm applied to the embedding stream
    hs = ad.hidden_states(batch)
    first_norm = modules[norm_names[0].format(i=0)]
    with torch.no_grad():
        assert torch.allclose(first_norm(hs[0]), ri.branches["layers.0.attn_in"], atol=1e-5)


def test_unknown_family_declares_no_residual_inputs(monkeypatch):
    monkeypatch.delitem(hf.SITE_SPECS, "llama")
    ad = HFModelAdapter(BACKBONES["llama"](0))
    assert ad.capabilities == frozenset({"hidden_states", "readout"})
    assert not supports(ad, "residual_inputs") and "final_norm" not in ad.feature_sites()
    with pytest.raises(NotImplementedError):
        ad.residual_inputs(_batch())
    # the same model with an explicit site spec (configuration, not new code) declares it again
    assert supports(HFModelAdapter(BACKBONES["llama"](0), sites=hf.LLAMA_LIKE), "residual_inputs")


class _NeedsChannels:
    name = "dynamics.needs_channels"
    version = "0.0.1"
    level = "channel"
    requires = {"update_package", "probes", "readout", "residual_inputs"}
    writes = {"channels"}

    def __init__(self):
        self.ran = False

    def run(self, inputs, probes, store, run_id):
        self.ran = True


def test_analysis_skips_with_reason_in_manifest(tmp_path, monkeypatch):
    st = Store(tmp_path)
    an = _NeedsChannels()
    monkeypatch.delitem(hf.SITE_SPECS, "llama")
    ad = HFModelAdapter(BACKBONES["llama"](0))
    assert not run_analysis(an, {}, [], st, "r", available_inputs={"probes"}, adapter=ad)
    assert not an.ran
    entry = st.manifest("r")["analyses"]["dynamics.needs_channels"]
    assert entry["status"] == "skipped"
    assert any("update_package" in r for r in entry["reasons"])
    assert any("residual_inputs" in r and "llama" in r for r in entry["reasons"])
    assert not any("'readout'" in r for r in entry["reasons"])
    monkeypatch.undo()
    ok = HFModelAdapter(BACKBONES["llama"](0))
    assert run_analysis(an, {}, [], st, "r", available_inputs={"update_package", "probes"}, adapter=ok, context="0_1")
    assert an.ran and st.manifest("r")["analyses"]["dynamics.needs_channels@0_1"]["status"] == "ran"
    an.requires = {"probes", "telepathy"}
    with pytest.raises(ValueError, match="telepathy"):
        unmet_requirements(an, {"probes"}, ok)
