import math

import pytest
import torch

from stepscope.observe.observables import decompose, option_logprobs, phi_ans, phi_fmt
from stepscope.observe.probes import RenderConfig, load_probe_file, render_item
from stepscope.observe.scoring import encode_probes
from tests.conftest import RENDER


def test_phi_formulas_match_manual():
    torch.manual_seed(0)
    logits = torch.randn(3, 50)
    opt_ids = torch.tensor([[5, 9, 12, 20]] * 3)
    correct = torch.tensor([0, 2, 3])
    opt = option_logprobs(logits, opt_ids)
    logp = torch.log_softmax(logits, -1)
    for i in range(3):
        ids = opt_ids[i].tolist()
        expect_fmt = math.log(sum(math.exp(logp[i, j]) for j in ids))
        c = ids[correct[i]]
        expect_ans = float(logp[i, c]) - math.log(sum(math.exp(logp[i, j]) for j in ids if j != c))
        assert abs(float(phi_fmt(opt)[i]) - expect_fmt) < 1e-5
        assert abs(float(phi_ans(opt, correct)[i]) - expect_ans) < 1e-5
    d = decompose(opt, correct)
    assert torch.allclose(d["p_correct"], d["p_valid"] * d["p_correct_given_valid"], atol=1e-6)


def test_option_ids_per_row_vs_shared():
    logits = torch.randn(2, 30)
    shared = option_logprobs(logits, [1, 2, 3])
    per_row = option_logprobs(logits, torch.tensor([[1, 2, 3], [1, 2, 3]]))
    assert torch.allclose(shared, per_row)


def test_encode_probes_resolves_labels_on_full_prompt(tokenizer, probe_file):
    ps = load_probe_file(probe_file)
    enc = encode_probes(tokenizer, ps.items, RENDER)
    assert len(enc) == 8 and enc.option_ids.shape == (8, 4)
    # char-level tokenizer: label token ids are the characters A..D, identical across prompts
    assert enc.option_ids.unique(dim=0).shape[0] == 1
    assert render_item(ps.items[0], RENDER).endswith("\nAnswer: ")
    assert enc.correct_idx.tolist() == [p.answer_idx for p in ps.items]


def test_encode_probes_rejects_merged_label(tokenizer, probe_file):
    """A two-character label is two tokens for the char tokenizer -> must raise (BPE merges would look the same)."""
    ps = load_probe_file(probe_file)
    with pytest.raises(ValueError, match="single clean token"):
        encode_probes(tokenizer, ps.items, RenderConfig(answer_prefix="\nAnswer:", label_format=" {label}"))


class _MergingTokenizer:
    """Fake tokenizer where ': A' merges into one token: the prefix-only check would pass, the full-prompt check must fail."""
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            if text.startswith(": A", i):
                ids.append(999); i += 3
            else:
                ids.append(ord(text[i]) % 500); i += 1
        return {"input_ids": ids}


def test_full_prompt_check_catches_merge(probe_file):
    ps = load_probe_file(probe_file)
    cfg = RenderConfig(answer_prefix="\nAnswer:", label_format=" {label}")
    with pytest.raises(ValueError, match="prefix preserved=False"):
        encode_probes(_MergingTokenizer(), ps.items, cfg)
