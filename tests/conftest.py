"""Test fixtures. The offline helpers live in ``stepscope.testing.tiny`` so the demo can reuse them."""
from __future__ import annotations

import pytest

from stepscope.testing.tiny import (  # noqa: F401 - re-exported for the test modules
    CHARS,
    PROBES,
    RENDER,
    build_char_tokenizer,
    build_tiny_model,
    make_checkpoint_pair,
    one_adamw_step,
    one_sgd_step,
    random_batch,
    write_probe_file,
)


@pytest.fixture(scope="session")
def tok_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    build_char_tokenizer(d)
    return d


@pytest.fixture(scope="session")
def tokenizer(tok_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(tok_dir))


@pytest.fixture(scope="session")
def tiny_model():
    return build_tiny_model(0)


@pytest.fixture(scope="session")
def probe_file(tmp_path_factory):
    return write_probe_file(tmp_path_factory.mktemp("probes") / "tiny_probes.jsonl")
