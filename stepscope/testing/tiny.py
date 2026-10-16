"""Tiny offline models and data for tests and the demo (no downloads, CPU only)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from ..observe.probes import RenderConfig

CHARS = [chr(i) for i in range(32, 127)] + ["\n"]
# Character-level tokenizer: a label is one token, so no leading space in label_format.
RENDER = RenderConfig(answer_prefix="\nAnswer: ", label_format="{label}")


def build_char_tokenizer(save_dir: str | Path):
    """Word-level tokenizer over single characters, saved as a loadable HF tokenizer."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2}
    for c in CHARS:
        vocab[c] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("[\\s\\S]"), behavior="isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(save_dir))
    return fast


def build_tiny_model(seed: int = 0):
    """Random Llama: vocab 128, hidden 32, 2 layers, eager attention, float32, no tied embeddings."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=False, attn_implementation="eager")
    torch.manual_seed(seed)
    m = LlamaForCausalLM(cfg).float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def random_batch(seed: int = 1, B: int = 4, T: int = 24):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 100, (B, T), generator=g)


def one_sgd_step(model, lr: float, seed: int = 1):
    """θ_to = θ_from − lr ∇L on one random LM batch. Returns a new model."""
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(True)
    ids = random_batch(seed)
    loss = m(input_ids=ids, labels=ids).loss
    grads = torch.autograd.grad(loss, list(m.parameters()))
    with torch.no_grad():
        for p, g in zip(m.parameters(), grads):
            p -= lr * g
            p.requires_grad_(False)
    return m.eval()


def one_adamw_step(model, lr: float, seed: int = 1):
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    ids = random_batch(seed)
    loss = m(input_ids=ids, labels=ids).loss
    loss.backward()
    opt.step()
    for p in m.parameters():
        p.requires_grad_(False)
        p.grad = None
    return m.eval()


PROBES = [
    {"id": "m1", "domain": "math", "question": "What is 2+2?", "options": ["3", "4", "5", "6"], "answer_idx": 1},
    {"id": "m2", "domain": "math", "question": "What is 3*3?", "options": ["6", "8", "9", "12"], "answer_idx": 2},
    {"id": "m3", "domain": "math", "question": "What is 10-7?", "options": ["3", "4", "2", "1"], "answer_idx": 0},
    {"id": "m4", "domain": "math", "question": "What is 8/2?", "options": ["2", "6", "3", "4"], "answer_idx": 3},
    {"id": "s1", "domain": "science", "question": "Water boils at?", "options": ["50C", "100C", "150C", "0C"], "answer_idx": 1},
    {"id": "s2", "domain": "science", "question": "Earth orbits?", "options": ["Mars", "Moon", "Sun", "Venus"], "answer_idx": 2},
    {"id": "s3", "domain": "science", "question": "H2O is?", "options": ["water", "salt", "air", "fire"], "answer_idx": 0},
    {"id": "s4", "domain": "science", "question": "Light speed is?", "options": ["slow", "medium", "zero", "fast"], "answer_idx": 3},
]


def write_probe_file(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(p) for p in PROBES) + "\n", encoding="utf-8")
    return path


def make_checkpoint_pair(root: str | Path, lr: float = 1e-2, steps=(0, 10), run_name: str = "runA"):
    """Save θ_from and θ_from − lr∇L as two HF checkpoints laid out like a verl export:
    <root>/<run_name>/global_step_<N>/actor/huggingface. Returns (path_from, path_to, probe_file)."""
    root = Path(root) / run_name
    m0 = build_tiny_model(0)
    m1 = one_sgd_step(m0, lr)
    paths = []
    for step, m in zip(steps, (m0, m1)):
        hf = root / f"global_step_{step}" / "actor" / "huggingface"
        m.save_pretrained(str(hf))
        build_char_tokenizer(hf)
        paths.append(hf)
    probes = write_probe_file(root / "probes.jsonl")
    return paths[0], paths[1], probes
