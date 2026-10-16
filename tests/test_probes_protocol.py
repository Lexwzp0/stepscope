"""Probe conversion from ood_ld build_mcq rows; protocol loading; chat-template rendering options; subset selection."""
import json

import pytest

from stepscope.cli.main import main
from stepscope.observe.convert import from_mcq_rows
from stepscope.observe.probes import RenderConfig, load_probe_file, render_item, select_items

MCQ_ROWS = [
    {"id": "gpqa_7#r0", "item_id": "gpqa_7", "source": "gpqa_diamond", "question": "Q7", "options": ["a", "b", "c", "d"],
     "answer_idx": 2, "rotation": 0, "split": "dev"},
    {"id": "gpqa_7#r1", "item_id": "gpqa_7", "source": "gpqa_diamond", "question": "Q7", "options": ["d", "a", "b", "c"],
     "answer_idx": 3, "rotation": 1, "split": "dev"},
    {"id": "mmlu_3#r0", "item_id": "mmlu_3", "source": "mmlu_redux", "question": "Q3", "options": ["x", "y", "z", "w"],
     "answer_idx": 0, "rotation": 0, "split": "test"},
]


def test_convert_keeps_order_and_answer(tmp_path):
    rows = from_mcq_rows(MCQ_ROWS)
    assert [r["domain"] for r in rows] == ["gpqa_diamond", "gpqa_diamond", "mmlu_redux"]
    assert rows[1]["options"] == ["d", "a", "b", "c"] and rows[1]["answer_idx"] == 3 and rows[1]["rotation"] == 1
    assert from_mcq_rows(MCQ_ROWS, split="dev")[-1]["id"] == "gpqa_7#r1"
    assert [r["id"] for r in from_mcq_rows(MCQ_ROWS, rotations="first")] == ["gpqa_7#r0", "mmlu_3#r0"]
    src = tmp_path / "mcq.jsonl"; src.write_text("\n".join(json.dumps(r) for r in MCQ_ROWS) + "\n")
    out = tmp_path / "probes.jsonl"
    assert main(["probes", "convert", str(src), "--out", str(out), "--split", "dev"]) == 0
    ps = load_probe_file(out)
    assert len(ps.items) == 2 and ps.items[0].meta["item_id"] == "gpqa_7" and ps.items[0].meta["split"] == "dev"


def test_select_items_is_deterministic(tmp_path):
    src = tmp_path / "p.jsonl"
    src.write_text("\n".join(json.dumps({"id": f"i{k}", "domain": "d", "question": "q", "options": ["a", "b"], "answer_idx": 0})
                             for k in range(20)) + "\n")
    ps = load_probe_file(src)
    a = select_items(ps, 5); b = select_items(ps, 5)
    assert [i.id for i in a.items] == [i.id for i in b.items] and len(a.items) == 5
    assert a.version.endswith("-n5") and select_items(ps, 50) is ps


class _ChatTok:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None, strftime_now=None):
        self.calls.append(dict(enable_thinking=enable_thinking, has_date=strftime_now is not None))
        date = strftime_now("%d %B %Y") if strftime_now else "today"
        sys_ = f"<sys date={date}>"
        user = messages[-1]["content"]
        return f"{sys_}<user>{user}</user><assistant>"


def test_protocol_loading_and_render(tmp_path):
    proto = {"direct": {"enable_thinking": False, "answer_prefix": "\nAnswer:", "label_format": " {}",
                        "labels": ["A", "B", "C", "D"], "no_think_suffix": "<think>\n\n</think>\n", "fixed_date": "2026-10-16"}}
    p = tmp_path / "template.json"; p.write_text(json.dumps(proto))
    cfg = RenderConfig.from_protocol(p)
    assert cfg.chat_template and cfg.enable_thinking is False and cfg.fixed_date == "2026-10-16"
    assert cfg.label_tokens(4) == [" A", " B", " C", " D"]          # positional "{}" format supported
    ps = load_probe_file(_write_one(tmp_path))
    tok = _ChatTok()
    text = render_item(ps.items[0], cfg, tok)
    assert text.endswith("<assistant><think>\n\n</think>\n\nAnswer:")
    assert "date=16 October 2026" in text and tok.calls[0] == {"enable_thinking": False, "has_date": True}
    cfg2 = RenderConfig.from_protocol(p, answer_prefix="\nFinal:")
    assert cfg2.answer_prefix == "\nFinal:" and cfg2.no_think_suffix == cfg.no_think_suffix


def _write_one(tmp_path):
    f = tmp_path / "one.jsonl"
    f.write_text(json.dumps({"id": "x", "domain": "d", "question": "Q?", "options": ["1", "2", "3", "4"], "answer_idx": 1}) + "\n")
    return f
