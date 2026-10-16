"""Convert other item formats into StepScope probe files.

``from_mcq_rows``: rows written by ood_ld ``data/build_mcq.py`` (``id, item_id, source, question,
options, answer_idx, rotation, split``) → probe rows. ``source`` becomes ``domain``; ``item_id``,
``rotation`` and ``split`` are kept; option order and ``answer_idx`` are untouched."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def from_mcq_rows(rows: Iterable[dict], split: str | None = None, rotations: str = "all") -> list[dict]:
    if rotations not in ("all", "first"):
        raise ValueError("rotations must be 'all' or 'first'")
    out = []
    for r in rows:
        if split is not None and r.get("split") != split:
            continue
        if rotations == "first" and int(r.get("rotation", 0)) != 0:
            continue
        domain = r.get("source") or r.get("domain")
        if not domain:
            raise ValueError(f"row {r.get('id')!r}: neither 'source' nor 'domain'")
        out.append({
            "id": str(r["id"]), "domain": str(domain), "question": str(r["question"]),
            "options": [str(o) for o in r["options"]], "answer_idx": int(r["answer_idx"]),
            "item_id": r.get("item_id"), "rotation": r.get("rotation"), "split": r.get("split"),
        })
    return out


def convert_file(src: str | Path, dst: str | Path, split: str | None = None, rotations: str = "all") -> int:
    rows = [json.loads(l) for l in Path(src).read_text(encoding="utf-8").splitlines() if l.strip()]
    probes = from_mcq_rows(rows, split=split, rotations=rotations)
    if not probes:
        raise ValueError(f"{src}: no rows selected (split={split}, rotations={rotations})")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in probes), encoding="utf-8")
    return len(probes)
