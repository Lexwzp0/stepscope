"""verl output directory → Run / IntervalRecords (blueprint §十四.6).

Layout written by verl's trainer (``ray_trainer._save_checkpoint`` + ``FSDPCheckpointManager``)::

    <default_local_dir>/
      latest_checkpointed_iteration.txt
      global_step_<N>/
        data.pt                       dataloader state
        actor/
          model_world_size_<W>_rank_<r>.pt   FSDP shards (always)
          optim_..., extra_state_...
          huggingface/                config + tokenizer (always); model weights only if
                                      checkpoint.save_contents includes hf_model

A checkpoint is usable by chain one only when ``actor/huggingface/`` holds HF weights. Otherwise it is
listed as pending with the merge command::

    python -m verl.model_merger merge --backend fsdp --local_dir <global_step_N>/actor \
        --target_dir <global_step_N>/actor/huggingface

Several optimizer steps happen inside one global step (ppo_epochs × mini-batches), so consecutive
global steps are IntervalRecords, never single-step update packages. Source logs are not read yet.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ...core.model import Checkpoint, IntervalRecord, Run
from ...core.registry import FRAMEWORKS

_STEP_DIR = re.compile(r"^global_step_(\d+)$")
WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")


@dataclass
class VerlCheckpoint:
    step: int
    actor_dir: str
    hf_dir: str
    has_weights: bool
    n_shards: int
    merge_cmd: str | None


def merge_command(actor_dir: str | Path) -> str:
    actor_dir = Path(actor_dir)
    return (f"python -m verl.model_merger merge --backend fsdp --local_dir {actor_dir} "
            f"--target_dir {actor_dir / 'huggingface'}")


def scan_verl_run(root: str | Path) -> list[VerlCheckpoint]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    out = []
    for d in sorted(root.iterdir()):
        m = _STEP_DIR.match(d.name)
        if not m or not d.is_dir():
            continue
        actor = d / "actor"
        hf = actor / "huggingface"
        has_w = hf.is_dir() and any(next(hf.glob(p), None) is not None for p in WEIGHT_PATTERNS)
        shards = len(list(actor.glob("model_world_size_*_rank_*.pt"))) if actor.is_dir() else 0
        out.append(VerlCheckpoint(step=int(m.group(1)), actor_dir=str(actor), hf_dir=str(hf), has_weights=bool(has_w),
                                  n_shards=shards, merge_cmd=None if has_w else (merge_command(actor) if shards else None)))
    out.sort(key=lambda c: c.step)
    if not out:
        raise ValueError(f"no global_step_* directories under {root}")
    return out


def run_from_verl(root: str | Path, run_id: str | None = None) -> Run:
    root = Path(root)
    cks = scan_verl_run(root)
    latest = root / "latest_checkpointed_iteration.txt"
    latest_step = latest.read_text(encoding="utf-8").strip() if latest.exists() else ""
    cfg_hash = hashlib.sha1(("verl:" + str(root.resolve()) + ":" + ",".join(str(c.step) for c in cks)).encode()).hexdigest()[:12]
    rid = run_id or (root.name or "verl_run")
    run = Run(id=rid, framework="verl", config_hash=cfg_hash, code_version="",
              checkpoints=[Checkpoint(rid, c.step, c.hf_dir) for c in cks if c.has_weights],
              source_log_path=None)
    run.meta = {
        "root": str(root), "latest_checkpointed_iteration": latest_step,
        "scanned": [asdict(c) for c in cks],
        "pending": [{"step": c.step, "merge_cmd": c.merge_cmd} for c in cks if not c.has_weights],
    }
    return run


@FRAMEWORKS.register("verl")
class VERLReader:
    name = "verl"

    def read_run(self, path: str, run_id: str | None = None) -> Run:
        return run_from_verl(path, run_id)

    def intervals(self, run: Run) -> Iterable[IntervalRecord]:
        cks = sorted(run.checkpoints, key=lambda c: c.step)
        for a, b in zip(cks, cks[1:]):
            yield IntervalRecord(run.id, a, b, run.source_log_path)

    def update_packages(self, run: Run):
        return iter(())   # capture of single optimizer steps is not implemented (blueprint §十二 不做)
