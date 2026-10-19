"""Prepare a small, pinned Open-MOPD example without private server paths.

CPU: python examples/prepare_real_case.py --download-only
GPU: python examples/prepare_real_case.py --generate

The output directory contains downloaded data and must stay outside version control.
No GPU work is performed by --download-only. No network access is needed by --generate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

MODEL_REVISIONS = {
    "MixSFT": "c9e7bad031667828656ead188d7e8ea162c048a4",
    "RL-Math": "5e901bb626b69d711074d2832c41ce1aa4232da8",
    "RL-Code": "e2ce9beec52381350edcdaefebfd08ea67c21b42",
    "RL-IF": "8948051a805883e2db988cae4522de3c29d1d112",
}
MODEL_PREFIX = "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-"
TRAIN_REPO = "BytedTsinghua-SIA/Open-MOPD-Data"
TRAIN_REVISION = "9e897efe3257599d4300e2d5ee865a1cc714af87"
TRAIN_SHA256 = "5243f3e61e7c0841a102389e509bd0822405c27f949e4d04cb7dd1615d4e5b2e"
PROBE_REPO = "edinburgh-dawg/mmlu-redux-2.0"
PROBE_REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"
SUBJECTS = ("abstract_algebra", "college_computer_science", "high_school_physics", "philosophy")
MODEL_FILES = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
               "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    temp = path.with_name(path.name + ".partial")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dev_item(item_id: str) -> bool:
    """Same 40% item-level dev split as the original OOD-LD input builder."""
    return int(hashlib.sha1(f"ood_ld:{item_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.4


def probe_rows(records: list[dict], subject: str, count: int) -> list[dict]:
    candidates = []
    for index, record in enumerate(records):
        item_id = f"mmlu_redux/{subject}/{index}"
        options = list(record["choices"])
        if (record.get("error_type") != "ok" or len(options) != 4
                or len(set(map(str, options))) != 4 or not dev_item(item_id)):
            continue
        candidates.append((item_id, index, record))
    candidates.sort(key=lambda row: hashlib.sha1(f"ood_ld_select:{row[0]}".encode()).hexdigest())
    if len(candidates) < count:
        raise ValueError(f"{subject}: only {len(candidates)} valid dev items, requested {count}")
    result = []
    for item_id, index, record in candidates[:count]:
        options, answer = list(map(str, record["choices"])), int(record["answer"])
        if not 0 <= answer < 4:
            raise ValueError(f"{item_id}: invalid answer {answer}")
        for target in range(4):
            shift = (target - answer) % 4
            result.append({"id": f"{item_id}#r{shift}", "item_id": item_id, "domain": subject,
                           "question": record["question"].strip(),
                           "options": [options[(i - shift) % 4].strip() for i in range(4)],
                           "answer_idx": target, "rotation": shift, "split": "dev", "source_row": index})
    return result


def read_arrow(path: Path) -> list[dict]:
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_stream(source).read_all().to_pylist()


def freeze_protocol(tokenizer, probes_path: Path, fixed_date: str) -> dict:
    from stepscope.observe.probes import RenderConfig, load_probe_file
    from stepscope.observe.scoring import encode_probes

    probes = load_probe_file(probes_path)
    # Selection uses token validity only, never accuracy or the outcome of the update.
    errors = []
    for prefix in ("Answer:", "The answer is"):
        cfg = RenderConfig(chat_template=True, enable_thinking=False, answer_prefix=prefix,
                           label_format=" {label}", fixed_date=fixed_date)
        try:
            encoded = encode_probes(tokenizer, probes.items, cfg)
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")
            continue
        return {"schema": "stepscope-real-case-protocol-v1", "chosen_by": "first token-valid prefix on all probes",
                "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "direct": {"enable_thinking": False, "answer_prefix": prefix, "label_format": " {label}",
                           "labels": ["A", "B", "C", "D"], "no_think_suffix": "", "fixed_date": fixed_date},
                "probe_count": len(probes.items), "encoded_probe_count": len(encoded),
                "option_token_ids": encoded.option_ids[0].tolist()}
    raise ValueError("No single-token answer-label protocol passed: " + "; ".join(errors))


def select_prompts(parquet: Path, tokenizer, per_source: int, max_tokens: int, fixed_date: str) -> list[dict]:
    import pyarrow.parquet as pq

    from stepscope.observe.probes import fixed_strftime

    picked = {source: [] for source in ("math", "code", "if")}
    row_index = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=128, columns=["prompt", "domain", "extra_info"]):
        for record in batch.to_pylist():
            index = row_index
            row_index += 1
            source = record["domain"]
            if source not in picked or len(picked[source]) >= per_source:
                continue
            text = tokenizer.apply_chat_template(record["prompt"], tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False, strftime_now=fixed_strftime(fixed_date))
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids or len(ids) > max_tokens:
                continue
            info = record.get("extra_info") or {}
            picked[source].append({"source": source, "source_row": index,
                                   "source_id": info.get("source_index") or info.get("index") or str(index),
                                   "prompt_ids": ids, "prompt_len": len(ids),
                                   "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()})
        if all(len(rows) == per_source for rows in picked.values()):
            break
    if any(len(rows) < per_source for rows in picked.values()):
        raise ValueError(f"Not enough prompts within {max_tokens} tokens: "
                         + str({key: len(value) for key, value in picked.items()}))
    return [row for rows in picked.values() for row in rows]


def dependency_versions() -> dict:
    result = {}
    for package in ("torch", "transformers", "huggingface-hub", "pyarrow"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def download(args) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    selection = {"per_source": args.per_source, "items_per_domain": args.items_per_domain,
                 "max_prompt_tokens": args.max_prompt_tokens, "fixed_date": args.fixed_date,
                 "subjects": list(SUBJECTS), "policy": "first short prompts per source; hashed dev probes"}
    manifest_path = root / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if old and old["selection"] != selection:
        raise ValueError("Selection differs from existing manifest; use a new --out directory.")
    paths = {}
    for name, revision in MODEL_REVISIONS.items():
        print(f"Downloading/reusing {name}@{revision}", flush=True)
        paths[name] = snapshot_download(MODEL_PREFIX + name, revision=revision, cache_dir=str(cache),
                                        allow_patterns=MODEL_FILES)
    tokenizer = AutoTokenizer.from_pretrained(paths["MixSFT"], local_files_only=True)
    token_hashes = {name: digest(Path(path) / "tokenizer.json") for name, path in paths.items()}
    template_hashes = {name: digest(Path(path) / "chat_template.jinja") for name, path in paths.items()}
    if len(set(token_hashes.values())) != 1 or len(set(template_hashes.values())) != 1:
        raise ValueError("Teacher and student tokenizer/template files differ.")
    probes, source_files = [], []
    for subject in SUBJECTS:
        path = Path(hf_hub_download(PROBE_REPO, f"{subject}/data-00000-of-00001.arrow", repo_type="dataset",
                                    revision=PROBE_REVISION, cache_dir=str(cache)))
        probes.extend(probe_rows(read_arrow(path), subject, args.items_per_domain))
        source_files.append({"file": f"{subject}/{path.name}", "sha256": digest(path)})
    write_rows(root / "probes.jsonl", probes)
    write_json(root / "protocol.json", freeze_protocol(tokenizer, root / "probes.jsonl", args.fixed_date))
    print("Downloading/reusing the 0.94 GB public prompt mixture", flush=True)
    train = Path(hf_hub_download(TRAIN_REPO, "rl_prompt_mix/train.parquet", repo_type="dataset",
                                revision=TRAIN_REVISION, cache_dir=str(cache)))
    if digest(train) != TRAIN_SHA256:
        raise ValueError("Training parquet digest differs from the pinned upstream manifest.")
    prompts = select_prompts(train, tokenizer, args.per_source, args.max_prompt_tokens, args.fixed_date)
    write_rows(root / "prompts.jsonl", prompts)
    manifest = {"schema": "stepscope-real-case-v1", "selection": selection,
                "models": {name: {"repo": MODEL_PREFIX + name, "revision": rev, "license": "apache-2.0"}
                           for name, rev in MODEL_REVISIONS.items()}, "model_paths": paths,
                "tokenizer_sha256": token_hashes, "chat_template_sha256": template_hashes,
                "training": {"repo": TRAIN_REPO, "revision": TRAIN_REVISION,
                             "file": "rl_prompt_mix/train.parquet", "sha256": TRAIN_SHA256,
                             "license": "upstream source terms; no unified license in the dataset card",
                             "rows": [{k: row[k] for k in ("source", "source_row", "source_id", "prompt_sha256")}
                                      for row in prompts]},
                "probes": {"repo": PROBE_REPO, "revision": PROBE_REVISION, "license": "cc-by-4.0",
                           "files": source_files, "rows": [row["id"] for row in probes]},
                "inputs": {name: digest(root / name) for name in ("probes.jsonl", "protocol.json", "prompts.jsonl")},
                "dependencies": dependency_versions()}
    if old and old.get("inputs") != manifest["inputs"]:
        raise ValueError("Prepared input content changed; use a new output directory.")
    if old and "generation" in old:
        manifest["generation"] = old["generation"]
    write_json(manifest_path, manifest)
    print(f"CPU preparation complete: {len(prompts)} prompts, {len(probes)} probe rows. Output: {root}")


def generate(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = args.out.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["inputs"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Input digest changed: {name}")
    prompts = read_rows(root / "prompts.jsonl")
    config = {"seed": args.seed, "max_new_tokens": args.max_new_tokens, "do_sample": True,
              "temperature": 0.6, "top_p": 0.95, "top_k": 20, "enable_thinking": False,
              "dtype": args.dtype, "device": args.device,
              "fixed_date": manifest["selection"]["fixed_date"], "inputs": manifest["inputs"]}
    job_hash = fingerprint(config)
    state_path, progress_path = root / "generation-state.json", root / "batch.progress.jsonl"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != job_hash:
            raise ValueError("Generation settings changed; use a new --out directory.")
    else:
        write_json(state_path, {"fingerprint": job_hash, "config": config})
    final = root / "batch.jsonl"
    if final.exists() and manifest.get("generation"):
        if digest(final) != manifest["generation"]["batch_sha256"]:
            raise ValueError("Saved batch content differs from the completed generation manifest.")
    completed = read_rows(final if final.exists() else progress_path) if final.exists() or progress_path.exists() else []
    if len(completed) > len(prompts):
        raise ValueError("Too many saved responses.")
    for index, row in enumerate(completed):
        prompt = prompts[index]
        if (row["input_ids"][:prompt["prompt_len"]] != prompt["prompt_ids"]
                or row["source"] != prompt["source"] or row["generation_seed"] != args.seed + index):
            raise ValueError(f"Saved response {index} does not match the input or generation seed.")
    if len(completed) < len(prompts):
        path = manifest["model_paths"]["MixSFT"]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, args.dtype),
                                                    attn_implementation="eager", local_files_only=True)
        model.to(args.device).eval()
        for index in range(len(completed), len(prompts)):
            prompt = prompts[index]
            torch.manual_seed(args.seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + index)
            ids = torch.tensor([prompt["prompt_ids"]], dtype=torch.long, device=args.device)
            with torch.inference_mode():
                output = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                        max_new_tokens=args.max_new_tokens, temperature=0.6, top_p=0.95, top_k=20,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tokens = output[0].tolist()
            response = tokens[prompt["prompt_len"]:]
            if not response:
                raise ValueError(f"Empty generated response for row {index}")
            row = {"source": prompt["source"], "input_ids": tokens, "prompt_len": prompt["prompt_len"],
                   "source_row": prompt["source_row"], "source_id": prompt["source_id"],
                   "generation_seed": args.seed + index, "response_tokens": len(response),
                   "ended_with_eos": response[-1] == tokenizer.eos_token_id,
                   "hit_token_limit": len(response) == args.max_new_tokens}
            completed.append(row)
            write_rows(progress_path, completed)
            print(f"Generated {index + 1}/{len(prompts)}: {prompt['source']}, {len(response)} response tokens", flush=True)
        del model
    write_rows(final, completed)
    manifest["generation"] = {"config": config, "fingerprint": job_hash, "batch_sha256": digest(final),
                              "rows": len(completed), "dependencies": dependency_versions(),
                              "note": "Short no-thinking calibration batch, not the original full training rollout."}
    write_json(manifest_path, manifest)
    print(f"Generation complete: {final}. Keep this batch fixed for SGD and AdamW.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument("--generate", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/real_case"))
    parser.add_argument("--per-source", type=int, default=2)
    parser.add_argument("--items-per-domain", type=int, default=5)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--fixed-date", default="2026-10-19")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    if min(args.per_source, args.items_per_domain, args.max_prompt_tokens, args.max_new_tokens) < 1:
        parser.error("Counts and token limits must be positive.")
    # Never implicitly pick up hub authentication for these entirely public downloads.
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    (download if args.download_only else generate)(args)


if __name__ == "__main__":
    main()
