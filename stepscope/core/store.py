"""Parquet store with a per-run manifest (blueprint §七). Engine writes here; report only reads.

The manifest lists every table part with its row count and, under ``analyses``, which analyses ran or were
skipped (with the reason, e.g. a model capability the adapter does not declare)."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import table_schema


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def run_dir(self, run_id: str) -> Path:
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, run_id: str, table: str, rows: list[dict], part: str) -> Path:
        schema = table_schema(table)
        tbl = pa.Table.from_pylist(rows, schema=schema)
        out = self.run_dir(run_id) / table / f"{part}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, out)
        self._edit_manifest(run_id, lambda d: d["tables"].setdefault(table, {}).__setitem__(part, {"rows": len(rows)}))
        return out

    def read(self, run_id: str, table: str) -> pa.Table:
        d = self.run_dir(run_id) / table
        files = sorted(d.glob("*.parquet"))
        if not files:
            return table_schema(table).empty_table()
        # Supply the current schema so optional fields added after an older part was written
        # become null there, including when legacy and current parts share one run.
        return pa.concat_tables([pq.read_table(f, schema=table_schema(table)) for f in files])

    def record_analysis(self, run_id: str, name: str, version: str, status: str, reasons: list[str] | None = None,
                        context: str | None = None) -> None:
        """Record that analysis ``name`` ran or was skipped (``status`` = ran | skipped) for ``context``."""
        entry = {"version": version, "status": status}
        if reasons:
            entry["reasons"] = list(reasons)
        key = f"{name}@{context}" if context else name
        self._edit_manifest(run_id, lambda d: d.setdefault("analyses", {}).__setitem__(key, entry))

    def manifest(self, run_id: str) -> dict:
        mf = self.run_dir(run_id) / "manifest.json"
        return json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"tables": {}}

    def _edit_manifest(self, run_id: str, fn) -> None:
        mf = self.run_dir(run_id) / "manifest.json"
        data = self.manifest(run_id)
        data.setdefault("tables", {})
        fn(data)
        mf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
