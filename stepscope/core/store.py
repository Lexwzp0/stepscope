"""Parquet store with a per-run manifest (blueprint §七). Engine writes here; report only reads."""
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
        self._update_manifest(run_id, table, part, len(rows))
        return out

    def read(self, run_id: str, table: str) -> pa.Table:
        d = self.run_dir(run_id) / table
        files = sorted(d.glob("*.parquet"))
        if not files:
            return table_schema(table).empty_table()
        return pa.concat_tables([pq.read_table(f) for f in files])

    def _update_manifest(self, run_id: str, table: str, part: str, n: int) -> None:
        mf = self.run_dir(run_id) / "manifest.json"
        data = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"tables": {}}
        data["tables"].setdefault(table, {})[part] = {"rows": n}
        mf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
