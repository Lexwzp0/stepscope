"""Smoke tests: schemas build, store round-trips, registries populate, CLI parses. No torch."""
import importlib

import pyarrow as pa

from stepscope.core import TABLES, table_schema
from stepscope.core.provenance import provenance
from stepscope.core.store import Store


def test_every_table_has_provenance_columns():
    for name in TABLES:
        s = table_schema(name)
        for col in ("run", "analysis", "analysis_version", "code_hash", "level", "cost_s", "cost_mem_gb"):
            assert col in s.names, (name, col)


def test_store_roundtrip(tmp_path):
    st = Store(tmp_path)
    prov = provenance("run0", "dynamics.actual_change", "0.0.1", "exact")
    rows = [dict(step_from=0, step_to=10, probe_set="gpqa", probe_version="v1", item="q1",
                 observable="phi_ans", actual=0.1, first_order=0.08, **prov)]
    st.write("run0", "effects", rows, part="0000")
    t = st.read("run0", "effects")
    assert t.num_rows == 1 and t.schema.equals(table_schema("effects"))
    assert (tmp_path / "run0" / "manifest.json").exists()


def test_registries_populate_on_import():
    for mod in ("stepscope.adapters.models.hf", "stepscope.adapters.objectives.mopd",
                "stepscope.adapters.optimizers.adamw", "stepscope.adapters.frameworks.verl",
                "stepscope.analyze.dynamics.first_order", "stepscope.analyze.interventions.data_intervention",
                "stepscope.studies.ood_spillover.study"):
        importlib.import_module(mod)
    from stepscope.core.registry import ANALYSES, FRAMEWORKS, MODELS, OBJECTIVES, OPTIMIZERS, STUDIES
    assert "hf" in MODELS.names() and "mopd" in OBJECTIVES.names() and "adamw" in OPTIMIZERS.names()
    assert "verl" in FRAMEWORKS.names() and "ood_spillover" in STUDIES.names()
    assert ANALYSES.get("dynamics.first_order").level == "first_order"
    assert ANALYSES.get("interventions.data_intervention").level == "counterfactual"
    assert "interventions" in ANALYSES.get("interventions.data_intervention").writes


def test_cli_parses_all_commands():
    from stepscope.cli.main import build_parser
    p = build_parser()
    assert p.parse_args(["ingest", "runs/x"]).cmd == "ingest"
    assert p.parse_args(["analyze", "run0", "--study", "ood_spillover"]).study == "ood_spillover"
