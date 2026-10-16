"""Table schemas (blueprint §七). One row unit per table; aggregation never double counts."""
from __future__ import annotations

import pyarrow as pa

_PROV = [
    ("run", pa.string()),
    ("analysis", pa.string()),
    ("analysis_version", pa.string()),
    ("code_hash", pa.string()),
    ("level", pa.string()),       # exact | first_order | channel | counterfactual
    ("cost_s", pa.float64()),
    ("cost_mem_gb", pa.float64()),
]

TABLES: dict[str, list[tuple[str, pa.DataType]]] = {
    # one update or interval × one probe item × one observable
    "effects": [
        ("step_from", pa.int64()), ("step_to", pa.int64()),
        ("probe_set", pa.string()), ("probe_version", pa.string()), ("item", pa.string()),
        ("domain", pa.string()), ("observable", pa.string()),
        ("value_from", pa.float64()), ("value_to", pa.float64()),
        ("actual", pa.float64()), ("first_order", pa.float64()),
        ("method", pa.string()),      # how first_order was computed: jvp | backward | none
    ],
    # effects row × one source (teacher/domain/sample/token) or momentum / weight_decay part.
    # Only decompositions whose parts sum to first_order.
    "contributions": [
        ("step_from", pa.int64()), ("step_to", pa.int64()),
        ("probe_set", pa.string()), ("probe_version", pa.string()), ("item", pa.string()),
        ("observable", pa.string()),
        ("source_kind", pa.string()), ("source", pa.string()),
        ("contrib", pa.float64()), ("method", pa.string()),
    ],
    # contributions row × one layer / channel (LDCL CH1/CH2; SGD-direction applicability)
    "channels": [
        ("step_from", pa.int64()), ("step_to", pa.int64()),
        ("probe_set", pa.string()), ("probe_version", pa.string()), ("item", pa.string()),
        ("observable", pa.string()),
        ("source_kind", pa.string()), ("source", pa.string()),
        ("layer", pa.string()), ("ch1", pa.float64()), ("ch2", pa.float64()),
        ("applicability", pa.string()),  # sgd | adamw_qualitative
    ],
    # one checkpoint × one probe item × one sample
    "behavior": [
        ("step", pa.int64()), ("probe_set", pa.string()), ("probe_version", pa.string()),
        ("item", pa.string()), ("domain", pa.string()), ("sample", pa.int64()),
        ("metric", pa.string()), ("score", pa.float64()), ("category", pa.string()),
        ("truncated", pa.bool_()), ("refused", pa.bool_()),
        ("wording", pa.string()), ("option_order", pa.string()),
    ],
    # one checkpoint × one layer × one metric × one probe set
    "representation": [
        ("step", pa.int64()), ("layer", pa.string()), ("metric", pa.string()),
        ("probe_set", pa.string()), ("probe_version", pa.string()), ("value", pa.float64()),
    ],
    # one probe set × one update/interval × one prediction kind. Decompositions only.
    "consistency": [
        ("step_from", pa.int64()), ("step_to", pa.int64()),
        ("probe_set", pa.string()), ("probe_version", pa.string()), ("observable", pa.string()),
        ("prediction", pa.string()),  # first_order | decomposition_sum | ch1_ch2
        ("n", pa.int64()),
        ("corr", pa.float64()), ("sign_agree", pa.float64()),
        ("resid_actual_minus_pred", pa.float64()),   # mean of (actual - pred)
        ("resid_rms", pa.float64()),                 # root mean square of (actual - pred)
        ("actual_rms", pa.float64()),                # scale reference
        ("resid_pred_minus_sum", pa.float64()),      # NaN unless a decomposition exists
    ],
    # one target × one training source × one method. Counterfactual; never added to effects.
    "interventions": [
        ("target", pa.string()), ("source_kind", pa.string()), ("source", pa.string()),
        ("method", pa.string()),   # source | approx_unrolling | tracin | influence | less
        ("estimate", pa.float64()), ("inputs_hash", pa.string()), ("validation", pa.string()),
    ],
}


def table_schema(name: str) -> pa.Schema:
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}; known: {sorted(TABLES)}")
    return pa.schema(TABLES[name] + _PROV)
