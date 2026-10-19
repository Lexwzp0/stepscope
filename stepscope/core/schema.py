"""Table schemas (blueprint §七). One row unit per table; aggregation never double counts."""
from __future__ import annotations

import pyarrow as pa

_PROV = [
    ("run", pa.string()),
    ("analysis", pa.string()),
    ("analysis_version", pa.string()),
    ("code_hash", pa.string()),
    ("level", pa.string()),       # exact | first_order | channel | path_integral | counterfactual
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
        ("first_order_loss", pa.float64()),  # nullable; SGD loss-only tangent, excluding nonzero weight decay
        ("method", pa.string()),      # how first_order was computed: jvp | backward | none
    ],
    # effects row × one component of one decomposition (source, momentum, weight decay, layer, teacher direction).
    # Decomposition protocol: every row of one decomposition shares ``decomp_id``; ``target`` names the effects
    # column its parts must sum to (first_order for per-source parts of an update, actual for path-integral
    # layer parts and teacher-direction projections); consistency checks Σ parts against that column.
    "contributions": [
        ("step_from", pa.int64()), ("step_to", pa.int64()),
        ("probe_set", pa.string()), ("probe_version", pa.string()), ("item", pa.string()),
        ("observable", pa.string()),
        ("decomp_id", pa.string()),
        ("target", pa.string()),         # first_order | actual
        ("source_kind", pa.string()), ("source", pa.string()),
        ("contrib", pa.float64()), ("method", pa.string()),
    ],
    # effects row × one source × one channel site. ch1 / ch2 are already multiplied by the effective step size,
    # so Σ over sources and sites of (ch1 + ch2) predicts the loss-gradient effect in observable units.
    # With weight decay, compare to effects.first_order_loss rather than the full-update first_order.
    # layer = "readout" carries CH1 (ch2 = 0); layer = a residual-branch site carries that site's CH2 (ch1 = 0).
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
    # one checkpoint × one layer × one metric × one probe set. Scalars live here; spectra, matrices and other
    # large arrays are analysis artefacts referenced by ``artifact`` ("<path>#sha1=<hash>", null when none).
    "representation": [
        ("step", pa.int64()), ("layer", pa.string()), ("metric", pa.string()),
        ("probe_set", pa.string()), ("probe_version", pa.string()), ("value", pa.float64()),
        ("artifact", pa.string()),
    ],
    # one probe set × one update/interval × one prediction or one decomposition.
    # corr / sign_agree / resid_* compare ``prediction`` with ``reference`` (actual unless stated; the channel
    # check of chain level 1 compares ch1_ch2 with first_order_loss when available, otherwise first_order).
    # For a decomposition, prediction is Σ parts and
    # resid_target_minus_sum is the RMS of (target − Σ parts), target read from the decomposition's rows.
    "consistency": [
        ("step_from", pa.int64()), ("step_to", pa.int64()),
        ("probe_set", pa.string()), ("probe_version", pa.string()), ("observable", pa.string()),
        ("prediction", pa.string()),  # first_order | ch1_ch2 | decomposition_sum
        ("decomp_id", pa.string()),   # null unless prediction == decomposition_sum
        ("reference", pa.string()),   # actual | first_order | first_order_loss
        ("n", pa.int64()),
        ("corr", pa.float64()), ("sign_agree", pa.float64()),
        ("resid_actual_minus_pred", pa.float64()),   # mean of (reference - prediction)
        ("resid_rms", pa.float64()),                 # root mean square of (reference - prediction)
        ("actual_rms", pa.float64()),                # scale reference: RMS of the reference column
        ("resid_target_minus_sum", pa.float64()),    # NaN unless prediction == decomposition_sum
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
