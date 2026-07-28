"""V2 trace shards: exact slot-addressed sources, self-verifying replay, and V1 compatibility.

The load-bearing invariant here is that a trace's SOURCE must be restored with its exact slot layout.
Actions address atoms by slot index, and a SMILES round-trip re-canonicalizes atom order: measured on real
corruption sources, 7 of 10 do NOT survive index-exactly. Storing only source_smiles would therefore
repoint every action at a different atom -- silently, since the result is still a valid molecule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compose_v4.chem.molecular_graph import (  # noqa: E402
    ORGANIC_VOCABULARY,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key  # noqa: E402
from compose_v4.rewrite.trace_shard import (  # noqa: E402
    TraceShardError,
    decode_state,
    decode_trace_record,
    encode_state,
    encode_trace_record,
    load_trace_records,
    write_shard,
)

_PANEL = [
    "O=C1NC(=O)c2ccccc21", "Cc1ccc(cc1)C(=O)Nc1ccccc1", "C1CCNCC1C(=O)O",
    "C[N+](C)(C)CCO", "CC(=O)[O-]", "O=C(O)c1ccc(cc1)S(=O)(=O)N",
]


def _records(seed: int = 11):
    from compose_v4.experiments.corrupted_source_prior import build_corrupted_prior_records
    from warmstart_dry_run import build_production_ring_catalog

    catalog = build_production_ring_catalog(40)
    recs, _log = build_corrupted_prior_records(
        _PANEL, n_slots=40, depth_max=5, seed=seed, catalog=catalog,
        vocabulary=ORGANIC_VOCABULARY, couplings_per_target=2,
    )
    return recs


def test_state_encoding_is_slot_exact():
    for rec in _records():
        src = rec.path.trace.source
        back = decode_state(encode_state(src))
        for field in ("atom_types", "formal_charges", "implicit_h_counts"):
            assert np.array_equal(getattr(src, field), getattr(back, field)), field
        assert np.array_equal(src.bonds, back.bonds)


def test_smiles_roundtrip_is_NOT_slot_exact_which_is_why_we_store_state():
    """Documents the hazard: this is why source_smiles alone cannot rebuild a trace source."""
    unstable = 0
    recs = _records()
    for rec in recs:
        src = rec.path.trace.source
        re_ = pad_molecular_graph(smiles_to_molecular_graph(molecular_graph_to_smiles(src)), 40)
        if not np.array_equal(src.atom_types, re_.atom_types) or not np.array_equal(src.bonds, re_.bonds):
            unstable += 1
    assert unstable > 0, (
        "expected some sources to reorder under a SMILES round-trip; if this ever becomes 0 the "
        "exact-state requirement is still correct, but this test no longer demonstrates why"
    )


def test_trace_records_replay_exactly():
    for i, rec in enumerate(_records()):
        record = encode_trace_record(
            rec.path.trace, n_slots=40, seed=11, trace_id=f"t{i}", partition="train"
        )
        trace = decode_trace_record(record, validate=True)   # replays every step, asserts each key
        assert canonical_state_key(trace.source) == record["source_key"]
        assert canonical_state_key(trace.target) == record["target_key"]
        assert len(trace.steps) == record["path_length"]


def test_shard_write_read_roundtrip_and_determinism():
    recs = _records()
    out = [
        encode_trace_record(r.path.trace, n_slots=40, seed=11, trace_id=f"t{i}", partition="train")
        for i, r in enumerate(recs)
    ]
    a = write_shard(Path("/tmp/_ts_a.jsonl.gz"), out)
    b = write_shard(Path("/tmp/_ts_b.jsonl.gz"), out)
    assert a["content_sha256"] == b["content_sha256"], "shard content must be deterministic"
    traces = load_trace_records(Path("/tmp/_ts_a.jsonl.gz"), validate=True)
    assert len(traces) == len(out)


def test_missing_source_state_is_rejected():
    rec = _records()[0]
    record = encode_trace_record(rec.path.trace, n_slots=40, seed=1, trace_id="x", partition="train")
    del record["source_state"]
    with pytest.raises(TraceShardError, match="no source_state"):
        decode_trace_record(record)


def test_tampered_successor_key_is_rejected():
    rec = next(r for r in _records() if r.path.trace.steps)
    record = encode_trace_record(rec.path.trace, n_slots=40, seed=1, trace_id="x", partition="train")
    record["steps"][0]["successor_key"] = "C"
    with pytest.raises(TraceShardError, match="replayed successor"):
        decode_trace_record(record, validate=True)
