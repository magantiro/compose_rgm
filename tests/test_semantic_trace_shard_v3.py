"""Exact persistence and process-identity gates for Editing-V2 traces."""

from __future__ import annotations

import hashlib
import json

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_v2_process_identity,
)
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import CycleCloseEdge
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard_v3 import (
    SemanticTraceShardError,
    decode_semantic_trace_record,
    encode_semantic_trace_record,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _record():
    source = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 16)
    step = RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1))
    target = editing_v2_semantic_rewrite_system().apply(
        source,
        step.rule_name,
        step.action,
    )
    trace = RewriteTrace(source, target, (step,), {"evidence": "test_fixture"})
    return encode_semantic_trace_record(
        trace,
        trace_id="fixture-1",
        split="development",
        source_address={"shard": "fixture", "row": 0},
        lineage={"kind": "programmatic_fixture"},
    )


def _rehash(record: dict[str, object]) -> None:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = _canonical_sha256(body)


def test_process_identity_is_full_and_binds_every_required_boundary() -> None:
    identity = editing_v2_process_identity()
    assert len(identity["process_identity_sha256"]) == 64
    assert len(identity["contract_sha256"]) == 64
    sources = set(identity["implementation_source_sha256"])
    for suffix in (
        "action_codec_v4.py",
        "trace_shard_v3.py",
        "kernel.py",
        "semantic_atom_restate.py",
        "semantic_cycle_close.py",
        "semantic_cycle_open.py",
        "ring_restate_semantics.py",
        "factorized_tracelet_rate_model.py",
        "production_successor_kernel.py",
        "reference_successor_kernel.py",
        "molecular_graph.py",
        "charge_policy.py",
    ):
        assert any(path.endswith(suffix) for path in sources), suffix
    assert all(
        len(digest) == 64
        for digest in identity["implementation_source_sha256"].values()
    )


def test_semantic_trace_round_trip_is_deterministic_and_exact() -> None:
    first = _record()
    second = _record()
    assert first == second
    trace = decode_semantic_trace_record(first)
    assert trace.steps == (RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1)),)
    assert trace.metadata == {"evidence": "test_fixture"}


def test_tampered_implicit_h_fails_after_valid_self_rehash() -> None:
    record = _record()
    record["states"][1]["implicit_h_counts"][0] += 1
    _rehash(record)
    with pytest.raises(SemanticTraceShardError, match="state|replay"):
        decode_semantic_trace_record(record)


def test_tampered_process_identity_fails_after_valid_self_rehash() -> None:
    record = _record()
    record["process_identity_sha256"] = "0" * 64
    _rehash(record)
    with pytest.raises(SemanticTraceShardError, match="process identity"):
        decode_semantic_trace_record(record)


def test_semantic_trace_requires_every_progress_state() -> None:
    record = _record()
    record["states"].pop()
    record["canonical_state_keys"].pop()
    _rehash(record)
    with pytest.raises(SemanticTraceShardError, match="one exact state"):
        decode_semantic_trace_record(record)
