"""Strict exact-state trace persistence for the Editing-V2 semantic process."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.editing_v2_process_identity import (
    PROCESS_SEMANTICS,
    editing_v2_process_identity,
    require_editing_v2_process_identity,
    validate_frozen_process_identity,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import decode_state, encode_state

TRACE_SCHEMA = "compose.rewrite.trace"
TRACE_SCHEMA_VERSION = 3
MAX_ACTIVE_ATOMS = 40
_TRACE_RECORD_FIELDS = {
    "schema",
    "schema_version",
    "process_semantics",
    "process_contract_sha256",
    "process_identity_sha256",
    "action_codec_schema_version",
    "action_codec_implementation_hash",
    "trace_id",
    "data_lane",
    "split",
    "source_address",
    "lineage",
    "path_length",
    "states",
    "canonical_state_keys",
    "steps",
    "family_histogram",
    "metadata",
    "record_sha256",
}


class SemanticTraceShardError(ValueError):
    """A semantic trace record violates exact process or replay identity."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _exact_state_equal(left: MolecularGraph, right: MolecularGraph) -> bool:
    return all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in ("atom_types", "formal_charges", "implicit_h_counts", "bonds")
    )


def _decode_exact_state(payload: object, *, path: str) -> MolecularGraph:
    if not isinstance(payload, dict):
        raise SemanticTraceShardError(f"{path}: exact state must be an object")
    expected = {
        "n_slots",
        "atom_types",
        "formal_charges",
        "implicit_h_counts",
        "bonds",
    }
    if set(payload) != expected:
        raise SemanticTraceShardError(
            f"{path}: exact-state fields disagree with the trace-v3 schema"
        )
    try:
        state = decode_state(payload)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise SemanticTraceShardError(f"{path}: malformed exact state") from error
    if encode_state(state) != payload:
        raise SemanticTraceShardError(
            f"{path}: exact state is not in deterministic canonical wire form"
        )
    if not is_valid_state(state) or not is_connected_or_null(state):
        raise SemanticTraceShardError(f"{path}: state is invalid or disconnected")
    if int(is_element(state.atom_types).sum()) > MAX_ACTIVE_ATOMS:
        raise SemanticTraceShardError(f"{path}: state exceeds the 40-atom support")
    return state


def _record_self_hash(record: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def encode_semantic_trace_record(
    trace: RewriteTrace,
    *,
    trace_id: str,
    data_lane: str,
    split: str,
    source_address: Mapping[str, object],
    lineage: Mapping[str, object],
) -> dict[str, object]:
    """Replay and persist every exact Active8 intermediate under one identity."""

    if not trace_id or not data_lane or not split:
        raise SemanticTraceShardError("trace_id, data_lane, and split must be nonempty")
    identity = editing_v2_process_identity()
    runtime = editing_v2_semantic_rewrite_system()
    state = trace.source
    states = [state]
    steps: list[dict[str, object]] = []
    families: Counter[str] = Counter()
    for index, step in enumerate(trace.steps):
        try:
            action = action_codec_v4.encode_action(step.rule_name, step.action)
            successor = runtime.apply(state, step.rule_name, step.action)
        except ValueError as error:
            raise SemanticTraceShardError(
                f"step {index}: action is outside the exact Editing-V2 process"
            ) from error
        steps.append(
            {
                "action": action,
                "source_key": canonical_state_key(state),
                "successor_key": canonical_state_key(successor),
            }
        )
        families[str(action["model_family"])] += 1
        states.append(successor)
        state = successor
    if not _exact_state_equal(state, trace.target):
        raise SemanticTraceShardError(
            "trace target differs from the exact semantic-runtime replay target"
        )
    body: dict[str, object] = {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        "process_semantics": PROCESS_SEMANTICS,
        "process_contract_sha256": identity["contract_sha256"],
        "process_identity_sha256": identity["process_identity_sha256"],
        "action_codec_schema_version": action_codec_v4.SCHEMA_VERSION,
        "action_codec_implementation_hash": action_codec_v4.codec_implementation_hash(),
        "trace_id": trace_id,
        "data_lane": data_lane,
        "split": split,
        "source_address": dict(source_address),
        "lineage": dict(lineage),
        "path_length": len(steps),
        "states": [encode_state(item) for item in states],
        "canonical_state_keys": [canonical_state_key(item) for item in states],
        "steps": steps,
        "family_histogram": dict(sorted(families.items())),
        "metadata": dict(trace.metadata),
    }
    return {**body, "record_sha256": _canonical_sha256(body)}


def decode_semantic_trace_record(
    record: object,
    *,
    validate_replay: bool = True,
    expected_process_identity: Mapping[str, object] | None = None,
) -> RewriteTrace:
    """Validate one exact semantic trace, optionally replaying its transitions.

    ``validate_replay=False`` is reserved for a physically hashed semantic
    packed artifact whose build already performed exact replay.  It still
    validates the record self-hash, full process identity, exact state wire
    form, stored key consistency, and every V4 action, but avoids both executor
    and RDKit calls.  Ordinary callers should retain the default and re-execute
    every step.

    ``expected_process_identity`` defaults to ``None``, which requires the
    record's process identity to be the LIVE one and is byte-identical to the
    historical behavior.  Supplying a complete historical identity object
    instead requires the record to match exactly that pinned identity.  This is
    the only honest way to read an IMMUTABLE payload whose process identity has
    been superseded: the caller must name the superseded identity explicitly,
    and the pinned object is itself validated for self-consistency.  It proves
    provenance, never currency, so a caller that needs the current process must
    keep the default.  It mirrors the identical contract on the semantic packed
    store's readers, which is where it is threaded from.
    """

    if not isinstance(record, dict):
        raise SemanticTraceShardError("semantic trace record must be an object")
    if set(record) != _TRACE_RECORD_FIELDS:
        raise SemanticTraceShardError(
            "semantic trace top-level fields disagree with the trace-v3 schema"
        )
    if record.get("schema") != TRACE_SCHEMA:
        raise SemanticTraceShardError("unknown semantic trace schema")
    if record.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise SemanticTraceShardError("unknown semantic trace schema version")
    supplied_hash = record.get("record_sha256")
    if not isinstance(supplied_hash, str) or supplied_hash != _record_self_hash(record):
        raise SemanticTraceShardError("semantic trace record self-hash disagrees")
    process_sha256 = record.get("process_identity_sha256")
    if not isinstance(process_sha256, str):
        raise SemanticTraceShardError("semantic trace lacks a process identity")
    if expected_process_identity is None:
        try:
            identity = require_editing_v2_process_identity(process_sha256)
        except ValueError as error:
            raise SemanticTraceShardError(
                "semantic trace process identity disagrees"
            ) from error
    else:
        try:
            identity = validate_frozen_process_identity(expected_process_identity)
        except ValueError as error:
            raise SemanticTraceShardError(
                "pinned semantic trace process identity is not self-consistent"
            ) from error
        if process_sha256 != identity["process_identity_sha256"]:
            raise SemanticTraceShardError(
                "semantic trace process identity disagrees with the pinned identity"
            )
    if record.get("process_semantics") != identity["process_semantics"]:
        raise SemanticTraceShardError("semantic trace process label disagrees")
    if record.get("process_contract_sha256") != identity["contract_sha256"]:
        raise SemanticTraceShardError("semantic trace process contract disagrees")
    if record.get("action_codec_schema_version") != action_codec_v4.SCHEMA_VERSION:
        raise SemanticTraceShardError("semantic trace action-codec version disagrees")
    if (
        record.get("action_codec_implementation_hash")
        != action_codec_v4.codec_implementation_hash()
    ):
        raise SemanticTraceShardError("semantic trace action-codec identity disagrees")

    states_payload = record.get("states")
    steps_payload = record.get("steps")
    keys_payload = record.get("canonical_state_keys")
    if not isinstance(states_payload, list) or not isinstance(steps_payload, list):
        raise SemanticTraceShardError("semantic trace states and steps must be arrays")
    if not isinstance(keys_payload, list):
        raise SemanticTraceShardError("semantic trace canonical keys must be an array")
    if not all(isinstance(key, str) and key for key in keys_payload):
        raise SemanticTraceShardError(
            "semantic trace canonical keys must be nonempty strings"
        )
    if len(states_payload) != len(steps_payload) + 1:
        raise SemanticTraceShardError(
            "semantic trace requires one exact state per progress point"
        )
    if len(keys_payload) != len(states_payload):
        raise SemanticTraceShardError("semantic trace canonical-key count disagrees")
    if record.get("path_length") != len(steps_payload):
        raise SemanticTraceShardError("semantic trace path length disagrees")
    for field in ("trace_id", "data_lane", "split"):
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise SemanticTraceShardError(
                f"semantic trace {field} must be a nonempty string"
            )
    for field in ("source_address", "lineage", "metadata"):
        if not isinstance(record.get(field), dict):
            raise SemanticTraceShardError(f"semantic trace {field} must be an object")
    if not record["source_address"]:
        raise SemanticTraceShardError("semantic trace source_address must be nonempty")
    states = tuple(
        _decode_exact_state(payload, path=f"states[{index}]")
        for index, payload in enumerate(states_payload)
    )
    if (
        validate_replay
        and [canonical_state_key(state) for state in states] != keys_payload
    ):
        raise SemanticTraceShardError("semantic trace canonical state keys disagree")

    runtime = editing_v2_semantic_rewrite_system() if validate_replay else None
    steps: list[RewriteStep] = []
    families: Counter[str] = Counter()
    for index, entry in enumerate(steps_payload):
        if not isinstance(entry, dict) or set(entry) != {
            "action",
            "source_key",
            "successor_key",
        }:
            raise SemanticTraceShardError(f"step {index}: fields disagree")
        try:
            rule, action = action_codec_v4.decode_action(entry["action"])
        except ValueError as error:
            raise SemanticTraceShardError(
                f"step {index}: semantic action decoding failed"
            ) from error
        if entry["source_key"] != keys_payload[index]:
            raise SemanticTraceShardError(f"step {index}: source key disagrees")
        if entry["successor_key"] != keys_payload[index + 1]:
            raise SemanticTraceShardError(f"step {index}: successor key disagrees")
        if runtime is not None:
            try:
                successor = runtime.apply(states[index], rule, action)
            except ValueError as error:
                raise SemanticTraceShardError(
                    f"step {index}: semantic replay failed"
                ) from error
            if not _exact_state_equal(successor, states[index + 1]):
                raise SemanticTraceShardError(
                    f"step {index}: replay differs from the exact persisted successor"
                )
        steps.append(RewriteStep(rule, action))
        families[str(entry["action"]["model_family"])] += 1
    if record["family_histogram"] != dict(sorted(families.items())):
        raise SemanticTraceShardError("semantic trace family histogram disagrees")
    return RewriteTrace(
        source=states[0],
        target=states[-1],
        steps=tuple(steps),
        metadata=dict(record.get("metadata", {})),
    )


__all__ = [
    "MAX_ACTIVE_ATOMS",
    "TRACE_SCHEMA",
    "TRACE_SCHEMA_VERSION",
    "SemanticTraceShardError",
    "decode_semantic_trace_record",
    "encode_semantic_trace_record",
]
