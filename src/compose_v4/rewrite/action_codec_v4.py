"""Complete Editing-V2 semantic-action codec.

Version 4 replaces the persisted raw ``bond_insert``/``BondInsert`` cycle
closure with ``cycle_close``/``CycleCloseEdge`` and raw atom restatement with
``atom_restate_semantic``/``SemanticAtomRestate``. Version 3 and Version 2
remain strict historical readers and writers. No record is silently upgraded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import is_dataclass
from typing import Any

from compose_v4.rewrite import action_codec_v3 as v3
from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.rewrite.operators import CycleCloseEdge, SemanticAtomRestate

SCHEMA = v3.SCHEMA
SCHEMA_VERSION = 4
SEMANTIC_CYCLE_CLOSE_RULE = "cycle_close"
SEMANTIC_CYCLE_CLOSE_FAMILY = "cycle_insert"
SEMANTIC_CYCLE_CLOSE_PAYLOAD = "CycleCloseEdge"
SEMANTIC_ATOM_RESTATE_RULE = "atom_restate_semantic"
SEMANTIC_ATOM_RESTATE_FAMILY = "atom_restate"
SEMANTIC_ATOM_RESTATE_PAYLOAD = "SemanticAtomRestate"
ACTIVE8_EXECUTOR_RULES = (
    "atom_delete",
    "atom_insert",
    SEMANTIC_ATOM_RESTATE_RULE,
    "bond_reorder",
    "bond_reroute",
    SEMANTIC_CYCLE_CLOSE_RULE,
    "cycle_open",
    "ring_system_restate",
)


class ActionCodecV4Error(ValueError):
    """A V4 record violates the frozen Editing-V2 action ontology."""


def _strict_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionCodecV4Error(f"{path}: expected int, got {type(value).__name__}")
    return int(value)


def canonical_family(executor_rule: str) -> str:
    if executor_rule not in ACTIVE8_EXECUTOR_RULES:
        raise ActionCodecV4Error(
            f"executor rule {executor_rule!r} is outside the frozen Active8 "
            "Editing-V2 persistence surface"
        )
    if executor_rule == SEMANTIC_ATOM_RESTATE_RULE:
        return SEMANTIC_ATOM_RESTATE_FAMILY
    if executor_rule == SEMANTIC_CYCLE_CLOSE_RULE:
        return SEMANTIC_CYCLE_CLOSE_FAMILY
    try:
        return v3.canonical_family(executor_rule)
    except v3.ActionCodecV3Error as error:
        raise ActionCodecV4Error(str(error)) from error


def public_operator_name(family: str) -> str:
    return v3.public_operator_name(family)


def _encode_cycle_close(action: Any) -> dict[str, object]:
    if not is_dataclass(action) or type(action) is not CycleCloseEdge:
        raise ActionCodecV4Error(
            "executor rule 'cycle_close' requires payload CycleCloseEdge"
        )
    left = _strict_int(action.a, path="payload.a")
    right = _strict_int(action.b, path="payload.b")
    order = _strict_int(action.order, path="payload.order")
    if left >= right:
        raise ActionCodecV4Error(
            "payload endpoints must be in strictly increasing undirected order"
        )
    if order not in {1, 2, 3}:
        raise ActionCodecV4Error("payload.order must be one of 1, 2, or 3")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "executor_rule": SEMANTIC_CYCLE_CLOSE_RULE,
        "model_family": SEMANTIC_CYCLE_CLOSE_FAMILY,
        "payload_type": SEMANTIC_CYCLE_CLOSE_PAYLOAD,
        "payload": {"a": left, "b": right, "order": order},
    }


def _encode_semantic_atom_restate(action: Any) -> dict[str, object]:
    if not is_dataclass(action) or type(action) is not SemanticAtomRestate:
        raise ActionCodecV4Error(
            "executor rule 'atom_restate_semantic' requires payload SemanticAtomRestate"
        )
    vertex = _strict_int(action.v, path="payload.v")
    target_class_index = _strict_int(
        action.target_class_index,
        path="payload.target_class_index",
    )
    if vertex < 0:
        raise ActionCodecV4Error("payload.v must be non-negative")
    if not 0 <= target_class_index < len(ORGANIC_VOCABULARY):
        raise ActionCodecV4Error(
            "payload.target_class_index is outside the broad-organic vocabulary"
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "executor_rule": SEMANTIC_ATOM_RESTATE_RULE,
        "model_family": SEMANTIC_ATOM_RESTATE_FAMILY,
        "payload_type": SEMANTIC_ATOM_RESTATE_PAYLOAD,
        "payload": {
            "v": vertex,
            "target_class_index": target_class_index,
        },
    }


def encode_action(executor_rule: str, action: Any) -> dict[str, object]:
    """Encode one action under the complete Editing-V2 cycle ontology."""

    canonical_family(executor_rule)
    if executor_rule == SEMANTIC_ATOM_RESTATE_RULE:
        return _encode_semantic_atom_restate(action)
    if executor_rule == SEMANTIC_CYCLE_CLOSE_RULE:
        return _encode_cycle_close(action)
    try:
        record = v3.encode_action(executor_rule, action)
    except v3.ActionCodecV3Error as error:
        raise ActionCodecV4Error(str(error)) from error
    return {**record, "schema_version": SCHEMA_VERSION}


def _validate_record_shape(record: object) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ActionCodecV4Error(
            f"action record must be an object, got {type(record).__name__}"
        )
    required = {
        "schema",
        "schema_version",
        "executor_rule",
        "model_family",
        "payload_type",
        "payload",
    }
    got = set(record)
    if got - required:
        raise ActionCodecV4Error(
            f"unexpected top-level field(s) {sorted(got - required)}"
        )
    if required - got:
        raise ActionCodecV4Error(f"missing top-level field(s) {sorted(required - got)}")
    if record["schema"] != SCHEMA:
        raise ActionCodecV4Error(
            f"unknown schema {record['schema']!r} (expected {SCHEMA!r})"
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise ActionCodecV4Error(
            f"unknown schema_version {record['schema_version']!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    return record


def _decode_cycle_close(record: dict[str, Any]) -> tuple[str, CycleCloseEdge]:
    if record["model_family"] != SEMANTIC_CYCLE_CLOSE_FAMILY:
        raise ActionCodecV4Error(
            "ontology disagreement: cycle_close must map to cycle_insert"
        )
    if record["payload_type"] != SEMANTIC_CYCLE_CLOSE_PAYLOAD:
        raise ActionCodecV4Error(
            "executor rule 'cycle_close' requires payload CycleCloseEdge"
        )
    payload = record["payload"]
    if not isinstance(payload, dict):
        raise ActionCodecV4Error(
            f"payload: expected object, got {type(payload).__name__}"
        )
    expected = {"a", "b", "order"}
    if set(payload) != expected:
        raise ActionCodecV4Error(
            "cycle-close payload fields disagree: "
            f"missing={sorted(expected - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected)}"
        )
    left = _strict_int(payload["a"], path="payload.a")
    right = _strict_int(payload["b"], path="payload.b")
    order = _strict_int(payload["order"], path="payload.order")
    if left >= right:
        raise ActionCodecV4Error(
            "payload endpoints must be in strictly increasing undirected order"
        )
    if order not in {1, 2, 3}:
        raise ActionCodecV4Error("payload.order must be one of 1, 2, or 3")
    return SEMANTIC_CYCLE_CLOSE_RULE, CycleCloseEdge(left, right, order)


def _decode_semantic_atom_restate(
    record: dict[str, Any],
) -> tuple[str, SemanticAtomRestate]:
    if record["model_family"] != SEMANTIC_ATOM_RESTATE_FAMILY:
        raise ActionCodecV4Error(
            "ontology disagreement: atom_restate_semantic must map to atom_restate"
        )
    if record["payload_type"] != SEMANTIC_ATOM_RESTATE_PAYLOAD:
        raise ActionCodecV4Error(
            "executor rule 'atom_restate_semantic' requires payload SemanticAtomRestate"
        )
    payload = record["payload"]
    if not isinstance(payload, dict):
        raise ActionCodecV4Error(
            f"payload: expected object, got {type(payload).__name__}"
        )
    expected = {"v", "target_class_index"}
    if set(payload) != expected:
        raise ActionCodecV4Error(
            "semantic atom-restatement payload fields disagree: "
            f"missing={sorted(expected - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected)}"
        )
    vertex = _strict_int(payload["v"], path="payload.v")
    target_class_index = _strict_int(
        payload["target_class_index"],
        path="payload.target_class_index",
    )
    if vertex < 0:
        raise ActionCodecV4Error("payload.v must be non-negative")
    if not 0 <= target_class_index < len(ORGANIC_VOCABULARY):
        raise ActionCodecV4Error(
            "payload.target_class_index is outside the broad-organic vocabulary"
        )
    return SEMANTIC_ATOM_RESTATE_RULE, SemanticAtomRestate(
        vertex,
        target_class_index,
    )


def decode_action(record: object) -> tuple[str, Any]:
    """Decode only V4 records; callers must select older readers explicitly."""

    value = _validate_record_shape(record)
    executor_rule = value["executor_rule"]
    if not isinstance(executor_rule, str):
        raise ActionCodecV4Error("executor_rule must be a string")
    canonical_family(executor_rule)
    if executor_rule == SEMANTIC_ATOM_RESTATE_RULE:
        return _decode_semantic_atom_restate(value)
    if executor_rule == SEMANTIC_CYCLE_CLOSE_RULE:
        return _decode_cycle_close(value)
    translated = {**value, "schema_version": v3.SCHEMA_VERSION}
    try:
        return v3.decode_action(translated)
    except v3.ActionCodecV3Error as error:
        raise ActionCodecV4Error(str(error)) from error


def canonical_json(record: dict[str, object]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def action_fingerprint(record: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(record).encode()).hexdigest()[:16]


def supported_executor_rules() -> tuple[str, ...]:
    return ACTIVE8_EXECUTOR_RULES


def codec_implementation_hash() -> str:
    surface = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "base_v3_codec_implementation_hash": v3.codec_implementation_hash(),
        "supported_executor_rules": supported_executor_rules(),
        "semantic_cycle_close": {
            "rule": SEMANTIC_CYCLE_CLOSE_RULE,
            "family": SEMANTIC_CYCLE_CLOSE_FAMILY,
            "payload": SEMANTIC_CYCLE_CLOSE_PAYLOAD,
            "fields": ("a", "b", "order"),
            "endpoint_policy": "strictly_increasing",
            "order_policy": (1, 2, 3),
        },
        "semantic_atom_restate": {
            "rule": SEMANTIC_ATOM_RESTATE_RULE,
            "family": SEMANTIC_ATOM_RESTATE_FAMILY,
            "payload": SEMANTIC_ATOM_RESTATE_PAYLOAD,
            "fields": ("v", "target_class_index"),
            "target_vocabulary": tuple(ORGANIC_VOCABULARY.classes),
        },
    }
    return hashlib.sha256(
        json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


__all__ = [
    "ACTIVE8_EXECUTOR_RULES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "ActionCodecV4Error",
    "action_fingerprint",
    "canonical_family",
    "canonical_json",
    "codec_implementation_hash",
    "decode_action",
    "encode_action",
    "public_operator_name",
    "supported_executor_rules",
]
