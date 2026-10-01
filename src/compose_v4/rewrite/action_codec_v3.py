"""Editing-V2 action codec with explicit semantic cycle opening.

Version 2 remains the historical reader and writer for existing packed
artifacts. Version 3 changes the active cycle-opening executor identity from
``bond_delete``/``BondDelete`` to ``cycle_open``/``CycleOpenEdge``. It never
rewrites or silently upgrades a V2 record.

All unchanged payloads delegate to the strict V2 type-directed codec after an
explicit version translation. Raw ``bond_delete`` is intentionally excluded
from the V3 persistence surface even though the micro executor still exists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import is_dataclass
from typing import Any

from compose_v4.rewrite import action_codec as v2
from compose_v4.rewrite.operators import CycleOpenEdge

SCHEMA = v2.SCHEMA
SCHEMA_VERSION = 3
SEMANTIC_CYCLE_OPEN_RULE = "cycle_open"
SEMANTIC_CYCLE_OPEN_FAMILY = "cycle_attach"
SEMANTIC_CYCLE_OPEN_PAYLOAD = "CycleOpenEdge"


class ActionCodecV3Error(ValueError):
    """A V3 record violates the frozen Editing-V2 action ontology."""


def _strict_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionCodecV3Error(f"{path}: expected int, got {type(value).__name__}")
    return int(value)


def canonical_family(executor_rule: str) -> str:
    if executor_rule == SEMANTIC_CYCLE_OPEN_RULE:
        return SEMANTIC_CYCLE_OPEN_FAMILY
    if executor_rule == "bond_delete":
        raise ActionCodecV3Error(
            "raw bond_delete is an internal legacy micro rule and cannot be "
            "persisted as an Editing-V2 cycle-open action"
        )
    try:
        return v2.canonical_family(executor_rule)
    except v2.ActionCodecError as error:
        raise ActionCodecV3Error(str(error)) from error


def public_operator_name(family: str) -> str:
    return v2.public_operator_name(family)


def _encode_cycle_open(action: Any) -> dict[str, object]:
    if not is_dataclass(action) or type(action) is not CycleOpenEdge:
        raise ActionCodecV3Error(
            "executor rule 'cycle_open' requires payload CycleOpenEdge"
        )
    left = _strict_int(action.a, path="payload.a")
    right = _strict_int(action.b, path="payload.b")
    if left >= right:
        raise ActionCodecV3Error(
            "payload endpoints must be in strictly increasing undirected order"
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "executor_rule": SEMANTIC_CYCLE_OPEN_RULE,
        "model_family": SEMANTIC_CYCLE_OPEN_FAMILY,
        "payload_type": SEMANTIC_CYCLE_OPEN_PAYLOAD,
        "payload": {"a": left, "b": right},
    }


def encode_action(executor_rule: str, action: Any) -> dict[str, object]:
    """Encode one action under the prospective Editing-V2 ontology."""

    canonical_family(executor_rule)
    if executor_rule == SEMANTIC_CYCLE_OPEN_RULE:
        return _encode_cycle_open(action)
    try:
        record = v2.encode_action(executor_rule, action)
    except v2.ActionCodecError as error:
        raise ActionCodecV3Error(str(error)) from error
    return {**record, "schema_version": SCHEMA_VERSION}


def _validate_record_shape(record: object) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ActionCodecV3Error(
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
        raise ActionCodecV3Error(
            f"unexpected top-level field(s) {sorted(got - required)}"
        )
    if required - got:
        raise ActionCodecV3Error(f"missing top-level field(s) {sorted(required - got)}")
    if record["schema"] != SCHEMA:
        raise ActionCodecV3Error(
            f"unknown schema {record['schema']!r} (expected {SCHEMA!r})"
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise ActionCodecV3Error(
            f"unknown schema_version {record['schema_version']!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    return record


def _decode_cycle_open(record: dict[str, Any]) -> tuple[str, CycleOpenEdge]:
    if record["model_family"] != SEMANTIC_CYCLE_OPEN_FAMILY:
        raise ActionCodecV3Error(
            "ontology disagreement: cycle_open must map to cycle_attach"
        )
    if record["payload_type"] != SEMANTIC_CYCLE_OPEN_PAYLOAD:
        raise ActionCodecV3Error(
            "executor rule 'cycle_open' requires payload CycleOpenEdge"
        )
    payload = record["payload"]
    if not isinstance(payload, dict):
        raise ActionCodecV3Error(
            f"payload: expected object, got {type(payload).__name__}"
        )
    if set(payload) != {"a", "b"}:
        missing = {"a", "b"} - set(payload)
        extra = set(payload) - {"a", "b"}
        raise ActionCodecV3Error(
            f"cycle-open payload fields disagree: missing={sorted(missing)}, "
            f"unexpected={sorted(extra)}"
        )
    left = _strict_int(payload["a"], path="payload.a")
    right = _strict_int(payload["b"], path="payload.b")
    if left >= right:
        raise ActionCodecV3Error(
            "payload endpoints must be in strictly increasing undirected order"
        )
    return SEMANTIC_CYCLE_OPEN_RULE, CycleOpenEdge(left, right)


def decode_action(record: object) -> tuple[str, Any]:
    """Decode only V3 records; callers must use V2 for historical artifacts."""

    value = _validate_record_shape(record)
    executor_rule = value["executor_rule"]
    if not isinstance(executor_rule, str):
        raise ActionCodecV3Error("executor_rule must be a string")
    canonical_family(executor_rule)
    if executor_rule == SEMANTIC_CYCLE_OPEN_RULE:
        return _decode_cycle_open(value)
    translated = {**value, "schema_version": v2.SCHEMA_VERSION}
    try:
        return v2.decode_action(translated)
    except v2.ActionCodecError as error:
        raise ActionCodecV3Error(str(error)) from error


def canonical_json(record: dict[str, object]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def action_fingerprint(record: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(record).encode()).hexdigest()[:16]


def supported_executor_rules() -> tuple[str, ...]:
    return tuple(
        sorted(
            set(v2.supported_executor_rules()) - {"bond_delete"}
            | {SEMANTIC_CYCLE_OPEN_RULE}
        )
    )


def codec_implementation_hash() -> str:
    surface = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "base_v2_codec_implementation_hash": v2.codec_implementation_hash(),
        "supported_executor_rules": supported_executor_rules(),
        "semantic_cycle_open": {
            "rule": SEMANTIC_CYCLE_OPEN_RULE,
            "family": SEMANTIC_CYCLE_OPEN_FAMILY,
            "payload": SEMANTIC_CYCLE_OPEN_PAYLOAD,
            "fields": ("a", "b"),
            "endpoint_policy": "strictly_increasing",
        },
    }
    return hashlib.sha256(
        json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "ActionCodecV3Error",
    "action_fingerprint",
    "canonical_family",
    "canonical_json",
    "codec_implementation_hash",
    "decode_action",
    "encode_action",
    "public_operator_name",
    "supported_executor_rules",
]
