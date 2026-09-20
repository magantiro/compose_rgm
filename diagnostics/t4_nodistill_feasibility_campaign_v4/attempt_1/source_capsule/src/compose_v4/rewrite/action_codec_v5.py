"""Editing-V3 codec with one explicit protonation-state rewrite.

Version 5 extends the strict V4 Active8 surface with exactly one public action,
``atom_protonation_restate``.  V4 remains the historical reader/writer and is
never upgraded implicitly.  The new action addresses one persistent slot and a
named state from the fixed reversible tertiary-amine pair; it cannot encode raw
charge, hydrogen, element, or bond values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import is_dataclass
from typing import Any

from compose_v4.rewrite import action_codec_v4 as v4
from compose_v4.rewrite.operators import (
    PROTONATION_RESTATE_TARGET_STATES,
    AtomProtonationRestate,
)

SCHEMA = v4.SCHEMA
SCHEMA_VERSION = 5
PROTONATION_RESTATE_RULE = "atom_protonation_restate"
PROTONATION_RESTATE_FAMILY = "atom_protonation_restate"
PROTONATION_RESTATE_PAYLOAD = "AtomProtonationRestate"
ACTIVE9_EXECUTOR_RULES = (*v4.ACTIVE8_EXECUTOR_RULES, PROTONATION_RESTATE_RULE)


class ActionCodecV5Error(ValueError):
    """A V5 record violates the explicit Editing-V3 action ontology."""


def _strict_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionCodecV5Error(f"{path}: expected int, got {type(value).__name__}")
    return int(value)


def canonical_family(executor_rule: str) -> str:
    if executor_rule == PROTONATION_RESTATE_RULE:
        return PROTONATION_RESTATE_FAMILY
    try:
        return v4.canonical_family(executor_rule)
    except v4.ActionCodecV4Error as error:
        raise ActionCodecV5Error(str(error)) from error


def public_operator_name(family: str) -> str:
    if family == PROTONATION_RESTATE_FAMILY:
        return PROTONATION_RESTATE_RULE
    return v4.public_operator_name(family)


def _encode_protonation_restate(action: Any) -> dict[str, object]:
    if not is_dataclass(action) or type(action) is not AtomProtonationRestate:
        raise ActionCodecV5Error(
            "executor rule 'atom_protonation_restate' requires payload "
            "AtomProtonationRestate"
        )
    vertex = _strict_int(action.v, path="payload.v")
    if vertex < 0:
        raise ActionCodecV5Error("payload.v must be non-negative")
    if type(action.target_state) is not str:
        raise ActionCodecV5Error("payload.target_state must be a string")
    if action.target_state not in PROTONATION_RESTATE_TARGET_STATES:
        raise ActionCodecV5Error(
            "payload.target_state is outside the fixed tertiary-amine pair"
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "executor_rule": PROTONATION_RESTATE_RULE,
        "model_family": PROTONATION_RESTATE_FAMILY,
        "payload_type": PROTONATION_RESTATE_PAYLOAD,
        "payload": {"v": vertex, "target_state": action.target_state},
    }


def encode_action(executor_rule: str, action: Any) -> dict[str, object]:
    """Encode one action under the strict Editing-V3 support surface."""

    canonical_family(executor_rule)
    if executor_rule == PROTONATION_RESTATE_RULE:
        return _encode_protonation_restate(action)
    try:
        record = v4.encode_action(executor_rule, action)
    except v4.ActionCodecV4Error as error:
        raise ActionCodecV5Error(str(error)) from error
    return {**record, "schema_version": SCHEMA_VERSION}


def _validate_record_shape(record: object) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ActionCodecV5Error(
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
        raise ActionCodecV5Error(
            f"unexpected top-level field(s) {sorted(got - required)}"
        )
    if required - got:
        raise ActionCodecV5Error(f"missing top-level field(s) {sorted(required - got)}")
    if record["schema"] != SCHEMA:
        raise ActionCodecV5Error(
            f"unknown schema {record['schema']!r} (expected {SCHEMA!r})"
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise ActionCodecV5Error(
            f"unknown schema_version {record['schema_version']!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    return record


def _decode_protonation_restate(
    record: dict[str, Any],
) -> tuple[str, AtomProtonationRestate]:
    if record["model_family"] != PROTONATION_RESTATE_FAMILY:
        raise ActionCodecV5Error(
            "ontology disagreement: atom_protonation_restate must map to itself"
        )
    if record["payload_type"] != PROTONATION_RESTATE_PAYLOAD:
        raise ActionCodecV5Error(
            "executor rule 'atom_protonation_restate' requires payload "
            "AtomProtonationRestate"
        )
    payload = record["payload"]
    if not isinstance(payload, dict):
        raise ActionCodecV5Error(
            f"payload: expected object, got {type(payload).__name__}"
        )
    expected = {"v", "target_state"}
    if set(payload) != expected:
        raise ActionCodecV5Error(
            "protonation-restatement payload fields disagree: "
            f"missing={sorted(expected - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected)}"
        )
    vertex = _strict_int(payload["v"], path="payload.v")
    if vertex < 0:
        raise ActionCodecV5Error("payload.v must be non-negative")
    target_state = payload["target_state"]
    if type(target_state) is not str:
        raise ActionCodecV5Error("payload.target_state must be a string")
    if target_state not in PROTONATION_RESTATE_TARGET_STATES:
        raise ActionCodecV5Error(
            "payload.target_state is outside the fixed tertiary-amine pair"
        )
    return PROTONATION_RESTATE_RULE, AtomProtonationRestate(vertex, target_state)


def decode_action(record: object) -> tuple[str, Any]:
    """Decode only V5 records; historical codecs remain explicit readers."""

    value = _validate_record_shape(record)
    executor_rule = value["executor_rule"]
    if not isinstance(executor_rule, str):
        raise ActionCodecV5Error("executor_rule must be a string")
    canonical_family(executor_rule)
    if executor_rule == PROTONATION_RESTATE_RULE:
        return _decode_protonation_restate(value)
    translated = {**value, "schema_version": v4.SCHEMA_VERSION}
    try:
        return v4.decode_action(translated)
    except v4.ActionCodecV4Error as error:
        raise ActionCodecV5Error(str(error)) from error


def canonical_json(record: dict[str, object]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def action_fingerprint(record: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(record).encode()).hexdigest()[:16]


def supported_executor_rules() -> tuple[str, ...]:
    return ACTIVE9_EXECUTOR_RULES


def codec_implementation_hash() -> str:
    surface = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "base_v4_codec_implementation_hash": v4.codec_implementation_hash(),
        "supported_executor_rules": supported_executor_rules(),
        "atom_protonation_restate": {
            "rule": PROTONATION_RESTATE_RULE,
            "family": PROTONATION_RESTATE_FAMILY,
            "payload": PROTONATION_RESTATE_PAYLOAD,
            "fields": ("v", "target_state"),
            "target_states": PROTONATION_RESTATE_TARGET_STATES,
        },
    }
    return hashlib.sha256(
        json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


__all__ = [
    "ACTIVE9_EXECUTOR_RULES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "ActionCodecV5Error",
    "action_fingerprint",
    "canonical_family",
    "canonical_json",
    "codec_implementation_hash",
    "decode_action",
    "encode_action",
    "public_operator_name",
    "supported_executor_rules",
]
