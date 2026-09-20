"""RewriteActionCodecV2 -- strict, versioned, semantic (de)serialization of executable rewrite actions.

WHY THIS EXISTS. The trace-pool format historically encoded only ``atom_insert``/``atom_delete`` (see
``experiments/analogue_prior.py``); every other production family raised. Corruption traces use seven-plus
families, so corruption could never be persisted and was regenerated in memory at every launch -- which is
why launches carried preflight-scale record counts and the trained prior was data-starved. This codec is
the production data-path fix:

    MISSING_COMPLETE_ACTION_CODEC -> IN_MEMORY_SERIAL_CORRUPTION -> PREFLIGHT_SCALE_DATA
                                 -> SCIENTIFIC_DATA_STARVATION

DESIGN CONSTRAINTS (deliberate, do not relax):
  * NOT a generic dataclass dump. Payload types come from an explicit ALLOWLIST; the persisted schema names
    types by a stable string, never by Python module path, so refactors cannot silently change the format.
  * ``dataclasses.asdict`` is NOT used: it recurses into nested dataclasses AND erases tuple-vs-list, both
    of which matter here (actions compare by equality, and tuples are semantic).
  * No pickle, no dynamic import of serialized module paths.
  * Type-DIRECTED decoding: every allowlisted payload declares an explicit field-type spec, so decoding
    reconstructs exact Python types (tuple, int, str) rather than trusting the wire form.
  * Unknown schema/version/type/rule, missing fields, extra fields, or an ontology disagreement all FAIL
    LOUDLY. No silent defaults, no repair, no lossy normalization.

ONTOLOGY. Executor rule names and model FAMILY names are distinct namespaces and must never be conflated:
the compositional cycle operators execute as ``bond_insert``/``bond_delete`` but score under the model
families ``cycle_insert``/``cycle_attach``. Confusingly, ``cycle_insert``/``cycle_attach`` ALSO exist as
legacy executor rules -- those are dead null-prior operators and are rejected here (deviation D1). Decode
asserts ``canonical_family(executor_rule) == model_family``.

``ring_system_grow`` is DISABLED under the RingCore-V1 production contract and is rejected by the writer;
it may not be persisted as an active production family.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from typing import Any

from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.tracelets import (
    AtomPayload,
    BondOrderChange,
    RingBond,
    RingSystemDelete,
    RingSystemRestate,
)

SCHEMA = "compose.rewrite.action"
SCHEMA_VERSION = 2

# ---- Ontology: executor rule -> model family (authoritative) ----------------------------------------
# Identity everywhere except the compositional cycle operators, which execute as bond ops but score under
# the cycle family slots. Mirrors _CYCLE_OP_EXECUTOR_TO_FAMILY in the rate model.
_EXECUTOR_TO_FAMILY: dict[str, str] = {
    "atom_insert": "atom_insert",
    "atom_delete": "atom_delete",
    "atom_restate": "atom_restate",
    "bond_reorder": "bond_reorder",
    "bond_reroute": "bond_reroute",
    "bond_insert": "cycle_insert",   # cycle_close
    "bond_delete": "cycle_attach",   # cycle_open
    "ring_system_delete": "ring_system_delete",
    "ring_system_restate": "ring_system_restate",
}

# Public operator names (paper-facing); internal slot names are engineering aliases.
_FAMILY_PUBLIC_NAME: dict[str, str] = {
    "cycle_insert": "cycle_close",
    "cycle_attach": "cycle_open",
    "ring_system_delete": "ring_delete",
    "ring_system_restate": "ring_aromaticity_restate",
}

# Disabled under the RingCore-V1 contract -- never persist as an active family.
DISABLED_EXECUTOR_RULES = frozenset({"ring_system_grow"})
# Legacy null-prior executor rules, dead for B and B-edit. Their NAMES collide with live model families;
# rejecting them explicitly prevents that collision from becoming a silent mis-encode (deviation D1).
LEGACY_DEAD_EXECUTOR_RULES = frozenset({
    "cycle_insert", "cycle_attach", "cycle_delete", "cycle_detach",
    "ring_ear_insert", "ring_ear_delete",
})

# ---- Allowlisted payload types + explicit field-type specs -------------------------------------------
# Type language: "int" | "str" | ("tuple", <spec>) | ("dc", "<TypeName>")
_INT = "int"
_STR = "str"
_PAIR = ("tuple", _INT)  # tuple[int, int] encodes as a 2-element array

_PAYLOAD_TYPES: dict[str, type] = {
    "AtomInsert": AtomInsert,
    "AtomDelete": AtomDelete,
    "AtomRestate": AtomRestate,
    "BondInsert": BondInsert,
    "BondDelete": BondDelete,
    "BondReorder": BondReorder,
    "BondReroute": BondReroute,
    "RingSystemDelete": RingSystemDelete,
    "RingSystemRestate": RingSystemRestate,
    # nested payloads
    "AtomPayload": AtomPayload,
    "RingBond": RingBond,
    "BondOrderChange": BondOrderChange,
}

_FIELD_SPECS: dict[str, tuple[tuple[str, Any], ...]] = {
    "AtomInsert": (
        ("slot", _INT), ("atom_type", _INT), ("formal_charge", _INT),
        ("implicit_h_count", _INT), ("neighbors", ("tuple", _PAIR)),
    ),
    "AtomDelete": (("v", _INT),),
    "AtomRestate": (
        ("v", _INT), ("atom_type", _INT), ("formal_charge", _INT), ("implicit_h_count", _INT),
    ),
    "BondInsert": (("a", _INT), ("b", _INT), ("order", _INT)),
    "BondDelete": (("a", _INT), ("b", _INT)),
    "BondReorder": (("a", _INT), ("b", _INT), ("new_order", _INT)),
    "BondReroute": (("a", _INT), ("b", _INT), ("u", _INT), ("v", _INT), ("new_order", _INT)),
    "AtomPayload": (
        ("slot", _INT), ("atom_type", _INT), ("formal_charge", _INT), ("implicit_h_count", _INT),
    ),
    "RingBond": (("a", _INT), ("b", _INT), ("order", _INT)),
    "BondOrderChange": (("a", _INT), ("b", _INT), ("new_order", _INT)),
    "RingSystemRestate": (("changes", ("tuple", ("dc", "BondOrderChange"))),),
    "RingSystemDelete": (
        ("system_atoms", ("tuple", _INT)),
        ("retained_system_atoms", ("tuple", _INT)),
        ("bond_deletions", ("tuple", ("dc", "RingBond"))),
        ("atom_deletions", ("tuple", _INT)),
        ("atom_payloads", ("tuple", ("dc", "AtomPayload"))),
        ("bond_reorders", ("tuple", ("dc", "BondOrderChange"))),
        ("source_aromatic_edges", ("tuple", _PAIR)),
        ("aromatic_edges", ("tuple", _PAIR)),
        ("topology_class", _STR),
    ),
}

# executor rule -> the payload type it must carry
_EXECUTOR_TO_PAYLOAD: dict[str, str] = {
    "atom_insert": "AtomInsert",
    "atom_delete": "AtomDelete",
    "atom_restate": "AtomRestate",
    "bond_insert": "BondInsert",
    "bond_delete": "BondDelete",
    "bond_reorder": "BondReorder",
    "bond_reroute": "BondReroute",
    "ring_system_delete": "RingSystemDelete",
    "ring_system_restate": "RingSystemRestate",
}
_TYPE_TO_NAME: dict[type, str] = {v: k for k, v in _PAYLOAD_TYPES.items()}


class ActionCodecError(ValueError):
    """Any encode/decode violation. Always loud -- never repaired."""


def canonical_family(executor_rule: str) -> str:
    """Authoritative executor-rule -> model-family mapping. Raises on disabled/legacy/unknown rules."""
    if executor_rule in DISABLED_EXECUTOR_RULES:
        raise ActionCodecError(
            f"executor rule {executor_rule!r} is DISABLED under the RingCore-V1 contract and must not be "
            f"persisted as an active production family"
        )
    if executor_rule in LEGACY_DEAD_EXECUTOR_RULES:
        raise ActionCodecError(
            f"executor rule {executor_rule!r} is a legacy null-prior operator, dead in production. Note its "
            f"name collides with a live MODEL FAMILY: the compositional cycle ops execute as "
            f"bond_insert/bond_delete and score under families cycle_insert/cycle_attach."
        )
    try:
        return _EXECUTOR_TO_FAMILY[executor_rule]
    except KeyError:
        raise ActionCodecError(f"unknown executor rule {executor_rule!r}") from None


def public_operator_name(family: str) -> str:
    """Paper-facing operator name for a model family (internal slot names are engineering aliases)."""
    return _FAMILY_PUBLIC_NAME.get(family, family)


# ---- Recursive value codec (type-directed) ------------------------------------------------------------


def _encode_value(value: Any, spec: Any, path: str) -> Any:
    if spec == _INT:
        # bool is a subclass of int; reject it so True/1 can never round-trip ambiguously
        if isinstance(value, bool) or not isinstance(value, int):
            raise ActionCodecError(f"{path}: expected int, got {type(value).__name__}")
        return int(value)
    if spec == _STR:
        if not isinstance(value, str):
            raise ActionCodecError(f"{path}: expected str, got {type(value).__name__}")
        return value
    if isinstance(spec, tuple) and spec[0] == "tuple":
        if not isinstance(value, tuple):
            raise ActionCodecError(
                f"{path}: expected tuple (tuple-vs-list is semantic here), got {type(value).__name__}"
            )
        return [_encode_value(v, spec[1], f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(spec, tuple) and spec[0] == "dc":
        return _encode_dataclass(value, spec[1], path)
    raise ActionCodecError(f"{path}: unknown field spec {spec!r}")


def _decode_value(value: Any, spec: Any, path: str) -> Any:
    if spec == _INT:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ActionCodecError(f"{path}: expected int, got {type(value).__name__}")
        return int(value)
    if spec == _STR:
        if not isinstance(value, str):
            raise ActionCodecError(f"{path}: expected str, got {type(value).__name__}")
        return value
    if isinstance(spec, tuple) and spec[0] == "tuple":
        if not isinstance(value, list):
            raise ActionCodecError(f"{path}: expected JSON array, got {type(value).__name__}")
        return tuple(_decode_value(v, spec[1], f"{path}[{i}]") for i, v in enumerate(value))
    if isinstance(spec, tuple) and spec[0] == "dc":
        return _decode_dataclass(value, spec[1], path)
    raise ActionCodecError(f"{path}: unknown field spec {spec!r}")


def _encode_dataclass(obj: Any, type_name: str, path: str) -> dict:
    cls = _PAYLOAD_TYPES.get(type_name)
    if cls is None:
        raise ActionCodecError(f"{path}: {type_name!r} is not an allowlisted payload type")
    if not isinstance(obj, cls):
        raise ActionCodecError(f"{path}: expected {type_name}, got {type(obj).__name__}")
    spec = _FIELD_SPECS[type_name]
    declared = {name for name, _ in spec}
    actual = {f.name for f in fields(cls)}
    if declared != actual:
        raise ActionCodecError(
            f"{path}: field-spec drift for {type_name}: spec={sorted(declared)} dataclass={sorted(actual)}. "
            f"The codec spec must be updated in lockstep with the dataclass."
        )
    return {name: _encode_value(getattr(obj, name), fspec, f"{path}.{name}") for name, fspec in spec}


def _decode_dataclass(payload: Any, type_name: str, path: str) -> Any:
    cls = _PAYLOAD_TYPES.get(type_name)
    if cls is None:
        raise ActionCodecError(f"{path}: {type_name!r} is not an allowlisted payload type")
    if not isinstance(payload, dict):
        raise ActionCodecError(f"{path}: expected object for {type_name}, got {type(payload).__name__}")
    spec = _FIELD_SPECS[type_name]
    expected = {name for name, _ in spec}
    got = set(payload)
    if got - expected:
        raise ActionCodecError(f"{path}: unexpected field(s) {sorted(got - expected)} for {type_name}")
    if expected - got:
        raise ActionCodecError(f"{path}: missing field(s) {sorted(expected - got)} for {type_name}")
    kwargs = {name: _decode_value(payload[name], fspec, f"{path}.{name}") for name, fspec in spec}
    return cls(**kwargs)


# ---- Public API ---------------------------------------------------------------------------------------


def encode_action(executor_rule: str, action: Any) -> dict:
    """Encode one executable action into the versioned V2 wire record. Raises on any violation."""
    family = canonical_family(executor_rule)  # rejects disabled/legacy/unknown
    expected_payload = _EXECUTOR_TO_PAYLOAD[executor_rule]
    if not is_dataclass(action):
        raise ActionCodecError(f"action for {executor_rule!r} is not a dataclass: {type(action).__name__}")
    actual_payload = _TYPE_TO_NAME.get(type(action))
    if actual_payload is None:
        raise ActionCodecError(
            f"payload type {type(action).__name__!r} is not allowlisted for persistence"
        )
    if actual_payload != expected_payload:
        raise ActionCodecError(
            f"executor rule {executor_rule!r} requires payload {expected_payload}, got {actual_payload}"
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "executor_rule": executor_rule,
        "model_family": family,
        "payload_type": actual_payload,
        "payload": _encode_dataclass(action, actual_payload, "payload"),
    }


def decode_action(record: dict) -> tuple[str, Any]:
    """Decode a V2 wire record to ``(executor_rule, action)``. Raises on any violation."""
    if not isinstance(record, dict):
        raise ActionCodecError(f"action record must be an object, got {type(record).__name__}")
    required = {"schema", "schema_version", "executor_rule", "model_family", "payload_type", "payload"}
    got = set(record)
    if got - required:
        raise ActionCodecError(f"unexpected top-level field(s) {sorted(got - required)}")
    if required - got:
        raise ActionCodecError(f"missing top-level field(s) {sorted(required - got)}")
    if record["schema"] != SCHEMA:
        raise ActionCodecError(f"unknown schema {record['schema']!r} (expected {SCHEMA!r})")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ActionCodecError(
            f"unknown schema_version {record['schema_version']!r} (expected {SCHEMA_VERSION})"
        )
    executor_rule = record["executor_rule"]
    if not isinstance(executor_rule, str):
        raise ActionCodecError("executor_rule must be a string")
    family = canonical_family(executor_rule)  # rejects disabled/legacy/unknown
    if record["model_family"] != family:
        raise ActionCodecError(
            f"ontology disagreement: executor_rule {executor_rule!r} maps to family {family!r} but the "
            f"record declares {record['model_family']!r}"
        )
    expected_payload = _EXECUTOR_TO_PAYLOAD[executor_rule]
    if record["payload_type"] != expected_payload:
        raise ActionCodecError(
            f"executor rule {executor_rule!r} requires payload {expected_payload}, record declares "
            f"{record['payload_type']!r}"
        )
    action = _decode_dataclass(record["payload"], expected_payload, "payload")
    return executor_rule, action


def canonical_json(record: dict) -> str:
    """Deterministic byte-stable encoding: encode(decode(encode(a))) == encode(a) byte-for-byte."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def action_fingerprint(record: dict) -> str:
    """Stable semantic fingerprint of an encoded action, for provenance and debugging ONLY.

    NOT a replacement for action equality -- the teacher-in-candidate invariant still compares actions
    directly. Two actions with the same fingerprint are byte-identical under canonical encoding."""
    return hashlib.sha256(canonical_json(record).encode()).hexdigest()[:16]


def codec_implementation_hash() -> str:
    """Hash of the codec's semantic surface (ontology + allowlist + field specs). Persisted in shard
    manifests so a format change is detectable without diffing code."""
    surface = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "executor_to_family": dict(sorted(_EXECUTOR_TO_FAMILY.items())),
        "executor_to_payload": dict(sorted(_EXECUTOR_TO_PAYLOAD.items())),
        "field_specs": {k: [[n, repr(s)] for n, s in v] for k, v in sorted(_FIELD_SPECS.items())},
        "disabled": sorted(DISABLED_EXECUTOR_RULES),
        "legacy_dead": sorted(LEGACY_DEAD_EXECUTOR_RULES),
    }
    return hashlib.sha256(
        json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def supported_executor_rules() -> tuple[str, ...]:
    """Active production executor rules this codec persists."""
    return tuple(sorted(_EXECUTOR_TO_PAYLOAD))
