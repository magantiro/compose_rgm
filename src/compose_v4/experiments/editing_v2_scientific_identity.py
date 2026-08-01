"""Fail-closed scientific identity for Editing-V2 runs and checkpoints.

The identity deliberately excludes mutable implementation-provenance allowances.
It binds the semantic process, exact model modes, and every scientific artifact
that authorizes the bounded Editing-V2 training sequence. A checkpoint without
this complete object is legacy-only and cannot be interpreted as Editing V2.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from compose_v4.model.factorized_tracelet_rate_model import (
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    OperatorCapabilities,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_v2_process_identity,
)

SCIENTIFIC_IDENTITY_SCHEMA = "compose.editing_v2_scientific_identity"
SCIENTIFIC_IDENTITY_SCHEMA_VERSION = 1
SCIENTIFIC_IDENTITY_STATUS = "COMPLETE_EDITING_V2_SCIENTIFIC_IDENTITY"
CHECKPOINT_IDENTITY_FIELD = "editing_v2_scientific_identity"
PRODUCTIVE_TRAINING_OBJECT = "canonical_productive_molecular_successor_probability"

SEMANTIC_CYCLE_OPEN_SCORER_MODE = "pair_linear"

ARTIFACT_KEYS = (
    "semantic_corpus_global_registry",
    "active8_admission",
    "successor_cache",
    "sampling_sidecar",
    "gate_zero",
    "t1",
    "p50_recipe",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAPABILITY_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{16}$")
_ARTIFACT_FIELDS = frozenset({"physical_sha256", "semantic_sha256"})
_MODEL_FIELDS = frozenset(
    {
        "mark_dim",
        "operator_capability_fingerprint",
        "compute_ring_grow_support",
        "compute_ring_restates",
        "compute_cyclic_graft",
        "compute_ring_opening",
        "compute_ring_system_delete",
        "enable_cycle_ops",
        "use_aromatic_bond_view",
        "editing_process_semantics",
        "atom_restate_action_semantics",
        "ring_restate_scorer_mode",
        "cycle_close_action_semantics",
        "cycle_open_action_semantics",
        "cycle_open_scorer_mode",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "training_object",
        "process_identity",
        "model_identity",
        "artifact_identities",
        "scientific_identity_sha256",
    }
)


class EditingV2ScientificIdentityError(ValueError):
    """An Editing-V2 scientific identity is absent, incomplete, or mismatched."""


class LegacyOnlyCheckpointError(EditingV2ScientificIdentityError):
    """A checkpoint lacks the complete identity required for Editing V2."""


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic finite JSON bytes used for every identity hash."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingV2ScientificIdentityError(
            f"scientific identity is not finite deterministic JSON: {error}"
        ) from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_json_copy(value: object) -> Any:
    """Detach an identity from mutable cached production mappings."""

    return json.loads(canonical_json_bytes(value))


def _require_exact_mapping(
    value: object,
    *,
    fields: frozenset[str],
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingV2ScientificIdentityError(f"{field} must be an object")
    payload = dict(value)
    if set(payload) != fields:
        raise EditingV2ScientificIdentityError(
            f"{field} fields disagree; missing={sorted(fields - set(payload))}, "
            f"unexpected={sorted(set(payload) - fields)}"
        )
    return payload


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if _SHA256_RE.fullmatch(digest) is None:
        raise EditingV2ScientificIdentityError(
            f"{field} must be a full lowercase SHA-256"
        )
    return digest


def _normalize_artifact_identity(value: object, *, field: str) -> dict[str, str]:
    payload = _require_exact_mapping(value, fields=_ARTIFACT_FIELDS, field=field)
    return {
        "physical_sha256": _require_sha256(
            payload["physical_sha256"],
            field=f"{field}.physical_sha256",
        ),
        "semantic_sha256": _require_sha256(
            payload["semantic_sha256"],
            field=f"{field}.semantic_sha256",
        ),
    }


def _expected_capability_fingerprint(model: Mapping[str, Any]) -> str:
    capabilities = OperatorCapabilities(
        compute_ring_grow_support=model["compute_ring_grow_support"],
        compute_ring_restates=model["compute_ring_restates"],
        compute_cyclic_graft=model["compute_cyclic_graft"],
        compute_ring_opening=model["compute_ring_opening"],
        compute_ring_system_delete=model["compute_ring_system_delete"],
        editing_process_semantics=model["editing_process_semantics"],
        atom_restate_action_semantics=model["atom_restate_action_semantics"],
        ring_restate_scorer_mode=model["ring_restate_scorer_mode"],
        cycle_close_action_semantics=model["cycle_close_action_semantics"],
        cycle_open_action_semantics=model["cycle_open_action_semantics"],
    )
    return capabilities.fingerprint()


def _normalize_model_identity(value: object) -> dict[str, Any]:
    payload = _require_exact_mapping(
        value, fields=_MODEL_FIELDS, field="model_identity"
    )
    if type(payload["mark_dim"]) is not int or payload["mark_dim"] <= 0:
        raise EditingV2ScientificIdentityError(
            "model_identity.mark_dim must be positive"
        )
    for field in (
        "compute_ring_grow_support",
        "compute_ring_restates",
        "compute_cyclic_graft",
        "compute_ring_opening",
        "compute_ring_system_delete",
        "enable_cycle_ops",
        "use_aromatic_bond_view",
    ):
        if type(payload[field]) is not bool:
            raise EditingV2ScientificIdentityError(
                f"model_identity.{field} must be boolean"
            )

    expected_values = {
        "compute_ring_grow_support": False,
        "compute_ring_restates": True,
        "compute_cyclic_graft": True,
        "compute_ring_opening": True,
        "compute_ring_system_delete": False,
        "enable_cycle_ops": True,
        "use_aromatic_bond_view": True,
        "editing_process_semantics": SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
        "atom_restate_action_semantics": SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
        "ring_restate_scorer_mode": SEMANTIC_RING_RESTATE_SCORER_MODE,
        "cycle_close_action_semantics": SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
        "cycle_open_action_semantics": SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
        "cycle_open_scorer_mode": SEMANTIC_CYCLE_OPEN_SCORER_MODE,
    }
    mismatches = {
        field: {"expected": expected, "observed": payload[field]}
        for field, expected in expected_values.items()
        if payload[field] != expected
    }
    if mismatches:
        raise EditingV2ScientificIdentityError(
            f"model_identity is not the exact semantic Editing-V2 model: {mismatches}"
        )

    fingerprint = payload["operator_capability_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or _CAPABILITY_FINGERPRINT_RE.fullmatch(fingerprint) is None
    ):
        raise EditingV2ScientificIdentityError(
            "model_identity.operator_capability_fingerprint must be 16 lowercase hex characters"
        )
    expected_fingerprint = _expected_capability_fingerprint(payload)
    if fingerprint != expected_fingerprint:
        raise EditingV2ScientificIdentityError(
            "model_identity.operator_capability_fingerprint disagrees with the exact modes"
        )
    return payload


def _normalize_artifacts(value: object) -> dict[str, dict[str, str]]:
    fields = frozenset(ARTIFACT_KEYS)
    payload = _require_exact_mapping(value, fields=fields, field="artifact_identities")
    return {
        key: _normalize_artifact_identity(
            payload[key],
            field=f"artifact_identities.{key}",
        )
        for key in ARTIFACT_KEYS
    }


def _require_complete_identity_shape(payload: Mapping[str, Any]) -> None:
    """Classify every structurally incomplete identity as legacy-only."""

    nested_fields = (
        (
            "process_identity",
            payload.get("process_identity"),
            frozenset(editing_v2_process_identity()),
        ),
        ("model_identity", payload.get("model_identity"), _MODEL_FIELDS),
        (
            "artifact_identities",
            payload.get("artifact_identities"),
            frozenset(ARTIFACT_KEYS),
        ),
    )
    for field, value, fields in nested_fields:
        if not isinstance(value, Mapping) or set(value) != fields:
            observed = set(value) if isinstance(value, Mapping) else set()
            raise LegacyOnlyCheckpointError(
                f"complete Editing-V2 {field} is missing; artifact is legacy-only; "
                f"missing={sorted(fields - observed)}, "
                f"unexpected={sorted(observed - fields)}"
            )
    artifacts = payload["artifact_identities"]
    for key in ARTIFACT_KEYS:
        artifact = artifacts[key]
        if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_FIELDS:
            observed = set(artifact) if isinstance(artifact, Mapping) else set()
            raise LegacyOnlyCheckpointError(
                "complete Editing-V2 artifact identity is missing; "
                f"artifact is legacy-only; field=artifact_identities.{key}; "
                f"missing={sorted(_ARTIFACT_FIELDS - observed)}, "
                f"unexpected={sorted(observed - _ARTIFACT_FIELDS)}"
            )


def semantic_model_identity(
    *,
    mark_dim: int,
    operator_capability_fingerprint: str,
) -> dict[str, Any]:
    """Build the exact semantic model-mode identity for Editing V2."""

    return _normalize_model_identity(
        {
            "mark_dim": mark_dim,
            "operator_capability_fingerprint": operator_capability_fingerprint,
            "compute_ring_grow_support": False,
            "compute_ring_restates": True,
            "compute_cyclic_graft": True,
            "compute_ring_opening": True,
            "compute_ring_system_delete": False,
            "enable_cycle_ops": True,
            "use_aromatic_bond_view": True,
            "editing_process_semantics": SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
            "atom_restate_action_semantics": SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
            "ring_restate_scorer_mode": SEMANTIC_RING_RESTATE_SCORER_MODE,
            "cycle_close_action_semantics": SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
            "cycle_open_action_semantics": SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
            "cycle_open_scorer_mode": SEMANTIC_CYCLE_OPEN_SCORER_MODE,
        }
    )


def build_editing_v2_scientific_identity(
    *,
    model_identity: Mapping[str, Any],
    semantic_corpus_global_registry: Mapping[str, str],
    active8_admission: Mapping[str, str],
    successor_cache: Mapping[str, str],
    sampling_sidecar: Mapping[str, str],
    gate_zero: Mapping[str, str],
    t1: Mapping[str, str],
    p50_recipe: Mapping[str, str],
    process_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one complete Editing-V2 checkpoint/run scientific identity."""

    current_process = _canonical_json_copy(editing_v2_process_identity())
    supplied_process = (
        current_process
        if process_identity is None
        else _canonical_json_copy(process_identity)
    )
    if supplied_process != current_process:
        raise EditingV2ScientificIdentityError(
            "process_identity differs from the complete current Editing-V2 process"
        )
    artifacts = _normalize_artifacts(
        {
            "semantic_corpus_global_registry": semantic_corpus_global_registry,
            "active8_admission": active8_admission,
            "successor_cache": successor_cache,
            "sampling_sidecar": sampling_sidecar,
            "gate_zero": gate_zero,
            "t1": t1,
            "p50_recipe": p50_recipe,
        }
    )
    body = {
        "schema": SCIENTIFIC_IDENTITY_SCHEMA,
        "schema_version": SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
        "status": SCIENTIFIC_IDENTITY_STATUS,
        "training_object": PRODUCTIVE_TRAINING_OBJECT,
        "process_identity": current_process,
        "model_identity": _normalize_model_identity(model_identity),
        "artifact_identities": artifacts,
    }
    return {**body, "scientific_identity_sha256": canonical_sha256(body)}


def validate_editing_v2_scientific_identity(value: object) -> dict[str, Any]:
    """Validate and return the canonical complete identity, rejecting legacy forms."""

    if not isinstance(value, Mapping):
        raise LegacyOnlyCheckpointError(
            "complete Editing-V2 scientific identity is missing; artifact is legacy-only"
        )
    payload = dict(value)
    if set(payload) != _IDENTITY_FIELDS:
        raise LegacyOnlyCheckpointError(
            "complete Editing-V2 scientific identity fields are missing; artifact is legacy-only; "
            f"missing={sorted(_IDENTITY_FIELDS - set(payload))}, "
            f"unexpected={sorted(set(payload) - _IDENTITY_FIELDS)}"
        )
    _require_complete_identity_shape(payload)
    if (
        payload["schema"] != SCIENTIFIC_IDENTITY_SCHEMA
        or payload["schema_version"] != SCIENTIFIC_IDENTITY_SCHEMA_VERSION
        or payload["status"] != SCIENTIFIC_IDENTITY_STATUS
        or payload["training_object"] != PRODUCTIVE_TRAINING_OBJECT
    ):
        raise EditingV2ScientificIdentityError(
            "scientific identity schema, status, or training object is not Editing V2"
        )
    process_identity = _canonical_json_copy(payload["process_identity"])
    if process_identity != _canonical_json_copy(editing_v2_process_identity()):
        raise EditingV2ScientificIdentityError(
            "scientific identity process differs from the complete current Editing-V2 process"
        )
    model_identity = _normalize_model_identity(payload["model_identity"])
    artifacts = _normalize_artifacts(payload["artifact_identities"])
    body = {
        "schema": SCIENTIFIC_IDENTITY_SCHEMA,
        "schema_version": SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
        "status": SCIENTIFIC_IDENTITY_STATUS,
        "training_object": PRODUCTIVE_TRAINING_OBJECT,
        "process_identity": process_identity,
        "model_identity": model_identity,
        "artifact_identities": artifacts,
    }
    supplied_sha256 = _require_sha256(
        payload["scientific_identity_sha256"],
        field="scientific_identity_sha256",
    )
    if supplied_sha256 != canonical_sha256(body):
        raise EditingV2ScientificIdentityError("scientific identity SHA-256 mismatch")
    canonical = {**body, "scientific_identity_sha256": supplied_sha256}
    if payload != canonical:
        raise EditingV2ScientificIdentityError("scientific identity is not canonical")
    return canonical


def editing_v2_scientific_identities_equal(left: object, right: object) -> bool:
    """Return exact scientific equality after strict validation of both identities."""

    return validate_editing_v2_scientific_identity(
        left
    ) == validate_editing_v2_scientific_identity(right)


def require_editing_v2_scientific_identity_equal(
    expected: object,
    observed: object,
) -> dict[str, Any]:
    """Require exact equality; no process, corpus, or provenance waiver exists."""

    expected_identity = validate_editing_v2_scientific_identity(expected)
    observed_identity = validate_editing_v2_scientific_identity(observed)
    if expected_identity != observed_identity:
        raise EditingV2ScientificIdentityError(
            "Editing-V2 scientific identities differ; mismatch cannot be waived"
        )
    return observed_identity


def editing_v2_identity_from_checkpoint(
    checkpoint: object,
    *,
    field: str = CHECKPOINT_IDENTITY_FIELD,
) -> dict[str, Any]:
    """Extract a strict identity; missing or partial checkpoints are legacy-only."""

    if not isinstance(checkpoint, Mapping) or field not in checkpoint:
        raise LegacyOnlyCheckpointError(
            "checkpoint lacks a complete Editing-V2 scientific identity and is legacy-only"
        )
    return validate_editing_v2_scientific_identity(checkpoint[field])


__all__ = [
    "ARTIFACT_KEYS",
    "CHECKPOINT_IDENTITY_FIELD",
    "PRODUCTIVE_TRAINING_OBJECT",
    "SCIENTIFIC_IDENTITY_SCHEMA",
    "SCIENTIFIC_IDENTITY_SCHEMA_VERSION",
    "SCIENTIFIC_IDENTITY_STATUS",
    "SEMANTIC_CYCLE_OPEN_SCORER_MODE",
    "EditingV2ScientificIdentityError",
    "LegacyOnlyCheckpointError",
    "build_editing_v2_scientific_identity",
    "canonical_json_bytes",
    "canonical_sha256",
    "editing_v2_identity_from_checkpoint",
    "editing_v2_scientific_identities_equal",
    "require_editing_v2_scientific_identity_equal",
    "semantic_model_identity",
    "validate_editing_v2_scientific_identity",
]
