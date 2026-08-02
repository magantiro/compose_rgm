"""Fail-closed scientific identity for Editing-V2 runs and checkpoints.

The identity deliberately excludes mutable implementation-provenance allowances.
It binds the semantic process, exact model modes, and every scientific artifact
that authorizes the bounded Editing-V2 training sequence. A checkpoint without
this complete object is legacy-only and cannot be interpreted as Editing V2.

Two semantic process versions of the editing lane are expressible, each under its
own frozen contract:

``compose.editing_v2_scientific_identity``
    Process V1. Its schema, status, field set, expected modes, body shape, and
    therefore its ``scientific_identity_sha256`` derivation are UNCHANGED. A
    historical V1 payload validates byte-for-byte as it always did.

``compose.editing_v2_scientific_identity.process_v2``
    Process V2. It binds the Process-V2 process identity and additionally binds
    ``atom_delete_action_semantics``, which the V1 contract does not carry at all.

The two contracts are disjoint: a V1 payload cannot declare the atom-delete mode
and a V2 payload must. Selecting the contract by schema, and requiring the model
identity's own process semantics to select the SAME contract, is what makes a
mixed V1/V2 identity a loud failure rather than a silent legacy default.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from compose_v4.model.factorized_tracelet_rate_model import (
    LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    OperatorCapabilities,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_process_v2_identity,
    editing_v2_process_identity,
)

SCIENTIFIC_IDENTITY_SCHEMA = "compose.editing_v2_scientific_identity"
SCIENTIFIC_IDENTITY_SCHEMA_VERSION = 1
SCIENTIFIC_IDENTITY_STATUS = "COMPLETE_EDITING_V2_SCIENTIFIC_IDENTITY"
PROCESS_V2_SCIENTIFIC_IDENTITY_SCHEMA = (
    "compose.editing_v2_scientific_identity.process_v2"
)
PROCESS_V2_SCIENTIFIC_IDENTITY_SCHEMA_VERSION = 1
PROCESS_V2_SCIENTIFIC_IDENTITY_STATUS = (
    "COMPLETE_EDITING_V2_PROCESS_V2_SCIENTIFIC_IDENTITY"
)
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
_PROCESS_V2_MODEL_FIELDS = frozenset({*_MODEL_FIELDS, "atom_delete_action_semantics"})
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

# Mode values every semantic Editing-V2 model shares, whatever its process version.
_SHARED_EXPECTED_MODEL_VALUES: dict[str, Any] = {
    "compute_ring_grow_support": False,
    "compute_ring_restates": True,
    "compute_cyclic_graft": True,
    "compute_ring_opening": True,
    "compute_ring_system_delete": False,
    "enable_cycle_ops": True,
    "use_aromatic_bond_view": True,
    "atom_restate_action_semantics": SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    "ring_restate_scorer_mode": SEMANTIC_RING_RESTATE_SCORER_MODE,
    "cycle_close_action_semantics": SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    "cycle_open_action_semantics": SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    "cycle_open_scorer_mode": SEMANTIC_CYCLE_OPEN_SCORER_MODE,
}


class EditingV2ScientificIdentityError(ValueError):
    """An Editing-V2 scientific identity is absent, incomplete, or mismatched."""


class LegacyOnlyCheckpointError(EditingV2ScientificIdentityError):
    """A checkpoint lacks the complete identity required for Editing V2."""


@dataclass(frozen=True)
class _ScientificIdentityContract:
    """One frozen scientific-identity contract, bound to one semantic process version."""

    schema: str
    schema_version: int
    status: str
    editing_process_semantics: str
    model_fields: frozenset[str]
    expected_model_values: Mapping[str, Any]
    process_identity: Callable[[], Mapping[str, Any]]


_V1_CONTRACT = _ScientificIdentityContract(
    schema=SCIENTIFIC_IDENTITY_SCHEMA,
    schema_version=SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
    status=SCIENTIFIC_IDENTITY_STATUS,
    editing_process_semantics=SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    model_fields=_MODEL_FIELDS,
    expected_model_values={
        **_SHARED_EXPECTED_MODEL_VALUES,
        "editing_process_semantics": SEMANTIC_EDITING_V2_PROCESS_SEMANTICS,
    },
    process_identity=editing_v2_process_identity,
)
_PROCESS_V2_CONTRACT = _ScientificIdentityContract(
    schema=PROCESS_V2_SCIENTIFIC_IDENTITY_SCHEMA,
    schema_version=PROCESS_V2_SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
    status=PROCESS_V2_SCIENTIFIC_IDENTITY_STATUS,
    editing_process_semantics=PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    model_fields=_PROCESS_V2_MODEL_FIELDS,
    expected_model_values={
        **_SHARED_EXPECTED_MODEL_VALUES,
        "editing_process_semantics": PROCESS_V2_EDITING_PROCESS_SEMANTICS,
        "atom_delete_action_semantics": PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    },
    process_identity=editing_process_v2_identity,
)
_CONTRACTS_BY_SCHEMA: dict[str, _ScientificIdentityContract] = {
    contract.schema: contract for contract in (_V1_CONTRACT, _PROCESS_V2_CONTRACT)
}
_CONTRACTS_BY_PROCESS_SEMANTICS: dict[str, _ScientificIdentityContract] = {
    contract.editing_process_semantics: contract
    for contract in (_V1_CONTRACT, _PROCESS_V2_CONTRACT)
}


def _contract_for_schema(value: object) -> _ScientificIdentityContract:
    """Select the contract a payload declares.

    An unrecognized schema resolves to the V1 contract deliberately: the V1
    validator already rejects it, with the exact error it always produced, so the
    historical failure modes of a tampered or legacy payload are preserved.
    """

    if isinstance(value, str):
        contract = _CONTRACTS_BY_SCHEMA.get(value)
        if contract is not None:
            return contract
    return _V1_CONTRACT


def _contract_for_model_identity(value: object) -> _ScientificIdentityContract:
    """Select the contract a model identity's own process semantics implies.

    An unrecognized (or absent) process semantics resolves to V1 for the same
    reason: the V1 expected-mode check then rejects it as "not the exact semantic
    Editing-V2 model", which is the historical behaviour.
    """

    semantics = value.get("editing_process_semantics") if isinstance(value, Mapping) else None
    if isinstance(semantics, str):
        contract = _CONTRACTS_BY_PROCESS_SEMANTICS.get(semantics)
        if contract is not None:
            return contract
    return _V1_CONTRACT


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
    # An ABSENT atom-delete key means the legacy mode, which is the constructor default, so a V1 model
    # identity produces exactly the fingerprint it always did.
    try:
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
            atom_delete_action_semantics=model.get(
                "atom_delete_action_semantics",
                LEGACY_ATOM_DELETE_ACTION_SEMANTICS,
            ),
        )
    except ValueError as error:
        raise EditingV2ScientificIdentityError(
            f"model_identity is not a constructible operator capability: {error}"
        ) from error
    return capabilities.fingerprint()


def _normalize_model_identity(
    value: object,
    *,
    contract: _ScientificIdentityContract | None = None,
) -> dict[str, Any]:
    selected = _contract_for_model_identity(value) if contract is None else contract
    payload = _require_exact_mapping(
        value, fields=selected.model_fields, field="model_identity"
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

    expected_values = selected.expected_model_values
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


def _require_complete_identity_shape(
    payload: Mapping[str, Any],
    *,
    contract: _ScientificIdentityContract,
) -> None:
    """Classify every structurally incomplete identity as legacy-only."""

    nested_fields = (
        (
            "process_identity",
            payload.get("process_identity"),
            frozenset(contract.process_identity()),
        ),
        ("model_identity", payload.get("model_identity"), contract.model_fields),
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


def _model_identity_for_contract(
    contract: _ScientificIdentityContract,
    *,
    mark_dim: int,
    operator_capability_fingerprint: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mark_dim": mark_dim,
        "operator_capability_fingerprint": operator_capability_fingerprint,
        **contract.expected_model_values,
    }
    return _normalize_model_identity(payload, contract=contract)


def semantic_model_identity(
    *,
    mark_dim: int,
    operator_capability_fingerprint: str,
) -> dict[str, Any]:
    """Build the exact semantic model-mode identity for Editing V2 Process V1.

    This is the historical identity and carries no atom-delete key at all: the V1
    field set is exact, so an atom-delete mode in a V1 identity is rejected as an
    unexpected field rather than accepted and ignored.
    """

    return _model_identity_for_contract(
        _V1_CONTRACT,
        mark_dim=mark_dim,
        operator_capability_fingerprint=operator_capability_fingerprint,
    )


def process_v2_semantic_model_identity(
    *,
    mark_dim: int,
    operator_capability_fingerprint: str,
) -> dict[str, Any]:
    """Build the exact semantic model-mode identity for Editing V2 Process V2.

    It differs from the V1 identity in exactly two bound values: the semantic
    process version, and the atom-delete action semantics the expanded delete
    fiber is defined by.
    """

    return _model_identity_for_contract(
        _PROCESS_V2_CONTRACT,
        mark_dim=mark_dim,
        operator_capability_fingerprint=operator_capability_fingerprint,
    )


def _build_scientific_identity(
    contract: _ScientificIdentityContract,
    *,
    model_identity: Mapping[str, Any],
    artifact_identities: Mapping[str, Mapping[str, str]],
    process_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current_process = _canonical_json_copy(contract.process_identity())
    supplied_process = (
        current_process
        if process_identity is None
        else _canonical_json_copy(process_identity)
    )
    if supplied_process != current_process:
        raise EditingV2ScientificIdentityError(
            "process_identity differs from the complete current Editing-V2 process"
        )
    body = {
        "schema": contract.schema,
        "schema_version": contract.schema_version,
        "status": contract.status,
        "training_object": PRODUCTIVE_TRAINING_OBJECT,
        "process_identity": current_process,
        "model_identity": _normalize_model_identity(
            model_identity, contract=contract
        ),
        "artifact_identities": _normalize_artifacts(artifact_identities),
    }
    return {**body, "scientific_identity_sha256": canonical_sha256(body)}


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
    """Build one complete Editing-V2 checkpoint/run scientific identity.

    The contract follows the supplied model identity's own semantic process
    version, so a Process-V2 model produces a Process-V2 identity bound to the
    Process-V2 process. A model identity whose process version is unrecognized
    resolves to V1 and is then rejected by the V1 expected-mode check.
    """

    return _build_scientific_identity(
        _contract_for_model_identity(model_identity),
        model_identity=model_identity,
        artifact_identities={
            "semantic_corpus_global_registry": semantic_corpus_global_registry,
            "active8_admission": active8_admission,
            "successor_cache": successor_cache,
            "sampling_sidecar": sampling_sidecar,
            "gate_zero": gate_zero,
            "t1": t1,
            "p50_recipe": p50_recipe,
        },
        process_identity=process_identity,
    )


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
    contract = _contract_for_schema(payload["schema"])
    _require_complete_identity_shape(payload, contract=contract)
    if (
        payload["schema"] != contract.schema
        or payload["schema_version"] != contract.schema_version
        or payload["status"] != contract.status
        or payload["training_object"] != PRODUCTIVE_TRAINING_OBJECT
    ):
        raise EditingV2ScientificIdentityError(
            "scientific identity schema, status, or training object is not Editing V2"
        )
    # The schema and the model's own process semantics must select the SAME contract.  Without this a
    # payload could declare the V1 schema while carrying Process-V2 modes (or the reverse), and the
    # weaker of the two contracts would silently govern.
    model_contract = _contract_for_model_identity(payload["model_identity"])
    if model_contract is not contract:
        raise EditingV2ScientificIdentityError(
            f"scientific identity schema {contract.schema!r} binds process "
            f"{contract.editing_process_semantics!r}, but model_identity declares "
            f"{model_contract.editing_process_semantics!r}"
        )
    process_identity = _canonical_json_copy(payload["process_identity"])
    if process_identity != _canonical_json_copy(contract.process_identity()):
        raise EditingV2ScientificIdentityError(
            "scientific identity process differs from the complete current Editing-V2 process"
        )
    model_identity = _normalize_model_identity(
        payload["model_identity"], contract=contract
    )
    artifacts = _normalize_artifacts(payload["artifact_identities"])
    body = {
        "schema": contract.schema,
        "schema_version": contract.schema_version,
        "status": contract.status,
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
    "PROCESS_V2_SCIENTIFIC_IDENTITY_SCHEMA",
    "PROCESS_V2_SCIENTIFIC_IDENTITY_SCHEMA_VERSION",
    "PROCESS_V2_SCIENTIFIC_IDENTITY_STATUS",
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
    "process_v2_semantic_model_identity",
    "require_editing_v2_scientific_identity_equal",
    "semantic_model_identity",
    "validate_editing_v2_scientific_identity",
]
