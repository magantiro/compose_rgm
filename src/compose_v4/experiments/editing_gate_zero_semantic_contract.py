"""Frozen semantic model/process identity for Editing-V2 Gate 0.

This contract is deliberately narrower than the Gate-0 runtime contract.  It
binds the model modes and production process that a future semantic Gate-0 run
must use, but it contains no corpus evidence and grants neither Gate-0 nor
training authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.experiments.editing_v2_scientific_identity import (
    EditingV2ScientificIdentityError,
    semantic_model_identity,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_v2_process_identity,
)

CONTRACT_SCHEMA = "compose.editing.gate_zero_semantic_model_process_contract"
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_ID = "editing_v2_active8_gate0_model_process_v1"
CONTRACT_STATUS = "FROZEN_SEMANTIC_MODEL_PROCESS_CONTRACT_NO_AUTHORITY"

EXPECTED_MARK_DIM = 32
EXPECTED_OPERATOR_CAPABILITY_FINGERPRINT = "d246bc88d8440d31"
LEGACY_OPERATOR_CAPABILITY_FINGERPRINT = "e787c852410c6b63"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "contract_id",
        "status",
        "training_authorized",
        "active_families",
        "model_identity",
        "process_identity_sha256",
        "contract_sha256",
    }
)


class GateZeroSemanticContractError(ValueError):
    """The semantic Gate-0 model/process contract is absent or inconsistent."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GateZeroSemanticContractError(
            f"semantic Gate-0 contract is not finite deterministic JSON: {error}"
        ) from error


def _canonical_copy(value: object) -> Any:
    return json.loads(_canonical_json_bytes(value))


def gate_zero_semantic_contract_sha256(value: Mapping[str, Any]) -> str:
    """Hash a contract body after removing its self-hash field."""

    body = dict(value)
    body.pop("contract_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _current_process_sha256() -> str:
    value = editing_v2_process_identity().get("process_identity_sha256")
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateZeroSemanticContractError(
            "production Editing-V2 process identity lacks a full SHA-256"
        )
    return value


def _expected_model_identity() -> dict[str, Any]:
    try:
        return semantic_model_identity(
            mark_dim=EXPECTED_MARK_DIM,
            operator_capability_fingerprint=(EXPECTED_OPERATOR_CAPABILITY_FINGERPRINT),
        )
    except EditingV2ScientificIdentityError as error:
        raise GateZeroSemanticContractError(
            "cannot construct the exact semantic Editing-V2 model identity"
        ) from error


def build_gate_zero_semantic_contract() -> dict[str, Any]:
    """Build the non-authorizing contract from production identity APIs."""

    body: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "status": CONTRACT_STATUS,
        "training_authorized": False,
        "active_families": list(ACTIVE8_FAMILIES),
        "model_identity": _expected_model_identity(),
        "process_identity_sha256": _current_process_sha256(),
    }
    return {**body, "contract_sha256": gate_zero_semantic_contract_sha256(body)}


def validate_gate_zero_semantic_contract(value: object) -> dict[str, Any]:
    """Validate exact semantic modes and reject legacy or mixed identities."""

    if not isinstance(value, Mapping):
        raise GateZeroSemanticContractError(
            "semantic Gate-0 model/process contract must be an object"
        )
    payload = _canonical_copy(value)
    observed_fields = set(payload)
    if observed_fields != _CONTRACT_FIELDS:
        raise GateZeroSemanticContractError(
            "semantic Gate-0 contract fields disagree; "
            f"missing={sorted(_CONTRACT_FIELDS - observed_fields)}, "
            f"unexpected={sorted(observed_fields - _CONTRACT_FIELDS)}"
        )
    if (
        payload["schema"] != CONTRACT_SCHEMA
        or payload["schema_version"] != CONTRACT_SCHEMA_VERSION
        or payload["contract_id"] != CONTRACT_ID
        or payload["status"] != CONTRACT_STATUS
        or payload["training_authorized"] is not False
    ):
        raise GateZeroSemanticContractError(
            "semantic Gate-0 contract schema, status, or authority is invalid"
        )
    if payload["active_families"] != list(ACTIVE8_FAMILIES):
        raise GateZeroSemanticContractError(
            "semantic Gate-0 contract does not preserve exact Active8 order"
        )

    model_identity = payload["model_identity"]
    if not isinstance(model_identity, Mapping):
        raise GateZeroSemanticContractError("model_identity must be an object")
    fingerprint = model_identity.get("operator_capability_fingerprint")
    if fingerprint == LEGACY_OPERATOR_CAPABILITY_FINGERPRINT:
        raise GateZeroSemanticContractError(
            "legacy Gate-0 capability fingerprint is forbidden for Editing V2"
        )
    expected_model = _expected_model_identity()
    if model_identity != expected_model:
        raise GateZeroSemanticContractError(
            "model_identity is missing or mixes nonsemantic Editing-V2 modes"
        )

    process_sha256 = payload["process_identity_sha256"]
    if (
        not isinstance(process_sha256, str)
        or _SHA256_RE.fullmatch(process_sha256) is None
        or process_sha256 != _current_process_sha256()
    ):
        raise GateZeroSemanticContractError(
            "semantic Gate-0 contract names another production process identity"
        )

    contract_sha256 = payload["contract_sha256"]
    if (
        not isinstance(contract_sha256, str)
        or _SHA256_RE.fullmatch(contract_sha256) is None
        or contract_sha256 != gate_zero_semantic_contract_sha256(payload)
    ):
        raise GateZeroSemanticContractError(
            "semantic Gate-0 contract self-hash does not match its contents"
        )
    return payload


@dataclass(frozen=True)
class FrozenGateZeroSemanticContract:
    """A validated logical contract plus its physical file identity."""

    source: Path
    payload: dict[str, Any]
    file_sha256: str

    @property
    def sha256(self) -> str:
        return str(self.payload["contract_sha256"])


def load_gate_zero_semantic_contract(
    path: str | Path,
) -> FrozenGateZeroSemanticContract:
    """Load a frozen semantic contract without granting runtime authority."""

    source = Path(path)
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except OSError as error:
        raise GateZeroSemanticContractError(
            f"semantic Gate-0 contract is absent: {source}"
        ) from error
    except json.JSONDecodeError as error:
        raise GateZeroSemanticContractError(
            f"semantic Gate-0 contract is invalid JSON: {source}"
        ) from error
    payload = validate_gate_zero_semantic_contract(value)
    return FrozenGateZeroSemanticContract(
        source=source.resolve(),
        payload=payload,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_SCHEMA",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_STATUS",
    "EXPECTED_MARK_DIM",
    "EXPECTED_OPERATOR_CAPABILITY_FINGERPRINT",
    "LEGACY_OPERATOR_CAPABILITY_FINGERPRINT",
    "FrozenGateZeroSemanticContract",
    "GateZeroSemanticContractError",
    "build_gate_zero_semantic_contract",
    "gate_zero_semantic_contract_sha256",
    "load_gate_zero_semantic_contract",
    "validate_gate_zero_semantic_contract",
]
