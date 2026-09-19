"""Seal and validate the payload-bound nine-cell T4 scored campaign.

The preparation contract remains immutable.  This module binds its exact
controller and cell matrix to the passed zero-oracle gate, an exact-commit source
capsule, and the stale-query reconciliation.  Scored execution additionally
requires a separate authorization receipt containing the exact contract payload
identity, so code or configuration cannot change after authorization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compose_v4.experiments.t4_shared_controller_completion_contract import (
    CONTRACT_RELATIVE_PATH,
    EXPECTED_CELLS,
    EXPECTED_CONTROLLER,
    payload_identity,
    sha256_file,
    validate_preparation_contract,
    validate_zero_oracle_support_artifact,
    verify_exact_commit_capsule,
)

SCHEMA_VERSION = "t4_shared_controller_completion_scored_contract_v1"
AUTHORIZATION_SCHEMA_VERSION = "t4_shared_controller_completion_scored_authorization_v1"
FINAL_CONTRACT_RELATIVE_PATH = (
    "diagnostics/t4_shared_controller_completion_v1/scored_contract.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "diagnostics/t4_shared_controller_completion_v1/scored_authorization.json"
)
RECONCILIATION_SCHEMA_VERSION = "t4_braf_v4_stale_query_reconciliation_v1"
SUPPORT_RELATIVE_PATH = "diagnostics/t4_nine_cell_support_preflight_v1/attempt_1"
CAPSULE_ROOT_RELATIVE_PATH = (
    "diagnostics/t4_shared_controller_completion_v1/source_capsule"
)
CAPSULE_MANIFEST_RELATIVE_PATH = (
    "diagnostics/t4_shared_controller_completion_v1/source_capsule_manifest.json"
)
CAPSULE_INCLUDE = (
    "configs/t4_nine_cell_support_preflight_v1.json",
    "configs/t4_shared_controller_completion_v1.json",
    "diagnostics/t4_nine_cell_support_preflight_v1/attempt_1",
    "diagnostics/t4_retained_core_shared_support_v1/attempt_1/scientific_result.json",
    "diagnostics/t4_shared_controller_completion_v1/stale_braf_v4_reconciliation.json",
    "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json",
    "modal_apps/t4_shared_controller_completion_v1_app.py",
    "src/compose_v4",
)
RETAINED_CORE_SUPPORT_RELATIVE_PATH = (
    "diagnostics/t4_retained_core_shared_support_v1/attempt_1/scientific_result.json"
)
OPERATIONAL_SETTINGS = {
    "proposal_wait_seconds": 2_400,
    "query_wait_seconds": 1_800,
    "driver_poll_seconds": 10,
    "retained_core": {
        "maximum_fragment_atoms": 16,
        "maximum_stages": 2,
        "maximum_primitives": 32,
        "maximum_prefixes": 4_096,
        "program_family": "retained_core_prune",
        "proposal_expert": "route_complete_region",
    },
    "complete_combination_particles": {
        "enabled": True,
        "semantics": "additive_route_complete_region_subproducer",
        "binding_particles": [0, 1, 2, 3],
        "supported_complete_depths": [1, 2, 3],
        "scheduled_shard_counts": {"1": 1, "2": 1, "3": 5},
        "jobs_per_parent": 28,
        "maximum_combinations_per_job": 4_096,
        "missing_or_failed_receipt": "abstain_to_legacy_route_pool",
    },
}


def required_authorization_sentence(contract_payload_sha256: str) -> str:
    return (
        "I authorize the nine-cell shared COMPOSE completion campaign under "
        "scored-contract payload SHA-256 "
        f"{contract_payload_sha256}, capped at 49 charged docking calls per cell "
        "and 441 total, with zero retry, replacement, backfill, prior-outcome "
        "reuse, winner injection, endpoint injection, teacher injection, or "
        "runtime comparator."
    )


def _load_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    identity = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or payload_identity(payload) != identity:
        raise ValueError(f"invalid self-hashed envelope: {path}")
    return payload, str(identity)


def _publish_once(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed artifact: {path}")
    identity = payload_identity(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"payload": payload, "payload_sha256": identity},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    temporary.replace(path)
    return identity


def _validate_reconciliation(path: Path) -> tuple[dict[str, Any], str]:
    payload, identity = _load_envelope(path)
    if payload.get("schema_version") != RECONCILIATION_SCHEMA_VERSION:
        raise ValueError("unexpected stale-query reconciliation schema")
    if payload.get("charged_complete_receipts") != 43:
        raise ValueError("stale BRAF charged-receipt census drift")
    if payload.get("unresolved_locked_queries") != 0:
        raise ValueError("stale BRAF run contains unresolved locked queries")
    exclusions = payload.get("excluded_canonical_smiles_sha256")
    if not isinstance(exclusions, dict) or set(exclusions) != {"braf_0", "braf_1"}:
        raise ValueError("stale BRAF exclusion cells drift")
    for values in exclusions.values():
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or any(len(value) != 64 for value in values)
        ):
            raise ValueError("stale BRAF query identities are invalid")
    forbidden = {"score", "scores", "answer", "best", "winner", "smiles"}
    if forbidden.intersection(payload):
        raise ValueError("stale-query reconciliation leaks prior outcomes")
    return payload, identity


def _validate_retained_core_support(path: Path) -> tuple[dict[str, Any], str]:
    payload, identity = _load_envelope(path)
    if payload.get("schema_version") != (
        "t4_retained_core_shared_support_scientific_v1"
    ):
        raise ValueError("unexpected retained-core support schema")
    expected_configuration = {
        **OPERATIONAL_SETTINGS["retained_core"],
        "support": "compose_valid",
        "target_conditioning": False,
    }
    if payload.get("configuration") != expected_configuration:
        raise ValueError("retained-core support configuration drift")
    cells = payload.get("cells")
    if (
        not isinstance(cells, list)
        or [row.get("cell_key") for row in cells]
        != [row["cell_key"] for row in EXPECTED_CELLS]
        or any(int(row.get("eligible_unique", 0)) < 1 for row in cells)
    ):
        raise ValueError("retained-core nine-cell support census drift")
    if payload.get("gate") != {
        "complete_nine_cell_census": True,
        "every_cell_has_nonzero_eligible_support": True,
    }:
        raise ValueError("retained-core support gate did not pass")
    if payload.get("costs") != {
        "oracle_calls": 0,
        "docking_calls": 0,
        "modal_launches": 0,
    }:
        raise ValueError("retained-core support evidence is not zero-oracle")
    return payload, identity


def seal_scored_contract(
    *,
    repository_root: Path,
    capsule_root: Path,
    capsule_manifest_path: Path,
    reconciliation_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind every launch input while leaving scored execution unauthorized."""

    root = repository_root.resolve()
    preparation_path = root / CONTRACT_RELATIVE_PATH
    validate_preparation_contract(root, preparation_path)
    preparation = json.loads(preparation_path.read_text())
    support_root = root / SUPPORT_RELATIVE_PATH
    support = validate_zero_oracle_support_artifact(
        repository_root=root, artifact_root=support_root
    )
    retained_core_support_path = root / RETAINED_CORE_SUPPORT_RELATIVE_PATH
    _, retained_core_support_identity = _validate_retained_core_support(
        retained_core_support_path
    )
    capsule = verify_exact_commit_capsule(capsule_root, capsule_manifest_path)
    capsule_manifest, _ = _load_envelope(capsule_manifest_path)
    if capsule_manifest.get("include") != list(CAPSULE_INCLUDE):
        raise ValueError("source capsule include set drift")
    reconciliation, reconciliation_payload_sha256 = _validate_reconciliation(
        reconciliation_path
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_PENDING_EXACT_USER_AUTHORIZATION",
        "scored_calls_requested": 441,
        "charged_calls_per_cell": 49,
        "total_charged_call_ceiling": 441,
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
        "support": "compose_valid",
        "controller": EXPECTED_CONTROLLER,
        "cells": list(EXPECTED_CELLS),
        "shared_route_checkpoint": preparation["shared_route_checkpoint"],
        "receipt_policy": preparation["receipt_policy"],
        "operational_settings": OPERATIONAL_SETTINGS,
        "preparation_contract": {
            "path": CONTRACT_RELATIVE_PATH,
            "sha256": sha256_file(preparation_path),
        },
        "zero_oracle_support_gate": {
            "path": SUPPORT_RELATIVE_PATH,
            **support,
        },
        "retained_core_support_gate": {
            "path": RETAINED_CORE_SUPPORT_RELATIVE_PATH,
            "sha256": sha256_file(retained_core_support_path),
            "payload_sha256": retained_core_support_identity,
        },
        "source_capsule": {
            **capsule,
            "root_relative_path": CAPSULE_ROOT_RELATIVE_PATH,
            "manifest_relative_path": CAPSULE_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": sha256_file(capsule_manifest_path),
        },
        "stale_braf_v4_reconciliation": {
            "path": str(reconciliation_path.relative_to(root)),
            "sha256": sha256_file(reconciliation_path),
            "payload_sha256": reconciliation_payload_sha256,
            "excluded_canonical_smiles_sha256": reconciliation[
                "excluded_canonical_smiles_sha256"
            ],
        },
        "authorization": {
            "state": "PENDING_EXACT_PAYLOAD_BOUND_USER_AUTHORIZATION",
            "receipt_path": AUTHORIZATION_RELATIVE_PATH,
        },
        "claim_boundary": (
            "Prospective nine-cell development completion campaign under one exact "
            "shared controller and one docking seed; not the final frozen 30-cell run"
        ),
    }
    identity = _publish_once(output_path, payload)
    return {
        "contract_path": str(output_path),
        "contract_sha256": sha256_file(output_path),
        "contract_payload_sha256": identity,
        "required_authorization_sentence": required_authorization_sentence(identity),
    }


def publish_authorization_receipt(
    *, contract_path: Path, authorization_path: Path, user_statement: str
) -> dict[str, Any]:
    contract, contract_identity = _load_envelope(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected scored contract schema")
    required = required_authorization_sentence(contract_identity)
    if user_statement != required:
        raise ValueError(
            "scored authorization does not exactly match the sealed payload"
        )
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "contract_payload_sha256": contract_identity,
        "authorized_scored_calls": 441,
        "user_statement": user_statement,
    }
    identity = _publish_once(authorization_path, payload)
    return {
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "authorization_payload_sha256": identity,
    }


def validate_scored_contract(
    *,
    repository_root: Path,
    contract_path: Path,
    authorization_path: Path,
    capsule_root: Path,
    capsule_manifest_path: Path,
) -> dict[str, Any]:
    """Fail closed unless contract, capsule, evidence, and authorization agree."""

    root = repository_root.resolve()
    contract, contract_identity = _load_envelope(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected scored contract schema")
    if contract.get("status") != "SEALED_PENDING_EXACT_USER_AUTHORIZATION":
        raise ValueError("scored contract status drift")
    if contract.get("controller") != EXPECTED_CONTROLLER:
        raise ValueError("scored controller drift")
    if contract.get("operational_settings") != OPERATIONAL_SETTINGS:
        raise ValueError("scored operational settings drift")
    if contract.get("cells") != list(EXPECTED_CELLS):
        raise ValueError("scored cell matrix drift")
    if (
        contract.get("scored_calls_requested") != 441
        or contract.get("charged_calls_per_cell") != 49
        or contract.get("total_charged_call_ceiling") != 441
        or contract.get("automatic_retries") != 0
        or contract.get("replacement") is not False
        or contract.get("backfill") is not False
    ):
        raise ValueError("scored budget or retry policy drift")
    if contract.get("support") != "compose_valid":
        raise ValueError("scored endpoint-support drift")
    preparation_path = root / CONTRACT_RELATIVE_PATH
    validate_preparation_contract(root, preparation_path)
    if contract.get("preparation_contract") != {
        "path": CONTRACT_RELATIVE_PATH,
        "sha256": sha256_file(preparation_path),
    }:
        raise ValueError("bound preparation contract drift")
    preparation = json.loads(preparation_path.read_text())
    if contract.get("shared_route_checkpoint") != preparation.get(
        "shared_route_checkpoint"
    ):
        raise ValueError("bound shared-route checkpoint drift")
    if contract.get("receipt_policy") != preparation.get("receipt_policy"):
        raise ValueError("bound receipt policy drift")
    support = validate_zero_oracle_support_artifact(
        repository_root=root, artifact_root=root / SUPPORT_RELATIVE_PATH
    )
    for key, value in support.items():
        if contract["zero_oracle_support_gate"].get(key) != value:
            raise ValueError(f"bound support evidence drift: {key}")
    retained_core_support_path = root / RETAINED_CORE_SUPPORT_RELATIVE_PATH
    _, retained_core_support_identity = _validate_retained_core_support(
        retained_core_support_path
    )
    if contract.get("retained_core_support_gate") != {
        "path": RETAINED_CORE_SUPPORT_RELATIVE_PATH,
        "sha256": sha256_file(retained_core_support_path),
        "payload_sha256": retained_core_support_identity,
    }:
        raise ValueError("bound retained-core support evidence drift")
    capsule = verify_exact_commit_capsule(capsule_root, capsule_manifest_path)
    capsule_manifest, _ = _load_envelope(capsule_manifest_path)
    if capsule_manifest.get("include") != list(CAPSULE_INCLUDE):
        raise ValueError("source capsule include set drift")
    for key, value in capsule.items():
        if contract["source_capsule"].get(key) != value:
            raise ValueError(f"bound source capsule drift: {key}")
    if (
        contract["source_capsule"].get("root_relative_path")
        != CAPSULE_ROOT_RELATIVE_PATH
        or contract["source_capsule"].get("manifest_relative_path")
        != CAPSULE_MANIFEST_RELATIVE_PATH
        or contract["source_capsule"].get("manifest_sha256")
        != sha256_file(capsule_manifest_path)
    ):
        raise ValueError("bound source capsule path or manifest drift")
    reconciliation_path = root / contract["stale_braf_v4_reconciliation"]["path"]
    reconciliation, reconciliation_identity = _validate_reconciliation(
        reconciliation_path
    )
    if (
        sha256_file(reconciliation_path)
        != contract["stale_braf_v4_reconciliation"]["sha256"]
        or reconciliation_identity
        != contract["stale_braf_v4_reconciliation"]["payload_sha256"]
        or reconciliation["excluded_canonical_smiles_sha256"]
        != contract["stale_braf_v4_reconciliation"]["excluded_canonical_smiles_sha256"]
    ):
        raise ValueError("bound stale-query reconciliation drift")
    authorization, _ = _load_envelope(authorization_path)
    if contract.get("authorization") != {
        "state": "PENDING_EXACT_PAYLOAD_BOUND_USER_AUTHORIZATION",
        "receipt_path": AUTHORIZATION_RELATIVE_PATH,
    }:
        raise ValueError("bound authorization policy drift")
    if authorization != {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "contract_payload_sha256": contract_identity,
        "authorized_scored_calls": 441,
        "user_statement": required_authorization_sentence(contract_identity),
    }:
        raise ValueError("scored authorization receipt does not match contract")
    return contract


__all__ = [
    "AUTHORIZATION_RELATIVE_PATH",
    "AUTHORIZATION_SCHEMA_VERSION",
    "CAPSULE_INCLUDE",
    "CAPSULE_MANIFEST_RELATIVE_PATH",
    "CAPSULE_ROOT_RELATIVE_PATH",
    "FINAL_CONTRACT_RELATIVE_PATH",
    "OPERATIONAL_SETTINGS",
    "RECONCILIATION_SCHEMA_VERSION",
    "RETAINED_CORE_SUPPORT_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "publish_authorization_receipt",
    "required_authorization_sentence",
    "seal_scored_contract",
    "validate_scored_contract",
]
