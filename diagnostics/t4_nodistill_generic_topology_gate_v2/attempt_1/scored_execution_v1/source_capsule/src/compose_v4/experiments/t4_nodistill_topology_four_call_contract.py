"""Authority and exact-source binding for the four-call topology diagnostic.

The scientific scored-call payload and its user authorization are immutable.
This module adds only an operational source capsule and execution identity.  It
does not add calls, candidates, retries, replacements, or backfill.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from compose_v4.experiments.t4_shared_controller_completion_contract import (
    build_exact_commit_capsule,
    payload_identity,
    sha256_file,
    verify_exact_commit_capsule,
)

PREPARATION_ROOT = (
    "diagnostics/t4_nodistill_generic_topology_gate_v2/attempt_1/"
    "scored_preparation_v1"
)
SCORED_CONTRACT_RELATIVE_PATH = f"{PREPARATION_ROOT}/scored_contract.json"
AUTHORIZATION_RELATIVE_PATH = f"{PREPARATION_ROOT}/scored_authorization.json"
PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH = (
    f"{PREPARATION_ROOT}/source_capsule_manifest.json"
)
CANDIDATE_PROPOSAL_RELATIVE_PATH = (
    "diagnostics/t4_nodistill_generic_topology_gate_v2/attempt_1/"
    "scored_candidate_proposal.json"
)
EXECUTION_ROOT = (
    "diagnostics/t4_nodistill_generic_topology_gate_v2/attempt_1/" "scored_execution_v1"
)
EXECUTION_CONTRACT_RELATIVE_PATH = f"{EXECUTION_ROOT}/execution_contract.json"
CAPSULE_ROOT_RELATIVE_PATH = f"{EXECUTION_ROOT}/source_capsule"
CAPSULE_MANIFEST_RELATIVE_PATH = f"{EXECUTION_ROOT}/source_capsule_manifest.json"

SCIENTIFIC_CONTRACT_PAYLOAD_SHA256 = (
    "4f973c171dd9392f23819b362c6b1946d846e712103d8f58b8a2ce29f70a8da6"
)
AUTHORIZATION_PAYLOAD_SHA256 = (
    "fa086fb8e3175a38873fa620878bc66eec636385cb0c35679cfcc9e01a28492e"
)
PREPARATION_CAPSULE_PAYLOAD_SHA256 = (
    "a039a0e5058a9a32b3d7cf435f4ddfb3ea30866acb777a27a00ef6d9e3a4cc9d"
)
REQUIRED_AUTHORIZATION_SENTENCE = (
    "I authorize exactly 4 scored docking calls under contract payload "
    f"{SCIENTIFIC_CONTRACT_PAYLOAD_SHA256}, with zero retries, replacements, "
    "or backfill."
)

APP_NAME = "compose-t4-nodistill-topology-four-call-v1"
VOLUME_NAME = "compose-t4-nodistill-topology-four-call-v1"
VOLUME_ROOT = "/artifacts/t4_nodistill_topology_four_call_v1"
EXECUTION_SCHEMA = "t4_nodistill_topology_four_call_execution_contract_v1"

APP_SOURCE = "modal_apps/t4_nodistill_topology_four_call_app.py"
LAUNCHER_SOURCE = "tools/launch_t4_nodistill_topology_four_call.py"
SEALER_SOURCE = "tools/seal_t4_nodistill_topology_four_call.py"
CONTRACT_SOURCE = (
    "src/compose_v4/experiments/t4_nodistill_topology_four_call_contract.py"
)
RUNTIME_SOURCE = "src/compose_v4/experiments/t4_nodistill_topology_four_call_runtime.py"
DOCKING_ADAPTER_SOURCE = "src/compose_v4/experiments/t4_docking_adapter.py"

CAPSULE_INCLUDE = (
    SCORED_CONTRACT_RELATIVE_PATH,
    AUTHORIZATION_RELATIVE_PATH,
    PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH,
    CANDIDATE_PROPOSAL_RELATIVE_PATH,
    APP_SOURCE,
    "modal_apps/genmol_t4_opt_app.py",
    "modal_apps/run_process_v2_p50_app.py",
    LAUNCHER_SOURCE,
    SEALER_SOURCE,
    CONTRACT_SOURCE,
    RUNTIME_SOURCE,
    DOCKING_ADAPTER_SOURCE,
    "src/compose_v4/control/docking_value.py",
    "src/compose_v4/experiments/t4_shared_controller_completion_contract.py",
)


def _load_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or claimed != payload_identity(payload):
        raise ValueError(f"artifact is not a valid self-hashed envelope: {path}")
    return payload, claimed


def _publish_once(path: Path, payload: dict[str, Any]) -> str:
    envelope = {"payload": payload, "payload_sha256": payload_identity(payload)}
    content = (
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
        return envelope["payload_sha256"]
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(path)
    return envelope["payload_sha256"]


def validate_scored_authority(
    repository_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    """Validate the unchanged four-query payload and exact user receipt."""

    root = repository_root.resolve()
    contract_path = root / SCORED_CONTRACT_RELATIVE_PATH
    authorization_path = root / AUTHORIZATION_RELATIVE_PATH
    candidate_path = root / CANDIDATE_PROPOSAL_RELATIVE_PATH
    preparation_manifest_path = root / PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH

    contract, contract_identity = _load_envelope(contract_path)
    if contract_identity != SCIENTIFIC_CONTRACT_PAYLOAD_SHA256:
        raise ValueError("four-call scientific contract payload changed")
    if contract.get("schema_version") != (
        "t4_nodistill_generic_topology_four_call_scored_contract_v1"
    ):
        raise ValueError("unexpected four-call scientific contract schema")
    if (
        contract.get("status")
        != "SEALED_PENDING_EXACT_PAYLOAD_BOUND_USER_AUTHORIZATION"
    ):
        raise ValueError("four-call scientific contract status changed")
    if contract.get("budget") != {
        "unique_query_count": 4,
        "total_charged_call_ceiling": 4,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "replicate_calls": 0,
    }:
        raise ValueError("four-call budget changed")
    queries = contract.get("queries")
    if (
        not isinstance(queries, list)
        or len(queries) != 4
        or len({row.get("query_id") for row in queries}) != 4
        or any(row.get("attempt_ceiling") != 1 for row in queries)
        or any(
            row.get("cell_key") not in {"parp1_0_d04", "parp1_1_d04"} for row in queries
        )
    ):
        raise ValueError("four-call query census changed")
    if contract.get("evaluator") != {
        "target": "parp1",
        "docking_box": [
            [26.413, 11.282, 27.238],
            [18.521, 17.479, 19.995],
        ],
        "qvina02_sha256": (
            "f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0"
        ),
        "receptor_sha256": (
            "8d0891ddf915f51cf3108f39dd9dc01dfcf86be342c566021956afdd79eb4ad9"
        ),
        "docking_seed": 20260919,
        "asset_hash_verification": "required_immediately_before_launch",
    }:
        raise ValueError("four-call evaluator changed")
    if contract.get("receipt_policy") != {
        "publish_query_lock_before_first_docking_call": True,
        "failed_or_timed_out_query": (
            "record_once_as_terminal_failure_without_retry_replacement_or_backfill"
        ),
        "partial_campaign": (
            "publish_all_completed_and_failed_receipts_without_filling_unused_slots"
        ),
        "candidate_mutation_after_authorization": False,
        "score_visibility_before_lock_change": False,
    }:
        raise ValueError("four-call receipt policy changed")

    proposal, proposal_identity = _load_envelope(candidate_path)
    candidate_binding = contract.get("candidate_lock", {})
    if (
        sha256_file(candidate_path) != candidate_binding.get("sha256")
        or proposal_identity != candidate_binding.get("payload_sha256")
        or proposal.get("candidate_count") != 4
    ):
        raise ValueError("four-call candidate proposal changed")
    expected = [
        (row["cell_key"], row["endpoint_key_sha256"], row["canonical_smiles"])
        for row in proposal["candidates"]
    ]
    observed = [
        (row["cell_key"], row["endpoint_key_sha256"], row["canonical_smiles"])
        for row in queries
    ]
    if observed != expected:
        raise ValueError("four-call candidates differ from the prospective lock")

    authorization, authorization_identity = _load_envelope(authorization_path)
    expected_authorization = {
        "schema_version": "t4_exact_payload_bound_scored_authorization_v1",
        "contract_path": SCORED_CONTRACT_RELATIVE_PATH,
        "contract_payload_sha256": SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
        "authorized_scored_calls": 4,
        "user_statement": REQUIRED_AUTHORIZATION_SENTENCE,
    }
    if (
        authorization_identity != AUTHORIZATION_PAYLOAD_SHA256
        or authorization != expected_authorization
    ):
        raise ValueError("four-call exact authorization receipt changed")

    preparation_manifest, preparation_identity = _load_envelope(
        preparation_manifest_path
    )
    if (
        preparation_identity != PREPARATION_CAPSULE_PAYLOAD_SHA256
        or preparation_manifest.get("status") != "SEALED_PREPARATION_ONLY"
        or preparation_manifest.get("code_revision")
        != "7c2f34f8574c7b247a315c0fefa09033e6241542"
    ):
        raise ValueError("preparation-only source capsule changed")
    return contract, contract_identity, authorization, authorization_identity


def _execution_policy() -> dict[str, Any]:
    return {
        "app_name": APP_NAME,
        "private_volume": VOLUME_NAME,
        "volume_root": VOLUME_ROOT,
        "query_count": 4,
        "maximum_parallel_workers": 4,
        "cpu_per_worker": 1,
        "attempts_per_query": 1,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "replicate_calls": 0,
        "publish_all_reservations_before_first_worker_spawn": True,
        "worker_requires_complete_durable_query_lock": True,
        "one_terminal_receipt_per_started_query": True,
        "deterministic_reduction_order": "scientific_contract_query_order",
    }


def seal_execution_contract(
    *,
    repository_root: Path,
    capsule_root: Path,
    capsule_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    _, contract_identity, _, authorization_identity = validate_scored_authority(root)
    capsule = verify_exact_commit_capsule(capsule_root, capsule_manifest_path)
    manifest, _ = _load_envelope(capsule_manifest_path)
    if manifest.get("include") != sorted(CAPSULE_INCLUDE):
        raise ValueError("four-call execution capsule include set changed")
    payload = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "SOURCE_BOUND_EXECUTABLE_UNLAUNCHED",
        "scientific_scored_contract": {
            "path": SCORED_CONTRACT_RELATIVE_PATH,
            "sha256": sha256_file(root / SCORED_CONTRACT_RELATIVE_PATH),
            "payload_sha256": contract_identity,
        },
        "authorization_receipt": {
            "path": AUTHORIZATION_RELATIVE_PATH,
            "sha256": sha256_file(root / AUTHORIZATION_RELATIVE_PATH),
            "payload_sha256": authorization_identity,
        },
        "candidate_proposal": {
            "path": CANDIDATE_PROPOSAL_RELATIVE_PATH,
            "sha256": sha256_file(root / CANDIDATE_PROPOSAL_RELATIVE_PATH),
        },
        "preparation_source_capsule": {
            "path": PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH,
            "sha256": sha256_file(root / PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH),
            "payload_sha256": PREPARATION_CAPSULE_PAYLOAD_SHA256,
            "scope": "preparation_only_not_runtime_authority",
        },
        "execution_source_capsule": {
            **capsule,
            "root_relative_path": CAPSULE_ROOT_RELATIVE_PATH,
            "manifest_relative_path": CAPSULE_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": sha256_file(capsule_manifest_path),
        },
        "execution": _execution_policy(),
        "costs_spent_during_packaging": {
            "docking_calls": 0,
            "oracle_calls": 0,
            "modal_calls": 0,
        },
        "claim_boundary": (
            "Operational binding for the already authorized four-candidate prospective "
            "utility diagnostic. It does not change the scored-call payload, candidates, "
            "evaluator, call ceiling, or scientific claim."
        ),
    }
    identity = _publish_once(output_path, payload)
    return {
        "execution_contract_path": str(output_path),
        "execution_contract_payload_sha256": identity,
        "execution_contract_sha256": sha256_file(output_path),
        "scientific_contract_payload_sha256": contract_identity,
        "authorization_payload_sha256": authorization_identity,
        "authorized_scored_calls": 4,
        "modal_calls_created": 0,
    }


def validate_execution_contract(
    repository_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    root = repository_root.resolve()
    scientific, scientific_identity, _, authorization_identity = (
        validate_scored_authority(root)
    )
    path = root / EXECUTION_CONTRACT_RELATIVE_PATH
    execution, execution_identity = _load_envelope(path)
    if (
        execution.get("schema_version") != EXECUTION_SCHEMA
        or execution.get("status") != "SOURCE_BOUND_EXECUTABLE_UNLAUNCHED"
        or execution.get("execution") != _execution_policy()
    ):
        raise ValueError("four-call execution contract changed")
    capsule_root = root / CAPSULE_ROOT_RELATIVE_PATH
    manifest_path = root / CAPSULE_MANIFEST_RELATIVE_PATH
    capsule = verify_exact_commit_capsule(capsule_root, manifest_path)
    manifest, _ = _load_envelope(manifest_path)
    if manifest.get("include") != sorted(CAPSULE_INCLUDE):
        raise ValueError("four-call execution capsule include set changed")
    expected_capsule = {
        **capsule,
        "root_relative_path": CAPSULE_ROOT_RELATIVE_PATH,
        "manifest_relative_path": CAPSULE_MANIFEST_RELATIVE_PATH,
        "manifest_sha256": sha256_file(manifest_path),
    }
    if execution.get("execution_source_capsule") != expected_capsule:
        raise ValueError("four-call execution source capsule changed")
    if execution.get("scientific_scored_contract") != {
        "path": SCORED_CONTRACT_RELATIVE_PATH,
        "sha256": sha256_file(root / SCORED_CONTRACT_RELATIVE_PATH),
        "payload_sha256": scientific_identity,
    }:
        raise ValueError("four-call execution/scientific contract binding changed")
    if execution.get("authorization_receipt") != {
        "path": AUTHORIZATION_RELATIVE_PATH,
        "sha256": sha256_file(root / AUTHORIZATION_RELATIVE_PATH),
        "payload_sha256": authorization_identity,
    }:
        raise ValueError("four-call execution/authorization binding changed")
    if execution.get("candidate_proposal") != {
        "path": CANDIDATE_PROPOSAL_RELATIVE_PATH,
        "sha256": sha256_file(root / CANDIDATE_PROPOSAL_RELATIVE_PATH),
    }:
        raise ValueError("four-call execution/candidate binding changed")
    if execution.get("preparation_source_capsule") != {
        "path": PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH,
        "sha256": sha256_file(root / PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH),
        "payload_sha256": PREPARATION_CAPSULE_PAYLOAD_SHA256,
        "scope": "preparation_only_not_runtime_authority",
    }:
        raise ValueError("four-call preparation-capsule scope changed")
    if execution.get("costs_spent_during_packaging") != {
        "docking_calls": 0,
        "oracle_calls": 0,
        "modal_calls": 0,
    }:
        raise ValueError("four-call packaging cost record changed")
    return execution, execution_identity, scientific


def file_sha256(path: Path) -> str:
    """Local alias used by thin launch tooling."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "APP_NAME",
    "APP_SOURCE",
    "AUTHORIZATION_PAYLOAD_SHA256",
    "AUTHORIZATION_RELATIVE_PATH",
    "CANDIDATE_PROPOSAL_RELATIVE_PATH",
    "CAPSULE_INCLUDE",
    "CAPSULE_MANIFEST_RELATIVE_PATH",
    "CAPSULE_ROOT_RELATIVE_PATH",
    "CONTRACT_SOURCE",
    "DOCKING_ADAPTER_SOURCE",
    "EXECUTION_CONTRACT_RELATIVE_PATH",
    "EXECUTION_ROOT",
    "LAUNCHER_SOURCE",
    "PREPARATION_CAPSULE_MANIFEST_RELATIVE_PATH",
    "RUNTIME_SOURCE",
    "SCIENTIFIC_CONTRACT_PAYLOAD_SHA256",
    "SCORED_CONTRACT_RELATIVE_PATH",
    "SEALER_SOURCE",
    "VOLUME_NAME",
    "VOLUME_ROOT",
    "build_exact_commit_capsule",
    "file_sha256",
    "seal_execution_contract",
    "validate_execution_contract",
    "validate_scored_authority",
]
