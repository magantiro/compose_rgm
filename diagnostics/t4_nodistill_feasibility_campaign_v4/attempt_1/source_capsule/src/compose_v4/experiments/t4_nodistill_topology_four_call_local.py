"""No-egress local execution binding for the locked four-call PARP1 diagnostic.

This module can seal and validate an operational binding without building an
image or making an oracle call.  Scoring remains impossible until a separate
environment attestation and an exact binding-specific authorization receipt
exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from compose_v4.experiments.t4_nodistill_topology_four_call_contract import (
    AUTHORIZATION_PAYLOAD_SHA256,
    AUTHORIZATION_RELATIVE_PATH,
    CANDIDATE_PROPOSAL_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
    SCORED_CONTRACT_RELATIVE_PATH,
    validate_execution_contract,
    validate_scored_authority,
)

LOCAL_EXECUTION_ROOT = (
    "diagnostics/t4_nodistill_generic_topology_gate_v2/attempt_1/scored_execution_local_v1"
)
LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH = f"{LOCAL_EXECUTION_ROOT}/execution_contract.json"
LOCAL_ENVIRONMENT_ATTESTATION_RELATIVE_PATH = f"{LOCAL_EXECUTION_ROOT}/environment_attestation.json"
LOCAL_LAUNCH_AUTHORIZATION_RELATIVE_PATH = f"{LOCAL_EXECUTION_ROOT}/launch_authorization.json"
LOCAL_RECEIPT_ROOT_RELATIVE_PATH = f"{LOCAL_EXECUTION_ROOT}/runs"

LOCAL_EXECUTION_SCHEMA = "t4_nodistill_topology_four_call_local_execution_contract_v1"
LOCAL_ENVIRONMENT_SCHEMA = "t4_nodistill_topology_four_call_local_environment_attestation_v1"
LOCAL_AUTHORIZATION_SCHEMA = "t4_nodistill_topology_four_call_local_launch_authorization_v1"

CONTAINER_RECIPE_RELATIVE_PATH = "containers/t4_nodistill_topology_four_call_local/Dockerfile"
LOCAL_BINDING_SOURCE = "src/compose_v4/experiments/t4_nodistill_topology_four_call_local.py"
LOCAL_RUNTIME_SOURCE = "src/compose_v4/experiments/t4_nodistill_topology_four_call_local_runtime.py"
LOCAL_TOOL_SOURCE = "tools/t4_nodistill_topology_four_call_local.py"
DOCKING_ADAPTER_SOURCE = "src/compose_v4/experiments/t4_docking_adapter.py"

LOCAL_SOURCE_FILES = (
    CONTAINER_RECIPE_RELATIVE_PATH,
    DOCKING_ADAPTER_SOURCE,
    LOCAL_BINDING_SOURCE,
    LOCAL_RUNTIME_SOURCE,
    LOCAL_TOOL_SOURCE,
)

LOCAL_QVINA_PATH = Path("/private/tmp/qvina02")
LOCAL_RECEPTOR_PATH = Path(
    "/private/tmp/fragment_constrained_audit/"
    "NVIDIA-BioNeMo-genmol-add09fc/scripts/exps/lead/docking/parp1.pdbqt"
)
QVINA_SHA256 = "f8ac045235025e98b15fd90aae6617edfdcc125081f72a5a5315db22be1f46e0"
RECEPTOR_SHA256 = "8d0891ddf915f51cf3108f39dd9dc01dfcf86be342c566021956afdd79eb4ad9"
DOCKING_ADAPTER_SHA256 = "4d501837c1e61e36b34582be4d60b4f73818a2cbdd61906e0d0f5af20a917d2e"
MODAL_EXECUTION_PAYLOAD_SHA256 = "1b930b39e8491c293664787faf984c999e3663f7ad625016faa9517ed5aa7a2f"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def payload_identity(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or claimed != payload_identity(payload):
        raise ValueError(f"artifact is not a valid self-hashed envelope: {path}")
    return payload, claimed


def publish_once_durable(path: Path, payload: dict[str, Any]) -> str:
    """Atomically publish immutable canonical JSON and fsync file and directory."""

    envelope = {"payload": payload, "payload_sha256": payload_identity(payload)}
    content = _canonical_bytes(envelope) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
        return envelope["payload_sha256"]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return envelope["payload_sha256"]


def _git(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repository_root, text=True).strip()


def build_source_binding(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    revision = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    files: dict[str, dict[str, str]] = {}
    for relative in sorted(LOCAL_SOURCE_FILES):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"local execution source is absent: {path}")
        blob_oid = _git(root, "rev-parse", f"{revision}:{relative}")
        current_blob = _git(root, "hash-object", relative)
        if current_blob != blob_oid:
            raise ValueError(f"local execution source is not committed: {relative}")
        files[relative] = {
            "git_blob_oid": blob_oid,
            "sha256": file_sha256(path),
        }
    return {
        "schema_version": "t4_exact_commit_local_source_binding_v1",
        "code_revision": revision,
        "git_tree": tree,
        "file_count": len(files),
        "files": files,
    }


def validate_source_binding(repository_root: Path, binding: dict[str, Any]) -> None:
    root = repository_root.resolve()
    if (
        binding.get("schema_version") != "t4_exact_commit_local_source_binding_v1"
        or binding.get("file_count") != len(LOCAL_SOURCE_FILES)
        or sorted(binding.get("files", {})) != sorted(LOCAL_SOURCE_FILES)
    ):
        raise ValueError("local execution source binding schema changed")
    revision = binding.get("code_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("local execution source revision is malformed")
    if _git(root, "rev-parse", f"{revision}^{{tree}}") != binding.get("git_tree"):
        raise ValueError("local execution source tree changed")
    for relative, expected in binding["files"].items():
        path = root / relative
        if file_sha256(path) != expected.get("sha256"):
            raise ValueError(f"local execution source changed: {relative}")
        if _git(root, "rev-parse", f"{revision}:{relative}") != expected.get("git_blob_oid"):
            raise ValueError(f"local execution source blob changed: {relative}")


def _protocol(evaluator: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": evaluator["target"],
        "qvina02_sha256": evaluator["qvina02_sha256"],
        "receptor_sha256": evaluator["receptor_sha256"],
        "docking_box": evaluator["docking_box"],
        "docking_seed": evaluator["docking_seed"],
        "cpu": 1,
        "num_modes": 10,
        "exhaustiveness": 1,
        "ligand_preparation": [
            ["obabel", "-:{canonical_smiles}", "--gen3D", "-O", "{ligand_mol}"],
            ["obabel", "{ligand_mol}", "-O", "{ligand_pdbqt}"],
        ],
        "docking_command": [
            "/opt/dock/qvina02",
            "--receptor",
            "/opt/dock/receptors/parp1.pdbqt",
            "--ligand",
            "{ligand_pdbqt}",
            "--out",
            "{output_pdbqt}",
            "--center_x",
            "26.413",
            "--center_y",
            "11.282",
            "--center_z",
            "27.238",
            "--size_x",
            "18.521",
            "--size_y",
            "17.479",
            "--size_z",
            "19.995",
            "--cpu",
            "1",
            "--num_modes",
            "10",
            "--exhaustiveness",
            "1",
            "--seed",
            "20260919",
        ],
        "timeouts_seconds": {
            "obabel_gen3d": 120,
            "obabel_pdbqt": 60,
            "qvina02": 300,
        },
        "score_parser": {
            "line_prefix": "REMARK VINA RESULT",
            "whitespace_field_index": 3,
            "score_direction": "minimize",
            "score_unit": "kcal/mol",
        },
        "adapter": {
            "path": DOCKING_ADAPTER_SOURCE,
            "sha256": DOCKING_ADAPTER_SHA256,
            "identical_to_modal_execution_v2_capsule": True,
        },
    }


def _execution_policy() -> dict[str, Any]:
    return {
        "query_count": 4,
        "execution_order": "scientific_contract_query_order_sequential",
        "maximum_parallel_workers": 1,
        "cpu_per_worker": 1,
        "attempts_per_query": 1,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "replicate_calls": 0,
        "publish_complete_query_lock_before_first_container_score_process": True,
        "publish_all_reservations_before_first_container_score_process": True,
        "started_receipt_precedes_each_possible_score_call": True,
        "one_terminal_receipt_per_started_query": True,
        "started_without_terminal_is_terminally_incomplete_and_never_retried": True,
        "deterministic_reduction_order": "scientific_contract_query_order",
    }


def _receipt_policy() -> dict[str, Any]:
    return {
        "root_relative_path": LOCAL_RECEIPT_ROOT_RELATIVE_PATH,
        "format": "canonical_json_self_hashed_sha256_envelope",
        "publication": "exclusive_temp_file_fsync_atomic_replace_parent_fsync",
        "overwrite_policy": "same_bytes_idempotent_otherwise_refuse",
        "query_lock": "complete_four_query_payload_before_first_score_process",
        "reservation": "all_four_published_before_first_score_process",
        "terminal": "exactly_one_result_or_failure_after_started",
        "partial_campaign": "preserve_all_receipts_without_retry_or_backfill",
    }


def build_local_execution_payload(
    repository_root: Path, source_binding: dict[str, Any]
) -> dict[str, Any]:
    root = repository_root.resolve()
    scientific, scientific_identity, _authorization, authorization_identity = (
        validate_scored_authority(root)
    )
    modal_execution, modal_identity, _ = validate_execution_contract(root)
    if modal_identity != MODAL_EXECUTION_PAYLOAD_SHA256:
        raise ValueError("Modal execution binding identity changed")
    modal_manifest, _ = load_envelope(
        root / modal_execution["execution_source_capsule"]["manifest_relative_path"]
    )
    if modal_manifest["files"][DOCKING_ADAPTER_SOURCE]["sha256"] != DOCKING_ADAPTER_SHA256:
        raise ValueError("Modal execution capsule docking adapter changed")
    if scientific["evaluator"]["qvina02_sha256"] != QVINA_SHA256:
        raise ValueError("scientific qvina identity changed")
    if scientific["evaluator"]["receptor_sha256"] != RECEPTOR_SHA256:
        raise ValueError("scientific receptor identity changed")
    if file_sha256(root / DOCKING_ADAPTER_SOURCE) != DOCKING_ADAPTER_SHA256:
        raise ValueError("docking adapter differs from the Modal-bound adapter")
    queries = scientific["queries"]
    return {
        "schema_version": LOCAL_EXECUTION_SCHEMA,
        "status": "PENDING_EXACT_BINDING_SPECIFIC_LAUNCH_AUTHORIZATION",
        "scientific_scored_contract": {
            "path": SCORED_CONTRACT_RELATIVE_PATH,
            "sha256": file_sha256(root / SCORED_CONTRACT_RELATIVE_PATH),
            "payload_sha256": scientific_identity,
        },
        "existing_scored_authorization": {
            "path": AUTHORIZATION_RELATIVE_PATH,
            "sha256": file_sha256(root / AUTHORIZATION_RELATIVE_PATH),
            "payload_sha256": authorization_identity,
            "scope": "exact_scientific_four_call_payload_only",
        },
        "candidate_proposal": {
            "path": CANDIDATE_PROPOSAL_RELATIVE_PATH,
            "sha256": file_sha256(root / CANDIDATE_PROPOSAL_RELATIVE_PATH),
        },
        "prior_modal_execution_binding": {
            "path": EXECUTION_CONTRACT_RELATIVE_PATH,
            "sha256": file_sha256(root / EXECUTION_CONTRACT_RELATIVE_PATH),
            "payload_sha256": modal_identity,
            "status": modal_execution["status"],
            "scope": "protocol_and_receipt_equivalence_reference_not_launch_authority",
        },
        "query_lock": {
            "source": "scientific_scored_contract.queries",
            "ordered_queries_payload_sha256": payload_identity(queries),
            "ordered_query_ids": [row["query_id"] for row in queries],
            "query_count": len(queries),
            "attempt_ceiling_each": 1,
            "candidate_mutation": False,
        },
        "evaluator_protocol": _protocol(scientific["evaluator"]),
        "decision_equivalence": {
            "reference": "prior_modal_execution_binding",
            "identical_fields": [
                "ordered_locked_queries",
                "target",
                "qvina02_sha256",
                "receptor_sha256",
                "docking_box",
                "docking_seed",
                "ligand_preparation_argv",
                "qvina02_argv",
                "cpu",
                "num_modes",
                "exhaustiveness",
                "score_parser",
                "one_attempt_no_retry_replacement_backfill_or_replication",
            ],
            "operational_differences_only": {
                "compute": "local_Docker_Linux_amd64_instead_of_Modal",
                "dispatch": "sequential_instead_of_up_to_four_parallel_workers",
                "durability": "local_atomic_fsync_receipts_instead_of_Modal_Volume_commit",
            },
            "scientific_decision_change": False,
        },
        "container": {
            "engine": "Docker",
            "platform": "linux/amd64",
            "runtime_image_tag": "compose-t4-topology-four-call-local:v1",
            "recipe": {
                "path": CONTAINER_RECIPE_RELATIVE_PATH,
                "sha256": file_sha256(root / CONTAINER_RECIPE_RELATIVE_PATH),
                "build_context": ("containers/t4_nodistill_topology_four_call_local"),
                "contains_private_source": False,
            },
            "runtime_network": "none",
            "repository_mount": {"container_path": "/workspace", "mode": "read_only"},
            "receipt_mount": {"container_path": "/receipts", "mode": "read_write"},
            "qvina_mount": {"container_path": "/inputs/qvina02", "mode": "read_only"},
            "receptor_mount": {
                "container_path": "/inputs/parp1.pdbqt",
                "mode": "read_only",
            },
            "ephemeral_paths": ["/opt/dock", "/tmp"],
            "source_or_image_push": "forbidden",
            "environment_attestation": {
                "path": LOCAL_ENVIRONMENT_ATTESTATION_RELATIVE_PATH,
                "required_before_launch_authorization": True,
                "must_bind": [
                    "local_execution_contract_payload_sha256",
                    "recipe_sha256",
                    "image_id",
                    "operating_system_linux",
                    "architecture_amd64",
                    "python_version",
                    "openbabel_version",
                    "openbabel_sha256",
                    "qvina02_sha256",
                    "receptor_sha256",
                    "zero_oracle_preflight",
                ],
            },
        },
        "local_assets": {
            "qvina02": {"path": str(LOCAL_QVINA_PATH), "sha256": QVINA_SHA256},
            "parp1_receptor": {
                "path": str(LOCAL_RECEPTOR_PATH),
                "sha256": RECEPTOR_SHA256,
            },
            "verification": (
                "required_during_environment_attestation_and_immediately_before_first_score_process"
            ),
        },
        "execution": _execution_policy(),
        "receipts": _receipt_policy(),
        "source_binding": source_binding,
        "launch_authorization": {
            "state": "PENDING_EXACT_BINDING_SPECIFIC_LAUNCH_AUTHORIZATION",
            "path": LOCAL_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
            "required_schema_version": LOCAL_AUTHORIZATION_SCHEMA,
            "must_bind": [
                "scientific_contract_payload_sha256",
                "existing_scored_authorization_payload_sha256",
                "local_execution_contract_payload_sha256",
                "environment_attestation_payload_sha256",
                "authorized_scored_calls_exactly_4",
                "zero_retries_replacements_backfill_or_replicates",
            ],
            "launch_rule": "refuse_until_exact_self_hashed_receipt_exists",
        },
        "costs_spent_during_preparation": {
            "docking_calls": 0,
            "oracle_calls": 0,
            "modal_calls": 0,
            "container_builds": 0,
            "container_runs": 0,
        },
        "claim_boundary": (
            "Separate no-egress operational fallback for the unchanged authorized "
            "four-query diagnostic. This binding makes no score, changes no locked "
            "candidate, and is not launch authority."
        ),
    }


def seal_local_execution_binding(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    source_binding = build_source_binding(root)
    payload = build_local_execution_payload(root, source_binding)
    path = root / LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH
    identity = publish_once_durable(path, payload)
    return {
        "path": str(path),
        "payload_sha256": identity,
        "sha256": file_sha256(path),
        "status": payload["status"],
        "docking_calls": 0,
        "container_runs": 0,
    }


def validate_local_assets() -> dict[str, str]:
    observed = {
        "qvina02_sha256": file_sha256(LOCAL_QVINA_PATH),
        "receptor_sha256": file_sha256(LOCAL_RECEPTOR_PATH),
    }
    expected = {
        "qvina02_sha256": QVINA_SHA256,
        "receptor_sha256": RECEPTOR_SHA256,
    }
    if observed != expected:
        raise ValueError("local evaluator assets differ from the scientific contract")
    return observed


def validate_local_execution_binding(
    repository_root: Path, *, require_local_assets: bool = False
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    root = repository_root.resolve()
    path = root / LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH
    payload, identity = load_envelope(path)
    source_binding = payload.get("source_binding")
    if not isinstance(source_binding, dict):
        raise TypeError("local execution source binding is absent")
    validate_source_binding(root, source_binding)
    expected = build_local_execution_payload(root, source_binding)
    if payload != expected:
        raise ValueError("local execution binding changed")
    if require_local_assets:
        validate_local_assets()
    scientific, _, _, _ = validate_scored_authority(root)
    return payload, identity, scientific


def validate_launch_authorization(
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], str]:
    """Require a future exact binding and environment-specific launch receipt."""

    root = repository_root.resolve()
    binding, binding_identity, _ = validate_local_execution_binding(root, require_local_assets=True)
    environment, environment_identity = load_envelope(
        root / LOCAL_ENVIRONMENT_ATTESTATION_RELATIVE_PATH
    )
    preflight = environment.get("zero_oracle_preflight")
    if (
        environment.get("schema_version") != LOCAL_ENVIRONMENT_SCHEMA
        or environment.get("status") != "PASS_ZERO_ORACLE"
        or environment.get("local_execution_contract_payload_sha256") != binding_identity
        or environment.get("recipe_sha256") != binding["container"]["recipe"]["sha256"]
        or environment.get("image_tag") != binding["container"]["runtime_image_tag"]
        or not isinstance(environment.get("image_id"), str)
        or not environment["image_id"].startswith("sha256:")
        or not isinstance(environment.get("repo_digests"), list)
        or environment.get("platform") != {"os": "linux", "architecture": "amd64"}
        or not isinstance(environment.get("python_version"), str)
        or not isinstance(environment.get("openbabel_version"), str)
        or not isinstance(environment.get("openbabel_sha256"), str)
        or len(environment["openbabel_sha256"]) != 64
        or environment.get("qvina02_sha256") != QVINA_SHA256
        or environment.get("receptor_sha256") != RECEPTOR_SHA256
        or environment.get("runtime_network") != "none"
        or not isinstance(preflight, dict)
        or preflight.get("status") != "PASS_ZERO_ORACLE"
        or preflight.get("local_execution_contract_payload_sha256") != binding_identity
        or preflight.get("platform") != {"os": "linux", "architecture": "x86_64"}
        or preflight.get("python_version") != environment.get("python_version")
        or preflight.get("openbabel_version") != environment.get("openbabel_version")
        or preflight.get("openbabel_sha256") != environment.get("openbabel_sha256")
        or preflight.get("qvina02_sha256") != QVINA_SHA256
        or preflight.get("receptor_sha256") != RECEPTOR_SHA256
        or preflight.get("docking_calls") != 0
        or preflight.get("oracle_calls") != 0
        or environment.get("docking_calls") != 0
        or environment.get("oracle_calls") != 0
    ):
        raise ValueError("local container environment attestation is not exact")
    authorization, authorization_identity = load_envelope(
        root / LOCAL_LAUNCH_AUTHORIZATION_RELATIVE_PATH
    )
    expected = {
        "schema_version": LOCAL_AUTHORIZATION_SCHEMA,
        "scientific_contract_payload_sha256": SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
        "existing_scored_authorization_payload_sha256": (AUTHORIZATION_PAYLOAD_SHA256),
        "local_execution_contract_payload_sha256": binding_identity,
        "environment_attestation_payload_sha256": environment_identity,
        "authorized_scored_calls": 4,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "replicate_calls": 0,
        "user_statement": (
            "I authorize exactly 4 scored docking calls under scientific contract "
            f"{SCIENTIFIC_CONTRACT_PAYLOAD_SHA256} using local execution binding "
            f"{binding_identity} and environment attestation {environment_identity}, "
            "with zero retries, replacements, backfill, or replicates."
        ),
    }
    if authorization != expected:
        raise ValueError("local launch authorization is not exact")
    return binding, environment, environment_identity, authorization, authorization_identity


__all__ = [
    "CONTAINER_RECIPE_RELATIVE_PATH",
    "LOCAL_AUTHORIZATION_SCHEMA",
    "LOCAL_ENVIRONMENT_ATTESTATION_RELATIVE_PATH",
    "LOCAL_EXECUTION_CONTRACT_RELATIVE_PATH",
    "LOCAL_EXECUTION_ROOT",
    "LOCAL_LAUNCH_AUTHORIZATION_RELATIVE_PATH",
    "LOCAL_QVINA_PATH",
    "LOCAL_RECEIPT_ROOT_RELATIVE_PATH",
    "LOCAL_RECEPTOR_PATH",
    "LOCAL_RUNTIME_SOURCE",
    "LOCAL_TOOL_SOURCE",
    "build_local_execution_payload",
    "build_source_binding",
    "file_sha256",
    "load_envelope",
    "payload_identity",
    "publish_once_durable",
    "seal_local_execution_binding",
    "validate_launch_authorization",
    "validate_local_assets",
    "validate_local_execution_binding",
]
