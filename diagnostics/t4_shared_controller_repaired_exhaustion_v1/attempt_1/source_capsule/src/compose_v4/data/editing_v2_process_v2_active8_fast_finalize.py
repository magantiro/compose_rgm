"""Fast, provenance-bound publication of an already prepared Active8 run.

The original finalizer recomputes the complete reduction from molecular row
artifacts after the preparation has already frozen that reduction.  This
publisher instead authenticates the immutable handoff at its content boundary:

* the plan and reduction preparation are validated once;
* every task receipt and summary is authenticated;
* the row and transition payloads are streamed once for SHA-256 verification,
  without JSON decoding or molecular work;
* every release-sentinel result is read once and reduced by the production
  sentinel reducer; and
* the canonical production completion is reproduced byte for byte.

This module is downstream of the frozen Active8 map identity.  A separate
receipt records the exact publisher implementation that authenticated and
committed the completion.  It grants no training or experimental authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from compose_v4.data.editing_process_v2_rebind import mounted_process_v2_artifact_path
from compose_v4.data.editing_v2_process_v2_active8_map import (
    RECEIPT_FILENAME,
    ROWS_FILENAME,
    SUMMARY_FIELDS,
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA,
    SUMMARY_SCHEMA_VERSION,
    TASK_STATUS,
    TRANSITIONS_FILENAME,
)
from compose_v4.data.editing_v2_process_v2_active8_plan import (
    validate_process_v2_active8_plan,
)
from compose_v4.data.editing_v2_process_v2_active8_reduce import (
    COMPLETION_FIELDS,
    COMPLETION_FILENAME,
    task_output_path,
    validate_process_v2_active8_completion,
    validate_process_v2_active8_reduction_preparation,
)
from compose_v4.data.editing_v2_process_v2_active8_sentinel import (
    ProcessV2Active8SentinelError,
    reduce_release_sentinel_partitions,
    require_sentinel_passed,
    sentinel_partition_identities,
    sentinel_partition_result_path,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACTIVE8_COMPLETION_SCHEMA,
    ACTIVE8_COMPLETION_SCHEMA_VERSION,
    ACTIVE8_RECEIPT_FIELDS,
    ACTIVE8_TASK_SCHEMA,
    ACTIVE8_TASK_SCHEMA_VERSION,
    PIPELINE_STATUS_NO_AUTHORITY,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
    self_hashed,
    verify_self_hash,
)
from compose_v4.data.immutable_artifact import (
    ImmutableArtifactError,
    write_bytes_if_absent,
)

FAST_FINALIZATION_RECEIPT_FILENAME = "PROCESS_V2_ACTIVE8_FAST_FINALIZATION_RECEIPT.json"
FAST_FINALIZATION_RECEIPT_SCHEMA = (
    "compose.data.editing_v2_process_v2_active8_fast_finalization_receipt"
)
FAST_FINALIZATION_RECEIPT_SCHEMA_VERSION = 1
FAST_FINALIZATION_STATUS = "PROCESS_V2_PIPELINE_EVIDENCE_ONLY_NO_AUTHORITY"
VERIFICATION_MODE = "streaming_payload_sha256_without_row_decode"
DEFAULT_HASH_WORKERS = 16
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

FAST_FINALIZATION_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "verification_mode",
        "plan_sha256",
        "preparation_sha256",
        "task_result_content_inventory_sha256",
        "sentinel_plan_sha256",
        "sentinel_result_inventory_sha256",
        "verified_task_count",
        "verified_sentinel_partition_count",
        "completion_sha256",
        "publisher_commit",
        "publisher_tree",
        "publisher_image_revision_sha256",
        "publisher_serialized_sources_sha256",
        "receipt_sha256",
    }
)


class ProcessV2Active8FastFinalizeError(RuntimeError):
    """The prepared Active8 evidence cannot be authenticated cheaply."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    except OSError as error:
        raise ProcessV2Active8FastFinalizeError(
            f"cannot read immutable Active8 payload: {path}"
        ) from error
    return digest.hexdigest()


def _canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except OSError as error:
        raise ProcessV2Active8FastFinalizeError(f"{label} is absent: {path}") from error
    except json.JSONDecodeError as error:
        raise ProcessV2Active8FastFinalizeError(
            f"{label} is not valid JSON: {path}"
        ) from error
    if not isinstance(value, dict) or canonical_bytes(value) + b"\n" != raw:
        raise ProcessV2Active8FastFinalizeError(
            f"{label} is not a canonical JSON mapping: {path}"
        )
    return value


def _require_exact_task_directory(output: Path) -> None:
    if output.is_symlink() or not output.is_dir():
        raise ProcessV2Active8FastFinalizeError(
            f"the Active8 task output is not a real directory: {output}"
        )
    entries = tuple(output.iterdir())
    expected = {
        ROWS_FILENAME,
        TRANSITIONS_FILENAME,
        RECEIPT_FILENAME,
        SUMMARY_FILENAME,
    }
    if (
        any(path.is_symlink() or not path.is_file() for path in entries)
        or {path.name for path in entries} != expected
    ):
        raise ProcessV2Active8FastFinalizeError(
            f"the Active8 task output inventory is incomplete or unexpected: {output}"
        )


def _verify_task_payload(
    task: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    artifact_root: Path,
    expected_receipt_sha256: str,
    expected_summary_sha256: str,
) -> tuple[str, str, str]:
    identity = str(task["task_identity_sha256"])
    output = task_output_path(plan, task, artifact_root=artifact_root)
    _require_exact_task_directory(output)
    receipt = _canonical_json(output / RECEIPT_FILENAME, label="the Active8 receipt")
    summary = _canonical_json(output / SUMMARY_FILENAME, label="the Active8 task summary")
    if set(receipt) != set(ACTIVE8_RECEIPT_FIELDS):
        raise ProcessV2Active8FastFinalizeError(
            f"the Active8 receipt field set disagrees for task {identity}"
        )
    if set(summary) != set(SUMMARY_FIELDS):
        raise ProcessV2Active8FastFinalizeError(
            f"the Active8 summary field set disagrees for task {identity}"
        )
    verify_self_hash(receipt, field="receipt_sha256", label="the Active8 receipt")
    verify_self_hash(summary, field="summary_sha256", label="the Active8 task summary")
    if (
        receipt["schema"] != ACTIVE8_TASK_SCHEMA
        or receipt["schema_version"] != ACTIVE8_TASK_SCHEMA_VERSION
        or receipt["status"] != TASK_STATUS
        or summary["schema"] != SUMMARY_SCHEMA
        or summary["schema_version"] != SUMMARY_SCHEMA_VERSION
        or summary["status"] != TASK_STATUS
        or summary["classification_affects_admission"] is not False
        or receipt["task_identity_sha256"] != identity
        or summary["task_identity_sha256"] != identity
        or summary["receipt_sha256"] != receipt["receipt_sha256"]
        or summary["binding_sha256"] != plan["binding_sha256"]
        or summary["plan_sha256"] != plan["plan_sha256"]
        or summary["run_identity_sha256"] != plan["run_identity_sha256"]
        or receipt["receipt_sha256"] != expected_receipt_sha256
        or summary["summary_sha256"] != expected_summary_sha256
    ):
        raise ProcessV2Active8FastFinalizeError(
            f"the Active8 task metadata disagrees with the preparation: {identity}"
        )
    rows_sha256 = _file_sha256(output / ROWS_FILENAME)
    transitions_sha256 = _file_sha256(output / TRANSITIONS_FILENAME)
    if rows_sha256 != summary["rows_file_sha256"]:
        raise ProcessV2Active8FastFinalizeError(
            f"the Active8 rows payload changed after preparation: {identity}"
        )
    if transitions_sha256 != receipt["decision_shard_sha256"]:
        raise ProcessV2Active8FastFinalizeError(
            f"the Active8 transition payload changed after preparation: {identity}"
        )
    return identity, str(receipt["receipt_sha256"]), str(summary["summary_sha256"])


def verify_task_payload_inventory(
    plan: Mapping[str, Any],
    preparation: Mapping[str, Any],
    *,
    artifact_root: Path,
    hash_workers: int = DEFAULT_HASH_WORKERS,
) -> list[list[str]]:
    """Verify every frozen task payload once without parsing molecular rows."""

    if type(hash_workers) is not int or hash_workers <= 0 or hash_workers > 64:
        raise ProcessV2Active8FastFinalizeError("hash_workers must be an integer in [1, 64]")
    expected = preparation["task_result_content_inventory"]
    tasks = list(plan["tasks"])
    if (
        not isinstance(expected, list)
        or len(expected) != len(tasks)
        or any(not isinstance(item, list) or len(item) != 3 for item in expected)
        or [item[0] for item in expected]
        != [str(task["task_identity_sha256"]) for task in tasks]
    ):
        raise ProcessV2Active8FastFinalizeError(
            "the prepared task-content inventory does not match the plan"
        )

    arguments = [
        (
            task,
            str(expected[index][1]),
            str(expected[index][2]),
        )
        for index, task in enumerate(tasks)
    ]

    def verify(argument: tuple[Mapping[str, Any], str, str]) -> tuple[str, str, str]:
        task, receipt_sha256, summary_sha256 = argument
        return _verify_task_payload(
            task,
            plan=plan,
            artifact_root=Path(artifact_root),
            expected_receipt_sha256=receipt_sha256,
            expected_summary_sha256=summary_sha256,
        )

    with ThreadPoolExecutor(max_workers=min(hash_workers, len(tasks))) as executor:
        observed = list(executor.map(verify, arguments))
    normalized = [list(item) for item in observed]
    if normalized != expected or canonical_sha256(normalized) != preparation[
        "task_result_content_inventory_sha256"
    ]:
        raise ProcessV2Active8FastFinalizeError(
            "the current Active8 task payload inventory disagrees with preparation"
        )
    return normalized


def _load_sentinel_results_once(
    plan: Mapping[str, Any],
    sentinel_plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], str]:
    identities = sentinel_partition_identities(sentinel_plan)
    results: list[dict[str, Any]] = []
    inventory: list[list[str]] = []
    for identity in identities:
        path = sentinel_partition_result_path(
            plan,
            identity,
            artifact_root=Path(artifact_root),
        )
        result = _canonical_json(path, label="the Active8 sentinel partition result")
        results.append(result)
        result_sha256 = result.get("partition_result_sha256")
        if not isinstance(result_sha256, str) or _SHA256.fullmatch(result_sha256) is None:
            raise ProcessV2Active8FastFinalizeError(
                f"the sentinel result has no valid self-hash: {identity}"
            )
        inventory.append([identity, result_sha256])
    return results, canonical_sha256(inventory)


def _completion_from_preparation(
    preparation: Mapping[str, Any], sentinel: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "schema": ACTIVE8_COMPLETION_SCHEMA,
        "schema_version": ACTIVE8_COMPLETION_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "accepted_transitions": int(preparation["accepted_transitions"]),
        "action_family_histogram": dict(preparation["action_family_histogram"]),
        "active8_exclusion_histogram": dict(preparation["active8_exclusion_histogram"]),
        "binding_sha256": str(preparation["binding_sha256"]),
        "candidate_totals": dict(preparation["candidate_totals"]),
        "capability_cell_histogram": dict(preparation["capability_cell_histogram"]),
        "census": dict(preparation["census"]),
        "classification_affects_admission": False,
        "plan_sha256": str(preparation["plan_sha256"]),
        "result_inventory": list(preparation["result_inventory"]),
        "result_inventory_sha256": str(preparation["result_inventory_sha256"]),
        "run_artifact_root": str(preparation["run_artifact_root"]),
        "run_identity_sha256": str(preparation["run_identity_sha256"]),
        "sentinel": dict(sentinel),
        "task_inventory_sha256": str(preparation["task_inventory_sha256"]),
    }
    completion = self_hashed(body, field="completion_sha256")
    if set(completion) != COMPLETION_FIELDS:
        raise ProcessV2Active8FastFinalizeError("the fast completion field set disagrees")
    return validate_process_v2_active8_completion(completion)


def _publisher_identity(revision: Mapping[str, Any]) -> dict[str, str]:
    sources = revision.get("serialized_sources")
    values = {
        "publisher_commit": revision.get("commit"),
        "publisher_tree": revision.get("tree"),
        "publisher_image_revision_sha256": revision.get("image_revision_sha256"),
        "publisher_serialized_sources_sha256": (
            canonical_sha256(sources) if isinstance(sources, Mapping) else None
        ),
    }
    if (
        not isinstance(values["publisher_commit"], str)
        or _COMMIT.fullmatch(values["publisher_commit"]) is None
        or not isinstance(values["publisher_tree"], str)
        or _COMMIT.fullmatch(values["publisher_tree"]) is None
        or any(
            not isinstance(values[field], str) or _SHA256.fullmatch(values[field]) is None
            for field in (
                "publisher_image_revision_sha256",
                "publisher_serialized_sources_sha256",
            )
        )
    ):
        raise ProcessV2Active8FastFinalizeError(
            "the fast-finalizer publisher revision is incomplete"
        )
    return {key: str(value) for key, value in values.items()}


def validate_fast_finalization_receipt(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != FAST_FINALIZATION_RECEIPT_FIELDS:
        raise ProcessV2Active8FastFinalizeError(
            "the fast-finalization receipt field set disagrees"
        )
    receipt = dict(value)
    verify_self_hash(receipt, field="receipt_sha256", label="the fast-finalization receipt")
    try:
        require_authority_false(receipt, label="the fast-finalization receipt")
    except ProcessV2SchemaError as error:
        raise ProcessV2Active8FastFinalizeError(str(error)) from error
    if (
        receipt["schema"] != FAST_FINALIZATION_RECEIPT_SCHEMA
        or receipt["schema_version"] != FAST_FINALIZATION_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != FAST_FINALIZATION_STATUS
        or receipt["verification_mode"] != VERIFICATION_MODE
        or type(receipt["verified_task_count"]) is not int
        or receipt["verified_task_count"] <= 0
        or type(receipt["verified_sentinel_partition_count"]) is not int
        or receipt["verified_sentinel_partition_count"] <= 0
    ):
        raise ProcessV2Active8FastFinalizeError(
            "the fast-finalization receipt contract disagrees"
        )
    for field in (
        "plan_sha256",
        "preparation_sha256",
        "task_result_content_inventory_sha256",
        "sentinel_plan_sha256",
        "sentinel_result_inventory_sha256",
        "completion_sha256",
        "publisher_image_revision_sha256",
        "publisher_serialized_sources_sha256",
    ):
        if not isinstance(receipt[field], str) or _SHA256.fullmatch(receipt[field]) is None:
            raise ProcessV2Active8FastFinalizeError(
                f"the fast-finalization receipt {field} is not a SHA-256"
            )
    for field in ("publisher_commit", "publisher_tree"):
        if not isinstance(receipt[field], str) or _COMMIT.fullmatch(receipt[field]) is None:
            raise ProcessV2Active8FastFinalizeError(
                f"the fast-finalization receipt {field} is not a Git identity"
            )
    return receipt


def authenticate_fast_finalization_publication(
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the completion and its exact-code fast-publisher receipt."""

    root = Path(run_root)
    completion = validate_process_v2_active8_completion(
        _canonical_json(root / COMPLETION_FILENAME, label="the Active8 completion")
    )
    receipt = validate_fast_finalization_receipt(
        _canonical_json(
            root / FAST_FINALIZATION_RECEIPT_FILENAME,
            label="the Active8 fast-finalization receipt",
        )
    )
    if (
        receipt["completion_sha256"] != completion["completion_sha256"]
        or receipt["plan_sha256"] != completion["plan_sha256"]
        or receipt["verified_task_count"] != len(completion["result_inventory"])
    ):
        raise ProcessV2Active8FastFinalizeError(
            "the fast-finalization receipt does not authenticate this completion"
        )
    return completion, receipt


def fast_finalize_process_v2_active8(
    plan: Mapping[str, Any],
    preparation: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
    publisher_revision: Mapping[str, Any],
    hash_workers: int = DEFAULT_HASH_WORKERS,
    publish: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate frozen evidence and publish the canonical completion quickly."""

    validated_plan = validate_process_v2_active8_plan(plan, repo_root=Path(repo_root))
    validated_preparation = validate_process_v2_active8_reduction_preparation(
        preparation,
        active8_plan=validated_plan,
    )
    publisher = _publisher_identity(publisher_revision)
    verified_tasks = verify_task_payload_inventory(
        validated_plan,
        validated_preparation,
        artifact_root=Path(artifact_root),
        hash_workers=hash_workers,
    )
    results, sentinel_result_inventory_sha256 = _load_sentinel_results_once(
        validated_plan,
        validated_preparation["sentinel_plan"],
        artifact_root=Path(artifact_root),
    )
    try:
        sentinel = reduce_release_sentinel_partitions(
            validated_plan,
            validated_preparation["sentinel_plan"],
            results,
        )
        require_sentinel_passed(
            sentinel,
            binding_sha256=str(validated_preparation["binding_sha256"]),
            plan_sha256=str(validated_preparation["plan_sha256"]),
            run_identity_sha256=str(validated_preparation["run_identity_sha256"]),
            task_inventory_sha256=str(validated_preparation["task_inventory_sha256"]),
            result_inventory_sha256=str(validated_preparation["result_inventory_sha256"]),
        )
    except ProcessV2Active8SentinelError as error:
        raise ProcessV2Active8FastFinalizeError(str(error)) from error

    completion = _completion_from_preparation(validated_preparation, sentinel)
    receipt_body = {
        "schema": FAST_FINALIZATION_RECEIPT_SCHEMA,
        "schema_version": FAST_FINALIZATION_RECEIPT_SCHEMA_VERSION,
        "status": FAST_FINALIZATION_STATUS,
        **authority_false_block(),
        "verification_mode": VERIFICATION_MODE,
        "plan_sha256": str(validated_plan["plan_sha256"]),
        "preparation_sha256": str(validated_preparation["preparation_sha256"]),
        "task_result_content_inventory_sha256": canonical_sha256(verified_tasks),
        "sentinel_plan_sha256": str(
            validated_preparation["sentinel_plan"]["sentinel_plan_sha256"]
        ),
        "sentinel_result_inventory_sha256": sentinel_result_inventory_sha256,
        "verified_task_count": len(verified_tasks),
        "verified_sentinel_partition_count": len(results),
        "completion_sha256": str(completion["completion_sha256"]),
        **publisher,
    }
    receipt = validate_fast_finalization_receipt(
        self_hashed(receipt_body, field="receipt_sha256")
    )
    if publish:
        run_root = mounted_process_v2_artifact_path(
            str(validated_plan["run_artifact_root"]),
            artifact_root=Path(artifact_root),
            field="plan.run_artifact_root",
        )
        run_root.mkdir(parents=True, exist_ok=True)
        try:
            write_bytes_if_absent(
                run_root / FAST_FINALIZATION_RECEIPT_FILENAME,
                canonical_bytes(receipt) + b"\n",
            )
            write_bytes_if_absent(
                run_root / COMPLETION_FILENAME,
                canonical_bytes(completion) + b"\n",
            )
        except ImmutableArtifactError as error:
            raise ProcessV2Active8FastFinalizeError(
                "immutable fast-finalization publication failed"
            ) from error
    return completion, receipt


__all__ = [
    "DEFAULT_HASH_WORKERS",
    "FAST_FINALIZATION_RECEIPT_FILENAME",
    "ProcessV2Active8FastFinalizeError",
    "authenticate_fast_finalization_publication",
    "fast_finalize_process_v2_active8",
    "validate_fast_finalization_receipt",
    "verify_task_payload_inventory",
]
