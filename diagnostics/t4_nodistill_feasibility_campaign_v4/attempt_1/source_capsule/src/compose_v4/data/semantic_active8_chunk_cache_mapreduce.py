"""Restart-safe orchestration for the 20-cell semantic Active8 chunk cache.

This layer freezes one exact source task per semantic lane/role cell, delegates
only mechanical chunk construction, and publishes a global content-addressed
completion after independently reloading every per-source cache.  It grants no
Active8, Gate 0, training, checkpoint-selection, or final-test authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    ACTIVE8_ADMISSION_STATUS,
    SOURCE_INVENTORY_STATUS,
    EditingV2SemanticActive8SourceInventory,
)
from compose_v4.data.semantic_active8_chunk_cache import (
    build_semantic_active8_chunk_cache,
    load_semantic_active8_chunk_cache,
    semantic_active8_chunk_cache_builder_identity,
)
from compose_v4.data.semantic_trace_migration_mapreduce import EXPECTED_TASK_COUNT

PLAN_SCHEMA = "compose.data.semantic_active8_chunk_cache_plan"
PLAN_SCHEMA_VERSION = 1
PLAN_STATUS = "FROZEN_MECHANICAL_CACHE_PLAN_NO_ACTIVE8_AUTHORITY"
SOURCE_RECEIPT_SCHEMA = "compose.data.semantic_active8_chunk_cache_source_receipt"
SOURCE_RECEIPT_SCHEMA_VERSION = 1
SOURCE_RECEIPT_STATUS = "COMPLETE_MECHANICAL_SOURCE_CACHE_NO_ACTIVE8_AUTHORITY"
GLOBAL_COMPLETION_SCHEMA = "compose.data.semantic_active8_chunk_cache_global_completion"
GLOBAL_COMPLETION_SCHEMA_VERSION = 1
GLOBAL_COMPLETION_STATUS = "COMPLETE_20_SOURCE_CACHE_NO_ACTIVE8_AUTHORITY"
PLAN_FILENAME = "PLAN.json"
GLOBAL_COMPLETION_FILENAME = "GLOBAL_COMPLETE.json"

_AUTHORITY = {
    "training_authorized": False,
    "active8_admission_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_SOURCE_REVISION_FIELDS = {
    "schema",
    "schema_version",
    "commit",
    "tree",
    "worktree_clean",
    "serialized_sources",
    "chunk_builder_identity",
    "source_revision_sha256",
}
_TASK_FIELDS = {
    "task_identity_sha256",
    "task_index",
    "data_lane",
    "partition_role",
    "migration_task_identity_sha256",
    "semantic_artifact_path",
    "semantic_shard_sha256",
    "semantic_manifest_file_sha256",
    "semantic_manifest_sha256",
    "semantic_completion_file_sha256",
    "semantic_completion_sha256",
    "semantic_record_stream_sha256",
    "process_identity_sha256",
    "action_codec_schema_version",
    "action_codec_implementation_hash",
    "source_binding",
    "family_histogram",
    "counts",
}
_RECEIPT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *_AUTHORITY,
    "global_run_identity_sha256",
    "task_identity_sha256",
    "source_revision_sha256",
    "source_task",
    "cache_run_identity_sha256",
    "cache_completion_sha256",
    "cache_chunk_count",
    "cache_chunk_inventory_sha256",
    "cache_source_identity_sha256",
    "receipt_sha256",
}


class SemanticActive8ChunkCacheMapReduceError(RuntimeError):
    """The global semantic chunk-cache boundary cannot be established."""


class SemanticActive8ChunkCacheIncomplete(SemanticActive8ChunkCacheMapReduceError):
    """At least one of the exact 20 source caches is absent."""


@dataclass(frozen=True, slots=True)
class SemanticActive8ChunkCacheReductionWitness:
    """One strict reduction plus the exact cache objects it validated in memory."""

    reduction: dict[str, Any]
    validated_caches: tuple[dict[str, Any], ...]


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SemanticActive8ChunkCacheMapReduceError(f"{field} must be a lowercase SHA-256")
    return value


def _require_artifact_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise SemanticActive8ChunkCacheMapReduceError(
            f"{field} must be a normalized /artifacts path"
        )
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise SemanticActive8ChunkCacheMapReduceError(
            f"{field} must be a normalized /artifacts path"
        )
    return value


def _mounted_path(value: object, *, artifact_root: Path, field: str) -> Path:
    address = _require_artifact_path(value, field=field)
    pure = PurePosixPath(address)
    root = Path(artifact_root).resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise SemanticActive8ChunkCacheMapReduceError(
            f"{field} resolves outside the artifact root"
        ) from error
    return resolved


def _publish_immutable(path: Path, content: bytes) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(content).hexdigest()
    if target.exists():
        if _file_sha256(target) != expected or target.read_bytes() != content:
            raise SemanticActive8ChunkCacheMapReduceError(
                f"immutable semantic chunk-cache orchestration collision at {target}"
            )
        return True
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        if _file_sha256(target) != expected or target.read_bytes() != content:
            raise SemanticActive8ChunkCacheMapReduceError(
                f"immutable semantic chunk-cache orchestration collision at {target}"
            )
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return False


def _validate_source_revision(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SOURCE_REVISION_FIELDS:
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache source revision fields disagree"
        )
    body = {key: item for key, item in value.items() if key != "source_revision_sha256"}
    if value["source_revision_sha256"] != _sha256(body):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache source revision self-hash disagrees"
        )
    if value.get("worktree_clean") is not True:
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache source revision is not clean"
        )
    for field in ("commit", "tree"):
        item = value.get(field)
        if (
            not isinstance(item, str)
            or len(item) != 40
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise SemanticActive8ChunkCacheMapReduceError(
                f"source_revision.{field} must be a full Git object"
            )
    sources = value.get("serialized_sources")
    if (
        not isinstance(sources, dict)
        or not sources
        or any(
            not isinstance(path, str)
            or not path
            or len(path) != len(path.strip())
            or _require_sha256(digest, field=f"serialized_sources.{path}") != digest
            for path, digest in sources.items()
        )
    ):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache serialized source hashes are malformed"
        )
    if value["chunk_builder_identity"] != semantic_active8_chunk_cache_builder_identity():
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache builder identity is stale"
        )
    return value


def _count_flow(source: object) -> dict[str, int]:
    counts = source.counts
    return {
        "candidate_traces": counts.candidate_traces,
        "migration_source_traces": counts.migration_source_traces,
        "migration_admitted_traces": counts.migration_admitted_traces,
        "migration_rejected_traces": counts.migration_rejected_traces,
        "semantic_entries": counts.semantic_entries,
        "semantic_states": counts.semantic_states,
        "semantic_actions": counts.semantic_actions,
    }


def _migration_identity(
    inventory: EditingV2SemanticActive8SourceInventory,
) -> dict[str, object]:
    return {
        "source_inventory_status": inventory.status,
        "source_training_authorized": inventory.training_authorized,
        "active8_admission_status": inventory.active8_admission_status,
        "migration_completion_file_sha256": inventory.migration_completion_file_sha256,
        "migration_completion_sha256": inventory.migration_completion_sha256,
        "migration_plan_file_sha256": inventory.migration_plan_file_sha256,
        "migration_plan_sha256": inventory.migration_plan_sha256,
        "migration_run_identity_sha256": inventory.run_identity_sha256,
        "migration_source_revision_sha256": inventory.source_revision_sha256,
        "migration_source_inventory_sha256": inventory.source_inventory_sha256,
        "migration_task_inventory_sha256": inventory.task_inventory_sha256,
        "migration_result_inventory_sha256": inventory.result_inventory_sha256,
        "process_identity_sha256": inventory.process_identity_sha256,
        "migration_builder_identity_sha256": inventory.builder_identity_sha256,
        "candidate_materialization_manifest_sha256": (
            inventory.candidate_materialization_manifest_sha256
        ),
        "candidate_provenance_source_stream_sha256": (
            inventory.candidate_provenance_source_stream_sha256
        ),
        "split_assignment_sha256": inventory.split_assignment_sha256,
        "lane_registry_sha256": inventory.lane_registry_sha256,
        "membership_receipt_file_sha256": inventory.membership_receipt_file_sha256,
        "membership_receipt_sha256": inventory.membership_receipt_sha256,
        "family_histogram": dict(inventory.family_histogram),
        "rejection_histogram": dict(inventory.rejection_histogram),
        "counts": _count_flow(inventory),
    }


def plan_semantic_active8_chunk_cache(
    inventory: EditingV2SemanticActive8SourceInventory,
    *,
    source_revision: Mapping[str, object],
    output_artifact_root: str,
    target_rows_per_chunk: int,
) -> dict[str, Any]:
    """Freeze the exact 20 semantic sources and one task per source."""

    revision = _validate_source_revision(dict(source_revision))
    output_artifact_root = _require_artifact_path(
        output_artifact_root,
        field="output_artifact_root",
    )
    if type(target_rows_per_chunk) is not int or target_rows_per_chunk <= 0:
        raise ValueError("target_rows_per_chunk must be a positive integer")
    expected_cells = tuple(
        (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
    )
    if len(expected_cells) != EXPECTED_TASK_COUNT or len(inventory.sources) != EXPECTED_TASK_COUNT:
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache planning requires exactly 20 sources"
        )
    observed_cells = tuple(
        (source.data_lane, source.partition_role) for source in inventory.sources
    )
    if observed_cells != expected_cells:
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache sources are not the exact ordered lane/role grid"
        )
    if (
        inventory.status != SOURCE_INVENTORY_STATUS
        or inventory.training_authorized is not False
        or inventory.active8_admission_status != ACTIVE8_ADMISSION_STATUS
    ):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic source inventory crosses its pre-Active8 boundary"
        )

    policy_body = {
        "algorithm": "one_mechanical_cache_per_semantic_source_v1",
        "target_rows_per_chunk": target_rows_per_chunk,
        "source_task_count": EXPECTED_TASK_COUNT,
    }
    policy = {**policy_body, "policy_sha256": _sha256(policy_body)}
    migration_identity = _migration_identity(inventory)
    task_bodies: list[dict[str, object]] = []
    for task_index, source in enumerate(inventory.sources):
        semantic_artifact_path = str(PurePosixPath(source.task_output_artifact_path) / "semantic")
        body: dict[str, object] = {
            "task_index": task_index,
            "data_lane": source.data_lane,
            "partition_role": source.partition_role,
            "migration_task_identity_sha256": source.task_identity_sha256,
            "semantic_artifact_path": _require_artifact_path(
                semantic_artifact_path,
                field=f"sources[{task_index}].semantic_artifact_path",
            ),
            "semantic_shard_sha256": source.semantic_shard_sha256,
            "semantic_manifest_file_sha256": source.semantic_manifest_file_sha256,
            "semantic_manifest_sha256": source.semantic_manifest_sha256,
            "semantic_completion_file_sha256": source.semantic_completion_file_sha256,
            "semantic_completion_sha256": source.semantic_completion_sha256,
            "semantic_record_stream_sha256": source.semantic_record_stream_sha256,
            "process_identity_sha256": source.process_identity_sha256,
            "action_codec_schema_version": source.action_codec_schema_version,
            "action_codec_implementation_hash": source.action_codec_implementation_hash,
            "source_binding": source.source_binding.as_mapping(),
            "family_histogram": dict(source.family_histogram),
            "counts": _count_flow(source),
        }
        task_identity = _sha256(
            {
                "source_revision_sha256": revision["source_revision_sha256"],
                "policy_sha256": policy["policy_sha256"],
                "source": body,
            }
        )
        task_bodies.append({"task_identity_sha256": task_identity, **body})
    run_body: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_revision": revision,
        "source_revision_sha256": revision["source_revision_sha256"],
        "migration_identity": migration_identity,
        "migration_identity_sha256": _sha256(migration_identity),
        "policy": policy,
        "output_artifact_root": output_artifact_root,
        "tasks": task_bodies,
        "task_inventory_sha256": _sha256(task_bodies),
    }
    run_identity = _sha256(run_body)
    plan_body: dict[str, object] = {
        **run_body,
        "status": PLAN_STATUS,
        **_AUTHORITY,
        "run_identity_sha256": run_identity,
        "expected_task_count": EXPECTED_TASK_COUNT,
    }
    return {**plan_body, "plan_sha256": _sha256(plan_body)}


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(plan)
    if (
        value.get("schema") != PLAN_SCHEMA
        or value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("status") != PLAN_STATUS
        or any(value.get(field) is not expected for field, expected in _AUTHORITY.items())
        or value.get("expected_task_count") != EXPECTED_TASK_COUNT
        or not isinstance(value.get("tasks"), list)
        or len(value["tasks"]) != EXPECTED_TASK_COUNT
    ):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache plan boundary disagrees"
        )
    plan_body = {key: item for key, item in value.items() if key != "plan_sha256"}
    if value.get("plan_sha256") != _sha256(plan_body):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache plan self-hash disagrees"
        )
    revision = _validate_source_revision(value["source_revision"])
    if value.get("source_revision_sha256") != revision["source_revision_sha256"]:
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache plan source revision disagrees"
        )
    migration_identity = value.get("migration_identity")
    if (
        not isinstance(migration_identity, dict)
        or value.get("migration_identity_sha256") != _sha256(migration_identity)
        or migration_identity.get("source_inventory_status") != SOURCE_INVENTORY_STATUS
        or migration_identity.get("source_training_authorized") is not False
        or migration_identity.get("active8_admission_status") != ACTIVE8_ADMISSION_STATUS
    ):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache migration identity disagrees"
        )
    policy = value.get("policy")
    if not isinstance(policy, dict):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache plan policy is malformed"
        )
    policy_body = {key: item for key, item in policy.items() if key != "policy_sha256"}
    if (
        policy.get("algorithm") != "one_mechanical_cache_per_semantic_source_v1"
        or type(policy.get("target_rows_per_chunk")) is not int
        or policy["target_rows_per_chunk"] <= 0
        or policy.get("source_task_count") != EXPECTED_TASK_COUNT
        or policy.get("policy_sha256") != _sha256(policy_body)
    ):
        raise SemanticActive8ChunkCacheMapReduceError("semantic chunk-cache plan policy disagrees")
    _require_artifact_path(
        value.get("output_artifact_root"),
        field="plan.output_artifact_root",
    )
    tasks = value["tasks"]
    if any(not isinstance(task, dict) or set(task) != _TASK_FIELDS for task in tasks):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache plan task fields disagree"
        )
    expected_cells = tuple(
        (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
    )
    if tuple((task["data_lane"], task["partition_role"]) for task in tasks) != expected_cells:
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache plan task grid disagrees"
        )
    for task_index, task in enumerate(tasks):
        if task.get("task_index") != task_index:
            raise SemanticActive8ChunkCacheMapReduceError(
                "semantic chunk-cache plan task index disagrees"
            )
        task_body = {key: item for key, item in task.items() if key != "task_identity_sha256"}
        expected_task_identity = _sha256(
            {
                "source_revision_sha256": revision["source_revision_sha256"],
                "policy_sha256": policy["policy_sha256"],
                "source": task_body,
            }
        )
        if task.get("task_identity_sha256") != expected_task_identity:
            raise SemanticActive8ChunkCacheMapReduceError(
                "semantic chunk-cache plan task identity disagrees"
            )
        _require_artifact_path(
            task.get("semantic_artifact_path"),
            field=f"tasks[{task_index}].semantic_artifact_path",
        )
        for field in (
            "migration_task_identity_sha256",
            "semantic_shard_sha256",
            "semantic_manifest_file_sha256",
            "semantic_manifest_sha256",
            "semantic_completion_file_sha256",
            "semantic_completion_sha256",
            "semantic_record_stream_sha256",
            "process_identity_sha256",
        ):
            _require_sha256(task.get(field), field=f"tasks[{task_index}].{field}")
    if value.get("task_inventory_sha256") != _sha256(tasks):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache task inventory disagrees"
        )
    run_body = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "status",
            *_AUTHORITY,
            "run_identity_sha256",
            "expected_task_count",
            "plan_sha256",
        }
    }
    if value.get("run_identity_sha256") != _sha256(run_body):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache global run identity disagrees"
        )
    return value


def _run_root(plan: Mapping[str, Any], *, artifact_root: Path) -> Path:
    output = _mounted_path(
        plan["output_artifact_root"],
        artifact_root=artifact_root,
        field="plan.output_artifact_root",
    )
    return output / "global_runs" / plan["run_identity_sha256"]


def write_semantic_active8_chunk_cache_plan(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> Path:
    value = _validate_plan(plan)
    path = _run_root(value, artifact_root=artifact_root) / PLAN_FILENAME
    _publish_immutable(path, _canonical_bytes(value, newline=True))
    return path


def _load_receipt(
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any] | None:
    path = (
        _run_root(plan, artifact_root=artifact_root)
        / "source_receipts"
        / f"{task['task_identity_sha256']}.json"
    )
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticActive8ChunkCacheMapReduceError(
            f"semantic chunk-cache source receipt is unreadable: {path}"
        ) from error
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _RECEIPT_FIELDS
        or _canonical_bytes(receipt, newline=True) != raw
        or receipt.get("schema") != SOURCE_RECEIPT_SCHEMA
        or receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != SOURCE_RECEIPT_STATUS
        or any(receipt.get(field) is not expected for field, expected in _AUTHORITY.items())
    ):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache source receipt boundary disagrees"
        )
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(body):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache source receipt self-hash disagrees"
        )
    if (
        receipt["global_run_identity_sha256"] != plan["run_identity_sha256"]
        or receipt["task_identity_sha256"] != task["task_identity_sha256"]
        or receipt["source_revision_sha256"] != plan["source_revision_sha256"]
        or receipt["source_task"] != task
    ):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache source receipt identity disagrees"
        )
    return receipt


def _validate_receipt_cache(
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    cache_root = (
        _mounted_path(
            plan["output_artifact_root"],
            artifact_root=artifact_root,
            field="plan.output_artifact_root",
        )
        / "cache"
    )
    cache = load_semantic_active8_chunk_cache(
        cache_root,
        run_identity_sha256=receipt["cache_run_identity_sha256"],
        expected_source_shard_sha256=task["semantic_shard_sha256"],
        expected_source_manifest_sha256=task["semantic_manifest_file_sha256"],
    )
    source_identity = cache["source_identity"]
    expected = {
        "cache_completion_sha256": cache["completion_sha256"],
        "cache_chunk_count": cache["chunk_count"],
        "cache_chunk_inventory_sha256": cache["chunk_inventory_sha256"],
        "cache_source_identity_sha256": source_identity["source_identity_sha256"],
    }
    if any(receipt[field] != value for field, value in expected.items()):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache source receipt and cache disagree"
        )
    if (
        source_identity["semantic_manifest_sha256"] != task["semantic_manifest_sha256"]
        or source_identity["record_stream_sha256"] != task["semantic_record_stream_sha256"]
        or source_identity["process_identity_sha256"] != task["process_identity_sha256"]
        or source_identity["entries"] != task["counts"]["semantic_entries"]
        or source_identity["states"] != task["counts"]["semantic_states"]
        or source_identity["actions"] != task["counts"]["semantic_actions"]
        or source_identity["family_histogram"] != task["family_histogram"]
    ):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk cache does not reconstruct its source inventory cell"
        )
    return cache


def execute_semantic_active8_chunk_cache_task(
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Build or reuse one source cache and publish its immutable receipt."""

    value = _validate_plan(plan)
    matches = [
        task for task in value["tasks"] if task["task_identity_sha256"] == task_identity_sha256
    ]
    if len(matches) != 1:
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache task identity is absent or duplicated"
        )
    task = matches[0]
    existing = _load_receipt(value, task, artifact_root=artifact_root)
    if existing is not None:
        _validate_receipt_cache(
            value,
            task,
            existing,
            artifact_root=artifact_root,
        )
        return existing
    semantic_artifact = _mounted_path(
        task["semantic_artifact_path"],
        artifact_root=artifact_root,
        field="task.semantic_artifact_path",
    )
    cache_root = (
        _mounted_path(
            value["output_artifact_root"],
            artifact_root=artifact_root,
            field="plan.output_artifact_root",
        )
        / "cache"
    )
    cache = build_semantic_active8_chunk_cache(
        semantic_artifact,
        expected_shard_sha256=task["semantic_shard_sha256"],
        expected_manifest_sha256=task["semantic_manifest_file_sha256"],
        expected_source_binding=task["source_binding"],
        output_root=cache_root,
        target_rows_per_chunk=value["policy"]["target_rows_per_chunk"],
    )
    receipt_body: dict[str, object] = {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "status": SOURCE_RECEIPT_STATUS,
        **_AUTHORITY,
        "global_run_identity_sha256": value["run_identity_sha256"],
        "task_identity_sha256": task["task_identity_sha256"],
        "source_revision_sha256": value["source_revision_sha256"],
        "source_task": task,
        "cache_run_identity_sha256": cache["run_identity_sha256"],
        "cache_completion_sha256": cache["completion_sha256"],
        "cache_chunk_count": cache["chunk_count"],
        "cache_chunk_inventory_sha256": cache["chunk_inventory_sha256"],
        "cache_source_identity_sha256": cache["source_identity"]["source_identity_sha256"],
    }
    receipt = {**receipt_body, "receipt_sha256": _sha256(receipt_body)}
    receipt_path = (
        _run_root(value, artifact_root=artifact_root)
        / "source_receipts"
        / f"{task['task_identity_sha256']}.json"
    )
    _publish_immutable(receipt_path, _canonical_bytes(receipt, newline=True))
    return receipt


def completed_semantic_active8_chunk_cache_task_ids(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> frozenset[str]:
    """Return verified completed task IDs, rejecting unexpected receipts."""

    value = _validate_plan(plan)
    receipt_root = _run_root(value, artifact_root=artifact_root) / "source_receipts"
    expected = {f"{task['task_identity_sha256']}.json" for task in value["tasks"]}
    observed = {path.name for path in receipt_root.iterdir()} if receipt_root.is_dir() else set()
    extras = observed - expected
    if extras:
        raise SemanticActive8ChunkCacheMapReduceError(
            f"semantic chunk-cache receipt directory has unexpected objects: {sorted(extras)}"
        )
    completed: set[str] = set()
    for task in value["tasks"]:
        receipt = _load_receipt(value, task, artifact_root=artifact_root)
        if receipt is None:
            continue
        _validate_receipt_cache(
            value,
            task,
            receipt,
            artifact_root=artifact_root,
        )
        completed.add(task["task_identity_sha256"])
    return frozenset(completed)


def _reduce_semantic_active8_chunk_caches_with_witness(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> SemanticActive8ChunkCacheReductionWitness:
    """Reload exactly 20 caches and publish one content-addressed completion."""

    value = _validate_plan(plan)
    plan_path = _run_root(value, artifact_root=artifact_root) / PLAN_FILENAME
    if not plan_path.is_file() or plan_path.read_bytes() != _canonical_bytes(value, newline=True):
        raise SemanticActive8ChunkCacheMapReduceError(
            "semantic chunk-cache reduction requires the exact published plan"
        )
    receipt_root = _run_root(value, artifact_root=artifact_root) / "source_receipts"
    expected_receipts = {f"{task['task_identity_sha256']}.json" for task in value["tasks"]}
    observed_receipts = (
        {path.name for path in receipt_root.iterdir()} if receipt_root.is_dir() else set()
    )
    extras = observed_receipts - expected_receipts
    if extras:
        raise SemanticActive8ChunkCacheMapReduceError(
            f"semantic chunk-cache receipt directory has unexpected objects: {sorted(extras)}"
        )
    missing = expected_receipts - observed_receipts
    if missing:
        raise SemanticActive8ChunkCacheIncomplete(
            "semantic chunk cache has "
            f"{EXPECTED_TASK_COUNT - len(missing)}/{EXPECTED_TASK_COUNT} complete sources"
        )
    cache_inventory: list[dict[str, object]] = []
    validated_caches: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    for task in value["tasks"]:
        receipt = _load_receipt(value, task, artifact_root=artifact_root)
        assert receipt is not None
        cache = _validate_receipt_cache(
            value,
            task,
            receipt,
            artifact_root=artifact_root,
        )
        validated_caches.append(cache)
        source_identity = cache["source_identity"]
        totals["semantic_entries"] += source_identity["entries"]
        totals["semantic_states"] += source_identity["states"]
        totals["semantic_actions"] += source_identity["actions"]
        family_totals.update(source_identity["family_histogram"])
        cache_inventory.append(
            {
                "task_identity_sha256": task["task_identity_sha256"],
                "data_lane": task["data_lane"],
                "partition_role": task["partition_role"],
                "semantic_shard_sha256": task["semantic_shard_sha256"],
                "semantic_manifest_file_sha256": task["semantic_manifest_file_sha256"],
                "cache_run_identity_sha256": cache["run_identity_sha256"],
                "cache_completion_sha256": cache["completion_sha256"],
                "cache_source_identity_sha256": source_identity["source_identity_sha256"],
                "chunk_count": cache["chunk_count"],
                "chunk_inventory_sha256": cache["chunk_inventory_sha256"],
                "source_receipt_sha256": receipt["receipt_sha256"],
            }
        )
    migration = value["migration_identity"]
    expected_counts = migration["counts"]
    if (
        totals["semantic_entries"] != expected_counts["semantic_entries"]
        or totals["semantic_states"] != expected_counts["semantic_states"]
        or totals["semantic_actions"] != expected_counts["semantic_actions"]
        or dict(sorted(family_totals.items())) != migration["family_histogram"]
    ):
        raise SemanticActive8ChunkCacheMapReduceError(
            "global semantic chunk-cache census disagrees with migration"
        )
    completion_body: dict[str, object] = {
        "schema": GLOBAL_COMPLETION_SCHEMA,
        "schema_version": GLOBAL_COMPLETION_SCHEMA_VERSION,
        "status": GLOBAL_COMPLETION_STATUS,
        **_AUTHORITY,
        "run_identity_sha256": value["run_identity_sha256"],
        "plan_sha256": value["plan_sha256"],
        "source_revision_sha256": value["source_revision_sha256"],
        "migration_identity": migration,
        "migration_identity_sha256": value["migration_identity_sha256"],
        "policy": value["policy"],
        "chunk_builder_identity": value["source_revision"]["chunk_builder_identity"],
        "source_task_count": EXPECTED_TASK_COUNT,
        "cache_inventory": cache_inventory,
        "cache_inventory_sha256": _sha256(cache_inventory),
        "counts": dict(totals),
        "family_histogram": dict(sorted(family_totals.items())),
    }
    completion = {
        **completion_body,
        "completion_sha256": _sha256(completion_body),
    }
    content = _canonical_bytes(completion, newline=True)
    file_sha256 = hashlib.sha256(content).hexdigest()
    output_root = _mounted_path(
        value["output_artifact_root"],
        artifact_root=artifact_root,
        field="plan.output_artifact_root",
    )
    relative = Path("objects") / "global_completions" / file_sha256[:2] / f"{file_sha256}.json"
    _publish_immutable(output_root / relative, content)
    pointer = {
        "schema": "compose.data.semantic_active8_chunk_cache_global_pointer",
        "schema_version": 1,
        "global_completion_object_path": relative.as_posix(),
        "global_completion_file_sha256": file_sha256,
        "global_completion_sha256": completion["completion_sha256"],
    }
    pointer["pointer_sha256"] = _sha256(pointer)
    _publish_immutable(
        _run_root(value, artifact_root=artifact_root) / GLOBAL_COMPLETION_FILENAME,
        _canonical_bytes(pointer, newline=True),
    )
    reduction = {
        **pointer,
        "completion": completion,
    }
    return SemanticActive8ChunkCacheReductionWitness(
        reduction=reduction,
        validated_caches=tuple(validated_caches),
    )


def reduce_semantic_active8_chunk_caches(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Reload exactly 20 caches and publish the unchanged public reduction."""

    return _reduce_semantic_active8_chunk_caches_with_witness(
        plan,
        artifact_root=artifact_root,
    ).reduction


def reduce_semantic_active8_chunk_caches_with_witness(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> SemanticActive8ChunkCacheReductionWitness:
    """Strictly reduce once while retaining verified cache metadata in memory."""

    return _reduce_semantic_active8_chunk_caches_with_witness(
        plan,
        artifact_root=artifact_root,
    )


__all__ = [
    "GLOBAL_COMPLETION_FILENAME",
    "GLOBAL_COMPLETION_SCHEMA",
    "GLOBAL_COMPLETION_SCHEMA_VERSION",
    "GLOBAL_COMPLETION_STATUS",
    "PLAN_FILENAME",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "PLAN_STATUS",
    "SemanticActive8ChunkCacheIncomplete",
    "SemanticActive8ChunkCacheMapReduceError",
    "SemanticActive8ChunkCacheReductionWitness",
    "completed_semantic_active8_chunk_cache_task_ids",
    "execute_semantic_active8_chunk_cache_task",
    "plan_semantic_active8_chunk_cache",
    "reduce_semantic_active8_chunk_caches",
    "reduce_semantic_active8_chunk_caches_with_witness",
    "write_semantic_active8_chunk_cache_plan",
]
