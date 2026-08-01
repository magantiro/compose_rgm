"""Deterministic map/reduce for semantic Active8 whole-trace decisions.

The planner expands the verified twenty-source chunk cache into one task per
content-addressed chunk.  Workers read exactly one chunk and apply the frozen
semantic Active8 admission primitive.  The reducer proves exact global-address
coverage and reconciles all decision counts.  These artifacts are evidence
only; they grant no Gate 0, T1, P50, training, checkpoint-selection, or final
test authority.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_v2_semantic_active8_admission import (
    ProductionSemanticExactCandidateChecker,
    SemanticActive8AdmissionPolicy,
    SemanticActive8TraceDecision,
    build_semantic_active8_admission_policy,
    evaluate_semantic_active8_trace,
    validate_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    ACTIVE8_ADMISSION_STATUS,
    SOURCE_INVENTORY_STATUS,
    EditingV2SemanticActive8SourceInventory,
)
from compose_v4.data.semantic_active8_chunk_cache import (
    load_semantic_active8_chunk_cache,
    read_semantic_active8_chunk,
)
from compose_v4.data.semantic_active8_chunk_cache_mapreduce import (
    reduce_semantic_active8_chunk_caches,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel

PLAN_SCHEMA = "compose.data.editing_v2_semantic_active8_decision_plan"
PLAN_SCHEMA_VERSION = 1
PLAN_STATUS = "FROZEN_DECISION_TASKS_NO_DOWNSTREAM_AUTHORITY"
RECEIPT_SCHEMA = "compose.data.editing_v2_semantic_active8_decision_receipt"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_STATUS = "COMPLETE_ACTIVE8_DECISIONS_NO_DOWNSTREAM_AUTHORITY"
COMPLETION_SCHEMA = "compose.data.editing_v2_semantic_active8_decision_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_STATUS = "COMPLETE_ACTIVE8_CENSUS_NO_DOWNSTREAM_AUTHORITY"
PLAN_FILENAME = "PLAN.json"
RECEIPT_FILENAME = "RECEIPT.json"
DECISION_FILENAME = "decisions.jsonl.gz"
COMPLETION_FILENAME = "COMPLETE.json"

_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_COUNT_KEYS = (
    "traces",
    "accepted_traces",
    "excluded_traces",
    "actions",
    "progress_rows",
    "raw_candidate_marks",
    "canonical_candidate_successors",
    "matching_candidate_marks",
    "successor_aliases",
)
_RECEIPT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *_AUTHORITY,
    "run_identity_sha256",
    "plan_sha256",
    "task",
    "policy_sha256",
    "model_runtime_identity_sha256",
    "decision_file_sha256",
    "decision_stream_sha256",
    "counts",
    "action_family_histogram",
    "active8_exclusion_reason_histogram",
    "receipt_sha256",
}
_TASK_FIELDS = {
    "task_identity_sha256",
    "task_index",
    "data_lane",
    "partition_role",
    "source_task_identity_sha256",
    "source_shard_sha256",
    "source_manifest_file_sha256",
    "source_manifest_sha256",
    "source_entry_count",
    "source_action_count",
    "cache_root",
    "cache_run_identity_sha256",
    "cache_completion_sha256",
    "cache_source_identity_sha256",
    "cache_chunk_inventory_sha256",
    "chunk_index",
    "entry_start",
    "entry_stop",
    "row_count",
    "chunk_object_path",
    "chunk_physical_sha256",
    "chunk_receipt_object_path",
    "chunk_receipt_file_sha256",
    "chunk_receipt_sha256",
    "chunk_cache_source_task_identity_sha256",
}
_PLAN_FIELDS = {
    "schema",
    "schema_version",
    "source_identity",
    "chunk_cache_global_run_identity_sha256",
    "chunk_cache_global_completion_sha256",
    "chunk_cache_inventory_sha256",
    "policy",
    "model_runtime_identity",
    "output_artifact_root",
    "tasks",
    "task_inventory_sha256",
    "status",
    *_AUTHORITY,
    "run_identity_sha256",
    "task_count",
    "plan_sha256",
}


class SemanticActive8DecisionMapReduceError(RuntimeError):
    """Decision plan, worker result, or reduction is incomplete or inconsistent."""


class SemanticActive8DecisionIncomplete(SemanticActive8DecisionMapReduceError):
    """At least one planned decision task is absent."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SemanticActive8DecisionMapReduceError(
            f"{field} must be a full lowercase SHA-256"
        )
    return value


def _artifact_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise SemanticActive8DecisionMapReduceError(
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
        raise SemanticActive8DecisionMapReduceError(
            f"{field} must be a normalized /artifacts path"
        )
    return value


def _relative_object_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise SemanticActive8DecisionMapReduceError(
            f"{field} must be a normalized relative object path"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise SemanticActive8DecisionMapReduceError(
            f"{field} must be a normalized relative object path"
        )
    return value


def _mounted(value: object, *, artifact_root: Path, field: str) -> Path:
    address = PurePosixPath(_artifact_path(value, field=field))
    root = Path(artifact_root).resolve()
    result = (root / Path(*address.parts[2:])).resolve()
    if not result.is_relative_to(root):
        raise SemanticActive8DecisionMapReduceError(f"{field} escapes artifact_root")
    return result


def _publish(path: Path, content: bytes) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(content).hexdigest()
    if target.exists():
        if (
            not target.is_file()
            or _file_sha(target) != expected
            or target.read_bytes() != content
        ):
            raise SemanticActive8DecisionMapReduceError(
                f"immutable Active8 decision collision at {target}"
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
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return False


def _model_runtime_identity(value: Mapping[str, object]) -> dict[str, Any]:
    identity = dict(value)
    supplied = _require_sha(
        identity.get("identity_sha256"), field="model_runtime.identity_sha256"
    )
    body = {key: item for key, item in identity.items() if key != "identity_sha256"}
    if not body or supplied != _sha(body):
        raise SemanticActive8DecisionMapReduceError(
            "model runtime identity is empty or its self-hash disagrees"
        )
    return identity


def _inventory_identity(
    inventory: EditingV2SemanticActive8SourceInventory,
) -> dict[str, object]:
    if (
        inventory.status != SOURCE_INVENTORY_STATUS
        or inventory.training_authorized is not False
        or inventory.active8_admission_status != ACTIVE8_ADMISSION_STATUS
    ):
        raise SemanticActive8DecisionMapReduceError(
            "source inventory crosses its pre-Active8 boundary"
        )
    cells = tuple(
        (source.data_lane, source.partition_role) for source in inventory.sources
    )
    expected = tuple(
        (lane, role)
        for lane in REQUIRED_DATA_LANES
        for role in REQUIRED_PARTITION_ROLES
    )
    if cells != expected:
        raise SemanticActive8DecisionMapReduceError(
            "source inventory is not the exact ordered twenty-cell grid"
        )
    counts = inventory.counts
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
        "counts": {
            "candidate_traces": counts.candidate_traces,
            "migration_source_traces": counts.migration_source_traces,
            "migration_admitted_traces": counts.migration_admitted_traces,
            "migration_rejected_traces": counts.migration_rejected_traces,
            "semantic_entries": counts.semantic_entries,
            "semantic_states": counts.semantic_states,
            "semantic_actions": counts.semantic_actions,
        },
    }


def _run_root(plan: Mapping[str, Any], *, artifact_root: Path) -> Path:
    return (
        _mounted(
            plan["output_artifact_root"],
            artifact_root=artifact_root,
            field="output_artifact_root",
        )
        / "runs"
        / str(plan["run_identity_sha256"])
    )


def plan_semantic_active8_decisions(
    inventory: EditingV2SemanticActive8SourceInventory,
    *,
    chunk_cache_plan: Mapping[str, Any],
    model_runtime_identity: Mapping[str, object],
    output_artifact_root: str,
    artifact_root: Path,
    policy: SemanticActive8AdmissionPolicy | None = None,
) -> dict[str, Any]:
    """Verify all cached sources and freeze one deterministic task per chunk."""

    selected_policy = validate_semantic_active8_admission_policy(
        policy or build_semantic_active8_admission_policy()
    )
    source_identity = _inventory_identity(inventory)
    runtime = _model_runtime_identity(model_runtime_identity)
    output_artifact_root = _artifact_path(
        output_artifact_root, field="output_artifact_root"
    )
    cache_reduction = reduce_semantic_active8_chunk_caches(
        chunk_cache_plan, artifact_root=artifact_root
    )
    cache_completion = cache_reduction["completion"]
    if cache_completion["migration_identity"] != source_identity:
        raise SemanticActive8DecisionMapReduceError(
            "verified chunk-cache inventory differs from the exact source inventory"
        )
    source_by_cell = {
        (source.data_lane, source.partition_role): source
        for source in inventory.sources
    }
    cache_task_by_cell = {
        (task["data_lane"], task["partition_role"]): task
        for task in chunk_cache_plan["tasks"]
    }
    cache_root = str(PurePosixPath(chunk_cache_plan["output_artifact_root"]) / "cache")
    tasks: list[dict[str, object]] = []
    for cache_source in cache_completion["cache_inventory"]:
        cell = (cache_source["data_lane"], cache_source["partition_role"])
        source = source_by_cell[cell]
        cache_task = cache_task_by_cell[cell]
        cache = load_semantic_active8_chunk_cache(
            _mounted(cache_root, artifact_root=artifact_root, field="cache_root"),
            run_identity_sha256=cache_source["cache_run_identity_sha256"],
            expected_source_shard_sha256=source.semantic_shard_sha256,
            expected_source_manifest_sha256=source.semantic_manifest_file_sha256,
        )
        for chunk in cache["chunk_inventory"]:
            body: dict[str, object] = {
                "task_index": len(tasks),
                "data_lane": cell[0],
                "partition_role": cell[1],
                "source_task_identity_sha256": source.task_identity_sha256,
                "source_shard_sha256": source.semantic_shard_sha256,
                "source_manifest_file_sha256": source.semantic_manifest_file_sha256,
                "source_manifest_sha256": source.semantic_manifest_sha256,
                "source_entry_count": source.entry_count,
                "source_action_count": source.counts.semantic_actions,
                "cache_root": cache_root,
                "cache_run_identity_sha256": cache["run_identity_sha256"],
                "cache_completion_sha256": cache["completion_sha256"],
                "cache_source_identity_sha256": cache["source_identity"][
                    "source_identity_sha256"
                ],
                "cache_chunk_inventory_sha256": cache["chunk_inventory_sha256"],
                "chunk_index": chunk["chunk_index"],
                "entry_start": chunk["entry_start"],
                "entry_stop": chunk["entry_stop"],
                "row_count": chunk["row_count"],
                "chunk_object_path": chunk["chunk_object_path"],
                "chunk_physical_sha256": chunk["chunk_physical_sha256"],
                "chunk_receipt_object_path": chunk["receipt_object_path"],
                "chunk_receipt_file_sha256": chunk["receipt_file_sha256"],
                "chunk_receipt_sha256": chunk["receipt_sha256"],
                "chunk_cache_source_task_identity_sha256": cache_task[
                    "task_identity_sha256"
                ],
            }
            task_id = _sha(
                {
                    "source_identity": source_identity,
                    "cache_global_completion_sha256": cache_completion[
                        "completion_sha256"
                    ],
                    "policy_sha256": selected_policy.policy_sha256,
                    "model_runtime_identity_sha256": runtime["identity_sha256"],
                    "task": body,
                }
            )
            tasks.append({"task_identity_sha256": task_id, **body})
    run_body: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_identity": source_identity,
        "chunk_cache_global_run_identity_sha256": cache_completion[
            "run_identity_sha256"
        ],
        "chunk_cache_global_completion_sha256": cache_completion["completion_sha256"],
        "chunk_cache_inventory_sha256": cache_completion["cache_inventory_sha256"],
        "policy": selected_policy.as_payload(),
        "model_runtime_identity": runtime,
        "output_artifact_root": output_artifact_root,
        "tasks": tasks,
        "task_inventory_sha256": _sha(tasks),
    }
    run_identity = _sha(run_body)
    body = {
        **run_body,
        "status": PLAN_STATUS,
        **_AUTHORITY,
        "run_identity_sha256": run_identity,
        "task_count": len(tasks),
    }
    return {**body, "plan_sha256": _sha(body)}


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(plan)
    if (
        set(value) != _PLAN_FIELDS
        or value.get("schema") != PLAN_SCHEMA
        or value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("status") != PLAN_STATUS
        or any(value.get(key) is not expected for key, expected in _AUTHORITY.items())
        or value.get("task_count") != len(value.get("tasks", ()))
    ):
        raise SemanticActive8DecisionMapReduceError("decision plan boundary disagrees")
    body = {key: item for key, item in value.items() if key != "plan_sha256"}
    if value.get("plan_sha256") != _sha(body):
        raise SemanticActive8DecisionMapReduceError("decision plan self-hash disagrees")
    if value.get("policy") != build_semantic_active8_admission_policy().as_payload():
        raise SemanticActive8DecisionMapReduceError("decision plan policy is stale")
    _model_runtime_identity(value["model_runtime_identity"])
    if value.get("task_inventory_sha256") != _sha(value["tasks"]):
        raise SemanticActive8DecisionMapReduceError("decision task inventory disagrees")
    expected_index = 0
    for task in value["tasks"]:
        if not isinstance(task, dict) or set(task) != _TASK_FIELDS:
            raise SemanticActive8DecisionMapReduceError("decision task fields disagree")
        if task.get("task_index") != expected_index:
            raise SemanticActive8DecisionMapReduceError(
                "decision task ordering disagrees"
            )
        expected_index += 1
        for field in (
            "task_identity_sha256",
            "source_task_identity_sha256",
            "source_shard_sha256",
            "source_manifest_file_sha256",
            "source_manifest_sha256",
            "cache_run_identity_sha256",
            "cache_completion_sha256",
            "cache_source_identity_sha256",
            "cache_chunk_inventory_sha256",
            "chunk_physical_sha256",
            "chunk_receipt_file_sha256",
            "chunk_receipt_sha256",
            "chunk_cache_source_task_identity_sha256",
        ):
            _require_sha(task.get(field), field=f"task.{field}")
        _artifact_path(task.get("cache_root"), field="task.cache_root")
        for field in ("chunk_object_path", "chunk_receipt_object_path"):
            _relative_object_path(task.get(field), field=f"task.{field}")
        for field in (
            "task_index",
            "source_entry_count",
            "source_action_count",
            "chunk_index",
            "entry_start",
            "entry_stop",
            "row_count",
        ):
            if type(task.get(field)) is not int or task[field] < 0:
                raise SemanticActive8DecisionMapReduceError(
                    f"task.{field} must be nonnegative"
                )
        if task["row_count"] != task["entry_stop"] - task["entry_start"]:
            raise SemanticActive8DecisionMapReduceError("decision task range disagrees")
        task_body = {
            key: item for key, item in task.items() if key != "task_identity_sha256"
        }
        expected_task_id = _sha(
            {
                "source_identity": value["source_identity"],
                "cache_global_completion_sha256": value[
                    "chunk_cache_global_completion_sha256"
                ],
                "policy_sha256": value["policy"]["policy_sha256"],
                "model_runtime_identity_sha256": value["model_runtime_identity"][
                    "identity_sha256"
                ],
                "task": task_body,
            }
        )
        if task["task_identity_sha256"] != expected_task_id:
            raise SemanticActive8DecisionMapReduceError(
                "decision task self-identity disagrees"
            )
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for task in value["tasks"]:
        grouped.setdefault((task["data_lane"], task["partition_role"]), []).append(
            (task["entry_start"], task["entry_stop"])
        )
    expected_cells = tuple(
        (lane, role)
        for lane in REQUIRED_DATA_LANES
        for role in REQUIRED_PARTITION_ROLES
    )
    if tuple(grouped) != expected_cells:
        raise SemanticActive8DecisionMapReduceError(
            "decision tasks omit or add a source cell"
        )
    for cell, ranges in grouped.items():
        cursor = 0
        cell_tasks = [
            task
            for task in value["tasks"]
            if (task["data_lane"], task["partition_role"]) == cell
        ]
        for start, stop in ranges:
            if start != cursor or stop < start:
                raise SemanticActive8DecisionMapReduceError(
                    "decision task ranges are not gap-free"
                )
            cursor = stop
        if cursor != cell_tasks[0]["source_entry_count"]:
            raise SemanticActive8DecisionMapReduceError(
                "decision tasks do not cover a source"
            )
    run_body = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "status",
            *_AUTHORITY,
            "run_identity_sha256",
            "task_count",
            "plan_sha256",
        }
    }
    if value.get("run_identity_sha256") != _sha(run_body):
        raise SemanticActive8DecisionMapReduceError(
            "decision plan run identity disagrees"
        )
    return value


def write_semantic_active8_decision_plan(
    plan: Mapping[str, Any], *, artifact_root: Path
) -> Path:
    value = _validate_plan(plan)
    path = _run_root(value, artifact_root=artifact_root) / PLAN_FILENAME
    _publish(path, _canonical_bytes(value, newline=True))
    return path


def _decision_row(
    addressed: Any, decision: SemanticActive8TraceDecision
) -> dict[str, object]:
    address = addressed.address
    actions: list[dict[str, object]] = []
    counts = Counter()
    families: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for item in decision.action_decisions:
        classification = item.classification
        evidence = item.candidate_evidence
        if classification.model_family is not None:
            families[classification.model_family] += 1
        if evidence is not None:
            counts["raw_candidate_marks"] += evidence.raw_mark_count
            counts[
                "canonical_candidate_successors"
            ] += evidence.canonical_successor_count
            counts["matching_candidate_marks"] += evidence.matching_mark_count
            counts["successor_aliases"] += evidence.successor_alias_count
        actions.append(
            {
                "step_index": classification.step_index,
                "executor_rule": classification.executor_rule,
                "model_family": classification.model_family,
                "action_sha256": classification.action_sha256,
                "policy_eligible": classification.policy_eligible,
                "exclusion_reason": classification.exclusion_reason,
                "candidate_evidence": (
                    None
                    if evidence is None
                    else {
                        "supported": evidence.supported,
                        "action_sha256": evidence.action_sha256,
                        "source_state_sha256": evidence.source_state_sha256,
                        "target_state_sha256": evidence.target_state_sha256,
                        "canonical_successor_key": evidence.canonical_successor_key,
                        "raw_mark_count": evidence.raw_mark_count,
                        "canonical_successor_count": evidence.canonical_successor_count,
                        "matching_mark_count": evidence.matching_mark_count,
                        "successor_alias_count": evidence.successor_alias_count,
                        "exclusion_reason": evidence.exclusion_reason,
                    }
                ),
            }
        )
    exclusions = [
        {
            "stage": item.stage,
            "step_index": item.step_index,
            "executor_rule": item.executor_rule,
            "model_family": item.model_family,
            "reason": item.reason,
        }
        for item in decision.active8_exclusions
    ]
    reasons.update(item["reason"] for item in exclusions)
    progress = []
    if decision.emits_progress_rows:
        progress = [
            {
                "progress_index": index,
                "terminal": index == address.path_length,
                "state_sha256": persistent_slot_state_sha256(
                    addressed.path.state_at(index)
                ),
            }
            for index in range(address.path_length + 1)
        ]
    row_body: dict[str, object] = {
        "schema": "compose.data.editing_v2_semantic_active8_trace_decision",
        "schema_version": 1,
        "address": {
            "packed_shard_content_sha256": address.packed_shard_content_sha256,
            "packed_shard_name": address.packed_shard_name,
            "entry_index": address.entry_index,
            "trace_id": address.trace_id,
            "layer": address.layer,
            "partition": address.partition,
            "source_key": address.source_key,
            "target_key": address.target_key,
            "path_length": address.path_length,
        },
        "policy_sha256": decision.policy_sha256,
        "semantic_migration_status": decision.semantic_migration_status,
        "semantic_migration_rejection": None,
        "active8_status": decision.active8_status,
        "emits_progress_rows": decision.emits_progress_rows,
        "actions": actions,
        "active8_exclusions": exclusions,
        "progress_rows": progress,
        "candidate_totals": {key: counts[key] for key in _COUNT_KEYS[5:]},
        "action_family_histogram": dict(sorted(families.items())),
        "active8_exclusion_reason_histogram": dict(sorted(reasons.items())),
    }
    return {**row_body, "decision_sha256": _sha(row_body)}


def _task_root(
    plan: Mapping[str, Any], task: Mapping[str, Any], *, artifact_root: Path
) -> Path:
    return (
        _run_root(plan, artifact_root=artifact_root)
        / "tasks"
        / task["task_identity_sha256"]
    )


def execute_semantic_active8_decision_task(
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    *,
    artifact_root: Path,
    model: FactorizedTraceletRateModel,
    exact_candidate_checker: ProductionSemanticExactCandidateChecker,
    model_runtime_identity_resolver: Callable[
        [FactorizedTraceletRateModel], Mapping[str, object]
    ],
) -> dict[str, Any]:
    """Apply the production checker to exactly one independently reusable chunk."""

    value = _validate_plan(plan)
    matches = [
        task
        for task in value["tasks"]
        if task["task_identity_sha256"] == task_identity_sha256
    ]
    if len(matches) != 1:
        raise SemanticActive8DecisionMapReduceError(
            "decision task identity is absent or duplicated"
        )
    task = matches[0]
    if (
        not isinstance(exact_candidate_checker, ProductionSemanticExactCandidateChecker)
        or exact_candidate_checker.model is not model
        or exact_candidate_checker.policy.policy_sha256
        != value["policy"]["policy_sha256"]
    ):
        raise SemanticActive8DecisionMapReduceError(
            "worker requires the exact production model/checker pair"
        )
    observed_runtime = _model_runtime_identity(model_runtime_identity_resolver(model))
    if observed_runtime != value["model_runtime_identity"]:
        raise SemanticActive8DecisionMapReduceError(
            "live model runtime identity differs from the plan"
        )
    task_root = _task_root(value, task, artifact_root=artifact_root)
    if task_root.exists():
        receipt_path = task_root / RECEIPT_FILENAME
        decision_path = task_root / DECISION_FILENAME
        if receipt_path.exists() or decision_path.exists():
            receipt, _ = _load_task_result(value, task, artifact_root=artifact_root)
            return receipt
    rows = tuple(
        read_semantic_active8_chunk(
            _mounted(
                task["cache_root"], artifact_root=artifact_root, field="task.cache_root"
            ),
            receipt_object_path=task["chunk_receipt_object_path"],
            expected_receipt_file_sha256=task["chunk_receipt_file_sha256"],
            expected_source_shard_sha256=task["source_shard_sha256"],
            expected_source_manifest_sha256=task["source_manifest_file_sha256"],
        )
    )
    if tuple(row.address.entry_index for row in rows) != tuple(
        range(task["entry_start"], task["entry_stop"])
    ):
        raise SemanticActive8DecisionMapReduceError(
            "chunk reader changed original global addresses"
        )
    decisions = [
        _decision_row(
            addressed,
            evaluate_semantic_active8_trace(
                addressed,
                exact_candidate_checker=exact_candidate_checker,
                policy=exact_candidate_checker.policy,
            ),
        )
        for addressed in rows
    ]
    raw = b"".join(_canonical_bytes(row, newline=True) for row in decisions)
    stream_sha = hashlib.sha256(raw).hexdigest()
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
        handle.write(raw)
    decision_bytes = output.getvalue()
    totals = Counter()
    families: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in decisions:
        totals["traces"] += 1
        totals[f"{row['active8_status']}_traces"] += 1
        totals["actions"] += len(row["actions"])
        totals["progress_rows"] += len(row["progress_rows"])
        totals.update(row["candidate_totals"])
        families.update(row["action_family_histogram"])
        reasons.update(row["active8_exclusion_reason_histogram"])
        if row["active8_status"] == "excluded" and row["progress_rows"]:
            raise SemanticActive8DecisionMapReduceError(
                "excluded trace emitted progress rows"
            )
    counts = {key: totals[key] for key in _COUNT_KEYS}
    _publish(task_root / DECISION_FILENAME, decision_bytes)
    receipt_body: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        **_AUTHORITY,
        "run_identity_sha256": value["run_identity_sha256"],
        "plan_sha256": value["plan_sha256"],
        "task": task,
        "policy_sha256": value["policy"]["policy_sha256"],
        "model_runtime_identity_sha256": value["model_runtime_identity"][
            "identity_sha256"
        ],
        "decision_file_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "decision_stream_sha256": stream_sha,
        "counts": counts,
        "action_family_histogram": dict(sorted(families.items())),
        "active8_exclusion_reason_histogram": dict(sorted(reasons.items())),
    }
    receipt = {**receipt_body, "receipt_sha256": _sha(receipt_body)}
    _publish(task_root / RECEIPT_FILENAME, _canonical_bytes(receipt, newline=True))
    return receipt


def _load_task_result(
    plan: Mapping[str, Any], task: Mapping[str, Any], *, artifact_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = _task_root(plan, task, artifact_root=artifact_root)
    receipt_path = root / RECEIPT_FILENAME
    decision_path = root / DECISION_FILENAME
    if not receipt_path.is_file() or not decision_path.is_file():
        raise SemanticActive8DecisionIncomplete(
            f"decision task {task['task_identity_sha256']} is absent"
        )
    if {path.name for path in root.iterdir()} != {
        RECEIPT_FILENAME,
        DECISION_FILENAME,
    }:
        raise SemanticActive8DecisionMapReduceError(
            "decision task directory contains unexpected objects"
        )
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _RECEIPT_FIELDS
        or _canonical_bytes(receipt, newline=True) != receipt_bytes
    ):
        raise SemanticActive8DecisionMapReduceError(
            "decision receipt is not canonical JSON"
        )
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != RECEIPT_STATUS
        or any(receipt.get(key) is not expected for key, expected in _AUTHORITY.items())
        or receipt.get("receipt_sha256") != _sha(body)
        or receipt.get("task") != task
        or receipt.get("run_identity_sha256") != plan["run_identity_sha256"]
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("policy_sha256") != plan["policy"]["policy_sha256"]
        or receipt.get("model_runtime_identity_sha256")
        != plan["model_runtime_identity"]["identity_sha256"]
        or set(receipt.get("counts", {})) != set(_COUNT_KEYS)
    ):
        raise SemanticActive8DecisionMapReduceError(
            "decision receipt identity disagrees"
        )
    content = decision_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != receipt["decision_file_sha256"]:
        raise SemanticActive8DecisionMapReduceError(
            "decision bytes disagree with receipt"
        )
    try:
        raw = gzip.decompress(content)
    except OSError as error:
        raise SemanticActive8DecisionMapReduceError(
            "decision file is not gzip"
        ) from error
    if hashlib.sha256(raw).hexdigest() != receipt["decision_stream_sha256"]:
        raise SemanticActive8DecisionMapReduceError(
            "decision stream disagrees with receipt"
        )
    rows = [json.loads(line) for line in raw.splitlines()]
    if b"".join(_canonical_bytes(row, newline=True) for row in rows) != raw:
        raise SemanticActive8DecisionMapReduceError("decision JSONL is not canonical")
    deterministic = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=deterministic, mtime=0
    ) as handle:
        handle.write(raw)
    if deterministic.getvalue() != content:
        raise SemanticActive8DecisionMapReduceError(
            "decision file is not deterministic gzip"
        )
    return receipt, rows


def reduce_semantic_active8_decisions(
    plan: Mapping[str, Any],
    *,
    inventory: EditingV2SemanticActive8SourceInventory,
    artifact_root: Path,
) -> dict[str, Any]:
    """Prove exact twenty-source coverage and publish a non-authorizing census."""

    value = _validate_plan(plan)
    if _inventory_identity(inventory) != value["source_identity"]:
        raise SemanticActive8DecisionMapReduceError(
            "reducer source inventory differs from plan"
        )
    plan_path = _run_root(value, artifact_root=artifact_root) / PLAN_FILENAME
    if not plan_path.is_file() or plan_path.read_bytes() != _canonical_bytes(
        value, newline=True
    ):
        raise SemanticActive8DecisionMapReduceError(
            "reduction requires the exact published plan"
        )
    task_parent = _run_root(value, artifact_root=artifact_root) / "tasks"
    expected_dirs = {task["task_identity_sha256"] for task in value["tasks"]}
    observed_dirs = (
        {path.name for path in task_parent.iterdir()} if task_parent.is_dir() else set()
    )
    if observed_dirs - expected_dirs:
        raise SemanticActive8DecisionMapReduceError(
            "unexpected decision task outputs are present"
        )
    totals = Counter()
    families: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    result_inventory: list[dict[str, object]] = []
    seen_addresses: set[tuple[str, int]] = set()
    for task in value["tasks"]:
        receipt, rows = _load_task_result(value, task, artifact_root=artifact_root)
        source_rows = tuple(
            read_semantic_active8_chunk(
                _mounted(
                    task["cache_root"],
                    artifact_root=artifact_root,
                    field="task.cache_root",
                ),
                receipt_object_path=task["chunk_receipt_object_path"],
                expected_receipt_file_sha256=task["chunk_receipt_file_sha256"],
                expected_source_shard_sha256=task["source_shard_sha256"],
                expected_source_manifest_sha256=task["source_manifest_file_sha256"],
            )
        )
        if len(source_rows) != len(rows):
            raise SemanticActive8DecisionMapReduceError(
                "decision rows do not cover the exact cached chunk"
            )
        observed = Counter()
        observed_families: Counter[str] = Counter()
        observed_reasons: Counter[str] = Counter()
        if tuple(row["address"]["entry_index"] for row in rows) != tuple(
            range(task["entry_start"], task["entry_stop"])
        ):
            raise SemanticActive8DecisionMapReduceError(
                "decision rows are not exact task-address coverage"
            )
        for row, source_row in zip(rows, source_rows, strict=True):
            body = {key: item for key, item in row.items() if key != "decision_sha256"}
            if row.get("decision_sha256") != _sha(body):
                raise SemanticActive8DecisionMapReduceError(
                    "decision row self-hash disagrees"
                )
            address = row["address"]
            original = source_row.address
            expected_address = {
                "packed_shard_content_sha256": original.packed_shard_content_sha256,
                "packed_shard_name": original.packed_shard_name,
                "entry_index": original.entry_index,
                "trace_id": original.trace_id,
                "layer": original.layer,
                "partition": original.partition,
                "source_key": original.source_key,
                "target_key": original.target_key,
                "path_length": original.path_length,
            }
            if address != expected_address:
                raise SemanticActive8DecisionMapReduceError(
                    "decision row changed its original global address"
                )
            key = (address["packed_shard_content_sha256"], address["entry_index"])
            if key in seen_addresses:
                raise SemanticActive8DecisionMapReduceError(
                    "duplicate original global address"
                )
            seen_addresses.add(key)
            if address["packed_shard_content_sha256"] != task["source_shard_sha256"]:
                raise SemanticActive8DecisionMapReduceError(
                    "decision row source address changed"
                )
            if (
                row.get("semantic_migration_status") != "admitted"
                or row.get("semantic_migration_rejection") is not None
                or row.get("active8_status") not in {"accepted", "excluded"}
                or len(row.get("actions", ())) != address["path_length"]
            ):
                raise SemanticActive8DecisionMapReduceError(
                    "decision row crosses the semantic/Active8 boundary"
                )
            accepted = row["active8_status"] == "accepted"
            expected_progress_count = address["path_length"] + 1 if accepted else 0
            if (
                row.get("emits_progress_rows") is not accepted
                or len(row.get("progress_rows", ())) != expected_progress_count
            ):
                raise SemanticActive8DecisionMapReduceError(
                    "whole-trace progress-row admission disagrees"
                )
            derived_candidates = Counter()
            derived_families: Counter[str] = Counter()
            for action in row["actions"]:
                family = action.get("model_family")
                if family is not None:
                    derived_families[family] += 1
                evidence = action.get("candidate_evidence")
                if evidence is not None:
                    derived_candidates["raw_candidate_marks"] += evidence[
                        "raw_mark_count"
                    ]
                    derived_candidates["canonical_candidate_successors"] += evidence[
                        "canonical_successor_count"
                    ]
                    derived_candidates["matching_candidate_marks"] += evidence[
                        "matching_mark_count"
                    ]
                    derived_candidates["successor_aliases"] += evidence[
                        "successor_alias_count"
                    ]
            derived_candidate_totals = {
                field: derived_candidates[field] for field in _COUNT_KEYS[5:]
            }
            derived_reasons = Counter(
                exclusion["reason"] for exclusion in row["active8_exclusions"]
            )
            if (
                row.get("candidate_totals") != derived_candidate_totals
                or row.get("action_family_histogram")
                != dict(sorted(derived_families.items()))
                or row.get("active8_exclusion_reason_histogram")
                != dict(sorted(derived_reasons.items()))
                or accepted != (not row["active8_exclusions"])
            ):
                raise SemanticActive8DecisionMapReduceError(
                    "decision row derived census disagrees"
                )
            observed["traces"] += 1
            observed[f"{row['active8_status']}_traces"] += 1
            observed["actions"] += len(row["actions"])
            observed["progress_rows"] += len(row["progress_rows"])
            observed.update(derived_candidate_totals)
            observed_families.update(derived_families)
            observed_reasons.update(derived_reasons)
        observed_counts = {key: observed[key] for key in _COUNT_KEYS}
        if (
            observed_counts != receipt["counts"]
            or dict(sorted(observed_families.items()))
            != receipt["action_family_histogram"]
            or dict(sorted(observed_reasons.items()))
            != receipt["active8_exclusion_reason_histogram"]
        ):
            raise SemanticActive8DecisionMapReduceError(
                "decision task receipt census disagrees"
            )
        totals.update(observed_counts)
        families.update(observed_families)
        reasons.update(observed_reasons)
        result_inventory.append(
            {
                "task_identity_sha256": task["task_identity_sha256"],
                "data_lane": task["data_lane"],
                "partition_role": task["partition_role"],
                "entry_start": task["entry_start"],
                "entry_stop": task["entry_stop"],
                "receipt_sha256": receipt["receipt_sha256"],
                "decision_file_sha256": receipt["decision_file_sha256"],
                "counts": observed_counts,
            }
        )
    source = value["source_identity"]
    if (
        totals["traces"] != source["counts"]["semantic_entries"]
        or totals["actions"] != source["counts"]["semantic_actions"]
        or totals["accepted_traces"] + totals["excluded_traces"] != totals["traces"]
    ):
        raise SemanticActive8DecisionMapReduceError(
            "global Active8 census disagrees with semantic sources"
        )
    completion_body: dict[str, object] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **_AUTHORITY,
        "run_identity_sha256": value["run_identity_sha256"],
        "plan_sha256": value["plan_sha256"],
        "source_identity": source,
        "chunk_cache_global_completion_sha256": value[
            "chunk_cache_global_completion_sha256"
        ],
        "policy_sha256": value["policy"]["policy_sha256"],
        "model_runtime_identity_sha256": value["model_runtime_identity"][
            "identity_sha256"
        ],
        "task_count": len(value["tasks"]),
        "result_inventory": result_inventory,
        "result_inventory_sha256": _sha(result_inventory),
        "active8_counts": {key: totals[key] for key in _COUNT_KEYS},
        "active8_action_family_histogram": dict(sorted(families.items())),
        "active8_exclusion_reason_histogram": dict(sorted(reasons.items())),
        "migration_rejection_census": {
            "rejected_traces": source["counts"]["migration_rejected_traces"],
            "rejections_by_code": source["rejection_histogram"],
            "rejected_traces_reevaluated_by_active8": 0,
        },
    }
    completion = {**completion_body, "completion_sha256": _sha(completion_body)}
    _publish(
        _run_root(value, artifact_root=artifact_root) / COMPLETION_FILENAME,
        _canonical_bytes(completion, newline=True),
    )
    return completion


__all__ = [
    "COMPLETION_FILENAME",
    "DECISION_FILENAME",
    "PLAN_FILENAME",
    "RECEIPT_FILENAME",
    "SemanticActive8DecisionIncomplete",
    "SemanticActive8DecisionMapReduceError",
    "execute_semantic_active8_decision_task",
    "plan_semantic_active8_decisions",
    "reduce_semantic_active8_decisions",
    "write_semantic_active8_decision_plan",
]
