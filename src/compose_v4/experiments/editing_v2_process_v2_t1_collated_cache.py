"""Authenticated CPU-collated panel cache for Process-V2 T1.

The expensive exact successor fibers remain owned by the prepared-input
artifact.  This module derives only the model-ready ``FactorizedMarkBatch``
rows from those frozen states, in independent restart-safe tasks.  The GPU
therefore performs model evaluation and optimization, not chemistry collation.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)
from compose_v4.data.immutable_artifact import ImmutableArtifactError, write_bytes_if_absent
from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
    _MaterializedSemanticT1Panel,
    _batch_for_ids,
    _collator,
)
from compose_v4.experiments.factorized_mark_conditional import (
    _concatenate_factorized_mark_batches,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
)

PLAN_FILENAME = "PROCESS_V2_T1_COLLATED_PLAN.json"
COMPLETION_FILENAME = "PROCESS_V2_T1_COLLATED_COMPLETE.json"
TASKS_DIRNAME = "collated_tasks"
PAYLOAD_FILENAME = "COLLATED_BATCH.pt"
RECEIPT_FILENAME = "COLLATED_BATCH_RECEIPT.json"

PLAN_SCHEMA = "compose.editing_v2.process_v2_t1_collated_plan"
PLAN_SCHEMA_VERSION = 1
PLAN_STATUS = "PLANNED_PROCESS_V2_T1_CPU_COLLATION_NO_DOWNSTREAM_AUTHORITY"
RECEIPT_SCHEMA = "compose.editing_v2.process_v2_t1_collated_receipt"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_STATUS = "COMPLETE_PROCESS_V2_T1_CPU_COLLATION_NO_DOWNSTREAM_AUTHORITY"
PAYLOAD_SCHEMA = "compose.editing_v2.process_v2_t1_collated_batch"
PAYLOAD_SCHEMA_VERSION = 1
COMPLETION_SCHEMA = "compose.editing_v2.process_v2_t1_collated_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_STATUS = "COMPLETE_PROCESS_V2_T1_COLLATED_CACHE_NO_DOWNSTREAM_AUTHORITY"

_IMPLEMENTATION_FILES = (
    "src/compose_v4/experiments/editing_v2_process_v2_t1_collated_cache.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_capacity_runner.py",
    "src/compose_v4/experiments/factorized_mark_conditional.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
)

_SCORE_REVISION_PREDECESSOR = {
    "plan_sha256": "1f13c83cc778bcbe5d34414547356c7a20e714d1a55798276d549b6a931f5682",
    "plan_file_sha256": (
        "399877de32e7016dc11e3aacc8a00b30d1a80fbe57e9bdae4c8f1640d785067d"
    ),
    "run_identity_sha256": (
        "61c0f4ccb4cc4d70d37d471be36cf0596b8687ef177cc24af685522332ea3cd9"
    ),
    "implementation_sha256": (
        "41df1271d9400ff442e65c073c0985b3373f0e6be01cefd99af6fc0d83dbc295"
    ),
    "prepared_completion_sha256": (
        "a93c0b65b08c6b7726deeae727213b7145b88407830f95563c5060b86c4fc865"
    ),
    "completion_sha256": (
        "77f01ca5ecf8a94ef8fb2d90d53eb180f6ddedd3725bbe678b2beab95c79f504"
    ),
    "completion_file_sha256": (
        "08c28b1d2ede162db69661fd0049386d5682f33abee5b8a370fe75d21d0ea25b"
    ),
}


class ProcessV2T1CollatedCacheError(RuntimeError):
    """The CPU-collated T1 cache is incomplete, stale, or inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256(repo_root: Path) -> str:
    root = Path(repo_root).resolve()
    rows: list[dict[str, str]] = []
    for relative in _IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise ProcessV2T1CollatedCacheError(
                f"T1 collated-cache implementation source is absent: {relative}"
            )
        rows.append({"path": relative, "file_sha256": _file_sha256(path)})
    return canonical_sha256(rows)


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessV2T1CollatedCacheError(f"{field} must be a lowercase SHA-256")
    return value


def _runtime_panel_ids(runtime: Any) -> tuple[str, ...]:
    identifiers = tuple(str(entry["panel_entry_sha256"]) for entry in runtime.entries)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ProcessV2T1CollatedCacheError(
            "T1 runtime panel identifiers are empty or repeated"
        )
    return identifiers


def _validate_source_revision(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1CollatedCacheError("collated-cache source revision is absent")
    revision = dict(value)
    required = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "image_revision_sha256",
        "serialized_source_inventory_sha256",
        "source_revision_sha256",
    }
    if set(revision) != required:
        raise ProcessV2T1CollatedCacheError("collated-cache source revision fields disagree")
    body = {key: item for key, item in revision.items() if key != "source_revision_sha256"}
    if revision["source_revision_sha256"] != canonical_sha256(body):
        raise ProcessV2T1CollatedCacheError("collated-cache source revision self-hash disagrees")
    _require_sha(revision["source_revision_sha256"], field="source revision")
    return revision


def build_collated_cache_plan(
    runtime: Any,
    *,
    prepared_completion_path: Path,
    prepared_completion_sha256: str,
    source_revision: Mapping[str, Any],
    software_versions: Mapping[str, str],
    repo_root: Path,
    entries_per_task: int = 16,
) -> dict[str, Any]:
    """Freeze deterministic independent collation tasks over the prepared panel."""

    if type(entries_per_task) is not int or entries_per_task <= 0:
        raise ProcessV2T1CollatedCacheError("entries_per_task must be a positive integer")
    revision = _validate_source_revision(source_revision)
    panel_ids = _runtime_panel_ids(runtime)
    versions = dict(software_versions)
    if set(versions) != {"python", "torch", "numpy", "rdkit"} or any(
        not isinstance(value, str) or not value for value in versions.values()
    ):
        raise ProcessV2T1CollatedCacheError("collated-cache software versions disagree")
    prepared_path = str(Path(prepared_completion_path))
    _require_sha(prepared_completion_sha256, field="prepared completion")
    implementation_sha256 = _implementation_sha256(repo_root)
    task_bodies: list[dict[str, Any]] = []
    for start in range(0, len(panel_ids), entries_per_task):
        selected = panel_ids[start : start + entries_per_task]
        body = {
            "task_index": len(task_bodies),
            "start": start,
            "stop": start + len(selected),
            "panel_entry_sha256s": list(selected),
            "panel_entry_inventory_sha256": canonical_sha256(list(selected)),
        }
        task_bodies.append({**body, "task_identity_sha256": canonical_sha256(body)})
    identity = {
        "prepared_completion_path": prepared_path,
        "prepared_completion_sha256": prepared_completion_sha256,
        "panel_entry_inventory_sha256": canonical_sha256(list(panel_ids)),
        "implementation_sha256": implementation_sha256,
        "source_revision_sha256": revision["source_revision_sha256"],
        "software_versions": versions,
        "entries_per_task": entries_per_task,
        "task_inventory_sha256": canonical_sha256(task_bodies),
    }
    body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PLAN_STATUS,
        **authority_false_block(),
        "source_revision": revision,
        "software_versions": versions,
        "implementation_sha256": implementation_sha256,
        "prepared_completion_path": prepared_path,
        "prepared_completion_sha256": prepared_completion_sha256,
        "panel_entry_count": len(panel_ids),
        "panel_entry_inventory_sha256": canonical_sha256(list(panel_ids)),
        "entries_per_task": entries_per_task,
        "task_count": len(task_bodies),
        "task_inventory_sha256": canonical_sha256(task_bodies),
        "tasks": task_bodies,
        "run_identity_sha256": canonical_sha256(identity),
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def validate_collated_cache_plan(
    value: object,
    *,
    runtime: Any,
    repo_root: Path,
    score_revision_rebind: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1CollatedCacheError("T1 collated-cache plan must be an object")
    plan = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "source_revision",
        "software_versions",
        "implementation_sha256",
        "prepared_completion_path",
        "prepared_completion_sha256",
        "panel_entry_count",
        "panel_entry_inventory_sha256",
        "entries_per_task",
        "task_count",
        "task_inventory_sha256",
        "tasks",
        "run_identity_sha256",
        "plan_sha256",
    }
    if set(plan) != expected_fields:
        raise ProcessV2T1CollatedCacheError("T1 collated-cache plan fields disagree")
    try:
        verify_self_hash(plan, field="plan_sha256", label="the T1 collated-cache plan")
        require_authority_false(plan, label="the T1 collated-cache plan")
    except ValueError as error:
        raise ProcessV2T1CollatedCacheError(str(error)) from error
    expected_implementation_sha256 = (
        _SCORE_REVISION_PREDECESSOR["implementation_sha256"]
        if score_revision_rebind
        else _implementation_sha256(repo_root)
    )
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["schema_version"] != PLAN_SCHEMA_VERSION
        or plan["status"] != PLAN_STATUS
        or plan["implementation_sha256"] != expected_implementation_sha256
        or type(plan["entries_per_task"]) is not int
        or plan["entries_per_task"] <= 0
    ):
        raise ProcessV2T1CollatedCacheError("T1 collated-cache plan identity disagrees")
    _validate_source_revision(plan["source_revision"])
    panel_ids = _runtime_panel_ids(runtime)
    tasks = plan["tasks"]
    if (
        not isinstance(tasks, list)
        or plan["panel_entry_count"] != len(panel_ids)
        or plan["panel_entry_inventory_sha256"] != canonical_sha256(list(panel_ids))
        or plan["task_count"] != len(tasks)
        or plan["task_inventory_sha256"] != canonical_sha256(tasks)
    ):
        raise ProcessV2T1CollatedCacheError("T1 collated-cache plan census disagrees")
    observed: list[str] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise ProcessV2T1CollatedCacheError("T1 collated-cache task is malformed")
        task = dict(task)
        body = {key: item for key, item in task.items() if key != "task_identity_sha256"}
        selected = task.get("panel_entry_sha256s")
        if (
            set(task)
            != {
                "task_index",
                "start",
                "stop",
                "panel_entry_sha256s",
                "panel_entry_inventory_sha256",
                "task_identity_sha256",
            }
            or task["task_index"] != index
            or not isinstance(selected, list)
            or not selected
            or task["start"] != len(observed)
            or task["stop"] != len(observed) + len(selected)
            or task["panel_entry_inventory_sha256"] != canonical_sha256(selected)
            or task["task_identity_sha256"] != canonical_sha256(body)
        ):
            raise ProcessV2T1CollatedCacheError("T1 collated-cache task identity disagrees")
        observed.extend(str(item) for item in selected)
    if tuple(observed) != panel_ids:
        raise ProcessV2T1CollatedCacheError("T1 collated-cache tasks changed panel order")
    identity = {
        "prepared_completion_path": plan["prepared_completion_path"],
        "prepared_completion_sha256": plan["prepared_completion_sha256"],
        "panel_entry_inventory_sha256": plan["panel_entry_inventory_sha256"],
        "implementation_sha256": plan["implementation_sha256"],
        "source_revision_sha256": plan["source_revision"]["source_revision_sha256"],
        "software_versions": plan["software_versions"],
        "entries_per_task": plan["entries_per_task"],
        "task_inventory_sha256": plan["task_inventory_sha256"],
    }
    if plan["run_identity_sha256"] != canonical_sha256(identity):
        raise ProcessV2T1CollatedCacheError("T1 collated-cache run identity disagrees")
    if score_revision_rebind and any(
        plan[field] != expected
        for field, expected in _SCORE_REVISION_PREDECESSOR.items()
        if field
        in {
            "plan_sha256",
            "run_identity_sha256",
            "implementation_sha256",
            "prepared_completion_sha256",
        }
    ):
        raise ProcessV2T1CollatedCacheError(
            "T1 collated-cache plan is not the exact allowlisted predecessor"
        )
    return plan


def write_collated_cache_plan(path: Path, plan: Mapping[str, Any]) -> None:
    try:
        write_bytes_if_absent(Path(path), _canonical_bytes(plan) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1CollatedCacheError(str(error)) from error


def load_collated_cache_plan(
    path: Path,
    *,
    runtime: Any,
    repo_root: Path,
    score_revision_rebind: bool = False,
) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2T1CollatedCacheError("T1 collated-cache plan is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise ProcessV2T1CollatedCacheError("T1 collated-cache plan is not canonical JSON")
    if score_revision_rebind and _file_sha256(path) != _SCORE_REVISION_PREDECESSOR[
        "plan_file_sha256"
    ]:
        raise ProcessV2T1CollatedCacheError(
            "T1 collated-cache plan file is not the exact allowlisted predecessor"
        )
    return validate_collated_cache_plan(
        value,
        runtime=runtime,
        repo_root=repo_root,
        score_revision_rebind=score_revision_rebind,
    )


def _task_for_identity(plan: Mapping[str, Any], task_identity_sha256: str) -> dict[str, Any]:
    matches = [
        dict(task)
        for task in plan["tasks"]
        if task["task_identity_sha256"] == task_identity_sha256
    ]
    if len(matches) != 1:
        raise ProcessV2T1CollatedCacheError("T1 collated task is absent or repeated")
    return matches[0]


def compile_collated_cache_leaf(
    runtime: Any,
    model: FactorizedTraceletRateModel,
    *,
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    run_root: Path,
) -> Path:
    task = _task_for_identity(plan, task_identity_sha256)
    panel_ids = tuple(str(item) for item in task["panel_entry_sha256s"])
    batch, _fibers, _partitions, entries = _batch_for_ids(
        runtime, model, _collator(model), panel_ids
    )
    if tuple(str(entry["panel_entry_sha256"]) for entry in entries) != panel_ids:
        raise ProcessV2T1CollatedCacheError("T1 collated leaf changed panel order")
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "task_identity_sha256": task_identity_sha256,
        "panel_entry_sha256s": list(panel_ids),
        "batch": batch,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    payload_bytes = buffer.getvalue()
    task_root = Path(run_root) / TASKS_DIRNAME / task_identity_sha256
    payload_path = task_root / PAYLOAD_FILENAME
    try:
        write_bytes_if_absent(payload_path, payload_bytes)
    except ImmutableArtifactError as error:
        raise ProcessV2T1CollatedCacheError(str(error)) from error
    body = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        **authority_false_block(),
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "implementation_sha256": plan["implementation_sha256"],
        "prepared_completion_sha256": plan["prepared_completion_sha256"],
        "task_identity_sha256": task_identity_sha256,
        "panel_entry_count": len(panel_ids),
        "panel_entry_inventory_sha256": canonical_sha256(list(panel_ids)),
        "payload_filename": PAYLOAD_FILENAME,
        "payload_file_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_file_bytes": len(payload_bytes),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    receipt_path = task_root / RECEIPT_FILENAME
    try:
        write_bytes_if_absent(receipt_path, _canonical_bytes(receipt) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1CollatedCacheError(str(error)) from error
    return receipt_path


def _load_receipt(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2T1CollatedCacheError("T1 collated receipt is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise ProcessV2T1CollatedCacheError("T1 collated receipt is not canonical JSON")
    return value


def validate_collated_cache_leaf(
    receipt_path: Path,
    *,
    plan: Mapping[str, Any],
    runtime: Any,
    load_payload: bool = True,
) -> tuple[dict[str, Any], FactorizedMarkBatch | None]:
    receipt = _load_receipt(receipt_path)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "plan_sha256",
        "run_identity_sha256",
        "implementation_sha256",
        "prepared_completion_sha256",
        "task_identity_sha256",
        "panel_entry_count",
        "panel_entry_inventory_sha256",
        "payload_filename",
        "payload_file_sha256",
        "payload_file_bytes",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise ProcessV2T1CollatedCacheError("T1 collated receipt fields disagree")
    try:
        verify_self_hash(receipt, field="receipt_sha256", label="a T1 collated receipt")
        require_authority_false(receipt, label="a T1 collated receipt")
    except ValueError as error:
        raise ProcessV2T1CollatedCacheError(str(error)) from error
    task = _task_for_identity(plan, str(receipt["task_identity_sha256"]))
    panel_ids = tuple(str(item) for item in task["panel_entry_sha256s"])
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
        or receipt["status"] != RECEIPT_STATUS
        or receipt["plan_sha256"] != plan["plan_sha256"]
        or receipt["run_identity_sha256"] != plan["run_identity_sha256"]
        or receipt["implementation_sha256"] != plan["implementation_sha256"]
        or receipt["prepared_completion_sha256"] != plan["prepared_completion_sha256"]
        or receipt["panel_entry_count"] != len(panel_ids)
        or receipt["panel_entry_inventory_sha256"] != canonical_sha256(list(panel_ids))
        or receipt["payload_filename"] != PAYLOAD_FILENAME
    ):
        raise ProcessV2T1CollatedCacheError("T1 collated receipt identity disagrees")
    payload_path = Path(receipt_path).parent / PAYLOAD_FILENAME
    if (
        not payload_path.is_file()
        or payload_path.stat().st_size != receipt["payload_file_bytes"]
        or _file_sha256(payload_path) != receipt["payload_file_sha256"]
    ):
        raise ProcessV2T1CollatedCacheError("T1 collated payload bytes disagree")
    if not load_payload:
        return receipt, None
    payload = torch.load(payload_path, map_location="cpu", weights_only=False, mmap=True)
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {
            "schema",
            "schema_version",
            "task_identity_sha256",
            "panel_entry_sha256s",
            "batch",
        }
        or payload["schema"] != PAYLOAD_SCHEMA
        or payload["schema_version"] != PAYLOAD_SCHEMA_VERSION
        or payload["task_identity_sha256"] != task["task_identity_sha256"]
        or tuple(payload["panel_entry_sha256s"]) != panel_ids
        or not isinstance(payload["batch"], FactorizedMarkBatch)
        or payload["batch"].batch_size != len(panel_ids)
    ):
        raise ProcessV2T1CollatedCacheError("T1 collated payload identity disagrees")
    batch = payload["batch"]
    expected_states = runtime.prepared.states_by_panel_entry_sha256
    if tuple(persistent_slot_state_sha256(state) for state in batch.states) != tuple(
        persistent_slot_state_sha256(expected_states[panel_id]) for panel_id in panel_ids
    ):
        raise ProcessV2T1CollatedCacheError("T1 collated payload states disagree")
    entries_by_id = {
        str(entry["panel_entry_sha256"]): entry for entry in runtime.entries
    }
    expected_families = tuple(
        str(entries_by_id[panel_id]["model_family"]) for panel_id in panel_ids
    )
    if (
        any(action is not None for action in batch.teacher_actions)
        or tuple(batch.teacher_rule_names) != expected_families
    ):
        raise ProcessV2T1CollatedCacheError("T1 collated teacher coordinates disagree")
    return receipt, batch


def completed_collated_task_identities(
    *, run_root: Path, plan: Mapping[str, Any], runtime: Any
) -> tuple[str, ...]:
    completed: list[str] = []
    for task in plan["tasks"]:
        identifier = str(task["task_identity_sha256"])
        receipt_path = Path(run_root) / TASKS_DIRNAME / identifier / RECEIPT_FILENAME
        if not receipt_path.is_file():
            continue
        validate_collated_cache_leaf(
            receipt_path, plan=plan, runtime=runtime, load_payload=False
        )
        completed.append(identifier)
    return tuple(completed)


def publish_collated_cache_completion(
    *, run_root: Path, plan: Mapping[str, Any], runtime: Any
) -> Path:
    receipts: list[dict[str, Any]] = []
    for task in plan["tasks"]:
        identifier = str(task["task_identity_sha256"])
        receipt_path = Path(run_root) / TASKS_DIRNAME / identifier / RECEIPT_FILENAME
        if not receipt_path.is_file():
            raise ProcessV2T1CollatedCacheError(
                f"T1 collated task receipt is absent: {identifier}"
            )
        receipt, _ = validate_collated_cache_leaf(
            receipt_path, plan=plan, runtime=runtime, load_payload=False
        )
        receipts.append(
            {
                "task_identity_sha256": identifier,
                "receipt_sha256": receipt["receipt_sha256"],
                "payload_file_sha256": receipt["payload_file_sha256"],
                "payload_file_bytes": receipt["payload_file_bytes"],
            }
        )
    body = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **authority_false_block(),
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "implementation_sha256": plan["implementation_sha256"],
        "prepared_completion_sha256": plan["prepared_completion_sha256"],
        "panel_entry_count": plan["panel_entry_count"],
        "panel_entry_inventory_sha256": plan["panel_entry_inventory_sha256"],
        "task_count": len(receipts),
        "task_receipt_inventory_sha256": canonical_sha256(receipts),
        "task_receipts": receipts,
        "gpu_training_launched": False,
    }
    completion = {**body, "completion_sha256": canonical_sha256(body)}
    completion_path = Path(run_root) / COMPLETION_FILENAME
    try:
        write_bytes_if_absent(completion_path, _canonical_bytes(completion) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1CollatedCacheError(str(error)) from error
    return completion_path


def load_materialized_collated_panel(
    completion_path: Path,
    *,
    plan: Mapping[str, Any],
    runtime: Any,
    score_revision_rebind: bool = False,
) -> _MaterializedSemanticT1Panel:
    raw = Path(completion_path).read_bytes()
    try:
        completion = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2T1CollatedCacheError("T1 collated completion is unreadable") from error
    if not isinstance(completion, dict) or raw != _canonical_bytes(completion) + b"\n":
        raise ProcessV2T1CollatedCacheError("T1 collated completion is not canonical JSON")
    if score_revision_rebind and (
        hashlib.sha256(raw).hexdigest()
        != _SCORE_REVISION_PREDECESSOR["completion_file_sha256"]
        or completion.get("completion_sha256")
        != _SCORE_REVISION_PREDECESSOR["completion_sha256"]
    ):
        raise ProcessV2T1CollatedCacheError(
            "T1 collated completion is not the exact allowlisted predecessor"
        )
    try:
        verify_self_hash(
            completion, field="completion_sha256", label="the T1 collated completion"
        )
        require_authority_false(completion, label="the T1 collated completion")
    except ValueError as error:
        raise ProcessV2T1CollatedCacheError(str(error)) from error
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "plan_sha256",
        "run_identity_sha256",
        "implementation_sha256",
        "prepared_completion_sha256",
        "panel_entry_count",
        "panel_entry_inventory_sha256",
        "task_count",
        "task_receipt_inventory_sha256",
        "task_receipts",
        "gpu_training_launched",
        "completion_sha256",
    }
    if set(completion) != expected_fields:
        raise ProcessV2T1CollatedCacheError("T1 collated completion fields disagree")
    expected_receipts = completion["task_receipts"]
    if (
        completion["schema"] != COMPLETION_SCHEMA
        or completion["schema_version"] != COMPLETION_SCHEMA_VERSION
        or completion["status"] != COMPLETION_STATUS
        or completion["plan_sha256"] != plan["plan_sha256"]
        or completion["run_identity_sha256"] != plan["run_identity_sha256"]
        or completion["implementation_sha256"] != plan["implementation_sha256"]
        or completion["prepared_completion_sha256"] != plan["prepared_completion_sha256"]
        or completion["panel_entry_count"] != plan["panel_entry_count"]
        or completion["panel_entry_inventory_sha256"]
        != plan["panel_entry_inventory_sha256"]
        or completion["task_count"] != len(plan["tasks"])
        or completion["task_count"] != len(expected_receipts)
        or completion["task_receipt_inventory_sha256"]
        != canonical_sha256(expected_receipts)
        or completion["gpu_training_launched"] is not False
    ):
        raise ProcessV2T1CollatedCacheError("T1 collated completion identity disagrees")
    batches: list[FactorizedMarkBatch] = []
    for task, summary in zip(plan["tasks"], expected_receipts, strict=True):
        identifier = str(task["task_identity_sha256"])
        receipt_path = Path(completion_path).parent / TASKS_DIRNAME / identifier / RECEIPT_FILENAME
        receipt, batch = validate_collated_cache_leaf(
            receipt_path, plan=plan, runtime=runtime, load_payload=True
        )
        if summary != {
            "task_identity_sha256": identifier,
            "receipt_sha256": receipt["receipt_sha256"],
            "payload_file_sha256": receipt["payload_file_sha256"],
            "payload_file_bytes": receipt["payload_file_bytes"],
        }:
            raise ProcessV2T1CollatedCacheError(
                "T1 collated completion receipt inventory disagrees"
            )
        assert batch is not None
        batches.append(batch)
    batch = _concatenate_factorized_mark_batches(tuple(batches))
    panel_ids = _runtime_panel_ids(runtime)
    entries_by_id = {
        str(entry["panel_entry_sha256"]): entry for entry in runtime.entries
    }
    entries = tuple(entries_by_id[panel_id] for panel_id in panel_ids)
    fibers = tuple(
        runtime.cache.record_for_panel_entry_sha256(panel_id).teacher_fiber
        for panel_id in panel_ids
    )
    partitions = tuple(
        runtime.prepared.partitions_by_panel_entry_sha256[panel_id]
        for panel_id in panel_ids
    )
    if batch.batch_size != len(panel_ids):
        raise ProcessV2T1CollatedCacheError("T1 collated completion batch size disagrees")
    return _MaterializedSemanticT1Panel(
        batch=batch,
        panel_ids=panel_ids,
        fibers=fibers,
        partitions=partitions,
        entries=entries,
        index_by_panel_id={panel_id: index for index, panel_id in enumerate(panel_ids)},
    )


__all__ = [
    "COMPLETION_FILENAME",
    "PLAN_FILENAME",
    "RECEIPT_FILENAME",
    "TASKS_DIRNAME",
    "ProcessV2T1CollatedCacheError",
    "build_collated_cache_plan",
    "compile_collated_cache_leaf",
    "completed_collated_task_identities",
    "load_collated_cache_plan",
    "load_materialized_collated_panel",
    "publish_collated_cache_completion",
    "validate_collated_cache_leaf",
    "validate_collated_cache_plan",
    "write_collated_cache_plan",
]
