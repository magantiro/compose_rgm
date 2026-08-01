"""Restart-safe map/reduce for the frozen Editing-V2 role/lane derivative.

The reference role/lane materializer is intentionally strict and produces one
atomic 20-cell derivative.  Its original execution path scans all immutable
source shards in one process, so a late worker failure loses the expensive
source validation and envelope-rewrite work.  This module changes only the
physical execution plan:

* planning freezes one task per immutable source shard;
* mapping validates and rewrites that shard once, then publishes an immutable
  content-addressed receipt and per-cell fragments;
* reduction validates the exact receipt set, restores the reference global
  source-shard/source-entry order, and calls the reference publication helpers.

The final packed rows, memberships, manifests, hashes, authority flags, and
downstream schema are the reference materializer's.  Map/reduce artifacts are
pre-Active8 corpus derivatives and never authorize Gate 0 or training.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from compose_v4.data import editing_v2_role_lane_packed_materializer as reference
from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
from compose_v4.data.editing_v2_active8_source_adapter import (
    ACTIVE8_ADMISSION_STATUS,
    CANDIDATE_AUTHORITY,
    REQUIRED_BLOCKERS,
    RESOLVED_PACKED_MEMBERSHIP_SCHEMA,
    RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION,
    RESOLVED_PACKED_MEMBERSHIP_STATUS,
    EditingV2Active8SourceAdapterError,
    _candidate_input_identity,
    _expected_original_address,
    _split_input_identity,
    _validate_split_assignment,
)
from compose_v4.data.editing_v2_candidate_provenance_bridge import (
    EditingV2CandidateProvenanceBridgeError,
    validate_candidate_provenance_bridge,
)
from compose_v4.data.editing_v2_lane_registry import (
    LANE_REGISTRY_SCHEMA,
    LANE_REGISTRY_SCHEMA_VERSION,
    LANE_REGISTRY_STATUS,
    editing_corpus_contract_identity,
    editing_v2_lane_definitions,
    file_sha256,
)
from compose_v4.data.editing_v2_packed_candidate_materializer import (
    CANDIDATE_ROWS_FILENAME,
    MATERIALIZATION_FILENAME,
    MATERIALIZATION_SCHEMA,
    MATERIALIZATION_SCHEMA_VERSION,
    MATERIALIZATION_STATUS,
    canonical_json_bytes,
    canonical_sha256,
)
from compose_v4.data.editing_v2_split_assignment import (
    EditingV2SplitAssignmentError,
    validate_candidate_source_stream,
)
from compose_v4.data.editing_v2_split_census import PARTITION_ROLES
from compose_v4.data.packed_trace_store import sampler_contract

PLAN_SCHEMA = "compose.editing_v2_role_lane_mapreduce_plan"
PLAN_SCHEMA_VERSION = 1
PLAN_STATUS = "FROZEN_ROLE_LANE_SHARD_PLAN_NO_TRAINING_AUTHORITY"
TASK_RECEIPT_SCHEMA = "compose.editing_v2_role_lane_shard_receipt"
TASK_RECEIPT_SCHEMA_VERSION = 1
TASK_RECEIPT_STATUS = "COMPLETE_ROLE_LANE_SHARD_FRAGMENTS_NO_TRAINING_AUTHORITY"
CELL_RECEIPT_SCHEMA = "compose.editing_v2_role_lane_cell_receipt"
CELL_RECEIPT_SCHEMA_VERSION = 1
CELL_RECEIPT_STATUS = "COMPLETE_ROLE_LANE_CELL_NO_TRAINING_AUTHORITY"

PLAN_FILENAME = "ROLE_LANE_MAPREDUCE_PLAN.json"
TASK_INPUT_FILENAME = "selected_candidates.jsonl.gz"
TASK_RECEIPT_FILENAME = "ROLE_LANE_SHARD_COMPLETE.json"
TASK_PROGRESS_FILENAME = "ROLE_LANE_SHARD_PROGRESS.json"
CELL_RECEIPT_FILENAME = "ROLE_LANE_CELL_COMPLETE.json"
CELL_MEMBERSHIP_FILENAME = "cell_memberships.jsonl"
CELL_PROGRESS_FILENAME = "ROLE_LANE_CELL_PROGRESS.json"
FRAGMENT_ROWS_FILENAME = "rows.jsonl.gz"
FRAGMENT_MEMBERSHIP_FILENAME = "memberships.jsonl"
DEFAULT_EXECUTION_ARTIFACT_PREFIX = "/artifacts/editing_v2/role_lane_mapreduce"
DEFAULT_EXPECTED_SOURCE_SHARDS = 62

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EditingV2RoleLaneMapReduceError(RuntimeError):
    """A role/lane map/reduce artifact is invalid, stale, or inconsistent."""


class EditingV2RoleLaneMapReduceIncomplete(EditingV2RoleLaneMapReduceError):
    """At least one exact planned source-shard task is not complete."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = canonical_json_bytes(value)
    return encoded + (b"\n" if newline else b"")


def _self_hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != field}
    return {**body, field: canonical_sha256(body)}


def _require_self_hash(value: object, *, field: str, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingV2RoleLaneMapReduceError(f"{name} must be an object")
    result = dict(value)
    observed = result.get(field)
    if not isinstance(observed, str) or _SHA256_RE.fullmatch(observed) is None:
        raise EditingV2RoleLaneMapReduceError(f"{name}.{field} must be a SHA-256")
    body = {key: item for key, item in result.items() if key != field}
    if observed != canonical_sha256(body):
        raise EditingV2RoleLaneMapReduceError(f"{name} self-hash disagrees")
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: object) -> None:
    """Replace one operational progress record without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_task_progress(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    artifact_root: Path,
    phase: str,
    started_at: str,
    elapsed_seconds: float,
    processed_candidates: int,
    detail: Mapping[str, Any] | None = None,
) -> None:
    """Persist nonauthorizing operational state separately from scientific receipts."""

    task_id = task["task_identity_sha256"]
    path = _mounted_path(
        f"{plan['run_artifact_root']}/progress/{task_id}/{TASK_PROGRESS_FILENAME}",
        artifact_root=artifact_root,
        field="task progress path",
    )
    body = {
        "schema": "compose.editing_v2_role_lane_shard_progress",
        "schema_version": 1,
        "status": phase,
        "training_authorized": False,
        "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
        "plan_sha256": plan["plan_sha256"],
        "task_identity_sha256": task_id,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "processed_candidates": processed_candidates,
        "expected_candidates": task["selected_candidates"],
        "detail": dict(detail or {}),
    }
    _write_json_atomic(path, _self_hashed(body, "progress_sha256"))


def _write_cell_progress(
    *,
    plan: Mapping[str, Any],
    artifact_root: Path,
    lane: str,
    role: str,
    phase: str,
    started_at: str,
    elapsed_seconds: float,
    processed_records: int,
    detail: Mapping[str, Any] | None = None,
) -> None:
    path = _mounted_path(
        f"{plan['run_artifact_root']}/cell_progress/{lane}/{role}/{CELL_PROGRESS_FILENAME}",
        artifact_root=artifact_root,
        field="cell progress path",
    )
    body = {
        "schema": "compose.editing_v2_role_lane_cell_progress",
        "schema_version": 1,
        "status": phase,
        "training_authorized": False,
        "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
        "plan_sha256": plan["plan_sha256"],
        "data_lane": lane,
        "partition_role": role,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "processed_records": processed_records,
        "detail": dict(detail or {}),
    }
    _write_json_atomic(path, _self_hashed(body, "progress_sha256"))


def _write_jsonl(path: Path, rows: Iterable[object]) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        handle = (
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
            if path.suffix == ".gz"
            else raw
        )
        try:
            for row in rows:
                encoded = _canonical_bytes(row, newline=True)
                handle.write(encoded)
                digest.update(encoded)
                count += 1
        finally:
            if handle is not raw:
                handle.close()
        raw.flush()
        os.fsync(raw.fileno())
    return count, digest.hexdigest()


def _iter_jsonl(path: Path, *, name: str) -> Iterable[dict[str, Any]]:
    try:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rb") as handle:
            for index, raw in enumerate(handle):
                if not raw.endswith(b"\n") or not raw.strip():
                    raise EditingV2RoleLaneMapReduceError(
                        f"{name} row {index} is empty or lacks a newline"
                    )
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise EditingV2RoleLaneMapReduceError(
                        f"{name} row {index} is invalid JSON"
                    ) from error
                if not isinstance(value, Mapping) or raw != _canonical_bytes(
                    value, newline=True
                ):
                    raise EditingV2RoleLaneMapReduceError(
                        f"{name} row {index} is not a canonical object"
                    )
                yield dict(value)
    except OSError as error:
        raise EditingV2RoleLaneMapReduceError(f"cannot read {name}: {path}") from error


def _stream_file_identity(path: Path, *, name: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in _iter_jsonl(path, name=name):
        digest.update(_canonical_bytes(row, newline=True))
        count += 1
    return count, digest.hexdigest()


def _copy_verified_bytes(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    """Copy immutable bytes and reopen the destination against its receipt."""

    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise EditingV2RoleLaneMapReduceError(
            "copied payload receipt must contain a full lowercase SHA-256"
        )
    try:
        with source.open("rb") as source_handle, destination.open("xb") as output:
            shutil.copyfileobj(source_handle, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        observed = file_sha256(destination)
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise EditingV2RoleLaneMapReduceError(
            f"cannot copy verified payload: {source} -> {destination}"
        ) from error
    if observed != expected_sha256:
        destination.unlink(missing_ok=True)
        raise EditingV2RoleLaneMapReduceError(
            "copied payload SHA-256 disagrees with its immutable receipt"
        )


def _artifact_address(path: Path, *, artifact_root: Path, field: str) -> str:
    root = artifact_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise EditingV2RoleLaneMapReduceError(
            f"{field} must resolve below artifact_root"
        ) from error
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _mounted_path(value: object, *, artifact_root: Path, field: str) -> Path:
    raw = value if isinstance(value, str) else ""
    pure = PurePosixPath(raw)
    if (
        not raw
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or str(pure) != raw
    ):
        raise EditingV2RoleLaneMapReduceError(
            f"{field} must be a normalized path below /artifacts"
        )
    root = artifact_root.resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EditingV2RoleLaneMapReduceError(
            f"{field} resolves outside artifact_root"
        ) from error
    return resolved


def _implementation_sha256() -> str:
    return file_sha256(Path(__file__))


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2RoleLaneMapReduceError(f"cannot load {name}: {path}") from error
    if not isinstance(value, Mapping):
        raise EditingV2RoleLaneMapReduceError(f"{name} must be an object")
    return dict(value)


def _load_candidate_manifest_header(candidate_root: Path) -> dict[str, Any]:
    """Validate immutable manifest identity without scanning candidate rows."""

    manifest = _load_json(
        candidate_root / MATERIALIZATION_FILENAME,
        name="candidate materialization manifest",
    )
    supplied = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    rows = manifest.get("rows")
    rows_path = candidate_root / CANDIDATE_ROWS_FILENAME
    if (
        not isinstance(supplied, str)
        or _SHA256_RE.fullmatch(supplied) is None
        or manifest.get("schema") != MATERIALIZATION_SCHEMA
        or manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or manifest.get("status") != MATERIALIZATION_STATUS
        or manifest.get("training_authorized") is not False
        or canonical_sha256(body) != supplied
        or not isinstance(rows, Mapping)
        or rows.get("relative_path") != CANDIDATE_ROWS_FILENAME
        or not rows_path.is_file()
    ):
        raise EditingV2RoleLaneMapReduceError(
            "candidate materialization header identity, authority, or row path disagrees"
        )
    return manifest


def _prepare_reference_identity(
    *,
    candidate_materialization_dir: Path,
    candidate_provenance_bridge_dir: Path,
    candidate_provenance_registry_path: Path,
    split_assignment_path: Path,
    editing_corpus_contract_path: Path,
    code_revision: str,
    final_output_artifact_prefix: str,
    candidate_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str], tuple[str, ...], Any]:
    """Validate the exact reference inputs and reconstruct its run identity."""

    if _COMMIT_RE.fullmatch(code_revision) is None:
        raise EditingV2RoleLaneMapReduceError(
            "code_revision must be a full lowercase 40-character Git SHA"
        )
    final_output_artifact_prefix = reference._artifact_prefix(
        final_output_artifact_prefix
    )
    if candidate_manifest is None:
        candidate_manifest = _load_candidate_manifest_header(
            candidate_materialization_dir
        )
    candidate_identity = _candidate_input_identity(
        candidate_materialization_dir,
        candidate_manifest,
    )
    try:
        assignment, assigned_roles = _validate_split_assignment(split_assignment_path)
        split_identity = _split_input_identity(split_assignment_path, assignment)
    except EditingV2Active8SourceAdapterError as error:
        raise EditingV2RoleLaneMapReduceError(
            "split assignment validation failed before planning"
        ) from error
    contract = load_editing_corpus_contract(editing_corpus_contract_path)
    contract_identity = editing_corpus_contract_identity(
        contract,
        contract_file_sha256=file_sha256(editing_corpus_contract_path),
    )
    lane_definitions = editing_v2_lane_definitions(contract)
    lane_ids = tuple(lane.lane_id for lane in lane_definitions)
    try:
        bridge = validate_candidate_provenance_bridge(
            candidate_provenance_bridge_dir,
            candidate_root=candidate_materialization_dir,
            provenance_registry_path=candidate_provenance_registry_path,
            editing_corpus_contract_path=editing_corpus_contract_path,
            _validated_candidate_materialization=candidate_manifest,
        )
        source_stream = validate_candidate_source_stream(
            bridge.get("source_stream"),
            expected_candidate_materialization=candidate_identity,
            expected_nonempty_rows=len(assigned_roles),
        )
    except (
        EditingV2CandidateProvenanceBridgeError,
        EditingV2SplitAssignmentError,
        OSError,
    ) as error:
        raise EditingV2RoleLaneMapReduceError(
            "candidate provenance bridge validation failed before planning"
        ) from error
    if source_stream != assignment["source_stream"]:
        raise EditingV2RoleLaneMapReduceError(
            "split assignment candidate provenance source stream is stale"
        )
    run_identity_body = {
        "schema": reference.ROLE_LANE_MATERIALIZATION_SCHEMA,
        "schema_version": reference.ROLE_LANE_MATERIALIZATION_SCHEMA_VERSION,
        "code_revision": code_revision,
        "materializer_implementation_sha256": file_sha256(Path(reference.__file__)),
        "candidate_materialization": candidate_identity,
        "candidate_provenance_source_stream": source_stream,
        "split_assignment": split_identity,
        "editing_corpus_contract": contract_identity,
        "lane_order": list(lane_ids),
        "role_order": list(PARTITION_ROLES),
        "envelope_rewrite_contract": reference.ENVELOPE_REWRITE_CONTRACT,
        "sampler_contract": sampler_contract(),
        "output_artifact_prefix": final_output_artifact_prefix,
    }
    run_identity_sha256 = canonical_sha256(run_identity_body)
    return (
        {
            "run_identity_body": run_identity_body,
            "run_identity_sha256": run_identity_sha256,
            "run_artifact_root": (
                f"{final_output_artifact_prefix}/{run_identity_sha256}"
            ),
            "candidate_root": str(candidate_materialization_dir),
            "split_assignment_path": str(split_assignment_path),
            "editing_corpus_contract_path": str(editing_corpus_contract_path),
        },
        assigned_roles,
        lane_ids,
        lane_definitions,
    )


def _validate_task_input(
    task: Mapping[str, Any], *, artifact_root: Path
) -> list[dict[str, Any]]:
    path = _mounted_path(
        task.get("selection_artifact_path"),
        artifact_root=artifact_root,
        field="task selection_artifact_path",
    )
    if not path.is_file() or file_sha256(path) != task.get("selection_file_sha256"):
        raise EditingV2RoleLaneMapReduceError(
            f"task selection file is absent or tampered: {path}"
        )
    rows = list(_iter_jsonl(path, name="task selection"))
    if len(rows) != task.get("selected_candidates") or canonical_sha256(
        rows
    ) != task.get("selection_semantic_sha256"):
        raise EditingV2RoleLaneMapReduceError(
            "task selection stream identity disagrees"
        )
    previous = -1
    candidate_ids: set[str] = set()
    for row in rows:
        if set(row) != {"assigned_role", "candidate"}:
            raise EditingV2RoleLaneMapReduceError("task selection row fields disagree")
        candidate = row["candidate"]
        if not isinstance(candidate, Mapping):
            raise EditingV2RoleLaneMapReduceError("task candidate must be an object")
        index = candidate.get("packed_address", {}).get("entry_index")
        candidate_id = candidate.get("candidate_id")
        if (
            type(index) is not int
            or index <= previous
            or not isinstance(candidate_id, str)
            or candidate_id in candidate_ids
        ):
            raise EditingV2RoleLaneMapReduceError(
                "task candidates must have unique IDs and increasing source entries"
            )
        previous = index
        candidate_ids.add(candidate_id)
    return rows


def validate_role_lane_mapreduce_plan(
    plan_path: str | Path,
    *,
    artifact_root: str | Path,
    validate_task_inputs: bool = False,
) -> dict[str, Any]:
    """Validate a frozen plan and optionally every physical task input."""

    path = Path(plan_path)
    plan = _require_self_hash(
        _load_json(path, name="role/lane map/reduce plan"),
        field="plan_sha256",
        name="role/lane map/reduce plan",
    )
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("training_authorized") is not False
        or plan.get("active8_admission_status") != ACTIVE8_ADMISSION_STATUS
        or plan.get("mapreduce_implementation_sha256") != _implementation_sha256()
    ):
        raise EditingV2RoleLaneMapReduceError("map/reduce plan identity disagrees")
    reference_identity = plan.get("reference_materialization")
    tasks = plan.get("tasks")
    if (
        not isinstance(reference_identity, Mapping)
        or not isinstance(tasks, list)
        or not tasks
    ):
        raise EditingV2RoleLaneMapReduceError("map/reduce plan lacks identity or tasks")
    run_body = reference_identity.get("run_identity_body")
    if (
        not isinstance(run_body, Mapping)
        or canonical_sha256(run_body) != reference_identity.get("run_identity_sha256")
        or reference_identity.get("run_artifact_root")
        != f"{run_body.get('output_artifact_prefix')}/{reference_identity.get('run_identity_sha256')}"
        or run_body.get("materializer_implementation_sha256")
        != file_sha256(Path(reference.__file__))
    ):
        raise EditingV2RoleLaneMapReduceError(
            "reference materialization identity disagrees"
        )
    expected_ids: list[str] = []
    seen_sources: set[tuple[str, str]] = set()
    previous_source: tuple[str, str] | None = None
    root = Path(artifact_root)
    for ordinal, raw_task in enumerate(tasks):
        if not isinstance(raw_task, Mapping):
            raise EditingV2RoleLaneMapReduceError("plan task must be an object")
        task = dict(raw_task)
        identity_body = task.get("task_identity_body")
        task_id = task.get("task_identity_sha256")
        source_key = (task.get("source_relative_path"), task.get("source_shard_sha256"))
        if (
            task.get("task_ordinal") != ordinal
            or not isinstance(identity_body, Mapping)
            or canonical_sha256(identity_body) != task_id
            or identity_body.get("reference_run_identity_sha256")
            != reference_identity.get("run_identity_sha256")
            or source_key in seen_sources
            or (previous_source is not None and source_key <= previous_source)
        ):
            raise EditingV2RoleLaneMapReduceError("plan task identity/order disagrees")
        expected_ids.append(task_id)
        seen_sources.add(source_key)
        previous_source = source_key
        if validate_task_inputs:
            _validate_task_input(task, artifact_root=root)
    if (
        expected_ids != plan.get("task_order")
        or canonical_sha256(tasks) != plan.get("task_inventory_sha256")
        or sum(int(task["selected_candidates"]) for task in tasks)
        != plan.get("selected_candidates")
    ):
        raise EditingV2RoleLaneMapReduceError("plan task inventory disagrees")
    return plan


def plan_role_lane_mapreduce(
    *,
    candidate_materialization_dir: str | Path,
    candidate_provenance_bridge_dir: str | Path,
    candidate_provenance_registry_path: str | Path,
    split_assignment_path: str | Path,
    editing_corpus_contract_path: str | Path,
    artifact_root: str | Path,
    code_revision: str,
    final_output_artifact_prefix: str = reference.OUTPUT_NAMESPACE,
    execution_artifact_prefix: str = DEFAULT_EXECUTION_ARTIFACT_PREFIX,
    expected_source_shards: int = DEFAULT_EXPECTED_SOURCE_SHARDS,
) -> dict[str, Any]:
    """Freeze immutable source-shard tasks without reading packed state payloads."""

    artifact_root = Path(artifact_root)
    if not artifact_root.is_dir():
        raise EditingV2RoleLaneMapReduceError(
            f"artifact root is absent: {artifact_root}"
        )
    if type(expected_source_shards) is not int or expected_source_shards <= 0:
        raise EditingV2RoleLaneMapReduceError(
            "expected_source_shards must be a positive integer"
        )
    execution_artifact_prefix = reference._artifact_prefix(execution_artifact_prefix)
    candidate_root = Path(candidate_materialization_dir)
    split_path = Path(split_assignment_path)
    contract_path = Path(editing_corpus_contract_path)
    candidate_manifest = _load_candidate_manifest_header(candidate_root)
    descriptor, database_name = tempfile.mkstemp(
        prefix="compose-editing-v2-role-lane-plan-", suffix=".sqlite3"
    )
    os.close(descriptor)
    connection = reference._database(Path(database_name))
    scratch = Path(tempfile.mkdtemp(prefix="compose-editing-v2-role-lane-plan-"))
    try:
        reference_identity, assigned_roles, lane_ids, _ = _prepare_reference_identity(
            candidate_materialization_dir=candidate_root,
            candidate_provenance_bridge_dir=Path(candidate_provenance_bridge_dir),
            candidate_provenance_registry_path=Path(candidate_provenance_registry_path),
            split_assignment_path=split_path,
            editing_corpus_contract_path=contract_path,
            code_revision=code_revision,
            final_output_artifact_prefix=final_output_artifact_prefix,
            candidate_manifest=candidate_manifest,
        )
        routed_count = reference._index_selected_candidates(
            connection,
            candidate_root=candidate_root,
            assigned_roles=assigned_roles,
            lane_ids=lane_ids,
        )
        source_groups = connection.execute("""
            SELECT DISTINCT source_relative_path, source_shard_sha256,
                            source_manifest_sha256, source_overlay_sha256
            FROM selected
            ORDER BY source_relative_path, source_shard_sha256
            """).fetchall()
        if len(source_groups) != expected_source_shards:
            raise EditingV2RoleLaneMapReduceError(
                f"expected exactly {expected_source_shards} selected source shards, "
                f"observed {len(source_groups)}"
            )
        task_drafts: list[dict[str, Any]] = []
        for ordinal, (
            relative_path,
            shard_sha256,
            manifest_sha256,
            overlay_sha256,
        ) in enumerate(source_groups):
            selections = [
                {
                    "assigned_role": role,
                    "candidate": json.loads(candidate_json),
                }
                for role, candidate_json in connection.execute(
                    """
                    SELECT assigned_role, candidate_json
                    FROM selected
                    WHERE source_relative_path = ?
                      AND source_shard_sha256 = ?
                      AND source_manifest_sha256 = ?
                      AND source_overlay_sha256 IS ?
                    ORDER BY source_entry_index
                    """,
                    (relative_path, shard_sha256, manifest_sha256, overlay_sha256),
                )
            ]
            selection_semantic_sha256 = canonical_sha256(selections)
            task_identity_body = {
                "reference_run_identity_sha256": reference_identity[
                    "run_identity_sha256"
                ],
                "source_relative_path": relative_path,
                "source_shard_sha256": shard_sha256,
                "source_manifest_sha256": manifest_sha256,
                "source_overlay_sha256": overlay_sha256,
                "selection_semantic_sha256": selection_semantic_sha256,
                "selected_candidates": len(selections),
            }
            task_id = canonical_sha256(task_identity_body)
            selection_path = scratch / task_id / TASK_INPUT_FILENAME
            count, _ = _write_jsonl(selection_path, selections)
            if count != len(selections):
                raise EditingV2RoleLaneMapReduceError(
                    "task selection write was incomplete"
                )
            task_drafts.append(
                {
                    "task_ordinal": ordinal,
                    "task_identity_body": task_identity_body,
                    "task_identity_sha256": task_id,
                    "source_relative_path": relative_path,
                    "source_shard_sha256": shard_sha256,
                    "source_manifest_sha256": manifest_sha256,
                    "source_overlay_sha256": overlay_sha256,
                    "selected_candidates": len(selections),
                    "selection_semantic_sha256": selection_semantic_sha256,
                    "selection_file_sha256": file_sha256(selection_path),
                    "_selection_path": selection_path,
                }
            )
        inventory_for_identity = [
            {key: item for key, item in task.items() if not key.startswith("_")}
            for task in task_drafts
        ]
        plan_identity_body = {
            "schema": PLAN_SCHEMA,
            "schema_version": PLAN_SCHEMA_VERSION,
            "code_revision": code_revision,
            "mapreduce_implementation_sha256": _implementation_sha256(),
            "reference_materialization": reference_identity,
            "task_inventory_without_paths_sha256": canonical_sha256(
                inventory_for_identity
            ),
            "selected_candidates": routed_count,
            "task_count": len(task_drafts),
            "expected_source_shards": expected_source_shards,
            "execution_artifact_prefix": execution_artifact_prefix,
        }
        run_identity_sha256 = canonical_sha256(plan_identity_body)
        run_artifact_root = f"{execution_artifact_prefix}/{run_identity_sha256}"
        target = _mounted_path(
            run_artifact_root,
            artifact_root=artifact_root,
            field="map/reduce run root",
        )
        tasks: list[dict[str, Any]] = []
        for draft in task_drafts:
            task_id = draft["task_identity_sha256"]
            tasks.append(
                {
                    **{
                        key: item
                        for key, item in draft.items()
                        if not key.startswith("_")
                    },
                    "selection_artifact_path": (
                        f"{run_artifact_root}/task_inputs/{task_id}/{TASK_INPUT_FILENAME}"
                    ),
                }
            )
        plan_body = {
            **plan_identity_body,
            "status": PLAN_STATUS,
            "training_authorized": False,
            "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
            "run_identity_sha256": run_identity_sha256,
            "run_artifact_root": run_artifact_root,
            "task_order": [task["task_identity_sha256"] for task in tasks],
            "tasks": tasks,
            "task_inventory_sha256": canonical_sha256(tasks),
        }
        plan = _self_hashed(plan_body, "plan_sha256")
        if target.exists():
            existing = validate_role_lane_mapreduce_plan(
                target / PLAN_FILENAME,
                artifact_root=artifact_root,
                validate_task_inputs=True,
            )
            if existing != plan:
                raise EditingV2RoleLaneMapReduceError(
                    "content-addressed plan collides with different content"
                )
            return existing
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                dir=target.parent,
                prefix=f".{run_identity_sha256}.",
                suffix=".staging",
            )
        )
        try:
            for draft in task_drafts:
                task_id = draft["task_identity_sha256"]
                destination = staging / "task_inputs" / task_id / TASK_INPUT_FILENAME
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(draft["_selection_path"], destination)
            _write_json(staging / PLAN_FILENAME, plan)
            os.replace(staging, target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return validate_role_lane_mapreduce_plan(
            target / PLAN_FILENAME,
            artifact_root=artifact_root,
            validate_task_inputs=True,
        )
    finally:
        connection.close()
        Path(database_name).unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)


def _task_by_id(plan: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    matches = [
        dict(task)
        for task in plan["tasks"]
        if task.get("task_identity_sha256") == task_id
    ]
    if len(matches) != 1:
        raise EditingV2RoleLaneMapReduceError(
            f"task ID is absent or duplicated in plan: {task_id}"
        )
    return matches[0]


def _fragment_paths(root: Path, lane: str, role: str) -> tuple[Path, Path]:
    cell = root / "fragments" / lane / role
    return cell / FRAGMENT_ROWS_FILENAME, cell / FRAGMENT_MEMBERSHIP_FILENAME


def _validate_fragment(
    fragment: Mapping[str, Any], *, task_root: Path, artifact_root: Path
) -> None:
    lane = fragment.get("data_lane")
    role = fragment.get("partition_role")
    rows_path = _mounted_path(
        fragment.get("rows_artifact_path"),
        artifact_root=artifact_root,
        field="fragment rows_artifact_path",
    )
    membership_path = _mounted_path(
        fragment.get("membership_artifact_path"),
        artifact_root=artifact_root,
        field="fragment membership_artifact_path",
    )
    if not rows_path.is_relative_to(task_root) or not membership_path.is_relative_to(
        task_root
    ):
        raise EditingV2RoleLaneMapReduceError(
            "fragment escapes its immutable task root"
        )
    if (
        not rows_path.is_file()
        or file_sha256(rows_path) != fragment.get("rows_file_sha256")
        or not membership_path.is_file()
        or file_sha256(membership_path) != fragment.get("membership_file_sha256")
    ):
        raise EditingV2RoleLaneMapReduceError(
            f"fragment files are missing or tampered: {lane}/{role}"
        )
    memberships = list(_iter_jsonl(membership_path, name="fragment membership"))
    membership_digest = hashlib.sha256()
    for member in memberships:
        membership_digest.update(_canonical_bytes(member, newline=True))
    row_count = 0
    output_digest = hashlib.sha256()
    original_digest = hashlib.sha256()
    try:
        with gzip.open(rows_path, "rb") as row_handle:
            for local_index, (raw, member) in enumerate(
                zip(row_handle, memberships, strict=True)
            ):
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise EditingV2RoleLaneMapReduceError(
                        f"fragment row {local_index} is invalid JSON"
                    ) from error
                encoded = _canonical_bytes(row)
                if (
                    raw != encoded + b"\n"
                    or member.get("local_entry_index") != local_index
                    or member.get("data_lane") != lane
                    or member.get("partition_role") != role
                    or member.get("output_packed_row_sha256")
                    != hashlib.sha256(encoded).hexdigest()
                    or row.get("trace", {}).get("layer") != lane
                    or row.get("trace", {}).get("partition") != role
                ):
                    raise EditingV2RoleLaneMapReduceError(
                        f"fragment row/membership disagrees at {lane}/{role}/{local_index}"
                    )
                output_digest.update(member["output_packed_row_sha256"].encode("ascii"))
                output_digest.update(b"\n")
                original_address = member.get("original_packed_address", {}).get(
                    "packed_address", {}
                )
                address_sha256 = original_address.get("address_sha256")
                if not isinstance(address_sha256, str):
                    raise EditingV2RoleLaneMapReduceError(
                        "fragment membership lacks original address identity"
                    )
                original_digest.update(address_sha256.encode("ascii"))
                original_digest.update(b"\n")
                row_count += 1
    except OSError as error:
        raise EditingV2RoleLaneMapReduceError(
            f"cannot read fragment rows: {rows_path}"
        ) from error
    if (
        row_count != fragment.get("records")
        or row_count != len(memberships)
        or output_digest.hexdigest() != fragment.get("output_row_stream_sha256")
        or original_digest.hexdigest() != fragment.get("original_address_stream_sha256")
        or membership_digest.hexdigest() != fragment.get("membership_stream_sha256")
    ):
        raise EditingV2RoleLaneMapReduceError(
            f"fragment stream identity disagrees: {lane}/{role}"
        )


def validate_role_lane_task_receipt(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    artifact_root: str | Path,
) -> dict[str, Any]:
    root = Path(artifact_root)
    task_id = task["task_identity_sha256"]
    task_root = _mounted_path(
        f"{plan['run_artifact_root']}/task_results/{task_id}",
        artifact_root=root,
        field="task result root",
    )
    receipt = _require_self_hash(
        _load_json(task_root / TASK_RECEIPT_FILENAME, name="role/lane task receipt"),
        field="receipt_sha256",
        name="role/lane task receipt",
    )
    fragments = receipt.get("fragments")
    if (
        receipt.get("schema") != TASK_RECEIPT_SCHEMA
        or receipt.get("schema_version") != TASK_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != TASK_RECEIPT_STATUS
        or receipt.get("training_authorized") is not False
        or receipt.get("active8_admission_status") != ACTIVE8_ADMISSION_STATUS
        or receipt.get("plan_sha256") != plan.get("plan_sha256")
        or receipt.get("task_identity_sha256") != task_id
        or receipt.get("task_identity_body") != task.get("task_identity_body")
        or receipt.get("selection_file_sha256") != task.get("selection_file_sha256")
        or receipt.get("selection_semantic_sha256")
        != task.get("selection_semantic_sha256")
        or not isinstance(fragments, list)
        or not fragments
    ):
        raise EditingV2RoleLaneMapReduceError(
            f"task receipt identity disagrees: {task_id}"
        )
    cells: set[tuple[str, str]] = set()
    records = 0
    states = 0
    transitions = 0
    for fragment in fragments:
        if not isinstance(fragment, Mapping):
            raise EditingV2RoleLaneMapReduceError("task fragment must be an object")
        cell = (fragment.get("data_lane"), fragment.get("partition_role"))
        if cell in cells:
            raise EditingV2RoleLaneMapReduceError(
                "task receipt duplicates a cell fragment"
            )
        cells.add(cell)
        _validate_fragment(fragment, task_root=task_root, artifact_root=root)
        records += int(fragment["records"])
        states += int(fragment["states"])
        transitions += int(fragment["transitions"])
    if receipt.get("totals") != {
        "records": records,
        "states": states,
        "transitions": transitions,
        "fragments": len(fragments),
    } or records != task.get("selected_candidates"):
        raise EditingV2RoleLaneMapReduceError("task receipt totals disagree")
    return receipt


def map_role_lane_source_shard(
    *,
    plan_path: str | Path,
    task_id: str,
    artifact_root: str | Path,
    max_source_row_bytes: int = reference.DEFAULT_MAX_SOURCE_ROW_BYTES,
) -> dict[str, Any]:
    """Materialize one immutable source shard and safely reuse a valid receipt."""

    root = Path(artifact_root)
    plan = validate_role_lane_mapreduce_plan(plan_path, artifact_root=root)
    task = _task_by_id(plan, task_id)
    selections = _validate_task_input(task, artifact_root=root)
    started_at = _utc_now()
    started = time.perf_counter()
    processed_candidates = 0
    _write_task_progress(
        plan=plan,
        task=task,
        artifact_root=root,
        phase="STARTED",
        started_at=started_at,
        elapsed_seconds=0.0,
        processed_candidates=0,
    )
    target = _mounted_path(
        f"{plan['run_artifact_root']}/task_results/{task_id}",
        artifact_root=root,
        field="task result root",
    )
    if target.exists():
        receipt = validate_role_lane_task_receipt(
            plan=plan,
            task=task,
            artifact_root=root,
        )
        _write_task_progress(
            plan=plan,
            task=task,
            artifact_root=root,
            phase="REUSED_COMPLETE",
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            processed_candidates=task["selected_candidates"],
            detail={"receipt_sha256": receipt["receipt_sha256"]},
        )
        return receipt
    source_path = reference._verify_frozen_source(
        artifact_root=root,
        relative_path=task["source_relative_path"],
        shard_sha256=task["source_shard_sha256"],
        manifest_sha256=task["source_manifest_sha256"],
        overlay_sha256=task["source_overlay_sha256"],
    )
    _write_task_progress(
        plan=plan,
        task=task,
        artifact_root=root,
        phase="SOURCE_VERIFIED",
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - started,
        processed_candidates=0,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{task_id}.",
            suffix=".staging",
        )
    )
    handles: dict[tuple[str, str], tuple[Any, Any, Path]] = {}
    memberships: dict[tuple[str, str], list[dict[str, Any]]] = {}
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    published = False
    try:
        wanted = iter(selections)
        try:
            selection = next(wanted)
        except StopIteration as error:
            raise EditingV2RoleLaneMapReduceError(
                "map task has no selected candidates"
            ) from error
        wanted_index = selection["candidate"]["packed_address"]["entry_index"]
        for source_index, entry in reference._iter_bounded_gzip_jsonl(
            source_path,
            max_row_bytes=max_source_row_bytes,
        ):
            if source_index < wanted_index:
                continue
            if source_index > wanted_index:
                raise EditingV2RoleLaneMapReduceError(
                    f"selected source entry is missing: {wanted_index}"
                )
            candidate = selection["candidate"]
            role = selection["assigned_role"]
            lane = candidate["data_lane"]
            reference._validate_original_candidate(entry, candidate)
            rewritten = reference._rewrite_envelope_only(entry, lane=lane, role=role)
            reference._assert_envelope_only(entry, rewritten, lane=lane, role=role)
            cell = (lane, role)
            if cell not in handles:
                rows_path, _ = _fragment_paths(staging, lane, role)
                rows_path.parent.mkdir(parents=True, exist_ok=True)
                raw = rows_path.open("wb")
                compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
                handles[cell] = (raw, compressed, rows_path)
                memberships[cell] = []
                stats[cell] = {
                    "records": 0,
                    "states": 0,
                    "transitions": 0,
                    "original_digest": hashlib.sha256(),
                    "output_digest": hashlib.sha256(),
                }
            encoded = _canonical_bytes(rewritten)
            handles[cell][1].write(encoded + b"\n")
            original_address = _expected_original_address(candidate)
            original_row_sha256 = canonical_sha256(entry)
            output_row_sha256 = hashlib.sha256(encoded).hexdigest()
            trace = rewritten["trace"]
            local_index = stats[cell]["records"]
            membership = {
                "local_entry_index": local_index,
                "source_entry_index": source_index,
                "candidate_id": candidate["candidate_id"],
                "data_lane": lane,
                "partition_role": role,
                "original_packed_address": original_address,
                "original_packed_row_sha256": original_row_sha256,
                "output_packed_row_sha256": output_row_sha256,
                "trace_id": trace["trace_id"],
                "source_key": trace["source_key"],
                "target_key": trace["target_key"],
                "path_length": trace["path_length"],
                "state_count": candidate["exact_states"]["state_count"],
            }
            memberships[cell].append(membership)
            address_sha256 = original_address["packed_address"]["address_sha256"]
            stats[cell]["original_digest"].update(address_sha256.encode("ascii"))
            stats[cell]["original_digest"].update(b"\n")
            stats[cell]["output_digest"].update(output_row_sha256.encode("ascii"))
            stats[cell]["output_digest"].update(b"\n")
            stats[cell]["records"] += 1
            stats[cell]["states"] += int(membership["state_count"])
            stats[cell]["transitions"] += int(membership["path_length"])
            processed_candidates += 1
            if processed_candidates % 4096 == 0:
                _write_task_progress(
                    plan=plan,
                    task=task,
                    artifact_root=root,
                    phase="MAPPING",
                    started_at=started_at,
                    elapsed_seconds=time.perf_counter() - started,
                    processed_candidates=processed_candidates,
                )
            try:
                selection = next(wanted)
                wanted_index = selection["candidate"]["packed_address"]["entry_index"]
            except StopIteration:
                selection = None
                break
        if selection is not None:
            raise EditingV2RoleLaneMapReduceError(
                f"selected source entry is missing: {wanted_index}"
            )
        for raw, compressed, _ in handles.values():
            compressed.close()
            raw.flush()
            os.fsync(raw.fileno())
            raw.close()
        handles.clear()
        lane_order = tuple(
            plan["reference_materialization"]["run_identity_body"]["lane_order"]
        )
        fragments: list[dict[str, Any]] = []
        for lane in lane_order:
            for role in PARTITION_ROLES:
                cell = (lane, role)
                if cell not in stats:
                    continue
                rows_path, membership_path = _fragment_paths(staging, lane, role)
                _, membership_stream_sha256 = _write_jsonl(
                    membership_path,
                    memberships[cell],
                )
                fragments.append(
                    {
                        "data_lane": lane,
                        "partition_role": role,
                        "records": stats[cell]["records"],
                        "states": stats[cell]["states"],
                        "transitions": stats[cell]["transitions"],
                        "rows_artifact_path": (
                            f"{plan['run_artifact_root']}/task_results/{task_id}/"
                            f"fragments/{lane}/{role}/{FRAGMENT_ROWS_FILENAME}"
                        ),
                        "rows_file_sha256": file_sha256(rows_path),
                        "membership_artifact_path": (
                            f"{plan['run_artifact_root']}/task_results/{task_id}/"
                            f"fragments/{lane}/{role}/{FRAGMENT_MEMBERSHIP_FILENAME}"
                        ),
                        "membership_file_sha256": file_sha256(membership_path),
                        "membership_stream_sha256": membership_stream_sha256,
                        "original_address_stream_sha256": stats[cell][
                            "original_digest"
                        ].hexdigest(),
                        "output_row_stream_sha256": stats[cell][
                            "output_digest"
                        ].hexdigest(),
                    }
                )
        receipt_body = {
            "schema": TASK_RECEIPT_SCHEMA,
            "schema_version": TASK_RECEIPT_SCHEMA_VERSION,
            "status": TASK_RECEIPT_STATUS,
            "training_authorized": False,
            "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
            "plan_sha256": plan["plan_sha256"],
            "task_identity_sha256": task_id,
            "task_identity_body": task["task_identity_body"],
            "selection_file_sha256": task["selection_file_sha256"],
            "selection_semantic_sha256": task["selection_semantic_sha256"],
            "fragments": fragments,
            "totals": {
                "records": sum(item["records"] for item in fragments),
                "states": sum(item["states"] for item in fragments),
                "transitions": sum(item["transitions"] for item in fragments),
                "fragments": len(fragments),
            },
        }
        receipt = _self_hashed(receipt_body, "receipt_sha256")
        _write_json(staging / TASK_RECEIPT_FILENAME, receipt)
        try:
            os.replace(staging, target)
            published = True
        except OSError:
            if not target.exists():
                raise
        validated = validate_role_lane_task_receipt(
            plan=plan,
            task=task,
            artifact_root=root,
        )
        _write_task_progress(
            plan=plan,
            task=task,
            artifact_root=root,
            phase="COMPLETE",
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            processed_candidates=processed_candidates,
            detail={"receipt_sha256": validated["receipt_sha256"]},
        )
        return validated
    except Exception as error:
        try:
            _write_task_progress(
                plan=plan,
                task=task,
                artifact_root=root,
                phase="FAILED",
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - started,
                processed_candidates=processed_candidates,
                detail={
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except OSError:
            pass
        raise
    finally:
        for raw, compressed, _ in handles.values():
            try:
                compressed.close()
            finally:
                raw.close()
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _expected_receipts(
    plan: Mapping[str, Any], *, artifact_root: Path
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    result_parent = _mounted_path(
        f"{plan['run_artifact_root']}/task_results",
        artifact_root=artifact_root,
        field="task result parent",
    )
    expected = set(plan["task_order"])
    observed = (
        {
            path.name
            for path in result_parent.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        if result_parent.is_dir()
        else set()
    )
    extra = observed - expected
    missing = expected - observed
    if extra:
        raise EditingV2RoleLaneMapReduceError(
            f"task result set contains unplanned receipts: {sorted(extra)}"
        )
    if missing:
        raise EditingV2RoleLaneMapReduceIncomplete(
            f"task result set is incomplete: {len(missing)} missing"
        )
    receipts: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_id in plan["task_order"]:
        task = _task_by_id(plan, task_id)
        receipt = validate_role_lane_task_receipt(
            plan=plan,
            task=task,
            artifact_root=artifact_root,
        )
        receipts.append((task, receipt))
    return receipts


def _load_task_receipt_header(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    """Validate receipt identity without rereading every fragment payload."""

    task_id = task["task_identity_sha256"]
    task_root = _mounted_path(
        f"{plan['run_artifact_root']}/task_results/{task_id}",
        artifact_root=artifact_root,
        field="task result root",
    )
    receipt = _require_self_hash(
        _load_json(task_root / TASK_RECEIPT_FILENAME, name="role/lane task receipt"),
        field="receipt_sha256",
        name="role/lane task receipt",
    )
    fragments = receipt.get("fragments")
    if (
        receipt.get("schema") != TASK_RECEIPT_SCHEMA
        or receipt.get("schema_version") != TASK_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != TASK_RECEIPT_STATUS
        or receipt.get("training_authorized") is not False
        or receipt.get("active8_admission_status") != ACTIVE8_ADMISSION_STATUS
        or receipt.get("plan_sha256") != plan.get("plan_sha256")
        or receipt.get("task_identity_sha256") != task_id
        or receipt.get("task_identity_body") != task.get("task_identity_body")
        or receipt.get("selection_file_sha256") != task.get("selection_file_sha256")
        or receipt.get("selection_semantic_sha256")
        != task.get("selection_semantic_sha256")
        or not isinstance(fragments, list)
        or not fragments
    ):
        raise EditingV2RoleLaneMapReduceError(
            f"task receipt header identity disagrees: {task_id}"
        )
    lane_order = tuple(
        plan["reference_materialization"]["run_identity_body"]["lane_order"]
    )
    allowed = {(lane, role) for lane in lane_order for role in PARTITION_ROLES}
    cells: set[tuple[str, str]] = set()
    totals = {"records": 0, "states": 0, "transitions": 0}
    for fragment in fragments:
        if not isinstance(fragment, Mapping):
            raise EditingV2RoleLaneMapReduceError("task fragment must be an object")
        cell = (fragment.get("data_lane"), fragment.get("partition_role"))
        if cell not in allowed or cell in cells:
            raise EditingV2RoleLaneMapReduceError(
                "task receipt has an unknown or duplicate cell fragment"
            )
        cells.add(cell)
        for field in totals:
            value = fragment.get(field)
            if type(value) is not int or value <= 0:
                raise EditingV2RoleLaneMapReduceError(
                    f"task fragment {field} must be positive"
                )
            totals[field] += value
    expected_totals = {
        **totals,
        "fragments": len(fragments),
    }
    if receipt.get("totals") != expected_totals or totals["records"] != task.get(
        "selected_candidates"
    ):
        raise EditingV2RoleLaneMapReduceError("task receipt header totals disagree")
    return receipt


def _task_receipt_headers(
    plan: Mapping[str, Any], *, artifact_root: Path
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Load the exact planned receipt headers in frozen source order."""

    result_parent = _mounted_path(
        f"{plan['run_artifact_root']}/task_results",
        artifact_root=artifact_root,
        field="task result parent",
    )
    expected = set(plan["task_order"])
    observed = (
        {
            path.name
            for path in result_parent.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        if result_parent.is_dir()
        else set()
    )
    if observed - expected:
        raise EditingV2RoleLaneMapReduceError(
            f"task result set contains unplanned receipts: {sorted(observed - expected)}"
        )
    if expected - observed:
        raise EditingV2RoleLaneMapReduceIncomplete(
            f"task result set is incomplete: {len(expected - observed)} missing"
        )
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_id in plan["task_order"]:
        task = _task_by_id(plan, task_id)
        result.append(
            (
                task,
                _load_task_receipt_header(
                    plan=plan,
                    task=task,
                    artifact_root=artifact_root,
                ),
            )
        )
    return result


def _cell_identity(
    *,
    plan: Mapping[str, Any],
    headers: list[tuple[dict[str, Any], dict[str, Any]]],
    lane: str,
    role: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs: list[dict[str, Any]] = []
    for task, receipt in headers:
        matches = [
            fragment
            for fragment in receipt["fragments"]
            if fragment["data_lane"] == lane and fragment["partition_role"] == role
        ]
        if len(matches) > 1:
            raise EditingV2RoleLaneMapReduceError(
                f"task duplicates cell fragment: {lane}/{role}"
            )
        if matches:
            inputs.append(
                {
                    "task_ordinal": task["task_ordinal"],
                    "task_identity_sha256": task["task_identity_sha256"],
                    "task_receipt_sha256": receipt["receipt_sha256"],
                    "fragment": dict(matches[0]),
                }
            )
    if not inputs:
        raise EditingV2RoleLaneMapReduceError(
            f"planned role/lane cell has no mapped records: {lane}/{role}"
        )
    body = {
        "plan_sha256": plan["plan_sha256"],
        "reference_run_identity_sha256": plan["reference_materialization"][
            "run_identity_sha256"
        ],
        "mapreduce_implementation_sha256": _implementation_sha256(),
        "data_lane": lane,
        "partition_role": role,
        "ordered_fragment_inventory_sha256": canonical_sha256(inputs),
        "ordered_fragments": inputs,
    }
    return body, inputs


def validate_role_lane_cell_receipt(
    *,
    plan: Mapping[str, Any],
    headers: list[tuple[dict[str, Any], dict[str, Any]]],
    lane: str,
    role: str,
    artifact_root: str | Path,
) -> dict[str, Any]:
    root = Path(artifact_root)
    identity_body, _ = _cell_identity(
        plan=plan,
        headers=headers,
        lane=lane,
        role=role,
    )
    cell_id = canonical_sha256(identity_body)
    cell_root = _mounted_path(
        f"{plan['run_artifact_root']}/cell_results/{lane}/{role}/{cell_id}",
        artifact_root=root,
        field="cell result root",
    )
    receipt = _require_self_hash(
        _load_json(cell_root / CELL_RECEIPT_FILENAME, name="role/lane cell receipt"),
        field="receipt_sha256",
        name="role/lane cell receipt",
    )
    if (
        receipt.get("schema") != CELL_RECEIPT_SCHEMA
        or receipt.get("schema_version") != CELL_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != CELL_RECEIPT_STATUS
        or receipt.get("training_authorized") is not False
        or receipt.get("active8_admission_status") != ACTIVE8_ADMISSION_STATUS
        or receipt.get("cell_identity_body") != identity_body
        or receipt.get("cell_identity_sha256") != cell_id
    ):
        raise EditingV2RoleLaneMapReduceError(
            f"cell receipt identity disagrees: {lane}/{role}"
        )
    rows_path = _mounted_path(
        receipt.get("rows_artifact_path"),
        artifact_root=root,
        field="cell rows artifact",
    )
    membership_path = _mounted_path(
        receipt.get("membership_artifact_path"),
        artifact_root=root,
        field="cell membership artifact",
    )
    if (
        not rows_path.is_relative_to(cell_root)
        or not membership_path.is_relative_to(cell_root)
        or not rows_path.is_file()
        or file_sha256(rows_path) != receipt.get("rows_file_sha256")
        or not membership_path.is_file()
        or file_sha256(membership_path) != receipt.get("membership_file_sha256")
    ):
        raise EditingV2RoleLaneMapReduceError(
            f"cell output is missing or tampered: {lane}/{role}"
        )
    count, semantic = _stream_file_identity(
        membership_path,
        name="cell membership",
    )
    if count != receipt.get("records") or semantic != receipt.get(
        "membership_stream_sha256"
    ):
        raise EditingV2RoleLaneMapReduceError(
            f"cell membership stream disagrees: {lane}/{role}"
        )
    return receipt


def reduce_role_lane_cell(
    *,
    plan_path: str | Path,
    lane: str,
    role: str,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Reduce one lane/role cell into an independently reusable immutable result."""

    root = Path(artifact_root)
    plan = validate_role_lane_mapreduce_plan(plan_path, artifact_root=root)
    started_at = _utc_now()
    started = time.perf_counter()
    _write_cell_progress(
        plan=plan,
        artifact_root=root,
        lane=lane,
        role=role,
        phase="STARTED",
        started_at=started_at,
        elapsed_seconds=0.0,
        processed_records=0,
    )
    lane_order = tuple(
        plan["reference_materialization"]["run_identity_body"]["lane_order"]
    )
    if lane not in lane_order or role not in PARTITION_ROLES:
        raise EditingV2RoleLaneMapReduceError(f"unknown role/lane cell: {lane}/{role}")
    headers = _task_receipt_headers(plan, artifact_root=root)
    identity_body, inputs = _cell_identity(
        plan=plan,
        headers=headers,
        lane=lane,
        role=role,
    )
    cell_id = canonical_sha256(identity_body)
    target = _mounted_path(
        f"{plan['run_artifact_root']}/cell_results/{lane}/{role}/{cell_id}",
        artifact_root=root,
        field="cell result root",
    )
    if target.exists():
        receipt = validate_role_lane_cell_receipt(
            plan=plan,
            headers=headers,
            lane=lane,
            role=role,
            artifact_root=root,
        )
        _write_cell_progress(
            plan=plan,
            artifact_root=root,
            lane=lane,
            role=role,
            phase="REUSED_COMPLETE",
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            processed_records=receipt["records"],
            detail={"receipt_sha256": receipt["receipt_sha256"]},
        )
        return receipt
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{cell_id}.",
            suffix=".staging",
        )
    )
    rows_path = staging / reference._CELL_SHARD_NAME
    membership_path = staging / CELL_MEMBERSHIP_FILENAME
    original_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    membership_digest = hashlib.sha256()
    records = 0
    states = 0
    transitions = 0
    published = False
    try:
        with (
            rows_path.open("wb") as raw_output,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_output, mtime=0
            ) as output,
            membership_path.open("wb") as membership_output,
        ):
            for item in inputs:
                fragment = item["fragment"]
                task_id = item["task_identity_sha256"]
                task_root = _mounted_path(
                    f"{plan['run_artifact_root']}/task_results/{task_id}",
                    artifact_root=root,
                    field="task result root",
                )
                rows_input = _mounted_path(
                    fragment["rows_artifact_path"],
                    artifact_root=root,
                    field="fragment rows",
                )
                membership_input = _mounted_path(
                    fragment["membership_artifact_path"],
                    artifact_root=root,
                    field="fragment membership",
                )
                if (
                    not rows_input.is_relative_to(task_root)
                    or not membership_input.is_relative_to(task_root)
                    or file_sha256(rows_input) != fragment["rows_file_sha256"]
                    or file_sha256(membership_input)
                    != fragment["membership_file_sha256"]
                ):
                    raise EditingV2RoleLaneMapReduceError(
                        f"mapped fragment is absent or tampered: {lane}/{role}/{task_id}"
                    )
                members = list(
                    _iter_jsonl(membership_input, name="fragment membership")
                )
                fragment_original = hashlib.sha256()
                fragment_output = hashlib.sha256()
                fragment_membership = hashlib.sha256()
                fragment_records = 0
                with gzip.open(rows_input, "rb") as rows:
                    for local_index, (raw, member) in enumerate(
                        zip(rows, members, strict=True)
                    ):
                        row = json.loads(raw)
                        encoded = _canonical_bytes(row)
                        output_row_sha256 = hashlib.sha256(encoded).hexdigest()
                        address_sha256 = member["original_packed_address"][
                            "packed_address"
                        ]["address_sha256"]
                        if (
                            raw != encoded + b"\n"
                            or member.get("local_entry_index") != local_index
                            or member.get("data_lane") != lane
                            or member.get("partition_role") != role
                            or member.get("output_packed_row_sha256")
                            != output_row_sha256
                            or row.get("trace", {}).get("layer") != lane
                            or row.get("trace", {}).get("partition") != role
                        ):
                            raise EditingV2RoleLaneMapReduceError(
                                f"mapped row/membership disagrees: {lane}/{role}/{task_id}"
                            )
                        output.write(encoded + b"\n")
                        member_bytes = _canonical_bytes(member, newline=True)
                        membership_output.write(member_bytes)
                        original_digest.update(address_sha256.encode("ascii"))
                        original_digest.update(b"\n")
                        output_digest.update(output_row_sha256.encode("ascii"))
                        output_digest.update(b"\n")
                        membership_digest.update(member_bytes)
                        fragment_original.update(address_sha256.encode("ascii"))
                        fragment_original.update(b"\n")
                        fragment_output.update(output_row_sha256.encode("ascii"))
                        fragment_output.update(b"\n")
                        fragment_membership.update(member_bytes)
                        fragment_records += 1
                        records += 1
                        states += int(member["state_count"])
                        transitions += int(member["path_length"])
                if (
                    fragment_records != fragment["records"]
                    or fragment_original.hexdigest()
                    != fragment["original_address_stream_sha256"]
                    or fragment_output.hexdigest()
                    != fragment["output_row_stream_sha256"]
                    or fragment_membership.hexdigest()
                    != fragment["membership_stream_sha256"]
                ):
                    raise EditingV2RoleLaneMapReduceError(
                        f"mapped fragment stream identity disagrees: {lane}/{role}/{task_id}"
                    )
                _write_cell_progress(
                    plan=plan,
                    artifact_root=root,
                    lane=lane,
                    role=role,
                    phase="REDUCING",
                    started_at=started_at,
                    elapsed_seconds=time.perf_counter() - started,
                    processed_records=records,
                    detail={"last_task_identity_sha256": task_id},
                )
            membership_output.flush()
            os.fsync(membership_output.fileno())
        with rows_path.open("rb") as rows_handle:
            os.fsync(rows_handle.fileno())
        receipt_body = {
            "schema": CELL_RECEIPT_SCHEMA,
            "schema_version": CELL_RECEIPT_SCHEMA_VERSION,
            "status": CELL_RECEIPT_STATUS,
            "training_authorized": False,
            "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
            "cell_identity_body": identity_body,
            "cell_identity_sha256": cell_id,
            "records": records,
            "states": states,
            "transitions": transitions,
            "rows_artifact_path": (
                f"{plan['run_artifact_root']}/cell_results/{lane}/{role}/{cell_id}/"
                f"{reference._CELL_SHARD_NAME}"
            ),
            "rows_file_sha256": file_sha256(rows_path),
            "membership_artifact_path": (
                f"{plan['run_artifact_root']}/cell_results/{lane}/{role}/{cell_id}/"
                f"{CELL_MEMBERSHIP_FILENAME}"
            ),
            "membership_file_sha256": file_sha256(membership_path),
            "membership_stream_sha256": membership_digest.hexdigest(),
            "original_address_stream_sha256": original_digest.hexdigest(),
            "output_row_stream_sha256": output_digest.hexdigest(),
        }
        receipt = _self_hashed(receipt_body, "receipt_sha256")
        _write_json(staging / CELL_RECEIPT_FILENAME, receipt)
        try:
            os.replace(staging, target)
            published = True
        except OSError:
            if not target.exists():
                raise
        validated = validate_role_lane_cell_receipt(
            plan=plan,
            headers=headers,
            lane=lane,
            role=role,
            artifact_root=root,
        )
        _write_cell_progress(
            plan=plan,
            artifact_root=root,
            lane=lane,
            role=role,
            phase="COMPLETE",
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - started,
            processed_records=records,
            detail={"receipt_sha256": validated["receipt_sha256"]},
        )
        return validated
    except Exception as error:
        try:
            _write_cell_progress(
                plan=plan,
                artifact_root=root,
                lane=lane,
                role=role,
                phase="FAILED",
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - started,
                processed_records=records,
                detail={
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except OSError:
            pass
        raise
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _expected_cell_receipts(plan: Mapping[str, Any], *, artifact_root: Path) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    dict[tuple[str, str], dict[str, Any]],
]:
    headers = _task_receipt_headers(plan, artifact_root=artifact_root)
    lane_order = tuple(
        plan["reference_materialization"]["run_identity_body"]["lane_order"]
    )
    receipts: dict[tuple[str, str], dict[str, Any]] = {}
    for lane in lane_order:
        for role in PARTITION_ROLES:
            identity_body, _ = _cell_identity(
                plan=plan,
                headers=headers,
                lane=lane,
                role=role,
            )
            expected_id = canonical_sha256(identity_body)
            parent = _mounted_path(
                f"{plan['run_artifact_root']}/cell_results/{lane}/{role}",
                artifact_root=artifact_root,
                field="cell result parent",
            )
            observed = (
                {
                    path.name
                    for path in parent.iterdir()
                    if path.is_dir() and not path.name.startswith(".")
                }
                if parent.is_dir()
                else set()
            )
            if observed - {expected_id}:
                raise EditingV2RoleLaneMapReduceError(
                    f"cell result contains stale/duplicate outputs: {lane}/{role}"
                )
            if expected_id not in observed:
                raise EditingV2RoleLaneMapReduceIncomplete(
                    f"cell result is incomplete: {lane}/{role}"
                )
            receipts[(lane, role)] = validate_role_lane_cell_receipt(
                plan=plan,
                headers=headers,
                lane=lane,
                role=role,
                artifact_root=artifact_root,
            )
    return headers, receipts


def _publish_reduced_reference(
    *,
    plan: Mapping[str, Any],
    cell_receipts: Mapping[tuple[str, str], Mapping[str, Any]],
    artifact_root: Path,
) -> dict[str, Any]:
    identity = plan["reference_materialization"]
    run_body = identity["run_identity_body"]
    run_artifact_root = identity["run_artifact_root"]
    target = _mounted_path(
        run_artifact_root,
        artifact_root=artifact_root,
        field="reference output root",
    )
    if target.exists():
        from compose_v4.data.editing_v2_refined_role_lane_continuation import (
            validate_role_lane_output,
        )

        manifest, _, _ = validate_role_lane_output(
            role_root=target,
            candidate_root=Path(identity["candidate_root"]),
            split_assignment_path=Path(identity["split_assignment_path"]),
            editing_corpus_contract_path=Path(identity["editing_corpus_contract_path"]),
            artifact_root=artifact_root,
            expected_output_prefix=run_body["output_artifact_prefix"],
            expected_code_revision=run_body["code_revision"],
            expected_candidate_identity=run_body["candidate_materialization"],
            expected_source_stream=run_body["candidate_provenance_source_stream"],
        )
        return manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{identity['run_identity_sha256']}.",
            suffix=".staging",
        )
    )
    descriptor, database_name = tempfile.mkstemp(
        prefix="compose-editing-v2-role-lane-reduce-", suffix=".sqlite3"
    )
    os.close(descriptor)
    connection = reference._database(Path(database_name))
    lane_order = tuple(run_body["lane_order"])
    writers: dict[tuple[str, str], Any] = {}
    seen_candidates: set[str] = set()
    seen_addresses: set[str] = set()
    published = False
    try:
        for lane in lane_order:
            for role in PARTITION_ROLES:
                cell = (lane, role)
                receipt = cell_receipts.get(cell)
                if receipt is None:
                    raise EditingV2RoleLaneMapReduceIncomplete(
                        f"final assembly lacks cell receipt: {lane}/{role}"
                    )
                source_rows = _mounted_path(
                    receipt["rows_artifact_path"],
                    artifact_root=artifact_root,
                    field="cell rows",
                )
                provisional = (
                    staging / "_cells" / lane / role / reference._CELL_SHARD_NAME
                )
                provisional.parent.mkdir(parents=True, exist_ok=True)
                # Preserve the reusable cell artifact while remaining compatible
                # with legacy Modal Volumes that do not support hard links.
                _copy_verified_bytes(
                    source_rows,
                    provisional,
                    expected_sha256=receipt["rows_file_sha256"],
                )
                writers[cell] = SimpleNamespace(
                    lane=lane,
                    role=role,
                    provisional_path=provisional,
                    records=receipt["records"],
                    states=receipt["states"],
                    transitions=receipt["transitions"],
                    original_address_digest=SimpleNamespace(
                        hexdigest=lambda value=receipt[
                            "original_address_stream_sha256"
                        ]: value
                    ),
                    output_row_digest=SimpleNamespace(
                        hexdigest=lambda value=receipt[
                            "output_row_stream_sha256"
                        ]: value
                    ),
                )
                membership_path = _mounted_path(
                    receipt["membership_artifact_path"],
                    artifact_root=artifact_root,
                    field="cell memberships",
                )
                for output_index, member in enumerate(
                    _iter_jsonl(membership_path, name="cell membership")
                ):
                    candidate_id = member["candidate_id"]
                    original_address = member["original_packed_address"]
                    address_sha256 = original_address["packed_address"][
                        "address_sha256"
                    ]
                    if (
                        candidate_id in seen_candidates
                        or address_sha256 in seen_addresses
                    ):
                        raise EditingV2RoleLaneMapReduceError(
                            "final assembly observed duplicate candidate/address membership"
                        )
                    seen_candidates.add(candidate_id)
                    seen_addresses.add(address_sha256)
                    try:
                        connection.execute(
                            """
                            INSERT INTO memberships VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                lane,
                                role,
                                output_index,
                                candidate_id,
                                canonical_json_bytes(original_address).decode("utf-8"),
                                member["original_packed_row_sha256"],
                                member["output_packed_row_sha256"],
                                member["trace_id"],
                                member["source_key"],
                                member["target_key"],
                                member["path_length"],
                            ),
                        )
                    except sqlite3.IntegrityError as error:
                        raise EditingV2RoleLaneMapReduceError(
                            "final membership uniqueness failed"
                        ) from error
        connection.commit()
        if len(seen_candidates) != plan["selected_candidates"]:
            raise EditingV2RoleLaneMapReduceError(
                "reducer did not consume every selected candidate"
            )
        contract_path = Path(identity["editing_corpus_contract_path"])
        contract = load_editing_corpus_contract(contract_path)
        lane_definitions = editing_v2_lane_definitions(contract)
        registry_lanes, published_cells = reference._publish_lane_files(
            staging,
            run_artifact_root=run_artifact_root,
            lane_definitions=lane_definitions,
            writers=writers,
            code_revision=run_body["code_revision"],
            implementation_sha256=run_body["materializer_implementation_sha256"],
            candidate_identity=run_body["candidate_materialization"],
            candidate_provenance_source_stream=run_body[
                "candidate_provenance_source_stream"
            ],
            split_identity=run_body["split_assignment"],
            contract_identity=run_body["editing_corpus_contract"],
        )
        registry_body = {
            "schema": LANE_REGISTRY_SCHEMA,
            "schema_version": LANE_REGISTRY_SCHEMA_VERSION,
            "status": LANE_REGISTRY_STATUS,
            "training_authorized": False,
            "editing_corpus_contract": run_body["editing_corpus_contract"],
            "partition_roles": list(PARTITION_ROLES),
            "lanes": registry_lanes,
        }
        registry = {
            **registry_body,
            "registry_sha256": canonical_sha256(registry_body),
        }
        registry_path = staging / reference.LANE_REGISTRY_FILENAME
        reference._write_json(registry_path, registry)
        completion_file_sha256: dict[str, str] = {}
        completion_sha256: dict[str, str] = {}
        inventory_sha256: dict[str, str] = {}
        for lane in registry_lanes:
            local_completion = staging / PurePosixPath(
                lane["completion_manifest_path"]
            ).relative_to(PurePosixPath(run_artifact_root))
            completion = json.loads(local_completion.read_bytes())
            lane_id = lane["lane_id"]
            completion_file_sha256[lane_id] = file_sha256(local_completion)
            completion_sha256[lane_id] = completion["completion_sha256"]
            inventory_sha256[lane_id] = completion["shard_inventory_sha256"]
        receipt_fixed = {
            "schema": RESOLVED_PACKED_MEMBERSHIP_SCHEMA,
            "schema_version": RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION,
            "status": RESOLVED_PACKED_MEMBERSHIP_STATUS,
            "training_authorized": False,
            "candidate_materialization_authority": CANDIDATE_AUTHORITY,
            "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
            "membership_complete": True,
            "blockers": list(REQUIRED_BLOCKERS),
            "inputs": {
                "candidate_materialization": run_body["candidate_materialization"],
                "split_assignment": run_body["split_assignment"],
                "lane_registry": {
                    "registry_file_sha256": file_sha256(registry_path),
                    "registry_sha256": registry["registry_sha256"],
                    "editing_corpus_contract_file_sha256": run_body[
                        "editing_corpus_contract"
                    ]["file_sha256"],
                    "editing_corpus_contract_id": run_body["editing_corpus_contract"][
                        "contract_id"
                    ],
                    "lane_completion_manifest_file_sha256": completion_file_sha256,
                    "lane_completion_sha256": completion_sha256,
                    "lane_shard_inventory_sha256": inventory_sha256,
                },
            },
        }
        receipt_path = staging / reference.RESOLVED_MEMBERSHIP_FILENAME
        receipt_sha256, receipt_file_sha256 = reference._write_membership_receipt(
            receipt_path,
            fixed_fields=receipt_fixed,
            lane_order=lane_order,
            published_cells=published_cells,
            connection=connection,
        )
        totals = {
            "routed_candidates": plan["selected_candidates"],
            "output_shards": len(published_cells),
            "output_records": sum(
                int(cell["records"]) for cell in published_cells.values()
            ),
            "output_states": sum(
                int(cell["states"]) for cell in published_cells.values()
            ),
            "output_transitions": sum(
                int(cell["transitions"]) for cell in published_cells.values()
            ),
        }
        materialization_body = {
            **run_body,
            "status": reference.ROLE_LANE_MATERIALIZATION_STATUS,
            "training_authorized": False,
            "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
            "run_identity_sha256": identity["run_identity_sha256"],
            "run_artifact_root": run_artifact_root,
            "selection_policy": "mechanical_application_of_frozen_split_no_metric_selection",
            "final_test_selection_use": "forbidden_not_performed",
            "lane_registry": {
                "relative_path": reference.LANE_REGISTRY_FILENAME,
                "file_sha256": file_sha256(registry_path),
                "registry_sha256": registry["registry_sha256"],
            },
            "resolved_packed_membership": {
                "relative_path": reference.RESOLVED_MEMBERSHIP_FILENAME,
                "file_sha256": receipt_file_sha256,
                "receipt_sha256": receipt_sha256,
                "schema_version": RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION,
            },
            "totals": totals,
            "atomic_publication": "same_parent_directory_rename",
        }
        materialization = _self_hashed(materialization_body, "manifest_sha256")
        reference._write_json(
            staging / reference.ROLE_LANE_MATERIALIZATION_FILENAME,
            materialization,
        )
        shutil.rmtree(staging / "_cells")
        os.replace(staging, target)
        published = True
        return materialization
    finally:
        connection.close()
        Path(database_name).unlink(missing_ok=True)
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def reduce_role_lane_mapreduce(
    *,
    plan_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Fail closed on the exact receipt set and publish the reference result."""

    root = Path(artifact_root)
    plan = validate_role_lane_mapreduce_plan(plan_path, artifact_root=root)
    _, cell_receipts = _expected_cell_receipts(plan, artifact_root=root)
    return _publish_reduced_reference(
        plan=plan,
        cell_receipts=cell_receipts,
        artifact_root=root,
    )


__all__ = [
    "CELL_RECEIPT_FILENAME",
    "CELL_RECEIPT_SCHEMA",
    "CELL_RECEIPT_SCHEMA_VERSION",
    "CELL_RECEIPT_STATUS",
    "DEFAULT_EXECUTION_ARTIFACT_PREFIX",
    "DEFAULT_EXPECTED_SOURCE_SHARDS",
    "FRAGMENT_MEMBERSHIP_FILENAME",
    "FRAGMENT_ROWS_FILENAME",
    "PLAN_FILENAME",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "PLAN_STATUS",
    "TASK_INPUT_FILENAME",
    "TASK_RECEIPT_FILENAME",
    "TASK_RECEIPT_SCHEMA",
    "TASK_RECEIPT_SCHEMA_VERSION",
    "TASK_RECEIPT_STATUS",
    "EditingV2RoleLaneMapReduceError",
    "EditingV2RoleLaneMapReduceIncomplete",
    "map_role_lane_source_shard",
    "plan_role_lane_mapreduce",
    "reduce_role_lane_cell",
    "reduce_role_lane_mapreduce",
    "validate_role_lane_cell_receipt",
    "validate_role_lane_mapreduce_plan",
    "validate_role_lane_task_receipt",
]
