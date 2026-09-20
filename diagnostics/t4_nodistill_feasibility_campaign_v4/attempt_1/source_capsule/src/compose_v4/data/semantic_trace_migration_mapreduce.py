"""Restart-safe map/reduce orchestration for semantic Editing V2 migration.

The planner consumes the physically verified five-lane by four-role packed
corpus returned by :func:`resolve_editing_v2_active8_sources`.  It creates one
immutable task per physical shard.  Workers may finish in any order and may be
retried because each task publishes into its own content-addressed directory.
The reducer trusts only the exact on-disk receipts named by the plan.

This module creates corpus derivatives only.  Neither a complete plan nor a
complete reduction grants Gate 0, T1, P50, checkpoint-selection, or training
authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_v2_active8_source_adapter import (
    ResolvedEditingV2Active8Sources,
)
from compose_v4.data.editing_v2_split_assignment import (
    EditingV2SplitAssignmentError,
    validate_candidate_source_stream,
)
from compose_v4.data.semantic_packed_trace_store import (
    semantic_packed_builder_identity,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    SemanticTraceMaterializationError,
    SemanticTraceMigrationTask,
    materialize_frozen_packed_shard,
    validate_semantic_trace_materialization,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_v2_process_identity,
)

PLAN_SCHEMA = "compose.data.semantic_trace_migration_plan"
PLAN_SCHEMA_VERSION = 1
PLAN_STATUS = "FROZEN_COMPLETE_NO_TRAINING_AUTHORITY"
SOURCE_REVISION_SCHEMA = "compose.data.semantic_trace_migration_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
COMPLETION_SCHEMA = "compose.data.semantic_trace_migration_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_STATUS = "COMPLETE_GATE0_NOT_RUN_NO_TRAINING_AUTHORITY"

PLAN_FILENAME = "SEMANTIC_MIGRATION_PLAN.json"
COMPLETION_FILENAME = "SEMANTIC_MIGRATION_COMPLETE.json"
DEFAULT_OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/semantic_migration"
EXPECTED_TASK_COUNT = len(REQUIRED_DATA_LANES) * len(REQUIRED_PARTITION_ROLES)

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_SOURCE_FILES = (
    "modal_apps/materialize_editing_v2_semantic_corpus_app.py",
    "src/compose_v4/data/semantic_trace_migration_mapreduce.py",
    "src/compose_v4/data/semantic_trace_migration_materializer.py",
    "src/compose_v4/data/semantic_packed_trace_store.py",
    "src/compose_v4/data/editing_v2_active8_source_adapter.py",
    "src/compose_v4/data/editing_v2_candidate_provenance_bridge.py",
    "src/compose_v4/data/editing_v2_lane_registry.py",
    "src/compose_v4/data/editing_v2_split_assignment.py",
    "src/compose_v4/data/editing_v2_split_census.py",
    "src/compose_v4/rewrite/semantic_trace_migration.py",
    "src/compose_v4/rewrite/editing_v2_process_identity.py",
    "src/compose_v4/rewrite/trace_shard_v3.py",
    "src/compose_v4/rewrite/action_codec_v4.py",
)
_SOURCE_BINDING_FIELDS = {
    "data_lane",
    "split",
    "source_artifact_path",
    "source_shard_sha256",
    "source_manifest_artifact_path",
    "source_manifest_sha256",
    "source_overlay_artifact_path",
    "source_overlay_sha256",
    "source_entry_count",
}
_TASK_FIELDS = _SOURCE_BINDING_FIELDS | {
    "task_identity_sha256",
    "output_artifact_path",
}
_UPSTREAM_FIELDS = {
    "candidate_materialization_manifest_sha256",
    "candidate_provenance_source_stream",
    "split_assignment_sha256",
    "lane_registry_sha256",
    "membership_receipt_file_sha256",
    "membership_receipt_sha256",
}


class SemanticTraceMigrationMapReduceError(RuntimeError):
    """The migration plan, task set, or reduction is incomplete or stale."""


class SemanticTraceMigrationIncomplete(SemanticTraceMigrationMapReduceError):
    """At least one exact planned task has not published a complete result."""


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or _HEX_RE.fullmatch(value) is None:
        raise SemanticTraceMigrationMapReduceError(f"{field} must be a lowercase SHA-256")
    return value


def _require_git_object(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) not in {40, 64} or _HEX_RE.fullmatch(value) is None:
        raise SemanticTraceMigrationMapReduceError(
            f"{field} must be a lowercase Git-style object identity"
        )
    return value


def _require_artifact_path(value: object, *, field: str) -> str:
    raw = value if isinstance(value, str) else ""
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or not path.is_absolute()
        or len(path.parts) < 3
        or path.parts[1] != "artifacts"
        or ".." in path.parts
        or str(path) != raw
        or raw.endswith("/")
    ):
        raise SemanticTraceMigrationMapReduceError(
            f"{field} must be a normalized path below /artifacts"
        )
    return raw


def _artifact_prefix(value: str) -> str:
    return _require_artifact_path(value, field="output_artifact_prefix")


def _mounted_artifact_path(
    artifact_path: str,
    *,
    artifact_root: Path,
    field: str,
) -> Path:
    normalized = _require_artifact_path(artifact_path, field=field)
    root = Path(artifact_root).resolve()
    resolved = (root / PurePosixPath(normalized).relative_to("/artifacts")).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise SemanticTraceMigrationMapReduceError(
            f"{field} resolves outside the artifact root"
        ) from error
    return resolved


def _implementation_files(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root)
    files: dict[str, str] = {}
    for relative in _SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise SemanticTraceMigrationMapReduceError(
                f"semantic migration implementation source is absent: {source}"
            )
        files[relative] = _file_sha256(source)
    return files


def build_semantic_migration_source_revision(
    *,
    commit: str,
    tree: str,
    repo_root: Path,
    worktree_clean: bool,
) -> dict[str, Any]:
    """Bind the complete serialized implementation to one Git revision."""

    commit = _require_git_object(commit, field="source revision commit")
    tree = _require_git_object(tree, field="source revision tree")
    if worktree_clean is not True:
        raise SemanticTraceMigrationMapReduceError(
            "semantic migration requires a clean committed worktree"
        )
    process_identity = editing_v2_process_identity()
    builder_identity = semantic_packed_builder_identity()
    implementation_files = _implementation_files(Path(repo_root))
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "implementation_files": implementation_files,
        "implementation_files_sha256": _canonical_sha256(implementation_files),
        "process_identity_sha256": process_identity["process_identity_sha256"],
        "builder_identity_sha256": builder_identity["identity_sha256"],
    }
    return {**body, "source_revision_sha256": _canonical_sha256(body)}


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SemanticTraceMigrationMapReduceError(
            f"cannot establish semantic migration Git identity: git {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def repository_semantic_migration_source_revision(*, repo_root: Path) -> dict[str, Any]:
    """Return a revision only when every serialized file belongs to clean HEAD."""

    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SemanticTraceMigrationMapReduceError(
            "semantic migration refuses a dirty serialized-code tree"
        )
    for relative in _SOURCE_FILES:
        head_blob = _git(root, "rev-parse", f"HEAD:{relative}")
        working_blob = _git(root, "hash-object", "--", relative)
        if head_blob != working_blob:
            raise SemanticTraceMigrationMapReduceError(
                f"semantic migration source is off the bound revision: {relative}"
            )
    return build_semantic_migration_source_revision(
        commit=commit,
        tree=tree,
        repo_root=root,
        worktree_clean=True,
    )


def validate_semantic_migration_source_revision(
    value: object,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate a passed clean-tree binding against the live serialized files."""

    if not isinstance(value, Mapping):
        raise SemanticTraceMigrationMapReduceError("source revision must be an object")
    payload = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "worktree_clean",
        "implementation_files",
        "implementation_files_sha256",
        "process_identity_sha256",
        "builder_identity_sha256",
        "source_revision_sha256",
    }
    if set(payload) != expected_fields:
        raise SemanticTraceMigrationMapReduceError(
            "semantic migration source revision fields disagree"
        )
    _require_git_object(payload["commit"], field="source revision commit")
    _require_git_object(payload["tree"], field="source revision tree")
    live_files = _implementation_files(Path(repo_root))
    process_identity = editing_v2_process_identity()
    builder_identity = semantic_packed_builder_identity()
    body = {key: item for key, item in payload.items() if key != "source_revision_sha256"}
    if (
        payload["schema"] != SOURCE_REVISION_SCHEMA
        or payload["schema_version"] != SOURCE_REVISION_SCHEMA_VERSION
        or payload["worktree_clean"] is not True
        or payload["implementation_files"] != live_files
        or payload["implementation_files_sha256"] != _canonical_sha256(live_files)
        or payload["process_identity_sha256"] != process_identity["process_identity_sha256"]
        or payload["builder_identity_sha256"] != builder_identity["identity_sha256"]
        or payload["source_revision_sha256"] != _canonical_sha256(body)
    ):
        raise SemanticTraceMigrationMapReduceError(
            "semantic migration source revision is stale or malformed"
        )
    return payload


def _source_inventory(
    resolved: ResolvedEditingV2Active8Sources,
) -> list[dict[str, Any]]:
    expected_cells = [
        (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
    ]
    observed_cells: list[tuple[str, str]] = []
    inventory: list[dict[str, Any]] = []
    seen_artifacts: set[str] = set()
    seen_digests: set[str] = set()
    for index, binding in enumerate(resolved.bindings):
        source = binding.source
        cell = (source.manifest_layer, source.partition)
        observed_cells.append(cell)
        if source.envelope_layer != source.manifest_layer:
            raise SemanticTraceMigrationMapReduceError(
                f"resolved shard {index} has different manifest and envelope lanes"
            )
        artifact_path = _require_artifact_path(
            binding.artifact_path,
            field=f"resolved shard {index}.artifact_path",
        )
        manifest_artifact_path = _require_artifact_path(
            binding.packed_manifest_artifact_path,
            field=f"resolved shard {index}.packed_manifest_artifact_path",
        )
        overlay_artifact_path = binding.packed_provenance_overlay_artifact_path
        if overlay_artifact_path is not None:
            overlay_artifact_path = _require_artifact_path(
                overlay_artifact_path,
                field=f"resolved shard {index}.packed_provenance_overlay_artifact_path",
            )
        shard_sha256 = _require_sha256(
            binding.packed_shard_file_sha256,
            field=f"resolved shard {index}.packed_shard_file_sha256",
        )
        if artifact_path in seen_artifacts or shard_sha256 in seen_digests:
            raise SemanticTraceMigrationMapReduceError(
                "resolved migration sources repeat a physical path or shard digest"
            )
        seen_artifacts.add(artifact_path)
        seen_digests.add(shard_sha256)
        if not binding.candidate_ids:
            raise SemanticTraceMigrationMapReduceError(
                f"resolved shard {index} has no exact source entries"
            )
        inventory.append(
            {
                "data_lane": source.manifest_layer,
                "split": source.partition,
                "source_artifact_path": artifact_path,
                "source_shard_sha256": shard_sha256,
                "source_manifest_artifact_path": manifest_artifact_path,
                "source_manifest_sha256": _require_sha256(
                    binding.packed_manifest_file_sha256,
                    field=f"resolved shard {index}.packed_manifest_file_sha256",
                ),
                "source_overlay_artifact_path": overlay_artifact_path,
                "source_overlay_sha256": (
                    _require_sha256(
                        binding.packed_provenance_overlay_file_sha256,
                        field=(f"resolved shard {index}.packed_provenance_overlay_file_sha256"),
                    )
                    if binding.packed_provenance_overlay_file_sha256 is not None
                    else None
                ),
                "source_entry_count": len(binding.candidate_ids),
            }
        )
    if observed_cells != expected_cells or len(inventory) != EXPECTED_TASK_COUNT:
        raise SemanticTraceMigrationMapReduceError(
            "semantic migration requires exactly the ordered five-lane by four-role shard grid"
        )
    return inventory


def plan_semantic_trace_migration(
    resolved: ResolvedEditingV2Active8Sources,
    *,
    source_revision: Mapping[str, Any],
    repo_root: Path,
    output_artifact_prefix: str = DEFAULT_OUTPUT_ARTIFACT_PREFIX,
) -> dict[str, Any]:
    """Build a deterministic task manifest over exactly 20 verified shards."""

    revision = validate_semantic_migration_source_revision(
        source_revision,
        repo_root=repo_root,
    )
    prefix = _artifact_prefix(output_artifact_prefix)
    inventory = _source_inventory(resolved)
    source_inventory_sha256 = _canonical_sha256(inventory)
    try:
        candidate_source_stream = validate_candidate_source_stream(
            resolved.candidate_provenance_source_stream,
            expected_nonempty_rows=sum(source["source_entry_count"] for source in inventory),
        )
    except EditingV2SplitAssignmentError as error:
        raise SemanticTraceMigrationMapReduceError(
            "semantic migration candidate provenance source stream is invalid"
        ) from error
    if (
        candidate_source_stream["candidate_materialization"]["manifest_sha256"]
        != resolved.candidate_materialization_manifest_sha256
    ):
        raise SemanticTraceMigrationMapReduceError(
            "semantic migration candidate materialization and source stream disagree"
        )
    upstream = {
        "candidate_materialization_manifest_sha256": _require_sha256(
            resolved.candidate_materialization_manifest_sha256,
            field="candidate materialization manifest SHA-256",
        ),
        "candidate_provenance_source_stream": candidate_source_stream,
        "split_assignment_sha256": _require_sha256(
            resolved.split_assignment_sha256,
            field="split assignment SHA-256",
        ),
        "lane_registry_sha256": _require_sha256(
            resolved.lane_registry_sha256,
            field="lane registry SHA-256",
        ),
        "membership_receipt_file_sha256": _require_sha256(
            resolved.membership_receipt_file_sha256,
            field="membership receipt file SHA-256",
        ),
        "membership_receipt_sha256": _require_sha256(
            resolved.membership_receipt_sha256,
            field="membership receipt SHA-256",
        ),
    }
    process_identity = editing_v2_process_identity()
    builder_identity = semantic_packed_builder_identity()
    run_identity_body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_revision_sha256": revision["source_revision_sha256"],
        "upstream": upstream,
        "source_inventory_sha256": source_inventory_sha256,
        "process_identity_sha256": process_identity["process_identity_sha256"],
        "builder_identity_sha256": builder_identity["identity_sha256"],
        "output_artifact_prefix": prefix,
    }
    run_identity_sha256 = _canonical_sha256(run_identity_body)
    run_artifact_root = f"{prefix}/{run_identity_sha256}"
    tasks: list[dict[str, Any]] = []
    for source in inventory:
        identity_body = {
            "run_identity_sha256": run_identity_sha256,
            "source": source,
        }
        task_identity_sha256 = _canonical_sha256(identity_body)
        tasks.append(
            {
                **source,
                "task_identity_sha256": task_identity_sha256,
                "output_artifact_path": (f"{run_artifact_root}/tasks/{task_identity_sha256}"),
            }
        )
    body: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PLAN_STATUS,
        "training_authorized": False,
        "implementation_revision": revision["commit"],
        "source_revision": revision,
        "upstream": upstream,
        "process_identity": process_identity,
        "builder_identity": builder_identity,
        "source_inventory": inventory,
        "source_inventory_sha256": source_inventory_sha256,
        "run_identity_sha256": run_identity_sha256,
        "run_artifact_root": run_artifact_root,
        "output_artifact_prefix": prefix,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "expected_source_entry_count": sum(source["source_entry_count"] for source in inventory),
        "tasks": tasks,
        "task_inventory_sha256": _canonical_sha256(tasks),
    }
    plan = {**body, "plan_sha256": _canonical_sha256(body)}
    validate_semantic_trace_migration_plan(plan, repo_root=repo_root)
    return plan


def validate_semantic_trace_migration_plan(
    value: object,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate plan identity, source revision, task order, and path ownership."""

    if not isinstance(value, Mapping):
        raise SemanticTraceMigrationMapReduceError("migration plan must be an object")
    plan = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        "training_authorized",
        "implementation_revision",
        "source_revision",
        "upstream",
        "process_identity",
        "builder_identity",
        "source_inventory",
        "source_inventory_sha256",
        "run_identity_sha256",
        "run_artifact_root",
        "output_artifact_prefix",
        "expected_task_count",
        "expected_source_entry_count",
        "tasks",
        "task_inventory_sha256",
        "plan_sha256",
    }
    if set(plan) != expected_fields:
        raise SemanticTraceMigrationMapReduceError("migration plan fields disagree")
    revision = validate_semantic_migration_source_revision(
        plan["source_revision"],
        repo_root=repo_root,
    )
    process_identity = editing_v2_process_identity()
    builder_identity = semantic_packed_builder_identity()
    inventory = plan.get("source_inventory")
    tasks = plan.get("tasks")
    if not isinstance(inventory, list) or not isinstance(tasks, list):
        raise SemanticTraceMigrationMapReduceError(
            "migration source and task inventories must be lists"
        )
    prefix = _artifact_prefix(str(plan.get("output_artifact_prefix", "")))
    upstream = plan.get("upstream")
    if not isinstance(upstream, Mapping) or set(upstream) != _UPSTREAM_FIELDS:
        raise SemanticTraceMigrationMapReduceError("migration upstream identity is malformed")
    for field, item in upstream.items():
        if field != "candidate_provenance_source_stream":
            _require_sha256(item, field=f"upstream.{field}")
    try:
        candidate_source_stream = validate_candidate_source_stream(
            upstream["candidate_provenance_source_stream"],
        )
    except EditingV2SplitAssignmentError as error:
        raise SemanticTraceMigrationMapReduceError(
            "migration upstream candidate provenance source stream is invalid"
        ) from error
    if (
        candidate_source_stream["candidate_materialization"]["manifest_sha256"]
        != upstream["candidate_materialization_manifest_sha256"]
    ):
        raise SemanticTraceMigrationMapReduceError(
            "migration upstream candidate materialization and source stream disagree"
        )
    source_inventory_sha256 = _canonical_sha256(inventory)
    run_identity_body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_revision_sha256": revision["source_revision_sha256"],
        "upstream": dict(upstream),
        "source_inventory_sha256": source_inventory_sha256,
        "process_identity_sha256": process_identity["process_identity_sha256"],
        "builder_identity_sha256": builder_identity["identity_sha256"],
        "output_artifact_prefix": prefix,
    }
    run_identity_sha256 = _canonical_sha256(run_identity_body)
    run_root = f"{prefix}/{run_identity_sha256}"
    expected_cells = [
        (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
    ]
    observed_cells: list[tuple[str, str]] = []
    source_artifacts: set[str] = set()
    source_digests: set[str] = set()
    task_ids: set[str] = set()
    if len(inventory) != EXPECTED_TASK_COUNT or len(tasks) != EXPECTED_TASK_COUNT:
        raise SemanticTraceMigrationMapReduceError(
            "migration plan does not contain exactly 20 source shards and tasks"
        )
    for index, (source, task) in enumerate(zip(inventory, tasks, strict=True)):
        if not isinstance(source, Mapping) or not isinstance(task, Mapping):
            raise SemanticTraceMigrationMapReduceError(
                f"migration source/task {index} must be an object"
            )
        source = dict(source)
        task = dict(task)
        if set(source) != _SOURCE_BINDING_FIELDS or set(task) != _TASK_FIELDS:
            raise SemanticTraceMigrationMapReduceError(
                f"migration source/task {index} fields disagree"
            )
        if {key: task[key] for key in source} != source:
            raise SemanticTraceMigrationMapReduceError(
                f"migration task {index} source binding disagrees"
            )
        observed_cells.append((str(source.get("data_lane")), str(source.get("split"))))
        source_artifact = _require_artifact_path(
            source.get("source_artifact_path"),
            field=f"source_inventory[{index}].source_artifact_path",
        )
        _require_artifact_path(
            source.get("source_manifest_artifact_path"),
            field=f"source_inventory[{index}].source_manifest_artifact_path",
        )
        overlay_path = source.get("source_overlay_artifact_path")
        overlay_sha = source.get("source_overlay_sha256")
        if (overlay_path is None) != (overlay_sha is None):
            raise SemanticTraceMigrationMapReduceError(
                f"source_inventory[{index}] overlay path and hash must be jointly null or present"
            )
        if overlay_path is not None:
            _require_artifact_path(
                overlay_path,
                field=f"source_inventory[{index}].source_overlay_artifact_path",
            )
            _require_sha256(
                overlay_sha,
                field=f"source_inventory[{index}].source_overlay_sha256",
            )
        source_digest = _require_sha256(
            source.get("source_shard_sha256"),
            field=f"source_inventory[{index}].source_shard_sha256",
        )
        _require_sha256(
            source.get("source_manifest_sha256"),
            field=f"source_inventory[{index}].source_manifest_sha256",
        )
        count = source.get("source_entry_count")
        if type(count) is not int or count <= 0:
            raise SemanticTraceMigrationMapReduceError(
                f"source_inventory[{index}].source_entry_count must be positive"
            )
        if source_artifact in source_artifacts or source_digest in source_digests:
            raise SemanticTraceMigrationMapReduceError(
                "migration plan repeats a source path or physical digest"
            )
        source_artifacts.add(source_artifact)
        source_digests.add(source_digest)
        identity_body = {
            "run_identity_sha256": run_identity_sha256,
            "source": source,
        }
        task_id = _canonical_sha256(identity_body)
        if (
            task.get("task_identity_sha256") != task_id
            or task.get("output_artifact_path") != f"{run_root}/tasks/{task_id}"
            or task_id in task_ids
        ):
            raise SemanticTraceMigrationMapReduceError(
                f"migration task {index} identity or output path disagrees"
            )
        task_ids.add(task_id)
    body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["schema_version"] != PLAN_SCHEMA_VERSION
        or plan["status"] != PLAN_STATUS
        or plan["training_authorized"] is not False
        or plan["implementation_revision"] != revision["commit"]
        or plan["process_identity"] != process_identity
        or plan["builder_identity"] != builder_identity
        or observed_cells != expected_cells
        or plan["source_inventory_sha256"] != source_inventory_sha256
        or plan["run_identity_sha256"] != run_identity_sha256
        or plan["run_artifact_root"] != run_root
        or plan["expected_task_count"] != EXPECTED_TASK_COUNT
        or plan["expected_source_entry_count"]
        != sum(int(source["source_entry_count"]) for source in inventory)
        or candidate_source_stream["nonempty_jsonl_rows"] != plan["expected_source_entry_count"]
        or plan["task_inventory_sha256"] != _canonical_sha256(tasks)
        or plan["plan_sha256"] != _canonical_sha256(body)
    ):
        raise SemanticTraceMigrationMapReduceError(
            "migration plan identity, order, provenance, or authority disagrees"
        )
    return plan


def write_semantic_trace_migration_plan(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
) -> Path:
    """Atomically publish or exactly reuse the content-addressed plan."""

    validated = validate_semantic_trace_migration_plan(plan, repo_root=repo_root)
    run_root = _mounted_artifact_path(
        validated["run_artifact_root"],
        artifact_root=artifact_root,
        field="plan.run_artifact_root",
    )
    run_root.mkdir(parents=True, exist_ok=True)
    target = run_root / PLAN_FILENAME
    content = _canonical_json_bytes(validated, newline=True)
    if target.exists():
        if target.read_bytes() != content:
            raise SemanticTraceMigrationMapReduceError(
                f"immutable semantic migration plan collision at {target}"
            )
        return target
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=run_root,
            prefix=f".{PLAN_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target


def load_semantic_trace_migration_plan(
    path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Load one plan and verify its exact semantic identity."""

    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticTraceMigrationMapReduceError(
            f"semantic migration plan is unreadable: {path}"
        ) from error
    return validate_semantic_trace_migration_plan(payload, repo_root=repo_root)


def _task_by_identity(plan: Mapping[str, Any], task_identity_sha256: str) -> dict[str, Any]:
    _require_sha256(task_identity_sha256, field="task_identity_sha256")
    matches = [
        task for task in plan["tasks"] if task["task_identity_sha256"] == task_identity_sha256
    ]
    if len(matches) != 1:
        raise SemanticTraceMigrationMapReduceError(
            "task identity is absent or duplicated in the frozen plan"
        )
    return dict(matches[0])


def _materialization_task(
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> SemanticTraceMigrationTask:
    source = _mounted_artifact_path(
        task["source_artifact_path"],
        artifact_root=artifact_root,
        field="task.source_artifact_path",
    )
    expected_manifest = _mounted_artifact_path(
        task["source_manifest_artifact_path"],
        artifact_root=artifact_root,
        field="task.source_manifest_artifact_path",
    )
    # The core materializer derives the standard manifest path from the source.
    # Refuse a plan that points at another file even if its bytes happen to match.
    from compose_v4.data.packed_trace_store import manifest_path_for

    if manifest_path_for(source).resolve() != expected_manifest:
        raise SemanticTraceMigrationMapReduceError(
            "task source manifest path is not the standard packed-manifest address"
        )
    overlay_path = task["source_overlay_artifact_path"]
    if overlay_path is not None:
        from compose_v4.data.provenance_overlay import overlay_path_for

        expected_overlay = _mounted_artifact_path(
            overlay_path,
            artifact_root=artifact_root,
            field="task.source_overlay_artifact_path",
        )
        if overlay_path_for(source).resolve() != expected_overlay:
            raise SemanticTraceMigrationMapReduceError(
                "task source overlay path is not the standard overlay address"
            )
    output = _mounted_artifact_path(
        task["output_artifact_path"],
        artifact_root=artifact_root,
        field="task.output_artifact_path",
    )
    return SemanticTraceMigrationTask(
        source_path=source,
        source_shard_sha256=task["source_shard_sha256"],
        source_manifest_sha256=task["source_manifest_sha256"],
        source_overlay_sha256=task["source_overlay_sha256"],
        source_unified_manifest_sha256=plan["source_inventory_sha256"],
        source_entry_count=task["source_entry_count"],
        implementation_revision=plan["implementation_revision"],
        data_lane=task["data_lane"],
        split=task["split"],
        output_dir=output,
    )


def execute_semantic_trace_migration_task(
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    *,
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Execute or exactly reuse one whole-physical-shard migration task."""

    validated = validate_semantic_trace_migration_plan(plan, repo_root=repo_root)
    task = _task_by_identity(validated, task_identity_sha256)
    materialization_task = _materialization_task(
        validated,
        task,
        artifact_root=artifact_root,
    )
    output_existed = materialization_task.output_dir.exists()
    receipt = materialize_frozen_packed_shard(materialization_task)
    return {
        "task_identity_sha256": task_identity_sha256,
        "output_artifact_path": task["output_artifact_path"],
        "receipt_sha256": receipt["receipt_sha256"],
        "counts": receipt["counts"],
        "reused": output_existed,
    }


def _validate_task_result(
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    output = _mounted_artifact_path(
        task["output_artifact_path"],
        artifact_root=artifact_root,
        field="task.output_artifact_path",
    )
    try:
        receipt = validate_semantic_trace_materialization(output)
    except SemanticTraceMaterializationError as error:
        raise SemanticTraceMigrationMapReduceError(
            f"semantic migration task result is invalid: {task['task_identity_sha256']}"
        ) from error
    expected_source_binding = {
        "source_shard_name": PurePosixPath(task["source_artifact_path"]).name,
        "source_shard_sha256": task["source_shard_sha256"],
        "source_manifest_sha256": task["source_manifest_sha256"],
        "source_overlay_sha256": task["source_overlay_sha256"],
        "source_unified_manifest_sha256": plan["source_inventory_sha256"],
        "source_entry_count": task["source_entry_count"],
    }
    if (
        receipt["source_binding"] != expected_source_binding
        or receipt["data_lane"] != task["data_lane"]
        or receipt["split"] != task["split"]
        or receipt["implementation_revision"] != plan["implementation_revision"]
        or receipt["process_identity"] != plan["process_identity"]
        or receipt["builder_identity"] != plan["builder_identity"]
    ):
        raise SemanticTraceMigrationMapReduceError(
            f"semantic migration result mismatches task {task['task_identity_sha256']}"
        )
    return receipt


def completed_semantic_trace_migration_task_ids(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
) -> set[str]:
    """Return exact reusable task identities and reject unexpected task objects."""

    validated = validate_semantic_trace_migration_plan(plan, repo_root=repo_root)
    task_root = _mounted_artifact_path(
        f"{validated['run_artifact_root']}/tasks",
        artifact_root=artifact_root,
        field="plan task root",
    )
    expected = {task["task_identity_sha256"] for task in validated["tasks"]}
    if not task_root.exists():
        return set()
    observed = {path.name for path in task_root.iterdir()}
    # The core materializer publishes by directory rename.  A hard container
    # termination can leave only its hidden, non-authoritative staging sibling.
    # Such a sibling is not a result and must not prevent the exact task from
    # being retried.  Every other unexpected object remains a hard failure.
    private_staging = {
        name
        for name in observed
        if any(
            name.startswith(f".{task_id}.") and name.endswith(".staging") for task_id in expected
        )
    }
    published = observed - private_staging
    unexpected = published - expected
    if unexpected:
        raise SemanticTraceMigrationMapReduceError(
            f"semantic migration task namespace contains unexpected objects: {sorted(unexpected)}"
        )
    complete: set[str] = set()
    tasks_by_id = {task["task_identity_sha256"]: task for task in validated["tasks"]}
    for task_id in sorted(published):
        _validate_task_result(
            validated,
            tasks_by_id[task_id],
            artifact_root=artifact_root,
        )
        complete.add(task_id)
    return complete


def _counter_sum(target: Counter[str], value: object, *, field: str) -> None:
    if not isinstance(value, Mapping):
        raise SemanticTraceMigrationMapReduceError(f"{field} must be an object")
    for key, count in value.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            raise SemanticTraceMigrationMapReduceError(
                f"{field} must map strings to nonnegative integers"
            )
        target[key] += count


def reduce_semantic_trace_migration(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Publish completion only after an exact, mismatch-free task reduction."""

    validated = validate_semantic_trace_migration_plan(plan, repo_root=repo_root)
    run_root = _mounted_artifact_path(
        validated["run_artifact_root"],
        artifact_root=artifact_root,
        field="plan.run_artifact_root",
    )
    published_plan_path = run_root / PLAN_FILENAME
    expected_plan_bytes = _canonical_json_bytes(validated, newline=True)
    if not published_plan_path.is_file() or published_plan_path.read_bytes() != expected_plan_bytes:
        raise SemanticTraceMigrationMapReduceError(
            "semantic migration reduction requires the exact published plan bytes"
        )
    plan_file_sha256 = hashlib.sha256(expected_plan_bytes).hexdigest()
    complete = completed_semantic_trace_migration_task_ids(
        validated,
        artifact_root=artifact_root,
        repo_root=repo_root,
    )
    expected = {task["task_identity_sha256"] for task in validated["tasks"]}
    missing = expected - complete
    if missing:
        raise SemanticTraceMigrationIncomplete(
            f"semantic migration is missing {len(missing)} planned task results"
        )
    results: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    rejection_totals: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    for task in validated["tasks"]:
        receipt = _validate_task_result(
            validated,
            task,
            artifact_root=artifact_root,
        )
        _counter_sum(totals, receipt["counts"], field="task receipt counts")
        _counter_sum(
            rejection_totals,
            receipt["rejections_by_code"],
            field="task receipt rejections_by_code",
        )
        _counter_sum(
            family_totals,
            receipt["accepted_family_histogram"],
            field="task receipt accepted_family_histogram",
        )
        results.append(
            {
                "task_identity_sha256": task["task_identity_sha256"],
                "data_lane": task["data_lane"],
                "split": task["split"],
                "source_artifact_path": task["source_artifact_path"],
                "output_artifact_path": task["output_artifact_path"],
                "receipt_sha256": receipt["receipt_sha256"],
                "semantic_shard_sha256": receipt["semantic_shard_sha256"],
                "semantic_manifest_sha256": receipt["semantic_manifest_sha256"],
                "counts": receipt["counts"],
            }
        )
    if (
        totals["source"] != validated["expected_source_entry_count"]
        or totals["source"] != totals["admitted"] + totals["rejected"]
    ):
        raise SemanticTraceMigrationMapReduceError(
            "semantic migration reduction does not account for every source trace"
        )
    body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        "training_authorized": False,
        "gate_zero_run": False,
        "run_identity_sha256": validated["run_identity_sha256"],
        "plan_sha256": validated["plan_sha256"],
        "plan_file_sha256": plan_file_sha256,
        "source_revision_sha256": validated["source_revision"]["source_revision_sha256"],
        "source_inventory_sha256": validated["source_inventory_sha256"],
        "process_identity_sha256": validated["process_identity"]["process_identity_sha256"],
        "builder_identity_sha256": validated["builder_identity"]["identity_sha256"],
        "task_inventory_sha256": validated["task_inventory_sha256"],
        "task_count": len(results),
        "result_inventory": results,
        "result_inventory_sha256": _canonical_sha256(results),
        "counts": {key: totals[key] for key in ("source", "admitted", "rejected")},
        "rejections_by_code": dict(sorted(rejection_totals.items())),
        "accepted_family_histogram": dict(sorted(family_totals.items())),
    }
    completion = {**body, "completion_sha256": _canonical_sha256(body)}
    run_root.mkdir(parents=True, exist_ok=True)
    target = run_root / COMPLETION_FILENAME
    content = _canonical_json_bytes(completion, newline=True)
    if target.exists():
        if target.read_bytes() != content:
            raise SemanticTraceMigrationMapReduceError(
                f"immutable semantic migration completion collision at {target}"
            )
        return completion
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=run_root,
            prefix=f".{COMPLETION_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return completion


__all__ = [
    "COMPLETION_FILENAME",
    "COMPLETION_SCHEMA",
    "COMPLETION_SCHEMA_VERSION",
    "COMPLETION_STATUS",
    "DEFAULT_OUTPUT_ARTIFACT_PREFIX",
    "EXPECTED_TASK_COUNT",
    "PLAN_FILENAME",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "PLAN_STATUS",
    "SemanticTraceMigrationIncomplete",
    "SemanticTraceMigrationMapReduceError",
    "build_semantic_migration_source_revision",
    "completed_semantic_trace_migration_task_ids",
    "execute_semantic_trace_migration_task",
    "load_semantic_trace_migration_plan",
    "plan_semantic_trace_migration",
    "reduce_semantic_trace_migration",
    "repository_semantic_migration_source_revision",
    "validate_semantic_migration_source_revision",
    "validate_semantic_trace_migration_plan",
    "write_semantic_trace_migration_plan",
]
