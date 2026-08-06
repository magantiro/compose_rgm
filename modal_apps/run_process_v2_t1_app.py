"""Prepare and run the bounded Process-V2 T1 capacity gate on Modal.

This file is orchestration only.  The Process-V2 panel, exact successor
compiler, optimizer, and result builder remain in their production modules.
One detached driver performs the stages in order:

1. build the bounded unique-state panel and immutable preparation plan;
2. compile missing plan leaves on at most forty replenishing CPU containers;
3. validate, hash, and deterministically reduce every planned leaf;
4. run one scratch-model successor-capacity job on one GPU.

Importing this module launches nothing.  The launcher binds every serialized
source byte to an exact clean Git commit.  It never launches P50 and never
grants downstream authority.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_t1_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")

PREPARED_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_t1_prepared"
COLLATED_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_t1_collated"
RUN_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_t1_capacity"
FAILURE_SCOPE_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_t1_failure_scope"
SCORE_REVISION_REPAIR_OUTPUT_PREFIX = (
    "/artifacts/editing_v2/process_v2_t1_score_revision_repair"
)
FINAL_SCOPE_REPAIR_OUTPUT_PREFIX = (
    "/artifacts/editing_v2/process_v2_t1_final_scope_repair"
)
CAPACITY_POLICY_SOURCE = "configs/editing_v2_process_v2_t1_capacity_policy.json"
PANEL_FILENAME = "PROCESS_V2_T1_PANEL.json"
PANEL_CACHE_DIRNAME = "_panel_cache"
PLAN_FILENAME = "PROCESS_V2_T1_PREPARED_PLAN.json"
LEAF_FILENAME = "PROCESS_V2_T1_PREPARED_LEAF.json"
PREPARED_COMPLETION_FILENAME = "PROCESS_V2_T1_PREPARED_COMPLETE.json"
PROCESS_V2_RESULT_FILENAME = "PROCESS_V2_T1_CAPACITY_RESULT.json"
FAILURE_SCOPE_RESULT_FILENAME = "PROCESS_V2_T1_FAILURE_SCOPE_RESULT.json"
SCORE_REVISION_REPAIR_FILENAME = "PROCESS_V2_T1_SCORE_REVISION_REPAIR.json"
FINAL_SCOPE_REPAIR_FILENAME = "PROCESS_V2_T1_FINAL_SCOPE_REPAIR.json"
STEP_TEN_CHECKPOINT_FILENAME = "step_0010.pt"
PUBLICATION_RECOVERY_RECEIPT_FILENAME = "PROCESS_V2_T1_PUBLICATION_RECOVERY.json"
PUBLICATION_RECOVERY_OUTPUT_PREFIX = (
    "/artifacts/editing_v2/process_v2_t1_publication_recovery"
)
TERMINAL_AUDIT_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_t1_terminal_audit"
TERMINAL_AUDIT_FILENAME = "PROCESS_V2_T1_TERMINAL_AUDIT.json"
PUBLICATION_RECOVERY_GPU = "A10G"

MAX_CPU_CONTAINERS = 40
COLLATED_ENTRIES_PER_TASK = 16
CPU_PER_LEAF = 1.0
CPU_MEMORY_MB = 8 * 1024
CPU_LEAF_TIMEOUT_SECONDS = 45 * 60
COORDINATOR_TIMEOUT_SECONDS = 6 * 3600
GPU_TIMEOUT_SECONDS = 8 * 3600
FAILURE_SCOPE_GPU_TIMEOUT_SECONDS = 20 * 60
SCORE_REVISION_REPAIR_GPU_TIMEOUT_SECONDS = 60 * 60
HEARTBEAT_SECONDS = 30
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

REVISION_SCHEMA = "compose.editing_v2.process_v2_t1_modal_image_revision"
REVISION_SCHEMA_VERSION = 1
SOURCE_REVISION_SCHEMA = "compose.editing_v2.process_v2_t1_authenticated_image_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_SCORE_REVISION_REPAIR_FAMILIES = (
    "bond_reroute",
    "cycle_attach",
    "ring_system_restate",
)
_PREDECESSOR_CAPACITY_RESULT_FILE_SHA256 = (
    "947732cdfbaf425ebb09ecb8bff453bd6023aaba3445131937ee727d021f60f9"
)
_PREDECESSOR_CAPACITY_RESULT_SHA256 = (
    "dd60265b494e74a236c6902d0d658a0cdf4a10a4733b8d8351ba8e6109b86f4e"
)
_ATOM_INSERT_PASS_RESULT_SHA256 = (
    "06e5c1c991740ba849fb26326326f104976e716d53588bdd80b96de6f0942c45"
)
_FINAL_SCOPE_REPAIR_FAMILIES = (
    "cycle_attach",
    "ring_system_restate",
)
_SCORE_REVISION_REPAIR_RESULT_FILE_SHA256 = (
    "423d61cd814c457f41c36ef2ee06be5f56265b0ca6c9f6dd51fd9603eb19a316"
)
_SCORE_REVISION_REPAIR_RESULT_SHA256 = (
    "a13bbdad41318343c78e70a91c855104c32327b5058b15fab78ea333bd0fcd17"
)
_FINAL_SCOPE_PREDECESSORS = {
    "cycle_attach": {
        "file_sha256": "c66b73ef1ed09053341ab58bca421230902d036f5f75c2b83e7cf69cc6579d20",
        "result_sha256": "6f6631ec8bde598c8295c19e5c71749504b32e361e3dcd90627f013bb987fbe5",
    },
    "ring_system_restate": {
        "file_sha256": "f25ead40949cb5169a218484731d284f4c66c54544b3092bc64c1bdcebdbb778",
        "result_sha256": "b00853616c4cbe0af4f3db412064fdda93f42e7d4567cdc97d8ff92560eb514b",
    },
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
)
for source_directory in IMAGE_SOURCE_DIRECTORIES:
    image = image.add_local_dir(
        ROOT / source_directory,
        str(REMOTE_ROOT / source_directory),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
image = image.add_local_file(
    ROOT / LAUNCHER_SOURCE,
    str(REMOTE_ROOT / LAUNCHER_SOURCE),
    copy=True,
)

app = modal.App("compose-v4-process-v2-t1")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)
_VALIDATED_REVISION_SHA256: str | None = None


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot establish Process-V2 T1 Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE]
    for directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / directory).rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("Process-V2 T1 serialized source inventory repeats a path")
    return tuple(paths)


def local_image_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind the exact clean commit and every byte serialized into the image."""

    if _COMMIT_RE.fullmatch(str(expected_commit)) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError("Process-V2 T1 launch requires the exact clean commit")
    sources = {
        relative: _file_sha256(root / relative) for relative in _serialized_source_paths(root)
    }
    tracked = set(_git(root, "ls-files").splitlines())
    if not set(sources).issubset(tracked):
        raise RuntimeError("every serialized Process-V2 T1 source must be Git-tracked")
    body = {
        "schema": REVISION_SCHEMA,
        "schema_version": REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    return {**body, "image_revision_sha256": _sha256(body)}


def _validate_remote_revision(value: Mapping[str, Any]) -> None:
    """Re-hash the serialized image once per warm container."""

    global _VALIDATED_REVISION_SHA256
    supplied = str(value.get("image_revision_sha256"))
    if _VALIDATED_REVISION_SHA256 == supplied:
        return
    body = {key: item for key, item in value.items() if key != "image_revision_sha256"}
    sources = value.get("serialized_sources")
    if (
        value.get("schema") != REVISION_SCHEMA
        or value.get("schema_version") != REVISION_SCHEMA_VERSION
        or value.get("worktree_clean") is not True
        or _COMMIT_RE.fullmatch(str(value.get("commit"))) is None
        or _COMMIT_RE.fullmatch(str(value.get("tree"))) is None
        or _SHA256_RE.fullmatch(supplied) is None
        or supplied != _sha256(body)
        or not isinstance(sources, Mapping)
        or set(sources) != set(_serialized_source_paths(REMOTE_ROOT))
    ):
        raise RuntimeError("Process-V2 T1 image revision disagrees")
    for relative, digest in sources.items():
        if _file_sha256(REMOTE_ROOT / str(relative)) != digest:
            raise RuntimeError(f"serialized Process-V2 T1 source differs: {relative}")
    _VALIDATED_REVISION_SHA256 = supplied


def _require_artifact_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a normalized absolute artifact path")
    logical_root = ARTIFACT_ROOT
    physical_root = logical_root.resolve()
    try:
        relative = path.relative_to(logical_root)
    except ValueError:
        try:
            relative = path.relative_to(physical_root)
        except ValueError as error:
            raise ValueError(f"{field} must be below {ARTIFACT_ROOT}") from error
    resolved = (physical_root / relative).resolve()
    try:
        resolved.relative_to(physical_root)
    except ValueError as error:
        raise ValueError(f"{field} resolves outside {ARTIFACT_ROOT}") from error
    # Preserve the logical /artifacts spelling when supplied.  Upstream
    # Process-V2 source identities bind that stable address, not Modal's
    # container-specific physical volume target.
    return path


def _require_physical_artifact_path(value: str, *, field: str) -> Path:
    """Resolve a validated artifact address for immutable local writes."""

    path = _require_artifact_path(value, field=field)
    logical_root = ARTIFACT_ROOT
    physical_root = logical_root.resolve()
    try:
        relative = path.relative_to(logical_root)
    except ValueError:
        relative = path.relative_to(physical_root)
    return (physical_root / relative).resolve()


def _read_canonical_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise RuntimeError(f"{label} is not canonical newline-framed JSON")
    return value


def _source_revision(revision: Mapping[str, Any]) -> dict[str, Any]:
    commit = str(revision.get("commit"))
    tree = str(revision.get("tree"))
    image_revision_sha256 = str(revision.get("image_revision_sha256"))
    sources = revision.get("serialized_sources")
    if (
        _COMMIT_RE.fullmatch(commit) is None
        or _COMMIT_RE.fullmatch(tree) is None
        or _SHA256_RE.fullmatch(image_revision_sha256) is None
        or not isinstance(sources, Mapping)
    ):
        raise RuntimeError("Process-V2 T1 image lacks an authenticated source revision")
    body = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "image_revision_sha256": image_revision_sha256,
        "serialized_source_inventory_sha256": _sha256(dict(sources)),
    }
    return {**body, "source_revision_sha256": _sha256(body)}


def _imports() -> dict[str, Any]:
    source_root = str(REMOTE_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    import numpy as np
    import rdkit
    import torch

    from compose_v4.data.immutable_artifact import write_bytes_if_absent
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        build_process_v2_t1_panel,
        load_process_v2_t1_panel,
        open_process_v2_t1_source,
        write_process_v2_t1_panel,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_collated_cache import (
        COMPLETION_FILENAME as COLLATED_COMPLETION_FILENAME,
        PLAN_FILENAME as COLLATED_PLAN_FILENAME,
        build_collated_cache_plan,
        compile_collated_cache_leaf,
        completed_collated_task_identities,
        load_collated_cache_plan,
        load_materialized_collated_panel,
        publish_collated_cache_completion,
        write_collated_cache_plan,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_result import (
        DECISION_FILENAME,
        RESULT_FILENAME,
        build_process_v2_t1_capacity_decision,
        build_process_v2_t1_capacity_result,
        process_v2_t1_runner_implementation_sha256,
        project_process_v2_t1_result_provenance,
        validate_process_v2_t1_capacity_decision,
        validate_process_v2_t1_capacity_result,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_failure_scope import (
        failing_families_from_capacity_result,
        next_scope_for_family,
        run_final_failure_scope,
        run_heads_only_failure_scope,
        run_next_failure_scope,
        validate_failure_scope_result,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        TASKS_DIRNAME,
        build_process_v2_t1_prepared_plan,
        build_process_v2_t1_scratch_runtime,
        compile_authenticated_process_v2_t1_prepared_leaf,
        load_process_v2_t1_capacity_policy,
        load_process_v2_t1_runtime_inputs,
        publish_process_v2_t1_prepared_inputs,
        publish_reused_process_v2_t1_prepared_inputs,
        validate_process_v2_t1_prepared_leaf,
        write_process_v2_t1_prepared_leaf,
        write_process_v2_t1_prepared_plan,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        _MaterializedSemanticT1Panel,
        _batch_for_ids,
        _collator,
        _metric_rows,
        run_semantic_t1_capacity,
        semantic_t1_threshold_checks,
        summarize_semantic_t1_metrics,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_checkpoint import (
        CHECKPOINT_SCHEMA,
        CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_STATUS,
        NO_AUTHORITY,
        validate_semantic_t1_selected_checkpoint,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_decision import (
        build_semantic_t1_execution_environment,
    )

    return {
        "np": np,
        "rdkit": rdkit,
        "torch": torch,
        "DECISION_FILENAME": DECISION_FILENAME,
        "RESULT_FILENAME": RESULT_FILENAME,
        "TASKS_DIRNAME": TASKS_DIRNAME,
        "build_panel": build_process_v2_t1_panel,
        "load_panel": load_process_v2_t1_panel,
        "open_source": open_process_v2_t1_source,
        "write_panel": write_process_v2_t1_panel,
        "build_plan": build_process_v2_t1_prepared_plan,
        "build_scratch": build_process_v2_t1_scratch_runtime,
        "compile_leaf": compile_authenticated_process_v2_t1_prepared_leaf,
        "load_capacity_policy": load_process_v2_t1_capacity_policy,
        "load_runtime": load_process_v2_t1_runtime_inputs,
        "publish_prepared": publish_process_v2_t1_prepared_inputs,
        "publish_reused_prepared": publish_reused_process_v2_t1_prepared_inputs,
        "validate_leaf": validate_process_v2_t1_prepared_leaf,
        "write_leaf": write_process_v2_t1_prepared_leaf,
        "write_plan": write_process_v2_t1_prepared_plan,
        "COLLATED_COMPLETION_FILENAME": COLLATED_COMPLETION_FILENAME,
        "COLLATED_PLAN_FILENAME": COLLATED_PLAN_FILENAME,
        "build_collated_plan": build_collated_cache_plan,
        "compile_collated_leaf": compile_collated_cache_leaf,
        "completed_collated_tasks": completed_collated_task_identities,
        "load_collated_plan": load_collated_cache_plan,
        "load_materialized_collated_panel": load_materialized_collated_panel,
        "publish_collated_completion": publish_collated_cache_completion,
        "write_collated_plan": write_collated_cache_plan,
        "build_result": build_process_v2_t1_capacity_result,
        "build_decision": build_process_v2_t1_capacity_decision,
        "runner_implementation_sha256": process_v2_t1_runner_implementation_sha256,
        "project_result_provenance": project_process_v2_t1_result_provenance,
        "validate_decision": validate_process_v2_t1_capacity_decision,
        "validate_result": validate_process_v2_t1_capacity_result,
        "failing_families": failing_families_from_capacity_result,
        "next_failure_scope": next_scope_for_family,
        "run_final_failure_scope": run_final_failure_scope,
        "run_failure_scope": run_heads_only_failure_scope,
        "run_next_failure_scope": run_next_failure_scope,
        "validate_failure_scope": validate_failure_scope_result,
        "write_bytes_if_absent": write_bytes_if_absent,
        "build_environment": build_semantic_t1_execution_environment,
        "run_capacity": run_semantic_t1_capacity,
        "MaterializedT1Panel": _MaterializedSemanticT1Panel,
        "batch_for_ids": _batch_for_ids,
        "collator": _collator,
        "metric_rows": _metric_rows,
        "threshold_checks": semantic_t1_threshold_checks,
        "summarize_metrics": summarize_semantic_t1_metrics,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
        "CHECKPOINT_SCHEMA_VERSION": CHECKPOINT_SCHEMA_VERSION,
        "CHECKPOINT_STATUS": CHECKPOINT_STATUS,
        "NO_AUTHORITY": NO_AUTHORITY,
        "validate_selected_checkpoint": validate_semantic_t1_selected_checkpoint,
    }


def _publication_recovery_result_builder(
    *,
    build_result: Callable[..., Mapping[str, Any]],
    capacity_policy: Mapping[str, Any],
    provenance: Mapping[str, Any],
    run_integrity: Mapping[str, Any],
    evaluation_trajectory: Sequence[Mapping[str, Any]],
    entry_metrics: Sequence[Mapping[str, Any]],
    gradient_evidence: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Repair publication metadata after a completed, uninterrupted optimizer run."""

    maximum_steps = int(capacity_policy["optimization"]["maximum_optimizer_steps"])
    if (
        run_integrity.get("optimizer_steps_completed") != maximum_steps
        or run_integrity.get("resume_requested") is not True
        or run_integrity.get("resume_count") != 1
        or not evaluation_trajectory
        or evaluation_trajectory[-1].get("step") != maximum_steps
    ):
        raise RuntimeError(
            "Process-V2 T1 publication recovery did not reopen exactly one completed run"
        )
    families = tuple(str(item) for item in capacity_policy["required_families"])
    family_indices = {family: index for index, family in enumerate(families)}
    metadata = sorted(
        (
            {
                "panel_entry_sha256": str(row["panel_entry_sha256"]),
                "family": str(row["family"]),
                "semantic_cell_id": str(row["semantic_cell_id"]),
            }
            for row in entry_metrics
        ),
        key=lambda row: (
            family_indices[row["family"]],
            row["semantic_cell_id"],
            row["panel_entry_sha256"],
        ),
    )
    corrected_provenance = {
        **dict(provenance),
        "panel_entry_metadata_sha256": _sha256(metadata),
    }
    corrected_integrity = {
        **dict(run_integrity),
        # These fields describe optimization, not this publication-only
        # reconstruction.  The trusted terminal checkpoint below proves the
        # original optimizer ran uninterrupted with resume_count == 0.
        "resume_requested": False,
        "resume_count": 0,
    }
    return build_result(
        provenance=corrected_provenance,
        run_integrity=corrected_integrity,
        evaluation_trajectory=evaluation_trajectory,
        entry_metrics=entry_metrics,
        gradient_evidence=gradient_evidence,
        capacity_policy=capacity_policy,
    )


def _validate_publication_recovery_checkpoint(
    payload: object,
    *,
    loaded: Mapping[str, Any],
    runtime: Any,
    expected_runner_implementation_sha256: str,
    expected_runner_source_revision_sha256: str,
) -> dict[str, Any]:
    """Validate the metadata needed before the runner rechecks all tensor state."""

    if not isinstance(payload, dict):
        raise RuntimeError("Process-V2 T1 recovery checkpoint is not a mapping")
    authority = loaded["NO_AUTHORITY"]
    identity = payload.get("identity")
    maximum_steps = int(runtime.capacity_policy["optimization"]["maximum_optimizer_steps"])
    if (
        payload.get("schema") != loaded["CHECKPOINT_SCHEMA"]
        or payload.get("schema_version") != loaded["CHECKPOINT_SCHEMA_VERSION"]
        or payload.get("status") != loaded["CHECKPOINT_STATUS"]
        or any(payload.get(name) is not value for name, value in authority.items())
        or payload.get("completed_steps") != maximum_steps
        or payload.get("resume_count") != 0
        or not isinstance(identity, Mapping)
        or identity.get("capacity_policy_sha256")
        != runtime.capacity_policy["policy_sha256"]
        or identity.get("prepared_input_artifact_sha256")
        != runtime.prepared.artifact["artifact_sha256"]
        or identity.get("cache_completion_sha256")
        != runtime.cache.completion["completion_sha256"]
        or identity.get("cache_manifest_sha256")
        != runtime.cache.manifest["manifest_sha256"]
        or identity.get("initial_model_state_sha256")
        != runtime.cache.completion["initial_model_state_sha256"]
        or identity.get("runner_implementation_sha256")
        != expected_runner_implementation_sha256
        or identity.get("runner_source_revision_sha256")
        != expected_runner_source_revision_sha256
    ):
        raise RuntimeError("Process-V2 T1 publication-recovery checkpoint disagrees")
    return payload


def _open_source(
    *, active8_run_root: Path, gate_zero_decision_path: Path, loaded: Mapping[str, Any]
) -> Any:
    return loaded["open_source"](
        active8_run_root,
        gate_zero_decision_path=gate_zero_decision_path,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )


def _panel_cache_identity(source: Any, *, source_revision: Mapping[str, Any]) -> str:
    """Address one panel selection before the expensive stream is scanned."""

    return _sha256(
        {
            "schema": "compose.editing_v2.process_v2_t1_panel_cache_address",
            "schema_version": 1,
            "source_revision_sha256": source_revision["source_revision_sha256"],
            "process_identity_sha256": source.contracts.process_identity_sha256,
            "active8_completion_sha256": source.index.active8_completion_sha256,
            "active8_sentinel_sha256": source.index.active8_sentinel_sha256,
            "active8_plan_sha256": source.plan["plan_sha256"],
            "gate_zero_decision_sha256": source.decision["decision_sha256"],
            "panel_policy_sha256": source.policy["contract_sha256"],
        }
    )


def _progress(phase: str, **fields: object) -> None:
    print(
        json.dumps({"phase": phase, "time_unix_seconds": time.time(), **fields}, sort_keys=True),
        flush=True,
    )


@contextmanager
def _heartbeat(phase: str, *, interval_seconds: int = HEARTBEAT_SECONDS) -> Iterator[None]:
    """Print liveness without touching model, optimizer, or artifact state."""

    stop = threading.Event()
    started = time.monotonic()

    def emit() -> None:
        while not stop.wait(interval_seconds):
            _progress(phase, elapsed_seconds=time.monotonic() - started)

    thread = threading.Thread(target=emit, name=f"{phase}-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=min(float(interval_seconds), 1.0))


def _groups(values: Sequence[str], *, maximum: int) -> tuple[tuple[str, ...], ...]:
    if type(maximum) is not int or not 1 <= maximum <= MAX_CPU_CONTAINERS:
        raise ValueError(f"maximum must lie in [1, {MAX_CPU_CONTAINERS}]")
    ordered = tuple(values)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Process-V2 T1 task inventory repeats an identity")
    return tuple(
        tuple(ordered[index : index + maximum]) for index in range(0, len(ordered), maximum)
    )


@contextmanager
def _gpu_checkpoint_heartbeat(run_root: Path) -> Iterator[None]:
    """Publish liveness and commit the step-10 checkpoint as soon as it exists."""

    stop = threading.Event()
    started = time.monotonic()
    checkpoint_path = Path(run_root) / "checkpoints" / STEP_TEN_CHECKPOINT_FILENAME
    checkpoint_committed = threading.Event()

    def emit() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            if checkpoint_path.is_file() and not checkpoint_committed.is_set():
                artifact_volume.commit()
                checkpoint_committed.set()
                _progress(
                    "process_v2_t1_step_ten_checkpoint_committed",
                    checkpoint_path=str(checkpoint_path),
                    checkpoint_file_sha256=_file_sha256(checkpoint_path),
                )
            _progress(
                "process_v2_t1_gpu_heartbeat",
                elapsed_seconds=time.monotonic() - started,
                step_ten_checkpoint_committed=checkpoint_committed.is_set(),
            )

    thread = threading.Thread(
        target=emit,
        name="process-v2-t1-gpu-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        if checkpoint_path.is_file() and not checkpoint_committed.is_set():
            artifact_volume.commit()
            checkpoint_committed.set()
            _progress(
                "process_v2_t1_step_ten_checkpoint_committed",
                checkpoint_path=str(checkpoint_path),
                checkpoint_file_sha256=_file_sha256(checkpoint_path),
            )


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_plan_remote(
    active8_run_root: str,
    gate_zero_decision_path: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Build the panel and one immutable CPU task per selected source chunk."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    active8_root = _require_artifact_path(active8_run_root, field="active8_run_root")
    gate_zero_path = _require_artifact_path(
        gate_zero_decision_path, field="gate_zero_decision_path"
    )
    prefix = _require_physical_artifact_path(output_prefix, field="output_prefix")
    source = _open_source(
        active8_run_root=active8_root,
        gate_zero_decision_path=gate_zero_path,
        loaded=loaded,
    )
    source_revision = _source_revision(revision)
    panel_cache_identity = _panel_cache_identity(
        source,
        source_revision=source_revision,
    )
    panel_cache_root = prefix / PANEL_CACHE_DIRNAME / panel_cache_identity
    panel_cache_path = panel_cache_root / PANEL_FILENAME
    panel_cache_hit = panel_cache_path.is_file()
    if panel_cache_hit:
        panel = loaded["load_panel"](panel_cache_path, source=source)
        _progress(
            "process_v2_t1_panel_cache_hit",
            panel_cache_path=str(panel_cache_path),
            panel_sha256=panel["panel_sha256"],
        )
    else:
        with _heartbeat("process_v2_t1_panel_selection"):
            panel = loaded["build_panel"](source)
        panel_cache_path = loaded["write_panel"](
            panel,
            output_root=panel_cache_root,
            source=source,
        )
        # The panel is the expensive selection result.  Commit it before model
        # reconstruction or plan publication so any later failure resumes here.
        artifact_volume.commit()
        _progress(
            "process_v2_t1_panel_cache_published",
            panel_cache_path=str(panel_cache_path),
            panel_sha256=panel["panel_sha256"],
        )
    # The runtime derives and validates the clean repository revision itself.
    plan = loaded["build_plan"](
        panel,
        source=source,
        source_revision=source_revision,
    )
    run_root = prefix / str(plan["run_identity_sha256"])
    panel_path = loaded["write_panel"](panel, output_root=run_root, source=source)
    plan_path = loaded["write_plan"](
        plan,
        output_root=prefix,
        panel=panel,
        source=source,
    )
    if plan_path.parent != run_root:
        raise RuntimeError("Process-V2 T1 plan and panel resolved to different run roots")
    artifact_volume.commit()
    result = {
        "phase": "process_v2_t1_plan_complete",
        "run_root": str(run_root),
        "panel_path": str(panel_path),
        "panel_cache_path": str(panel_cache_path),
        "panel_cache_hit": panel_cache_hit,
        "plan_path": str(plan_path),
        "plan": plan,
        "task_count": int(plan["task_count"]),
        "selected_entry_count": int(plan["selected_entry_count"]),
        "image_revision_sha256": revision["image_revision_sha256"],
        "training_launched": False,
        "bounded_p50_authorized": False,
    }
    _progress(**result)
    return result


@app.function(
    image=image,
    cpu=1.0,
    memory=4 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def scan_completed_remote(plan_path: str, run_root: str, revision: dict[str, Any]) -> list[str]:
    """Return only leaves that pass the production leaf validator."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    plan = _read_canonical_object(
        _require_artifact_path(plan_path, field="plan_path"),
        label="the Process-V2 T1 prepared plan",
    )
    root = _require_artifact_path(run_root, field="run_root")
    completed: list[str] = []
    for task in plan["tasks"]:
        identity = str(task["task_identity_sha256"])
        path = root / loaded["TASKS_DIRNAME"] / identity / LEAF_FILENAME
        if not path.exists():
            continue
        leaf = _read_canonical_object(path, label="a Process-V2 T1 prepared leaf")
        loaded["validate_leaf"](leaf, plan=plan)
        completed.append(identity)
    return completed


@app.function(
    image=image,
    cpu=CPU_PER_LEAF,
    memory=CPU_MEMORY_MB,
    timeout=CPU_LEAF_TIMEOUT_SECONDS,
    max_containers=MAX_CPU_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_leaf_remote(
    plan_path: str,
    task_identity_sha256: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    panel_path: str,
    run_root: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Compile and immutably publish one restart-safe exact successor leaf."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    plan = _read_canonical_object(
        _require_artifact_path(plan_path, field="plan_path"),
        label="the Process-V2 T1 prepared plan",
    )
    source = _open_source(
        active8_run_root=_require_artifact_path(active8_run_root, field="active8_run_root"),
        gate_zero_decision_path=_require_artifact_path(
            gate_zero_decision_path, field="gate_zero_decision_path"
        ),
        loaded=loaded,
    )
    panel = _read_canonical_object(
        _require_artifact_path(panel_path, field="panel_path"),
        label="the Process-V2 T1 panel",
    )
    scratch, _binding = loaded["build_scratch"](source)

    def report(event: Mapping[str, Any]) -> None:
        _progress(**dict(event))

    with _heartbeat(
        "process_v2_t1_prepare_leaf_heartbeat",
        interval_seconds=HEARTBEAT_SECONDS,
    ):
        leaf = loaded["compile_leaf"](
            plan,
            task_identity_sha256=task_identity_sha256,
            panel=panel,
            source=source,
            scratch_runtime=scratch,
            progress_callback=report,
        )
    path = loaded["write_leaf"](
        leaf, plan=plan, run_root=_require_artifact_path(run_root, field="run_root")
    )
    artifact_volume.commit()
    result = {
        "task_identity_sha256": task_identity_sha256,
        "leaf_sha256": leaf["leaf_sha256"],
        "leaf_path": str(path),
        "entry_count": int(leaf["entry_count"]),
    }
    _progress("process_v2_t1_prepare_leaf_complete", **result)
    return result


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def finalize_prepared_remote(
    plan_path: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    panel_path: str,
    run_root: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Validate, hash, and deterministically reduce every planned leaf."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    plan = _read_canonical_object(
        _require_artifact_path(plan_path, field="plan_path"),
        label="the Process-V2 T1 prepared plan",
    )
    source = _open_source(
        active8_run_root=_require_artifact_path(active8_run_root, field="active8_run_root"),
        gate_zero_decision_path=_require_artifact_path(
            gate_zero_decision_path, field="gate_zero_decision_path"
        ),
        loaded=loaded,
    )
    panel = _read_canonical_object(
        _require_artifact_path(panel_path, field="panel_path"),
        label="the Process-V2 T1 panel",
    )
    root = _require_physical_artifact_path(run_root, field="run_root")
    with _heartbeat("process_v2_t1_prepared_leaf_reduction"):
        completion_path = loaded["publish_prepared"](
            plan,
            panel=panel,
            source=source,
            run_root=root,
        )
    artifact_volume.commit()
    completion = _read_canonical_object(
        completion_path, label="the Process-V2 T1 prepared completion"
    )
    result = {
        "phase": "process_v2_t1_prepared_complete",
        "run_root": str(root),
        "prepared_completion_path": str(completion_path),
        "prepared_completion_sha256": completion["completion_sha256"],
        "entry_count": int(completion["entry_count"]),
        "training_launched": False,
        "bounded_p50_authorized": False,
    }
    _progress(**result)
    return result


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def finalize_reused_prepared_remote(
    plan_path: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    panel_path: str,
    run_root: str,
    expected_plan_sha256: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Reduce one exact completed leaf run without recomputing any successor fiber."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    plan = _read_canonical_object(
        _require_artifact_path(plan_path, field="plan_path"),
        label="the reusable Process-V2 T1 prepared plan",
    )
    source = _open_source(
        active8_run_root=_require_artifact_path(active8_run_root, field="active8_run_root"),
        gate_zero_decision_path=_require_artifact_path(
            gate_zero_decision_path, field="gate_zero_decision_path"
        ),
        loaded=loaded,
    )
    panel = _read_canonical_object(
        _require_artifact_path(panel_path, field="panel_path"),
        label="the reusable Process-V2 T1 panel",
    )
    leaf_root = _require_physical_artifact_path(run_root, field="run_root")
    source_revision = _source_revision(revision)
    reduction_identity = _sha256(
        {
            "schema": "compose.editing_v2.process_v2_t1_reused_reduction",
            "schema_version": 1,
            "leaf_plan_sha256": expected_plan_sha256,
            "source_revision_sha256": source_revision["source_revision_sha256"],
        }
    )
    root = leaf_root / "reductions" / reduction_identity
    _progress(
        "process_v2_t1_reused_reduction_start",
        run_root=str(root),
        expected_plan_sha256=expected_plan_sha256,
    )
    with _heartbeat("process_v2_t1_reused_leaf_reduction"):
        completion_path = loaded["publish_reused_prepared"](
            plan,
            panel=panel,
            source=source,
            leaf_run_root=leaf_root,
            output_root=root,
            source_revision=source_revision,
            expected_plan_sha256=expected_plan_sha256,
        )
    artifact_volume.commit()
    completion = _read_canonical_object(
        completion_path, label="the reused Process-V2 T1 prepared completion"
    )
    result = {
        "phase": "process_v2_t1_reused_prepared_complete",
        "run_root": str(root),
        "prepared_completion_path": str(completion_path),
        "prepared_completion_sha256": completion["completion_sha256"],
        "entry_count": int(completion["entry_count"]),
        "training_launched": False,
        "bounded_p50_authorized": False,
    }
    _progress(**result)
    return result


def _collated_software_versions(loaded: Mapping[str, Any]) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(loaded["torch"].__version__),
        "numpy": str(loaded["np"].__version__),
        "rdkit": str(loaded["rdkit"].__version__),
    }


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_collated_plan_remote(
    prepared_completion_path: str,
    collated_output_prefix: str,
    entries_per_task: int,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Plan CPU-only collation over an already-published exact T1 panel."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    completion_path = _require_artifact_path(
        prepared_completion_path, field="prepared_completion_path"
    )
    runtime, _scratch, provenance = loaded["load_runtime"](
        completion_path,
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    plan = loaded["build_collated_plan"](
        runtime,
        prepared_completion_path=completion_path,
        prepared_completion_sha256=provenance["prepared_completion_sha256"],
        source_revision=_source_revision(revision),
        software_versions=_collated_software_versions(loaded),
        repo_root=REMOTE_ROOT,
        entries_per_task=int(entries_per_task),
    )
    run_root = (
        _require_physical_artifact_path(
            collated_output_prefix, field="collated_output_prefix"
        )
        / plan["run_identity_sha256"]
    )
    plan_path = run_root / loaded["COLLATED_PLAN_FILENAME"]
    loaded["write_collated_plan"](plan_path, plan)
    artifact_volume.commit()
    response = {
        "phase": "process_v2_t1_collated_plan_complete",
        "run_root": str(run_root),
        "plan_path": str(plan_path),
        "plan": plan,
        "plan_sha256": plan["plan_sha256"],
        "task_count": plan["task_count"],
        "panel_entry_count": plan["panel_entry_count"],
        "training_launched": False,
        "bounded_p50_authorized": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=1.0,
    memory=16 * 1024,
    timeout=45 * 60,
    max_containers=MAX_CPU_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def collate_t1_leaf_remote(
    prepared_completion_path: str,
    plan_path: str,
    run_root: str,
    task_identity_sha256: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Collate one bounded T1 panel slice and publish it immutably."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    runtime, scratch, provenance = loaded["load_runtime"](
        _require_artifact_path(
            prepared_completion_path, field="prepared_completion_path"
        ),
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    plan = loaded["load_collated_plan"](
        _require_artifact_path(plan_path, field="plan_path"),
        runtime=runtime,
        repo_root=REMOTE_ROOT,
    )
    if (
        plan["source_revision"] != _source_revision(revision)
        or plan["prepared_completion_sha256"]
        != provenance["prepared_completion_sha256"]
    ):
        raise RuntimeError("Process-V2 T1 collated leaf binds another input or image")
    with _heartbeat(
        "process_v2_t1_collated_leaf_heartbeat",
        interval_seconds=HEARTBEAT_SECONDS,
    ):
        receipt_path = loaded["compile_collated_leaf"](
            runtime,
            scratch.model,
            plan=plan,
            task_identity_sha256=task_identity_sha256,
            run_root=_require_physical_artifact_path(run_root, field="run_root"),
        )
    artifact_volume.commit()
    response = {
        "phase": "process_v2_t1_collated_leaf_complete",
        "task_identity_sha256": task_identity_sha256,
        "receipt_path": str(receipt_path),
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def scan_collated_remote(
    prepared_completion_path: str,
    plan_path: str,
    run_root: str,
    revision: dict[str, Any],
) -> list[str]:
    """Return only physically complete, authenticated collated tasks."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    runtime, _scratch, _provenance = loaded["load_runtime"](
        _require_artifact_path(
            prepared_completion_path, field="prepared_completion_path"
        ),
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    plan = loaded["load_collated_plan"](
        _require_artifact_path(plan_path, field="plan_path"),
        runtime=runtime,
        repo_root=REMOTE_ROOT,
    )
    return list(
        loaded["completed_collated_tasks"](
            run_root=_require_physical_artifact_path(run_root, field="run_root"),
            plan=plan,
            runtime=runtime,
        )
    )


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def finalize_collated_remote(
    prepared_completion_path: str,
    plan_path: str,
    run_root: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Reduce the complete CPU-collation receipts without recomputing rows."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    runtime, _scratch, provenance = loaded["load_runtime"](
        _require_artifact_path(
            prepared_completion_path, field="prepared_completion_path"
        ),
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    plan = loaded["load_collated_plan"](
        _require_artifact_path(plan_path, field="plan_path"),
        runtime=runtime,
        repo_root=REMOTE_ROOT,
    )
    if plan["prepared_completion_sha256"] != provenance["prepared_completion_sha256"]:
        raise RuntimeError("Process-V2 T1 collated completion binds another prepared input")
    completion_path = loaded["publish_collated_completion"](
        run_root=_require_physical_artifact_path(run_root, field="run_root"),
        plan=plan,
        runtime=runtime,
    )
    artifact_volume.commit()
    completion = _read_canonical_object(
        completion_path, label="the Process-V2 T1 collated completion"
    )
    response = {
        "phase": "process_v2_t1_collated_complete",
        "completion_path": str(completion_path),
        "completion_sha256": completion["completion_sha256"],
        "task_count": completion["task_count"],
        "panel_entry_count": completion["panel_entry_count"],
        "training_launched": False,
        "bounded_p50_authorized": False,
    }
    _progress(**response)
    return response


def _execution_environment(loaded: Mapping[str, Any], *, batch_size: int) -> dict[str, Any]:
    torch = loaded["torch"]
    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    return loaded["build_environment"](
        hardware_class="gpu",
        device_name=torch.cuda.get_device_name(device_index),
        device_capability=f"{capability[0]}.{capability[1]}",
        accelerator_class="gpu",
        dtype="float32",
        mixed_precision=False,
        batch_size=batch_size,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda),
        cudnn_version=str(torch.backends.cudnn.version()),
        rdkit_version=str(loaded["rdkit"].__version__),
        numpy_version=str(loaded["np"].__version__),
    )


def _invoke_process_v2_capacity_runner(
    run_capacity: Callable[..., Mapping[str, Any]],
    *,
    model: Any,
    runtime: Any,
    output_directory: Path,
    provenance: Mapping[str, Any],
    result_builder: Callable[..., Mapping[str, Any]],
    runner_implementation_sha256: str,
) -> Mapping[str, Any]:
    """Keep the two Process-V2 callback seams at one explicit call site."""

    return run_capacity(
        model,
        runtime,
        output_directory=output_directory,
        provenance=provenance,
        result_builder=result_builder,
        result_filename=PROCESS_V2_RESULT_FILENAME,
        expected_runner_implementation_sha256=runner_implementation_sha256,
    )


@app.function(
    image=image,
    gpu=PUBLICATION_RECOVERY_GPU,
    cpu=8.0,
    memory=64 * 1024,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_t1_gpu_remote(
    prepared_completion_path: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Run one hazard-free joint scratch-model T1 job and stop before P50."""

    _validate_remote_revision(revision)
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "Process-V2 T1 requires deterministic cuBLAS before materialization"
        )
    artifact_volume.reload()
    loaded = _imports()
    runtime, scratch, provenance = loaded["load_runtime"](
        _require_artifact_path(prepared_completion_path, field="prepared_completion_path"),
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    torch = loaded["torch"]
    model = scratch.model.to(device="cuda", dtype=torch.float32)
    environment = _execution_environment(
        loaded, batch_size=int(runtime.capacity_policy["optimization"]["batch_size"])
    )
    runner_hash = loaded["runner_implementation_sha256"](repo_root=REMOTE_ROOT)
    source_revision = _source_revision(revision)
    if dict(runtime.prepared.artifact["source_revision"]) != source_revision:
        raise RuntimeError("Process-V2 T1 prepared inputs bind another image revision")
    provenance = loaded["project_result_provenance"](
        {
            **dict(provenance),
            "runner_source_revision_sha256": source_revision["source_revision_sha256"],
            "runner_implementation_sha256": runner_hash,
            "execution_environment": environment,
        }
    )
    run_identity_sha256 = _sha256(
        {
            "prepared_completion_sha256": provenance["prepared_completion_sha256"],
            "capacity_policy_sha256": provenance["capacity_policy_sha256"],
            "runner_source_revision_sha256": provenance["runner_source_revision_sha256"],
            "runner_implementation_sha256": runner_hash,
            "execution_environment_sha256": environment["environment_sha256"],
        }
    )
    run_root = (
        _require_physical_artifact_path(output_prefix, field="output_prefix") / run_identity_sha256
    )
    builder = functools.partial(loaded["build_result"], capacity_policy=runtime.capacity_policy)
    _progress(
        "process_v2_t1_gpu_start",
        run_root=str(run_root),
        maximum_optimizer_steps=int(
            runtime.capacity_policy["optimization"]["maximum_optimizer_steps"]
        ),
        prepared_entry_count=len(runtime.entries),
    )
    with _gpu_checkpoint_heartbeat(run_root):
        run = _invoke_process_v2_capacity_runner(
            loaded["run_capacity"],
            model=model,
            runtime=runtime,
            output_directory=run_root,
            provenance=provenance,
            result_builder=builder,
            runner_implementation_sha256=runner_hash,
        )
    result = loaded["validate_result"](run["result"], capacity_policy=runtime.capacity_policy)
    if loaded["RESULT_FILENAME"] != PROCESS_V2_RESULT_FILENAME:
        raise RuntimeError("Process-V2 T1 result filename contract changed")
    result_path = Path(run["result_path"])
    if result_path != run_root / PROCESS_V2_RESULT_FILENAME:
        raise RuntimeError("Process-V2 T1 runner published the result at another path")
    result_file_sha256 = _file_sha256(result_path)
    decision = loaded["build_decision"](
        result,
        capacity_policy=runtime.capacity_policy,
        result_file_sha256=result_file_sha256,
    )
    decision_path = run_root / loaded["DECISION_FILENAME"]
    loaded["write_bytes_if_absent"](
        decision_path,
        _canonical_bytes(decision) + b"\n",
    )
    reopened_decision = _read_canonical_object(
        decision_path, label="the Process-V2 T1 capacity decision"
    )
    decision = loaded["validate_decision"](
        reopened_decision,
        result=result,
        capacity_policy=runtime.capacity_policy,
        result_file_sha256=result_file_sha256,
    )
    completed_steps = int(result["run_integrity"]["optimizer_steps_completed"])
    step_ten_path = run_root / "checkpoints" / STEP_TEN_CHECKPOINT_FILENAME
    if completed_steps >= 10 and not step_ten_path.is_file():
        raise RuntimeError("Process-V2 T1 completed step 10 without its durable checkpoint")
    step_ten_receipts = [
        dict(receipt)
        for receipt in run["recovery_checkpoints"]
        if int(receipt["completed_steps"]) == 10
    ]
    if completed_steps >= 10 and (
        len(step_ten_receipts) != 1
        or step_ten_receipts[0]["path"] != str(step_ten_path)
        or step_ten_receipts[0]["file_sha256"] != _file_sha256(step_ten_path)
    ):
        raise RuntimeError("Process-V2 T1 step-10 checkpoint receipt disagrees")
    artifact_volume.commit()
    response = {
        "phase": "process_v2_t1_gpu_complete",
        "run_root": str(run_root),
        "result_path": str(result_path),
        "result_file_sha256": result_file_sha256,
        "result_sha256": result["result_sha256"],
        "decision_path": str(decision_path),
        "decision_sha256": decision["decision_sha256"],
        "decision_status": decision["status"],
        "selected_checkpoint_path": str(run["selected_checkpoint_path"]),
        "selected_checkpoint_file_sha256": run["selected_checkpoint_file_sha256"],
        "step_ten_checkpoint_path": (None if completed_steps < 10 else str(step_ten_path)),
        "step_ten_checkpoint_file_sha256": (
            None if completed_steps < 10 else _file_sha256(step_ten_path)
        ),
        "step_ten_checkpoint_receipt": (None if completed_steps < 10 else step_ten_receipts[0]),
        "optimizer_steps_completed": completed_steps,
        "threshold_checks": result["threshold_checks"],
        "all_threshold_checks_pass": all(result["threshold_checks"].values()),
        "image_revision_sha256": revision["image_revision_sha256"],
        "training_launched": True,
        "bounded_p50_authorized": decision["bounded_p50_authorized"],
        "p50_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=64 * 1024,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_t1_collated_gpu_remote(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Run T1 from authenticated CPU-collated tensors with no GPU collation."""

    _validate_remote_revision(revision)
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError("Process-V2 T1 requires deterministic cuBLAS")
    artifact_volume.reload()
    loaded = _imports()
    prepared_path = _require_artifact_path(
        prepared_completion_path, field="prepared_completion_path"
    )
    runtime, scratch, provenance = loaded["load_runtime"](
        prepared_path,
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    plan = loaded["load_collated_plan"](
        _require_artifact_path(collated_plan_path, field="collated_plan_path"),
        runtime=runtime,
        repo_root=REMOTE_ROOT,
    )
    source_revision = _source_revision(revision)
    if (
        plan["source_revision"] != source_revision
        or plan["prepared_completion_path"] != str(prepared_path)
        or plan["prepared_completion_sha256"]
        != provenance["prepared_completion_sha256"]
    ):
        raise RuntimeError("Process-V2 T1 collated cache binds another input or image")
    collated_path = _require_artifact_path(
        collated_completion_path, field="collated_completion_path"
    )
    materialized_panel = loaded["load_materialized_collated_panel"](
        collated_path,
        plan=plan,
        runtime=runtime,
    )
    collated_completion = _read_canonical_object(
        collated_path, label="the Process-V2 T1 collated completion"
    )
    torch = loaded["torch"]
    model = scratch.model.to(device="cuda", dtype=torch.float32)
    environment = _execution_environment(
        loaded, batch_size=int(runtime.capacity_policy["optimization"]["batch_size"])
    )
    runner_hash = loaded["runner_implementation_sha256"](repo_root=REMOTE_ROOT)
    provenance = loaded["project_result_provenance"](
        {
            **dict(provenance),
            "runner_source_revision_sha256": source_revision["source_revision_sha256"],
            "runner_implementation_sha256": runner_hash,
            "execution_environment": environment,
        }
    )
    run_identity_sha256 = _sha256(
        {
            "prepared_completion_sha256": provenance["prepared_completion_sha256"],
            "collated_completion_sha256": collated_completion["completion_sha256"],
            "capacity_policy_sha256": provenance["capacity_policy_sha256"],
            "runner_source_revision_sha256": provenance[
                "runner_source_revision_sha256"
            ],
            "runner_implementation_sha256": runner_hash,
            "execution_environment_sha256": environment["environment_sha256"],
        }
    )
    run_root = (
        _require_physical_artifact_path(output_prefix, field="output_prefix")
        / run_identity_sha256
    )
    builder = functools.partial(
        loaded["build_result"], capacity_policy=runtime.capacity_policy
    )
    _progress(
        "process_v2_t1_collated_gpu_start",
        run_root=str(run_root),
        maximum_optimizer_steps=int(
            runtime.capacity_policy["optimization"]["maximum_optimizer_steps"]
        ),
        prepared_entry_count=len(runtime.entries),
        collated_task_count=collated_completion["task_count"],
        gpu_side_collation_count=0,
    )
    with _gpu_checkpoint_heartbeat(run_root):
        run = loaded["run_capacity"](
            model,
            runtime,
            output_directory=run_root,
            provenance=provenance,
            materialized_panel=materialized_panel,
            result_builder=builder,
            result_filename=PROCESS_V2_RESULT_FILENAME,
            expected_runner_implementation_sha256=runner_hash,
            expected_runner_source_revision_sha256=source_revision[
                "source_revision_sha256"
            ],
        )
    result = loaded["validate_result"](
        run["result"], capacity_policy=runtime.capacity_policy
    )
    if loaded["RESULT_FILENAME"] != PROCESS_V2_RESULT_FILENAME:
        raise RuntimeError("Process-V2 T1 result filename contract changed")
    result_path = Path(run["result_path"])
    if result_path != run_root / PROCESS_V2_RESULT_FILENAME:
        raise RuntimeError("Process-V2 T1 runner published the result at another path")
    result_file_sha256 = _file_sha256(result_path)
    decision = loaded["build_decision"](
        result,
        capacity_policy=runtime.capacity_policy,
        result_file_sha256=result_file_sha256,
    )
    decision_path = run_root / loaded["DECISION_FILENAME"]
    loaded["write_bytes_if_absent"](
        decision_path, _canonical_bytes(decision) + b"\n"
    )
    decision = loaded["validate_decision"](
        _read_canonical_object(
            decision_path, label="the Process-V2 T1 capacity decision"
        ),
        result=result,
        capacity_policy=runtime.capacity_policy,
        result_file_sha256=result_file_sha256,
    )
    completed_steps = int(result["run_integrity"]["optimizer_steps_completed"])
    step_ten_path = run_root / "checkpoints" / STEP_TEN_CHECKPOINT_FILENAME
    step_ten_receipts = [
        dict(receipt)
        for receipt in run["recovery_checkpoints"]
        if int(receipt["completed_steps"]) == 10
    ]
    if completed_steps >= 10 and (
        not step_ten_path.is_file()
        or len(step_ten_receipts) != 1
        or step_ten_receipts[0]["path"] != str(step_ten_path)
        or step_ten_receipts[0]["file_sha256"] != _file_sha256(step_ten_path)
    ):
        raise RuntimeError("Process-V2 T1 step-10 checkpoint receipt disagrees")
    artifact_volume.commit()
    response = {
        "phase": "process_v2_t1_collated_gpu_complete",
        "run_root": str(run_root),
        "result_path": str(result_path),
        "result_file_sha256": result_file_sha256,
        "result_sha256": result["result_sha256"],
        "decision_path": str(decision_path),
        "decision_sha256": decision["decision_sha256"],
        "decision_status": decision["status"],
        "selected_checkpoint_path": str(run["selected_checkpoint_path"]),
        "selected_checkpoint_file_sha256": run[
            "selected_checkpoint_file_sha256"
        ],
        "step_ten_checkpoint_path": (
            None if completed_steps < 10 else str(step_ten_path)
        ),
        "optimizer_steps_completed": completed_steps,
        "threshold_checks": result["threshold_checks"],
        "all_threshold_checks_pass": all(result["threshold_checks"].values()),
        "collated_completion_path": str(collated_path),
        "collated_completion_sha256": collated_completion["completion_sha256"],
        "gpu_side_collation_count": 0,
        "image_revision_sha256": revision["image_revision_sha256"],
        "training_launched": True,
        "bounded_p50_authorized": decision["bounded_p50_authorized"],
        "p50_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=64 * 1024,
    timeout=SCORE_REVISION_REPAIR_GPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_t1_score_revision_repair_remote(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    output_prefix: str,
    revision: dict[str, Any],
    repair_families: tuple[str, ...] = _SCORE_REVISION_REPAIR_FAMILIES,
) -> dict[str, Any]:
    """Run an exact nonempty subset of unresolved T1 families from cached tensors."""

    _validate_remote_revision(revision)
    if (
        not repair_families
        or len(set(repair_families)) != len(repair_families)
        or tuple(
            family
            for family in _SCORE_REVISION_REPAIR_FAMILIES
            if family in repair_families
        )
        != repair_families
    ):
        raise ValueError(
            "repair_families must be a nonempty canonical-order subset of "
            "the score-revision repair families"
        )
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError("Process-V2 T1 score repair requires deterministic cuBLAS")
    artifact_volume.reload()
    loaded = _imports()
    prepared_path = _require_artifact_path(
        prepared_completion_path, field="prepared_completion_path"
    )
    runtime, scratch, provenance = loaded["load_runtime"](
        prepared_path,
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
        score_revision_rebind=True,
    )
    bridge = provenance.pop("score_revision_rebind", None)
    if not isinstance(bridge, Mapping):
        raise RuntimeError("Process-V2 T1 score-revision bridge identity is absent")
    plan = loaded["load_collated_plan"](
        _require_artifact_path(collated_plan_path, field="collated_plan_path"),
        runtime=runtime,
        repo_root=REMOTE_ROOT,
        score_revision_rebind=True,
    )
    collated_path = _require_artifact_path(
        collated_completion_path, field="collated_completion_path"
    )
    materialized_panel = loaded["load_materialized_collated_panel"](
        collated_path,
        plan=plan,
        runtime=runtime,
        score_revision_rebind=True,
    )
    collated_completion = _read_canonical_object(
        collated_path, label="the Process-V2 T1 collated completion"
    )
    policy = runtime.capacity_policy
    if tuple(
        family for family in policy["required_families"] if family in repair_families
    ) != repair_families:
        raise RuntimeError("Process-V2 T1 repair family order disagrees with the policy")

    source_revision = _source_revision(revision)
    identity = {
        "score_revision_rebind_sha256": bridge["score_revision_rebind_sha256"],
        "collated_completion_sha256": collated_completion["completion_sha256"],
        "capacity_policy_sha256": policy["policy_sha256"],
        "initial_model_state_sha256": provenance["initial_model_state_sha256"],
        "predecessor_capacity_result_file_sha256": (
            _PREDECESSOR_CAPACITY_RESULT_FILE_SHA256
        ),
        "predecessor_capacity_result_sha256": _PREDECESSOR_CAPACITY_RESULT_SHA256,
        "atom_insert_pass_result_sha256": _ATOM_INSERT_PASS_RESULT_SHA256,
        "repair_families": list(repair_families),
        "runner_source_revision_sha256": source_revision[
            "source_revision_sha256"
        ],
    }
    run_root = (
        _require_physical_artifact_path(output_prefix, field="output_prefix")
        / _sha256(identity)
    )
    _progress(
        "process_v2_t1_score_revision_repair_start",
        run_root=str(run_root),
        families=list(repair_families),
        gpu_containers=1,
        fiber_recomputation_count=0,
        gpu_side_collation_count=0,
    )

    family_results: list[dict[str, Any]] = []
    for family in repair_families:
        family_root = run_root / family

        def report(row: Mapping[str, Any], *, _family: str = family) -> None:
            _progress(
                "process_v2_t1_score_revision_repair_heartbeat",
                family=_family,
                **dict(row),
            )

        heads_model = copy.deepcopy(scratch.model).to(
            device="cuda", dtype=loaded["torch"].float32
        )
        heads_result = loaded["run_failure_scope"](
            heads_model,
            materialized_panel,
            family=family,
            failing_families=repair_families,
            capacity_policy=policy,
            input_capacity_result_file_sha256=(
                _PREDECESSOR_CAPACITY_RESULT_FILE_SHA256
            ),
            input_capacity_result_sha256=_PREDECESSOR_CAPACITY_RESULT_SHA256,
            collated_completion_sha256=collated_completion["completion_sha256"],
            initial_model_state_sha256=provenance["initial_model_state_sha256"],
            progress=report,
        )
        heads_path = family_root / "heads_only" / FAILURE_SCOPE_RESULT_FILENAME
        loaded["write_bytes_if_absent"](
            heads_path, _canonical_bytes(heads_result) + b"\n"
        )
        selected = heads_result
        selected_path = heads_path
        if not heads_result["diagnostic_passed"]:
            next_model = copy.deepcopy(scratch.model).to(
                device="cuda", dtype=loaded["torch"].float32
            )
            selected = loaded["run_next_failure_scope"](
                next_model,
                materialized_panel,
                family=family,
                failing_families=repair_families,
                capacity_policy=policy,
                prior_scope_result=heads_result,
                prior_scope_result_file_sha256=_file_sha256(heads_path),
                input_capacity_result_file_sha256=(
                    _PREDECESSOR_CAPACITY_RESULT_FILE_SHA256
                ),
                input_capacity_result_sha256=_PREDECESSOR_CAPACITY_RESULT_SHA256,
                collated_completion_sha256=collated_completion[
                    "completion_sha256"
                ],
                initial_model_state_sha256=provenance[
                    "initial_model_state_sha256"
                ],
                progress=report,
            )
            selected_path = (
                family_root / selected["scope"] / FAILURE_SCOPE_RESULT_FILENAME
            )
            loaded["write_bytes_if_absent"](
                selected_path, _canonical_bytes(selected) + b"\n"
            )
        family_results.append(
            {
                "family": family,
                "heads_result_sha256": heads_result["result_sha256"],
                "selected_scope": selected["scope"],
                "selected_result_path": str(selected_path),
                "selected_result_file_sha256": _file_sha256(selected_path),
                "selected_result_sha256": selected["result_sha256"],
                "diagnostic_passed": selected["diagnostic_passed"],
            }
        )
        del heads_model
        if "next_model" in locals():
            del next_model

    passed = all(row["diagnostic_passed"] for row in family_results)
    body = {
        "schema": "compose.editing_v2.process_v2_t1_score_revision_repair",
        "schema_version": 1,
        "status": (
            "PASS_PROCESS_V2_T1_SCORE_REVISION_REPAIR_NO_DOWNSTREAM_AUTHORITY"
            if passed
            else "FAIL_PROCESS_V2_T1_SCORE_REVISION_REPAIR_NO_DOWNSTREAM_AUTHORITY"
        ),
        **loaded["NO_AUTHORITY"],
        **identity,
        "score_revision_rebind": dict(bridge),
        "prepared_completion_path": str(prepared_path),
        "collated_completion_path": str(collated_path),
        "family_results": family_results,
        "all_repair_families_passed": passed,
        "fiber_recomputation_count": 0,
        "gpu_side_collation_count": 0,
        "p50_launched": False,
    }
    result = {**body, "repair_sha256": _sha256(body)}
    result_path = run_root / SCORE_REVISION_REPAIR_FILENAME
    loaded["write_bytes_if_absent"](
        result_path, _canonical_bytes(result) + b"\n"
    )
    artifact_volume.commit()
    response = {
        "phase": "process_v2_t1_score_revision_repair_complete",
        "run_root": str(run_root),
        "result_path": str(result_path),
        "repair_sha256": result["repair_sha256"],
        "family_results": family_results,
        "all_repair_families_passed": passed,
        "fiber_recomputation_count": 0,
        "gpu_side_collation_count": 0,
        "bounded_p50_authorized": False,
        "p50_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=64 * 1024,
    timeout=SCORE_REVISION_REPAIR_GPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_t1_final_scope_repair_remote(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    repair_result_path: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Run only the missing ``all`` rung for two failed local-adapter arms."""

    _validate_remote_revision(revision)
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError("Process-V2 T1 final scope requires deterministic cuBLAS")
    artifact_volume.reload()
    loaded = _imports()
    prepared_path = _require_artifact_path(
        prepared_completion_path, field="prepared_completion_path"
    )
    runtime, scratch, provenance = loaded["load_runtime"](
        prepared_path,
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
        score_revision_rebind=True,
    )
    bridge = provenance.pop("score_revision_rebind", None)
    if not isinstance(bridge, Mapping):
        raise RuntimeError("Process-V2 T1 final scope lost the score-revision bridge")
    plan = loaded["load_collated_plan"](
        _require_artifact_path(collated_plan_path, field="collated_plan_path"),
        runtime=runtime,
        repo_root=REMOTE_ROOT,
        score_revision_rebind=True,
    )
    collated_path = _require_artifact_path(
        collated_completion_path, field="collated_completion_path"
    )
    materialized_panel = loaded["load_materialized_collated_panel"](
        collated_path,
        plan=plan,
        runtime=runtime,
        score_revision_rebind=True,
    )
    collated_completion = _read_canonical_object(
        collated_path, label="the Process-V2 T1 collated completion"
    )
    repair_path = _require_artifact_path(
        repair_result_path, field="repair_result_path"
    )
    repair = _read_canonical_object(
        repair_path, label="the Process-V2 T1 score-revision repair"
    )
    if (
        _file_sha256(repair_path) != _SCORE_REVISION_REPAIR_RESULT_FILE_SHA256
        or repair.get("repair_sha256") != _SCORE_REVISION_REPAIR_RESULT_SHA256
        or repair.get("schema")
        != "compose.editing_v2.process_v2_t1_score_revision_repair"
        or repair.get("status")
        != "FAIL_PROCESS_V2_T1_SCORE_REVISION_REPAIR_NO_DOWNSTREAM_AUTHORITY"
        or repair.get("all_repair_families_passed") is not False
        or repair.get("initial_model_state_sha256")
        != provenance["initial_model_state_sha256"]
        or repair.get("collated_completion_sha256")
        != collated_completion["completion_sha256"]
        or repair.get("capacity_policy_sha256")
        != runtime.capacity_policy["policy_sha256"]
        or any(repair.get(name) is not False for name in loaded["NO_AUTHORITY"])
    ):
        raise RuntimeError("Process-V2 T1 final scope repair predecessor disagrees")
    family_rows = {
        str(row["family"]): dict(row)
        for row in repair.get("family_results", ())
        if isinstance(row, Mapping) and "family" in row
    }
    if set(family_rows) != set(_SCORE_REVISION_REPAIR_FAMILIES):
        raise RuntimeError("Process-V2 T1 repair predecessor lost a family")

    source_revision = _source_revision(revision)
    identity = {
        "repair_result_file_sha256": _SCORE_REVISION_REPAIR_RESULT_FILE_SHA256,
        "repair_result_sha256": _SCORE_REVISION_REPAIR_RESULT_SHA256,
        "collated_completion_sha256": collated_completion["completion_sha256"],
        "capacity_policy_sha256": runtime.capacity_policy["policy_sha256"],
        "initial_model_state_sha256": provenance["initial_model_state_sha256"],
        "score_revision_rebind_sha256": bridge["score_revision_rebind_sha256"],
        "families": list(_FINAL_SCOPE_REPAIR_FAMILIES),
        "scope": "all",
        "runner_source_revision_sha256": source_revision[
            "source_revision_sha256"
        ],
    }
    run_root = (
        _require_physical_artifact_path(output_prefix, field="output_prefix")
        / _sha256(identity)
    )
    _progress(
        "process_v2_t1_final_scope_repair_start",
        run_root=str(run_root),
        families=list(_FINAL_SCOPE_REPAIR_FAMILIES),
        scope="all",
        gpu_containers=1,
        fiber_recomputation_count=0,
        gpu_side_collation_count=0,
    )

    family_results: list[dict[str, Any]] = []
    for family in _FINAL_SCOPE_REPAIR_FAMILIES:
        row = family_rows[family]
        expected = _FINAL_SCOPE_PREDECESSORS[family]
        prior_path = _require_artifact_path(
            str(row["selected_result_path"]),
            field=f"{family}_prior_scope_result_path",
        )
        prior = loaded["validate_failure_scope"](
            _read_canonical_object(
                prior_path,
                label=f"the Process-V2 T1 {family} local-adapter result",
            )
        )
        if (
            row.get("selected_scope") != "heads_plus_local_adapter"
            or row.get("diagnostic_passed") is not False
            or row.get("selected_result_file_sha256") != expected["file_sha256"]
            or row.get("selected_result_sha256") != expected["result_sha256"]
            or _file_sha256(prior_path) != expected["file_sha256"]
            or prior["result_sha256"] != expected["result_sha256"]
            or prior["family"] != family
            or prior["scope"] != "heads_plus_local_adapter"
            or prior["diagnostic_passed"] is not False
            or prior["initial_model_state_sha256"]
            != provenance["initial_model_state_sha256"]
            or prior["collated_completion_sha256"]
            != collated_completion["completion_sha256"]
        ):
            raise RuntimeError(f"Process-V2 T1 {family} predecessor disagrees")

        result_path = run_root / family / "all" / FAILURE_SCOPE_RESULT_FILENAME
        if result_path.is_file():
            result = loaded["validate_failure_scope"](
                _read_canonical_object(
                    result_path,
                    label=f"the Process-V2 T1 {family} all-scope result",
                )
            )
            reused = True
        else:
            model = copy.deepcopy(scratch.model).to(
                device="cuda", dtype=loaded["torch"].float32
            )

            def report(metric: Mapping[str, Any], *, _family: str = family) -> None:
                _progress(
                    "process_v2_t1_final_scope_repair_heartbeat",
                    family=_family,
                    scope="all",
                    **dict(metric),
                )

            result = loaded["run_final_failure_scope"](
                model,
                materialized_panel,
                family=family,
                failing_families=_FINAL_SCOPE_REPAIR_FAMILIES,
                capacity_policy=runtime.capacity_policy,
                prior_scope_result=prior,
                prior_scope_result_file_sha256=expected["file_sha256"],
                input_capacity_result_file_sha256=(
                    _PREDECESSOR_CAPACITY_RESULT_FILE_SHA256
                ),
                input_capacity_result_sha256=_PREDECESSOR_CAPACITY_RESULT_SHA256,
                collated_completion_sha256=collated_completion["completion_sha256"],
                initial_model_state_sha256=provenance[
                    "initial_model_state_sha256"
                ],
                progress=report,
            )
            loaded["write_bytes_if_absent"](
                result_path, _canonical_bytes(result) + b"\n"
            )
            artifact_volume.commit()
            del model
            reused = False
        family_results.append(
            {
                "family": family,
                "scope": "all",
                "result_path": str(result_path),
                "result_file_sha256": _file_sha256(result_path),
                "result_sha256": result["result_sha256"],
                "diagnostic_passed": result["diagnostic_passed"],
                "reused": reused,
            }
        )

    passed = all(row["diagnostic_passed"] for row in family_results)
    body = {
        "schema": "compose.editing_v2.process_v2_t1_final_scope_repair",
        "schema_version": 1,
        "status": (
            "PASS_PROCESS_V2_T1_FINAL_SCOPE_REPAIR_NO_DOWNSTREAM_AUTHORITY"
            if passed
            else "FAIL_PROCESS_V2_T1_FINAL_SCOPE_REPAIR_NO_DOWNSTREAM_AUTHORITY"
        ),
        **loaded["NO_AUTHORITY"],
        **identity,
        "repair_result_path": str(repair_path),
        "family_results": family_results,
        "all_final_scope_families_passed": passed,
        "fiber_recomputation_count": 0,
        "gpu_side_collation_count": 0,
        "p50_launched": False,
    }
    result = {**body, "final_scope_repair_sha256": _sha256(body)}
    wrapper_path = run_root / FINAL_SCOPE_REPAIR_FILENAME
    loaded["write_bytes_if_absent"](
        wrapper_path, _canonical_bytes(result) + b"\n"
    )
    artifact_volume.commit()
    response = {
        "phase": "process_v2_t1_final_scope_repair_complete",
        "run_root": str(run_root),
        "result_path": str(wrapper_path),
        "family_results": family_results,
        "all_final_scope_families_passed": passed,
        "fiber_recomputation_count": 0,
        "gpu_side_collation_count": 0,
        "bounded_p50_authorized": False,
        "p50_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=1.0,
    memory=4 * 1024,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def failure_scope_driver(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    capacity_result_path: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Derive the failing families and run their independent arms in parallel."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    policy, _policy_file_sha256 = loaded["load_capacity_policy"](
        REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    result_path = _require_artifact_path(
        capacity_result_path, field="capacity_result_path"
    )
    capacity_result = loaded["validate_result"](
        _read_canonical_object(
            result_path, label="the Process-V2 T1 capacity result"
        ),
        capacity_policy=policy,
    )
    families = loaded["failing_families"](
        capacity_result, capacity_policy=policy
    )
    run_t1_failure_scope_gpu_remote.update_autoscaler(
        max_containers=len(families)
    )
    _progress(
        "process_v2_t1_failure_scope_parallel_plan",
        families=list(families),
        gpu_containers=len(families),
        fiber_recomputation_count=0,
    )
    results = list(
        run_t1_failure_scope_gpu_remote.starmap(
            [
                (
                    prepared_completion_path,
                    collated_plan_path,
                    collated_completion_path,
                    capacity_result_path,
                    family,
                    output_prefix,
                    revision,
                )
                for family in families
            ]
        )
    )
    by_family = {str(item["family"]): dict(item) for item in results}
    if set(by_family) != set(families) or len(results) != len(families):
        raise RuntimeError("Process-V2 T1 failure-scope fanout lost a family")
    response = {
        "phase": "process_v2_t1_failure_scope_complete",
        "scope": "heads_only",
        "families": list(families),
        "gpu_containers": len(families),
        "results": [by_family[family] for family in families],
        "all_diagnostics_passed": all(
            by_family[family]["diagnostic_passed"] for family in families
        ),
        "fiber_recomputation_count": 0,
        "gpu_side_collation_count": 0,
        "image_revision_sha256": revision["image_revision_sha256"],
        "bounded_p50_authorized": False,
        "p50_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=1.0,
    memory=4 * 1024,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def next_failure_scope_driver(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    capacity_result_path: str,
    prior_scope_result_paths: dict[str, str],
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Run the first applicable post-heads scope for each failed family."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    policy, _policy_file_sha256 = loaded["load_capacity_policy"](
        REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    result_path = _require_artifact_path(
        capacity_result_path, field="capacity_result_path"
    )
    capacity_result = loaded["validate_result"](
        _read_canonical_object(
            result_path, label="the Process-V2 T1 capacity result"
        ),
        capacity_policy=policy,
    )
    families = loaded["failing_families"](
        capacity_result, capacity_policy=policy
    )
    if set(prior_scope_result_paths) != set(families) or any(
        not isinstance(path, str) or not path
        for path in prior_scope_result_paths.values()
    ):
        raise RuntimeError(
            "Process-V2 T1 next-scope predecessor paths do not match the failures"
        )
    scopes = {
        family: loaded["next_failure_scope"](
            family, capacity_policy=policy
        )
        for family in families
    }
    # This post-heads diagnostic is intentionally serialized onto one A10.
    # Each starmap item remains independently content-addressed and restartable,
    # while the single-container ceiling bounds concurrent GPU spend.
    run_t1_failure_scope_gpu_remote.update_autoscaler(max_containers=1)
    _progress(
        "process_v2_t1_next_failure_scope_sequential_plan",
        families=list(families),
        scopes=scopes,
        gpu_containers=1,
        execution_mode="one_gpu_sequential_arms",
        fiber_recomputation_count=0,
    )
    results = list(
        run_t1_failure_scope_gpu_remote.starmap(
            [
                (
                    prepared_completion_path,
                    collated_plan_path,
                    collated_completion_path,
                    capacity_result_path,
                    family,
                    output_prefix,
                    revision,
                    scopes[family],
                    prior_scope_result_paths[family],
                )
                for family in families
            ]
        )
    )
    by_family = {str(item["family"]): dict(item) for item in results}
    if set(by_family) != set(families) or len(results) != len(families):
        raise RuntimeError("Process-V2 T1 next-scope fanout lost a family")
    response = {
        "phase": "process_v2_t1_next_failure_scope_complete",
        "scopes": scopes,
        "families": list(families),
        "gpu_containers": 1,
        "execution_mode": "one_gpu_sequential_arms",
        "results": [by_family[family] for family in families],
        "all_diagnostics_passed": all(
            by_family[family]["diagnostic_passed"] for family in families
        ),
        "fiber_recomputation_count": 0,
        "gpu_side_collation_count": 0,
        "image_revision_sha256": revision["image_revision_sha256"],
        "bounded_p50_authorized": False,
        "p50_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=64 * 1024,
    timeout=FAILURE_SCOPE_GPU_TIMEOUT_SECONDS,
    max_containers=4,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_t1_failure_scope_gpu_remote(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    capacity_result_path: str,
    family: str,
    output_prefix: str,
    revision: dict[str, Any],
    scope: str = "heads_only",
    prior_scope_result_path: str | None = None,
) -> dict[str, Any]:
    """Run one independent scope arm for a family that failed T1's tail.

    The collated cache is an authenticated immutable input produced by its own
    source revision.  This diagnostic binds that upstream revision separately
    from its current runner revision, and never rebuilds a fiber or retrains a
    family whose aggregate and entry checks already passed.
    """

    _validate_remote_revision(revision)
    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError("Process-V2 T1 failure scope requires deterministic cuBLAS")
    artifact_volume.reload()
    loaded = _imports()
    prepared_path = _require_artifact_path(
        prepared_completion_path, field="prepared_completion_path"
    )
    runtime, scratch, provenance = loaded["load_runtime"](
        prepared_path,
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    plan_path = _require_artifact_path(collated_plan_path, field="collated_plan_path")
    plan = loaded["load_collated_plan"](
        plan_path,
        runtime=runtime,
        repo_root=REMOTE_ROOT,
    )
    if (
        plan["prepared_completion_path"] != str(prepared_path)
        or plan["prepared_completion_sha256"]
        != provenance["prepared_completion_sha256"]
    ):
        raise RuntimeError("Process-V2 T1 failure scope cache binds another input")
    collated_path = _require_artifact_path(
        collated_completion_path, field="collated_completion_path"
    )
    materialized_panel = loaded["load_materialized_collated_panel"](
        collated_path,
        plan=plan,
        runtime=runtime,
    )
    collated_completion = _read_canonical_object(
        collated_path, label="the Process-V2 T1 collated completion"
    )
    capacity_path = _require_artifact_path(
        capacity_result_path, field="capacity_result_path"
    )
    capacity_result = loaded["validate_result"](
        _read_canonical_object(
            capacity_path, label="the Process-V2 T1 capacity result"
        ),
        capacity_policy=runtime.capacity_policy,
    )
    capacity_result_file_sha256 = _file_sha256(capacity_path)
    if (
        capacity_result["provenance"]["prepared_completion_sha256"]
        != provenance["prepared_completion_sha256"]
        or capacity_result["provenance"]["initial_model_state_sha256"]
        != provenance["initial_model_state_sha256"]
        or capacity_result["provenance"]["capacity_policy_sha256"]
        != provenance["capacity_policy_sha256"]
    ):
        raise RuntimeError("Process-V2 T1 failure scope result binds another run")
    failing_families = loaded["failing_families"](
        capacity_result, capacity_policy=runtime.capacity_policy
    )
    if family not in failing_families:
        raise RuntimeError("Process-V2 T1 failure scope received a passing family")
    prior_scope_result: dict[str, Any] | None = None
    prior_scope_result_file_sha256: str | None = None
    expected_scope = "heads_only"
    if prior_scope_result_path is not None:
        expected_scope = loaded["next_failure_scope"](
            family, capacity_policy=runtime.capacity_policy
        )
        prior_path = _require_artifact_path(
            prior_scope_result_path, field="prior_scope_result_path"
        )
        prior_scope_result = loaded["validate_failure_scope"](
            _read_canonical_object(
                prior_path,
                label=f"the Process-V2 T1 {family} prior-scope result",
            )
        )
        prior_scope_result_file_sha256 = _file_sha256(prior_path)
    if scope != expected_scope or (
        scope == "heads_only" and prior_scope_result_path is not None
    ) or (scope != "heads_only" and prior_scope_result_path is None):
        raise RuntimeError("Process-V2 T1 failure-scope ladder was skipped")
    source_revision = _source_revision(revision)
    identity = {
        "capacity_result_file_sha256": capacity_result_file_sha256,
        "capacity_result_sha256": capacity_result["result_sha256"],
        "collated_completion_sha256": collated_completion["completion_sha256"],
        "capacity_policy_sha256": provenance["capacity_policy_sha256"],
        "initial_model_state_sha256": provenance["initial_model_state_sha256"],
        "scope": scope,
        "family": family,
        "prior_scope_result_file_sha256": prior_scope_result_file_sha256,
        "prior_scope_result_sha256": (
            None
            if prior_scope_result is None
            else prior_scope_result["result_sha256"]
        ),
        "runner_source_revision_sha256": source_revision["source_revision_sha256"],
    }
    run_root = (
        _require_physical_artifact_path(output_prefix, field="output_prefix")
        / _sha256(identity)
    )
    _progress(
        "process_v2_t1_failure_scope_start",
        run_root=str(run_root),
        family=family,
        scope=scope,
        fiber_recomputation_count=0,
        gpu_side_collation_count=0,
    )
    result_path = run_root / FAILURE_SCOPE_RESULT_FILENAME
    if result_path.is_file():
        result = loaded["validate_failure_scope"](
            _read_canonical_object(
                result_path,
                label=f"the Process-V2 T1 {family} failure-scope result",
            )
        )
        reused = True
    else:
        model = scratch.model.to(device="cuda", dtype=loaded["torch"].float32)

        def report(row: Mapping[str, Any]) -> None:
            _progress(
                "process_v2_t1_failure_scope_heartbeat",
                family=family,
                scope=scope,
                **dict(row),
            )

        common = {
            "family": family,
            "failing_families": failing_families,
            "capacity_policy": runtime.capacity_policy,
            "input_capacity_result_file_sha256": capacity_result_file_sha256,
            "input_capacity_result_sha256": capacity_result["result_sha256"],
            "collated_completion_sha256": collated_completion["completion_sha256"],
            "initial_model_state_sha256": provenance["initial_model_state_sha256"],
            "progress": report,
        }
        if scope == "heads_only":
            result = loaded["run_failure_scope"](
                model,
                materialized_panel,
                **common,
            )
        else:
            if prior_scope_result is None or prior_scope_result_file_sha256 is None:
                raise RuntimeError("Process-V2 T1 next scope lost its predecessor")
            result = loaded["run_next_failure_scope"](
                model,
                materialized_panel,
                prior_scope_result=prior_scope_result,
                prior_scope_result_file_sha256=prior_scope_result_file_sha256,
                **common,
            )
        loaded["write_bytes_if_absent"](
            result_path, _canonical_bytes(result) + b"\n"
        )
        artifact_volume.commit()
        reused = False
    if (
        result["family"] != family
        or result["scope"] != scope
        or result["input_capacity_result_file_sha256"]
        != capacity_result_file_sha256
        or result["input_capacity_result_sha256"]
        != capacity_result["result_sha256"]
        or result["collated_completion_sha256"]
        != collated_completion["completion_sha256"]
        or result["initial_model_state_sha256"]
        != provenance["initial_model_state_sha256"]
        or (
            scope != "heads_only"
            and (
                result["input_prior_scope_result_file_sha256"]
                != prior_scope_result_file_sha256
                or result["input_prior_scope_result_sha256"]
                != prior_scope_result["result_sha256"]
            )
        )
    ):
        raise RuntimeError("Process-V2 T1 failure-scope publication drifted")
    response = {
        "phase": "process_v2_t1_failure_scope_family_complete",
        "run_root": str(run_root),
        "scope": scope,
        "family": family,
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_sha256": result["result_sha256"],
        "status": result["status"],
        "diagnostic_passed": result["diagnostic_passed"],
        "selected_step": result["selected_step"],
        "reused": reused,
        "fiber_recomputation_count": 0,
        "gpu_side_collation_count": 0,
        "image_revision_sha256": revision["image_revision_sha256"],
        "bounded_p50_authorized": False,
        "p50_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    gpu=PUBLICATION_RECOVERY_GPU,
    cpu=8.0,
    memory=64 * 1024,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def recover_t1_publication_remote(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    terminal_checkpoint_path: str,
    expected_terminal_checkpoint_file_sha256: str,
    selected_checkpoint_path: str,
    expected_selected_checkpoint_file_sha256: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Recover missing result publication after an uninterrupted terminal step."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    torch = loaded["torch"]
    completion = _require_artifact_path(
        prepared_completion_path, field="prepared_completion_path"
    )
    terminal_path = _require_artifact_path(
        terminal_checkpoint_path, field="terminal_checkpoint_path"
    )
    selected_path = _require_artifact_path(
        selected_checkpoint_path, field="selected_checkpoint_path"
    )
    for field, value, path in (
        (
            "expected_terminal_checkpoint_file_sha256",
            expected_terminal_checkpoint_file_sha256,
            terminal_path,
        ),
        (
            "expected_selected_checkpoint_file_sha256",
            expected_selected_checkpoint_file_sha256,
            selected_path,
        ),
    ):
        if _SHA256_RE.fullmatch(value) is None or _file_sha256(path) != value:
            raise RuntimeError(f"Process-V2 T1 publication recovery {field} disagrees")

    runtime, scratch, runtime_provenance = loaded["load_runtime"](
        completion,
        capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    plan_path = _require_artifact_path(
        collated_plan_path, field="collated_plan_path"
    )
    plan = loaded["load_collated_plan"](
        plan_path,
        runtime=runtime,
        repo_root=REMOTE_ROOT,
    )
    if (
        plan["prepared_completion_path"] != str(completion)
        or plan["prepared_completion_sha256"]
        != runtime_provenance["prepared_completion_sha256"]
    ):
        raise RuntimeError(
            "Process-V2 T1 recovery collated plan binds another prepared input"
        )
    collated_path = _require_artifact_path(
        collated_completion_path, field="collated_completion_path"
    )
    materialized_panel = loaded["load_materialized_collated_panel"](
        collated_path,
        plan=plan,
        runtime=runtime,
    )
    collated_completion = _read_canonical_object(
        collated_path, label="the Process-V2 T1 collated completion"
    )
    try:
        checkpoint = torch.load(terminal_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("Process-V2 T1 terminal checkpoint is unreadable") from error
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("identity"), Mapping
    ):
        raise RuntimeError("Process-V2 T1 terminal checkpoint identity is absent")
    checkpoint_identity = dict(checkpoint["identity"])
    runner_hash = loaded["runner_implementation_sha256"](repo_root=REMOTE_ROOT)
    runner_source_revision_sha256 = str(
        checkpoint_identity.get("runner_source_revision_sha256")
    )
    checkpoint = _validate_publication_recovery_checkpoint(
        checkpoint,
        loaded=loaded,
        runtime=runtime,
        expected_runner_implementation_sha256=runner_hash,
        expected_runner_source_revision_sha256=runner_source_revision_sha256,
    )

    model = scratch.model.to(device="cuda", dtype=torch.float32)
    environment = _execution_environment(
        loaded, batch_size=int(runtime.capacity_policy["optimization"]["batch_size"])
    )
    if environment != checkpoint_identity.get("execution_environment"):
        expected_environment = checkpoint_identity.get("execution_environment")
        expected_mapping = (
            dict(expected_environment)
            if isinstance(expected_environment, Mapping)
            else {"value": expected_environment}
        )
        differing_fields = {
            name: {
                "expected": expected_mapping.get(name),
                "observed": environment.get(name),
            }
            for name in sorted(set(expected_mapping) | set(environment))
            if expected_mapping.get(name) != environment.get(name)
        }
        raise RuntimeError(
            "Process-V2 T1 publication recovery execution environment disagrees: "
            f"{differing_fields}"
        )
    loaded["validate_selected_checkpoint"](
        selected_path,
        expected_file_sha256=expected_selected_checkpoint_file_sha256,
        expected_selected_step=int(checkpoint["selected_step"]),
        expected_model_state_sha256=str(checkpoint["selected_model_state_sha256"]),
        expected_stream_sha256=str(checkpoint["stream_sha256"]),
        expected_identity_fields=checkpoint_identity,
    )
    provenance = loaded["project_result_provenance"](
        {
            **dict(runtime_provenance),
            "runner_source_revision_sha256": runner_source_revision_sha256,
            "runner_implementation_sha256": runner_hash,
            "execution_environment": environment,
        }
    )
    recovery_revision = _source_revision(revision)
    recovery_identity_sha256 = _sha256(
        {
            "prepared_completion_sha256": provenance["prepared_completion_sha256"],
            "collated_completion_sha256": collated_completion["completion_sha256"],
            "terminal_checkpoint_file_sha256": (
                expected_terminal_checkpoint_file_sha256
            ),
            "selected_checkpoint_file_sha256": (
                expected_selected_checkpoint_file_sha256
            ),
            "training_runner_source_revision_sha256": runner_source_revision_sha256,
            "training_runner_implementation_sha256": runner_hash,
            "recovery_source_revision_sha256": recovery_revision[
                "source_revision_sha256"
            ],
        }
    )
    run_root = (
        _require_physical_artifact_path(output_prefix, field="output_prefix")
        / recovery_identity_sha256
    )
    builder = functools.partial(
        _publication_recovery_result_builder,
        build_result=loaded["build_result"],
        capacity_policy=runtime.capacity_policy,
    )
    _progress(
        "process_v2_t1_publication_recovery_start",
        run_root=str(run_root),
        optimizer_updates_planned=0,
        prepared_entry_count=len(runtime.entries),
        collated_task_count=collated_completion["task_count"],
        gpu_side_collation_count=0,
    )
    with _heartbeat("process_v2_t1_publication_recovery_heartbeat"):
        run = loaded["run_capacity"](
            model,
            runtime,
            output_directory=run_root,
            provenance=provenance,
            resume_checkpoint_path=terminal_path,
            resume_checkpoint_file_sha256=expected_terminal_checkpoint_file_sha256,
            materialized_panel=materialized_panel,
            result_builder=builder,
            result_filename=PROCESS_V2_RESULT_FILENAME,
            expected_runner_implementation_sha256=runner_hash,
            expected_runner_source_revision_sha256=(
                runner_source_revision_sha256
            ),
        )
    result = loaded["validate_result"](
        run["result"], capacity_policy=runtime.capacity_policy
    )
    recovered_selected_path = Path(run["selected_checkpoint_path"])
    if (
        result["run_integrity"]["optimizer_steps_completed"]
        != int(runtime.capacity_policy["optimization"]["maximum_optimizer_steps"])
        or result["run_integrity"]["resume_requested"] is not False
        or result["run_integrity"]["resume_count"] != 0
        or result["run_integrity"]["optimizer_state_sha256"]
        != checkpoint["optimizer_state_sha256"]
        or _file_sha256(recovered_selected_path)
        != expected_selected_checkpoint_file_sha256
        or run["recovery_checkpoints"] != []
    ):
        raise RuntimeError(
            "Process-V2 T1 publication recovery changed optimization or selection"
        )
    result_path = Path(run["result_path"])
    result_file_sha256 = _file_sha256(result_path)
    decision = loaded["build_decision"](
        result,
        capacity_policy=runtime.capacity_policy,
        result_file_sha256=result_file_sha256,
    )
    decision_path = run_root / loaded["DECISION_FILENAME"]
    loaded["write_bytes_if_absent"](
        decision_path, _canonical_bytes(decision) + b"\n"
    )
    decision = loaded["validate_decision"](
        _read_canonical_object(
            decision_path, label="the recovered Process-V2 T1 capacity decision"
        ),
        result=result,
        capacity_policy=runtime.capacity_policy,
        result_file_sha256=result_file_sha256,
    )
    receipt_body = {
        "schema": "compose.editing_v2.process_v2_t1_publication_recovery",
        "schema_version": 2,
        "status": "COMPLETE_T1_PUBLICATION_RECOVERY_NO_DOWNSTREAM_AUTHORITY",
        **loaded["NO_AUTHORITY"],
        "prepared_completion_sha256": provenance["prepared_completion_sha256"],
        "collated_completion_sha256": collated_completion["completion_sha256"],
        "terminal_checkpoint_file_sha256": (
            expected_terminal_checkpoint_file_sha256
        ),
        "selected_checkpoint_file_sha256": (
            expected_selected_checkpoint_file_sha256
        ),
        "training_runner_source_revision_sha256": runner_source_revision_sha256,
        "training_runner_implementation_sha256": runner_hash,
        "recovery_source_revision": recovery_revision,
        "optimizer_updates_executed": 0,
        "panel_materialization_count": 1,
        "gpu_side_collation_count": 0,
        "result_file_sha256": result_file_sha256,
        "result_sha256": result["result_sha256"],
        "decision_sha256": decision["decision_sha256"],
    }
    receipt = {
        **receipt_body,
        "recovery_receipt_sha256": _sha256(receipt_body),
    }
    receipt_path = run_root / PUBLICATION_RECOVERY_RECEIPT_FILENAME
    loaded["write_bytes_if_absent"](
        receipt_path, _canonical_bytes(receipt) + b"\n"
    )
    artifact_volume.commit()
    response = {
        "phase": "process_v2_t1_publication_recovery_complete",
        "run_root": str(run_root),
        "result_path": str(result_path),
        "result_file_sha256": result_file_sha256,
        "result_sha256": result["result_sha256"],
        "decision_path": str(decision_path),
        "decision_sha256": decision["decision_sha256"],
        "decision_status": decision["status"],
        "recovery_receipt_path": str(receipt_path),
        "recovery_receipt_sha256": receipt["recovery_receipt_sha256"],
        "optimizer_updates_executed": 0,
        "gpu_side_collation_count": 0,
        "bounded_p50_authorized": decision["bounded_p50_authorized"],
        "p50_launched": False,
    }
    _progress(**response)
    return response


def _terminal_audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach only prepared-fiber census fields to deterministic metric rows."""

    entry_by_id = {str(entry["panel_entry_sha256"]): entry for entry in entries}
    if len(entry_by_id) != len(entries):
        raise RuntimeError("Process-V2 T1 terminal audit repeats a prepared entry")
    result: list[dict[str, Any]] = []
    for row in rows:
        identifier = str(row["panel_entry_sha256"])
        try:
            entry = entry_by_id[identifier]
        except KeyError as error:
            raise RuntimeError(
                "Process-V2 T1 terminal metric references another prepared panel"
            ) from error
        result.append(
            {
                **dict(row),
                "raw_mark_count": int(entry["raw_mark_count"]),
                "productive_alias_count": int(entry["productive_alias_count"]),
                "virtual_alias_count": int(entry["virtual_alias_count"]),
                "teacher_alias_multiplicity": int(
                    entry["production_successor_alias_multiplicity"]
                ),
            }
        )
    return result


def _terminal_audit_environment_disposition(
    observed: Mapping[str, Any],
    expected: object,
) -> str:
    """Classify the one known Modal A10 device-name alias for this audit.

    This does not modify the frozen T1 execution receipt.  The terminal audit
    carries no downstream authority and separately requires exact reproduction
    of every selected-checkpoint metric before publishing terminal metrics.
    """

    if not isinstance(expected, Mapping):
        raise RuntimeError("Process-V2 T1 checkpoint execution environment is absent")
    if observed == expected:
        return "EXACT"
    nonidentity_fields = {"device_name", "environment_sha256"}
    observed_substantive = {
        key: value for key, value in observed.items() if key not in nonidentity_fields
    }
    expected_substantive = {
        key: value for key, value in expected.items() if key not in nonidentity_fields
    }
    if (
        observed_substantive == expected_substantive
        and (str(expected.get("device_name")), str(observed.get("device_name")))
        == ("NVIDIA A10", "NVIDIA A10G")
    ):
        return "MODAL_A10_DEVICE_NAME_ALIAS_ONLY"
    differing_fields = {
        key: {
            "expected": expected.get(key),
            "observed": observed.get(key),
        }
        for key in sorted(set(observed) | set(expected))
        if expected.get(key) != observed.get(key)
    }
    raise RuntimeError(
        "Process-V2 T1 terminal-audit environment disagrees: "
        f"{differing_fields}"
    )


def _terminal_audit_checkpoint_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must be identical across independently scored families."""

    return {
        "selected_step": result["selected_step"],
        "terminal_step": result["terminal_step"],
        "selected_model_state_sha256": result["selected_model_state_sha256"],
        "terminal_model_state_sha256": result["terminal_model_state_sha256"],
        "gradient_evidence": result["gradient_evidence"],
        "checkpoint_execution_environment": result[
            "checkpoint_execution_environment"
        ],
    }


def _terminal_audit_worker_execution_receipts(
    by_family: Mapping[str, Mapping[str, Any]],
    *,
    family_order: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate and retain each worker's observed execution environment.

    Modal can expose the same A10 accelerator as either ``NVIDIA A10`` or
    ``NVIDIA A10G``.  Each worker is checked independently against the frozen
    checkpoint environment.  Requiring the observed spelling to agree across
    workers would incorrectly turn an accepted infrastructure alias into a
    checkpoint-identity disagreement.
    """

    receipts: list[dict[str, Any]] = []
    for family in family_order:
        result = by_family[family]
        observed = result["audit_execution_environment"]
        expected = result["checkpoint_execution_environment"]
        disposition = _terminal_audit_environment_disposition(observed, expected)
        if disposition != result["execution_environment_disposition"]:
            raise RuntimeError(
                "Process-V2 T1 terminal family audit environment disposition disagrees"
            )
        receipts.append(
            {
                "family": family,
                "audit_execution_environment": observed,
                "execution_environment_disposition": disposition,
            }
        )
    return receipts


@app.function(
    image=image,
    gpu=PUBLICATION_RECOVERY_GPU,
    cpu=8.0,
    memory=32 * 1024,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=8,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def audit_t1_family_remote(
    prepared_completion_path: str,
    step_500_checkpoint_path: str,
    expected_step_500_file_sha256: str,
    family: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Score one exact 64-entry family at selected and terminal checkpoints."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    torch = loaded["torch"]
    completion = _require_artifact_path(
        prepared_completion_path, field="prepared_completion_path"
    )
    step_500_path = _require_artifact_path(
        step_500_checkpoint_path, field="step_500_checkpoint_path"
    )
    if (
        _SHA256_RE.fullmatch(expected_step_500_file_sha256) is None
        or _file_sha256(step_500_path) != expected_step_500_file_sha256
    ):
        raise RuntimeError("Process-V2 T1 terminal-audit checkpoint SHA-256 disagrees")

    with _heartbeat(f"process_v2_t1_terminal_audit_{family}"):
        runtime, scratch, _provenance = loaded["load_runtime"](
            completion,
            capacity_policy_path=REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
            repo_root=REMOTE_ROOT,
        )
        required_families = tuple(str(item) for item in runtime.capacity_policy["required_families"])
        if family not in required_families:
            raise RuntimeError(f"Process-V2 T1 terminal audit received unknown family {family!r}")
        try:
            checkpoint = torch.load(step_500_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError("Process-V2 T1 terminal-audit checkpoint is unreadable") from error
        if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("identity"), Mapping
        ):
            raise RuntimeError("Process-V2 T1 terminal-audit checkpoint identity is absent")
        checkpoint_identity = dict(checkpoint["identity"])
        runner_hash = loaded["runner_implementation_sha256"](repo_root=REMOTE_ROOT)
        checkpoint = _validate_publication_recovery_checkpoint(
            checkpoint,
            loaded=loaded,
            runtime=runtime,
            expected_runner_implementation_sha256=runner_hash,
            expected_runner_source_revision_sha256=str(
                checkpoint_identity["runner_source_revision_sha256"]
            ),
        )
        model = scratch.model.to(device="cuda", dtype=torch.float32)
        environment = _execution_environment(
            loaded, batch_size=int(runtime.capacity_policy["optimization"]["batch_size"])
        )
        expected_environment = checkpoint_identity.get("execution_environment")
        environment_disposition = _terminal_audit_environment_disposition(
            environment,
            expected_environment,
        )
        panel_ids = tuple(
            str(entry["panel_entry_sha256"])
            for entry in runtime.entries
            if entry["model_family"] == family
        )
        if len(panel_ids) != 64:
            raise RuntimeError(
                f"Process-V2 T1 terminal audit expected 64 {family} entries, got {len(panel_ids)}"
            )
        batch, fibers, partitions, entries = loaded["batch_for_ids"](
            runtime, model, loaded["collator"](model), panel_ids
        )
        panel = loaded["MaterializedT1Panel"](
            batch=batch,
            panel_ids=panel_ids,
            fibers=fibers,
            partitions=partitions,
            entries=entries,
            index_by_panel_id={identifier: index for index, identifier in enumerate(panel_ids)},
        )

        def evaluate(state: Mapping[str, Any]) -> list[dict[str, Any]]:
            model.load_state_dict(state, strict=True)
            return _terminal_audit_rows(
                loaded["metric_rows"](panel, model, batch_size=64),
                entries=entries,
            )

        selected_rows = evaluate(checkpoint["selected_model_state"])
        terminal_rows = evaluate(checkpoint["model_state"])

    response = {
        "family": family,
        "entry_count": len(panel_ids),
        "selected_step": int(checkpoint["selected_step"]),
        "terminal_step": int(checkpoint["completed_steps"]),
        "selected_model_state_sha256": str(checkpoint["selected_model_state_sha256"]),
        "terminal_model_state_sha256": str(checkpoint["model_state_sha256"]),
        "gradient_evidence": checkpoint["gradient_evidence"],
        "checkpoint_execution_environment": expected_environment,
        "audit_execution_environment": environment,
        "execution_environment_disposition": environment_disposition,
        "selected_rows": selected_rows,
        "terminal_rows": terminal_rows,
    }
    _progress(
        "process_v2_t1_terminal_family_audit_complete",
        family=family,
        entry_count=len(panel_ids),
        selected_step=response["selected_step"],
        terminal_step=response["terminal_step"],
    )
    return response


def _metric_result_projection(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_order: Sequence[str],
) -> list[dict[str, Any]]:
    indices = {family: index for index, family in enumerate(family_order)}
    return sorted(
        (
            {
                "panel_entry_sha256": str(row["panel_entry_sha256"]),
                "family": str(row["model_family"]),
                "semantic_cell_id": str(row["capability_cell_id"]),
                "teacher_successor_probability": float(row["teacher_successor_probability"]),
                "canonical_successor_nll": float(row["teacher_successor_nll"]),
                "teacher_successor_rank": int(row["teacher_successor_rank"]),
                "teacher_successor_top1": bool(row["teacher_successor_top1"]),
            }
            for row in rows
        ),
        key=lambda row: (
            indices[row["family"]],
            row["semantic_cell_id"],
            row["panel_entry_sha256"],
        ),
    )


@app.function(
    image=image,
    cpu=1.0,
    memory=4 * 1024,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def audit_t1_terminal_driver(
    prepared_completion_path: str,
    step_500_checkpoint_path: str,
    expected_step_500_file_sha256: str,
    recovered_result_path: str,
    expected_recovered_result_file_sha256: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Fan out family scoring and publish one non-authorizing terminal audit."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    result_path = _require_artifact_path(recovered_result_path, field="recovered_result_path")
    if (
        _SHA256_RE.fullmatch(expected_recovered_result_file_sha256) is None
        or _file_sha256(result_path) != expected_recovered_result_file_sha256
    ):
        raise RuntimeError("Process-V2 T1 recovered result SHA-256 disagrees")
    policy, _policy_file_sha256 = loaded["load_capacity_policy"](
        REMOTE_ROOT / CAPACITY_POLICY_SOURCE,
        repo_root=REMOTE_ROOT,
    )
    recovered_result = loaded["validate_result"](
        _read_canonical_object(result_path, label="the recovered Process-V2 T1 result"),
        capacity_policy=policy,
    )
    family_order = tuple(str(item) for item in policy["required_families"])
    audit_t1_family_remote.update_autoscaler(max_containers=len(family_order))
    family_results = list(
        audit_t1_family_remote.starmap(
            [
                (
                    prepared_completion_path,
                    step_500_checkpoint_path,
                    expected_step_500_file_sha256,
                    family,
                    revision,
                )
                for family in family_order
            ]
        )
    )
    by_family = {str(item["family"]): item for item in family_results}
    if set(by_family) != set(family_order):
        raise RuntimeError("Process-V2 T1 terminal audit lost a family result")
    first = by_family[family_order[0]]
    checkpoint_projection = _terminal_audit_checkpoint_projection(first)
    if any(
        _terminal_audit_checkpoint_projection(item) != checkpoint_projection
        for item in by_family.values()
    ):
        raise RuntimeError("Process-V2 T1 terminal family audits disagree on checkpoint identity")
    worker_execution_receipts = _terminal_audit_worker_execution_receipts(
        by_family,
        family_order=family_order,
    )
    selected_rows = [
        row for family in family_order for row in by_family[family]["selected_rows"]
    ]
    terminal_rows = [
        row for family in family_order for row in by_family[family]["terminal_rows"]
    ]
    selected_projection = _metric_result_projection(selected_rows, family_order=family_order)
    expected_projection = list(recovered_result["entry_metrics"])
    if selected_projection != expected_projection:
        raise RuntimeError(
            "Process-V2 T1 terminal audit does not reproduce the authoritative selected metrics"
        )
    selected_metrics = loaded["summarize_metrics"](selected_rows)
    terminal_metrics = loaded["summarize_metrics"](terminal_rows)
    gradients = first["gradient_evidence"]
    thresholds = policy["thresholds"]
    selected_checks = loaded["threshold_checks"](
        selected_metrics, thresholds=thresholds, gradient_evidence=gradients
    )
    terminal_checks = loaded["threshold_checks"](
        terminal_metrics, thresholds=thresholds, gradient_evidence=gradients
    )
    if selected_checks != recovered_result["threshold_checks"]:
        raise RuntimeError("Process-V2 T1 terminal audit threshold cross-check disagrees")

    selected_by_id = {str(row["panel_entry_sha256"]): row for row in selected_rows}
    terminal_by_id = {str(row["panel_entry_sha256"]): row for row in terminal_rows}
    entry_comparisons = []
    for family in family_order:
        for identifier in sorted(
            row_id
            for row_id, row in terminal_by_id.items()
            if row["model_family"] == family
        ):
            selected = selected_by_id[identifier]
            terminal = terminal_by_id[identifier]
            entry_comparisons.append(
                {
                    "panel_entry_sha256": identifier,
                    "family": family,
                    "semantic_cell_id": terminal["capability_cell_id"],
                    "canonical_successor_count": int(terminal["canonical_successor_count"]),
                    "raw_mark_count": int(terminal["raw_mark_count"]),
                    "productive_alias_count": int(terminal["productive_alias_count"]),
                    "virtual_alias_count": int(terminal["virtual_alias_count"]),
                    "teacher_alias_multiplicity": int(terminal["teacher_alias_multiplicity"]),
                    "selected_probability": float(selected["teacher_successor_probability"]),
                    "terminal_probability": float(terminal["teacher_successor_probability"]),
                    "probability_delta": float(terminal["teacher_successor_probability"])
                    - float(selected["teacher_successor_probability"]),
                    "selected_nll": float(selected["teacher_successor_nll"]),
                    "terminal_nll": float(terminal["teacher_successor_nll"]),
                    "selected_rank": int(selected["teacher_successor_rank"]),
                    "terminal_rank": int(terminal["teacher_successor_rank"]),
                    "selected_top1": bool(selected["teacher_successor_top1"]),
                    "terminal_top1": bool(terminal["teacher_successor_top1"]),
                }
            )
    revision_body = _source_revision(revision)
    identity = {
        "prepared_completion_path": prepared_completion_path,
        "step_500_checkpoint_path": step_500_checkpoint_path,
        "step_500_checkpoint_file_sha256": expected_step_500_file_sha256,
        "recovered_result_path": recovered_result_path,
        "recovered_result_file_sha256": expected_recovered_result_file_sha256,
        "recovered_result_sha256": recovered_result["result_sha256"],
        "audit_source_revision_sha256": revision_body["source_revision_sha256"],
    }
    artifact_body = {
        "schema": "compose.editing_v2.process_v2_t1_terminal_audit",
        "schema_version": 1,
        "status": "COMPLETE_PROCESS_V2_T1_TERMINAL_AUDIT_NO_DOWNSTREAM_AUTHORITY",
        **loaded["NO_AUTHORITY"],
        "identity": identity,
        "selected_step": int(first["selected_step"]),
        "terminal_step": int(first["terminal_step"]),
        "selected_model_state_sha256": first["selected_model_state_sha256"],
        "terminal_model_state_sha256": first["terminal_model_state_sha256"],
        "checkpoint_execution_environment": first["checkpoint_execution_environment"],
        "worker_execution_receipts": worker_execution_receipts,
        "family_worker_count": len(family_results),
        "entry_count": len(entry_comparisons),
        "selected_result_reproduced_exactly": True,
        "selected_threshold_checks": selected_checks,
        "terminal_threshold_checks": terminal_checks,
        "selected_metrics": {
            key: value for key, value in selected_metrics.items() if key != "per_entry"
        },
        "terminal_metrics": {
            key: value for key, value in terminal_metrics.items() if key != "per_entry"
        },
        "entry_comparisons": entry_comparisons,
        "optimizer_updates_executed": 0,
        "p50_launched": False,
    }
    artifact = {**artifact_body, "audit_sha256": _sha256(artifact_body)}
    run_root = (
        _require_physical_artifact_path(output_prefix, field="output_prefix")
        / _sha256(identity)
    )
    audit_path = run_root / TERMINAL_AUDIT_FILENAME
    loaded["write_bytes_if_absent"](audit_path, _canonical_bytes(artifact) + b"\n")
    reopened = _read_canonical_object(audit_path, label="the Process-V2 T1 terminal audit")
    if reopened != artifact:
        raise RuntimeError("published Process-V2 T1 terminal audit changed on reopen")
    artifact_volume.commit()
    response = {
        "phase": "process_v2_t1_terminal_audit_complete",
        "audit_path": str(audit_path),
        "audit_file_sha256": _file_sha256(audit_path),
        "audit_sha256": artifact["audit_sha256"],
        "selected_result_reproduced_exactly": True,
        "terminal_threshold_checks": terminal_checks,
        "optimizer_updates_executed": 0,
        "p50_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=0.25,
    memory=1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
)
def driver(
    active8_run_root: str,
    gate_zero_decision_path: str,
    prepared_output_prefix: str,
    run_output_prefix: str,
    max_cpu_containers: int,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Run detached restart-safe preparation, then exactly one GPU job."""

    _validate_remote_revision(revision)
    if type(max_cpu_containers) is not int or not 1 <= max_cpu_containers <= MAX_CPU_CONTAINERS:
        raise ValueError(f"max_cpu_containers must lie in [1, {MAX_CPU_CONTAINERS}]")
    prepared = prepare_plan_remote.remote(
        active8_run_root,
        gate_zero_decision_path,
        prepared_output_prefix,
        revision,
    )
    plan = prepared["plan"]
    completed = set(
        scan_completed_remote.remote(prepared["plan_path"], prepared["run_root"], revision)
    )
    expected = [str(task["task_identity_sha256"]) for task in plan["tasks"]]
    missing = [identity for identity in expected if identity not in completed]
    waves = _groups(missing, maximum=max_cpu_containers)
    _progress(
        "process_v2_t1_cpu_map_plan",
        expected_tasks=len(expected),
        already_complete=len(completed),
        missing_tasks=len(missing),
        max_cpu_containers=max_cpu_containers,
        submission_waves=len(waves),
    )
    prepare_leaf_remote.update_autoscaler(max_containers=max_cpu_containers)
    if missing:
        # One starmap lets Modal replenish the bounded pool immediately when a
        # leaf finishes.  Submitting manual waves would strand slots behind the
        # slowest leaf in each wave.
        results = list(
            prepare_leaf_remote.starmap(
                [
                    (
                        prepared["plan_path"],
                        identity,
                        active8_run_root,
                        gate_zero_decision_path,
                        prepared["panel_path"],
                        prepared["run_root"],
                        revision,
                    )
                    for identity in missing
                ]
            )
        )
        if {str(result["task_identity_sha256"]) for result in results} != set(missing):
            raise RuntimeError("Process-V2 T1 CPU map lost a task result")
        _progress(
            "process_v2_t1_cpu_map_complete",
            submission_waves=len(waves),
            submitted_tasks=len(missing),
            completed_tasks=len(completed) + len(missing),
            expected_tasks=len(expected),
        )
    complete = set(
        scan_completed_remote.remote(prepared["plan_path"], prepared["run_root"], revision)
    )
    if complete != set(expected):
        raise RuntimeError(
            f"Process-V2 T1 preparation is incomplete: missing={len(set(expected) - complete)}"
        )
    finalized = finalize_prepared_remote.remote(
        prepared["plan_path"],
        active8_run_root,
        gate_zero_decision_path,
        prepared["panel_path"],
        prepared["run_root"],
        revision,
    )
    gpu = run_t1_gpu_remote.remote(
        finalized["prepared_completion_path"],
        run_output_prefix,
        revision,
    )
    result = {
        "phase": "process_v2_t1_driver_complete",
        "prepared": finalized,
        "capacity": gpu,
        "max_cpu_containers": max_cpu_containers,
        "cpu_submission_waves": len(waves),
        "image_revision": revision,
        "bounded_p50_authorized": gpu["bounded_p50_authorized"],
        "p50_launched": False,
    }
    _progress(
        "process_v2_t1_driver_complete",
        prepared_completion_sha256=finalized["prepared_completion_sha256"],
        result_sha256=gpu["result_sha256"],
        decision_sha256=gpu["decision_sha256"],
        decision_status=gpu["decision_status"],
        bounded_p50_authorized=gpu["bounded_p50_authorized"],
        p50_launched=False,
    )
    return result


@app.function(
    image=image,
    cpu=0.25,
    memory=1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
)
def reuse_driver(
    active8_run_root: str,
    gate_zero_decision_path: str,
    reusable_run_root: str,
    expected_plan_sha256: str,
    run_output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Reduce an exact completed leaf run, then launch exactly one GPU T1 job."""

    _validate_remote_revision(revision)
    root = _require_artifact_path(reusable_run_root, field="reusable_run_root")
    finalized = finalize_reused_prepared_remote.remote(
        str(root / PLAN_FILENAME),
        active8_run_root,
        gate_zero_decision_path,
        str(root / PANEL_FILENAME),
        str(root),
        expected_plan_sha256,
        revision,
    )
    gpu = run_t1_gpu_remote.remote(
        finalized["prepared_completion_path"],
        run_output_prefix,
        revision,
    )
    result = {
        "phase": "process_v2_t1_reuse_driver_complete",
        "prepared": finalized,
        "capacity": gpu,
        "image_revision": revision,
        "fiber_recomputation_count": 0,
        "bounded_p50_authorized": gpu["bounded_p50_authorized"],
        "p50_launched": False,
    }
    _progress(
        "process_v2_t1_reuse_driver_complete",
        prepared_completion_sha256=finalized["prepared_completion_sha256"],
        result_sha256=gpu["result_sha256"],
        decision_sha256=gpu["decision_sha256"],
        decision_status=gpu["decision_status"],
        fiber_recomputation_count=0,
        bounded_p50_authorized=gpu["bounded_p50_authorized"],
        p50_launched=False,
    )
    return result


@app.function(
    image=image,
    cpu=0.25,
    memory=1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
)
def collated_cache_only_driver(
    prepared_completion_path: str,
    collated_output_prefix: str,
    entries_per_task: int,
    max_cpu_containers: int,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Fan out CPU collation over frozen fibers and stop before any GPU."""

    _validate_remote_revision(revision)
    if type(max_cpu_containers) is not int or not 1 <= max_cpu_containers <= MAX_CPU_CONTAINERS:
        raise ValueError(f"max_cpu_containers must lie in [1, {MAX_CPU_CONTAINERS}]")
    planned = prepare_collated_plan_remote.remote(
        prepared_completion_path,
        collated_output_prefix,
        int(entries_per_task),
        revision,
    )
    plan = planned["plan"]
    expected = [str(task["task_identity_sha256"]) for task in plan["tasks"]]
    completed = set(
        scan_collated_remote.remote(
            prepared_completion_path,
            planned["plan_path"],
            planned["run_root"],
            revision,
        )
    )
    missing = [identifier for identifier in expected if identifier not in completed]
    _progress(
        "process_v2_t1_collated_cpu_map_plan",
        expected_tasks=len(expected),
        already_complete=len(completed),
        missing_tasks=len(missing),
        max_cpu_containers=max_cpu_containers,
    )
    collate_t1_leaf_remote.update_autoscaler(max_containers=max_cpu_containers)
    if missing:
        results = list(
            collate_t1_leaf_remote.starmap(
                [
                    (
                        prepared_completion_path,
                        planned["plan_path"],
                        planned["run_root"],
                        identifier,
                        revision,
                    )
                    for identifier in missing
                ]
            )
        )
        if {str(result["task_identity_sha256"]) for result in results} != set(missing):
            raise RuntimeError("Process-V2 T1 CPU collation lost a task result")
    complete = set(
        scan_collated_remote.remote(
            prepared_completion_path,
            planned["plan_path"],
            planned["run_root"],
            revision,
        )
    )
    if complete != set(expected):
        raise RuntimeError(
            f"Process-V2 T1 CPU collation is incomplete: missing={len(set(expected) - complete)}"
        )
    finalized = finalize_collated_remote.remote(
        prepared_completion_path,
        planned["plan_path"],
        planned["run_root"],
        revision,
    )
    result = {
        "phase": "process_v2_t1_collated_cache_only_complete",
        "prepared_completion_path": prepared_completion_path,
        "plan_path": planned["plan_path"],
        "collated": finalized,
        "max_cpu_containers": max_cpu_containers,
        "entries_per_task": entries_per_task,
        "fiber_recomputation_count": 0,
        "gpu_side_collation_count": 0,
        "image_revision": revision,
        "training_launched": False,
        "bounded_p50_authorized": False,
        "p50_launched": False,
    }
    _progress(
        "process_v2_t1_collated_cache_only_complete",
        collated_completion_sha256=finalized["completion_sha256"],
        task_count=finalized["task_count"],
        fiber_recomputation_count=0,
        gpu_side_collation_count=0,
        training_launched=False,
        bounded_p50_authorized=False,
        p50_launched=False,
    )
    return result


@app.function(
    image=image,
    cpu=0.25,
    memory=1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
)
def collated_reuse_driver(
    prepared_completion_path: str,
    collated_output_prefix: str,
    run_output_prefix: str,
    entries_per_task: int,
    max_cpu_containers: int,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Complete the CPU cache, then launch the GPU-only T1 function."""

    cached = collated_cache_only_driver.remote(
        prepared_completion_path,
        collated_output_prefix,
        entries_per_task,
        max_cpu_containers,
        revision,
    )
    gpu = run_t1_collated_gpu_remote.remote(
        prepared_completion_path,
        cached["plan_path"],
        cached["collated"]["completion_path"],
        run_output_prefix,
        revision,
    )
    result = {
        "phase": "process_v2_t1_collated_driver_complete",
        "collated": cached["collated"],
        "capacity": gpu,
        "max_cpu_containers": max_cpu_containers,
        "entries_per_task": entries_per_task,
        "fiber_recomputation_count": 0,
        "gpu_side_collation_count": 0,
        "image_revision": revision,
        "bounded_p50_authorized": gpu["bounded_p50_authorized"],
        "p50_launched": False,
    }
    _progress(
        "process_v2_t1_collated_driver_complete",
        collated_completion_sha256=cached["collated"]["completion_sha256"],
        decision_sha256=gpu["decision_sha256"],
        decision_status=gpu["decision_status"],
        bounded_p50_authorized=gpu["bounded_p50_authorized"],
        fiber_recomputation_count=0,
        gpu_side_collation_count=0,
        p50_launched=False,
    )
    return result


@app.local_entrypoint()
def main(
    active8_run_root: str,
    gate_zero_decision_path: str,
    expected_commit: str,
    prepared_output_prefix: str = PREPARED_OUTPUT_PREFIX,
    run_output_prefix: str = RUN_OUTPUT_PREFIX,
    max_cpu_containers: int = MAX_CPU_CONTAINERS,
    reusable_run_root: str = "",
    expected_reuse_plan_sha256: str = "",
    wait_for_completion: bool = False,
) -> None:
    """Spawn one disconnect-safe Process-V2 T1 driver and return immediately."""

    revision = local_image_revision(expected_commit=expected_commit)
    reuse_requested = bool(reusable_run_root or expected_reuse_plan_sha256)
    if reuse_requested and not (reusable_run_root and expected_reuse_plan_sha256):
        raise ValueError(
            "reusable_run_root and expected_reuse_plan_sha256 must be supplied together"
        )
    if reuse_requested:
        arguments = (
            active8_run_root,
            gate_zero_decision_path,
            reusable_run_root,
            expected_reuse_plan_sha256,
            run_output_prefix,
            revision,
        )
        if wait_for_completion:
            print(json.dumps(reuse_driver.remote(*arguments), indent=2, sort_keys=True))
            return
        call = reuse_driver.spawn(*arguments)
    else:
        call = driver.spawn(
            active8_run_root,
            gate_zero_decision_path,
            prepared_output_prefix,
            run_output_prefix,
            int(max_cpu_containers),
            revision,
        )
    print(
        json.dumps(
            {
                "phase": "process_v2_t1_driver_launched",
                "driver_call_id": call.object_id,
                "max_cpu_containers": int(max_cpu_containers),
                "reused_precomputed_leaves": reuse_requested,
                "fiber_recomputation_count": 0 if reuse_requested else None,
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "bounded_p50_authorized": False,
                "p50_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def cached_main(
    prepared_completion_path: str,
    expected_commit: str,
    collated_output_prefix: str = COLLATED_OUTPUT_PREFIX,
    run_output_prefix: str = RUN_OUTPUT_PREFIX,
    entries_per_task: int = COLLATED_ENTRIES_PER_TASK,
    max_cpu_containers: int = MAX_CPU_CONTAINERS,
    wait_for_completion: bool = False,
) -> None:
    """Launch the optimized T1 path from an existing prepared completion."""

    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        prepared_completion_path,
        collated_output_prefix,
        run_output_prefix,
        int(entries_per_task),
        int(max_cpu_containers),
        revision,
    )
    if wait_for_completion:
        print(
            json.dumps(
                collated_reuse_driver.remote(*arguments), indent=2, sort_keys=True
            )
        )
        return
    call = collated_reuse_driver.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_t1_collated_driver_launched",
                "driver_call_id": call.object_id,
                "prepared_completion_path": prepared_completion_path,
                "entries_per_task": int(entries_per_task),
                "max_cpu_containers": int(max_cpu_containers),
                "fiber_recomputation_count": 0,
                "gpu_side_collation_count": 0,
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "bounded_p50_authorized": False,
                "p50_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def cache_only_main(
    prepared_completion_path: str,
    expected_commit: str,
    collated_output_prefix: str = COLLATED_OUTPUT_PREFIX,
    entries_per_task: int = COLLATED_ENTRIES_PER_TASK,
    max_cpu_containers: int = MAX_CPU_CONTAINERS,
    wait_for_completion: bool = False,
) -> None:
    """Materialize the authenticated CPU cache and allocate no GPU."""

    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        prepared_completion_path,
        collated_output_prefix,
        int(entries_per_task),
        int(max_cpu_containers),
        revision,
    )
    if wait_for_completion:
        print(
            json.dumps(
                collated_cache_only_driver.remote(*arguments),
                indent=2,
                sort_keys=True,
            )
        )
        return
    call = collated_cache_only_driver.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_t1_collated_cache_only_launched",
                "driver_call_id": call.object_id,
                "prepared_completion_path": prepared_completion_path,
                "entries_per_task": int(entries_per_task),
                "max_cpu_containers": int(max_cpu_containers),
                "fiber_recomputation_count": 0,
                "gpu_allocated": False,
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "bounded_p50_authorized": False,
                "p50_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def gpu_from_cache_main(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    expected_commit: str,
    run_output_prefix: str = RUN_OUTPUT_PREFIX,
    wait_for_completion: bool = False,
) -> None:
    """Launch only the GPU phase from a complete authenticated CPU cache."""

    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        prepared_completion_path,
        collated_plan_path,
        collated_completion_path,
        run_output_prefix,
        revision,
    )
    if wait_for_completion:
        print(
            json.dumps(
                run_t1_collated_gpu_remote.remote(*arguments),
                indent=2,
                sort_keys=True,
            )
        )
        return
    call = run_t1_collated_gpu_remote.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_t1_collated_gpu_launched",
                "gpu_call_id": call.object_id,
                "prepared_completion_path": prepared_completion_path,
                "collated_plan_path": collated_plan_path,
                "collated_completion_path": collated_completion_path,
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "bounded_p50_authorized": False,
                "p50_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def score_revision_repair_main(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    expected_commit: str,
    output_prefix: str = SCORE_REVISION_REPAIR_OUTPUT_PREFIX,
    repair_family: str = "",
    wait_for_completion: bool = False,
) -> None:
    """Launch one GPU for all unresolved families or one exact requested family."""

    revision = local_image_revision(expected_commit=expected_commit)
    repair_families = (
        _SCORE_REVISION_REPAIR_FAMILIES
        if not repair_family
        else (repair_family,)
    )
    if any(family not in _SCORE_REVISION_REPAIR_FAMILIES for family in repair_families):
        raise ValueError(
            "repair_family must be empty or name one score-revision repair family"
        )
    arguments = (
        prepared_completion_path,
        collated_plan_path,
        collated_completion_path,
        output_prefix,
        revision,
        repair_families,
    )
    if wait_for_completion:
        print(
            json.dumps(
                run_t1_score_revision_repair_remote.remote(*arguments),
                indent=2,
                sort_keys=True,
            )
        )
        return
    call = run_t1_score_revision_repair_remote.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_t1_score_revision_repair_launched",
                "function_call_id": call.object_id,
                "families": list(repair_families),
                "gpu_containers": 1,
                "fiber_recomputation_count": 0,
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "bounded_p50_authorized": False,
                "p50_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def final_scope_repair_main(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    repair_result_path: str,
    expected_commit: str,
    output_prefix: str = FINAL_SCOPE_REPAIR_OUTPUT_PREFIX,
    wait_for_completion: bool = False,
) -> None:
    """Launch only the final all-parameter rung for two failed families."""

    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        prepared_completion_path,
        collated_plan_path,
        collated_completion_path,
        repair_result_path,
        output_prefix,
        revision,
    )
    if wait_for_completion:
        print(
            json.dumps(
                run_t1_final_scope_repair_remote.remote(*arguments),
                indent=2,
                sort_keys=True,
            )
        )
        return
    call = run_t1_final_scope_repair_remote.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_t1_final_scope_repair_launched",
                "function_call_id": call.object_id,
                "families": list(_FINAL_SCOPE_REPAIR_FAMILIES),
                "scope": "all",
                "gpu_containers": 1,
                "fiber_recomputation_count": 0,
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "bounded_p50_authorized": False,
                "p50_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def failure_scope_main(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    capacity_result_path: str,
    expected_commit: str,
    output_prefix: str = FAILURE_SCOPE_OUTPUT_PREFIX,
    wait_for_completion: bool = False,
) -> None:
    """Launch one detached driver that fans out the failing families."""

    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        prepared_completion_path,
        collated_plan_path,
        collated_completion_path,
        capacity_result_path,
        output_prefix,
        revision,
    )
    if wait_for_completion:
        print(
            json.dumps(
                failure_scope_driver.remote(*arguments),
                indent=2,
                sort_keys=True,
            )
        )
        return
    call = failure_scope_driver.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_t1_failure_scope_launched",
                "driver_call_id": call.object_id,
                "scope": "heads_only",
                "parallel_family_arms": True,
                "fiber_recomputation_count": 0,
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "bounded_p50_authorized": False,
                "p50_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def next_failure_scope_main(
    prepared_completion_path: str,
    collated_plan_path: str,
    collated_completion_path: str,
    capacity_result_path: str,
    prior_scope_result_paths_json: str,
    expected_commit: str,
    output_prefix: str = FAILURE_SCOPE_OUTPUT_PREFIX,
    wait_for_completion: bool = False,
) -> None:
    """Launch the detached first post-heads parameter-scope wave."""

    parsed = json.loads(prior_scope_result_paths_json)
    if (
        not isinstance(parsed, dict)
        or not parsed
        or any(
            not isinstance(family, str)
            or not family
            or not isinstance(path, str)
            or not path
            for family, path in parsed.items()
        )
    ):
        raise RuntimeError(
            "prior_scope_result_paths_json must map family names to artifact paths"
        )
    prior_paths = {str(family): str(path) for family, path in parsed.items()}
    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        prepared_completion_path,
        collated_plan_path,
        collated_completion_path,
        capacity_result_path,
        prior_paths,
        output_prefix,
        revision,
    )
    if wait_for_completion:
        print(
            json.dumps(
                next_failure_scope_driver.remote(*arguments),
                indent=2,
                sort_keys=True,
            )
        )
        return
    call = next_failure_scope_driver.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_t1_next_failure_scope_launched",
                "driver_call_id": call.object_id,
                "parallel_family_arms": False,
                "execution_mode": "one_gpu_sequential_arms",
                "fiber_recomputation_count": 0,
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "bounded_p50_authorized": False,
                "p50_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "COLLATED_OUTPUT_PREFIX",
    "FAILURE_SCOPE_OUTPUT_PREFIX",
    "MAX_CPU_CONTAINERS",
    "PREPARED_OUTPUT_PREFIX",
    "PROCESS_V2_RESULT_FILENAME",
    "RUN_OUTPUT_PREFIX",
    "app",
    "cached_main",
    "cache_only_main",
    "collate_t1_leaf_remote",
    "collated_cache_only_driver",
    "collated_reuse_driver",
    "driver",
    "failure_scope_driver",
    "final_scope_repair_main",
    "next_failure_scope_driver",
    "finalize_collated_remote",
    "finalize_prepared_remote",
    "finalize_reused_prepared_remote",
    "gpu_from_cache_main",
    "failure_scope_main",
    "next_failure_scope_main",
    "local_image_revision",
    "main",
    "prepare_collated_plan_remote",
    "prepare_leaf_remote",
    "prepare_plan_remote",
    "run_t1_collated_gpu_remote",
    "run_t1_failure_scope_gpu_remote",
    "run_t1_final_scope_repair_remote",
    "run_t1_score_revision_repair_remote",
    "run_t1_gpu_remote",
    "score_revision_repair_main",
    "reuse_driver",
    "scan_collated_remote",
    "scan_completed_remote",
]
