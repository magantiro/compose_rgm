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

import functools
import hashlib
import json
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
RUN_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_t1_capacity"
CAPACITY_POLICY_SOURCE = "configs/editing_v2_process_v2_t1_capacity_policy.json"
PANEL_FILENAME = "PROCESS_V2_T1_PANEL.json"
PANEL_CACHE_DIRNAME = "_panel_cache"
PLAN_FILENAME = "PROCESS_V2_T1_PREPARED_PLAN.json"
LEAF_FILENAME = "PROCESS_V2_T1_PREPARED_LEAF.json"
PREPARED_COMPLETION_FILENAME = "PROCESS_V2_T1_PREPARED_COMPLETE.json"
PROCESS_V2_RESULT_FILENAME = "PROCESS_V2_T1_CAPACITY_RESULT.json"
STEP_TEN_CHECKPOINT_FILENAME = "step_0010.pt"

MAX_CPU_CONTAINERS = 40
CPU_PER_LEAF = 1.0
CPU_MEMORY_MB = 8 * 1024
CPU_LEAF_TIMEOUT_SECONDS = 45 * 60
COORDINATOR_TIMEOUT_SECONDS = 6 * 3600
GPU_TIMEOUT_SECONDS = 8 * 3600
HEARTBEAT_SECONDS = 30

REVISION_SCHEMA = "compose.editing_v2.process_v2_t1_modal_image_revision"
REVISION_SCHEMA_VERSION = 1
SOURCE_REVISION_SCHEMA = "compose.editing_v2.process_v2_t1_authenticated_image_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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
    from compose_v4.experiments.editing_v2_process_v2_t1_result import (
        DECISION_FILENAME,
        RESULT_FILENAME,
        build_process_v2_t1_capacity_decision,
        build_process_v2_t1_capacity_result,
        process_v2_t1_runner_implementation_sha256,
        validate_process_v2_t1_capacity_decision,
        validate_process_v2_t1_capacity_result,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        TASKS_DIRNAME,
        build_process_v2_t1_prepared_plan,
        build_process_v2_t1_scratch_runtime,
        compile_authenticated_process_v2_t1_prepared_leaf,
        load_process_v2_t1_runtime_inputs,
        publish_process_v2_t1_prepared_inputs,
        validate_process_v2_t1_prepared_leaf,
        write_process_v2_t1_prepared_leaf,
        write_process_v2_t1_prepared_plan,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        run_semantic_t1_capacity,
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
        "load_runtime": load_process_v2_t1_runtime_inputs,
        "publish_prepared": publish_process_v2_t1_prepared_inputs,
        "validate_leaf": validate_process_v2_t1_prepared_leaf,
        "write_leaf": write_process_v2_t1_prepared_leaf,
        "write_plan": write_process_v2_t1_prepared_plan,
        "build_result": build_process_v2_t1_capacity_result,
        "build_decision": build_process_v2_t1_capacity_decision,
        "runner_implementation_sha256": process_v2_t1_runner_implementation_sha256,
        "validate_decision": validate_process_v2_t1_capacity_decision,
        "validate_result": validate_process_v2_t1_capacity_result,
        "write_bytes_if_absent": write_bytes_if_absent,
        "build_environment": build_semantic_t1_execution_environment,
        "run_capacity": run_semantic_t1_capacity,
    }


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
    root = _require_artifact_path(run_root, field="run_root")
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
    gpu="A10G",
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
    provenance = {
        **dict(provenance),
        "runner_source_revision_sha256": source_revision["source_revision_sha256"],
        "runner_implementation_sha256": runner_hash,
        "execution_environment": environment,
    }
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


@app.local_entrypoint()
def main(
    active8_run_root: str,
    gate_zero_decision_path: str,
    expected_commit: str,
    prepared_output_prefix: str = PREPARED_OUTPUT_PREFIX,
    run_output_prefix: str = RUN_OUTPUT_PREFIX,
    max_cpu_containers: int = MAX_CPU_CONTAINERS,
) -> None:
    """Spawn one disconnect-safe Process-V2 T1 driver and return immediately."""

    revision = local_image_revision(expected_commit=expected_commit)
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
    "MAX_CPU_CONTAINERS",
    "PREPARED_OUTPUT_PREFIX",
    "PROCESS_V2_RESULT_FILENAME",
    "RUN_OUTPUT_PREFIX",
    "app",
    "driver",
    "finalize_prepared_remote",
    "local_image_revision",
    "main",
    "prepare_leaf_remote",
    "prepare_plan_remote",
    "run_t1_gpu_remote",
    "scan_completed_remote",
]
