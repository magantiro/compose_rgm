"""Recover one exact semantic-migration task on nonpreemptible CPU.

This narrow Modal surface exists only for an immutable migration plan whose
ordinary preemptible workers left a small number of tasks incomplete.  One
invocation authenticates one published plan and one task identity, executes
the unchanged production task implementation, and writes a content-addressed
recovery receipt.  It does not plan, reduce, launch training, or grant any
downstream authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")

LAUNCHER_SOURCE = "modal_apps/recover_editing_v2_semantic_migration_task_app.py"
MIGRATION_LAUNCHER_SOURCE = "modal_apps/materialize_editing_v2_semantic_corpus_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
IMAGE_SOURCE_FILES = (LAUNCHER_SOURCE, MIGRATION_LAUNCHER_SOURCE)

REQUEST_SCHEMA = "compose.editing_v2_semantic_migration_task_recovery_request"
REQUEST_SCHEMA_VERSION = 1
RECEIPT_SCHEMA = "compose.editing_v2_semantic_migration_task_recovery_receipt"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_STATUS = "COMPLETE_CORPUS_TASK_RECOVERY_NO_DOWNSTREAM_AUTHORITY"
RECEIPT_FILENAME = "SEMANTIC_MIGRATION_TASK_RECOVERY_RECEIPT.json"
RECOVERY_OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_migration_task_recovery"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHORITY_FIELDS = {
    "active8_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
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
for source_file in IMAGE_SOURCE_FILES:
    image = image.add_local_file(
        ROOT / source_file,
        str(REMOTE_ROOT / source_file),
        copy=True,
    )

app = modal.App("compose-v4-editing-v2-semantic-migration-task-recovery")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{field} must be a lowercase SHA-256")
    return value


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise RuntimeError(f"{field} must be a full lowercase Git commit")
    return value


def _artifact_path(value: str, *, artifact_root: Path, field: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or str(pure) != value
        or value.endswith("/")
    ):
        raise RuntimeError(f"{field} must be a normalized path below /artifacts")
    root = Path(artifact_root).resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"{field} resolves outside artifact_root")
    return resolved


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    relative = path.resolve().relative_to(artifact_root.resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


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
            f"cannot establish recovery-launcher Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = list(IMAGE_SOURCE_FILES)
    for source_directory in IMAGE_SOURCE_DIRECTORIES:
        directory = root / source_directory
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    return tuple(sorted(set(paths)))


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    return {relative: _file_sha256(root / relative) for relative in _serialized_source_paths(root)}


def local_source_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind every serialized file from an exact clean committed worktree."""

    _require_commit(expected_commit, field="expected_commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError("recovery requires the exact clean committed worktree")
    hashes = _serialized_source_hashes(root)
    body: dict[str, Any] = {
        "schema": "compose.editing_v2_semantic_migration_recovery_source_revision",
        "schema_version": 1,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": _sha(hashes),
    }
    return {**body, "source_revision_sha256": _sha(body)}


def _validate_source_revision(value: object, *, remote_root: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("recovery source revision must be an object")
    revision = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "worktree_clean",
        "serialized_source_hashes",
        "serialized_source_hashes_sha256",
        "source_revision_sha256",
    }
    body = {key: item for key, item in revision.items() if key != "source_revision_sha256"}
    live_hashes = _serialized_source_hashes(remote_root)
    if (
        set(revision) != expected_fields
        or revision.get("schema")
        != "compose.editing_v2_semantic_migration_recovery_source_revision"
        or revision.get("schema_version") != 1
        or revision.get("worktree_clean") is not True
        or revision.get("serialized_source_hashes") != live_hashes
        or revision.get("serialized_source_hashes_sha256") != _sha(live_hashes)
        or revision.get("source_revision_sha256") != _sha(body)
    ):
        raise RuntimeError("serialized recovery source revision disagrees")
    _require_commit(revision.get("commit"), field="source_revision.commit")
    _require_commit(revision.get("tree"), field="source_revision.tree")
    return revision


def build_recovery_request(
    *,
    source_revision: Mapping[str, Any],
    plan_artifact_path: str,
    expected_plan_file_sha256: str,
    expected_plan_sha256: str,
    expected_run_identity_sha256: str,
    expected_core_source_revision_sha256: str,
    task_identity_sha256: str,
    expected_task_output_artifact_path: str,
) -> dict[str, Any]:
    """Freeze one exact plan and task before starting the remote worker."""

    _artifact_path(
        plan_artifact_path,
        artifact_root=ARTIFACT_ROOT,
        field="plan_artifact_path",
    )
    _artifact_path(
        expected_task_output_artifact_path,
        artifact_root=ARTIFACT_ROOT,
        field="expected_task_output_artifact_path",
    )
    body: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "schema_version": REQUEST_SCHEMA_VERSION,
        "status": "FROZEN_SINGLE_TASK_RECOVERY_NO_DOWNSTREAM_AUTHORITY",
        **_AUTHORITY_FIELDS,
        "source_revision": dict(source_revision),
        "plan_artifact_path": plan_artifact_path,
        "expected_plan_file_sha256": _require_sha256(
            expected_plan_file_sha256, field="expected_plan_file_sha256"
        ),
        "expected_plan_sha256": _require_sha256(expected_plan_sha256, field="expected_plan_sha256"),
        "expected_run_identity_sha256": _require_sha256(
            expected_run_identity_sha256, field="expected_run_identity_sha256"
        ),
        "expected_core_source_revision_sha256": _require_sha256(
            expected_core_source_revision_sha256,
            field="expected_core_source_revision_sha256",
        ),
        "task_identity_sha256": _require_sha256(task_identity_sha256, field="task_identity_sha256"),
        "expected_task_output_artifact_path": expected_task_output_artifact_path,
    }
    return {**body, "request_sha256": _sha(body)}


def _validate_request(value: object, *, remote_root: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("recovery request must be an object")
    request = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *_AUTHORITY_FIELDS,
        "source_revision",
        "plan_artifact_path",
        "expected_plan_file_sha256",
        "expected_plan_sha256",
        "expected_run_identity_sha256",
        "expected_core_source_revision_sha256",
        "task_identity_sha256",
        "expected_task_output_artifact_path",
        "request_sha256",
    }
    body = {key: item for key, item in request.items() if key != "request_sha256"}
    if (
        set(request) != expected_fields
        or request.get("schema") != REQUEST_SCHEMA
        or request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or request.get("status") != "FROZEN_SINGLE_TASK_RECOVERY_NO_DOWNSTREAM_AUTHORITY"
        or any(request.get(field) is not False for field in _AUTHORITY_FIELDS)
        or request.get("request_sha256") != _sha(body)
    ):
        raise RuntimeError("recovery request identity or authority disagrees")
    _validate_source_revision(request["source_revision"], remote_root=remote_root)
    _artifact_path(
        str(request["plan_artifact_path"]),
        artifact_root=ARTIFACT_ROOT,
        field="plan_artifact_path",
    )
    _artifact_path(
        str(request["expected_task_output_artifact_path"]),
        artifact_root=ARTIFACT_ROOT,
        field="expected_task_output_artifact_path",
    )
    for field in (
        "expected_plan_file_sha256",
        "expected_plan_sha256",
        "expected_run_identity_sha256",
        "expected_core_source_revision_sha256",
        "task_identity_sha256",
    ):
        _require_sha256(request[field], field=field)
    return request


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    import sys

    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.semantic_trace_migration_mapreduce import (
        PLAN_FILENAME,
        execute_semantic_trace_migration_task,
        load_semantic_trace_migration_plan,
    )

    return {
        "PLAN_FILENAME": PLAN_FILENAME,
        "execute_semantic_trace_migration_task": execute_semantic_trace_migration_task,
        "load_semantic_trace_migration_plan": load_semantic_trace_migration_plan,
    }


def _load_exact_target(
    request: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
    loaded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Open and authenticate exactly one task from one immutable core plan."""

    plan_path = _artifact_path(
        str(request["plan_artifact_path"]),
        artifact_root=artifact_root,
        field="plan_artifact_path",
    )
    if plan_path.name != loaded["PLAN_FILENAME"]:
        raise RuntimeError("recovery input is not a semantic migration plan path")
    if not plan_path.is_file():
        raise RuntimeError(f"semantic migration plan is absent: {plan_path}")
    raw_plan = plan_path.read_bytes()
    if hashlib.sha256(raw_plan).hexdigest() != request["expected_plan_file_sha256"]:
        raise RuntimeError("semantic migration plan file SHA-256 disagrees")
    plan = loaded["load_semantic_trace_migration_plan"](
        plan_path,
        repo_root=repo_root,
    )
    if raw_plan != _canonical_bytes(plan) + b"\n":
        raise RuntimeError("semantic migration plan bytes are not canonical")
    expected_plan_path = _artifact_path(
        f"{plan['run_artifact_root']}/{loaded['PLAN_FILENAME']}",
        artifact_root=artifact_root,
        field="plan.run_artifact_root",
    )
    if plan_path.resolve() != expected_plan_path.resolve():
        raise RuntimeError("semantic migration plan is outside its declared run root")
    if (
        plan.get("plan_sha256") != request["expected_plan_sha256"]
        or plan.get("run_identity_sha256") != request["expected_run_identity_sha256"]
        or plan.get("source_revision", {}).get("source_revision_sha256")
        != request["expected_core_source_revision_sha256"]
        or plan.get("training_authorized") is not False
    ):
        raise RuntimeError("semantic migration plan identity or authority disagrees")
    matches = [
        dict(task)
        for task in plan.get("tasks", [])
        if isinstance(task, Mapping)
        and task.get("task_identity_sha256") == request["task_identity_sha256"]
    ]
    if len(matches) != 1:
        raise RuntimeError("recovery task is absent or duplicated in the exact plan")
    task = matches[0]
    if task.get("output_artifact_path") != request["expected_task_output_artifact_path"]:
        raise RuntimeError("recovery task output address disagrees")
    return plan, task, plan_path


def _build_receipt(
    *,
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    if result.get("task_identity_sha256") != task.get("task_identity_sha256") or result.get(
        "output_artifact_path"
    ) != task.get("output_artifact_path"):
        raise RuntimeError("production recovery result disagrees with the exact task")
    plan_binding = {
        "plan_artifact_path": request["plan_artifact_path"],
        "plan_file_sha256": plan_file_sha256,
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "source_revision_sha256": plan["source_revision"]["source_revision_sha256"],
        "implementation_revision": plan["implementation_revision"],
        "process_identity_sha256": plan["process_identity"]["process_identity_sha256"],
        "builder_identity_sha256": plan["builder_identity"]["identity_sha256"],
        "task_inventory_sha256": plan["task_inventory_sha256"],
    }
    task_binding = {
        key: task[key]
        for key in (
            "task_identity_sha256",
            "data_lane",
            "split",
            "source_artifact_path",
            "source_shard_sha256",
            "output_artifact_path",
        )
    }
    result_binding = {
        key: result[key]
        for key in (
            "task_identity_sha256",
            "output_artifact_path",
            "receipt_sha256",
            "counts",
            "reused",
        )
    }
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        **_AUTHORITY_FIELDS,
        "training_launched": False,
        "downstream_stage_launched": False,
        "request_sha256": request["request_sha256"],
        "wrapper_source_revision": dict(request["source_revision"]),
        "upstream_plan": plan_binding,
        "task": task_binding,
        "result": result_binding,
        "blockers": [
            "semantic_migration_reduction.not_run_by_recovery_wrapper",
            "active8.not_run_by_recovery_wrapper",
            "gate_zero.not_run_by_recovery_wrapper",
            "t1.not_run_by_recovery_wrapper",
            "bounded_p50.not_run_by_recovery_wrapper",
            "training.not_authorized",
        ],
    }
    return {**body, "receipt_sha256": _sha(body)}


def _write_immutable_json(path: Path, value: object) -> bool:
    encoded = _canonical_bytes(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable recovery receipt collision at {path}")
        return False
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".staging",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return True


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=8 * 3600,
    nonpreemptible=True,
    max_containers=2,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def recover_one_task(request: dict[str, object]) -> dict[str, object]:
    """Execute exactly one authenticated task and durably attest recovery."""

    validated_request = _validate_request(request, remote_root=REMOTE_ROOT)
    loaded = _imports()
    artifact_volume.reload()
    plan, task, plan_path = _load_exact_target(
        validated_request,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
        loaded=loaded,
    )
    result = loaded["execute_semantic_trace_migration_task"](
        plan,
        task["task_identity_sha256"],
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    receipt = _build_receipt(
        request=validated_request,
        plan=plan,
        task=task,
        result=result,
        plan_file_sha256=_file_sha256(plan_path),
    )
    receipt_path = _artifact_path(
        (f"{RECOVERY_OUTPUT_PREFIX}/{validated_request['request_sha256']}/{RECEIPT_FILENAME}"),
        artifact_root=ARTIFACT_ROOT,
        field="recovery receipt path",
    )
    _write_immutable_json(receipt_path, receipt)
    artifact_volume.commit()
    return {
        "status": RECEIPT_STATUS,
        **_AUTHORITY_FIELDS,
        "request_sha256": validated_request["request_sha256"],
        "task_identity_sha256": task["task_identity_sha256"],
        "recovery_receipt_artifact_path": _artifact_address(
            receipt_path, artifact_root=ARTIFACT_ROOT
        ),
        "recovery_receipt_sha256": receipt["receipt_sha256"],
        "recovery_receipt_file_sha256": _file_sha256(receipt_path),
        "production_task_receipt_sha256": result["receipt_sha256"],
        "task_result_reused": result["reused"],
    }


@app.local_entrypoint()
def main(
    plan_artifact_path: str,
    expected_plan_file_sha256: str,
    expected_plan_sha256: str,
    expected_run_identity_sha256: str,
    expected_core_source_revision_sha256: str,
    task_identity_sha256: str,
    expected_task_output_artifact_path: str,
    expected_commit: str,
) -> None:
    source_revision = local_source_revision(expected_commit=expected_commit)
    request = build_recovery_request(
        source_revision=source_revision,
        plan_artifact_path=plan_artifact_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        expected_plan_sha256=expected_plan_sha256,
        expected_run_identity_sha256=expected_run_identity_sha256,
        expected_core_source_revision_sha256=expected_core_source_revision_sha256,
        task_identity_sha256=task_identity_sha256,
        expected_task_output_artifact_path=expected_task_output_artifact_path,
    )
    result = recover_one_task.remote(request)
    print(
        json.dumps(
            {
                "phase": "semantic_migration_single_task_recovery_complete",
                "request": request,
                "result": result,
                "downstream_authority_granted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
