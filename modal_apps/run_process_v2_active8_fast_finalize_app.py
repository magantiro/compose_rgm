"""Publish a prepared Process-V2 Active8 run without replaying its reduction.

The remote function performs content-hash verification only.  It never opens a
molecular cache, decodes an Active8 row, enumerates a rewrite fiber, or trains a
model.  Importing this module launches nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_active8_fast_finalize_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
PLAN_FILENAME = "PROCESS_V2_ACTIVE8_PLAN.json"
PREPARATION_FILENAME = "PROCESS_V2_ACTIVE8_REDUCTION_PREPARED.json"
COMPLETION_FILENAME = "PROCESS_V2_ACTIVE8_COMPLETE.json"
RECEIPT_FILENAME = "PROCESS_V2_ACTIVE8_FAST_FINALIZATION_RECEIPT.json"
TIMEOUT_SECONDS = 2 * 60 * 60
HASH_WORKERS = 16

REVISION_SCHEMA = "compose.data.process_v2_active8_fast_finalize_modal_image_revision"
REVISION_SCHEMA_VERSION = 1
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy==1.26.4",
        "networkx==3.3",
        "rdkit==2024.3.5",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src"), "PYTHONUNBUFFERED": "1"})
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

app = modal.App("compose-v4-process-v2-active8-fast-finalize")
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
            f"cannot establish fast-finalizer Git identity: git {' '.join(arguments)}"
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
        raise RuntimeError("the fast-finalizer serialized source inventory repeats a path")
    return tuple(paths)


def local_image_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind a clean exact commit and every byte serialized into the image."""

    if _COMMIT_RE.fullmatch(str(expected_commit)) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError("the Active8 fast finalizer requires the exact clean commit")
    sources = {
        relative: _file_sha256(root / relative)
        for relative in _serialized_source_paths(root)
    }
    tracked = set(_git(root, "ls-files").splitlines())
    if not set(sources).issubset(tracked):
        raise RuntimeError("every serialized fast-finalizer source must be Git-tracked")
    body = {
        "schema": REVISION_SCHEMA,
        "schema_version": REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    return {**body, "image_revision_sha256": _sha256(body)}


def _validate_remote_revision(value: dict[str, Any]) -> None:
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
        or not isinstance(sources, dict)
        or set(sources) != set(_serialized_source_paths(REMOTE_ROOT))
    ):
        raise RuntimeError("the Active8 fast-finalizer image revision disagrees")
    for relative, digest in sources.items():
        if _file_sha256(REMOTE_ROOT / relative) != digest:
            raise RuntimeError(f"serialized fast-finalizer source differs: {relative}")
    _VALIDATED_REVISION_SHA256 = supplied


def _require_artifact_path(value: str, *, field: str) -> Path:
    path = Path(value)
    try:
        path.relative_to(ARTIFACT_ROOT)
    except ValueError as error:
        raise ValueError(f"{field} must be below {ARTIFACT_ROOT}") from error
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a normalized absolute artifact path")
    return path


@app.function(
    image=image,
    cpu=4.0,
    memory=8 * 1024,
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def fast_finalize_remote(run_root: str, revision: dict[str, Any]) -> dict[str, Any]:
    """Hash frozen payloads once and publish the canonical Active8 completion."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    mounted_run_root = _require_artifact_path(run_root, field="run_root")
    plan_path = mounted_run_root / PLAN_FILENAME
    preparation_path = mounted_run_root / PREPARATION_FILENAME
    if not plan_path.is_file() or not preparation_path.is_file():
        raise RuntimeError("the fast finalizer requires the plan and reduction preparation")

    source_root = str(REMOTE_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.editing_v2_process_v2_active8_fast_finalize import (
        fast_finalize_process_v2_active8,
    )
    from compose_v4.data.editing_v2_process_v2_active8_plan import (
        load_process_v2_active8_plan,
    )
    from compose_v4.data.editing_v2_process_v2_active8_reduce import (
        load_process_v2_active8_reduction_preparation,
    )

    plan = load_process_v2_active8_plan(plan_path, repo_root=REMOTE_ROOT)
    prepared = load_process_v2_active8_reduction_preparation(
        str(plan["run_artifact_root"]),
        active8_plan=plan,
        artifact_root=ARTIFACT_ROOT,
    )
    print(
        json.dumps(
            {
                "phase": "fast_finalization_started",
                "task_count": len(plan["tasks"]),
                "sentinel_partition_count": len(prepared["sentinel_plan"]["partitions"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completion, receipt = fast_finalize_process_v2_active8(
        plan,
        prepared,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
        publisher_revision=revision,
        hash_workers=HASH_WORKERS,
        publish=True,
    )
    artifact_volume.commit()
    if not (mounted_run_root / COMPLETION_FILENAME).is_file() or not (
        mounted_run_root / RECEIPT_FILENAME
    ).is_file():
        raise RuntimeError("the fast finalizer returned without publishing both artifacts")
    result = {
        "phase": "fast_finalization_complete",
        "run_root": str(mounted_run_root),
        "completion_sha256": completion["completion_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "verified_task_count": receipt["verified_task_count"],
        "verified_sentinel_partition_count": receipt[
            "verified_sentinel_partition_count"
        ],
        "image_revision_sha256": revision["image_revision_sha256"],
        "training_launched": False,
        "training_authorized": False,
        "bounded_p50_authorized": False,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main(run_root: str, expected_commit: str) -> None:
    """Spawn one disconnect-safe exact-commit fast finalizer."""

    revision = local_image_revision(expected_commit=expected_commit)
    call = fast_finalize_remote.spawn(run_root, revision)
    print(
        json.dumps(
            {
                "phase": "fast_finalization_launched",
                "call_id": call.object_id,
                "run_root": run_root,
                "image_revision": revision,
                "training_launched": False,
                "training_authorized": False,
                "bounded_p50_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
