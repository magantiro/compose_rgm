"""Build the CPU-only 20-source semantic Active8 mechanical chunk cache.

The caller supplies the exact ``SEMANTIC_MIGRATION_COMPLETE.json``.  The driver
validates that migration once, freezes one cache task per lane/role source,
fans out at most 20 CPU workers, and invokes a strict reducer.  The reducer
reloads all 20 cache completions before publishing a content-addressed global
completion.  This app performs no chemistry decision or Active8 admission and
grants no Gate 0, T1, P50, training, checkpoint, or final-test authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/build_semantic_active8_chunk_cache_app.py"
OUTPUT_ARTIFACT_ROOT = "/artifacts/editing_v2/semantic_active8_chunk_cache_v1"
MAX_MAP_CONTAINERS = 20
EXPECTED_SOURCE_TASKS = 20
DEFAULT_TARGET_ROWS_PER_CHUNK = 500

SOURCE_REVISION_SCHEMA = "compose.data.semantic_active8_chunk_cache_modal_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
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
    .add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
    .add_local_file(
        ROOT / LAUNCHER_SOURCE,
        str(REMOTE_ROOT / LAUNCHER_SOURCE),
        copy=True,
    )
)

app = modal.App("compose-v4-semantic-active8-chunk-cache")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


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


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    """Return every Python source serialized into the Modal image."""

    source_root = Path(root) / "src" / "compose_v4"
    paths = [LAUNCHER_SOURCE]
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted(source_root.rglob("*.py"))
        if path.is_file()
    )
    return tuple(paths)


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
            f"cannot establish semantic chunk-cache Git identity: git {' '.join(arguments)}"
        ) from error


def _require_artifact_path(value: str, *, field: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise RuntimeError(f"{field} must be a normalized path below /artifacts")
    return value


def _mounted_artifact_path(value: str, *, field: str) -> Path:
    address = _require_artifact_path(value, field=field)
    pure = PurePosixPath(address)
    resolved = (ARTIFACT_ROOT / Path(*pure.parts[2:])).resolve()
    try:
        resolved.relative_to(ARTIFACT_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"{field} resolves outside /artifacts") from error
    return resolved


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    import sys

    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
        resolve_editing_v2_semantic_active8_sources,
    )
    from compose_v4.data.semantic_active8_chunk_cache import (
        semantic_active8_chunk_cache_builder_identity,
    )
    from compose_v4.data.semantic_active8_chunk_cache_mapreduce import (
        completed_semantic_active8_chunk_cache_task_ids,
        execute_semantic_active8_chunk_cache_task,
        plan_semantic_active8_chunk_cache,
        reduce_semantic_active8_chunk_caches,
        write_semantic_active8_chunk_cache_plan,
    )

    return {
        "resolve_editing_v2_semantic_active8_sources": (
            resolve_editing_v2_semantic_active8_sources
        ),
        "semantic_active8_chunk_cache_builder_identity": (
            semantic_active8_chunk_cache_builder_identity
        ),
        "completed_semantic_active8_chunk_cache_task_ids": (
            completed_semantic_active8_chunk_cache_task_ids
        ),
        "execute_semantic_active8_chunk_cache_task": (execute_semantic_active8_chunk_cache_task),
        "plan_semantic_active8_chunk_cache": plan_semantic_active8_chunk_cache,
        "reduce_semantic_active8_chunk_caches": reduce_semantic_active8_chunk_caches,
        "write_semantic_active8_chunk_cache_plan": (write_semantic_active8_chunk_cache_plan),
    }


def local_source_revision(
    *,
    expected_commit: str,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Require one exact clean commit/tree and hash every serialized source."""

    if not isinstance(expected_commit, str) or _COMMIT_RE.fullmatch(expected_commit) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "semantic chunk-cache launch requires the exact clean committed worktree"
        )
    loaded = _imports(root)
    sources = {
        relative: _file_sha256(root / relative) for relative in _serialized_source_paths(root)
    }
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
        "chunk_builder_identity": loaded["semantic_active8_chunk_cache_builder_identity"](),
    }
    return {**body, "source_revision_sha256": _sha256(body)}


def _validate_remote_source_revision(
    value: dict[str, Any],
    *,
    loaded: dict[str, Any],
    remote_root: Path = REMOTE_ROOT,
) -> None:
    body = {key: item for key, item in value.items() if key != "source_revision_sha256"}
    if (
        value.get("schema") != SOURCE_REVISION_SCHEMA
        or value.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or value.get("worktree_clean") is not True
        or value.get("source_revision_sha256") != _sha256(body)
        or value.get("chunk_builder_identity")
        != loaded["semantic_active8_chunk_cache_builder_identity"]()
    ):
        raise RuntimeError("semantic chunk-cache source revision disagrees")
    sources = value.get("serialized_sources")
    expected_sources = set(_serialized_source_paths(remote_root))
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise RuntimeError("semantic chunk-cache serialized source inventory disagrees")
    for relative, expected_sha256 in sources.items():
        if _file_sha256(Path(remote_root) / relative) != expected_sha256:
            raise RuntimeError(f"serialized semantic chunk-cache source differs: {relative}")


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=4 * 3600,
    max_containers=MAX_MAP_CONTAINERS,
    volumes={"/artifacts": artifact_volume},
)
def build_one_source_cache(
    plan: dict[str, Any],
    task_identity_sha256: str,
) -> dict[str, Any]:
    """Build or verify one mechanical source cache and commit it durably."""

    loaded = _imports()
    _validate_remote_source_revision(plan["source_revision"], loaded=loaded)
    artifact_volume.reload()
    receipt = loaded["execute_semantic_active8_chunk_cache_task"](
        plan,
        task_identity_sha256,
        artifact_root=ARTIFACT_ROOT,
    )
    artifact_volume.commit()
    return {
        "task_identity_sha256": task_identity_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=4 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def reduce_global_cache(plan: dict[str, Any]) -> dict[str, Any]:
    """Strictly reload all 20 caches and publish the global completion."""

    loaded = _imports()
    _validate_remote_source_revision(plan["source_revision"], loaded=loaded)
    artifact_volume.reload()
    result = loaded["reduce_semantic_active8_chunk_caches"](
        plan,
        artifact_root=ARTIFACT_ROOT,
    )
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=24 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def driver(
    semantic_migration_completion: str,
    output_artifact_root: str,
    target_rows_per_chunk: int,
    source_revision: dict[str, Any],
) -> dict[str, Any]:
    """Validate the caller's migration once, fan out missing tasks, reduce."""

    if type(target_rows_per_chunk) is not int or target_rows_per_chunk <= 0:
        raise ValueError("target_rows_per_chunk must be a positive integer")
    loaded = _imports()
    _validate_remote_source_revision(source_revision, loaded=loaded)
    completion_path = _mounted_artifact_path(
        semantic_migration_completion,
        field="semantic_migration_completion",
    )
    _require_artifact_path(output_artifact_root, field="output_artifact_root")
    artifact_volume.reload()
    source_inventory = loaded["resolve_editing_v2_semantic_active8_sources"](
        completion_path,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    if len(source_inventory.sources) != EXPECTED_SOURCE_TASKS:
        raise RuntimeError("semantic migration did not resolve exactly 20 sources")
    plan = loaded["plan_semantic_active8_chunk_cache"](
        source_inventory,
        source_revision=source_revision,
        output_artifact_root=output_artifact_root,
        target_rows_per_chunk=target_rows_per_chunk,
    )
    loaded["write_semantic_active8_chunk_cache_plan"](
        plan,
        artifact_root=ARTIFACT_ROOT,
    )
    artifact_volume.commit()
    completed = loaded["completed_semantic_active8_chunk_cache_task_ids"](
        plan,
        artifact_root=ARTIFACT_ROOT,
    )
    missing = [
        task["task_identity_sha256"]
        for task in plan["tasks"]
        if task["task_identity_sha256"] not in completed
    ]
    if missing:
        results = list(
            build_one_source_cache.starmap(
                [(plan, task_identity_sha256) for task_identity_sha256 in missing]
            )
        )
        if len(results) != len(missing):
            raise RuntimeError("semantic chunk-cache map lost source results")
    return reduce_global_cache.remote(plan)


@app.local_entrypoint()
def main(
    semantic_migration_completion: str,
    expected_commit: str,
    output_artifact_root: str = OUTPUT_ARTIFACT_ROOT,
    target_rows_per_chunk: int = DEFAULT_TARGET_ROWS_PER_CHUNK,
):
    """Wait for the durable CPU driver and print its non-authorizing result."""

    _require_artifact_path(
        semantic_migration_completion,
        field="semantic_migration_completion",
    )
    _require_artifact_path(output_artifact_root, field="output_artifact_root")
    source_revision = local_source_revision(expected_commit=expected_commit)
    result = driver.remote(
        semantic_migration_completion,
        output_artifact_root,
        target_rows_per_chunk,
        source_revision,
    )
    print(
        json.dumps(
            {
                "semantic_migration_completion": semantic_migration_completion,
                "output_artifact_root": output_artifact_root,
                "target_rows_per_chunk": target_rows_per_chunk,
                "source_revision": source_revision,
                "source_task_count": EXPECTED_SOURCE_TASKS,
                "max_map_containers": MAX_MAP_CONTAINERS,
                "training_launched": False,
                "active8_admission_run": False,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
