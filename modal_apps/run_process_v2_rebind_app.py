"""Prove a completed V1 semantic migration against the Process-V2 authority.

The V1 migration payload is consumed as **immutable chemical data**.  Nothing is
relabelled: every record keeps its V1 identity, and the V2 admission decision is
recorded alongside it.  The driver freezes a content-addressed plan, resumes only
the range tasks that are not already durably present, maps them across
independent CPU containers, and hands off to the strict reducer.

Three properties this app must preserve, and how it does:

**No partial publication.**  A task publishes by renaming a private staging
directory, so a container that dies leaves the task simply not complete rather
than half written.  An integrity mismatch raises before any staging directory is
created, so it publishes nothing at all.  Only the reducer may declare the run
complete, and it independently re-proves that every planned task is present with
no gap, duplicate, or extra.

**No training authority.**  Every envelope this app prints carries the authority
fields explicitly false.  Proving the corpus is not authorization to train on it,
and Active8 materialization, Gate 0, T1 and P50 all remain separately gated.

**The schedule cannot change the artifact.**  Worker count is chosen here, at map
time.  It is deliberately not an input to the plan: `entries_per_task` is folded
into `run_identity_sha256`, so letting fleet size pick it would address a
different run for the same data.

This app launches nothing by importing it.  Run it explicitly, from a clean
committed worktree, with `--detach`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_rebind_app.py"
PLAN_DRIVER_SOURCE = "scripts/plan_process_v2_rebind.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")

OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_rebind"
# The owner's fleet bound. Each range task publishes its own immutable directory
# by rename, so tasks never contend for one destination; this bounds concurrent
# volume writers, which is the real constraint. The repository's cache and pack
# apps use 5 for Volume v1 small-commit writers, and its per-task-receipt
# inventory app uses 64. 20 sits between them by instruction and is overridable.
MAX_MAP_CONTAINERS = 20
MAP_CPU = 4.0
MAP_MEMORY_MB = 16384
MAP_TIMEOUT_SECONDS = 8 * 3600
REDUCE_TIMEOUT_SECONDS = 8 * 3600
DRIVER_TIMEOUT_SECONDS = 24 * 3600

IMAGE_REVISION_SCHEMA = "compose.data.process_v2_rebind_modal_image_revision"
IMAGE_REVISION_SCHEMA_VERSION = 1
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
            "OMP_NUM_THREADS": "4",
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
for source_file in (LAUNCHER_SOURCE, PLAN_DRIVER_SOURCE):
    image = image.add_local_file(
        ROOT / source_file, str(REMOTE_ROOT / source_file), copy=True
    )

app = modal.App("compose-v4-process-v2-rebind")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


# ---- Deterministic hashing and Git identity ----


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
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
    paths = [LAUNCHER_SOURCE, PLAN_DRIVER_SOURCE]
    for source_directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / source_directory).rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("Process-V2 rebind serialized source inventory repeats a path")
    return tuple(paths)


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments), cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot establish Process-V2 rebind Git identity: git {' '.join(arguments)}"
        ) from error


def local_image_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Require the exact clean commit and hash every serialized image input.

    This is the *image* identity. The scientific source revision is the
    library's own `build_process_v2_rebind_source_revision`, which the plan
    carries and which every task and the reducer revalidate; the two are kept
    separate so the app cannot weaken the library's boundary.
    """

    if not isinstance(expected_commit, str) or _COMMIT_RE.fullmatch(expected_commit) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "Process-V2 rebind launch requires the exact clean committed worktree"
        )
    sources = {
        relative: _file_sha256(root / relative) for relative in _serialized_source_paths(root)
    }
    tracked = set(_git(root, "ls-files").splitlines())
    if not sources or not set(sources).issubset(tracked):
        raise RuntimeError("every Process-V2 rebind serialized source must be Git-tracked")
    body = {
        "schema": IMAGE_REVISION_SCHEMA,
        "schema_version": IMAGE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    return {**body, "image_revision_sha256": _sha256(body)}


def _validate_remote_image_revision(
    value: dict[str, Any], *, remote_root: Path = REMOTE_ROOT
) -> None:
    body = {key: item for key, item in value.items() if key != "image_revision_sha256"}
    if (
        value.get("schema") != IMAGE_REVISION_SCHEMA
        or value.get("schema_version") != IMAGE_REVISION_SCHEMA_VERSION
        or value.get("worktree_clean") is not True
        or value.get("image_revision_sha256") != _sha256(body)
    ):
        raise RuntimeError("Process-V2 rebind image revision disagrees")
    sources = value.get("serialized_sources")
    if not isinstance(sources, dict) or set(sources) != set(
        _serialized_source_paths(remote_root)
    ):
        raise RuntimeError("Process-V2 rebind serialized source inventory disagrees")
    for relative, digest in sources.items():
        if _file_sha256(Path(remote_root) / relative) != digest:
            raise RuntimeError(f"serialized Process-V2 rebind source differs: {relative}")


def _authority_envelope() -> dict[str, bool]:
    """Proving a corpus authorizes nothing downstream. Stated, never implied."""

    return {
        "training_launched": False,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
    }


# ---- Remote functions ----


@app.function(
    image=image,
    cpu=MAP_CPU,
    memory=MAP_MEMORY_MB,
    timeout=MAP_TIMEOUT_SECONDS,
    max_containers=MAX_MAP_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prove_one_range(
    plan: dict[str, Any], task_identity_sha256: str, image_revision: dict[str, Any]
) -> dict[str, Any]:
    """One independent range task: prove it, or publish nothing."""

    from compose_v4.data.editing_process_v2_rebind import execute_process_v2_rebind_task

    _validate_remote_image_revision(image_revision)
    result = execute_process_v2_rebind_task(
        plan,
        task_identity_sha256,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    cpu=MAP_CPU,
    memory=MAP_MEMORY_MB,
    timeout=REDUCE_TIMEOUT_SECONDS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def reduce_rebind(plan: dict[str, Any], image_revision: dict[str, Any]) -> dict[str, Any]:
    """Only the reducer may declare the run complete."""

    from compose_v4.data.editing_process_v2_rebind import reduce_process_v2_rebind

    _validate_remote_image_revision(image_revision)
    completion = reduce_process_v2_rebind(
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    artifact_volume.commit()
    return completion


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=DRIVER_TIMEOUT_SECONDS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def driver(
    *,
    v1_payload_root: str,
    expected_v1_process_identity: str,
    output_artifact_prefix: str,
    entries_per_task: int,
    max_map_containers: int,
    image_revision: dict[str, Any],
) -> dict[str, Any]:
    """Plan, publish, resume, map the missing ranges, then reduce."""

    import sys

    if str(REMOTE_ROOT) not in sys.path:
        sys.path.insert(0, str(REMOTE_ROOT))
    if str(REMOTE_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REMOTE_ROOT / "scripts"))

    from compose_v4.data.editing_process_v2_admitted_source import (
        resolve_process_v2_admitted_source,
    )
    from compose_v4.data.editing_process_v2_rebind import (
        completed_process_v2_rebind_task_ids,
        write_process_v2_rebind_plan,
    )
    from plan_process_v2_rebind import build_plan, plan_envelope

    _validate_remote_image_revision(image_revision)

    # The plan driver is imported, not reimplemented: the pinned-identity
    # discovery and the refusal to plan against an unexpected historical
    # process must be identical locally and here.
    plan = build_plan(
        artifact_root=ARTIFACT_ROOT,
        v1_payload_root_artifact_path=v1_payload_root,
        expected_process_identity_sha256=expected_v1_process_identity,
        output_artifact_prefix=output_artifact_prefix,
        entries_per_task=entries_per_task,
    )
    write_process_v2_rebind_plan(plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT)
    artifact_volume.commit()

    completed = completed_process_v2_rebind_task_ids(
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    missing = [
        str(task["task_identity_sha256"])
        for task in plan["tasks"]
        if str(task["task_identity_sha256"]) not in completed
    ]
    print(
        json.dumps(
            {
                "phase": "process_v2_rebind_map_plan",
                "expected_task_count": int(plan["expected_task_count"]),
                "already_complete": len(completed),
                "missing_tasks": len(missing),
                "max_map_containers": int(max_map_containers),
            }
        ),
        flush=True,
    )

    if missing:
        results = list(
            prove_one_range.starmap(
                [(plan, task_id, image_revision) for task_id in missing]
            )
        )
        if len(results) != len(missing):
            raise RuntimeError("Process-V2 rebind map lost or repeated a range task")

    completion = reduce_rebind.remote(plan, image_revision)

    # Resolving the admitted source here proves the downstream adapter can read
    # what was just published, before anything else is asked to consume it.
    admitted = resolve_process_v2_admitted_source(
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    return {
        "phase": "process_v2_rebind_complete",
        "plan": plan_envelope(plan, written=None),
        "completion_sha256": completion["completion_sha256"],
        "run_identity_sha256": completion["run_identity_sha256"],
        "run_artifact_root": plan["run_artifact_root"],
        "counts": completion["counts"],
        "rejected_traces_by_code": completion["rejected_traces_by_code"],
        "admitted_source_identity": admitted.identity(),
        **_authority_envelope(),
    }


@app.local_entrypoint()
def main(
    v1_payload_root: str,
    expected_commit: str,
    expected_v1_process_identity: str = "",
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
    entries_per_task: int = 0,
    max_map_containers: int = MAX_MAP_CONTAINERS,
) -> None:
    import sys

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from compose_v4.data.editing_process_v2_rebind import DEFAULT_ENTRIES_PER_TASK
    from compose_v4.rewrite.editing_v2_process_identity import (
        SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
    )

    if not 1 <= int(max_map_containers) <= MAX_MAP_CONTAINERS:
        raise RuntimeError(
            f"max_map_containers must lie in [1, {MAX_MAP_CONTAINERS}]; the map fan-out "
            "bounds concurrent volume writers"
        )
    resolved_identity = expected_v1_process_identity or SUPERSEDED_V1_PROCESS_IDENTITY_SHA256
    resolved_entries = int(entries_per_task) or DEFAULT_ENTRIES_PER_TASK

    image_revision = local_image_revision(expected_commit=expected_commit)
    report = driver.remote(
        v1_payload_root=v1_payload_root,
        expected_v1_process_identity=resolved_identity,
        output_artifact_prefix=output_artifact_prefix,
        entries_per_task=resolved_entries,
        max_map_containers=int(max_map_containers),
        image_revision=image_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
