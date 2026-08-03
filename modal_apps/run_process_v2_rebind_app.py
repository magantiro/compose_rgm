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

Three defects this module used to carry, and what replaced them:

**Planning could not run remotely.**  The driver called `build_plan`, which
called `repository_process_v2_rebind_source_revision`, which shells out to
`git rev-parse HEAD` in `/root/compose`.  The image has source files but no
`.git`, and `debian_slim` has no `git` binary, so that raised before any work
began.  The revision is now computed and verified *locally*, passed in, and
revalidated remotely against the image's own files, then cross-checked against
the image revision.  No Git call exists in any remote body.

**`--max-map-containers` was cosmetic.**  The value was validated and printed
while the decorator capped at 20 and one `starmap` submitted every task at once.
The bound is now real: `run_bounded_map` submits deterministic sequential waves
of at most the requested size and lowers the function's own ceiling through
`Function.update_autoscaler`, which is the surface Modal 1.3.5 exposes.

**Volume visibility was implicit.**  A Modal volume is a snapshot. `reload()`
now happens before the driver scans reusable tasks, at the top of every worker
and the reducer, and in the driver after the reducer returns.

This app launches nothing by importing it.  Run it explicitly, from a clean
committed worktree, with `--detach`.
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
if str(ROOT / "src") not in sys.path:  # pragma: no cover - import-time path setup
    sys.path.insert(0, str(ROOT / "src"))

from compose_v4.data.editing_v2_process_v2_chunk_cache import (  # noqa: E402
    DEFAULT_CHUNK_MAP_CONTAINERS,
    PLATFORM_MAX_MAP_CONTAINERS,
    plan_submission_waves,
    run_bounded_map,
    validate_map_container_bound,
)

REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_rebind_app.py"
PLAN_DRIVER_SOURCE = "scripts/plan_process_v2_rebind.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")

OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_rebind"
# The fleet bound is `PLATFORM_MAX_MAP_CONTAINERS`, imported from the shared
# Process-V2 module so the ceiling has one home. Each range task publishes its
# own immutable directory by rename, so tasks never contend for one destination;
# the bound exists to limit concurrent volume writers. The decorator sets the
# platform ceiling; `run_bounded_map` sets the per-run one.
DEFAULT_MAX_MAP_CONTAINERS = DEFAULT_CHUNK_MAP_CONTAINERS
# One range task is a sequential replay-and-hash loop with no BLAS in it, so the
# previous request of four CPUs paid for parallelism the code cannot use.
MAP_CPU = 2.0
MAP_MEMORY_MB = 8192
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


def validate_remote_source_revision(
    source_revision: dict[str, Any],
    image_revision: dict[str, Any],
    *,
    remote_root: Path = REMOTE_ROOT,
) -> dict[str, Any]:
    """Revalidate the supplied scientific source revision, without Git.

    Two separate identities meet here and both are required.  The *image*
    revision pins every serialized file this launcher shipped; the *source*
    revision is the library's own boundary, which the plan carries and which
    every task and the reducer revalidate.  They are cross-checked so a caller
    cannot pair one commit's code with another commit's revision object.

    ``validate_process_v2_rebind_source_revision`` rehashes the implementation
    files against the image and calls no Git, which is precisely why it can run
    where ``.git`` does not exist.
    """

    if str(remote_root) not in sys.path:
        sys.path.insert(0, str(remote_root))
    from compose_v4.data.editing_process_v2_rebind import (
        validate_process_v2_rebind_source_revision,
    )

    revision = validate_process_v2_rebind_source_revision(
        source_revision, repo_root=Path(remote_root)
    )
    if (
        revision["commit"] != image_revision.get("commit")
        or revision["tree"] != image_revision.get("tree")
    ):
        raise RuntimeError(
            "the supplied Process-V2 source revision and the image revision name "
            f"different trees: {revision['commit']} vs {image_revision.get('commit')}"
        )
    return revision


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
    max_containers=PLATFORM_MAX_MAP_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prove_one_range(
    plan: dict[str, Any],
    task_identity_sha256: str,
    image_revision: dict[str, Any],
    source_revision: dict[str, Any],
) -> dict[str, Any]:
    """One independent range task: prove it, or publish nothing."""

    from compose_v4.data.editing_process_v2_rebind import execute_process_v2_rebind_task

    _validate_remote_image_revision(image_revision)
    validate_remote_source_revision(source_revision, image_revision)
    # This worker reads the V1 payload and the plan another container wrote.
    artifact_volume.reload()
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
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def reduce_rebind(
    plan: dict[str, Any], image_revision: dict[str, Any], source_revision: dict[str, Any]
) -> dict[str, Any]:
    """Only the serialized reducer may declare the run complete."""

    from compose_v4.data.editing_process_v2_rebind import reduce_process_v2_rebind

    _validate_remote_image_revision(image_revision)
    validate_remote_source_revision(source_revision, image_revision)
    # Reduction reads every range result, each written by another container.
    artifact_volume.reload()
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
    max_containers=1,
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
    source_revision: dict[str, Any],
) -> dict[str, Any]:
    """Plan, publish, resume, map the missing ranges in waves, then reduce.

    No Git call appears in this body or in anything it calls: ``build_plan``
    receives the already-verified ``source_revision`` and validates it against
    the image's own files.
    """

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
    revision = validate_remote_source_revision(source_revision, image_revision)
    bound = validate_map_container_bound(max_map_containers)
    # Before reading the V1 payload the caller named.
    artifact_volume.reload()

    # The plan driver is imported, not reimplemented: the pinned-identity
    # discovery and the refusal to plan against an unexpected historical
    # process must be identical locally and here.
    plan = build_plan(
        artifact_root=ARTIFACT_ROOT,
        v1_payload_root_artifact_path=v1_payload_root,
        expected_process_identity_sha256=expected_v1_process_identity,
        output_artifact_prefix=output_artifact_prefix,
        entries_per_task=entries_per_task,
        source_revision=revision,
        repo_root=REMOTE_ROOT,
    )
    write_process_v2_rebind_plan(plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT)
    artifact_volume.commit()

    # Before scanning reusable tasks: a previous run's results live in a volume
    # snapshot this container did not necessarily boot with.
    artifact_volume.reload()
    completed = completed_process_v2_rebind_task_ids(
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    missing = [
        str(task["task_identity_sha256"])
        for task in plan["tasks"]
        if str(task["task_identity_sha256"]) not in completed
    ]
    waves = plan_submission_waves(missing, bound) if missing else ()
    print(
        json.dumps(
            {
                "phase": "process_v2_rebind_map_plan",
                "expected_task_count": int(plan["expected_task_count"]),
                "already_complete": len(completed),
                "missing_tasks": len(missing),
                "max_map_containers": bound,
                "submission_waves": len(waves),
            }
        ),
        flush=True,
    )

    if missing:
        run_bounded_map(
            missing,
            max_map_containers=bound,
            submit=lambda wave: prove_one_range.starmap(
                [(plan, task_id, image_revision, source_revision) for task_id in wave]
            ),
            set_autoscaler=prove_one_range.update_autoscaler,
        )

    completion = reduce_rebind.remote(plan, image_revision, source_revision)
    # After the reducer returns and before resolving completion: the completion
    # document was written by that other container.
    artifact_volume.reload()

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
        "max_map_containers": bound,
        "submission_waves": len(waves),
        **_authority_envelope(),
    }


@app.local_entrypoint()
def main(
    v1_payload_root: str,
    expected_commit: str,
    expected_v1_process_identity: str = "",
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
    entries_per_task: int = 0,
    max_map_containers: int = DEFAULT_MAX_MAP_CONTAINERS,
) -> None:
    from compose_v4.data.editing_process_v2_rebind import (
        DEFAULT_ENTRIES_PER_TASK,
        repository_process_v2_rebind_source_revision,
    )
    from compose_v4.rewrite.editing_v2_process_identity import (
        SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
    )

    bound = validate_map_container_bound(int(max_map_containers))
    resolved_identity = expected_v1_process_identity or SUPERSEDED_V1_PROCESS_IDENTITY_SHA256
    resolved_entries = int(entries_per_task) or DEFAULT_ENTRIES_PER_TASK

    image_revision = local_image_revision(expected_commit=expected_commit)
    # Computed here, where `.git` exists. The remote side revalidates it against
    # the image and never runs Git.
    source_revision = repository_process_v2_rebind_source_revision(repo_root=ROOT)
    if source_revision["commit"] != image_revision["commit"]:
        raise RuntimeError("the local source and image revisions name different commits")
    report = driver.remote(
        v1_payload_root=v1_payload_root,
        expected_v1_process_identity=resolved_identity,
        output_artifact_prefix=output_artifact_prefix,
        entries_per_task=resolved_entries,
        max_map_containers=bound,
        image_revision=image_revision,
        source_revision=source_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
