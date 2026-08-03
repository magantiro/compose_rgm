"""Build the Process-V2 chunk cache: one fused pass per source shard.

This launcher exists to make three properties true at run time rather than in a
docstring.

**Planning never needs Git.**  The image carries source files but no ``.git``,
so a remote driver that shells out to ``git rev-parse`` cannot plan at all.  The
clean image revision is therefore computed and verified *locally*, serialized
into the call, and revalidated remotely against the image's own files.  This is
the established semantic-v4 launcher pattern
(``materialize_editing_v2_semantic_v4_migration_app.local_source_revision``),
not a new one.

**The image revision is execution provenance and nothing else.**  It hashes the
launcher, the plan driver and every file under ``src`` and ``configs``, so it
moves whenever anything anywhere in the tree moves.  It used to be handed
straight to the library as the plan's scientific ``source_revision``, where it
was hashed into ``cache_physical_identity_sha256`` -- which meant editing any
file under ``src`` or ``configs`` relocated every cached byte.  The scientific
revision is now the library's own **narrow** ``cache_implementation_revision``,
computed by the library from the modules that actually decide what the cache
holds.  Both are validated in every remote body, and they are cross-checked: the
narrow revision's files must appear in the image inventory with identical
digests, so a caller cannot pair one tree's code with another tree's revision.

**The container bound is real.**  A decorator ``max_containers`` is a ceiling,
not a schedule, and a single ``starmap`` of every task submits every task at
once.  ``run_bounded_map`` submits deterministic sequential waves of at most the
requested size *and* lowers the installed autoscaler's own ceiling through
``Function.update_autoscaler``, which is the concurrency surface Modal 1.3.5
actually exposes.  ``Function.with_options`` does not exist in this version and
is not used.  Submitting at most N tasks is not the same as N simultaneously
live containers, and nothing here claims otherwise.

**Volume visibility is explicit.**  A Modal volume is a snapshot, not shared
memory.  ``reload()`` is called before the driver scans reusable work, at the
top of every worker and the reducer, and again in the driver after the reducer
returns, because each of those is a point where one container must see bytes
another container committed.

The cache is mechanical infrastructure.  Every envelope this app prints carries
the authority fields explicitly false, and no stage of it authorizes the rebind,
Active8, Gate 0, T1 or P50.  Run it explicitly, from a clean committed worktree,
with ``--detach``.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - import-time path setup
    sys.path.insert(0, str(ROOT / "src"))

from compose_v4.data.editing_v2_process_v2_chunk_cache import (  # noqa: E402
    CACHE_IMPLEMENTATION_FILES,
    DEFAULT_CACHE_MAP_CONTAINERS,
    PLATFORM_MAX_MAP_CONTAINERS,
    plan_submission_waves,
    run_bounded_map,
    validate_map_container_bound,
)

REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/build_process_v2_chunk_cache_app.py"
# The plan driver owns ``read_pinned_identities``, which recovers the payload's
# own historical identities and refuses a root that mixes migrations. It is
# imported rather than reimplemented, so it is serialized into the image.
PLAN_DRIVER_SOURCE = "scripts/plan_process_v2_rebind.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")

OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_chunk_cache"

# One fused pass is a single-threaded gzip and JSON loop, so a second CPU buys
# nothing; the previous app requested four for a sequential Python loop. Memory
# is sized from the local synthetic instrument: peak scan memory stayed under
# 5 MB for a 135 MB decompressed shard, so the request is dominated by the
# interpreter and one chunk buffer, not by the corpus.
MAP_CPU = 1.0
MAP_MEMORY_MB = 4096
REDUCE_CPU = 2.0
REDUCE_MEMORY_MB = 4096
MAP_TIMEOUT_SECONDS = 4 * 3600
REDUCE_TIMEOUT_SECONDS = 2 * 3600
DRIVER_TIMEOUT_SECONDS = 12 * 3600

# The *image* revision: execution-environment provenance, deliberately absent
# from every scientific identity. The narrow, artifact-addressing revision is
# `compose_v4.data.editing_v2_process_v2_chunk_cache.build_cache_implementation_revision`.
IMAGE_REVISION_SCHEMA = "compose.data.process_v2_chunk_cache_modal_image_revision"
IMAGE_REVISION_SCHEMA_VERSION = 2
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
for _source_directory in IMAGE_SOURCE_DIRECTORIES:
    image = image.add_local_dir(
        ROOT / _source_directory,
        str(REMOTE_ROOT / _source_directory),
        copy=True,
        ignore=("**/__pycache__/**", "**/*.pyc"),
    )
for _source_file in (LAUNCHER_SOURCE, PLAN_DRIVER_SOURCE):
    image = image.add_local_file(
        ROOT / _source_file, str(REMOTE_ROOT / _source_file), copy=True
    )

app = modal.App("compose-v4-process-v2-chunk-cache")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


# ---- Deterministic hashing -----------------------------------------------------


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


def _add_remote_paths(remote_root: Path = REMOTE_ROOT) -> None:
    """Put the image's own roots on ``sys.path`` inside a container."""

    for entry in (str(remote_root), str(Path(remote_root) / "scripts")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE, PLAN_DRIVER_SOURCE]
    for source_directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / source_directory).rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("the chunk-cache serialized source inventory repeats a path")
    return tuple(paths)


def _git(root: Path, *arguments: str) -> str:
    """Run Git. Called only from the *local* entrypoint, never in a container."""

    try:
        return subprocess.run(
            ("git", *arguments), cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot establish the chunk-cache Git identity: git {' '.join(arguments)}"
        ) from error


def local_image_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Compute and verify the clean image revision locally, once.

    The image has no ``.git``, and the ``debian_slim`` base has no ``git``
    binary either, so this cannot run remotely and must not be attempted there.
    This is execution-environment provenance: it proves the container runs the
    tree the launcher committed, and it addresses no artifact.
    """

    if not isinstance(expected_commit, str) or _COMMIT_RE.fullmatch(expected_commit) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "the Process-V2 chunk cache requires the exact clean committed worktree"
        )
    sources = {relative: _file_sha256(root / relative) for relative in serialized_source_paths(root)}
    tracked = set(_git(root, "ls-files").splitlines())
    if not sources or not set(sources).issubset(tracked):
        raise RuntimeError("every serialized chunk-cache source must be Git-tracked")
    body = {
        "schema": IMAGE_REVISION_SCHEMA,
        "schema_version": IMAGE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    return {**body, "image_revision_sha256": _sha256(body)}


def validate_remote_image_revision(
    value: dict[str, Any], *, remote_root: Path = REMOTE_ROOT
) -> dict[str, Any]:
    """Revalidate the supplied image revision against the image, without Git.

    Every serialized file is rehashed from the image, so a revision that names
    a different tree than the one actually baked in cannot pass.
    """

    if not isinstance(value, dict):
        raise RuntimeError("the chunk-cache image revision must be an object")
    body = {key: item for key, item in value.items() if key != "image_revision_sha256"}
    if (
        value.get("schema") != IMAGE_REVISION_SCHEMA
        or value.get("schema_version") != IMAGE_REVISION_SCHEMA_VERSION
        or value.get("worktree_clean") is not True
        or _COMMIT_RE.fullmatch(str(value.get("commit", ""))) is None
        or value.get("image_revision_sha256") != _sha256(body)
    ):
        raise RuntimeError("the chunk-cache image revision is malformed or self-inconsistent")
    sources = value.get("serialized_sources")
    expected = set(serialized_source_paths(Path(remote_root)))
    if not isinstance(sources, dict) or set(sources) != expected:
        raise RuntimeError(
            "the chunk-cache serialized source inventory disagrees with the image; "
            f"missing={sorted(expected - set(sources or {}))}, "
            f"extras={sorted(set(sources or {}) - expected)}"
        )
    for relative, digest in sources.items():
        if _file_sha256(Path(remote_root) / relative) != digest:
            raise RuntimeError(f"a serialized chunk-cache source differs in the image: {relative}")
    return value


def validate_remote_source_revision(
    cache_implementation_revision: dict[str, Any],
    image_revision: dict[str, Any],
    *,
    remote_root: Path = REMOTE_ROOT,
) -> dict[str, Any]:
    """Revalidate both revisions and cross-check them. Calls no Git.

    The *narrow* revision is the one that addresses the artifact, and the
    library owns it: ``validate_cache_implementation_revision`` recomputes it
    from the modules that are actually present and refuses anything else, so a
    caller-supplied value is a claim rather than an authority.

    The cross-check is by digest, not by commit.  Every behaviour-affecting
    module the narrow revision names must appear in the image inventory with the
    identical hash, so pairing one tree's code with another tree's revision
    object fails on the exact file that differs.

    That cross-check is currently unreachable by construction, and is kept
    deliberately: both validators rehash the same image, and the image inventory
    is a superset of the narrow one, so if both pass they cannot disagree.  It
    exists so that a future change to either file set -- a narrow module that
    stops being serialized, an image that stops carrying ``src`` whole -- becomes
    a loud refusal instead of a silent gap.
    """

    if str(remote_root) not in sys.path:
        sys.path.insert(0, str(remote_root))
    from compose_v4.data.editing_v2_process_v2_chunk_cache import (
        validate_cache_implementation_revision,
    )

    validate_remote_image_revision(image_revision, remote_root=remote_root)
    revision = validate_cache_implementation_revision(
        cache_implementation_revision, repo_root=Path(remote_root)
    )
    serialized = image_revision.get("serialized_sources") or {}
    disagreeing = [
        relative
        for relative in CACHE_IMPLEMENTATION_FILES
        if serialized.get(relative) != revision["implementation_files"][relative]
    ]
    if disagreeing:
        raise RuntimeError(
            "the chunk-cache implementation revision and the image revision describe "
            f"different files: {sorted(disagreeing)}"
        )
    return revision


def authority_envelope() -> dict[str, bool]:
    """Building a cache authorizes nothing. Stated, never implied."""

    from compose_v4.data.editing_v2_process_v2_schema import authority_false_block

    return {"training_launched": False, "cache_materialized": True, **authority_false_block()}


# ---- Remote functions ----------------------------------------------------------


@app.function(
    image=image,
    cpu=MAP_CPU,
    memory=MAP_MEMORY_MB,
    timeout=MAP_TIMEOUT_SECONDS,
    max_containers=PLATFORM_MAX_MAP_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def build_one_source_cache(
    plan: dict[str, Any],
    task_identity_sha256: str,
    cache_implementation_revision: dict[str, Any],
    image_revision: dict[str, Any],
) -> dict[str, Any]:
    """One source shard, one fused pass, one atomic publication."""

    from compose_v4.data.editing_v2_process_v2_chunk_cache import (
        ProcessV2ChunkCacheError,
        execute_process_v2_chunk_cache_task,
        write_chunk_cache_refusal_report,
    )

    validate_remote_source_revision(cache_implementation_revision, image_revision)
    # This worker reads a shard another container wrote; the volume snapshot it
    # booted with may predate it.
    artifact_volume.reload()
    try:
        result = execute_process_v2_chunk_cache_task(
            plan,
            task_identity_sha256,
            artifact_root=ARTIFACT_ROOT,
            repo_root=REMOTE_ROOT,
        )
    except ProcessV2ChunkCacheError as error:
        # Nothing was published under the run namespace; the report goes to a
        # separate content-addressed diagnostic namespace.
        write_chunk_cache_refusal_report(
            diagnostic_root=ARTIFACT_ROOT / "editing_v2" / "process_v2_chunk_cache_refusals",
            plan=plan,
            stage="build_one_source_cache",
            error=error,
            detail={"task_identity_sha256": task_identity_sha256},
        )
        artifact_volume.commit()
        raise
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    cpu=REDUCE_CPU,
    memory=REDUCE_MEMORY_MB,
    timeout=REDUCE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def reduce_chunk_cache(
    plan: dict[str, Any],
    cache_implementation_revision: dict[str, Any],
    image_revision: dict[str, Any],
) -> dict[str, Any]:
    """Only the serialized reducer may declare the cache complete."""

    from compose_v4.data.editing_v2_process_v2_chunk_cache import reduce_process_v2_chunk_cache

    validate_remote_source_revision(cache_implementation_revision, image_revision)
    # Reduction reads twenty source caches written by twenty other containers.
    artifact_volume.reload()
    completion = reduce_process_v2_chunk_cache(
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
    semantic_migration_completion: str,
    cache_implementation_revision: dict[str, Any],
    image_revision: dict[str, Any],
    output_artifact_prefix: str,
    records_per_chunk: int,
    max_map_containers: int,
) -> dict[str, Any]:
    """Bind the exact completion, plan, resume, map in waves, then reduce.

    No Git call appears anywhere in this body or anything it calls: both
    revisions arrive already computed and are revalidated against the image.
    """

    _add_remote_paths()

    from compose_v4.data.editing_v2_process_v2_chunk_cache import (
        completed_process_v2_chunk_cache_task_ids,
        plan_process_v2_chunk_cache,
        write_process_v2_chunk_cache_plan,
    )
    from compose_v4.data.editing_v2_process_v2_completion_binder import (
        PRODUCTION_COMPLETION_EXPECTATION,
        bind_exact_semantic_migration_completion,
    )
    from plan_process_v2_rebind import read_pinned_identities

    revision = validate_remote_source_revision(cache_implementation_revision, image_revision)
    bound = validate_map_container_bound(max_map_containers)
    # Before reading anything the caller named on the volume.
    artifact_volume.reload()

    expectation = PRODUCTION_COMPLETION_EXPECTATION
    if semantic_migration_completion != expectation.completion_artifact_path:
        raise RuntimeError(
            "the chunk cache is pinned to the exact declared completion path; "
            f"{semantic_migration_completion!r} is not it"
        )
    pinned_process_identity, pinned_builder_identity = pinned_identities_for(
        semantic_migration_completion,
        artifact_root=ARTIFACT_ROOT,
        expected_process_identity_sha256=expectation.process_identity_sha256,
        expected_builder_identity_sha256=expectation.builder_identity_sha256,
        read_pinned_identities=read_pinned_identities,
    )
    binding = bind_exact_semantic_migration_completion(
        completion_artifact_path=semantic_migration_completion,
        artifact_root=ARTIFACT_ROOT,
        pinned_process_identity=pinned_process_identity,
        pinned_builder_identity=pinned_builder_identity,
        expectation=expectation,
    )
    plan = plan_process_v2_chunk_cache(
        binding,
        repo_root=REMOTE_ROOT,
        cache_implementation_revision=revision,
        output_artifact_prefix=output_artifact_prefix,
        records_per_chunk=records_per_chunk,
    )
    write_process_v2_chunk_cache_plan(
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    artifact_volume.commit()

    # Before scanning reusable work: a previous run's outputs live in a snapshot
    # this container did not necessarily boot with.
    artifact_volume.reload()
    complete = completed_process_v2_chunk_cache_task_ids(
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    missing = [
        str(task["task_identity_sha256"])
        for task in plan["tasks"]
        if str(task["task_identity_sha256"]) not in complete
    ]
    waves = plan_submission_waves(missing, bound) if missing else ()
    print(
        json.dumps(
            {
                "phase": "process_v2_chunk_cache_map_plan",
                "expected_task_count": int(plan["expected_task_count"]),
                "already_complete": len(complete),
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
            submit=lambda wave: build_one_source_cache.starmap(
                [(plan, task_id, revision, image_revision) for task_id in wave]
            ),
            set_autoscaler=build_one_source_cache.update_autoscaler,
        )

    completion = reduce_chunk_cache.remote(plan, revision, image_revision)
    # After the reducer returns and before resolving completion: the completion
    # document was written by that other container.
    artifact_volume.reload()
    return {
        "phase": "process_v2_chunk_cache_complete",
        "run_artifact_root": plan["run_artifact_root"],
        "cache_implementation_sha256": completion["cache_implementation_sha256"],
        # Execution-environment provenance, reported and never hashed into the
        # artifact address.
        "image_revision_sha256": image_revision["image_revision_sha256"],
        "cache_semantic_identity_sha256": completion["cache_semantic_identity_sha256"],
        "cache_physical_identity_sha256": completion["cache_physical_identity_sha256"],
        "completion_sha256": completion["completion_sha256"],
        "source_count": completion["source_count"],
        "entries": completion["entries"],
        "chunk_count": completion["chunk_count"],
        "max_map_containers": bound,
        "submission_waves": len(waves),
        **authority_envelope(),
    }


def pinned_identities_for(
    completion_artifact_path: str,
    *,
    artifact_root: Path,
    expected_process_identity_sha256: str,
    expected_builder_identity_sha256: str,
    read_pinned_identities: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover the payload's own pinned identities and require both to agree.

    The identity objects are read from the immutable receipts by the plan
    driver's ``read_pinned_identities``, which already refuses a payload root
    that mixes migrations and one whose process identity is not the expected
    historical value.  Only the builder check is added here.
    """

    payload_root = Path(artifact_root) / PurePosixPath(
        completion_artifact_path
    ).parent.relative_to("/artifacts")
    process_identity, builder_identity = read_pinned_identities(
        payload_root,
        expected_process_identity_sha256=expected_process_identity_sha256,
    )
    if builder_identity.get("identity_sha256") != expected_builder_identity_sha256:
        raise RuntimeError(
            f"the payload builder identity is {builder_identity.get('identity_sha256')}, "
            f"not the declared {expected_builder_identity_sha256}"
        )
    return process_identity, builder_identity


@app.local_entrypoint()
def main(
    expected_commit: str,
    semantic_migration_completion: str = "",
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
    records_per_chunk: int = 0,
    max_map_containers: int = DEFAULT_CACHE_MAP_CONTAINERS,
) -> None:
    from compose_v4.data.editing_v2_process_v2_chunk_cache import (
        DEFAULT_RECORDS_PER_CHUNK,
        build_cache_implementation_revision,
    )
    from compose_v4.data.editing_v2_process_v2_completion_binder import (
        PRODUCTION_COMPLETION_ARTIFACT_PATH,
    )

    bound = validate_map_container_bound(int(max_map_containers))
    completion_path = semantic_migration_completion or PRODUCTION_COMPLETION_ARTIFACT_PATH
    # Computed here, where `.git` exists. The remote side revalidates both
    # against the image and never runs Git.
    image_revision = local_image_revision(expected_commit=expected_commit)
    cache_implementation_revision = build_cache_implementation_revision(repo_root=ROOT)
    report = driver.remote(
        semantic_migration_completion=completion_path,
        cache_implementation_revision=cache_implementation_revision,
        image_revision=image_revision,
        output_artifact_prefix=output_artifact_prefix,
        records_per_chunk=int(records_per_chunk) or DEFAULT_RECORDS_PER_CHUNK,
        max_map_containers=bound,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
