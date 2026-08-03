"""Compute Process-V2 Active8 decisions on Modal.  Never yet executed.

This is the map/reduce stage: the corpus is a committed chunk cache and a
committed rebind generation, one task decides one chunk, and only the serialized
reducer may declare the generation complete.  Gate 0 consumes what this
publishes and is a separate, single-container streaming job -- deliberately not
another fan-out system.

FAN-OUT IS BOUNDED AT FORTY, TWICE
----------------------------------
``PLATFORM_MAX_MAP_CONTAINERS`` is the platform ceiling and is imported from the
shared Process-V2 module so the bound has one home.  It is applied in the
decorator, which is the platform ceiling, AND through ``run_bounded_map``, which
submits deterministic sequential waves and lowers the function's own autoscaler.
Both are needed: the chunk cache launcher previously validated and printed a
bound while a single ``starmap`` submitted every task at once, so the bound was
cosmetic.  Neither mechanism promises forty simultaneously live containers; the
guarantee is an upper bound, which is the direction that matters for a volume
writer.

RESTART IS RECEIPT-EXACT, NOT OPTIMISTIC
----------------------------------------
A task publishes by renaming a private staging directory, so an interrupted
container leaves its task simply not complete rather than half written.  Before
mapping, the driver reloads the volume and asks the stage which task identities
already have a *valid* receipt; only the remainder are submitted.  A receipt is
reused because it was verified, never because a file exists at the expected
path.

WHAT IT REFUSES
---------------
A dirty or unexpected worktree; a container whose files differ from the ones
bound; an uncommitted chunk cache or rebind generation, which are absent rather
than partial because their completion markers are written last; an input that
grants authority at any depth; and a decision generation computed under a
process identity that is not the live Process-V2 one.

NOTHING RUNS ON IMPORT, AND THE STAGE IS RESOLVED AT EXECUTION
--------------------------------------------------------------
The Active8 policy, map/reduce and index modules are named here as constants and
imported inside the remote bodies.  That keeps this launcher importable and
testable on its own, and turns an absent stage into a refusal that names the
exact missing module or symbol instead of an import error at collection time.
``describe`` prints the exact command and artifact roots a launch WOULD use and
executes nothing.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - import-time path setup
    sys.path.insert(0, str(ROOT / "src"))

from compose_v4.data.editing_v2_process_v2_chunk_cache import (  # noqa: E402
    PLATFORM_MAX_MAP_CONTAINERS,
    plan_submission_waves,
    run_bounded_map,
    validate_map_container_bound,
)
from compose_v4.data.editing_v2_process_v2_launch_binding import (  # noqa: E402
    build_implementation_revision,
    require_tracked,
    validate_implementation_revision,
)
from compose_v4.data.editing_v2_process_v2_schema import (  # noqa: E402
    AUTHORITY_FIELDS,
    authority_false_block,
    canonical_sha256,
    require_no_granted_authority,
)

REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_active8_decisions_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_active8_decisions"

#: The Active8 stage, owned by the Active8 workstream.  Declared as constants and
#: resolved at EXECUTION: this launcher is written against the frozen index
#: protocol, so it must import and be testable before the stage exists, and an
#: absent stage must refuse by name rather than break collection.  This is the
#: declared integration seam; if the stage lands under other names, exactly this
#: mapping moves.
ACTIVE8_STAGE_SYMBOLS: Mapping[str, tuple[str, ...]] = {
    "compose_v4.data.editing_v2_process_v2_active8_policy": (
        "build_process_v2_active8_admission_policy",
    ),
    "compose_v4.data.editing_v2_process_v2_active8_mapreduce": (
        "completed_process_v2_active8_task_ids",
        "execute_process_v2_active8_decision_task",
        "plan_process_v2_active8_decisions",
        "reduce_process_v2_active8_decisions",
        "write_process_v2_active8_decision_plan",
    ),
    "compose_v4.data.editing_v2_process_v2_active8_index": (
        "resolve_process_v2_active8_decision_index",
    ),
}
ACTIVE8_ENTRY_MODULES: tuple[str, ...] = tuple(sorted(ACTIVE8_STAGE_SYMBOLS))

#: The input-binding code this launcher owns outright: it exists now, so its
#: narrow revision is buildable and testable today.
INPUT_BINDING_ENTRY_MODULES: tuple[str, ...] = (
    "compose_v4.data.editing_process_v2_admitted_source",
    "compose_v4.data.editing_v2_process_v2_chunk_cache",
)

RUN_REQUEST_SCHEMA = "compose.editing_v2.process_v2.active8_run_request"
RUN_REQUEST_SCHEMA_VERSION = 1
RUN_REQUEST_STATUS = "PROCESS_V2_ACTIVE8_RUN_REQUEST_NO_DOWNSTREAM_AUTHORITY"
IMAGE_REVISION_SCHEMA = "compose.editing_v2.process_v2.active8_modal_image_revision"
IMAGE_REVISION_SCHEMA_VERSION = 1

MAP_CPU = 4.0
MAP_MEMORY_MB = 16384
MAP_TIMEOUT_SECONDS = 8 * 3600
REDUCE_TIMEOUT_SECONDS = 8 * 3600
DRIVER_TIMEOUT_SECONDS = 24 * 3600
#: The whole fan-out, and the default.  Forty is the platform ceiling for
#: concurrent writers to this volume; it is refused, never clamped, above that.
DEFAULT_MAX_MAP_CONTAINERS = PLATFORM_MAX_MAP_CONTAINERS

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
image = image.add_local_file(
    ROOT / LAUNCHER_SOURCE, str(REMOTE_ROOT / LAUNCHER_SOURCE), copy=True
)

app = modal.App("compose-v4-process-v2-active8-decisions")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


# ---- Deterministic identity ----


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE]
    for source_directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / source_directory).rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("the Active8 serialized source inventory repeats a path")
    return tuple(paths)


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments), cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot establish Process-V2 Active8 Git identity: git {' '.join(arguments)}"
        ) from error


def local_image_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Require the exact clean commit and hash every serialized image input."""

    if not isinstance(expected_commit, str) or _COMMIT_RE.fullmatch(expected_commit) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "Process-V2 Active8 requires the exact clean committed worktree; a dirty tree "
            "cannot be bound to a scientific artifact"
        )
    sources = {
        relative: _file_sha256(root / relative) for relative in _serialized_source_paths(root)
    }
    require_tracked(sources, tracked=_git(root, "ls-files").splitlines())
    body = {
        "schema": IMAGE_REVISION_SCHEMA,
        "schema_version": IMAGE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    return {**body, "image_revision_sha256": canonical_sha256(body)}


def validate_remote_image_revision(
    value: Mapping[str, Any], *, remote_root: Path = REMOTE_ROOT
) -> dict[str, Any]:
    """Re-derive the image identity from the files in hand.  No Git is called."""

    body = {key: item for key, item in value.items() if key != "image_revision_sha256"}
    if (
        value.get("schema") != IMAGE_REVISION_SCHEMA
        or value.get("schema_version") != IMAGE_REVISION_SCHEMA_VERSION
        or value.get("worktree_clean") is not True
        or value.get("image_revision_sha256") != canonical_sha256(body)
    ):
        raise RuntimeError("the Process-V2 Active8 image revision disagrees")
    sources = value.get("serialized_sources")
    expected = _serialized_source_paths(Path(remote_root))
    if not isinstance(sources, dict) or set(sources) != set(expected):
        raise RuntimeError("the Process-V2 Active8 serialized source inventory disagrees")
    for relative, digest in sources.items():
        if _file_sha256(Path(remote_root) / relative) != digest:
            raise RuntimeError(f"a serialized Active8 source differs in this container: {relative}")
    return dict(value)


def authority_envelope() -> dict[str, bool]:
    """Every authority field false.  Deciding a corpus authorizes nothing."""

    return {"training_launched": False, **authority_false_block()}


# ---- Artifact paths ----


def artifact_path(value: str, *, field: str, artifact_root: Path = ARTIFACT_ROOT) -> Path:
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
    root = Path(artifact_root).resolve()
    resolved = (root / Path(*pure.parts[2:])).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"{field} resolves outside the artifact root")
    return resolved


def artifact_address(path: Path, *, artifact_root: Path = ARTIFACT_ROOT) -> str:
    relative = Path(path).resolve().relative_to(Path(artifact_root).resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def resolve_active8_stage(*, remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    """Import the Active8 stage at execution, or refuse with the exact missing names."""

    if str(remote_root / "src") not in sys.path:
        sys.path.insert(0, str(remote_root / "src"))
    resolved: dict[str, Any] = {}
    missing_modules: list[str] = []
    missing_symbols: list[str] = []
    for module_name, symbols in ACTIVE8_STAGE_SYMBOLS.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing_modules.append(module_name)
            continue
        for symbol in symbols:
            value = getattr(module, symbol, None)
            if value is None:
                missing_symbols.append(f"{module_name}.{symbol}")
            else:
                resolved[symbol] = value
    if missing_modules or missing_symbols:
        raise RuntimeError(
            "the Process-V2 Active8 stage is not present in this tree, so there is nothing "
            f"to run; missing modules={sorted(missing_modules)} "
            f"missing symbols={sorted(missing_symbols)}"
        )
    return resolved


def build_run_request(
    *,
    image_revision: Mapping[str, Any],
    input_binding_revision: Mapping[str, Any],
    active8_revision: Mapping[str, Any],
    inputs: Mapping[str, str],
    output_artifact_prefix: str,
) -> dict[str, Any]:
    """Content-address one run from its inputs and every revision it binds.

    The fan-out is deliberately absent from this body.  Worker count is a
    schedule, and letting it into the identity would address a different run for
    the same data.
    """

    for name, value in inputs.items():
        artifact_path(value, field=name)
    artifact_path(f"{output_artifact_prefix}/placeholder", field="output_artifact_prefix")
    body: dict[str, Any] = {
        "schema": RUN_REQUEST_SCHEMA,
        "schema_version": RUN_REQUEST_SCHEMA_VERSION,
        "status": RUN_REQUEST_STATUS,
        **authority_false_block(),
        "inputs": dict(sorted(inputs.items())),
        "image_revision_sha256": str(image_revision["image_revision_sha256"]),
        "input_binding_implementation_sha256": str(
            input_binding_revision["implementation_sha256"]
        ),
        "active8_implementation_sha256": str(active8_revision["implementation_sha256"]),
        "output_artifact_prefix": output_artifact_prefix,
        "python_runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }
    request = {**body, "run_identity_sha256": canonical_sha256(body)}
    require_no_granted_authority(request, label="Process-V2 Active8 run request")
    return request


def launch_command(
    *,
    entrypoint: str,
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    expected_commit: str,
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
    max_map_containers: int = DEFAULT_MAX_MAP_CONTAINERS,
) -> str:
    """The exact command a launch would use.  Printing it executes nothing."""

    return " ".join(
        (
            "modal run --detach",
            f"{LAUNCHER_SOURCE}::{entrypoint}",
            f"--cache-run-artifact-root {cache_run_artifact_root}",
            f"--rebind-run-artifact-root {rebind_run_artifact_root}",
            f"--expected-commit {expected_commit}",
            f"--output-artifact-prefix {output_artifact_prefix}",
            f"--max-map-containers {max_map_containers}",
        )
    )


def require_committed_inputs(
    *,
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    artifact_root: Path,
    remote_root: Path,
) -> dict[str, Any]:
    """Require both upstream generations to be COMMITTED, never merely present."""

    from compose_v4.data.editing_process_v2_admitted_source import (
        resolve_process_v2_admitted_source,
    )
    from compose_v4.data.editing_process_v2_rebind import PLAN_FILENAME
    from compose_v4.data.editing_v2_process_v2_chunk_cache import (
        load_committed_process_v2_chunk_cache_completion,
    )

    cache_completion = load_committed_process_v2_chunk_cache_completion(
        cache_run_artifact_root, artifact_root=artifact_root, repo_root=remote_root
    )
    require_no_granted_authority(cache_completion, label="the chunk-cache completion")
    rebind_root = artifact_path(
        rebind_run_artifact_root, field="rebind_run_artifact_root", artifact_root=artifact_root
    )
    plan_path = rebind_root / PLAN_FILENAME
    if not plan_path.is_file():
        raise RuntimeError(
            f"the Process-V2 rebind generation is not committed; no {PLAN_FILENAME} at "
            f"{rebind_run_artifact_root}"
        )
    plan = json.loads(plan_path.read_bytes())
    admitted = resolve_process_v2_admitted_source(
        plan, artifact_root=artifact_root, repo_root=remote_root
    )
    identity = dict(admitted.identity())
    require_no_granted_authority(identity, label="the admitted-source identity")
    return {
        "admitted_source": admitted,
        "cache_completion": cache_completion,
        "admitted_source_identity": identity,
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
def decide_one_chunk(
    plan: dict[str, Any],
    task_identity_sha256: str,
    image_revision: dict[str, Any],
    input_binding_revision: dict[str, Any],
    active8_revision: dict[str, Any],
) -> dict[str, Any]:
    """One independent chunk task: decide it, or publish nothing at all."""

    validate_remote_image_revision(image_revision)
    validate_implementation_revision(input_binding_revision, repo_root=REMOTE_ROOT)
    validate_implementation_revision(active8_revision, repo_root=REMOTE_ROOT)
    stage = resolve_active8_stage()
    # This worker reads one cache chunk and a plan another container wrote.
    artifact_volume.reload()
    receipt = stage["execute_process_v2_active8_decision_task"](
        plan,
        task_identity_sha256,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    artifact_volume.commit()
    return {
        "task_identity_sha256": task_identity_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
    }


@app.function(
    image=image,
    cpu=MAP_CPU,
    memory=MAP_MEMORY_MB,
    timeout=REDUCE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def reduce_decisions(
    plan: dict[str, Any],
    image_revision: dict[str, Any],
    input_binding_revision: dict[str, Any],
    active8_revision: dict[str, Any],
) -> dict[str, Any]:
    """Only the serialized reducer may declare the generation complete."""

    validate_remote_image_revision(image_revision)
    validate_implementation_revision(input_binding_revision, repo_root=REMOTE_ROOT)
    validate_implementation_revision(active8_revision, repo_root=REMOTE_ROOT)
    stage = resolve_active8_stage()
    # Reduction reads every task receipt, each written by another container.
    artifact_volume.reload()
    completion = stage["reduce_process_v2_active8_decisions"](
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    require_no_granted_authority(completion, label="the Active8 completion")
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
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    output_artifact_prefix: str,
    max_map_containers: int,
    image_revision: dict[str, Any],
    input_binding_revision: dict[str, Any],
    active8_revision: dict[str, Any],
) -> dict[str, Any]:
    """Bind, plan, resume the incomplete tasks in bounded waves, then reduce."""

    validate_remote_image_revision(image_revision)
    validate_implementation_revision(input_binding_revision, repo_root=REMOTE_ROOT)
    validate_implementation_revision(active8_revision, repo_root=REMOTE_ROOT)
    stage = resolve_active8_stage()
    bound = validate_map_container_bound(max_map_containers)

    # Before reading the committed cache and the committed rebind generation.
    artifact_volume.reload()
    upstream = require_committed_inputs(
        cache_run_artifact_root=cache_run_artifact_root,
        rebind_run_artifact_root=rebind_run_artifact_root,
        artifact_root=ARTIFACT_ROOT,
        remote_root=REMOTE_ROOT,
    )
    request = build_run_request(
        image_revision=image_revision,
        input_binding_revision=input_binding_revision,
        active8_revision=active8_revision,
        inputs={
            "cache_run_artifact_root": cache_run_artifact_root,
            "rebind_run_artifact_root": rebind_run_artifact_root,
        },
        output_artifact_prefix=output_artifact_prefix,
    )
    run_artifact_root = f"{output_artifact_prefix}/{request['run_identity_sha256']}"
    policy = stage["build_process_v2_active8_admission_policy"]()
    plan = stage["plan_process_v2_active8_decisions"](
        upstream["admitted_source"],
        cache_run_artifact_root=cache_run_artifact_root,
        policy=policy,
        run_request=request,
        run_artifact_root=run_artifact_root,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    stage["write_process_v2_active8_decision_plan"](
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    artifact_volume.commit()

    # Before scanning reusable receipts: a previous run's results live in a
    # volume snapshot this container did not necessarily boot with.
    artifact_volume.reload()
    complete = stage["completed_process_v2_active8_task_ids"](
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
                "phase": "process_v2_active8_map_plan",
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
            submit=lambda wave: decide_one_chunk.starmap(
                [
                    (
                        plan,
                        task_id,
                        image_revision,
                        input_binding_revision,
                        active8_revision,
                    )
                    for task_id in wave
                ]
            ),
            set_autoscaler=decide_one_chunk.update_autoscaler,
        )

    completion = reduce_decisions.remote(
        plan, image_revision, input_binding_revision, active8_revision
    )
    # After the reducer returns and before resolving the published generation:
    # the completion document was written by that other container.
    artifact_volume.reload()
    index = stage["resolve_process_v2_active8_decision_index"](
        run_artifact_root, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    report = {
        "phase": "process_v2_active8_complete",
        "executed": True,
        "run_identity_sha256": request["run_identity_sha256"],
        "run_artifact_root": run_artifact_root,
        "completion_sha256": completion["completion_sha256"],
        "decision_index_completion_sha256": index.completion_sha256,
        "process_identity_sha256": index.process_identity_sha256,
        "counts": dict(index.counts()),
        "rejected_traces_by_code": dict(index.rejected_traces_by_code()),
        "cache_run_artifact_root": cache_run_artifact_root,
        "rebind_run_artifact_root": rebind_run_artifact_root,
        "admitted_source_identity": upstream["admitted_source_identity"],
        "image_revision_sha256": image_revision["image_revision_sha256"],
        "input_binding_implementation_sha256": input_binding_revision["implementation_sha256"],
        "active8_implementation_sha256": active8_revision["implementation_sha256"],
        "max_map_containers": bound,
        "submission_waves": len(waves),
        **authority_envelope(),
    }
    require_no_granted_authority(report, label="the Active8 run report")
    return report


# ---- Local entry points ----


@app.local_entrypoint()
def describe(
    cache_run_artifact_root: str = "/artifacts/editing_v2/process_v2_chunk_cache/<generation>",
    rebind_run_artifact_root: str = "/artifacts/editing_v2/process_v2_rebind/<run>",
    expected_commit: str = "<full-40-hex-commit>",
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
    max_map_containers: int = DEFAULT_MAX_MAP_CONTAINERS,
) -> None:
    """Print the exact command and artifact roots a launch WOULD use.

    Executes no remote function, resolves no artifact, and calls no Git.
    ``executed`` is false because nothing ran, and the wording is conditional so
    this output can never be quoted as evidence that Active8 has been computed.
    """

    bound = validate_map_container_bound(int(max_map_containers))
    print(
        json.dumps(
            {
                "phase": "process_v2_active8_launch_description",
                "executed": False,
                "status": "DESCRIPTION_ONLY_NOTHING_WAS_EXECUTED",
                "would_run_command": launch_command(
                    entrypoint="main",
                    cache_run_artifact_root=cache_run_artifact_root,
                    rebind_run_artifact_root=rebind_run_artifact_root,
                    expected_commit=expected_commit,
                    output_artifact_prefix=output_artifact_prefix,
                    max_map_containers=bound,
                ),
                "would_read_artifact_roots": {
                    "chunk_cache": cache_run_artifact_root,
                    "rebind": rebind_run_artifact_root,
                },
                "would_write_artifact_root": (
                    f"{output_artifact_prefix}/<run_identity_sha256>"
                ),
                "would_bind_stage_modules": {
                    module: list(symbols)
                    for module, symbols in sorted(ACTIVE8_STAGE_SYMBOLS.items())
                },
                "max_map_containers": bound,
                "platform_max_map_containers": PLATFORM_MAX_MAP_CONTAINERS,
                "execution_shape": "bounded_map_then_serialized_reduce",
                **authority_envelope(),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def main(
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    expected_commit: str,
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
    max_map_containers: int = DEFAULT_MAX_MAP_CONTAINERS,
) -> None:
    """Bind the exact clean commit and both revisions, then drive the stage."""

    for field, value in (
        ("cache_run_artifact_root", cache_run_artifact_root),
        ("rebind_run_artifact_root", rebind_run_artifact_root),
    ):
        artifact_path(value, field=field)
    bound = validate_map_container_bound(int(max_map_containers))
    image_revision = local_image_revision(expected_commit=expected_commit)
    input_binding_revision = build_implementation_revision(
        INPUT_BINDING_ENTRY_MODULES, repo_root=ROOT
    )
    active8_revision = build_implementation_revision(ACTIVE8_ENTRY_MODULES, repo_root=ROOT)
    report = driver.remote(
        cache_run_artifact_root=cache_run_artifact_root,
        rebind_run_artifact_root=rebind_run_artifact_root,
        output_artifact_prefix=output_artifact_prefix,
        max_map_containers=bound,
        image_revision=image_revision,
        input_binding_revision=input_binding_revision,
        active8_revision=active8_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


__all__ = [
    "ACTIVE8_ENTRY_MODULES",
    "ACTIVE8_STAGE_SYMBOLS",
    "AUTHORITY_FIELDS",
    "DEFAULT_MAX_MAP_CONTAINERS",
    "INPUT_BINDING_ENTRY_MODULES",
    "OUTPUT_ARTIFACT_PREFIX",
    "PLATFORM_MAX_MAP_CONTAINERS",
    "app",
    "artifact_address",
    "artifact_path",
    "authority_envelope",
    "build_run_request",
    "decide_one_chunk",
    "describe",
    "driver",
    "launch_command",
    "local_image_revision",
    "main",
    "reduce_decisions",
    "require_committed_inputs",
    "resolve_active8_stage",
    "validate_remote_image_revision",
]
