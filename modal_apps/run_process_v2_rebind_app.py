"""Prove a completed V1 semantic migration against the Process-V2 authority.

The V1 migration payload is consumed as **immutable chemical data**.  Nothing is
relabelled: every record keeps its V1 identity, and the V2 admission decision is
recorded alongside it.  The driver freezes a content-addressed plan, resumes only
the chunk tasks that are not already durably present, maps them across
independent CPU containers, and hands off to the strict reducer.

**The corpus is a committed chunk cache, and its root is the only source input.**
One proof task reads exactly one verified chunk; no task opens a packed shard.
The V1 payload root is *derived* from the pinned production completion path and
the exact completion is bound, so this launcher no longer takes a payload root at
all.  It used to take one as a bare positional with no default, no declared
expectation and no equality check, which flowed unchecked into the plan -- the
chunk-cache launcher beside it had pinned its own path all along.

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
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
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
SMOKE_OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_rebind_smoke"

# Prospectively frozen before the production cache or any Process-V2 rebind
# result exists. This is a full, non-tail 2,048-row chunk from the largest
# training source. The address is corpus identity, not a scheduling choice.
SMOKE_V1_TASK_IDENTITY_SHA256 = (
    "af176e806b3a9d053cf6797140fc28c3cfd0d5ab152761b70c72fe3853c6eae5"
)
SMOKE_DATA_LANE = "reversible_synthetic_walk"
SMOKE_SPLIT = "train"
SMOKE_CHUNK_INDEX = 64
SMOKE_ENTRY_START = 131072
SMOKE_ENTRY_STOP = 133120
SMOKE_EXPECTED_ROWS = SMOKE_ENTRY_STOP - SMOKE_ENTRY_START

# Operational stop thresholds only. They grant no scientific or training
# authority and are deliberately stricter than the worker allocation/timeout.
SMOKE_MAX_PEAK_RSS_MIB = 4096.0
SMOKE_MAX_WALL_SECONDS = 4 * 3600.0
SMOKE_SCHEMA = "compose.data.process_v2_rebind_operational_smoke"
SMOKE_SCHEMA_VERSION = 1
SMOKE_STATUS = "MEASURED_OPERATIONAL_SMOKE_NO_DOWNSTREAM_AUTHORITY"
SMOKE_FILENAME = "SMOKE_RESULT.json"
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


def _smoke_authority_envelope() -> dict[str, bool]:
    """The complete authority vocabulary, explicitly false for a smoke."""

    return {
        "training_launched": False,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
    }


def _select_frozen_smoke_task(plan: dict[str, Any]) -> dict[str, Any]:
    """Resolve exactly the preregistered full chunk, or refuse the smoke."""

    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("the Process-V2 smoke plan has no ordered task list")
    matches = [
        task
        for task in tasks
        if isinstance(task, dict)
        and task.get("v1_task_identity_sha256") == SMOKE_V1_TASK_IDENTITY_SHA256
        and task.get("chunk_index") == SMOKE_CHUNK_INDEX
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "the Process-V2 smoke address does not resolve to exactly one cache chunk"
        )
    task = dict(matches[0])
    expected = {
        "data_lane": SMOKE_DATA_LANE,
        "split": SMOKE_SPLIT,
        "entry_start": SMOKE_ENTRY_START,
        "entry_stop": SMOKE_ENTRY_STOP,
        "chunk_row_count": SMOKE_EXPECTED_ROWS,
    }
    disagreements = {
        field: {"expected": value, "observed": task.get(field)}
        for field, value in expected.items()
        if task.get(field) != value
    }
    if disagreements:
        raise RuntimeError(
            "the prospectively frozen Process-V2 smoke chunk moved: "
            f"{disagreements}"
        )
    return task


def _smoke_artifact_path(plan: dict[str, Any], task: dict[str, Any]) -> Path:
    """Return the diagnostic path outside the normal production run root."""

    prefix = PurePosixPath(SMOKE_OUTPUT_ARTIFACT_PREFIX)
    relative = prefix.relative_to("/artifacts")
    return (
        ARTIFACT_ROOT
        / Path(relative)
        / str(plan["run_identity_sha256"])
        / str(task["task_identity_sha256"])
        / SMOKE_FILENAME
    )


def _validate_smoke_record(
    value: object,
    *,
    plan: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    """Validate the non-authorizing operational result after reopening it."""

    if not isinstance(value, dict):
        raise RuntimeError("the Process-V2 smoke result must be an object")
    record = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *_smoke_authority_envelope(),
        "timestamp_utc",
        "code_commit",
        "code_tree",
        "image_revision_sha256",
        "source_revision_sha256",
        "cache_run_artifact_root",
        "plan_sha256",
        "run_identity_sha256",
        "run_artifact_root",
        "task_identity_sha256",
        "task_address",
        "receipt_sha256",
        "counts",
        "resource_request",
        "thresholds",
        "measurements",
        "environment",
        "smoke_result",
        "diagnostic_sha256",
    }
    if set(record) != expected_fields:
        raise RuntimeError("the Process-V2 smoke result fields disagree")
    body = {key: item for key, item in record.items() if key != "diagnostic_sha256"}
    if (
        record.get("schema") != SMOKE_SCHEMA
        or record.get("schema_version") != SMOKE_SCHEMA_VERSION
        or record.get("status") != SMOKE_STATUS
        or record.get("diagnostic_sha256") != _sha256(body)
        or any(record.get(key) is not False for key in _smoke_authority_envelope())
        or record.get("plan_sha256") != plan.get("plan_sha256")
        or record.get("run_identity_sha256") != plan.get("run_identity_sha256")
        or record.get("run_artifact_root") != plan.get("run_artifact_root")
        or record.get("task_identity_sha256") != task.get("task_identity_sha256")
        or record.get("source_revision_sha256")
        != plan.get("source_revision", {}).get("source_revision_sha256")
        or record.get("cache_run_artifact_root")
        != plan.get("cache_binding", {}).get("cache_run_artifact_root")
    ):
        raise RuntimeError("the Process-V2 smoke result is stale or self-inconsistent")
    for field in (
        "image_revision_sha256",
        "source_revision_sha256",
        "plan_sha256",
        "run_identity_sha256",
        "task_identity_sha256",
        "receipt_sha256",
        "diagnostic_sha256",
    ):
        if not isinstance(record.get(field), str) or _SHA256_RE.fullmatch(
            str(record[field])
        ) is None:
            raise RuntimeError(f"the Process-V2 smoke {field} is not a SHA-256")
    if (
        not isinstance(record.get("code_commit"), str)
        or _COMMIT_RE.fullmatch(str(record["code_commit"])) is None
        or not isinstance(record.get("code_tree"), str)
        or _COMMIT_RE.fullmatch(str(record["code_tree"])) is None
    ):
        raise RuntimeError("the Process-V2 smoke code revision is malformed")
    try:
        timestamp = datetime.fromisoformat(str(record["timestamp_utc"]))
    except ValueError as error:
        raise RuntimeError("the Process-V2 smoke timestamp is malformed") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise RuntimeError("the Process-V2 smoke timestamp is not UTC")
    if not str(record.get("cache_run_artifact_root", "")).startswith(
        "/artifacts/editing_v2/process_v2_chunk_cache/"
    ):
        raise RuntimeError("the Process-V2 smoke does not name a production cache root")
    address = record.get("task_address")
    if address != {
        "v1_task_identity_sha256": SMOKE_V1_TASK_IDENTITY_SHA256,
        "data_lane": SMOKE_DATA_LANE,
        "split": SMOKE_SPLIT,
        "chunk_index": SMOKE_CHUNK_INDEX,
        "entry_start": SMOKE_ENTRY_START,
        "entry_stop": SMOKE_ENTRY_STOP,
        "row_count": SMOKE_EXPECTED_ROWS,
    }:
        raise RuntimeError("the Process-V2 smoke result names another chunk")
    measurements = record.get("measurements")
    if not isinstance(measurements, dict) or set(measurements) != {
        "wall_seconds",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "peak_rss_mib",
    }:
        raise RuntimeError("the Process-V2 smoke measurements are malformed")
    if any(
        not isinstance(measurements[field], (int, float))
        or isinstance(measurements[field], bool)
        or not math.isfinite(float(measurements[field]))
        or float(measurements[field]) < 0.0
        for field in measurements
    ):
        raise RuntimeError("the Process-V2 smoke measurements must be finite and nonnegative")
    if record.get("thresholds") != {
        "max_wall_seconds": SMOKE_MAX_WALL_SECONDS,
        "max_peak_rss_mib": SMOKE_MAX_PEAK_RSS_MIB,
    }:
        raise RuntimeError("the Process-V2 smoke thresholds differ from the frozen values")
    if record.get("resource_request") != {
        "cpu": MAP_CPU,
        "memory_mib": MAP_MEMORY_MB,
        "timeout_seconds": MAP_TIMEOUT_SECONDS,
        "worker_count": 1,
    }:
        raise RuntimeError("the Process-V2 smoke resource request disagrees")
    counts = record.get("counts")
    if (
        not isinstance(counts, dict)
        or type(counts.get("source_entries")) is not int
        or type(counts.get("admitted_entries")) is not int
        or type(counts.get("rejected_entries")) is not int
        or counts["source_entries"] != SMOKE_EXPECTED_ROWS
        or counts["admitted_entries"] + counts["rejected_entries"]
        != counts["source_entries"]
    ):
        raise RuntimeError("the Process-V2 smoke task census does not reconcile")
    environment = record.get("environment")
    if not isinstance(environment, dict) or set(environment) != {
        "python",
        "platform",
        "processor",
        "logical_cpu_count",
        "torch",
        "numpy",
        "rdkit",
        "modal",
        "precision",
    }:
        raise RuntimeError("the Process-V2 smoke environment is incomplete")
    expected_result = (
        "PASS"
        if float(measurements["wall_seconds"]) <= SMOKE_MAX_WALL_SECONDS
        and float(measurements["peak_rss_mib"]) <= SMOKE_MAX_PEAK_RSS_MIB
        else "FAIL"
    )
    if record.get("smoke_result") != expected_result:
        raise RuntimeError("the Process-V2 smoke decision disagrees with its frozen thresholds")
    return record


def _publish_smoke_record(
    record: dict[str, Any],
    *,
    output: Path,
    plan: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish and physically reopen one diagnostic result."""

    if output.exists():
        return _validate_smoke_record(
            json.loads(output.read_text(encoding="utf-8")), plan=plan, task=task
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=".smoke.", suffix=".staging"))
    staged_file = staging / SMOKE_FILENAME
    try:
        staged_file.write_bytes(_canonical_bytes(record) + b"\n")
        _validate_smoke_record(
            json.loads(staged_file.read_text(encoding="utf-8")), plan=plan, task=task
        )
        os.replace(staged_file, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return _validate_smoke_record(
        json.loads(output.read_text(encoding="utf-8")), plan=plan, task=task
    )


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
    """One independent chunk task: prove it, or publish nothing."""

    from compose_v4.data.editing_process_v2_rebind import (
        execute_process_v2_rebind_task,
        require_production_source_geometry,
    )

    _validate_remote_image_revision(image_revision)
    validate_remote_source_revision(source_revision, image_revision)
    require_production_source_geometry(plan, label="the Process-V2 rebind plan")
    # This worker reads one cache chunk and the plan another container wrote.
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
    timeout=MAP_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def smoke_one_chunk_driver(
    *,
    cache_run_artifact_root: str,
    output_artifact_prefix: str,
    image_revision: dict[str, Any],
    source_revision: dict[str, Any],
) -> dict[str, Any]:
    """Execute one frozen production task and publish operational evidence.

    The normal task is written into the normal content-addressed rebind run, so
    a later full driver reuses it. The separate smoke record carries timing and
    memory only, skips reduction, and grants no downstream authority.
    """

    if str(REMOTE_ROOT) not in sys.path:
        sys.path.insert(0, str(REMOTE_ROOT))
    if str(REMOTE_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REMOTE_ROOT / "scripts"))

    from compose_v4.data.editing_process_v2_rebind import (
        completed_process_v2_rebind_task_ids,
        execute_process_v2_rebind_task,
        require_production_source_geometry,
        write_process_v2_rebind_plan,
    )
    from plan_process_v2_rebind import build_cache_fed_plan

    _validate_remote_image_revision(image_revision)
    revision = validate_remote_source_revision(source_revision, image_revision)
    if not sys.platform.startswith("linux"):
        raise RuntimeError("the Process-V2 RSS smoke is defined on the Linux Modal image")
    artifact_volume.reload()
    plan = build_cache_fed_plan(
        artifact_root=ARTIFACT_ROOT,
        cache_run_artifact_root=cache_run_artifact_root,
        output_artifact_prefix=output_artifact_prefix,
        source_revision=revision,
        repo_root=REMOTE_ROOT,
    )
    require_production_source_geometry(plan, label="the Process-V2 smoke plan")
    write_process_v2_rebind_plan(plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT)
    artifact_volume.commit()

    task = _select_frozen_smoke_task(plan)
    task_identity = str(task["task_identity_sha256"])
    diagnostic_path = _smoke_artifact_path(plan, task)
    artifact_volume.reload()
    completed_before = completed_process_v2_rebind_task_ids(
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    if diagnostic_path.exists():
        if task_identity not in completed_before:
            raise RuntimeError(
                "the Process-V2 smoke diagnostic exists without its production task"
            )
        existing = _validate_smoke_record(
            json.loads(diagnostic_path.read_text(encoding="utf-8")),
            plan=plan,
            task=task,
        )
        reopened = execute_process_v2_rebind_task(
            plan,
            task_identity,
            artifact_root=ARTIFACT_ROOT,
            repo_root=REMOTE_ROOT,
        )
        if (
            reopened.get("reused") is not True
            or reopened.get("receipt_sha256") != existing["receipt_sha256"]
            or reopened.get("counts") != existing["counts"]
        ):
            raise RuntimeError(
                "the Process-V2 smoke diagnostic disagrees with its production task"
            )
        return {**existing, "diagnostic_artifact_path": str(diagnostic_path)}
    if task_identity in completed_before:
        raise RuntimeError(
            "the frozen Process-V2 smoke task is already complete without a smoke "
            "diagnostic; refusing to mismeasure a reused task"
        )

    before = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.perf_counter()
    result = execute_process_v2_rebind_task(
        plan,
        task_identity,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    wall_seconds = time.perf_counter() - wall_start
    after = resource.getrusage(resource.RUSAGE_SELF)
    if result.get("reused") is not False:
        raise RuntimeError("the Process-V2 smoke did not execute a fresh production task")
    completed_after = completed_process_v2_rebind_task_ids(
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    if task_identity not in completed_after:
        raise RuntimeError("the Process-V2 smoke task does not reopen through production")

    # Linux reports ru_maxrss in KiB. This function refuses any other platform
    # above, so the unit conversion is explicit rather than heuristic.
    measurements = {
        "wall_seconds": float(wall_seconds),
        "user_cpu_seconds": float(after.ru_utime - before.ru_utime),
        "system_cpu_seconds": float(after.ru_stime - before.ru_stime),
        "peak_rss_mib": float(after.ru_maxrss) / 1024.0,
    }
    smoke_result = (
        "PASS"
        if measurements["wall_seconds"] <= SMOKE_MAX_WALL_SECONDS
        and measurements["peak_rss_mib"] <= SMOKE_MAX_PEAK_RSS_MIB
        else "FAIL"
    )
    import numpy
    import rdkit
    import torch

    body: dict[str, Any] = {
        "schema": SMOKE_SCHEMA,
        "schema_version": SMOKE_SCHEMA_VERSION,
        "status": SMOKE_STATUS,
        **_smoke_authority_envelope(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": image_revision["commit"],
        "code_tree": image_revision["tree"],
        "image_revision_sha256": image_revision["image_revision_sha256"],
        "source_revision_sha256": revision["source_revision_sha256"],
        "cache_run_artifact_root": cache_run_artifact_root,
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "run_artifact_root": plan["run_artifact_root"],
        "task_identity_sha256": task_identity,
        "task_address": {
            "v1_task_identity_sha256": SMOKE_V1_TASK_IDENTITY_SHA256,
            "data_lane": SMOKE_DATA_LANE,
            "split": SMOKE_SPLIT,
            "chunk_index": SMOKE_CHUNK_INDEX,
            "entry_start": SMOKE_ENTRY_START,
            "entry_stop": SMOKE_ENTRY_STOP,
            "row_count": SMOKE_EXPECTED_ROWS,
        },
        "receipt_sha256": result["receipt_sha256"],
        "counts": result["counts"],
        "resource_request": {
            "cpu": MAP_CPU,
            "memory_mib": MAP_MEMORY_MB,
            "timeout_seconds": MAP_TIMEOUT_SECONDS,
            "worker_count": 1,
        },
        "thresholds": {
            "max_wall_seconds": SMOKE_MAX_WALL_SECONDS,
            "max_peak_rss_mib": SMOKE_MAX_PEAK_RSS_MIB,
        },
        "measurements": measurements,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "rdkit": rdkit.__version__,
            "modal": getattr(modal, "__version__", "unknown"),
            "precision": "not_applicable_cpu_rebind",
        },
        "smoke_result": smoke_result,
    }
    record = {**body, "diagnostic_sha256": _sha256(body)}
    published = _publish_smoke_record(
        record, output=diagnostic_path, plan=plan, task=task
    )
    artifact_volume.commit()
    return {**published, "diagnostic_artifact_path": str(diagnostic_path)}


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

    from compose_v4.data.editing_process_v2_rebind import (
        reduce_process_v2_rebind,
        require_production_source_geometry,
    )

    _validate_remote_image_revision(image_revision)
    validate_remote_source_revision(source_revision, image_revision)
    require_production_source_geometry(plan, label="the Process-V2 rebind plan")
    # Reduction reads every chunk result, each written by another container.
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
    cache_run_artifact_root: str,
    output_artifact_prefix: str,
    max_map_containers: int,
    image_revision: dict[str, Any],
    source_revision: dict[str, Any],
) -> dict[str, Any]:
    """Plan, publish, resume, map the missing chunks in waves, then reduce.

    No Git call appears in this body or in anything it calls: the plan driver
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
        require_production_source_geometry,
        write_process_v2_rebind_plan,
    )
    from plan_process_v2_rebind import build_cache_fed_plan, plan_envelope

    _validate_remote_image_revision(image_revision)
    revision = validate_remote_source_revision(source_revision, image_revision)
    bound = validate_map_container_bound(max_map_containers)
    # Before reading the committed cache and the pinned payload.
    artifact_volume.reload()

    # The plan driver is imported, not reimplemented: the pinned-identity
    # discovery, the exact-completion binding and the refusal to plan against an
    # unexpected historical process must be identical locally and here.
    plan = build_cache_fed_plan(
        artifact_root=ARTIFACT_ROOT,
        cache_run_artifact_root=cache_run_artifact_root,
        output_artifact_prefix=output_artifact_prefix,
        source_revision=revision,
        repo_root=REMOTE_ROOT,
    )
    require_production_source_geometry(plan, label="the Process-V2 rebind plan")
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
        "source_geometry": completion["source_geometry"],
        "cache_run_artifact_root": cache_run_artifact_root,
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
    cache_run_artifact_root: str,
    expected_commit: str,
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
    max_map_containers: int = DEFAULT_MAX_MAP_CONTAINERS,
) -> None:
    """The production launcher: a committed cache root, and nothing else.

    There is deliberately no ``v1_payload_root`` and no ``entries_per_task``.
    The payload root is derived from the pinned production completion, and the
    task size is the cache's own chunk size, because the chunk boundary is the
    task boundary. Both used to be caller-supplied and unchecked.
    """

    from compose_v4.data.editing_process_v2_rebind import (
        repository_process_v2_rebind_source_revision,
    )

    bound = validate_map_container_bound(int(max_map_containers))
    image_revision = local_image_revision(expected_commit=expected_commit)
    # Computed here, where `.git` exists. The remote side revalidates it against
    # the image and never runs Git.
    source_revision = repository_process_v2_rebind_source_revision(repo_root=ROOT)
    if source_revision["commit"] != image_revision["commit"]:
        raise RuntimeError("the local source and image revisions name different commits")
    report = driver.remote(
        cache_run_artifact_root=cache_run_artifact_root,
        output_artifact_prefix=output_artifact_prefix,
        max_map_containers=bound,
        image_revision=image_revision,
        source_revision=source_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


@app.local_entrypoint()
def smoke_one_chunk(
    cache_run_artifact_root: str,
    expected_commit: str,
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
) -> None:
    """Measure one frozen full chunk without reducing or granting authority."""

    from compose_v4.data.editing_process_v2_rebind import (
        repository_process_v2_rebind_source_revision,
    )

    image_revision = local_image_revision(expected_commit=expected_commit)
    source_revision = repository_process_v2_rebind_source_revision(repo_root=ROOT)
    if source_revision["commit"] != image_revision["commit"]:
        raise RuntimeError("the local smoke source and image revisions disagree")
    report = smoke_one_chunk_driver.remote(
        cache_run_artifact_root=cache_run_artifact_root,
        output_artifact_prefix=output_artifact_prefix,
        image_revision=image_revision,
        source_revision=source_revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
