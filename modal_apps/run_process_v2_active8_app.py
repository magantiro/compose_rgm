"""Run the Process-V2 Active8 map and release sentinel on Modal.

This is orchestration only.  The production library owns planning, teacher
support, immutable publication, reduction and the sentinel.  A spawned remote
driver survives client disconnection and handles one deterministic phase. The
default operational pilot runs three independently measured one-CPU canaries.
The explicit full map retains at most eighty independently replenished Volume-v1
writer containers, extending the proven Process-V2 rebind publication pattern,
with one restart-safe task per container. A final reduction invocation reuses every
validated task, runs the parallel sentinel, and publishes completion.

Importing this module launches nothing.  Run only from the exact clean commit.
Every returned envelope keeps downstream authority explicitly false.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Hashable, Iterable

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - launcher path setup
    sys.path.insert(0, str(ROOT / "src"))
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_active8_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
RUNTIME_CONTRACT_SOURCE = "configs/editing_v2_process_v2_active8_decision_runtime.json"
SEMANTIC_CONTRACT_SOURCE = "configs/editing_gate_zero_semantic_model_process_v2.json"
OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_active8"

MAX_MAP_CONTAINERS = 80
MAP_CPU = 1.0
FULL_MAP_TASKS_PER_CONTAINER = 1
PILOT_GROUP_COUNT = 3
PILOT_GROUP_SIZE = 1
PILOT_TASK_COUNT = 3
# The full-map geometry remains frozen until the independent canary supplies a
# production-task time and memory bound. It is not used by the default pilot.
MAP_MEMORY_MB = 4 * 1024
MAP_TIMEOUT_SECONDS = 12 * 3600
SENTINEL_TIMEOUT_SECONDS = 6 * 3600
SERIAL_TIMEOUT_SECONDS = 4 * 3600
DEFAULT_SENTINEL_PAIRS_PER_PARTITION = 256
DEFAULT_SMOKE_TASK_SELECTOR = {
    "split": "train",
    "data_lane": "real_endpoint_multistep_path",
    "chunk_index": 24,
    "entry_start": 49_152,
    "entry_stop": 51_200,
    "chunk_row_count": 2_048,
}
DEFAULT_SYNTHETIC_CANARY_SELECTOR = {
    "split": "train",
    "data_lane": "reversible_synthetic_walk",
    "chunk_row_count": 2_048,
}
PROGRESS_PROBE_SELECTOR = {
    "split": "train",
    "data_lane": "real_endpoint_multistep_path",
    "chunk_index": 58,
    "entry_start": 118_784,
    "entry_stop": 120_832,
    "chunk_row_count": 2_048,
}

REVISION_SCHEMA = "compose.data.process_v2_active8_modal_image_revision"
REVISION_SCHEMA_VERSION = 1
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
image = image.add_local_file(ROOT / LAUNCHER_SOURCE, str(REMOTE_ROOT / LAUNCHER_SOURCE), copy=True)

app = modal.App("compose-v4-process-v2-active8")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)

_RUNTIME: Any | None = None
_VALIDATED_REVISION_SHA256: str | None = None


def _source_reuse_statistics(keys: Iterable[Hashable]) -> dict[str, Any]:
    """Summarize exact source-state reuse without inspecting held-out labels."""

    multiplicities = Counter(keys)
    total = sum(multiplicities.values())
    unique = len(multiplicities)
    repeated_queries = total - unique
    histogram = Counter(multiplicities.values())
    return {
        "query_count": total,
        "unique_source_state_count": unique,
        "repeated_source_query_count": repeated_queries,
        "repeated_source_query_fraction": (
            0.0 if total == 0 else repeated_queries / total
        ),
        "source_states_with_multiple_queries": sum(
            count for multiplicity, count in histogram.items() if multiplicity > 1
        ),
        "maximum_queries_per_source_state": max(multiplicities.values(), default=0),
        "source_query_multiplicity_histogram": {
            str(multiplicity): histogram[multiplicity]
            for multiplicity in sorted(histogram)
        },
    }


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
            f"cannot establish Process-V2 Active8 Git identity: git {' '.join(arguments)}"
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
        raise RuntimeError("Process-V2 Active8 serialized source inventory repeats a path")
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
        raise RuntimeError("Process-V2 Active8 launch requires the exact clean commit")
    sources = {
        relative: _file_sha256(root / relative) for relative in _serialized_source_paths(root)
    }
    tracked = set(_git(root, "ls-files").splitlines())
    if not set(sources).issubset(tracked):
        raise RuntimeError("every serialized Active8 source must be Git-tracked")
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
        or not isinstance(sources, dict)
        or set(sources) != set(_serialized_source_paths(REMOTE_ROOT))
    ):
        raise RuntimeError("Process-V2 Active8 image revision disagrees")
    for relative, digest in sources.items():
        if _file_sha256(REMOTE_ROOT / relative) != digest:
            raise RuntimeError(f"serialized Active8 source differs: {relative}")
    _VALIDATED_REVISION_SHA256 = supplied


def _imports() -> dict[str, Any]:
    source_root = str(REMOTE_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.editing_process_v2_admitted_source import (
        resolve_process_v2_admitted_source,
    )
    from compose_v4.data.editing_process_v2_rebind import (
        COMPLETION_FILENAME as REBIND_COMPLETION_FILENAME,
    )
    from compose_v4.data.editing_process_v2_rebind import (
        PLAN_FILENAME as REBIND_PLAN_FILENAME,
    )
    from compose_v4.data.editing_process_v2_rebind import (
        load_process_v2_rebind_plan,
        mounted_process_v2_artifact_path,
    )
    from compose_v4.data.editing_v2_process_v2_active8_map import (
        execute_process_v2_active8_task,
        validate_process_v2_active8_task_result,
    )
    from compose_v4.data.editing_v2_process_v2_active8_plan import (
        PLAN_FILENAME as ACTIVE8_PLAN_FILENAME,
    )
    from compose_v4.data.editing_v2_process_v2_active8_plan import (
        build_process_v2_active8_binding,
        load_process_v2_active8_plan,
        model_runtime_descriptor,
        plan_process_v2_active8,
        write_process_v2_active8_plan,
    )
    from compose_v4.data.editing_v2_process_v2_active8_reduce import (
        completed_process_v2_active8_task_ids,
        finalize_process_v2_active8_reduction,
        load_process_v2_active8_reduction_preparation,
        prepare_process_v2_active8_reduction,
        run_process_v2_active8_sentinel_partition,
    )
    from compose_v4.data.editing_v2_process_v2_active8_sentinel import (
        completed_release_sentinel_partition_ids,
        load_release_sentinel_partition_result,
        sentinel_partition_identities,
    )
    from compose_v4.data.editing_v2_process_v2_chunk_cache import (
        load_committed_process_v2_chunk_cache_completion,
    )
    from compose_v4.experiments.editing_gate_zero_semantic_contract import (
        load_gate_zero_semantic_contract,
    )
    from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
        ACTIVE8_DECISION_RUNTIME,
        validate_process_v2_chain_artifact,
    )
    from compose_v4.experiments.editing_v2_semantic_runtime import (
        SemanticScratchModelConfig,
        build_semantic_scratch_runtime,
    )

    return {
        "resolve_process_v2_admitted_source": resolve_process_v2_admitted_source,
        "REBIND_COMPLETION_FILENAME": REBIND_COMPLETION_FILENAME,
        "REBIND_PLAN_FILENAME": REBIND_PLAN_FILENAME,
        "load_process_v2_rebind_plan": load_process_v2_rebind_plan,
        "mounted_process_v2_artifact_path": mounted_process_v2_artifact_path,
        "execute_process_v2_active8_task": execute_process_v2_active8_task,
        "validate_process_v2_active8_task_result": (validate_process_v2_active8_task_result),
        "ACTIVE8_PLAN_FILENAME": ACTIVE8_PLAN_FILENAME,
        "build_process_v2_active8_binding": build_process_v2_active8_binding,
        "load_process_v2_active8_plan": load_process_v2_active8_plan,
        "model_runtime_descriptor": model_runtime_descriptor,
        "plan_process_v2_active8": plan_process_v2_active8,
        "write_process_v2_active8_plan": write_process_v2_active8_plan,
        "completed_process_v2_active8_task_ids": (completed_process_v2_active8_task_ids),
        "finalize_process_v2_active8_reduction": (finalize_process_v2_active8_reduction),
        "load_process_v2_active8_reduction_preparation": (
            load_process_v2_active8_reduction_preparation
        ),
        "prepare_process_v2_active8_reduction": prepare_process_v2_active8_reduction,
        "run_process_v2_active8_sentinel_partition": (run_process_v2_active8_sentinel_partition),
        "completed_release_sentinel_partition_ids": (completed_release_sentinel_partition_ids),
        "load_release_sentinel_partition_result": (load_release_sentinel_partition_result),
        "sentinel_partition_identities": sentinel_partition_identities,
        "load_committed_process_v2_chunk_cache_completion": (
            load_committed_process_v2_chunk_cache_completion
        ),
        "load_gate_zero_semantic_contract": load_gate_zero_semantic_contract,
        "ACTIVE8_DECISION_RUNTIME": ACTIVE8_DECISION_RUNTIME,
        "validate_process_v2_chain_artifact": validate_process_v2_chain_artifact,
        "SemanticScratchModelConfig": SemanticScratchModelConfig,
        "build_semantic_scratch_runtime": build_semantic_scratch_runtime,
    }


def _runtime() -> Any:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    import torch

    loaded = _imports()
    runtime_path = REMOTE_ROOT / RUNTIME_CONTRACT_SOURCE
    payload = json.loads(runtime_path.read_bytes())
    loaded["validate_process_v2_chain_artifact"](
        payload,
        name=loaded["ACTIVE8_DECISION_RUNTIME"],
        repo_root=REMOTE_ROOT,
    )
    observed_software = {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "torch": importlib.metadata.version("torch"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "networkx": importlib.metadata.version("networkx"),
        "rdkit": importlib.metadata.version("rdkit"),
    }
    if observed_software != payload["software"]:
        raise RuntimeError(f"Process-V2 Active8 software differs: {observed_software}")
    model = payload["model"]
    torch.set_num_threads(1)
    config = loaded["SemanticScratchModelConfig"](
        initialization_seed=int(model["initialization_seed"]),
        max_atoms=int(model["max_atoms"]),
        hidden_dim=int(model["hidden_dim"]),
        message_passing_steps=int(model["message_passing_steps"]),
        mark_dim=int(model["mark_dim"]),
        dtype=str(model["dtype"]),
        atom_vocabulary_class_count=int(model["atom_vocabulary_class_count"]),
        catalog_fingerprint=str(model["catalog_fingerprint"]),
    )
    semantic = loaded["load_gate_zero_semantic_contract"](REMOTE_ROOT / SEMANTIC_CONTRACT_SOURCE)
    _RUNTIME = loaded["build_semantic_scratch_runtime"](config, semantic)
    descriptor = loaded["model_runtime_descriptor"](_RUNTIME)
    for field in (
        "initialization_seed",
        "max_atoms",
        "hidden_dim",
        "message_passing_steps",
        "mark_dim",
        "dtype",
        "atom_vocabulary_class_count",
        "catalog_fingerprint",
    ):
        if descriptor[field] != model[field]:
            raise RuntimeError(f"constructed Active8 runtime differs at {field}")
    return _RUNTIME


def _active8_plan_path(plan: dict[str, Any]) -> Path:
    loaded = _imports()
    root = loaded["mounted_process_v2_artifact_path"](
        str(plan["run_artifact_root"]),
        artifact_root=ARTIFACT_ROOT,
        field="plan.run_artifact_root",
    )
    return root / loaded["ACTIVE8_PLAN_FILENAME"]


def _require_plan_revision(plan: dict[str, Any], revision: dict[str, Any]) -> None:
    """Require the plan namespace to bind the exact validated execution commit."""

    if plan.get("binding", {}).get("execution_commit") != revision.get("commit"):
        raise RuntimeError("the Active8 plan execution commit differs from the image revision")


def _build_task_workloads(
    plan: dict[str, Any], rebind_completion: dict[str, Any]
) -> dict[str, int]:
    """Bind every Active8 task to its frozen admitted-transition workload.

    Workload affects scheduling only.  It is derived from the exact rebind
    completion already bound by the Active8 plan and cannot alter task content,
    admission, or reduction.  Requiring an exact one-to-one join prevents a
    missing or duplicated upstream result from becoming a scheduling hint.
    """

    binding = plan.get("binding")
    if not isinstance(binding, dict) or rebind_completion.get("completion_sha256") != binding.get(
        "rebind_completion_sha256"
    ):
        raise RuntimeError("the Active8 workload source is not the bound rebind completion")
    result_inventory = rebind_completion.get("result_inventory")
    tasks = plan.get("tasks")
    if not isinstance(result_inventory, list) or not isinstance(tasks, list):
        raise RuntimeError("the Active8 workload inventories must be lists")

    workload_by_rebind_task: dict[str, int] = {}
    for result in result_inventory:
        if not isinstance(result, dict) or not isinstance(result.get("counts"), dict):
            raise RuntimeError("a rebind workload result is malformed")
        identity = result.get("task_identity_sha256")
        transitions = result["counts"].get("admitted_transitions")
        if (
            not isinstance(identity, str)
            or _SHA256_RE.fullmatch(identity) is None
            or type(transitions) is not int
            or transitions < 0
            or identity in workload_by_rebind_task
        ):
            raise RuntimeError("a rebind workload result is invalid or duplicated")
        workload_by_rebind_task[identity] = transitions

    task_workloads: dict[str, int] = {}
    joined_rebind_identities: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise RuntimeError("an Active8 workload task is malformed")
        identity = task.get("task_identity_sha256")
        rebind_identity = task.get("rebind_task_identity_sha256")
        if (
            not isinstance(identity, str)
            or _SHA256_RE.fullmatch(identity) is None
            or not isinstance(rebind_identity, str)
            or _SHA256_RE.fullmatch(rebind_identity) is None
            or identity in task_workloads
            or rebind_identity in joined_rebind_identities
            or rebind_identity not in workload_by_rebind_task
        ):
            raise RuntimeError("the Active8 workload join is incomplete or duplicated")
        joined_rebind_identities.add(rebind_identity)
        task_workloads[identity] = workload_by_rebind_task[rebind_identity]
    if joined_rebind_identities != set(workload_by_rebind_task):
        raise RuntimeError("the Active8 workload join does not exhaust the rebind completion")

    total = sum(task_workloads.values())
    counts = rebind_completion.get("counts")
    if (
        not isinstance(counts, dict)
        or type(counts.get("admitted_transitions")) is not int
        or counts["admitted_transitions"] != total
    ):
        raise RuntimeError("the Active8 workload total disagrees with the rebind completion")
    return task_workloads


def _validate_task_workloads(plan: dict[str, Any], task_workloads: Any) -> dict[str, int]:
    """Require a complete typed scheduling map for the exact plan."""

    if not isinstance(task_workloads, dict):
        raise RuntimeError("the Active8 task workload map is absent")
    workload_by_task: dict[str, int] = {}
    for identity, transitions in task_workloads.items():
        if (
            not isinstance(identity, str)
            or _SHA256_RE.fullmatch(identity) is None
            or type(transitions) is not int
            or transitions < 0
        ):
            raise RuntimeError("an Active8 task workload is invalid")
        workload_by_task[identity] = transitions
    planned = {str(task["task_identity_sha256"]) for task in plan.get("tasks", [])}
    if set(workload_by_task) != planned:
        raise RuntimeError("the Active8 task workloads do not cover the plan exactly")
    return workload_by_task


def _partition_task_ids(
    plan: dict[str, Any],
    *,
    partition_count: int,
    partition_index: int,
    completed: set[str],
) -> list[str]:
    """Select a disjoint deterministic map partition in plan order."""

    if type(partition_count) is not int or partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if type(partition_index) is not int or not 0 <= partition_index < partition_count:
        raise ValueError("partition_index lies outside partition_count")
    return [
        str(task["task_identity_sha256"])
        for index, task in enumerate(plan["tasks"])
        if index % partition_count == partition_index
        and str(task["task_identity_sha256"]) not in completed
    ]


def _task_groups(plan: dict[str, Any], task_ids: list[str], *, group_size: int) -> list[list[str]]:
    """Build deterministic bounded groups in the supplied plan order."""

    if type(group_size) is not int or group_size <= 0 or group_size > FULL_MAP_TASKS_PER_CONTAINER:
        raise ValueError(f"group_size must lie in [1, {FULL_MAP_TASKS_PER_CONTAINER}]")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("an Active8 task group input repeats an identity")
    planned = {str(task["task_identity_sha256"]): task for task in plan["tasks"]}
    if not set(task_ids).issubset(planned):
        raise ValueError("an Active8 task group input names an unplanned task")
    groups = [task_ids[start : start + group_size] for start in range(0, len(task_ids), group_size)]
    return groups


def _workload_ordered_task_groups(
    plan: dict[str, Any],
    task_ids: list[str],
    *,
    group_size: int,
    task_workloads: dict[str, int],
) -> list[list[str]]:
    """Submit the heaviest restart-safe tasks first.

    Sorting by the prospectively frozen rebind transition count bounds the tail
    without changing task membership or outputs. One task per container lets
    Modal replenish every freed worker immediately.
    """

    # Reuse the structural checks and exact task-membership validation.
    _task_groups(plan, task_ids, group_size=group_size)
    workload_by_task = _validate_task_workloads(plan, task_workloads)
    plan_order = {
        str(task["task_identity_sha256"]): index for index, task in enumerate(plan["tasks"])
    }
    ordered = sorted(
        task_ids,
        key=lambda identity: (-workload_by_task[identity], plan_order[identity]),
    )
    return [ordered[start : start + group_size] for start in range(0, len(ordered), group_size)]


def _pilot_task_ids(plan: dict[str, Any]) -> list[str]:
    """Freeze three train-only operational canaries in plan order.

    The canaries cover publication overhead, the multistep lane that dominated
    the stopped pilot, and the other full-corpus dominant lane.  Capability or
    outcome values never influence this selection.
    """

    train_tasks = [task for task in plan["tasks"] if task.get("split") == "train"]
    heavy = [
        task
        for task in train_tasks
        if all(task.get(field) == value for field, value in DEFAULT_SMOKE_TASK_SELECTOR.items())
    ]
    if len(heavy) != 1:
        raise ValueError("the pinned Active8 train canary source chunk is absent or ambiguous")
    synthetic = [
        task
        for task in train_tasks
        if all(
            task.get(field) == value for field, value in DEFAULT_SYNTHETIC_CANARY_SELECTOR.items()
        )
    ]
    if not synthetic:
        raise ValueError("the Active8 train partition lacks a full synthetic-walk canary")
    tiny_size = min(int(task["chunk_row_count"]) for task in train_tasks)
    tiny = next(task for task in train_tasks if int(task["chunk_row_count"]) == tiny_size)
    selected = {
        str(tiny["task_identity_sha256"]),
        str(heavy[0]["task_identity_sha256"]),
        str(synthetic[0]["task_identity_sha256"]),
    }
    if len(selected) != PILOT_TASK_COUNT:
        raise ValueError("the Active8 operational canary roles do not resolve distinctly")
    cohort = [
        str(task["task_identity_sha256"])
        for task in plan["tasks"]
        if str(task["task_identity_sha256"]) in selected
    ]
    if len(cohort) != PILOT_TASK_COUNT:
        raise RuntimeError("the Active8 bounded pilot cohort is inconsistent")
    return cohort


def _submission_groups(
    plan: dict[str, Any],
    missing: list[str],
    *,
    completed: set[str],
    group_size: int,
    group_limit: int,
    pilot: bool,
    full_map: bool,
    task_workloads: dict[str, int] | None = None,
) -> list[list[str]]:
    """Select the fixed bounded pilot or an explicitly requested full map."""

    if type(group_limit) is not int or group_limit < 0:
        raise ValueError("group_limit must be a nonnegative integer")
    if pilot == full_map:
        raise ValueError("select exactly one of the bounded pilot or full Active8 map")
    if pilot:
        if group_size != PILOT_GROUP_SIZE or group_limit != PILOT_GROUP_COUNT:
            raise ValueError("the Active8 pilot requires three independent canary tasks")
        cohort = _pilot_task_ids(plan)
        remaining = [identity for identity in cohort if identity not in completed]
        if not set(remaining).issubset(missing):
            raise ValueError("the Active8 pilot cohort lies outside the selected partition")
        return [[identity] for identity in remaining]
    groups = _workload_ordered_task_groups(
        plan,
        missing,
        group_size=group_size,
        task_workloads=task_workloads,
    )
    return groups if group_limit == 0 else groups[:group_limit]


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / 1e6


@app.function(
    image=image,
    cpu=1.0,
    memory=8192,
    timeout=SERIAL_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_plan(
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    output_artifact_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Resolve exact upstream completions and publish one immutable plan."""

    _validate_remote_revision(revision)
    loaded = _imports()
    artifact_volume.reload()
    cache_completion = loaded["load_committed_process_v2_chunk_cache_completion"](
        cache_run_artifact_root,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    rebind_root = loaded["mounted_process_v2_artifact_path"](
        rebind_run_artifact_root,
        artifact_root=ARTIFACT_ROOT,
        field="rebind_run_artifact_root",
    )
    rebind_plan = loaded["load_process_v2_rebind_plan"](
        rebind_root / loaded["REBIND_PLAN_FILENAME"], repo_root=REMOTE_ROOT
    )
    if rebind_plan["run_artifact_root"] != rebind_run_artifact_root:
        raise RuntimeError("the rebind plan names another run root")
    admitted = loaded["resolve_process_v2_admitted_source"](
        rebind_plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    rebind_completion = json.loads(
        (rebind_root / loaded["REBIND_COMPLETION_FILENAME"]).read_bytes()
    )
    runtime = _runtime()
    binding = loaded["build_process_v2_active8_binding"](
        cache_completion=cache_completion,
        rebind_plan=rebind_plan,
        rebind_completion=rebind_completion,
        admitted_source_identity=admitted.identity(),
        model_runtime=loaded["model_runtime_descriptor"](runtime),
        execution_commit=str(revision["commit"]),
        repo_root=REMOTE_ROOT,
    )
    plan = loaded["plan_process_v2_active8"](
        binding,
        rebind_plan=rebind_plan,
        output_artifact_prefix=output_artifact_prefix,
    )
    _require_plan_revision(plan, revision)
    loaded["write_process_v2_active8_plan"](
        plan, artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT
    )
    task_workloads = _build_task_workloads(plan, rebind_completion)
    artifact_volume.commit()
    return {"plan": plan, "task_workloads": task_workloads}


@app.function(
    image=image,
    cpu=1.0,
    memory=4096,
    timeout=SERIAL_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def scan_completed(plan_path: str, revision: dict[str, Any]) -> list[str]:
    _validate_remote_revision(revision)
    loaded = _imports()
    artifact_volume.reload()
    plan = loaded["load_process_v2_active8_plan"](Path(plan_path), repo_root=REMOTE_ROOT)
    _require_plan_revision(plan, revision)
    return sorted(
        loaded["completed_process_v2_active8_task_ids"](plan, artifact_root=ARTIFACT_ROOT)
    )


@app.function(
    image=image,
    cpu=MAP_CPU,
    memory=MAP_MEMORY_MB,
    timeout=MAP_TIMEOUT_SECONDS,
    max_containers=MAX_MAP_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def decide_task(
    plan_path: str, task_identity_sha256: str, revision: dict[str, Any]
) -> dict[str, Any]:
    """Measure and publish one restart-safe task with no cohort barrier."""

    _validate_remote_revision(revision)
    loaded = _imports()
    artifact_volume.reload()
    plan = loaded["load_process_v2_active8_plan"](Path(plan_path), repo_root=REMOTE_ROOT)
    _require_plan_revision(plan, revision)
    tasks = {str(task["task_identity_sha256"]): task for task in plan["tasks"]}
    if task_identity_sha256 not in tasks:
        raise RuntimeError("the Active8 map names an unplanned task")

    import torch

    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    with torch.inference_mode():
        receipt = loaded["execute_process_v2_active8_task"](
            plan,
            task_identity_sha256,
            runtime=_runtime(),
            artifact_root=ARTIFACT_ROOT,
            repo_root=REMOTE_ROOT,
        )
    wall_seconds = time.perf_counter() - wall_before
    cpu_seconds = time.process_time() - cpu_before
    task = tasks[task_identity_sha256]
    output = loaded["mounted_process_v2_artifact_path"](
        str(task["output_artifact_path"]),
        artifact_root=ARTIFACT_ROOT,
        field="task.output_artifact_path",
    )
    reopened, summary = loaded["validate_process_v2_active8_task_result"](output)
    if not (
        reopened["task_identity_sha256"] == task_identity_sha256
        and reopened["receipt_sha256"] == receipt["receipt_sha256"]
        and summary["binding_sha256"] == plan["binding_sha256"]
        and summary["plan_sha256"] == plan["plan_sha256"]
        and summary["run_identity_sha256"] == plan["run_identity_sha256"]
    ):
        raise RuntimeError("the Active8 task output does not bind its plan")
    artifact_volume.commit()
    return {
        "task_identity_sha256": task_identity_sha256,
        "data_lane": str(task["data_lane"]),
        "chunk_row_count": int(task["chunk_row_count"]),
        "teacher_action_count": int(sum(summary["action_family_histogram"].values())),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "process_peak_rss_mb": _peak_rss_mb(),
        "receipt_sha256": str(receipt["receipt_sha256"]),
    }


@app.function(
    image=image,
    cpu=MAP_CPU,
    memory=MAP_MEMORY_MB,
    timeout=MAP_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def probe_task_progress(
    plan_path: str,
    task_identity_sha256: str,
    revision: dict[str, Any],
    progress_batch_size: int,
    stats_only: bool,
) -> dict[str, Any]:
    """Measure one queued task with bounded heartbeats and no publication.

    This operational probe uses the production reader, policy, model, support
    checker, executor, and structural row construction.  It deliberately wraps
    only the checker's ``evaluate_many`` call so that each bounded block emits
    elapsed time, CPU time, resident memory, throughput, and ETA.  It writes no
    artifact and grants no downstream authority.
    """

    if type(progress_batch_size) is not int or progress_batch_size <= 0:
        raise ValueError("progress_batch_size must be positive")
    if type(stats_only) is not bool:
        raise ValueError("stats_only must be boolean")
    _validate_remote_revision(revision)
    loaded = _imports()
    artifact_volume.reload()
    plan = loaded["load_process_v2_active8_plan"](Path(plan_path), repo_root=REMOTE_ROOT)
    _require_plan_revision(plan, revision)
    tasks = {str(task["task_identity_sha256"]): task for task in plan["tasks"]}
    if task_identity_sha256 not in tasks:
        raise RuntimeError("the Active8 progress probe names an unplanned task")
    task = tasks[task_identity_sha256]

    from compose_v4.data.editing_process_v2_rebind import (
        mounted_process_v2_artifact_path,
    )
    from compose_v4.data.editing_v2_process_v2_active8_admission import (
        ProductionProcessV2BatchedTeacherSupportChecker,
        build_process_v2_semantic_active8_admission_policy,
        validate_process_v2_semantic_active8_admission_policy,
    )
    from compose_v4.data.editing_v2_process_v2_active8_map import (
        TEACHER_SUPPORT_BATCH_SIZE,
        _decide_chunk,
        read_rebind_chunk_decisions,
    )
    from compose_v4.data.editing_v2_semantic_capability_cells import (
        load_semantic_capability_cell_registry,
    )
    from compose_v4.model.factorized_tracelet_rate_model import (
        molecular_state_cache_key,
    )

    runtime = _runtime()
    binding = plan["binding"]
    if loaded["model_runtime_descriptor"](runtime) != dict(binding["model_runtime"]):
        raise RuntimeError("the progress probe runtime is not the one bound by the plan")
    policy = validate_process_v2_semantic_active8_admission_policy(
        build_process_v2_semantic_active8_admission_policy()
    )
    if policy.policy_sha256 != binding["active8_policy_sha256"]:
        raise RuntimeError("the progress probe policy is not the one bound by the plan")
    registry = load_semantic_capability_cell_registry()
    if registry.registry_sha256 != binding["capability_cell_registry_sha256"]:
        raise RuntimeError("the progress probe cell registry is not the one bound by the plan")
    inner = ProductionProcessV2BatchedTeacherSupportChecker(
        runtime.model,
        policy=policy,
        batch_size=TEACHER_SUPPORT_BATCH_SIZE,
    )

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    class StatsOnlyComplete(Exception):
        """Stop after the exact production query-preparation boundary."""

    class ProgressChecker:
        def __init__(self) -> None:
            self.policy = inner.policy
            self.processed = 0
            self.total = 0
            self.workload_statistics: dict[str, Any] | None = None

        def evaluate_many(self, queries: Any) -> tuple[Any, ...]:
            self.total = len(queries)
            source_keys = (
                molecular_state_cache_key(addressed.path.state_at(step_index))
                for addressed, step_index in queries
            )
            rule_histogram = Counter(
                str(addressed.trace.steps[step_index].rule_name)
                for addressed, step_index in queries
            )
            self.workload_statistics = {
                **_source_reuse_statistics(source_keys),
                "teacher_rule_histogram": {
                    rule_name: rule_histogram[rule_name]
                    for rule_name in sorted(rule_histogram)
                },
                "query_preparation_wall_seconds": time.perf_counter() - wall_start,
                "query_preparation_cpu_seconds": time.process_time() - cpu_start,
                "process_peak_rss_mb": _peak_rss_mb(),
            }
            print(
                json.dumps(
                    {
                        "event": "ACTIVE8_PROGRESS_PROBE_WORKLOAD",
                        "task_identity_sha256": task_identity_sha256,
                        **self.workload_statistics,
                        "stats_only": stats_only,
                        "scientific_authority": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if stats_only:
                raise StatsOnlyComplete
            next_report = min(progress_batch_size, self.total)

            def report(processed: int, total: int) -> None:
                nonlocal next_report
                self.processed = processed
                if processed < next_report and processed != total:
                    return
                wall_seconds = time.perf_counter() - wall_start
                cpu_seconds = time.process_time() - cpu_start
                rate = self.processed / wall_seconds if wall_seconds > 0.0 else 0.0
                eta_seconds = (
                    None if rate <= 0.0 else (self.total - self.processed) / rate
                )
                print(
                    json.dumps(
                        {
                            "event": "ACTIVE8_PROGRESS_PROBE_HEARTBEAT",
                            "task_identity_sha256": task_identity_sha256,
                            "processed_transitions": self.processed,
                            "total_transitions": self.total,
                            "wall_seconds": wall_seconds,
                            "cpu_seconds": cpu_seconds,
                            "transitions_per_second": rate,
                            "eta_seconds": eta_seconds,
                            "process_peak_rss_mb": _peak_rss_mb(),
                            "scientific_authority": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                while next_report <= processed:
                    next_report += progress_batch_size

            return inner.evaluate_many(queries, progress_callback=report)

    checker = ProgressChecker()
    rebind_output = mounted_process_v2_artifact_path(
        str(task["rebind_task_artifact_path"]),
        artifact_root=ARTIFACT_ROOT,
        field="task.rebind_task_artifact_path",
    )
    decisions, pinned_process_identity = read_rebind_chunk_decisions(
        rebind_output,
        entry_start=int(task["entry_start"]),
        entry_stop=int(task["entry_stop"]),
        task_identity_sha256=str(task["rebind_task_identity_sha256"]),
        chunk_file_sha256=str(task["chunk_file_sha256"]),
        pinned_process_identity_sha256=str(task["pinned_process_identity_sha256"]),
    )
    row_count = 0
    produced_transition_count = 0
    try:
        for _row, produced in _decide_chunk(
            task,
            decisions=decisions,
            checker=checker,
            namespace=registry.namespace,
            expected_process_identity=pinned_process_identity,
            artifact_root=ARTIFACT_ROOT,
            repo_root=REMOTE_ROOT,
        ):
            row_count += 1
            produced_transition_count += len(produced)
    except StatsOnlyComplete:
        pass
    result = {
        "event": (
            "ACTIVE8_PROGRESS_PROBE_STATS_COMPLETE"
            if stats_only
            else "ACTIVE8_PROGRESS_PROBE_COMPLETE"
        ),
        "task_identity_sha256": task_identity_sha256,
        "source_rows": None if stats_only else row_count,
        "planned_source_rows": int(task["chunk_row_count"]),
        "processed_transitions": checker.processed,
        "total_transitions": checker.total,
        "produced_transitions": None if stats_only else produced_transition_count,
        "wall_seconds": time.perf_counter() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
        "process_peak_rss_mb": _peak_rss_mb(),
        "stats_only": stats_only,
        "workload_statistics": checker.workload_statistics,
        "scientific_authority": False,
        "artifact_published": False,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    cpu=1.0,
    memory=8192,
    timeout=SERIAL_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_sentinel(
    plan_path: str, pairs_per_partition: int, revision: dict[str, Any]
) -> dict[str, Any]:
    _validate_remote_revision(revision)
    loaded = _imports()
    artifact_volume.reload()
    plan = loaded["load_process_v2_active8_plan"](Path(plan_path), repo_root=REMOTE_ROOT)
    _require_plan_revision(plan, revision)
    prepared = loaded["prepare_process_v2_active8_reduction"](
        plan,
        runtime=_runtime(),
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
        max_sentinel_pairs_per_partition=pairs_per_partition,
        publish=True,
    )
    artifact_volume.commit()
    completed = loaded["completed_release_sentinel_partition_ids"](
        active8_plan=plan,
        sentinel_plan=prepared["sentinel_plan"],
        artifact_root=ARTIFACT_ROOT,
    )
    return {
        "run_artifact_root": plan["run_artifact_root"],
        "preparation_sha256": prepared["preparation_sha256"],
        "partition_ids": loaded["sentinel_partition_identities"](prepared["sentinel_plan"]),
        "completed_partition_ids": sorted(completed),
    }


@app.function(
    image=image,
    cpu=1.0,
    memory=8192,
    timeout=SENTINEL_TIMEOUT_SECONDS,
    max_containers=MAX_MAP_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_sentinel_partition(
    plan_path: str, partition_identity_sha256: str, revision: dict[str, Any]
) -> dict[str, Any]:
    _validate_remote_revision(revision)
    loaded = _imports()
    artifact_volume.reload()
    plan = loaded["load_process_v2_active8_plan"](Path(plan_path), repo_root=REMOTE_ROOT)
    _require_plan_revision(plan, revision)
    prepared = loaded["load_process_v2_active8_reduction_preparation"](
        str(plan["run_artifact_root"]),
        active8_plan=plan,
        artifact_root=ARTIFACT_ROOT,
    )
    result = loaded["run_process_v2_active8_sentinel_partition"](
        plan,
        prepared,
        partition_identity_sha256,
        runtime=_runtime(),
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
        publish=True,
    )
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    cpu=1.0,
    memory=8192,
    timeout=SERIAL_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def finalize(plan_path: str, revision: dict[str, Any]) -> dict[str, Any]:
    _validate_remote_revision(revision)
    loaded = _imports()
    artifact_volume.reload()
    plan = loaded["load_process_v2_active8_plan"](Path(plan_path), repo_root=REMOTE_ROOT)
    _require_plan_revision(plan, revision)
    prepared = loaded["load_process_v2_active8_reduction_preparation"](
        str(plan["run_artifact_root"]),
        active8_plan=plan,
        artifact_root=ARTIFACT_ROOT,
    )
    partition_ids = loaded["sentinel_partition_identities"](prepared["sentinel_plan"])
    completed = loaded["completed_release_sentinel_partition_ids"](
        active8_plan=plan,
        sentinel_plan=prepared["sentinel_plan"],
        artifact_root=ARTIFACT_ROOT,
    )
    if set(completed) != set(partition_ids):
        raise RuntimeError("the Active8 sentinel partition inventory is incomplete")
    results = [
        loaded["load_release_sentinel_partition_result"](
            identity,
            active8_plan=plan,
            sentinel_plan=prepared["sentinel_plan"],
            artifact_root=ARTIFACT_ROOT,
        )
        for identity in partition_ids
    ]
    completion = loaded["finalize_process_v2_active8_reduction"](
        plan,
        prepared,
        results,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
        publish=True,
    )
    artifact_volume.commit()
    return completion


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=MAP_TIMEOUT_SECONDS,
    max_containers=1,
)
def driver(
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    output_artifact_prefix: str,
    partition_count: int,
    partition_index: int,
    group_size: int,
    group_limit: int,
    max_map_containers: int,
    pilot: bool,
    full_map: bool,
    reduce: bool,
    sentinel_pairs_per_partition: int,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Run detached orchestration remotely so client exit cannot stop fan-out."""

    _validate_remote_revision(revision)
    if (
        type(max_map_containers) is not int
        or max_map_containers <= 0
        or max_map_containers > MAX_MAP_CONTAINERS
    ):
        raise ValueError(f"max_map_containers must lie in [1, {MAX_MAP_CONTAINERS}]")
    if sum((bool(pilot), bool(full_map), bool(reduce))) != 1:
        raise ValueError("select exactly one Active8 phase: pilot, full_map, or reduce")
    if (pilot or reduce) and (int(partition_count) != 1 or int(partition_index) != 0):
        raise ValueError("the pilot and reduction require the complete unpartitioned plan")
    if pilot and (int(group_size) != PILOT_GROUP_SIZE or int(group_limit) != PILOT_GROUP_COUNT):
        raise ValueError("the Active8 pilot requires the frozen canary geometry")
    if full_map and (int(group_size) != FULL_MAP_TASKS_PER_CONTAINER or int(group_limit) != 0):
        raise ValueError(
            "the full Active8 map requires "
            f"group_size={FULL_MAP_TASKS_PER_CONTAINER} and group_limit=0"
        )

    prepared_plan = prepare_plan.remote(
        cache_run_artifact_root,
        rebind_run_artifact_root,
        output_artifact_prefix,
        revision,
    )
    plan = prepared_plan["plan"]
    workload_by_task = _validate_task_workloads(plan, prepared_plan["task_workloads"])
    plan_path = str(_active8_plan_path(plan))
    complete = set(scan_completed.remote(plan_path, revision))
    missing = _partition_task_ids(
        plan,
        partition_count=int(partition_count),
        partition_index=int(partition_index),
        completed=complete,
    )
    submitted_groups = []
    if not reduce:
        submitted_groups = _submission_groups(
            plan,
            missing,
            completed=complete,
            group_size=int(group_size),
            group_limit=int(group_limit),
            pilot=bool(pilot),
            full_map=bool(full_map),
            task_workloads=workload_by_task,
        )
    submitted_tasks = sum(len(group) for group in submitted_groups)
    submitted_transition_work = sum(
        workload_by_task[identity] for group in submitted_groups for identity in group
    )
    submitted_group_capacity = sum(
        max(workload_by_task[identity] for identity in group) * len(group)
        for group in submitted_groups
    )
    canary_measurements: list[dict[str, Any]] = []
    if submitted_groups:
        if any(len(group) != 1 for group in submitted_groups):
            raise RuntimeError("the Active8 map requires one task per replenishable container")
        decide_task.update_autoscaler(max_containers=max_map_containers)
        task_results = list(
            decide_task.starmap(
                [(plan_path, group[0], revision) for group in submitted_groups]
            )
        )
        if len(task_results) != len(submitted_groups) or {
            result["task_identity_sha256"] for result in task_results
        } != {group[0] for group in submitted_groups}:
            raise RuntimeError("the Active8 map lost a task result")
        if pilot:
            canary_measurements = task_results

    completion = None
    sentinel_partitions = 0
    if reduce:
        all_complete = set(scan_completed.remote(plan_path, revision))
        expected = {str(task["task_identity_sha256"]) for task in plan["tasks"]}
        if all_complete != expected:
            raise RuntimeError(
                f"Active8 reduction requires all tasks; missing={len(expected - all_complete)}"
            )
        prepared = prepare_sentinel.remote(plan_path, int(sentinel_pairs_per_partition), revision)
        partition_ids = list(prepared["partition_ids"])
        completed_partition_ids = set(prepared["completed_partition_ids"])
        missing_partition_ids = [
            identity for identity in partition_ids if identity not in completed_partition_ids
        ]
        sentinel_partitions = len(missing_partition_ids)
        if missing_partition_ids:
            results = list(
                run_sentinel_partition.starmap(
                    [(plan_path, identity, revision) for identity in missing_partition_ids]
                )
            )
            if len(results) != len(missing_partition_ids):
                raise RuntimeError("the Active8 sentinel lost a partition result")
        completion = finalize.remote(plan_path, revision)

    result = {
        "phase": "process_v2_active8_complete" if reduce else "process_v2_active8_map",
        "run_artifact_root": plan["run_artifact_root"],
        "expected_tasks": plan["expected_task_count"],
        "already_complete": len(complete),
        "missing_tasks_before_submit": len(missing),
        "submitted_groups": len(submitted_groups),
        "submitted_tasks": submitted_tasks,
        "submitted_admitted_transitions": submitted_transition_work,
        "submitted_group_capacity": submitted_group_capacity,
        "submitted_group_utilization": (
            None
            if submitted_group_capacity == 0
            else submitted_transition_work / submitted_group_capacity
        ),
        "task_workloads_sha256": _sha256(
            [
                [
                    str(task["task_identity_sha256"]),
                    workload_by_task[str(task["task_identity_sha256"])],
                ]
                for task in plan["tasks"]
            ]
        ),
        "canary_measurements": canary_measurements,
        "group_size": int(group_size),
        "group_limit": int(group_limit),
        "max_map_containers": max_map_containers,
        "pilot": bool(pilot),
        "full_map": bool(full_map),
        "partition_count": int(partition_count),
        "partition_index": int(partition_index),
        "sentinel_partitions": sentinel_partitions,
        "completion_sha256": None if completion is None else completion["completion_sha256"],
        "image_revision": revision,
        "training_launched": False,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(image=image, cpu=0.125, memory=512, timeout=SERIAL_TIMEOUT_SECONDS)
def progress_probe_orchestrator(
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    output_artifact_prefix: str,
    progress_batch_size: int,
    stats_only: bool,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Prepare and spawn one non-authoritative queued-task progress probe."""

    _validate_remote_revision(revision)
    if type(progress_batch_size) is not int or progress_batch_size <= 0:
        raise ValueError("progress_batch_size must be positive")
    if type(stats_only) is not bool:
        raise ValueError("stats_only must be boolean")
    prepared_plan = prepare_plan.remote(
        cache_run_artifact_root,
        rebind_run_artifact_root,
        output_artifact_prefix,
        revision,
    )
    plan = prepared_plan["plan"]
    selected = [
        task
        for task in plan["tasks"]
        if all(task.get(field) == value for field, value in PROGRESS_PROBE_SELECTOR.items())
    ]
    if len(selected) != 1:
        raise RuntimeError("the Active8 progress probe selector is absent or ambiguous")
    plan_path = str(_active8_plan_path(plan))
    call = probe_task_progress.spawn(
        plan_path,
        str(selected[0]["task_identity_sha256"]),
        revision,
        int(progress_batch_size),
        stats_only,
    )
    result = {
        "phase": "process_v2_active8_progress_probe_spawned",
        "probe_call_id": call.object_id,
        "selector": PROGRESS_PROBE_SELECTOR,
        "progress_batch_size": int(progress_batch_size),
        "stats_only": stats_only,
        "image_revision": revision,
        "artifact_publication": False,
        "scientific_authority": False,
        "training_launched": False,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main(
    cache_run_artifact_root: str,
    rebind_run_artifact_root: str,
    expected_commit: str,
    output_artifact_prefix: str = OUTPUT_ARTIFACT_PREFIX,
    partition_count: int = 1,
    partition_index: int = 0,
    group_size: int = PILOT_GROUP_SIZE,
    group_limit: int = PILOT_GROUP_COUNT,
    max_map_containers: int = MAX_MAP_CONTAINERS,
    pilot: bool = True,
    full_map: bool = False,
    reduce: bool = False,
    sentinel_pairs_per_partition: int = DEFAULT_SENTINEL_PAIRS_PER_PARTITION,
    progress_probe: bool = False,
    progress_batch_size: int = 64,
    progress_stats_only: bool = False,
) -> None:
    """Spawn one disconnect-safe remote pilot, full-map, or reduction driver."""

    revision = local_image_revision(expected_commit=expected_commit)
    if progress_probe:
        call = progress_probe_orchestrator.spawn(
            cache_run_artifact_root,
            rebind_run_artifact_root,
            output_artifact_prefix,
            int(progress_batch_size),
            bool(progress_stats_only),
            revision,
        )
        print(
            json.dumps(
                {
                    "phase": "process_v2_active8_progress_probe_launched",
                    "probe_orchestrator_call_id": call.object_id,
                    "selector": PROGRESS_PROBE_SELECTOR,
                    "progress_batch_size": int(progress_batch_size),
                    "stats_only": bool(progress_stats_only),
                    "image_revision": revision,
                    "artifact_publication": False,
                    "scientific_authority": False,
                    "training_launched": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    call = driver.spawn(
        cache_run_artifact_root,
        rebind_run_artifact_root,
        output_artifact_prefix,
        int(partition_count),
        int(partition_index),
        int(group_size),
        int(group_limit),
        int(max_map_containers),
        bool(pilot),
        bool(full_map),
        bool(reduce),
        int(sentinel_pairs_per_partition),
        revision,
    )
    print(
        json.dumps(
            {
                "phase": "process_v2_active8_driver_launched",
                "driver_call_id": call.object_id,
                "pilot": bool(pilot),
                "full_map": bool(full_map),
                "reduce": bool(reduce),
                "max_map_containers": int(max_map_containers),
                "image_revision": revision,
                "training_launched": False,
                "training_authorized": False,
                "gate_zero_authorized": False,
                "t1_authorized": False,
                "bounded_p50_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "MAX_MAP_CONTAINERS",
    "PROGRESS_PROBE_SELECTOR",
    "_source_reuse_statistics",
    "_partition_task_ids",
    "local_image_revision",
]
