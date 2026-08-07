"""Prepare and run the bounded Process-V2 P50 pilot on Modal.

The launcher is intentionally thin.  Production modules own selection,
successor compilation, the fifty-step scratch optimization, result validation,
and the P500 decision.  This file only supplies exact-source attestation,
restart-safe CPU fan-out, immutable publication, and one detached driver.

Importing this module launches nothing.  P50 starts only when the current
Process-V2 Gate-0 and T1 artifacts form an exact bounded-P50 GO chain.  The T1
selected checkpoint is never accepted as an input and is never opened.
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_process_v2_p50_app.py"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")

PREPARED_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_p50_prepared"
COLLATED_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_p50_collated"
LEGACY_PREPARED_RUN_ROOT = (
    PREPARED_OUTPUT_PREFIX + "/d9ffc03094161dcadb3d1fc670d9c4e9595ec31787cca7964560a7a591b2f1dc"
)
RUN_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_p50"
P50_POLICY_SOURCE = "configs/editing_v2_process_v2_p50_recipe_policy.json"
T1_EVIDENCE_SOURCE = "configs/editing_v2_process_v2_p50_t1_evidence.json"
SCOPED_T1_OUTPUT_PREFIX = "/artifacts/editing_v2/process_v2_t1_scoped_capacity"
SELECTION_FILENAME = "PROCESS_V2_P50_SELECTION.json"
PREPARED_FILENAME = "PROCESS_V2_P50_PREPARED_INPUTS.json"
LEAF_FILENAME = "PROCESS_V2_P50_PREPARED_LEAF.json"
TASKS_DIRNAME = "tasks"
TERMINAL_CHECKPOINT_FILENAME = "PROCESS_V2_P50_TERMINAL.pt"
COLLATED_PLAN_FILENAME = "PROCESS_V2_P50_COLLATED_PLAN.json"
COLLATED_COMPLETION_FILENAME = "PROCESS_V2_P50_COLLATED_COMPLETE.json"
COLLATED_PAYLOAD_FILENAME = "PROCESS_V2_P50_COLLATED_BATCH.pt"
COLLATED_RECEIPT_FILENAME = "PROCESS_V2_P50_COLLATED_RECEIPT.json"
COLLATED_TASKS_DIRNAME = "collated_tasks"

MAX_CPU_CONTAINERS = 100
CPU_PER_LEAF = 1.0
#: Cores per PREP leaf. Modal bills core-seconds, so N cores in one container
#: cost the same as N containers -- but containers are tier-capped and cores are
#: not, which makes this the only lever that buys wall clock without buying
#: compute. Memory scales with it because each forked child holds its own
#: chemistry cache.
PREP_CPU_PER_LEAF = 8.0
PREP_MEMORY_MB = 32 * 1024
PREP_COMPILE_WORKERS = 8
CPU_MEMORY_MB = 8 * 1024
CPU_LEAF_TIMEOUT_SECONDS = 45 * 60
#: Held back from a prep slice's compile budget for the collation,
#: publication and volume commit that still run after the compile ends.
PREP_SLICE_RESERVE_SECONDS = 5 * 60
#: Written once per output root to prove writability BEFORE the compile.
PREP_PREFLIGHT_FILENAME = "PROCESS_V2_PREP_WRITABILITY_PREFLIGHT.json"
COORDINATOR_TIMEOUT_SECONDS = 6 * 3600
GPU_TIMEOUT_SECONDS = 20 * 60
HEARTBEAT_SECONDS = 30
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

REVISION_SCHEMA = "compose.editing_v2.process_v2_p50_modal_image_revision"
REVISION_SCHEMA_VERSION = 1
SOURCE_REVISION_SCHEMA = "compose.editing_v2.process_v2_p50_authenticated_image_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "compose.editing_v2.process_v2_p50_terminal_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STATUS = "COMPLETE_PROCESS_V2_P50_TERMINAL_NO_DOWNSTREAM_AUTHORITY"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

COLLATED_PLAN_SCHEMA = "compose.editing_v2.process_v2_p50_collated_plan"
COLLATED_RECEIPT_SCHEMA = "compose.editing_v2.process_v2_p50_collated_receipt"
COLLATED_COMPLETION_SCHEMA = "compose.editing_v2.process_v2_p50_collated_completion"
COLLATED_PAYLOAD_SCHEMA = "compose.editing_v2.process_v2_p50_collated_batch"
COLLATED_SCHEMA_VERSION = 1

_RUNNER_IMPLEMENTATION_SOURCES = (
    LAUNCHER_SOURCE,
    "src/compose_v4/experiments/editing_v2_process_v2_p50_prerequisites.py",
    "src/compose_v4/experiments/editing_v2_process_v2_p50_result.py",
    "src/compose_v4/experiments/editing_v2_process_v2_p50_runtime.py",
    "src/compose_v4/experiments/editing_v2_process_v2_t1_scoped_result.py",
    "src/compose_v4/experiments/editing_v2_process_v2_t1_runtime.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/score_free_successor_support.py",
)

_COLLATION_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/editing_v2_process_v2_p50_runtime.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_capacity_runner.py",
    "src/compose_v4/experiments/factorized_mark_conditional.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
)

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
            "CUBLAS_WORKSPACE_CONFIG": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "NVIDIA_TF32_OVERRIDE": "0",
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

app = modal.App("compose-v4-process-v2-p50")
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
            f"cannot establish Process-V2 P50 Git identity: git {' '.join(arguments)}"
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
        raise RuntimeError("Process-V2 P50 serialized source inventory repeats a path")
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
        raise RuntimeError("Process-V2 P50 launch requires the exact clean commit")
    sources = {
        relative: _file_sha256(root / relative) for relative in _serialized_source_paths(root)
    }
    tracked = set(_git(root, "ls-files").splitlines())
    if not set(sources).issubset(tracked):
        raise RuntimeError("every serialized Process-V2 P50 source must be Git-tracked")
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
        raise RuntimeError("Process-V2 P50 image revision disagrees")
    for relative, digest in sources.items():
        if _file_sha256(REMOTE_ROOT / str(relative)) != digest:
            raise RuntimeError(f"serialized Process-V2 P50 source differs: {relative}")
    _VALIDATED_REVISION_SHA256 = supplied


def _source_revision(revision: Mapping[str, Any]) -> dict[str, Any]:
    sources = revision.get("serialized_sources")
    if not isinstance(sources, Mapping):
        raise RuntimeError("Process-V2 P50 image lacks serialized sources")
    body = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": str(revision.get("commit")),
        "tree": str(revision.get("tree")),
        "image_revision_sha256": str(revision.get("image_revision_sha256")),
        "serialized_source_inventory_sha256": _sha256(dict(sources)),
    }
    if (
        _COMMIT_RE.fullmatch(body["commit"]) is None
        or _COMMIT_RE.fullmatch(body["tree"]) is None
        or _SHA256_RE.fullmatch(body["image_revision_sha256"]) is None
    ):
        raise RuntimeError("Process-V2 P50 image revision identity is malformed")
    return {**body, "source_revision_sha256": _sha256(body)}


def _runner_implementation_sha256(revision: Mapping[str, Any]) -> str:
    sources = revision.get("serialized_sources")
    if not isinstance(sources, Mapping) or any(
        path not in sources for path in _RUNNER_IMPLEMENTATION_SOURCES
    ):
        raise RuntimeError("Process-V2 P50 runner source closure is incomplete")
    return _sha256(
        {
            "algorithm": "compose.process_v2_p50_runner_implementation.v1",
            "sources": {path: sources[path] for path in _RUNNER_IMPLEMENTATION_SOURCES},
        }
    )


def _collation_implementation_sha256(revision: Mapping[str, Any]) -> str:
    """Bind only code that can change model-ready tensor contents."""

    sources = revision.get("serialized_sources")
    if not isinstance(sources, Mapping) or any(
        path not in sources for path in _COLLATION_IMPLEMENTATION_SOURCES
    ):
        raise RuntimeError("Process-V2 P50 collation source closure is incomplete")
    return _sha256(
        {
            "algorithm": "compose.process_v2_p50_cpu_collation.v1",
            "sources": {path: sources[path] for path in _COLLATION_IMPLEMENTATION_SOURCES},
        }
    )


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
    return path


def _require_physical_artifact_path(value: str, *, field: str) -> Path:
    path = _require_artifact_path(value, field=field)
    physical_root = ARTIFACT_ROOT.resolve()
    try:
        relative = path.relative_to(ARTIFACT_ROOT)
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


def _validate_self_hash(value: Mapping[str, Any], *, field: str, label: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != field}
    if set(value) == set(body) or value.get(field) != _sha256(body):
        raise RuntimeError(f"{label} self-hash disagrees")
    return dict(value)


def _groups(values: Sequence[str], *, maximum: int) -> tuple[tuple[str, ...], ...]:
    if type(maximum) is not int or not 1 <= maximum <= MAX_CPU_CONTAINERS:
        raise ValueError(f"maximum must lie in [1, {MAX_CPU_CONTAINERS}]")
    ordered = tuple(values)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Process-V2 P50 task inventory repeats an identity")
    return tuple(
        tuple(ordered[index : index + maximum]) for index in range(0, len(ordered), maximum)
    )


def _progress(phase: str, **fields: object) -> None:
    print(
        json.dumps({"phase": phase, "time_unix_seconds": time.time(), **fields}, sort_keys=True),
        flush=True,
    )


@contextmanager
def _heartbeat(phase: str, *, interval_seconds: int = HEARTBEAT_SECONDS) -> Iterator[None]:
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


def _imports() -> dict[str, Any]:
    source_root = str(REMOTE_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    import numpy as np
    import rdkit
    import torch

    from compose_v4.data.editing_v2_process_v2_schema import authority_false_block
    from compose_v4.data.immutable_artifact import write_bytes_if_absent
    from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
    from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
        P50_RECIPE_POLICY,
        T1_CAPACITY_POLICY,
        load_process_v2_chain_artifact,
    )
    from compose_v4.experiments.editing_v2_process_v2_p50_prerequisites import (
        load_process_v2_p50_scoped_prerequisites,
    )
    from compose_v4.experiments.editing_v2_process_v2_p50_result import (
        DECISION_FILENAME,
        RESULT_FILENAME,
        build_process_v2_p50_decision,
        build_process_v2_p50_result,
        validate_process_v2_p50_decision,
        validate_process_v2_p50_result,
    )
    from compose_v4.experiments.editing_v2_process_v2_p50_runtime import (
        build_process_v2_p50_prepared_inputs,
        build_process_v2_p50_selection,
        collate_process_v2_p50_entries,
        compile_process_v2_p50_entries,
        convert_process_v2_p50_v1_leaf,
        load_process_v2_p50_inputs,
        materialize_process_v2_p50_batch,
        run_process_v2_p50,
        validate_process_v2_p50_prepared_inputs,
        validate_process_v2_p50_selection,
    )
    from compose_v4.experiments.editing_v2_process_v2_chunk_compile import (
        compile_process_v2_chunk_shard,
        publish_process_v2_chunk_shard,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_scoped_result import (
        DECISION_FILENAME as SCOPED_T1_DECISION_FILENAME,
        RESULT_FILENAME as SCOPED_T1_RESULT_FILENAME,
        build_process_v2_t1_scoped_capacity_decision,
        build_process_v2_t1_scoped_capacity_result,
        validate_process_v2_t1_scoped_capacity_decision,
        validate_process_v2_t1_scoped_capacity_result,
        validate_t1_evidence_source_manifest,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        _index_factorized_batch,
        optimizer_state_semantic_sha256,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_decision import (
        build_semantic_t1_execution_environment,
    )
    from compose_v4.experiments.factorized_mark_conditional import (
        _concatenate_factorized_mark_batches,
    )
    from compose_v4.model.factorized_tracelet_rate_model import FactorizedMarkBatch

    return {
        "np": np,
        "rdkit": rdkit,
        "torch": torch,
        "authority_false_block": authority_false_block,
        "write_bytes_if_absent": write_bytes_if_absent,
        "state_dict_sha256": state_dict_semantic_sha256,
        "P50_RECIPE_POLICY": P50_RECIPE_POLICY,
        "T1_CAPACITY_POLICY": T1_CAPACITY_POLICY,
        "load_policy": load_process_v2_chain_artifact,
        "load_prerequisites": load_process_v2_p50_scoped_prerequisites,
        "open_source": open_process_v2_t1_source,
        "build_scratch": build_process_v2_score_revised_scratch_runtime,
        "build_scoped_t1_result": build_process_v2_t1_scoped_capacity_result,
        "build_scoped_t1_decision": build_process_v2_t1_scoped_capacity_decision,
        "validate_scoped_t1_result": validate_process_v2_t1_scoped_capacity_result,
        "validate_scoped_t1_decision": validate_process_v2_t1_scoped_capacity_decision,
        "validate_t1_evidence_manifest": validate_t1_evidence_source_manifest,
        "SCOPED_T1_RESULT_FILENAME": SCOPED_T1_RESULT_FILENAME,
        "SCOPED_T1_DECISION_FILENAME": SCOPED_T1_DECISION_FILENAME,
        "build_selection": build_process_v2_p50_selection,
        "validate_selection": validate_process_v2_p50_selection,
        "compile_entries": compile_process_v2_p50_entries,
        "compile_chunk": compile_process_v2_chunk_shard,
        "publish_chunk": publish_process_v2_chunk_shard,
        "convert_v1_leaf": convert_process_v2_p50_v1_leaf,
        "build_prepared": build_process_v2_p50_prepared_inputs,
        "validate_prepared": validate_process_v2_p50_prepared_inputs,
        "collate_entries": collate_process_v2_p50_entries,
        "load_runtime": load_process_v2_p50_inputs,
        "materialize_batch": materialize_process_v2_p50_batch,
        "run_p50": run_process_v2_p50,
        "build_result": build_process_v2_p50_result,
        "validate_result": validate_process_v2_p50_result,
        "build_decision": build_process_v2_p50_decision,
        "validate_decision": validate_process_v2_p50_decision,
        "RESULT_FILENAME": RESULT_FILENAME,
        "DECISION_FILENAME": DECISION_FILENAME,
        "optimizer_state_sha256": optimizer_state_semantic_sha256,
        "build_environment": build_semantic_t1_execution_environment,
        "concatenate_batches": _concatenate_factorized_mark_batches,
        "index_batch": _index_factorized_batch,
        "FactorizedMarkBatch": FactorizedMarkBatch,
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


def _load_prerequisites(
    *,
    gate_zero_decision_path: Path,
    t1_result_path: Path,
    t1_decision_path: Path,
    loaded: Mapping[str, Any],
) -> Any:
    return loaded["load_prerequisites"](
        gate_zero_decision_path=gate_zero_decision_path,
        t1_result_path=t1_result_path,
        t1_decision_path=t1_decision_path,
        repo_root=REMOTE_ROOT,
    )


def _publish_canonical(path: Path, value: Mapping[str, Any], *, loaded: Mapping[str, Any]) -> Path:
    loaded["write_bytes_if_absent"](Path(path), _canonical_bytes(dict(value)) + b"\n")
    return Path(path)


def _leaf_path(run_root: Path, task_identity_sha256: str) -> Path:
    return Path(run_root) / TASKS_DIRNAME / task_identity_sha256 / LEAF_FILENAME


def _validate_leaf(
    value: Mapping[str, Any], *, selection_sha256: str, task_identity_sha256: str
) -> dict[str, Any]:
    leaf = _validate_self_hash(value, field="leaf_sha256", label="a Process-V2 P50 leaf")
    if (
        set(leaf)
        != {
            "task_identity_sha256",
            "selection_sha256",
            "entry_count",
            "entries",
            "leaf_sha256",
        }
        or leaf["selection_sha256"] != selection_sha256
        or leaf["task_identity_sha256"] != task_identity_sha256
        or type(leaf["entry_count"]) is not int
        or leaf["entry_count"] <= 0
        or leaf["entry_count"] != len(leaf["entries"])
    ):
        raise RuntimeError("a Process-V2 P50 leaf binding disagrees")
    return leaf


def _collated_task_root(run_root: Path, task_identity_sha256: str) -> Path:
    return Path(run_root) / COLLATED_TASKS_DIRNAME / task_identity_sha256


def _balanced_collated_tasks(
    leaves: Sequence[Mapping[str, Any]], *, maximum: int
) -> list[dict[str, Any]]:
    """Greedily balance immutable leaves by measured candidate-mark work."""

    if type(maximum) is not int or not 1 <= maximum <= MAX_CPU_CONTAINERS:
        raise ValueError(f"maximum must lie in [1, {MAX_CPU_CONTAINERS}]")
    descriptors: list[dict[str, Any]] = []
    observed_entries: set[str] = set()
    observed_leaves: set[str] = set()
    for raw in leaves:
        leaf = dict(raw)
        leaf_id = str(leaf["task_identity_sha256"])
        entry_ids = tuple(sorted(str(entry["p50_entry_sha256"]) for entry in leaf["entries"]))
        if (
            leaf_id in observed_leaves
            or not entry_ids
            or len(entry_ids) != len(set(entry_ids))
            or observed_entries.intersection(entry_ids)
        ):
            raise RuntimeError("Process-V2 P50 collated source inventory repeats an identity")
        observed_leaves.add(leaf_id)
        observed_entries.update(entry_ids)
        workload = sum(int(entry["raw_mark_count"]) for entry in leaf["entries"])
        if workload <= 0:
            raise RuntimeError("Process-V2 P50 collated source has no candidate work")
        descriptors.append(
            {
                "leaf_task_identity_sha256": leaf_id,
                "leaf_sha256": str(leaf["leaf_sha256"]),
                "entry_count": len(entry_ids),
                "p50_entry_sha256s": list(entry_ids),
                "raw_mark_count": workload,
            }
        )
    worker_count = min(maximum, len(descriptors))
    bins: list[dict[str, Any]] = [
        {"raw_mark_count": 0, "leaves": []} for _ in range(worker_count)
    ]
    for descriptor in sorted(
        descriptors,
        key=lambda item: (-int(item["raw_mark_count"]), item["leaf_task_identity_sha256"]),
    ):
        index = min(
            range(worker_count),
            key=lambda item: (
                int(bins[item]["raw_mark_count"]),
                len(bins[item]["leaves"]),
                item,
            ),
        )
        bins[index]["leaves"].append(descriptor)
        bins[index]["raw_mark_count"] += int(descriptor["raw_mark_count"])
    tasks: list[dict[str, Any]] = []
    for index, bucket in enumerate(bins):
        bucket_leaves = sorted(
            bucket["leaves"], key=lambda item: item["leaf_task_identity_sha256"]
        )
        entry_ids = sorted(
            entry_id for leaf in bucket_leaves for entry_id in leaf["p50_entry_sha256s"]
        )
        body = {
            "task_index": index,
            "leaves": bucket_leaves,
            "leaf_count": len(bucket_leaves),
            "entry_count": len(entry_ids),
            "p50_entry_sha256s": entry_ids,
            "entry_inventory_sha256": _sha256(entry_ids),
            "raw_mark_count": int(bucket["raw_mark_count"]),
        }
        tasks.append({**body, "task_identity_sha256": _sha256(body)})
    return tasks


def _require_false_authority(
    value: Mapping[str, Any],
    *,
    label: str,
    authority_false: Mapping[str, bool],
) -> None:
    fields = {key for key in value if key.endswith("_authorized")}
    expected = set(authority_false)
    if (
        any(item is not False for item in authority_false.values())
        or fields != expected
        or any(value[field] is not False for field in fields)
    ):
        raise RuntimeError(f"{label} grants or omits authority")


def _validate_collated_plan(
    value: Mapping[str, Any],
    *,
    authority_false: Mapping[str, bool],
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = _validate_self_hash(
        value, field="plan_sha256", label="the Process-V2 P50 collated plan"
    )
    _require_false_authority(
        plan,
        label="the Process-V2 P50 collated plan",
        authority_false=authority_false,
    )
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false,
        "prepared_path",
        "prepared_file_sha256",
        "prepared_sha256",
        "selection_sha256",
        "initial_model_state_sha256",
        "collation_implementation_sha256",
        "source_revision_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "task_count",
        "task_inventory_sha256",
        "tasks",
        "run_identity_sha256",
        "plan_sha256",
    }
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks or any(
        not isinstance(task, Mapping) for task in tasks
    ):
        raise RuntimeError("Process-V2 P50 collated plan tasks are malformed")
    expected_ids = (
        tuple(sorted(str(entry["p50_entry_sha256"]) for entry in prepared["entries"]))
        if prepared is not None
        else tuple(
            sorted(
                str(identifier)
                for task in tasks or ()
                for identifier in task.get("p50_entry_sha256s", ())
            )
        )
    )
    if (
        set(plan) != expected_fields
        or plan.get("schema") != COLLATED_PLAN_SCHEMA
        or plan.get("schema_version") != COLLATED_SCHEMA_VERSION
        or plan.get("status") != "PLANNED_CPU_COLLATION_NO_DOWNSTREAM_AUTHORITY"
        or (
            prepared is not None
            and (
                plan.get("prepared_sha256") != prepared["prepared_sha256"]
                or plan.get("selection_sha256") != prepared["selection_sha256"]
                or plan.get("initial_model_state_sha256")
                != prepared["initial_model_state_sha256"]
            )
        )
        or plan.get("task_count") != len(tasks)
        or plan.get("task_inventory_sha256") != _sha256(tasks)
        or plan.get("entry_count") != len(expected_ids)
        or plan.get("entry_inventory_sha256") != _sha256(expected_ids)
    ):
        raise RuntimeError("Process-V2 P50 collated plan identity disagrees")
    observed_ids: list[str] = []
    observed_leaves: set[str] = set()
    for index, raw_task in enumerate(tasks):
        if not isinstance(raw_task, Mapping):
            raise RuntimeError("Process-V2 P50 collated task is malformed")
        task = dict(raw_task)
        body = {key: item for key, item in task.items() if key != "task_identity_sha256"}
        leaves = task.get("leaves")
        entry_ids = task.get("p50_entry_sha256s")
        if (
            set(task)
            != {
                "task_index",
                "leaves",
                "leaf_count",
                "entry_count",
                "p50_entry_sha256s",
                "entry_inventory_sha256",
                "raw_mark_count",
                "task_identity_sha256",
            }
            or task.get("task_index") != index
            or task.get("task_identity_sha256") != _sha256(body)
            or not isinstance(leaves, list)
            or not isinstance(entry_ids, list)
            or entry_ids != sorted(entry_ids)
            or task.get("leaf_count") != len(leaves)
            or task.get("entry_count") != len(entry_ids)
            or task.get("entry_inventory_sha256") != _sha256(entry_ids)
            or task.get("raw_mark_count")
            != sum(int(leaf["raw_mark_count"]) for leaf in leaves)
        ):
            raise RuntimeError("Process-V2 P50 collated task identity disagrees")
        for leaf in leaves:
            if (
                not isinstance(leaf, Mapping)
                or set(leaf)
                != {
                    "leaf_task_identity_sha256",
                    "leaf_sha256",
                    "entry_count",
                    "p50_entry_sha256s",
                    "raw_mark_count",
                }
                or _SHA256_RE.fullmatch(str(leaf["leaf_task_identity_sha256"])) is None
                or _SHA256_RE.fullmatch(str(leaf["leaf_sha256"])) is None
                or leaf["entry_count"] != len(leaf["p50_entry_sha256s"])
                or leaf["p50_entry_sha256s"] != sorted(leaf["p50_entry_sha256s"])
                or any(
                    _SHA256_RE.fullmatch(str(identifier)) is None
                    for identifier in leaf["p50_entry_sha256s"]
                )
                or type(leaf["raw_mark_count"]) is not int
                or leaf["raw_mark_count"] <= 0
            ):
                raise RuntimeError("Process-V2 P50 collated source leaf is malformed")
            leaf_id = str(leaf["leaf_task_identity_sha256"])
            if leaf_id in observed_leaves:
                raise RuntimeError("Process-V2 P50 collated plan repeats a source leaf")
            observed_leaves.add(leaf_id)
        observed_ids.extend(str(item) for item in entry_ids)
    if tuple(sorted(observed_ids)) != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("Process-V2 P50 collated plan changed the entry inventory")
    identity = {
        "prepared_file_sha256": plan["prepared_file_sha256"],
        "prepared_sha256": plan["prepared_sha256"],
        "collation_implementation_sha256": plan["collation_implementation_sha256"],
        "task_inventory_sha256": plan["task_inventory_sha256"],
    }
    if plan.get("run_identity_sha256") != _sha256(identity):
        raise RuntimeError("Process-V2 P50 collated run identity disagrees")
    return plan


def _collated_task(plan: Mapping[str, Any], task_identity_sha256: str) -> dict[str, Any]:
    matches = [
        dict(task)
        for task in plan["tasks"]
        if task["task_identity_sha256"] == task_identity_sha256
    ]
    if len(matches) != 1:
        raise RuntimeError("Process-V2 P50 collated task is absent or repeated")
    return matches[0]


def _validate_collated_receipt(
    value: Mapping[str, Any],
    *,
    authority_false: Mapping[str, bool],
    plan: Mapping[str, Any],
    task_identity_sha256: str,
) -> dict[str, Any]:
    receipt = _validate_self_hash(
        value, field="receipt_sha256", label="a Process-V2 P50 collated receipt"
    )
    _require_false_authority(
        receipt,
        label="a Process-V2 P50 collated receipt",
        authority_false=authority_false,
    )
    task = _collated_task(plan, task_identity_sha256)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false,
        "plan_sha256",
        "run_identity_sha256",
        "prepared_sha256",
        "collation_implementation_sha256",
        "task_identity_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "payload_filename",
        "payload_file_sha256",
        "payload_file_bytes",
        "receipt_sha256",
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != COLLATED_RECEIPT_SCHEMA
        or receipt.get("schema_version") != COLLATED_SCHEMA_VERSION
        or receipt.get("status") != "COMPLETE_CPU_COLLATION_NO_DOWNSTREAM_AUTHORITY"
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("run_identity_sha256") != plan["run_identity_sha256"]
        or receipt.get("prepared_sha256") != plan["prepared_sha256"]
        or receipt.get("collation_implementation_sha256")
        != plan["collation_implementation_sha256"]
        or receipt.get("task_identity_sha256") != task_identity_sha256
        or receipt.get("entry_count") != task["entry_count"]
        or receipt.get("entry_inventory_sha256") != task["entry_inventory_sha256"]
        or receipt.get("payload_filename") != COLLATED_PAYLOAD_FILENAME
        or type(receipt.get("payload_file_bytes")) is not int
        or receipt["payload_file_bytes"] <= 0
        or _SHA256_RE.fullmatch(str(receipt.get("payload_file_sha256"))) is None
    ):
        raise RuntimeError("Process-V2 P50 collated receipt identity disagrees")
    return receipt


def _load_collated_payload(
    receipt_path: Path,
    *,
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    loaded: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    receipt = _validate_collated_receipt(
        _read_canonical_object(receipt_path, label="a Process-V2 P50 collated receipt"),
        authority_false=loaded["authority_false_block"](),
        plan=plan,
        task_identity_sha256=task_identity_sha256,
    )
    payload_path = Path(receipt_path).parent / COLLATED_PAYLOAD_FILENAME
    if (
        not payload_path.is_file()
        or payload_path.stat().st_size != receipt["payload_file_bytes"]
        or _file_sha256(payload_path) != receipt["payload_file_sha256"]
    ):
        raise RuntimeError("Process-V2 P50 collated payload bytes disagree")
    payload = loaded["torch"].load(
        payload_path, map_location="cpu", weights_only=False, mmap=True
    )
    task = _collated_task(plan, task_identity_sha256)
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {
            "schema",
            "schema_version",
            "task_identity_sha256",
            "p50_entry_sha256s",
            "batch",
        }
        or payload.get("schema") != COLLATED_PAYLOAD_SCHEMA
        or payload.get("schema_version") != COLLATED_SCHEMA_VERSION
        or payload.get("task_identity_sha256") != task_identity_sha256
        or payload.get("p50_entry_sha256s") != task["p50_entry_sha256s"]
        or not isinstance(payload.get("batch"), loaded["FactorizedMarkBatch"])
        or payload["batch"].batch_size != task["entry_count"]
    ):
        raise RuntimeError("Process-V2 P50 collated payload identity disagrees")
    return receipt, payload["batch"]


def _load_collated_completion(
    completion_path: Path,
    *,
    plan: Mapping[str, Any],
    prepared: Mapping[str, Any],
    loaded: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, list[str]]:
    completion = _validate_self_hash(
        _read_canonical_object(
            completion_path, label="the Process-V2 P50 collated completion"
        ),
        field="completion_sha256",
        label="the Process-V2 P50 collated completion",
    )
    authority_false = loaded["authority_false_block"]()
    _require_false_authority(
        completion,
        label="the Process-V2 P50 collated completion",
        authority_false=authority_false,
    )
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false,
        "plan_sha256",
        "run_identity_sha256",
        "prepared_path",
        "prepared_file_sha256",
        "prepared_sha256",
        "collation_implementation_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "task_count",
        "task_receipt_inventory_sha256",
        "task_receipts",
        "payload_filename",
        "payload_file_sha256",
        "payload_file_bytes",
        "gpu_training_launched",
        "completion_sha256",
    }
    summaries = completion.get("task_receipts")
    if (
        not isinstance(summaries, list)
        or len(summaries) != len(plan["tasks"])
        or any(
            not isinstance(summary, Mapping)
            or set(summary)
            != {
                "task_identity_sha256",
                "receipt_sha256",
                "payload_file_sha256",
            }
            or summary["task_identity_sha256"] != task["task_identity_sha256"]
            or _SHA256_RE.fullmatch(str(summary["receipt_sha256"])) is None
            or _SHA256_RE.fullmatch(str(summary["payload_file_sha256"])) is None
            for task, summary in zip(plan["tasks"], summaries, strict=True)
        )
    ):
        raise RuntimeError("Process-V2 P50 collated completion receipts disagree")
    if (
        set(completion) != expected_fields
        or completion.get("schema") != COLLATED_COMPLETION_SCHEMA
        or completion.get("schema_version") != COLLATED_SCHEMA_VERSION
        or completion.get("status") != "COMPLETE_CPU_COLLATION_NO_DOWNSTREAM_AUTHORITY"
        or completion.get("plan_sha256") != plan["plan_sha256"]
        or completion.get("run_identity_sha256") != plan["run_identity_sha256"]
        or completion.get("prepared_path") != plan["prepared_path"]
        or completion.get("prepared_file_sha256") != plan["prepared_file_sha256"]
        or completion.get("prepared_sha256") != prepared["prepared_sha256"]
        or completion.get("collation_implementation_sha256")
        != plan["collation_implementation_sha256"]
        or completion.get("entry_count") != prepared["entry_count"]
        or completion.get("entry_inventory_sha256") != plan["entry_inventory_sha256"]
        or completion.get("task_count") != plan["task_count"]
        or completion.get("task_receipt_inventory_sha256")
        != _sha256(summaries)
        or completion.get("payload_filename") != COLLATED_PAYLOAD_FILENAME
        or completion.get("gpu_training_launched") is not False
    ):
        raise RuntimeError("Process-V2 P50 collated completion identity disagrees")
    payload_path = Path(completion_path).parent / COLLATED_PAYLOAD_FILENAME
    if (
        not payload_path.is_file()
        or payload_path.stat().st_size != completion["payload_file_bytes"]
        or _file_sha256(payload_path) != completion["payload_file_sha256"]
    ):
        raise RuntimeError("Process-V2 P50 consolidated payload bytes disagree")
    payload = loaded["torch"].load(
        payload_path, map_location="cpu", weights_only=False, mmap=True
    )
    expected_ids = sorted(str(entry["p50_entry_sha256"]) for entry in prepared["entries"])
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {
            "schema",
            "schema_version",
            "task_identity_sha256",
            "p50_entry_sha256s",
            "batch",
        }
        or payload.get("schema") != COLLATED_PAYLOAD_SCHEMA
        or payload.get("schema_version") != COLLATED_SCHEMA_VERSION
        or payload.get("task_identity_sha256") != plan["run_identity_sha256"]
        or payload.get("p50_entry_sha256s") != expected_ids
        or not isinstance(payload.get("batch"), loaded["FactorizedMarkBatch"])
        or payload["batch"].batch_size != len(expected_ids)
    ):
        raise RuntimeError("Process-V2 P50 consolidated payload identity disagrees")
    return completion, payload["batch"], expected_ids


def _require_prepared_prerequisite_binding(
    prepared: Mapping[str, Any], *, prerequisites: Any
) -> None:
    if (
        prepared.get("prerequisites_binding_sha256") != prerequisites.binding_sha256
        or prepared.get("initial_model_state_sha256") != prerequisites.t1_initial_model_state_sha256
    ):
        raise RuntimeError("Process-V2 P50 prepared prerequisite binding disagrees")


def _require_selection_prerequisite_binding(
    selection: Mapping[str, Any], *, prerequisites: Any
) -> None:
    if (
        selection.get("prerequisites_binding_sha256") != prerequisites.binding_sha256
        or selection.get("prerequisites") != prerequisites.as_payload()
    ):
        raise RuntimeError("Process-V2 P50 selection prerequisite binding disagrees")


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


def _torch_bytes(value: object, *, torch_module: Any) -> bytes:
    output = io.BytesIO()
    torch_module.save(value, output)
    return output.getvalue()


def _checkpoint_payload(
    run: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    provenance: Mapping[str, Any],
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer_sha = loaded["optimizer_state_sha256"](run["optimizer_state"])
    body = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": CHECKPOINT_STATUS,
        **loaded["authority_false_block"](),
        "completed_optimizer_steps": int(run["optimizer_steps_completed"]),
        "resume_count": 0,
        "configuration": dict(policy),
        "provenance": dict(provenance),
        "model_state_sha256": str(run["final_model_state_sha256"]),
        "optimizer_state_sha256": optimizer_sha,
        "hazard_initial_state_sha256": str(run["hazard_initial_state_sha256"]),
        "hazard_final_state_sha256": str(run["hazard_final_state_sha256"]),
    }
    return {
        **body,
        "model_state": run["final_model_state"],
        "optimizer_state": run["optimizer_state"],
        "rng_state": run["rng_state"],
        "checkpoint_sha256": _sha256(body),
    }


def _validate_checkpoint_payload(
    value: Mapping[str, Any], *, loaded: Mapping[str, Any]
) -> dict[str, Any]:
    payload = dict(value)
    tensor_fields = {"model_state", "optimizer_state", "rng_state", "checkpoint_sha256"}
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *loaded["authority_false_block"](),
        "completed_optimizer_steps",
        "resume_count",
        "configuration",
        "provenance",
        "model_state_sha256",
        "optimizer_state_sha256",
        "hazard_initial_state_sha256",
        "hazard_final_state_sha256",
        *tensor_fields,
    }
    body = {key: item for key, item in payload.items() if key not in tensor_fields}
    if (
        set(payload) != expected_fields
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("status") != CHECKPOINT_STATUS
        or payload.get("completed_optimizer_steps") != 50
        or payload.get("resume_count") != 0
        or payload.get("checkpoint_sha256") != _sha256(body)
        or payload.get("model_state_sha256") != loaded["state_dict_sha256"](payload["model_state"])
        or payload.get("optimizer_state_sha256")
        != loaded["optimizer_state_sha256"](payload["optimizer_state"])
        or payload.get("hazard_initial_state_sha256") != payload.get("hazard_final_state_sha256")
        or not isinstance(payload.get("configuration"), Mapping)
        or not isinstance(payload.get("provenance"), Mapping)
        or not isinstance(payload.get("rng_state"), Mapping)
    ):
        raise RuntimeError("Process-V2 P50 terminal checkpoint disagrees")
    return payload


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def materialize_scoped_t1_remote(
    active8_run_root: str,
    gate_zero_decision_path: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Compose existing per-family receipts and publish the P50 GO decision."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    manifest = loaded["validate_t1_evidence_manifest"](
        _read_canonical_object(
            REMOTE_ROOT / T1_EVIDENCE_SOURCE,
            label="the Process-V2 P50 T1 evidence source manifest",
        )
    )
    source = _open_source(
        active8_run_root=_require_artifact_path(active8_run_root, field="active8_run_root"),
        gate_zero_decision_path=_require_artifact_path(
            gate_zero_decision_path, field="gate_zero_decision_path"
        ),
        loaded=loaded,
    )
    _scratch, _binding, score_receipt = loaded["build_scratch"](source)
    capacity_policy = loaded["load_policy"](loaded["T1_CAPACITY_POLICY"], repo_root=REMOTE_ROOT)
    sources: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for name, descriptor in manifest["input_sources"].items():
        path = _require_artifact_path(str(descriptor["path"]), field=f"{name}_path")
        sources[name] = _read_canonical_object(path, label=f"the {name} T1 receipt")
        source_hashes[name] = _file_sha256(path)
    result = loaded["build_scoped_t1_result"](
        source_manifest=manifest,
        source_file_sha256s=source_hashes,
        base_result=sources.pop("base"),
        repair_results=sources,
        capacity_policy=capacity_policy,
        score_revision_receipt=score_receipt,
        current_process_identity_sha256=source.contracts.process_identity_sha256,
        current_active8_completion_sha256=source.index.active8_completion_sha256,
        current_gate_zero_decision_sha256=source.decision["decision_sha256"],
    )
    result = loaded["validate_scoped_t1_result"](result)
    run_identity = _sha256(
        {
            "schema": "compose.editing_v2.process_v2_t1_scoped_capacity_address",
            "schema_version": 1,
            "source_manifest_sha256": manifest["manifest_sha256"],
            "capacity_policy_sha256": capacity_policy["contract_sha256"],
            "score_revision_receipt_sha256": score_receipt["receipt_sha256"],
            "result_sha256": result["result_sha256"],
        }
    )
    run_root = _require_physical_artifact_path(output_prefix, field="output_prefix") / run_identity
    result_path = run_root / loaded["SCOPED_T1_RESULT_FILENAME"]
    decision_path = run_root / loaded["SCOPED_T1_DECISION_FILENAME"]
    _publish_canonical(result_path, result, loaded=loaded)
    result_file_sha256 = _file_sha256(result_path)
    decision = loaded["build_scoped_t1_decision"](result, result_file_sha256=result_file_sha256)
    _publish_canonical(decision_path, decision, loaded=loaded)
    decision = loaded["validate_scoped_t1_decision"](
        _read_canonical_object(decision_path, label="the scoped T1 decision"),
        result=result,
        result_file_sha256=result_file_sha256,
        require_p50_go=True,
    )
    artifact_volume.commit()
    response = {
        "phase": "process_v2_t1_scoped_capacity_complete",
        "run_root": str(run_root),
        "result_path": str(result_path),
        "result_sha256": result["result_sha256"],
        "decision_path": str(decision_path),
        "decision_sha256": decision["decision_sha256"],
        "bounded_p50_authorized": decision["bounded_p50_authorized"],
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_selection_remote(
    active8_run_root: str,
    gate_zero_decision_path: str,
    t1_result_path: str,
    t1_decision_path: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Scan authenticated train/validation metadata once and freeze addresses."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    active8_root = _require_artifact_path(active8_run_root, field="active8_run_root")
    gate_path = _require_artifact_path(gate_zero_decision_path, field="gate_zero_decision_path")
    prerequisites = _load_prerequisites(
        gate_zero_decision_path=gate_path,
        t1_result_path=_require_artifact_path(t1_result_path, field="t1_result_path"),
        t1_decision_path=_require_artifact_path(t1_decision_path, field="t1_decision_path"),
        loaded=loaded,
    )
    source_revision = _source_revision(revision)
    run_identity = _sha256(
        {
            "schema": "compose.editing_v2.process_v2_p50_prepared_run_address",
            "schema_version": 1,
            "prerequisites_binding_sha256": prerequisites.binding_sha256,
            "source_revision_sha256": source_revision["source_revision_sha256"],
        }
    )
    run_root = _require_physical_artifact_path(output_prefix, field="output_prefix") / run_identity
    selection_path = run_root / SELECTION_FILENAME
    selection_cache_hit = selection_path.is_file()
    if selection_cache_hit:
        selection = loaded["validate_selection"](
            _read_canonical_object(selection_path, label="the cached Process-V2 P50 selection")
        )
        _require_selection_prerequisite_binding(selection, prerequisites=prerequisites)
    else:
        source = _open_source(
            active8_run_root=active8_root,
            gate_zero_decision_path=gate_path,
            loaded=loaded,
        )
        with _heartbeat("process_v2_p50_metadata_selection"):
            selection = loaded["build_selection"](source, prerequisites=prerequisites)
        selection = loaded["validate_selection"](selection)
        _require_selection_prerequisite_binding(selection, prerequisites=prerequisites)
        _publish_canonical(selection_path, selection, loaded=loaded)
        artifact_volume.commit()
    task_ids = sorted({str(entry["task_identity_sha256"]) for entry in selection["entries"]})
    task_entry_counts = {
        task_id: sum(entry["task_identity_sha256"] == task_id for entry in selection["entries"])
        for task_id in task_ids
    }
    response = {
        "phase": "process_v2_p50_selection_complete",
        "run_root": str(run_root),
        "selection_path": str(selection_path),
        "selection_sha256": selection["selection_sha256"],
        "selection_cache_hit": selection_cache_hit,
        "task_identity_sha256s": task_ids,
        "task_entry_counts": task_entry_counts,
        "task_count": len(task_ids),
        "unique_entry_count": int(selection["unique_entry_count"]),
        "source_revision": source_revision,
        "p50_training_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=CPU_PER_LEAF,
    memory=CPU_MEMORY_MB,
    timeout=CPU_LEAF_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def chunk_pilot_remote(
    active8_run_root: str,
    gate_zero_decision_path: str,
    task_identity_sha256: str,
    limit: int,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Compile a bounded slice of ONE chunk selection-free, and time it.

    This publishes nothing and trains nothing.  It exists to replace a
    projection with a measurement before the full corpus pass is paid for: the
    fused compile shares one chemistry pass with collation, and a real chunk is
    the only place the within-chunk source reuse can actually be observed.
    """

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    source = _open_source(
        active8_run_root=_require_artifact_path(active8_run_root, field="active8_run_root"),
        gate_zero_decision_path=_require_artifact_path(
            gate_zero_decision_path, field="gate_zero_decision_path"
        ),
        loaded=loaded,
    )
    scratch, _binding, _receipt = loaded["build_scratch"](source)
    started = time.monotonic()
    result = loaded["compile_chunk"](
        source,
        scratch.model,
        task_identity_sha256=task_identity_sha256,
        limit=int(limit),
    )
    elapsed = time.monotonic() - started
    entries = int(result["entry_count"])
    payload = {
        "phase": "process_v2_chunk_pilot_complete",
        "task_identity_sha256": task_identity_sha256,
        "partition_role": result["partition_role"],
        "entry_count": entries,
        "elapsed_seconds": elapsed,
        "ms_per_entry": 1000.0 * elapsed / max(entries, 1),
        "chemistry_states_cached": int(result["chemistry_states_cached"]),
        "distinct_source_fraction": (
            int(result["chemistry_states_cached"]) / entries if entries else None
        ),
        "family_counts": result["family_counts"],
        "capability_cell_counts": result["capability_cell_counts"],
        "p50_training_launched": False,
        "artifact_published": False,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


#: Set by the parent before forking; children inherit it copy-on-write.  A
#: module global rather than an argument because fork inheritance is the point:
#: the opened source and the built model must NOT be pickled to the children.
_FORK_STATE: dict[str, Any] = {}


def _compile_subrange_child(sub_offset: int, sub_limit: int) -> dict[str, Any]:
    """Compile and publish ONE sub-range in a forked child.

    Each child publishes its own offset-addressed slice, so the parent never
    merges tensors across processes and a child that stops on its deadline
    degrades exactly the way a whole slice would.
    """

    state = _FORK_STATE
    loaded = state["loaded"]
    started = time.monotonic()
    result = loaded["compile_chunk"](
        state["source"],
        state["model"],
        task_identity_sha256=state["task_identity_sha256"],
        offset=int(sub_offset),
        limit=int(sub_limit),
        deadline_seconds=float(state["deadline_seconds"]),
        progress_callback=None,
    )
    receipt = loaded["publish_chunk"](result, output_root=Path(state["output_root"]))
    elapsed = time.monotonic() - started
    payload = {
        "entry_offset": int(sub_offset),
        "requested_entry_count": int(sub_limit),
        "entry_count": int(receipt["entry_count"]),
        "stopped_on_deadline": bool(result["stopped_on_deadline"]),
        "next_offset": int(result["next_offset"]),
        "unfinished_entry_count": int(result["unfinished_entry_count"]),
        "elapsed_seconds": elapsed,
        "ms_per_entry": 1000.0 * elapsed / max(int(receipt["entry_count"]), 1),
        "receipt_sha256": receipt["receipt_sha256"],
    }
    print(
        json.dumps({"phase": "process_v2_prep_subrange_complete", **payload}, sort_keys=True),
        flush=True,
    )
    return payload


@app.function(
    image=image,
    cpu=PREP_CPU_PER_LEAF,
    memory=PREP_MEMORY_MB,
    timeout=CPU_LEAF_TIMEOUT_SECONDS,
    max_containers=MAX_CPU_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prep_slice_remote(
    active8_run_root: str,
    gate_zero_decision_path: str,
    output_root: str,
    task_identity_sha256: str,
    offset: int,
    limit: int,
    revision: dict[str, Any],
    compile_workers: int = 1,
) -> dict[str, Any]:
    """Compile ONE slice of one chunk and publish it. Trains nothing."""

    # FIRST statement, so every term inside the worker wall is measured.  The
    # wall covers the shard reads and the scratch build as well as the compile,
    # and a budget derived from a compile-only rate silently omits them -- which
    # is how an 800-entry slice sized at "27 of 45 minutes" hit the 2,700s wall.
    entered = time.monotonic()

    def _emit(event: dict[str, Any]) -> None:
        # Printed as it happens, not accumulated and returned.  A worker that
        # reports only on completion tells you nothing while it is dying, and a
        # slice that is about to blow the wall is exactly when you need to know.
        print(
            json.dumps(
                {
                    **event,
                    "task_identity_sha256": task_identity_sha256,
                    "entry_offset": int(offset),
                    "since_entry_seconds": time.monotonic() - entered,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # Emitted BEFORE any work, so an operator can tell "still starting up" from
    # "never started" without waiting for the first compile sub-batch.
    _emit({"phase": "process_v2_prep_slice_entered", "limit": int(limit)})
    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    _emit({"phase": "process_v2_prep_slice_imports_ready"})

    # PROVE the output is writable before paying for the compile.  Publication
    # runs last, so a write-time fault -- a symlinked component, a bad root, a
    # permission problem -- surfaces only after the full slice has been
    # computed, and every container discards its work at once.  This exercises
    # the real writer on the real path, so it fails in seconds instead of in
    # tens of minutes.
    preflight_root = _require_physical_artifact_path(output_root, field="output_root")
    try:
        loaded["write_bytes_if_absent"](
            preflight_root / PREP_PREFLIGHT_FILENAME,
            _canonical_bytes(
                {
                    "schema": "compose.editing_v2.process_v2_prep_writability_preflight",
                    "schema_version": 1,
                    "status": "PROCESS_V2_PREP_PREFLIGHT_EVIDENCE_ONLY_NO_AUTHORITY",
                    **loaded["authority_false_block"](),
                }
            )
            + b"\n",
        )
    except Exception as error:  # noqa: BLE001 - surface the real cause remotely
        raise RuntimeError(
            f"prep output root is not writable before any work was done: "
            f"{preflight_root}: {type(error).__name__}: {error}"
        ) from error
    _emit({"phase": "process_v2_prep_slice_output_writable"})
    source = _open_source(
        active8_run_root=_require_artifact_path(active8_run_root, field="active8_run_root"),
        gate_zero_decision_path=_require_artifact_path(
            gate_zero_decision_path, field="gate_zero_decision_path"
        ),
        loaded=loaded,
    )
    _emit({"phase": "process_v2_prep_slice_source_open"})
    scratch, _binding, _receipt = loaded["build_scratch"](source)
    started = time.monotonic()
    # The number every future sizing decision needs, reported the moment it is
    # known rather than only on success.
    _emit(
        {
            "phase": "process_v2_prep_slice_setup_complete",
            "setup_seconds": started - entered,
            "worker_timeout_seconds": CPU_LEAF_TIMEOUT_SECONDS,
        }
    )
    _progress = _emit

    # Everything already spent on setup comes out of the wall, and a margin is
    # held back for collation, publication and the volume commit that still
    # follow the compile.
    deadline = (
        CPU_LEAF_TIMEOUT_SECONDS - (started - entered) - PREP_SLICE_RESERVE_SECONDS
    )
    if deadline <= 0:
        raise RuntimeError(
            f"setup consumed {started - entered:.0f}s of the "
            f"{CPU_LEAF_TIMEOUT_SECONDS}s worker wall, leaving no compile budget"
        )
    # Fan the slice across processes INSIDE the container.  Modal bills
    # core-seconds, so N cores in one container cost the same as N containers
    # -- but the container cap is a tier limit and the core count is not, which
    # makes this the only lever that buys wall-clock without buying compute.
    # The fork happens AFTER setup so the ~110s source open and scratch build
    # are paid once and inherited copy-on-write, and each child publishes its
    # OWN offset-addressed slice, so nothing has to be merged back.
    workers = max(1, int(compile_workers))
    if workers > 1 and int(limit) > workers:
        global _FORK_STATE
        _FORK_STATE = {
            "source": source,
            "model": scratch.model,
            "loaded": loaded,
            "task_identity_sha256": task_identity_sha256,
            "output_root": str(preflight_root),
            "deadline_seconds": deadline,
        }
        span = int(limit) // workers
        ranges = [
            (int(offset) + index * span, span if index < workers - 1 else int(limit) - span * (workers - 1))
            for index in range(workers)
        ]
        _emit(
            {
                "phase": "process_v2_prep_slice_fanned",
                "compile_workers": workers,
                "sub_ranges": ranges,
            }
        )
        context = multiprocessing.get_context("fork")
        with context.Pool(processes=workers) as pool:
            children = pool.starmap(_compile_subrange_child, ranges)
        entries_published = sum(int(row["entry_count"]) for row in children)
        payload = {
            "phase": "process_v2_prep_slice_complete",
            "task_identity_sha256": task_identity_sha256,
            "entry_offset": int(offset),
            "entry_count": entries_published,
            "requested_entry_count": int(limit),
            "compile_workers": workers,
            "sub_slices": children,
            "stopped_on_deadline": any(bool(row["stopped_on_deadline"]) for row in children),
            "next_offset": int(offset) + entries_published,
            "unfinished_entry_count": int(limit) - entries_published,
            "elapsed_seconds": time.monotonic() - started,
            "setup_seconds": started - entered,
            "worker_seconds": time.monotonic() - entered,
            "worker_timeout_seconds": CPU_LEAF_TIMEOUT_SECONDS,
            "ms_per_entry_wall": (
                1000.0 * (time.monotonic() - started) / max(entries_published, 1)
            ),
            "p50_training_launched": False,
        }
        artifact_volume.commit()
        print(json.dumps(payload, sort_keys=True), flush=True)
        return payload

    result = loaded["compile_chunk"](
        source,
        scratch.model,
        task_identity_sha256=task_identity_sha256,
        offset=int(offset),
        limit=int(limit),
        deadline_seconds=deadline,
        progress_callback=_progress,
    )
    receipt = loaded["publish_chunk"](
        result,
        # PHYSICAL, not logical.  `/artifacts` is a symlink in the container and
        # the immutable-artifact writer refuses any path with a symlinked
        # component, so a write target resolved the read way fails AFTER the
        # whole compile has been paid for.  Reads take the logical path; writes
        # take the resolved one, as every other publishing site here does.
        output_root=_require_physical_artifact_path(output_root, field="output_root"),
    )
    artifact_volume.commit()
    payload = {
        "phase": "process_v2_prep_slice_complete",
        "task_identity_sha256": task_identity_sha256,
        "entry_offset": int(offset),
        "entry_count": int(receipt["entry_count"]),
        "requested_entry_count": int(result["requested_entry_count"]),
        "stopped_on_deadline": bool(result["stopped_on_deadline"]),
        "next_offset": int(result["next_offset"]),
        "unfinished_entry_count": int(result["unfinished_entry_count"]),
        "elapsed_seconds": time.monotonic() - started,
        "setup_seconds": started - entered,
        "worker_seconds": time.monotonic() - entered,
        "worker_timeout_seconds": CPU_LEAF_TIMEOUT_SECONDS,
        "receipt_sha256": receipt["receipt_sha256"],
        "batch_bytes": int(receipt["batch_bytes"]),
        "p50_training_launched": False,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


@app.local_entrypoint()
def prep_subset(
    active8_run_root: str,
    gate_zero_decision_path: str,
    output_root: str,
    expected_commit: str,
    chunk_plan_path: str,
    slice_size: int = 300,
    max_slices: int = 0,
    compile_workers: int = PREP_COMPILE_WORKERS,
) -> None:
    """Fan a pinned chunk list out as EQUAL slices, one worker per slice.

    The unit is a slice rather than a chunk because chunk sizes differ by 4.5x
    here, and one-container-per-chunk floors the wall clock at the largest chunk
    however many containers run -- the straggler that made the earlier bounded
    pilot slow.

    ``slice_size`` defaults to 300 rather than 800 because 800 DID time out.  It
    was sized from a compile-only rate of 2,046 ms/entry measured on a single
    chunk, which put a slice at 27 of the 45 available minutes -- but the worker
    timeout also covers source open and scratch build, and the rate itself
    varies with a chunk's distinct-source fraction.  300 leaves margin for both
    at the cost of paying setup more often; the ``setup_seconds`` a slice now
    reports is what should size this properly, so the next run measures the term
    that was missing instead of guessing again.
    """

    revision = local_image_revision(expected_commit=expected_commit)
    plan = json.loads(Path(chunk_plan_path).read_text())
    work: list[tuple[Any, ...]] = []
    for entry in plan["chunks"]:
        total = int(entry["transitions"])
        for offset in range(0, total, int(slice_size)):
            work.append(
                (
                    active8_run_root,
                    gate_zero_decision_path,
                    output_root,
                    str(entry["task_identity_sha256"]),
                    offset,
                    min(int(slice_size), total - offset),
                    revision,
                    int(compile_workers),
                )
            )
    if int(max_slices) > 0:
        work = work[: int(max_slices)]
    print(
        json.dumps(
            {
                "phase": "process_v2_prep_subset_planned",
                "chunks": len(plan["chunks"]),
                "slices": len(work),
                "slice_size": int(slice_size),
                "planned_transitions": sum(int(c["transitions"]) for c in plan["chunks"]),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    published = 0
    entries = 0
    failures: list[dict[str, Any]] = []
    unfinished: list[dict[str, Any]] = []
    slowest = 0.0
    # `return_exceptions` so ONE bad slice costs one slice, not the run.  The
    # default raises on the first failure and abandons every slice still in
    # flight, even though each worker commits its own output on success -- so a
    # single 45-minute timeout used to discard an otherwise complete fan-out.
    # Re-running is safe: publication is write-if-absent, so a slice that
    # already landed is a no-op and only the failures are recomputed.
    for index, result in enumerate(
        prep_slice_remote.starmap(work, return_exceptions=True)
    ):
        if isinstance(result, Exception):
            failures.append(
                {
                    "task_identity_sha256": str(work[index][3]),
                    "entry_offset": int(work[index][4]),
                    "limit": int(work[index][5]),
                    "error": f"{type(result).__name__}: {result}"[:200],
                }
            )
            continue
        published += 1
        entries += int(result["entry_count"])
        slowest = max(slowest, float(result.get("worker_seconds", 0.0)))
        if int(result.get("unfinished_entry_count", 0)):
            # A slice that stopped on its deadline published what it finished.
            # The remainder is named rather than dropped, so re-running is a
            # bounded top-up instead of a rerun of everything.
            unfinished.append(
                {
                    "task_identity_sha256": str(result["task_identity_sha256"]),
                    "entry_offset": int(result["next_offset"]),
                    "limit": int(result["unfinished_entry_count"]),
                    "reason": "stopped_on_deadline",
                }
            )
    summary = {
        "phase": "process_v2_prep_subset_complete",
        "slices_planned": len(work),
        "slices_published": published,
        "slices_failed": len(failures),
        "failures": failures,
        "slices_short": len(unfinished),
        "unfinished": unfinished,
        "entries_published": entries,
        "entries_unfinished": sum(int(row["limit"]) for row in unfinished),
        "slowest_worker_seconds": slowest,
        "worker_timeout_seconds": CPU_LEAF_TIMEOUT_SECONDS,
        # Loud, because a partial fan-out that reads as complete is how a
        # training set silently ends up smaller than it claims to be.
        "complete": not failures and not unfinished,
        "output_root": output_root,
        "p50_training_launched": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


@app.local_entrypoint()
def chunk_pilot(
    active8_run_root: str,
    gate_zero_decision_path: str,
    expected_commit: str,
    task_identity_sha256: str,
    limit: int = 256,
) -> None:
    """Measure the fused chunk compile on one real chunk. Publishes nothing."""

    revision = local_image_revision(expected_commit=expected_commit)
    print(
        json.dumps(
            chunk_pilot_remote.remote(
                active8_run_root,
                gate_zero_decision_path,
                task_identity_sha256,
                int(limit),
                revision,
            ),
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def canary(
    active8_run_root: str,
    gate_zero_decision_path: str,
    expected_commit: str,
    scoped_t1_output_prefix: str = SCOPED_T1_OUTPUT_PREFIX,
    prepared_output_prefix: str = PREPARED_OUTPUT_PREFIX,
) -> None:
    """Compile and publish the largest selected leaf without launching P50."""

    revision = local_image_revision(expected_commit=expected_commit)
    scoped_t1 = materialize_scoped_t1_remote.remote(
        active8_run_root,
        gate_zero_decision_path,
        scoped_t1_output_prefix,
        revision,
    )
    selected = prepare_selection_remote.remote(
        active8_run_root,
        gate_zero_decision_path,
        str(scoped_t1["result_path"]),
        str(scoped_t1["decision_path"]),
        prepared_output_prefix,
        revision,
    )
    counts = dict(selected["task_entry_counts"])
    task_identity = min(counts, key=lambda identity: (-int(counts[identity]), identity))
    started = time.monotonic()
    leaf = prepare_leaf_remote.remote(
        selected["selection_path"],
        task_identity,
        active8_run_root,
        gate_zero_decision_path,
        selected["run_root"],
        revision,
    )
    print(
        json.dumps(
            {
                "phase": "process_v2_p50_prepare_canary_complete",
                "elapsed_seconds": time.monotonic() - started,
                "selected_task_entry_count": int(counts[task_identity]),
                "selection_path": selected["selection_path"],
                "run_root": selected["run_root"],
                "leaf": leaf,
                "p50_training_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.function(
    image=image,
    cpu=0.5,
    memory=2 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def scan_completed_remote(
    selection_path: str,
    run_root: str,
    task_identity_sha256s: list[str],
    revision: dict[str, Any],
) -> list[str]:
    """Return only immutable leaves that pass their exact local binding."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    selection = loaded["validate_selection"](
        _read_canonical_object(
            _require_artifact_path(selection_path, field="selection_path"),
            label="the Process-V2 P50 selection",
        ),
    )
    root = _require_artifact_path(run_root, field="run_root")
    completed: list[str] = []
    for task_id in task_identity_sha256s:
        path = _leaf_path(root, task_id)
        if not path.is_file():
            continue
        _validate_leaf(
            _read_canonical_object(path, label="a Process-V2 P50 leaf"),
            selection_sha256=str(selection["selection_sha256"]),
            task_identity_sha256=task_id,
        )
        completed.append(task_id)
    return completed


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def migrate_v1_leaves_remote(
    selection_path: str,
    legacy_run_root: str,
    run_root: str,
    task_identity_sha256s: list[str],
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Convert compatible exhaustive leaves without molecular computation."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    selection = loaded["validate_selection"](
        _read_canonical_object(
            _require_artifact_path(selection_path, field="selection_path"),
            label="the Process-V2 P50 selection",
        )
    )
    old_root = _require_artifact_path(legacy_run_root, field="legacy_run_root")
    new_root = _require_physical_artifact_path(run_root, field="run_root")
    converted = 0
    already_current = 0
    with _heartbeat("process_v2_p50_legacy_leaf_conversion"):
        for task_id in task_identity_sha256s:
            destination = _leaf_path(new_root, task_id)
            if destination.is_file():
                _validate_leaf(
                    _read_canonical_object(destination, label="a Process-V2 P50 leaf"),
                    selection_sha256=str(selection["selection_sha256"]),
                    task_identity_sha256=task_id,
                )
                already_current += 1
                continue
            source = _leaf_path(old_root, task_id)
            if not source.is_file():
                continue
            converted_leaf = loaded["convert_v1_leaf"](
                selection,
                _read_canonical_object(source, label="a legacy Process-V2 P50 leaf"),
            )
            converted_leaf = _validate_leaf(
                converted_leaf,
                selection_sha256=str(selection["selection_sha256"]),
                task_identity_sha256=task_id,
            )
            _publish_canonical(destination, converted_leaf, loaded=loaded)
            converted += 1
    artifact_volume.commit()
    response = {
        "phase": "process_v2_p50_legacy_leaf_conversion_complete",
        "legacy_run_root": str(old_root),
        "run_root": str(new_root),
        "expected_task_count": len(task_identity_sha256s),
        "converted_task_count": converted,
        "already_current_task_count": already_current,
        "molecular_reenumeration_count": 0,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=CPU_PER_LEAF,
    memory=CPU_MEMORY_MB,
    timeout=CPU_LEAF_TIMEOUT_SECONDS,
    max_containers=MAX_CPU_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_leaf_remote(
    selection_path: str,
    task_identity_sha256: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    run_root: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Compile and immutably publish one selected source-chunk leaf."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    selection = loaded["validate_selection"](
        _read_canonical_object(
            _require_artifact_path(selection_path, field="selection_path"),
            label="the Process-V2 P50 selection",
        ),
    )
    source = _open_source(
        active8_run_root=_require_artifact_path(active8_run_root, field="active8_run_root"),
        gate_zero_decision_path=_require_artifact_path(
            gate_zero_decision_path, field="gate_zero_decision_path"
        ),
        loaded=loaded,
    )

    def report(event: Mapping[str, Any]) -> None:
        _progress(**dict(event))

    with _heartbeat("process_v2_p50_prepare_leaf_heartbeat"):
        leaf = loaded["compile_entries"](
            selection,
            source=source,
            task_identity_sha256=task_identity_sha256,
            progress_callback=report,
        )
    leaf = _validate_leaf(
        leaf,
        selection_sha256=str(selection["selection_sha256"]),
        task_identity_sha256=task_identity_sha256,
    )
    path = _publish_canonical(
        _leaf_path(
            _require_physical_artifact_path(run_root, field="run_root"),
            task_identity_sha256,
        ),
        leaf,
        loaded=loaded,
    )
    artifact_volume.commit()
    response = {
        "task_identity_sha256": task_identity_sha256,
        "leaf_sha256": leaf["leaf_sha256"],
        "leaf_path": str(path),
        "entry_count": int(leaf["entry_count"]),
    }
    _progress("process_v2_p50_prepare_leaf_complete", **response)
    return response


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def finalize_prepared_remote(
    selection_path: str,
    run_root: str,
    task_identity_sha256s: list[str],
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Validate and deterministically reduce every selected compilation leaf."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    selection = loaded["validate_selection"](
        _read_canonical_object(
            _require_artifact_path(selection_path, field="selection_path"),
            label="the Process-V2 P50 selection",
        ),
    )
    root = _require_physical_artifact_path(run_root, field="run_root")
    leaves = [
        _validate_leaf(
            _read_canonical_object(_leaf_path(root, task_id), label="a Process-V2 P50 leaf"),
            selection_sha256=str(selection["selection_sha256"]),
            task_identity_sha256=task_id,
        )
        for task_id in sorted(task_identity_sha256s)
    ]
    with _heartbeat("process_v2_p50_prepared_reduction"):
        prepared = loaded["build_prepared"](selection, leaves=leaves)
    prepared = loaded["validate_prepared"](prepared)
    prepared_path = _publish_canonical(root / PREPARED_FILENAME, prepared, loaded=loaded)
    artifact_volume.commit()
    response = {
        "phase": "process_v2_p50_prepared_complete",
        "run_root": str(root),
        "prepared_path": str(prepared_path),
        "prepared_sha256": prepared["prepared_sha256"],
        "entry_count": int(prepared["entry_count"]),
        "p50_training_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_collated_plan_remote(
    prepared_path: str,
    output_prefix: str,
    max_cpu_containers: int,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Plan balanced CPU-only collation from already compiled immutable leaves."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    prepared_file = _require_artifact_path(prepared_path, field="prepared_path")
    prepared = loaded["validate_prepared"](
        _read_canonical_object(prepared_file, label="the Process-V2 P50 prepared input")
    )
    selection = loaded["validate_selection"](
        _read_canonical_object(
            prepared_file.parent / SELECTION_FILENAME,
            label="the Process-V2 P50 selection beside the prepared input",
        )
    )
    if selection["selection_sha256"] != prepared["selection_sha256"]:
        raise RuntimeError("Process-V2 P50 prepared input and selection disagree")
    task_ids = sorted({str(entry["task_identity_sha256"]) for entry in selection["entries"]})
    leaves = [
        _validate_leaf(
            _read_canonical_object(
                _leaf_path(prepared_file.parent, task_id),
                label="a Process-V2 P50 prepared leaf",
            ),
            selection_sha256=str(selection["selection_sha256"]),
            task_identity_sha256=task_id,
        )
        for task_id in task_ids
    ]
    tasks = _balanced_collated_tasks(leaves, maximum=max_cpu_containers)
    source_revision = _source_revision(revision)
    implementation_sha256 = _collation_implementation_sha256(revision)
    prepared_file_sha256 = _file_sha256(prepared_file)
    entry_ids = tuple(
        sorted(str(entry["p50_entry_sha256"]) for entry in prepared["entries"])
    )
    identity = {
        "prepared_file_sha256": prepared_file_sha256,
        "prepared_sha256": prepared["prepared_sha256"],
        "collation_implementation_sha256": implementation_sha256,
        "task_inventory_sha256": _sha256(tasks),
    }
    run_identity = _sha256(identity)
    run_root = _require_physical_artifact_path(output_prefix, field="output_prefix") / run_identity
    plan_path = run_root / COLLATED_PLAN_FILENAME
    if plan_path.is_file():
        plan = _validate_collated_plan(
            _read_canonical_object(plan_path, label="the cached Process-V2 P50 collated plan"),
            authority_false=loaded["authority_false_block"](),
            prepared=prepared,
        )
    else:
        body = {
            "schema": COLLATED_PLAN_SCHEMA,
            "schema_version": COLLATED_SCHEMA_VERSION,
            "status": "PLANNED_CPU_COLLATION_NO_DOWNSTREAM_AUTHORITY",
            **loaded["authority_false_block"](),
            "prepared_path": str(prepared_file),
            "prepared_file_sha256": prepared_file_sha256,
            "prepared_sha256": prepared["prepared_sha256"],
            "selection_sha256": prepared["selection_sha256"],
            "initial_model_state_sha256": prepared["initial_model_state_sha256"],
            "collation_implementation_sha256": implementation_sha256,
            "source_revision_sha256": source_revision["source_revision_sha256"],
            "entry_count": len(entry_ids),
            "entry_inventory_sha256": _sha256(entry_ids),
            "task_count": len(tasks),
            "task_inventory_sha256": _sha256(tasks),
            "tasks": tasks,
            "run_identity_sha256": run_identity,
        }
        plan = _validate_collated_plan(
            {**body, "plan_sha256": _sha256(body)},
            authority_false=loaded["authority_false_block"](),
            prepared=prepared,
        )
        _publish_canonical(plan_path, plan, loaded=loaded)
        artifact_volume.commit()
    response = {
        "phase": "process_v2_p50_collated_plan_complete",
        "run_root": str(run_root),
        "plan_path": str(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "task_identity_sha256s": [task["task_identity_sha256"] for task in plan["tasks"]],
        "task_count": plan["task_count"],
        "entry_count": plan["entry_count"],
        "estimated_raw_mark_count": sum(task["raw_mark_count"] for task in plan["tasks"]),
        "gpu_training_launched": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    cpu=1.0,
    memory=8 * 1024,
    timeout=20 * 60,
    max_containers=MAX_CPU_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def collate_p50_task_remote(
    plan_path: str,
    task_identity_sha256: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Collate one balanced bundle without chemistry or GPU allocation."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    plan_file = _require_artifact_path(plan_path, field="plan_path")
    plan = _validate_collated_plan(
        _read_canonical_object(plan_file, label="the Process-V2 P50 collated plan"),
        authority_false=loaded["authority_false_block"](),
    )
    if plan["collation_implementation_sha256"] != _collation_implementation_sha256(revision):
        raise RuntimeError("Process-V2 P50 collated plan binds another tensor implementation")
    task = _collated_task(plan, task_identity_sha256)
    task_root = _collated_task_root(plan_file.parent, task_identity_sha256)
    receipt_path = task_root / COLLATED_RECEIPT_FILENAME
    if receipt_path.is_file():
        receipt, _batch = _load_collated_payload(
            receipt_path,
            plan=plan,
            task_identity_sha256=task_identity_sha256,
            loaded=loaded,
        )
        return {
            "task_identity_sha256": task_identity_sha256,
            "receipt_sha256": receipt["receipt_sha256"],
            "entry_count": task["entry_count"],
            "reused": True,
        }
    source = _open_source(
        active8_run_root=_require_artifact_path(active8_run_root, field="active8_run_root"),
        gate_zero_decision_path=_require_artifact_path(
            gate_zero_decision_path, field="gate_zero_decision_path"
        ),
        loaded=loaded,
    )
    scratch, _binding, _score_receipt = loaded["build_scratch"](source)
    if loaded["state_dict_sha256"](scratch.model.state_dict()) != plan[
        "initial_model_state_sha256"
    ]:
        raise RuntimeError("Process-V2 P50 collated task model identity disagrees")
    entries: list[dict[str, Any]] = []
    for descriptor in task["leaves"]:
        leaf_id = str(descriptor["leaf_task_identity_sha256"])
        leaf = _validate_leaf(
            _read_canonical_object(
                _leaf_path(Path(plan["prepared_path"]).parent, leaf_id),
                label="a Process-V2 P50 collation source leaf",
            ),
            selection_sha256=str(plan["selection_sha256"]),
            task_identity_sha256=leaf_id,
        )
        if (
            leaf["leaf_sha256"] != descriptor["leaf_sha256"]
            or sorted(str(entry["p50_entry_sha256"]) for entry in leaf["entries"])
            != descriptor["p50_entry_sha256s"]
        ):
            raise RuntimeError("Process-V2 P50 collation source leaf changed")
        entries.extend(dict(entry) for entry in leaf["entries"])
    entries.sort(key=lambda entry: entry["p50_entry_sha256"])
    with _heartbeat("process_v2_p50_cpu_collation_heartbeat"):
        batch, entry_ids = loaded["collate_entries"](entries, scratch.model)
    if list(entry_ids) != task["p50_entry_sha256s"]:
        raise RuntimeError("Process-V2 P50 collated task changed entry order")
    payload = {
        "schema": COLLATED_PAYLOAD_SCHEMA,
        "schema_version": COLLATED_SCHEMA_VERSION,
        "task_identity_sha256": task_identity_sha256,
        "p50_entry_sha256s": list(entry_ids),
        "batch": batch,
    }
    payload_bytes = _torch_bytes(payload, torch_module=loaded["torch"])
    loaded["write_bytes_if_absent"](task_root / COLLATED_PAYLOAD_FILENAME, payload_bytes)
    body = {
        "schema": COLLATED_RECEIPT_SCHEMA,
        "schema_version": COLLATED_SCHEMA_VERSION,
        "status": "COMPLETE_CPU_COLLATION_NO_DOWNSTREAM_AUTHORITY",
        **loaded["authority_false_block"](),
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "prepared_sha256": plan["prepared_sha256"],
        "collation_implementation_sha256": plan["collation_implementation_sha256"],
        "task_identity_sha256": task_identity_sha256,
        "entry_count": task["entry_count"],
        "entry_inventory_sha256": task["entry_inventory_sha256"],
        "payload_filename": COLLATED_PAYLOAD_FILENAME,
        "payload_file_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_file_bytes": len(payload_bytes),
    }
    receipt = _validate_collated_receipt(
        {**body, "receipt_sha256": _sha256(body)},
        authority_false=loaded["authority_false_block"](),
        plan=plan,
        task_identity_sha256=task_identity_sha256,
    )
    _publish_canonical(receipt_path, receipt, loaded=loaded)
    artifact_volume.commit()
    response = {
        "task_identity_sha256": task_identity_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "entry_count": task["entry_count"],
        "payload_file_bytes": len(payload_bytes),
        "reused": False,
    }
    _progress("process_v2_p50_cpu_collation_complete", **response)
    return response


@app.function(
    image=image,
    cpu=1.0,
    memory=4 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def scan_collated_remote(
    plan_path: str, task_identity_sha256s: list[str], revision: dict[str, Any]
) -> list[str]:
    """Return only complete, byte-authenticated collation tasks."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    plan_file = _require_artifact_path(plan_path, field="plan_path")
    plan = _validate_collated_plan(
        _read_canonical_object(plan_file, label="the Process-V2 P50 collated plan"),
        authority_false=loaded["authority_false_block"](),
    )
    completed: list[str] = []
    for task_id in task_identity_sha256s:
        receipt_path = _collated_task_root(plan_file.parent, task_id) / COLLATED_RECEIPT_FILENAME
        if not receipt_path.is_file():
            continue
        receipt = _validate_collated_receipt(
            _read_canonical_object(receipt_path, label="a Process-V2 P50 collated receipt"),
            authority_false=loaded["authority_false_block"](),
            plan=plan,
            task_identity_sha256=task_id,
        )
        payload_path = receipt_path.parent / COLLATED_PAYLOAD_FILENAME
        if (
            not payload_path.is_file()
            or payload_path.stat().st_size != receipt["payload_file_bytes"]
            or _file_sha256(payload_path) != receipt["payload_file_sha256"]
        ):
            raise RuntimeError("Process-V2 P50 collated payload bytes disagree")
        completed.append(task_id)
    return completed


@app.function(
    image=image,
    cpu=8.0,
    memory=64 * 1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def finalize_collated_remote(
    plan_path: str, task_identity_sha256s: list[str], revision: dict[str, Any]
) -> dict[str, Any]:
    """Concatenate and reorder CPU batches once; publish completion last."""

    _validate_remote_revision(revision)
    artifact_volume.reload()
    loaded = _imports()
    plan_file = _require_artifact_path(plan_path, field="plan_path")
    plan = _validate_collated_plan(
        _read_canonical_object(plan_file, label="the Process-V2 P50 collated plan"),
        authority_false=loaded["authority_false_block"](),
    )
    completion_path = plan_file.parent / COLLATED_COMPLETION_FILENAME
    if completion_path.is_file():
        prepared = loaded["validate_prepared"](
            _read_canonical_object(
                _require_artifact_path(plan["prepared_path"], field="prepared_path"),
                label="the Process-V2 P50 prepared input",
            )
        )
        completion, _batch, _entry_ids = _load_collated_completion(
            completion_path, plan=plan, prepared=prepared, loaded=loaded
        )
        return {
            "phase": "process_v2_p50_collated_cache_reused",
            "completion_path": str(completion_path),
            "completion_sha256": completion["completion_sha256"],
            "entry_count": completion["entry_count"],
            "reused": True,
        }
    expected_tasks = [task["task_identity_sha256"] for task in plan["tasks"]]
    if task_identity_sha256s != expected_tasks:
        raise RuntimeError("Process-V2 P50 collation finalizer task inventory disagrees")
    batches: list[Any] = []
    concatenated_ids: list[str] = []
    receipts: list[dict[str, Any]] = []
    with _heartbeat("process_v2_p50_collated_finalization_heartbeat"):
        for task_id in expected_tasks:
            receipt_path = (
                _collated_task_root(plan_file.parent, task_id) / COLLATED_RECEIPT_FILENAME
            )
            receipt, batch = _load_collated_payload(
                receipt_path,
                plan=plan,
                task_identity_sha256=task_id,
                loaded=loaded,
            )
            batches.append(batch)
            concatenated_ids.extend(_collated_task(plan, task_id)["p50_entry_sha256s"])
            receipts.append(
                {
                    "task_identity_sha256": task_id,
                    "receipt_sha256": receipt["receipt_sha256"],
                    "payload_file_sha256": receipt["payload_file_sha256"],
                }
            )
        merged = loaded["concatenate_batches"](tuple(batches))
        expected_ids = sorted(concatenated_ids)
        if (
            len(expected_ids) != plan["entry_count"]
            or len(expected_ids) != len(set(expected_ids))
            or _sha256(expected_ids) != plan["entry_inventory_sha256"]
        ):
            raise RuntimeError("Process-V2 P50 collated reduction changed entry inventory")
        position = {identifier: index for index, identifier in enumerate(concatenated_ids)}
        merged = loaded["index_batch"](merged, tuple(position[item] for item in expected_ids))
        payload = {
            "schema": COLLATED_PAYLOAD_SCHEMA,
            "schema_version": COLLATED_SCHEMA_VERSION,
            "task_identity_sha256": plan["run_identity_sha256"],
            "p50_entry_sha256s": expected_ids,
            "batch": merged,
        }
        payload_bytes = _torch_bytes(payload, torch_module=loaded["torch"])
        payload_path = plan_file.parent / COLLATED_PAYLOAD_FILENAME
        loaded["write_bytes_if_absent"](payload_path, payload_bytes)
    body = {
        "schema": COLLATED_COMPLETION_SCHEMA,
        "schema_version": COLLATED_SCHEMA_VERSION,
        "status": "COMPLETE_CPU_COLLATION_NO_DOWNSTREAM_AUTHORITY",
        **loaded["authority_false_block"](),
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "prepared_path": plan["prepared_path"],
        "prepared_file_sha256": plan["prepared_file_sha256"],
        "prepared_sha256": plan["prepared_sha256"],
        "collation_implementation_sha256": plan["collation_implementation_sha256"],
        "entry_count": plan["entry_count"],
        "entry_inventory_sha256": plan["entry_inventory_sha256"],
        "task_count": plan["task_count"],
        "task_receipt_inventory_sha256": _sha256(receipts),
        "task_receipts": receipts,
        "payload_filename": COLLATED_PAYLOAD_FILENAME,
        "payload_file_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_file_bytes": len(payload_bytes),
        "gpu_training_launched": False,
    }
    completion = {**body, "completion_sha256": _sha256(body)}
    _publish_canonical(completion_path, completion, loaded=loaded)
    artifact_volume.commit()
    response = {
        "phase": "process_v2_p50_collated_cache_complete",
        "completion_path": str(completion_path),
        "completion_sha256": completion["completion_sha256"],
        "entry_count": completion["entry_count"],
        "payload_file_bytes": completion["payload_file_bytes"],
        "reused": False,
    }
    _progress(**response)
    return response


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=64 * 1024,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_p50_gpu_remote(
    prepared_path: str,
    collated_completion_path: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    t1_result_path: str,
    t1_decision_path: str,
    output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Run one exact fifty-step scratch pilot and publish its decision last."""

    _validate_remote_revision(revision)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("Process-V2 P50 requires deterministic cuBLAS before materialization")
    artifact_volume.reload()
    loaded = _imports()
    gate_path = _require_artifact_path(gate_zero_decision_path, field="gate_zero_decision_path")
    prerequisites = _load_prerequisites(
        gate_zero_decision_path=gate_path,
        t1_result_path=_require_artifact_path(t1_result_path, field="t1_result_path"),
        t1_decision_path=_require_artifact_path(t1_decision_path, field="t1_decision_path"),
        loaded=loaded,
    )
    source = _open_source(
        active8_run_root=_require_artifact_path(active8_run_root, field="active8_run_root"),
        gate_zero_decision_path=gate_path,
        loaded=loaded,
    )
    prepared = loaded["validate_prepared"](
        _read_canonical_object(
            _require_artifact_path(prepared_path, field="prepared_path"),
            label="the Process-V2 P50 prepared input",
        ),
    )
    collated_completion_file = _require_artifact_path(
        collated_completion_path, field="collated_completion_path"
    )
    plan = _validate_collated_plan(
        _read_canonical_object(
            collated_completion_file.parent / COLLATED_PLAN_FILENAME,
            label="the Process-V2 P50 collated plan",
        ),
        authority_false=loaded["authority_false_block"](),
        prepared=prepared,
    )
    if (
        plan["collation_implementation_sha256"] != _collation_implementation_sha256(revision)
        or plan["prepared_path"] != str(prepared_path)
        or plan["prepared_file_sha256"] != _file_sha256(Path(prepared_path))
    ):
        raise RuntimeError("Process-V2 P50 collated cache binds another scientific input")
    collated_completion, collated_batch, collated_entry_ids = _load_collated_completion(
        collated_completion_file, plan=plan, prepared=prepared, loaded=loaded
    )
    _require_prepared_prerequisite_binding(prepared, prerequisites=prerequisites)
    runtime = loaded["load_runtime"](prepared)
    materialized = loaded["materialize_batch"](
        runtime, batch=collated_batch, entry_ids=collated_entry_ids
    )
    scratch, _binding, score_receipt = loaded["build_scratch"](source)
    policy = loaded["load_policy"](loaded["P50_RECIPE_POLICY"], repo_root=REMOTE_ROOT)
    source_revision = _source_revision(revision)
    runner_hash = _runner_implementation_sha256(revision)
    environment = _execution_environment(loaded, batch_size=prerequisites.batch_size)
    expected_runtime_provenance = {
        "prepared_inputs_sha256": prepared["prepared_sha256"],
        "training_stream_sha256": prepared["training_stream_sha256"],
        "validation_stream_sha256": prepared["validation_inventory_sha256"],
        "runner_implementation_sha256": runner_hash,
        "runner_source_revision_sha256": source_revision["source_revision_sha256"],
        "collated_completion_sha256": collated_completion["completion_sha256"],
        "collated_payload_file_sha256": collated_completion["payload_file_sha256"],
    }
    provenance = {
        **prerequisites.as_payload(),
        "p50_prerequisite_binding_sha256": prerequisites.binding_sha256,
        "prepared_inputs_sha256": expected_runtime_provenance["prepared_inputs_sha256"],
        "training_stream_sha256": expected_runtime_provenance["training_stream_sha256"],
        "validation_stream_sha256": expected_runtime_provenance["validation_stream_sha256"],
        "initial_model_state_sha256": prepared["initial_model_state_sha256"],
        "runner_implementation_sha256": runner_hash,
        "runner_source_revision_sha256": source_revision["source_revision_sha256"],
        "collated_completion_sha256": collated_completion["completion_sha256"],
        "collated_payload_file_sha256": collated_completion["payload_file_sha256"],
        "execution_environment": environment,
    }
    if (
        score_receipt["receipt_sha256"] != prerequisites.t1_score_revision_receipt_sha256
        or score_receipt["current_initial_model_state_sha256"]
        != prerequisites.t1_initial_model_state_sha256
        or loaded["state_dict_sha256"](scratch.model.state_dict())
        != prerequisites.t1_initial_model_state_sha256
    ):
        raise RuntimeError("Process-V2 P50 score-revision scratch binding disagrees")
    # ``active_families``, ``optimizer_steps``, and ``batch_size`` belong to
    # the validated prerequisite object but are not result provenance fields.
    for field in ("active_families", "optimizer_steps", "batch_size"):
        provenance.pop(field, None)
    run_identity = _sha256(
        {
            "prepared_sha256": prepared["prepared_sha256"],
            "prerequisites_binding_sha256": prerequisites.binding_sha256,
            "runner_implementation_sha256": runner_hash,
            "runner_source_revision_sha256": source_revision["source_revision_sha256"],
            "collated_completion_sha256": collated_completion["completion_sha256"],
            "collated_payload_file_sha256": collated_completion["payload_file_sha256"],
            "environment_sha256": environment["environment_sha256"],
        }
    )
    run_root = _require_physical_artifact_path(output_prefix, field="output_prefix") / run_identity
    result_path = run_root / loaded["RESULT_FILENAME"]
    decision_path = run_root / loaded["DECISION_FILENAME"]
    checkpoint_path = run_root / TERMINAL_CHECKPOINT_FILENAME

    if decision_path.is_file():
        if not result_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError("a Process-V2 P50 decision exists without its complete run")
        result = loaded["validate_result"](
            _read_canonical_object(result_path, label="the Process-V2 P50 result"),
            recipe_policy=policy,
            prerequisites=prerequisites,
            required_cell_ids=prepared["required_cells"],
            validation_unsupported_required_cells=prepared["validation_unsupported_required_cells"],
            expected_runtime_provenance=expected_runtime_provenance,
        )
        if (
            _file_sha256(checkpoint_path)
            != result["run_integrity"]["terminal_checkpoint_file_sha256"]
        ):
            raise RuntimeError("Process-V2 P50 terminal checkpoint file hash disagrees")
        decision = loaded["validate_decision"](
            _read_canonical_object(decision_path, label="the Process-V2 P50 decision"),
            result=result,
            recipe_policy=policy,
            prerequisites=prerequisites,
            required_cell_ids=prepared["required_cells"],
            validation_unsupported_required_cells=prepared["validation_unsupported_required_cells"],
            expected_runtime_provenance=expected_runtime_provenance,
            result_file_sha256=_file_sha256(result_path),
        )
        return {
            "phase": "process_v2_p50_gpu_reused_complete",
            "run_root": str(run_root),
            "result_path": str(result_path),
            "decision_path": str(decision_path),
            "decision_status": decision["status"],
            "p500_authorized": decision["p500_authorized"],
            "optimizer_steps_completed": result["run_integrity"]["optimizer_steps_completed"],
            "training_reused": True,
        }

    torch = loaded["torch"]
    model = scratch.model.to(device="cuda", dtype=torch.float32)
    _progress(
        "process_v2_p50_gpu_start",
        run_root=str(run_root),
        optimizer_steps=prerequisites.optimizer_steps,
        prepared_entry_count=int(prepared["entry_count"]),
    )

    def report(event: Mapping[str, Any]) -> None:
        _progress(**dict(event))

    with _heartbeat("process_v2_p50_gpu_heartbeat"):
        run = loaded["run_p50"](
            model,
            runtime,
            policy=policy,
            progress_callback=report,
            materialized=materialized,
        )
    checkpoint = _checkpoint_payload(
        run,
        policy=policy,
        provenance={
            **provenance,
            "prepared_input_path": str(prepared_path),
            "prepared_sha256": prepared["prepared_sha256"],
            "selection_sha256": prepared["selection_sha256"],
            "collated_completion_path": str(collated_completion_file),
            "collated_completion_sha256": collated_completion["completion_sha256"],
            "collated_payload_file_sha256": collated_completion["payload_file_sha256"],
        },
        loaded=loaded,
    )
    checkpoint = _validate_checkpoint_payload(checkpoint, loaded=loaded)
    checkpoint_bytes = _torch_bytes(checkpoint, torch_module=torch)
    loaded["write_bytes_if_absent"](checkpoint_path, checkpoint_bytes)
    checkpoint_file_sha256 = _file_sha256(checkpoint_path)
    optimizer_state_sha256 = loaded["optimizer_state_sha256"](run["optimizer_state"])
    run_integrity = {
        "optimizer_steps_completed": int(run["optimizer_steps_completed"]),
        "scheduled_example_count": len(prepared["training_stream"]),
        "batch_size": prerequisites.batch_size,
        "initialization": "scratch_from_t1_bound_initial_model_state",
        "resume_requested": False,
        "resume_count": 0,
        "abort_triggered": False,
        "abort_reasons": [],
        "nonfinite_event_count": 0,
        "unsupported_teacher_count": 0,
        "missing_candidate_count": 0,
        "provenance_drift_detected": False,
        "stream_identity_drift_detected": False,
        "deterministic_algorithms_enabled": True,
        "dtype": "float32",
        "mixed_precision": False,
        "hazard_included": False,
        "t1_selected_checkpoint_loaded": False,
        "initial_model_state_sha256": prepared["initial_model_state_sha256"],
        "terminal_model_state_sha256": run["final_model_state_sha256"],
        "optimizer_state_sha256": optimizer_state_sha256,
        "terminal_checkpoint_file_sha256": checkpoint_file_sha256,
        "hazard_initial_state_sha256": run["hazard_initial_state_sha256"],
        "hazard_final_state_sha256": run["hazard_final_state_sha256"],
    }
    result = loaded["build_result"](
        recipe_policy=policy,
        prerequisites=prerequisites,
        required_cell_ids=prepared["required_cells"],
        validation_unsupported_required_cells=prepared["validation_unsupported_required_cells"],
        expected_runtime_provenance=expected_runtime_provenance,
        provenance=provenance,
        run_integrity=run_integrity,
        trajectory=run["trajectory"],
        validation_entry_metrics=run["validation_entry_metrics"],
        family_training_evidence=run["family_training_evidence"],
        semantic_cell_training_evidence=run["semantic_cell_training_evidence"],
    )
    _publish_canonical(result_path, result, loaded=loaded)
    result_file_sha256 = _file_sha256(result_path)
    decision = loaded["build_decision"](
        result,
        recipe_policy=policy,
        prerequisites=prerequisites,
        required_cell_ids=prepared["required_cells"],
        validation_unsupported_required_cells=prepared["validation_unsupported_required_cells"],
        expected_runtime_provenance=expected_runtime_provenance,
        result_file_sha256=result_file_sha256,
    )
    # Decision publication is the atomic completion marker.  Readers never
    # treat a checkpoint or result without this recomputed object as complete.
    _publish_canonical(decision_path, decision, loaded=loaded)
    reopened = loaded["validate_decision"](
        _read_canonical_object(decision_path, label="the Process-V2 P50 decision"),
        result=result,
        recipe_policy=policy,
        prerequisites=prerequisites,
        required_cell_ids=prepared["required_cells"],
        validation_unsupported_required_cells=prepared["validation_unsupported_required_cells"],
        expected_runtime_provenance=expected_runtime_provenance,
        result_file_sha256=result_file_sha256,
    )
    artifact_volume.commit()
    response = {
        "phase": "process_v2_p50_gpu_complete",
        "run_root": str(run_root),
        "result_path": str(result_path),
        "result_file_sha256": result_file_sha256,
        "result_sha256": result["result_sha256"],
        "decision_path": str(decision_path),
        "decision_sha256": reopened["decision_sha256"],
        "decision_status": reopened["status"],
        "checkpoint_path": str(checkpoint_path),
        "terminal_checkpoint_file_sha256": checkpoint_file_sha256,
        "optimizer_steps_completed": int(run["optimizer_steps_completed"]),
        "threshold_checks": result["threshold_checks"],
        "p500_authorized": reopened["p500_authorized"],
        "training_reused": False,
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
def prepare_driver(
    active8_run_root: str,
    gate_zero_decision_path: str,
    scoped_t1_output_prefix: str,
    prepared_output_prefix: str,
    max_cpu_containers: int,
    legacy_prepared_run_root: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Publish complete restart-safe CPU inputs without allocating a GPU."""

    _validate_remote_revision(revision)
    if type(max_cpu_containers) is not int or not 1 <= max_cpu_containers <= MAX_CPU_CONTAINERS:
        raise ValueError(f"max_cpu_containers must lie in [1, {MAX_CPU_CONTAINERS}]")
    scoped_t1 = materialize_scoped_t1_remote.remote(
        active8_run_root,
        gate_zero_decision_path,
        scoped_t1_output_prefix,
        revision,
    )
    t1_result_path = str(scoped_t1["result_path"])
    t1_decision_path = str(scoped_t1["decision_path"])
    selected = prepare_selection_remote.remote(
        active8_run_root,
        gate_zero_decision_path,
        t1_result_path,
        t1_decision_path,
        prepared_output_prefix,
        revision,
    )
    expected = list(selected["task_identity_sha256s"])
    migration = migrate_v1_leaves_remote.remote(
        selected["selection_path"],
        legacy_prepared_run_root,
        selected["run_root"],
        expected,
        revision,
    )
    completed = set(
        scan_completed_remote.remote(
            selected["selection_path"], selected["run_root"], expected, revision
        )
    )
    missing = [identity for identity in expected if identity not in completed]
    waves = _groups(missing, maximum=max_cpu_containers)
    _progress(
        "process_v2_p50_cpu_map_plan",
        expected_tasks=len(expected),
        already_complete=len(completed),
        missing_tasks=len(missing),
        max_cpu_containers=max_cpu_containers,
        submission_waves=len(waves),
    )
    prepare_leaf_remote.update_autoscaler(max_containers=max_cpu_containers)
    if missing:
        results = list(
            prepare_leaf_remote.starmap(
                [
                    (
                        selected["selection_path"],
                        identity,
                        active8_run_root,
                        gate_zero_decision_path,
                        selected["run_root"],
                        revision,
                    )
                    for identity in missing
                ]
            )
        )
        if {str(result["task_identity_sha256"]) for result in results} != set(missing):
            raise RuntimeError("Process-V2 P50 CPU map lost a task result")
    complete = set(
        scan_completed_remote.remote(
            selected["selection_path"], selected["run_root"], expected, revision
        )
    )
    if complete != set(expected):
        raise RuntimeError(
            f"Process-V2 P50 preparation is incomplete: missing={len(set(expected) - complete)}"
        )
    prepared = finalize_prepared_remote.remote(
        selected["selection_path"], selected["run_root"], expected, revision
    )
    response = {
        "phase": "process_v2_p50_preparation_complete",
        "prepared": prepared,
        "scoped_t1": scoped_t1,
        "max_cpu_containers": max_cpu_containers,
        "cpu_submission_waves": len(waves),
        "image_revision": revision,
        "legacy_leaf_migration": migration,
        "p50_training_launched": False,
        "p500_authorized": False,
        "p500_launched": False,
    }
    _progress(
        "process_v2_p50_preparation_complete",
        prepared_sha256=prepared["prepared_sha256"],
        reused_legacy_tasks=migration["converted_task_count"],
        computed_tasks=len(missing),
        p50_training_launched=False,
        p500_authorized=False,
        p500_launched=False,
    )
    return response


@app.function(
    image=image,
    cpu=0.25,
    memory=1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
)
def collate_driver(
    prepared_path: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    collated_output_prefix: str,
    max_cpu_containers: int,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Build the reusable CPU tensor cache before any GPU allocation."""

    _validate_remote_revision(revision)
    if type(max_cpu_containers) is not int or not 1 <= max_cpu_containers <= MAX_CPU_CONTAINERS:
        raise ValueError(f"max_cpu_containers must lie in [1, {MAX_CPU_CONTAINERS}]")
    planned = prepare_collated_plan_remote.remote(
        prepared_path, collated_output_prefix, max_cpu_containers, revision
    )
    expected = list(planned["task_identity_sha256s"])
    completed = set(scan_collated_remote.remote(planned["plan_path"], expected, revision))
    missing = [task_id for task_id in expected if task_id not in completed]
    _progress(
        "process_v2_p50_cpu_collation_map_plan",
        expected_tasks=len(expected),
        already_complete=len(completed),
        missing_tasks=len(missing),
        max_cpu_containers=max_cpu_containers,
        gpu_training_launched=False,
    )
    collate_p50_task_remote.update_autoscaler(max_containers=max_cpu_containers)
    if missing:
        results = list(
            collate_p50_task_remote.starmap(
                [
                    (
                        planned["plan_path"],
                        task_id,
                        active8_run_root,
                        gate_zero_decision_path,
                        revision,
                    )
                    for task_id in missing
                ]
            )
        )
        if {result["task_identity_sha256"] for result in results} != set(missing):
            raise RuntimeError("Process-V2 P50 CPU collation lost a task result")
    complete = set(scan_collated_remote.remote(planned["plan_path"], expected, revision))
    if complete != set(expected):
        raise RuntimeError(
            f"Process-V2 P50 CPU collation is incomplete: "
            f"missing={len(set(expected) - complete)}"
        )
    finalized = finalize_collated_remote.remote(planned["plan_path"], expected, revision)
    response = {
        "phase": "process_v2_p50_cpu_collation_driver_complete",
        "plan": planned,
        "completion": finalized,
        "max_cpu_containers": max_cpu_containers,
        "computed_tasks": len(missing),
        "reused_tasks": len(completed),
        "gpu_training_launched": False,
    }
    _progress(
        "process_v2_p50_cpu_collation_driver_complete",
        completion_sha256=finalized["completion_sha256"],
        computed_tasks=len(missing),
        reused_tasks=len(completed),
        gpu_training_launched=False,
    )
    return response


@app.function(
    image=image,
    cpu=0.25,
    memory=1024,
    timeout=COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
)
def train_driver(
    prepared_path: str,
    collated_completion_path: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    t1_result_path: str,
    t1_decision_path: str,
    run_output_prefix: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Run only the bounded GPU pilot from an immutable prepared artifact."""

    _validate_remote_revision(revision)
    gpu = run_p50_gpu_remote.remote(
        prepared_path,
        collated_completion_path,
        active8_run_root,
        gate_zero_decision_path,
        t1_result_path,
        t1_decision_path,
        run_output_prefix,
        revision,
    )
    response = {
        "phase": "process_v2_p50_training_complete",
        "pilot": gpu,
        "image_revision": revision,
        "p500_authorized": gpu["p500_authorized"],
        "p500_launched": False,
    }
    _progress(
        "process_v2_p50_training_complete",
        decision_status=gpu["decision_status"],
        p500_authorized=gpu["p500_authorized"],
        p500_launched=False,
    )
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
    scoped_t1_output_prefix: str,
    prepared_output_prefix: str,
    collated_output_prefix: str,
    run_output_prefix: str,
    max_cpu_containers: int,
    legacy_prepared_run_root: str,
    revision: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility path that composes the separate prepare and train drivers."""

    _validate_remote_revision(revision)
    prepared = prepare_driver.remote(
        active8_run_root,
        gate_zero_decision_path,
        scoped_t1_output_prefix,
        prepared_output_prefix,
        max_cpu_containers,
        legacy_prepared_run_root,
        revision,
    )
    collated = collate_driver.remote(
        prepared["prepared"]["prepared_path"],
        active8_run_root,
        gate_zero_decision_path,
        collated_output_prefix,
        max_cpu_containers,
        revision,
    )
    gpu = train_driver.remote(
        prepared["prepared"]["prepared_path"],
        collated["completion"]["completion_path"],
        active8_run_root,
        gate_zero_decision_path,
        prepared["scoped_t1"]["result_path"],
        prepared["scoped_t1"]["decision_path"],
        run_output_prefix,
        revision,
    )
    return {
        **prepared,
        "collated": collated,
        "phase": "process_v2_p50_driver_complete",
        "pilot": gpu["pilot"],
        "p500_authorized": gpu["p500_authorized"],
        "p500_launched": False,
    }


@app.local_entrypoint()
def prepare(
    active8_run_root: str,
    gate_zero_decision_path: str,
    expected_commit: str,
    scoped_t1_output_prefix: str = SCOPED_T1_OUTPUT_PREFIX,
    prepared_output_prefix: str = PREPARED_OUTPUT_PREFIX,
    max_cpu_containers: int = MAX_CPU_CONTAINERS,
    legacy_prepared_run_root: str = LEGACY_PREPARED_RUN_ROOT,
    wait_for_completion: bool = False,
) -> None:
    """Spawn CPU-only preparation and return without allocating a GPU."""

    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        active8_run_root,
        gate_zero_decision_path,
        scoped_t1_output_prefix,
        prepared_output_prefix,
        int(max_cpu_containers),
        legacy_prepared_run_root,
        revision,
    )
    if wait_for_completion:
        print(json.dumps(prepare_driver.remote(*arguments), indent=2, sort_keys=True))
        return
    call = prepare_driver.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_p50_preparation_launched",
                "driver_call_id": call.object_id,
                "max_cpu_containers": int(max_cpu_containers),
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "p50_training_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def collate(
    prepared_path: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    expected_commit: str,
    collated_output_prefix: str = COLLATED_OUTPUT_PREFIX,
    max_cpu_containers: int = MAX_CPU_CONTAINERS,
    wait_for_completion: bool = False,
) -> None:
    """Spawn reusable CPU-only tensor collation and return immediately."""

    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        prepared_path,
        active8_run_root,
        gate_zero_decision_path,
        collated_output_prefix,
        int(max_cpu_containers),
        revision,
    )
    if wait_for_completion:
        print(json.dumps(collate_driver.remote(*arguments), indent=2, sort_keys=True))
        return
    call = collate_driver.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_p50_cpu_collation_launched",
                "driver_call_id": call.object_id,
                "prepared_path": prepared_path,
                "max_cpu_containers": int(max_cpu_containers),
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "gpu_training_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def train(
    prepared_path: str,
    collated_completion_path: str,
    active8_run_root: str,
    gate_zero_decision_path: str,
    t1_result_path: str,
    t1_decision_path: str,
    expected_commit: str,
    run_output_prefix: str = RUN_OUTPUT_PREFIX,
    wait_for_completion: bool = False,
) -> None:
    """Spawn GPU-only P50 from one explicit immutable prepared artifact."""

    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        prepared_path,
        collated_completion_path,
        active8_run_root,
        gate_zero_decision_path,
        t1_result_path,
        t1_decision_path,
        run_output_prefix,
        revision,
    )
    if wait_for_completion:
        print(json.dumps(train_driver.remote(*arguments), indent=2, sort_keys=True))
        return
    call = train_driver.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_p50_training_launched",
                "driver_call_id": call.object_id,
                "prepared_path": prepared_path,
                "collated_completion_path": collated_completion_path,
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "p50_training_launched": True,
                "p500_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def main(
    active8_run_root: str,
    gate_zero_decision_path: str,
    expected_commit: str,
    scoped_t1_output_prefix: str = SCOPED_T1_OUTPUT_PREFIX,
    prepared_output_prefix: str = PREPARED_OUTPUT_PREFIX,
    collated_output_prefix: str = COLLATED_OUTPUT_PREFIX,
    run_output_prefix: str = RUN_OUTPUT_PREFIX,
    max_cpu_containers: int = MAX_CPU_CONTAINERS,
    legacy_prepared_run_root: str = LEGACY_PREPARED_RUN_ROOT,
    wait_for_completion: bool = False,
) -> None:
    """Spawn one disconnect-safe Process-V2 P50 driver and return immediately."""

    revision = local_image_revision(expected_commit=expected_commit)
    arguments = (
        active8_run_root,
        gate_zero_decision_path,
        scoped_t1_output_prefix,
        prepared_output_prefix,
        collated_output_prefix,
        run_output_prefix,
        int(max_cpu_containers),
        legacy_prepared_run_root,
        revision,
    )
    if wait_for_completion:
        print(json.dumps(driver.remote(*arguments), indent=2, sort_keys=True))
        return
    call = driver.spawn(*arguments)
    print(
        json.dumps(
            {
                "phase": "process_v2_p50_driver_launched",
                "driver_call_id": call.object_id,
                "max_cpu_containers": int(max_cpu_containers),
                "commit": revision["commit"],
                "image_revision_sha256": revision["image_revision_sha256"],
                "p50_driver_launched": True,
                "p50_training_launched": False,
                "p500_authorized": False,
                "p500_launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "COLLATED_OUTPUT_PREFIX",
    "MAX_CPU_CONTAINERS",
    "LEGACY_PREPARED_RUN_ROOT",
    "PREPARED_OUTPUT_PREFIX",
    "RUN_OUTPUT_PREFIX",
    "app",
    "collate",
    "collate_driver",
    "collate_p50_task_remote",
    "driver",
    "finalize_prepared_remote",
    "finalize_collated_remote",
    "local_image_revision",
    "main",
    "materialize_scoped_t1_remote",
    "migrate_v1_leaves_remote",
    "prepare_leaf_remote",
    "prepare_driver",
    "prepare",
    "prepare_selection_remote",
    "prepare_collated_plan_remote",
    "run_p50_gpu_remote",
    "scan_completed_remote",
    "scan_collated_remote",
    "train_driver",
    "train",
]
