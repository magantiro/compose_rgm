"""Build the strict combined semantic P50 successor cache on Modal CPUs.

The driver physically reopens the prepared recipe, complete semantic Active8
source, Gate 0 evidence, and T1 decision before freezing a content-addressed
plan.  At most five CPU workers compile immutable score-independent leaves.
Each leaf covers one exact train or validation task and is committed
independently, so a retry reuses every already validated leaf.

The reducer publishes a manifest and completion receipt only after all leaves
exist, then invokes the strict combined-cache opener.  No model score,
probability, hazard coordinate, optimizer state, training result, or training
authority is produced.  The only final output is the combined cache completion
path; validation consumers use that cache rather than a second cache format.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/build_editing_v2_semantic_p50_successor_cache_app.py"
SEMANTIC_MODEL_PROCESS_SOURCE = "configs/editing_gate_zero_semantic_model_process_v1.json"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_p50_successor_cache"
SOURCE_REVISION_SCHEMA = "compose.editing_v2.semantic_p50_successor_cache_modal_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 2
MAX_VOLUME_WRITERS = 5
SUPPORT_COMPILATION_TIME = 0.5

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_INPUT_FIELDS = {
    "prepared_recipe",
    "source_inventory",
    "migration_completion",
    "chunk_cache_plan",
    "chunk_cache_global_completion",
    "decision_plan",
    "decision_completion",
    "gate_zero_evidence",
    "t1_decision",
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
image = image.add_local_file(
    ROOT / LAUNCHER_SOURCE,
    str(REMOTE_ROOT / LAUNCHER_SOURCE),
    copy=True,
)

app = modal.App("compose-v4-editing-v2-semantic-p50-successor-cache")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
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
            f"cannot establish semantic P50 cache Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE]
    for source_directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / source_directory).rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("semantic P50 cache source inventory repeats a path")
    return tuple(paths)


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    return {relative: _file_sha(root / relative) for relative in _serialized_source_paths(root)}


def local_source_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind the exact clean commit and every serialized source/config byte."""

    if not isinstance(expected_commit, str) or not _COMMIT_RE.fullmatch(expected_commit):
        raise ValueError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "semantic P50 successor cache requires the exact clean committed worktree"
        )
    hashes = _serialized_source_hashes(root)
    tracked = set(_git(root, "ls-files").splitlines())
    if not hashes or not set(hashes).issubset(tracked):
        raise RuntimeError("every semantic P50 cache serialized source/config must be Git-tracked")
    loaded = _imports(root)
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "execution_source_revision": loaded["build_execution_source_revision"](
            commit=commit,
            tree=tree,
        ),
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": _sha(hashes),
    }
    return {**body, "source_revision_sha256": _sha(body)}


def _validate_source_revision(value: object, *, remote_root: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic P50 cache source revision must be an object")
    revision = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "worktree_clean",
        "execution_source_revision",
        "serialized_source_hashes",
        "serialized_source_hashes_sha256",
        "source_revision_sha256",
    }
    body = dict(revision)
    supplied_sha = body.pop("source_revision_sha256", None)
    hashes = _serialized_source_hashes(remote_root)
    loaded = _imports(remote_root)
    try:
        execution_revision = loaded["validate_execution_source_revision"](
            revision.get("execution_source_revision")
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("semantic P50 cache execution-source revision disagrees") from error
    if (
        set(revision) != expected_fields
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or execution_revision.get("commit") != revision.get("commit")
        or execution_revision.get("tree") != revision.get("tree")
        or revision.get("serialized_source_hashes") != hashes
        or revision.get("serialized_source_hashes_sha256") != _sha(hashes)
        or supplied_sha != _sha(body)
        or not isinstance(revision.get("commit"), str)
        or not _COMMIT_RE.fullmatch(revision["commit"])
        or not isinstance(revision.get("tree"), str)
        or not _COMMIT_RE.fullmatch(revision["tree"])
    ):
        raise RuntimeError("semantic P50 cache serialized source revision disagrees")
    return revision


def _artifact_path(value: str, *, artifact_root: Path, field_name: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a /artifacts path")
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise ValueError(f"{field_name} must be a normalized /artifacts path")
    root = Path(artifact_root).resolve()
    result = (root / Path(*pure.parts[2:])).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"{field_name} resolves outside artifact_root")
    return result


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    root = Path(artifact_root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact resolves outside artifact_root")
    relative = resolved.relative_to(root)
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _validate_input_addresses(value: object, *, artifact_root: Path) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _INPUT_FIELDS:
        raise TypeError("semantic P50 cache input-address fields disagree")
    result = dict(value)
    for name, address in result.items():
        _artifact_path(
            address,
            artifact_root=artifact_root,
            field_name=f"input_addresses.{name}",
        )
    return result


def _load_canonical_object(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {field_name}: {path}") from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise RuntimeError(f"{field_name} is not canonical JSON")
    return payload


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.editing_v2_semantic_capability_cells import (
        load_semantic_capability_cell_registry,
    )
    from compose_v4.data.successor_fiber_cache import SuccessorFiberCacheAddress
    from compose_v4.experiments import (
        editing_v2_semantic_p50_successor_cache as cache,
    )
    from compose_v4.experiments.cnof_conditional import PathRecord
    from compose_v4.experiments.editing_gate_zero_semantic_contract import (
        load_gate_zero_semantic_contract,
    )
    from compose_v4.experiments.editing_v2_execution_source_revision import (
        build_execution_source_revision,
        validate_execution_source_revision,
    )
    from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
        SemanticP50Prerequisites,
        validate_semantic_p50_prerequisite_relationships,
    )
    from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
        SemanticP50SourceInventoryBinding,
        load_semantic_p50_source_inventory,
    )
    from compose_v4.experiments.editing_v2_semantic_runtime import (
        SemanticScratchModelConfig,
        build_semantic_scratch_runtime,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_decision import (
        validate_semantic_gate_zero_evidence_receipt,
        validate_semantic_t1_capacity_decision,
    )
    from compose_v4.experiments.successor_fiber_cache_builder import (
        compile_successor_fiber_trace_union,
    )

    return {
        "cache": cache,
        "load_registry": load_semantic_capability_cell_registry,
        "SuccessorFiberCacheAddress": SuccessorFiberCacheAddress,
        "PathRecord": PathRecord,
        "load_semantic_contract": load_gate_zero_semantic_contract,
        "build_execution_source_revision": build_execution_source_revision,
        "validate_execution_source_revision": validate_execution_source_revision,
        "SemanticP50Prerequisites": SemanticP50Prerequisites,
        "validate_prerequisites": validate_semantic_p50_prerequisite_relationships,
        "SemanticP50SourceInventoryBinding": SemanticP50SourceInventoryBinding,
        "load_source": load_semantic_p50_source_inventory,
        "SemanticScratchModelConfig": SemanticScratchModelConfig,
        "build_scratch": build_semantic_scratch_runtime,
        "validate_gate_zero": validate_semantic_gate_zero_evidence_receipt,
        "validate_t1": validate_semantic_t1_capacity_decision,
        "compile_trace_union": compile_successor_fiber_trace_union,
    }


def _source_binding(prerequisites: Mapping[str, Any], *, loaded: Mapping[str, Any]) -> Any:
    return loaded["SemanticP50SourceInventoryBinding"](
        source_inventory_file_sha256=prerequisites["source_inventory_file_sha256"],
        source_inventory_sha256=prerequisites["source_inventory_sha256"],
        process_identity_sha256=prerequisites["process_identity_sha256"],
        model_runtime_identity_sha256=prerequisites["model_runtime_identity_sha256"],
        active8_policy_sha256=prerequisites["active8_policy_sha256"],
        operator_capability_fingerprint=prerequisites["operator_capability_fingerprint"],
        decision_source_implementation_sha256=prerequisites[
            "decision_source_implementation_sha256"
        ],
    )


def _scratch_runtime(
    gate_zero: Mapping[str, Any], *, loaded: Mapping[str, Any], remote_root: Path
) -> Any:
    runtime = gate_zero.get("model_runtime_identity")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("architecture"), Mapping):
        raise RuntimeError("Gate 0 model runtime identity is incomplete")
    architecture = runtime["architecture"]
    config = loaded["SemanticScratchModelConfig"](
        initialization_seed=runtime["initialization_seed"],
        max_atoms=architecture["max_atoms"],
        hidden_dim=architecture["hidden_dim"],
        message_passing_steps=architecture["message_passing_steps"],
        mark_dim=architecture["mark_dim"],
        dtype=architecture["dtype"],
        atom_vocabulary_class_count=architecture["atom_vocabulary_class_count"],
        catalog_fingerprint=architecture["catalog_fingerprint"],
    )
    contract = loaded["load_semantic_contract"](Path(remote_root) / SEMANTIC_MODEL_PROCESS_SOURCE)
    scratch = loaded["build_scratch"](config, contract)
    if (
        scratch.architecture.as_payload() != dict(architecture)
        or scratch.initial_model_state_sha256 != runtime.get("initial_model_state_sha256")
        or scratch.process_identity_sha256 != runtime.get("process_identity_sha256")
        or scratch.semantic_model_process_contract_sha256
        != runtime.get("semantic_model_process_contract_sha256")
        or scratch.semantic_model_identity != runtime.get("semantic_model_identity")
    ):
        raise RuntimeError("scratch runtime differs from the physically reopened Gate 0 identity")
    scratch.model.eval()
    return scratch


def _open_exact_prerequisites(
    input_addresses: Mapping[str, str],
    *,
    artifact_root: Path,
    remote_root: Path,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    addresses = _validate_input_addresses(input_addresses, artifact_root=artifact_root)
    paths = {
        name: _artifact_path(
            address,
            artifact_root=artifact_root,
            field_name=f"input_addresses.{name}",
        )
        for name, address in addresses.items()
    }
    cache = loaded["cache"]
    prepared, _ = cache.load_semantic_p50_prepared_recipe(paths["prepared_recipe"])
    prerequisites = loaded["SemanticP50Prerequisites"](**dict(prepared["prerequisites"]))
    binding = _source_binding(prepared["prerequisites"], loaded=loaded)
    source = loaded["load_source"](
        paths["source_inventory"],
        expected_binding=binding,
        migration_completion_path=paths["migration_completion"],
        chunk_cache_plan_path=paths["chunk_cache_plan"],
        chunk_cache_global_completion_path=paths["chunk_cache_global_completion"],
        decision_plan_path=paths["decision_plan"],
        decision_completion_path=paths["decision_completion"],
        artifact_root=artifact_root,
        repo_root=remote_root,
    )
    gate_file_sha = _file_sha(paths["gate_zero_evidence"])
    gate_zero = loaded["validate_gate_zero"](
        paths["gate_zero_evidence"],
        expected_file_sha256=gate_file_sha,
    )
    t1_payload = _load_canonical_object(paths["t1_decision"], field_name="semantic T1 decision")
    t1_decision = loaded["validate_t1"](
        t1_payload,
        decision_path=paths["t1_decision"],
        repo_root=remote_root,
        expected_gate_zero_evidence_file_sha256=gate_file_sha,
        require_p50_go=True,
    )
    registry = loaded["load_registry"]()
    reconstructed = loaded["validate_prerequisites"](
        source=source,
        gate_zero=gate_zero,
        gate_zero_file_sha256=gate_file_sha,
        t1_decision=t1_decision,
        t1_decision_file_sha256=_file_sha(paths["t1_decision"]),
        registry=registry,
    )
    if reconstructed != prerequisites:
        raise RuntimeError("prepared P50 recipe differs from the reopened source, Gate 0, or T1")
    scratch = _scratch_runtime(gate_zero, loaded=loaded, remote_root=remote_root)
    if scratch.initial_model_state_sha256 != prerequisites.scratch_initial_model_state_sha256:
        raise RuntimeError("prepared P50 recipe names another scratch model state")
    source_paths = cache.SemanticP50SourceReopenPaths(
        source_inventory_path=paths["source_inventory"],
        migration_completion_path=paths["migration_completion"],
        chunk_cache_plan_path=paths["chunk_cache_plan"],
        chunk_cache_global_completion_path=paths["chunk_cache_global_completion"],
        decision_plan_path=paths["decision_plan"],
        decision_completion_path=paths["decision_completion"],
    )
    expected_identity = (
        cache.expected_semantic_p50_successor_cache_identity_from_reopened_prerequisites(
            prepared_recipe_sha256=prepared["prepared_recipe_sha256"],
            source=source,
            gate_zero_evidence_path=paths["gate_zero_evidence"],
            t1_decision_path=paths["t1_decision"],
            repo_root=remote_root,
        )
    )
    return {
        "prepared": prepared,
        "source": source,
        "scratch": scratch,
        "registry": registry,
        "source_paths": source_paths,
        "expected_identity": expected_identity,
    }


def _compiler_runtime(
    *,
    implementation_sha256: str,
    source_revision_sha256: str,
    execution_source_revision_sha256: str,
) -> dict[str, Any]:
    import rdkit
    import torch

    body = {
        "device": "cpu",
        "dtype": "torch.float32",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "rdkit_version": rdkit.__version__,
        "implementation_sha256": implementation_sha256,
        "source_revision_sha256": source_revision_sha256,
        "execution_source_revision_sha256": execution_source_revision_sha256,
    }
    return {**body, "runtime_sha256": _sha(body)}


def _prepare_plan(
    *,
    source_revision: Mapping[str, Any],
    input_addresses: Mapping[str, str],
    output_prefix: str,
    artifact_root: Path,
    remote_root: Path,
    stage_commit: Any | None,
) -> tuple[dict[str, Any], Path]:
    revision = _validate_source_revision(source_revision, remote_root=remote_root)
    loaded = _imports(remote_root)
    opened = _open_exact_prerequisites(
        input_addresses,
        artifact_root=artifact_root,
        remote_root=remote_root,
        loaded=loaded,
    )
    prepared_path = _artifact_path(
        input_addresses["prepared_recipe"],
        artifact_root=artifact_root,
        field_name="input_addresses.prepared_recipe",
    )
    cache = loaded["cache"]
    if cache.SUPPORT_COMPILATION_TIME != SUPPORT_COMPILATION_TIME:
        raise RuntimeError("launcher and cache support-compilation time disagree")
    plan = cache.build_semantic_p50_successor_cache_plan(
        prepared_recipe_path=prepared_path,
        verified_source=opened["source"],
        source_revision_sha256=revision["source_revision_sha256"],
        execution_source_revision_sha256=revision["execution_source_revision"][
            "source_revision_sha256"
        ],
        artifact_root=artifact_root,
        repo_root=remote_root,
        registry=opened["registry"],
        output_prefix=output_prefix,
    )
    run_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=artifact_root,
        field_name="plan.run_artifact_root",
    )
    plan_path = run_root / cache.PLAN_FILENAME
    created = cache.write_semantic_p50_successor_cache_artifact(plan_path, plan)
    if created and stage_commit is not None:
        stage_commit()
    physical = _load_canonical_object(plan_path, field_name="semantic P50 cache plan")
    cache.validate_semantic_p50_successor_cache_plan(
        physical, artifact_root=artifact_root, repo_root=remote_root
    )
    return plan, plan_path


def _task_batches(
    tasks: Sequence[Mapping[str, Any]], *, maximum_batches: int
) -> tuple[tuple[str, ...], ...]:
    """Greedily balance immutable tasks without exceeding the writer cap."""

    if type(maximum_batches) is not int or maximum_batches <= 0:
        raise ValueError("maximum_batches must be positive")
    if not tasks:
        return ()
    batch_count = min(maximum_batches, len(tasks))
    bins: list[list[str]] = [[] for _ in range(batch_count)]
    loads = [0] * batch_count
    ordered = sorted(
        tasks,
        key=lambda task: (
            -int(task["closure_record_count"]),
            str(task["task_identity_sha256"]),
        ),
    )
    for task in ordered:
        index = min(range(batch_count), key=lambda item: (loads[item], item))
        bins[index].append(str(task["task_identity_sha256"]))
        loads[index] += int(task["closure_record_count"])
    return tuple(tuple(items) for items in bins if items)


def _path_records_by_task(
    source: Any,
    *,
    plan: Mapping[str, Any],
    task_identities: Sequence[str],
    loaded: Mapping[str, Any],
) -> dict[str, tuple[Any, ...]]:
    requested_task_identities = set(task_identities)
    selected_tasks = {
        task["task_identity_sha256"]: task
        for task in plan["tasks"]
        if task["task_identity_sha256"] in requested_task_identities
    }
    if set(selected_tasks) != requested_task_identities:
        raise RuntimeError("semantic P50 cache worker received an unknown task")
    owner_by_trace: dict[tuple[Any, ...], str] = {}
    expected_rows_by_trace: defaultdict[tuple[Any, ...], set[Any]] = defaultdict(set)
    for task_identity, task in selected_tasks.items():
        for row in task["closure_rows"]:
            address = loaded["SuccessorFiberCacheAddress"](**row["address"])
            existing = owner_by_trace.setdefault(address.trace_key, task_identity)
            if existing != task_identity:
                raise RuntimeError("one physical trace belongs to multiple cache tasks")
            expected_rows_by_trace[address.trace_key].add(address)
    records: defaultdict[str, list[Any]] = defaultdict(list)
    observed: set[tuple[Any, ...]] = set()
    for partition_role in ("train", "validation"):
        for trace in source.index.iter_accepted_traces_for_partition(partition_role):
            packed = trace.addressed_trace.address
            first = loaded["SuccessorFiberCacheAddress"].from_packed_trace(packed, progress_index=0)
            task_identity = owner_by_trace.get(first.trace_key)
            if task_identity is None:
                continue
            if first.trace_key in observed:
                raise RuntimeError("semantic P50 source repeats a selected trace")
            expected_rows = expected_rows_by_trace[first.trace_key]
            physical_rows = {
                loaded["SuccessorFiberCacheAddress"].from_packed_trace(
                    packed, progress_index=progress_index
                )
                for progress_index in range(packed.path_length + 1)
            }
            if expected_rows != physical_rows:
                raise RuntimeError("semantic P50 worker trace differs from its exact task closure")
            transitions = tuple(source.index.accepted_transitions_for(trace))
            if len(transitions) != packed.path_length:
                raise RuntimeError("semantic P50 worker source omits a transition")
            records[task_identity].append(
                loaded["PathRecord"](
                    target_key=packed.target_key,
                    path=trace.addressed_trace.path,
                    corpus_address=packed,
                )
            )
            observed.add(first.trace_key)
    if observed != set(owner_by_trace):
        raise RuntimeError("semantic P50 worker did not reopen every selected trace")
    return {task_identity: tuple(items) for task_identity, items in records.items()}


@app.function(
    image=image,
    cpu=4.0,
    memory=32768,
    timeout=12 * 3600,
    max_containers=MAX_VOLUME_WRITERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def compile_semantic_p50_cache_task_batch(
    plan_artifact_path: str,
    source_revision: dict[str, Any],
    input_addresses: dict[str, str],
    task_identities: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Compile and independently commit one immutable leaf per missing task."""

    revision = _validate_source_revision(source_revision, remote_root=REMOTE_ROOT)
    artifact_volume.reload()
    loaded = _imports()
    cache = loaded["cache"]
    plan_path = _artifact_path(
        plan_artifact_path,
        artifact_root=ARTIFACT_ROOT,
        field_name="plan_artifact_path",
    )
    plan = cache.validate_semantic_p50_successor_cache_plan(
        _load_canonical_object(plan_path, field_name="semantic P50 cache plan"),
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    if plan["source_revision_sha256"] != revision["source_revision_sha256"]:
        raise RuntimeError("semantic P50 plan names another serialized source")
    opened = _open_exact_prerequisites(
        input_addresses,
        artifact_root=ARTIFACT_ROOT,
        remote_root=REMOTE_ROOT,
        loaded=loaded,
    )
    path_records = _path_records_by_task(
        opened["source"],
        plan=plan,
        task_identities=task_identities,
        loaded=loaded,
    )
    compiler_runtime = _compiler_runtime(
        implementation_sha256=plan["implementation_sha256"],
        source_revision_sha256=plan["source_revision_sha256"],
        execution_source_revision_sha256=plan["execution_source_revision_sha256"],
    )
    results: list[dict[str, Any]] = []
    for task_identity in task_identities:
        compiled = loaded["compile_trace_union"](
            opened["scratch"].model,
            path_records[task_identity],
            time=SUPPORT_COMPILATION_TIME,
        )
        leaf = cache.build_semantic_p50_successor_cache_leaf(
            plan,
            task_identity_sha256=task_identity,
            records=compiled,
            compiler_runtime=compiler_runtime,
            artifact_root=ARTIFACT_ROOT,
            repo_root=REMOTE_ROOT,
        )
        leaf_path = (
            _artifact_path(
                plan["run_artifact_root"],
                artifact_root=ARTIFACT_ROOT,
                field_name="plan.run_artifact_root",
            )
            / "tasks"
            / task_identity
            / cache.LEAF_FILENAME
        )
        reused = not cache.write_semantic_p50_successor_cache_artifact(leaf_path, leaf)
        artifact_volume.commit()
        results.append(
            {
                "task_identity_sha256": task_identity,
                "leaf_sha256": leaf["leaf_sha256"],
                "record_count": leaf["record_count"],
                "leaf_reused": reused,
            }
        )
    return results


@app.function(
    image=image,
    cpu=4.0,
    memory=32768,
    timeout=8 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def reduce_semantic_p50_successor_cache(
    plan_artifact_path: str,
    source_revision: dict[str, Any],
    input_addresses: dict[str, str],
) -> str:
    """Reduce all leaves, strictly reopen the combined cache, then publish."""

    revision = _validate_source_revision(source_revision, remote_root=REMOTE_ROOT)
    artifact_volume.reload()
    loaded = _imports()
    cache = loaded["cache"]
    plan_path = _artifact_path(
        plan_artifact_path,
        artifact_root=ARTIFACT_ROOT,
        field_name="plan_artifact_path",
    )
    plan = cache.validate_semantic_p50_successor_cache_plan(
        _load_canonical_object(plan_path, field_name="semantic P50 cache plan"),
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    if plan["source_revision_sha256"] != revision["source_revision_sha256"]:
        raise RuntimeError("semantic P50 plan names another serialized source")
    opened = _open_exact_prerequisites(
        input_addresses,
        artifact_root=ARTIFACT_ROOT,
        remote_root=REMOTE_ROOT,
        loaded=loaded,
    )
    run_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=ARTIFACT_ROOT,
        field_name="plan.run_artifact_root",
    )
    manifest_path = run_root / cache.MANIFEST_FILENAME
    completion_path = run_root / cache.COMPLETION_FILENAME
    manifest = cache.build_semantic_p50_successor_cache_manifest(
        plan,
        plan_path=plan_path,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    cache.write_semantic_p50_successor_cache_artifact(manifest_path, manifest)
    completion = cache.build_semantic_p50_successor_cache_completion(
        plan,
        manifest,
        plan_path=plan_path,
        manifest_path=manifest_path,
        artifact_root=ARTIFACT_ROOT,
    )
    cache.write_semantic_p50_successor_cache_artifact(completion_path, completion)
    opened_cache = cache.open_semantic_p50_successor_cache(
        completion_path,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
        expected_identity=opened["expected_identity"],
        source_reopen_paths=opened["source_paths"],
        scratch_runtime=opened["scratch"],
        registry=opened["registry"],
    )
    if (
        len(opened_cache.train_requested_records) != completion["train_requested_address_count"]
        or len(opened_cache.validation_requested_records)
        != completion["validation_requested_address_count"]
        or len(opened_cache.records) != completion["closure_record_count"]
    ):
        raise RuntimeError("strictly reopened semantic P50 cache census changed")
    artifact_volume.commit()
    return _artifact_address(completion_path, artifact_root=ARTIFACT_ROOT)


@app.function(
    image=image,
    cpu=4.0,
    memory=32768,
    timeout=24 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def driver(
    *,
    source_revision: dict[str, Any],
    input_addresses: dict[str, str],
    output_prefix: str,
) -> str:
    """Freeze the plan, resume missing leaves, then strictly reduce."""

    artifact_volume.reload()
    plan, plan_path = _prepare_plan(
        source_revision=source_revision,
        input_addresses=input_addresses,
        output_prefix=output_prefix,
        artifact_root=ARTIFACT_ROOT,
        remote_root=REMOTE_ROOT,
        stage_commit=artifact_volume.commit,
    )
    cache = _imports()["cache"]
    run_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=ARTIFACT_ROOT,
        field_name="plan.run_artifact_root",
    )
    missing: list[Mapping[str, Any]] = []
    for task in plan["tasks"]:
        leaf_path = run_root / "tasks" / task["task_identity_sha256"] / cache.LEAF_FILENAME
        if not leaf_path.is_file():
            missing.append(task)
            continue
        leaf = _load_canonical_object(leaf_path, field_name="semantic P50 cache leaf")
        cache.validate_semantic_p50_successor_cache_leaf_for_plan(leaf, plan=plan)
    batches = _task_batches(missing, maximum_batches=MAX_VOLUME_WRITERS)
    if batches:
        plan_address = _artifact_address(plan_path, artifact_root=ARTIFACT_ROOT)
        mapped = list(
            compile_semantic_p50_cache_task_batch.starmap(
                [
                    (
                        plan_address,
                        source_revision,
                        input_addresses,
                        batch,
                    )
                    for batch in batches
                ]
            )
        )
        mapped_items = [item for result_batch in mapped for item in result_batch]
        observed = {item["task_identity_sha256"] for item in mapped_items}
        expected = {task["task_identity_sha256"] for task in missing}
        if (
            observed != expected
            or len(mapped_items) != len(expected)
            or len(observed) != len(mapped_items)
        ):
            raise RuntimeError("semantic P50 cache map lost or repeated a task")
    return reduce_semantic_p50_successor_cache.remote(
        _artifact_address(plan_path, artifact_root=ARTIFACT_ROOT),
        source_revision,
        input_addresses,
    )


@app.local_entrypoint()
def main(
    prepared_recipe: str,
    source_inventory: str,
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    gate_zero_evidence: str,
    t1_decision: str,
    expected_commit: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    source_revision = local_source_revision(expected_commit=expected_commit)
    input_addresses = {
        "prepared_recipe": prepared_recipe,
        "source_inventory": source_inventory,
        "migration_completion": migration_completion,
        "chunk_cache_plan": chunk_cache_plan,
        "chunk_cache_global_completion": chunk_cache_global_completion,
        "decision_plan": decision_plan,
        "decision_completion": decision_completion,
        "gate_zero_evidence": gate_zero_evidence,
        "t1_decision": t1_decision,
    }
    completion_path = driver.remote(
        source_revision=source_revision,
        input_addresses=input_addresses,
        output_prefix=output_prefix,
    )
    print(completion_path)


__all__ = [
    "MAX_VOLUME_WRITERS",
    "OUTPUT_PREFIX",
    "SUPPORT_COMPILATION_TIME",
    "app",
    "compile_semantic_p50_cache_task_batch",
    "driver",
    "local_source_revision",
    "main",
    "reduce_semantic_p50_successor_cache",
]
