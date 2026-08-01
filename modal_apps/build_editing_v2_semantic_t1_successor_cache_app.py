"""Compile the committed semantic T1 unique-state successor cache on CPUs.

The driver reopens the exact semantic panel, Active8 decision source, Gate 0
evidence, and semantic migration inventory.  It freezes one content-addressed
task per selected train shard, fans those tasks out to bounded CPU workers, and
reduces only complete immutable leaves.  Workers compile production
canonical-successor coordinates for every row of each selected complete trace.

No model score, probability, hazard coordinate, repeated-state empirical law,
optimizer, threshold, or training authority is stored here.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/build_editing_v2_semantic_t1_successor_cache_app.py"
POLICY_SOURCE = "configs/editing_v2_semantic_t1_successor_cache_v1.json"
SEMANTIC_MODEL_PROCESS_SOURCE = (
    "configs/editing_gate_zero_semantic_model_process_v1.json"
)
IMAGE_SOURCE_DIRECTORIES = ("src", "configs", "modal_apps")
OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_t1_successor_cache"
SOURCE_REVISION_SCHEMA = (
    "compose.editing.semantic_t1_successor_cache_modal_source_revision"
)
SOURCE_REVISION_SCHEMA_VERSION = 1
MAX_MAP_CONTAINERS = 5

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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

app = modal.App("compose-v4-editing-v2-semantic-t1-successor-cache")
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts", create_if_missing=False
)


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
            f"cannot establish semantic cache Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE]
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "src" / "compose_v4").rglob("*.py"))
        if path.is_file()
    )
    paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "configs").rglob("*.json"))
        if path.is_file()
    )
    return tuple(paths)


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    return {
        relative: _file_sha(root / relative)
        for relative in _serialized_source_paths(root)
    }


def local_source_revision(
    *, expected_commit: str, repo_root: Path = ROOT
) -> dict[str, Any]:
    """Bind the exact clean commit and every serialized code/config byte."""

    if not isinstance(expected_commit, str) or not _COMMIT_RE.fullmatch(
        expected_commit
    ):
        raise ValueError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "semantic T1 successor cache requires the exact clean committed worktree"
        )
    hashes = _serialized_source_hashes(root)
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": _sha(hashes),
    }
    return {**body, "source_revision_sha256": _sha(body)}


def _validate_source_revision(value: object, *, remote_root: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic cache source revision must be an object")
    revision = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "worktree_clean",
        "serialized_source_hashes",
        "serialized_source_hashes_sha256",
        "source_revision_sha256",
    }
    body = dict(revision)
    supplied_sha256 = body.pop("source_revision_sha256", None)
    hashes = _serialized_source_hashes(remote_root)
    if (
        set(revision) != expected_fields
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or revision.get("serialized_source_hashes") != hashes
        or revision.get("serialized_source_hashes_sha256") != _sha(hashes)
        or supplied_sha256 != _sha(body)
        or not isinstance(revision.get("commit"), str)
        or not _COMMIT_RE.fullmatch(revision["commit"])
        or not isinstance(revision.get("tree"), str)
        or not _COMMIT_RE.fullmatch(revision["tree"])
    ):
        raise RuntimeError("semantic cache serialized source revision disagrees")
    return revision


def _artifact_path(value: str, *, artifact_root: Path, field_name: str) -> Path:
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
    relative = Path(path).resolve().relative_to(Path(artifact_root).resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.editing_v2_semantic_active8_decision_source import (
        resolve_editing_v2_semantic_active8_decision_source,
    )
    from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
        resolve_editing_v2_semantic_active8_sources,
    )
    from compose_v4.data.semantic_packed_trace_store import (
        read_semantic_packed_artifact,
    )
    from compose_v4.experiments import editing_v2_semantic_gate_zero as gate_zero
    from compose_v4.experiments import (
        editing_v2_semantic_t1_successor_cache as cache,
    )
    from compose_v4.experiments.cnof_conditional import PathRecord
    from compose_v4.experiments.editing_gate_zero_semantic_contract import (
        load_gate_zero_semantic_contract,
    )
    from compose_v4.experiments.editing_v2_semantic_runtime import (
        SemanticScratchModelConfig,
        build_semantic_scratch_runtime,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_panel_cache import (
        bind_verified_semantic_gate_zero_pass,
        resolve_semantic_t1_cache_trace_inputs,
    )

    return {
        "cache": cache,
        "gate_zero": gate_zero,
        "resolve_decision_source": (
            resolve_editing_v2_semantic_active8_decision_source
        ),
        "resolve_sources": resolve_editing_v2_semantic_active8_sources,
        "read_semantic_packed_artifact": read_semantic_packed_artifact,
        "load_gate_zero_semantic_contract": load_gate_zero_semantic_contract,
        "SemanticScratchModelConfig": SemanticScratchModelConfig,
        "build_semantic_scratch_runtime": build_semantic_scratch_runtime,
        "bind_gate_zero": bind_verified_semantic_gate_zero_pass,
        "resolve_cache_inputs": resolve_semantic_t1_cache_trace_inputs,
        "PathRecord": PathRecord,
    }


def _load_canonical_object(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {field_name}: {path}") from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise RuntimeError(f"{field_name} is not canonical JSON")
    return payload


def _mounted_panel_input(
    bundle: Any,
    name: str,
    *,
    artifact_root: Path,
) -> Path:
    try:
        record = bundle.request["inputs"][name]
        address = record["artifact_path"]
        expected_sha256 = record["file_sha256"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"panel request input {name} is absent") from error
    path = _artifact_path(
        address,
        artifact_root=artifact_root,
        field_name=f"panel input {name}",
    )
    if _file_sha(path) != expected_sha256:
        raise RuntimeError(f"panel request input {name} bytes changed")
    return path


def _load_exact_prerequisites(
    bundle: Any,
    *,
    artifact_root: Path,
    remote_root: Path,
    loaded: Mapping[str, Any],
) -> tuple[Any, Any, Mapping[str, Any], Any]:
    migration_completion = _mounted_panel_input(
        bundle,
        "migration_completion",
        artifact_root=artifact_root,
    )
    decision_plan = _mounted_panel_input(
        bundle,
        "decision_plan",
        artifact_root=artifact_root,
    )
    index = loaded["resolve_decision_source"](
        migration_completion_path=migration_completion,
        chunk_cache_plan_path=_mounted_panel_input(
            bundle, "chunk_cache_plan", artifact_root=artifact_root
        ),
        chunk_cache_global_completion_path=_mounted_panel_input(
            bundle,
            "chunk_cache_global_completion",
            artifact_root=artifact_root,
        ),
        decision_plan_path=decision_plan,
        decision_completion_path=_mounted_panel_input(
            bundle, "decision_completion", artifact_root=artifact_root
        ),
        artifact_root=artifact_root,
        repo_root=remote_root,
    )
    gate_zero = loaded["gate_zero"]
    gate_zero_paths = {
        name: _mounted_panel_input(bundle, name, artifact_root=artifact_root)
        for name in (
            "gate_zero_evidence",
            "gate_zero_decision",
            "gate_zero_completion",
        )
    }
    if (
        len({path.parent.resolve() for path in gate_zero_paths.values()}) != 1
        or gate_zero_paths["gate_zero_evidence"].name != gate_zero.EVIDENCE_FILENAME
        or gate_zero_paths["gate_zero_decision"].name != gate_zero.DECISION_FILENAME
        or gate_zero_paths["gate_zero_completion"].name != gate_zero.COMPLETION_FILENAME
    ):
        raise RuntimeError("Gate 0 artifacts are not the exact sibling set")
    gate_zero_contract = gate_zero.load_semantic_gate_zero_structural_contract(
        repo_root=remote_root
    )
    gate_zero_artifacts = gate_zero.load_semantic_gate_zero_structural_artifacts(
        output_directory=gate_zero_paths["gate_zero_evidence"].parent,
        index=index,
        decision_plan_path=decision_plan,
        contract=gate_zero_contract,
        repo_root=remote_root,
    )
    binding = loaded["bind_gate_zero"](
        index,
        gate_zero_artifacts,
        decision_plan_path=decision_plan,
        contract=gate_zero_contract,
        repo_root=remote_root,
    )
    if binding != bundle.panel.gate_zero_binding:
        raise RuntimeError("live Gate 0 binding differs from the committed panel")
    source_inventory = loaded["resolve_sources"](
        migration_completion,
        artifact_root=artifact_root,
        repo_root=remote_root,
    )
    return index, source_inventory, gate_zero_artifacts, decision_plan


def _prepare_plan(
    *,
    source_revision: Mapping[str, Any],
    panel_completion: str,
    output_prefix: str,
    artifact_root: Path,
    remote_root: Path,
    stage_commit: Any | None,
) -> tuple[dict[str, Any], Path]:
    revision = _validate_source_revision(source_revision, remote_root=remote_root)
    loaded = _imports(remote_root)
    cache = loaded["cache"]
    panel_completion_path = _artifact_path(
        panel_completion,
        artifact_root=artifact_root,
        field_name="panel_completion",
    )
    bundle = cache.load_verified_semantic_t1_panel_bundle(
        panel_completion_path,
        artifact_root=artifact_root,
        repo_root=remote_root,
    )
    index, source_inventory, gate_zero_artifacts, _ = _load_exact_prerequisites(
        bundle,
        artifact_root=artifact_root,
        remote_root=remote_root,
        loaded=loaded,
    )
    resolved = loaded["resolve_cache_inputs"](
        index,
        bundle.panel,
        repo_root=remote_root,
    )
    if len(resolved) != len(bundle.panel.cache_trace_inputs):
        raise RuntimeError("semantic T1 complete-trace cache union is incomplete")
    runtime = gate_zero_artifacts["evidence"].get("model_runtime_identity")
    if not isinstance(runtime, Mapping):
        raise TypeError("Gate 0 model runtime identity is absent")
    semantic_contract = loaded["load_gate_zero_semantic_contract"](
        remote_root / SEMANTIC_MODEL_PROCESS_SOURCE
    )
    semantic_contract_binding = {
        "path": SEMANTIC_MODEL_PROCESS_SOURCE,
        "file_sha256": semantic_contract.file_sha256,
        "contract_sha256": semantic_contract.sha256,
    }
    plan = cache.build_semantic_t1_successor_cache_plan(
        bundle,
        source_inventory=source_inventory,
        model_runtime_identity=runtime,
        semantic_model_process_contract=semantic_contract_binding,
        source_revision=revision,
        artifact_root=artifact_root,
        repo_root=remote_root,
        output_prefix=output_prefix,
    )
    if plan["policy"]["maximum_concurrent_volume_writers"] != MAX_MAP_CONTAINERS:
        raise RuntimeError(
            "semantic cache launcher concurrency differs from the frozen policy"
        )
    run_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=artifact_root,
        field_name="cache run_artifact_root",
    )
    plan_path = run_root / cache.CACHE_PLAN_FILENAME
    created = cache.write_semantic_t1_successor_cache_artifact(plan_path, plan)
    if created and stage_commit is not None:
        stage_commit()
    persisted = _load_canonical_object(plan_path, field_name="semantic cache plan")
    cache.validate_semantic_t1_successor_cache_plan(
        persisted,
        repo_root=remote_root,
        artifact_root=artifact_root,
        expected_source_revision=revision,
    )
    return plan, plan_path


def _task_records(
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    loaded: Mapping[str, Any],
    artifact_root: Path,
) -> tuple[Any, ...]:
    source = task["semantic_source"]
    expected_by_index = {
        item["packed_entry_index"]: item for item in task["cache_trace_inputs"]
    }
    selected: list[Any] = []
    source_path = _artifact_path(
        source["semantic_artifact_path"],
        artifact_root=artifact_root,
        field_name="task semantic_artifact_path",
    )
    for addressed in loaded["read_semantic_packed_artifact"](
        source_path,
        expected_shard_sha256=source["semantic_shard_sha256"],
        expected_manifest_sha256=source["semantic_manifest_sha256"],
        expected_source_binding=source["source_binding"],
    ):
        expected = expected_by_index.get(addressed.address.entry_index)
        if expected is None:
            continue
        address = addressed.address
        if (
            address.packed_shard_content_sha256
            != expected["packed_shard_content_sha256"]
            or address.packed_shard_name != expected["packed_shard_name"]
            or address.trace_id != expected["trace_id"]
            or address.layer != expected["data_lane"]
            or address.partition != "train"
            or address.source_key != expected["trace_source_key"]
            or address.target_key != expected["trace_target_key"]
            or address.path_length != expected["path_length"]
        ):
            raise RuntimeError("semantic cache selected trace envelope changed")
        selected.append(
            loaded["PathRecord"](
                target_key=address.target_key,
                path=addressed.path,
                corpus_address=address,
            )
        )
    if len(selected) != len(expected_by_index):
        raise RuntimeError("semantic cache worker did not reopen every selected trace")
    return tuple(selected)


def _compiler_runtime(
    *, implementation_sha256: str, source_revision_sha256: str
) -> dict[str, Any]:
    import rdkit
    import torch

    return {
        "device": "cpu",
        "dtype": "torch.float32",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "rdkit_version": rdkit.__version__,
        "implementation_sha256": implementation_sha256,
        "source_revision_sha256": source_revision_sha256,
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=32768,
    timeout=8 * 3600,
    max_containers=MAX_MAP_CONTAINERS,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def compile_one_semantic_t1_cache_leaf(
    plan: dict[str, Any], task_identity_sha256: str
) -> dict[str, Any]:
    """Compile or byte-identically reopen one selected train-shard leaf."""

    loaded = _imports()
    cache = loaded["cache"]
    _validate_source_revision(plan["source_revision"], remote_root=REMOTE_ROOT)
    artifact_volume.reload()
    validated = cache.validate_semantic_t1_successor_cache_plan(
        plan,
        repo_root=REMOTE_ROOT,
        artifact_root=ARTIFACT_ROOT,
        expected_source_revision=plan["source_revision"],
    )
    task = next(
        item
        for item in validated["tasks"]
        if item["task_identity_sha256"] == task_identity_sha256
    )
    bundle = cache.load_verified_semantic_t1_panel_bundle(
        _artifact_path(
            validated["panel_binding"]["completion_artifact_path"],
            artifact_root=ARTIFACT_ROOT,
            field_name="plan panel completion",
        ),
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    architecture = validated["model_runtime_identity"]["architecture"]
    config = loaded["SemanticScratchModelConfig"](
        initialization_seed=validated["model_runtime_identity"]["initialization_seed"],
        max_atoms=architecture["max_atoms"],
        hidden_dim=architecture["hidden_dim"],
        message_passing_steps=architecture["message_passing_steps"],
        mark_dim=architecture["mark_dim"],
        dtype=architecture["dtype"],
        atom_vocabulary_class_count=architecture["atom_vocabulary_class_count"],
        catalog_fingerprint=architecture["catalog_fingerprint"],
    )
    semantic_contract = loaded["load_gate_zero_semantic_contract"](
        REMOTE_ROOT / validated["semantic_model_process_contract"]["path"]
    )
    if (
        semantic_contract.file_sha256
        != validated["semantic_model_process_contract"]["file_sha256"]
        or semantic_contract.sha256
        != validated["semantic_model_process_contract"]["contract_sha256"]
    ):
        raise RuntimeError("semantic model/process contract bytes changed")
    scratch = loaded["build_semantic_scratch_runtime"](config, semantic_contract)
    scratch.model.eval()
    records = _task_records(
        validated,
        task,
        loaded=loaded,
        artifact_root=ARTIFACT_ROOT,
    )
    leaf = cache.compile_semantic_t1_successor_cache_leaf(
        validated,
        task_identity_sha256=task_identity_sha256,
        panel=bundle.panel,
        records=records,
        model=scratch.model,
        compiler_runtime=_compiler_runtime(
            implementation_sha256=validated["implementation_sha256"],
            source_revision_sha256=validated["source_revision"][
                "source_revision_sha256"
            ],
        ),
        repo_root=REMOTE_ROOT,
        artifact_root=ARTIFACT_ROOT,
    )
    leaf_path = (
        _artifact_path(
            validated["run_artifact_root"],
            artifact_root=ARTIFACT_ROOT,
            field_name="cache run_artifact_root",
        )
        / "tasks"
        / task_identity_sha256
        / cache.CACHE_LEAF_FILENAME
    )
    reused = not cache.write_semantic_t1_successor_cache_leaf(leaf_path, leaf)
    artifact_volume.commit()
    return {
        "task_identity_sha256": task_identity_sha256,
        "leaf_sha256": leaf["leaf_sha256"],
        "record_count": leaf["record_count"],
        "panel_entry_binding_count": leaf["panel_entry_binding_count"],
        "leaf_reused": reused,
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=4 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def reduce_semantic_t1_successor_cache(plan: dict[str, Any]) -> dict[str, Any]:
    """Strictly reduce all task leaves and publish manifest plus completion."""

    loaded = _imports()
    cache = loaded["cache"]
    _validate_source_revision(plan["source_revision"], remote_root=REMOTE_ROOT)
    artifact_volume.reload()
    run_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=ARTIFACT_ROOT,
        field_name="cache run_artifact_root",
    )
    plan_path = run_root / cache.CACHE_PLAN_FILENAME
    manifest_path = run_root / cache.CACHE_MANIFEST_FILENAME
    completion_path = run_root / cache.CACHE_COMPLETION_FILENAME
    manifest = cache.build_semantic_t1_successor_cache_manifest(
        plan,
        plan_path=plan_path,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    cache.write_semantic_t1_successor_cache_artifact(manifest_path, manifest)
    completion = cache.build_semantic_t1_successor_cache_completion(
        plan,
        manifest,
        plan_path=plan_path,
        manifest_path=manifest_path,
        artifact_root=ARTIFACT_ROOT,
    )
    cache.write_semantic_t1_successor_cache_artifact(
        completion_path,
        completion,
    )
    artifact_volume.commit()
    return {
        "run_root": _artifact_address(run_root, artifact_root=ARTIFACT_ROOT),
        "completion_artifact_path": _artifact_address(
            completion_path, artifact_root=ARTIFACT_ROOT
        ),
        "completion_sha256": completion["completion_sha256"],
        "record_count": completion["record_count"],
        "panel_entry_binding_count": completion["panel_entry_binding_count"],
        "unique_state_successor_cache_compiled": True,
        "repeated_state_empirical_law_compiled": False,
        "training_authorized": False,
        "bounded_p50_authorized": False,
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=24 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def driver(
    *,
    source_revision: dict[str, Any],
    panel_completion: str,
    output_prefix: str,
) -> dict[str, Any]:
    """Freeze the plan, fan out missing immutable leaves, and reduce."""

    artifact_volume.reload()
    plan, _ = _prepare_plan(
        source_revision=source_revision,
        panel_completion=panel_completion,
        output_prefix=output_prefix,
        artifact_root=ARTIFACT_ROOT,
        remote_root=REMOTE_ROOT,
        stage_commit=artifact_volume.commit,
    )
    cache = _imports()["cache"]
    missing: list[str] = []
    run_root = _artifact_path(
        plan["run_artifact_root"],
        artifact_root=ARTIFACT_ROOT,
        field_name="cache run_artifact_root",
    )
    for task in plan["tasks"]:
        leaf_path = (
            run_root
            / "tasks"
            / task["task_identity_sha256"]
            / cache.CACHE_LEAF_FILENAME
        )
        if not leaf_path.is_file():
            missing.append(task["task_identity_sha256"])
            continue
        leaf = _load_canonical_object(leaf_path, field_name="semantic cache leaf")
        cache.validate_semantic_t1_successor_cache_leaf_for_plan(
            leaf,
            plan=plan,
        )
    if missing:
        results = list(
            compile_one_semantic_t1_cache_leaf.starmap(
                [(plan, task_identity_sha256) for task_identity_sha256 in missing]
            )
        )
        if len(results) != len(missing):
            raise RuntimeError("semantic T1 cache map lost a task result")
    return reduce_semantic_t1_successor_cache.remote(plan)


@app.local_entrypoint()
def main(
    panel_completion: str,
    expected_commit: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    source_revision = local_source_revision(expected_commit=expected_commit)
    result = driver.remote(
        source_revision=source_revision,
        panel_completion=panel_completion,
        output_prefix=output_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "MAX_MAP_CONTAINERS",
    "OUTPUT_PREFIX",
    "POLICY_SOURCE",
    "app",
    "compile_one_semantic_t1_cache_leaf",
    "driver",
    "local_source_revision",
    "main",
    "reduce_semantic_t1_successor_cache",
]
