"""Run CPU-only semantic Active8 decisions over verified cached chunks.

The caller supplies the exact semantic-migration completion and exact global
chunk-cache plan/completion.  The driver reconstructs the twenty semantic
sources, verifies the reusable chunk cache, freezes one decision task per
chunk, resumes only verified missing tasks, and invokes the strict reducer.
The model is deterministic scratch initialization under the committed semantic
model/process and decision-runtime contracts.  This app grants no downstream
scientific authority and is never a training launcher.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_semantic_active8_decisions_app.py"
MIGRATION_LAUNCHER_SOURCE = "modal_apps/materialize_editing_v2_semantic_corpus_app.py"
RUNTIME_CONTRACT_SOURCE = "configs/editing_v2_semantic_active8_decision_runtime_v1.json"
SEMANTIC_CONTRACT_SOURCE = "configs/editing_gate_zero_semantic_model_process_v1.json"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
OUTPUT_ARTIFACT_ROOT = "/artifacts/editing_v2/semantic_active8_decisions_v1"
EXPECTED_SOURCE_TASKS = 20
MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS = 5
MAX_MAP_CONTAINERS = MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS

SOURCE_REVISION_SCHEMA = "compose.data.semantic_active8_decision_modal_revision"
SOURCE_REVISION_SCHEMA_VERSION = 2
MODEL_RUNTIME_SCHEMA = "compose.data.semantic_active8_exact_model_runtime"
MODEL_RUNTIME_SCHEMA_VERSION = 2
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
image = image.add_local_file(
    ROOT / LAUNCHER_SOURCE,
    str(REMOTE_ROOT / LAUNCHER_SOURCE),
    copy=True,
)
image = image.add_local_file(
    ROOT / MIGRATION_LAUNCHER_SOURCE,
    str(REMOTE_ROOT / MIGRATION_LAUNCHER_SOURCE),
    copy=True,
)

app = modal.App("compose-v4-semantic-active8-decisions")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)
_WORKER_MODEL_CACHE: dict[str, tuple[Any, dict[str, Any], dict[str, Any], Any, Any]] = {}


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE, MIGRATION_LAUNCHER_SOURCE]
    for source_directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / source_directory).rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("semantic Active8 serialized source inventory repeats a path")
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
            f"cannot establish semantic Active8 Git identity: git {' '.join(arguments)}"
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
    address = PurePosixPath(_require_artifact_path(value, field=field))
    result = (ARTIFACT_ROOT / Path(*address.parts[2:])).resolve()
    if not result.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise RuntimeError(f"{field} resolves outside /artifacts")
    return result


def _artifact_address(path: Path) -> str:
    """Return one stable lexical /artifacts address for a mounted path."""

    relative = Path(path).resolve().relative_to(ARTIFACT_ROOT.resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    import sys

    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.data.editing_v2_semantic_active8_admission import (
        ProductionSemanticExactCandidateChecker,
    )
    from compose_v4.data.editing_v2_semantic_active8_decision_mapreduce import (
        completed_semantic_active8_decision_task_ids,
        execute_semantic_active8_decision_task,
        plan_semantic_active8_decisions,
        reduce_semantic_active8_decisions,
        write_semantic_active8_decision_plan,
    )
    from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
        resolve_editing_v2_semantic_active8_sources,
    )
    from compose_v4.data.semantic_active8_chunk_cache_mapreduce import (
        reduce_semantic_active8_chunk_caches_with_witness,
    )
    from compose_v4.experiments.editing_gate_zero_semantic_contract import (
        load_gate_zero_semantic_contract,
    )
    from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
    from compose_v4.experiments.editing_v2_execution_source_revision import (
        build_execution_source_revision,
        validate_execution_source_revision,
    )
    from compose_v4.experiments.editing_v2_semantic_runtime import (
        SemanticScratchModelConfig,
        build_semantic_scratch_runtime,
    )
    from compose_v4.rewrite.typed_ring_catalog import ring_catalog_fingerprint

    return {
        "ProductionSemanticExactCandidateChecker": ProductionSemanticExactCandidateChecker,
        "SemanticScratchModelConfig": SemanticScratchModelConfig,
        "build_semantic_scratch_runtime": build_semantic_scratch_runtime,
        "ring_catalog_fingerprint": ring_catalog_fingerprint,
        "load_gate_zero_semantic_contract": load_gate_zero_semantic_contract,
        "state_dict_semantic_sha256": state_dict_semantic_sha256,
        "build_execution_source_revision": build_execution_source_revision,
        "validate_execution_source_revision": validate_execution_source_revision,
        "resolve_editing_v2_semantic_active8_sources": (
            resolve_editing_v2_semantic_active8_sources
        ),
        "reduce_semantic_active8_chunk_caches_with_witness": (
            reduce_semantic_active8_chunk_caches_with_witness
        ),
        "completed_semantic_active8_decision_task_ids": (
            completed_semantic_active8_decision_task_ids
        ),
        "execute_semantic_active8_decision_task": execute_semantic_active8_decision_task,
        "plan_semantic_active8_decisions": plan_semantic_active8_decisions,
        "reduce_semantic_active8_decisions": reduce_semantic_active8_decisions,
        "write_semantic_active8_decision_plan": write_semantic_active8_decision_plan,
    }


def _load_json_exact(path: Path, *, field: str, require_canonical: bool = True) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{field} is absent or invalid JSON: {path}") from error
    if not isinstance(value, dict) or (
        require_canonical and _canonical_bytes(value, newline=True) != raw
    ):
        raise RuntimeError(f"{field} is not canonical JSON")
    return value


def _load_runtime_contract(*, root: Path, loaded: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    path = Path(root) / RUNTIME_CONTRACT_SOURCE
    runtime = _load_json_exact(path, field="decision runtime contract", require_canonical=False)
    body = {key: item for key, item in runtime.items() if key != "runtime_contract_sha256"}
    if (
        runtime.get("schema") != "compose.editing.semantic_active8_decision_runtime_contract"
        or runtime.get("schema_version") != 1
        or runtime.get("status") != "FROZEN_CPU_DECISION_RUNTIME_NO_DOWNSTREAM_AUTHORITY"
        or runtime.get("training_authorized") is not False
        or runtime.get("gate_zero_authorized") is not False
        or runtime.get("t1_authorized") is not False
        or runtime.get("bounded_p50_authorized") is not False
        or runtime.get("runtime_contract_sha256") != _sha256(body)
    ):
        raise RuntimeError("semantic Active8 decision runtime contract disagrees")
    parent = runtime.get("semantic_model_process_contract")
    if not isinstance(parent, dict) or set(parent) != {
        "path",
        "file_sha256",
        "contract_sha256",
    }:
        raise RuntimeError("semantic model/process parent identity is malformed")
    parent_path = Path(root) / str(parent["path"])
    if (
        str(parent["path"]) != SEMANTIC_CONTRACT_SOURCE
        or _file_sha256(parent_path) != parent["file_sha256"]
    ):
        raise RuntimeError("semantic model/process contract bytes disagree")
    semantic = loaded["load_gate_zero_semantic_contract"](parent_path)
    if semantic.sha256 != parent["contract_sha256"]:
        raise RuntimeError("semantic model/process contract identity disagrees")
    return runtime, semantic


def _software_identity(runtime: dict[str, Any]) -> dict[str, str]:
    expected = runtime["software"]
    observed = {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "torch": importlib.metadata.version("torch"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "networkx": importlib.metadata.version("networkx"),
        "rdkit": importlib.metadata.version("rdkit"),
    }
    if observed != expected:
        raise RuntimeError(
            f"semantic Active8 software identity disagrees: expected={expected}, observed={observed}"
        )
    return observed


def local_source_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Require the exact clean commit/tree and hash every serialized input."""

    if not isinstance(expected_commit, str) or _COMMIT_RE.fullmatch(expected_commit) is None:
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError("semantic Active8 launch requires the exact clean committed worktree")
    loaded = _imports(root)
    runtime, semantic = _load_runtime_contract(root=root, loaded=loaded)
    sources = {
        relative: _file_sha256(root / relative) for relative in _serialized_source_paths(root)
    }
    tracked = set(_git(root, "ls-files").splitlines())
    if not sources or not set(sources).issubset(tracked):
        raise RuntimeError("every semantic Active8 serialized source/config must be Git-tracked")
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
        "serialized_sources": sources,
        "runtime_contract_sha256": runtime["runtime_contract_sha256"],
        "semantic_model_process_contract_sha256": semantic.sha256,
    }
    return {**body, "source_revision_sha256": _sha256(body)}


def _validate_remote_source_revision(
    value: dict[str, Any], *, loaded: dict[str, Any], remote_root: Path = REMOTE_ROOT
) -> tuple[dict[str, Any], Any]:
    body = {key: item for key, item in value.items() if key != "source_revision_sha256"}
    runtime, semantic = _load_runtime_contract(root=remote_root, loaded=loaded)
    expected_fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "worktree_clean",
        "execution_source_revision",
        "serialized_sources",
        "runtime_contract_sha256",
        "semantic_model_process_contract_sha256",
        "source_revision_sha256",
    }
    try:
        execution_revision = loaded["validate_execution_source_revision"](
            value.get("execution_source_revision")
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("semantic Active8 execution-source revision disagrees") from error
    if (
        set(value) != expected_fields
        or value.get("schema") != SOURCE_REVISION_SCHEMA
        or value.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or value.get("worktree_clean") is not True
        or value.get("source_revision_sha256") != _sha256(body)
        or execution_revision.get("commit") != value.get("commit")
        or execution_revision.get("tree") != value.get("tree")
        or value.get("runtime_contract_sha256") != runtime["runtime_contract_sha256"]
        or value.get("semantic_model_process_contract_sha256") != semantic.sha256
    ):
        raise RuntimeError("semantic Active8 source revision disagrees")
    sources = value.get("serialized_sources")
    expected = set(_serialized_source_paths(remote_root))
    if not isinstance(sources, dict) or set(sources) != expected:
        raise RuntimeError("semantic Active8 serialized source inventory disagrees")
    for relative, digest in sources.items():
        if _file_sha256(Path(remote_root) / relative) != digest:
            raise RuntimeError(f"serialized semantic Active8 source differs: {relative}")
    return runtime, semantic


def _model_runtime_identity(
    model: Any,
    *,
    runtime: dict[str, Any],
    semantic: Any,
    source_revision: dict[str, Any],
    loaded: dict[str, Any],
) -> dict[str, Any]:
    import torch

    model_config = runtime["model"]
    state = model.state_dict()
    dtypes = sorted({str(tensor.dtype) for tensor in state.values()})
    architecture = {
        "max_atoms": model_config["max_atoms"],
        "hidden_dim": model.hidden_dim,
        "message_passing_steps": model.message_passing_steps,
        "mark_dim": model.mark_dim,
        "dtype": str(next(model.parameters()).dtype),
        "parameter_dtypes": dtypes,
        "atom_vocabulary_class_count": len(model.atom_vocabulary.classes),
        "catalog_fingerprint": loaded["ring_catalog_fingerprint"](model.ring_catalog),
        "operator_capability_fingerprint": model.operator_capabilities.fingerprint(),
    }
    expected_architecture = {
        "max_atoms": model_config["max_atoms"],
        "hidden_dim": model_config["hidden_dim"],
        "message_passing_steps": model_config["message_passing_steps"],
        "mark_dim": model_config["mark_dim"],
        "dtype": model_config["dtype"],
        "parameter_dtypes": [model_config["dtype"]],
        "atom_vocabulary_class_count": model_config["atom_vocabulary_class_count"],
        "catalog_fingerprint": model_config["catalog_fingerprint"],
        "operator_capability_fingerprint": semantic.payload["model_identity"][
            "operator_capability_fingerprint"
        ],
    }
    if architecture != expected_architecture or model.training or torch.is_grad_enabled():
        raise RuntimeError("live semantic Active8 model architecture or inference mode disagrees")
    body: dict[str, Any] = {
        "schema": MODEL_RUNTIME_SCHEMA,
        "schema_version": MODEL_RUNTIME_SCHEMA_VERSION,
        "runtime_contract_sha256": runtime["runtime_contract_sha256"],
        "semantic_model_process_contract_sha256": semantic.sha256,
        "semantic_model_identity": semantic.payload["model_identity"],
        "process_identity_sha256": semantic.payload["process_identity_sha256"],
        "architecture": architecture,
        "initialization_seed": model_config["initialization_seed"],
        "initial_model_state_sha256": loaded["state_dict_semantic_sha256"](state),
        "software": _software_identity(runtime),
        "producer_source_revision_sha256": source_revision["source_revision_sha256"],
        "execution_source_revision_sha256": source_revision["execution_source_revision"][
            "source_revision_sha256"
        ],
    }
    return {**body, "identity_sha256": _sha256(body)}


def build_exact_semantic_model(
    *,
    runtime: dict[str, Any],
    semantic: Any,
    source_revision: dict[str, Any],
    loaded: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Construct deterministic scratch semantic Active8 inference state."""

    import torch

    config = runtime["model"]
    scratch = loaded["build_semantic_scratch_runtime"](
        loaded["SemanticScratchModelConfig"](
            initialization_seed=int(config["initialization_seed"]),
            max_atoms=int(config["max_atoms"]),
            hidden_dim=int(config["hidden_dim"]),
            message_passing_steps=int(config["message_passing_steps"]),
            mark_dim=int(config["mark_dim"]),
            dtype=str(config["dtype"]),
            atom_vocabulary_class_count=int(config["atom_vocabulary_class_count"]),
            catalog_fingerprint=str(config["catalog_fingerprint"]),
        ),
        semantic,
    )
    model = scratch.model
    with torch.inference_mode():
        runtime_identity = _model_runtime_identity(
            model,
            runtime=runtime,
            semantic=semantic,
            source_revision=source_revision,
            loaded=loaded,
        )
    if (
        runtime_identity["initial_model_state_sha256"] != scratch.initial_model_state_sha256
        or runtime_identity["architecture"] != scratch.architecture.as_payload()
        or runtime_identity["semantic_model_identity"] != scratch.semantic_model_identity
        or runtime_identity["semantic_model_process_contract_sha256"]
        != scratch.semantic_model_process_contract_sha256
        or runtime_identity["process_identity_sha256"] != scratch.process_identity_sha256
    ):
        raise RuntimeError("shared semantic scratch runtime differs from decision runtime identity")
    return model, runtime_identity


def _worker_model_and_checker(
    *, source_revision: dict[str, Any], loaded: dict[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any], Any, Any]:
    """Reuse immutable inference state and its successor cache within a container."""

    key = str(source_revision["source_revision_sha256"])
    cached = _WORKER_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    runtime, semantic = _validate_remote_source_revision(source_revision, loaded=loaded)
    model, runtime_identity = build_exact_semantic_model(
        runtime=runtime,
        semantic=semantic,
        source_revision=source_revision,
        loaded=loaded,
    )
    checker = loaded["ProductionSemanticExactCandidateChecker"](
        model,
        cache_size=int(runtime["model"]["candidate_cache_size"]),
        time=float(runtime["model"]["candidate_time"]),
    )
    result = (model, runtime_identity, runtime, semantic, checker)
    _WORKER_MODEL_CACHE[key] = result
    return result


def _load_verified_chunk_inputs(
    *, plan_path: Path, global_completion_path: Path, loaded: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    cache_plan = _load_json_exact(plan_path, field="semantic chunk-cache plan")
    caller_pointer = _load_json_exact(
        global_completion_path, field="semantic chunk-cache global completion"
    )
    witness = loaded["reduce_semantic_active8_chunk_caches_with_witness"](
        cache_plan, artifact_root=ARTIFACT_ROOT
    )
    reduced = witness.reduction
    expected_pointer = {
        key: value for key, value in reduced.items() if key != "completion"
    }
    if caller_pointer != expected_pointer:
        raise RuntimeError(
            "caller chunk-cache GLOBAL_COMPLETE differs from strict reduction"
        )
    return cache_plan, witness


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=8 * 3600,
    max_containers=MAX_MAP_CONTAINERS,
    volumes={"/artifacts": artifact_volume},
)
def decide_one_chunk(
    plan_artifact_path: str,
    task_identity_sha256: str,
    source_revision: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact model, decide one chunk, and commit its durable receipt."""

    import torch

    loaded = _imports()
    model, runtime_identity, runtime, semantic, checker = _worker_model_and_checker(
        source_revision=source_revision,
        loaded=loaded,
    )
    artifact_volume.reload()
    plan = _load_json_exact(
        _mounted_artifact_path(plan_artifact_path, field="plan_artifact_path"),
        field="published semantic Active8 decision plan",
    )
    with torch.inference_mode():
        receipt = loaded["execute_semantic_active8_decision_task"](
            plan,
            task_identity_sha256,
            artifact_root=ARTIFACT_ROOT,
            model=model,
            exact_candidate_checker=checker,
            model_runtime_identity_resolver=lambda live: _model_runtime_identity(
                live,
                runtime=runtime,
                semantic=semantic,
                source_revision=source_revision,
                loaded=loaded,
            ),
        )
    if runtime_identity["identity_sha256"] != plan["model_runtime_identity"]["identity_sha256"]:
        raise RuntimeError("worker model identity differs from decision plan")
    artifact_volume.commit()
    return {
        "task_identity_sha256": task_identity_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=8 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def reduce_decisions(
    plan: dict[str, Any],
    semantic_migration_completion: str,
    source_revision: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact source inventory and strictly reduce all chunks."""

    loaded = _imports()
    _validate_remote_source_revision(source_revision, loaded=loaded)
    artifact_volume.reload()
    inventory = loaded["resolve_editing_v2_semantic_active8_sources"](
        _mounted_artifact_path(
            semantic_migration_completion, field="semantic_migration_completion"
        ),
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    result = loaded["reduce_semantic_active8_decisions"](
        plan, inventory=inventory, artifact_root=ARTIFACT_ROOT
    )
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=24 * 3600,
    volumes={"/artifacts": artifact_volume},
)
def driver(
    semantic_migration_completion: str,
    semantic_chunk_cache_plan: str,
    semantic_chunk_cache_global_completion: str,
    output_artifact_root: str,
    source_revision: dict[str, Any],
) -> dict[str, Any]:
    """Validate exact upstreams, resume durable chunk tasks, and reduce."""

    loaded = _imports()
    runtime, semantic = _validate_remote_source_revision(source_revision, loaded=loaded)
    artifact_volume.reload()
    inventory = loaded["resolve_editing_v2_semantic_active8_sources"](
        _mounted_artifact_path(
            semantic_migration_completion, field="semantic_migration_completion"
        ),
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    if len(inventory.sources) != EXPECTED_SOURCE_TASKS:
        raise RuntimeError("semantic migration did not resolve exactly twenty sources")
    cache_plan, cache_witness = _load_verified_chunk_inputs(
        plan_path=_mounted_artifact_path(
            semantic_chunk_cache_plan, field="semantic_chunk_cache_plan"
        ),
        global_completion_path=_mounted_artifact_path(
            semantic_chunk_cache_global_completion,
            field="semantic_chunk_cache_global_completion",
        ),
        loaded=loaded,
    )
    model, runtime_identity = build_exact_semantic_model(
        runtime=runtime,
        semantic=semantic,
        source_revision=source_revision,
        loaded=loaded,
    )
    del model
    plan = loaded["plan_semantic_active8_decisions"](
        inventory,
        chunk_cache_plan=cache_plan,
        chunk_cache_witness=cache_witness,
        model_runtime_identity=runtime_identity,
        output_artifact_root=_require_artifact_path(
            output_artifact_root, field="output_artifact_root"
        ),
        artifact_root=ARTIFACT_ROOT,
    )
    plan_path = loaded["write_semantic_active8_decision_plan"](plan, artifact_root=ARTIFACT_ROOT)
    artifact_volume.commit()
    completed = loaded["completed_semantic_active8_decision_task_ids"](
        plan, artifact_root=ARTIFACT_ROOT
    )
    missing = [
        task["task_identity_sha256"]
        for task in plan["tasks"]
        if task["task_identity_sha256"] not in completed
    ]
    if missing:
        plan_artifact_path = _artifact_address(plan_path)
        results = list(
            decide_one_chunk.starmap(
                [(plan_artifact_path, task_identity, source_revision) for task_identity in missing]
            )
        )
        if len(results) != len(missing):
            raise RuntimeError("semantic Active8 map lost decision task results")
    return reduce_decisions.remote(plan, semantic_migration_completion, source_revision)


@app.local_entrypoint()
def main(
    semantic_migration_completion: str,
    semantic_chunk_cache_plan: str,
    semantic_chunk_cache_global_completion: str,
    expected_commit: str,
    output_artifact_root: str = OUTPUT_ARTIFACT_ROOT,
):
    """Wait for the CPU driver and print only non-authorizing evidence."""

    for field, value in (
        ("semantic_migration_completion", semantic_migration_completion),
        ("semantic_chunk_cache_plan", semantic_chunk_cache_plan),
        (
            "semantic_chunk_cache_global_completion",
            semantic_chunk_cache_global_completion,
        ),
        ("output_artifact_root", output_artifact_root),
    ):
        _require_artifact_path(value, field=field)
    source_revision = local_source_revision(expected_commit=expected_commit)
    result = driver.remote(
        semantic_migration_completion,
        semantic_chunk_cache_plan,
        semantic_chunk_cache_global_completion,
        output_artifact_root,
        source_revision,
    )
    print(
        json.dumps(
            {
                "semantic_migration_completion": semantic_migration_completion,
                "semantic_chunk_cache_plan": semantic_chunk_cache_plan,
                "semantic_chunk_cache_global_completion": (semantic_chunk_cache_global_completion),
                "output_artifact_root": output_artifact_root,
                "source_revision": source_revision,
                "max_map_containers": MAX_MAP_CONTAINERS,
                "training_launched": False,
                "gate_zero_authorized": False,
                "t1_authorized": False,
                "bounded_p50_authorized": False,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
