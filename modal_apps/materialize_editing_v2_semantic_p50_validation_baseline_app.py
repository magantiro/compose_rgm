"""Materialize the prospective semantic P50 validation baseline on Modal.

The CPU stage strictly reopens the source, prepared recipe, scratch runtime,
and combined successor cache before publishing and reopening the deterministic
validation-only inventory.  The GPU stage independently repeats those
reopeners, scores the untouched scratch model at the frozen validation times
through the production canonical-successor evaluator, publishes and reopens a
physical evaluation-environment receipt, and materializes and reopens the
baseline.

All outputs are immutable and content addressed.  This producer has no
optimizer, update, checkpoint-selection, training, or downstream authority.
Remote source attestation compares every serialized source byte and never
depends on a remote Git checkout.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/materialize_editing_v2_semantic_p50_validation_baseline_app.py"
SEMANTIC_MODEL_PROCESS_SOURCE = "configs/editing_gate_zero_semantic_model_process_v1.json"
P50_RECIPE_POLICY_SOURCE = "configs/editing_v2_semantic_p50_recipe_policy_v1.json"
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_p50_validation_baseline"
REQUEST_FILENAME = "SEMANTIC_P50_VALIDATION_BASELINE_REQUEST.json"
EVALUATION_ENVIRONMENT_FILENAME = "SEMANTIC_P50_VALIDATION_EVALUATION_ENVIRONMENT.json"
PRODUCER_COMPLETION_FILENAME = "SEMANTIC_P50_VALIDATION_BASELINE_PRODUCER_COMPLETE.json"
SOURCE_REVISION_SCHEMA = "compose.editing_v2.semantic_p50_validation_baseline_modal_source_revision"
REQUEST_SCHEMA = "compose.editing_v2.semantic_p50_validation_baseline_request"
PRODUCER_COMPLETION_SCHEMA = (
    "compose.editing_v2.semantic_p50_validation_baseline_producer_completion"
)
SCHEMA_VERSION = 1
REQUEST_STATUS = "FROZEN_CPU_GPU_VALIDATION_BASELINE_REQUEST_NO_AUTHORITY"
PRODUCER_COMPLETION_STATUS = "COMPLETE_VALIDATION_BASELINE_PRODUCER_NO_AUTHORITY"
GPU_TYPE = "A10G"
EVALUATION_BATCH_SIZE = 64

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "p500_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_INPUT_NAMES = (
    "prepared_recipe",
    "source_inventory",
    "migration_completion",
    "chunk_cache_plan",
    "chunk_cache_global_completion",
    "decision_plan",
    "decision_completion",
    "gate_zero_evidence",
    "t1_decision",
    "successor_cache_completion",
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
            "OMP_NUM_THREADS": "8",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
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

app = modal.App("compose-v4-editing-v2-semantic-p50-validation-baseline")
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


def _load_canonical(path: Path, *, field: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"{field} is absent: {source}")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{field} is unreadable: {source}") from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise RuntimeError(f"{field} must be canonical newline-terminated JSON")
    return payload


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _artifact_path(value: str, *, artifact_root: Path, field: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a /artifacts path")
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
        raise ValueError(f"{field} must be a normalized path below /artifacts")
    root = Path(artifact_root).resolve()
    result = (root / Path(*pure.parts[2:])).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"{field} resolves outside artifact_root")
    return result


def _artifact_address(path: Path, *, artifact_root: Path) -> str:
    root = Path(artifact_root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact resolves outside artifact_root")
    relative = resolved.relative_to(root)
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


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
            f"cannot establish validation-baseline source identity: git {' '.join(arguments)}"
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
        raise RuntimeError("validation-baseline serialized source inventory repeats a path")
    return tuple(paths)


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    return {relative: _file_sha(root / relative) for relative in _serialized_source_paths(root)}


def local_source_revision(*, expected_commit: str, repo_root: Path = ROOT) -> dict[str, Any]:
    """Bind a clean local commit and every byte serialized into the image."""

    if not isinstance(expected_commit, str) or _COMMIT_RE.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "validation-baseline production requires the exact clean committed worktree"
        )
    hashes = _serialized_source_hashes(root)
    tracked = set(_git(root, "ls-files").splitlines())
    if not hashes or not set(hashes).issubset(tracked):
        raise RuntimeError("every validation-baseline serialized source/config must be Git-tracked")
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": _sha(hashes),
    }
    return {**body, "source_revision_sha256": _sha(body)}


def _validate_source_revision(value: object, *, remote_root: Path) -> dict[str, Any]:
    """Attest serialized bytes remotely without consulting a remote .git tree."""

    if not isinstance(value, Mapping):
        raise TypeError("validation-baseline source revision must be an object")
    revision = dict(value)
    body = dict(revision)
    supplied = body.pop("source_revision_sha256", None)
    hashes = _serialized_source_hashes(remote_root)
    if (
        set(revision)
        != {
            "schema",
            "schema_version",
            "commit",
            "tree",
            "worktree_clean",
            "serialized_source_hashes",
            "serialized_source_hashes_sha256",
            "source_revision_sha256",
        }
        or revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SCHEMA_VERSION
        or revision.get("worktree_clean") is not True
        or not isinstance(revision.get("commit"), str)
        or _COMMIT_RE.fullmatch(revision["commit"]) is None
        or not isinstance(revision.get("tree"), str)
        or _COMMIT_RE.fullmatch(revision["tree"]) is None
        or revision.get("serialized_source_hashes") != hashes
        or revision.get("serialized_source_hashes_sha256") != _sha(hashes)
        or supplied != _sha(body)
    ):
        raise RuntimeError("validation-baseline serialized source bytes disagree")
    return revision


def _validate_input_addresses(value: object, *, artifact_root: Path) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_INPUT_NAMES):
        raise TypeError("validation-baseline input-address fields disagree")
    result = dict(value)
    for name in _INPUT_NAMES:
        path = _artifact_path(
            result[name], artifact_root=artifact_root, field=f"input_addresses.{name}"
        )
        if not path.is_file():
            raise RuntimeError(f"validation-baseline input {name} is absent: {path}")
    return result


def _snapshot_physical_inputs(
    input_addresses: Mapping[str, str], *, artifact_root: Path
) -> dict[str, dict[str, str]]:
    """Capture the exact ten physical input bytes before any scientific open."""

    addresses = _validate_input_addresses(input_addresses, artifact_root=artifact_root)
    return {
        name: {
            "artifact_path": addresses[name],
            "file_sha256": _file_sha(
                _artifact_path(
                    addresses[name],
                    artifact_root=artifact_root,
                    field=f"input_addresses.{name}",
                )
            ),
        }
        for name in _INPUT_NAMES
    }


def _require_unchanged_physical_inputs(
    snapshot: Mapping[str, Mapping[str, str]],
    *,
    input_addresses: Mapping[str, str],
    artifact_root: Path,
) -> None:
    """Fail if any input path or byte changed after the initial snapshot."""

    if dict(snapshot) != _snapshot_physical_inputs(
        input_addresses,
        artifact_root=artifact_root,
    ):
        raise RuntimeError(
            "validation-baseline physical inputs changed after their initial snapshot"
        )


def build_run_request(
    *,
    source_revision: Mapping[str, Any],
    input_records: Mapping[str, Mapping[str, str]],
    prepared_recipe_sha256: str,
    source_inventory_sha256: str,
    successor_cache_completion_sha256: str,
    scratch_initial_model_state_sha256: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Content-address one exact validation-only CPU/GPU execution."""

    if set(input_records) != set(_INPUT_NAMES):
        raise ValueError("validation-baseline input records disagree")
    normalized: dict[str, dict[str, str]] = {}
    for name in _INPUT_NAMES:
        record = input_records[name]
        if not isinstance(record, Mapping) or set(record) != {
            "artifact_path",
            "file_sha256",
        }:
            raise ValueError(f"validation-baseline input record {name} fields disagree")
        path = str(record["artifact_path"])
        _artifact_path(path, artifact_root=ARTIFACT_ROOT, field=name)
        normalized[name] = {
            "artifact_path": path,
            "file_sha256": _require_sha(record["file_sha256"], field=f"{name}.file_sha256"),
        }
    _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=ARTIFACT_ROOT,
        field="output_prefix",
    )
    body: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": REQUEST_STATUS,
        **_NO_AUTHORITY,
        "source_revision": dict(source_revision),
        "inputs": normalized,
        "prepared_recipe_sha256": _require_sha(
            prepared_recipe_sha256, field="prepared_recipe_sha256"
        ),
        "source_inventory_sha256": _require_sha(
            source_inventory_sha256, field="source_inventory_sha256"
        ),
        "successor_cache_completion_sha256": _require_sha(
            successor_cache_completion_sha256,
            field="successor_cache_completion_sha256",
        ),
        "scratch_initial_model_state_sha256": _require_sha(
            scratch_initial_model_state_sha256,
            field="scratch_initial_model_state_sha256",
        ),
        "evaluation": {
            "accelerator_class": "gpu",
            "modal_gpu_type": GPU_TYPE,
            "dtype": "torch.float32",
            "mixed_precision": False,
            "batch_size": EVALUATION_BATCH_SIZE,
            "deterministic_algorithms_required": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_tf32_allowed": False,
            "cuda_matmul_tf32_allowed": False,
            "cublas_workspace_config": ":4096:8",
            "nvidia_tf32_override": "0",
            "objective": "productive_embedded_canonical_successor_nll",
            "model_state": "untouched_scratch_initial_state",
        },
        "output_prefix": output_prefix,
    }
    return {**body, "run_identity_sha256": _sha(body)}


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    import numpy as np
    import rdkit
    import torch

    from compose_v4.chem.persistent_state_identity import (
        persistent_slot_state_sha256,
    )
    from compose_v4.data.editing_v2_semantic_capability_cells import (
        load_semantic_capability_cell_registry,
    )
    from compose_v4.data.immutable_artifact import write_bytes_if_absent
    from compose_v4.data.successor_fiber_cache import (
        SuccessorFiberCacheAddress,
        successor_fiber_cache_record_payload,
    )
    from compose_v4.experiments import (
        editing_v2_semantic_p50_successor_cache as cache,
    )
    from compose_v4.experiments import (
        editing_v2_semantic_p50_validation_baseline as baseline,
    )
    from compose_v4.experiments.editing_gate_zero_semantic_contract import (
        load_gate_zero_semantic_contract,
    )
    from compose_v4.experiments.editing_p50_gate import (
        state_dict_semantic_sha256,
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
    from compose_v4.experiments.factorized_mark_conditional import (
        FactorizedMarkCollator,
        FactorizedMarkExample,
    )
    from compose_v4.experiments.factorized_successor_training import (
        SuccessorTrainingError,
        forward_teacher_successor_batch,
    )

    return {
        "np": np,
        "rdkit": rdkit,
        "torch": torch,
        "baseline": baseline,
        "cache": cache,
        "load_registry": load_semantic_capability_cell_registry,
        "write_bytes_if_absent": write_bytes_if_absent,
        "SuccessorFiberCacheAddress": SuccessorFiberCacheAddress,
        "successor_fiber_cache_record_payload": successor_fiber_cache_record_payload,
        "persistent_slot_state_sha256": persistent_slot_state_sha256,
        "load_semantic_contract": load_gate_zero_semantic_contract,
        "state_dict_semantic_sha256": state_dict_semantic_sha256,
        "SemanticP50Prerequisites": SemanticP50Prerequisites,
        "validate_prerequisites": validate_semantic_p50_prerequisite_relationships,
        "SemanticP50SourceInventoryBinding": SemanticP50SourceInventoryBinding,
        "load_source": load_semantic_p50_source_inventory,
        "SemanticScratchModelConfig": SemanticScratchModelConfig,
        "build_scratch": build_semantic_scratch_runtime,
        "validate_gate_zero": validate_semantic_gate_zero_evidence_receipt,
        "validate_t1": validate_semantic_t1_capacity_decision,
        "FactorizedMarkCollator": FactorizedMarkCollator,
        "FactorizedMarkExample": FactorizedMarkExample,
        "SuccessorTrainingError": SuccessorTrainingError,
        "forward_teacher_successor_batch": forward_teacher_successor_batch,
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
        or loaded["state_dict_semantic_sha256"](scratch.model.state_dict())
        != scratch.initial_model_state_sha256
    ):
        raise RuntimeError("scratch runtime differs from the physical Gate 0 identity")
    scratch.model.eval()
    return scratch


def _open_exact_prerequisites(
    *,
    input_addresses: Mapping[str, str],
    artifact_root: Path,
    remote_root: Path,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen recipe, source, Gate 0/T1-bound scratch, and combined cache."""

    addresses = _validate_input_addresses(input_addresses, artifact_root=artifact_root)
    paths = {
        name: _artifact_path(
            addresses[name], artifact_root=artifact_root, field=f"input_addresses.{name}"
        )
        for name in _INPUT_NAMES
    }
    cache = loaded["cache"]
    prepared, prepared_file_sha256 = cache.load_semantic_p50_prepared_recipe(
        paths["prepared_recipe"]
    )
    prerequisites = loaded["SemanticP50Prerequisites"](**dict(prepared["prerequisites"]))
    source = loaded["load_source"](
        paths["source_inventory"],
        expected_binding=_source_binding(prepared["prerequisites"], loaded=loaded),
        migration_completion_path=paths["migration_completion"],
        chunk_cache_plan_path=paths["chunk_cache_plan"],
        chunk_cache_global_completion_path=paths["chunk_cache_global_completion"],
        decision_plan_path=paths["decision_plan"],
        decision_completion_path=paths["decision_completion"],
        artifact_root=artifact_root,
        repo_root=remote_root,
    )
    gate_file_sha256 = _file_sha(paths["gate_zero_evidence"])
    gate_zero = loaded["validate_gate_zero"](
        paths["gate_zero_evidence"], expected_file_sha256=gate_file_sha256
    )
    t1_payload = _load_canonical(paths["t1_decision"], field="semantic T1 decision")
    t1_decision = loaded["validate_t1"](
        t1_payload,
        decision_path=paths["t1_decision"],
        repo_root=remote_root,
        expected_gate_zero_evidence_file_sha256=gate_file_sha256,
        require_p50_go=True,
    )
    registry = loaded["load_registry"]()
    reconstructed = loaded["validate_prerequisites"](
        source=source,
        gate_zero=gate_zero,
        gate_zero_file_sha256=gate_file_sha256,
        t1_decision=t1_decision,
        t1_decision_file_sha256=_file_sha(paths["t1_decision"]),
        registry=registry,
    )
    if reconstructed != prerequisites:
        raise RuntimeError("prepared recipe differs from reopened source, Gate 0, or T1")
    scratch = _scratch_runtime(gate_zero, loaded=loaded, remote_root=remote_root)
    if scratch.initial_model_state_sha256 != prerequisites.scratch_initial_model_state_sha256:
        raise RuntimeError("prepared recipe names another scratch model state")
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
    opened_cache = cache.open_semantic_p50_successor_cache(
        paths["successor_cache_completion"],
        artifact_root=artifact_root,
        repo_root=remote_root,
        expected_identity=expected_identity,
        source_reopen_paths=source_paths,
        scratch_runtime=scratch,
        registry=registry,
    )
    if (
        opened_cache.completion.get("prepared_recipe_sha256") != prepared["prepared_recipe_sha256"]
        or opened_cache.completion.get("scratch_initial_model_state_sha256")
        != scratch.initial_model_state_sha256
    ):
        raise RuntimeError("opened successor cache differs from recipe or scratch")
    return {
        "paths": paths,
        "prepared": prepared,
        "prepared_file_sha256": prepared_file_sha256,
        "source": source,
        "gate_zero": gate_zero,
        "t1_decision": t1_decision,
        "registry": registry,
        "scratch": scratch,
        "cache": opened_cache,
    }


def _request_for_opened(
    *,
    revision: Mapping[str, Any],
    input_snapshot: Mapping[str, Mapping[str, str]],
    opened: Mapping[str, Any],
    output_prefix: str,
) -> dict[str, Any]:
    return build_run_request(
        source_revision=revision,
        input_records=input_snapshot,
        prepared_recipe_sha256=opened["prepared"]["prepared_recipe_sha256"],
        source_inventory_sha256=opened["source"].binding.source_inventory_sha256,
        successor_cache_completion_sha256=opened["cache"].completion["completion_sha256"],
        scratch_initial_model_state_sha256=opened["scratch"].initial_model_state_sha256,
        output_prefix=output_prefix,
    )


def _run_root(request: Mapping[str, Any], *, artifact_root: Path, output_prefix: str) -> Path:
    prefix_parent = _artifact_path(
        f"{output_prefix}/placeholder",
        artifact_root=artifact_root,
        field="output_prefix",
    ).parent
    return prefix_parent / str(request["run_identity_sha256"])


def _collator(model: Any, *, loaded: Mapping[str, Any]) -> Any:
    # Build from the immutable capability object rather than a keyword list: the list dropped the
    # atom-delete mode, so a Process-V2 model could not be evaluated through this app at all.
    return loaded["FactorizedMarkCollator"].from_capabilities(
        model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
    )


def _address_sha(address: Any) -> str:
    return _sha(asdict(address))


def _validation_states(inventory: Any, *, source: Any, loaded: Mapping[str, Any]) -> dict[str, Any]:
    """Recover exact persistent-slot states for the sealed validation addresses."""

    expected = {candidate.address: candidate for candidate in inventory.candidates}
    states: dict[str, Any] = {}
    observed_addresses: set[Any] = set()
    address_class = loaded["SuccessorFiberCacheAddress"]
    for trace in source.index.iter_accepted_traces_for_partition("validation"):
        for transition in source.index.accepted_transitions_for(trace):
            address = address_class.from_packed_trace(
                transition.addressed_trace.address,
                progress_index=transition.step_index,
            )
            candidate = expected.get(address)
            if candidate is None:
                continue
            if address in observed_addresses:
                raise RuntimeError("validation source repeats an inventory address")
            state = transition.addressed_trace.path.state_at(transition.step_index)
            state_sha256 = loaded["persistent_slot_state_sha256"](state)
            if state_sha256 != candidate.source_state_sha256:
                raise RuntimeError("validation source state differs from inventory")
            states[_address_sha(address)] = state
            observed_addresses.add(address)
    if observed_addresses != set(expected) or len(states) != len(expected):
        raise RuntimeError("validation source omits an inventory state")
    return states


def _evaluate_scratch(
    *,
    inventory: Any,
    successor_cache: Any,
    scratch_runtime: Any,
    states: Mapping[str, Any],
    loaded: Mapping[str, Any],
    batch_size: int = EVALUATION_BATCH_SIZE,
) -> tuple[Any, ...]:
    """Evaluate canonical-successor NLLs without updates or mark reconstruction."""

    torch = loaded["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("semantic P50 baseline GPU is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = scratch_runtime.model.to(torch.device("cuda"))
    model.eval()
    initial_state_sha256 = scratch_runtime.initial_model_state_sha256
    if loaded["state_dict_semantic_sha256"](model.state_dict()) != initial_state_sha256:
        raise RuntimeError("moving scratch model to the evaluator changed its state")
    collator = _collator(model, loaded=loaded)
    evaluations: list[Any] = []
    candidates = tuple(inventory.candidates)
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch_candidates = candidates[start : start + batch_size]
            examples: list[Any] = []
            fibers: list[Any] = []
            records: list[Any] = []
            for candidate in batch_candidates:
                address_sha256 = _address_sha(candidate.address)
                record = successor_cache.record_for_address(candidate.address)
                if (
                    record.teacher_fiber is None
                    or record.source_state_sha256 != candidate.source_state_sha256
                    or address_sha256 not in states
                ):
                    raise RuntimeError("validation cache/state teacher identity disagrees")
                examples.append(
                    loaded["FactorizedMarkExample"](
                        state=states[address_sha256],
                        time=float.fromhex(candidate.time_hex),
                        teacher_action=None,
                        teacher_rule_name=candidate.family,
                        teacher_rate=1.0,
                        importance_weight=1.0,
                    )
                )
                fibers.append(record.teacher_fiber)
                records.append(record)
            batch = collator(examples).to(model.device)
            try:
                prediction = loaded["forward_teacher_successor_batch"](model, batch, tuple(fibers))
            except (ValueError, RuntimeError, loaded["SuccessorTrainingError"]) as error:
                raise RuntimeError(
                    "production canonical-successor validation failed closed"
                ) from error
            log_probabilities = prediction.selected_productive_successor_log_probability
            if log_probabilities.shape != (len(batch_candidates),) or not bool(
                torch.isfinite(log_probabilities).all()
            ):
                raise RuntimeError("validation scorer returned nonfinite probabilities")
            for candidate, record, log_probability in zip(
                batch_candidates,
                records,
                log_probabilities.detach().cpu(),
                strict=True,
            ):
                nll = -float(log_probability)
                if not math.isfinite(nll) or nll < 0.0:
                    raise RuntimeError("validation canonical-successor NLL is invalid")
                evaluations.append(
                    loaded["baseline"].SemanticP50ValidationEvaluation(
                        address=candidate.address,
                        family=candidate.family,
                        semantic_cell_id=candidate.semantic_cell_id,
                        cache_record_sha256=_sha(
                            loaded["successor_fiber_cache_record_payload"](record)
                        ),
                        time_hex=candidate.time_hex,
                        canonical_successor_nll_nats=nll,
                    )
                )
    if (
        len(evaluations) != len(candidates)
        or loaded["state_dict_semantic_sha256"](model.state_dict()) != initial_state_sha256
    ):
        raise RuntimeError("validation evaluation changed or incompletely scored scratch")
    return tuple(evaluations)


def _execution_environment(scratch_runtime: Any, *, loaded: Mapping[str, Any]) -> dict[str, Any]:
    torch = loaded["torch"]
    parameter = next(scratch_runtime.model.parameters())
    device = parameter.device
    if device.type != "cuda":
        raise RuntimeError("evaluation environment requires the actual GPU model")
    capability = torch.cuda.get_device_capability(device)
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    if cuda_version is None or cudnn_version is None:
        raise RuntimeError("GPU evaluation lacks CUDA or cuDNN runtime identity")
    body: dict[str, Any] = {
        "hardware_class": platform.machine(),
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": f"{capability[0]}.{capability[1]}",
        "accelerator_class": "gpu",
        "dtype": str(parameter.dtype),
        "mixed_precision": False,
        "batch_size": EVALUATION_BATCH_SIZE,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(cuda_version),
        "cudnn_version": str(cudnn_version),
        "rdkit_version": str(loaded["rdkit"].__version__),
        "numpy_version": str(loaded["np"].__version__),
    }
    return {**body, "environment_sha256": _sha(body)}


def _prepare_inventory_impl(
    *,
    source_revision: Mapping[str, Any],
    input_addresses: Mapping[str, str],
    output_prefix: str,
    artifact_root: Path,
    remote_root: Path,
    stage_commit: Any | None,
) -> dict[str, Any]:
    revision = _validate_source_revision(source_revision, remote_root=remote_root)
    loaded = _imports(remote_root)
    addresses = _validate_input_addresses(input_addresses, artifact_root=artifact_root)
    input_snapshot = _snapshot_physical_inputs(addresses, artifact_root=artifact_root)
    opened = _open_exact_prerequisites(
        input_addresses=addresses,
        artifact_root=artifact_root,
        remote_root=remote_root,
        loaded=loaded,
    )
    _require_unchanged_physical_inputs(
        input_snapshot,
        input_addresses=addresses,
        artifact_root=artifact_root,
    )
    request = _request_for_opened(
        revision=revision,
        input_snapshot=input_snapshot,
        opened=opened,
        output_prefix=output_prefix,
    )
    run_root = _run_root(request, artifact_root=artifact_root, output_prefix=output_prefix)
    request_path = run_root / REQUEST_FILENAME
    created = loaded["write_bytes_if_absent"](request_path, _canonical_bytes(request, newline=True))
    if created and stage_commit is not None:
        stage_commit()
    baseline = loaded["baseline"]
    inventory_completion_path = baseline.materialize_semantic_p50_validation_inventory(
        prepared_recipe_path=opened["paths"]["prepared_recipe"],
        source=opened["source"],
        scratch_runtime=opened["scratch"],
        successor_cache=opened["cache"],
        artifact_root=artifact_root,
        output_root=run_root / "inventory",
        recipe_policy_path=remote_root / P50_RECIPE_POLICY_SOURCE,
        registry=opened["registry"],
    )
    verified_inventory = baseline.open_semantic_p50_validation_inventory(
        inventory_completion_path,
        prepared_recipe_path=opened["paths"]["prepared_recipe"],
        source=opened["source"],
        scratch_runtime=opened["scratch"],
        successor_cache=opened["cache"],
        artifact_root=artifact_root,
        recipe_policy_path=remote_root / P50_RECIPE_POLICY_SOURCE,
        registry=opened["registry"],
    )
    _require_unchanged_physical_inputs(
        input_snapshot,
        input_addresses=addresses,
        artifact_root=artifact_root,
    )
    if stage_commit is not None:
        stage_commit()
    return {
        "run_identity_sha256": request["run_identity_sha256"],
        "run_root": _artifact_address(run_root, artifact_root=artifact_root),
        "request_artifact_path": _artifact_address(request_path, artifact_root=artifact_root),
        "inventory_completion_artifact_path": _artifact_address(
            inventory_completion_path, artifact_root=artifact_root
        ),
        "validation_candidate_count": len(verified_inventory.candidates),
        "training_launched": False,
        "training_authorized": False,
        "bounded_p50_authorized": False,
    }


def _evaluate_baseline_impl(
    *,
    source_revision: Mapping[str, Any],
    input_addresses: Mapping[str, str],
    inventory_completion: str,
    request_artifact_path: str,
    output_prefix: str,
    artifact_root: Path,
    remote_root: Path,
    stage_commit: Any | None,
) -> dict[str, Any]:
    revision = _validate_source_revision(source_revision, remote_root=remote_root)
    loaded = _imports(remote_root)
    addresses = _validate_input_addresses(input_addresses, artifact_root=artifact_root)
    input_snapshot = _snapshot_physical_inputs(addresses, artifact_root=artifact_root)
    opened = _open_exact_prerequisites(
        input_addresses=addresses,
        artifact_root=artifact_root,
        remote_root=remote_root,
        loaded=loaded,
    )
    _require_unchanged_physical_inputs(
        input_snapshot,
        input_addresses=addresses,
        artifact_root=artifact_root,
    )
    request = _request_for_opened(
        revision=revision,
        input_snapshot=input_snapshot,
        opened=opened,
        output_prefix=output_prefix,
    )
    run_root = _run_root(request, artifact_root=artifact_root, output_prefix=output_prefix)
    request_path = _artifact_path(
        request_artifact_path, artifact_root=artifact_root, field="request_artifact_path"
    )
    if (
        request_path != run_root / REQUEST_FILENAME
        or _load_canonical(request_path, field="validation-baseline request") != request
    ):
        raise RuntimeError("GPU stage request differs from the CPU content address")
    inventory_path = _artifact_path(
        inventory_completion,
        artifact_root=artifact_root,
        field="inventory_completion",
    )
    if not inventory_path.is_relative_to(run_root / "inventory"):
        raise RuntimeError("validation inventory lies outside the request run root")
    baseline = loaded["baseline"]
    policy_path = remote_root / P50_RECIPE_POLICY_SOURCE
    inventory = baseline.open_semantic_p50_validation_inventory(
        inventory_path,
        prepared_recipe_path=opened["paths"]["prepared_recipe"],
        source=opened["source"],
        scratch_runtime=opened["scratch"],
        successor_cache=opened["cache"],
        artifact_root=artifact_root,
        recipe_policy_path=policy_path,
        registry=opened["registry"],
    )
    states = _validation_states(inventory, source=opened["source"], loaded=loaded)
    _require_unchanged_physical_inputs(
        input_snapshot,
        input_addresses=addresses,
        artifact_root=artifact_root,
    )
    evaluations = _evaluate_scratch(
        inventory=inventory,
        successor_cache=opened["cache"],
        scratch_runtime=opened["scratch"],
        states=states,
        loaded=loaded,
    )
    environment = _execution_environment(opened["scratch"], loaded=loaded)
    receipt = baseline.build_semantic_p50_validation_evaluation_environment_receipt(
        repo_root=remote_root,
        execution_environment=environment,
        scratch_runtime=opened["scratch"],
        successor_cache=opened["cache"],
        evaluations=evaluations,
    )
    environment_path = (
        run_root
        / "evaluation_environment"
        / receipt["receipt_sha256"]
        / EVALUATION_ENVIRONMENT_FILENAME
    )
    environment_created = loaded["write_bytes_if_absent"](
        environment_path, _canonical_bytes(receipt, newline=True)
    )
    if environment_created and stage_commit is not None:
        stage_commit()
    verified_environment = baseline.open_semantic_p50_validation_evaluation_environment_receipt(
        environment_path,
        artifact_root=artifact_root,
        repo_root=remote_root,
        scratch_runtime=opened["scratch"],
        successor_cache=opened["cache"],
        evaluations=evaluations,
    )
    baseline_completion_path = baseline.materialize_semantic_p50_validation_baseline(
        inventory_completion_path=inventory_path,
        prepared_recipe_path=opened["paths"]["prepared_recipe"],
        source=opened["source"],
        scratch_runtime=opened["scratch"],
        artifact_root=artifact_root,
        evaluations=evaluations,
        successor_cache=opened["cache"],
        evaluation_environment_receipt=verified_environment,
        repo_root=remote_root,
        output_root=run_root / "baseline",
        recipe_policy_path=policy_path,
        registry=opened["registry"],
    )
    if stage_commit is not None:
        stage_commit()
    verified_baseline = baseline.open_semantic_p50_validation_baseline(
        baseline_completion_path,
        inventory_completion_path=inventory_path,
        prepared_recipe_path=opened["paths"]["prepared_recipe"],
        source=opened["source"],
        scratch_runtime=opened["scratch"],
        artifact_root=artifact_root,
        successor_cache=opened["cache"],
        evaluation_environment_receipt=verified_environment,
        repo_root=remote_root,
        recipe_policy_path=policy_path,
        registry=opened["registry"],
    )
    _require_unchanged_physical_inputs(
        input_snapshot,
        input_addresses=addresses,
        artifact_root=artifact_root,
    )
    completion_body: dict[str, Any] = {
        "schema": PRODUCER_COMPLETION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": PRODUCER_COMPLETION_STATUS,
        **_NO_AUTHORITY,
        "training_launched": False,
        "run_identity_sha256": request["run_identity_sha256"],
        "request_artifact_path": request_artifact_path,
        "request_file_sha256": _file_sha(request_path),
        "inventory_completion_artifact_path": inventory_completion,
        "inventory_completion_file_sha256": _file_sha(inventory_path),
        "inventory_completion_sha256": inventory.completion["completion_sha256"],
        "evaluation_environment_artifact_path": _artifact_address(
            environment_path, artifact_root=artifact_root
        ),
        "evaluation_environment_file_sha256": _file_sha(environment_path),
        "evaluation_environment_receipt_sha256": receipt["receipt_sha256"],
        "baseline_completion_artifact_path": _artifact_address(
            baseline_completion_path, artifact_root=artifact_root
        ),
        "baseline_completion_file_sha256": _file_sha(baseline_completion_path),
        "baseline_completion_sha256": verified_baseline.completion["completion_sha256"],
        "baseline_result_sha256": verified_baseline.result["result_sha256"],
        "evaluated_model_state_sha256": opened["scratch"].initial_model_state_sha256,
        "example_count": len(evaluations),
        "source_revision_sha256": revision["source_revision_sha256"],
    }
    producer_completion = {
        **completion_body,
        "completion_sha256": _sha(completion_body),
    }
    producer_completion_path = (
        run_root
        / "producer_completions"
        / producer_completion["completion_sha256"]
        / PRODUCER_COMPLETION_FILENAME
    )
    completion_created = loaded["write_bytes_if_absent"](
        producer_completion_path,
        _canonical_bytes(producer_completion, newline=True),
    )
    if completion_created and stage_commit is not None:
        stage_commit()
    if (
        _load_canonical(
            producer_completion_path,
            field="validation-baseline producer completion",
        )
        != producer_completion
    ):
        raise RuntimeError("producer completion differs after immutable publication")
    _require_unchanged_physical_inputs(
        input_snapshot,
        input_addresses=addresses,
        artifact_root=artifact_root,
    )
    return {
        "run_identity_sha256": request["run_identity_sha256"],
        "producer_completion_artifact_path": _artifact_address(
            producer_completion_path, artifact_root=artifact_root
        ),
        "inventory_completion_artifact_path": inventory_completion,
        "evaluation_environment_artifact_path": _artifact_address(
            environment_path, artifact_root=artifact_root
        ),
        "baseline_completion_artifact_path": _artifact_address(
            baseline_completion_path, artifact_root=artifact_root
        ),
        "example_count": len(evaluations),
        "environment_reused": not environment_created,
        "producer_completion_reused": not completion_created,
        "training_launched": False,
        "training_authorized": False,
        "bounded_p50_authorized": False,
    }


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=24 * 3600,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_validation_inventory(
    *,
    source_revision: dict[str, Any],
    input_addresses: dict[str, str],
    output_prefix: str,
) -> dict[str, Any]:
    artifact_volume.reload()
    result = _prepare_inventory_impl(
        source_revision=source_revision,
        input_addresses=input_addresses,
        output_prefix=output_prefix,
        artifact_root=ARTIFACT_ROOT,
        remote_root=REMOTE_ROOT,
        stage_commit=artifact_volume.commit,
    )
    artifact_volume.commit()
    return result


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=8.0,
    memory=65536,
    timeout=24 * 3600,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def evaluate_validation_baseline(
    *,
    source_revision: dict[str, Any],
    input_addresses: dict[str, str],
    inventory_completion: str,
    request_artifact_path: str,
    output_prefix: str,
) -> dict[str, Any]:
    artifact_volume.reload()
    result = _evaluate_baseline_impl(
        source_revision=source_revision,
        input_addresses=input_addresses,
        inventory_completion=inventory_completion,
        request_artifact_path=request_artifact_path,
        output_prefix=output_prefix,
        artifact_root=ARTIFACT_ROOT,
        remote_root=REMOTE_ROOT,
        stage_commit=artifact_volume.commit,
    )
    artifact_volume.commit()
    return result


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
    successor_cache_completion: str,
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
        "successor_cache_completion": successor_cache_completion,
    }
    prepared = prepare_validation_inventory.remote(
        source_revision=source_revision,
        input_addresses=input_addresses,
        output_prefix=output_prefix,
    )
    result = evaluate_validation_baseline.remote(
        source_revision=source_revision,
        input_addresses=input_addresses,
        inventory_completion=prepared["inventory_completion_artifact_path"],
        request_artifact_path=prepared["request_artifact_path"],
        output_prefix=output_prefix,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "EVALUATION_BATCH_SIZE",
    "GPU_TYPE",
    "OUTPUT_PREFIX",
    "app",
    "build_run_request",
    "evaluate_validation_baseline",
    "local_source_revision",
    "main",
    "prepare_validation_inventory",
]
