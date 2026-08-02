"""Run exactly one permit-authorized semantic Editing-V2 P50 pilot on Modal.

The local entrypoint attests one clean committed source tree and every source or
configuration file serialized into the image.  The GPU worker validates those
bytes again, strictly reopens the content-addressed P50-only execution permit
and every physical prerequisite behind it, observes the live container, and
atomically reserves the no-resume output before moving the scratch model to
CUDA.  It executes only the frozen 50 by 64 fp32 pilot and publishes one
content-addressed result/checkpoint/completion directory.  It grants no P500,
checkpoint-selection, long-training, or final-test authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_editing_v2_semantic_p50_app.py"
TRAINER_SOURCE = "src/compose_v4/experiments/editing_v2_semantic_p50_runner.py"
SEMANTIC_MODEL_PROCESS_SOURCE = (
    "configs/editing_gate_zero_semantic_model_process_v1.json"
)
IMAGE_SOURCE_DIRECTORIES = ("src", "configs")
APP_NAME = "compose-v4-editing-v2-semantic-p50"
MODAL_GPU_TYPE = "A10G"
PREFLIGHT_CONTRACT_ROOT_RELATIVE = "editing_v2/semantic_p50/execution_contracts"
SOURCE_REVISION_SCHEMA = "compose.editing_v2.semantic_p50_modal_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 2

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_BUILD_SPEC = {
    "base": "modal.Image.debian_slim",
    "python": "3.11",
    "pip": [
        "torch==2.4.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "networkx==3.3",
        "rdkit==2024.3.5",
    ],
    "gpu": MODAL_GPU_TYPE,
    "cpu": 8.0,
    "memory_mb": 65536,
    "dtype": "float32",
    "mixed_precision": False,
    "cublas_workspace_config": ":4096:8",
    "nvidia_tf32_override": "0",
}


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
            ("git", "-C", str(Path(root).resolve()), *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot establish semantic P50 Git identity: git {' '.join(arguments)}"
        ) from error


def _serialized_source_paths(root: Path) -> tuple[str, ...]:
    paths = [LAUNCHER_SOURCE]
    for source_directory in IMAGE_SOURCE_DIRECTORIES:
        paths.extend(
            path.relative_to(root).as_posix()
            for path in sorted((Path(root) / source_directory).rglob("*"))
            if path.is_file()
            and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
        )
    if len(paths) != len(set(paths)):
        raise RuntimeError("semantic P50 serialized source inventory repeats a path")
    return tuple(paths)


def _serialized_source_hashes(root: Path) -> dict[str, str]:
    return {
        relative: _file_sha(Path(root) / relative)
        for relative in _serialized_source_paths(Path(root))
    }


def _image_content_sha256(source_hashes: Mapping[str, str]) -> str:
    return _sha(
        {
            "image_build_spec": _IMAGE_BUILD_SPEC,
            "serialized_source_hashes": dict(source_hashes),
        }
    )


def _execution_source_revision_module(root: Path) -> Any:
    source_root = str(Path(root).resolve() / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from compose_v4.experiments import editing_v2_execution_source_revision

    return editing_v2_execution_source_revision


def local_source_revision(
    *, expected_commit: str, repo_root: Path = ROOT
) -> dict[str, Any]:
    """Bind a clean Git revision and every byte serialized into the image."""

    if (
        not isinstance(expected_commit, str)
        or _COMMIT_RE.fullmatch(expected_commit) is None
    ):
        raise RuntimeError("expected_commit must be a full lowercase Git commit")
    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if commit != expected_commit or dirty:
        raise RuntimeError(
            "semantic P50 launch requires the exact clean committed worktree"
        )
    serialized = _serialized_source_hashes(root)
    tracked = set(_git(root, "ls-files").splitlines())
    if not serialized or not set(serialized).issubset(tracked):
        raise RuntimeError(
            "every semantic P50 serialized source/config must be tracked"
        )
    execution_revision = _execution_source_revision_module(
        root
    ).build_execution_source_revision(commit=commit, tree=tree)
    body: dict[str, Any] = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "execution_source_revision": execution_revision,
        "serialized_source_hashes": serialized,
        "serialized_source_hashes_sha256": _sha(serialized),
        "image_build_spec": _IMAGE_BUILD_SPEC,
        "image_content_sha256": _image_content_sha256(serialized),
    }
    return {**body, "source_revision_sha256": _sha(body)}


def _validate_source_revision(
    value: object, *, remote_root: Path = REMOTE_ROOT
) -> dict[str, Any]:
    fields = {
        "schema",
        "schema_version",
        "execution_source_revision",
        "serialized_source_hashes",
        "serialized_source_hashes_sha256",
        "image_build_spec",
        "image_content_sha256",
        "source_revision_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError("semantic P50 source revision fields disagree")
    revision = dict(value)
    body = dict(revision)
    supplied = body.pop("source_revision_sha256")
    execution = revision.get("execution_source_revision")
    live_hashes = _serialized_source_hashes(Path(remote_root))
    try:
        validated_execution = _execution_source_revision_module(
            remote_root
        ).validate_execution_source_revision(execution)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "semantic P50 execution-source revision disagrees"
        ) from error
    if (
        revision.get("schema") != SOURCE_REVISION_SCHEMA
        or revision.get("schema_version") != SOURCE_REVISION_SCHEMA_VERSION
        or supplied != _sha(body)
        or validated_execution != execution
        or revision.get("serialized_source_hashes") != live_hashes
        or revision.get("serialized_source_hashes_sha256") != _sha(live_hashes)
        or revision.get("image_build_spec") != _IMAGE_BUILD_SPEC
        or revision.get("image_content_sha256") != _image_content_sha256(live_hashes)
    ):
        raise RuntimeError("semantic P50 serialized source revision disagrees")
    return revision


_LOCAL_SOURCE_HASHES = _serialized_source_hashes(ROOT)
_LOCAL_IMAGE_CONTENT_SHA256 = _image_content_sha256(_LOCAL_SOURCE_HASHES)
IMAGE_REFERENCE = f"{APP_NAME}@sha256:{_LOCAL_IMAGE_CONTENT_SHA256}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_IMAGE_BUILD_SPEC["pip"])
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
            "COMPOSE_IMAGE_CONTENT_SHA256": _LOCAL_IMAGE_CONTENT_SHA256,
            "MODAL_GPU_TYPE": MODAL_GPU_TYPE,
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

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(
    "compose-v4-artifacts", create_if_missing=False
)


def _artifact_path(value: str, *, field: str) -> Path:
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
    root = Path(ARTIFACT_ROOT).absolute()
    result = (root / Path(*pure.parts[2:])).absolute()
    resolved_root = root.resolve()
    resolved_result = result.resolve()
    if (
        not result.is_relative_to(root)
        or not resolved_result.is_relative_to(resolved_root)
        or resolved_result != result
    ):
        raise RuntimeError(f"{field} resolves outside /artifacts")
    return result


def _artifact_address(path: Path) -> str:
    relative = Path(path).resolve().relative_to(ARTIFACT_ROOT.resolve())
    return str(PurePosixPath("/artifacts") / PurePosixPath(relative.as_posix()))


def _load_canonical(path: Path, *, field: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{field} is absent or unreadable") from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise RuntimeError(f"{field} must be canonical newline-terminated JSON")
    return payload


def _content_addressed_contract_root(kind: str) -> Path:
    if kind not in {"runtime", "environment", "launch", "permit"}:
        raise RuntimeError("semantic P50 contract kind is unsupported")
    namespace = (
        Path(ARTIFACT_ROOT).absolute() / PREFLIGHT_CONTRACT_ROOT_RELATIVE
    ).absolute()
    destination = (namespace / kind).absolute()
    resolved_namespace = namespace.resolve()
    resolved_destination = destination.resolve()
    if (
        not destination.is_relative_to(namespace)
        or not resolved_destination.is_relative_to(resolved_namespace)
        or resolved_destination != destination
    ):
        raise RuntimeError(
            "semantic P50 content-addressed path escapes through a symbolic link"
        )
    return destination


def _content_addressed_contract_path(
    *, kind: str, identity_sha256: str, filename: str
) -> Path:
    if (
        not isinstance(identity_sha256, str)
        or _SHA_RE.fullmatch(identity_sha256) is None
    ):
        raise RuntimeError("semantic P50 contract identity must be a lowercase SHA-256")
    if not filename or Path(filename).name != filename:
        raise RuntimeError("semantic P50 contract filename is malformed")
    root = _content_addressed_contract_root(kind)
    destination = (root / identity_sha256 / filename).absolute()
    if destination.resolve() != destination:
        raise RuntimeError(
            "semantic P50 content-addressed path escapes through a symbolic link"
        )
    return destination


def _publish_content_addressed_contract(
    *,
    kind: str,
    identity_sha256: str,
    filename: str,
    payload: Mapping[str, Any],
    loaded: Mapping[str, Any],
) -> Path:
    destination = _content_addressed_contract_path(
        kind=kind,
        identity_sha256=identity_sha256,
        filename=filename,
    )
    content = loaded["canonical_contract_bytes"](payload)
    try:
        loaded["write_bytes_if_absent"](destination, content)
    except loaded["ImmutableArtifactError"] as error:
        raise RuntimeError(
            f"immutable semantic P50 contract collision: {destination}"
        ) from error
    return destination


def _preflight_imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    source_root = str(Path(remote_root) / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from compose_v4.experiments.editing_gate_zero_semantic_contract import (
        load_gate_zero_semantic_contract,
    )
    from compose_v4.data.immutable_artifact import (
        ImmutableArtifactError,
        write_bytes_if_absent,
    )
    from compose_v4.experiments.editing_v2_semantic_p50_execution_contracts import (
        ENVIRONMENT_FILENAME,
        ENVIRONMENT_RECEIPT_FILENAME,
        LAUNCH_FILENAME,
        PERMIT_FILENAME,
        RUNTIME_FILENAME,
        SemanticP50ExecutionPrerequisitePaths,
        build_semantic_p50_environment_contract,
        build_semantic_p50_launch_projection,
        build_semantic_p50_observed_environment_receipt,
        build_semantic_p50_runtime_contract,
        canonical_semantic_p50_contract_bytes,
        materialize_semantic_p50_execution_permit,
        observe_semantic_p50_physical_environment,
        open_semantic_p50_environment_contract,
        open_semantic_p50_execution_permit,
        open_semantic_p50_execution_prerequisites,
        open_semantic_p50_launch_projection,
        open_semantic_p50_observed_environment_receipt,
        open_semantic_p50_runtime_contract,
        publish_semantic_p50_observed_environment_receipt,
        require_semantic_p50_output_available,
    )
    from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
        SemanticP50SourceInventoryBinding,
    )
    from compose_v4.experiments.editing_v2_semantic_p50_successor_cache import (
        load_semantic_p50_prepared_recipe,
    )
    from compose_v4.experiments.editing_v2_semantic_runtime import (
        SemanticScratchModelConfig,
        build_semantic_scratch_runtime,
    )

    return {
        "load_semantic_contract": load_gate_zero_semantic_contract,
        "SemanticP50ExecutionPrerequisitePaths": SemanticP50ExecutionPrerequisitePaths,
        "RUNTIME_FILENAME": RUNTIME_FILENAME,
        "ENVIRONMENT_FILENAME": ENVIRONMENT_FILENAME,
        "LAUNCH_FILENAME": LAUNCH_FILENAME,
        "PERMIT_FILENAME": PERMIT_FILENAME,
        "ENVIRONMENT_RECEIPT_FILENAME": ENVIRONMENT_RECEIPT_FILENAME,
        "build_runtime_contract": build_semantic_p50_runtime_contract,
        "open_runtime_contract": open_semantic_p50_runtime_contract,
        "build_environment_contract": build_semantic_p50_environment_contract,
        "open_environment_contract": open_semantic_p50_environment_contract,
        "build_launch_projection": build_semantic_p50_launch_projection,
        "open_launch_projection": open_semantic_p50_launch_projection,
        "materialize_execution_permit": materialize_semantic_p50_execution_permit,
        "open_execution_permit": open_semantic_p50_execution_permit,
        "open_execution_prerequisites": open_semantic_p50_execution_prerequisites,
        "observe_physical_environment": observe_semantic_p50_physical_environment,
        "canonical_contract_bytes": canonical_semantic_p50_contract_bytes,
        "ImmutableArtifactError": ImmutableArtifactError,
        "write_bytes_if_absent": write_bytes_if_absent,
        "build_observed_environment_receipt": build_semantic_p50_observed_environment_receipt,
        "publish_observed_environment_receipt": publish_semantic_p50_observed_environment_receipt,
        "open_observed_environment_receipt": open_semantic_p50_observed_environment_receipt,
        "require_output_available": require_semantic_p50_output_available,
        "SemanticP50SourceInventoryBinding": SemanticP50SourceInventoryBinding,
        "load_prepared_recipe": load_semantic_p50_prepared_recipe,
        "SemanticScratchModelConfig": SemanticScratchModelConfig,
        "build_scratch_runtime": build_semantic_scratch_runtime,
    }


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    loaded = _preflight_imports(remote_root)
    import torch

    from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
        authorize_semantic_p50_recipe,
    )
    from compose_v4.experiments.editing_v2_semantic_p50_runner import (
        open_semantic_p50_run_artifacts,
        publish_semantic_p50_run_artifacts,
        run_semantic_p50,
    )

    return {
        **loaded,
        "torch": torch,
        "authorize_recipe": authorize_semantic_p50_recipe,
        "run_p50": run_semantic_p50,
        "publish_run_artifacts": publish_semantic_p50_run_artifacts,
        "open_run_artifacts": open_semantic_p50_run_artifacts,
    }


def _scratch_runtime(
    decision_plan_path: Path, *, loaded: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    plan = _load_canonical(decision_plan_path, field="semantic Active8 decision plan")
    runtime = plan.get("model_runtime_identity")
    if not isinstance(runtime, Mapping):
        raise TypeError("semantic Active8 decision plan lacks model runtime identity")
    architecture = runtime.get("architecture")
    if not isinstance(architecture, Mapping):
        raise TypeError("semantic Active8 model architecture is absent")
    contract = loaded["load_semantic_contract"](
        REMOTE_ROOT / SEMANTIC_MODEL_PROCESS_SOURCE
    )
    config = loaded["SemanticScratchModelConfig"](
        initialization_seed=int(runtime["initialization_seed"]),
        max_atoms=int(architecture["max_atoms"]),
        hidden_dim=int(architecture["hidden_dim"]),
        message_passing_steps=int(architecture["message_passing_steps"]),
        mark_dim=int(architecture["mark_dim"]),
        dtype=str(architecture["dtype"]),
        atom_vocabulary_class_count=int(architecture["atom_vocabulary_class_count"]),
        catalog_fingerprint=str(architecture["catalog_fingerprint"]),
    )
    scratch = loaded["build_scratch_runtime"](config, contract)
    if (
        scratch.initial_model_state_sha256 != runtime.get("initial_model_state_sha256")
        or scratch.process_identity_sha256 != runtime.get("process_identity_sha256")
        or scratch.semantic_model_process_contract_sha256
        != runtime.get("semantic_model_process_contract_sha256")
        or scratch.semantic_model_identity != runtime.get("semantic_model_identity")
        or scratch.architecture.as_payload() != architecture
    ):
        raise RuntimeError("reconstructed semantic P50 scratch runtime disagrees")
    return scratch, dict(runtime)


def _source_binding(prepared: Mapping[str, Any], *, loaded: Mapping[str, Any]) -> Any:
    prerequisites = prepared.get("prerequisites")
    if not isinstance(prerequisites, Mapping):
        raise TypeError("semantic P50 prepared recipe lacks prerequisites")
    names = (
        "source_inventory_file_sha256",
        "source_inventory_sha256",
        "process_identity_sha256",
        "model_runtime_identity_sha256",
        "active8_policy_sha256",
        "operator_capability_fingerprint",
        "decision_source_implementation_sha256",
    )
    try:
        return loaded["SemanticP50SourceInventoryBinding"](
            **{name: prerequisites[name] for name in names}
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("semantic P50 source binding is malformed") from error


_PREREQUISITE_ADDRESS_FIELDS = {
    "source_inventory": "source_inventory_path",
    "migration_completion": "migration_completion_path",
    "chunk_cache_plan": "chunk_cache_plan_path",
    "chunk_cache_global_completion": "chunk_cache_global_completion_path",
    "decision_plan": "decision_plan_path",
    "decision_completion": "decision_completion_path",
    "gate_zero_evidence": "gate_zero_evidence_path",
    "t1_decision": "t1_decision_path",
    "prepared_recipe": "prepared_recipe_path",
    "successor_cache_completion": "successor_cache_completion_path",
    "validation_inventory_completion": "validation_inventory_completion_path",
    "validation_baseline_completion": "validation_baseline_completion_path",
    "validation_evaluation_environment_receipt": (
        "validation_evaluation_environment_receipt_path"
    ),
}


def _mounted_execution_prerequisites(
    addresses: Mapping[str, str], *, loaded: Mapping[str, Any]
) -> tuple[dict[str, Path], Any]:
    if set(addresses) != set(_PREREQUISITE_ADDRESS_FIELDS):
        raise RuntimeError("semantic P50 prerequisite address fields disagree")
    mounted = {
        field: _artifact_path(addresses[address], field=field)
        for address, field in _PREREQUISITE_ADDRESS_FIELDS.items()
    }
    paths = loaded["SemanticP50ExecutionPrerequisitePaths"](
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
        **mounted,
    )
    return mounted, paths


def _snapshot_prerequisite_files(
    mounted: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    if set(mounted) != set(_PREREQUISITE_ADDRESS_FIELDS.values()) or len(mounted) != 13:
        raise RuntimeError(
            "semantic P50 prerequisite snapshot must contain exactly 13 files"
        )
    snapshot: dict[str, dict[str, Any]] = {}
    for field, path in sorted(mounted.items()):
        physical = Path(path).absolute()
        try:
            metadata = physical.lstat()
        except OSError as error:
            raise RuntimeError(f"{field} cannot be snapshotted") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{field} must be a physical regular file")
        snapshot[field] = {
            "artifact_path": _artifact_address(physical),
            "file_bytes": metadata.st_size,
            "file_sha256": _file_sha(physical),
        }
    return snapshot


def _require_unchanged_prerequisite_snapshot(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> None:
    if before != after:
        changed = sorted(
            set(before).symmetric_difference(after)
            | {
                field
                for field in set(before).intersection(after)
                if before[field] != after[field]
            }
        )
        raise RuntimeError(
            f"semantic P50 prerequisite bytes changed during preflight: {changed[:4]}"
        )


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _assert_result_environment(
    result: Mapping[str, Any], *, observed_receipt: Mapping[str, Any]
) -> None:
    environment = result.get("execution_environment")
    software = observed_receipt.get("observed_software")
    hardware = observed_receipt.get("observed_hardware")
    if (
        not isinstance(environment, Mapping)
        or not isinstance(software, Mapping)
        or not isinstance(hardware, Mapping)
        or environment.get("device_type") != "cuda"
        or environment.get("device_name") != hardware.get("device_name")
        or environment.get("dtype") != "torch.float32"
        or environment.get("mixed_precision") is not False
        or environment.get("cuda_matmul_tf32_allowed") is not False
        or environment.get("cudnn_tf32_allowed") is not False
        or environment.get("deterministic_algorithms_enabled") is not True
        or environment.get("python_version") != software.get("python_version")
        or environment.get("torch_version") != software.get("torch_version")
        or str(environment.get("cuda_version")) != software.get("cuda_version")
        or result.get("learning_demonstrated") is not False
        or result.get("next_stage_authorized") is not None
        or result.get("p500_authorized") is not False
        or result.get("checkpoint_selection_authorized") is not False
        or result.get("final_test_selection_authorized") is not False
    ):
        raise RuntimeError(
            "semantic P50 result environment or downstream authority disagrees"
        )


def _validate_action_inputs(
    *,
    action: str,
    output_prefix_relative: str,
    runtime_contract: str,
    environment_contract: str,
    launch_projection: str,
    execution_permit: str,
) -> str:
    execution_artifacts = (
        runtime_contract,
        environment_contract,
        launch_projection,
        execution_permit,
    )
    if action == "preflight":
        if any(execution_artifacts):
            raise RuntimeError(
                "preflight action forbids pre-existing execution artifacts"
            )
        return action
    if action == "run":
        if not all(execution_artifacts):
            raise RuntimeError(
                "run action requires all four preflight execution artifacts"
            )
        if output_prefix_relative != "editing_v2/semantic_p50":
            raise RuntimeError(
                "run action takes its output prefix only from the launch projection"
            )
        return action
    raise RuntimeError("action must be exactly 'preflight' or 'run'")


@app.function(
    image=image,
    gpu=MODAL_GPU_TYPE,
    cpu=8.0,
    memory=65536,
    timeout=8 * 3600,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def materialize_bounded_semantic_p50_preflight(
    *,
    source_revision: dict[str, Any],
    source_inventory: str,
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    gate_zero_evidence: str,
    t1_decision: str,
    prepared_recipe: str,
    successor_cache_completion: str,
    validation_inventory_completion: str,
    validation_baseline_completion: str,
    validation_evaluation_environment_receipt: str,
    output_prefix_relative: str,
) -> dict[str, Any]:
    """Materialize the prospective A10G execution chain without training."""

    revision = _validate_source_revision(source_revision)
    artifact_volume.reload()
    loaded = _preflight_imports()
    addresses = {
        "source_inventory": source_inventory,
        "migration_completion": migration_completion,
        "chunk_cache_plan": chunk_cache_plan,
        "chunk_cache_global_completion": chunk_cache_global_completion,
        "decision_plan": decision_plan,
        "decision_completion": decision_completion,
        "gate_zero_evidence": gate_zero_evidence,
        "t1_decision": t1_decision,
        "prepared_recipe": prepared_recipe,
        "successor_cache_completion": successor_cache_completion,
        "validation_inventory_completion": validation_inventory_completion,
        "validation_baseline_completion": validation_baseline_completion,
        "validation_evaluation_environment_receipt": (
            validation_evaluation_environment_receipt
        ),
    }
    mounted, paths = _mounted_execution_prerequisites(addresses, loaded=loaded)
    prerequisite_snapshot = _snapshot_prerequisite_files(mounted)
    prepared, _ = loaded["load_prepared_recipe"](mounted["prepared_recipe_path"])
    source_binding = _source_binding(prepared, loaded=loaded)
    scratch, model_runtime = _scratch_runtime(
        mounted["decision_plan_path"], loaded=loaded
    )
    execution_revision = revision["execution_source_revision"]
    prerequisites = loaded["open_execution_prerequisites"](
        paths,
        scratch_runtime=scratch,
        expected_source_binding=source_binding,
        expected_source_revision_sha256=execution_revision["source_revision_sha256"],
    )
    t1 = prerequisites.t1_decision
    if (
        t1.get("bounded_p50_authorized") is not True
        or t1.get("p500_authorized") is not False
        or t1.get("checkpoint_selection_authorized") is not False
        or t1.get("final_test_selection_authorized") is not False
    ):
        raise RuntimeError(
            "semantic P50 preflight requires the exact current T1 P50-only GO"
        )
    if any(parameter.device.type != "cpu" for parameter in scratch.model.parameters()):
        raise RuntimeError("semantic P50 preflight scratch model must remain on CPU")

    observed = loaded["observe_physical_environment"]()
    hardware = observed.get("hardware") if isinstance(observed, Mapping) else None
    software = observed.get("software") if isinstance(observed, Mapping) else None
    expected_image_reference = f"{APP_NAME}@sha256:{revision['image_content_sha256']}"
    if (
        not isinstance(hardware, Mapping)
        or not isinstance(software, Mapping)
        or observed.get("image_content_sha256") != revision["image_content_sha256"]
        or expected_image_reference != IMAGE_REFERENCE
        or hardware.get("modal_gpu_type") != MODAL_GPU_TYPE
        or hardware.get("accelerator_class") != "gpu"
        or hardware.get("device_type") != "cuda"
        or hardware.get("cuda_device_count") != 1
    ):
        raise RuntimeError("live semantic P50 A10G image or environment disagrees")

    common = {
        "scratch_runtime": scratch,
        "prerequisite_paths": paths,
        "expected_source_binding": source_binding,
        "trainer_source_path": REMOTE_ROOT / TRAINER_SOURCE,
        "expected_source_revision": execution_revision,
    }
    runtime = loaded["build_runtime_contract"](
        model_runtime_identity=model_runtime,
        **common,
    )
    runtime_path = _publish_content_addressed_contract(
        kind="runtime",
        identity_sha256=runtime["contract_sha256"],
        filename=loaded["RUNTIME_FILENAME"],
        payload=runtime,
        loaded=loaded,
    )
    artifact_volume.commit()
    artifact_volume.reload()
    reopened_runtime = loaded["open_runtime_contract"](runtime_path, **common)
    if reopened_runtime != runtime:
        raise RuntimeError("semantic P50 runtime changed across publication")

    environment = loaded["build_environment_contract"](
        runtime_contract_path=runtime_path,
        image_reference=IMAGE_REFERENCE,
        image_definition_path=REMOTE_ROOT / LAUNCHER_SOURCE,
        expected_hardware=hardware,
        expected_software=software,
        **common,
    )
    environment_path = _publish_content_addressed_contract(
        kind="environment",
        identity_sha256=environment["contract_sha256"],
        filename=loaded["ENVIRONMENT_FILENAME"],
        payload=environment,
        loaded=loaded,
    )
    artifact_volume.commit()
    artifact_volume.reload()
    environment_open_kwargs = {
        "runtime_contract_path": runtime_path,
        "image_definition_path": REMOTE_ROOT / LAUNCHER_SOURCE,
        **common,
    }
    reopened_environment = loaded["open_environment_contract"](
        environment_path,
        **environment_open_kwargs,
    )
    if reopened_environment != environment:
        raise RuntimeError("semantic P50 environment changed across publication")

    launch = loaded["build_launch_projection"](
        runtime_contract_path=runtime_path,
        environment_contract_path=environment_path,
        image_definition_path=REMOTE_ROOT / LAUNCHER_SOURCE,
        output_prefix_relative=output_prefix_relative,
        **common,
    )
    launch_path = _publish_content_addressed_contract(
        kind="launch",
        identity_sha256=launch["projection_sha256"],
        filename=loaded["LAUNCH_FILENAME"],
        payload=launch,
        loaded=loaded,
    )
    artifact_volume.commit()
    artifact_volume.reload()
    launch_open_kwargs = {
        "runtime_contract_path": runtime_path,
        "environment_contract_path": environment_path,
        "image_definition_path": REMOTE_ROOT / LAUNCHER_SOURCE,
        **common,
    }
    reopened_launch = loaded["open_launch_projection"](
        launch_path,
        **launch_open_kwargs,
    )
    if reopened_launch != launch:
        raise RuntimeError("semantic P50 launch changed across publication")

    permit_path = loaded["materialize_execution_permit"](
        output_root=_content_addressed_contract_root("permit"),
        prepared=prepared,
        launch_projection_path=launch_path,
        **launch_open_kwargs,
    )
    artifact_volume.commit()
    artifact_volume.reload()
    verified = loaded["open_execution_permit"](
        permit_path,
        prepared=prepared,
        launch_projection_path=launch_path,
        **launch_open_kwargs,
    )
    _require_unchanged_prerequisite_snapshot(
        prerequisite_snapshot,
        _snapshot_prerequisite_files(mounted),
    )
    permit = verified.permit
    if (
        verified.path != Path(permit_path).resolve()
        or permit_path
        != _content_addressed_contract_path(
            kind="permit",
            identity_sha256=permit["permit_sha256"],
            filename=loaded["PERMIT_FILENAME"],
        )
        or verified.launch_projection != launch
        or launch.get("image") != environment.get("image")
        or environment.get("image", {}).get("reference") != IMAGE_REFERENCE
        or environment.get("image", {}).get("content_sha256")
        != revision["image_content_sha256"]
        or permit.get("optimizer_steps") != 50
        or permit.get("batch_size") != 64
        or permit.get("resume") is not False
        or permit.get("training_authorized") is not True
        or permit.get("bounded_p50_authorized") is not True
        or permit.get("p500_authorized") is not False
        or permit.get("checkpoint_selection_authorized") is not False
        or permit.get("final_test_selection_authorized") is not False
        or any(
            parameter.device.type != "cpu" for parameter in scratch.model.parameters()
        )
    ):
        raise RuntimeError(
            "semantic P50 prospective permit or validation boundary disagrees"
        )
    return {
        "schema": "compose.editing_v2.semantic_p50_modal_preflight_receipt",
        "schema_version": 1,
        "status": "PREFLIGHT_COMPLETE_EXACT_ONE_SHOT_PERMIT_NO_EXECUTION",
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "p500_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "source_revision_sha256": revision["source_revision_sha256"],
        "execution_source_revision_sha256": execution_revision[
            "source_revision_sha256"
        ],
        "image_reference": IMAGE_REFERENCE,
        "image_content_sha256": revision["image_content_sha256"],
        "observed_hardware": dict(hardware),
        "observed_software": dict(software),
        "runtime_contract_artifact_path": _artifact_address(runtime_path),
        "runtime_contract_sha256": runtime["contract_sha256"],
        "environment_contract_artifact_path": _artifact_address(environment_path),
        "environment_contract_sha256": environment["contract_sha256"],
        "launch_projection_artifact_path": _artifact_address(launch_path),
        "launch_projection_sha256": launch["projection_sha256"],
        "execution_permit_artifact_path": _artifact_address(permit_path),
        "execution_permit_sha256": permit["permit_sha256"],
        "optimizer_constructed": False,
        "optimizer_steps_completed": 0,
        "model_device": "cpu",
        "next_stage_authorized": None,
    }


@app.function(
    image=image,
    gpu=MODAL_GPU_TYPE,
    cpu=8.0,
    memory=65536,
    timeout=8 * 3600,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_bounded_semantic_p50(
    *,
    source_revision: dict[str, Any],
    source_inventory: str,
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    gate_zero_evidence: str,
    t1_decision: str,
    prepared_recipe: str,
    successor_cache_completion: str,
    validation_inventory_completion: str,
    validation_baseline_completion: str,
    validation_evaluation_environment_receipt: str,
    runtime_contract: str,
    environment_contract: str,
    launch_projection: str,
    execution_permit: str,
) -> dict[str, Any]:
    """Strictly reopen one permit and execute its exact one-shot GPU run."""

    revision = _validate_source_revision(source_revision)
    artifact_volume.reload()
    loaded = _imports()
    addresses = {
        "source_inventory": source_inventory,
        "migration_completion": migration_completion,
        "chunk_cache_plan": chunk_cache_plan,
        "chunk_cache_global_completion": chunk_cache_global_completion,
        "decision_plan": decision_plan,
        "decision_completion": decision_completion,
        "gate_zero_evidence": gate_zero_evidence,
        "t1_decision": t1_decision,
        "prepared_recipe": prepared_recipe,
        "successor_cache_completion": successor_cache_completion,
        "validation_inventory_completion": validation_inventory_completion,
        "validation_baseline_completion": validation_baseline_completion,
        "validation_evaluation_environment_receipt": (
            validation_evaluation_environment_receipt
        ),
    }
    mounted, paths = _mounted_execution_prerequisites(addresses, loaded=loaded)
    runtime_path = _artifact_path(runtime_contract, field="runtime_contract")
    environment_path = _artifact_path(
        environment_contract, field="environment_contract"
    )
    launch_path = _artifact_path(launch_projection, field="launch_projection")
    permit_path = _artifact_path(execution_permit, field="execution_permit")
    prepared, _ = loaded["load_prepared_recipe"](mounted["prepared_recipe_path"])
    source_binding = _source_binding(prepared, loaded=loaded)
    scratch, _model_runtime = _scratch_runtime(
        mounted["decision_plan_path"], loaded=loaded
    )
    execution_revision = revision["execution_source_revision"]
    physical_kwargs = {
        "runtime_contract_path": runtime_path,
        "environment_contract_path": environment_path,
        "scratch_runtime": scratch,
        "prerequisite_paths": paths,
        "expected_source_binding": source_binding,
        "trainer_source_path": REMOTE_ROOT / TRAINER_SOURCE,
        "image_definition_path": REMOTE_ROOT / LAUNCHER_SOURCE,
        "expected_source_revision": execution_revision,
    }
    permit_kwargs = {
        "launch_projection_path": launch_path,
        **physical_kwargs,
    }
    authorized = loaded["authorize_recipe"](
        prepared=prepared,
        permit_path=permit_path,
        **permit_kwargs,
    )
    if authorized.path != permit_path.resolve():
        raise RuntimeError("semantic P50 authorizer returned another permit")
    verified = loaded["open_execution_permit"](
        permit_path,
        prepared=prepared,
        **permit_kwargs,
    )
    if (
        verified.path != permit_path.resolve()
        or verified.permit.get("source_revision_sha256")
        != execution_revision["source_revision_sha256"]
        or verified.launch_projection.get("image", {}).get("reference")
        != IMAGE_REFERENCE
        or verified.launch_projection.get("image", {}).get("content_sha256")
        != revision["image_content_sha256"]
        or verified.launch_projection.get("image", {}).get("definition_relative_path")
        != LAUNCHER_SOURCE
        or verified.permit.get("optimizer_steps") != 50
        or verified.permit.get("batch_size") != 64
        or verified.permit.get("resume") is not False
        or verified.permit.get("training_authorized") is not True
        or verified.permit.get("bounded_p50_authorized") is not True
        or verified.permit.get("p500_authorized") is not False
        or verified.permit.get("checkpoint_selection_authorized") is not False
        or verified.permit.get("final_test_selection_authorized") is not False
    ):
        raise RuntimeError("semantic P50 execution permit scope disagrees")
    observed_receipt_path = permit_path.parent / loaded["ENVIRONMENT_RECEIPT_FILENAME"]
    observed = loaded["build_observed_environment_receipt"](
        launch_projection_path=launch_path,
        observed_at_utc=_utc_now(),
        **physical_kwargs,
    )
    loaded["publish_observed_environment_receipt"](observed_receipt_path, observed)
    artifact_volume.commit()
    artifact_volume.reload()
    observed = loaded["open_observed_environment_receipt"](
        observed_receipt_path,
        launch_projection_path=launch_path,
        reobserve=True,
        **physical_kwargs,
    )
    destination = loaded["require_output_available"](
        launch_path,
        execution_permit_path=permit_path,
        environment_receipt_path=observed_receipt_path,
        artifact_root=ARTIFACT_ROOT,
        **physical_kwargs,
    )
    expected_destination = (
        ARTIFACT_ROOT / verified.permit["output_relative_path"]
    ).resolve()
    if destination.resolve() != expected_destination:
        raise RuntimeError("semantic P50 reserved another output identity")
    artifact_volume.commit()

    inputs = verified.prerequisites.runtime_inputs
    model = inputs.scratch_runtime.model
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("semantic P50 scratch model was not rebuilt on CPU")
    model.to(device="cuda", dtype=loaded["torch"].float32)
    if (
        not loaded["torch"].cuda.is_available()
        or loaded["torch"].cuda.device_count() != 1
        or any(
            parameter.device.type != "cuda"
            or parameter.dtype != loaded["torch"].float32
            for parameter in model.parameters()
        )
    ):
        raise RuntimeError("semantic P50 model is not exact single-GPU fp32")
    artifacts = loaded["run_p50"](inputs)
    _assert_result_environment(artifacts.result, observed_receipt=observed)
    completion_path = loaded["publish_run_artifacts"](
        destination,
        artifacts=artifacts,
        inputs=inputs,
    )
    loaded["open_run_artifacts"](
        completion_path,
        artifact_root=destination,
        inputs=inputs,
    )
    artifact_volume.commit()
    artifact_volume.reload()
    reopened = loaded["open_run_artifacts"](
        completion_path,
        artifact_root=destination,
        inputs=inputs,
    )
    return {
        "schema": "compose.editing_v2.semantic_p50_modal_run_receipt",
        "schema_version": 1,
        "status": "COMPLETE_EXACT_ONE_SHOT_P50_NO_DOWNSTREAM_AUTHORITY",
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "p500_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "source_revision_sha256": revision["source_revision_sha256"],
        "execution_permit_artifact_path": _artifact_address(permit_path),
        "execution_permit_sha256": verified.permit["permit_sha256"],
        "observed_environment_artifact_path": _artifact_address(observed_receipt_path),
        "run_artifact_root": _artifact_address(destination),
        "completion_artifact_path": _artifact_address(reopened.completion_path),
        "completion_sha256": reopened.completion["completion_sha256"],
        "result_artifact_path": _artifact_address(reopened.result_path),
        "result_sha256": reopened.result["result_sha256"],
        "checkpoint_artifact_path": _artifact_address(reopened.checkpoint_path),
        "checkpoint_sha256": reopened.checkpoint["checkpoint_sha256"],
        "completed_optimizer_steps": reopened.result["completed_optimizer_steps"],
        "learning_demonstrated": False,
        "next_stage_authorized": None,
    }


@app.local_entrypoint()
def main(
    source_inventory: str,
    migration_completion: str,
    chunk_cache_plan: str,
    chunk_cache_global_completion: str,
    decision_plan: str,
    decision_completion: str,
    gate_zero_evidence: str,
    t1_decision: str,
    prepared_recipe: str,
    successor_cache_completion: str,
    validation_inventory_completion: str,
    validation_baseline_completion: str,
    validation_evaluation_environment_receipt: str,
    expected_commit: str,
    action: str,
    output_prefix_relative: str = "editing_v2/semantic_p50",
    runtime_contract: str = "",
    environment_contract: str = "",
    launch_projection: str = "",
    execution_permit: str = "",
) -> None:
    """Run either validation-only preflight or one separately authorized P50."""

    revision = local_source_revision(expected_commit=expected_commit)
    action = _validate_action_inputs(
        action=action,
        output_prefix_relative=output_prefix_relative,
        runtime_contract=runtime_contract,
        environment_contract=environment_contract,
        launch_projection=launch_projection,
        execution_permit=execution_permit,
    )
    common = {
        "source_revision": revision,
        "source_inventory": source_inventory,
        "migration_completion": migration_completion,
        "chunk_cache_plan": chunk_cache_plan,
        "chunk_cache_global_completion": chunk_cache_global_completion,
        "decision_plan": decision_plan,
        "decision_completion": decision_completion,
        "gate_zero_evidence": gate_zero_evidence,
        "t1_decision": t1_decision,
        "prepared_recipe": prepared_recipe,
        "successor_cache_completion": successor_cache_completion,
        "validation_inventory_completion": validation_inventory_completion,
        "validation_baseline_completion": validation_baseline_completion,
        "validation_evaluation_environment_receipt": (
            validation_evaluation_environment_receipt
        ),
    }
    if action == "preflight":
        result = materialize_bounded_semantic_p50_preflight.remote(
            output_prefix_relative=output_prefix_relative,
            **common,
        )
    elif action == "run":
        result = run_bounded_semantic_p50.remote(
            runtime_contract=runtime_contract,
            environment_contract=environment_contract,
            launch_projection=launch_projection,
            execution_permit=execution_permit,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "APP_NAME",
    "IMAGE_REFERENCE",
    "LAUNCHER_SOURCE",
    "MODAL_GPU_TYPE",
    "PREFLIGHT_CONTRACT_ROOT_RELATIVE",
    "SOURCE_REVISION_SCHEMA",
    "app",
    "image",
    "local_source_revision",
    "main",
    "materialize_bounded_semantic_p50_preflight",
    "run_bounded_semantic_p50",
]
