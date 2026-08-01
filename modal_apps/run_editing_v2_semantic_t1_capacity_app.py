"""Prepare exact semantic T1 inputs on CPU and run the frozen GPU capacity gate.

The CPU stage reopens only the selected packed traces named by the already
completed strict successor cache, materializes exact persistent-slot states,
and compiles each unique state's complete score-independent successor
partition once.  The GPU stage consumes only that immutable artifact and the
strict cache.  It never executes chemistry or reconstructs a state from
SMILES, and it does not launch P50.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import modal

from modal_apps import build_editing_v2_semantic_t1_successor_cache_app as cache_app

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
LAUNCHER_SOURCE = "modal_apps/run_editing_v2_semantic_t1_capacity_app.py"
CAPACITY_POLICY_SOURCE = "configs/editing_v2_semantic_t1_capacity_policy_v1.json"
SEMANTIC_MODEL_PROCESS_SOURCE = "configs/editing_gate_zero_semantic_model_process_v1.json"
PREPARED_OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_t1_prepared_inputs"
PREPARED_COMPLETION_FILENAME = "SEMANTIC_T1_PREPARED_INPUTS_COMPLETE.json"
RUN_OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_t1_capacity"
FAILURE_FILENAME = "SEMANTIC_T1_CAPACITY_FAILURE.json"
FAILURE_OUTPUT_PREFIX = "/artifacts/editing_v2/semantic_t1_capacity_failures"
RESULT_FILENAME = "SEMANTIC_T1_CAPACITY_RESULT.json"
COMPLETION_FILENAME = "SEMANTIC_T1_CAPACITY_COMPLETE.json"
DECISION_FILENAME = "SEMANTIC_T1_CAPACITY_DECISION.json"
SOURCE_REVISION_SCHEMA = "compose.editing_v2.semantic_t1_runner_source_revision"
SOURCE_REVISION_SCHEMA_VERSION = 1

image = cache_app.image.add_local_file(
    ROOT / LAUNCHER_SOURCE,
    str(REMOTE_ROOT / LAUNCHER_SOURCE),
    copy=True,
)
artifact_volume = cache_app.artifact_volume
app = modal.App("compose-v4-editing-v2-semantic-t1-capacity")


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


def local_runner_source_revision(*, expected_commit: str) -> dict[str, Any]:
    """Bind a clean base serialization plus this launcher exactly."""

    base = cache_app.local_source_revision(
        expected_commit=expected_commit,
        repo_root=ROOT,
    )
    body = {
        "schema": SOURCE_REVISION_SCHEMA,
        "schema_version": SOURCE_REVISION_SCHEMA_VERSION,
        "commit": base["commit"],
        "tree": base["tree"],
        "worktree_clean": True,
        "base_serialized_source_revision": base,
        "launcher_relative_path": LAUNCHER_SOURCE,
        "launcher_file_sha256": _file_sha(ROOT / LAUNCHER_SOURCE),
    }
    return {**body, "source_revision_sha256": _sha(body)}


def _validate_runner_source_revision(
    value: object,
    *,
    remote_root: Path,
) -> dict[str, Any]:
    fields = {
        "schema",
        "schema_version",
        "commit",
        "tree",
        "worktree_clean",
        "base_serialized_source_revision",
        "launcher_relative_path",
        "launcher_file_sha256",
        "source_revision_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError("semantic T1 runner source revision is malformed")
    revision = dict(value)
    body = dict(revision)
    supplied = body.pop("source_revision_sha256")
    base = cache_app._validate_source_revision(
        revision["base_serialized_source_revision"],
        remote_root=remote_root,
    )
    if (
        revision["schema"] != SOURCE_REVISION_SCHEMA
        or revision["schema_version"] != SOURCE_REVISION_SCHEMA_VERSION
        or revision["commit"] != base["commit"]
        or revision["tree"] != base["tree"]
        or revision["worktree_clean"] is not True
        or revision["launcher_relative_path"] != LAUNCHER_SOURCE
        or revision["launcher_file_sha256"] != _file_sha(remote_root / LAUNCHER_SOURCE)
        or supplied != _sha(body)
    ):
        raise RuntimeError("semantic T1 runner source revision changed")
    return revision


def _imports(remote_root: Path = REMOTE_ROOT) -> dict[str, Any]:
    source_root = str(remote_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    import numpy as np
    import rdkit
    import torch

    from compose_v4.data.immutable_artifact import write_bytes_if_absent
    from compose_v4.experiments import editing_v2_semantic_t1_decision as decision
    from compose_v4.experiments import (
        editing_v2_semantic_t1_prepared_inputs as prepared,
    )
    from compose_v4.experiments import (
        editing_v2_semantic_t1_successor_cache as cache,
    )
    from compose_v4.experiments.editing_gate_zero_semantic_contract import (
        load_gate_zero_semantic_contract,
    )
    from compose_v4.experiments.editing_v2_semantic_runtime import (
        SemanticScratchModelConfig,
        build_semantic_scratch_runtime,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        SemanticT1CapacityRunnerError,
        SemanticT1RuntimeInputs,
        load_capacity_policy_for_runner,
        run_semantic_t1_capacity,
    )
    from compose_v4.experiments.factorized_successor_training import (
        SuccessorTrainingError,
        compile_state_successor_map,
    )

    return {
        "np": np,
        "rdkit": rdkit,
        "torch": torch,
        "cache": cache,
        "decision": decision,
        "prepared": prepared,
        "write_bytes_if_absent": write_bytes_if_absent,
        "SemanticScratchModelConfig": SemanticScratchModelConfig,
        "build_semantic_scratch_runtime": build_semantic_scratch_runtime,
        "SemanticT1RuntimeInputs": SemanticT1RuntimeInputs,
        "SemanticT1CapacityRunnerError": SemanticT1CapacityRunnerError,
        "SuccessorTrainingError": SuccessorTrainingError,
        "load_capacity_policy": load_capacity_policy_for_runner,
        "run_capacity": run_semantic_t1_capacity,
        "compile_state_successor_map": compile_state_successor_map,
        "load_semantic_contract": load_gate_zero_semantic_contract,
    }


def _artifact_path(value: str, *, field: str) -> Path:
    return cache_app._artifact_path(
        value,
        artifact_root=ARTIFACT_ROOT,
        field_name=field,
    )


def _artifact_address(path: Path) -> str:
    return cache_app._artifact_address(path, artifact_root=ARTIFACT_ROOT)


def _load_cache(
    completion_path: Path,
    *,
    loaded: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    cache = loaded["cache"]
    completion = cache_app._load_canonical_object(
        completion_path,
        field_name="semantic T1 successor-cache completion",
    )
    completion = cache.validate_semantic_t1_successor_cache_completion(completion)
    opened = cache.open_semantic_t1_successor_cache(
        completion_path,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
        expected_panel_completion_sha256=completion["panel_completion_sha256"],
        expected_panel_artifact_sha256=completion["panel_artifact_sha256"],
        expected_decision_source_inventory_sha256=completion["decision_source_inventory_sha256"],
        expected_initial_model_state_sha256=completion["initial_model_state_sha256"],
    )
    return completion, opened


def _scratch_runtime(cache: Any, *, loaded: Mapping[str, Any]) -> Any:
    runtime = cache.manifest["model_runtime_identity"]
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
    binding = cache.manifest["semantic_model_process_contract"]
    contract = loaded["load_semantic_contract"](REMOTE_ROOT / binding["path"])
    if (
        contract.file_sha256 != binding["file_sha256"]
        or contract.sha256 != binding["contract_sha256"]
    ):
        raise RuntimeError("semantic model/process contract bytes changed")
    return loaded["build_semantic_scratch_runtime"](config, contract)


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=4 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def prepare_exact_t1_inputs(
    *,
    runner_source_revision: dict[str, Any],
    cache_completion: str,
    panel_completion: str,
    output_prefix: str = PREPARED_OUTPUT_PREFIX,
) -> dict[str, Any]:
    """Compile exact states and complete partitions once on CPU."""

    revision = _validate_runner_source_revision(
        runner_source_revision,
        remote_root=REMOTE_ROOT,
    )
    artifact_volume.reload()
    loaded = _imports()
    cache_completion_path = _artifact_path(
        cache_completion,
        field="cache_completion",
    )
    completion, opened_cache = _load_cache(
        cache_completion_path,
        loaded=loaded,
    )
    panel_completion_path = _artifact_path(
        panel_completion,
        field="panel_completion",
    )
    bundle = loaded["cache"].load_verified_semantic_t1_panel_bundle(
        panel_completion_path,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    if (
        bundle.completion["completion_sha256"] != completion["panel_completion_sha256"]
        or bundle.panel.artifact_sha256 != completion["panel_artifact_sha256"]
    ):
        raise RuntimeError("panel and successor cache identities disagree")

    plan_path = _artifact_path(
        completion["plan_artifact_path"],
        field="cache plan",
    )
    plan = cache_app._load_canonical_object(
        plan_path,
        field_name="semantic T1 successor-cache plan",
    )
    plan = loaded["cache"].validate_semantic_t1_successor_cache_plan(
        plan,
        repo_root=REMOTE_ROOT,
        artifact_root=ARTIFACT_ROOT,
    )
    records_by_trace: dict[tuple[str, int, str], Any] = {}
    cache_loaded = cache_app._imports(REMOTE_ROOT)
    for task in plan["tasks"]:
        records = cache_app._task_records(
            plan,
            task,
            loaded=cache_loaded,
            artifact_root=ARTIFACT_ROOT,
        )
        for record in records:
            address = record.corpus_address
            if address is None:
                raise RuntimeError("prepared-input source trace lacks an address")
            key = (
                address.packed_shard_content_sha256,
                address.entry_index,
                address.trace_id,
            )
            if key in records_by_trace:
                raise RuntimeError("prepared-input trace appears in multiple tasks")
            records_by_trace[key] = record

    scratch = _scratch_runtime(opened_cache, loaded=loaded)
    scratch.model.eval()
    states: dict[str, Any] = {}
    partitions: dict[str, Any] = {}
    partition_by_state_sha256: dict[str, Any] = {}
    for panel_entry in bundle.panel.entries:
        panel_id = panel_entry.panel_entry_sha256
        cache_record = opened_cache.record_for_panel_entry_sha256(panel_id)
        address = cache_record.address
        trace = records_by_trace.get(
            (
                address.packed_shard_content_sha256,
                address.entry_index,
                address.trace_id,
            )
        )
        if trace is None:
            raise RuntimeError("prepared-input selected trace was not reopened")
        state = trace.path.state_at(address.progress_index)
        states[panel_id] = state
        partition = partition_by_state_sha256.get(panel_entry.source_state_sha256)
        if partition is None:
            partition = loaded["compile_state_successor_map"](
                scratch.model,
                state,
                time=bundle.panel.request.support_time,
            )
            partition_by_state_sha256[panel_entry.source_state_sha256] = partition
        partitions[panel_id] = partition

    policy_path = REMOTE_ROOT / CAPACITY_POLICY_SOURCE
    policy = loaded["load_capacity_policy"](policy_path)
    artifact = loaded["prepared"].build_semantic_t1_prepared_inputs(
        panel=bundle.panel,
        cache=opened_cache,
        source_states_by_panel_entry_sha256=states,
        successor_partitions_by_panel_entry_sha256=partitions,
        capacity_policy=policy,
        capacity_policy_file_sha256=_file_sha(policy_path),
        source_revision=revision,
        repo_root=REMOTE_ROOT,
    )
    prefix = _artifact_path(output_prefix, field="prepared output_prefix")
    artifact_path = prefix / artifact["artifact_sha256"] / loaded["prepared"].FILENAME
    loaded["prepared"].write_semantic_t1_prepared_inputs(artifact_path, artifact)
    completion_receipt = _prepared_completion(
        artifact_path=artifact_path,
        artifact=artifact,
        source_revision=revision,
        panel_completion_sha256=bundle.completion["completion_sha256"],
        unique_exact_source_state_count=len(partition_by_state_sha256),
    )
    completion_path = artifact_path.parent / PREPARED_COMPLETION_FILENAME
    loaded["write_bytes_if_absent"](
        completion_path,
        _canonical_bytes(completion_receipt, newline=True),
    )
    artifact_volume.commit()
    return {
        "prepared_input_completion_path": _artifact_address(completion_path),
        "prepared_input_completion_file_sha256": _file_sha(completion_path),
        "prepared_input_completion_sha256": completion_receipt["completion_sha256"],
        "prepared_input_artifact_path": _artifact_address(artifact_path),
        "prepared_input_file_sha256": _file_sha(artifact_path),
        "prepared_input_artifact_sha256": artifact["artifact_sha256"],
        "prepared_input_implementation_sha256": artifact["implementation_sha256"],
        "unique_panel_entry_count": len(states),
        "unique_exact_source_state_count": len(partition_by_state_sha256),
        "chemistry_compiled_on_gpu": False,
        "training_launched": False,
        "bounded_p50_authorized": False,
    }


def _execution_environment(loaded: Mapping[str, Any], *, batch_size: int) -> dict[str, Any]:
    torch = loaded["torch"]
    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    return loaded["decision"].build_semantic_t1_execution_environment(
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


def _copy_immutable(source: Path, destination: Path, *, loaded: Mapping[str, Any]) -> None:
    created = loaded["write_bytes_if_absent"](destination, source.read_bytes())
    if not created and _file_sha(source) != _file_sha(destination):
        raise RuntimeError("staged immutable T1 input differs from its source")


def _prepared_completion(
    *,
    artifact_path: Path,
    artifact: Mapping[str, Any],
    source_revision: Mapping[str, Any],
    panel_completion_sha256: str,
    unique_exact_source_state_count: int,
) -> dict[str, Any]:
    body = {
        "schema": "compose.editing_v2.semantic_t1_prepared_inputs_completion",
        "schema_version": 1,
        "status": "COMPLETE_EXACT_FULL_PARTITIONS_NO_T1_AUTHORITY",
        "training_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "prepared_input_artifact_path": _artifact_address(artifact_path),
        "prepared_input_file_sha256": _file_sha(artifact_path),
        "prepared_input_file_bytes": artifact_path.stat().st_size,
        "prepared_input_artifact_sha256": artifact["artifact_sha256"],
        "compiler_implementation_sha256": artifact["implementation_sha256"],
        "runner_source_revision": dict(source_revision),
        "runner_source_revision_sha256": source_revision["source_revision_sha256"],
        "panel_completion_sha256": panel_completion_sha256,
        "panel_artifact_sha256": artifact["panel_artifact_sha256"],
        "cache_completion_sha256": artifact["cache_completion_sha256"],
        "cache_manifest_sha256": artifact["cache_manifest_sha256"],
        "panel_entry_count": artifact["entry_count"],
        "unique_exact_source_state_count": unique_exact_source_state_count,
        "panel_entry_inventory_sha256": artifact["panel_entry_inventory_sha256"],
        "prepared_entry_inventory_sha256": artifact["entry_inventory_sha256"],
        "full_nonteacher_partitions_authenticated_by": (
            "content_addressed_cpu_compiler_completion"
        ),
        "teacher_and_census_cross_checked_against_successor_cache": True,
    }
    return {**body, "completion_sha256": _sha(body)}


def _load_prepared_completion(path: Path) -> tuple[dict[str, Any], Path]:
    if Path(path).name != PREPARED_COMPLETION_FILENAME:
        raise RuntimeError(f"prepared-input completion must name {PREPARED_COMPLETION_FILENAME}")
    completion = cache_app._load_canonical_object(
        path,
        field_name="semantic T1 prepared-input completion",
    )
    body = dict(completion)
    supplied = body.pop("completion_sha256", None)
    required = {
        "schema",
        "schema_version",
        "status",
        "training_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
        "long_training_authorized",
        "checkpoint_selection_authorized",
        "final_test_selection_authorized",
        "prepared_input_artifact_path",
        "prepared_input_file_sha256",
        "prepared_input_file_bytes",
        "prepared_input_artifact_sha256",
        "compiler_implementation_sha256",
        "runner_source_revision",
        "runner_source_revision_sha256",
        "panel_completion_sha256",
        "panel_artifact_sha256",
        "cache_completion_sha256",
        "cache_manifest_sha256",
        "panel_entry_count",
        "unique_exact_source_state_count",
        "panel_entry_inventory_sha256",
        "prepared_entry_inventory_sha256",
        "full_nonteacher_partitions_authenticated_by",
        "teacher_and_census_cross_checked_against_successor_cache",
    }
    if (
        set(body) != required
        or completion.get("schema") != "compose.editing_v2.semantic_t1_prepared_inputs_completion"
        or completion.get("schema_version") != 1
        or completion.get("status") != "COMPLETE_EXACT_FULL_PARTITIONS_NO_T1_AUTHORITY"
        or supplied != _sha(body)
        or any(
            completion.get(name) is not False
            for name in (
                "training_authorized",
                "t1_authorized",
                "bounded_p50_authorized",
                "long_training_authorized",
                "checkpoint_selection_authorized",
                "final_test_selection_authorized",
            )
        )
        or completion.get("teacher_and_census_cross_checked_against_successor_cache") is not True
        or completion.get("full_nonteacher_partitions_authenticated_by")
        != "content_addressed_cpu_compiler_completion"
    ):
        raise RuntimeError("semantic T1 prepared-input completion is malformed")
    artifact_path = _artifact_path(
        completion["prepared_input_artifact_path"],
        field="prepared completion artifact path",
    )
    if (
        artifact_path.name != "SEMANTIC_T1_PREPARED_INPUTS.json"
        or artifact_path.parent != Path(path).resolve().parent
        or _file_sha(artifact_path) != completion["prepared_input_file_sha256"]
        or artifact_path.stat().st_size != completion["prepared_input_file_bytes"]
    ):
        raise RuntimeError("semantic T1 prepared-input completion physical bytes disagree")
    return {**body, "completion_sha256": supplied}, artifact_path


def _failure_receipt(
    *,
    provenance: Mapping[str, Any] | None,
    error: Exception,
    run_root: Path,
    failure_stage: str,
    request_identity_sha256: str,
    request_bindings: Mapping[str, Any],
    recovery_commit_succeeded: bool,
    recovery_commit_failure: Exception | None = None,
) -> dict[str, Any]:
    checkpoints = [
        {
            "relative_path": path.relative_to(run_root).as_posix(),
            "file_sha256": _file_sha(path),
            "file_bytes": path.stat().st_size,
        }
        for path in sorted((run_root / "checkpoints").glob("step_*.pt"))
    ]
    body = {
        "schema": "compose.editing_v2.semantic_t1_capacity_failure",
        "schema_version": 2,
        "status": "FAILED_CLOSED_NO_DOWNSTREAM_AUTHORITY",
        "training_authorized": False,
        "bounded_p50_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "request_identity_sha256": request_identity_sha256,
        "request_bindings": dict(request_bindings),
        "provenance": None if provenance is None else dict(provenance),
        "failure_stage": failure_stage,
        "failure_category": _classify_failure(failure_stage, error),
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "recovery_checkpoints": checkpoints,
        "pre_receipt_recovery_commit_succeeded": recovery_commit_succeeded,
        "pre_receipt_recovery_commit_failure": (
            None
            if recovery_commit_failure is None
            else {
                "type": type(recovery_commit_failure).__name__,
                "message": str(recovery_commit_failure),
            }
        ),
        "scientific_result_published": (run_root / RESULT_FILENAME).is_file(),
        "completion_published": (run_root / COMPLETION_FILENAME).is_file(),
        "decision_published": (run_root / DECISION_FILENAME).is_file(),
        "p50_launched": False,
    }
    return {**body, "failure_sha256": _sha(body)}


def _classify_failure(failure_stage: str, error: Exception) -> str:
    message = str(error).lower()
    name = type(error).__name__.lower()
    if "outofmemory" in name or "out of memory" in message:
        return "CUDA_OUT_OF_MEMORY"
    if failure_stage in {
        "runner_source_revision",
        "volume_reload",
        "imports",
        "input_resolution",
        "cache_validation",
        "panel_authentication",
        "prepared_input_authentication",
        "gate_zero_input",
        "provenance_construction",
        "resume_input",
    }:
        return "PRERUN_INPUT_OR_PROVENANCE_FAILURE"
    if failure_stage in {
        "result_publication",
        "completion_publication",
        "decision_publication",
        "volume_commit",
    }:
        return "ARTIFACT_PUBLICATION_FAILURE"
    return "RUNTIME_FAILURE"


def _local_write_bytes_if_absent(path: Path, payload: bytes) -> bool:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"immutable failure artifact differs at {destination}")
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _publish_failure_receipt(
    *,
    loaded: Mapping[str, Any] | None,
    run_root: Path,
    provenance: Mapping[str, Any] | None,
    error: Exception,
    failure_stage: str,
    request_identity_sha256: str,
    request_bindings: Mapping[str, Any],
    commit: Any,
) -> tuple[dict[str, Any], Path]:
    """Commit recovery bytes first, then publish a nonauthorizing receipt."""

    recovery_commit_error: Exception | None = None
    try:
        commit()
        recovery_commit_succeeded = True
    except Exception as commit_error:  # explicit scientific-job boundary
        recovery_commit_succeeded = False
        recovery_commit_error = commit_error
    failure = _failure_receipt(
        provenance=provenance,
        error=error,
        run_root=run_root,
        failure_stage=failure_stage,
        request_identity_sha256=request_identity_sha256,
        request_bindings=request_bindings,
        recovery_commit_succeeded=recovery_commit_succeeded,
        recovery_commit_failure=recovery_commit_error,
    )
    failure_path = run_root / FAILURE_FILENAME
    writer = _local_write_bytes_if_absent if loaded is None else loaded["write_bytes_if_absent"]
    writer(
        failure_path,
        _canonical_bytes(failure, newline=True),
    )
    commit()
    return failure, failure_path


def _run_t1_capacity_success(
    *,
    runner_source_revision: dict[str, Any],
    prepared_completion: str,
    cache_completion: str,
    gate_zero_evidence: str,
    output_prefix: str = RUN_OUTPUT_PREFIX,
    resume_checkpoint: str | None = None,
    resume_checkpoint_file_sha256: str | None = None,
    failure_context: dict[str, Any],
) -> dict[str, Any]:
    """Run only the frozen semantic T1 gate and publish its decision chain."""

    failure_context["stage"] = "runner_source_revision"
    revision = _validate_runner_source_revision(
        runner_source_revision,
        remote_root=REMOTE_ROOT,
    )
    failure_context["stage"] = "volume_reload"
    artifact_volume.reload()
    failure_context["stage"] = "imports"
    loaded = _imports()
    failure_context["loaded"] = loaded
    failure_context["stage"] = "input_resolution"
    cache_completion_path = _artifact_path(
        cache_completion,
        field="cache_completion",
    )
    completion, opened_cache = _load_cache(
        cache_completion_path,
        loaded=loaded,
    )
    failure_context["stage"] = "prepared_input_authentication"
    policy = loaded["load_capacity_policy"](REMOTE_ROOT / CAPACITY_POLICY_SOURCE)
    prepared_completion_path = _artifact_path(
        prepared_completion,
        field="prepared_completion",
    )
    prepared_completion_receipt, prepared_path = _load_prepared_completion(prepared_completion_path)
    prepared = loaded["prepared"].load_semantic_t1_prepared_inputs(
        prepared_path,
        expected_capacity_policy_sha256=policy["policy_sha256"],
        expected_panel_artifact_sha256=completion["panel_artifact_sha256"],
        expected_cache_completion_sha256=completion["completion_sha256"],
        expected_initial_model_state_sha256=completion["initial_model_state_sha256"],
        repo_root=REMOTE_ROOT,
    )
    if (
        prepared_completion_receipt["runner_source_revision"] != revision
        or prepared.artifact["source_revision"] != revision
        or prepared_completion_receipt["prepared_input_artifact_sha256"]
        != prepared.artifact["artifact_sha256"]
        or prepared_completion_receipt["compiler_implementation_sha256"]
        != prepared.artifact["implementation_sha256"]
        or prepared_completion_receipt["panel_artifact_sha256"]
        != completion["panel_artifact_sha256"]
        or prepared_completion_receipt["cache_completion_sha256"] != completion["completion_sha256"]
        or prepared_completion_receipt["cache_manifest_sha256"]
        != opened_cache.manifest["manifest_sha256"]
        or prepared_completion_receipt["panel_entry_count"] != prepared.artifact["entry_count"]
    ):
        raise RuntimeError("prepared-input completion, artifact, runner, panel, or cache disagree")
    failure_context["stage"] = "panel_authentication"
    panel_completion_path = _artifact_path(
        opened_cache.manifest["panel_binding"]["completion_artifact_path"],
        field="bound panel completion",
    )
    panel_bundle = loaded["cache"].load_verified_semantic_t1_panel_bundle(
        panel_completion_path,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    if (
        panel_bundle.completion["completion_sha256"] != completion["panel_completion_sha256"]
        or panel_bundle.panel.artifact_sha256 != completion["panel_artifact_sha256"]
    ):
        raise RuntimeError("reopened semantic T1 panel identity disagrees")
    loaded["prepared"].authenticate_semantic_t1_prepared_inputs(
        prepared,
        panel=panel_bundle.panel,
        cache=opened_cache,
    )
    runtime = loaded["SemanticT1RuntimeInputs"](
        prepared,
        panel_bundle.panel,
        opened_cache,
        policy,
    )
    failure_context["stage"] = "scratch_runtime"
    scratch = _scratch_runtime(opened_cache, loaded=loaded)
    model = scratch.model.to(device="cuda", dtype=loaded["torch"].float32)
    environment = _execution_environment(
        loaded,
        batch_size=int(policy["optimization"]["batch_size"]),
    )
    runner_implementation_sha256 = loaded["decision"].semantic_t1_runner_implementation_sha256(
        repo_root=REMOTE_ROOT
    )
    run_identity = _sha(
        {
            "capacity_policy_sha256": policy["policy_sha256"],
            "prepared_input_artifact_sha256": prepared.artifact["artifact_sha256"],
            "prepared_input_completion_sha256": prepared_completion_receipt["completion_sha256"],
            "cache_completion_sha256": completion["completion_sha256"],
            "runner_source_revision_sha256": revision["source_revision_sha256"],
            "runner_implementation_sha256": runner_implementation_sha256,
            "execution_environment_sha256": environment["environment_sha256"],
        }
    )
    run_root = _artifact_path(output_prefix, field="run output_prefix") / run_identity
    failure_context["run_root"] = run_root
    failure_context["stage"] = "gate_zero_input"
    staged_prepared = run_root / "inputs" / loaded["prepared"].FILENAME
    staged_prepared_completion = run_root / "inputs" / PREPARED_COMPLETION_FILENAME
    staged_cache = run_root / "inputs" / loaded["cache"].CACHE_COMPLETION_FILENAME
    staged_gate_zero = run_root / "inputs" / Path(gate_zero_evidence).name
    gate_zero_path = _artifact_path(gate_zero_evidence, field="gate_zero_evidence")
    _copy_immutable(prepared_path, staged_prepared, loaded=loaded)
    _copy_immutable(
        prepared_completion_path,
        staged_prepared_completion,
        loaded=loaded,
    )
    _copy_immutable(cache_completion_path, staged_cache, loaded=loaded)
    _copy_immutable(gate_zero_path, staged_gate_zero, loaded=loaded)

    failure_context["stage"] = "provenance_construction"
    provenance = loaded["decision"].build_semantic_t1_result_provenance(
        cache_completion_path=staged_cache,
        gate_zero_evidence_path=staged_gate_zero,
        prepared_input_path=staged_prepared,
        runner_source_revision_sha256=revision["source_revision_sha256"],
        execution_environment=environment,
        repo_root=REMOTE_ROOT,
    )
    failure_context["provenance"] = provenance
    failure_context["stage"] = "resume_input"
    resume_path = (
        None
        if resume_checkpoint is None
        else _artifact_path(resume_checkpoint, field="resume_checkpoint")
    )
    failure_context["stage"] = "capacity_runtime"
    run = loaded["run_capacity"](
        model,
        runtime,
        output_directory=run_root,
        provenance=provenance,
        resume_checkpoint_path=resume_path,
        resume_checkpoint_file_sha256=resume_checkpoint_file_sha256,
    )
    failure_context["stage"] = "completion_publication"
    completion_artifact = loaded["decision"].build_semantic_t1_capacity_completion(
        artifact_directory=run_root,
        result_path=run["result_path"],
        cache_completion_path=staged_cache,
        gate_zero_evidence_path=staged_gate_zero,
        prepared_input_path=staged_prepared,
        selected_checkpoint_path=run["selected_checkpoint_path"],
        repo_root=REMOTE_ROOT,
    )
    completion_path = run_root / loaded["decision"].COMPLETION_FILENAME
    loaded["write_bytes_if_absent"](
        completion_path,
        _canonical_bytes(completion_artifact, newline=True),
    )
    failure_context["stage"] = "decision_publication"
    decision_artifact = loaded["decision"].build_semantic_t1_capacity_decision(
        completion_path=completion_path,
        repo_root=REMOTE_ROOT,
    )
    decision_path = run_root / loaded["decision"].DECISION_FILENAME
    loaded["write_bytes_if_absent"](
        decision_path,
        _canonical_bytes(decision_artifact, newline=True),
    )
    failure_context["stage"] = "volume_commit"
    artifact_volume.commit()
    return {
        "run_artifact_root": _artifact_address(run_root),
        "result_artifact_path": _artifact_address(run["result_path"]),
        "completion_artifact_path": _artifact_address(completion_path),
        "decision_artifact_path": _artifact_address(decision_path),
        "decision_status": decision_artifact["status"],
        "bounded_p50_authorized": decision_artifact["bounded_p50_authorized"],
        "selected_checkpoint_artifact_path": _artifact_address(run["selected_checkpoint_path"]),
        "selected_checkpoint_file_sha256": run["selected_checkpoint_file_sha256"],
        "p50_launched": False,
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=65536,
    timeout=8 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_t1_capacity(
    *,
    runner_source_revision: dict[str, Any],
    prepared_completion: str,
    cache_completion: str,
    gate_zero_evidence: str,
    output_prefix: str = RUN_OUTPUT_PREFIX,
    resume_checkpoint: str | None = None,
    resume_checkpoint_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Run T1 while preserving a receipt for every recoverable failure stage."""

    request_bindings = {
        "runner_source_revision": runner_source_revision,
        "prepared_completion": prepared_completion,
        "cache_completion": cache_completion,
        "gate_zero_evidence": gate_zero_evidence,
        "output_prefix": output_prefix,
        "resume_checkpoint": resume_checkpoint,
        "resume_checkpoint_file_sha256": resume_checkpoint_file_sha256,
    }
    request_identity_sha256 = _sha(request_bindings)
    failure_context: dict[str, Any] = {
        "stage": "runner_source_revision",
        "loaded": None,
        "provenance": None,
        "run_root": (
            ARTIFACT_ROOT / "editing_v2" / "semantic_t1_capacity_failures" / request_identity_sha256
        ),
    }
    try:
        return _run_t1_capacity_success(
            runner_source_revision=runner_source_revision,
            prepared_completion=prepared_completion,
            cache_completion=cache_completion,
            gate_zero_evidence=gate_zero_evidence,
            output_prefix=output_prefix,
            resume_checkpoint=resume_checkpoint,
            resume_checkpoint_file_sha256=resume_checkpoint_file_sha256,
            failure_context=failure_context,
        )
    except Exception as error:  # explicit Modal scientific-job boundary
        failure, failure_path = _publish_failure_receipt(
            loaded=failure_context["loaded"],
            run_root=failure_context["run_root"],
            provenance=failure_context["provenance"],
            error=error,
            failure_stage=str(failure_context["stage"]),
            request_identity_sha256=request_identity_sha256,
            request_bindings=request_bindings,
            commit=artifact_volume.commit,
        )
        return {
            "run_artifact_root": _artifact_address(failure_context["run_root"]),
            "failure_artifact_path": _artifact_address(failure_path),
            "failure_sha256": failure["failure_sha256"],
            "failure_category": failure["failure_category"],
            "failure_type": failure["failure_type"],
            "bounded_p50_authorized": False,
            "p50_launched": False,
        }


@app.local_entrypoint()
def main(
    cache_completion: str,
    panel_completion: str,
    gate_zero_evidence: str,
    expected_commit: str,
    prepared_output_prefix: str = PREPARED_OUTPUT_PREFIX,
    run_output_prefix: str = RUN_OUTPUT_PREFIX,
) -> None:
    revision = local_runner_source_revision(expected_commit=expected_commit)
    prepared = prepare_exact_t1_inputs.remote(
        runner_source_revision=revision,
        cache_completion=cache_completion,
        panel_completion=panel_completion,
        output_prefix=prepared_output_prefix,
    )
    result = run_t1_capacity.remote(
        runner_source_revision=revision,
        prepared_completion=prepared["prepared_input_completion_path"],
        cache_completion=cache_completion,
        gate_zero_evidence=gate_zero_evidence,
        output_prefix=run_output_prefix,
    )
    print(json.dumps({"prepared": prepared, "capacity": result}, indent=2, sort_keys=True))


__all__ = [
    "FAILURE_FILENAME",
    "PREPARED_OUTPUT_PREFIX",
    "RUN_OUTPUT_PREFIX",
    "app",
    "local_runner_source_revision",
    "main",
    "prepare_exact_t1_inputs",
    "run_t1_capacity",
]
