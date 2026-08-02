"""Fail-closed physical execution contracts for the semantic P50 pilot.

This module grants no training authority.  It deliberately has no generic
"validate a self-hashed JSON file" prerequisite API.  A prerequisite enters a
runtime or launch contract only after its production, purpose-specific opener
has reconstructed the physical lineage that gives the digest meaning.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_semantic_capability_cells import (
    load_semantic_capability_cell_registry,
)
from compose_v4.data.successor_fiber_cache import SuccessorFiberCacheAddress
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_training_gate import (
    EditingTrainingGateError,
    assert_p50_launch_authorized,
    load_editing_training_gate,
)
from compose_v4.experiments.editing_v2_semantic_gate_zero import (
    EVIDENCE_FILENAME as GATE_ZERO_EVIDENCE_FILENAME,
)
from compose_v4.experiments.editing_v2_semantic_p50_recipe_stream import (
    REQUIRED_BINDING_PURPOSES,
    SemanticP50Prerequisites,
    validate_semantic_p50_prerequisite_relationships,
)
from compose_v4.experiments.editing_v2_semantic_p50_runner import (
    SemanticP50RuntimeInputs,
)
from compose_v4.experiments.editing_v2_execution_source_revision import (
    EXECUTION_SOURCE_REVISION_SCHEMA,
    EditingV2ExecutionSourceRevisionError,
    build_execution_source_revision,
    validate_execution_source_revision,
)
from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    SEMANTIC_P50_SOURCE_INVENTORY_FILENAME,
    SemanticP50SourceInventoryBinding,
    load_semantic_p50_source_inventory,
)
from compose_v4.experiments.editing_v2_semantic_p50_successor_cache import (
    COMPLETION_FILENAME as CACHE_COMPLETION_FILENAME,
)
from compose_v4.experiments.editing_v2_semantic_p50_successor_cache import (
    SemanticP50SourceReopenPaths,
    expected_semantic_p50_successor_cache_identity_from_reopened_prerequisites,
    load_semantic_p50_prepared_recipe,
    open_semantic_p50_successor_cache,
)
from compose_v4.experiments.editing_v2_semantic_p50_validation_baseline import (
    BASELINE_COMPLETION_FILENAME,
    open_semantic_p50_validation_baseline_from_paths,
)
from compose_v4.experiments.editing_v2_semantic_runtime import SemanticScratchRuntime
from compose_v4.experiments.editing_v2_semantic_t1_decision import (
    DECISION_FILENAME as T1_DECISION_FILENAME,
)
from compose_v4.experiments.editing_v2_semantic_t1_decision import (
    validate_semantic_gate_zero_evidence_receipt,
    validate_semantic_t1_capacity_decision,
)

RUNTIME_SCHEMA = "compose.editing_v2.semantic_p50_runtime_contract"
RUNTIME_SCHEMA_VERSION = 2
RUNTIME_STATUS = "FROZEN_P50_RUNTIME_COORDINATES_NO_TRAINING_AUTHORITY"
RUNTIME_FILENAME = "SEMANTIC_P50_RUNTIME_CONTRACT.json"
ENVIRONMENT_SCHEMA = "compose.editing_v2.semantic_p50_environment_contract"
ENVIRONMENT_SCHEMA_VERSION = 2
ENVIRONMENT_STATUS = "FROZEN_P50_ENVIRONMENT_COORDINATES_NO_TRAINING_AUTHORITY"
ENVIRONMENT_FILENAME = "SEMANTIC_P50_ENVIRONMENT_CONTRACT.json"
ENVIRONMENT_RECEIPT_SCHEMA = "compose.editing_v2.semantic_p50_observed_environment_receipt"
ENVIRONMENT_RECEIPT_SCHEMA_VERSION = 2
ENVIRONMENT_RECEIPT_STATUS = "OBSERVED_P50_ENVIRONMENT_NO_TRAINING_AUTHORITY"
ENVIRONMENT_RECEIPT_FILENAME = "SEMANTIC_P50_OBSERVED_ENVIRONMENT.json"
LAUNCH_SCHEMA = "compose.editing_v2.semantic_p50_launch_projection"
LAUNCH_SCHEMA_VERSION = 2
LAUNCH_STATUS = "FROZEN_P50_LAUNCH_PROJECTION_NO_TRAINING_AUTHORITY"
LAUNCH_FILENAME = "SEMANTIC_P50_LAUNCH_PROJECTION.json"
SOURCE_REVISION_SCHEMA = EXECUTION_SOURCE_REVISION_SCHEMA
ARGV_SCHEMA = "compose.editing_v2.semantic_p50_exact_argv_v1"
RESERVATION_FILENAME = "SEMANTIC_P50_OUTPUT_RESERVED.json"
PERMIT_SCHEMA = "compose.editing_v2.semantic_p50_execution_permit"
PERMIT_STATUS = "AUTHORIZED_EXACT_ONE_SHOT_50_STEP_P50_ONLY"
PERMIT_FILENAME = "SEMANTIC_P50_EXECUTION_PERMIT.json"

NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "p500_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
P50_ONLY_AUTHORITY = {
    "training_authorized": True,
    "bounded_p50_authorized": True,
    "p500_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}

SOURCE_INVENTORY_ROLE = "semantic_source_inventory"
UPSTREAM_RUNTIME_ROLES = (
    SOURCE_INVENTORY_ROLE,
    "semantic_gate_zero_evidence",
    "semantic_t1_decision",
    "p50_stream_union_successor_cache_completion",
    "sampling_and_exposure_contract",
    "semantic_p50_validation_baseline",
)
LAUNCH_ARTIFACT_ROLES = (
    *UPSTREAM_RUNTIME_ROLES,
    "semantic_p50_environment_contract",
    "semantic_p50_runtime_contract",
)
EXECUTION_ORDER = (
    "reopen_purpose_specific_physical_prerequisites",
    "rebuild_and_hash_exact_scratch_model_on_cpu",
    "verify_clean_committed_source_tree",
    "verify_pinned_image_and_physical_definition",
    "verify_exact_schema_bound_argv",
    "collect_and_publish_observed_container_environment",
    "atomically_reserve_content_addressed_output",
    "move_model_to_cuda_float32",
    "seed_python_numpy_torch_and_cuda",
    "enable_torch_deterministic_algorithms",
    "construct_adamw_without_hazard_parameters",
    "execute_exactly_50_optimizer_steps",
    "publish_failure_or_completion_atomically",
)
OPTIMIZER_CONTRACT = {
    "kind": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "scheduler_family": "constant",
    "gradient_clip_norm": 10.0,
}
NUMERICS_CONTRACT = {
    "dtype": "float32",
    "mixed_precision": False,
    "deterministic_algorithms_required": True,
}
OBJECTIVE_CONTRACT = {
    "unit": "productive_embedded_canonical_successor",
    "hazard_included": False,
    "hazard_weight": 0.0,
    "terminal_rows": "excluded",
}
TRAINING_CONTRACT = {
    "initialization_regime": "scratch",
    "optimizer_steps": 50,
    "batch_size": 64,
    "scheduled_nonterminal_examples": 3200,
    "seed": 31,
    "resume": False,
}

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_RE = re.compile(r"^.+@sha256:([0-9a-f]{64})$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_MAX_JSON_BYTES = 256 << 20
_MODEL_RUNTIME_FIELDS = {
    "schema",
    "schema_version",
    "runtime_contract_sha256",
    "semantic_model_process_contract_sha256",
    "semantic_model_identity",
    "process_identity_sha256",
    "architecture",
    "initialization_seed",
    "initial_model_state_sha256",
    "software",
    "producer_source_revision_sha256",
    "execution_source_revision_sha256",
    "identity_sha256",
}
_HARDWARE_FIELDS = {
    "accelerator_class",
    "modal_gpu_type",
    "device_type",
    "device_name",
    "device_capability",
    "cuda_device_count",
    "cpu_count",
    "memory_mb",
}
_SOFTWARE_FIELDS = {
    "python_version",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "rdkit_version",
    "numpy_version",
}


class SemanticP50ExecutionContractError(RuntimeError):
    """A physical P50 execution prerequisite or coordinate failed closed."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticP50ExecutionContractError(
            "P50 artifact is not finite canonical JSON"
        ) from error
    return raw + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    except OSError as error:
        raise SemanticP50ExecutionContractError(f"cannot hash physical artifact: {path}") from error
    return digest.hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise SemanticP50ExecutionContractError(f"{field} must be a lowercase SHA-256")
    return value


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise SemanticP50ExecutionContractError(f"{field} must be normalized nonempty text")
    return value


def _require_nonauthorizing(value: Mapping[str, Any], *, field: str) -> None:
    if any(value.get(name) is not expected for name, expected in NO_AUTHORITY.items()):
        raise SemanticP50ExecutionContractError(f"{field} crosses its authority boundary")


def _require_inside(path: Path, *, root: Path, field: str) -> Path:
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise SemanticP50ExecutionContractError(f"{field} resolves outside its physical root")
    return resolved


def _relative(path: Path, *, root: Path, field: str) -> str:
    return (
        _require_inside(path, root=root, field=field).relative_to(Path(root).resolve()).as_posix()
    )


def _load_canonical(
    path: Path, *, field: str, filename: str | None = None
) -> tuple[dict[str, Any], bytes]:
    source = Path(path).resolve()
    if filename is not None and source.name != filename:
        raise SemanticP50ExecutionContractError(f"{field} must name {filename}")
    if not source.is_file() or not 0 < source.stat().st_size <= _MAX_JSON_BYTES:
        raise SemanticP50ExecutionContractError(f"{field} is absent or outside its byte bound")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticP50ExecutionContractError(f"{field} is unreadable") from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise SemanticP50ExecutionContractError(
            f"{field} must be canonical newline-terminated JSON"
        )
    return payload, raw


def _publish_once(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_bytes(dict(payload), newline=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if destination.is_file() and destination.read_bytes() == content:
            return destination
        raise SemanticP50ExecutionContractError(f"immutable artifact collision: {destination}")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


@dataclass(frozen=True, slots=True, order=True)
class SemanticP50PhysicalArtifactBinding:
    """A receipt emitted only after a purpose-specific physical opener succeeds."""

    role: str
    relative_path: str
    filename: str
    file_sha256: str
    file_bytes: int
    schema: str
    schema_version: int
    status: str
    semantic_sha256_field: str
    semantic_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.role, field="binding.role")
        path = PurePosixPath(_require_text(self.relative_path, field="binding.relative_path"))
        if path.is_absolute() or ".." in path.parts or path.name != self.filename:
            raise SemanticP50ExecutionContractError("binding relative path is not normalized")
        _require_sha(self.file_sha256, field="binding.file_sha256")
        _require_sha(self.semantic_sha256, field="binding.semantic_sha256")
        if type(self.file_bytes) is not int or self.file_bytes <= 0:
            raise SemanticP50ExecutionContractError("binding.file_bytes must be positive")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise SemanticP50ExecutionContractError("binding.schema_version must be positive")
        for field in ("filename", "schema", "status", "semantic_sha256_field"):
            _require_text(getattr(self, field), field=f"binding.{field}")

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticP50ExecutionPrerequisitePaths:
    """Every path needed by the five purpose-specific prerequisite openers."""

    artifact_root: Path
    repo_root: Path
    source_inventory_path: Path
    migration_completion_path: Path
    chunk_cache_plan_path: Path
    chunk_cache_global_completion_path: Path
    decision_plan_path: Path
    decision_completion_path: Path
    gate_zero_evidence_path: Path
    t1_decision_path: Path
    prepared_recipe_path: Path
    successor_cache_completion_path: Path
    validation_inventory_completion_path: Path
    validation_baseline_completion_path: Path
    validation_evaluation_environment_receipt_path: Path


@dataclass(frozen=True, slots=True)
class VerifiedSemanticP50ExecutionPrerequisites:
    """Freshly reopened prerequisite objects and their cross-lineage receipts."""

    source: object
    gate_zero: Mapping[str, Any]
    t1_decision: Mapping[str, Any]
    prepared_recipe: Mapping[str, Any]
    successor_cache: object
    validation_baseline: object
    scratch_runtime: SemanticScratchRuntime
    runtime_inputs: SemanticP50RuntimeInputs
    bindings: tuple[SemanticP50PhysicalArtifactBinding, ...]


@dataclass(frozen=True, slots=True)
class VerifiedSemanticP50ExecutionPermit:
    """A strictly reopened one-shot P50-only authorization artifact."""

    path: Path
    permit: Mapping[str, Any]
    prerequisites: VerifiedSemanticP50ExecutionPrerequisites
    launch_projection: Mapping[str, Any]


def _binding_from_opened(
    path: Path,
    *,
    artifact_root: Path,
    role: str,
    payload: Mapping[str, Any],
    semantic_sha256_field: str,
    expected_filename: str,
) -> SemanticP50PhysicalArtifactBinding:
    source = _require_inside(path, root=artifact_root, field=role)
    physical, raw = _load_canonical(source, field=role, filename=expected_filename)
    if physical != dict(payload):
        raise SemanticP50ExecutionContractError(f"{role} changed after its purpose-specific opener")
    semantic_sha = _require_sha(
        payload.get(semantic_sha256_field), field=f"{role}.{semantic_sha256_field}"
    )
    return SemanticP50PhysicalArtifactBinding(
        role=role,
        relative_path=source.relative_to(Path(artifact_root).resolve()).as_posix(),
        filename=source.name,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        file_bytes=len(raw),
        schema=_require_text(payload.get("schema"), field=f"{role}.schema"),
        schema_version=payload.get("schema_version"),
        status=_require_text(payload.get("status"), field=f"{role}.status"),
        semantic_sha256_field=semantic_sha256_field,
        semantic_sha256=semantic_sha,
    )


def bind_canonical_semantic_p50_artifact(
    *args: object, **kwargs: object
) -> SemanticP50PhysicalArtifactBinding:
    """Reject the retired generic self-hash binding path."""

    del args, kwargs
    raise SemanticP50ExecutionContractError(
        "generic self-hash prerequisite binding is forbidden; use the purpose-specific physical opener"
    )


def _exact_requested_source_states(*, source: object, successor_cache: object) -> dict[str, Any]:
    requested = {
        _sha(asdict(record.address)): record for record in successor_cache.requested_records
    }
    if len(requested) != len(successor_cache.requested_records):
        raise SemanticP50ExecutionContractError(
            "combined successor cache repeats a requested address"
        )
    states: dict[str, Any] = {}
    for partition_role in ("train", "validation"):
        for trace in source.index.iter_accepted_traces_for_partition(partition_role):
            for transition in source.index.accepted_transitions_for(trace):
                address = transition.addressed_trace.address
                cache_address = SuccessorFiberCacheAddress.from_packed_trace(
                    address,
                    progress_index=transition.step_index,
                )
                address_sha256 = _sha(asdict(cache_address))
                if address_sha256 not in requested:
                    continue
                state = transition.addressed_trace.path.state_at(transition.step_index)
                if address_sha256 in states and persistent_slot_state_sha256(
                    states[address_sha256]
                ) != persistent_slot_state_sha256(state):
                    raise SemanticP50ExecutionContractError(
                        "physical semantic source repeats a requested address with another state"
                    )
                states[address_sha256] = state
    if set(states) != set(requested):
        missing = sorted(set(requested).difference(states))
        raise SemanticP50ExecutionContractError(
            f"physical semantic source omits combined-cache requested states: {missing[:4]}"
        )
    return states


def open_semantic_p50_execution_prerequisites(
    paths: SemanticP50ExecutionPrerequisitePaths,
    *,
    scratch_runtime: SemanticScratchRuntime,
    expected_source_binding: SemanticP50SourceInventoryBinding,
    expected_source_revision_sha256: str,
) -> VerifiedSemanticP50ExecutionPrerequisites:
    """Reopen all five prerequisite domains and require one exact lineage."""

    if not isinstance(paths, SemanticP50ExecutionPrerequisitePaths):
        raise TypeError("paths must be SemanticP50ExecutionPrerequisitePaths")
    if not isinstance(expected_source_binding, SemanticP50SourceInventoryBinding):
        raise TypeError("expected_source_binding must be SemanticP50SourceInventoryBinding")
    root = Path(paths.artifact_root).resolve()
    for field in paths.__dataclass_fields__:
        if field not in {"artifact_root", "repo_root"}:
            _require_inside(getattr(paths, field), root=root, field=field)
    source = load_semantic_p50_source_inventory(
        paths.source_inventory_path,
        expected_binding=expected_source_binding,
        migration_completion_path=paths.migration_completion_path,
        chunk_cache_plan_path=paths.chunk_cache_plan_path,
        chunk_cache_global_completion_path=paths.chunk_cache_global_completion_path,
        decision_plan_path=paths.decision_plan_path,
        decision_completion_path=paths.decision_completion_path,
        artifact_root=root,
        repo_root=paths.repo_root,
    )
    gate_file_sha = _file_sha(paths.gate_zero_evidence_path)
    gate_zero = validate_semantic_gate_zero_evidence_receipt(
        paths.gate_zero_evidence_path,
        expected_file_sha256=gate_file_sha,
    )
    t1_physical, _ = _load_canonical(
        paths.t1_decision_path,
        field="semantic T1 decision",
        filename=T1_DECISION_FILENAME,
    )
    t1_decision = validate_semantic_t1_capacity_decision(
        t1_physical,
        decision_path=paths.t1_decision_path,
        repo_root=paths.repo_root,
        expected_gate_zero_evidence_file_sha256=gate_file_sha,
        require_p50_go=True,
    )
    registry = load_semantic_capability_cell_registry()
    relationships = validate_semantic_p50_prerequisite_relationships(
        source=source,
        gate_zero=gate_zero,
        gate_zero_file_sha256=gate_file_sha,
        t1_decision=t1_decision,
        t1_decision_file_sha256=_file_sha(paths.t1_decision_path),
        registry=registry,
    )
    prepared, _ = load_semantic_p50_prepared_recipe(paths.prepared_recipe_path)
    try:
        prepared_prerequisites = SemanticP50Prerequisites(**dict(prepared["prerequisites"]))
    except (KeyError, TypeError, ValueError) as error:
        raise SemanticP50ExecutionContractError(
            "prepared P50 prerequisite fields disagree"
        ) from error
    if prepared_prerequisites != relationships:
        raise SemanticP50ExecutionContractError(
            "prepared recipe differs from physically reconstructed source, Gate0, or T1 lineage"
        )
    cache = open_semantic_p50_successor_cache(
        paths.successor_cache_completion_path,
        artifact_root=root,
        repo_root=paths.repo_root,
        expected_identity=(
            expected_semantic_p50_successor_cache_identity_from_reopened_prerequisites(
                prepared_recipe_sha256=prepared["prepared_recipe_sha256"],
                source=source,
                gate_zero_evidence_path=paths.gate_zero_evidence_path,
                t1_decision_path=paths.t1_decision_path,
                repo_root=paths.repo_root,
            )
        ),
        source_reopen_paths=SemanticP50SourceReopenPaths(
            source_inventory_path=paths.source_inventory_path,
            migration_completion_path=paths.migration_completion_path,
            chunk_cache_plan_path=paths.chunk_cache_plan_path,
            chunk_cache_global_completion_path=paths.chunk_cache_global_completion_path,
            decision_plan_path=paths.decision_plan_path,
            decision_completion_path=paths.decision_completion_path,
        ),
    )
    source_revision_sha = _require_sha(
        expected_source_revision_sha256,
        field="expected_source_revision_sha256",
    )
    if cache.completion.get("execution_source_revision_sha256") != source_revision_sha:
        raise SemanticP50ExecutionContractError(
            "successor cache names another clean source revision"
        )
    baseline = open_semantic_p50_validation_baseline_from_paths(
        paths.validation_baseline_completion_path,
        inventory_completion_path=paths.validation_inventory_completion_path,
        evaluation_environment_receipt_path=(paths.validation_evaluation_environment_receipt_path),
        prepared_recipe_path=paths.prepared_recipe_path,
        source=source,
        scratch_runtime=scratch_runtime,
        artifact_root=root,
        successor_cache=cache,
        repo_root=paths.repo_root,
    )
    baseline_binding = baseline.result.get("binding")
    if (
        not isinstance(baseline_binding, Mapping)
        or baseline.result.get("evaluated_model_state_sha256")
        != relationships.scratch_initial_model_state_sha256
        or baseline_binding.get("source_inventory_file_sha256")
        != expected_source_binding.source_inventory_file_sha256
        or baseline_binding.get("source_inventory_sha256")
        != expected_source_binding.source_inventory_sha256
        or baseline_binding.get("prepared_recipe_file_sha256")
        != _file_sha(paths.prepared_recipe_path)
        or baseline_binding.get("prepared_recipe_sha256") != prepared.get("prepared_recipe_sha256")
        or baseline_binding.get("scratch_initial_model_state_sha256")
        != relationships.scratch_initial_model_state_sha256
    ):
        raise SemanticP50ExecutionContractError(
            "validation baseline differs from the reopened source, recipe, scratch state, or cache"
        )
    runtime_inputs = SemanticP50RuntimeInputs(
        prepared_recipe=prepared,
        cache=cache,
        scratch_runtime=scratch_runtime,
        validation_baseline=baseline,
        states_by_address_sha256=_exact_requested_source_states(
            source=source,
            successor_cache=cache,
        ),
    )
    bindings = (
        _binding_from_opened(
            paths.source_inventory_path,
            artifact_root=root,
            role=SOURCE_INVENTORY_ROLE,
            payload=source.index.identity_payload(),
            semantic_sha256_field="inventory_sha256",
            expected_filename=SEMANTIC_P50_SOURCE_INVENTORY_FILENAME,
        ),
        _binding_from_opened(
            paths.gate_zero_evidence_path,
            artifact_root=root,
            role="semantic_gate_zero_evidence",
            payload=gate_zero,
            semantic_sha256_field="evidence_sha256",
            expected_filename=GATE_ZERO_EVIDENCE_FILENAME,
        ),
        _binding_from_opened(
            paths.t1_decision_path,
            artifact_root=root,
            role="semantic_t1_decision",
            payload=t1_decision,
            semantic_sha256_field="decision_sha256",
            expected_filename=T1_DECISION_FILENAME,
        ),
        _binding_from_opened(
            paths.successor_cache_completion_path,
            artifact_root=root,
            role="p50_stream_union_successor_cache_completion",
            payload=cache.completion,
            semantic_sha256_field="completion_sha256",
            expected_filename=CACHE_COMPLETION_FILENAME,
        ),
        _binding_from_opened(
            paths.prepared_recipe_path,
            artifact_root=root,
            role="sampling_and_exposure_contract",
            payload=prepared,
            semantic_sha256_field="prepared_recipe_sha256",
            expected_filename=Path(paths.prepared_recipe_path).name,
        ),
        _binding_from_opened(
            paths.validation_baseline_completion_path,
            artifact_root=root,
            role="semantic_p50_validation_baseline",
            payload=baseline.completion,
            semantic_sha256_field="completion_sha256",
            expected_filename=BASELINE_COMPLETION_FILENAME,
        ),
    )
    return VerifiedSemanticP50ExecutionPrerequisites(
        source=source,
        gate_zero=gate_zero,
        t1_decision=t1_decision,
        prepared_recipe=prepared,
        successor_cache=cache,
        validation_baseline=baseline,
        scratch_runtime=scratch_runtime,
        runtime_inputs=runtime_inputs,
        bindings=tuple(sorted(bindings)),
    )


def _git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(Path(repo_root).resolve()), *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SemanticP50ExecutionContractError(
            "cannot verify the physical Git source tree"
        ) from error
    return completed.stdout.strip()


def observe_clean_source_revision(repo_root: Path) -> dict[str, Any]:
    """Observe HEAD, HEAD tree, and cleanliness from Git itself."""

    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SemanticP50ExecutionContractError(
            "P50 launch requires an exact clean committed source tree"
        )
    try:
        return build_execution_source_revision(commit=commit, tree=tree)
    except EditingV2ExecutionSourceRevisionError as error:
        raise SemanticP50ExecutionContractError(
            "P50 Git commit or tree identity is malformed"
        ) from error


def _validate_source_revision_attestation(value: object) -> dict[str, Any]:
    try:
        return validate_execution_source_revision(value)
    except EditingV2ExecutionSourceRevisionError as error:
        raise SemanticP50ExecutionContractError(
            "source revision attestation identity disagrees"
        ) from error


def _source_revision_for_validation(
    repo_root: Path,
    *,
    expected_source_revision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if expected_source_revision is None:
        return observe_clean_source_revision(repo_root)
    return _validate_source_revision_attestation(expected_source_revision)


def _trainer_identity(
    path: Path, *, repo_root: Path, require_git_tracking: bool = True
) -> dict[str, str]:
    source = _require_inside(path, root=repo_root, field="trainer_source_path")
    relative = source.relative_to(Path(repo_root).resolve()).as_posix()
    if (
        not source.is_file()
        or source.suffix != ".py"
        or (
            require_git_tracking
            and _git(repo_root, "ls-files", "--error-unmatch", relative) != relative
        )
    ):
        raise SemanticP50ExecutionContractError(
            "semantic P50 trainer must be a tracked Python source file"
        )
    if not relative.startswith("src/"):
        raise SemanticP50ExecutionContractError(
            "semantic P50 trainer must be an importable src module"
        )
    module = relative[4:-3].replace("/", ".")
    return {
        "relative_path": relative,
        "module": module,
        "file_sha256": _file_sha(source),
    }


def _validate_model_runtime_identity(
    value: object,
    *,
    scratch_runtime: SemanticScratchRuntime,
    source_binding: SemanticP50SourceInventoryBinding,
    source_revision_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MODEL_RUNTIME_FIELDS:
        raise SemanticP50ExecutionContractError("model runtime identity fields disagree")
    runtime = dict(value)
    body = dict(runtime)
    supplied = body.pop("identity_sha256")
    if (
        supplied != _sha(body)
        or supplied != source_binding.model_runtime_identity_sha256
        or runtime.get("schema") != "compose.data.semantic_active8_exact_model_runtime"
        or runtime.get("schema_version") != 2
        or runtime.get("process_identity_sha256") != scratch_runtime.process_identity_sha256
        or runtime.get("process_identity_sha256") != source_binding.process_identity_sha256
        or runtime.get("semantic_model_process_contract_sha256")
        != scratch_runtime.semantic_model_process_contract_sha256
        or runtime.get("semantic_model_identity") != scratch_runtime.semantic_model_identity
        or runtime.get("architecture") != scratch_runtime.architecture.as_payload()
        or runtime.get("initialization_seed") != scratch_runtime.config.initialization_seed
        or runtime.get("initial_model_state_sha256") != scratch_runtime.initial_model_state_sha256
        or runtime.get("initial_model_state_sha256")
        != state_dict_semantic_sha256(scratch_runtime.model.state_dict())
        or runtime.get("execution_source_revision_sha256") != source_revision_sha256
        or not isinstance(runtime.get("producer_source_revision_sha256"), str)
        or _SHA_RE.fullmatch(runtime["producer_source_revision_sha256"]) is None
        or scratch_runtime.architecture.operator_capability_fingerprint
        != source_binding.operator_capability_fingerprint
        or not isinstance(runtime.get("software"), Mapping)
        or not runtime["software"]
    ):
        raise SemanticP50ExecutionContractError(
            "scratch model, process, initial state, source revision, or source runtime identity disagrees"
        )
    _require_sha(
        runtime.get("runtime_contract_sha256"),
        field="model_runtime.runtime_contract_sha256",
    )
    return runtime


def build_semantic_p50_runtime_contract(
    *,
    scratch_runtime: SemanticScratchRuntime,
    model_runtime_identity: Mapping[str, Any],
    prerequisite_paths: SemanticP50ExecutionPrerequisitePaths,
    expected_source_binding: SemanticP50SourceInventoryBinding,
    trainer_source_path: Path,
    expected_source_revision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    revision = _source_revision_for_validation(
        prerequisite_paths.repo_root,
        expected_source_revision=expected_source_revision,
    )
    prerequisites = open_semantic_p50_execution_prerequisites(
        prerequisite_paths,
        scratch_runtime=scratch_runtime,
        expected_source_binding=expected_source_binding,
        expected_source_revision_sha256=revision["source_revision_sha256"],
    )
    trainer = _trainer_identity(
        trainer_source_path,
        repo_root=prerequisite_paths.repo_root,
        require_git_tracking=expected_source_revision is None,
    )
    body = {
        "schema": RUNTIME_SCHEMA,
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "status": RUNTIME_STATUS,
        **NO_AUTHORITY,
        "source_revision": revision,
        "source_inventory_binding": expected_source_binding.as_payload(),
        "model_runtime_identity": dict(model_runtime_identity),
        "trainer_source": trainer,
        "initialization_regime": "scratch",
        "initial_model_state_sha256": scratch_runtime.initial_model_state_sha256,
        "process_identity_sha256": scratch_runtime.process_identity_sha256,
        "semantic_model_process_contract_sha256": scratch_runtime.semantic_model_process_contract_sha256,
        "operator_capability_fingerprint": scratch_runtime.architecture.operator_capability_fingerprint,
        "optimizer": dict(OPTIMIZER_CONTRACT),
        "numerics": dict(NUMERICS_CONTRACT),
        "objective": dict(OBJECTIVE_CONTRACT),
        "training": dict(TRAINING_CONTRACT),
        "execution_order": list(EXECUTION_ORDER),
        "physical_inputs": [item.as_payload() for item in prerequisites.bindings],
    }
    return validate_semantic_p50_runtime_contract(
        {**body, "contract_sha256": _sha(body)},
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        expected_source_revision=expected_source_revision,
    )


def validate_semantic_p50_runtime_contract(
    value: object,
    *,
    scratch_runtime: SemanticScratchRuntime,
    prerequisite_paths: SemanticP50ExecutionPrerequisitePaths,
    expected_source_binding: SemanticP50SourceInventoryBinding,
    trainer_source_path: Path,
    expected_source_revision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "schema_version",
        "status",
        *NO_AUTHORITY,
        "source_revision",
        "source_inventory_binding",
        "model_runtime_identity",
        "trainer_source",
        "initialization_regime",
        "initial_model_state_sha256",
        "process_identity_sha256",
        "semantic_model_process_contract_sha256",
        "operator_capability_fingerprint",
        "optimizer",
        "numerics",
        "objective",
        "training",
        "execution_order",
        "physical_inputs",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SemanticP50ExecutionContractError("semantic P50 runtime fields disagree")
    contract = dict(value)
    body = dict(contract)
    supplied = body.pop("contract_sha256")
    _require_nonauthorizing(contract, field="semantic P50 runtime contract")
    revision = _source_revision_for_validation(
        prerequisite_paths.repo_root,
        expected_source_revision=expected_source_revision,
    )
    if contract.get("source_revision") != revision:
        raise SemanticP50ExecutionContractError(
            "runtime contract names another clean source revision"
        )
    prerequisites = open_semantic_p50_execution_prerequisites(
        prerequisite_paths,
        scratch_runtime=scratch_runtime,
        expected_source_binding=expected_source_binding,
        expected_source_revision_sha256=revision["source_revision_sha256"],
    )
    runtime = _validate_model_runtime_identity(
        contract.get("model_runtime_identity"),
        scratch_runtime=scratch_runtime,
        source_binding=expected_source_binding,
        source_revision_sha256=revision["source_revision_sha256"],
    )
    expected_inputs = [item.as_payload() for item in prerequisites.bindings]
    if (
        contract.get("schema") != RUNTIME_SCHEMA
        or contract.get("schema_version") != RUNTIME_SCHEMA_VERSION
        or contract.get("status") != RUNTIME_STATUS
        or supplied != _sha(body)
        or contract.get("source_inventory_binding") != expected_source_binding.as_payload()
        or contract.get("trainer_source")
        != _trainer_identity(
            trainer_source_path,
            repo_root=prerequisite_paths.repo_root,
            require_git_tracking=expected_source_revision is None,
        )
        or contract.get("initialization_regime") != "scratch"
        or contract.get("initial_model_state_sha256") != scratch_runtime.initial_model_state_sha256
        or contract.get("process_identity_sha256") != scratch_runtime.process_identity_sha256
        or contract.get("semantic_model_process_contract_sha256")
        != scratch_runtime.semantic_model_process_contract_sha256
        or contract.get("operator_capability_fingerprint")
        != expected_source_binding.operator_capability_fingerprint
        or contract.get("optimizer") != OPTIMIZER_CONTRACT
        or contract.get("numerics") != NUMERICS_CONTRACT
        or contract.get("objective") != OBJECTIVE_CONTRACT
        or contract.get("training") != TRAINING_CONTRACT
        or tuple(contract.get("execution_order", ())) != EXECUTION_ORDER
        or contract.get("physical_inputs") != expected_inputs
        or runtime.get("identity_sha256") != expected_source_binding.model_runtime_identity_sha256
    ):
        raise SemanticP50ExecutionContractError(
            "semantic P50 runtime, scratch state, or purpose-specific physical lineage disagrees"
        )
    return contract


def open_semantic_p50_runtime_contract(
    path: Path,
    **validation_kwargs: Any,
) -> dict[str, Any]:
    payload, _ = _load_canonical(
        path, field="semantic P50 runtime contract", filename=RUNTIME_FILENAME
    )
    return validate_semantic_p50_runtime_contract(payload, **validation_kwargs)


def _validate_expected_environment(hardware: object, software: object) -> None:
    if not isinstance(hardware, Mapping) or set(hardware) != _HARDWARE_FIELDS:
        raise SemanticP50ExecutionContractError("environment hardware fields disagree")
    if not isinstance(software, Mapping) or set(software) != _SOFTWARE_FIELDS:
        raise SemanticP50ExecutionContractError("environment software fields disagree")
    if (
        hardware.get("accelerator_class") != "gpu"
        or hardware.get("device_type") != "cuda"
        or hardware.get("cuda_device_count") != 1
        or any(
            type(hardware.get(name)) is not int or hardware[name] <= 0
            for name in ("cpu_count", "memory_mb")
        )
    ):
        raise SemanticP50ExecutionContractError(
            "environment requires one exact positive GPU allocation"
        )
    for name in ("modal_gpu_type", "device_name", "device_capability"):
        _require_text(hardware.get(name), field=f"hardware.{name}")
    for name in _SOFTWARE_FIELDS:
        _require_text(software.get(name), field=f"software.{name}")


def _image_identity(
    reference: str,
    definition_path: Path,
    *,
    repo_root: Path,
    require_git_tracking: bool = True,
) -> dict[str, str]:
    match = _IMAGE_RE.fullmatch(_require_text(reference, field="image_reference"))
    if match is None:
        raise SemanticP50ExecutionContractError("image reference must be pinned by sha256 digest")
    source = _require_inside(definition_path, root=repo_root, field="image_definition_path")
    relative = source.relative_to(Path(repo_root).resolve()).as_posix()
    if not source.is_file() or (
        require_git_tracking
        and _git(repo_root, "ls-files", "--error-unmatch", relative) != relative
    ):
        raise SemanticP50ExecutionContractError(
            "image definition must be a tracked physical source file"
        )
    return {
        "reference": reference,
        "content_sha256": match.group(1),
        "definition_relative_path": relative,
        "definition_file_sha256": _file_sha(source),
    }


def build_semantic_p50_environment_contract(
    *,
    runtime_contract_path: Path,
    scratch_runtime: SemanticScratchRuntime,
    prerequisite_paths: SemanticP50ExecutionPrerequisitePaths,
    expected_source_binding: SemanticP50SourceInventoryBinding,
    trainer_source_path: Path,
    image_reference: str,
    image_definition_path: Path,
    expected_hardware: Mapping[str, Any],
    expected_software: Mapping[str, Any],
    expected_source_revision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = open_semantic_p50_runtime_contract(
        runtime_contract_path,
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        expected_source_revision=expected_source_revision,
    )
    _validate_expected_environment(expected_hardware, expected_software)
    body = {
        "schema": ENVIRONMENT_SCHEMA,
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "status": ENVIRONMENT_STATUS,
        **NO_AUTHORITY,
        "runtime_contract_relative_path": _relative(
            runtime_contract_path,
            root=prerequisite_paths.artifact_root,
            field="runtime_contract_path",
        ),
        "runtime_contract_file_sha256": _file_sha(runtime_contract_path),
        "runtime_contract_sha256": runtime["contract_sha256"],
        "source_revision_sha256": runtime["source_revision"]["source_revision_sha256"],
        "image": _image_identity(
            image_reference,
            image_definition_path,
            repo_root=prerequisite_paths.repo_root,
            require_git_tracking=expected_source_revision is None,
        ),
        "hardware": dict(expected_hardware),
        "software": dict(expected_software),
        "numerics": dict(NUMERICS_CONTRACT),
        "execution_order": list(EXECUTION_ORDER),
    }
    return validate_semantic_p50_environment_contract(
        {**body, "contract_sha256": _sha(body)},
        runtime_contract_path=runtime_contract_path,
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        image_definition_path=image_definition_path,
        expected_source_revision=expected_source_revision,
    )


def validate_semantic_p50_environment_contract(
    value: object,
    *,
    runtime_contract_path: Path,
    scratch_runtime: SemanticScratchRuntime,
    prerequisite_paths: SemanticP50ExecutionPrerequisitePaths,
    expected_source_binding: SemanticP50SourceInventoryBinding,
    trainer_source_path: Path,
    image_definition_path: Path,
    expected_source_revision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "schema_version",
        "status",
        *NO_AUTHORITY,
        "runtime_contract_relative_path",
        "runtime_contract_file_sha256",
        "runtime_contract_sha256",
        "source_revision_sha256",
        "image",
        "hardware",
        "software",
        "numerics",
        "execution_order",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SemanticP50ExecutionContractError("P50 environment contract fields disagree")
    contract = dict(value)
    body = dict(contract)
    supplied = body.pop("contract_sha256")
    _require_nonauthorizing(contract, field="semantic P50 environment contract")
    runtime = open_semantic_p50_runtime_contract(
        runtime_contract_path,
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        expected_source_revision=expected_source_revision,
    )
    image = contract.get("image")
    expected_image = None
    if isinstance(image, Mapping) and isinstance(image.get("reference"), str):
        expected_image = _image_identity(
            image["reference"],
            image_definition_path,
            repo_root=prerequisite_paths.repo_root,
            require_git_tracking=expected_source_revision is None,
        )
    _validate_expected_environment(contract.get("hardware"), contract.get("software"))
    if (
        contract.get("schema") != ENVIRONMENT_SCHEMA
        or contract.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION
        or contract.get("status") != ENVIRONMENT_STATUS
        or supplied != _sha(body)
        or contract.get("runtime_contract_relative_path")
        != _relative(
            runtime_contract_path,
            root=prerequisite_paths.artifact_root,
            field="runtime_contract_path",
        )
        or contract.get("runtime_contract_file_sha256") != _file_sha(runtime_contract_path)
        or contract.get("runtime_contract_sha256") != runtime["contract_sha256"]
        or contract.get("source_revision_sha256")
        != runtime["source_revision"]["source_revision_sha256"]
        or contract.get("image") != expected_image
        or contract.get("numerics") != NUMERICS_CONTRACT
        or tuple(contract.get("execution_order", ())) != EXECUTION_ORDER
    ):
        raise SemanticP50ExecutionContractError(
            "P50 environment physical runtime, image, or coordinates disagree"
        )
    return contract


def open_semantic_p50_environment_contract(path: Path, **validation_kwargs: Any) -> dict[str, Any]:
    payload, _ = _load_canonical(
        path, field="semantic P50 environment contract", filename=ENVIRONMENT_FILENAME
    )
    return validate_semantic_p50_environment_contract(payload, **validation_kwargs)


def _expected_argv(
    *,
    trainer_module: str,
    runtime_relative: str,
    environment_relative: str,
    prepared_relative: str,
    cache_relative: str,
    output_relative: str,
) -> list[str]:
    return [
        "python3",
        "-m",
        trainer_module,
        "--runtime-contract",
        runtime_relative,
        "--environment-contract",
        environment_relative,
        "--prepared-recipe",
        prepared_relative,
        "--successor-cache-completion",
        cache_relative,
        "--output",
        output_relative,
        "--optimizer-steps",
        "50",
        "--batch-size",
        "64",
        "--seed",
        "31",
        "--dtype",
        "float32",
        "--no-resume",
        "--hazard-weight",
        "0.0",
    ]


def _validate_output_prefix(value: object) -> str:
    text = _require_text(value, field="output_prefix_relative")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise SemanticP50ExecutionContractError("output prefix must be a normalized relative path")
    return text.rstrip("/")


def _opened_launch_inputs(
    *,
    runtime_contract_path: Path,
    environment_contract_path: Path,
    scratch_runtime: SemanticScratchRuntime,
    prerequisite_paths: SemanticP50ExecutionPrerequisitePaths,
    expected_source_binding: SemanticP50SourceInventoryBinding,
    trainer_source_path: Path,
    image_definition_path: Path,
    expected_source_revision: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[SemanticP50PhysicalArtifactBinding, ...]]:
    runtime = open_semantic_p50_runtime_contract(
        runtime_contract_path,
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        expected_source_revision=expected_source_revision,
    )
    environment = open_semantic_p50_environment_contract(
        environment_contract_path,
        runtime_contract_path=runtime_contract_path,
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        image_definition_path=image_definition_path,
        expected_source_revision=expected_source_revision,
    )
    reopened = open_semantic_p50_execution_prerequisites(
        prerequisite_paths,
        scratch_runtime=scratch_runtime,
        expected_source_binding=expected_source_binding,
        expected_source_revision_sha256=runtime["source_revision"]["source_revision_sha256"],
    )
    root = prerequisite_paths.artifact_root
    runtime_binding = _binding_from_opened(
        runtime_contract_path,
        artifact_root=root,
        role="semantic_p50_runtime_contract",
        payload=runtime,
        semantic_sha256_field="contract_sha256",
        expected_filename=RUNTIME_FILENAME,
    )
    environment_binding = _binding_from_opened(
        environment_contract_path,
        artifact_root=root,
        role="semantic_p50_environment_contract",
        payload=environment,
        semantic_sha256_field="contract_sha256",
        expected_filename=ENVIRONMENT_FILENAME,
    )
    bindings = tuple(sorted((*reopened.bindings, runtime_binding, environment_binding)))
    if {item.role for item in bindings} != set(LAUNCH_ARTIFACT_ROLES):
        raise SemanticP50ExecutionContractError("launch physical roles are incomplete")
    return runtime, environment, bindings


def build_semantic_p50_launch_projection(
    *,
    runtime_contract_path: Path,
    environment_contract_path: Path,
    scratch_runtime: SemanticScratchRuntime,
    prerequisite_paths: SemanticP50ExecutionPrerequisitePaths,
    expected_source_binding: SemanticP50SourceInventoryBinding,
    trainer_source_path: Path,
    image_definition_path: Path,
    output_prefix_relative: str,
    expected_source_revision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime, environment, bindings = _opened_launch_inputs(
        runtime_contract_path=runtime_contract_path,
        environment_contract_path=environment_contract_path,
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        image_definition_path=image_definition_path,
        expected_source_revision=expected_source_revision,
    )
    revision = _source_revision_for_validation(
        prerequisite_paths.repo_root,
        expected_source_revision=expected_source_revision,
    )
    if runtime["source_revision"] != revision:
        raise SemanticP50ExecutionContractError("launch source changed after runtime freeze")
    prefix = _validate_output_prefix(output_prefix_relative)
    artifact_rows = [item.as_payload() for item in bindings]
    identity = {
        "source_revision_sha256": revision["source_revision_sha256"],
        "image": environment["image"],
        "runtime_contract_sha256": runtime["contract_sha256"],
        "environment_contract_sha256": environment["contract_sha256"],
        "physical_artifact_inventory_sha256": _sha(artifact_rows),
        "output_prefix_relative": prefix,
        "resume": False,
    }
    run_identity = _sha(identity)
    output_relative = f"{prefix}/{run_identity}"
    trainer = runtime["trainer_source"]
    argv = _expected_argv(
        trainer_module=trainer["module"],
        runtime_relative=_relative(
            runtime_contract_path,
            root=prerequisite_paths.artifact_root,
            field="runtime_contract_path",
        ),
        environment_relative=_relative(
            environment_contract_path,
            root=prerequisite_paths.artifact_root,
            field="environment_contract_path",
        ),
        prepared_relative=_relative(
            prerequisite_paths.prepared_recipe_path,
            root=prerequisite_paths.artifact_root,
            field="prepared_recipe_path",
        ),
        cache_relative=_relative(
            prerequisite_paths.successor_cache_completion_path,
            root=prerequisite_paths.artifact_root,
            field="cache_completion_path",
        ),
        output_relative=output_relative,
    )
    body = {
        "schema": LAUNCH_SCHEMA,
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "status": LAUNCH_STATUS,
        **NO_AUTHORITY,
        "source_revision": revision,
        "image": environment["image"],
        "argv_schema": ARGV_SCHEMA,
        "argv": argv,
        "argv_sha256": _sha(argv),
        "runtime_contract_sha256": runtime["contract_sha256"],
        "environment_contract_sha256": environment["contract_sha256"],
        "physical_artifacts": artifact_rows,
        "physical_artifact_inventory_sha256": identity["physical_artifact_inventory_sha256"],
        "execution_order": list(EXECUTION_ORDER),
        "optimizer_steps": 50,
        "batch_size": 64,
        "resume": False,
        "output_prefix_relative": prefix,
        "run_identity_sha256": run_identity,
        "output_relative_path": output_relative,
    }
    return validate_semantic_p50_launch_projection(
        {**body, "projection_sha256": _sha(body)},
        runtime_contract_path=runtime_contract_path,
        environment_contract_path=environment_contract_path,
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        image_definition_path=image_definition_path,
        expected_source_revision=expected_source_revision,
    )


def validate_semantic_p50_launch_projection(
    value: object,
    **physical_kwargs: Any,
) -> dict[str, Any]:
    fields = {
        "schema",
        "schema_version",
        "status",
        *NO_AUTHORITY,
        "source_revision",
        "image",
        "argv_schema",
        "argv",
        "argv_sha256",
        "runtime_contract_sha256",
        "environment_contract_sha256",
        "physical_artifacts",
        "physical_artifact_inventory_sha256",
        "execution_order",
        "optimizer_steps",
        "batch_size",
        "resume",
        "output_prefix_relative",
        "run_identity_sha256",
        "output_relative_path",
        "projection_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SemanticP50ExecutionContractError("P50 launch projection fields disagree")
    projection = dict(value)
    body = dict(projection)
    supplied = body.pop("projection_sha256")
    _require_nonauthorizing(projection, field="semantic P50 launch projection")
    runtime, environment, bindings = _opened_launch_inputs(**physical_kwargs)
    paths = physical_kwargs["prerequisite_paths"]
    revision = _source_revision_for_validation(
        paths.repo_root,
        expected_source_revision=physical_kwargs.get("expected_source_revision"),
    )
    prefix = _validate_output_prefix(projection.get("output_prefix_relative"))
    artifact_rows = [item.as_payload() for item in bindings]
    identity = {
        "source_revision_sha256": revision["source_revision_sha256"],
        "image": environment["image"],
        "runtime_contract_sha256": runtime["contract_sha256"],
        "environment_contract_sha256": environment["contract_sha256"],
        "physical_artifact_inventory_sha256": _sha(artifact_rows),
        "output_prefix_relative": prefix,
        "resume": False,
    }
    run_identity = _sha(identity)
    output_relative = f"{prefix}/{run_identity}"
    expected_argv = _expected_argv(
        trainer_module=runtime["trainer_source"]["module"],
        runtime_relative=_relative(
            physical_kwargs["runtime_contract_path"],
            root=paths.artifact_root,
            field="runtime_contract_path",
        ),
        environment_relative=_relative(
            physical_kwargs["environment_contract_path"],
            root=paths.artifact_root,
            field="environment_contract_path",
        ),
        prepared_relative=_relative(
            paths.prepared_recipe_path,
            root=paths.artifact_root,
            field="prepared_recipe_path",
        ),
        cache_relative=_relative(
            paths.successor_cache_completion_path,
            root=paths.artifact_root,
            field="cache_completion_path",
        ),
        output_relative=output_relative,
    )
    if (
        projection.get("schema") != LAUNCH_SCHEMA
        or projection.get("schema_version") != LAUNCH_SCHEMA_VERSION
        or projection.get("status") != LAUNCH_STATUS
        or supplied != _sha(body)
        or projection.get("source_revision") != revision
        or projection.get("image") != environment["image"]
        or projection.get("argv_schema") != ARGV_SCHEMA
        or projection.get("argv") != expected_argv
        or projection.get("argv_sha256") != _sha(expected_argv)
        or projection.get("runtime_contract_sha256") != runtime["contract_sha256"]
        or projection.get("environment_contract_sha256") != environment["contract_sha256"]
        or projection.get("physical_artifacts") != artifact_rows
        or projection.get("physical_artifact_inventory_sha256") != _sha(artifact_rows)
        or tuple(projection.get("execution_order", ())) != EXECUTION_ORDER
        or projection.get("optimizer_steps") != 50
        or projection.get("batch_size") != 64
        or projection.get("resume") is not False
        or projection.get("run_identity_sha256") != run_identity
        or projection.get("output_relative_path") != output_relative
    ):
        raise SemanticP50ExecutionContractError(
            "P50 launch source, image, argv, physical artifacts, or output identity disagrees"
        )
    return projection


def open_semantic_p50_launch_projection(path: Path, **physical_kwargs: Any) -> dict[str, Any]:
    payload, _ = _load_canonical(
        path, field="semantic P50 launch projection", filename=LAUNCH_FILENAME
    )
    return validate_semantic_p50_launch_projection(payload, **physical_kwargs)


def _semantic_p50_training_gate_refinement(
    *,
    prepared: Mapping[str, Any],
    prerequisites: VerifiedSemanticP50ExecutionPrerequisites,
    repo_root: Path,
) -> dict[str, Any]:
    """Resolve the legacy design gate explicitly for this one scratch P50 only."""

    gate_path = Path(repo_root).resolve() / "configs/editing_training_v2_gate.json"
    base = load_editing_training_gate(gate_path)
    p50_gate = next(
        gate for gate in base["gates"] if gate["id"] == "P50_gradient_and_collapse_sentinel"
    )
    thresholds = p50_gate["numeric_thresholds"]
    evidence = p50_gate["prerequisite_evidence"]
    if (
        base.get("status") != "DESIGN_NOT_TRAINING_AUTHORIZED"
        or base.get("bounded_p50_authorized") is not False
        or any(value is not None for value in thresholds.values())
        or any(value is not None for value in evidence.values())
    ):
        raise SemanticP50ExecutionContractError(
            "base Editing-V2 training gate is not the expected unresolved design contract"
        )
    required_families = tuple(prepared["required_families"])
    minimum_by_family = dict(prepared["minimum_nonzero_gradient_updates_by_family"])
    nonincrease = prepared["validation_contract"][
        "maximum_family_final_minus_baseline_for_p50_nonincrease_nats"
    ]
    source_identity = prerequisites.source.index.identity_payload()
    prerequisite_evidence = {
        "frozen_source_corpus_inventory_sha256": source_identity["inventory_sha256"],
        "gate_zero_structural_evidence_sha256": prerequisites.gate_zero["evidence_sha256"],
        "t1_successor_gate_decision_sha256": prerequisites.t1_decision["decision_sha256"],
        "frozen_p50_recipe_sha256": prepared["prepared_recipe_sha256"],
    }
    numeric_thresholds = {
        "minimum_gradient_updates_per_required_slice": {
            family: minimum_by_family[family] for family in required_families
        },
        "maximum_required_slice_successor_nll_regression": {
            family: nonincrease for family in required_families
        },
        "maximum_inherited_probe_nll_regression": {
            "scratch": "NOT_APPLICABLE",
            "compatible_warm_start": "NOT_AUTHORIZED",
            "compatible_warm_start_with_retention": "NOT_AUTHORIZED",
        },
    }
    resolved = copy.deepcopy(base)
    resolved["status"] = "FROZEN_BOUNDED_P50_AUTHORIZED"
    resolved["bounded_p50_authorized"] = True
    resolved_p50 = next(
        gate for gate in resolved["gates"] if gate["id"] == "P50_gradient_and_collapse_sentinel"
    )
    resolved_p50["prerequisite_evidence"] = prerequisite_evidence
    resolved_p50["numeric_thresholds"] = numeric_thresholds
    try:
        exact_thresholds = assert_p50_launch_authorized(
            resolved,
            initialization_regime="scratch",
            required_families=required_families,
        )
    except EditingTrainingGateError as error:
        raise SemanticP50ExecutionContractError(
            "task-specific semantic P50 refinement does not satisfy the global Editing-V2 gate"
        ) from error
    refinement_body = {
        "schema": "compose.editing_v2.semantic_p50_training_gate_refinement",
        "schema_version": 1,
        "status": "FROZEN_ONE_SHOT_P50_REFINEMENT_OF_DESIGN_GATE",
        "base_contract_relative_path": "configs/editing_training_v2_gate.json",
        "base_contract_file_sha256": _file_sha(gate_path),
        "base_contract_id": base["contract_id"],
        "base_status": base["status"],
        "base_bounded_p50_authorized": base["bounded_p50_authorized"],
        "resolved_status": resolved["status"],
        "resolved_bounded_p50_authorized": resolved["bounded_p50_authorized"],
        "initialization_regime": "scratch",
        "required_families": list(required_families),
        "prerequisite_evidence": prerequisite_evidence,
        "numeric_thresholds": numeric_thresholds,
        "resolved_thresholds": {
            "minimum_gradient_updates": dict(exact_thresholds.minimum_gradient_updates),
            "maximum_family_nll_regression": dict(exact_thresholds.maximum_family_nll_regression),
            "inherited_retention_status": exact_thresholds.inherited_retention_status,
            "maximum_inherited_probe_nll_regression": (
                exact_thresholds.maximum_inherited_probe_nll_regression
            ),
        },
        "authority_scope": "exact_one_shot_scratch_semantic_p50_only",
        "p500_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "rationale": (
            "content_addressed_task_specific_resolution_of_predeclared_P50_nulls;"
            "base_design_contract_is_not_mutated_or_ignored"
        ),
    }
    return {
        **refinement_body,
        "refinement_sha256": _sha(refinement_body),
    }


def _expected_semantic_p50_execution_permit(
    *,
    prepared: Mapping[str, Any],
    launch_projection_path: Path,
    runtime_contract_path: Path,
    environment_contract_path: Path,
    scratch_runtime: SemanticScratchRuntime,
    prerequisite_paths: SemanticP50ExecutionPrerequisitePaths,
    expected_source_binding: SemanticP50SourceInventoryBinding,
    trainer_source_path: Path,
    image_definition_path: Path,
    expected_source_revision: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    VerifiedSemanticP50ExecutionPrerequisites,
    dict[str, Any],
]:
    physical_prepared, prepared_file_sha256 = load_semantic_p50_prepared_recipe(
        prerequisite_paths.prepared_recipe_path
    )
    if dict(prepared) != physical_prepared:
        raise SemanticP50ExecutionContractError(
            "authorization requires the exact physical prepared recipe"
        )
    launch_kwargs = {
        "runtime_contract_path": runtime_contract_path,
        "environment_contract_path": environment_contract_path,
        "scratch_runtime": scratch_runtime,
        "prerequisite_paths": prerequisite_paths,
        "expected_source_binding": expected_source_binding,
        "trainer_source_path": trainer_source_path,
        "image_definition_path": image_definition_path,
        "expected_source_revision": expected_source_revision,
    }
    projection = open_semantic_p50_launch_projection(
        launch_projection_path,
        **launch_kwargs,
    )
    prerequisites = open_semantic_p50_execution_prerequisites(
        prerequisite_paths,
        scratch_runtime=scratch_runtime,
        expected_source_binding=expected_source_binding,
        expected_source_revision_sha256=projection["source_revision"]["source_revision_sha256"],
    )
    t1 = prerequisites.t1_decision
    if (
        t1.get("bounded_p50_authorized") is not True
        or t1.get("p500_authorized") is not False
        or t1.get("checkpoint_selection_authorized") is not False
        or t1.get("final_test_selection_authorized") is not False
        or t1.get("decision_sha256") != physical_prepared["prerequisites"]["t1_decision_sha256"]
        or physical_prepared.get("bounded_p50_authorized") is not False
        or tuple(physical_prepared.get("unresolved_physical_bindings", ()))
        != REQUIRED_BINDING_PURPOSES
    ):
        raise SemanticP50ExecutionContractError(
            "execution permit requires the exact current T1 P50-only GO and all five unresolved bindings"
        )
    runtime = open_semantic_p50_runtime_contract(
        runtime_contract_path,
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        expected_source_revision=expected_source_revision,
    )
    environment = open_semantic_p50_environment_contract(
        environment_contract_path,
        runtime_contract_path=runtime_contract_path,
        scratch_runtime=scratch_runtime,
        prerequisite_paths=prerequisite_paths,
        expected_source_binding=expected_source_binding,
        trainer_source_path=trainer_source_path,
        image_definition_path=image_definition_path,
        expected_source_revision=expected_source_revision,
    )
    by_role = {item.role: item for item in prerequisites.bindings}
    by_role["semantic_p50_runtime_contract"] = _binding_from_opened(
        runtime_contract_path,
        artifact_root=prerequisite_paths.artifact_root,
        role="semantic_p50_runtime_contract",
        payload=runtime,
        semantic_sha256_field="contract_sha256",
        expected_filename=RUNTIME_FILENAME,
    )
    by_role["semantic_p50_environment_contract"] = _binding_from_opened(
        environment_contract_path,
        artifact_root=prerequisite_paths.artifact_root,
        role="semantic_p50_environment_contract",
        payload=environment,
        semantic_sha256_field="contract_sha256",
        expected_filename=ENVIRONMENT_FILENAME,
    )
    by_role["semantic_p50_launch_projection"] = _binding_from_opened(
        launch_projection_path,
        artifact_root=prerequisite_paths.artifact_root,
        role="semantic_p50_launch_projection",
        payload=projection,
        semantic_sha256_field="projection_sha256",
        expected_filename=LAUNCH_FILENAME,
    )
    purpose_roles = {
        "stream_union_successor_cache": "p50_stream_union_successor_cache_completion",
        "validation_baseline": "semantic_p50_validation_baseline",
        "trainer_runtime": "semantic_p50_runtime_contract",
        "execution_environment": "semantic_p50_environment_contract",
        "launch_projection": "semantic_p50_launch_projection",
    }
    resolved_bindings = [
        {
            "purpose": purpose,
            "artifact": by_role[purpose_roles[purpose]].as_payload(),
        }
        for purpose in REQUIRED_BINDING_PURPOSES
    ]
    training_gate_refinement = _semantic_p50_training_gate_refinement(
        prepared=physical_prepared,
        prerequisites=prerequisites,
        repo_root=prerequisite_paths.repo_root,
    )
    body = {
        "schema": PERMIT_SCHEMA,
        "schema_version": 1,
        "status": PERMIT_STATUS,
        **P50_ONLY_AUTHORITY,
        "authorization_scope": "exact_one_shot_50_step_64_batch_semantic_p50_only",
        "optimizer_steps": 50,
        "batch_size": 64,
        "scheduled_nonterminal_examples": 3200,
        "resume": False,
        "resolved_physical_binding_purposes": list(REQUIRED_BINDING_PURPOSES),
        "unresolved_physical_bindings": [],
        "resolved_physical_bindings": resolved_bindings,
        "editing_training_gate_refinement": training_gate_refinement,
        "prepared_recipe_file_sha256": prepared_file_sha256,
        "prepared_recipe_sha256": physical_prepared["prepared_recipe_sha256"],
        "t1_decision_file_sha256": _file_sha(prerequisite_paths.t1_decision_path),
        "t1_decision_sha256": t1["decision_sha256"],
        "launch_projection_file_sha256": _file_sha(launch_projection_path),
        "launch_projection_sha256": projection["projection_sha256"],
        "run_identity_sha256": projection["run_identity_sha256"],
        "output_relative_path": projection["output_relative_path"],
        "source_revision_sha256": projection["source_revision"]["source_revision_sha256"],
    }
    return body, prerequisites, projection


def materialize_semantic_p50_execution_permit(*, output_root: Path, **physical_kwargs: Any) -> Path:
    """Publish a content-addressed, P50-only permit after all strict reopeners pass."""

    body, _, _ = _expected_semantic_p50_execution_permit(**physical_kwargs)
    permit = {**body, "permit_sha256": _sha(body)}
    artifact_root = Path(physical_kwargs["prerequisite_paths"].artifact_root).resolve()
    root = _require_inside(output_root, root=artifact_root, field="permit output_root")
    path = root / permit["permit_sha256"] / PERMIT_FILENAME
    _publish_once(path, permit)
    open_semantic_p50_execution_permit(path, **physical_kwargs)
    return path


def open_semantic_p50_execution_permit(
    path: Path, **physical_kwargs: Any
) -> VerifiedSemanticP50ExecutionPermit:
    """Strictly reopen a content-addressed one-shot P50 permit."""

    artifact_root = Path(physical_kwargs["prerequisite_paths"].artifact_root).resolve()
    physical_path = _require_inside(path, root=artifact_root, field="execution permit")
    permit, _ = _load_canonical(
        physical_path,
        field="semantic P50 execution permit",
        filename=PERMIT_FILENAME,
    )
    body = dict(permit)
    supplied = body.pop("permit_sha256", None)
    expected, prerequisites, projection = _expected_semantic_p50_execution_permit(**physical_kwargs)
    if (
        permit != {**expected, "permit_sha256": _sha(expected)}
        or supplied != _sha(body)
        or physical_path.parent.name != supplied
        or any(permit.get(name) is not value for name, value in P50_ONLY_AUTHORITY.items())
        or permit.get("unresolved_physical_bindings") != []
        or tuple(permit.get("resolved_physical_binding_purposes", ())) != REQUIRED_BINDING_PURPOSES
    ):
        raise SemanticP50ExecutionContractError(
            "semantic P50 execution permit identity or authority disagrees"
        )
    return VerifiedSemanticP50ExecutionPermit(
        path=physical_path,
        permit=permit,
        prerequisites=prerequisites,
        launch_projection=projection,
    )


def _physical_observed_environment() -> dict[str, Any]:
    """Collect the current container state; absence of a GPU or image identity fails closed."""

    import numpy
    import torch
    from rdkit import rdBase

    image_sha = os.environ.get("COMPOSE_IMAGE_CONTENT_SHA256", "")
    _require_sha(image_sha, field="COMPOSE_IMAGE_CONTENT_SHA256")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SemanticP50ExecutionContractError(
            "observed P50 environment requires exactly one CUDA GPU"
        )
    memory_mb = 0
    if hasattr(os, "sysconf"):
        try:
            memory_mb = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1 << 20))
        except (OSError, ValueError):
            memory_mb = 0
    hardware = {
        "accelerator_class": "gpu",
        "modal_gpu_type": _require_text(os.environ.get("MODAL_GPU_TYPE"), field="MODAL_GPU_TYPE"),
        "device_type": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": ".".join(str(item) for item in torch.cuda.get_device_capability(0)),
        "cuda_device_count": 1,
        "cpu_count": os.cpu_count() or 0,
        "memory_mb": memory_mb,
    }
    software = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "rdkit_version": str(rdBase.rdkitVersion),
        "numpy_version": str(numpy.__version__),
    }
    _validate_expected_environment(hardware, software)
    return {
        "image_content_sha256": image_sha,
        "hardware": hardware,
        "software": software,
    }


def observe_semantic_p50_physical_environment() -> dict[str, Any]:
    """Observe the live prospective P50 container without granting authority."""

    return _physical_observed_environment()


def build_semantic_p50_observed_environment_receipt(
    *,
    launch_projection_path: Path,
    observed_at_utc: str,
    **physical_kwargs: Any,
) -> dict[str, Any]:
    projection = open_semantic_p50_launch_projection(launch_projection_path, **physical_kwargs)
    environment_path = physical_kwargs["environment_contract_path"]
    environment, environment_raw = _load_canonical(
        environment_path,
        field="semantic P50 environment contract",
        filename=ENVIRONMENT_FILENAME,
    )
    observed = observe_semantic_p50_physical_environment()
    if (
        observed["image_content_sha256"] != environment["image"]["content_sha256"]
        or observed["hardware"] != environment["hardware"]
        or observed["software"] != environment["software"]
    ):
        raise SemanticP50ExecutionContractError(
            "observed container differs from prospective environment"
        )
    if not isinstance(observed_at_utc, str) or _UTC_RE.fullmatch(observed_at_utc) is None:
        raise SemanticP50ExecutionContractError("observed_at_utc must be second-resolution UTC")
    _, launch_raw = _load_canonical(
        launch_projection_path,
        field="semantic P50 launch projection",
        filename=LAUNCH_FILENAME,
    )
    body = {
        "schema": ENVIRONMENT_RECEIPT_SCHEMA,
        "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
        "status": ENVIRONMENT_RECEIPT_STATUS,
        **NO_AUTHORITY,
        "environment_contract_file_sha256": hashlib.sha256(environment_raw).hexdigest(),
        "environment_contract_sha256": environment["contract_sha256"],
        "launch_projection_file_sha256": hashlib.sha256(launch_raw).hexdigest(),
        "launch_projection_sha256": projection["projection_sha256"],
        "observed_at_utc": observed_at_utc,
        "observed_image_content_sha256": observed["image_content_sha256"],
        "observed_hardware": observed["hardware"],
        "observed_software": observed["software"],
        "completed_preflight_checks": list(EXECUTION_ORDER[:6]),
    }
    return {**body, "receipt_sha256": _sha(body)}


def publish_semantic_p50_observed_environment_receipt(
    path: Path, receipt: Mapping[str, Any]
) -> Path:
    return _publish_once(path, receipt)


def open_semantic_p50_observed_environment_receipt(
    path: Path,
    *,
    launch_projection_path: Path,
    reobserve: bool = True,
    **physical_kwargs: Any,
) -> dict[str, Any]:
    receipt, _ = _load_canonical(
        path,
        field="observed environment receipt",
        filename=ENVIRONMENT_RECEIPT_FILENAME,
    )
    fields = {
        "schema",
        "schema_version",
        "status",
        *NO_AUTHORITY,
        "environment_contract_file_sha256",
        "environment_contract_sha256",
        "launch_projection_file_sha256",
        "launch_projection_sha256",
        "observed_at_utc",
        "observed_image_content_sha256",
        "observed_hardware",
        "observed_software",
        "completed_preflight_checks",
        "receipt_sha256",
    }
    if set(receipt) != fields:
        raise SemanticP50ExecutionContractError("observed environment receipt fields disagree")
    body = dict(receipt)
    supplied = body.pop("receipt_sha256")
    _require_nonauthorizing(receipt, field="observed environment receipt")
    projection = open_semantic_p50_launch_projection(launch_projection_path, **physical_kwargs)
    environment_path = physical_kwargs["environment_contract_path"]
    environment = open_semantic_p50_environment_contract(
        environment_path,
        runtime_contract_path=physical_kwargs["runtime_contract_path"],
        scratch_runtime=physical_kwargs["scratch_runtime"],
        prerequisite_paths=physical_kwargs["prerequisite_paths"],
        expected_source_binding=physical_kwargs["expected_source_binding"],
        trainer_source_path=physical_kwargs["trainer_source_path"],
        image_definition_path=physical_kwargs["image_definition_path"],
        expected_source_revision=physical_kwargs.get("expected_source_revision"),
    )
    if (
        receipt.get("schema") != ENVIRONMENT_RECEIPT_SCHEMA
        or receipt.get("schema_version") != ENVIRONMENT_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != ENVIRONMENT_RECEIPT_STATUS
        or supplied != _sha(body)
        or receipt.get("environment_contract_file_sha256") != _file_sha(environment_path)
        or receipt.get("environment_contract_sha256") != environment["contract_sha256"]
        or receipt.get("launch_projection_file_sha256") != _file_sha(launch_projection_path)
        or receipt.get("launch_projection_sha256") != projection["projection_sha256"]
        or not isinstance(receipt.get("observed_at_utc"), str)
        or _UTC_RE.fullmatch(receipt["observed_at_utc"]) is None
        or receipt.get("observed_image_content_sha256") != environment["image"]["content_sha256"]
        or receipt.get("observed_hardware") != environment["hardware"]
        or receipt.get("observed_software") != environment["software"]
        or tuple(receipt.get("completed_preflight_checks", ())) != EXECUTION_ORDER[:6]
    ):
        raise SemanticP50ExecutionContractError("observed environment physical binding disagrees")
    if reobserve:
        observed = observe_semantic_p50_physical_environment()
        if (
            observed["image_content_sha256"] != receipt["observed_image_content_sha256"]
            or observed["hardware"] != receipt["observed_hardware"]
            or observed["software"] != receipt["observed_software"]
        ):
            raise SemanticP50ExecutionContractError(
                "current container differs from observed receipt"
            )
    return receipt


def require_semantic_p50_output_available(
    launch_projection_path: Path,
    *,
    execution_permit_path: Path,
    environment_receipt_path: Path,
    artifact_root: Path,
    **physical_kwargs: Any,
) -> Path:
    """Validate physical launch state and atomically reserve its no-resume output."""

    projection = open_semantic_p50_launch_projection(launch_projection_path, **physical_kwargs)
    prepared, _ = load_semantic_p50_prepared_recipe(
        physical_kwargs["prerequisite_paths"].prepared_recipe_path
    )
    permit = open_semantic_p50_execution_permit(
        execution_permit_path,
        prepared=prepared,
        launch_projection_path=launch_projection_path,
        **physical_kwargs,
    )
    if permit.launch_projection != projection:
        raise SemanticP50ExecutionContractError("execution permit names another launch projection")
    receipt = open_semantic_p50_observed_environment_receipt(
        environment_receipt_path,
        launch_projection_path=launch_projection_path,
        reobserve=True,
        **physical_kwargs,
    )
    root = Path(artifact_root).resolve()
    prerequisite_paths = physical_kwargs.get("prerequisite_paths")
    if (
        not isinstance(prerequisite_paths, SemanticP50ExecutionPrerequisitePaths)
        or Path(prerequisite_paths.artifact_root).resolve() != root
    ):
        raise SemanticP50ExecutionContractError(
            "reservation artifact_root differs from the physically reopened prerequisite root"
        )
    destination = _require_inside(
        root / projection["output_relative_path"], root=root, field="P50 output"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as error:
        raise SemanticP50ExecutionContractError(
            f"semantic P50 no-resume output collision: {destination}"
        ) from error
    reservation_body = {
        "schema": "compose.editing_v2.semantic_p50_output_reservation",
        "schema_version": 1,
        **NO_AUTHORITY,
        "run_identity_sha256": projection["run_identity_sha256"],
        "launch_projection_file_sha256": _file_sha(launch_projection_path),
        "launch_projection_sha256": projection["projection_sha256"],
        "environment_receipt_file_sha256": _file_sha(environment_receipt_path),
        "environment_receipt_sha256": receipt["receipt_sha256"],
        "execution_permit_file_sha256": _file_sha(execution_permit_path),
        "execution_permit_sha256": permit.permit["permit_sha256"],
    }
    reservation = {**reservation_body, "reservation_sha256": _sha(reservation_body)}
    try:
        _publish_once(destination / RESERVATION_FILENAME, reservation)
    except BaseException:
        try:
            destination.rmdir()
        except OSError:
            pass
        raise
    return destination


def canonical_semantic_p50_contract_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(dict(value), newline=True)


__all__ = [
    "ARGV_SCHEMA",
    "ENVIRONMENT_FILENAME",
    "ENVIRONMENT_RECEIPT_FILENAME",
    "ENVIRONMENT_RECEIPT_SCHEMA",
    "ENVIRONMENT_RECEIPT_STATUS",
    "ENVIRONMENT_SCHEMA",
    "ENVIRONMENT_STATUS",
    "EXECUTION_ORDER",
    "LAUNCH_ARTIFACT_ROLES",
    "LAUNCH_FILENAME",
    "LAUNCH_SCHEMA",
    "NO_AUTHORITY",
    "OBJECTIVE_CONTRACT",
    "OPTIMIZER_CONTRACT",
    "P50_ONLY_AUTHORITY",
    "PERMIT_FILENAME",
    "PERMIT_SCHEMA",
    "RUNTIME_FILENAME",
    "RUNTIME_SCHEMA",
    "RUNTIME_STATUS",
    "SOURCE_INVENTORY_ROLE",
    "TRAINING_CONTRACT",
    "UPSTREAM_RUNTIME_ROLES",
    "SemanticP50ExecutionContractError",
    "SemanticP50ExecutionPrerequisitePaths",
    "SemanticP50PhysicalArtifactBinding",
    "VerifiedSemanticP50ExecutionPermit",
    "VerifiedSemanticP50ExecutionPrerequisites",
    "bind_canonical_semantic_p50_artifact",
    "build_semantic_p50_environment_contract",
    "build_semantic_p50_launch_projection",
    "build_semantic_p50_observed_environment_receipt",
    "build_semantic_p50_runtime_contract",
    "canonical_semantic_p50_contract_bytes",
    "materialize_semantic_p50_execution_permit",
    "observe_clean_source_revision",
    "observe_semantic_p50_physical_environment",
    "open_semantic_p50_environment_contract",
    "open_semantic_p50_execution_permit",
    "open_semantic_p50_execution_prerequisites",
    "open_semantic_p50_launch_projection",
    "open_semantic_p50_observed_environment_receipt",
    "open_semantic_p50_runtime_contract",
    "publish_semantic_p50_observed_environment_receipt",
    "require_semantic_p50_output_available",
    "validate_semantic_p50_environment_contract",
    "validate_semantic_p50_launch_projection",
    "validate_semantic_p50_runtime_contract",
]
