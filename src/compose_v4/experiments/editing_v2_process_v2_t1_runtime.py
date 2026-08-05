"""Minimal Process-V2 T1 prepared-input and runtime boundary.

The Process-V2 Active8 and Gate-0 chain deliberately stops before molecular
successor compilation.  This module performs that one missing CPU operation for
the bounded, unique-state T1 panel:

* reopen the exact persistent-slot source and target states through the
  Process-V2 cache/rebind path;
* enumerate the complete productive canonical-successor partition once on CPU;
* verify the selected teacher action and exact successor are present;
* publish restart-safe plan, leaf, prepared-input, and completion artifacts; and
* expose a small runtime view consumed by the existing successor-level T1
  optimization core.

It does not train, score, select a checkpoint, grant P50 authority, or rebuild
any upstream corpus artifact.  The map boundary is the already-authenticated
Active8 source chunk, so each selected chunk is reopened at most once per leaf.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_process_v2_active8_plan import (
    model_runtime_descriptor,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)
from compose_v4.data.immutable_artifact import ImmutableArtifactError, write_bytes_if_absent
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    load_gate_zero_semantic_contract,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    ACTIVE8_DECISION_RUNTIME,
    DEVELOPMENT_CELL_ROLES,
    GATE_ZERO_MODEL_PROCESS_V2,
    T1_CAPACITY_POLICY,
    load_process_v2_chain_artifact,
)
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
    ProcessV2T1PanelError,
    ProcessV2T1Source,
    resolve_process_v2_t1_entries,
    validate_process_v2_t1_panel,
)
from compose_v4.experiments.editing_v2_semantic_runtime import (
    SemanticScratchModelConfig,
    SemanticScratchRuntime,
    build_semantic_scratch_runtime,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    compiled_successor_map_from_payload,
    compiled_successor_map_payload,
)
from compose_v4.experiments.factorized_successor_training import (
    CompiledStateSuccessorMap,
    SuccessorTrainingError,
    TeacherSuccessorFiber,
    compile_state_successor_map,
    require_exact_successor_action_identity,
    rewrite_action_codec_sha256,
    teacher_successor_fiber_from_exact_digest,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from compose_v4.rewrite import action_codec_v4

PLAN_SCHEMA = "compose.editing_v2.process_v2_t1_prepared_plan"
PLAN_SCHEMA_VERSION = 1
PLAN_STATUS = "PROCESS_V2_T1_PREPARED_PLAN_NO_DOWNSTREAM_AUTHORITY"
PLAN_FILENAME = "PROCESS_V2_T1_PREPARED_PLAN.json"
LEAF_SCHEMA = "compose.editing_v2.process_v2_t1_prepared_leaf"
LEAF_SCHEMA_VERSION = 1
LEAF_STATUS = "PROCESS_V2_T1_PREPARED_LEAF_NO_DOWNSTREAM_AUTHORITY"
LEAF_FILENAME = "PROCESS_V2_T1_PREPARED_LEAF.json"
PREPARED_SCHEMA = "compose.editing_v2.process_v2_t1_prepared_inputs"
PREPARED_SCHEMA_VERSION = 1
PREPARED_STATUS = "PROCESS_V2_T1_EXACT_SUCCESSOR_INPUTS_NO_DOWNSTREAM_AUTHORITY"
PREPARED_FILENAME = "PROCESS_V2_T1_PREPARED_INPUTS.json"
COMPLETION_SCHEMA = "compose.editing_v2.process_v2_t1_prepared_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_STATUS = "PROCESS_V2_T1_PREPARED_COMPLETE_NO_DOWNSTREAM_AUTHORITY"
COMPLETION_FILENAME = "PROCESS_V2_T1_PREPARED_COMPLETE.json"
TASKS_DIRNAME = "tasks"

_PREPARED_ENTRY_FIELDS = frozenset(
    {
        "panel_entry_sha256",
        "model_family",
        "capability_cell_id",
        "support_time_hex",
        "source_state_sha256",
        "target_state_sha256",
        "successor_canonical_key",
        "teacher_action_sha256",
        "objective_coefficient",
        "raw_mark_count",
        "canonical_successor_count",
        "production_successor_alias_multiplicity",
        "exact_state",
        "successor_partition",
        "productive_alias_count",
        "virtual_alias_count",
        "model_scores_or_probabilities_stored",
        "hazard_included",
        "entry_sha256",
    }
)

_IMPLEMENTATION_FILES = (
    "src/compose_v4/experiments/editing_v2_process_v2_t1_runtime.py",
    "src/compose_v4/experiments/editing_v2_process_v2_t1_panel.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_prepared_inputs.py",
    "src/compose_v4/experiments/editing_v2_semantic_runtime.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
)


class ProcessV2T1RuntimeError(RuntimeError):
    """A Process-V2 T1 prerequisite or exact successor partition is invalid."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessV2T1RuntimeError(f"{field} must be a full lowercase SHA-256")
    return value


def _load_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2T1RuntimeError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict) or raw != canonical_bytes(value) + b"\n":
        raise ProcessV2T1RuntimeError(f"{label} is not canonical newline-framed JSON")
    return value, raw


def _implementation_sha256(repo_root: Path) -> str:
    root = Path(repo_root).resolve()
    rows: list[dict[str, str]] = []
    for relative in _IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise ProcessV2T1RuntimeError(f"T1 implementation source is absent: {relative}")
        rows.append({"path": relative, "file_sha256": _file_sha256(path)})
    return canonical_sha256(rows)


def _validate_source_revision(value: Mapping[str, Any]) -> dict[str, Any]:
    revision = dict(value)
    if set(revision) != {"commit", "source_revision_sha256"}:
        raise ProcessV2T1RuntimeError("T1 source revision fields disagree")
    commit = revision["commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or revision["source_revision_sha256"]
        != canonical_sha256({"commit": commit})
    ):
        raise ProcessV2T1RuntimeError("T1 source revision is not an exact commit binding")
    return revision


def load_process_v2_t1_capacity_policy(
    path: Path, *, repo_root: Path
) -> tuple[dict[str, Any], str]:
    """Load only the current Process-V2 sparse capacity protocol."""

    source = Path(path).resolve()
    expected = (Path(repo_root) / T1_CAPACITY_POLICY).resolve()
    if source != expected:
        raise ProcessV2T1RuntimeError("T1 capacity policy path is not the Process-V2 contract")
    policy = load_process_v2_chain_artifact(T1_CAPACITY_POLICY, repo_root=repo_root)
    if (
        policy.get("schema") != "compose.editing_v2.process_v2_t1_capacity_policy"
        or tuple(policy.get("required_families", ()))
        != (
            "atom_insert",
            "atom_delete",
            "atom_restate",
            "bond_reorder",
            "bond_reroute",
            "cycle_insert",
            "cycle_attach",
            "ring_system_restate",
        )
        or policy.get("panel_kind")
        != "unique_state_single_target_canonical_successor_capacity"
        or policy.get("hazard_included") is not False
        or policy.get("training_authorized") is not False
        or policy.get("bounded_p50_authorized") is not False
        or policy.get("optimization", {}).get("trajectory_evaluation")
        != "initial_report_points_and_terminal_state"
    ):
        raise ProcessV2T1RuntimeError("T1 capacity policy is outside Process-V2 T1 scope")
    return {**policy, "policy_sha256": policy["contract_sha256"]}, _file_sha256(source)


def _scratch_runtime_from_descriptor(
    descriptor: Mapping[str, Any], *, repo_root: Path
) -> tuple[SemanticScratchRuntime, dict[str, Any]]:
    runtime_contract = load_process_v2_chain_artifact(
        ACTIVE8_DECISION_RUNTIME, repo_root=repo_root
    )
    semantic_path = Path(repo_root) / GATE_ZERO_MODEL_PROCESS_V2
    semantic = load_gate_zero_semantic_contract(semantic_path)
    expected_parent = runtime_contract["parents"]["semantic_model_process"]
    if (
        semantic.file_sha256 != expected_parent["physical"]["sha256"]
        or semantic.sha256 != expected_parent["semantic"]["sha256"]
    ):
        raise ProcessV2T1RuntimeError("Process-V2 semantic model contract bytes disagree")
    config = SemanticScratchModelConfig(
        initialization_seed=int(descriptor["initialization_seed"]),
        max_atoms=int(descriptor["max_atoms"]),
        hidden_dim=int(descriptor["hidden_dim"]),
        message_passing_steps=int(descriptor["message_passing_steps"]),
        mark_dim=int(descriptor["mark_dim"]),
        dtype=str(descriptor["dtype"]),
        atom_vocabulary_class_count=int(descriptor["atom_vocabulary_class_count"]),
        catalog_fingerprint=str(descriptor["catalog_fingerprint"]),
    )
    runtime = build_semantic_scratch_runtime(config, semantic)
    observed = model_runtime_descriptor(runtime)
    if observed != dict(descriptor) or observed[
        "initial_model_state_sha256"
    ] != state_dict_semantic_sha256(runtime.model.state_dict()):
        raise ProcessV2T1RuntimeError("scratch model differs from its bound descriptor")
    binding = {
        "active8_decision_runtime_file_sha256": _file_sha256(
            Path(repo_root) / ACTIVE8_DECISION_RUNTIME
        ),
        "active8_decision_runtime_sha256": runtime_contract["contract_sha256"],
        "semantic_model_process_file_sha256": semantic.file_sha256,
        "semantic_model_process_sha256": semantic.sha256,
        "model_runtime": observed,
    }
    return runtime, binding


def build_process_v2_t1_scratch_runtime(
    source: ProcessV2T1Source,
) -> tuple[SemanticScratchRuntime, dict[str, Any]]:
    """Reconstruct the exact scratch model already bound by Active8."""

    bound = source.plan.get("binding", {}).get("model_runtime")
    if not isinstance(bound, Mapping):
        raise ProcessV2T1RuntimeError("Active8 plan lacks its model runtime descriptor")
    runtime, binding = _scratch_runtime_from_descriptor(bound, repo_root=source.repo_root)
    if runtime.process_identity_sha256 != source.contracts.process_identity_sha256:
        raise ProcessV2T1RuntimeError("scratch model differs from the Active8-bound runtime")
    return runtime, binding


def build_process_v2_t1_prepared_plan(
    panel: Mapping[str, Any],
    *,
    source: ProcessV2T1Source,
    source_revision: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze one CPU compilation task per selected Active8 source chunk."""

    try:
        validated_panel = validate_process_v2_t1_panel(panel, source=source)
    except ProcessV2T1PanelError as error:
        raise ProcessV2T1RuntimeError(f"T1 panel is invalid: {error}") from error
    revision = _validate_source_revision(source_revision)
    _runtime, model_binding = build_process_v2_t1_scratch_runtime(source)
    entries_by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in validated_panel["entries"]:
        identity = str(entry["task_identity_sha256"])
        entries_by_task.setdefault(identity, []).append(dict(entry))
    active8_tasks = {
        str(task["task_identity_sha256"]): dict(task) for task in source.plan["tasks"]
    }
    if set(entries_by_task) - set(active8_tasks):
        raise ProcessV2T1RuntimeError("T1 panel names an Active8 task outside its plan")
    panel_bytes = canonical_bytes(validated_panel) + b"\n"
    base = {
        "source_revision": revision,
        "implementation_sha256": _implementation_sha256(source.repo_root),
        "process_identity_sha256": source.contracts.process_identity_sha256,
        "active8_completion_sha256": source.index.active8_completion_sha256,
        "active8_sentinel_sha256": source.index.active8_sentinel_sha256,
        "active8_plan_sha256": source.plan["plan_sha256"],
        "active8_run_identity_sha256": source.plan["run_identity_sha256"],
        "gate_zero_decision_sha256": source.decision["decision_sha256"],
        "panel_sha256": validated_panel["panel_sha256"],
        "panel_file_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "panel_file_bytes": len(panel_bytes),
        "panel_entry_inventory_sha256": canonical_sha256(
            sorted(str(entry["panel_entry_sha256"]) for entry in validated_panel["entries"])
        ),
        "support_time_hex": validated_panel["support_time_hex"],
        "model_binding": model_binding,
    }
    build_identity_sha256 = canonical_sha256(base)
    tasks: list[dict[str, Any]] = []
    for task_index, active8_identity in enumerate(sorted(entries_by_task)):
        entries = sorted(
            entries_by_task[active8_identity], key=lambda item: item["panel_entry_sha256"]
        )
        body = {
            "task_index": task_index,
            "active8_task_identity_sha256": active8_identity,
            "active8_chunk_file_sha256": active8_tasks[active8_identity][
                "chunk_file_sha256"
            ],
            "panel_entry_sha256s": [entry["panel_entry_sha256"] for entry in entries],
            "panel_entry_count": len(entries),
        }
        tasks.append(
            {
                **body,
                "task_identity_sha256": canonical_sha256(
                    {"build_identity_sha256": build_identity_sha256, "task": body}
                ),
            }
        )
    body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PLAN_STATUS,
        **authority_false_block(),
        **base,
        "build_identity_sha256": build_identity_sha256,
        "task_count": len(tasks),
        "task_inventory_sha256": canonical_sha256(tasks),
        "selected_entry_count": len(validated_panel["entries"]),
        "tasks": tasks,
    }
    with_run = {**body, "run_identity_sha256": canonical_sha256(body)}
    return {**with_run, "plan_sha256": canonical_sha256(with_run)}


def validate_process_v2_t1_prepared_plan(
    value: object,
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
) -> dict[str, Any]:
    """Rebuild a plan from its exact inputs and require byte identity."""

    if not isinstance(value, Mapping):
        raise ProcessV2T1RuntimeError("T1 prepared plan must be an object")
    plan = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "source_revision",
        "implementation_sha256",
        "process_identity_sha256",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "active8_plan_sha256",
        "active8_run_identity_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "panel_file_sha256",
        "panel_file_bytes",
        "panel_entry_inventory_sha256",
        "support_time_hex",
        "model_binding",
        "build_identity_sha256",
        "task_count",
        "task_inventory_sha256",
        "selected_entry_count",
        "tasks",
        "run_identity_sha256",
        "plan_sha256",
    }
    if set(plan) != expected_fields:
        raise ProcessV2T1RuntimeError("T1 prepared plan field set disagrees")
    try:
        verify_self_hash(plan, field="plan_sha256", label="the T1 prepared plan")
        require_authority_false(plan, label="the T1 prepared plan")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["schema_version"] != PLAN_SCHEMA_VERSION
        or plan["status"] != PLAN_STATUS
        or plan["task_count"] != len(plan["tasks"])
        or plan["task_inventory_sha256"] != canonical_sha256(plan["tasks"])
        or plan["selected_entry_count"]
        != sum(int(task["panel_entry_count"]) for task in plan["tasks"])
    ):
        raise ProcessV2T1RuntimeError("T1 prepared plan identity or census disagrees")
    rebuilt = build_process_v2_t1_prepared_plan(
        panel,
        source=source,
        source_revision=_validate_source_revision(plan["source_revision"]),
    )
    if canonical_bytes(rebuilt) != canonical_bytes(plan):
        raise ProcessV2T1RuntimeError("T1 prepared plan differs from exact recomputation")
    return plan


def authenticate_process_v2_t1_prepared_plan_for_worker(
    value: object,
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
) -> dict[str, Any]:
    """Authenticate coordinator-validated inputs without rescanning the corpus.

    The coordinator builds the panel from the complete eligible decision stream
    and freezes its canonical bytes in the plan.  A map worker must authenticate
    those exact bytes and the plan's derivable identities, but must not repeat
    the global panel-selection scan before compiling its assigned states.
    """

    if not isinstance(value, Mapping) or not isinstance(panel, Mapping):
        raise ProcessV2T1RuntimeError("T1 worker plan and panel must be objects")
    plan = dict(value)
    panel_value = dict(panel)
    expected_plan_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "source_revision",
        "implementation_sha256",
        "process_identity_sha256",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "active8_plan_sha256",
        "active8_run_identity_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "panel_file_sha256",
        "panel_file_bytes",
        "panel_entry_inventory_sha256",
        "support_time_hex",
        "model_binding",
        "build_identity_sha256",
        "task_count",
        "task_inventory_sha256",
        "selected_entry_count",
        "tasks",
        "run_identity_sha256",
        "plan_sha256",
    }
    if set(plan) != expected_plan_fields:
        raise ProcessV2T1RuntimeError("T1 worker plan field set disagrees")
    panel_bytes = canonical_bytes(panel_value) + b"\n"
    try:
        verify_self_hash(plan, field="plan_sha256", label="the T1 prepared plan")
        require_authority_false(plan, label="the T1 prepared plan")
        verify_self_hash(panel_value, field="panel_sha256", label="the T1 panel")
        require_authority_false(panel_value, label="the T1 panel")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    source_revision = _validate_source_revision(plan.get("source_revision", {}))
    without_plan_sha = {
        key: item for key, item in plan.items() if key != "plan_sha256"
    }
    without_run_identity = {
        key: item
        for key, item in without_plan_sha.items()
        if key != "run_identity_sha256"
    }
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ProcessV2T1RuntimeError("T1 worker plan tasks must be a list")
    expected_task_fields = {
        "task_index",
        "active8_task_identity_sha256",
        "active8_chunk_file_sha256",
        "panel_entry_sha256s",
        "panel_entry_count",
        "task_identity_sha256",
    }
    active8_tasks = {
        str(task["task_identity_sha256"]): task for task in source.plan["tasks"]
    }
    task_ids: list[str] = []
    entry_ids: list[str] = []
    for index, raw_task in enumerate(tasks):
        if not isinstance(raw_task, Mapping) or set(raw_task) != expected_task_fields:
            raise ProcessV2T1RuntimeError("T1 worker task field set disagrees")
        task = dict(raw_task)
        active8_identity = str(task["active8_task_identity_sha256"])
        active8_task = active8_tasks.get(active8_identity)
        identifiers = task["panel_entry_sha256s"]
        task_body = {
            key: item for key, item in task.items() if key != "task_identity_sha256"
        }
        if (
            type(task["task_index"]) is not int
            or task["task_index"] != index
            or active8_task is None
            or task["active8_chunk_file_sha256"]
            != active8_task["chunk_file_sha256"]
            or not isinstance(identifiers, list)
            or not identifiers
            or identifiers != sorted(identifiers)
            or len(identifiers) != len(set(identifiers))
            or type(task["panel_entry_count"]) is not int
            or task["panel_entry_count"] != len(identifiers)
            or task["task_identity_sha256"]
            != canonical_sha256(
                {
                    "build_identity_sha256": plan.get("build_identity_sha256"),
                    "task": task_body,
                }
            )
        ):
            raise ProcessV2T1RuntimeError("T1 worker task identity or census disagrees")
        task_ids.append(_require_sha(task["task_identity_sha256"], field="task identity"))
        entry_ids.extend(_require_sha(item, field="panel entry") for item in identifiers)
    base_keys = {
        "source_revision",
        "implementation_sha256",
        "process_identity_sha256",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "active8_plan_sha256",
        "active8_run_identity_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "panel_file_sha256",
        "panel_file_bytes",
        "panel_entry_inventory_sha256",
        "support_time_hex",
        "model_binding",
    }
    base = {key: plan.get(key) for key in base_keys}
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("implementation_sha256")
        != _implementation_sha256(source.repo_root)
        or plan.get("process_identity_sha256")
        != source.contracts.process_identity_sha256
        or plan.get("active8_completion_sha256")
        != source.index.active8_completion_sha256
        or plan.get("active8_sentinel_sha256") != source.index.active8_sentinel_sha256
        or plan.get("active8_plan_sha256") != source.plan["plan_sha256"]
        or plan.get("active8_run_identity_sha256")
        != source.plan["run_identity_sha256"]
        or plan.get("gate_zero_decision_sha256")
        != source.decision["decision_sha256"]
        or plan.get("panel_sha256") != panel_value.get("panel_sha256")
        or plan.get("panel_file_sha256") != hashlib.sha256(panel_bytes).hexdigest()
        or plan.get("panel_file_bytes") != len(panel_bytes)
        or plan.get("panel_entry_inventory_sha256")
        != canonical_sha256(sorted(entry_ids))
        or plan.get("build_identity_sha256") != canonical_sha256(base)
        or plan.get("run_identity_sha256") != canonical_sha256(without_run_identity)
        or plan.get("task_count") != len(tasks)
        or plan.get("task_inventory_sha256") != canonical_sha256(tasks)
        or plan.get("selected_entry_count") != len(entry_ids)
        or len(task_ids) != len(set(task_ids))
        or len(entry_ids) != len(set(entry_ids))
        or source_revision != plan.get("source_revision")
    ):
        raise ProcessV2T1RuntimeError("T1 worker plan, panel, or source binding disagrees")
    panel_entries = panel_value.get("entries")
    if not isinstance(panel_entries, list) or sorted(
        str(entry.get("panel_entry_sha256"))
        for entry in panel_entries
        if isinstance(entry, Mapping)
    ) != sorted(entry_ids):
        raise ProcessV2T1RuntimeError("T1 worker panel entry inventory disagrees")
    return plan


def write_process_v2_t1_prepared_plan(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
) -> Path:
    validated = validate_process_v2_t1_prepared_plan(plan, panel=panel, source=source)
    path = Path(output_root) / validated["run_identity_sha256"] / PLAN_FILENAME
    try:
        write_bytes_if_absent(path, canonical_bytes(validated) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    return path


def _entry_from_exact_transition(
    panel_entry: Mapping[str, Any],
    *,
    addressed: Any,
    scratch_runtime: SemanticScratchRuntime,
    support_time: float,
) -> dict[str, Any]:
    progress = int(panel_entry["progress_index"])
    source_state = addressed.path.state_at(progress)
    target_state = addressed.path.state_at(progress + 1)
    step = addressed.trace.steps[progress]
    observed_action_sha256 = rewrite_action_codec_sha256(
        step.rule_name,
        step.action,
        schema_version=action_codec_v4.SCHEMA_VERSION,
    )
    if (
        observed_action_sha256 != panel_entry["action_sha256"]
        or persistent_slot_state_sha256(source_state) != panel_entry["source_state_sha256"]
        or persistent_slot_state_sha256(target_state) != panel_entry["target_state_sha256"]
    ):
        raise ProcessV2T1RuntimeError("T1 selected action or exact state changed after paneling")
    try:
        partition = compile_state_successor_map(
            scratch_runtime.model,
            source_state,
            time=support_time,
        )
        require_exact_successor_action_identity(
            partition,
            target_state_sha256=str(panel_entry["target_state_sha256"]),
            action_sha256=str(panel_entry["action_sha256"]),
        )
        teacher = teacher_successor_fiber_from_exact_digest(
            partition, str(panel_entry["target_state_sha256"])
        )
    except (SuccessorTrainingError, ValueError) as error:
        raise ProcessV2T1RuntimeError(
            "T1 teacher is absent from its exact Process-V2 successor partition"
        ) from error
    state_payload = encode_state(source_state)
    if encode_state(decode_state(state_payload)) != state_payload:
        raise ProcessV2T1RuntimeError("T1 exact-state encoding is not byte-stable")
    partition_payload = compiled_successor_map_payload(partition)
    productive_alias_count = sum(len(group.marks) for group in partition.successor_groups)
    virtual_alias_count = len(partition.virtual_marks)
    body = {
        "panel_entry_sha256": panel_entry["panel_entry_sha256"],
        "model_family": panel_entry["model_family"],
        "capability_cell_id": panel_entry["capability_cell_id"],
        "support_time_hex": float(support_time).hex(),
        "source_state_sha256": panel_entry["source_state_sha256"],
        "target_state_sha256": panel_entry["target_state_sha256"],
        "successor_canonical_key": panel_entry["canonical_successor_key"],
        "teacher_action_sha256": panel_entry["action_sha256"],
        "objective_coefficient": 1,
        "raw_mark_count": productive_alias_count + virtual_alias_count,
        "canonical_successor_count": len(partition.successor_groups),
        "production_successor_alias_multiplicity": len(teacher.aliases),
        "exact_state": state_payload,
        "successor_partition": partition_payload,
        "productive_alias_count": productive_alias_count,
        "virtual_alias_count": virtual_alias_count,
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    return {**body, "entry_sha256": canonical_sha256(body)}


def compile_process_v2_t1_prepared_leaf(
    plan: Mapping[str, Any],
    *,
    task_identity_sha256: str,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    scratch_runtime: SemanticScratchRuntime,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile exact successor partitions for one selected Active8 chunk."""

    validated = validate_process_v2_t1_prepared_plan(plan, panel=panel, source=source)
    return _compile_validated_process_v2_t1_prepared_leaf(
        validated,
        task_identity_sha256=task_identity_sha256,
        panel=panel,
        source=source,
        scratch_runtime=scratch_runtime,
        progress_callback=progress_callback,
    )


def compile_authenticated_process_v2_t1_prepared_leaf(
    plan: Mapping[str, Any],
    *,
    task_identity_sha256: str,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    scratch_runtime: SemanticScratchRuntime,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compile one worker leaf from a coordinator-validated bound panel."""

    validated = authenticate_process_v2_t1_prepared_plan_for_worker(
        plan, panel=panel, source=source
    )
    return _compile_validated_process_v2_t1_prepared_leaf(
        validated,
        task_identity_sha256=task_identity_sha256,
        panel=panel,
        source=source,
        scratch_runtime=scratch_runtime,
        progress_callback=progress_callback,
    )


def _compile_validated_process_v2_t1_prepared_leaf(
    validated: Mapping[str, Any],
    *,
    task_identity_sha256: str,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    scratch_runtime: SemanticScratchRuntime,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> dict[str, Any]:
    task = next(
        (
            candidate
            for candidate in validated["tasks"]
            if candidate["task_identity_sha256"] == task_identity_sha256
        ),
        None,
    )
    if task is None:
        raise ProcessV2T1RuntimeError("T1 prepared task is absent from the plan")
    descriptor = model_runtime_descriptor(scratch_runtime)
    if descriptor != validated["model_binding"]["model_runtime"]:
        raise ProcessV2T1RuntimeError("T1 leaf scratch runtime differs from the plan")
    panel_by_id = {
        str(entry["panel_entry_sha256"]): dict(entry) for entry in panel["entries"]
    }
    try:
        selected = [panel_by_id[str(item)] for item in task["panel_entry_sha256s"]]
    except KeyError as error:
        raise ProcessV2T1RuntimeError("T1 prepared task names an absent panel entry") from error
    if any(
        entry["task_identity_sha256"] != task["active8_task_identity_sha256"]
        for entry in selected
    ):
        raise ProcessV2T1RuntimeError("T1 prepared task crosses Active8 source chunks")
    try:
        resolved = resolve_process_v2_t1_entries(source, selected)
    except ProcessV2T1PanelError as error:
        raise ProcessV2T1RuntimeError(f"exact T1 states could not be reopened: {error}") from error
    addressed_by_entry: dict[str, Any] = {}
    for item in resolved:
        for entry in item.panel_entries:
            identifier = str(entry["panel_entry_sha256"])
            if identifier in addressed_by_entry:
                raise ProcessV2T1RuntimeError("T1 exact entry was reopened twice")
            addressed_by_entry[identifier] = item.addressed_trace
    if set(addressed_by_entry) != set(task["panel_entry_sha256s"]):
        raise ProcessV2T1RuntimeError("T1 leaf did not reopen its complete exact entry set")
    support_time = float.fromhex(str(validated["support_time_hex"]))
    started_at = time.monotonic()
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "process_v2_t1_prepare_task_start",
                "task_identity_sha256": task["task_identity_sha256"],
                "entry_count": len(selected),
            }
        )
    entries: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(selected, start=1):
        entries.append(
            _entry_from_exact_transition(
                entry,
                addressed=addressed_by_entry[str(entry["panel_entry_sha256"])],
                scratch_runtime=scratch_runtime,
                support_time=support_time,
            )
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "process_v2_t1_prepare_entry_complete",
                    "task_identity_sha256": task["task_identity_sha256"],
                    "entry_index": entry_index,
                    "entry_count": len(selected),
                    "elapsed_seconds": time.monotonic() - started_at,
                }
            )
    entries.sort(key=lambda item: item["panel_entry_sha256"])
    body = {
        "schema": LEAF_SCHEMA,
        "schema_version": LEAF_SCHEMA_VERSION,
        "status": LEAF_STATUS,
        **authority_false_block(),
        "plan_sha256": validated["plan_sha256"],
        "run_identity_sha256": validated["run_identity_sha256"],
        "build_identity_sha256": validated["build_identity_sha256"],
        "task_identity_sha256": task["task_identity_sha256"],
        "active8_task_identity_sha256": task["active8_task_identity_sha256"],
        "panel_sha256": validated["panel_sha256"],
        "initial_model_state_sha256": descriptor["initial_model_state_sha256"],
        "entry_count": len(entries),
        "entry_inventory_sha256": canonical_sha256(entries),
        "entries": entries,
        "model_scores_or_probabilities_stored": False,
        "hazard_included": False,
    }
    leaf = {**body, "leaf_sha256": canonical_sha256(body)}
    return validate_process_v2_t1_prepared_leaf(leaf, plan=validated)


def validate_process_v2_t1_prepared_leaf(
    value: object, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1RuntimeError("T1 prepared leaf must be an object")
    leaf = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "plan_sha256",
        "run_identity_sha256",
        "build_identity_sha256",
        "task_identity_sha256",
        "active8_task_identity_sha256",
        "panel_sha256",
        "initial_model_state_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "entries",
        "model_scores_or_probabilities_stored",
        "hazard_included",
        "leaf_sha256",
    }
    if set(leaf) != expected_fields:
        raise ProcessV2T1RuntimeError("T1 prepared leaf field set disagrees")
    try:
        verify_self_hash(leaf, field="leaf_sha256", label="a T1 prepared leaf")
        require_authority_false(leaf, label="a T1 prepared leaf")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    task = next(
        (
            candidate
            for candidate in plan["tasks"]
            if candidate["task_identity_sha256"] == leaf["task_identity_sha256"]
        ),
        None,
    )
    if (
        task is None
        or leaf["schema"] != LEAF_SCHEMA
        or leaf["schema_version"] != LEAF_SCHEMA_VERSION
        or leaf["status"] != LEAF_STATUS
        or leaf["plan_sha256"] != plan["plan_sha256"]
        or leaf["run_identity_sha256"] != plan["run_identity_sha256"]
        or leaf["build_identity_sha256"] != plan["build_identity_sha256"]
        or leaf["panel_sha256"] != plan["panel_sha256"]
        or leaf["active8_task_identity_sha256"]
        != task["active8_task_identity_sha256"]
        or leaf["initial_model_state_sha256"]
        != plan["model_binding"]["model_runtime"]["initial_model_state_sha256"]
        or leaf["entry_count"] != len(leaf["entries"])
        or leaf["entry_inventory_sha256"] != canonical_sha256(leaf["entries"])
        or leaf["model_scores_or_probabilities_stored"] is not False
        or leaf["hazard_included"] is not False
    ):
        raise ProcessV2T1RuntimeError("T1 prepared leaf identity or census disagrees")
    observed_ids: list[str] = []
    for raw in leaf["entries"]:
        if not isinstance(raw, Mapping) or set(raw) != _PREPARED_ENTRY_FIELDS:
            raise ProcessV2T1RuntimeError("T1 prepared entry field set disagrees")
        entry = dict(raw)
        try:
            verify_self_hash(entry, field="entry_sha256", label="a T1 prepared entry")
            state = decode_state(entry["exact_state"])
            partition = compiled_successor_map_from_payload(entry["successor_partition"])
            teacher = teacher_successor_fiber_from_exact_digest(
                partition, str(entry["target_state_sha256"])
            )
            require_exact_successor_action_identity(
                partition,
                target_state_sha256=str(entry["target_state_sha256"]),
                action_sha256=str(entry["teacher_action_sha256"]),
            )
        except (ProcessV2SchemaError, SuccessorTrainingError, TypeError, ValueError) as error:
            raise ProcessV2T1RuntimeError("T1 prepared entry is internally invalid") from error
        productive = sum(len(group.marks) for group in partition.successor_groups)
        virtual = len(partition.virtual_marks)
        if (
            encode_state(state) != entry["exact_state"]
            or persistent_slot_state_sha256(state) != entry["source_state_sha256"]
            or partition.source_state_sha256 != entry["source_state_sha256"]
            or partition.source_key == entry["successor_canonical_key"]
            or teacher.target_key != entry["successor_canonical_key"]
            or entry["objective_coefficient"] != 1
            or entry["raw_mark_count"] != productive + virtual
            or entry["canonical_successor_count"] != len(partition.successor_groups)
            or entry["production_successor_alias_multiplicity"] != len(teacher.aliases)
            or entry["productive_alias_count"] != productive
            or entry["virtual_alias_count"] != virtual
            or entry["model_scores_or_probabilities_stored"] is not False
            or entry["hazard_included"] is not False
        ):
            raise ProcessV2T1RuntimeError("T1 prepared entry state or successor census disagrees")
        observed_ids.append(_require_sha(entry["panel_entry_sha256"], field="panel entry"))
    if (
        observed_ids != sorted(task["panel_entry_sha256s"])
        or len(set(observed_ids)) != len(observed_ids)
    ):
        raise ProcessV2T1RuntimeError("T1 prepared leaf entry inventory disagrees")
    return leaf


def write_process_v2_t1_prepared_leaf(
    leaf: Mapping[str, Any], *, plan: Mapping[str, Any], run_root: Path
) -> Path:
    validated = validate_process_v2_t1_prepared_leaf(leaf, plan=plan)
    path = (
        Path(run_root)
        / TASKS_DIRNAME
        / str(validated["task_identity_sha256"])
        / LEAF_FILENAME
    )
    try:
        write_bytes_if_absent(path, canonical_bytes(validated) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    return path


def _load_prepared_leaf(path: Path, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    leaf, _raw = _load_canonical(path, label="a T1 prepared leaf")
    return validate_process_v2_t1_prepared_leaf(leaf, plan=plan)


def build_process_v2_t1_prepared_inputs(
    plan: Mapping[str, Any],
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    run_root: Path,
) -> dict[str, Any]:
    """Reduce every exact task leaf into one score-independent runtime input."""

    validated = validate_process_v2_t1_prepared_plan(plan, panel=panel, source=source)
    leaves: list[dict[str, Any]] = []
    for task in validated["tasks"]:
        path = (
            Path(run_root)
            / TASKS_DIRNAME
            / str(task["task_identity_sha256"])
            / LEAF_FILENAME
        )
        if not path.is_file():
            raise ProcessV2T1RuntimeError(f"T1 prepared leaf is absent: {path}")
        leaves.append(_load_prepared_leaf(path, plan=validated))
    if len(leaves) != validated["task_count"]:
        raise ProcessV2T1RuntimeError("T1 prepared leaf count differs from the plan")
    entries = [dict(entry) for leaf in leaves for entry in leaf["entries"]]
    entries.sort(
        key=lambda item: (
            item["model_family"],
            item["capability_cell_id"],
            item["panel_entry_sha256"],
        )
    )
    observed_ids = [str(entry["panel_entry_sha256"]) for entry in entries]
    expected_ids = sorted(
        str(identifier)
        for task in validated["tasks"]
        for identifier in task["panel_entry_sha256s"]
    )
    if observed_ids != sorted(expected_ids) or len(set(observed_ids)) != len(observed_ids):
        raise ProcessV2T1RuntimeError("T1 prepared reduction omits or repeats a panel entry")
    leaf_receipts = [
        {
            "task_identity_sha256": leaf["task_identity_sha256"],
            "leaf_sha256": leaf["leaf_sha256"],
            "entry_count": leaf["entry_count"],
        }
        for leaf in sorted(leaves, key=lambda item: item["task_identity_sha256"])
    ]
    manifest_body = {
        "plan_sha256": validated["plan_sha256"],
        "run_identity_sha256": validated["run_identity_sha256"],
        "panel_sha256": validated["panel_sha256"],
        "leaf_inventory_sha256": canonical_sha256(leaf_receipts),
        "entry_inventory_sha256": canonical_sha256(observed_ids),
        "entry_count": len(entries),
    }
    manifest_sha256 = canonical_sha256(manifest_body)
    body = {
        "schema": PREPARED_SCHEMA,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": PREPARED_STATUS,
        **authority_false_block(),
        "source_revision": validated["source_revision"],
        "implementation_sha256": validated["implementation_sha256"],
        "process_identity_sha256": validated["process_identity_sha256"],
        "active8_completion_sha256": validated["active8_completion_sha256"],
        "active8_sentinel_sha256": validated["active8_sentinel_sha256"],
        "gate_zero_decision_sha256": validated["gate_zero_decision_sha256"],
        "panel_sha256": validated["panel_sha256"],
        "panel_entry_inventory_sha256": validated["panel_entry_inventory_sha256"],
        "plan_sha256": validated["plan_sha256"],
        "run_identity_sha256": validated["run_identity_sha256"],
        "build_identity_sha256": validated["build_identity_sha256"],
        "manifest_sha256": manifest_sha256,
        "model_binding": validated["model_binding"],
        "support_time_hex": validated["support_time_hex"],
        "state_encoding": "compose.rewrite.trace.encoded_state_v2_exact_slots",
        "successor_partition_contract": {
            "productive_groups_complete": True,
            "productive_aliases_disjoint": True,
            "virtual_aliases_disjoint": True,
            "self_events_excluded_from_embedded_jump_chain": True,
            "model_scores_or_probabilities_stored": False,
            "hazard_included": False,
        },
        "leaf_count": len(leaves),
        "leaf_inventory_sha256": canonical_sha256(leaf_receipts),
        "entry_count": len(entries),
        "entry_inventory_sha256": canonical_sha256(entries),
        "entries": entries,
    }
    artifact = {**body, "artifact_sha256": canonical_sha256(body)}
    return validate_process_v2_t1_prepared_inputs(
        artifact,
        expected_plan=validated,
    )


def validate_process_v2_t1_prepared_inputs(
    value: object,
    *,
    expected_plan: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1RuntimeError("T1 prepared inputs must be an object")
    artifact = dict(value)
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "source_revision",
        "implementation_sha256",
        "process_identity_sha256",
        "active8_completion_sha256",
        "active8_sentinel_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "panel_entry_inventory_sha256",
        "plan_sha256",
        "run_identity_sha256",
        "build_identity_sha256",
        "manifest_sha256",
        "model_binding",
        "support_time_hex",
        "state_encoding",
        "successor_partition_contract",
        "leaf_count",
        "leaf_inventory_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "entries",
        "artifact_sha256",
    }
    if set(artifact) != expected_fields:
        raise ProcessV2T1RuntimeError("T1 prepared input field set disagrees")
    try:
        verify_self_hash(artifact, field="artifact_sha256", label="the T1 prepared inputs")
        require_authority_false(artifact, label="the T1 prepared inputs")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    if (
        artifact["schema"] != PREPARED_SCHEMA
        or artifact["schema_version"] != PREPARED_SCHEMA_VERSION
        or artifact["status"] != PREPARED_STATUS
        or artifact["state_encoding"]
        != "compose.rewrite.trace.encoded_state_v2_exact_slots"
        or artifact["entry_count"] != len(artifact["entries"])
        or artifact["entry_inventory_sha256"] != canonical_sha256(artifact["entries"])
        or artifact["leaf_count"] <= 0
    ):
        raise ProcessV2T1RuntimeError("T1 prepared input identity or census disagrees")
    _validate_source_revision(artifact["source_revision"])
    if repo_root is not None and artifact["implementation_sha256"] != _implementation_sha256(
        repo_root
    ):
        raise ProcessV2T1RuntimeError("T1 prepared input implementation is stale")
    if expected_plan is not None:
        comparisons = {
            "source_revision": expected_plan["source_revision"],
            "implementation_sha256": expected_plan["implementation_sha256"],
            "process_identity_sha256": expected_plan["process_identity_sha256"],
            "active8_completion_sha256": expected_plan["active8_completion_sha256"],
            "active8_sentinel_sha256": expected_plan["active8_sentinel_sha256"],
            "gate_zero_decision_sha256": expected_plan["gate_zero_decision_sha256"],
            "panel_sha256": expected_plan["panel_sha256"],
            "panel_entry_inventory_sha256": expected_plan[
                "panel_entry_inventory_sha256"
            ],
            "plan_sha256": expected_plan["plan_sha256"],
            "run_identity_sha256": expected_plan["run_identity_sha256"],
            "build_identity_sha256": expected_plan["build_identity_sha256"],
            "model_binding": expected_plan["model_binding"],
            "support_time_hex": expected_plan["support_time_hex"],
        }
        if any(artifact.get(name) != expected for name, expected in comparisons.items()):
            raise ProcessV2T1RuntimeError("T1 prepared inputs bind another plan")
    observed: list[str] = []
    observed_sources: set[str] = set()
    for entry in artifact["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != _PREPARED_ENTRY_FIELDS:
            raise ProcessV2T1RuntimeError("T1 prepared entry field set disagrees")
        # Validate the entry body directly without pretending the synthetic
        # envelope belongs to a plan task.
        try:
            verify_self_hash(entry, field="entry_sha256", label="a T1 prepared entry")
            state = decode_state(entry["exact_state"])
            partition = compiled_successor_map_from_payload(entry["successor_partition"])
            teacher = teacher_successor_fiber_from_exact_digest(
                partition, str(entry["target_state_sha256"])
            )
            require_exact_successor_action_identity(
                partition,
                target_state_sha256=str(entry["target_state_sha256"]),
                action_sha256=str(entry["teacher_action_sha256"]),
            )
        except (ProcessV2SchemaError, SuccessorTrainingError, TypeError, ValueError) as error:
            raise ProcessV2T1RuntimeError("T1 prepared entry is internally invalid") from error
        productive = sum(len(group.marks) for group in partition.successor_groups)
        virtual = len(partition.virtual_marks)
        if (
            encode_state(state) != entry["exact_state"]
            or persistent_slot_state_sha256(state) != entry["source_state_sha256"]
            or partition.source_state_sha256 != entry["source_state_sha256"]
            or teacher.target_key != entry["successor_canonical_key"]
            or entry["raw_mark_count"] != productive + virtual
            or entry["canonical_successor_count"] != len(partition.successor_groups)
            or entry["production_successor_alias_multiplicity"] != len(teacher.aliases)
            or entry["productive_alias_count"] != productive
            or entry["virtual_alias_count"] != virtual
            or entry["model_scores_or_probabilities_stored"] is not False
            or entry["hazard_included"] is not False
        ):
            raise ProcessV2T1RuntimeError("T1 prepared entry state or successor census disagrees")
        observed.append(str(entry["panel_entry_sha256"]))
        source_sha256 = str(entry["source_state_sha256"])
        if source_sha256 in observed_sources:
            raise ProcessV2T1RuntimeError("T1 prepared inputs repeat an exact source state")
        observed_sources.add(source_sha256)
    if len(set(observed)) != len(observed):
        raise ProcessV2T1RuntimeError("T1 prepared inputs repeat a panel entry")
    return artifact


def publish_process_v2_t1_prepared_inputs(
    plan: Mapping[str, Any],
    *,
    panel: Mapping[str, Any],
    source: ProcessV2T1Source,
    run_root: Path,
) -> Path:
    """Atomically publish prepared bytes, then their completion receipt."""

    validated_plan = validate_process_v2_t1_prepared_plan(
        plan, panel=panel, source=source
    )
    artifact = build_process_v2_t1_prepared_inputs(
        validated_plan,
        panel=panel,
        source=source,
        run_root=run_root,
    )
    root = Path(run_root)
    artifact_path = root / PREPARED_FILENAME
    try:
        write_bytes_if_absent(artifact_path, canonical_bytes(artifact) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    raw = artifact_path.read_bytes()
    if raw != canonical_bytes(artifact) + b"\n":
        raise ProcessV2T1RuntimeError("published T1 prepared bytes changed on reopen")
    body = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **authority_false_block(),
        "plan_sha256": validated_plan["plan_sha256"],
        "run_identity_sha256": validated_plan["run_identity_sha256"],
        "process_identity_sha256": validated_plan["process_identity_sha256"],
        "active8_completion_sha256": validated_plan["active8_completion_sha256"],
        "gate_zero_decision_sha256": validated_plan["gate_zero_decision_sha256"],
        "panel_sha256": validated_plan["panel_sha256"],
        "manifest_sha256": artifact["manifest_sha256"],
        "prepared_filename": PREPARED_FILENAME,
        "prepared_file_sha256": hashlib.sha256(raw).hexdigest(),
        "prepared_file_bytes": len(raw),
        "prepared_artifact_sha256": artifact["artifact_sha256"],
        "initial_model_state_sha256": artifact["model_binding"]["model_runtime"][
            "initial_model_state_sha256"
        ],
        "entry_count": artifact["entry_count"],
        "entry_inventory_sha256": artifact["entry_inventory_sha256"],
        "training_launched": False,
    }
    completion = {**body, "completion_sha256": canonical_sha256(body)}
    completion_path = root / COMPLETION_FILENAME
    try:
        write_bytes_if_absent(completion_path, canonical_bytes(completion) + b"\n")
    except ImmutableArtifactError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    return completion_path


def _load_prepared_completion(
    completion_path: Path, *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    completion, completion_raw = _load_canonical(
        completion_path, label="the T1 prepared completion"
    )
    expected_fields = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "plan_sha256",
        "run_identity_sha256",
        "process_identity_sha256",
        "active8_completion_sha256",
        "gate_zero_decision_sha256",
        "panel_sha256",
        "manifest_sha256",
        "prepared_filename",
        "prepared_file_sha256",
        "prepared_file_bytes",
        "prepared_artifact_sha256",
        "initial_model_state_sha256",
        "entry_count",
        "entry_inventory_sha256",
        "training_launched",
        "completion_sha256",
    }
    if set(completion) != expected_fields:
        raise ProcessV2T1RuntimeError("T1 prepared completion field set disagrees")
    try:
        verify_self_hash(
            completion, field="completion_sha256", label="the T1 prepared completion"
        )
        require_authority_false(completion, label="the T1 prepared completion")
    except ProcessV2SchemaError as error:
        raise ProcessV2T1RuntimeError(str(error)) from error
    if (
        completion["schema"] != COMPLETION_SCHEMA
        or completion["schema_version"] != COMPLETION_SCHEMA_VERSION
        or completion["status"] != COMPLETION_STATUS
        or completion["prepared_filename"] != PREPARED_FILENAME
        or completion["training_launched"] is not False
    ):
        raise ProcessV2T1RuntimeError("T1 prepared completion identity disagrees")
    artifact_path = Path(completion_path).resolve().parent / PREPARED_FILENAME
    artifact, artifact_raw = _load_canonical(artifact_path, label="the T1 prepared inputs")
    artifact = validate_process_v2_t1_prepared_inputs(artifact, repo_root=repo_root)
    scratch, binding = _scratch_runtime_from_descriptor(
        artifact["model_binding"]["model_runtime"], repo_root=repo_root
    )
    if (
        completion["prepared_file_sha256"] != hashlib.sha256(artifact_raw).hexdigest()
        or completion["prepared_file_bytes"] != len(artifact_raw)
        or completion["prepared_artifact_sha256"] != artifact["artifact_sha256"]
        or completion["process_identity_sha256"] != artifact["process_identity_sha256"]
        or completion["active8_completion_sha256"]
        != artifact["active8_completion_sha256"]
        or completion["gate_zero_decision_sha256"]
        != artifact["gate_zero_decision_sha256"]
        or completion["panel_sha256"] != artifact["panel_sha256"]
        or completion["manifest_sha256"] != artifact["manifest_sha256"]
        or completion["initial_model_state_sha256"]
        != artifact["model_binding"]["model_runtime"]["initial_model_state_sha256"]
        or completion["entry_count"] != artifact["entry_count"]
        or completion["entry_inventory_sha256"] != artifact["entry_inventory_sha256"]
        or artifact["model_binding"] != binding
        or scratch.process_identity_sha256 != artifact["process_identity_sha256"]
    ):
        raise ProcessV2T1RuntimeError(
            "T1 completion, prepared bytes, and current scratch runtime disagree"
        )
    return (
        completion,
        artifact,
        hashlib.sha256(completion_raw).hexdigest(),
        hashlib.sha256(artifact_raw).hexdigest(),
    )


@dataclass(frozen=True)
class LoadedProcessV2T1PreparedInputs:
    artifact: Mapping[str, Any]
    states_by_panel_entry_sha256: Mapping[str, Any]
    partitions_by_panel_entry_sha256: Mapping[str, CompiledStateSuccessorMap]


@dataclass(frozen=True)
class ProcessV2T1CacheView:
    completion: Mapping[str, Any]
    manifest: Mapping[str, Any]
    teacher_fibers_by_panel_entry_sha256: Mapping[str, TeacherSuccessorFiber]

    def record_for_panel_entry_sha256(self, identifier: str) -> Any:
        try:
            fiber = self.teacher_fibers_by_panel_entry_sha256[identifier]
        except KeyError as error:
            raise ProcessV2T1RuntimeError(
                "T1 runtime requested a teacher outside the prepared panel"
            ) from error
        return SimpleNamespace(teacher_fiber=fiber)


@dataclass(frozen=True)
class ProcessV2T1RuntimeInputs:
    """Duck-typed runtime consumed by the existing successor-level T1 core."""

    prepared: LoadedProcessV2T1PreparedInputs
    cache: ProcessV2T1CacheView
    capacity_policy: Mapping[str, Any]

    @property
    def entries(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.prepared.artifact["entries"])


def load_process_v2_t1_runtime_inputs(
    completion_path: Path,
    *,
    capacity_policy_path: Path,
    repo_root: Path,
) -> tuple[ProcessV2T1RuntimeInputs, SemanticScratchRuntime, dict[str, Any]]:
    """Reopen CPU-prepared partitions and reconstruct the bound scratch model."""

    completion, artifact, completion_file_sha256, prepared_file_sha256 = (
        _load_prepared_completion(completion_path, repo_root=repo_root)
    )
    policy, policy_file_sha256 = load_process_v2_t1_capacity_policy(
        capacity_policy_path, repo_root=repo_root
    )
    scratch, binding = _scratch_runtime_from_descriptor(
        artifact["model_binding"]["model_runtime"], repo_root=repo_root
    )
    if binding != artifact["model_binding"]:
        raise ProcessV2T1RuntimeError("T1 scratch runtime differs from prepared inputs")
    states: dict[str, Any] = {}
    partitions: dict[str, CompiledStateSuccessorMap] = {}
    fibers: dict[str, TeacherSuccessorFiber] = {}
    observed_cells: set[str] = set()
    family_counts: dict[str, int] = {}
    for entry in artifact["entries"]:
        identifier = str(entry["panel_entry_sha256"])
        state = decode_state(entry["exact_state"])
        partition = compiled_successor_map_from_payload(entry["successor_partition"])
        fiber = teacher_successor_fiber_from_exact_digest(
            partition, str(entry["target_state_sha256"])
        )
        states[identifier] = state
        partitions[identifier] = partition
        fibers[identifier] = fiber
        observed_cells.add(str(entry["capability_cell_id"]))
        family = str(entry["model_family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    roles = load_process_v2_chain_artifact(DEVELOPMENT_CELL_ROLES, repo_root=repo_root)
    required_cells = {
        str(cell)
        for cell, role in roles["cell_roles"].items()
        if role == "required_editing"
    }
    if observed_cells != required_cells:
        raise ProcessV2T1RuntimeError(
            "T1 prepared inputs do not exactly cover the required Process-V2 cells"
        )
    minimums = policy.get("panel_cardinality", {}).get("minimum_entries_by_family", {})
    maximums = policy.get("panel_cardinality", {}).get("maximum_entries_by_family", {})
    if any(
        not int(minimums[family]) <= family_counts.get(family, 0) <= int(maximums[family])
        for family in policy["required_families"]
    ):
        raise ProcessV2T1RuntimeError("T1 prepared family counts violate the capacity policy")
    prepared = LoadedProcessV2T1PreparedInputs(
        artifact=MappingProxyType(dict(artifact)),
        states_by_panel_entry_sha256=MappingProxyType(states),
        partitions_by_panel_entry_sha256=MappingProxyType(partitions),
    )
    manifest = {
        "manifest_sha256": artifact["manifest_sha256"],
        "source_revision": artifact["source_revision"],
        "semantic_model_process_contract": {
            "file_sha256": binding["semantic_model_process_file_sha256"],
            "contract_sha256": binding["semantic_model_process_sha256"],
        },
        "panel_binding": {"panel_artifact_sha256": artifact["panel_sha256"]},
    }
    cache_completion = {
        **dict(completion),
        "initial_model_state_sha256": completion["initial_model_state_sha256"],
        "manifest_sha256": artifact["manifest_sha256"],
    }
    cache = ProcessV2T1CacheView(
        completion=MappingProxyType(cache_completion),
        manifest=MappingProxyType(manifest),
        teacher_fibers_by_panel_entry_sha256=MappingProxyType(fibers),
    )
    runtime = ProcessV2T1RuntimeInputs(
        prepared=prepared,
        cache=cache,
        capacity_policy=MappingProxyType(dict(policy)),
    )
    provenance = {
        "capacity_policy_file_sha256": policy_file_sha256,
        "capacity_policy_sha256": policy["policy_sha256"],
        "prepared_completion_file_sha256": completion_file_sha256,
        "prepared_completion_sha256": completion["completion_sha256"],
        "prepared_input_file_sha256": prepared_file_sha256,
        "prepared_input_artifact_sha256": artifact["artifact_sha256"],
        "panel_sha256": artifact["panel_sha256"],
        "panel_entry_inventory_sha256": artifact["panel_entry_inventory_sha256"],
        "gate_zero_decision_sha256": artifact["gate_zero_decision_sha256"],
        "active8_completion_sha256": artifact["active8_completion_sha256"],
        "process_identity_sha256": artifact["process_identity_sha256"],
        "initial_model_state_sha256": completion["initial_model_state_sha256"],
    }
    return runtime, scratch, provenance


__all__ = [
    "COMPLETION_FILENAME",
    "LEAF_FILENAME",
    "PLAN_FILENAME",
    "PREPARED_FILENAME",
    "LoadedProcessV2T1PreparedInputs",
    "ProcessV2T1CacheView",
    "ProcessV2T1RuntimeError",
    "ProcessV2T1RuntimeInputs",
    "authenticate_process_v2_t1_prepared_plan_for_worker",
    "build_process_v2_t1_prepared_inputs",
    "build_process_v2_t1_prepared_plan",
    "build_process_v2_t1_scratch_runtime",
    "compile_process_v2_t1_prepared_leaf",
    "compile_authenticated_process_v2_t1_prepared_leaf",
    "load_process_v2_t1_capacity_policy",
    "load_process_v2_t1_runtime_inputs",
    "publish_process_v2_t1_prepared_inputs",
    "validate_process_v2_t1_prepared_inputs",
    "validate_process_v2_t1_prepared_leaf",
    "validate_process_v2_t1_prepared_plan",
    "write_process_v2_t1_prepared_leaf",
    "write_process_v2_t1_prepared_plan",
]
