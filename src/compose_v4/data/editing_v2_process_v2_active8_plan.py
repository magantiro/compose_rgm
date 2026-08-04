"""The content-addressed plan for the Process-V2 Active8 stage.

One plan per run.  It addresses everything a task's answer depends on, so two
runs whose plans have the same ``plan_sha256`` cannot disagree about what was
evaluated, and a run whose inputs moved cannot reuse the previous run's
namespace:

* the **chunk-cache completion** the source rows are read from;
* the **rebind completion** whose per-entry decision says which of those rows
  Process V2 admitted;
* the **admitted-source identity**, the resolved join of the two above;
* the **Process-V2 identity**, which moves whenever the frozen fiber moves;
* the **Active8 admission policy**, which decides what a supported teacher is;
* the **model runtime**, because the candidate fiber is enumerated by a model
  and a different model is a different fiber;
* the **code revision** of the modules that decide, classify or publish here.

One Active8 map task per source chunk.  The chunk is the unit the cache already
publishes and the unit the rebind already decided over, so a task reads exactly
one verified chunk and exactly one rebind task result, and never a packed shard.

This module holds no evidence and grants no authority.  Every field of the
frozen authority vocabulary is ``False``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compose_v4.data.editing_process_v2_rebind import (
    PRODUCTION_SOURCE_GEOMETRY,
    chunk_target_for_task,
    require_production_source_geometry,
)
from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_SCHEMA as REBIND_COMPLETION_SCHEMA,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    ProcessV2ChunkTarget,
    require_process_v2_artifact_path,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACTIVE8_TASKS_DIRNAME,
    PIPELINE_STATUS_NO_AUTHORITY,
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    self_hashed,
    verify_self_hash,
)
from compose_v4.data.editing_v2_process_v2_active8_admission import (
    SemanticActive8AdmissionPolicy,
    build_process_v2_semantic_active8_admission_policy,
    validate_process_v2_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    load_semantic_capability_cell_registry,
)
from compose_v4.experiments.editing_v2_semantic_development_cell_roles import (
    load_semantic_development_cell_roles,
)
from compose_v4.rewrite.editing_v2_process_identity import editing_process_v2_identity

DEFAULT_OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_active8"
TASK_DIRNAME = ACTIVE8_TASKS_DIRNAME
PLAN_FILENAME = "PROCESS_V2_ACTIVE8_PLAN.json"

BINDING_SCHEMA = "compose.data.editing_v2_process_v2_active8_binding"
BINDING_SCHEMA_VERSION = 1

#: Exactly the modules that decide, classify or publish an Active8 answer.  A
#: change to any of them relocates the run rather than silently reusing it.
IMPLEMENTATION_FILES: tuple[str, ...] = (
    "src/compose_v4/data/editing_v2_process_v2_active8_map.py",
    "src/compose_v4/data/editing_v2_process_v2_active8_plan.py",
    "src/compose_v4/data/editing_v2_process_v2_active8_reduce.py",
    "src/compose_v4/data/editing_v2_process_v2_active8_sentinel.py",
    "src/compose_v4/data/editing_v2_process_v2_pipeline_schema.py",
    "src/compose_v4/data/editing_v2_process_v2_active8_admission.py",
    "src/compose_v4/data/editing_v2_semantic_active8_admission.py",
    "src/compose_v4/data/editing_v2_semantic_capability_cells.py",
)

MODEL_RUNTIME_FIELDS: tuple[str, ...] = (
    "atom_vocabulary_class_count",
    "catalog_fingerprint",
    "dtype",
    "hidden_dim",
    "initial_model_state_sha256",
    "initialization_seed",
    "mark_dim",
    "max_atoms",
    "message_passing_steps",
    "model_identity_sha256",
    "operator_capability_fingerprint",
    "semantic_model_process_contract_sha256",
)

BINDING_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "active8_policy_sha256",
        "admitted_source_sha256",
        "cache_completion_sha256",
        "cache_physical_identity_sha256",
        "cache_run_artifact_root",
        "cache_semantic_identity_sha256",
        "capability_cell_registry_sha256",
        "cell_role_policy_sha256",
        "code_revision_sha256",
        "implementation_files_sha256",
        "model_runtime",
        "process_v2_identity_sha256",
        "rebind_completion_sha256",
        "rebind_plan_sha256",
        "rebind_run_artifact_root",
        "records_per_chunk",
        "required_cell_ids",
        "binding_sha256",
    }
)

TASK_FIELDS: frozenset[str] = frozenset(
    {
        "binding_sha256",
        "cache_physical_identity_sha256",
        "cache_semantic_identity_sha256",
        "cache_source_artifact_path",
        "cache_source_manifest_sha256",
        "cache_source_task_identity_sha256",
        "chunk_file_sha256",
        "chunk_filename",
        "chunk_index",
        "chunk_row_count",
        "chunk_uncompressed_sha256",
        "data_lane",
        "entry_start",
        "entry_stop",
        "output_artifact_path",
        "pinned_process_identity_sha256",
        "rebind_task_artifact_path",
        "rebind_task_identity_sha256",
        "split",
        "task_identity_sha256",
        "v1_semantic_shard_sha256",
        "v1_task_identity_sha256",
    }
)

PLAN_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "binding",
        "binding_sha256",
        "expected_entry_count",
        "expected_task_count",
        "output_artifact_prefix",
        "run_artifact_root",
        "run_identity_sha256",
        "source_geometry",
        "task_inventory_sha256",
        "tasks",
        "plan_sha256",
    }
)


class ProcessV2Active8PlanError(RuntimeError):
    """The Active8 plan is stale, malformed, or does not bind its inputs."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_files_sha256(*, repo_root: Path) -> dict[str, str]:
    """Hash every module whose contents change an Active8 answer."""

    root = Path(repo_root)
    revision: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise ProcessV2Active8PlanError(
                f"the Active8 implementation source is missing: {relative}"
            )
        revision[relative] = _file_sha256(path)
    return revision


def model_runtime_descriptor(runtime: Any) -> dict[str, Any]:
    """The deterministic model-runtime block a plan binds.

    Built from a constructed :class:`SemanticScratchRuntime` rather than from a
    request, so the plan addresses the model that exists and not the one that
    was asked for.
    """

    architecture = runtime.architecture
    config = runtime.config
    descriptor = {
        "atom_vocabulary_class_count": int(architecture.atom_vocabulary_class_count),
        "catalog_fingerprint": str(architecture.catalog_fingerprint),
        "dtype": str(architecture.dtype),
        "hidden_dim": int(architecture.hidden_dim),
        "initial_model_state_sha256": str(runtime.initial_model_state_sha256),
        "initialization_seed": int(config.initialization_seed),
        "mark_dim": int(architecture.mark_dim),
        "max_atoms": int(architecture.max_atoms),
        "message_passing_steps": int(architecture.message_passing_steps),
        "model_identity_sha256": canonical_sha256(dict(runtime.semantic_model_identity)),
        "operator_capability_fingerprint": str(
            architecture.operator_capability_fingerprint
        ),
        "semantic_model_process_contract_sha256": str(
            runtime.semantic_model_process_contract_sha256
        ),
    }
    if tuple(sorted(descriptor)) != MODEL_RUNTIME_FIELDS:
        raise ProcessV2Active8PlanError("the model runtime descriptor fields disagree")
    return descriptor


def build_process_v2_active8_binding(
    *,
    cache_completion: Mapping[str, Any],
    rebind_plan: Mapping[str, Any],
    rebind_completion: Mapping[str, Any],
    admitted_source_identity: Mapping[str, Any],
    model_runtime: Mapping[str, Any],
    repo_root: Path,
    policy: SemanticActive8AdmissionPolicy | None = None,
) -> dict[str, Any]:
    """Bind every input whose value changes what an Active8 task answers."""

    selected = validate_process_v2_semantic_active8_admission_policy(
        policy or build_process_v2_semantic_active8_admission_policy()
    )
    if sorted(model_runtime) != list(MODEL_RUNTIME_FIELDS):
        raise ProcessV2Active8PlanError("the bound model runtime descriptor is incomplete")
    if rebind_completion.get("schema") != REBIND_COMPLETION_SCHEMA:
        raise ProcessV2Active8PlanError("the bound rebind completion has another schema")
    require_production_source_geometry(rebind_plan, label="the bound Process-V2 rebind plan")
    live_identity = str(editing_process_v2_identity()["process_identity_sha256"])
    for label, value in (
        ("rebind plan", rebind_plan["process_v2_identity"]["process_identity_sha256"]),
        ("rebind completion", rebind_completion["process_v2_identity_sha256"]),
        ("admitted source", admitted_source_identity["process_v2_identity_sha256"]),
    ):
        if str(value) != live_identity:
            raise ProcessV2Active8PlanError(
                f"the bound {label} was produced under another Process-V2 identity"
            )
    if str(rebind_completion["plan_sha256"]) != str(rebind_plan["plan_sha256"]):
        raise ProcessV2Active8PlanError("the bound rebind completion seals another plan")
    if str(admitted_source_identity["plan_sha256"]) != str(rebind_plan["plan_sha256"]):
        raise ProcessV2Active8PlanError("the bound admitted source resolves another rebind plan")
    cache_binding = rebind_plan["cache_binding"]
    if str(cache_binding["cache_completion_sha256"]) != str(
        cache_completion["completion_sha256"]
    ):
        raise ProcessV2Active8PlanError(
            "the bound chunk cache is not the one the rebind was planned against"
        )
    revision = implementation_files_sha256(repo_root=repo_root)
    registry = load_semantic_capability_cell_registry()
    cell_roles = load_semantic_development_cell_roles()
    body = {
        "schema": BINDING_SCHEMA,
        "schema_version": BINDING_SCHEMA_VERSION,
        "active8_policy_sha256": selected.policy_sha256,
        "admitted_source_sha256": str(admitted_source_identity["admitted_source_sha256"]),
        "cache_completion_sha256": str(cache_completion["completion_sha256"]),
        "cache_physical_identity_sha256": str(
            cache_completion["cache_physical_identity_sha256"]
        ),
        "cache_run_artifact_root": str(cache_completion["run_artifact_root"]),
        "cache_semantic_identity_sha256": str(
            cache_binding["cache_semantic_identity_sha256"]
        ),
        "capability_cell_registry_sha256": registry.registry_sha256,
        "cell_role_policy_sha256": cell_roles.policy_sha256,
        "code_revision_sha256": canonical_sha256(revision),
        "implementation_files_sha256": dict(revision),
        "model_runtime": dict(model_runtime),
        "process_v2_identity_sha256": live_identity,
        "rebind_completion_sha256": str(rebind_completion["completion_sha256"]),
        "rebind_plan_sha256": str(rebind_plan["plan_sha256"]),
        "rebind_run_artifact_root": str(rebind_plan["run_artifact_root"]),
        "records_per_chunk": int(cache_completion["records_per_chunk"]),
        "required_cell_ids": list(cell_roles.required_cell_ids),
    }
    binding = self_hashed(body, field="binding_sha256")
    if set(binding) != BINDING_FIELDS:
        raise ProcessV2Active8PlanError("the Active8 binding field set disagrees")
    return binding


def _task_body(
    *,
    target: ProcessV2ChunkTarget,
    rebind_task: Mapping[str, Any],
    binding_sha256: str,
    pinned_process_identity_sha256: str,
) -> dict[str, Any]:
    return {
        "binding_sha256": binding_sha256,
        "cache_physical_identity_sha256": target.cache_physical_identity_sha256,
        "cache_semantic_identity_sha256": target.cache_semantic_identity_sha256,
        "cache_source_artifact_path": target.source_artifact_path,
        "cache_source_manifest_sha256": target.cache_source_manifest_sha256,
        "cache_source_task_identity_sha256": target.cache_source_task_identity_sha256,
        "chunk_file_sha256": target.chunk_file_sha256,
        "chunk_filename": target.chunk_filename,
        "chunk_index": target.chunk_index,
        "chunk_row_count": target.row_count,
        "chunk_uncompressed_sha256": target.chunk_uncompressed_sha256,
        "data_lane": target.data_lane,
        "entry_start": target.entry_start,
        "entry_stop": target.entry_stop,
        "pinned_process_identity_sha256": pinned_process_identity_sha256,
        "rebind_task_artifact_path": str(rebind_task["output_artifact_path"]),
        "rebind_task_identity_sha256": str(rebind_task["task_identity_sha256"]),
        "split": target.split,
        "v1_semantic_shard_sha256": target.semantic_shard_sha256,
        "v1_task_identity_sha256": target.v1_task_identity_sha256,
    }


def plan_process_v2_active8(
    binding: Mapping[str, Any],
    *,
    rebind_plan: Mapping[str, Any],
    output_artifact_prefix: str = DEFAULT_OUTPUT_ARTIFACT_PREFIX,
) -> dict[str, Any]:
    """Plan exactly one Active8 map task per source chunk.

    The chunk inventory is taken from the rebind plan's own cache-geometry
    tasks rather than re-derived from the cache, so an Active8 task and the
    rebind decision it consumes address the same chunk by construction.
    """

    if set(binding) != BINDING_FIELDS:
        raise ProcessV2Active8PlanError("the Active8 binding field set disagrees")
    verify_self_hash(binding, field="binding_sha256", label="the Active8 binding")
    require_process_v2_artifact_path(output_artifact_prefix, field="output_artifact_prefix")
    geometry = require_production_source_geometry(
        rebind_plan, label="the Process-V2 rebind plan"
    )
    if str(rebind_plan["plan_sha256"]) != str(binding["rebind_plan_sha256"]):
        raise ProcessV2Active8PlanError("the Active8 binding names another rebind plan")
    pinned = str(rebind_plan["pinned_process_identity"]["process_identity_sha256"])
    tasks: list[dict[str, Any]] = []
    seen_chunks: set[tuple[str, int]] = set()
    for rebind_task in rebind_plan["tasks"]:
        target = chunk_target_for_task(rebind_task, pinned_process_identity_sha256=pinned)
        chunk_key = (target.cache_source_task_identity_sha256, target.chunk_index)
        if chunk_key in seen_chunks:
            raise ProcessV2Active8PlanError(
                "the rebind plan reads one source chunk twice, so an Active8 task "
                "would decide the same entries twice"
            )
        seen_chunks.add(chunk_key)
        body = _task_body(
            target=target,
            rebind_task=rebind_task,
            binding_sha256=str(binding["binding_sha256"]),
            pinned_process_identity_sha256=pinned,
        )
        tasks.append({**body, "task_identity_sha256": canonical_sha256(body)})
    tasks.sort(key=lambda task: (task["data_lane"], task["split"], task["entry_start"]))
    identities = [str(task["task_identity_sha256"]) for task in tasks]
    if len(set(identities)) != len(identities):
        raise ProcessV2Active8PlanError("two Active8 tasks share one identity")
    task_inventory_sha256 = canonical_sha256(identities)
    run_identity_sha256 = canonical_sha256(
        {
            "binding_sha256": str(binding["binding_sha256"]),
            "output_artifact_prefix": output_artifact_prefix,
            "schema": PLAN_SCHEMA,
            "schema_version": PLAN_SCHEMA_VERSION,
            "task_inventory_sha256": task_inventory_sha256,
        }
    )
    run_artifact_root = f"{output_artifact_prefix}/{run_identity_sha256}"
    for task in tasks:
        task["output_artifact_path"] = (
            f"{run_artifact_root}/{TASK_DIRNAME}/{task['task_identity_sha256']}"
        )
        if set(task) != TASK_FIELDS:
            raise ProcessV2Active8PlanError("the Active8 task field set disagrees")
    body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "binding": dict(binding),
        "binding_sha256": str(binding["binding_sha256"]),
        "expected_entry_count": sum(int(task["chunk_row_count"]) for task in tasks),
        "expected_task_count": len(tasks),
        "output_artifact_prefix": output_artifact_prefix,
        "run_artifact_root": run_artifact_root,
        "run_identity_sha256": run_identity_sha256,
        "source_geometry": geometry,
        "task_inventory_sha256": task_inventory_sha256,
        "tasks": tasks,
    }
    return self_hashed(body, field="plan_sha256")


def validate_process_v2_active8_plan(
    value: object, *, repo_root: Path, live: bool = True
) -> dict[str, Any]:
    """Require the exact plan shape, its self-hashes, and its live identities.

    ``live`` re-derives the Process-V2 identity, the Active8 policy and the
    implementation revision and requires equality, so a plan whose meaning has
    moved cannot be executed under its old namespace.
    """

    if not isinstance(value, Mapping):
        raise ProcessV2Active8PlanError("the Active8 plan must be a mapping")
    plan = dict(value)
    if set(plan) != PLAN_FIELDS:
        raise ProcessV2Active8PlanError("the Active8 plan field set disagrees")
    if plan["schema"] != PLAN_SCHEMA or plan["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ProcessV2Active8PlanError("the Active8 plan schema disagrees")
    if plan["status"] != PIPELINE_STATUS_NO_AUTHORITY:
        raise ProcessV2Active8PlanError("the Active8 plan status disagrees")
    if plan["source_geometry"] != PRODUCTION_SOURCE_GEOMETRY:
        raise ProcessV2Active8PlanError("the Active8 plan is not on the production geometry")
    try:
        require_authority_false(plan, label="the Active8 plan")
    except ProcessV2SchemaError as error:
        raise ProcessV2Active8PlanError(str(error)) from error
    verify_self_hash(plan, field="plan_sha256", label="the Active8 plan")
    binding = plan["binding"]
    if not isinstance(binding, Mapping) or set(binding) != BINDING_FIELDS:
        raise ProcessV2Active8PlanError("the Active8 plan binding field set disagrees")
    verify_self_hash(binding, field="binding_sha256", label="the Active8 plan binding")
    if plan["binding_sha256"] != binding["binding_sha256"]:
        raise ProcessV2Active8PlanError("the Active8 plan and its binding disagree")
    tasks = plan["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ProcessV2Active8PlanError("the Active8 plan publishes no task")
    identities: list[str] = []
    entries = 0
    for task in tasks:
        if not isinstance(task, Mapping) or set(task) != TASK_FIELDS:
            raise ProcessV2Active8PlanError("the Active8 task field set disagrees")
        body = {
            key: item
            for key, item in task.items()
            if key not in {"task_identity_sha256", "output_artifact_path"}
        }
        if canonical_sha256(body) != task["task_identity_sha256"]:
            raise ProcessV2Active8PlanError("an Active8 task identity disagrees with its body")
        if task["binding_sha256"] != binding["binding_sha256"]:
            raise ProcessV2Active8PlanError("an Active8 task binds another input set")
        expected_path = (
            f"{plan['run_artifact_root']}/{TASK_DIRNAME}/{task['task_identity_sha256']}"
        )
        if task["output_artifact_path"] != expected_path:
            raise ProcessV2Active8PlanError("an Active8 task publishes outside its run namespace")
        identities.append(str(task["task_identity_sha256"]))
        entries += int(task["chunk_row_count"])
    if sorted(identities) != sorted(set(identities)):
        raise ProcessV2Active8PlanError("two Active8 tasks share one identity")
    ordered = [
        str(task["task_identity_sha256"])
        for task in sorted(
            tasks, key=lambda item: (item["data_lane"], item["split"], item["entry_start"])
        )
    ]
    if identities != ordered:
        raise ProcessV2Active8PlanError(
            "the Active8 task inventory is not in lane, split, entry order"
        )
    if canonical_sha256(identities) != plan["task_inventory_sha256"]:
        raise ProcessV2Active8PlanError("the Active8 task inventory hash disagrees")
    if plan["expected_task_count"] != len(tasks) or plan["expected_entry_count"] != entries:
        raise ProcessV2Active8PlanError("the Active8 plan census disagrees with its tasks")
    run_identity_sha256 = canonical_sha256(
        {
            "binding_sha256": str(binding["binding_sha256"]),
            "output_artifact_prefix": str(plan["output_artifact_prefix"]),
            "schema": PLAN_SCHEMA,
            "schema_version": PLAN_SCHEMA_VERSION,
            "task_inventory_sha256": str(plan["task_inventory_sha256"]),
        }
    )
    if plan["run_identity_sha256"] != run_identity_sha256 or plan["run_artifact_root"] != (
        f"{plan['output_artifact_prefix']}/{run_identity_sha256}"
    ):
        raise ProcessV2Active8PlanError("the Active8 run identity disagrees with its plan")
    if live:
        _require_live_binding(binding, repo_root=repo_root)
    return plan


def _require_live_binding(binding: Mapping[str, Any], *, repo_root: Path) -> None:
    live_identity = str(editing_process_v2_identity()["process_identity_sha256"])
    if binding["process_v2_identity_sha256"] != live_identity:
        raise ProcessV2Active8PlanError(
            "the Active8 plan was built under a superseded Process-V2 identity"
        )
    policy = build_process_v2_semantic_active8_admission_policy()
    if binding["active8_policy_sha256"] != policy.policy_sha256:
        raise ProcessV2Active8PlanError(
            "the Active8 plan was built under a superseded admission policy"
        )
    revision = implementation_files_sha256(repo_root=repo_root)
    if binding["implementation_files_sha256"] != revision or binding[
        "code_revision_sha256"
    ] != canonical_sha256(revision):
        raise ProcessV2Active8PlanError(
            "the Active8 implementation has moved since the plan was built"
        )
    registry = load_semantic_capability_cell_registry()
    if binding["capability_cell_registry_sha256"] != registry.registry_sha256:
        raise ProcessV2Active8PlanError(
            "the Active8 plan binds a superseded capability-cell registry"
        )
    cell_roles = load_semantic_development_cell_roles()
    if binding["cell_role_policy_sha256"] != cell_roles.policy_sha256 or list(
        binding["required_cell_ids"]
    ) != list(cell_roles.required_cell_ids):
        raise ProcessV2Active8PlanError(
            "the Active8 plan binds a superseded capability-cell role policy"
        )


def task_by_identity(plan: Mapping[str, Any], task_identity_sha256: str) -> dict[str, Any]:
    """Return exactly one planned task, or refuse."""

    matches = [
        dict(task)
        for task in plan["tasks"]
        if str(task["task_identity_sha256"]) == str(task_identity_sha256)
    ]
    if len(matches) != 1:
        raise ProcessV2Active8PlanError(
            f"the Active8 plan does not name exactly one task {task_identity_sha256}"
        )
    return matches[0]


__all__ = [
    "BINDING_FIELDS",
    "BINDING_SCHEMA",
    "BINDING_SCHEMA_VERSION",
    "DEFAULT_OUTPUT_ARTIFACT_PREFIX",
    "IMPLEMENTATION_FILES",
    "MODEL_RUNTIME_FIELDS",
    "PLAN_FIELDS",
    "PLAN_FILENAME",
    "TASK_DIRNAME",
    "TASK_FIELDS",
    "ProcessV2Active8PlanError",
    "build_process_v2_active8_binding",
    "implementation_files_sha256",
    "model_runtime_descriptor",
    "plan_process_v2_active8",
    "task_by_identity",
    "validate_process_v2_active8_plan",
]
