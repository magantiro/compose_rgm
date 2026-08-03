"""Deterministic reduction of the Active8 stage, gated by the release sentinel.

Three properties, and each of them is a refusal rather than a best effort:

* **Order and concurrency independent.**  The reduction reads the planned task
  inventory, not the directory, and folds every task in the plan's own order.
  Twenty containers finishing in any order reduce to the same bytes.
* **Complete or nothing.**  A single missing or unreconcilable task result
  raises :class:`ProcessV2Active8Incomplete` and publishes NOTHING.  There is no
  partial completion, because a partial completion of an admission census is
  indistinguishable from a smaller corpus.
* **Sentinel-gated.**  The reduction is computed, then the release sentinel runs
  over the published evidence, and only a passing sentinel is followed by the
  authoritative completion.  A mismatch leaves the run with published tasks and
  no completion, which is exactly the state a rerun can recover from.

Every aggregate here is recomputed from the published ROWS.  The task receipts
are revalidated against their own rows and then used only as an integrity
check -- never as the source of a number.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compose_v4.data.editing_process_v2_rebind import mounted_process_v2_artifact_path
from compose_v4.data.editing_v2_process_v2_active8_map import (
    ACCEPTED,
    EXCLUDED,
    NOT_EVALUATED,
    ProcessV2Active8MapError,
    derive_action_family_histogram,
    derive_candidate_totals,
    execute_process_v2_active8_task,
    read_task_rows,
    read_task_transitions,
    validate_process_v2_active8_task_result,
)
from compose_v4.data.editing_v2_process_v2_active8_plan import (
    model_runtime_descriptor,
    validate_process_v2_active8_plan,
)
from compose_v4.data.editing_v2_process_v2_active8_sentinel import (
    require_sentinel_passed,
    run_release_sentinel,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_COMPLETION_SCHEMA,
    ACTIVE8_COMPLETION_SCHEMA_VERSION,
    CANDIDATE_TOTAL_FIELDS,
    PIPELINE_STATUS_NO_AUTHORITY,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    self_hashed,
    verify_self_hash,
)
from compose_v4.data.editing_v2_semantic_active8_admission import (
    ProductionSemanticExactCandidateChecker,
    build_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    load_semantic_capability_cell_registry,
)

COMPLETION_FILENAME = "PROCESS_V2_ACTIVE8_COMPLETE.json"

COMPLETION_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "accepted_transitions",
        "action_family_histogram",
        "active8_exclusion_histogram",
        "binding_sha256",
        "candidate_totals",
        "capability_cell_histogram",
        "census",
        "classification_affects_admission",
        "plan_sha256",
        "result_inventory",
        "result_inventory_sha256",
        "run_artifact_root",
        "run_identity_sha256",
        "sentinel",
        "task_inventory_sha256",
        "completion_sha256",
    }
)


class ProcessV2Active8ReduceError(RuntimeError):
    """The Active8 reduction cannot be computed from the published tasks."""


class ProcessV2Active8Incomplete(ProcessV2Active8ReduceError):
    """A planned Active8 task result is absent or does not reconcile."""


def task_output_path(
    plan: Mapping[str, Any], task: Mapping[str, Any], *, artifact_root: Path
) -> Path:
    return mounted_process_v2_artifact_path(
        str(task["output_artifact_path"]),
        artifact_root=Path(artifact_root),
        field="task.output_artifact_path",
    )


def completed_process_v2_active8_task_ids(
    plan: Mapping[str, Any], *, artifact_root: Path
) -> set[str]:
    """The planned tasks whose published result reconciles from its own bytes."""

    complete: set[str] = set()
    for task in plan["tasks"]:
        output = task_output_path(plan, task, artifact_root=artifact_root)
        if not output.is_dir():
            continue
        try:
            receipt, summary = validate_process_v2_active8_task_result(output)
        except (ProcessV2Active8MapError, OSError, ValueError):
            continue
        if receipt["task_identity_sha256"] == str(task["task_identity_sha256"]) and summary[
            "plan_sha256"
        ] == str(plan["plan_sha256"]):
            complete.add(str(task["task_identity_sha256"]))
    return complete


def run_process_v2_active8_tasks(
    plan: Mapping[str, Any],
    *,
    runtime: Any,
    artifact_root: Path,
    repo_root: Path,
    reuse: bool = True,
) -> dict[str, dict[str, Any]]:
    """Execute every planned task, reusing exact valid receipts on a restart."""

    receipts: dict[str, dict[str, Any]] = {}
    for task in plan["tasks"]:
        identity = str(task["task_identity_sha256"])
        receipts[identity] = execute_process_v2_active8_task(
            plan,
            identity,
            runtime=runtime,
            artifact_root=Path(artifact_root),
            repo_root=Path(repo_root),
            reuse=reuse,
        )
    return receipts


def reduce_process_v2_active8(
    plan: Mapping[str, Any],
    *,
    runtime: Any,
    artifact_root: Path,
    repo_root: Path,
    publish: bool = True,
) -> dict[str, Any]:
    """Fold every planned task, run the sentinel, then publish the completion."""

    validated = validate_process_v2_active8_plan(plan, repo_root=Path(repo_root))
    binding = validated["binding"]
    if model_runtime_descriptor(runtime) != dict(binding["model_runtime"]):
        raise ProcessV2Active8ReduceError(
            "the supplied model runtime is not the one the Active8 plan binds"
        )
    census = dict.fromkeys(ACTIVE8_CENSUS_FIELDS, 0)
    totals = dict.fromkeys(CANDIDATE_TOTAL_FIELDS, 0)
    families: dict[str, int] = {}
    cells: dict[str, int] = {}
    exclusions: dict[str, int] = {}
    inventory: list[list[str]] = []
    transitions: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, int]] = set()

    for task in validated["tasks"]:
        identity = str(task["task_identity_sha256"])
        output = task_output_path(validated, task, artifact_root=artifact_root)
        if not output.is_dir():
            raise ProcessV2Active8Incomplete(
                f"the Active8 run is missing the result of task {identity}"
            )
        try:
            receipt, summary = validate_process_v2_active8_task_result(output)
        except ProcessV2Active8MapError as error:
            raise ProcessV2Active8Incomplete(
                f"the Active8 result of task {identity} does not reconcile"
            ) from error
        if (
            receipt["task_identity_sha256"] != identity
            or summary["plan_sha256"] != str(validated["plan_sha256"])
            or summary["run_identity_sha256"] != str(validated["run_identity_sha256"])
            or summary["binding_sha256"] != str(binding["binding_sha256"])
        ):
            raise ProcessV2Active8Incomplete(
                f"the Active8 result of task {identity} belongs to another run"
            )
        inventory.append([identity, str(receipt["receipt_sha256"])])
        rows = read_task_rows(output)
        task_transitions = read_task_transitions(output)
        for row in rows:
            key = (str(row["v1_task_identity_sha256"]), int(row["entry_index"]))
            if key in seen_rows:
                raise ProcessV2Active8ReduceError(
                    f"two Active8 tasks decide the same source entry: {key}"
                )
            seen_rows.add(key)
            census["source_entries"] += 1
            status = str(row["admission_status"])
            if status == ACCEPTED:
                census["active8_accepted_entries"] += 1
            elif status == EXCLUDED:
                census["active8_excluded_entries"] += 1
            elif status == NOT_EVALUATED:
                census["upstream_rejected_entries"] += 1
            else:
                raise ProcessV2Active8ReduceError(
                    f"unknown Active8 admission status: {status!r}"
                )
            for name, value in derive_candidate_totals(row["actions"]).items():
                totals[name] += value
            for name, value in derive_action_family_histogram(row["actions"]).items():
                families[name] = families.get(name, 0) + value
            for action in row["actions"]:
                evidence = action["candidate_evidence"]
                if evidence is not None and not evidence["supported"]:
                    reason = str(evidence["exclusion_reason"])
                    exclusions[reason] = exclusions.get(reason, 0) + 1
        for transition in task_transitions:
            cell = transition["capability_cell_id"]
            if cell is not None:
                cells[str(cell)] = cells.get(str(cell), 0) + 1
        transitions.extend(task_transitions)

    if census["source_entries"] != int(validated["expected_entry_count"]):
        raise ProcessV2Active8Incomplete(
            "the Active8 reduction covers a different entry count than the plan requires"
        )
    if (
        census["source_entries"]
        != census["upstream_rejected_entries"]
        + census["active8_accepted_entries"]
        + census["active8_excluded_entries"]
    ):
        raise ProcessV2Active8ReduceError("the Active8 run census does not reconcile")

    checker = ProductionSemanticExactCandidateChecker(
        runtime.model, policy=build_semantic_active8_admission_policy(process_v2=True)
    )
    sentinel = run_release_sentinel(
        validated,
        transitions,
        checker=checker,
        namespace=load_semantic_capability_cell_registry().namespace,
        artifact_root=Path(artifact_root),
        repo_root=Path(repo_root),
    )
    require_sentinel_passed(sentinel)

    body = {
        "schema": ACTIVE8_COMPLETION_SCHEMA,
        "schema_version": ACTIVE8_COMPLETION_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "accepted_transitions": len(transitions),
        "action_family_histogram": dict(sorted(families.items())),
        "active8_exclusion_histogram": dict(sorted(exclusions.items())),
        "binding_sha256": str(binding["binding_sha256"]),
        "candidate_totals": totals,
        "capability_cell_histogram": dict(sorted(cells.items())),
        "census": census,
        "classification_affects_admission": False,
        "plan_sha256": str(validated["plan_sha256"]),
        "result_inventory": inventory,
        "result_inventory_sha256": canonical_sha256(inventory),
        "run_artifact_root": str(validated["run_artifact_root"]),
        "run_identity_sha256": str(validated["run_identity_sha256"]),
        "sentinel": sentinel,
        "task_inventory_sha256": str(validated["task_inventory_sha256"]),
    }
    completion = self_hashed(body, field="completion_sha256")
    if set(completion) != COMPLETION_FIELDS:
        raise ProcessV2Active8ReduceError("the Active8 completion field set disagrees")
    if publish:
        run_root = mounted_process_v2_artifact_path(
            str(validated["run_artifact_root"]),
            artifact_root=Path(artifact_root),
            field="plan.run_artifact_root",
        )
        run_root.mkdir(parents=True, exist_ok=True)
        target = run_root / COMPLETION_FILENAME
        staged = target.with_name(f"{target.name}.staged")
        try:
            staged.write_bytes(canonical_bytes(completion) + b"\n")
            os.replace(staged, target)
        finally:
            staged.unlink(missing_ok=True)
    return completion


def load_process_v2_active8_completion(
    run_artifact_root: str, *, artifact_root: Path
) -> dict[str, Any]:
    """Read one committed Active8 completion, or refuse the generation."""

    run_root = mounted_process_v2_artifact_path(
        run_artifact_root, artifact_root=Path(artifact_root), field="run_artifact_root"
    )
    path = run_root / COMPLETION_FILENAME
    if not path.is_file():
        raise ProcessV2Active8Incomplete(
            f"the Active8 generation is not committed; no {COMPLETION_FILENAME} at {run_root}"
        )
    completion = json.loads(path.read_bytes())
    if not isinstance(completion, dict) or set(completion) != COMPLETION_FIELDS:
        raise ProcessV2Active8ReduceError("the Active8 completion field set disagrees")
    verify_self_hash(completion, field="completion_sha256", label="the Active8 completion")
    if (
        completion["schema"] != ACTIVE8_COMPLETION_SCHEMA
        or completion["schema_version"] != ACTIVE8_COMPLETION_SCHEMA_VERSION
        or completion["status"] != PIPELINE_STATUS_NO_AUTHORITY
        or completion["classification_affects_admission"] is not False
    ):
        raise ProcessV2Active8ReduceError("the Active8 completion contract disagrees")
    return completion


__all__ = [
    "COMPLETION_FIELDS",
    "COMPLETION_FILENAME",
    "ProcessV2Active8Incomplete",
    "ProcessV2Active8ReduceError",
    "completed_process_v2_active8_task_ids",
    "load_process_v2_active8_completion",
    "reduce_process_v2_active8",
    "run_process_v2_active8_tasks",
    "task_output_path",
]
