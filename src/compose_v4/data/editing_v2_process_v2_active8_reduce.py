"""Deterministic reduction of the Active8 stage, gated by the release sentinel.

Three properties, and each of them is a refusal rather than a best effort:

* **Order and concurrency independent.**  The reduction reads the planned task
  inventory, not the directory, and folds every task in the plan's own order.
  Up to forty containers finishing in any order reduce to the same bytes.
* **Complete or nothing.**  A single missing or unreconcilable task result
  raises :class:`ProcessV2Active8Incomplete` and publishes NOTHING.  There is no
  partial completion, because a partial completion of an admission census is
  indistinguishable from a smaller corpus.
* **Sentinel-gated.**  Preparation freezes deterministic source-chunk partitions;
  each partition runs independently, and only their exact complete inventory can
  produce the final sentinel.  A mismatch leaves auditable task and partition
  evidence but no completion, which is exactly the state a rerun can recover from.

Every aggregate here is recomputed from the published ROWS.  The task receipts
are revalidated against their own rows and then used only as an integrity
check -- never as the source of a number.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import closing
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
    SELECTION_EXHAUSTIVE,
    ProcessV2Active8SentinelError,
    create_sentinel_pair_index,
    index_sentinel_transition,
    load_release_sentinel_partition_result,
    prepare_release_sentinel_plan_from_selection,
    reduce_release_sentinel_partitions,
    require_sentinel_passed,
    run_release_sentinel_partition,
    select_sentinel_pairs_from_index,
    sentinel_partition_identities,
    validate_release_sentinel_plan,
    write_release_sentinel_partition_result,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_COMPLETION_SCHEMA,
    ACTIVE8_COMPLETION_SCHEMA_VERSION,
    ACTIVE8_REDUCTION_PREPARED_FIELDS,
    ACTIVE8_REDUCTION_PREPARED_FILENAME,
    ACTIVE8_REDUCTION_PREPARED_SCHEMA,
    ACTIVE8_REDUCTION_PREPARED_SCHEMA_VERSION,
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
from compose_v4.data.immutable_artifact import (
    ImmutableArtifactError,
    write_bytes_if_absent,
)
from compose_v4.data.editing_v2_process_v2_active8_admission import (
    ProductionProcessV2SemanticExactCandidateChecker,
    build_process_v2_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    load_semantic_capability_cell_registry,
)

COMPLETION_FILENAME = "PROCESS_V2_ACTIVE8_COMPLETE.json"
PREPARATION_FILENAME = ACTIVE8_REDUCTION_PREPARED_FILENAME
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

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


def _create_reduction_index(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE source_rows (
            v1_task_identity_sha256 TEXT NOT NULL,
            entry_index INTEGER NOT NULL,
            PRIMARY KEY (v1_task_identity_sha256, entry_index)
        ) WITHOUT ROWID
        """
    )
    create_sentinel_pair_index(connection)


def _index_source_row(connection: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    key = (str(row["v1_task_identity_sha256"]), int(row["entry_index"]))
    try:
        connection.execute("INSERT INTO source_rows VALUES (?, ?)", key)
    except sqlite3.IntegrityError as error:
        raise ProcessV2Active8ReduceError(
            f"two Active8 tasks decide the same source entry: {key}"
        ) from error


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


def _validated_task_result(
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    identity = str(task["task_identity_sha256"])
    output = task_output_path(plan, task, artifact_root=artifact_root)
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
        or summary["plan_sha256"] != str(plan["plan_sha256"])
        or summary["run_identity_sha256"] != str(plan["run_identity_sha256"])
        or summary["binding_sha256"] != str(plan["binding_sha256"])
    ):
        raise ProcessV2Active8Incomplete(
            f"the Active8 result of task {identity} belongs to another run"
        )
    return output, receipt, summary


def _derive_reduction_inputs(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    max_sentinel_pairs_per_partition: int,
) -> dict[str, Any]:
    """Two-pass, disk-indexed derivation of aggregates and sentinel selection."""

    census = dict.fromkeys(ACTIVE8_CENSUS_FIELDS, 0)
    totals = dict.fromkeys(CANDIDATE_TOTAL_FIELDS, 0)
    families: dict[str, int] = {}
    cells: dict[str, int] = {}
    exclusions: dict[str, int] = {}
    inventory: list[list[str]] = []
    content_inventory: list[list[str]] = []
    accepted_transitions = 0
    selection: dict[str, Any]
    selected_transitions: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="compose-active8-reduce-") as temporary:
        database_path = Path(temporary) / "reduction.sqlite3"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute("PRAGMA temp_store = FILE")
            _create_reduction_index(connection)

            # Pass one derives every aggregate and stores only unique pair/cell
            # membership.  The database enforces the global source-row refusal.
            for task in plan["tasks"]:
                identity = str(task["task_identity_sha256"])
                output, receipt, summary = _validated_task_result(
                    plan, task, artifact_root=artifact_root
                )
                inventory.append([identity, str(receipt["receipt_sha256"])])
                content_inventory.append(
                    [
                        identity,
                        str(receipt["receipt_sha256"]),
                        str(summary["summary_sha256"]),
                    ]
                )
                for row in read_task_rows(output):
                    _index_source_row(connection, row)
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
                for transition in read_task_transitions(output):
                    accepted_transitions += 1
                    cell = transition["capability_cell_id"]
                    if cell is not None:
                        cells[str(cell)] = cells.get(str(cell), 0) + 1
                    index_sentinel_transition(connection, transition)
                connection.commit()

            selection = select_sentinel_pairs_from_index(
                connection,
                required_cell_ids=list(plan["binding"]["required_cell_ids"]),
            )
            selected_pairs = set(selection["selected"])

            # Pass two rereads accepted transitions but retains occurrence
            # payloads only for the bounded selected-pair union.
            second_pass_transitions = 0
            second_content_inventory: list[list[str]] = []
            for task in plan["tasks"]:
                identity = str(task["task_identity_sha256"])
                output = task_output_path(plan, task, artifact_root=artifact_root)
                task_transitions = read_task_transitions(output)
                for transition in task_transitions:
                    second_pass_transitions += 1
                    evidence = transition["candidate_evidence"]
                    pair = (
                        str(evidence["source_state_sha256"]),
                        str(evidence["action_sha256"]),
                    )
                    if pair in selected_pairs:
                        selected_transitions.append(transition)
                del task_transitions
                _, receipt, summary = _validated_task_result(
                    plan, task, artifact_root=artifact_root
                )
                second_content_inventory.append(
                    [
                        identity,
                        str(receipt["receipt_sha256"]),
                        str(summary["summary_sha256"]),
                    ]
                )
            if (
                second_pass_transitions != accepted_transitions
                or second_content_inventory != content_inventory
            ):
                raise ProcessV2Active8Incomplete(
                    "the Active8 map results changed during bounded reduction"
                )

    if census["source_entries"] != int(plan["expected_entry_count"]):
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

    result_inventory_sha256 = canonical_sha256(inventory)
    sentinel_plan = prepare_release_sentinel_plan_from_selection(
        plan,
        selection,
        selected_transitions,
        result_inventory_sha256=result_inventory_sha256,
        max_pairs_per_partition=max_sentinel_pairs_per_partition,
    )
    return {
        "census": census,
        "candidate_totals": totals,
        "action_family_histogram": dict(sorted(families.items())),
        "capability_cell_histogram": dict(sorted(cells.items())),
        "active8_exclusion_histogram": dict(sorted(exclusions.items())),
        "result_inventory": inventory,
        "result_inventory_sha256": result_inventory_sha256,
        "task_result_content_inventory": content_inventory,
        "task_result_content_inventory_sha256": canonical_sha256(content_inventory),
        "accepted_transitions": accepted_transitions,
        "sentinel_plan": sentinel_plan,
    }


def prepare_process_v2_active8_reduction(
    plan: Mapping[str, Any],
    *,
    runtime: Any,
    artifact_root: Path,
    repo_root: Path,
    max_sentinel_pairs_per_partition: int,
    publish: bool = True,
) -> dict[str, Any]:
    """Freeze aggregates and the exact parallel sentinel worker inventory."""

    validated = validate_process_v2_active8_plan(plan, repo_root=Path(repo_root))
    binding = validated["binding"]
    if model_runtime_descriptor(runtime) != dict(binding["model_runtime"]):
        raise ProcessV2Active8ReduceError(
            "the supplied model runtime is not the one the Active8 plan binds"
        )
    collected = _derive_reduction_inputs(
        validated,
        artifact_root=Path(artifact_root),
        max_sentinel_pairs_per_partition=max_sentinel_pairs_per_partition,
    )
    body = {
        "schema": ACTIVE8_REDUCTION_PREPARED_SCHEMA,
        "schema_version": ACTIVE8_REDUCTION_PREPARED_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "accepted_transitions": int(collected["accepted_transitions"]),
        "action_family_histogram": collected["action_family_histogram"],
        "active8_exclusion_histogram": collected["active8_exclusion_histogram"],
        "binding_sha256": str(binding["binding_sha256"]),
        "candidate_totals": collected["candidate_totals"],
        "capability_cell_histogram": collected["capability_cell_histogram"],
        "census": collected["census"],
        "classification_affects_admission": False,
        "plan_sha256": str(validated["plan_sha256"]),
        "result_inventory": collected["result_inventory"],
        "result_inventory_sha256": str(collected["result_inventory_sha256"]),
        "task_result_content_inventory": collected["task_result_content_inventory"],
        "task_result_content_inventory_sha256": str(
            collected["task_result_content_inventory_sha256"]
        ),
        "run_artifact_root": str(validated["run_artifact_root"]),
        "run_identity_sha256": str(validated["run_identity_sha256"]),
        "sentinel_plan": collected["sentinel_plan"],
        "task_inventory_sha256": str(validated["task_inventory_sha256"]),
    }
    prepared = self_hashed(body, field="preparation_sha256")
    validate_process_v2_active8_reduction_preparation(prepared, active8_plan=validated)
    if publish:
        run_root = mounted_process_v2_artifact_path(
            str(validated["run_artifact_root"]),
            artifact_root=Path(artifact_root),
            field="plan.run_artifact_root",
        )
        try:
            write_bytes_if_absent(
                run_root / PREPARATION_FILENAME,
                canonical_bytes(prepared) + b"\n",
            )
        except ImmutableArtifactError as error:
            raise ProcessV2Active8ReduceError(
                "immutable Active8 reduction-preparation publication failed"
            ) from error
    return prepared


def validate_process_v2_active8_reduction_preparation(
    value: Mapping[str, Any], *, active8_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the frozen handoff between map and sentinel workers."""

    if not isinstance(value, Mapping) or set(value) != set(ACTIVE8_REDUCTION_PREPARED_FIELDS):
        raise ProcessV2Active8ReduceError("the Active8 reduction preparation field set disagrees")
    prepared = dict(value)
    verify_self_hash(
        prepared,
        field="preparation_sha256",
        label="the Active8 reduction preparation",
    )
    if (
        prepared["schema"] != ACTIVE8_REDUCTION_PREPARED_SCHEMA
        or prepared["schema_version"] != ACTIVE8_REDUCTION_PREPARED_SCHEMA_VERSION
        or prepared["status"] != PIPELINE_STATUS_NO_AUTHORITY
        or prepared["classification_affects_admission"] is not False
    ):
        raise ProcessV2Active8ReduceError("the Active8 reduction preparation contract disagrees")
    for field, false_value in authority_false_block().items():
        if prepared[field] is not false_value:
            raise ProcessV2Active8ReduceError(
                f"the Active8 reduction preparation grants authority through {field}"
            )
    expected_context = {
        "binding_sha256": str(active8_plan["binding_sha256"]),
        "plan_sha256": str(active8_plan["plan_sha256"]),
        "run_artifact_root": str(active8_plan["run_artifact_root"]),
        "run_identity_sha256": str(active8_plan["run_identity_sha256"]),
        "task_inventory_sha256": str(active8_plan["task_inventory_sha256"]),
    }
    for field, expected in expected_context.items():
        if prepared[field] != expected:
            raise ProcessV2Active8ReduceError(
                f"the Active8 reduction preparation {field} addresses another run"
            )
    inventory = prepared["result_inventory"]
    if (
        not isinstance(inventory, list)
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or any(
                not isinstance(digest, str) or _SHA256.fullmatch(digest) is None for digest in item
            )
            for item in inventory
        )
        or [item[0] for item in inventory]
        != [str(task["task_identity_sha256"]) for task in active8_plan["tasks"]]
        or canonical_sha256(inventory) != prepared["result_inventory_sha256"]
    ):
        raise ProcessV2Active8ReduceError(
            "the Active8 reduction preparation result inventory disagrees"
        )
    content_inventory = prepared["task_result_content_inventory"]
    if (
        not isinstance(content_inventory, list)
        or any(
            not isinstance(item, list)
            or len(item) != 3
            or any(
                not isinstance(digest, str) or _SHA256.fullmatch(digest) is None for digest in item
            )
            for item in content_inventory
        )
        or [item[:2] for item in content_inventory] != inventory
        or canonical_sha256(content_inventory) != prepared["task_result_content_inventory_sha256"]
    ):
        raise ProcessV2Active8ReduceError(
            "the Active8 reduction preparation content inventory disagrees"
        )
    census = prepared["census"]
    if (
        not isinstance(census, Mapping)
        or set(census) != set(ACTIVE8_CENSUS_FIELDS)
        or any(type(value) is not int or value < 0 for value in census.values())
        or census["source_entries"] != int(active8_plan["expected_entry_count"])
        or census["source_entries"]
        != census["upstream_rejected_entries"]
        + census["active8_accepted_entries"]
        + census["active8_excluded_entries"]
    ):
        raise ProcessV2Active8ReduceError(
            "the Active8 reduction preparation census does not reconcile"
        )
    if type(prepared["accepted_transitions"]) is not int or prepared["accepted_transitions"] < 0:
        raise ProcessV2Active8ReduceError(
            "the Active8 reduction preparation transition count is invalid"
        )
    for field in (
        "action_family_histogram",
        "active8_exclusion_histogram",
        "capability_cell_histogram",
        "candidate_totals",
    ):
        aggregate = prepared[field]
        if (
            not isinstance(aggregate, Mapping)
            or any(not isinstance(key, str) for key in aggregate)
            or any(type(count) is not int or count < 0 for count in aggregate.values())
        ):
            raise ProcessV2Active8ReduceError(
                f"the Active8 reduction preparation {field} is malformed"
            )
    if set(prepared["candidate_totals"]) != set(CANDIDATE_TOTAL_FIELDS):
        raise ProcessV2Active8ReduceError(
            "the Active8 reduction preparation candidate totals disagree"
        )
    try:
        sentinel_plan = validate_release_sentinel_plan(
            prepared["sentinel_plan"], active8_plan=active8_plan
        )
    except ProcessV2Active8SentinelError as error:
        raise ProcessV2Active8ReduceError(str(error)) from error
    if sentinel_plan["result_inventory_sha256"] != prepared["result_inventory_sha256"]:
        raise ProcessV2Active8ReduceError(
            "the Active8 reduction preparation and sentinel plan disagree"
        )
    accepted_transitions = int(prepared["accepted_transitions"])
    unique_pairs = int(sentinel_plan["unique_accepted_pairs"])
    occurrence_count = sum(
        len(pair["occurrences"])
        for partition in sentinel_plan["partitions"]
        for pair in partition["pairs"]
    )
    if (
        unique_pairs > accepted_transitions
        or (accepted_transitions == 0) != (unique_pairs == 0)
        or occurrence_count > accepted_transitions
    ):
        raise ProcessV2Active8ReduceError(
            "the Active8 reduction preparation and sentinel census disagree"
        )
    if (
        sentinel_plan["selection_mode"] == SELECTION_EXHAUSTIVE
        and occurrence_count != accepted_transitions
    ):
        raise ProcessV2Active8ReduceError(
            "the exhaustive sentinel does not cover every accepted transition"
        )
    return prepared


def load_process_v2_active8_reduction_preparation(
    run_artifact_root: str,
    *,
    active8_plan: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    """Reopen the canonical prepared handoff for restart-safe sentinel work."""

    run_root = mounted_process_v2_artifact_path(
        run_artifact_root,
        artifact_root=Path(artifact_root),
        field="run_artifact_root",
    )
    path = run_root / PREPARATION_FILENAME
    try:
        raw = path.read_bytes()
        prepared = json.loads(raw)
    except OSError as error:
        raise ProcessV2Active8Incomplete(
            f"the Active8 reduction preparation is absent: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ProcessV2Active8ReduceError(
            "the Active8 reduction preparation is not valid JSON"
        ) from error
    if canonical_bytes(prepared) + b"\n" != raw:
        raise ProcessV2Active8ReduceError("the Active8 reduction preparation is not canonical JSON")
    return validate_process_v2_active8_reduction_preparation(prepared, active8_plan=active8_plan)


def run_process_v2_active8_sentinel_partition(
    plan: Mapping[str, Any],
    prepared: Mapping[str, Any],
    partition_identity_sha256: str,
    *,
    runtime: Any,
    artifact_root: Path,
    repo_root: Path,
    publish: bool = False,
) -> dict[str, Any]:
    """Modal-facing worker API for one independent sentinel partition."""

    validated = validate_process_v2_active8_plan(plan, repo_root=Path(repo_root))
    preparation = validate_process_v2_active8_reduction_preparation(
        prepared, active8_plan=validated
    )
    if model_runtime_descriptor(runtime) != dict(validated["binding"]["model_runtime"]):
        raise ProcessV2Active8ReduceError(
            "the supplied sentinel runtime is not the one the Active8 plan binds"
        )
    checker = ProductionProcessV2SemanticExactCandidateChecker(
        runtime.model, policy=build_process_v2_semantic_active8_admission_policy()
    )
    result = run_release_sentinel_partition(
        validated,
        preparation["sentinel_plan"],
        partition_identity_sha256,
        checker=checker,
        namespace=load_semantic_capability_cell_registry().namespace,
        artifact_root=Path(artifact_root),
        repo_root=Path(repo_root),
    )
    if publish:
        write_release_sentinel_partition_result(
            result,
            active8_plan=validated,
            sentinel_plan=preparation["sentinel_plan"],
            artifact_root=Path(artifact_root),
        )
    return result


def _require_current_result_inventory(
    plan: Mapping[str, Any],
    prepared: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> None:
    """Re-run the same bounded derivation from current immutable map bytes."""

    current = _derive_reduction_inputs(
        plan,
        artifact_root=Path(artifact_root),
        max_sentinel_pairs_per_partition=int(prepared["sentinel_plan"]["max_pairs_per_partition"]),
    )
    comparisons = {
        "accepted_transitions": current["accepted_transitions"],
        "action_family_histogram": current["action_family_histogram"],
        "active8_exclusion_histogram": current["active8_exclusion_histogram"],
        "candidate_totals": current["candidate_totals"],
        "capability_cell_histogram": current["capability_cell_histogram"],
        "census": current["census"],
        "result_inventory": current["result_inventory"],
        "result_inventory_sha256": current["result_inventory_sha256"],
        "task_result_content_inventory": current["task_result_content_inventory"],
        "task_result_content_inventory_sha256": current["task_result_content_inventory_sha256"],
        "sentinel_plan": current["sentinel_plan"],
    }
    for field, observed in comparisons.items():
        if canonical_bytes(prepared[field]) != canonical_bytes(observed):
            raise ProcessV2Active8Incomplete(
                f"the Active8 map-result {field} changed after sentinel preparation"
            )


def finalize_process_v2_active8_reduction(
    plan: Mapping[str, Any],
    prepared: Mapping[str, Any],
    partition_results: list[Mapping[str, Any]],
    *,
    artifact_root: Path,
    repo_root: Path,
    publish: bool = True,
) -> dict[str, Any]:
    """Require the exact map and sentinel inventories, then publish completion."""

    validated = validate_process_v2_active8_plan(plan, repo_root=Path(repo_root))
    preparation = validate_process_v2_active8_reduction_preparation(
        prepared, active8_plan=validated
    )
    if publish:
        reopened = load_process_v2_active8_reduction_preparation(
            str(validated["run_artifact_root"]),
            active8_plan=validated,
            artifact_root=Path(artifact_root),
        )
        if canonical_bytes(reopened) != canonical_bytes(preparation):
            raise ProcessV2Active8Incomplete(
                "the published Active8 reduction preparation differs from finalization"
            )
    _require_current_result_inventory(validated, preparation, artifact_root=Path(artifact_root))
    effective_results = list(partition_results)
    if publish:
        partition_ids = sentinel_partition_identities(preparation["sentinel_plan"])
        published_results = [
            load_release_sentinel_partition_result(
                identity,
                active8_plan=validated,
                sentinel_plan=preparation["sentinel_plan"],
                artifact_root=Path(artifact_root),
            )
            for identity in partition_ids
        ]
        supplied_by_identity = {
            str(result.get("partition_identity_sha256")): result for result in effective_results
        }
        if (
            len(supplied_by_identity) != len(effective_results)
            or set(supplied_by_identity) != set(partition_ids)
            or any(
                canonical_bytes(supplied_by_identity[identity]) != canonical_bytes(published)
                for identity, published in zip(partition_ids, published_results, strict=True)
            )
        ):
            raise ProcessV2Active8Incomplete(
                "the supplied sentinel results differ from the exact published inventory"
            )
        effective_results = published_results
    try:
        sentinel = reduce_release_sentinel_partitions(
            validated, preparation["sentinel_plan"], effective_results
        )
        require_sentinel_passed(
            sentinel,
            binding_sha256=str(preparation["binding_sha256"]),
            plan_sha256=str(preparation["plan_sha256"]),
            run_identity_sha256=str(preparation["run_identity_sha256"]),
            task_inventory_sha256=str(preparation["task_inventory_sha256"]),
            result_inventory_sha256=str(preparation["result_inventory_sha256"]),
        )
    except ProcessV2Active8SentinelError as error:
        raise ProcessV2Active8SentinelError(str(error)) from error

    body = {
        "schema": ACTIVE8_COMPLETION_SCHEMA,
        "schema_version": ACTIVE8_COMPLETION_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "accepted_transitions": int(preparation["accepted_transitions"]),
        "action_family_histogram": dict(preparation["action_family_histogram"]),
        "active8_exclusion_histogram": dict(preparation["active8_exclusion_histogram"]),
        "binding_sha256": str(preparation["binding_sha256"]),
        "candidate_totals": dict(preparation["candidate_totals"]),
        "capability_cell_histogram": dict(preparation["capability_cell_histogram"]),
        "census": dict(preparation["census"]),
        "classification_affects_admission": False,
        "plan_sha256": str(preparation["plan_sha256"]),
        "result_inventory": list(preparation["result_inventory"]),
        "result_inventory_sha256": str(preparation["result_inventory_sha256"]),
        "run_artifact_root": str(preparation["run_artifact_root"]),
        "run_identity_sha256": str(preparation["run_identity_sha256"]),
        "sentinel": sentinel,
        "task_inventory_sha256": str(preparation["task_inventory_sha256"]),
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
        try:
            write_bytes_if_absent(
                run_root / COMPLETION_FILENAME,
                canonical_bytes(completion) + b"\n",
            )
        except ImmutableArtifactError as error:
            raise ProcessV2Active8ReduceError(
                "immutable Active8 completion publication failed"
            ) from error
    return completion


def reduce_process_v2_active8(
    plan: Mapping[str, Any],
    *,
    runtime: Any,
    artifact_root: Path,
    repo_root: Path,
    publish: bool = True,
) -> dict[str, Any]:
    """Local serial wrapper over the production prepare/map/finalize interfaces."""

    prepared = prepare_process_v2_active8_reduction(
        plan,
        runtime=runtime,
        artifact_root=Path(artifact_root),
        repo_root=Path(repo_root),
        max_sentinel_pairs_per_partition=256,
        publish=publish,
    )
    results = [
        run_process_v2_active8_sentinel_partition(
            plan,
            prepared,
            identity,
            runtime=runtime,
            artifact_root=Path(artifact_root),
            repo_root=Path(repo_root),
            publish=publish,
        )
        for identity in sentinel_partition_identities(prepared["sentinel_plan"])
    ]
    return finalize_process_v2_active8_reduction(
        plan,
        prepared,
        results,
        artifact_root=Path(artifact_root),
        repo_root=Path(repo_root),
        publish=publish,
    )


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
    return validate_process_v2_active8_completion(completion)


def validate_process_v2_active8_completion(
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate one committed completion and its blocking sentinel."""

    if not isinstance(completion, Mapping) or set(completion) != COMPLETION_FIELDS:
        raise ProcessV2Active8ReduceError("the Active8 completion field set disagrees")
    verify_self_hash(completion, field="completion_sha256", label="the Active8 completion")
    if (
        completion["schema"] != ACTIVE8_COMPLETION_SCHEMA
        or completion["schema_version"] != ACTIVE8_COMPLETION_SCHEMA_VERSION
        or completion["status"] != PIPELINE_STATUS_NO_AUTHORITY
        or completion["classification_affects_admission"] is not False
    ):
        raise ProcessV2Active8ReduceError("the Active8 completion contract disagrees")
    for field, value in authority_false_block().items():
        if completion.get(field) is not value:
            raise ProcessV2Active8ReduceError(
                f"the Active8 completion grants forbidden authority through {field}"
            )
    inventory = completion["result_inventory"]
    if (
        not isinstance(inventory, list)
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or not all(
                isinstance(value, str) and _SHA256.fullmatch(value) is not None for value in item
            )
            for item in inventory
        )
        or len({item[0] for item in inventory}) != len(inventory)
        or canonical_sha256(inventory) != completion["result_inventory_sha256"]
    ):
        raise ProcessV2Active8ReduceError(
            "the Active8 completion result inventory does not reconcile"
        )
    identities = [item[0] for item in inventory]
    if canonical_sha256(identities) != completion["task_inventory_sha256"]:
        raise ProcessV2Active8ReduceError(
            "the Active8 completion task inventory does not reconcile"
        )
    for field in (
        "binding_sha256",
        "plan_sha256",
        "run_identity_sha256",
        "task_inventory_sha256",
        "result_inventory_sha256",
    ):
        value = completion[field]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ProcessV2Active8ReduceError(f"the Active8 completion {field} is not a SHA-256")
    census = completion["census"]
    if not isinstance(census, Mapping) or set(census) != set(ACTIVE8_CENSUS_FIELDS):
        raise ProcessV2Active8ReduceError("the Active8 completion census field set disagrees")
    if any(type(census[field]) is not int or census[field] < 0 for field in census):
        raise ProcessV2Active8ReduceError(
            "the Active8 completion census must contain nonnegative integers"
        )
    if census["source_entries"] != (
        census["upstream_rejected_entries"]
        + census["active8_accepted_entries"]
        + census["active8_excluded_entries"]
    ):
        raise ProcessV2Active8ReduceError("the Active8 completion census does not close")
    if (
        type(completion["accepted_transitions"]) is not int
        or completion["accepted_transitions"] < 0
    ):
        raise ProcessV2Active8ReduceError(
            "the Active8 completion accepted transition count is invalid"
        )
    totals = completion["candidate_totals"]
    if not isinstance(totals, Mapping) or set(totals) != set(CANDIDATE_TOTAL_FIELDS):
        raise ProcessV2Active8ReduceError(
            "the Active8 completion candidate total field set disagrees"
        )
    aggregate_fields = (
        "action_family_histogram",
        "active8_exclusion_histogram",
        "capability_cell_histogram",
        "candidate_totals",
    )
    for field in aggregate_fields:
        aggregate = completion[field]
        if (
            not isinstance(aggregate, Mapping)
            or any(not isinstance(key, str) for key in aggregate)
            or any(type(value) is not int or value < 0 for value in aggregate.values())
        ):
            raise ProcessV2Active8ReduceError(f"the Active8 completion {field} is malformed")
    try:
        require_sentinel_passed(
            completion["sentinel"],
            binding_sha256=str(completion["binding_sha256"]),
            plan_sha256=str(completion["plan_sha256"]),
            run_identity_sha256=str(completion["run_identity_sha256"]),
            task_inventory_sha256=str(completion["task_inventory_sha256"]),
            result_inventory_sha256=str(completion["result_inventory_sha256"]),
        )
    except ProcessV2Active8SentinelError as error:
        raise ProcessV2Active8ReduceError(str(error)) from error
    return dict(completion)


__all__ = [
    "COMPLETION_FIELDS",
    "COMPLETION_FILENAME",
    "PREPARATION_FILENAME",
    "ProcessV2Active8Incomplete",
    "ProcessV2Active8ReduceError",
    "completed_process_v2_active8_task_ids",
    "finalize_process_v2_active8_reduction",
    "load_process_v2_active8_completion",
    "load_process_v2_active8_reduction_preparation",
    "prepare_process_v2_active8_reduction",
    "reduce_process_v2_active8",
    "run_process_v2_active8_sentinel_partition",
    "run_process_v2_active8_tasks",
    "task_output_path",
    "validate_process_v2_active8_completion",
    "validate_process_v2_active8_reduction_preparation",
]
