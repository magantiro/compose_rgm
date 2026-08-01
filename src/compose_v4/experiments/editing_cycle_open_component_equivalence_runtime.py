"""Frozen planning and artifact identities for cycle-open equivalence.

The planner selects only validation shards whose immutable Active8 metadata
declares at least one accepted ``cycle_attach`` teacher row. It still
reconciles the complete validation shard inventory and total family-row count,
so skipping zero-support shards is a verified optimization rather than a
scientific support reduction.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from compose_v4.data.active8_trace_inventory import Active8TraceAdmission
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    NON_AUTHORIZING_STATUS,
    CycleOpenComponentEquivalenceError,
    file_sha256,
    semantic_sha256,
)


PLAN_SCHEMA = "compose.editing.cycle_open_component_equivalence_plan"
PLAN_SCHEMA_VERSION = 1


def pretty_json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CycleOpenComponentEquivalenceError(
            "runtime artifact is not finite deterministic JSON"
        ) from error


def load_json_artifact(path: str | Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CycleOpenComponentEquivalenceError(f"cannot load {description}: {path}") from error
    if not isinstance(payload, dict):
        raise CycleOpenComponentEquivalenceError(f"{description} must be a JSON object")
    return payload


def validate_parent_admission(
    contract: Mapping[str, Any], admission: Active8TraceAdmission
) -> None:
    parent = contract["parent_active8_identity"]
    observed = {
        "inventory_manifest_path": str(admission.manifest_path),
        "inventory_manifest_file_sha256": admission.manifest_file_sha256,
        "inventory_sha256": admission.inventory_sha256,
        "effective_source_corpus_cache_sha256": (admission.effective_source_corpus_cache_sha256),
        "support_contract_sha256": admission.support_contract_sha256,
        "unified_packed_manifest_sha256": admission.unified_packed_manifest_sha256,
    }
    if observed != parent:
        raise CycleOpenComponentEquivalenceError(
            f"Active8 admission identity disagrees with the contract: {observed}"
        )


def active8_partition_census(
    admission: Active8TraceAdmission,
) -> dict[str, dict[str, int]]:
    census: dict[str, Counter[str]] = {}
    for metadata in admission.shard_metadata_by_digest.values():
        partition = metadata.get("partition")
        counts = metadata.get("counts")
        family_rows = metadata.get("accepted_nonterminal_rows_by_family")
        if (
            not isinstance(partition, str)
            or not isinstance(counts, Mapping)
            or not isinstance(family_rows, Mapping)
        ):
            raise CycleOpenComponentEquivalenceError(
                "Active8 shard metadata cannot establish its partition census"
            )
        row = census.setdefault(partition, Counter())
        row["source_shards"] += 1
        for field in (
            "traces",
            "accepted_traces",
            "excluded_traces",
            "accepted_nonterminal_rows",
            "accepted_terminal_rows",
        ):
            count = counts.get(field)
            if type(count) is not int or count < 0:
                raise CycleOpenComponentEquivalenceError(
                    f"Active8 shard partition count is malformed: {field}"
                )
            row[field] += count
        cycle_attach_rows = family_rows.get("cycle_attach")
        if type(cycle_attach_rows) is not int or cycle_attach_rows < 0:
            raise CycleOpenComponentEquivalenceError(
                "Active8 shard lacks a cycle_attach teacher-row census"
            )
        row["cycle_attach_teacher_rows"] += cycle_attach_rows
    return {partition: dict(sorted(counts.items())) for partition, counts in sorted(census.items())}


def implementation_identity(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Bind every local source that can affect planning, decoding, or comparison."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    sources = (
        "modal_apps/audit_editing_cycle_open_component_equivalence.py",
        "scripts/audit_editing_cycle_open_component_equivalence.py",
        "src/compose_v4/experiments/editing_cycle_open_component_equivalence_runtime.py",
        "src/compose_v4/experiments/editing_cycle_open_component_equivalence.py",
        "src/compose_v4/experiments/aromatic_cycle_open_semantics.py",
        "src/compose_v4/chem/aromaticity.py",
        "src/compose_v4/chem/molecular_graph.py",
        "src/compose_v4/chem/persistent_state_identity.py",
        "src/compose_v4/chem/state.py",
        "src/compose_v4/data/active8_trace_inventory.py",
        "src/compose_v4/data/packed_charge_policy_audit.py",
        "src/compose_v4/data/packed_trace_store.py",
        "src/compose_v4/rewrite/action_codec.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/operators.py",
    )
    hashes = {relative: file_sha256(root / relative) for relative in sources}
    return {
        "sources": hashes,
        "implementation_sha256": semantic_sha256(hashes),
    }


def _task_body(
    *,
    task_index: int,
    declared: object,
    digest: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    counts = metadata.get("counts")
    family_rows = metadata.get("accepted_nonterminal_rows_by_family")
    if not isinstance(counts, Mapping) or not isinstance(family_rows, Mapping):
        raise CycleOpenComponentEquivalenceError("Active8 shard metadata lacks exact counts")
    required_counts = (
        "traces",
        "accepted_traces",
        "accepted_nonterminal_rows",
        "accepted_terminal_rows",
    )
    if any(type(counts.get(field)) is not int for field in required_counts):
        raise CycleOpenComponentEquivalenceError("Active8 shard metadata has malformed counts")
    cycle_attach_rows = family_rows.get("cycle_attach")
    if type(cycle_attach_rows) is not int or cycle_attach_rows <= 0:
        raise CycleOpenComponentEquivalenceError(
            "selected equivalence shard lacks positive cycle_attach support"
        )
    body = {
        "task_index": task_index,
        "manifest_layer": str(declared.manifest_layer),
        "layer": str(declared.envelope_layer),
        "partition": str(declared.partition),
        "relative_path": str(declared.relative_path),
        "packed_shard_name": str(declared.path.name),
        "packed_shard_content_sha256": digest,
        "packed_manifest_sha256": metadata["packed_manifest_sha256"],
        "packed_provenance_overlay_sha256": metadata["packed_provenance_overlay_sha256"],
        "expected_counts": {field: int(counts[field]) for field in required_counts},
        "expected_cycle_attach_teacher_rows": cycle_attach_rows,
    }
    body["task_sha256"] = semantic_sha256(body)
    body["receipt_filename"] = f"task_{task_index:05d}.{body['task_sha256']}.json"
    return body


def build_plan(
    contract: Mapping[str, Any],
    admission: Active8TraceAdmission,
    declared_shards: Sequence[object],
    *,
    unified_manifest_file_sha256: str,
    inputs: Mapping[str, str],
    code_revision: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic positive-support validation shard plan."""

    validate_parent_admission(contract, admission)
    census = active8_partition_census(admission)
    if set(census) != {"validation"} or census["validation"]["source_shards"] <= 0:
        raise CycleOpenComponentEquivalenceError(
            f"pinned admission is not exactly validation-only: {census}"
        )
    if census["validation"]["cycle_attach_teacher_rows"] <= 0:
        raise CycleOpenComponentEquivalenceError(
            "pinned validation admission has no cycle_attach teacher support"
        )
    parent = contract["parent_active8_identity"]
    if unified_manifest_file_sha256 != parent["unified_packed_manifest_sha256"]:
        raise CycleOpenComponentEquivalenceError(
            "unified packed manifest bytes disagree with the contract"
        )
    if code_revision.get("tree_dirty") is not False or not code_revision.get("commit"):
        raise CycleOpenComponentEquivalenceError("equivalence plan requires a clean committed tree")
    implementation_sha256 = implementation.get("implementation_sha256")
    if not isinstance(implementation_sha256, str) or len(implementation_sha256) != 64:
        raise CycleOpenComponentEquivalenceError("implementation identity lacks a full SHA-256")

    validation_shards = tuple(
        sorted(
            (shard for shard in declared_shards if shard.partition == "validation"),
            key=lambda shard: (str(shard.envelope_layer), str(shard.relative_path)),
        )
    )
    if not validation_shards:
        raise CycleOpenComponentEquivalenceError("plan resolved no validation shards")
    admission.assert_partition_shards(
        "validation",
        ((shard.envelope_layer, shard.partition, shard.path.name) for shard in validation_shards),
    )

    selected: list[tuple[object, str, Mapping[str, Any]]] = []
    skipped: list[dict[str, Any]] = []
    for shard in validation_shards:
        digest = admission.expected_source_digest(
            packed_shard_name=shard.path.name,
            layer=shard.envelope_layer,
            partition=shard.partition,
        )
        metadata = admission.shard_metadata_by_digest[digest]
        family_rows = metadata["accepted_nonterminal_rows_by_family"]
        cycle_attach_rows = family_rows["cycle_attach"]
        if cycle_attach_rows > 0:
            selected.append((shard, digest, metadata))
        else:
            skipped.append(
                {
                    "layer": str(shard.envelope_layer),
                    "partition": "validation",
                    "packed_shard_name": str(shard.path.name),
                    "packed_shard_content_sha256": digest,
                    "cycle_attach_teacher_rows": 0,
                }
            )
    tasks = [
        _task_body(
            task_index=index,
            declared=shard,
            digest=digest,
            metadata=metadata,
        )
        for index, (shard, digest, metadata) in enumerate(selected)
    ]
    if (
        not tasks
        or sum(task["expected_cycle_attach_teacher_rows"] for task in tasks)
        != census["validation"]["cycle_attach_teacher_rows"]
    ):
        raise CycleOpenComponentEquivalenceError(
            "selected positive-support shards do not reconcile the validation family census"
        )
    if len(tasks) + len(skipped) != census["validation"]["source_shards"]:
        raise CycleOpenComponentEquivalenceError(
            "selected plus skipped shards do not reconcile the validation inventory"
        )

    body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "contract_sha256": contract["contract_sha256"],
        "parent_active8_identity": dict(parent),
        "inputs": dict(sorted(inputs.items())),
        "code_revision": dict(code_revision),
        "implementation": dict(implementation),
        "partitions": ["validation"],
        "excluded_partitions": ["train", "controller_validation", "test"],
        "active8_partition_census": census,
        "selection_policy": "positive_cycle_attach_support_only_with_zero_support_ledger",
        "task_count": len(tasks),
        "tasks": tasks,
        "zero_cycle_attach_support_shards": skipped,
    }
    return {**body, "plan_sha256": semantic_sha256(body)}


def validate_plan(value: Mapping[str, Any], *, expected_contract_sha256: str) -> dict[str, Any]:
    plan = dict(value)
    plan_hash = plan.pop("plan_sha256", None)
    if plan_hash != semantic_sha256(plan):
        raise CycleOpenComponentEquivalenceError("equivalence plan self-hash disagrees")
    plan["plan_sha256"] = plan_hash
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != NON_AUTHORIZING_STATUS
        or plan.get("training_authorized") is not False
        or plan.get("contract_sha256") != expected_contract_sha256
        or plan.get("partitions") != ["validation"]
        or plan.get("excluded_partitions") != ["train", "controller_validation", "test"]
        or plan.get("selection_policy")
        != "positive_cycle_attach_support_only_with_zero_support_ledger"
    ):
        raise CycleOpenComponentEquivalenceError(
            "equivalence plan schema, scope, identity, or authority disagrees"
        )
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or plan.get("task_count") != len(tasks) or not tasks:
        raise CycleOpenComponentEquivalenceError("equivalence plan task census is invalid")
    for index, task in enumerate(tasks):
        if (
            not isinstance(task, Mapping)
            or task.get("task_index") != index
            or task.get("partition") != "validation"
            or task.get("expected_cycle_attach_teacher_rows", 0) <= 0
        ):
            raise CycleOpenComponentEquivalenceError("equivalence plan contains an invalid task")
        task_body = {
            key: item
            for key, item in task.items()
            if key not in {"task_sha256", "receipt_filename"}
        }
        if (
            semantic_sha256(task_body) != task.get("task_sha256")
            or task.get("receipt_filename") != f"task_{index:05d}.{task['task_sha256']}.json"
        ):
            raise CycleOpenComponentEquivalenceError(
                "equivalence task hash or receipt filename disagrees"
            )
    skipped = plan.get("zero_cycle_attach_support_shards")
    if (
        not isinstance(skipped, list)
        or any(
            row.get("partition") != "validation" or row.get("cycle_attach_teacher_rows") != 0
            for row in skipped
            if isinstance(row, Mapping)
        )
        or any(not isinstance(row, Mapping) for row in skipped)
    ):
        raise CycleOpenComponentEquivalenceError("zero-support shard ledger is malformed")
    census = plan.get("active8_partition_census")
    if not isinstance(census, Mapping) or set(census) != {"validation"}:
        raise CycleOpenComponentEquivalenceError(
            "equivalence plan lacks the exact validation census"
        )
    if (
        sum(task["expected_cycle_attach_teacher_rows"] for task in tasks)
        != census["validation"]["cycle_attach_teacher_rows"]
        or len(tasks) + len(skipped) != census["validation"]["source_shards"]
    ):
        raise CycleOpenComponentEquivalenceError(
            "equivalence plan selection does not reconcile its census"
        )
    return plan


__all__ = [
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "active8_partition_census",
    "build_plan",
    "implementation_identity",
    "load_json_artifact",
    "pretty_json_bytes",
    "validate_parent_admission",
    "validate_plan",
]
