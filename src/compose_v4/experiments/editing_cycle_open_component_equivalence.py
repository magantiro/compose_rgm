"""Deterministic validation records for component-factored cycle opening.

This module compares the non-authorizing component-factored resolver with the
independent exhaustive RDKit oracle. It does not alter production actions,
legal fibers, executors, codecs, corpora, or training authority.

Runtime is represented here by deterministic work counters. Wall-clock
benchmarking belongs in a separately identified performance artifact because
mixing nondeterministic timings into immutable equivalence receipts would make
safe receipt reuse impossible.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import BOND_AROMATIC, BOND_NULL, MolecularGraph, is_element
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.active8_trace_inventory import Active8TraceAdmission
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    COMPONENT_FACTORED_PROTOTYPE_STATUS,
    PROTOTYPE_STATUS,
    AromaticCycleOpenAliasOverflow,
    AromaticCycleOpenResolution,
    enumerate_component_factored_kekule_assignments,
    resolve_component_factored_cycle_open,
    resolve_edge_anchored_cycle_open,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.operators import BondDelete


SOURCE_RECORD_SCHEMA = "compose.editing.cycle_open_component_equivalence_source"
SOURCE_RECORD_SCHEMA_VERSION = 1
RESULT_SCHEMA = "compose.editing.cycle_open_component_equivalence_result"
RESULT_SCHEMA_VERSION = 1
RECEIPT_SCHEMA = "compose.editing.cycle_open_component_equivalence_shard_receipt"
RECEIPT_SCHEMA_VERSION = 1
NON_AUTHORIZING_STATUS = (
    "VALIDATION_ONLY_COMPONENT_FACTORED_CYCLE_OPEN_EQUIVALENCE_NO_TRAINING_AUTHORITY"
)
ORACLE_OVERFLOW = "exhaustive_oracle_overflow"
CONTRACT_SCHEMA = "compose.editing.cycle_open_component_equivalence_contract"
CONTRACT_SCHEMA_VERSION = 1
PINNED_CONTRACT_FILE_SHA256 = "c6aba8525f421e7357a589aa815324d9cbe503ac22133109134619c2a8d86c59"
PINNED_IMPACT_RESULT_LOGICAL_SHA256 = (
    "cc1893991b75a4e592d344b592f6f12cf1eef3428781d6b85bddeb4ad75f3cf2"
)

_CONTRACT_FIELDS = {
    "comparison_policy",
    "contract_sha256",
    "edge_scope",
    "equivalence_gate",
    "excluded_partitions",
    "parent_active8_identity",
    "partitions",
    "prerequisite_impact_result_logical_sha256",
    "schema",
    "schema_version",
    "source_scope",
    "status",
    "training_authorized",
}
_PARENT_FIELDS = {
    "effective_source_corpus_cache_sha256",
    "inventory_manifest_file_sha256",
    "inventory_manifest_path",
    "inventory_sha256",
    "support_contract_sha256",
    "unified_packed_manifest_sha256",
}
_COMPARISON_FIELDS = {
    "component_resolver",
    "equivalence_fields",
    "exhaustive_oracle",
    "maximum_oracle_structures",
    "oracle_overflow_policy",
    "work_policy",
}


class CycleOpenComponentEquivalenceError(RuntimeError):
    """The comparison cannot establish its declared evidence boundary."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the deterministic JSON encoding used for semantic hashes."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CycleOpenComponentEquivalenceError(
            "equivalence value is not finite deterministic JSON"
        ) from error


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    except OSError as error:
        raise CycleOpenComponentEquivalenceError(
            f"cannot hash required equivalence input: {path}"
        ) from error
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load the physically pinned validation-only equivalence contract."""

    if file_sha256(path) != PINNED_CONTRACT_FILE_SHA256:
        raise CycleOpenComponentEquivalenceError(
            "component equivalence contract physical bytes disagree"
        )
    try:
        contract = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CycleOpenComponentEquivalenceError(
            f"cannot load component equivalence contract: {path}"
        ) from error
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_FIELDS:
        raise CycleOpenComponentEquivalenceError("component equivalence contract fields disagree")
    contract_hash = contract.get("contract_sha256")
    body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if not _is_sha256(contract_hash) or semantic_sha256(body) != contract_hash:
        raise CycleOpenComponentEquivalenceError(
            "component equivalence contract self-hash disagrees"
        )
    if (
        contract["schema"] != CONTRACT_SCHEMA
        or contract["schema_version"] != CONTRACT_SCHEMA_VERSION
        or contract["status"] != NON_AUTHORIZING_STATUS
        or contract["training_authorized"] is not False
        or contract["partitions"] != ["validation"]
        or contract["excluded_partitions"] != ["train", "controller_validation", "test"]
        or contract["source_scope"]
        != "all_distinct_exact_source_states_of_accepted_cycle_attach_teacher_rows"
        or contract["edge_scope"]
        != "all_resonance_invariant_aromatic_edges_in_each_declared_source"
        or contract["equivalence_gate"]
        != "zero_mismatches_zero_oracle_overflows_complete_declared_edge_coverage"
        or contract["prerequisite_impact_result_logical_sha256"]
        != PINNED_IMPACT_RESULT_LOGICAL_SHA256
    ):
        raise CycleOpenComponentEquivalenceError(
            "component equivalence contract scope, authority, or prerequisite disagrees"
        )
    parent = contract["parent_active8_identity"]
    if (
        not isinstance(parent, dict)
        or set(parent) != _PARENT_FIELDS
        or not isinstance(parent["inventory_manifest_path"], str)
        or not parent["inventory_manifest_path"].startswith("/artifacts/")
        or any(
            not _is_sha256(value)
            for key, value in parent.items()
            if key != "inventory_manifest_path"
        )
    ):
        raise CycleOpenComponentEquivalenceError(
            "component equivalence parent Active8 identity is malformed"
        )
    comparison = contract["comparison_policy"]
    if (
        not isinstance(comparison, dict)
        or set(comparison) != _COMPARISON_FIELDS
        or comparison["maximum_oracle_structures"] != 4096
        or comparison["oracle_overflow_policy"] != "typed_incomplete_comparison_blocks_gate"
        or comparison["work_policy"]
        != "record_actual_executions_separately_from_implied_global_cardinality"
    ):
        raise CycleOpenComponentEquivalenceError(
            "component equivalence comparison policy is invalid"
        )
    return contract


def _semantic_aromatic_edges(
    state: MolecularGraph,
) -> tuple[tuple[int, int], ...]:
    perceived = resonance_invariant_bond_classes(state)
    real_slots = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    return tuple(
        (left, right)
        for left_offset, left in enumerate(real_slots)
        for right in real_slots[left_offset + 1 :]
        if int(perceived[left, right]) == BOND_AROMATIC
    )


def _bridge_edges(state: MolecularGraph) -> frozenset[tuple[int, int]]:
    real_slots = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real_slots)
    graph.add_edges_from(
        (left, right)
        for left_offset, left in enumerate(real_slots)
        for right in real_slots[left_offset + 1 :]
        if int(state.bonds[left, right]) != BOND_NULL
    )
    return frozenset(tuple(sorted((int(left), int(right)))) for left, right in nx.bridges(graph))


def _rejection_value(resolution: AromaticCycleOpenResolution) -> str | None:
    return resolution.rejection_code.value if resolution.rejection_code is not None else None


def _successor_identity(
    resolution: AromaticCycleOpenResolution,
) -> dict[str, str] | None:
    if resolution.successor is None:
        return None
    return {
        "canonical_key_sha256": semantic_sha256(canonical_state_key(resolution.successor)),
        "exact_state_sha256": persistent_slot_state_sha256(resolution.successor),
    }


def _resolution_semantics(resolution: AromaticCycleOpenResolution) -> dict[str, Any]:
    """Return fields that must agree across the two independent resolvers."""

    return {
        "admitted": resolution.admitted,
        "canonical_product_key_sha256s": [
            semantic_sha256(key) for key in resolution.canonical_product_keys
        ],
        "edge": list(resolution.edge),
        "enumerated_alias_count": resolution.enumerated_alias_count,
        "forced_single_alias_count": resolution.forced_single_alias_count,
        "inverse_bond_order": resolution.inverse_bond_order,
        "rejection_code": _rejection_value(resolution),
        "semantic_aromatic_edge": resolution.semantic_aromatic_edge,
        "source_key_sha256": (
            semantic_sha256(resolution.source_key) if resolution.source_key is not None else None
        ),
        "successor": _successor_identity(resolution),
    }


def compare_one_edge(
    state: MolecularGraph,
    edge: tuple[int, int],
    *,
    maximum_oracle_structures: int,
) -> dict[str, Any]:
    """Compare one semantic edge without treating implied work as executed."""

    if maximum_oracle_structures <= 0:
        raise ValueError("maximum_oracle_structures must be positive")
    normalized = tuple(sorted((int(edge[0]), int(edge[1]))))
    factored = resolve_component_factored_cycle_open(state, BondDelete(*normalized))
    if factored.prototype_status != COMPONENT_FACTORED_PROTOTYPE_STATUS:
        raise CycleOpenComponentEquivalenceError(
            "component resolver returned an unexpected prototype identity"
        )

    oracle: AromaticCycleOpenResolution | None
    try:
        oracle = resolve_edge_anchored_cycle_open(
            state,
            BondDelete(*normalized),
            maximum_aliases=maximum_oracle_structures,
        )
    except AromaticCycleOpenAliasOverflow:
        oracle = None

    factored_semantics = _resolution_semantics(factored)
    oracle_semantics = _resolution_semantics(oracle) if oracle is not None else None
    comparison_complete = oracle is not None
    semantic_equivalent = comparison_complete and oracle_semantics == factored_semantics
    if oracle is not None and oracle.prototype_status != PROTOTYPE_STATUS:
        raise CycleOpenComponentEquivalenceError(
            "exhaustive resolver returned an unexpected prototype identity"
        )

    work = {
        "factored_executed_products": factored.executed_product_count,
        "factored_global_alias_cardinality": factored.enumerated_alias_count,
        "factored_global_forced_single_cardinality": factored.forced_single_alias_count,
        "oracle_executed_products": (oracle.executed_product_count if oracle is not None else None),
    }
    if oracle is not None and factored.admitted:
        if oracle.executed_product_count != factored.forced_single_alias_count:
            raise CycleOpenComponentEquivalenceError(
                "exhaustive execution count does not match global forced-single cardinality"
            )
        if factored.executed_product_count > factored.forced_single_alias_count:
            raise CycleOpenComponentEquivalenceError(
                "component solver executed more products than the represented global fiber"
            )

    body = {
        "edge": list(normalized),
        "comparison_complete": comparison_complete,
        "semantic_equivalent": semantic_equivalent,
        "outcome": (
            "equivalent"
            if semantic_equivalent
            else ("mismatch" if comparison_complete else ORACLE_OVERFLOW)
        ),
        "oracle": oracle_semantics,
        "factored": factored_semantics,
        "work": work,
    }
    return {**body, "edge_record_sha256": semantic_sha256(body)}


def audit_one_source(
    state: MolecularGraph,
    *,
    maximum_oracle_structures: int = 4096,
) -> dict[str, Any]:
    """Build one deterministic record covering every semantic aromatic edge."""

    if type(maximum_oracle_structures) is not int or maximum_oracle_structures <= 0:
        raise ValueError("maximum_oracle_structures must be a positive integer")
    source_digest = persistent_slot_state_sha256(state)
    source_key = canonical_state_key(state)
    components = enumerate_component_factored_kekule_assignments(state)
    aromatic_edges = _semantic_aromatic_edges(state)
    bridge_edges = _bridge_edges(state)
    edge_records = tuple(
        compare_one_edge(
            state,
            edge,
            maximum_oracle_structures=maximum_oracle_structures,
        )
        for edge in aromatic_edges
    )
    outcomes = Counter(str(record["outcome"]) for record in edge_records)
    counts = {
        "aromatic_component_count": len(components),
        "semantic_aromatic_edge_count": len(aromatic_edges),
        "semantic_aromatic_bridge_edge_count": sum(edge in bridge_edges for edge in aromatic_edges),
        "complete_edge_comparison_count": sum(
            bool(record["comparison_complete"]) for record in edge_records
        ),
        "equivalent_edge_count": outcomes["equivalent"],
        "mismatch_edge_count": outcomes["mismatch"],
        "oracle_overflow_edge_count": outcomes[ORACLE_OVERFLOW],
        "factored_executed_product_count": sum(
            int(record["work"]["factored_executed_products"]) for record in edge_records
        ),
        "exhaustive_executed_product_count_on_complete_edges": sum(
            int(record["work"]["oracle_executed_products"] or 0) for record in edge_records
        ),
    }
    reconciliations = {
        "edge_partition": (
            counts["equivalent_edge_count"]
            + counts["mismatch_edge_count"]
            + counts["oracle_overflow_edge_count"]
            == counts["semantic_aromatic_edge_count"]
        ),
        "complete_partition": (
            counts["complete_edge_comparison_count"]
            == counts["equivalent_edge_count"] + counts["mismatch_edge_count"]
        ),
    }
    if not all(reconciliations.values()):
        raise CycleOpenComponentEquivalenceError(
            f"source comparison denominators do not reconcile: {reconciliations}"
        )
    body = {
        "schema": SOURCE_RECORD_SCHEMA,
        "schema_version": SOURCE_RECORD_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "maximum_oracle_structures": maximum_oracle_structures,
        "exact_source_state_sha256": source_digest,
        "source_canonical_key_sha256": semantic_sha256(source_key),
        "component_edge_counts": [len(component.edges) for component in components],
        "component_assignment_counts": [len(component.bond_orders) for component in components],
        "counts": counts,
        "outcomes": dict(sorted(outcomes.items())),
        "edge_records": list(edge_records),
        "reconciliations": reconciliations,
    }
    return {**body, "source_record_sha256": semantic_sha256(body)}


def _trace_identity(address: object) -> str:
    return semantic_sha256(
        {
            "packed_shard_content_sha256": address.packed_shard_content_sha256,
            "entry_index": address.entry_index,
            "trace_id": address.trace_id,
        }
    )


def audit_one_addressed_shard(
    task: Mapping[str, Any],
    addressed_traces: Iterable[object],
    admission: Active8TraceAdmission,
    *,
    plan_sha256: str,
    implementation_sha256: str,
    maximum_oracle_structures: int,
) -> dict[str, Any]:
    """Audit one exact physical validation shard and seal a reusable receipt."""

    if task.get("partition") != "validation":
        raise CycleOpenComponentEquivalenceError(
            "component equivalence receipts are validation-only"
        )
    if not _is_sha256(plan_sha256) or not _is_sha256(implementation_sha256):
        raise CycleOpenComponentEquivalenceError(
            "receipt requires plan and implementation identities"
        )
    if maximum_oracle_structures != 4096:
        raise CycleOpenComponentEquivalenceError(
            "frozen shard equivalence audit requires the contracted 4096-structure oracle cap"
        )
    counts: Counter[str] = Counter()
    family_rows: Counter[str] = Counter()
    source_states: dict[str, MolecularGraph] = {}
    source_trace_identities: dict[str, set[str]] = {}
    observed_digest: str | None = None

    for addressed in addressed_traces:
        address = addressed.address
        counts["physical_traces_scanned"] += 1
        if (
            address.packed_shard_name != task["packed_shard_name"]
            or address.layer != task["layer"]
            or address.partition != task["partition"]
            or address.packed_shard_content_sha256 != task["packed_shard_content_sha256"]
        ):
            raise CycleOpenComponentEquivalenceError(
                "decoded packed address disagrees with its exact shard task"
            )
        if observed_digest is None:
            observed_digest = address.packed_shard_content_sha256
        elif observed_digest != address.packed_shard_content_sha256:
            raise CycleOpenComponentEquivalenceError(
                "one shard stream yielded multiple physical digests"
            )
        if not admission.is_accepted(address):
            counts["excluded_traces_scanned"] += 1
            continue
        counts["accepted_traces_scanned"] += 1
        if len(addressed.trace.steps) != address.path_length:
            raise CycleOpenComponentEquivalenceError(
                "accepted packed trace path length disagrees with its address"
            )
        counts["accepted_nonterminal_teacher_rows"] += address.path_length
        trace_identity = _trace_identity(address)
        for progress_index, step in enumerate(addressed.trace.steps):
            family = canonical_family(step.rule_name)
            family_rows[family] += 1
            if family != "cycle_attach":
                continue
            counts["accepted_cycle_attach_teacher_rows"] += 1
            if step.rule_name != "bond_delete" or not isinstance(step.action, BondDelete):
                raise CycleOpenComponentEquivalenceError(
                    "cycle_attach teacher is not the frozen BondDelete action"
                )
            source = addressed.path.state_at(progress_index)
            source_digest = persistent_slot_state_sha256(source)
            source_states.setdefault(source_digest, source)
            source_trace_identities.setdefault(source_digest, set()).add(trace_identity)

    admission.assert_complete_source_shard(
        packed_shard_name=str(task["packed_shard_name"]),
        layer=str(task["layer"]),
        partition=str(task["partition"]),
        observed_digest=observed_digest,
        observed_entries=counts["physical_traces_scanned"],
    )
    expected = task["expected_counts"]
    reconciliations = {
        "physical_traces": counts["physical_traces_scanned"] == expected["traces"],
        "accepted_plus_excluded_traces": (
            counts["accepted_traces_scanned"] + counts["excluded_traces_scanned"]
            == counts["physical_traces_scanned"]
        ),
        "accepted_traces": counts["accepted_traces_scanned"] == expected["accepted_traces"],
        "accepted_terminal_rows": (
            counts["accepted_traces_scanned"] == expected["accepted_terminal_rows"]
        ),
        "accepted_nonterminal_rows": (
            counts["accepted_nonterminal_teacher_rows"] == expected["accepted_nonterminal_rows"]
        ),
        "family_rows": sum(family_rows.values()) == counts["accepted_nonterminal_teacher_rows"],
        "cycle_attach_rows": (
            counts["accepted_cycle_attach_teacher_rows"]
            == task["expected_cycle_attach_teacher_rows"]
        ),
        "source_trace_identity_keys": set(source_trace_identities) == set(source_states),
    }
    if not all(reconciliations.values()):
        raise CycleOpenComponentEquivalenceError(
            f"shard equivalence denominators do not reconcile: {reconciliations}"
        )

    source_records = [
        audit_one_source(
            source_states[source_digest],
            maximum_oracle_structures=maximum_oracle_structures,
        )
        for source_digest in sorted(source_states)
    ]
    body = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "plan_sha256": plan_sha256,
        "implementation_sha256": implementation_sha256,
        "maximum_oracle_structures": maximum_oracle_structures,
        "task": dict(task),
        "counts": dict(sorted(counts.items())),
        "accepted_teacher_rows_by_family": dict(sorted(family_rows.items())),
        "source_trace_identity_sha256s": {
            digest: sorted(source_trace_identities[digest]) for digest in sorted(source_states)
        },
        "source_records": source_records,
        "reconciliations": reconciliations,
    }
    return {**body, "receipt_sha256": semantic_sha256(body)}


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    expected_plan_sha256: str,
    expected_implementation_sha256: str,
    expected_task: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = dict(value)
    receipt_hash = receipt.pop("receipt_sha256", None)
    if receipt_hash != semantic_sha256(receipt):
        raise CycleOpenComponentEquivalenceError("shard receipt self-hash disagrees")
    receipt["receipt_sha256"] = receipt_hash
    reconciliations = receipt.get("reconciliations")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != NON_AUTHORIZING_STATUS
        or receipt.get("training_authorized") is not False
        or receipt.get("plan_sha256") != expected_plan_sha256
        or receipt.get("implementation_sha256") != expected_implementation_sha256
        or receipt.get("task") != dict(expected_task)
        or receipt.get("maximum_oracle_structures") != 4096
        or not isinstance(reconciliations, Mapping)
        or not reconciliations
        or not all(value is True for value in reconciliations.values())
    ):
        raise CycleOpenComponentEquivalenceError(
            "shard receipt schema, identity, task, authority, or reconciliation disagrees"
        )
    return receipt


def reduce_shard_receipts(
    receipts: Iterable[Mapping[str, Any]],
    *,
    expected_tasks: Iterable[Mapping[str, Any]],
    plan_sha256: str,
    implementation_sha256: str,
    evidence_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce exactly the planned shard set into one non-authorizing result."""

    if evidence_identity.get("partitions") != ["validation"]:
        raise CycleOpenComponentEquivalenceError(
            "shard reduction evidence identity must remain explicitly validation-only"
        )
    task_by_index = {int(task["task_index"]): dict(task) for task in expected_tasks}
    receipt_by_index: dict[int, dict[str, Any]] = {}
    for value in receipts:
        task = value.get("task")
        if not isinstance(task, Mapping) or type(task.get("task_index")) is not int:
            raise CycleOpenComponentEquivalenceError("receipt lacks a planned task index")
        task_index = int(task["task_index"])
        if task_index not in task_by_index or task_index in receipt_by_index:
            raise CycleOpenComponentEquivalenceError("receipt task is unplanned or duplicated")
        receipt_by_index[task_index] = _validate_receipt(
            value,
            expected_plan_sha256=plan_sha256,
            expected_implementation_sha256=implementation_sha256,
            expected_task=task_by_index[task_index],
        )
    if set(receipt_by_index) != set(task_by_index):
        raise CycleOpenComponentEquivalenceError(
            "receipt coverage does not equal the exact planned task set"
        )

    source_records = (
        record
        for task_index in sorted(receipt_by_index)
        for record in receipt_by_index[task_index]["source_records"]
    )
    result = reduce_source_records(source_records, evidence_identity=evidence_identity)
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    body.update(
        {
            "plan_sha256": plan_sha256,
            "implementation_sha256": implementation_sha256,
            "receipt_count": len(receipt_by_index),
            "receipt_sha256s": [
                receipt_by_index[index]["receipt_sha256"] for index in sorted(receipt_by_index)
            ],
        }
    )
    return {**body, "result_sha256": semantic_sha256(body)}


def reduce_source_records(
    records: Iterable[Mapping[str, Any]],
    *,
    evidence_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Deduplicate exact sources, reconcile denominators, and fail closed."""

    if evidence_identity.get("partitions") != ["validation"]:
        raise CycleOpenComponentEquivalenceError(
            "source reduction evidence identity must remain explicitly validation-only"
        )
    by_source: dict[str, dict[str, Any]] = {}
    duplicate_records = 0
    for value in records:
        record = dict(value)
        expected_hash = record.pop("source_record_sha256", None)
        if expected_hash != semantic_sha256(record):
            raise CycleOpenComponentEquivalenceError("source record self-hash disagrees")
        record["source_record_sha256"] = expected_hash
        if (
            record.get("schema") != SOURCE_RECORD_SCHEMA
            or record.get("schema_version") != SOURCE_RECORD_SCHEMA_VERSION
            or record.get("status") != NON_AUTHORIZING_STATUS
            or record.get("training_authorized") is not False
        ):
            raise CycleOpenComponentEquivalenceError(
                "source record schema, status, or authority disagrees"
            )
        if (
            type(record.get("maximum_oracle_structures")) is not int
            or record["maximum_oracle_structures"] <= 0
            or not isinstance(record.get("counts"), Mapping)
            or any(type(value) is not int or value < 0 for value in record["counts"].values())
        ):
            raise CycleOpenComponentEquivalenceError(
                "source record oracle cap or count fields are malformed"
            )
        reconciliations = record.get("reconciliations")
        if (
            not isinstance(reconciliations, Mapping)
            or not reconciliations
            or not all(value is True for value in reconciliations.values())
        ):
            raise CycleOpenComponentEquivalenceError(
                "source record reconciliation is absent or failed"
            )
        source_digest = record.get("exact_source_state_sha256")
        if not _is_sha256(source_digest):
            raise CycleOpenComponentEquivalenceError("source record lacks exact identity")
        previous = by_source.setdefault(source_digest, record)
        if previous != record:
            raise CycleOpenComponentEquivalenceError(
                "duplicate exact source produced nonidentical equivalence records"
            )
        if previous is not record:
            duplicate_records += 1

    oracle_caps = {int(record["maximum_oracle_structures"]) for record in by_source.values()}
    if len(oracle_caps) > 1:
        raise CycleOpenComponentEquivalenceError(
            "source records use multiple exhaustive-oracle caps"
        )
    totals: Counter[str] = Counter()
    for record in by_source.values():
        totals.update({key: int(value) for key, value in record["counts"].items()})
    complete = (
        len(by_source) > 0
        and totals["semantic_aromatic_edge_count"] > 0
        and totals["oracle_overflow_edge_count"] == 0
        and totals["mismatch_edge_count"] == 0
        and totals["complete_edge_comparison_count"] == totals["semantic_aromatic_edge_count"]
    )
    body = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "evidence_identity": dict(evidence_identity),
        "unique_source_count": len(by_source),
        "duplicate_source_record_count": duplicate_records,
        "totals": dict(sorted(totals.items())),
        "equivalence_gate": {
            "passed": complete,
            "required": (
                "zero mismatches, zero oracle overflows, and complete coverage of every "
                "semantic aromatic edge in the declared validation source set"
            ),
        },
        "source_record_sha256s": [
            by_source[digest]["source_record_sha256"] for digest in sorted(by_source)
        ],
    }
    return {**body, "result_sha256": semantic_sha256(body)}


__all__ = [
    "CONTRACT_SCHEMA",
    "CONTRACT_SCHEMA_VERSION",
    "NON_AUTHORIZING_STATUS",
    "ORACLE_OVERFLOW",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_VERSION",
    "RECEIPT_SCHEMA",
    "RECEIPT_SCHEMA_VERSION",
    "SOURCE_RECORD_SCHEMA",
    "SOURCE_RECORD_SCHEMA_VERSION",
    "CycleOpenComponentEquivalenceError",
    "audit_one_source",
    "audit_one_addressed_shard",
    "canonical_json_bytes",
    "compare_one_edge",
    "file_sha256",
    "load_contract",
    "reduce_source_records",
    "reduce_shard_receipts",
    "semantic_sha256",
]
