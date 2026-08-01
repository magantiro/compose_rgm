"""Tests for deterministic component-versus-exhaustive validation evidence."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import BOND_AROMATIC, BOND_SINGLE, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    NON_AUTHORIZING_STATUS,
    ORACLE_OVERFLOW,
    CycleOpenComponentEquivalenceError,
    audit_one_addressed_shard,
    audit_one_source,
    load_contract,
    reduce_shard_receipts,
    reduce_source_records,
    semantic_sha256,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import BondDelete, apply_bond_delete, is_valid_bond_delete
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace


SHARD_DIGEST = "1" * 64
PLAN_DIGEST = "2" * 64
IMPLEMENTATION_DIGEST = "3" * 64
CONTRACT_PATH = "configs/editing_cycle_open_component_equivalence_v1.json"


class _Admission:
    def __init__(self, *, accepted_entries: set[int], expected_entries: int) -> None:
        self.accepted_entries = accepted_entries
        self.expected_entries = expected_entries
        self.completed = False

    def is_accepted(self, address: PackedTraceAddress) -> bool:
        return address.entry_index in self.accepted_entries

    def assert_complete_source_shard(
        self,
        *,
        packed_shard_name: str,
        layer: str,
        partition: str,
        observed_digest: str | None,
        observed_entries: int,
    ) -> None:
        assert packed_shard_name == "fixture.jsonl.gz"
        assert layer == "cycle_ops"
        assert partition == "validation"
        assert observed_digest == SHARD_DIGEST
        assert observed_entries == self.expected_entries
        self.completed = True


def _addressed(smiles: str, *, entry_index: int) -> AddressedPackedTrace:
    source = pad_molecular_graph(smiles_to_molecular_graph(smiles), 20)
    perceived = resonance_invariant_bond_classes(source)
    candidates = tuple(
        BondDelete(left, right)
        for left in range(source.n_atoms)
        for right in range(left + 1, source.n_atoms)
        if int(perceived[left, right]) == BOND_AROMATIC
        and int(source.bonds[left, right]) == BOND_SINGLE
        and is_valid_bond_delete(source, BondDelete(left, right))
    )
    assert candidates
    action = candidates[0]
    target = apply_bond_delete(source, action)
    trace = RewriteTrace(
        source=source,
        target=target,
        steps=(RewriteStep("bond_delete", action),),
        metadata={"fixture": "component-equivalence"},
    )
    address = PackedTraceAddress(
        packed_shard_content_sha256=SHARD_DIGEST,
        packed_shard_name="fixture.jsonl.gz",
        entry_index=entry_index,
        trace_id=f"trace-{entry_index}",
        layer="cycle_ops",
        partition="validation",
        source_key=canonical_state_key(source),
        target_key=canonical_state_key(target),
        path_length=1,
    )
    return AddressedPackedTrace(
        address=address,
        trace=trace,
        path=PackedTraceProgress(trace, (source, target)),
    )


def _task(*, traces: int, accepted: int) -> dict:
    return {
        "task_index": 0,
        "layer": "cycle_ops",
        "partition": "validation",
        "packed_shard_name": "fixture.jsonl.gz",
        "packed_shard_content_sha256": SHARD_DIGEST,
        "expected_counts": {
            "traces": traces,
            "accepted_traces": accepted,
            "accepted_nonterminal_rows": accepted,
            "accepted_terminal_rows": accepted,
        },
        "expected_cycle_attach_teacher_rows": accepted,
    }


def test_source_record_covers_every_edge_and_separates_work_counts() -> None:
    record = audit_one_source(smiles_to_molecular_graph("c1ccccc1Oc2ccccc2"))

    assert record["status"] == NON_AUTHORIZING_STATUS
    assert record["training_authorized"] is False
    assert record["component_assignment_counts"] == [2, 2]
    assert record["counts"]["semantic_aromatic_edge_count"] == 12
    assert record["counts"]["equivalent_edge_count"] == 12
    assert record["counts"]["mismatch_edge_count"] == 0
    assert record["counts"]["oracle_overflow_edge_count"] == 0
    assert (
        record["counts"]["factored_executed_product_count"]
        < record["counts"]["exhaustive_executed_product_count_on_complete_edges"]
    )
    assert all(edge["semantic_equivalent"] for edge in record["edge_records"])


def test_contract_is_physically_pinned_self_hashed_and_validation_only(tmp_path) -> None:
    contract = load_contract(CONTRACT_PATH)
    assert contract["partitions"] == ["validation"]
    assert contract["excluded_partitions"] == ["train", "controller_validation", "test"]
    assert contract["training_authorized"] is False
    assert contract["comparison_policy"]["maximum_oracle_structures"] == 4096

    changed = tmp_path / "changed.json"
    changed.write_text(Path(CONTRACT_PATH).read_text().replace("4096", "8192"))
    with pytest.raises(CycleOpenComponentEquivalenceError, match="physical bytes"):
        load_contract(changed)


def test_oracle_overflow_is_recorded_and_cannot_pass_equivalence() -> None:
    record = audit_one_source(
        smiles_to_molecular_graph("c1ccccc1Oc2ccccc2"),
        maximum_oracle_structures=2,
    )

    assert record["counts"]["oracle_overflow_edge_count"] == 12
    assert {edge["outcome"] for edge in record["edge_records"]} == {ORACLE_OVERFLOW}
    result = reduce_source_records((record,), evidence_identity={"partitions": ["validation"]})
    assert result["equivalence_gate"]["passed"] is False


def test_reducer_deduplicates_only_identical_exact_source_records() -> None:
    record = audit_one_source(smiles_to_molecular_graph("c1ccccc1"))
    result = reduce_source_records(
        (record, copy.deepcopy(record)),
        evidence_identity={"partitions": ["validation"], "fixture": True},
    )

    assert result["unique_source_count"] == 1
    assert result["duplicate_source_record_count"] == 1
    assert result["equivalence_gate"]["passed"] is True
    assert result["training_authorized"] is False


def test_empty_evidence_cannot_pass_equivalence() -> None:
    result = reduce_source_records((), evidence_identity={"partitions": ["validation"]})
    assert result["unique_source_count"] == 0
    assert result["equivalence_gate"]["passed"] is False


def test_reducers_refuse_nonvalidation_evidence_identity() -> None:
    with pytest.raises(CycleOpenComponentEquivalenceError, match="validation-only"):
        reduce_source_records((), evidence_identity={"partitions": ["train"]})


def test_shard_receipt_reconciles_exact_admission_and_is_reducible() -> None:
    task = _task(traces=3, accepted=2)
    admission = _Admission(accepted_entries={0, 1}, expected_entries=3)
    rows = (
        _addressed("c1ccccc1", entry_index=0),
        _addressed("c1ccncc1", entry_index=1),
        _addressed("c1ccc2ccccc2c1", entry_index=2),
    )
    receipt = audit_one_addressed_shard(
        task,
        rows,
        admission,
        plan_sha256=PLAN_DIGEST,
        implementation_sha256=IMPLEMENTATION_DIGEST,
        maximum_oracle_structures=4096,
    )

    assert admission.completed
    assert receipt["counts"]["physical_traces_scanned"] == 3
    assert receipt["counts"]["accepted_traces_scanned"] == 2
    assert receipt["counts"]["excluded_traces_scanned"] == 1
    assert len(receipt["source_records"]) == 2
    result = reduce_shard_receipts(
        (receipt,),
        expected_tasks=(task,),
        plan_sha256=PLAN_DIGEST,
        implementation_sha256=IMPLEMENTATION_DIGEST,
        evidence_identity={"partitions": ["validation"], "fixture": True},
    )
    assert result["receipt_count"] == 1
    assert result["unique_source_count"] == 2
    assert result["equivalence_gate"]["passed"] is True


def test_shard_reducer_requires_exact_planned_receipt_set() -> None:
    with pytest.raises(CycleOpenComponentEquivalenceError, match="exact planned task set"):
        reduce_shard_receipts(
            (),
            expected_tasks=(_task(traces=1, accepted=1),),
            plan_sha256=PLAN_DIGEST,
            implementation_sha256=IMPLEMENTATION_DIGEST,
            evidence_identity={"partitions": ["validation"]},
        )


def test_reducer_rejects_tampered_or_conflicting_records() -> None:
    record = audit_one_source(smiles_to_molecular_graph("c1ccccc1"))
    tampered = copy.deepcopy(record)
    tampered["counts"]["equivalent_edge_count"] -= 1
    with pytest.raises(CycleOpenComponentEquivalenceError, match="self-hash"):
        reduce_source_records((tampered,), evidence_identity={"partitions": ["validation"]})

    failed_reconciliation = copy.deepcopy(record)
    failed_reconciliation["reconciliations"]["edge_partition"] = False
    body = {
        key: value for key, value in failed_reconciliation.items() if key != "source_record_sha256"
    }
    failed_reconciliation["source_record_sha256"] = semantic_sha256(body)
    with pytest.raises(CycleOpenComponentEquivalenceError, match="reconciliation"):
        reduce_source_records(
            (failed_reconciliation,),
            evidence_identity={"partitions": ["validation"]},
        )

    conflicting = copy.deepcopy(record)
    conflicting["outcomes"] = {"mismatch": 6}
    body = {key: value for key, value in conflicting.items() if key != "source_record_sha256"}
    conflicting["source_record_sha256"] = semantic_sha256(body)
    with pytest.raises(CycleOpenComponentEquivalenceError, match="nonidentical"):
        reduce_source_records(
            (record, conflicting),
            evidence_identity={"partitions": ["validation"]},
        )
