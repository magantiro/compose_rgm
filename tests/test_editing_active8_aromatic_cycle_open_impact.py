from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_SINGLE,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    AromaticCycleOpenAliasOverflow,
)
from compose_v4.experiments.editing_active8_aromatic_cycle_open_impact import (
    NON_AUTHORIZING_STATUS,
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    AromaticCycleOpenImpactAuditError,
    audit_one_shard,
    build_audit_plan,
    load_audit_contract,
    reduce_shard_receipts,
    semantic_sha256,
    validate_audit_plan,
    validate_shard_receipt,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import (
    BondDelete,
    apply_bond_delete,
    is_valid_bond_delete,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace


CONTRACT_PATH = "configs/editing_active8_aromatic_cycle_open_impact_audit_v1.json"
SHARD_DIGEST = "1" * 64
IMPLEMENTATION_DIGEST = "2" * 64


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


def _cycle_open_action(source, *, aromatic: bool) -> BondDelete:
    perceived = resonance_invariant_bond_classes(source)
    candidates = []
    for left in range(source.n_atoms):
        for right in range(left + 1, source.n_atoms):
            is_aromatic = int(perceived[left, right]) == BOND_AROMATIC
            if (
                is_aromatic == aromatic
                and int(source.bonds[left, right]) == BOND_SINGLE
                and is_valid_bond_delete(source, BondDelete(left, right))
            ):
                candidates.append(BondDelete(left, right))
    assert candidates
    return candidates[0]


def _addressed(
    smiles: str,
    *,
    entry_index: int,
    aromatic: bool,
) -> AddressedPackedTrace:
    source = pad_molecular_graph(smiles_to_molecular_graph(smiles), 12)
    action = _cycle_open_action(source, aromatic=aromatic)
    target = apply_bond_delete(source, action)
    trace = RewriteTrace(
        source=source,
        target=target,
        steps=(RewriteStep("bond_delete", action),),
        metadata={"fixture": "aromatic-cycle-open-impact"},
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


def _task(*, traces: int, accepted: int, cycle_attach_rows: int) -> dict:
    body = {
        "task_index": 0,
        "manifest_layer": "cycle_ops",
        "layer": "cycle_ops",
        "partition": "validation",
        "relative_path": "validation/fixture.jsonl.gz",
        "packed_shard_name": "fixture.jsonl.gz",
        "packed_shard_content_sha256": SHARD_DIGEST,
        "packed_manifest_sha256": "3" * 64,
        "packed_provenance_overlay_sha256": "4" * 64,
        "expected_counts": {
            "traces": traces,
            "accepted_traces": accepted,
            "accepted_nonterminal_rows": accepted,
            "accepted_terminal_rows": accepted,
        },
        "expected_cycle_attach_teacher_rows": cycle_attach_rows,
    }
    body["task_sha256"] = semantic_sha256(body)
    body["receipt_filename"] = f"task_00000.{body['task_sha256']}.json"
    return body


def _plan(task: dict, contract: dict) -> dict:
    body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "contract_sha256": contract["contract_sha256"],
        "parent_active8_identity": contract["parent_active8_identity"],
        "inputs": {"fixture": "immutable"},
        "code_revision": {"commit": "fixture", "tree_dirty": False},
        "implementation": {
            "sources": {},
            "implementation_sha256": IMPLEMENTATION_DIGEST,
        },
        "partitions": ["validation"],
        "excluded_partitions": ["train", "controller_validation", "test"],
        "active8_partition_census": {
            "validation": {
                "accepted_traces": task["expected_counts"]["accepted_traces"],
                "source_shards": 1,
            }
        },
        "task_count": 1,
        "tasks": [task],
    }
    return {**body, "plan_sha256": semantic_sha256(body)}


def test_contract_is_self_hashed_and_explicitly_excludes_final_test(tmp_path) -> None:
    contract = load_audit_contract(CONTRACT_PATH)
    assert contract["partitions"] == ["validation"]
    assert contract["excluded_partitions"] == [
        "train",
        "controller_validation",
        "test",
    ]
    assert contract["training_authorized"] is False

    changed = copy.deepcopy(contract)
    changed["partitions"] = ["validation", "test"]
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(AromaticCycleOpenImpactAuditError):
        load_audit_contract(changed_path)


def test_plan_reports_available_partition_and_blocks_a_missing_required_role() -> None:
    contract = load_audit_contract(CONTRACT_PATH)
    parent = contract["parent_active8_identity"]
    admission = SimpleNamespace(
        manifest_path=Path(parent["inventory_manifest_path"]),
        manifest_file_sha256=parent["inventory_manifest_file_sha256"],
        inventory_sha256=parent["inventory_sha256"],
        effective_source_corpus_cache_sha256=parent["effective_source_corpus_cache_sha256"],
        support_contract_sha256=parent["support_contract_sha256"],
        unified_packed_manifest_sha256=parent["unified_packed_manifest_sha256"],
        shard_metadata_by_digest={
            "7" * 64: {
                "partition": "train",
                "counts": {
                    "traces": 2,
                    "accepted_traces": 1,
                    "excluded_traces": 1,
                    "accepted_nonterminal_rows": 1,
                    "accepted_terminal_rows": 1,
                },
                "accepted_nonterminal_rows_by_family": {"cycle_attach": 1},
            }
        },
    )
    with pytest.raises(
        AromaticCycleOpenImpactAuditError,
        match="BLOCKED:.*exact_available_partition_census=.*train",
    ):
        build_audit_plan(
            contract,
            admission,
            (),
            unified_manifest_file_sha256=parent["unified_packed_manifest_sha256"],
            inputs={"fixture": "immutable"},
            code_revision={"commit": "fixture", "tree_dirty": False},
            implementation={"implementation_sha256": IMPLEMENTATION_DIGEST},
        )


def test_shard_audit_reconciles_teacher_impact_and_source_reachability() -> None:
    contract = load_audit_contract(CONTRACT_PATH)
    task = _task(traces=6, accepted=5, cycle_attach_rows=5)
    plan = _plan(task, contract)
    admission = _Admission(accepted_entries={0, 1, 2, 3, 4}, expected_entries=6)
    rows = (
        _addressed("c1ccccc1", entry_index=0, aromatic=True),
        _addressed("C1CCCCC1", entry_index=1, aromatic=False),
        _addressed("c1ccncc1", entry_index=2, aromatic=True),
        _addressed("c1ccc2ccccc2c1", entry_index=3, aromatic=True),
        _addressed("C1CC2CCC1CC2", entry_index=4, aromatic=False),
        _addressed("c1cc[nH]c1", entry_index=5, aromatic=True),
    )

    receipt = audit_one_shard(
        task,
        rows,
        admission,
        contract=contract,
        plan_sha256=plan["plan_sha256"],
        implementation_sha256=IMPLEMENTATION_DIGEST,
    )
    validate_shard_receipt(
        receipt,
        expected_contract_sha256=contract["contract_sha256"],
        expected_plan_sha256=plan["plan_sha256"],
        expected_task=task,
        expected_implementation_sha256=IMPLEMENTATION_DIGEST,
    )
    assert admission.completed
    assert receipt["counts"]["physical_traces_scanned"] == 6
    assert receipt["counts"]["excluded_traces_scanned"] == 1
    assert receipt["counts"]["accepted_cycle_attach_teacher_rows"] == 5
    assert receipt["counts"]["perceived_aromatic_bond_delete_teacher_rows"] == 3
    assert receipt["counts"]["nonaromatic_cycle_attach_teacher_rows"] == 2
    assert receipt["counts"]["traces_with_perceived_aromatic_bond_delete_teacher_in_shard"] == 3
    assert receipt["raw_deleted_bond_class"] == {"single": 3}

    records = receipt["cycle_open_source_reachability"]
    assert len(records) == 5
    benzene_record = next(
        record
        for record in records
        if record["topology"]["aromatic_system_atom_sizes"] == [6]
        and not record["topology"]["heteroaromatic"]
        and not record["topology"]["topology_flags"]["fused_aromatic_rings"]
    )
    fused_record = next(
        record for record in records if record["topology"]["topology_flags"]["fused_aromatic_rings"]
    )
    bridged_record = next(
        record for record in records if record["topology"]["topology_flags"]["bridged_ring_source"]
    )
    assert benzene_record["semantic_aromatic_cycle_edge_count"] == 6
    assert benzene_record["admitted_semantic_aromatic_cycle_edge_count"] == 6
    assert benzene_record["topology"]["topology_flags"]["isolated_aromatic_system"]
    assert fused_record["topology"]["aromatic_system_atom_sizes"] == [10]
    assert bridged_record["semantic_aromatic_cycle_edge_count"] == 0
    assert bridged_record["zero_admitted_semantic_aromatic_cycle_edges"]

    result = reduce_shard_receipts(
        plan,
        [receipt],
        contract=contract,
        implementation_sha256=IMPLEMENTATION_DIGEST,
    )
    summary = result["source_reachability"]["summary"]
    assert summary["counts"]["unique_cycle_open_teacher_source_states"] == 5
    assert summary["counts"]["sources_with_semantic_aromatic_cycle_edges"] == 3
    assert summary["counts"]["sources_without_semantic_aromatic_cycle_edges"] == 2
    assert summary["counts"]["zero_admitted_sources"] == 2
    assert summary["counts"]["heteroaromatic_sources"] == 1
    assert summary["counts"].get("zero_admitted_sources_with_semantic_aromatic_edges", 0) == 0
    assert summary["overlapping_topology_flag_counts"] == {
        "bridged_ring_source": 1,
        "fused_aromatic_rings": 1,
        "isolated_aromatic_system": 2,
    }
    assert result["source_reachability"]["threshold_applied"] is False
    assert result["training_authorized"] is False


def test_typed_alias_overflow_is_counted_without_truncation() -> None:
    contract = load_audit_contract(CONTRACT_PATH)
    task = _task(traces=1, accepted=1, cycle_attach_rows=1)
    plan = _plan(task, contract)
    row = _addressed("c1ccccc1", entry_index=0, aromatic=True)

    def overflow_resolver(*args, maximum_aliases: int, **kwargs):
        raise AromaticCycleOpenAliasOverflow(
            maximum_aliases=maximum_aliases,
            observed_structures=maximum_aliases + 1,
        )

    receipt = audit_one_shard(
        task,
        (row,),
        _Admission(accepted_entries={0}, expected_entries=1),
        contract=contract,
        plan_sha256=plan["plan_sha256"],
        implementation_sha256=IMPLEMENTATION_DIGEST,
        resolver=overflow_resolver,
    )
    assert receipt["resolver_outcomes"] == {"kekule_alias_enumeration_overflow": 1}
    source = receipt["cycle_open_source_reachability"][0]
    assert source["resolver_outcomes"] == {"kekule_alias_enumeration_overflow": 6}
    assert source["zero_admitted_semantic_aromatic_cycle_edges"]


def test_plan_validation_fails_closed_if_a_task_is_changed_to_test() -> None:
    contract = load_audit_contract(CONTRACT_PATH)
    task = _task(traces=1, accepted=1, cycle_attach_rows=1)
    plan = _plan(task, contract)
    validate_audit_plan(plan, expected_contract_sha256=contract["contract_sha256"])

    changed = copy.deepcopy(plan)
    changed_task = changed["tasks"][0]
    changed_task["partition"] = "test"
    task_body = {
        key: value
        for key, value in changed_task.items()
        if key not in {"task_sha256", "receipt_filename"}
    }
    changed_task["task_sha256"] = semantic_sha256(task_body)
    changed_task["receipt_filename"] = f"task_00000.{changed_task['task_sha256']}.json"
    plan_body = {key: value for key, value in changed.items() if key != "plan_sha256"}
    changed["plan_sha256"] = semantic_sha256(plan_body)
    with pytest.raises(AromaticCycleOpenImpactAuditError):
        validate_audit_plan(changed)
