"""Contract and record tests for the prospective V2 cycle-open audit."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    NON_AUTHORIZING_STATUS as V1_NON_AUTHORIZING_STATUS,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    RECEIPT_SCHEMA as V1_RECEIPT_SCHEMA,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    RECEIPT_SCHEMA_VERSION as V1_RECEIPT_SCHEMA_VERSION,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    RESULT_SCHEMA as V1_RESULT_SCHEMA,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    RESULT_SCHEMA_VERSION as V1_RESULT_SCHEMA_VERSION,
)
from compose_v4.experiments.editing_cycle_open_component_equivalence import (
    semantic_sha256,
)
from compose_v4.experiments.editing_cycle_open_global_equivalence import (
    CycleOpenGlobalEquivalenceError,
    _audit_one_source,
    assignment_product_cardinality,
    atomic_write_if_absent,
    load_contract,
    validate_v1_receipt,
    validate_v1_result,
)


def test_frozen_v2_contract_validates() -> None:
    contract = load_contract("configs/editing_cycle_open_global_equivalence_v2.json")
    assert contract["partitions"] == ["validation"]
    assert contract["training_authorized"] is False
    assert contract["oracle_policy"]["maximum_assignments"] == 4096


def test_real_v1_failure_pattern_passes_complete_global_comparison() -> None:
    source = smiles_to_molecular_graph(
        "CC(C)(C)c1cc[n+](CCCCC[n+]2ccc(C(C)(C)C)cc2)cc1"
    )
    record = _audit_one_source(source, maximum_assignments=4096)
    assert (
        record["oracle_diagnostics"]["feasible_assignment_count_on_completed_sources"]
        == 4
    )
    assert (
        record["oracle_diagnostics"]["preserving_alias_count_on_completed_sources"] == 4
    )
    assert record["counts"]["semantic_aromatic_edge_count"] == 12
    assert record["counts"]["equivalent_edge_count"] == 12
    assert record["counts"]["mismatch_edge_count"] == 0
    assert record["nonpassing_edge_records"] == []
    assert all(record["reconciliations"].values())


def test_global_oracle_overflow_is_typed_and_blocks_every_edge() -> None:
    source = smiles_to_molecular_graph("c1ccccc1Oc2ccccc2")
    record = _audit_one_source(source, maximum_assignments=3)
    edge_count = record["counts"]["semantic_aromatic_edge_count"]
    assert edge_count == 12
    assert record["counts"]["complete_edge_comparison_count"] == 0
    assert record["counts"]["equivalent_edge_count"] == 0
    assert record["counts"]["mismatch_edge_count"] == 0
    assert record["counts"]["oracle_overflow_edge_count"] == edge_count
    assert record["oracle_diagnostics"]["completed_source_count"] == 0
    assert record["oracle_diagnostics"]["overflow_source_count"] == 1
    assert record["oracle_diagnostics"]["observed_assignments_at_overflow"] == 4
    assert len(record["nonpassing_edge_records"]) == edge_count
    assert all(record["reconciliations"].values())


def test_contract_physical_bytes_are_pinned(tmp_path) -> None:
    contract_path = tmp_path / "contract.json"
    contract = json.loads(
        Path("configs/editing_cycle_open_global_equivalence_v2.json").read_text()
    )
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(CycleOpenGlobalEquivalenceError, match="physical bytes"):
        load_contract(contract_path)


def _minimal_v1_receipt_and_contract() -> tuple[dict, dict]:
    contract = copy.deepcopy(
        load_contract("configs/editing_cycle_open_global_equivalence_v2.json")
    )
    body = {
        "schema": V1_RECEIPT_SCHEMA,
        "schema_version": V1_RECEIPT_SCHEMA_VERSION,
        "status": V1_NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "plan_sha256": contract["v1_negative_evidence"]["plan_sha256"],
        "implementation_sha256": contract["v1_negative_evidence"][
            "implementation_sha256"
        ],
        "task": {
            "packed_shard_content_sha256": contract["exact_validation_input"][
                "packed_shard_content_sha256"
            ]
        },
        "source_trace_identity_sha256s": {"a" * 64: ["b" * 64]},
        "reconciliations": {"complete": True},
    }
    receipt = {**body, "receipt_sha256": semantic_sha256(body)}
    contract["v1_negative_evidence"]["receipt_sha256"] = receipt["receipt_sha256"]
    return receipt, contract


def test_v1_source_ledger_requires_its_exact_self_hash() -> None:
    receipt, contract = _minimal_v1_receipt_and_contract()
    assert validate_v1_receipt(receipt, contract=contract) == receipt

    corrupted = copy.deepcopy(receipt)
    corrupted["source_trace_identity_sha256s"]["a" * 64].append("c" * 64)
    with pytest.raises(CycleOpenGlobalEquivalenceError, match="self-hash"):
        validate_v1_receipt(corrupted, contract=contract)


def test_v1_negative_result_must_remain_a_failed_gate() -> None:
    receipt, contract = _minimal_v1_receipt_and_contract()
    expected = contract["v1_negative_evidence"]
    body = {
        "schema": V1_RESULT_SCHEMA,
        "schema_version": V1_RESULT_SCHEMA_VERSION,
        "status": V1_NON_AUTHORIZING_STATUS,
        "training_authorized": False,
        "plan_sha256": expected["plan_sha256"],
        "implementation_sha256": expected["implementation_sha256"],
        "receipt_sha256s": [receipt["receipt_sha256"]],
        "totals": {
            "mismatch_edge_count": expected["mismatch_edge_count"],
            "semantic_aromatic_edge_count": expected["semantic_aromatic_edge_count"],
        },
        "nonpassing_source_records": [{}, {}],
        "equivalence_gate": {"passed": False},
    }
    result = {**body, "result_sha256": semantic_sha256(body)}
    contract["v1_negative_evidence"]["result_sha256"] = result["result_sha256"]
    assert validate_v1_result(result, contract=contract) == result

    falsely_passing = copy.deepcopy(result)
    falsely_passing["equivalence_gate"]["passed"] = True
    falsely_passing["result_sha256"] = semantic_sha256(
        {key: value for key, value in falsely_passing.items() if key != "result_sha256"}
    )
    contract["v1_negative_evidence"]["result_sha256"] = falsely_passing["result_sha256"]
    with pytest.raises(CycleOpenGlobalEquivalenceError, match="failure census"):
        validate_v1_result(falsely_passing, contract=contract)


def test_atomic_publication_reuses_identical_bytes_and_rejects_change(tmp_path) -> None:
    destination = tmp_path / "artifact.json"
    atomic_write_if_absent(destination, b"first\n")
    atomic_write_if_absent(destination, b"first\n")
    assert destination.read_bytes() == b"first\n"
    with pytest.raises(CycleOpenGlobalEquivalenceError, match="different bytes"):
        atomic_write_if_absent(destination, b"second\n")


def test_empty_component_product_contains_the_identity_assignment() -> None:
    assert assignment_product_cardinality(()) == 1
    assert assignment_product_cardinality((2, 3, 2)) == 12


def test_source_without_semantic_aromatic_edges_reconciles_without_vacuous_evidence() -> (
    None
):
    record = _audit_one_source(
        smiles_to_molecular_graph("CCC"), maximum_assignments=4096
    )
    assert record["counts"]["semantic_aromatic_edge_count"] == 0
    assert record["counts"]["complete_edge_comparison_count"] == 0
    assert (
        record["oracle_diagnostics"]["preserving_alias_count_on_completed_sources"] == 1
    )
    assert record["nonpassing_edge_records"] == []
    assert all(record["reconciliations"].values())
