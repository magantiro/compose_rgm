"""Cycle-close complete-oracle audit contract and bounded regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.editing_cycle_close_global_equivalence import (
    CycleCloseGlobalEquivalenceError,
    audit_one_source,
    load_contract,
)

CONTRACT = Path("configs/editing_cycle_close_global_equivalence_v1.json")


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)


def test_contract_is_self_hashed_validation_only_and_physically_pinned() -> None:
    contract = load_contract(CONTRACT)
    assert contract["contract_sha256"] == (
        "3e471533963bf867ce9e3618da07a98ef24ab78750c91060f1d39f1b2b95d545"
    )
    assert contract["training_authorized"] is False
    assert contract["partitions"] == ["validation"]
    assert contract["exact_validation_input"]["expected_unique_exact_sources"] == 1707


def test_contract_physical_mutation_fails_loudly(tmp_path: Path) -> None:
    mutated = tmp_path / CONTRACT.name
    mutated.write_bytes(CONTRACT.read_bytes() + b"\n")
    with pytest.raises(CycleCloseGlobalEquivalenceError, match="physical bytes"):
        load_contract(mutated)


def test_aromatic_fixture_matches_complete_global_oracle_and_records_contraction() -> (
    None
):
    record = audit_one_source(_state("Cc1cccc(Cl)c1"), maximum_assignments=4096)
    assert record["oracle_overflow"] is False
    assert record["oracle_diagnostics"] == {
        "semantic_aromatic_edge_count": 6,
        "feasible_assignment_count": 2,
        "preserving_alias_count": 2,
        "solver_call_count": 3,
    }
    assert record["counts"] == {
        "candidate_count": 8,
        "equivalent_count": 8,
        "mismatch_count": 0,
        "admitted_count": 7,
        "ambiguous_rejection_count": 1,
        "other_rejection_count": 0,
        "factored_executed_product_count": 16,
        "oracle_executed_product_count": 16,
    }
    assert record["nonpassing_comparisons"] == []


def test_nonaromatic_fixture_preserves_every_raw_cycle_close_candidate() -> None:
    record = audit_one_source(_state("CCCCCC"), maximum_assignments=4096)
    assert record["oracle_overflow"] is False
    assert record["counts"]["candidate_count"] == 21
    assert record["counts"]["admitted_count"] == 21
    assert record["counts"]["mismatch_count"] == 0
    assert record["counts"]["ambiguous_rejection_count"] == 0
    assert record["oracle_diagnostics"]["preserving_alias_count"] == 1
