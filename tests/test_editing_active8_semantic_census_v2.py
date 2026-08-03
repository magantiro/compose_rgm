from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import compose_v4.experiments.editing_active8_semantic_census_v2 as census_module
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    ELEMENT_TO_IDX,
    MolecularGraph,
)
from compose_v4.data.editing_v2_process_v2_schema import require_no_granted_authority
from compose_v4.experiments.editing_active8_semantic_census_v2 import (
    SEMANTIC_CENSUS_SCHEMA_VERSION,
    Active8SemanticCensusError,
    OrthogonalCandidateAliasMetrics,
    build_semantic_census,
    derive_semantic_observation,
    load_semantic_census_contract,
    semantic_census_bytes,
    validate_parent_t1_successor_gate,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/editing_active8_within_family_semantic_census_v2.json"
PARENT_T1_GATE = ROOT / "configs/editing_t1_successor_gate_v8.json"


def _state(
    symbols: list[str],
    edges: list[tuple[int, int, int]],
    *,
    charges: list[int] | None = None,
    hydrogens: list[int] | None = None,
) -> MolecularGraph:
    atom_types = np.asarray([ELEMENT_TO_IDX[symbol] for symbol in symbols], dtype=np.int32)
    n = len(symbols)
    bonds = np.zeros((n, n), dtype=np.int32)
    for a, b, order in edges:
        bonds[a, b] = bonds[b, a] = order
    return MolecularGraph(
        atom_types=atom_types,
        formal_charges=np.asarray(charges or [0] * n, dtype=np.int32),
        implicit_h_counts=np.asarray(hydrogens or [0] * n, dtype=np.int32),
        bonds=bonds,
    )


def test_derives_family_specific_graph_context_without_joint_cells() -> None:
    source = _state(
        ["C", "C", "O", "N"],
        [(0, 1, BOND_SINGLE), (1, 2, BOND_DOUBLE), (2, 3, BOND_SINGLE)],
    )
    successor = _state(
        ["C", "C", "O", "N"],
        [(0, 1, BOND_SINGLE), (0, 3, BOND_SINGLE), (2, 3, BOND_SINGLE)],
    )
    observation = derive_semantic_observation(
        source,
        successor,
        executor_rule="bond_reroute",
        action=BondReroute(a=1, b=2, u=0, v=3, new_order=BOND_SINGLE),
        data_lane="operator_aware_real_endpoint",
        step_index=1,
        path_length=3,
    )

    assert observation.family == "bond_reroute"
    assert observation.axes["trace_step_role"] == "interior"
    assert observation.axes["source_cycle_rank"] == "0"
    assert observation.axes["bond_order_transition"] == "double->single"
    assert observation.axes["cut_component_size_pair"] == "2|2"
    assert observation.axes["reused_removed_endpoint_count"] == "0"
    assert observation.axes["removed_edge_cycle_context"] == "bridge"
    assert "semantic_cell" not in observation.axes


def test_cycle_close_and_open_use_exact_shortest_paths() -> None:
    chain = _state(
        ["C", "C", "N"],
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE)],
    )
    ring = _state(
        ["C", "C", "N"],
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE), (0, 2, BOND_SINGLE)],
    )
    close = derive_semantic_observation(
        chain,
        ring,
        executor_rule="bond_insert",
        action=BondInsert(a=0, b=2, order=BOND_SINGLE),
        data_lane="cycle_ops",
        step_index=0,
        path_length=1,
    )
    opened = derive_semantic_observation(
        ring,
        chain,
        executor_rule="bond_delete",
        action=BondDelete(a=0, b=2),
        data_lane="cycle_ops",
        step_index=0,
        path_length=1,
    )

    assert close.axes["preclosure_shortest_path_edges"] == "2"
    assert close.axes["closed_shortest_cycle_size"] == "3"
    assert opened.axes["remaining_shortest_path_edges"] == "2"
    assert opened.axes["opened_shortest_cycle_size"] == "3"


def test_delete_restate_and_reorder_axes_are_row_local() -> None:
    bonded = _state(["C", "O"], [(0, 1, BOND_DOUBLE)], hydrogens=[2, 0])
    deleted = _state(["C", "null"], [], hydrogens=[4, 0])
    delete = derive_semantic_observation(
        bonded,
        deleted,
        executor_rule="atom_delete",
        action=AtomDelete(v=1),
        data_lane="mmp_analogue",
        step_index=0,
        path_length=1,
    )
    assert delete.axes["deletion_mode"] == "leaf"
    assert delete.axes["deleted_element"] == "O"
    assert delete.axes["incident_bond_signature"] == "double"

    triangle = _state(
        ["C", "C", "O"],
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE), (0, 2, BOND_SINGLE)],
        hydrogens=[1, 1, 0],
    )
    restated = _state(
        ["N", "C", "O"],
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE), (0, 2, BOND_SINGLE)],
        hydrogens=[0, 1, 0],
    )
    restate = derive_semantic_observation(
        triangle,
        restated,
        executor_rule="atom_restate",
        action=AtomRestate(
            v=0,
            atom_type=ELEMENT_TO_IDX["N"],
            formal_charge=0,
            implicit_h_count=0,
        ),
        data_lane="corruption",
        step_index=0,
        path_length=2,
    )
    assert restate.axes["element_transition"] == "C->N"
    assert restate.axes["implicit_h_delta"] == "-1"
    assert restate.axes["site_cycle_context"] == "cyclic"
    assert restate.axes["cyclic_target_support_class"] == ("cyclic_declared_ring_element")

    reordered = _state(["C", "O"], [(0, 1, BOND_SINGLE)])
    reorder = derive_semantic_observation(
        bonded,
        reordered,
        executor_rule="bond_reorder",
        action=BondReorder(a=0, b=1, new_order=BOND_SINGLE),
        data_lane="corruption",
        step_index=1,
        path_length=2,
    )
    assert reorder.axes["bond_order_transition"] == "double->single"
    assert reorder.axes["endpoint_element_pair"] == "C|O"
    assert reorder.axes["source_edge_cycle_context"] == "bridge"


def test_cyclic_carbon_to_iodine_is_retained_as_support_consistency_diagnostic() -> None:
    source = _state(
        ["C", "C", "C"],
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE), (0, 2, BOND_SINGLE)],
        hydrogens=[1, 1, 1],
    )
    successor = _state(
        ["I", "C", "C"],
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE), (0, 2, BOND_SINGLE)],
        hydrogens=[0, 1, 1],
    )
    observation = derive_semantic_observation(
        source,
        successor,
        executor_rule="atom_restate",
        action=AtomRestate(
            v=0,
            atom_type=ELEMENT_TO_IDX["I"],
            formal_charge=0,
            implicit_h_count=0,
        ),
        data_lane="forensic_floor_failure",
        step_index=0,
        path_length=1,
    )

    assert observation.axes["element_transition"] == "C->I"
    assert observation.axes["site_cycle_context"] == "cyclic"
    assert observation.axes["cyclic_target_support_class"] == "cyclic_nonring_element"


def test_ring_restate_reports_a_transition_multiset() -> None:
    source = _state(
        ["C", "N", "C"],
        [(0, 1, BOND_AROMATIC), (1, 2, BOND_AROMATIC), (0, 2, BOND_AROMATIC)],
    )
    successor = _state(
        ["C", "N", "C"],
        [(0, 1, BOND_SINGLE), (1, 2, BOND_DOUBLE), (0, 2, BOND_AROMATIC)],
    )
    observation = derive_semantic_observation(
        source,
        successor,
        executor_rule="ring_system_restate",
        action=RingSystemRestate(
            changes=(
                BondOrderChange(0, 1, BOND_SINGLE),
                BondOrderChange(1, 2, BOND_DOUBLE),
            )
        ),
        data_lane="reversible_synthetic_walk",
        step_index=0,
        path_length=1,
    )

    assert observation.axes["changed_bond_count"] == "2"
    assert observation.axes["affected_atom_count"] == "3"
    assert observation.axes["bond_order_transition_multiset"] == (
        "aromatic->double+aromatic->single"
    )
    assert observation.axes["changed_edge_cycle_context"] == "nonbridge"
    assert observation.axes["aromatic_transition_presence"] == "present"


def test_aggregation_keeps_alias_metrics_orthogonal_and_absence_nondecisional() -> None:
    contract = load_semantic_census_contract(CONTRACT)
    source = _state(["C", "null"], [])
    successor = _state(["C", "O"], [(0, 1, BOND_SINGLE)], hydrogens=[0, 1])
    observation = derive_semantic_observation(
        source,
        successor,
        executor_rule="atom_insert",
        action=AtomInsert(
            slot=1,
            atom_type=ELEMENT_TO_IDX["O"],
            formal_charge=0,
            implicit_h_count=1,
            neighbors=((0, BOND_SINGLE),),
        ),
        data_lane="operator_aware_real_endpoint",
        step_index=0,
        path_length=1,
        candidate_alias=OrthogonalCandidateAliasMetrics(12, 9, 1, 2),
    )
    report = build_semantic_census(
        [observation, observation],
        contract=contract,
        provenance={"fixture": "exact_slot_state"},
    )

    insert = report["families"]["atom_insert"]
    inserted_element = insert["axes"]["inserted_element"]
    assert inserted_element["observed_row_counts"] == {"O": 2}
    assert inserted_element["unique_exact_source_state_counts"] == {"O": 1}
    assert inserted_element["unique_exact_successor_state_counts"] == {"O": 1}
    assert "C" in insert["axes"]["inserted_element"]["absent_declared_values"]
    assert insert["axes"]["source_atom_count"]["absent_declared_values"] is None
    assert insert["rows"] == 2
    assert insert["unique_exact_source_states"] == 1
    assert insert["unique_exact_successor_states"] == 1
    assert insert["unique_scaffolds"] is None
    assert insert["scaffold_count_claimed"] is False
    orthogonal = insert["orthogonal_candidate_alias_diagnostics"]
    assert orthogonal["availability"] == "complete"
    assert orthogonal["metric_marginals"]["raw_legal_mark_count"] == {"12": 2}
    assert orthogonal["joined_to_semantic_axes"] is False
    assert report["scope"]["cartesian_joint_cells_constructed"] is False
    assert report["scope"]["scaffold_count_claimed"] is False
    assert report["counts"]["unique_scaffolds"] is None
    assert report["schema_version"] == SEMANTIC_CENSUS_SCHEMA_VERSION == 3
    assert report["decisions"] == {
        "corpus_failure": None,
        "model_learnability": None,
        "training_authorized": False,
        "bounded_p50_authorized": False,
    }
    # The retired spelling is gone from the whole report, not only from the block
    # that carried it, and the frozen guard now reaches this artifact.
    assert "p50_authorized" not in semantic_census_bytes(report).decode().replace(
        "bounded_p50_authorized", ""
    )
    require_no_granted_authority(report, label="the semantic census report")
    assert semantic_census_bytes(report).decode().count('"model_learnability"') == 1


def test_current_birth_factorization_rejects_multi_neighbor_insert() -> None:
    source = _state(["C", "C", "null"], [(0, 1, BOND_SINGLE)])
    successor = _state(
        ["C", "C", "O"],
        [(0, 1, BOND_SINGLE), (0, 2, BOND_SINGLE), (1, 2, BOND_SINGLE)],
    )
    with pytest.raises(Active8SemanticCensusError, match="exactly-one-neighbor"):
        derive_semantic_observation(
            source,
            successor,
            executor_rule="atom_insert",
            action=AtomInsert(
                slot=2,
                atom_type=ELEMENT_TO_IDX["O"],
                formal_charge=0,
                implicit_h_count=0,
                neighbors=((0, BOND_SINGLE), (1, BOND_SINGLE)),
            ),
            data_lane="synthetic",
            step_index=0,
            path_length=1,
        )


def test_atom_delete_rejects_successor_that_keeps_deleted_slot_element() -> None:
    source = _state(["C", "O"], [(0, 1, BOND_SINGLE)])
    with pytest.raises(Active8SemanticCensusError, match="leaves a real element"):
        derive_semantic_observation(
            source,
            source,
            executor_rule="atom_delete",
            action=AtomDelete(v=1),
            data_lane="fixture",
            step_index=0,
            path_length=1,
        )


def test_contract_is_self_validated_externally_pinned_and_bound_to_v8(tmp_path: Path) -> None:
    contract = load_semantic_census_contract(CONTRACT)
    parent = validate_parent_t1_successor_gate(contract, path=PARENT_T1_GATE)
    assert (
        parent["active8_inventory_manifest_file_sha256"]
        == (contract["parent_active8_identity"]["inventory_manifest_file_sha256"])
    )

    mutated = dict(contract)
    mutated["sparse_observed_value_max_rows"] = 5
    body = dict(mutated)
    body.pop("contract_sha256")
    mutated["contract_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path = tmp_path / "rehashed-mutated-contract.json"
    path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
    with pytest.raises(Active8SemanticCensusError, match="physical SHA-256 mismatch"):
        load_semantic_census_contract(path)


def test_contract_self_hash_is_checked_after_physical_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(CONTRACT.read_text())
    payload["contract_sha256"] = "0" * 64
    path = tmp_path / "stale-self-hash.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(
        census_module,
        "PINNED_SEMANTIC_CENSUS_CONTRACT_FILE_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(Active8SemanticCensusError, match="canonical self-hash mismatch"):
        load_semantic_census_contract(path)


def test_cli_refuses_a_dirty_scientific_tree_including_untracked_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = ROOT / "scripts/build_editing_active8_semantic_census_v2.py"
    spec = importlib.util.spec_from_file_location("active8_semantic_census_cli_test", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    outputs = iter(("a" * 40 + "\n", "?? src/compose_v4/untracked_policy.py\n"))

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="clean scientific tree"):
        cli._git_revision()
