from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
    dependency_region_summary,
)
from compose_v4.control.generic_legal_action_policy import enumerate_rule_successors
from compose_v4.rewrite.trace_shard import encode_state


def _methane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[0] = 2
    hydrogens[0] = 4
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def _propane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[:3] = 2
    hydrogens[:3] = (3, 2, 3)
    bonds[0, 1] = bonds[1, 0] = 1
    bonds[1, 2] = bonds[2, 1] = 1
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def test_created_handle_dependency_forms_one_exact_region():
    source = _methane()
    insert = enumerate_rule_successors(source, "atom_insert")[0]
    created_slot = insert.action_record["payload"]["slot"]
    restate = next(
        row
        for row in enumerate_rule_successors(insert.successor, "atom_restate_semantic")
        if row.action_record["payload"]["v"] == created_slot
    )
    states = (
        encode_state(source),
        encode_state(insert.successor),
        encode_state(restate.successor),
    )
    actions = (insert.action_record, restate.action_record)
    result = dependency_region_program(states, actions)
    assert result["component_count"] == 1
    assert result["created_dependency_edges"]
    assert result["exact_replay"] is True
    assert result["contiguous_component_schedule"] is True


def test_region_budget_abstains_without_changing_exact_replay():
    source = _propane()
    first = next(
        row
        for row in enumerate_rule_successors(source, "atom_restate_semantic")
        if row.action_record["payload"]["v"] == 0
    )
    second = next(
        row
        for row in enumerate_rule_successors(first.successor, "atom_restate_semantic")
        if row.action_record["payload"]["v"] == 2
    )
    states = (
        encode_state(source),
        encode_state(first.successor),
        encode_state(second.successor),
    )
    actions = (first.action_record, second.action_record)
    result = dependency_region_program(
        states,
        actions,
        DependencyRegionConfig(
            runtime_maximum_components=1, join_lifetime_neighbors=False
        ),
    )
    assert result["component_count"] == 2
    assert result["exact_replay"] is True
    assert result["complete_representation_supported"] is False
    assert result["abstention_reason"] == "component_budget_exceeded"

    summary = dependency_region_summary([{"dependency_region_program": result}])
    assert summary["exact_replay_precision"] == 1.0
    assert summary["complete_representation_coverage_all_routes"] == 0.0
