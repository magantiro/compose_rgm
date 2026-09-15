from __future__ import annotations

import json

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.generic_legal_action_policy import enumerate_rule_successors
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    attachment_bindings,
    extract_structural_goal,
    instantiate_goal,
)
from compose_v4.control.structural_subgoal_realizer import (
    RealizerConfig,
    realize_structural_goal,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def _methane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[0] = 2
    hydrogens[0] = 4
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def _created_handle_trace():
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
    return (
        source,
        restate.successor,
        states,
        (
            insert.action_record,
            restate.action_record,
        ),
    )


def test_structural_goal_omits_teacher_primitive_trace_and_reconstructs_endpoint():
    source, target, states, actions = _created_handle_trace()
    goal, teacher_bindings, regions = extract_structural_goal(states, actions)

    assert regions["component_count"] == 1
    serialized = json.dumps(goal.payload(), sort_keys=True)
    for forbidden in ("executor_rule", "primitive_indices", "marks", "actions"):
        assert forbidden not in serialized
    assert StructuralGoal.from_payload(goal.payload()) == goal

    product, receipt = instantiate_goal(source, goal, teacher_bindings)
    assert canonical_state_key(product) == canonical_state_key(target)
    assert receipt["primitive_teacher_actions_used"] == 0


def test_address_free_binding_enumeration_contains_teacher_assignment():
    source, _, states, actions = _created_handle_trace()
    goal, teacher_bindings, _ = extract_structural_goal(states, actions)

    census = attachment_bindings(goal.subgoals[0], source)
    assert teacher_bindings[0] in census.assignments
    assert census.visits > 0


def test_conditional_realizer_reaches_bound_goal_without_teacher_actions():
    source, target, states, actions = _created_handle_trace()
    goal, teacher_bindings, _ = extract_structural_goal(states, actions)
    progress = []

    result = realize_structural_goal(
        source,
        goal,
        teacher_bindings,
        config=RealizerConfig(maximum_expansions=64, children_per_expansion=8),
        progress=progress.append,
    )

    assert result["status"] == "realized"
    assert result["endpoint_matches_bound_target"]
    assert result["primitive_teacher_actions_used"] == 0
    assert canonical_state_key(decode_state(result["states"][-1])) == canonical_state_key(target)
    assert progress


def test_conditional_realizer_reports_bounded_abstention():
    source, _, states, actions = _created_handle_trace()
    goal, teacher_bindings, _ = extract_structural_goal(states, actions)

    result = realize_structural_goal(
        source,
        goal,
        teacher_bindings,
        config=RealizerConfig(maximum_primitives=1, maximum_expansions=1),
    )

    assert result["status"].endswith("abstention")
    assert not result["endpoint_matches_bound_target"]
    assert result["actions"] == []
