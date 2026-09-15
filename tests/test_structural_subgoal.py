from __future__ import annotations

import json

import numpy as np

from compose_v4.chem.molecular_graph import NULL_IDX, MolecularGraph
from compose_v4.control.edit_program import atom_signature, environment
from compose_v4.control.generic_legal_action_policy import enumerate_rule_successors
from compose_v4.control.structural_subgoal import (
    StructuralGoal,
    StructuralSubgoal,
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
    assert result["primitive_teacher_actions_used"] == 0
    assert result["output_role_slots"] == []
    assert result["logical_role_automorphisms"] >= 1
    assert result["subgoal_targets_match_within_complete_goal"] == [False]


def test_realizer_target_prefers_slots_empty_before_source_deletions():
    source, _, states, actions = _created_handle_trace()
    goal, teacher_bindings, _ = extract_structural_goal(states, actions)

    result = realize_structural_goal(
        source,
        goal,
        teacher_bindings,
        config=RealizerConfig(maximum_expansions=64, children_per_expansion=8),
    )

    receipt = result["bound_target_receipt"]
    initially_empty = {int(slot) for slot in np.flatnonzero(source.atom_types == NULL_IDX)}
    assert receipt["output_slot_policy"] == "initially_empty_then_freed"
    assert set(receipt["output_slots"][0]) <= initially_empty


def test_target_scheduler_does_not_fix_output_roles_to_creation_order():
    source = _methane()
    subgoal = StructuralSubgoal(
        input_atoms=(atom_signature(source, 0),),
        input_bonds=((0,),),
        environments=(environment(source, 0),),
        target_atoms=((2, 0, 3, 1),),
        # The terminal carbon deliberately precedes the boundary carbon. A
        # slot-fixed compiler cannot create role 0 into the first legal empty
        # slot because its required neighbor does not exist yet.
        output_atoms=((2, 0, 3, 1), (2, 0, 2, 2)),
        target_bonds=((0, 0, 1), (0, 0, 1), (1, 1, 0)),
    )

    result = realize_structural_goal(
        source,
        StructuralGoal((subgoal,)),
        ((0,),),
        config=RealizerConfig(maximum_expansions=32, children_per_expansion=8),
    )

    assert result["status"] == "realized"
    assert result["endpoint_matches_bound_target"]
    assert result["subgoal_targets_match_within_complete_goal"] == [True]
    inserted_slots = [
        action["payload"]["slot"]
        for action in result["actions"]
        if action["executor_rule"] == "atom_insert"
    ]
    assert result["compiler_strategy"] == "deterministic_graph_delta_schedule"
    assert result["output_role_slots"] == [1, 2]
    assert inserted_slots == [2, 1]
    assert result["primitive_teacher_actions_used"] == 0


def test_role_aware_realizer_quotients_automorphic_output_roles():
    source = _methane()
    subgoal = StructuralSubgoal(
        input_atoms=(atom_signature(source, 0),),
        input_bonds=((0,),),
        environments=(environment(source, 0),),
        target_atoms=((2, 0, 2, 2),),
        output_atoms=((2, 0, 3, 1), (2, 0, 3, 1)),
        target_bonds=((0, 1, 1), (1, 0, 0), (1, 0, 0)),
    )

    result = realize_structural_goal(
        source,
        StructuralGoal((subgoal,)),
        ((0,),),
        config=RealizerConfig(maximum_expansions=16, children_per_expansion=8),
    )

    assert result["status"] == "realized"
    assert result["endpoint_matches_bound_target"]
    assert result["logical_role_automorphisms"] == 2
    assert result["output_role_slots"] == [1, 2]
    assert result["expanded"] == 2


def test_structural_goal_rebinds_after_persistent_slot_permutation():
    source = _methane()
    subgoal = StructuralSubgoal(
        input_atoms=(atom_signature(source, 0),),
        input_bonds=((0,),),
        environments=(environment(source, 0),),
        target_atoms=((2, 0, 3, 1),),
        output_atoms=((2, 0, 3, 1),),
        target_bonds=((0, 1), (1, 0)),
    )
    goal = StructuralGoal((subgoal,))
    original = realize_structural_goal(source, goal, ((0,),))

    permutation = np.arange(source.n_atoms)
    permutation[[0, 7]] = permutation[[7, 0]]
    permuted = MolecularGraph(
        source.atom_types[permutation],
        source.formal_charges[permutation],
        source.implicit_h_counts[permutation],
        source.bonds[np.ix_(permutation, permutation)],
    )
    rebound = attachment_bindings(subgoal, permuted)
    assert rebound.assignments == ((7,),)
    moved = realize_structural_goal(permuted, goal, (rebound.assignments[0],))

    assert original["status"] == moved["status"] == "realized"
    assert original["primitive_teacher_actions_used"] == 0
    assert moved["primitive_teacher_actions_used"] == 0
    assert canonical_state_key(decode_state(original["states"][-1])) == canonical_state_key(
        decode_state(moved["states"][-1])
    )


def test_role_aware_cycle_closure_normalizes_nonmonotone_physical_slots():
    source = _methane()
    subgoal = StructuralSubgoal(
        input_atoms=(atom_signature(source, 0),),
        input_bonds=((0,),),
        environments=(environment(source, 0),),
        target_atoms=((2, 0, 2, 2),),
        output_atoms=((3, 0, 1, 2), (2, 0, 2, 2)),
        target_bonds=((0, 1, 1), (1, 0, 1), (1, 1, 0)),
    )

    result = realize_structural_goal(
        source,
        StructuralGoal((subgoal,)),
        ((0,),),
        config=RealizerConfig(maximum_expansions=32, children_per_expansion=8),
    )

    assert result["status"] == "realized"
    assert result["subgoal_targets_match_within_complete_goal"] == [True]
    cycle_closures = [
        action for action in result["actions"] if action["executor_rule"] == "cycle_close"
    ]
    assert cycle_closures
    assert all(action["payload"]["a"] < action["payload"]["b"] for action in cycle_closures)
