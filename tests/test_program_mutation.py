"""Supported variable programs and dependent attachments, using the exact executor."""

import json

import numpy as np
import pytest

from compose_v4.control.edit_program import execute_bound_program, extract_program
from compose_v4.control.edit_program_graph import combine_bound_programs
from compose_v4.control.program_mutation import (
    branch_components,
    mutate_parameter,
    parameter_choices,
    replace_branch,
    select_branch,
)
from tests.test_edit_program import graph, ring_and_carbonyl


def test_segment_extension_changes_topology_size_and_preserves_dependent_carbonyl():
    source, stages = ring_and_carbonyl()
    original, binding = extract_program(source, stages)
    # Extend an internal chain segment; the downstream carbonyl still refers to
    # the originally selected created atom, not a stale renumbered handle.
    choices = parameter_choices(original, "extend_segment")
    variant = mutate_parameter(original, "extend_segment", choices[2])
    endpoint, receipt = execute_bound_program(source, variant, binding)
    old, _ = execute_bound_program(source, original, binding)
    assert endpoint.n_real_atoms == old.n_real_atoms + 1
    assert len(receipt["actions"]) == len(original.marks) + 1
    assert receipt["endpoint"] != stages[-1]["endpoint"]
    assert sum(r["executor_rule"] == "cycle_close" for r in receipt["actions"]) == 1
    assert receipt["actions"][-1]["payload"]["neighbors"][0][1] == 2


def test_contraction_never_deletes_an_atom_referenced_by_a_later_refinement():
    source, stages = ring_and_carbonyl()
    original, binding = extract_program(source, stages)
    carbonyl_anchor = json.loads(original.marks[-1])["payload"]["neighbors"][0][0]
    for choice in parameter_choices(original, "contract_segment"):
        assert json.loads(original.marks[choice[0]])["payload"]["slot"] != carbonyl_anchor
        variant = mutate_parameter(original, "contract_segment", choice)
        product, _ = execute_bound_program(source, variant, binding)
        assert product.n_real_atoms == 8


def test_branch_selection_rejects_dangling_created_reference():
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    assert branch_components(program) == ((0, 1),)
    with pytest.raises(ValueError, match="dependency-closed"):
        select_branch(program, (1,))


def test_replace_independent_branch_keeps_other_bound_construction():
    source, stages = ring_and_carbonyl()
    branch, _ = extract_program(source, stages)
    parent = graph("CCCCCC")
    joint, binding = combine_bound_programs(
        parent, ((branch, (0,)), (branch, (5,))), max_primitives=32, max_blocks=8
    )
    components = branch_components(joint)
    assert len(components) == 2
    donor = mutate_parameter(
        branch, "extend_segment", parameter_choices(branch, "extend_segment")[2]
    )
    replacement, anchors = replace_branch(
        parent, joint, binding, components[1], donor, (5,), max_primitives=32, max_blocks=8
    )
    product, _ = execute_bound_program(
        parent, replacement, anchors, max_primitives=32, max_blocks=8
    )
    old, _ = execute_bound_program(parent, joint, binding, max_primitives=32, max_blocks=8)
    assert product.n_real_atoms == old.n_real_atoms + 1
    assert np.array_equal(parent.atom_types, graph("CCCCCC").atom_types)
