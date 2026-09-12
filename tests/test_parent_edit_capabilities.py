"""Focused invariants for the explicit new proposal recipe, no task oracle."""

from dataclasses import replace

import numpy as np
import pytest

from compose_v4.control.current_state_edits import current_state_program
from compose_v4.control.edit_program import ProgramBlock
from compose_v4.control.edit_program_graph import (
    combine_bound_programs,
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.program_decomposition import verified_branches
from compose_v4.control.program_mutation import branch_components, replace_branch
from compose_v4.rewrite.trace_shard import encode_state
from tests.test_adaptive_program_optimizer import optimizer
from tests.test_edit_program import graph
from tests.test_edit_program_graph import add_chain


def test_opaque_route_donates_one_verified_branch_without_replacing_the_other():
    source = graph("CCCCCCO")
    joint, binding = combine_bound_programs(source, (add_chain(source, 0), add_chain(source, 6)))
    opaque = replace(
        joint, blocks=(ProgramBlock("compiled_complete_transformation", len(joint.marks)),)
    )
    assert len(branch_components(opaque)) == 1
    decomposed, anchors, report = verified_branches(source, opaque, binding)
    assert report["status"] == "verified" and len(branch_components(decomposed)) == 2
    donor, donor_binding = add_chain(source, 0, length=2)
    changed, assigned = replace_branch(
        source,
        decomposed,
        anchors,
        branch_components(decomposed)[0],
        donor,
        donor_binding,
        max_primitives=8,
        max_blocks=4,
    )
    _, receipt = execute_program_graph(source, compile_program_graph(changed), assigned)
    assert receipt["endpoint"] != report["reference_endpoint"]
    # The other branch still adds the oxygen-bound methyl group.
    assert "OC" in receipt["endpoint"] or "CO" in receipt["endpoint"]
    assert len(receipt["actions"]) == 3


def test_fresh_proposal_can_continue_a_constructor_that_fills_its_entire_horizon():
    search = optimizer()
    entry = next(iter(search.entries.values()))
    n = len(entry["program"]["marks"])

    class ExactTail:
        def complete_reference_program(self, node, rng, *, max_options):
            self.budget = node.budget
            program, binding = add_chain(node.graph, 0)
            return program, binding, {"primitive_edits": 1, "complete_options": 1}

    # A tiny real-executor callback makes the horizon boundary deterministic.
    search.hierarchy = ExactTail()
    search.config = replace(search.config, max_primitives=n)
    with pytest.raises(ValueError, match="no remaining work"):
        search._broad(entry)
    search.config = replace(search.config, continuation_root="exact_current_state")
    source, tail, _binding, metadata = search._broad(entry)
    assert encode_state(source) == entry["trace"]["states"][-1]
    assert search.hierarchy.budget == n and len(tail.marks) == 1
    assert metadata["ancestral_primitive_edits"] == n
    assert metadata["original_seed_state"] == entry["source_state"]
    assert metadata["construction_ancestry"][0]["entry_id"] == entry["entry_id"]
    assert len(search.observations) == 1


def test_explicit_edit_modifies_existing_input_without_an_insertion():
    source = graph("CCCC")
    program, binding, detail = current_state_program(
        source, np.random.default_rng(4), family="atom_restate_semantic"
    )
    product, trace = execute_program_graph(source, compile_program_graph(program), binding)
    assert product.n_real_atoms == source.n_real_atoms
    assert trace["actual_changes"]["changed_original_slots"]
    assert trace["actual_changes"]["surviving_new_atoms"] == 0
    assert detail["enumerated_actions"] > 0


def test_neighbor_writes_keep_adjacent_deletions_in_one_reusable_branch():
    from compose_v4.control.edit_program import extract_program
    from compose_v4.control.macro_engine import ELEMENT_CODE
    from compose_v4.experiments.whole_ring_plan import execute_program
    from compose_v4.rewrite.action_codec_v4 import encode_action
    from compose_v4.rewrite.operators import AtomDelete, AtomInsert

    source = graph("CCCCCCO")
    actions = [encode_action("atom_delete", AtomDelete(i)) for i in (0, 1)]
    actions.append(encode_action("atom_insert", AtomInsert(7, ELEMENT_CODE["C"], 0, 3, ((6, 1),))))
    _, trace = execute_program(source, actions)
    program, binding = extract_program(source, [{"name": "opaque", **trace}])
    decomposed, _, report = verified_branches(source, program, binding)
    assert report["status"] == "verified"
    assert report["primitive_components"] == [[0, 1], [2]]
    assert len(decomposed.blocks) == 2


def test_maximize_mode_does_not_invert_pmo_parent_quality():
    search = optimizer()
    from tests.test_adaptive_program_optimizer import eligible, outcomes

    batch = search.propose_batch(eligible)
    search.observe_batch(batch["batch_id"], outcomes(search, batch, -5))
    search.duplicate_counts.clear()
    keys, minimizing = search.selection()
    search.config = replace(search.config, score_direction="maximize")
    _, maximizing = search.selection()
    original = next(
        k
        for k in keys
        if search.entries[k]["static_score"] == -1
        and search.entries[k]["endpoint"] == next(iter(search.observations.values()))["endpoint"]
    )
    assert maximizing[keys.index(original)] > minimizing[keys.index(original)]
