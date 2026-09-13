"""Executable program dependencies, transfer and failure, without network/oracles."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.edit_program import (
    EditProgram,
    ProgramExecutionError,
    attachment_bindings,
    execute_bound_program,
    extract_program,
)
from compose_v4.control.macro_engine import ELEMENT_CODE
from compose_v4.control.ring_program import RingSpec
from compose_v4.experiments.whole_ring_plan import RingRequest, compile_ring, execute_program
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.operators import AtomDelete, AtomInsert
from compose_v4.rewrite.trace_shard import encode_state


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


def ring_and_carbonyl():
    source = graph("CC")
    middle, ring = compile_ring(
        source,
        RingRequest(RingSpec("pendant", 6, (6, 0, 0), "saturated"), (1,), ("C",) * 6, (1,) * 6),
    )
    action = AtomInsert(8, ELEMENT_CODE["O"], 0, 0, ((ring["progress"]["path"][3], 2),))
    _, carbonyl = execute_program(middle, [encode_action("atom_insert", action)])
    stages = [{"name": "ring", **ring}, {"name": "carbonyl", **carbonyl}]
    return source, stages


def test_complete_program_binds_created_atom_across_options_and_roundtrips():
    source, stages = ring_and_carbonyl()
    program, assignment = extract_program(source, stages)
    clone = EditProgram.from_payload(json.loads(json.dumps(program.payload())))
    assert clone == program
    assert clone.program_id == program.program_id
    assert stages[-1]["endpoint"] not in json.dumps(program.payload())
    assert json.loads(program.marks[-1])["payload"]["neighbors"][0][0] == {"created": 3}
    _, receipt = execute_bound_program(source, clone, assignment)
    assert receipt["endpoint"] == stages[-1]["endpoint"]
    assert receipt["states"] == stages[0]["states"] + stages[1]["states"][1:]
    assert receipt["primitive_edits"] == 8


def test_transfer_to_different_parent_and_relabeling_preserve_program_semantics():
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    other = graph("CCCO")
    census = attachment_bindings(program, other)
    assert census.assignments and census.visits > 0
    _, receipt = execute_bound_program(other, program, (0,))
    assert receipt["endpoint"] != stages[-1]["endpoint"]
    perm = np.roll(np.arange(48), 7)
    inverse = np.argsort(perm)
    relabeled = deepcopy(other)
    for field in ("atom_types", "formal_charges", "implicit_h_counts"):
        setattr(relabeled, field, getattr(other, field)[perm].copy())
    relabeled.bonds = other.bonds[np.ix_(perm, perm)].copy()
    _, moved = execute_bound_program(relabeled, program, (int(inverse[0]),))
    assert moved["endpoint"] == receipt["endpoint"]
    assert encode_state(source) == stages[0]["states"][0]


def test_input_slot_reuse_is_a_new_identity_not_stale_reference():
    source = graph("CC")
    actions = [
        encode_action("atom_delete", AtomDelete(1)),
        encode_action("atom_insert", AtomInsert(1, ELEMENT_CODE["O"], 0, 1, ((0, 1),))),
        encode_action("atom_delete", AtomDelete(1)),
        encode_action("atom_insert", AtomInsert(1, ELEMENT_CODE["N"], 0, 2, ((0, 1),))),
    ]
    _, stage = execute_program(source, actions)
    program, assignment = extract_program(source, [stage])
    assert json.loads(program.marks[1])["payload"]["slot"] == {"created": 0}
    assert json.loads(program.marks[3])["payload"]["slot"] == {"created": 1}
    assert execute_bound_program(source, program, assignment)[1]["endpoint"] == stage["endpoint"]
    damaged = program.payload()
    record = json.loads(damaged["marks"][3])
    record["payload"]["neighbors"] = [[{"created": 0}, 1]]
    damaged["marks"] = (*damaged["marks"][:3], json.dumps(record))
    with pytest.raises(ValueError, match="unbound, deleted"):
        EditProgram.from_payload(damaged)


def test_invalid_or_overbudget_program_never_returns_partial_candidate():
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    before = encode_state(source)
    with pytest.raises(ValueError, match="budget"):
        execute_bound_program(source, program, (1,), max_primitives=7)
    with pytest.raises(ValueError, match="preserved context"):
        execute_bound_program(source, program, (1,), mutable_slots=frozenset({0}))
    saturated = graph("CC(C)(C)C")
    with pytest.raises(ProgramExecutionError) as failed:
        execute_bound_program(saturated, program, (1,))
    assert failed.value.receipt["complete"] is False
    assert "endpoint" not in failed.value.receipt
    assert encode_state(source) == before


def test_capped_binding_is_reported_and_saved_state_corruption_is_rejected():
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    census = attachment_bindings(program, source, max_bindings=1)
    assert len(census.assignments) == 1 and census.truncated
    bad = deepcopy(stages)
    bad[-1]["states"][-1] = encode_state(source)
    with pytest.raises(ValueError, match="exact executor replay"):
        extract_program(source, bad)


def test_binding_masks_enforce_zero_bonds_injectivity_order_and_visit_caps():
    from itertools import permutations

    source = graph("CCC")
    actions = [encode_action("atom_delete", AtomDelete(i)) for i in (0, 2)]
    _, stage = execute_program(source, actions)
    program, _ = extract_program(source, [stage])
    target = graph("CCCC")
    expected = tuple((a, b) for a, b in permutations(range(4), 2) if target.bonds[a, b] == 0)
    census = attachment_bindings(program, target, contextual=False)
    assert census.assignments == expected
    assert census.visits == 11 and not census.truncated
    capped = attachment_bindings(program, target, contextual=False, max_bindings=2)
    assert capped.assignments == expected[:2] and capped.visits == 4 and capped.truncated
    limited = attachment_bindings(program, target, contextual=False, max_visits=4)
    assert limited == capped
    masked = attachment_bindings(program, target, contextual=False, mutable_slots=frozenset({0, 3}))
    assert masked.assignments == ((0, 3), (3, 0))


def test_real_five_stage_plan_rebinds_without_endpoint_at_inference():
    path = Path("diagnostics/t4_whole_ring_plan/result.json")
    if not path.exists():
        pytest.skip("documented saved development plan absent")
    from compose_v4.rewrite.trace_shard import decode_state

    result = json.loads(path.read_text())
    source = decode_state(result["source_state"])
    stages = result["attempts"][0]["stages"]
    program, assignment = extract_program(source, stages)
    _, receipt = execute_bound_program(source, program, assignment)
    assert receipt["endpoint"] == result["target"]
    assert receipt["primitive_edits"] == 21
    assert len(program.blocks) == 5
