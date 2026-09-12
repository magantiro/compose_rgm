"""Multi-site coordination uses serial valid states, not parallel graph writes."""

import json
from pathlib import Path

import pytest
from test_edit_program import graph, ring_and_carbonyl

from compose_v4.control.edit_program import ProgramExecutionError, extract_program
from compose_v4.control.edit_program_graph import (
    combine_bound_programs,
    compile_program_graph,
    execute_program_graph,
    scheduled_program,
)
from compose_v4.control.macro_engine import ELEMENT_CODE
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.operators import AtomInsert, CycleOpenEdge
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def add_chain(source, anchor, length=1):
    marks = []
    start = source.n_real_atoms
    for offset in range(length):
        slot = start + offset
        marks.append(
            encode_action(
                "atom_insert",
                AtomInsert(
                    slot,
                    ELEMENT_CODE["C"],
                    0,
                    3,
                    ((anchor if offset == 0 else slot - 1, 1),),
                ),
            )
        )
    _, stage = execute_program(source, marks)
    return extract_program(source, [{"name": "chain", **stage}])


def test_disconnected_sites_are_one_completed_proposal_and_both_orders_work():
    source = graph("CCCCCC")
    left = add_chain(source, 0)
    right = add_chain(source, 5)
    program, assignment = combine_bound_programs(source, (left, right))
    dag = compile_program_graph(program)
    assert dag.dependencies == dag.conflicts == ()
    before = encode_state(source)
    _, first = execute_program_graph(source, dag, assignment)
    _, second = execute_program_graph(source, dag, assignment, priority=(1, 0))
    assert first["endpoint"] == second["endpoint"]
    assert first["block_order"] == (0, 1) and second["block_order"] == (1, 0)
    assert first["primitive_edits"] == 2 and first["complete"]
    assert encode_state(source) == before


def test_created_ring_dependency_is_not_permanently_excluded_as_a_conflict():
    source, stages = ring_and_carbonyl()
    program, assignment = extract_program(source, stages)
    dag = compile_program_graph(program)
    assert dag.dependencies == ((0, 1),)
    assert (0, 1) in dag.conflicts
    _, receipt = execute_program_graph(source, dag, assignment, priority=(1, 0))
    assert receipt["block_order"] == (0, 1)
    assert receipt["endpoint"] == stages[-1]["endpoint"]
    with pytest.raises(ValueError, match="cycle"):
        compile_program_graph(program, dependencies=((1, 0),))


def test_pairwise_disjoint_cycle_openings_still_revalidate_connectivity():
    source = graph("C1CCCCC1")
    parts = []
    for a, b in ((0, 1), (3, 4)):
        record = encode_action("cycle_open", CycleOpenEdge(a, b))
        _, stage = execute_program(source, [record])
        parts.append(extract_program(source, [stage]))
    program, assignment = combine_bound_programs(source, tuple(parts))
    dag = compile_program_graph(program)
    assert dag.conflicts == ()  # Explicit footprints are not a safety certificate.
    with pytest.raises(ProgramExecutionError) as caught:
        execute_program_graph(source, dag, assignment)
    assert caught.value.receipt["complete"] is False
    assert len(caught.value.receipt["actions"]) == 1
    assert "endpoint" not in caught.value.receipt


def test_triple_resource_overflow_is_rejected_even_when_every_pair_fits():
    source = graph("C" * 34)
    parts = tuple(add_chain(source, i, 3) for i in (0, 16, 33))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        program, _ = combine_bound_programs(source, (parts[a], parts[b]))
        _, _, timeline = scheduled_program(compile_program_graph(program), 34)
        assert timeline[-1]["end"] == 40
    program, _ = combine_bound_programs(source, parts)
    with pytest.raises(ValueError, match="peak-capacity"):
        scheduled_program(compile_program_graph(program), 34)


def test_real_route_records_dependencies_without_assuming_branch_independence():
    path = Path("diagnostics/t4_whole_ring_plan/result.json")
    if not path.exists():
        pytest.skip("documented saved development route absent")
    saved = json.loads(path.read_text())
    source = decode_state(saved["source_state"])
    program, assignment = extract_program(source, saved["attempts"][0]["stages"])
    dag = compile_program_graph(program)
    _, original = execute_program_graph(source, dag, assignment)
    _, preferred_core = execute_program_graph(source, dag, assignment, priority=(4, 0, 1, 2, 3))
    assert original["endpoint"] == preferred_core["endpoint"] == saved["target"]
    assert original["primitive_edits"] == preferred_core["primitive_edits"] == 21
