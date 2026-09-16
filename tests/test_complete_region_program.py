from __future__ import annotations

import json

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.complete_region_program import (
    CONTINUE,
    STOP,
    CompleteRegionDecision,
    CompleteRegionProgram,
    CreatedRoleReference,
    execute_complete_region_program,
)
from compose_v4.control.edit_program import atom_signature, environment
from compose_v4.control.structural_subgoal import StructuralSubgoal
from compose_v4.control.structural_subgoal_realizer import RealizerConfig
from compose_v4.rewrite.trace_shard import decode_state


def _methane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[0] = 2
    hydrogens[0] = 4
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def _append_carbon(graph: MolecularGraph, slot: int) -> StructuralSubgoal:
    target_degree = int(np.count_nonzero(graph.bonds[slot])) + 1
    return StructuralSubgoal(
        input_atoms=(atom_signature(graph, slot),),
        input_bonds=((0,),),
        environments=(environment(graph, slot),),
        target_atoms=((2, 0, int(graph.implicit_h_counts[slot]) - 1, target_degree),),
        output_atoms=((2, 0, 3, 1),),
        target_bonds=((0, 1), (1, 0)),
    )


def _dependent_two_region_program() -> tuple[MolecularGraph, CompleteRegionProgram]:
    source = _methane()
    first_patch = _append_carbon(source, 0)
    first_only = CompleteRegionProgram(
        (
            CompleteRegionDecision(
                first_patch,
                (None,),
                STOP,
            ),
        )
    )
    first_result = execute_complete_region_program(source, first_only)
    assert first_result["status"] == "committed"
    intermediate = decode_state(first_result["committed_endpoint_state"])
    created_slots = [
        slot
        for slot in range(source.n_atoms)
        if source.atom_types[slot] == 0 and intermediate.atom_types[slot] != 0
    ]
    assert created_slots == [1]
    second_patch = _append_carbon(intermediate, created_slots[0])
    return source, CompleteRegionProgram(
        (
            CompleteRegionDecision(first_patch, (None,), CONTINUE),
            CompleteRegionDecision(
                second_patch,
                (CreatedRoleReference(0, 0),),
                STOP,
            ),
        )
    )


def test_complete_region_program_preserves_created_role_provenance() -> None:
    source, program = _dependent_two_region_program()

    result = execute_complete_region_program(source, program)

    assert result["status"] == "committed"
    assert result["execution_mode"] == "sequential_created_role_dependencies"
    assert result["regions_completed_in_protected_state"] == 2
    assert result["stop_reached"] is True
    assert result["committed_endpoint_count"] == 1
    assert result["partial_endpoint_evaluations"] == 0
    assert result["partial_endpoint_locks"] == 0
    assert result["primitive_teacher_actions_used"] == 0
    endpoint = decode_state(result["committed_endpoint_state"])
    assert endpoint.n_real_atoms == 3

    serialized = json.dumps(program.payload(), sort_keys=True)
    assert "relative_created_role_reference_v1" in serialized
    for forbidden in ("executor_rule", "actions", "route_id", "target_id"):
        assert forbidden not in serialized
    assert CompleteRegionProgram.from_payload(program.payload()) == program


def test_protected_program_abort_never_commits_partial_endpoint() -> None:
    source, program = _dependent_two_region_program()

    result = execute_complete_region_program(
        source,
        program,
        config=RealizerConfig(maximum_primitives=1, maximum_expansions=16),
    )

    assert result["status"] == "total_primitive_budget_abstention"
    assert result["regions_completed_in_protected_state"] == 1
    assert result["stop_reached"] is False
    assert result["committed_endpoint_count"] == 0
    assert result["committed_endpoint_state"] is None
    assert result["realized_actions"] == []
    assert result["partial_endpoint_evaluations"] == 0
    assert result["partial_endpoint_locks"] == 0


def test_complete_region_program_rejects_early_stop() -> None:
    source = _methane()
    patch = _append_carbon(source, 0)
    with pytest.raises(ValueError, match="CONTINUE/STOP"):
        CompleteRegionProgram(
            (
                CompleteRegionDecision(patch, (None,), STOP),
                CompleteRegionDecision(patch, (None,), STOP),
            )
        )
