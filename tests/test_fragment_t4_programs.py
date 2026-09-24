from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentPrompt, FragmentTask
from compose_v4.benchmark.fragment_program_adapter import (
    ProgramConstraint,
    admit_complete_program,
    compile_region,
)
from compose_v4.benchmark.fragment_t4_programs import (
    ConstrainedProgramAbstention,
    bind_complete_program,
    refine_with_t4,
)
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.dynamic_program_synthesis import synthesize_named_module_sequence
from compose_v4.control.dynamic_program_synthesis_v1 import (
    RingRequest,
    RingSpec,
    SubstitutedRingRequest,
    compile_substituted_ring,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.progressive_structured_sampler import (
    synthesize_anchored_replacement_program,
)
from compose_v4.control.ring_program import construction_branches


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


def test_native_t4_multi_module_dependency_program_rebinds_to_required_interface():
    source = graph("CC")
    _, program, _, _, _ = synthesize_named_module_sequence(
        source, np.random.default_rng(2), ("segment_grow", "segment_grow")
    )
    candidate = bind_complete_program(
        source, program, ProgramConstraint((0, 1), (0,), ((0, 1),)), np.random.default_rng(3)
    )
    assert candidate.provenance["program_blocks"] == 2
    assert len(candidate.trace["actions"]) > 1
    assert candidate.provenance["dependencies"] or candidate.provenance["conflicts"]
    assert candidate.trace["scheduling"].startswith("deterministic serial")
    assert candidate.provenance["selected_binding"] == [0]


def test_native_t4_whole_ring_is_bound_without_copying_constructor():
    source = graph("CC")
    spec = RingSpec("pendant", 6, (6, 0, 0), "aromatic")
    anchors, pattern = construction_branches(source, {1}, spec)[0]
    _, stage = compile_substituted_ring(
        source, SubstitutedRingRequest(RingRequest(spec, anchors, ("C",) * 6, pattern), ())
    )
    program, _ = extract_program(source, [stage])
    candidate = bind_complete_program(
        source, program, ProgramConstraint((0, 1), (0,), ((0, 1),)), np.random.default_rng(0)
    )
    mol = Chem.MolFromSmiles(candidate.smiles)
    assert mol.GetRingInfo().NumRings() == 1
    assert candidate.provenance["selected_binding"] == [0]


def test_native_anchored_deletion_is_not_allowed_to_destroy_locked_core():
    source = graph("CCOC")
    _, program, _, _, _ = synthesize_anchored_replacement_program(source, np.random.default_rng(1))
    with pytest.raises(ConstrainedProgramAbstention) as caught:
        bind_complete_program(
            source,
            program,
            ProgramConstraint((0, 1, 2, 3), (0,), ((0, 1),)),
            np.random.default_rng(0),
        )
    assert caught.value.receipt["program_blocks"] == 2
    assert caught.value.receipt["program_primitives"] > 1


@pytest.mark.parametrize("lane", ["shallow", "structured", "anchored_replacement"])
def test_actual_t4_lane_refines_generated_content_in_one_bounded_program(lane):
    context = build_prompt_context(
        FragmentPrompt("unit_fixture", "CCC", FragmentTask.MOTIF_EXTENSION, ("[1*]CC",))
    )
    site = context.attachment.interfaces[0]
    trace, provenance = compile_region(context.start_state, "[1*]CCOC", (site,))
    seed = admit_complete_program(
        context.start_state, trace, ProgramConstraint.from_context(context), provenance=provenance
    )
    accepted = []
    for index in range(8):
        try:
            candidate = refine_with_t4(context, seed, np.random.default_rng(index), lane)
        except ConstrainedProgramAbstention:
            continue
        assert candidate.provenance["t4_lane"] == lane
        assert candidate.smiles != seed.smiles
        assert len(candidate.trace["actions"]) <= 32
        assert 2 <= len(candidate.provenance["program_block_lengths"]) <= 8
        assert candidate.provenance["dependencies"] or candidate.provenance["conflicts"]
        accepted.append(candidate)
    assert accepted, f"native T4 {lane} never produced a legal fixture refinement"
