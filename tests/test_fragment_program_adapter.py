from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.benchmark.fragment_program_adapter import (
    CompleteProgram,
    ProgramConstraint,
    admit_complete_program,
    compile_region,
    native_scoring_mark,
    sample_boundary_content,
    select_learned_program,
    select_learned_program_novelty4,
    validate_native_compound_membership,
)
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.model.factorized_tracelet_rate_model import (
    LEGACY_ATOM_RESTATE_ACTION_SEMANTICS,
    LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS,
    LEGACY_CYCLE_OPEN_ACTION_SEMANTICS,
    LEGACY_EDITING_PROCESS_SEMANTICS,
)
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.trace_shard import decode_state
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate


def source(smiles="CC"):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


def test_joint_capacity_reserves_observed_minimum_not_one_atom_per_later_site():
    entries = tuple(
        {"contexts": [key], "heavy_atoms": size, "occurrences": 1}
        for key, sizes in (("first", (2, 20)), ("later", (3, 20)))
        for size in sizes
    )
    keys = ["first"] + ["later"] * 5
    for seed in range(20):
        selected = sample_boundary_content(keys, entries, 26, np.random.default_rng(seed))
        assert len(selected) == 6
        assert sum(e["heavy_atoms"] for e in selected) <= 26
        assert selected[0]["heavy_atoms"] == 2
    with pytest.raises(ValueError, match="minima exceed"):
        sample_boundary_content(keys, entries, 16, np.random.default_rng(0))
    with pytest.raises(ValueError, match="required boundary context"):
        sample_boundary_content(["missing"], entries, 26, np.random.default_rng(0))


def test_compound_model_support_absence_is_an_explicit_abstention():
    action = RingSystemRestate((BondOrderChange(0, 1, 2),))
    with pytest.raises(ValueError, match="frozen model action fiber"):
        validate_native_compound_membership(
            SimpleNamespace(
                teacher_actions=(action,),
                ring_restate_actions=((),),
            )
        )
    validate_native_compound_membership(
        SimpleNamespace(
            teacher_actions=(action,),
            ring_restate_actions=((action,),),
        )
    )


@pytest.mark.parametrize(
    "region",
    [
        "[1*]CC(C)O",
        "[1*]c1ccccc1",
        "[1*]C1CNCCN1",
        "[1*]c1ccc2ccccc2c1",
        "[1*]CSC",
        "[1*]CC#N",
    ],
)
def test_complete_observed_region_uses_shared_compiler_and_replay(region):
    graph = source()
    trace, provenance = compile_region(graph, region, (0,))
    candidate = admit_complete_program(
        graph, trace, ProgramConstraint((0, 1), (0,), ((0, 1),)), provenance=provenance
    )
    assert len(candidate.trace["actions"]) > 1
    assert candidate.provenance["compiler_strategy"] == "deterministic_graph_delta_schedule"
    assert Chem.MolFromSmiles(candidate.smiles) is not None
    assert int(candidate.endpoint.bonds[0, 1]) == 1
    assert all(
        np.array_equal(decode_state(s).atom_types[:2], graph.atom_types[:2])
        for s in candidate.trace["states"]
    )


def test_two_boundaries_planned_together_but_not_claimed_as_official_linker():
    graph = source()
    trace, provenance = compile_region(graph, "[1*]CC[2*]", (0, 1))
    assert provenance["boundary_arity"] == 2
    endpoint = decode_state(trace["states"][-1])
    assert endpoint.n_real_atoms == 4
    assert any(a["executor_rule"] == "cycle_close" for a in trace["actions"])
    # Closing a path through one locked core is forbidden even though both
    # endpoints are declared interfaces. This is not a valid decoration.
    with pytest.raises(ValueError, match="immutable core"):
        admit_complete_program(
            graph,
            trace,
            ProgramConstraint((0, 1), (0, 1), ((0, 1), (1, 1))),
            provenance=provenance,
        )


def test_undeclared_interface_and_incomplete_program_fail_closed():
    graph = source()
    trace, provenance = compile_region(graph, "[1*]CCO", (0,))
    with pytest.raises(ValueError, match="immutable core"):
        admit_complete_program(
            graph, trace, ProgramConstraint((0, 1), (1,), ((1, 1),)), provenance=provenance
        )
    with pytest.raises(ValueError, match="unsatisfied"):
        admit_complete_program(
            graph,
            trace,
            ProgramConstraint((0, 1), (0, 1), ((0, 1), (1, 1))),
            provenance=provenance,
        )


def test_model_score_controls_selection_and_no_uniform_unsupported_fallback():
    first = CompleteProgram(source("CCC"), {}, {})
    second = CompleteProgram(source("CCO"), {}, {})
    picked, receipt = select_learned_program(
        (first, second), (-1000.0, 0.0), np.random.default_rng(0)
    )
    assert picked is second
    assert receipt["model_used"] is True
    with pytest.raises(ValueError, match="no model-supported"):
        select_learned_program((first,), (-np.inf,), np.random.default_rng(0))
    # Do not turn repeated representations into extra proposal mass.
    _, receipt = select_learned_program(
        (first, first, second), (0.0, 0.0, 0.0), np.random.default_rng(0)
    )
    assert receipt["unique_model_supported_endpoints"] == 2


def test_novelty4_keeps_positive_support_and_records_native_score():
    first = CompleteProgram(source("CCC"), {}, {})
    second = CompleteProgram(source("CCO"), {}, {})

    class CaptureRng:
        def choice(self, count, *, p):
            assert count == 2
            assert np.allclose(p, (0.2, 0.8))
            return 1

    selected, receipt = select_learned_program_novelty4(
        (first, second), (0.0, 0.0), CaptureRng(), frozenset((first.smiles,))
    )
    assert selected is second
    assert receipt["selected_probability"] == pytest.approx(0.8)
    assert receipt["mean_native_log_mark_probability"] == 0.0
    assert receipt["prior_unique_emissions"] == 1
    assert receipt["selected_already_emitted"] is False
    assert receipt["novelty_multiplier"] == 4.0


def test_novelty4_does_not_replace_duplicate_when_no_new_offer_exists():
    first = CompleteProgram(source("CCC"), {}, {})
    selected, receipt = select_learned_program_novelty4(
        (first,), (0.0,), np.random.default_rng(0), frozenset((first.smiles,))
    )
    assert selected is first
    assert receipt["selected_probability"] == 1.0
    assert receipt["selected_already_emitted"] is True
    with pytest.raises(ValueError, match="prior emitted endpoints"):
        select_learned_program_novelty4(
            (first,), (0.0,), np.random.default_rng(0), frozenset(("",))
        )


@pytest.mark.parametrize("region", ["[1*]c1ccccc1", "[1*]c1ccc2ccccc2c1"])
def test_semantic_ring_program_is_scored_only_via_exact_native_successors(region):
    model = SimpleNamespace(
        editing_process_semantics=LEGACY_EDITING_PROCESS_SEMANTICS,
        cycle_close_action_semantics=LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS,
        cycle_open_action_semantics=LEGACY_CYCLE_OPEN_ACTION_SEMANTICS,
        atom_restate_action_semantics=LEGACY_ATOM_RESTATE_ACTION_SEMANTICS,
    )
    graph = source()
    trace, _ = compile_region(graph, region, (0,))
    found_alias = False
    for index, record in enumerate(trace["actions"]):
        rule, action = decode_action(record)
        before, after = map(decode_state, trace["states"][index : index + 2])
        native_rule, _ = native_scoring_mark(model, before, after, rule, action)
        if rule == "cycle_close":
            assert native_rule == "bond_insert"
            found_alias = True
            with pytest.raises(ValueError, match="exact program successor"):
                native_scoring_mark(model, before, before, rule, action)
    assert found_alias


def test_actual_t4_substituted_ring_constructor_is_accepted_without_copying_it():
    from compose_v4.control.dynamic_program_synthesis_v1 import (
        RingRequest,
        RingSpec,
        SubstitutedRingRequest,
        compile_substituted_ring,
    )
    from compose_v4.control.ring_program import construction_branches

    graph = source()
    spec = RingSpec("pendant", 6, (6, 0, 0), "aromatic")
    anchors, pattern = next(
        b for b in construction_branches(graph, frozenset((0,)), spec) if b[0] == (0,)
    )
    _, stage = compile_substituted_ring(
        graph, SubstitutedRingRequest(RingRequest(spec, anchors, ("C",) * 6, pattern), ())
    )
    candidate = admit_complete_program(
        graph,
        stage,
        ProgramConstraint((0, 1), (0,), ((0, 1),)),
        provenance={"lane": "existing_t4_construct_substituted_ring"},
    )
    mol = Chem.MolFromSmiles(candidate.smiles)
    assert mol.GetRingInfo().NumRings() == 1
    assert len(candidate.trace["actions"]) == 7
