from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.dynamic_program_synthesis_v1 import (
    FunctionalizeRequest,
    PendantDeleteRequest,
    RingPathRemodelRequest,
    RingSubstituentAtom,
    SubstitutedRingRequest,
    compile_forced_module_sequence,
    compile_ring_path_remodel,
    enumerate_functionalizations,
    enumerate_pendant_deletions,
    enumerate_ring_path_remodels,
    initial_dynamic_program_batch_v1,
)
from compose_v4.control.ring_program import RingSpec
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.t4_dynamic_v1 import (
    SELECTED_UNITS,
    load_contract,
    v1_contract_payload,
)
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.whole_ring_plan import RingRequest
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def _diagnostic_source(cell: str):
    envelope = json.loads(
        (ROOT / "configs/t4_no_complete_routes_diagnostic_v1.json").read_text()
    )
    return decode_state(envelope["payload"]["cells"][cell]["source_state"])


def test_ring_path_remodel_contracts_a_cycle_with_exact_replay():
    source = production_state_from_smiles("C1CCCCC1", max_atoms=48)
    product, stage = compile_ring_path_remodel(
        source, RingPathRemodelRequest((0, 1, 2))
    )

    assert canonical_state_key(product) == "C1CCCC1"
    assert stage["name"] == "ring_path_remodel"
    assert stage["primitive_edits"] == 2
    assert stage["parameters"]["retained_boundaries"] == [0, 2]


def test_joint_substituted_ring_and_remodel_exactly_recover_jak2_teacher():
    source = _diagnostic_source("jak2_1")
    requests = (
        PendantDeleteRequest((0, 1), 2),
        SubstitutedRingRequest(
            RingRequest(
                RingSpec("pendant", 6, (4, 2, 0), "saturated"),
                (2,),
                ("N", "C", "C", "C", "N", "C"),
                (1, 1, 1, 1, 1, 1),
            ),
            (
                RingSubstituentAtom("ring", 1, "C"),
                RingSubstituentAtom("ring", 4, "C"),
                RingSubstituentAtom("substituent", 1, "O", 2),
                RingSubstituentAtom("substituent", 1, "O", 1),
            ),
        ),
        RingPathRemodelRequest((5, 6, 7)),
    )

    _, program, _, trace, metadata = compile_forced_module_sequence(source, requests)

    assert trace["endpoint"] == ("CC1CCN(C(=O)O)CN1C(=O)CC1c2ccccc2-c2ccnc3[nH]cc1c23")
    assert metadata["modules"] == [
        "substituent_delete",
        "construct_substituted_ring",
        "ring_path_remodel",
    ]
    assert metadata["module_count"] == 3
    assert len(program.marks) == 15
    assert metadata["intermediate_task_evaluations"] == 0


def test_delete_and_contextual_functionalize_recover_5ht1b_teacher_precursor():
    source = _diagnostic_source("5ht1b_0")
    _, program, _, trace, metadata = compile_forced_module_sequence(
        source,
        (
            PendantDeleteRequest((0, 1, 2, 38, 37), 3),
            FunctionalizeRequest(36, "O"),
        ),
    )

    assert trace["endpoint"] == (
        "Cc1ccc(-c2ccc(C(=O)N3CCc4cc5c(cc43)C3(CC[NH+](C)CC3)CO5)cc2)" "c(C)c1O"
    )
    assert metadata["modules"] == ["substituent_delete", "functionalize"]
    assert len(program.marks) == 6


def test_contextual_binding_panel_is_bounded_deterministic_and_prefers_anchor():
    source = production_state_from_smiles("CCO", max_atoms=48)
    first = enumerate_functionalizations(
        source, preferred_anchors=frozenset({1}), max_candidates=5
    )
    second = enumerate_functionalizations(
        source, preferred_anchors=frozenset({1}), max_candidates=5
    )

    assert 1 <= len(first) <= 5
    assert [candidate.structural_rank for candidate in first] == [
        candidate.structural_rank for candidate in second
    ]
    assert first[0].stage["parameters"]["anchor"] == 1


def test_contextual_deletion_reaches_a_pendant_ring_lobe_without_edge_cut():
    source = _diagnostic_source("5ht1b_0")
    candidates = enumerate_pendant_deletions(source, max_candidates=16)

    requests = {
        (
            tuple(candidate.stage["parameters"]["fragment_slots"]),
            candidate.stage["parameters"]["retained_anchor"],
        )
        for candidate in candidates
    }
    assert ((0, 1, 2, 37, 38), 3) in requests


def test_contextual_ring_path_panel_reaches_the_generic_jak2_contraction():
    source = _diagnostic_source("jak2_1")
    candidates = enumerate_ring_path_remodels(source, max_candidates=16)

    paths = {tuple(candidate.stage["parameters"]["path"]) for candidate in candidates}
    assert (5, 6, 7) in paths


def test_v1_initial_batch_is_seed_deterministic_and_loads_no_route():
    source = production_state_from_smiles("CCNCC", max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=19),
        attempts_per_batch=8,
        candidates_per_batch=2,
        wall_seconds=10,
    )
    kwargs = {
        "source_group": "test-source",
        "oracle_protocol": "structural-test",
        "eligibility": lambda _: {"oracle_eligible": True},
    }
    first = initial_dynamic_program_batch_v1(source, (), config, **kwargs)
    second = initial_dynamic_program_batch_v1(source, (), config, **kwargs)

    assert first["batch_id"] == second["batch_id"]
    assert first["initial_route_archive"] == []
    assert first["source_library_rows_loaded"] == 0
    assert first["new_oracle_calls"] == 0
    assert all(
        candidate["provenance"]["metadata"]["intermediate_task_evaluations"] == 0
        for candidate in first["candidates"]
    )


def test_v1_production_module_contains_no_development_cell_payload():
    source = (
        ROOT / "src/compose_v4/control/dynamic_program_synthesis_v1.py"
    ).read_text()

    for forbidden in (
        "jak2",
        "5ht1b",
        "braf",
        "CC1CCN(C(=O)O)CN1C(=O)",
        "Cc1ccc(-c2ccc(C(=O)N3",
    ):
        assert forbidden.lower() not in source.lower()


def test_v1_development_contract_is_three_cell_route_empty_and_hash_bound():
    contract = load_contract(ROOT)
    reference = unseal(ROOT / "diagnostics/t4_dynamic_v1/comparison_reference.json")

    assert contract["dynamic_v1_development"] == v1_contract_payload()
    assert contract["library_programs"] == 0
    assert contract["dynamic_v1_development"]["initial_route_archive"] == []
    assert contract["dynamic_v1_development"]["development_units"] == list(
        SELECTED_UNITS
    )
    assert set(reference["full_146"]) == set(SELECTED_UNITS)
    assert set(reference["dynamic_v0"]) == set(SELECTED_UNITS)
    assert reference["new_oracle_calls"] == 0
