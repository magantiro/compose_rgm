from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.compositional_structural_subgoal_generator import GeneratedPatch
from compose_v4.control.docking_value import identity
from compose_v4.control.macro_archive_integration import (
    AdmissionDecision,
    EligibilityReceipt,
    MacroCandidateReference,
    MacroLane,
    build_shared_archive,
    candidate_resolver,
    dynamic_v0_record,
    seed_sequences,
    selector_references,
)
from compose_v4.control.structural_subgoal import extract_structural_goal
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace_shard import encode_state

LOCK = "a" * 64
FILTER = "b" * 64


def test_integration_contracts_are_self_hashed_and_qualification_is_not_launchable():
    paths = (
        "configs/t4_macro_archive_v0_integration_v1.json",
        "configs/t4_macro_archive_v0_selector_binding_v1.json",
        "configs/t4_macro_archive_v0_five_cell_qualification_v1.json",
    )
    payloads = []
    for path in paths:
        value = json.loads(Path(path).read_text())
        assert value["contract_sha256"] == identity(value["payload"])
        payloads.append(value["payload"])
    assert payloads[0]["budgets"]["oracle_calls"] == 0
    assert payloads[1]["oracle_calls"] == 0
    assert payloads[2]["status"] == "prepared_not_authorized_for_launch"
    assert payloads[2]["budget"]["plateau_stopping"] is False


def _ethane():
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[:2] = 2
    hydrogens[:2] = 3
    bonds[0, 1] = bonds[1, 0] = 1
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def _candidate(source, *, slot=3):
    action = encode_action(
        "atom_insert",
        AtomInsert(
            slot=slot,
            atom_type=2,
            formal_charge=0,
            implicit_h_count=3,
            neighbors=((0, 1),),
        ),
    )
    endpoint, trace = execute_program(source, [action])
    goal, bindings, _ = extract_structural_goal(
        tuple(trace["states"]), tuple(trace["actions"])
    )
    return (
        GeneratedPatch(
            goal=goal,
            bindings=bindings,
            endpoint_state=encode_state(endpoint),
            actions=(action,),
            construction_dependencies=(),
            score=0.0,
            policy_id="balanced_joint_autoregressive",
            patch_ids=(identity(goal.payload()),),
            realization={
                "status": "realized",
                "endpoint_matches_bound_target": True,
                "primitive_teacher_actions_used": 0,
                "primitive_count": 1,
                "attempted": 1,
                "expanded": 0,
                "compiler_strategy": "test_exact",
            },
        ).payload(),
        trace,
    )


def _accept_all(rows):
    return [
        AdmissionDecision(row["endpoint_key"], True, rank, "fixture_accept_all")
        for rank, row in enumerate(sorted(rows, key=lambda item: item["endpoint_key"]))
    ]


def test_shared_archive_is_order_invariant_and_preserves_lane_collision():
    source = _ethane()
    raw, _ = _candidate(source)
    candidate_id = identity(raw)
    learned = MacroCandidateReference(
        "case", candidate_id, LOCK, MacroLane.LEARNED, 7, 0, 1.2
    )
    # A true cross-lane alias normally has a distinct serialized proposal. The
    # resolver fixture adds an inert provenance value so its candidate identity
    # differs while its exact endpoint stays the same.
    uniform_raw = deepcopy(raw)
    uniform_raw["score"] = -0.0
    uniform_id = identity(uniform_raw)
    uniform = MacroCandidateReference("case", uniform_id, LOCK, MacroLane.UNIFORM, 2)
    rows = {candidate_id: raw, uniform_id: uniform_raw}

    def resolver(ref):
        return encode_state(source), rows[ref.candidate_identity]

    receipts = {
        key: EligibilityReceipt(key, FILTER, True, {"strict_eligible": True})
        for key in rows
    }
    first = build_shared_archive(
        [learned, uniform],
        resolver=resolver,
        eligibility=receipts,
        admission_policy=_accept_all,
    )
    second = build_shared_archive(
        [uniform, learned],
        resolver=resolver,
        eligibility=receipts,
        admission_policy=_accept_all,
    )
    assert first == second
    assert first["candidate_count"] == 2
    assert first["unique_endpoint_count"] == 1
    assert len(first["entries"][0]["contributions"]) == 2
    assert first["entries"][0]["candidate_identity"] == candidate_id


def test_dynamic_v0_handoff_replays_as_existing_program_record():
    source = _ethane()
    raw, trace = _candidate(source)
    candidate_id = identity(raw)
    ref = MacroCandidateReference(
        "case", candidate_id, LOCK, MacroLane.LEARNED, 0, 0, 2.0
    )
    receipt = EligibilityReceipt(candidate_id, FILTER, True, {})
    archive = build_shared_archive(
        [ref],
        resolver=lambda _: (encode_state(source), raw),
        eligibility={candidate_id: receipt},
        admission_policy=_accept_all,
    )
    record = dynamic_v0_record(
        archive["entries"][0], source_group="case", oracle_protocol="p"
    )
    assert record["score"] is None
    assert (
        record["endpoint"]
        == trace["endpoint"]
        == canonical_state_key(execute_program(source, raw["actions"])[0])
    )
    assert record["provenance"]["oracle_calls"] == 0


def test_invalid_realization_fails_closed():
    source = _ethane()
    raw, _ = _candidate(source)
    bad = deepcopy(raw)
    bad["realization"]["primitive_teacher_actions_used"] = 1
    bad_id = identity(bad)
    ref = MacroCandidateReference("case", bad_id, LOCK, MacroLane.LEARNED, 0, 0, 0.0)
    receipt = EligibilityReceipt(bad_id, FILTER, True, {})
    try:
        build_shared_archive(
            [ref],
            resolver=lambda _: (encode_state(source), bad),
            eligibility={bad_id: receipt},
            admission_policy=_accept_all,
        )
    except ValueError as error:
        assert "teacher-free exact-realization" in str(error)
    else:
        raise AssertionError("invalid realization did not fail closed")


def test_rng_namespaces_are_deterministic_and_distinct():
    first = seed_sequences(20260915)
    assert first == seed_sequences(20260915)
    assert len({tuple(value) for value in first.values()}) == 4
    assert set(first) == {
        "learned_macro_generation",
        "uniform_macro_generation",
        "archive_admission",
        "dynamic_v0_refinement",
    }


def test_selector_and_generator_locks_resolve_by_identity_only():
    source = _ethane()
    raw, _ = _candidate(source)
    candidate_id = identity(raw)
    generator_payload = {
        "schema_version": "fixture_generator_lock",
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
        "folds": [
            {
                "fold": 0,
                "cases": [
                    {
                        "source_case_id": "case",
                        "source_state": encode_state(source),
                        "policies": [{"policy_id": "learned", "candidates": [raw]}],
                    }
                ],
            }
        ],
    }
    generator = {
        "payload": generator_payload,
        "payload_sha256": identity(generator_payload),
    }
    selector_payload = {
        "schema_version": "t4_complete_macro_selector_candidate_lock_v1",
        "teacher_fields_present": False,
        "task_identity_present": False,
        "candidate_generation_calls": 0,
        "folds": [
            {
                "fold": 0,
                "cases": [
                    {
                        "source_case_id": "case",
                        "candidate_count": 1,
                        "ranked_candidates": [
                            {
                                "candidate_identity": candidate_id,
                                "original_rank": 9,
                                "selector_rank": 0,
                                "selector_score": 1.5,
                                "complete_macro_signature": ["patch"],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    selector = {
        "payload": selector_payload,
        "payload_sha256": identity(selector_payload),
    }
    refs = selector_references(selector, generator_lock_sha256=LOCK)
    source_state, resolved = candidate_resolver(generator)(refs[0])
    assert source_state == encode_state(source)
    assert identity(resolved) == identity(raw)

    poisoned = deepcopy(selector)
    poisoned["payload"]["teacher_fields_present"] = True
    poisoned["payload_sha256"] = identity(poisoned["payload"])
    try:
        selector_references(poisoned, generator_lock_sha256=LOCK)
    except ValueError as error:
        assert "candidate-reference boundary" in str(error)
    else:
        raise AssertionError("teacher-bearing selector did not fail closed")
