import gzip
import json

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_delta06_utility_lock import (
    EligibilityThresholds,
    _sealed_bytes,
    build_request_records,
    choose_unit_candidates,
    endpoint_exclusion_reasons,
    load_contract,
    publish_bundle,
)

THRESHOLDS = EligibilityThresholds(
    similarity_gt=0.6,
    qed_gt=0.6,
    sa_lt=4.0,
    active_atoms_lte=40,
)


def _reasons(**overrides):
    values = {
        "valid": True,
        "connected": True,
        "is_null": False,
        "canonical_available": True,
        "active_heavy_match": True,
        "non_self": True,
        "similarity": 0.61,
        "qed": 0.61,
        "sa": 3.99,
        "active_atoms": 40,
        "thresholds": THRESHOLDS,
    }
    values.update(overrides)
    return endpoint_exclusion_reasons(**values)


def test_strict_eligibility_boundaries_and_all_reasons_are_preserved():
    assert _reasons() == []
    assert _reasons(similarity=0.6) == ["similarity_not_strictly_greater_than_0.6"]
    assert _reasons(qed=0.6) == ["qed_not_strictly_greater_than_0.6"]
    assert _reasons(sa=4.0) == ["sa_not_strictly_less_than_4.0"]
    assert _reasons(active_atoms=41) == ["active_atoms_greater_than_40"]
    assert _reasons(
        valid=False,
        connected=False,
        is_null=True,
        canonical_available=False,
        active_heavy_match=False,
        non_self=False,
        similarity=None,
        qed=None,
        sa=None,
        active_atoms=41,
    ) == [
        "invalid_molecular_graph",
        "disconnected_molecular_graph",
        "null_endpoint",
        "canonical_molecule_unavailable",
        "active_heavy_atom_count_mismatch",
        "self_endpoint",
        "similarity_not_strictly_greater_than_0.6",
        "qed_not_strictly_greater_than_0.6",
        "sa_not_strictly_less_than_4.0",
        "active_atoms_greater_than_40",
    ]


def test_unit_selection_is_rank_first_then_maximum_distance_with_frozen_ties():
    rows = [
        {"rank": 3, "canonical_smiles": "C", "attempt_id": "third"},
        {"rank": 1, "canonical_smiles": "N", "attempt_id": "first"},
        {"rank": 2, "canonical_smiles": "N", "attempt_id": "duplicate"},
        {"rank": 4, "canonical_smiles": "O", "attempt_id": "far-late"},
        {"rank": 2, "canonical_smiles": "F", "attempt_id": "far-early"},
    ]
    distances = {("N", "C"): 0.2, ("N", "O"): 0.8, ("N", "F"): 0.8}
    selected = choose_unit_candidates(
        rows,
        distance=lambda first, second: distances[
            (first["canonical_smiles"], second["canonical_smiles"])
        ],
    )
    assert [
        (role, row["attempt_id"], distance) for role, row, distance in selected
    ] == [
        ("lowest_rank_eligible", "first", None),
        ("maximally_morgan_distant", "far-early", 0.8),
    ]


def _membership(membership_id, *, policy_id, attempt_id):
    return {
        "membership_id": membership_id,
        "fold": 0,
        "cell": "braf_1",
        "target": "braf",
        "source_case_id": "source-1",
        "policy_id": policy_id,
        "selection_role": "lowest_rank_eligible",
        "rank": 1,
        "attempt_id": attempt_id,
        "canonical_smiles": "CC",
        "docking_seed": 1701,
        "evaluator": "evaluator-hash",
        "evaluator_payload": {"target": "braf"},
    }


def test_request_deduplication_preserves_every_source_policy_membership():
    memberships = [
        _membership(
            "m2", policy_id="source_balanced_marginal_subgoal", attempt_id="a2"
        ),
        _membership("m1", policy_id="graph_conditioned_subgoal", attempt_id="a1"),
    ]
    requests = build_request_records(memberships)
    assert len(requests) == 1
    assert requests[0]["membership_ids"] == ["m1", "m2"]
    assert [row["membership_id"] for row in requests[0]["memberships"]] == [
        "m1",
        "m2",
    ]


def test_sealed_gzip_is_byte_deterministic_and_self_hashed():
    payload = {"z": [2, 1], "a": "value"}
    first = _sealed_bytes(payload, compressed=True)
    second = _sealed_bytes(payload, compressed=True)
    assert first == second
    envelope = json.loads(gzip.decompress(first))
    assert envelope == {"payload": payload, "payload_sha256": identity(payload)}


def test_contract_loader_rejects_any_scored_authority(tmp_path):
    payload = {
        "schema_version": "t4_delta06_structural_subgoal_utility_lock_contract_v1",
        "oracle": {
            "calls_authorized": 0,
            "docking_calls_authorized": 1,
            "modal_launch_authorized": False,
            "live_run_access_authorized": False,
            "scored_launch_authorized": False,
        },
    }
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps({"payload": payload, "contract_sha256": identity(payload)})
    )
    with pytest.raises(ValueError, match="zero-oracle"):
        load_contract(contract)


def test_publication_refuses_to_overwrite_existing_lock(tmp_path):
    output = tmp_path / "attempt_1"
    output.mkdir()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        publish_bundle(output, {})
