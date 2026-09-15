import gzip
import json

import pytest

import compose_v4.experiments.t4_compositional_utility_lock as utility_lock
from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_compositional_utility_lock import (
    BINDING_SCHEMA,
    _sealed_bytes,
    build_request_records,
    endpoint_exclusion_reasons,
    load_binding,
    publish_bundle,
    select_cell_arm,
    sha256_file,
)


def _reasons(**overrides):
    values = {
        "replay_exact": True,
        "valid": True,
        "connected": True,
        "is_null": False,
        "canonical_available": True,
        "active_heavy_match": True,
        "non_self": True,
        "similarity": 0.41,
        "qed": 0.61,
        "sa": 3.99,
        "active_atoms": 40,
    }
    values.update(overrides)
    return endpoint_exclusion_reasons(**values)


def test_delta04_eligibility_is_strict_and_preserves_all_failures():
    assert _reasons() == []
    assert _reasons(similarity=0.4) == ["similarity_not_strictly_greater_than_0.4"]
    assert _reasons(qed=0.6) == ["qed_not_strictly_greater_than_0.6"]
    assert _reasons(sa=4.0) == ["sa_not_strictly_less_than_4.0"]
    assert _reasons(active_atoms=41) == ["active_atoms_greater_than_40"]
    assert _reasons(
        replay_exact=False,
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
        "candidate_actions_do_not_exactly_replay_endpoint_state",
        "invalid_molecular_graph",
        "disconnected_molecular_graph",
        "null_endpoint",
        "canonical_molecule_unavailable",
        "active_heavy_atom_count_mismatch",
        "self_endpoint",
        "similarity_not_strictly_greater_than_0.4",
        "qed_not_strictly_greater_than_0.6",
        "sa_not_strictly_less_than_4.0",
        "active_atoms_greater_than_40",
    ]


def _candidate(rank, canonical, *, eligible=True):
    return {
        "rank": rank,
        "canonical_smiles": canonical,
        "candidate_id": f"candidate-{rank}",
        "eligible": eligible,
        "selection_exclusion_reasons": [],
        "selected": False,
    }


def test_selection_is_eligibility_first_canonical_deduplicated_and_rank_first():
    ineligible = _candidate(0, "Cl", eligible=False)
    first = _candidate(1, "C")
    duplicate = _candidate(2, "C")
    later = _candidate(3, "N")
    rows = [later, duplicate, ineligible, first]
    selected = select_cell_arm(rows)
    assert selected is first
    assert first["selected"] is True
    assert duplicate["selection_exclusion_reasons"] == [
        "eligible_canonical_duplicate_within_cell_arm"
    ]
    assert later["selection_exclusion_reasons"] == [
        "eligible_not_lowest_rank_representative"
    ]
    assert ineligible["selection_exclusion_reasons"] == []


def _membership(membership_id, arm_id):
    return {
        "membership_id": membership_id,
        "generator_revision": arm_id.split("_")[0],
        "arm_id": arm_id,
        "policy_id": "uniform_joint_grammar",
        "fold": 0,
        "cell": "braf_1",
        "target": "braf",
        "source_case_id": "source-1",
        "rank": 1,
        "candidate_id": f"candidate-{membership_id}",
        "candidate_uid": f"uid-{membership_id}",
        "canonical_smiles": "CC",
        "docking_seed": 1701,
        "evaluator": "evaluator-hash",
        "evaluator_payload": {"target": "braf"},
    }


def test_cross_arm_request_deduplication_preserves_memberships_without_backfill():
    memberships = [
        _membership("m2", "expanded_uniform"),
        _membership("m1", "baseline_uniform"),
    ]
    requests = build_request_records(memberships)
    assert len(requests) == 1
    assert requests[0]["membership_ids"] == ["m1", "m2"]
    assert [row["arm_id"] for row in requests[0]["memberships"]] == [
        "baseline_uniform",
        "expanded_uniform",
    ]


def test_binding_verifies_both_physical_locks_before_payload_open(
    tmp_path, monkeypatch
):
    implementation = tmp_path / "src/compose_v4/experiments"
    implementation.mkdir(parents=True)
    implementation_file = implementation / "t4_compositional_utility_lock.py"
    implementation_file.write_text("selection implementation")
    baseline = tmp_path / "baseline.json.gz"
    expanded = tmp_path / "expanded.json.gz"
    baseline.write_bytes(b"not opened by binding validation")
    expanded.write_bytes(b"also not opened by binding validation")
    prelock_payload = {"selection_semantics": {"selection": "frozen"}}
    prelock = tmp_path / "prelock.json"
    prelock.write_text(
        json.dumps(
            {
                "payload": prelock_payload,
                "contract_sha256": identity(prelock_payload),
            }
        )
    )
    frozen = {
        "selection_semantics": {"selection": "frozen"},
        "candidate_bindings": {
            "baseline": {
                "logical_path": "baseline.json.gz",
                "expected_physical_sha256": sha256_file(baseline),
            },
            "expanded": {
                "logical_path": "expanded.json.gz",
                "expected_physical_sha256": "PENDING_ARTIFACT_ONLY_COMMIT",
            },
        },
    }
    binding_payload = {
        "schema_version": BINDING_SCHEMA,
        "prelock": {
            "path": "prelock.json",
            "sha256": sha256_file(prelock),
            "contract_sha256": identity(prelock_payload),
        },
        "selection_semantics_sha256": identity(frozen["selection_semantics"]),
        "candidate_locks": {
            "baseline": {
                "path": "baseline.json.gz",
                "sha256": sha256_file(baseline),
                "payload_sha256": "1" * 64,
                "source_revision": "a" * 40,
            },
            "expanded": {
                "path": "expanded.json.gz",
                "sha256": sha256_file(expanded),
                "payload_sha256": "2" * 64,
                "source_revision": "b" * 40,
            },
        },
        "selection_implementation": {
            "path": "src/compose_v4/experiments/t4_compositional_utility_lock.py",
            "sha256": sha256_file(implementation_file),
            "source_revision": "c" * 40,
        },
    }
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "payload": binding_payload,
                "payload_sha256": identity(binding_payload),
            }
        )
    )
    monkeypatch.setattr(
        utility_lock,
        "sha256_git_path",
        lambda root, revision, path: sha256_file(root / path),
    )
    assert load_binding(tmp_path, binding, prelock, frozen) == binding_payload
    expanded.write_bytes(b"changed after binding")
    with pytest.raises(ValueError, match="physical identity changed"):
        load_binding(tmp_path, binding, prelock, frozen)


def test_sealed_gzip_is_byte_deterministic_and_self_hashed():
    payload = {"z": [2, 1], "a": "value"}
    first = _sealed_bytes(payload, compressed=True)
    second = _sealed_bytes(payload, compressed=True)
    assert first == second
    assert json.loads(gzip.decompress(first)) == {
        "payload": payload,
        "payload_sha256": identity(payload),
    }


def test_publication_refuses_to_overwrite_existing_lock(tmp_path):
    output = tmp_path / "attempt_1"
    output.mkdir()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        publish_bundle(output, {})
