from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_candidate_provenance_decisions import (
    EditingV2CandidateProvenanceDecisionsError,
    candidate_provenance_decisions_self_hash,
    canonical_json_bytes,
    load_candidate_provenance_decisions,
    validate_candidate_provenance_decisions,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "editing_v2_candidate_provenance_decisions_v1.json"

MATERIALIZER_SHA = "bf332f1c43c438fea7a0e1b4cfed18d186312459905abbdb4ebd052f29a9cc86"
BRIDGE_SHA = "15f9c1ea8bac1a221227961c37c24d18fcaeaf60b435cac2b6a1497f34510b3e"
CORPUS_CONTRACT_SHA = "153ef7316a64d03a89ba739f6423976b1d734695af8569d2e2cdc7a1227eb519"
ROUTING_POLICY_SHA = "14c50c14675e3ec3efbb4fb7fa0b6c58866ebdc7693ceaec8e3f0e702af5ce64"


def _raw() -> dict:
    return json.loads(CONFIG_PATH.read_bytes())


def _rehash(value: dict) -> dict:
    value["decision_sha256"] = candidate_provenance_decisions_self_hash(value)
    return value


def _nested_hash(value: dict, hash_field: str) -> None:
    body = dict(value)
    body.pop(hash_field)
    value[hash_field] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_decisions_load_and_bind_completed_candidate_artifacts() -> None:
    decisions = load_candidate_provenance_decisions(CONFIG_PATH)
    artifact = decisions.candidate_artifact

    assert decisions.training_authorized is False
    assert decisions.decision_sha256 == candidate_provenance_decisions_self_hash(_raw())
    assert artifact.run_identity_sha256 == (
        "2f29883dafe6b2b7fd812c34a1a99d8c89270ce729387d37abc1111193f56c13"
    )
    assert artifact.build_commit == "720f22672ac66773e0cb7e1fbb8ce0f29491f33f"
    assert artifact.completion.file_sha256 == (
        "3fd9bce17054b518c98b0194e10d91228a407a9cb3a44912d43e820ef490ac4a"
    )
    assert artifact.completion.semantic_sha256 == (
        "a6dcd0dddbdadf0f7f131d0590383dedaa6a888ff4a671061595fb8a4bf215d9"
    )
    assert artifact.candidate_materialization.manifest_file_sha256 == (
        "4d9f1b62062b79b259e6c43f25beaff6fdb78775cfca9b0e20acd2ab36facbf0"
    )
    assert artifact.candidate_materialization.manifest_sha256 == (
        "753316fb9f7d484df2cf9c16ca25aee1355f98927bb6f95ec8510413f865720e"
    )
    assert artifact.candidate_materialization.rows_file_sha256 == (
        "7f4b6d99dd3780e5e96ab9f5687868810234ce6ed06dc77b81659d37a02c41f6"
    )
    assert artifact.candidate_materialization.rows_semantic_sha256 == (
        "c832de8efa269389b32dc86474b518e7053c7a10a8a248bc3d9f8cceb50ea1a9"
    )
    assert artifact.candidate_materialization.address_stream_sha256 == (
        "66297fd08b214e83acca236ce58a814216dc55bcc3966a105b9254bfaf7e78be"
    )
    assert artifact.source_manifest.file_sha256 == (
        "c8e2bc12a88bf293866f79656d66239031ed7ed9dae86a2115e7d430adfc36d9"
    )
    assert artifact.source_manifest.semantic_sha256 == (
        "3429cd4e8aa73168cd4427a6511cb856587f7d74e2e6e6ca0161ee9c80995193"
    )


def test_registry_inputs_are_exact_and_keep_semantic_policies_deferred() -> None:
    decisions = load_candidate_provenance_decisions(CONFIG_PATH)
    registry = decisions.registry_inputs()

    assert set(registry) == {
        "compiler_identity",
        "policy_bindings",
        "evidence_identity",
        "identity_definitions",
        "source_assets",
    }
    assert registry["compiler_identity"] == {
        "implementation_sha256": MATERIALIZER_SHA,
        "config_sha256": ROUTING_POLICY_SHA,
        "operator_contract_sha256": CORPUS_CONTRACT_SHA,
        "canonicalizer_sha256": decisions.candidate_stage_pass_through.identity_sha256,
    }
    assert registry["policy_bindings"] == {
        "mapping_policy_sha256": None,
        "metric_policy_sha256": None,
        "unresolved_mapping_policies": [
            "editing_v2.mapping_policy.resolve_after_semantic_migration"
        ],
        "unresolved_metric_policies": ["editing_v2.metric_policy.resolve_after_semantic_migration"],
    }
    assert registry["evidence_identity"]["implementation_sha256"] == BRIDGE_SHA
    assert registry["evidence_identity"]["component_reference_fields"] == {
        "source_endpoint": ["endpoint_identity.source_key"],
        "target_endpoint": ["endpoint_identity.target_key"],
        "pair_relationship": [
            "endpoint_identity.source_key",
            "endpoint_identity.target_key",
        ],
        "path": ["packed_address.trace_id", "trace_envelope_sha256"],
        "intermediates": ["exact_states.encoded_state_stream_sha256"],
        "action_sequence": ["operator_summary.action_stream_sha256"],
    }
    assert [source["source_asset_id"] for source in registry["source_assets"]] == [
        "guacamol-subset-500000-seed0",
        "mmp-analogue-pool-v2-252fbfaafe1c1fb3b16f",
    ]
    assert all(source["access_basis"] for source in registry["source_assets"])
    assert all(
        source["source_group_namespace"]["cross_lane_sharing_authorized"] is True
        for source in registry["source_assets"]
    )


def test_local_implementations_and_input_configs_match_frozen_physical_hashes() -> None:
    decisions = load_candidate_provenance_decisions(CONFIG_PATH)
    artifact = decisions.candidate_artifact

    assert (
        _file_sha256(
            ROOT / "src" / "compose_v4" / "data" / "editing_v2_candidate_provenance_bridge.py"
        )
        == decisions.evidence_identity.implementation_sha256
    )
    assert _file_sha256(ROOT / artifact.editing_corpus_contract.path) == (
        artifact.editing_corpus_contract.file_sha256
    )
    assert _file_sha256(ROOT / artifact.routing_policy.path) == artifact.routing_policy.file_sha256
    assert _file_sha256(ROOT / artifact.source_binding_registry.path) == (
        artifact.source_binding_registry.file_sha256
    )


def test_top_level_self_hash_rejects_tampering() -> None:
    tampered = _raw()
    tampered["decision_id"] = "tampered"
    with pytest.raises(
        EditingV2CandidateProvenanceDecisionsError,
        match="self-hash disagrees",
    ):
        validate_candidate_provenance_decisions(tampered)


@pytest.mark.parametrize("policy_type", ["mapping", "metric"])
@pytest.mark.parametrize("state", ["both", "neither"])
def test_each_policy_requires_exactly_one_resolved_or_unresolved_state(
    policy_type: str,
    state: str,
) -> None:
    value = copy.deepcopy(_raw())
    digest_field = f"{policy_type}_policy_sha256"
    unresolved_field = f"unresolved_{policy_type}_policies"
    if state == "both":
        value["policy_bindings"][digest_field] = "0" * 64
    else:
        value["policy_bindings"][unresolved_field] = []
    _rehash(value)

    with pytest.raises(
        EditingV2CandidateProvenanceDecisionsError,
        match=rf"policy_bindings\.{policy_type} must have exactly one",
    ):
        validate_candidate_provenance_decisions(value)


@pytest.mark.parametrize("policy_type", ["mapping", "metric"])
def test_resolved_policy_state_is_valid_when_unresolved_list_is_empty(
    policy_type: str,
) -> None:
    value = copy.deepcopy(_raw())
    value["policy_bindings"][f"{policy_type}_policy_sha256"] = "0" * 64
    value["policy_bindings"][f"unresolved_{policy_type}_policies"] = []
    _rehash(value)

    decisions = validate_candidate_provenance_decisions(value)
    assert getattr(decisions.policy_bindings, f"{policy_type}_policy_sha256") == "0" * 64


def test_pass_through_contract_cannot_silently_add_semantic_normalization() -> None:
    value = copy.deepcopy(_raw())
    value["candidate_stage_pass_through"]["semantic_normalization_performed"] = True
    _nested_hash(value["candidate_stage_pass_through"], "identity_sha256")
    value["compiler_identity"]["canonicalizer_sha256"] = value["candidate_stage_pass_through"][
        "identity_sha256"
    ]
    _rehash(value)

    with pytest.raises(
        EditingV2CandidateProvenanceDecisionsError,
        match="no-recanonicalization pass-through",
    ):
        validate_candidate_provenance_decisions(value)


def test_evidence_selectors_must_be_sorted_and_unique_even_after_rehash() -> None:
    value = copy.deepcopy(_raw())
    fields = value["evidence_identity"]["component_reference_fields"]["pair_relationship"]
    fields.reverse()
    _nested_hash(value["evidence_identity"], "evidence_identity_sha256")
    _rehash(value)

    with pytest.raises(
        EditingV2CandidateProvenanceDecisionsError,
        match="pair_relationship must be unique and deterministically sorted",
    ):
        validate_candidate_provenance_decisions(value)


def test_source_namespace_must_bind_its_exact_source_asset() -> None:
    value = copy.deepcopy(_raw())
    value["source_assets"][0]["source_group_namespace"]["source_asset_sha256"] = "0" * 64
    _rehash(value)

    with pytest.raises(
        EditingV2CandidateProvenanceDecisionsError,
        match="must bind the exact source asset",
    ):
        validate_candidate_provenance_decisions(value)
