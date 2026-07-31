from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_corpus_contract import (
    ACTIVE8_FAMILIES,
    DISABLED_FAMILIES,
    EXPECTED_SCHEMA_VERSION,
    REQUIRED_DATA_LANES,
    EditingCorpusContractError,
    assert_training_launch_authorized,
    load_editing_corpus_contract,
    relationship_definition_sha256,
    training_launch_blockers,
    validate_admitted_record_envelope,
    validate_editing_corpus_contract,
    validate_evidence_assignment_envelope,
    validate_partition_resolution,
    validate_record_membership_assignment,
    validate_relationship_group,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_corpus_v2_contract.json"


def _contract() -> dict[str, Any]:
    return load_editing_corpus_contract(CONTRACT_PATH)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {nested for child in value.values() for nested in _all_keys(child)}
    if isinstance(value, list):
        return {nested for child in value for nested in _all_keys(child)}
    return set()


def _freeze_thresholds(contract: dict[str, Any]) -> None:
    thresholds = contract["pretraining_gates"]["threshold_values"]
    for name in thresholds:
        if "share" in name or name.startswith("maximum_"):
            thresholds[name] = 0.25
        elif name == "minimum_effective_teacher_coefficient_per_capability":
            thresholds[name] = 0.01
        else:
            thresholds[name] = 1


def _assigned_partition_resolution() -> dict[str, Any]:
    return {
        "status": "assigned",
        "assigned_role": "train",
        "source_endpoint_role": "train",
        "target_endpoint_role": "train",
        "split_component_id": "component-1",
        "split_assignment_manifest_sha256": "a" * 64,
    }


def _evidence_components(
    contract: dict[str, Any],
    profile_id: str = "observed_pair_compiled_path",
) -> dict[str, str]:
    profile = next(
        item
        for item in contract["evidence_component_contract"]["profiles"]
        if item["id"] == profile_id
    )
    return copy.deepcopy(profile["components"])


def _admitted_record(contract: dict[str, Any]) -> dict[str, Any]:
    source_state = {
        "n_slots": 2,
        "atom_types": [1, 0],
        "formal_charges": [0, 0],
        "implicit_h_counts": [4, 0],
        "bonds": [],
    }
    target_state = {
        "n_slots": 2,
        "atom_types": [1, 1],
        "formal_charges": [0, 0],
        "implicit_h_counts": [3, 3],
        "bonds": [[0, 1, 1]],
    }
    return {
        "record_id": "trace-1",
        "partition_resolution": _assigned_partition_resolution(),
        "data_lane": "observed_local_analogue",
        "evidence_profile_id": "observed_pair_compiled_path",
        "evidence_components": _evidence_components(contract),
        "source_state_exact": source_state,
        "successor_states_exact": [target_state],
        "target_state_exact": copy.deepcopy(target_state),
        "canonical_state_keys": ["C", "CC"],
        "executable_actions": [{"model_family": "atom_insert"}],
        "operator_sequence": ["atom_insert"],
        "operator_family_set": ["atom_insert"],
        "record_membership_cells": ["cardinality_growth"],
        "relationship_group_ids": [],
        "source_group_id": "source-group-1",
        "scaffold_group_id": "scaffold-group-1",
        "series_group_id": None,
        "document_or_source_group_id": "document-group-1",
        "transformation_signature": "C>>CC",
        "constant_core_mapping": [[0, 0]],
        "protected_mapping": None,
        "atom_count_delta": 1,
        "graph_cycle_rank_delta": 0,
        "path_length": 1,
        "maximum_intermediate_source_distance": 1.0,
        "immediate_reversal_count": 0,
        "repeated_state_count": 0,
        "constraint_evaluator_identity": None,
        "compiler_candidates_considered": 1,
        "compiler_choice_reason": "minimum_registered_cost",
        "compiler_failure_reason": None,
        "source_provenance": {"source_record_id": "source-record-1"},
        "operator_contract_hash": "b" * 64,
        "canonicalizer_hash": "c" * 64,
        "compiler_hash": "d" * 64,
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _relationship_member(
    role: str,
    *,
    trace_id: str,
    entry: int,
    start: int,
    end: int,
) -> dict[str, Any]:
    member = {
        "trace_key": {
            "packed_shard_sha256": "a" * 64,
            "entry": entry,
            "trace_id": trace_id,
        },
        "segment": {
            "start_progress_index": start,
            "end_progress_index": end,
        },
        "role": role,
    }
    member["member_sha256"] = _canonical_sha256(member)
    return member


def _relationship_group(
    contract: dict[str, Any],
    relationship_type: str,
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    definition = next(
        item
        for item in contract["relationship_coverage_contract"]["group_types"]
        if item["id"] == relationship_type
    )
    group = {
        "type": relationship_type,
        "group_id": f"{relationship_type}-group-1",
        "definition_sha256": relationship_definition_sha256(
            contract,
            relationship_type,
        ),
        "evidence_profile_id": "observed_pair_compiled_path",
        "verification_receipt": {
            "receipt_type": definition["verification_receipt_type"],
            "verifier_identity_sha256": "b" * 64,
            "receipt_sha256": "c" * 64,
        },
        "members": members,
    }
    group["group_sha256"] = _canonical_sha256(group)
    return group


def _rehash_relationship_group(group: dict[str, Any]) -> None:
    for member in group["members"]:
        member["member_sha256"] = _canonical_sha256(
            {key: value for key, value in member.items() if key != "member_sha256"}
        )
    group["group_sha256"] = _canonical_sha256(
        {key: value for key, value in group.items() if key != "group_sha256"}
    )


def test_v2_contract_is_well_formed_but_intentionally_blocks_training() -> None:
    contract = _contract()
    assert contract["schema_version"] == EXPECTED_SCHEMA_VERSION == 2
    assert tuple(contract["operator_basis"]["active8"]) == ACTIVE8_FAMILIES
    assert tuple(contract["operator_basis"]["disabled"]) == DISABLED_FAMILIES
    assert tuple(lane["id"] for lane in contract["data_lanes"]) == REQUIRED_DATA_LANES
    assert contract["training_authorized"] is False
    assert contract["status"] == "DESIGN_NOT_TRAINING_AUTHORIZED"

    blockers = training_launch_blockers(contract)
    assert "training_authorized is not true" in blockers
    assert "status is not FROZEN_TRAINING_AUTHORIZED" in blockers
    assert any(item.startswith("unresolved threshold reference:") for item in blockers)
    assert any(
        item.startswith("unresolved record-membership cell: constraint_feasible_path")
        for item in blockers
    )
    assert {
        item for item in blockers if item.startswith("unresolved external authority requirement:")
    } == {
        "unresolved external authority requirement: "
        "authoritative_trace_derived_record_membership_classification "
        "(unresolved_no_go_pending_trace_derived_predicate_receipts)",
        "unresolved external authority requirement: "
        "physical_relationship_receipt_resolution "
        "(unresolved_no_go_pending_physical_receipt_resolver)",
    }
    with pytest.raises(EditingCorpusContractError, match="blocks training"):
        assert_training_launch_authorized(contract)


def test_evidence_is_componentwise_and_does_not_upgrade_missing_provenance() -> None:
    contract = _contract()
    profiles = {
        profile["id"]: profile for profile in contract["evidence_component_contract"]["profiles"]
    }
    assert profiles["observed_pair_observed_actions"]["admission_status"] == (
        "inadmissible_missing_true_action_source_manifest"
    )
    assert profiles["genuine_observed_series_compiled_path"]["admission_status"] == (
        "unavailable_missing_genuine_series_provenance"
    )
    assert (
        profiles["inferred_relation_compiled_path"]["components"]["pair_relationship"]
        == "structurally_inferred_relationship"
    )
    assert (
        profiles["inferred_relation_compiled_path"]["components"]["action_sequence"]
        == "compiler_generated_action_sequence"
    )
    assert "evidence_class" not in _all_keys(contract)
    assert "observed_series_path" not in {lane["id"] for lane in contract["data_lanes"]}


def test_contract_separates_record_relationship_and_corpus_direction_coverage() -> None:
    contract = _contract()
    record_cells = {cell["id"] for cell in contract["record_membership_contract"]["cells"]}
    assert {
        "cardinality_growth",
        "cardinality_shrinkage",
        "cycle_rank_increase",
        "cycle_rank_decrease",
        "coupled_cardinality_topology_path",
        "constraint_feasible_path",
    } <= record_cells
    relationships = {
        group["id"] for group in contract["relationship_coverage_contract"]["group_types"]
    }
    assert {
        "shared_prefix_branch",
        "alternative_route",
        "inverse_pair",
        "correction",
        "constraint_compatible_alternative_route",
    } == relationships
    direction_cells = {
        cell["id"]: cell["required_record_cells"]
        for cell in contract["corpus_direction_contract"]["cells"]
    }
    assert direction_cells["cardinality_bidirectional"] == [
        "cardinality_growth",
        "cardinality_shrinkage",
    ]
    assert direction_cells["cycle_rank_bidirectional"] == [
        "cycle_rank_increase",
        "cycle_rank_decrease",
    ]


def test_exact_trace_fields_apply_only_to_admitted_records() -> None:
    contract = _contract()
    admitted = contract["admitted_record_contract"]
    partition = contract["partition_resolution_contract"]
    assert admitted["applies_to"] == "accepted_post_split_compiled_traces_only"
    assert admitted["rejected_attempts_live_in"] == ("pre_admission_candidate_audit_ledger")
    assert admitted["structural_envelope_validation_scope"] == (
        "structural_envelope_only_no_trace_predicate_recomputation_or_"
        "relationship_receipt_resolution"
    )
    assert [item["id"] for item in admitted["external_authority_requirements"]] == [
        "authoritative_trace_derived_record_membership_classification",
        "physical_relationship_receipt_resolution",
    ]
    assert "partition_resolution" in admitted["required_fields"]
    assert "partition" not in admitted["required_fields"]
    assert partition["rejected_attempt_retention"] == ("pre_admission_candidate_audit_ledger")
    rules = {rule["status"]: rule for rule in partition["status_rules"]}
    assert rules["assigned"]["admitted_record_policy"] == "eligible"
    assert rules["cross_partition_rejected"]["assigned_role_policy"] == "must_be_null"
    assert rules["unassigned_rejected"]["endpoint_role_policy"] == (
        "at_least_one_null_other_may_be_partition_role"
    )


def test_evidence_assignment_envelope_requires_exact_admissible_lane_profile() -> None:
    contract = _contract()
    validate_evidence_assignment_envelope(
        contract,
        data_lane="observed_local_analogue",
        evidence_profile_id="observed_pair_compiled_path",
        evidence_components=_evidence_components(contract),
    )

    with pytest.raises(EditingCorpusContractError, match="reserved"):
        validate_evidence_assignment_envelope(
            contract,
            data_lane="observed_local_analogue",
            evidence_profile_id="observed_pair_observed_actions",
            evidence_components=_evidence_components(
                contract,
                "observed_pair_observed_actions",
            ),
        )
    with pytest.raises(EditingCorpusContractError, match="lane/profile mismatch"):
        validate_evidence_assignment_envelope(
            contract,
            data_lane="observed_local_analogue",
            evidence_profile_id="executor_generated_walk",
            evidence_components=_evidence_components(
                contract,
                "executor_generated_walk",
            ),
        )


def test_evidence_assignment_envelope_rejects_component_drift() -> None:
    contract = _contract()
    missing = _evidence_components(contract)
    missing.pop("action_sequence")
    with pytest.raises(EditingCorpusContractError, match="fields disagree"):
        validate_evidence_assignment_envelope(
            contract,
            data_lane="observed_local_analogue",
            evidence_profile_id="observed_pair_compiled_path",
            evidence_components=missing,
        )

    relabeled = _evidence_components(contract)
    relabeled["pair_relationship"] = "structurally_inferred_relationship"
    with pytest.raises(EditingCorpusContractError, match="do not match declared profile"):
        validate_evidence_assignment_envelope(
            contract,
            data_lane="observed_local_analogue",
            evidence_profile_id="observed_pair_compiled_path",
            evidence_components=relabeled,
        )


def test_admitted_record_envelope_accepts_a_structurally_aligned_record() -> None:
    contract = _contract()
    validate_admitted_record_envelope(contract, _admitted_record(contract))


def test_admitted_record_envelope_rejects_field_and_partition_drift() -> None:
    contract = _contract()
    extra_field = _admitted_record(contract)
    extra_field["partition"] = "train"
    with pytest.raises(EditingCorpusContractError, match="fields disagree"):
        validate_admitted_record_envelope(contract, extra_field)

    rejected_partition = _admitted_record(contract)
    rejected_partition["partition_resolution"] = {
        "status": "cross_partition_rejected",
        "assigned_role": None,
        "source_endpoint_role": "train",
        "target_endpoint_role": "validation",
        "split_component_id": "component-1",
        "split_assignment_manifest_sha256": "a" * 64,
    }
    with pytest.raises(EditingCorpusContractError, match="only an assigned"):
        validate_admitted_record_envelope(contract, rejected_partition)


def test_admitted_record_envelope_rejects_trace_alignment_and_target_drift() -> None:
    contract = _contract()
    bad_length = _admitted_record(contract)
    bad_length["path_length"] = 2
    with pytest.raises(EditingCorpusContractError, match="trace arrays disagree"):
        validate_admitted_record_envelope(contract, bad_length)

    bad_key_length = _admitted_record(contract)
    bad_key_length["canonical_state_keys"].append("CCC")
    with pytest.raises(EditingCorpusContractError, match="source plus one key"):
        validate_admitted_record_envelope(contract, bad_key_length)

    bad_target = _admitted_record(contract)
    bad_target["target_state_exact"]["bonds"] = []
    with pytest.raises(EditingCorpusContractError, match="final exact successor"):
        validate_admitted_record_envelope(contract, bad_target)


def test_admitted_record_envelope_rejects_non_active8_or_unresolved_membership() -> None:
    contract = _contract()
    disabled_family = _admitted_record(contract)
    disabled_family["operator_sequence"] = ["ring_system_delete"]
    disabled_family["operator_family_set"] = ["ring_system_delete"]
    with pytest.raises(EditingCorpusContractError, match="outside Active8"):
        validate_admitted_record_envelope(contract, disabled_family)

    mismatched_family_set = _admitted_record(contract)
    mismatched_family_set["operator_family_set"] = ["atom_delete"]
    with pytest.raises(EditingCorpusContractError, match="must equal"):
        validate_admitted_record_envelope(contract, mismatched_family_set)

    unresolved_membership = _admitted_record(contract)
    unresolved_membership["record_membership_cells"] = ["constraint_feasible_path"]
    with pytest.raises(EditingCorpusContractError, match="unresolved record-membership"):
        validate_admitted_record_envelope(contract, unresolved_membership)


def test_admitted_record_envelope_rejects_bad_mappings_hashes_and_failure_state() -> None:
    contract = _contract()
    duplicate_mapping = _admitted_record(contract)
    duplicate_mapping["constant_core_mapping"] = [[0, 0], [0, 1]]
    with pytest.raises(EditingCorpusContractError, match="one-to-one"):
        validate_admitted_record_envelope(contract, duplicate_mapping)

    bad_hash = _admitted_record(contract)
    bad_hash["compiler_hash"] = "not-a-hash"
    with pytest.raises(EditingCorpusContractError, match="full lowercase SHA-256"):
        validate_admitted_record_envelope(contract, bad_hash)

    failed_compilation = _admitted_record(contract)
    failed_compilation["compiler_failure_reason"] = "no executable path"
    with pytest.raises(
        EditingCorpusContractError, match="must set compiler_failure_reason to null"
    ):
        validate_admitted_record_envelope(contract, failed_compilation)


def test_partition_resolution_accepts_assigned_and_retained_reject_outcomes() -> None:
    contract = _contract()
    assigned = {
        "status": "assigned",
        "assigned_role": "train",
        "source_endpoint_role": "train",
        "target_endpoint_role": "train",
        "split_component_id": "component-1",
        "split_assignment_manifest_sha256": "a" * 64,
    }
    validate_partition_resolution(contract, assigned, admitted_record=True)

    cross_partition = {
        "status": "cross_partition_rejected",
        "assigned_role": None,
        "source_endpoint_role": "train",
        "target_endpoint_role": "validation",
        "split_component_id": "candidate-component-2",
        "split_assignment_manifest_sha256": "a" * 64,
    }
    validate_partition_resolution(
        contract,
        cross_partition,
        admitted_record=False,
    )

    unassigned = {
        "status": "unassigned_rejected",
        "assigned_role": None,
        "source_endpoint_role": "train",
        "target_endpoint_role": None,
        "split_component_id": "candidate-component-3",
        "split_assignment_manifest_sha256": "a" * 64,
    }
    validate_partition_resolution(contract, unassigned, admitted_record=False)


@pytest.mark.parametrize(
    ("mutation", "admitted_record", "match"),
    [
        (
            {"target_endpoint_role": "validation"},
            True,
            "must name the same partition role",
        ),
        (
            {
                "status": "cross_partition_rejected",
                "assigned_role": None,
                "source_endpoint_role": "train",
                "target_endpoint_role": "validation",
                "split_component_id": "candidate-component-2",
            },
            True,
            "only an assigned partition resolution",
        ),
        (
            {
                "status": "cross_partition_rejected",
                "assigned_role": None,
                "source_endpoint_role": "train",
                "target_endpoint_role": "train",
                "split_component_id": "candidate-component-2",
            },
            False,
            "requires distinct endpoint roles",
        ),
        (
            {
                "status": "unassigned_rejected",
                "assigned_role": None,
                "source_endpoint_role": "train",
                "target_endpoint_role": "validation",
                "split_component_id": "candidate-component-2",
            },
            False,
            "at least one unassigned endpoint role",
        ),
        (
            {
                "status": "cross_partition_rejected",
                "assigned_role": "train",
                "source_endpoint_role": "train",
                "target_endpoint_role": "validation",
                "split_component_id": "candidate-component-2",
            },
            False,
            "assigned_role to null",
        ),
    ],
)
def test_partition_resolution_rejects_fabricated_or_inconsistent_roles(
    mutation: dict[str, Any],
    admitted_record: bool,
    match: str,
) -> None:
    contract = _contract()
    resolution = {
        "status": "assigned",
        "assigned_role": "train",
        "source_endpoint_role": "train",
        "target_endpoint_role": "train",
        "split_component_id": "component-1",
        "split_assignment_manifest_sha256": "a" * 64,
    }
    resolution.update(mutation)
    with pytest.raises(EditingCorpusContractError, match=match):
        validate_partition_resolution(
            contract,
            resolution,
            admitted_record=admitted_record,
        )


def test_partition_resolution_rejects_unknown_fields_and_bad_manifest_hash() -> None:
    contract = _contract()
    resolution = {
        "status": "assigned",
        "assigned_role": "train",
        "source_endpoint_role": "train",
        "target_endpoint_role": "train",
        "split_component_id": "component-1",
        "split_assignment_manifest_sha256": "not-a-hash",
    }
    with pytest.raises(EditingCorpusContractError, match="full lowercase SHA-256"):
        validate_partition_resolution(contract, resolution, admitted_record=True)
    resolution["split_assignment_manifest_sha256"] = "a" * 64
    resolution["partition"] = "train"
    with pytest.raises(EditingCorpusContractError, match="fields disagree"):
        validate_partition_resolution(contract, resolution, admitted_record=True)


def test_partition_status_rule_drift_is_rejected() -> None:
    broken = copy.deepcopy(_contract())
    broken["partition_resolution_contract"]["status_rules"][1]["admitted_record_policy"] = (
        "eligible"
    )
    with pytest.raises(EditingCorpusContractError, match="status rules have drifted"):
        validate_editing_corpus_contract(broken)


def test_e3_e4_dynamic_and_pareto_coverage_have_the_intended_granularity() -> None:
    contract = _contract()
    capabilities = {
        capability["id"]: capability for capability in contract["capability_coverage_contracts"]
    }
    cardinality = capabilities["cardinality_growth_and_shrinkage"]
    topology = capabilities["ring_and_topology_adaptation"]
    dynamic = capabilities["dynamic_correction_and_partial_reversal"]
    pareto = capabilities["pareto_shared_prefix_branching"]

    assert "E3" in cardinality["experiments"]
    assert "coupled_cardinality_topology_path" in cardinality["required_corpus_record_cells"]
    assert "E4" in topology["experiments"]
    assert "coupled_cardinality_topology_path" in topology["required_corpus_record_cells"]
    assert dynamic["required_corpus_record_cells"] == []
    assert dynamic["required_relationship_group_types"] == [
        "inverse_pair",
        "correction",
    ]
    assert capabilities["multi_operator_compositional_transport"][
        "required_relationship_group_types"
    ] == ["alternative_route"]
    assert pareto["required_corpus_record_cells"] == []
    assert pareto["required_corpus_direction_cells"] == []
    assert pareto["required_relationship_group_types"] == ["shared_prefix_branch"]
    assert pareto["coverage_scope"] == "controller_neutral_prior_relationship"


def test_relationship_definitions_freeze_semantics_and_post_active8_row_shape() -> None:
    contract = _contract()
    definitions = {
        item["id"]: item for item in contract["relationship_coverage_contract"]["group_types"]
    }
    assert definitions["shared_prefix_branch"]["semantic_requirements"] == [
        "exact_anchored_prefix_identity",
        "divergent_continuations",
        "minimum_two_branch_continuations",
    ]
    assert definitions["alternative_route"]["semantic_requirements"][:2] == [
        "exact_source_endpoint_identity",
        "exact_target_endpoint_identity",
    ]
    assert definitions["inverse_pair"]["semantic_requirements"] == [
        "swapped_exact_endpoint_identity",
        "one_forward_and_one_reverse_member",
        "actionwise_inverse_not_required",
    ]
    assert definitions["correction"]["semantic_requirements"][-1] == ("exact_return_not_required")
    assert (
        definitions["constraint_compatible_alternative_route"]["verification_receipt_type"]
        == "constraint_evaluator_receipt"
    )
    ledger = contract["relationship_coverage_contract"]["post_active8_ledger_schema"]
    assert ledger["stage"] == "post_active8"
    assert ledger["coverage_count_policy"] == "complete_verified_groups_only"
    assert ledger["group_fields"] == [
        "type",
        "group_id",
        "definition_sha256",
        "evidence_profile_id",
        "verification_receipt",
        "members",
        "group_sha256",
    ]


def test_relationship_definition_semantic_drift_is_rejected() -> None:
    broken = copy.deepcopy(_contract())
    inverse = next(
        item
        for item in broken["relationship_coverage_contract"]["group_types"]
        if item["id"] == "inverse_pair"
    )
    inverse["semantic_requirements"].remove("actionwise_inverse_not_required")
    with pytest.raises(
        EditingCorpusContractError,
        match="relationship group definition has drifted",
    ):
        validate_editing_corpus_contract(broken)


def test_complete_relationship_groups_validate_for_all_frozen_types() -> None:
    contract = _contract()
    groups = [
        _relationship_group(
            contract,
            "shared_prefix_branch",
            [
                _relationship_member(
                    "shared_prefix_anchor",
                    trace_id="trace-anchor",
                    entry=0,
                    start=0,
                    end=1,
                ),
                _relationship_member(
                    "branch_continuation",
                    trace_id="trace-branch-a",
                    entry=1,
                    start=1,
                    end=2,
                ),
                _relationship_member(
                    "branch_continuation",
                    trace_id="trace-branch-b",
                    entry=2,
                    start=1,
                    end=2,
                ),
            ],
        ),
        _relationship_group(
            contract,
            "alternative_route",
            [
                _relationship_member(
                    "route",
                    trace_id="trace-route-a",
                    entry=0,
                    start=0,
                    end=2,
                ),
                _relationship_member(
                    "route",
                    trace_id="trace-route-b",
                    entry=1,
                    start=0,
                    end=3,
                ),
            ],
        ),
        _relationship_group(
            contract,
            "inverse_pair",
            [
                _relationship_member(
                    "forward",
                    trace_id="trace-forward",
                    entry=0,
                    start=0,
                    end=2,
                ),
                _relationship_member(
                    "reverse",
                    trace_id="trace-reverse",
                    entry=1,
                    start=0,
                    end=3,
                ),
            ],
        ),
        _relationship_group(
            contract,
            "correction",
            [
                _relationship_member(
                    "initial_change",
                    trace_id="trace-correction",
                    entry=0,
                    start=0,
                    end=1,
                ),
                _relationship_member(
                    "corrective_continuation",
                    trace_id="trace-correction",
                    entry=0,
                    start=1,
                    end=3,
                ),
            ],
        ),
        _relationship_group(
            contract,
            "constraint_compatible_alternative_route",
            [
                _relationship_member(
                    "constraint_feasible_route",
                    trace_id="trace-constraint-a",
                    entry=0,
                    start=0,
                    end=2,
                ),
                _relationship_member(
                    "constraint_feasible_route",
                    trace_id="trace-constraint-b",
                    entry=1,
                    start=0,
                    end=3,
                ),
            ],
        ),
    ]
    for group in groups:
        validate_relationship_group(contract, group)


def test_incomplete_relationship_groups_never_count() -> None:
    contract = _contract()
    shared_prefix = _relationship_group(
        contract,
        "shared_prefix_branch",
        [
            _relationship_member(
                "shared_prefix_anchor",
                trace_id="trace-anchor",
                entry=0,
                start=0,
                end=1,
            ),
            _relationship_member(
                "branch_continuation",
                trace_id="trace-branch-a",
                entry=1,
                start=1,
                end=2,
            ),
        ],
    )
    with pytest.raises(EditingCorpusContractError, match="incomplete"):
        validate_relationship_group(contract, shared_prefix)

    inverse = _relationship_group(
        contract,
        "inverse_pair",
        [
            _relationship_member(
                "forward",
                trace_id="trace-forward",
                entry=0,
                start=0,
                end=1,
            ),
            _relationship_member(
                "forward",
                trace_id="trace-other",
                entry=1,
                start=0,
                end=1,
            ),
        ],
    )
    with pytest.raises(EditingCorpusContractError, match="one forward and one reverse"):
        validate_relationship_group(contract, inverse)


def test_relationship_groups_reject_hash_receipt_and_contiguity_fabrication() -> None:
    contract = _contract()
    correction = _relationship_group(
        contract,
        "correction",
        [
            _relationship_member(
                "initial_change",
                trace_id="trace-correction",
                entry=0,
                start=0,
                end=1,
            ),
            _relationship_member(
                "corrective_continuation",
                trace_id="trace-correction",
                entry=0,
                start=2,
                end=3,
            ),
        ],
    )
    with pytest.raises(EditingCorpusContractError, match="must be contiguous"):
        validate_relationship_group(contract, correction)

    bad_receipt = copy.deepcopy(correction)
    bad_receipt["verification_receipt"]["receipt_type"] = "constraint_evaluator_receipt"
    _rehash_relationship_group(bad_receipt)
    with pytest.raises(EditingCorpusContractError, match="receipt type disagrees"):
        validate_relationship_group(contract, bad_receipt)

    bad_member_hash = copy.deepcopy(correction)
    bad_member_hash["members"][0]["member_sha256"] = "f" * 64
    with pytest.raises(EditingCorpusContractError, match="member hash disagrees"):
        validate_relationship_group(contract, bad_member_hash)

    bad_definition = copy.deepcopy(correction)
    bad_definition["definition_sha256"] = "f" * 64
    _rehash_relationship_group(bad_definition)
    with pytest.raises(EditingCorpusContractError, match="definition_sha256"):
        validate_relationship_group(contract, bad_definition)


def test_relationship_group_rejects_unavailable_evidence_profile() -> None:
    contract = _contract()
    group = _relationship_group(
        contract,
        "alternative_route",
        [
            _relationship_member(
                "route",
                trace_id="trace-a",
                entry=0,
                start=0,
                end=1,
            ),
            _relationship_member(
                "route",
                trace_id="trace-b",
                entry=1,
                start=0,
                end=1,
            ),
        ],
    )
    group["evidence_profile_id"] = "genuine_observed_series_compiled_path"
    _rehash_relationship_group(group)
    with pytest.raises(EditingCorpusContractError, match="unavailable evidence profile"):
        validate_relationship_group(contract, group)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("unexpected",), True, "fields disagree"),
        (("data_lanes", 0, "unexpected"), "value", "fields disagree"),
        (("training_authorized",), "false", "must be a boolean"),
        (("scientific_scope", "max_active_atoms"), True, "must equal 40"),
        (("data_lanes",), {}, "must be a list"),
    ],
)
def test_contract_fails_closed_on_unknown_keys_and_types(
    path: tuple[str | int, ...],
    value: object,
    match: str,
) -> None:
    broken = copy.deepcopy(_contract())
    target: Any = broken
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(EditingCorpusContractError, match=match):
        validate_editing_corpus_contract(broken)


@pytest.mark.parametrize(
    "stale_term",
    [
        "inverse_record_id",
        "branch_group_required",
        "required_endpoint_deltas",
        "observed_series_path",
    ],
)
def test_contract_rejects_stale_schema_v1_terms(stale_term: str) -> None:
    broken = copy.deepcopy(_contract())
    broken["admitted_record_contract"]["required_fields"].append(stale_term)
    with pytest.raises(EditingCorpusContractError, match="schema-v1 terms"):
        validate_editing_corpus_contract(broken)


@pytest.mark.parametrize(
    ("family", "match"),
    [
        ("ring_system_delete", "disabled families"),
        ("invented_operator", "unknown families"),
    ],
)
def test_contract_rejects_disabled_or_unknown_family_references(
    family: str,
    match: str,
) -> None:
    broken = copy.deepcopy(_contract())
    broken["record_membership_contract"]["cells"][0]["active8_families"].append(family)
    with pytest.raises(EditingCorpusContractError, match=match):
        validate_editing_corpus_contract(broken)


def test_contract_rejects_active8_drift() -> None:
    broken = copy.deepcopy(_contract())
    broken["operator_basis"]["active8"].remove("ring_system_restate")
    with pytest.raises(EditingCorpusContractError, match="exact Active8"):
        validate_editing_corpus_contract(broken)


def test_contract_rejects_unknown_or_unavailable_evidence_profile_admission() -> None:
    unknown = copy.deepcopy(_contract())
    unknown["data_lanes"][0]["admissible_evidence_profiles"] = ["invented_profile"]
    with pytest.raises(EditingCorpusContractError, match="unknown evidence profiles"):
        validate_editing_corpus_contract(unknown)

    unavailable = copy.deepcopy(_contract())
    unavailable["data_lanes"][0]["admissible_evidence_profiles"] = [
        "observed_pair_observed_actions"
    ]
    unavailable["data_lanes"][0]["reserved_evidence_profiles"] = []
    with pytest.raises(EditingCorpusContractError, match="admits unavailable profile"):
        validate_editing_corpus_contract(unavailable)


def test_contract_rejects_evidence_component_upgrade_by_relabeling() -> None:
    broken = copy.deepcopy(_contract())
    profiles = broken["evidence_component_contract"]["profiles"]
    inferred = next(
        profile for profile in profiles if profile["id"] == "inferred_relation_compiled_path"
    )
    inferred["components"]["pair_relationship"] = "observed_real_endpoint_pair"
    with pytest.raises(EditingCorpusContractError, match="profile definition has drifted"):
        validate_editing_corpus_contract(broken)


def test_contract_cannot_relabel_external_authority_as_verified() -> None:
    broken = copy.deepcopy(_contract())
    broken["admitted_record_contract"]["external_authority_requirements"][0]["status"] = "verified"
    with pytest.raises(
        EditingCorpusContractError,
        match="external authority requirements have drifted",
    ):
        validate_editing_corpus_contract(broken)


def test_one_record_cannot_claim_opposing_direction_cells() -> None:
    contract = _contract()
    validate_record_membership_assignment(
        contract,
        ("cardinality_growth", "cycle_rank_increase"),
    )
    with pytest.raises(EditingCorpusContractError, match="opposing atom_count"):
        validate_record_membership_assignment(
            contract,
            ("cardinality_growth", "cardinality_shrinkage"),
        )
    with pytest.raises(EditingCorpusContractError, match="opposing graph_cycle_rank"):
        validate_record_membership_assignment(
            contract,
            ("cycle_rank_increase", "cycle_rank_decrease"),
        )


def test_record_membership_assignment_rejects_unknown_and_duplicate_cells() -> None:
    contract = _contract()
    with pytest.raises(EditingCorpusContractError, match="at least one"):
        validate_record_membership_assignment(contract, ())
    with pytest.raises(EditingCorpusContractError, match="unknown cells"):
        validate_record_membership_assignment(contract, ("invented_cell",))
    with pytest.raises(EditingCorpusContractError, match="duplicates"):
        validate_record_membership_assignment(
            contract,
            ("attachment_relocation", "attachment_relocation"),
        )


def test_contract_rejects_per_record_opposing_direction_definition_drift() -> None:
    broken = copy.deepcopy(_contract())
    exclusions = broken["record_membership_contract"]["opposing_direction_memberships_forbidden"]
    exclusions[0]["record_cells"][1] = "cardinality_growth"
    with pytest.raises(EditingCorpusContractError, match="duplicates"):
        validate_editing_corpus_contract(broken)


@pytest.mark.parametrize(
    ("capability_id", "field", "replacement", "match"),
    [
        (
            "dynamic_correction_and_partial_reversal",
            "required_relationship_group_types",
            ["inverse_pair"],
            "capability coverage definition has drifted",
        ),
        (
            "dynamic_correction_and_partial_reversal",
            "required_corpus_record_cells",
            ["mixed_operator_path"],
            "capability coverage definition has drifted",
        ),
        (
            "pareto_shared_prefix_branching",
            "required_relationship_group_types",
            ["alternative_route"],
            "capability coverage definition has drifted",
        ),
        (
            "pareto_shared_prefix_branching",
            "required_corpus_record_cells",
            ["mixed_operator_path"],
            "capability coverage definition has drifted",
        ),
        (
            "ring_and_topology_adaptation",
            "experiments",
            ["E7"],
            "capability coverage definition has drifted",
        ),
    ],
)
def test_contract_rejects_missing_e2_e3_e4_e7_or_relationship_coverage(
    capability_id: str,
    field: str,
    replacement: list[str],
    match: str,
) -> None:
    broken = copy.deepcopy(_contract())
    capability = next(
        item for item in broken["capability_coverage_contracts"] if item["id"] == capability_id
    )
    capability[field] = replacement
    with pytest.raises(EditingCorpusContractError, match=match):
        validate_editing_corpus_contract(broken)


@pytest.mark.parametrize(
    ("field", "unknown_value", "match"),
    [
        ("evidence_lanes", "invented_lane", "unknown lanes"),
        ("required_corpus_record_cells", "invented_cell", "unknown record cells"),
        (
            "required_corpus_direction_cells",
            "invented_direction",
            "unknown direction cells",
        ),
        (
            "required_relationship_group_types",
            "invented_relationship",
            "unknown relationship types",
        ),
        ("threshold_refs", "invented_threshold", "unknown thresholds"),
    ],
)
def test_contract_rejects_unknown_capability_references(
    field: str,
    unknown_value: str,
    match: str,
) -> None:
    broken = copy.deepcopy(_contract())
    broken["capability_coverage_contracts"][0][field].append(unknown_value)
    with pytest.raises(EditingCorpusContractError, match=match):
        validate_editing_corpus_contract(broken)


@pytest.mark.parametrize(
    "roles",
    [
        ["train", "validation", "final_test"],
        ["train", "controller_validation", "validation", "final_test"],
    ],
)
def test_contract_rejects_missing_or_reordered_partition_roles(
    roles: list[str],
) -> None:
    broken = copy.deepcopy(_contract())
    broken["split_contract"]["partition_roles"] = roles
    with pytest.raises(EditingCorpusContractError, match="partition_roles"):
        validate_editing_corpus_contract(broken)


@pytest.mark.parametrize("bad_value", ["1", True, -1, float("inf")])
def test_contract_rejects_invalid_threshold_types_or_ranges(
    bad_value: object,
) -> None:
    broken = copy.deepcopy(_contract())
    broken["pretraining_gates"]["threshold_values"][
        "minimum_records_per_required_membership_cell"
    ] = bad_value
    with pytest.raises(EditingCorpusContractError, match="threshold_values"):
        validate_editing_corpus_contract(broken)


def test_every_unresolved_threshold_reference_blocks_launch() -> None:
    contract = _contract()
    blockers = training_launch_blockers(contract)
    null_thresholds = {
        name
        for name, value in contract["pretraining_gates"]["threshold_values"].items()
        if value is None
    }
    blocked_thresholds = {
        blocker.rsplit(" -> ", 1)[1]
        for blocker in blockers
        if blocker.startswith("unresolved threshold reference:")
    }
    assert blocked_thresholds == null_thresholds


def test_capability_cannot_drop_a_semantically_required_threshold_reference() -> None:
    broken = copy.deepcopy(_contract())
    pareto = next(
        item
        for item in broken["capability_coverage_contracts"]
        if item["id"] == "pareto_shared_prefix_branching"
    )
    pareto["threshold_refs"].remove("minimum_shared_prefix_branch_groups")
    with pytest.raises(
        EditingCorpusContractError,
        match="omits required threshold references",
    ):
        validate_editing_corpus_contract(broken)


def test_resolving_thresholds_does_not_override_status_or_external_authority() -> None:
    contract = copy.deepcopy(_contract())
    _freeze_thresholds(contract)
    blockers = training_launch_blockers(contract)
    assert not any(item.startswith("unresolved threshold reference:") for item in blockers)
    assert "training_authorized is not true" in blockers
    assert "status is not FROZEN_TRAINING_AUTHORIZED" in blockers
    assert any("constraint_feasible_path" in item for item in blockers)
    assert any("trace_derived_record_membership" in item for item in blockers)
    assert any("physical_relationship_receipt_resolution" in item for item in blockers)


def test_external_authority_gaps_block_even_an_asserted_frozen_status() -> None:
    contract = copy.deepcopy(_contract())
    _freeze_thresholds(contract)
    contract["status"] = "FROZEN_TRAINING_AUTHORIZED"
    contract["training_authorized"] = True
    blockers = training_launch_blockers(contract)
    assert blockers == [
        "unresolved record-membership cell: constraint_feasible_path "
        "(unresolved_no_go_pending_constraint_evaluator)",
        "unresolved external authority requirement: "
        "authoritative_trace_derived_record_membership_classification "
        "(unresolved_no_go_pending_trace_derived_predicate_receipts)",
        "unresolved external authority requirement: "
        "physical_relationship_receipt_resolution "
        "(unresolved_no_go_pending_physical_receipt_resolver)",
    ]
    with pytest.raises(EditingCorpusContractError, match="trace_derived"):
        assert_training_launch_authorized(contract)
