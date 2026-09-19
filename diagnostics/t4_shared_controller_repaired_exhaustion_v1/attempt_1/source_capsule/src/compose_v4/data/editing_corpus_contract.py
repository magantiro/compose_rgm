"""Fail-closed validation for the versioned COMPOSE editing-corpus contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "compose.editing_corpus_contract"
EXPECTED_SCHEMA_VERSION = 2
REQUIRED_EXPERIMENTS = frozenset({"E2", "E3", "E4", "E7"})
REQUIRED_PARTITION_ROLES = (
    "train",
    "validation",
    "controller_validation",
    "final_test",
)
REQUIRED_DATA_LANES = (
    "observed_local_analogue",
    "operator_aware_real_endpoint",
    "linker_positional_topology_analogue",
    "real_endpoint_multistep_path",
    "reversible_synthetic_walk",
)
ACTIVE8_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
    "ring_system_restate",
)
DISABLED_FAMILIES = ("ring_system_delete", "ring_system_grow")

_TOP_LEVEL_FIELDS = {
    "schema",
    "schema_version",
    "contract_id",
    "status",
    "scientific_scope",
    "operator_basis",
    "evidence_component_contract",
    "data_lanes",
    "record_membership_contract",
    "relationship_coverage_contract",
    "corpus_direction_contract",
    "capability_coverage_contracts",
    "partition_resolution_contract",
    "admitted_record_contract",
    "split_contract",
    "sampling_contract",
    "pretraining_gates",
    "training_authorized",
}
_SCIENTIFIC_SCOPE_FIELDS = {
    "model",
    "controller_only_inputs",
    "separate_lanes",
    "max_active_atoms",
    "stereochemistry",
    "formal_charge_changes",
}
_CONTROLLER_ONLY_INPUTS = (
    "immutable_source",
    "goal_or_preference",
    "remaining_budget",
    "protected_mapping",
    "path_constraint_state",
)
_SUPPORT_EXPANSIONS = (
    "multi_neighbor_atom_insert",
    "fragment_replace_macro",
    "ring_template_macro",
)
_EVIDENCE_COMPONENT_VOCABULARIES = {
    "source_endpoint": (
        "observed_real_endpoint",
        "executor_generated_endpoint",
    ),
    "target_endpoint": (
        "observed_real_endpoint",
        "executor_generated_endpoint",
    ),
    "pair_relationship": (
        "observed_real_endpoint_pair",
        "observed_real_series_relationship",
        "structurally_inferred_relationship",
        "executor_generated_pair",
    ),
    "path": (
        "observed_source_path",
        "compiled_path_between_real_endpoints",
        "compiled_path_through_observed_real_intermediates",
        "executor_generated_walk",
    ),
    "intermediates": (
        "observed_real_intermediates",
        "executor_generated_intermediates",
    ),
    "action_sequence": (
        "observed_source_action_sequence",
        "compiler_generated_action_sequence",
        "executor_generated_action_sequence",
    ),
}
_EVIDENCE_PROFILES = {
    "observed_pair_observed_actions": (
        "inadmissible_missing_true_action_source_manifest",
        {
            "source_endpoint": "observed_real_endpoint",
            "target_endpoint": "observed_real_endpoint",
            "pair_relationship": "observed_real_endpoint_pair",
            "path": "observed_source_path",
            "intermediates": "observed_real_intermediates",
            "action_sequence": "observed_source_action_sequence",
        },
    ),
    "observed_pair_compiled_path": (
        "admissible",
        {
            "source_endpoint": "observed_real_endpoint",
            "target_endpoint": "observed_real_endpoint",
            "pair_relationship": "observed_real_endpoint_pair",
            "path": "compiled_path_between_real_endpoints",
            "intermediates": "executor_generated_intermediates",
            "action_sequence": "compiler_generated_action_sequence",
        },
    ),
    "genuine_observed_series_compiled_path": (
        "unavailable_missing_genuine_series_provenance",
        {
            "source_endpoint": "observed_real_endpoint",
            "target_endpoint": "observed_real_endpoint",
            "pair_relationship": "observed_real_series_relationship",
            "path": "compiled_path_through_observed_real_intermediates",
            "intermediates": "observed_real_intermediates",
            "action_sequence": "compiler_generated_action_sequence",
        },
    ),
    "inferred_relation_compiled_path": (
        "admissible",
        {
            "source_endpoint": "observed_real_endpoint",
            "target_endpoint": "observed_real_endpoint",
            "pair_relationship": "structurally_inferred_relationship",
            "path": "compiled_path_between_real_endpoints",
            "intermediates": "executor_generated_intermediates",
            "action_sequence": "compiler_generated_action_sequence",
        },
    ),
    "executor_generated_walk": (
        "admissible",
        {
            "source_endpoint": "observed_real_endpoint",
            "target_endpoint": "executor_generated_endpoint",
            "pair_relationship": "executor_generated_pair",
            "path": "executor_generated_walk",
            "intermediates": "executor_generated_intermediates",
            "action_sequence": "executor_generated_action_sequence",
        },
    ),
}
_LANE_PROFILE_ASSIGNMENTS = {
    "observed_local_analogue": (
        ("observed_pair_compiled_path", "inferred_relation_compiled_path"),
        ("observed_pair_observed_actions",),
    ),
    "operator_aware_real_endpoint": (
        ("observed_pair_compiled_path", "inferred_relation_compiled_path"),
        ("observed_pair_observed_actions",),
    ),
    "linker_positional_topology_analogue": (
        ("observed_pair_compiled_path", "inferred_relation_compiled_path"),
        (),
    ),
    "real_endpoint_multistep_path": (
        ("observed_pair_compiled_path", "inferred_relation_compiled_path"),
        (
            "observed_pair_observed_actions",
            "genuine_observed_series_compiled_path",
        ),
    ),
    "reversible_synthetic_walk": (("executor_generated_walk",), ()),
}
_RECORD_CELL_DEFINITIONS = {
    "local_substituent_or_bioisostere": (
        "local_substituent_or_bioisostere_v1",
        ("atom_insert", "atom_delete", "atom_restate", "bond_reorder"),
        "none",
        "none",
        "admissible",
    ),
    "cardinality_growth": (
        "cardinality_growth_v1",
        ("atom_insert",),
        "atom_count",
        "increase",
        "admissible",
    ),
    "cardinality_shrinkage": (
        "cardinality_shrinkage_v1",
        ("atom_delete",),
        "atom_count",
        "decrease",
        "admissible",
    ),
    "attachment_relocation": (
        "attachment_relocation_v1",
        ("bond_reroute",),
        "none",
        "none",
        "admissible",
    ),
    "electronic_or_bond_state": (
        "electronic_or_bond_state_v1",
        ("atom_restate", "bond_reorder", "ring_system_restate"),
        "none",
        "none",
        "admissible",
    ),
    "cycle_rank_increase": (
        "cycle_rank_increase_v1",
        ("cycle_insert",),
        "graph_cycle_rank",
        "increase",
        "admissible",
    ),
    "cycle_rank_decrease": (
        "cycle_rank_decrease_v1",
        ("cycle_attach",),
        "graph_cycle_rank",
        "decrease",
        "admissible",
    ),
    "topology_restructure": (
        "topology_restructure_v1",
        ("cycle_insert", "cycle_attach", "ring_system_restate"),
        "none",
        "none",
        "admissible",
    ),
    "mixed_operator_path": (
        "mixed_operator_path_v1",
        ACTIVE8_FAMILIES,
        "none",
        "none",
        "admissible",
    ),
    "coupled_cardinality_topology_path": (
        "coupled_cardinality_topology_path_v1",
        (
            "atom_insert",
            "atom_delete",
            "cycle_insert",
            "cycle_attach",
            "ring_system_restate",
        ),
        "none",
        "none",
        "admissible",
    ),
    "constraint_feasible_path": (
        "constraint_feasible_path_v1",
        ACTIVE8_FAMILIES,
        "none",
        "none",
        "unresolved_no_go_pending_constraint_evaluator",
    ),
}
_OPPOSING_DIRECTION_CELLS = {
    "atom_count": ("cardinality_growth", "cardinality_shrinkage"),
    "graph_cycle_rank": ("cycle_rank_increase", "cycle_rank_decrease"),
}
_RELATIONSHIP_GROUP_TYPES = {
    "shared_prefix_branch": {
        "minimum_distinct_members": 3,
        "construction_scope": "controller_neutral_prior_corpus",
        "member_roles": ("shared_prefix_anchor", "branch_continuation"),
        "semantic_requirements": (
            "exact_anchored_prefix_identity",
            "divergent_continuations",
            "minimum_two_branch_continuations",
        ),
        "verification_receipt_type": "shared_prefix_divergence_receipt",
    },
    "alternative_route": {
        "minimum_distinct_members": 2,
        "construction_scope": "corpus_relationship",
        "member_roles": ("route",),
        "semantic_requirements": (
            "exact_source_endpoint_identity",
            "exact_target_endpoint_identity",
            "distinct_route_identity",
        ),
        "verification_receipt_type": "shared_endpoints_distinct_routes_receipt",
    },
    "inverse_pair": {
        "minimum_distinct_members": 2,
        "construction_scope": "corpus_relationship",
        "member_roles": ("forward", "reverse"),
        "semantic_requirements": (
            "swapped_exact_endpoint_identity",
            "one_forward_and_one_reverse_member",
            "actionwise_inverse_not_required",
        ),
        "verification_receipt_type": "swapped_endpoints_receipt",
    },
    "correction": {
        "minimum_distinct_members": 2,
        "construction_scope": "corpus_relationship",
        "member_roles": ("initial_change", "corrective_continuation"),
        "semantic_requirements": (
            "contiguous_anchored_segments",
            "initial_change_precedes_corrective_continuation",
            "correction_predicate_receipt_required",
            "exact_return_not_required",
        ),
        "verification_receipt_type": "correction_predicate_receipt",
    },
    "constraint_compatible_alternative_route": {
        "minimum_distinct_members": 2,
        "construction_scope": "corpus_relationship",
        "member_roles": ("constraint_feasible_route",),
        "semantic_requirements": (
            "exact_source_endpoint_identity",
            "exact_target_endpoint_identity",
            "distinct_route_identity",
            "constraint_evaluator_receipt_required",
        ),
        "verification_receipt_type": "constraint_evaluator_receipt",
    },
}
_RELATIONSHIP_LEDGER_GROUP_FIELDS = (
    "type",
    "group_id",
    "definition_sha256",
    "evidence_profile_id",
    "verification_receipt",
    "members",
    "group_sha256",
)
_RELATIONSHIP_LEDGER_RECEIPT_FIELDS = (
    "receipt_type",
    "verifier_identity_sha256",
    "receipt_sha256",
)
_RELATIONSHIP_LEDGER_MEMBER_FIELDS = (
    "trace_key",
    "segment",
    "role",
    "member_sha256",
)
_RELATIONSHIP_LEDGER_TRACE_KEY_FIELDS = (
    "packed_shard_sha256",
    "entry",
    "trace_id",
)
_RELATIONSHIP_LEDGER_SEGMENT_FIELDS = (
    "start_progress_index",
    "end_progress_index",
)
_CORPUS_DIRECTION_CELLS = {
    "cardinality_bidirectional": (
        "atom_count",
        ("cardinality_growth", "cardinality_shrinkage"),
    ),
    "cycle_rank_bidirectional": (
        "graph_cycle_rank",
        ("cycle_rank_increase", "cycle_rank_decrease"),
    ),
}
_CAPABILITY_REQUIREMENTS = {
    "local_substituent_and_bioisostere": {
        "experiments": ("E2", "E7"),
        "evidence_lanes": (
            "observed_local_analogue",
            "operator_aware_real_endpoint",
            "reversible_synthetic_walk",
        ),
        "record_cells": ("local_substituent_or_bioisostere",),
        "direction_cells": (),
        "relationships": (),
        "coverage_scope": "capability_evidence",
    },
    "cardinality_growth_and_shrinkage": {
        "experiments": ("E3", "E7"),
        "evidence_lanes": (
            "observed_local_analogue",
            "operator_aware_real_endpoint",
            "real_endpoint_multistep_path",
            "reversible_synthetic_walk",
        ),
        "record_cells": (
            "cardinality_growth",
            "cardinality_shrinkage",
            "coupled_cardinality_topology_path",
        ),
        "direction_cells": ("cardinality_bidirectional",),
        "relationships": (),
        "coverage_scope": "capability_evidence",
    },
    "linker_and_attachment_relocation": {
        "experiments": ("E2", "E7"),
        "evidence_lanes": (
            "operator_aware_real_endpoint",
            "linker_positional_topology_analogue",
            "real_endpoint_multistep_path",
            "reversible_synthetic_walk",
        ),
        "record_cells": ("attachment_relocation",),
        "direction_cells": (),
        "relationships": (),
        "coverage_scope": "capability_evidence",
    },
    "electronic_and_bond_state_adaptation": {
        "experiments": ("E2", "E7"),
        "evidence_lanes": (
            "observed_local_analogue",
            "operator_aware_real_endpoint",
            "real_endpoint_multistep_path",
            "reversible_synthetic_walk",
        ),
        "record_cells": ("electronic_or_bond_state",),
        "direction_cells": (),
        "relationships": (),
        "coverage_scope": "capability_evidence",
    },
    "ring_and_topology_adaptation": {
        "experiments": ("E4", "E7"),
        "evidence_lanes": (
            "operator_aware_real_endpoint",
            "linker_positional_topology_analogue",
            "real_endpoint_multistep_path",
            "reversible_synthetic_walk",
        ),
        "record_cells": (
            "cycle_rank_increase",
            "cycle_rank_decrease",
            "topology_restructure",
            "coupled_cardinality_topology_path",
        ),
        "direction_cells": ("cycle_rank_bidirectional",),
        "relationships": (),
        "coverage_scope": "capability_evidence",
    },
    "multi_operator_compositional_transport": {
        "experiments": ("E2", "E7"),
        "evidence_lanes": (
            "real_endpoint_multistep_path",
            "reversible_synthetic_walk",
        ),
        "record_cells": ("mixed_operator_path",),
        "direction_cells": (),
        "relationships": ("alternative_route",),
        "coverage_scope": "capability_evidence",
    },
    "dynamic_correction_and_partial_reversal": {
        "experiments": ("E3", "E4", "E7"),
        "evidence_lanes": (
            "real_endpoint_multistep_path",
            "reversible_synthetic_walk",
        ),
        "record_cells": (),
        "direction_cells": (),
        "relationships": ("inverse_pair", "correction"),
        "coverage_scope": "capability_evidence",
    },
    "constraint_compatible_alternative_routes": {
        "experiments": ("E7",),
        "evidence_lanes": (
            "real_endpoint_multistep_path",
            "reversible_synthetic_walk",
        ),
        "record_cells": ("constraint_feasible_path",),
        "direction_cells": (),
        "relationships": ("constraint_compatible_alternative_route",),
        "coverage_scope": "capability_evidence",
    },
    "pareto_shared_prefix_branching": {
        "experiments": ("E7",),
        "evidence_lanes": REQUIRED_DATA_LANES,
        "record_cells": (),
        "direction_cells": (),
        "relationships": ("shared_prefix_branch",),
        "coverage_scope": "controller_neutral_prior_relationship",
    },
}
_THRESHOLD_NAMES = {
    "minimum_records_per_required_membership_cell",
    "minimum_records_per_direction_member",
    "minimum_unique_sources_per_capability",
    "minimum_unique_scaffolds_per_capability",
    "minimum_effective_teacher_coefficient_per_capability",
    "minimum_relationship_groups_per_required_type",
    "minimum_real_endpoint_multistep_path_share",
    "minimum_coupled_cardinality_topology_paths",
    "minimum_constraint_feasible_paths",
    "minimum_shared_prefix_branch_groups",
    "maximum_single_series_share",
    "maximum_single_transformation_share",
    "maximum_delete_insert_fallback_share_where_direct_semantic_path_exists",
}
_GLOBAL_THRESHOLD_REFS = (
    "maximum_single_series_share",
    "maximum_single_transformation_share",
    "maximum_delete_insert_fallback_share_where_direct_semantic_path_exists",
)
_PARTITION_RESOLUTION_FIELDS = (
    "status",
    "assigned_role",
    "source_endpoint_role",
    "target_endpoint_role",
    "split_component_id",
    "split_assignment_manifest_sha256",
)
_PARTITION_RESOLUTION_STATUSES = (
    "assigned",
    "cross_partition_rejected",
    "unassigned_rejected",
)
_PARTITION_STATUS_RULES = {
    "assigned": (
        "required_partition_role",
        "both_equal_assigned_role",
        "required",
        "eligible",
    ),
    "cross_partition_rejected": (
        "must_be_null",
        "both_partition_roles_distinct",
        "required",
        "forbidden",
    ),
    "unassigned_rejected": (
        "must_be_null",
        "at_least_one_null_other_may_be_partition_role",
        "required",
        "forbidden",
    ),
}
_REQUIRED_RECORD_FIELDS = (
    "record_id",
    "partition_resolution",
    "data_lane",
    "evidence_profile_id",
    "evidence_components",
    "source_state_exact",
    "successor_states_exact",
    "target_state_exact",
    "canonical_state_keys",
    "executable_actions",
    "operator_sequence",
    "operator_family_set",
    "record_membership_cells",
    "relationship_group_ids",
    "source_group_id",
    "scaffold_group_id",
    "series_group_id",
    "document_or_source_group_id",
    "transformation_signature",
    "constant_core_mapping",
    "protected_mapping",
    "atom_count_delta",
    "graph_cycle_rank_delta",
    "path_length",
    "maximum_intermediate_source_distance",
    "immediate_reversal_count",
    "repeated_state_count",
    "constraint_evaluator_identity",
    "compiler_candidates_considered",
    "compiler_choice_reason",
    "compiler_failure_reason",
    "source_provenance",
    "operator_contract_hash",
    "canonicalizer_hash",
    "compiler_hash",
)
_ADMITTED_RECORD_VALIDATION_SCOPE = (
    "structural_envelope_only_no_trace_predicate_recomputation_or_relationship_receipt_resolution"
)
_EXTERNAL_AUTHORITY_REQUIREMENTS = (
    (
        "authoritative_trace_derived_record_membership_classification",
        "unresolved_no_go_pending_trace_derived_predicate_receipts",
    ),
    (
        "physical_relationship_receipt_resolution",
        "unresolved_no_go_pending_physical_receipt_resolver",
    ),
)
_STALE_V1_TERMS = {
    "observed_series_path",
    "evidence_class",
    "required_families_all",
    "required_families_any",
    "real_endpoint_required",
    "mixed_family_path_required",
    "branch_group_required",
    "alternative_path_group_required",
    "inverse_or_correction_pair_required",
    "required_endpoint_deltas",
    "minimum_distinct_operator_families_per_path",
    "inverse_record_id",
}


class EditingCorpusContractError(ValueError):
    """The proposed corpus contract is malformed or not launch-ready."""


def load_editing_corpus_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    validate_editing_corpus_contract(payload)
    return payload


def _require_object(
    value: object,
    *,
    field: str,
    expected_fields: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingCorpusContractError(f"{field} must be an object")
    if expected_fields is not None and set(value) != expected_fields:
        raise EditingCorpusContractError(
            f"{field} fields disagree; "
            f"missing={sorted(expected_fields - set(value))}, "
            f"unexpected={sorted(set(value) - expected_fields)}"
        )
    return value


def _require_list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise EditingCorpusContractError(f"{field} must be a list")
    return value


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EditingCorpusContractError(
            f"{field} must be a nonempty, whitespace-normalized string"
        )
    return value


def _require_optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field=field)


def _require_string_list(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    items = _require_list(value, field=field)
    normalized = tuple(
        _require_text(item, field=f"{field}[{index}]") for index, item in enumerate(items)
    )
    if not allow_empty and not normalized:
        raise EditingCorpusContractError(f"{field} must not be empty")
    duplicates = _duplicates(normalized)
    if duplicates:
        raise EditingCorpusContractError(f"{field} contains duplicates: {duplicates}")
    return normalized


def _require_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise EditingCorpusContractError(f"{field} must be a boolean")
    return value


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EditingCorpusContractError(f"{field} must be an integer")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EditingCorpusContractError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EditingCorpusContractError(f"{field} must be a nonnegative integer")
    return value


def _require_finite_nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditingCorpusContractError(f"{field} must be a finite nonnegative number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise EditingCorpusContractError(f"{field} must be a finite nonnegative number")
    return numeric


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise EditingCorpusContractError(f"{field} must be a full lowercase SHA-256")
    return digest


def _require_nonempty_json_object(value: object, *, field: str) -> Mapping[str, Any]:
    payload = _require_object(value, field=field)
    if not payload:
        raise EditingCorpusContractError(f"{field} must not be empty")
    try:
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise EditingCorpusContractError(f"{field} must be JSON-serializable") from error
    return payload


def _require_slot_mapping(value: object, *, field: str) -> tuple[tuple[int, int], ...] | None:
    if value is None:
        return None
    pairs = _require_list(value, field=field)
    normalized: list[tuple[int, int]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise EditingCorpusContractError(f"{field}[{index}] must be a two-slot list")
        source_slot = _require_nonnegative_int(
            pair[0],
            field=f"{field}[{index}].source_slot",
        )
        target_slot = _require_nonnegative_int(
            pair[1],
            field=f"{field}[{index}].target_slot",
        )
        normalized.append((source_slot, target_slot))
    source_slots = [source for source, _target in normalized]
    target_slots = [target for _source, target in normalized]
    if len(set(source_slots)) != len(source_slots) or len(set(target_slots)) != len(target_slots):
        raise EditingCorpusContractError(f"{field} must be one-to-one")
    if tuple(normalized) != tuple(sorted(normalized)):
        raise EditingCorpusContractError(f"{field} must use deterministic sorted order")
    return tuple(normalized)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate)


def _find_stale_terms(value: object, *, path: str = "contract") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _STALE_V1_TERMS:
                findings.append(f"{path}.{key}")
            findings.extend(_find_stale_terms(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_stale_terms(child, path=f"{path}[{index}]"))
    elif isinstance(value, str) and value in _STALE_V1_TERMS:
        findings.append(path)
    return findings


def _validate_scientific_scope(value: object) -> None:
    scope = _require_object(
        value,
        field="scientific_scope",
        expected_fields=_SCIENTIFIC_SCOPE_FIELDS,
    )
    if (
        _require_text(scope["model"], field="scientific_scope.model")
        != "property-agnostic canonical molecular-successor editing prior"
    ):
        raise EditingCorpusContractError("scientific_scope.model is not the editing prior")
    controller_inputs = _require_string_list(
        scope["controller_only_inputs"],
        field="scientific_scope.controller_only_inputs",
    )
    if controller_inputs != _CONTROLLER_ONLY_INPUTS:
        raise EditingCorpusContractError(
            "scientific_scope.controller_only_inputs disagrees with the frozen separation"
        )
    separate_lanes = _require_object(
        scope["separate_lanes"],
        field="scientific_scope.separate_lanes",
        expected_fields={"editing", "de_novo"},
    )
    if dict(separate_lanes) != {
        "editing": "fixed-budget embedded canonical-successor jump chain",
        "de_novo": "separate checkpoint and timed CTMC",
    }:
        raise EditingCorpusContractError("scientific_scope.separate_lanes has drifted")
    max_atoms = scope["max_active_atoms"]
    if isinstance(max_atoms, bool) or not isinstance(max_atoms, int) or max_atoms != 40:
        raise EditingCorpusContractError("scientific_scope.max_active_atoms must equal 40")
    if scope["stereochemistry"] != "out_of_scope":
        raise EditingCorpusContractError("scientific_scope.stereochemistry has drifted")
    if scope["formal_charge_changes"] != "out_of_scope_charge_preserving_only":
        raise EditingCorpusContractError("scientific_scope.formal_charge_changes has drifted")


def _validate_operator_basis(value: object) -> None:
    basis = _require_object(
        value,
        field="operator_basis",
        expected_fields={
            "active8",
            "disabled",
            "support_expansion_requires_recorded_decision",
        },
    )
    active8 = _require_string_list(basis["active8"], field="operator_basis.active8")
    disabled = _require_string_list(basis["disabled"], field="operator_basis.disabled")
    expansions = _require_string_list(
        basis["support_expansion_requires_recorded_decision"],
        field="operator_basis.support_expansion_requires_recorded_decision",
    )
    if active8 != ACTIVE8_FAMILIES:
        raise EditingCorpusContractError("operator_basis.active8 must equal exact Active8")
    if disabled != DISABLED_FAMILIES:
        raise EditingCorpusContractError(
            "operator_basis.disabled must equal ring_system_delete and ring_system_grow"
        )
    if expansions != _SUPPORT_EXPANSIONS:
        raise EditingCorpusContractError("operator_basis support-expansion boundary has drifted")
    overlap = set(active8) & set(disabled)
    if overlap:
        raise EditingCorpusContractError(
            f"operator families are both Active8 and disabled: {sorted(overlap)}"
        )


def _validate_evidence_contract(value: object) -> dict[str, str]:
    evidence = _require_object(
        value,
        field="evidence_component_contract",
        expected_fields={"components", "profiles"},
    )
    components = _require_object(
        evidence["components"],
        field="evidence_component_contract.components",
        expected_fields=set(_EVIDENCE_COMPONENT_VOCABULARIES),
    )
    for component, expected_vocabulary in _EVIDENCE_COMPONENT_VOCABULARIES.items():
        vocabulary = _require_string_list(
            components[component],
            field=f"evidence_component_contract.components.{component}",
        )
        if vocabulary != expected_vocabulary:
            raise EditingCorpusContractError(
                f"evidence component vocabulary has drifted: {component}"
            )

    profiles = _require_list(
        evidence["profiles"],
        field="evidence_component_contract.profiles",
    )
    profile_status: dict[str, str] = {}
    for index, raw_profile in enumerate(profiles):
        profile = _require_object(
            raw_profile,
            field=f"evidence_component_contract.profiles[{index}]",
            expected_fields={"id", "admission_status", "components"},
        )
        profile_id = _require_text(
            profile["id"],
            field=f"evidence_component_contract.profiles[{index}].id",
        )
        if profile_id in profile_status:
            raise EditingCorpusContractError(f"duplicate evidence profile: {profile_id}")
        if profile_id not in _EVIDENCE_PROFILES:
            raise EditingCorpusContractError(f"unknown evidence profile: {profile_id}")
        status = _require_text(
            profile["admission_status"],
            field=f"evidence profile {profile_id}.admission_status",
        )
        component_states = _require_object(
            profile["components"],
            field=f"evidence profile {profile_id}.components",
            expected_fields=set(_EVIDENCE_COMPONENT_VOCABULARIES),
        )
        for component, state in component_states.items():
            state_text = _require_text(
                state,
                field=f"evidence profile {profile_id}.components.{component}",
            )
            if state_text not in _EVIDENCE_COMPONENT_VOCABULARIES[component]:
                raise EditingCorpusContractError(
                    f"evidence profile {profile_id} uses unknown {component} state {state_text!r}"
                )
        expected_status, expected_components = _EVIDENCE_PROFILES[profile_id]
        if status != expected_status or dict(component_states) != expected_components:
            raise EditingCorpusContractError(
                f"evidence profile definition has drifted: {profile_id}"
            )
        profile_status[profile_id] = status
    if set(profile_status) != set(_EVIDENCE_PROFILES):
        raise EditingCorpusContractError(
            "evidence profiles are incomplete; "
            f"missing={sorted(set(_EVIDENCE_PROFILES) - set(profile_status))}"
        )
    return profile_status


def _validate_data_lanes(value: object, *, profile_status: Mapping[str, str]) -> None:
    lanes = _require_list(value, field="data_lanes")
    lane_ids: list[str] = []
    for index, raw_lane in enumerate(lanes):
        lane = _require_object(
            raw_lane,
            field=f"data_lanes[{index}]",
            expected_fields={
                "id",
                "purpose",
                "admissible_evidence_profiles",
                "reserved_evidence_profiles",
            },
        )
        lane_id = _require_text(lane["id"], field=f"data_lanes[{index}].id")
        if lane_id in lane_ids:
            raise EditingCorpusContractError(f"duplicate data lane: {lane_id}")
        if lane_id not in _LANE_PROFILE_ASSIGNMENTS:
            raise EditingCorpusContractError(f"unknown data lane: {lane_id}")
        _require_text(lane["purpose"], field=f"data lane {lane_id}.purpose")
        admitted = _require_string_list(
            lane["admissible_evidence_profiles"],
            field=f"data lane {lane_id}.admissible_evidence_profiles",
        )
        reserved = _require_string_list(
            lane["reserved_evidence_profiles"],
            field=f"data lane {lane_id}.reserved_evidence_profiles",
            allow_empty=True,
        )
        unknown = (set(admitted) | set(reserved)) - set(profile_status)
        if unknown:
            raise EditingCorpusContractError(
                f"data lane {lane_id} references unknown evidence profiles: {sorted(unknown)}"
            )
        overlap = set(admitted) & set(reserved)
        if overlap:
            raise EditingCorpusContractError(
                f"data lane {lane_id} both admits and reserves profiles: {sorted(overlap)}"
            )
        for profile_id in admitted:
            if profile_status[profile_id] != "admissible":
                raise EditingCorpusContractError(
                    f"data lane {lane_id} admits unavailable profile {profile_id}"
                )
        for profile_id in reserved:
            if profile_status[profile_id] == "admissible":
                raise EditingCorpusContractError(
                    f"data lane {lane_id} reserves an admissible profile {profile_id}"
                )
        if (admitted, reserved) != _LANE_PROFILE_ASSIGNMENTS[lane_id]:
            raise EditingCorpusContractError(
                f"data lane evidence-profile assignment has drifted: {lane_id}"
            )
        lane_ids.append(lane_id)
    if tuple(lane_ids) != REQUIRED_DATA_LANES:
        raise EditingCorpusContractError(
            "data-lane identifiers or ordering disagree with editing-corpus schema version 2"
        )


def _validate_record_membership_contract(value: object) -> None:
    membership = _require_object(
        value,
        field="record_membership_contract",
        expected_fields={
            "assignment_field",
            "cells",
            "opposing_direction_memberships_forbidden",
        },
    )
    if membership["assignment_field"] != "record_membership_cells":
        raise EditingCorpusContractError("record_membership_contract.assignment_field has drifted")
    cells = _require_list(
        membership["cells"],
        field="record_membership_contract.cells",
    )
    seen: set[str] = set()
    cell_directions: dict[str, tuple[str, str]] = {}
    for index, raw_cell in enumerate(cells):
        cell = _require_object(
            raw_cell,
            field=f"record_membership_contract.cells[{index}]",
            expected_fields={
                "id",
                "predicate_id",
                "active8_families",
                "direction_axis",
                "direction",
                "admission_status",
            },
        )
        cell_id = _require_text(
            cell["id"],
            field=f"record_membership_contract.cells[{index}].id",
        )
        if cell_id in seen:
            raise EditingCorpusContractError(f"duplicate record-membership cell: {cell_id}")
        if cell_id not in _RECORD_CELL_DEFINITIONS:
            raise EditingCorpusContractError(f"unknown record-membership cell: {cell_id}")
        families = _require_string_list(
            cell["active8_families"],
            field=f"record-membership cell {cell_id}.active8_families",
        )
        disabled = set(families) & set(DISABLED_FAMILIES)
        if disabled:
            raise EditingCorpusContractError(
                f"record-membership cell {cell_id} references disabled families: {sorted(disabled)}"
            )
        unknown = set(families) - set(ACTIVE8_FAMILIES)
        if unknown:
            raise EditingCorpusContractError(
                f"record-membership cell {cell_id} references unknown families: {sorted(unknown)}"
            )
        predicate = _require_text(
            cell["predicate_id"],
            field=f"record-membership cell {cell_id}.predicate_id",
        )
        axis = _require_text(
            cell["direction_axis"],
            field=f"record-membership cell {cell_id}.direction_axis",
        )
        direction = _require_text(
            cell["direction"],
            field=f"record-membership cell {cell_id}.direction",
        )
        status = _require_text(
            cell["admission_status"],
            field=f"record-membership cell {cell_id}.admission_status",
        )
        if (axis == "none") != (direction == "none"):
            raise EditingCorpusContractError(
                f"record-membership cell {cell_id} has an incomplete direction"
            )
        expected = _RECORD_CELL_DEFINITIONS[cell_id]
        if (predicate, families, axis, direction, status) != expected:
            raise EditingCorpusContractError(
                f"record-membership cell definition has drifted: {cell_id}"
            )
        cell_directions[cell_id] = (axis, direction)
        seen.add(cell_id)
    if set(seen) != set(_RECORD_CELL_DEFINITIONS):
        raise EditingCorpusContractError(
            "record-membership cells are incomplete; "
            f"missing={sorted(set(_RECORD_CELL_DEFINITIONS) - seen)}"
        )

    exclusions = _require_list(
        membership["opposing_direction_memberships_forbidden"],
        field="record_membership_contract.opposing_direction_memberships_forbidden",
    )
    observed_exclusions: dict[str, tuple[str, ...]] = {}
    for index, raw_exclusion in enumerate(exclusions):
        exclusion = _require_object(
            raw_exclusion,
            field=(f"record_membership_contract.opposing_direction_memberships_forbidden[{index}]"),
            expected_fields={"axis", "record_cells"},
        )
        axis = _require_text(
            exclusion["axis"],
            field=f"opposing-direction exclusion {index}.axis",
        )
        record_cells = _require_string_list(
            exclusion["record_cells"],
            field=f"opposing-direction exclusion {axis}.record_cells",
        )
        if axis in observed_exclusions:
            raise EditingCorpusContractError(f"duplicate opposing-direction axis: {axis}")
        unknown_cells = set(record_cells) - seen
        if unknown_cells:
            raise EditingCorpusContractError(
                f"opposing-direction exclusion {axis} references unknown cells: "
                f"{sorted(unknown_cells)}"
            )
        directions = {cell_directions[cell_id] for cell_id in record_cells}
        if directions != {(axis, "increase"), (axis, "decrease")}:
            raise EditingCorpusContractError(
                f"opposing-direction exclusion {axis} does not pair increase and decrease"
            )
        observed_exclusions[axis] = record_cells
    if observed_exclusions != _OPPOSING_DIRECTION_CELLS:
        raise EditingCorpusContractError("opposing per-record direction exclusions have drifted")


def _validate_relationship_contract(value: object) -> None:
    relationship = _require_object(
        value,
        field="relationship_coverage_contract",
        expected_fields={
            "assignment_field",
            "group_types",
            "post_active8_ledger_schema",
        },
    )
    if relationship["assignment_field"] != "relationship_group_ids":
        raise EditingCorpusContractError(
            "relationship_coverage_contract.assignment_field has drifted"
        )
    groups = _require_list(
        relationship["group_types"],
        field="relationship_coverage_contract.group_types",
    )
    seen: set[str] = set()
    for index, raw_group in enumerate(groups):
        group = _require_object(
            raw_group,
            field=f"relationship_coverage_contract.group_types[{index}]",
            expected_fields={
                "id",
                "minimum_distinct_members",
                "construction_scope",
                "member_roles",
                "semantic_requirements",
                "verification_receipt_type",
            },
        )
        group_id = _require_text(
            group["id"],
            field=f"relationship_coverage_contract.group_types[{index}].id",
        )
        if group_id in seen:
            raise EditingCorpusContractError(f"duplicate relationship group type: {group_id}")
        if group_id not in _RELATIONSHIP_GROUP_TYPES:
            raise EditingCorpusContractError(f"unknown relationship group type: {group_id}")
        minimum_members = _require_positive_int(
            group["minimum_distinct_members"],
            field=f"relationship group {group_id}.minimum_distinct_members",
        )
        scope = _require_text(
            group["construction_scope"],
            field=f"relationship group {group_id}.construction_scope",
        )
        member_roles = _require_string_list(
            group["member_roles"],
            field=f"relationship group {group_id}.member_roles",
        )
        requirements = _require_string_list(
            group["semantic_requirements"],
            field=f"relationship group {group_id}.semantic_requirements",
        )
        receipt_type = _require_text(
            group["verification_receipt_type"],
            field=f"relationship group {group_id}.verification_receipt_type",
        )
        expected = _RELATIONSHIP_GROUP_TYPES[group_id]
        observed = {
            "minimum_distinct_members": minimum_members,
            "construction_scope": scope,
            "member_roles": member_roles,
            "semantic_requirements": requirements,
            "verification_receipt_type": receipt_type,
        }
        if observed != expected:
            raise EditingCorpusContractError(
                f"relationship group definition has drifted: {group_id}"
            )
        seen.add(group_id)
    if seen != set(_RELATIONSHIP_GROUP_TYPES):
        raise EditingCorpusContractError(
            "relationship group types are incomplete; "
            f"missing={sorted(set(_RELATIONSHIP_GROUP_TYPES) - seen)}"
        )

    ledger = _require_object(
        relationship["post_active8_ledger_schema"],
        field="relationship_coverage_contract.post_active8_ledger_schema",
        expected_fields={
            "stage",
            "group_fields",
            "verification_receipt_fields",
            "member_fields",
            "trace_key_fields",
            "segment_fields",
            "coverage_count_policy",
        },
    )
    if ledger["stage"] != "post_active8":
        raise EditingCorpusContractError("relationship ledger must be constructed post-Active8")
    expected_ledger_fields = {
        "group_fields": _RELATIONSHIP_LEDGER_GROUP_FIELDS,
        "verification_receipt_fields": _RELATIONSHIP_LEDGER_RECEIPT_FIELDS,
        "member_fields": _RELATIONSHIP_LEDGER_MEMBER_FIELDS,
        "trace_key_fields": _RELATIONSHIP_LEDGER_TRACE_KEY_FIELDS,
        "segment_fields": _RELATIONSHIP_LEDGER_SEGMENT_FIELDS,
    }
    for field, expected_fields in expected_ledger_fields.items():
        observed_fields = _require_string_list(
            ledger[field],
            field=f"relationship ledger {field}",
        )
        if observed_fields != expected_fields:
            raise EditingCorpusContractError(f"relationship ledger {field} has drifted")
    if ledger["coverage_count_policy"] != "complete_verified_groups_only":
        raise EditingCorpusContractError(
            "incomplete relationship groups must not count toward coverage"
        )


def _validate_corpus_direction_contract(value: object) -> None:
    direction_contract = _require_object(
        value,
        field="corpus_direction_contract",
        expected_fields={"cells"},
    )
    cells = _require_list(
        direction_contract["cells"],
        field="corpus_direction_contract.cells",
    )
    seen: set[str] = set()
    for index, raw_cell in enumerate(cells):
        cell = _require_object(
            raw_cell,
            field=f"corpus_direction_contract.cells[{index}]",
            expected_fields={"id", "axis", "required_record_cells"},
        )
        cell_id = _require_text(
            cell["id"],
            field=f"corpus_direction_contract.cells[{index}].id",
        )
        if cell_id in seen:
            raise EditingCorpusContractError(f"duplicate corpus direction cell: {cell_id}")
        if cell_id not in _CORPUS_DIRECTION_CELLS:
            raise EditingCorpusContractError(f"unknown corpus direction cell: {cell_id}")
        axis = _require_text(
            cell["axis"],
            field=f"corpus direction cell {cell_id}.axis",
        )
        record_cells = _require_string_list(
            cell["required_record_cells"],
            field=f"corpus direction cell {cell_id}.required_record_cells",
        )
        if (axis, record_cells) != _CORPUS_DIRECTION_CELLS[cell_id]:
            raise EditingCorpusContractError(
                f"corpus direction cell definition has drifted: {cell_id}"
            )
        if record_cells != _OPPOSING_DIRECTION_CELLS[axis]:
            raise EditingCorpusContractError(
                f"corpus direction cell {cell_id} does not aggregate its opposing directions"
            )
        seen.add(cell_id)
    if seen != set(_CORPUS_DIRECTION_CELLS):
        raise EditingCorpusContractError(
            "corpus direction cells are incomplete; "
            f"missing={sorted(set(_CORPUS_DIRECTION_CELLS) - seen)}"
        )


def _validate_capabilities(value: object) -> set[str]:
    capabilities = _require_list(value, field="capability_coverage_contracts")
    seen: set[str] = set()
    covered_experiments: set[str] = set()
    covered_lanes: set[str] = set()
    covered_record_cells: set[str] = set()
    covered_direction_cells: set[str] = set()
    covered_relationships: set[str] = set()
    referenced_thresholds: set[str] = set()
    for index, raw_capability in enumerate(capabilities):
        capability = _require_object(
            raw_capability,
            field=f"capability_coverage_contracts[{index}]",
            expected_fields={
                "id",
                "experiments",
                "evidence_lanes",
                "required_corpus_record_cells",
                "required_corpus_direction_cells",
                "required_relationship_group_types",
                "coverage_scope",
                "threshold_refs",
            },
        )
        capability_id = _require_text(
            capability["id"],
            field=f"capability_coverage_contracts[{index}].id",
        )
        if capability_id in seen:
            raise EditingCorpusContractError(f"duplicate capability: {capability_id}")
        if capability_id not in _CAPABILITY_REQUIREMENTS:
            raise EditingCorpusContractError(f"unknown capability: {capability_id}")
        experiments = _require_string_list(
            capability["experiments"],
            field=f"capability {capability_id}.experiments",
        )
        evidence_lanes = _require_string_list(
            capability["evidence_lanes"],
            field=f"capability {capability_id}.evidence_lanes",
        )
        record_cells = _require_string_list(
            capability["required_corpus_record_cells"],
            field=f"capability {capability_id}.required_corpus_record_cells",
            allow_empty=True,
        )
        direction_cells = _require_string_list(
            capability["required_corpus_direction_cells"],
            field=f"capability {capability_id}.required_corpus_direction_cells",
            allow_empty=True,
        )
        relationships = _require_string_list(
            capability["required_relationship_group_types"],
            field=f"capability {capability_id}.required_relationship_group_types",
            allow_empty=True,
        )
        scope = _require_text(
            capability["coverage_scope"],
            field=f"capability {capability_id}.coverage_scope",
        )
        threshold_refs = _require_string_list(
            capability["threshold_refs"],
            field=f"capability {capability_id}.threshold_refs",
        )
        unknown_lanes = set(evidence_lanes) - set(REQUIRED_DATA_LANES)
        unknown_cells = set(record_cells) - set(_RECORD_CELL_DEFINITIONS)
        unknown_directions = set(direction_cells) - set(_CORPUS_DIRECTION_CELLS)
        unknown_relationships = set(relationships) - set(_RELATIONSHIP_GROUP_TYPES)
        unknown_thresholds = set(threshold_refs) - _THRESHOLD_NAMES
        if unknown_lanes:
            raise EditingCorpusContractError(
                f"capability {capability_id} references unknown lanes: {sorted(unknown_lanes)}"
            )
        if unknown_cells:
            raise EditingCorpusContractError(
                f"capability {capability_id} references unknown record cells: "
                f"{sorted(unknown_cells)}"
            )
        if unknown_directions:
            raise EditingCorpusContractError(
                f"capability {capability_id} references unknown direction cells: "
                f"{sorted(unknown_directions)}"
            )
        if unknown_relationships:
            raise EditingCorpusContractError(
                f"capability {capability_id} references unknown relationship types: "
                f"{sorted(unknown_relationships)}"
            )
        if unknown_thresholds:
            raise EditingCorpusContractError(
                f"capability {capability_id} references unknown thresholds: "
                f"{sorted(unknown_thresholds)}"
            )
        expected = _CAPABILITY_REQUIREMENTS[capability_id]
        observed = {
            "experiments": experiments,
            "evidence_lanes": evidence_lanes,
            "record_cells": record_cells,
            "direction_cells": direction_cells,
            "relationships": relationships,
            "coverage_scope": scope,
        }
        if observed != expected:
            raise EditingCorpusContractError(
                f"capability coverage definition has drifted: {capability_id}"
            )
        required_threshold_refs = {
            "minimum_unique_sources_per_capability",
            "minimum_unique_scaffolds_per_capability",
            "minimum_effective_teacher_coefficient_per_capability",
        }
        if record_cells:
            required_threshold_refs.add("minimum_records_per_required_membership_cell")
        if direction_cells:
            required_threshold_refs.add("minimum_records_per_direction_member")
        if relationships and capability_id != "pareto_shared_prefix_branching":
            required_threshold_refs.add("minimum_relationship_groups_per_required_type")
        if "coupled_cardinality_topology_path" in record_cells:
            required_threshold_refs.add("minimum_coupled_cardinality_topology_paths")
        if "constraint_feasible_path" in record_cells:
            required_threshold_refs.add("minimum_constraint_feasible_paths")
        if capability_id == "multi_operator_compositional_transport":
            required_threshold_refs.add("minimum_real_endpoint_multistep_path_share")
        if capability_id == "pareto_shared_prefix_branching":
            required_threshold_refs.add("minimum_shared_prefix_branch_groups")
        missing_threshold_refs = required_threshold_refs - set(threshold_refs)
        if missing_threshold_refs:
            raise EditingCorpusContractError(
                f"capability {capability_id} omits required threshold references: "
                f"{sorted(missing_threshold_refs)}"
            )
        if capability_id == "dynamic_correction_and_partial_reversal" and (
            record_cells or relationships != ("inverse_pair", "correction")
        ):
            raise EditingCorpusContractError(
                "dynamic correction must use inverse-pair and correction groups, "
                "not a per-record family predicate"
            )
        if capability_id == "pareto_shared_prefix_branching" and (
            record_cells
            or direction_cells
            or relationships != ("shared_prefix_branch",)
            or scope != "controller_neutral_prior_relationship"
        ):
            raise EditingCorpusContractError(
                "Pareto coverage must be controller-neutral shared-prefix branching only"
            )
        covered_experiments.update(experiments)
        covered_lanes.update(evidence_lanes)
        covered_record_cells.update(record_cells)
        covered_direction_cells.update(direction_cells)
        covered_relationships.update(relationships)
        referenced_thresholds.update(threshold_refs)
        seen.add(capability_id)
    if seen != set(_CAPABILITY_REQUIREMENTS):
        raise EditingCorpusContractError(
            "capability coverage contracts are incomplete; "
            f"missing={sorted(set(_CAPABILITY_REQUIREMENTS) - seen)}"
        )
    missing_experiments = REQUIRED_EXPERIMENTS - covered_experiments
    if missing_experiments:
        raise EditingCorpusContractError(
            f"capability coverage does not cover experiments: {sorted(missing_experiments)}"
        )
    unknown_experiments = covered_experiments - REQUIRED_EXPERIMENTS
    if unknown_experiments:
        raise EditingCorpusContractError(
            f"capability coverage names unknown experiments: {sorted(unknown_experiments)}"
        )
    uncovered = {
        "data lanes": set(REQUIRED_DATA_LANES) - covered_lanes,
        "record-membership cells": set(_RECORD_CELL_DEFINITIONS) - covered_record_cells,
        "corpus direction cells": set(_CORPUS_DIRECTION_CELLS) - covered_direction_cells,
        "relationship group types": set(_RELATIONSHIP_GROUP_TYPES) - covered_relationships,
    }
    missing_coverage = {name: sorted(items) for name, items in uncovered.items() if items}
    if missing_coverage:
        raise EditingCorpusContractError(
            f"declared corpus semantics lack capability coverage: {missing_coverage}"
        )
    return referenced_thresholds


def _validate_partition_resolution_contract(value: object) -> None:
    resolution = _require_object(
        value,
        field="partition_resolution_contract",
        expected_fields={
            "field",
            "resolution_fields",
            "statuses",
            "status_rules",
            "assigned_role_domain_ref",
            "accepted_record_status",
            "rejected_statuses",
            "rejected_attempt_retention",
        },
    )
    if resolution["field"] != "partition_resolution":
        raise EditingCorpusContractError("partition_resolution_contract.field has drifted")
    fields = _require_string_list(
        resolution["resolution_fields"],
        field="partition_resolution_contract.resolution_fields",
    )
    if fields != _PARTITION_RESOLUTION_FIELDS:
        raise EditingCorpusContractError("partition-resolution row fields have drifted")
    statuses = _require_string_list(
        resolution["statuses"],
        field="partition_resolution_contract.statuses",
    )
    if statuses != _PARTITION_RESOLUTION_STATUSES:
        raise EditingCorpusContractError("partition-resolution statuses have drifted")
    raw_rules = _require_list(
        resolution["status_rules"],
        field="partition_resolution_contract.status_rules",
    )
    observed_rules: dict[str, tuple[str, str, str, str]] = {}
    for index, raw_rule in enumerate(raw_rules):
        rule = _require_object(
            raw_rule,
            field=f"partition_resolution_contract.status_rules[{index}]",
            expected_fields={
                "status",
                "assigned_role_policy",
                "endpoint_role_policy",
                "split_component_id_policy",
                "admitted_record_policy",
            },
        )
        status = _require_text(
            rule["status"],
            field=f"partition status rule {index}.status",
        )
        if status in observed_rules:
            raise EditingCorpusContractError(f"duplicate partition status rule: {status}")
        observed_rules[status] = (
            _require_text(
                rule["assigned_role_policy"],
                field=f"partition status rule {status}.assigned_role_policy",
            ),
            _require_text(
                rule["endpoint_role_policy"],
                field=f"partition status rule {status}.endpoint_role_policy",
            ),
            _require_text(
                rule["split_component_id_policy"],
                field=f"partition status rule {status}.split_component_id_policy",
            ),
            _require_text(
                rule["admitted_record_policy"],
                field=f"partition status rule {status}.admitted_record_policy",
            ),
        )
    if observed_rules != _PARTITION_STATUS_RULES:
        raise EditingCorpusContractError("partition-resolution status rules have drifted")
    if resolution["assigned_role_domain_ref"] != "split_contract.partition_roles":
        raise EditingCorpusContractError(
            "partition assigned-role domain must reference split partition roles"
        )
    if resolution["accepted_record_status"] != "assigned":
        raise EditingCorpusContractError(
            "only assigned partition resolutions may become admitted records"
        )
    rejected = _require_string_list(
        resolution["rejected_statuses"],
        field="partition_resolution_contract.rejected_statuses",
    )
    if rejected != ("cross_partition_rejected", "unassigned_rejected"):
        raise EditingCorpusContractError("partition-resolution rejected statuses have drifted")
    if resolution["rejected_attempt_retention"] != "pre_admission_candidate_audit_ledger":
        raise EditingCorpusContractError(
            "rejected partition attempts must remain in the candidate audit ledger"
        )


def _validate_admitted_record_contract(value: object) -> None:
    admitted = _require_object(
        value,
        field="admitted_record_contract",
        expected_fields={
            "applies_to",
            "rejected_attempts_live_in",
            "structural_envelope_validation_scope",
            "external_authority_requirements",
            "required_fields",
        },
    )
    if admitted["applies_to"] != "accepted_post_split_compiled_traces_only":
        raise EditingCorpusContractError(
            "exact trace fields may be required only on accepted compiled traces"
        )
    if admitted["rejected_attempts_live_in"] != "pre_admission_candidate_audit_ledger":
        raise EditingCorpusContractError(
            "rejected attempts must remain in the pre-admission candidate audit ledger"
        )
    if admitted["structural_envelope_validation_scope"] != _ADMITTED_RECORD_VALIDATION_SCOPE:
        raise EditingCorpusContractError(
            "admitted-record envelope validation must not claim chemistry-predicate "
            "or relationship-receipt authority"
        )
    requirements = _require_list(
        admitted["external_authority_requirements"],
        field="admitted_record_contract.external_authority_requirements",
    )
    observed_requirements: list[tuple[str, str]] = []
    for index, raw_requirement in enumerate(requirements):
        requirement = _require_object(
            raw_requirement,
            field=f"admitted_record_contract.external_authority_requirements[{index}]",
            expected_fields={"id", "status"},
        )
        observed_requirements.append(
            (
                _require_text(
                    requirement["id"],
                    field=(f"admitted_record_contract.external_authority_requirements[{index}].id"),
                ),
                _require_text(
                    requirement["status"],
                    field=(
                        f"admitted_record_contract.external_authority_requirements[{index}].status"
                    ),
                ),
            )
        )
    if tuple(observed_requirements) != _EXTERNAL_AUTHORITY_REQUIREMENTS:
        raise EditingCorpusContractError(
            "admitted-record external authority requirements have drifted"
        )
    fields = _require_string_list(
        admitted["required_fields"],
        field="admitted_record_contract.required_fields",
    )
    if fields != _REQUIRED_RECORD_FIELDS:
        missing = set(_REQUIRED_RECORD_FIELDS) - set(fields)
        unexpected = set(fields) - set(_REQUIRED_RECORD_FIELDS)
        raise EditingCorpusContractError(
            "admitted-record required fields disagree with schema version 2; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if "partition" in fields or "partition_resolution" not in fields:
        raise EditingCorpusContractError(
            "admitted records require partition_resolution, not a scalar partition"
        )


def _validate_split_contract(value: object) -> None:
    split = _require_object(
        value,
        field="split_contract",
        expected_fields={
            "partition_roles",
            "group_keys",
            "cross_partition_pairs_forbidden",
            "new_sealed_final_holdout_required",
        },
    )
    roles = _require_string_list(
        split["partition_roles"],
        field="split_contract.partition_roles",
    )
    if roles != REQUIRED_PARTITION_ROLES:
        raise EditingCorpusContractError(
            "split_contract.partition_roles must declare the ordered train, validation, "
            "controller-validation, and final-test roles"
        )
    group_keys = _require_string_list(
        split["group_keys"],
        field="split_contract.group_keys",
    )
    if group_keys != (
        "molecule_identity",
        "scaffold_group_id",
        "series_group_id",
        "document_or_source_group_id",
        "relationship_group_ids",
        "transformation_signature",
    ):
        raise EditingCorpusContractError("split_contract.group_keys has drifted")
    if not _require_bool(
        split["cross_partition_pairs_forbidden"],
        field="split_contract.cross_partition_pairs_forbidden",
    ):
        raise EditingCorpusContractError("cross-partition endpoint pairs must be forbidden")
    if not _require_bool(
        split["new_sealed_final_holdout_required"],
        field="split_contract.new_sealed_final_holdout_required",
    ):
        raise EditingCorpusContractError("a new sealed final holdout must be required")


def _validate_sampling_contract(value: object) -> None:
    sampling = _require_object(
        value,
        field="sampling_contract",
        expected_fields={
            "hierarchy",
            "raw_pair_uniform_sampling_forbidden",
            "report_effective_teacher_coefficient",
            "report_path_position_coefficient",
        },
    )
    hierarchy = _require_string_list(
        sampling["hierarchy"],
        field="sampling_contract.hierarchy",
    )
    if hierarchy != (
        "data_lane",
        "series_scaffold_or_source_group",
        "semantic_capability_cell",
        "endpoint_pair_or_path",
        "progress_state",
    ):
        raise EditingCorpusContractError("sampling_contract.hierarchy has drifted")
    for field in (
        "raw_pair_uniform_sampling_forbidden",
        "report_effective_teacher_coefficient",
        "report_path_position_coefficient",
    ):
        if not _require_bool(sampling[field], field=f"sampling_contract.{field}"):
            raise EditingCorpusContractError(f"sampling_contract.{field} must remain true")


def _validate_threshold_value(name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditingCorpusContractError(
            f"pretraining_gates.threshold_values.{name} must be numeric or null"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EditingCorpusContractError(
            f"pretraining_gates.threshold_values.{name} must be finite"
        )
    if "share" in name or name.startswith("maximum_"):
        if not 0.0 <= numeric <= 1.0:
            raise EditingCorpusContractError(
                f"pretraining_gates.threshold_values.{name} must be in [0, 1]"
            )
    elif name == "minimum_effective_teacher_coefficient_per_capability":
        if numeric <= 0.0:
            raise EditingCorpusContractError(
                f"pretraining_gates.threshold_values.{name} must be positive"
            )
    elif not isinstance(value, int) or value < 1:
        raise EditingCorpusContractError(
            f"pretraining_gates.threshold_values.{name} must be a positive integer"
        )


def _validate_pretraining_gates(
    value: object,
    *,
    capability_threshold_refs: set[str],
) -> None:
    gates = _require_object(
        value,
        field="pretraining_gates",
        expected_fields={
            "structural",
            "threshold_values",
            "global_threshold_refs",
            "required_staged_decisions",
        },
    )
    structural_fields = {
        "all_actions_execute",
        "all_committed_states_supported_connected_and_valid",
        "teacher_successor_fibers_recoverable",
        "split_overlap_zero",
        "unsupported_pairs_retained_as_audit_records",
        "path_lengths_fit_registered_edit_budgets",
        "every_required_coverage_cell_nonempty",
        "evidence_components_match_declared_profile",
        "relationship_groups_resolve_to_distinct_records",
    }
    structural = _require_object(
        gates["structural"],
        field="pretraining_gates.structural",
        expected_fields=structural_fields,
    )
    for field in structural_fields:
        if not _require_bool(
            structural[field],
            field=f"pretraining_gates.structural.{field}",
        ):
            raise EditingCorpusContractError(
                f"pretraining_gates.structural.{field} must remain true"
            )

    thresholds = _require_object(
        gates["threshold_values"],
        field="pretraining_gates.threshold_values",
        expected_fields=_THRESHOLD_NAMES,
    )
    for name, threshold in thresholds.items():
        _validate_threshold_value(name, threshold)
    global_refs = _require_string_list(
        gates["global_threshold_refs"],
        field="pretraining_gates.global_threshold_refs",
    )
    if global_refs != _GLOBAL_THRESHOLD_REFS:
        raise EditingCorpusContractError("pretraining_gates.global_threshold_refs has drifted")
    unknown_global_refs = set(global_refs) - set(thresholds)
    if unknown_global_refs:
        raise EditingCorpusContractError(
            f"global gates reference unknown thresholds: {sorted(unknown_global_refs)}"
        )
    all_refs = capability_threshold_refs | set(global_refs)
    missing_refs = set(thresholds) - all_refs
    if missing_refs:
        raise EditingCorpusContractError(
            f"threshold values are not referenced by any gate: {sorted(missing_refs)}"
        )
    staged = _require_string_list(
        gates["required_staged_decisions"],
        field="pretraining_gates.required_staged_decisions",
    )
    if staged != ("micro_overfit", "step_500_stop_go", "step_2000_stop_go"):
        raise EditingCorpusContractError("pretraining_gates.required_staged_decisions has drifted")


def validate_editing_corpus_contract(contract: Mapping[str, Any]) -> None:
    root = _require_object(
        contract,
        field="contract",
        expected_fields=_TOP_LEVEL_FIELDS,
    )
    stale_terms = _find_stale_terms(root)
    if stale_terms:
        raise EditingCorpusContractError(
            f"schema-v1 terms are forbidden in schema version 2: {stale_terms}"
        )
    if root["schema"] != EXPECTED_SCHEMA:
        raise EditingCorpusContractError("unexpected editing-corpus schema")
    if root["schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise EditingCorpusContractError("unexpected editing-corpus schema version")
    _require_text(root["contract_id"], field="contract_id")
    status = _require_text(root["status"], field="status")
    if status not in {
        "DESIGN_NOT_TRAINING_AUTHORIZED",
        "FROZEN_TRAINING_AUTHORIZED",
    }:
        raise EditingCorpusContractError(f"unsupported contract status: {status}")
    _require_bool(root["training_authorized"], field="training_authorized")

    _validate_scientific_scope(root["scientific_scope"])
    _validate_operator_basis(root["operator_basis"])
    profile_status = _validate_evidence_contract(root["evidence_component_contract"])
    _validate_data_lanes(root["data_lanes"], profile_status=profile_status)
    _validate_record_membership_contract(root["record_membership_contract"])
    _validate_relationship_contract(root["relationship_coverage_contract"])
    _validate_corpus_direction_contract(root["corpus_direction_contract"])
    capability_threshold_refs = _validate_capabilities(root["capability_coverage_contracts"])
    _validate_partition_resolution_contract(root["partition_resolution_contract"])
    _validate_admitted_record_contract(root["admitted_record_contract"])
    _validate_split_contract(root["split_contract"])
    _validate_sampling_contract(root["sampling_contract"])
    _validate_pretraining_gates(
        root["pretraining_gates"],
        capability_threshold_refs=capability_threshold_refs,
    )


def validate_record_membership_assignment(
    contract: Mapping[str, Any],
    membership_cells: Sequence[str],
) -> None:
    """Reject unknown or directionally contradictory cells on one trace record."""

    validate_editing_corpus_contract(contract)
    _validate_record_membership_assignment_validated(contract, membership_cells)


def _validate_record_membership_assignment_validated(
    contract: Mapping[str, Any],
    membership_cells: Sequence[str],
) -> None:
    if isinstance(membership_cells, (str, bytes)) or not isinstance(membership_cells, Sequence):
        raise EditingCorpusContractError(
            "record membership assignment must be a sequence of cell identifiers"
        )
    normalized = tuple(
        _require_text(cell, field=f"record membership assignment[{index}]")
        for index, cell in enumerate(membership_cells)
    )
    if not normalized:
        raise EditingCorpusContractError(
            "record membership assignment must contain at least one cell"
        )
    duplicates = _duplicates(normalized)
    if duplicates:
        raise EditingCorpusContractError(
            f"record membership assignment contains duplicates: {duplicates}"
        )
    unknown = set(normalized) - set(_RECORD_CELL_DEFINITIONS)
    if unknown:
        raise EditingCorpusContractError(
            f"record membership assignment contains unknown cells: {sorted(unknown)}"
        )
    selected = set(normalized)
    for axis, opposing_cells in _OPPOSING_DIRECTION_CELLS.items():
        if set(opposing_cells) <= selected:
            raise EditingCorpusContractError(
                f"one record cannot carry opposing {axis} directions: {list(opposing_cells)}"
            )


def validate_partition_resolution(
    contract: Mapping[str, Any],
    resolution: Mapping[str, Any],
    *,
    admitted_record: bool,
) -> None:
    """Validate one split outcome without fabricating a partition for rejects."""

    validate_editing_corpus_contract(contract)
    _validate_partition_resolution_validated(
        contract,
        resolution,
        admitted_record=admitted_record,
    )


def _validate_partition_resolution_validated(
    contract: Mapping[str, Any],
    resolution: Mapping[str, Any],
    *,
    admitted_record: bool,
) -> None:
    _require_bool(admitted_record, field="admitted_record")
    payload = _require_object(
        resolution,
        field="partition_resolution",
        expected_fields=set(_PARTITION_RESOLUTION_FIELDS),
    )
    status = _require_text(payload["status"], field="partition_resolution.status")
    if status not in _PARTITION_RESOLUTION_STATUSES:
        raise EditingCorpusContractError(f"partition_resolution.status is unknown: {status!r}")
    roles = set(contract["split_contract"]["partition_roles"])
    assigned_role = payload["assigned_role"]
    source_role = payload["source_endpoint_role"]
    target_role = payload["target_endpoint_role"]
    split_component_id = payload["split_component_id"]
    _require_text(
        split_component_id,
        field="partition_resolution.split_component_id",
    )
    _require_sha256(
        payload["split_assignment_manifest_sha256"],
        field="partition_resolution.split_assignment_manifest_sha256",
    )

    if admitted_record and status != "assigned":
        raise EditingCorpusContractError(
            "only an assigned partition resolution may appear on an admitted record"
        )
    if status == "assigned":
        assigned = _require_text(
            assigned_role,
            field="partition_resolution.assigned_role",
        )
        source = _require_text(
            source_role,
            field="partition_resolution.source_endpoint_role",
        )
        target = _require_text(
            target_role,
            field="partition_resolution.target_endpoint_role",
        )
        if assigned not in roles or source not in roles or target not in roles:
            raise EditingCorpusContractError(
                "assigned partition roles must come from split_contract.partition_roles"
            )
        if not assigned == source == target:
            raise EditingCorpusContractError(
                "assigned endpoints and assigned_role must name the same partition role"
            )
        return

    if assigned_role is not None:
        raise EditingCorpusContractError(
            "rejected partition resolutions must set assigned_role to null"
        )
    if status == "cross_partition_rejected":
        source = _require_text(
            source_role,
            field="partition_resolution.source_endpoint_role",
        )
        target = _require_text(
            target_role,
            field="partition_resolution.target_endpoint_role",
        )
        if source not in roles or target not in roles:
            raise EditingCorpusContractError(
                "cross-partition endpoint roles must come from split partition roles"
            )
        if source == target:
            raise EditingCorpusContractError(
                "cross_partition_rejected requires distinct endpoint roles"
            )
        return

    endpoint_roles = (source_role, target_role)
    for index, role in enumerate(endpoint_roles):
        if role is not None and role not in roles:
            raise EditingCorpusContractError(
                "unassigned endpoint roles must be null or one of the split roles; "
                f"bad endpoint index={index}"
            )
    if all(role is not None for role in endpoint_roles):
        raise EditingCorpusContractError(
            "unassigned_rejected requires at least one unassigned endpoint role"
        )


def validate_evidence_assignment_envelope(
    contract: Mapping[str, Any],
    *,
    data_lane: str,
    evidence_profile_id: str,
    evidence_components: Mapping[str, Any],
) -> None:
    """Validate declared lane/profile/components, not the underlying evidence.

    This structural check cannot establish that endpoints were observed, that a
    relationship was real, or that a compiled path has the declared semantics.
    """

    validate_editing_corpus_contract(contract)
    _validate_evidence_assignment_envelope_validated(
        contract,
        data_lane=data_lane,
        evidence_profile_id=evidence_profile_id,
        evidence_components=evidence_components,
    )


def _validate_evidence_assignment_envelope_validated(
    contract: Mapping[str, Any],
    *,
    data_lane: str,
    evidence_profile_id: str,
    evidence_components: Mapping[str, Any],
) -> None:
    lane_id = _require_text(data_lane, field="data_lane")
    profile_id = _require_text(evidence_profile_id, field="evidence_profile_id")
    lanes = {lane["id"]: lane for lane in contract["data_lanes"]}
    if lane_id not in lanes:
        raise EditingCorpusContractError(f"unknown data lane: {lane_id}")
    profiles = {
        profile["id"]: profile for profile in contract["evidence_component_contract"]["profiles"]
    }
    if profile_id not in profiles:
        raise EditingCorpusContractError(f"unknown evidence profile: {profile_id}")

    lane = lanes[lane_id]
    profile = profiles[profile_id]
    if profile["admission_status"] != "admissible":
        location = (
            "reserved by this lane"
            if profile_id in lane["reserved_evidence_profiles"]
            else "unavailable"
        )
        raise EditingCorpusContractError(
            f"evidence profile {profile_id} is {location} and cannot be admitted"
        )
    if profile_id not in lane["admissible_evidence_profiles"]:
        raise EditingCorpusContractError(
            f"evidence lane/profile mismatch: {lane_id} does not admit {profile_id}"
        )

    components = _require_object(
        evidence_components,
        field="evidence_components",
        expected_fields=set(_EVIDENCE_COMPONENT_VOCABULARIES),
    )
    normalized_components = {
        component: _require_text(
            components[component],
            field=f"evidence_components.{component}",
        )
        for component in _EVIDENCE_COMPONENT_VOCABULARIES
    }
    if normalized_components != profile["components"]:
        raise EditingCorpusContractError(
            f"evidence components do not match declared profile {profile_id}"
        )


def validate_admitted_record_envelope(
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    """Validate an admitted-record structural envelope without certifying chemistry.

    The function checks exact fields, split assignment, declared evidence,
    trace-array alignment, Active8 membership, scalar shapes, and hash syntax.
    It deliberately does not execute actions, recompute canonical identities or
    membership predicates, or resolve relationship receipts against artifacts.
    """

    validate_editing_corpus_contract(contract)
    payload = _require_object(
        record,
        field="admitted_record",
        expected_fields=set(_REQUIRED_RECORD_FIELDS),
    )
    _require_text(payload["record_id"], field="admitted_record.record_id")
    _validate_partition_resolution_validated(
        contract,
        payload["partition_resolution"],
        admitted_record=True,
    )
    _validate_evidence_assignment_envelope_validated(
        contract,
        data_lane=payload["data_lane"],
        evidence_profile_id=payload["evidence_profile_id"],
        evidence_components=payload["evidence_components"],
    )

    path_length = _require_positive_int(
        payload["path_length"],
        field="admitted_record.path_length",
    )
    _require_nonempty_json_object(
        payload["source_state_exact"],
        field="admitted_record.source_state_exact",
    )
    successors = _require_list(
        payload["successor_states_exact"],
        field="admitted_record.successor_states_exact",
    )
    if not successors:
        raise EditingCorpusContractError("admitted_record.successor_states_exact must not be empty")
    normalized_successors = tuple(
        _require_nonempty_json_object(
            successor,
            field=f"admitted_record.successor_states_exact[{index}]",
        )
        for index, successor in enumerate(successors)
    )
    target_state = _require_nonempty_json_object(
        payload["target_state_exact"],
        field="admitted_record.target_state_exact",
    )

    actions = _require_list(
        payload["executable_actions"],
        field="admitted_record.executable_actions",
    )
    if not actions:
        raise EditingCorpusContractError("admitted_record.executable_actions must not be empty")
    for index, action in enumerate(actions):
        _require_nonempty_json_object(
            action,
            field=f"admitted_record.executable_actions[{index}]",
        )

    canonical_key_items = _require_list(
        payload["canonical_state_keys"],
        field="admitted_record.canonical_state_keys",
    )
    canonical_keys = tuple(
        _require_text(
            key,
            field=f"admitted_record.canonical_state_keys[{index}]",
        )
        for index, key in enumerate(canonical_key_items)
    )
    if not canonical_keys:
        raise EditingCorpusContractError("admitted_record.canonical_state_keys must not be empty")

    operator_items = _require_list(
        payload["operator_sequence"],
        field="admitted_record.operator_sequence",
    )
    operator_sequence = tuple(
        _require_text(
            family,
            field=f"admitted_record.operator_sequence[{index}]",
        )
        for index, family in enumerate(operator_items)
    )
    if not operator_sequence:
        raise EditingCorpusContractError("admitted_record.operator_sequence must not be empty")
    outside_active8 = set(operator_sequence) - set(ACTIVE8_FAMILIES)
    if outside_active8:
        raise EditingCorpusContractError(
            "admitted_record.operator_sequence contains families outside Active8: "
            f"{sorted(outside_active8)}"
        )

    family_set = _require_string_list(
        payload["operator_family_set"],
        field="admitted_record.operator_family_set",
    )
    outside_active8 = set(family_set) - set(ACTIVE8_FAMILIES)
    if outside_active8:
        raise EditingCorpusContractError(
            "admitted_record.operator_family_set contains families outside Active8: "
            f"{sorted(outside_active8)}"
        )
    expected_family_set = tuple(
        family for family in ACTIVE8_FAMILIES if family in set(operator_sequence)
    )
    if family_set != expected_family_set:
        raise EditingCorpusContractError(
            "admitted_record.operator_family_set must equal the deterministic "
            "Active8-ordered set induced by operator_sequence"
        )

    aligned_lengths = {
        "successor_states_exact": len(normalized_successors),
        "executable_actions": len(actions),
        "operator_sequence": len(operator_sequence),
    }
    bad_aligned_lengths = {
        name: length for name, length in aligned_lengths.items() if length != path_length
    }
    if bad_aligned_lengths:
        raise EditingCorpusContractError(
            "admitted-record trace arrays disagree with path_length; "
            f"path_length={path_length}, observed={bad_aligned_lengths}"
        )
    if len(canonical_keys) != path_length + 1:
        raise EditingCorpusContractError(
            "admitted_record.canonical_state_keys must contain source plus one key per successor"
        )
    if dict(target_state) != dict(normalized_successors[-1]):
        raise EditingCorpusContractError(
            "admitted_record.target_state_exact must equal the final exact successor state"
        )

    membership_cells = _require_string_list(
        payload["record_membership_cells"],
        field="admitted_record.record_membership_cells",
    )
    _validate_record_membership_assignment_validated(contract, membership_cells)
    membership_statuses = {
        cell["id"]: cell["admission_status"]
        for cell in contract["record_membership_contract"]["cells"]
    }
    unresolved_memberships = {
        cell_id: membership_statuses[cell_id]
        for cell_id in membership_cells
        if membership_statuses[cell_id] != "admissible"
    }
    if unresolved_memberships:
        raise EditingCorpusContractError(
            "admitted record references unresolved record-membership cells: "
            f"{unresolved_memberships}"
        )

    relationship_group_ids = _require_string_list(
        payload["relationship_group_ids"],
        field="admitted_record.relationship_group_ids",
        allow_empty=True,
    )
    if relationship_group_ids != tuple(sorted(relationship_group_ids)):
        raise EditingCorpusContractError(
            "admitted_record.relationship_group_ids must use deterministic sorted order"
        )

    for field in (
        "source_group_id",
        "scaffold_group_id",
        "document_or_source_group_id",
    ):
        _require_text(payload[field], field=f"admitted_record.{field}")
    _require_optional_text(
        payload["series_group_id"],
        field="admitted_record.series_group_id",
    )
    _require_optional_text(
        payload["transformation_signature"],
        field="admitted_record.transformation_signature",
    )
    _require_slot_mapping(
        payload["constant_core_mapping"],
        field="admitted_record.constant_core_mapping",
    )
    _require_slot_mapping(
        payload["protected_mapping"],
        field="admitted_record.protected_mapping",
    )

    _require_int(
        payload["atom_count_delta"],
        field="admitted_record.atom_count_delta",
    )
    _require_int(
        payload["graph_cycle_rank_delta"],
        field="admitted_record.graph_cycle_rank_delta",
    )
    _require_finite_nonnegative_number(
        payload["maximum_intermediate_source_distance"],
        field="admitted_record.maximum_intermediate_source_distance",
    )
    immediate_reversals = _require_nonnegative_int(
        payload["immediate_reversal_count"],
        field="admitted_record.immediate_reversal_count",
    )
    repeated_states = _require_nonnegative_int(
        payload["repeated_state_count"],
        field="admitted_record.repeated_state_count",
    )
    if immediate_reversals > max(path_length - 1, 0):
        raise EditingCorpusContractError(
            "admitted_record.immediate_reversal_count exceeds the path-length bound"
        )
    if repeated_states > path_length:
        raise EditingCorpusContractError(
            "admitted_record.repeated_state_count exceeds the path-length bound"
        )

    constraint_identity = payload["constraint_evaluator_identity"]
    if constraint_identity is not None:
        _require_sha256(
            constraint_identity,
            field="admitted_record.constraint_evaluator_identity",
        )
    _require_positive_int(
        payload["compiler_candidates_considered"],
        field="admitted_record.compiler_candidates_considered",
    )
    _require_text(
        payload["compiler_choice_reason"],
        field="admitted_record.compiler_choice_reason",
    )
    if payload["compiler_failure_reason"] is not None:
        raise EditingCorpusContractError(
            "an admitted record must set compiler_failure_reason to null"
        )
    _require_nonempty_json_object(
        payload["source_provenance"],
        field="admitted_record.source_provenance",
    )
    for field in ("operator_contract_hash", "canonicalizer_hash", "compiler_hash"):
        _require_sha256(payload[field], field=f"admitted_record.{field}")


def relationship_definition_sha256(
    contract: Mapping[str, Any],
    relationship_type: str,
) -> str:
    """Return the semantic hash a post-Active8 relationship row must bind."""

    validate_editing_corpus_contract(contract)
    relationship_id = _require_text(
        relationship_type,
        field="relationship_type",
    )
    definitions = {
        group["id"]: group for group in contract["relationship_coverage_contract"]["group_types"]
    }
    if relationship_id not in definitions:
        raise EditingCorpusContractError(f"unknown relationship group type: {relationship_id}")
    return _canonical_sha256(definitions[relationship_id])


def validate_relationship_group(
    contract: Mapping[str, Any],
    group: Mapping[str, Any],
) -> None:
    """Validate one complete post-Active8 relationship group and its hashes."""

    validate_editing_corpus_contract(contract)
    payload = _require_object(
        group,
        field="relationship_group",
        expected_fields=set(_RELATIONSHIP_LEDGER_GROUP_FIELDS),
    )
    relationship_type = _require_text(
        payload["type"],
        field="relationship_group.type",
    )
    definitions = {
        definition["id"]: definition
        for definition in contract["relationship_coverage_contract"]["group_types"]
    }
    if relationship_type not in definitions:
        raise EditingCorpusContractError(f"unknown relationship group type: {relationship_type}")
    definition = definitions[relationship_type]
    _require_text(payload["group_id"], field="relationship_group.group_id")
    definition_sha256 = _require_sha256(
        payload["definition_sha256"],
        field="relationship_group.definition_sha256",
    )
    if definition_sha256 != _canonical_sha256(definition):
        raise EditingCorpusContractError(
            "relationship group definition_sha256 disagrees with the contract"
        )

    evidence_profile_id = _require_text(
        payload["evidence_profile_id"],
        field="relationship_group.evidence_profile_id",
    )
    profile_statuses = {
        profile["id"]: profile["admission_status"]
        for profile in contract["evidence_component_contract"]["profiles"]
    }
    if evidence_profile_id not in profile_statuses:
        raise EditingCorpusContractError(
            f"relationship group references unknown evidence profile: {evidence_profile_id}"
        )
    if profile_statuses[evidence_profile_id] != "admissible":
        raise EditingCorpusContractError(
            f"relationship group references unavailable evidence profile: {evidence_profile_id}"
        )

    receipt = _require_object(
        payload["verification_receipt"],
        field="relationship_group.verification_receipt",
        expected_fields=set(_RELATIONSHIP_LEDGER_RECEIPT_FIELDS),
    )
    receipt_type = _require_text(
        receipt["receipt_type"],
        field="relationship_group.verification_receipt.receipt_type",
    )
    if receipt_type != definition["verification_receipt_type"]:
        raise EditingCorpusContractError(
            "relationship verification receipt type disagrees with its definition"
        )
    _require_sha256(
        receipt["verifier_identity_sha256"],
        field="relationship_group.verification_receipt.verifier_identity_sha256",
    )
    _require_sha256(
        receipt["receipt_sha256"],
        field="relationship_group.verification_receipt.receipt_sha256",
    )

    members = _require_list(payload["members"], field="relationship_group.members")
    minimum_members = definition["minimum_distinct_members"]
    if len(members) < minimum_members:
        raise EditingCorpusContractError(
            f"incomplete {relationship_type} group: expected at least {minimum_members} members"
        )
    normalized_members: list[Mapping[str, Any]] = []
    member_hashes: set[str] = set()
    member_identities: set[tuple[str, int, str, int, int, str]] = set()
    trace_identities: list[tuple[str, int, str]] = []
    roles: list[str] = []
    for index, raw_member in enumerate(members):
        member = _require_object(
            raw_member,
            field=f"relationship_group.members[{index}]",
            expected_fields=set(_RELATIONSHIP_LEDGER_MEMBER_FIELDS),
        )
        trace_key = _require_object(
            member["trace_key"],
            field=f"relationship_group.members[{index}].trace_key",
            expected_fields=set(_RELATIONSHIP_LEDGER_TRACE_KEY_FIELDS),
        )
        shard_sha = _require_sha256(
            trace_key["packed_shard_sha256"],
            field=(f"relationship_group.members[{index}].trace_key.packed_shard_sha256"),
        )
        entry = _require_nonnegative_int(
            trace_key["entry"],
            field=f"relationship_group.members[{index}].trace_key.entry",
        )
        trace_id = _require_text(
            trace_key["trace_id"],
            field=f"relationship_group.members[{index}].trace_key.trace_id",
        )
        segment = _require_object(
            member["segment"],
            field=f"relationship_group.members[{index}].segment",
            expected_fields=set(_RELATIONSHIP_LEDGER_SEGMENT_FIELDS),
        )
        start = _require_nonnegative_int(
            segment["start_progress_index"],
            field=(f"relationship_group.members[{index}].segment.start_progress_index"),
        )
        end = _require_nonnegative_int(
            segment["end_progress_index"],
            field=(f"relationship_group.members[{index}].segment.end_progress_index"),
        )
        if end <= start:
            raise EditingCorpusContractError(
                "relationship member segments must contain at least one transition"
            )
        role = _require_text(
            member["role"],
            field=f"relationship_group.members[{index}].role",
        )
        if role not in definition["member_roles"]:
            raise EditingCorpusContractError(
                f"relationship member role {role!r} is invalid for {relationship_type}"
            )
        member_sha = _require_sha256(
            member["member_sha256"],
            field=f"relationship_group.members[{index}].member_sha256",
        )
        member_without_hash = {
            key: value for key, value in member.items() if key != "member_sha256"
        }
        if member_sha != _canonical_sha256(member_without_hash):
            raise EditingCorpusContractError(f"relationship member hash disagrees at index {index}")
        identity = (shard_sha, entry, trace_id, start, end, role)
        if identity in member_identities or member_sha in member_hashes:
            raise EditingCorpusContractError("relationship group contains duplicate members")
        member_identities.add(identity)
        member_hashes.add(member_sha)
        trace_identities.append((shard_sha, entry, trace_id))
        roles.append(role)
        normalized_members.append(member)

    role_counts = {role: roles.count(role) for role in set(roles)}
    if relationship_type == "shared_prefix_branch":
        if (
            role_counts.get("shared_prefix_anchor", 0) < 1
            or role_counts.get("branch_continuation", 0) < 2
        ):
            raise EditingCorpusContractError(
                "incomplete shared-prefix group requires an anchor and two branches"
            )
        branch_traces = {
            trace_identities[index]
            for index, role in enumerate(roles)
            if role == "branch_continuation"
        }
        if len(branch_traces) < 2:
            raise EditingCorpusContractError(
                "shared-prefix branch continuations must resolve to distinct traces"
            )
    elif relationship_type in {
        "alternative_route",
        "constraint_compatible_alternative_route",
    }:
        if set(roles) != set(definition["member_roles"]) or len(set(trace_identities)) < 2:
            raise EditingCorpusContractError(
                f"incomplete {relationship_type} group requires distinct route traces"
            )
    elif relationship_type == "inverse_pair":
        if len(members) != 2 or role_counts != {"forward": 1, "reverse": 1}:
            raise EditingCorpusContractError(
                "inverse_pair requires exactly one forward and one reverse member"
            )
        if len(set(trace_identities)) != 2:
            raise EditingCorpusContractError("inverse_pair members must resolve to distinct traces")
    elif relationship_type == "correction":
        if len(members) != 2 or role_counts != {
            "initial_change": 1,
            "corrective_continuation": 1,
        }:
            raise EditingCorpusContractError(
                "correction requires one initial-change and one corrective segment"
            )
        initial_index = roles.index("initial_change")
        correction_index = roles.index("corrective_continuation")
        if trace_identities[initial_index] != trace_identities[correction_index]:
            raise EditingCorpusContractError(
                "correction segments must resolve to one anchored trace"
            )
        initial_segment = normalized_members[initial_index]["segment"]
        correction_segment = normalized_members[correction_index]["segment"]
        if initial_segment["end_progress_index"] != correction_segment["start_progress_index"]:
            raise EditingCorpusContractError(
                "correction initial and continuation segments must be contiguous"
            )

    group_sha = _require_sha256(
        payload["group_sha256"],
        field="relationship_group.group_sha256",
    )
    group_without_hash = {key: value for key, value in payload.items() if key != "group_sha256"}
    if group_sha != _canonical_sha256(group_without_hash):
        raise EditingCorpusContractError("relationship group_sha256 disagrees")


def training_launch_blockers(contract: Mapping[str, Any]) -> list[str]:
    """Return every unresolved item that must block a new training launch."""

    validate_editing_corpus_contract(contract)
    blockers: list[str] = []
    if contract["training_authorized"] is not True:
        blockers.append("training_authorized is not true")
    if contract["status"] != "FROZEN_TRAINING_AUTHORIZED":
        blockers.append("status is not FROZEN_TRAINING_AUTHORIZED")

    threshold_values = contract["pretraining_gates"]["threshold_values"]
    unresolved_refs: set[tuple[str, str]] = set()
    for capability in contract["capability_coverage_contracts"]:
        for threshold_ref in capability["threshold_refs"]:
            if threshold_values[threshold_ref] is None:
                unresolved_refs.add((f"capability {capability['id']}", threshold_ref))
    for threshold_ref in contract["pretraining_gates"]["global_threshold_refs"]:
        if threshold_values[threshold_ref] is None:
            unresolved_refs.add(("global gate", threshold_ref))
    blockers.extend(
        f"unresolved threshold reference: {owner} -> {threshold_ref}"
        for owner, threshold_ref in sorted(unresolved_refs)
    )

    record_status = {
        cell["id"]: cell["admission_status"]
        for cell in contract["record_membership_contract"]["cells"]
    }
    referenced_cells = {
        cell_id
        for capability in contract["capability_coverage_contracts"]
        for cell_id in capability["required_corpus_record_cells"]
    }
    blockers.extend(
        f"unresolved record-membership cell: {cell_id} ({record_status[cell_id]})"
        for cell_id in sorted(referenced_cells)
        if record_status[cell_id] != "admissible"
    )
    blockers.extend(
        f"unresolved external authority requirement: {requirement['id']} ({requirement['status']})"
        for requirement in contract["admitted_record_contract"]["external_authority_requirements"]
        if requirement["status"] != "verified"
    )
    return blockers


def assert_training_launch_authorized(contract: Mapping[str, Any]) -> None:
    blockers = training_launch_blockers(contract)
    if blockers:
        raise EditingCorpusContractError(
            "editing-corpus contract blocks training: " + "; ".join(blockers)
        )
