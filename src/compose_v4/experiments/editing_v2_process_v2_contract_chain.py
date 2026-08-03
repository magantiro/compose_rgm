"""Deterministic builder and validator for the Process-V2 downstream contract chain.

WHAT THIS IS
------------
Editing-V2 froze a seven-artifact prospective contract chain under the V1 semantic
process identity (``configs/editing_v2_semantic_*_v1.json``).  Process-V2 changed
``atom_delete`` admission, which moved ``editing_process_v2_identity()``, so every
one of those artifacts describes a process that is no longer the live one.  Because
the V1 artifacts are frozen and must never be relabelled, the Process-V2 chain is a
SEPARATE set of seven configs that MIRRORS the V1 structure while binding the live
Process-V2 identity:

===  ==================================================================  ==========
 #   ``configs/editing_v2_process_v2_<kind>.json``                        mirrors
===  ==================================================================  ==========
 1   ``..._active8_decision_runtime.json``                                the V1 CPU decision runtime
 2   ``..._capability_cells.json``                                        the V1 capability-cell registry
 3   ``..._development_cell_roles.json``                                  the V1 development cell roles
 4   ``..._gate_zero_structural.json``                                    the V1 Gate-0 structural contract
 5   ``..._t1_panel_policy.json``                                         the V1 T1 panel policy
 6   ``..._t1_capacity_policy.json``                                      the V1 T1 capacity policy
 7   ``..._p50_recipe_policy.json``                                       the V1 P50 recipe policy
===  ==================================================================  ==========

MIRROR, NOT REDESIGN
--------------------
Every policy value -- cell definitions, family contexts, data lanes, partition roles,
the ``editing_v2_active8_v1`` cell namespace, the 17 required / 3 conditional / 2
separate-lane cell counts, the panel cardinalities, every threshold, the optimizer
and sampling laws -- is copied verbatim from its V1 counterpart and is asserted
identical by ``tests/test_editing_v2_process_v2_contract_chain.py``.  The ONLY
differences are the identity, the pin shape, and the authority envelope:

* each artifact binds ``editing_process_v2_identity()`` (V2), never the V1 identity;
* each artifact binds the Active8 operator order from ``ACTIVE8_FAMILIES``;
* every parent pointer is the Gate-0 canonical triple
  ``{"path", "file_sha256", "semantic_sha256"}``, replacing V1's several
  inconsistent pin spellings;
* every artifact carries exactly one self-hash field, ``contract_sha256``;
* every artifact carries all seven authority flags, every one ``false``.

SEMANTIC HASH RULE
------------------
``semantic_sha256`` is the target's OWN self-hash -- the unique top-level
``*_sha256`` field that equals the canonical hash of the rest of the object.  A JSON
target that declares no such field is pinned by the canonical hash of its whole body
(this is what ``configs/editing_corpus_v2_contract.json`` needs, and it matches the
existing pin in ``configs/editing_v2_candidate_provenance_decisions_v1.json``).  A
non-JSON target has no separable semantic body, so its semantic hash is its physical
hash.

WHAT THIS DOES NOT AUTHORIZE
----------------------------
Nothing.  These are prospective binding contracts, not evidence and not permission.
Every artifact sets ``training_authorized``, ``gate_zero_authorized``,
``t1_authorized``, ``bounded_p50_authorized``, ``long_training_authorized``,
``checkpoint_selection_authorized`` and ``final_test_selection_authorized`` to
``false``, and every ``status`` ends in ``_NO_DOWNSTREAM_AUTHORITY``.  Building or
validating this chain does not run Gate 0, T1 or P50, does not authorize a Modal
launch or any training, and does not revalidate evidence produced under a superseded
identity -- such evidence stays invalid regardless.  ``admitted_source`` is present
on every artifact with null members so that a later rebind run can only FILL it,
never introduce a field the chain never declared.

INVARIANTS MAINTAINED (and tested)
----------------------------------
* build -> serialize -> validate -> load round-trips for all seven artifacts;
* ``write_process_v2_chain`` is byte-stable across repeated runs and writes
  parents-first, recomputing each child after its parent is sealed;
* every parent pin resolves to the target's live physical and semantic hashes;
* a V1 identity pin, a ``_v1`` parent path, a changed Active8 order, a non-``False``
  authority flag, a disagreeing self-hash, and a changed field set each raise
  :class:`ProcessV2ChainError` with a message naming that specific defect.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_process_v2_identity,
    editing_v2_process_identity,
)

# ---- Errors ----


class ProcessV2ChainError(ValueError):
    """A Process-V2 chain artifact is unbuildable, mixed with V1, or inconsistent."""


# ---- Chain membership ----

ACTIVE8_DECISION_RUNTIME = "configs/editing_v2_process_v2_active8_decision_runtime.json"
CAPABILITY_CELLS = "configs/editing_v2_process_v2_capability_cells.json"
DEVELOPMENT_CELL_ROLES = "configs/editing_v2_process_v2_development_cell_roles.json"
GATE_ZERO_STRUCTURAL = "configs/editing_v2_process_v2_gate_zero_structural.json"
T1_PANEL_POLICY = "configs/editing_v2_process_v2_t1_panel_policy.json"
T1_CAPACITY_POLICY = "configs/editing_v2_process_v2_t1_capacity_policy.json"
P50_RECIPE_POLICY = "configs/editing_v2_process_v2_p50_recipe_policy.json"

#: The seven chain artifacts in dependency order; parents always precede children.
PROCESS_V2_CHAIN_ARTIFACTS: tuple[str, ...] = (
    ACTIVE8_DECISION_RUNTIME,
    CAPABILITY_CELLS,
    DEVELOPMENT_CELL_ROLES,
    GATE_ZERO_STRUCTURAL,
    T1_PANEL_POLICY,
    T1_CAPACITY_POLICY,
    P50_RECIPE_POLICY,
)

# Frozen external parents.  Neither is a V1-named artifact.
GATE_ZERO_MODEL_PROCESS_V2 = "configs/editing_gate_zero_semantic_model_process_v2.json"
SEMANTIC_PROCESS_V2 = "configs/editing_v2_semantic_process_v2.json"
EDITING_CORPUS_V2_CONTRACT = "configs/editing_corpus_v2_contract.json"
CAPABILITY_CELL_CLASSIFIER = "src/compose_v4/data/editing_v2_semantic_capability_cells.py"

#: The single self-hash field name every chain artifact carries.
SELF_HASH_FIELD = "contract_sha256"

#: The exact authority flags every chain artifact must carry, every one ``False``.
AUTHORITY_FIELDS: tuple[str, ...] = (
    "training_authorized",
    "gate_zero_authorized",
    "t1_authorized",
    "bounded_p50_authorized",
    "long_training_authorized",
    "checkpoint_selection_authorized",
    "final_test_selection_authorized",
)

STATUS_SUFFIX = "_NO_DOWNSTREAM_AUTHORITY"

ADMITTED_SOURCE_SCHEMA = "compose.data.process_v2_admitted_source"
ADMITTED_SOURCE_FIELDS: tuple[str, ...] = (
    "schema",
    "completion_sha256",
    "run_identity_sha256",
)

PROCESS_IDENTITY_PIN_FIELD = "process_identity_sha256"
PROCESS_V2_IDENTITY_PROVIDER = "editing_process_v2_identity"
PROCESS_IDENTITY_MODULE = "src/compose_v4/rewrite/editing_v2_process_identity.py"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_V1_STEM = re.compile(r"_v1(?:$|_)")

# ---- Policy values mirrored verbatim from the frozen V1 chain ----
#
# These are COPIES, never re-derivations.  ``tests/test_editing_v2_process_v2_
# contract_chain.py`` reads the V1 configs and asserts each block below is equal to
# its V1 counterpart, so a silent redesign here fails the suite.

_V1_RUNTIME_MODEL: dict[str, Any] = {
    "atom_vocabulary_class_count": 15,
    "candidate_cache_size": 4096,
    "candidate_time": 0.5,
    "catalog_fingerprint": "639ff6078c32d43c",
    "dtype": "torch.float32",
    "hidden_dim": 256,
    "initialization_seed": 20260730,
    "mark_dim": 32,
    "max_atoms": 40,
    "message_passing_steps": 6,
}

_V1_RUNTIME_SOFTWARE: dict[str, Any] = {
    "networkx": "3.3",
    "numpy": "1.26.4",
    "python": "3.11",
    "rdkit": "2024.3.5",
    "scipy": "1.13.1",
    "torch": "2.4.0",
}

_V1_CELL_SCOPE: dict[str, Any] = {
    "formal_charge_changes": "out_of_scope_charge_preserving_only",
    "generated_object": "canonical_molecular_successor",
    "maximum_active_atoms": 40,
    "objective_lane": "productive_embedded_jump_chain",
    "row_unit": "accepted_nonterminal_progress_row_with_exact_action_v4_teacher",
    "stereochemistry": "out_of_scope",
    "terminal_rows": "outside_capability_cell_registry_hazard_is_separate",
}

_V1_CELL_IDENTITY_POLICY: dict[str, Any] = {
    "balancing_dimensions": ["model_family", "family_specific_context"],
    "namespace": "editing_v2_active8_v1",
    "rationale": (
        "keep_the_balancing_registry_compact_and_retain_provenance_and_difficulty"
        "_for_stratified_audits_without_a_cartesian_product"
    ),
    "separate_nonbalancing_dimensions": [
        "data_lane",
        "partition_role",
        "record_membership_cells",
        "endpoint_evidence_roles",
        "path_evidence_roles",
        "raw_mark_count_stratum",
        "canonical_successor_count_stratum",
        "successor_alias_multiplicity_stratum",
        "matching_mark_count",
        "semantic_groups",
        "endpoint_descriptor",
        "path_descriptor",
        "relationship_group_ids",
        "mappings",
        "sampling_coefficients",
        "atom_element_transition",
        "minimum_edited_cycle_length",
        "source_edge_aromatic",
        "successor_edge_aromatic",
    ],
}

_V1_FAMILY_CONTEXTS: dict[str, Any] = {
    "atom_delete": [
        "singleton_to_null_death",
        "leaf_death",
        "connected_nonleaf_death",
    ],
    "atom_insert": ["root_birth", "one_neighbor_birth"],
    "atom_restate": ["element_identity_change", "valence_state_change"],
    "bond_reorder": ["bond_order_increase", "bond_order_decrease"],
    "bond_reroute": [
        "single_atom_pendant_acyclic_source",
        "multi_atom_pendant_acyclic_source",
        "single_atom_pendant_cyclic_source",
        "multi_atom_pendant_cyclic_source",
    ],
    "cycle_attach": [
        "open_from_monocyclic_ring_system",
        "open_from_articulated_polycyclic_ring_system",
        "open_from_nonarticulated_polycyclic_ring_system",
    ],
    "cycle_insert": [
        "close_to_monocyclic_ring_system",
        "close_to_articulated_polycyclic_ring_system",
        "close_to_nonarticulated_polycyclic_ring_system",
    ],
    "ring_system_restate": [
        "aromatization",
        "dearomatization",
        "coordinated_ring_bond_state_change",
    ],
}

_V1_EXACT_EVIDENCE_STRATA: dict[str, Any] = {
    "canonical_successor_count": {
        "bins": [
            {"id": "successors_001_004", "maximum": 4, "minimum": 1},
            {"id": "successors_005_016", "maximum": 16, "minimum": 5},
            {"id": "successors_017_064", "maximum": 64, "minimum": 17},
            {"id": "successors_065_plus", "maximum": None, "minimum": 65},
        ],
        "scope": "all_productive_canonical_molecular_successors_at_the_exact_source",
    },
    "policy": (
        "prospective_non_data_derived_logarithmic_engineering_bins"
        "_not_balancing_dimensions"
    ),
    "raw_mark_count": {
        "bins": [
            {"id": "marks_001_004", "maximum": 4, "minimum": 1},
            {"id": "marks_005_016", "maximum": 16, "minimum": 5},
            {"id": "marks_017_064", "maximum": 64, "minimum": 17},
            {"id": "marks_065_plus", "maximum": None, "minimum": 65},
        ],
        "scope": "all_productive_action_v4_marks_emitted_at_the_exact_source",
    },
    "successor_alias_multiplicity": {
        "bins": [
            {"id": "aliases_001", "maximum": 1, "minimum": 1},
            {"id": "aliases_002_004", "maximum": 4, "minimum": 2},
            {"id": "aliases_005_plus", "maximum": None, "minimum": 5},
        ],
        "scope": "all_action_v4_marks_in_the_teacher_canonical_successor_fiber",
    },
}

_V1_FAIL_CLOSED_POLICY: dict[str, Any] = {
    "lane_or_partition_mismatch": "error",
    "missing_exact_candidate_evidence": "error",
    "process_identity_drift": "error",
    "registry_tamper": "error",
    "terminal_progress_row": "error",
    "unknown_family": "error",
    "unknown_family_context": "error",
    "unsupported_teacher": "error",
}

_V1_DATA_LANES: list[str] = [
    "observed_local_analogue",
    "operator_aware_real_endpoint",
    "linker_positional_topology_analogue",
    "real_endpoint_multistep_path",
    "reversible_synthetic_walk",
]

_V1_PARTITION_ROLES: list[str] = [
    "train",
    "validation",
    "controller_validation",
    "final_test",
]

_V1_ACTION_CODEC_SCHEMA_VERSION = 4

_V1_REQUIRED_CELL_IDS: list[str] = [
    "editing_v2_active8_v1:atom_insert:one_neighbor_birth",
    "editing_v2_active8_v1:atom_delete:leaf_death",
    "editing_v2_active8_v1:atom_delete:connected_nonleaf_death",
    "editing_v2_active8_v1:atom_restate:element_identity_change",
    "editing_v2_active8_v1:atom_restate:valence_state_change",
    "editing_v2_active8_v1:bond_reorder:bond_order_increase",
    "editing_v2_active8_v1:bond_reorder:bond_order_decrease",
    "editing_v2_active8_v1:bond_reroute:single_atom_pendant_acyclic_source",
    "editing_v2_active8_v1:bond_reroute:multi_atom_pendant_acyclic_source",
    "editing_v2_active8_v1:bond_reroute:single_atom_pendant_cyclic_source",
    "editing_v2_active8_v1:bond_reroute:multi_atom_pendant_cyclic_source",
    "editing_v2_active8_v1:cycle_insert:close_to_monocyclic_ring_system",
    "editing_v2_active8_v1:cycle_insert:close_to_nonarticulated_polycyclic_ring_system",
    "editing_v2_active8_v1:cycle_attach:open_from_monocyclic_ring_system",
    "editing_v2_active8_v1:cycle_attach:open_from_nonarticulated_polycyclic_ring_system",
    "editing_v2_active8_v1:ring_system_restate:aromatization",
    "editing_v2_active8_v1:ring_system_restate:dearomatization",
]

_V1_CONDITIONAL_CELL_IDS: list[str] = [
    "editing_v2_active8_v1:cycle_insert:close_to_articulated_polycyclic_ring_system",
    "editing_v2_active8_v1:cycle_attach:open_from_articulated_polycyclic_ring_system",
    "editing_v2_active8_v1:ring_system_restate:coordinated_ring_bond_state_change",
]

_V1_SEPARATE_LANE_CELL_IDS: list[str] = [
    "editing_v2_active8_v1:atom_insert:root_birth",
    "editing_v2_active8_v1:atom_delete:singleton_to_null_death",
]

_V1_CONDITIONAL_POLICY: dict[str, Any] = {
    "gate_zero_nonempty_required": False,
    "included_in_balanced_editing_p50": False,
    "included_in_unique_state_editing_t1": False,
    "reported_as_audit_context": True,
    "scientific_role": "conditional_source_conditioned_editing",
    "validate_when_present": True,
}

_V1_SEPARATE_LANE_POLICY: dict[str, Any] = {
    "editing_v2_authority": False,
    "gate_zero_nonempty_required": False,
    "included_in_balanced_editing_p50": False,
    "included_in_unique_state_editing_t1": False,
    "reported_as_audit_context": True,
    "scientific_role": "separate_timed_de_novo_null_boundary",
}

_V1_PARTITION_POLICY: dict[str, Any] = {
    "conditional_cell_count": 3,
    "registered_cells_must_be_classified_exactly_once": True,
    "required_cell_count": 17,
    "role_assignment_frozen_before_gate_zero_outputs": True,
    "separate_lane_cell_count": 2,
}

_V1_CELL_ROLE_SCOPE = "source_conditioned_editing_gate0_t1_p50"

_V1_REQUIRED_ARCHITECTURE: dict[str, Any] = {
    "atom_vocabulary_class_count": 15,
    "catalog_fingerprint": "639ff6078c32d43c",
    "dtype": "torch.float32",
    "hidden_dim": 256,
    "initialization_seed": 20260730,
    "mark_dim": 32,
    "max_atoms": 40,
    "message_passing_steps": 6,
}

_V1_STRUCTURAL_CHECKS: dict[str, Any] = {
    "capability_context_policy": (
        "classify_every_decision_eligible_teacher_require_frozen_required_editing"
        "_cells_report_conditional_and_separate_lane_contexts"
    ),
    "classification_failure_policy": "publish_bounded_typed_negative_receipts_and_fail",
    "classification_failure_receipt_limit": 100,
    "decision_eligible_partition_roles": ["train"],
    "excluded_trace_policy": "preserve_and_report_never_reclassify_as_teacher_coverage",
    "legacy_action_v2_evidence": "forbidden",
    "require_every_active8_family": True,
    "require_every_teacher_supported": True,
    "require_exactly_one_matching_mark": True,
    "require_one_assignment_per_accepted_action": True,
    "require_positive_canonical_successor_count": True,
    "require_positive_raw_mark_count": True,
    "require_positive_successor_alias_count": True,
    "require_zero_terminal_assignments": True,
    "sealed_nondecision_partition_roles": [
        "controller_validation",
        "final_test",
        "validation",
    ],
    "teacher_unit": "accepted_nonterminal_action_v4_transition",
    "trace_selection": "accepted_whole_traces_only",
}

_V1_DECISION_POLICY: dict[str, Any] = {
    "failure_policy": "publish_fail_closed_negative_evidence",
    "pass_grants_checkpoint_selection_authority": False,
    "pass_grants_final_test_selection_authority": False,
    "pass_grants_gate_zero_authority": False,
    "pass_grants_long_training_authority": False,
    "pass_grants_p50_authority": False,
    "pass_grants_t1_authority": False,
    "pass_grants_training_authority": False,
    "pass_meaning": (
        "structural_evidence_complete_for_all_active8_families_and_required_editing"
        "_cells_in_train_only"
    ),
}

_V1_PANEL_BODY: dict[str, Any] = {
    "cache_handoff": "complete_train_trace_union_for_cpu_successor_fiber_cache_v1",
    "empirical_multiplicity_receipts_included": False,
    "gate_thresholds_included": False,
    "hazard_included": False,
    "maximum_entries_by_family": {
        "atom_delete": 128,
        "atom_insert": 128,
        "atom_restate": 128,
        "bond_reorder": 128,
        "bond_reroute": 128,
        "cycle_attach": 128,
        "cycle_insert": 128,
        "ring_system_restate": 128,
    },
    "minimum_entries_by_family": {
        "atom_delete": 64,
        "atom_insert": 64,
        "atom_restate": 64,
        "bond_reorder": 64,
        "bond_reroute": 64,
        "cycle_attach": 64,
        "cycle_insert": 64,
        "ring_system_restate": 64,
    },
    "objective_unit": "exact_source_frozen_time_canonical_successor",
    "optimizer_policy_included": False,
    "p50_policy_included": False,
    "panel_kind": "unique_state_single_target_canonical_successor_capacity",
    "repeated_state_panel_included": False,
    "successor_fiber_cache_compiled": False,
    "support_time_hex": "0x1.0000000000000p-1",
}

_V1_CAPACITY_BODY: dict[str, Any] = {
    "empirical_repeated_state_gate": {
        "bounded_p50_capacity_prerequisite": False,
        "empirical_law_claims_authorized": False,
        "mark_alias_multiplicity_is_observation_count": False,
        "raw_record_multiplicity_is_observation_count": False,
        "required_receipt_kind": "independent_empirical_transition_v1",
        "status": "BLOCKED_PENDING_VERIFIED_INDEPENDENT_OBSERVATION_RECEIPTS",
    },
    "hazard_included": False,
    "objective_unit": "exact_source_frozen_time_canonical_successor",
    "optimization": {
        "accelerator_class": "gpu",
        "address_stream": "sha256_counter_stream_bound_to_policy_and_panel_identity",
        "batch_size": 64,
        "checkpoint_selection": (
            "maximize_minimum_entry_teacher_successor_probability__tie_lower_mean_nll"
            "__tie_earlier_step"
        ),
        "deterministic_algorithms_required": True,
        "dtype": "float32",
        "early_stop_rule": (
            "all_required_family_nonempty_cell_and_entry_thresholds_pass_at_one"
            "_evaluated_state"
        ),
        "failure_diagnostic_scope_order": [
            "heads_only",
            "heads_plus_local_adapter_if_distinct",
            "all",
        ],
        "failure_diagnostics_only_for_failing_families": True,
        "gradient_clip_norm": 10.0,
        "learning_rate": 0.001,
        "maximum_optimizer_steps": 500,
        "mixed_precision": False,
        "model_initialization": "scratch",
        "optimizer": "adamw",
        "parameter_scope": "all_trainable_active8_parameters",
        "report_points": [1, 10, 50, 100, 250, 500],
        "scheduler": "constant",
        "seed": 31,
        "trajectory_evaluation": "every_pre_update_state_and_terminal_state",
        "weight_decay": 0.0,
    },
    "panel_cardinality": {
        "maximum_entries_by_family": {
            "atom_delete": 128,
            "atom_insert": 128,
            "atom_restate": 128,
            "bond_reorder": 128,
            "bond_reroute": 128,
            "cycle_attach": 128,
            "cycle_insert": 128,
            "ring_system_restate": 128,
        },
        "minimum_entries_by_family": {
            "atom_delete": 64,
            "atom_insert": 64,
            "atom_restate": 64,
            "bond_reorder": 64,
            "bond_reroute": 64,
            "cycle_attach": 64,
            "cycle_insert": 64,
            "ring_system_restate": 64,
        },
    },
    "panel_kind": "unique_state_single_target_canonical_successor_capacity",
    "required_families": [
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    ],
    "sampling_law": {
        "importance_correction": "none",
        "order": [
            "model_family",
            "semantic_capability_cell",
            "unique_panel_entry",
        ],
        "probability_within_each_level": "uniform_over_nonempty_children",
        "target_coefficient": 1.0,
    },
    "support_time_hex": "0x1.0000000000000p-1",
    "thresholds": {
        "maximum_nonempty_cell_teacher_successor_nll": 0.22314355131420976,
        "maximum_unique_state_teacher_successor_nll": 0.22314355131420976,
        "minimum_every_unique_entry_teacher_successor_probability": 0.8,
        "minimum_nonempty_cell_teacher_successor_probability": 0.8,
        "minimum_nonempty_cell_teacher_successor_top1": 0.95,
        "minimum_unique_state_teacher_successor_probability": 0.8,
        "minimum_unique_state_teacher_successor_top1": 0.95,
        "require_every_unique_entry_teacher_successor_top1": True,
        "require_finite_nonzero_action_route_gradient_each_family": True,
        "require_finite_nonzero_family_gate_gradient_each_family": True,
    },
}

_V1_P50_BODY: dict[str, Any] = {
    "active_families": [
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_reorder",
        "bond_reroute",
        "cycle_insert",
        "cycle_attach",
        "ring_system_restate",
    ],
    "cache": {
        "closure_only_rows_schedulable": False,
        "coverage_mode": "complete_trace_closure_of_planned_address_union",
        "source_population": "full_admitted_semantic_active8_corpus",
        "terminal_rows_schedulable": False,
    },
    "objective": {
        "hazard_included": False,
        "hazard_weight": 0.0,
        "importance_correction": "none",
        "name": "balanced_semantic_cell_productive_identity",
        "path_position_coefficient": 1.0,
        "terminal_rows": "excluded",
        "unit": "productive_embedded_canonical_successor",
    },
    "optimization": {
        "batch_size": 64,
        "deterministic_algorithms_required": True,
        "dtype": "float32",
        "gradient_clip_norm": 10.0,
        "initialization": "scratch_from_t1_bound_initial_model_state",
        "learning_rate": 0.001,
        "mixed_precision": False,
        "optimizer": "adamw",
        "optimizer_steps": 50,
        "resume": False,
        "scheduled_nonterminal_examples": 3200,
        "scheduler": "constant",
        "seed": 31,
        "weight_decay": 0.0,
    },
    "p500_authorized": False,
    "required_physical_binding_purposes": [
        "stream_union_successor_cache",
        "validation_baseline",
        "trainer_runtime",
        "execution_environment",
        "launch_projection",
    ],
    "sampling": {
        "lane_probability": "not_a_sampling_level_preserved_as_audit_dimension",
        "policy_id": "stage_a_equal_semantic_cell_capability_exception_v1",
        "production_hierarchy_claim_authorized": False,
        "rationale": (
            "P50 tests load-bearing capability acquisition; production-law "
            "calibration remains a later separately frozen stage"
        ),
        "semantic_cell_probability": "equal_round_robin",
        "source_group_id_in_stream": None,
        "source_group_probability": "explicit_stage_a_exception_not_used",
        "within_cell_probability": "deterministic_uniform_cycle",
    },
    "scientific_scope": (
        "scratch_active8_stage_a_capability_pilot_not_production_law_calibration"
    ),
    "thresholds": {
        "baseline_definition": (
            "exact_pre_update_scratch_evaluation_on_the_frozen_validation_stream"
        ),
        "baseline_partition_role": "validation",
        "baseline_values_inspected_when_thresholds_frozen": False,
        "catastrophic_regression_sentinel_rationale": (
            "plus_0_25_nats_is_a_separate_abort_sentinel_and_not_a_learning_criterion"
        ),
        "gradient_floor_rule": (
            "max_1_ceil_fraction_times_planned_optimizer_step_opportunities"
        ),
        "gradient_opportunity_fraction": 0.8,
        "maximum_cell_final_minus_baseline_for_p50_nonincrease_nats": 1e-07,
        "maximum_cell_final_minus_baseline_successor_nll_nats": 0.25,
        "maximum_family_final_minus_baseline_for_p50_nonincrease_nats": 1e-07,
        "maximum_family_final_minus_baseline_successor_nll_nats": 0.25,
        "p50_nonincrease_numerical_equivalence_rationale": (
            "one_e_minus_seven_nats_allows_only_float32_reduction_equivalence"
            "_not_regression"
        ),
        "zero_planned_family_or_cell_opportunities_allowed": False,
    },
    "time_derivation": {
        "algorithm": "sha256_counter_open_unit_interval_53bit_v1",
        "seed": 31,
        "serialized_value": "python_float_hex_v1",
        "support": "strict_open_unit_interval",
    },
}

# ---- Per-artifact envelope metadata ----

_ENVELOPE: dict[str, dict[str, str]] = {
    ACTIVE8_DECISION_RUNTIME: {
        "schema": "compose.editing_v2.process_v2_active8_decision_runtime",
        "contract_id": "editing_v2_process_v2_active8_decision_runtime",
        "status": "FROZEN_PROCESS_V2_CPU_DECISION_RUNTIME" + STATUS_SUFFIX,
    },
    CAPABILITY_CELLS: {
        "schema": "compose.editing_v2.process_v2_capability_cells",
        "contract_id": "editing_v2_process_v2_capability_cells",
        "status": "FROZEN_PROCESS_V2_SEMANTIC_CAPABILITY_CELLS" + STATUS_SUFFIX,
    },
    DEVELOPMENT_CELL_ROLES: {
        "schema": "compose.editing_v2.process_v2_development_cell_roles",
        "contract_id": "editing_v2_process_v2_development_cell_roles",
        "status": "FROZEN_PROCESS_V2_DEVELOPMENT_CELL_ROLES" + STATUS_SUFFIX,
    },
    GATE_ZERO_STRUCTURAL: {
        "schema": "compose.editing_v2.process_v2_gate_zero_structural",
        "contract_id": "editing_v2_process_v2_gate_zero_structural",
        "status": "FROZEN_PROCESS_V2_GATE_ZERO_STRUCTURAL" + STATUS_SUFFIX,
    },
    T1_PANEL_POLICY: {
        "schema": "compose.editing_v2.process_v2_t1_panel_policy",
        "contract_id": "editing_v2_process_v2_t1_panel_policy",
        "status": "FROZEN_PROCESS_V2_T1_PANEL_POLICY" + STATUS_SUFFIX,
    },
    T1_CAPACITY_POLICY: {
        "schema": "compose.editing_v2.process_v2_t1_capacity_policy",
        "contract_id": "editing_v2_process_v2_t1_capacity_policy",
        "status": "FROZEN_PROCESS_V2_T1_CAPACITY_POLICY" + STATUS_SUFFIX,
    },
    P50_RECIPE_POLICY: {
        "schema": "compose.editing_v2.process_v2_p50_recipe_policy",
        "contract_id": "editing_v2_process_v2_p50_recipe_policy",
        "status": "FROZEN_PROCESS_V2_P50_RECIPE_POLICY" + STATUS_SUFFIX,
    },
}

#: ``artifact -> {parent role: parent relative path}``.  No value is a V1 config.
_PARENTS: dict[str, dict[str, str]] = {
    ACTIVE8_DECISION_RUNTIME: {
        "semantic_model_process": GATE_ZERO_MODEL_PROCESS_V2,
    },
    CAPABILITY_CELLS: {
        "classifier_implementation": CAPABILITY_CELL_CLASSIFIER,
        "editing_corpus_contract": EDITING_CORPUS_V2_CONTRACT,
    },
    DEVELOPMENT_CELL_ROLES: {
        "capability_cell_registry": CAPABILITY_CELLS,
    },
    GATE_ZERO_STRUCTURAL: {
        "capability_cell_registry": CAPABILITY_CELLS,
        "decision_runtime": ACTIVE8_DECISION_RUNTIME,
        "development_cell_roles": DEVELOPMENT_CELL_ROLES,
        "semantic_model_process": GATE_ZERO_MODEL_PROCESS_V2,
        "semantic_process": SEMANTIC_PROCESS_V2,
    },
    T1_PANEL_POLICY: {
        "development_cell_roles": DEVELOPMENT_CELL_ROLES,
    },
    T1_CAPACITY_POLICY: {
        "development_cell_roles": DEVELOPMENT_CELL_ROLES,
        "t1_panel_policy": T1_PANEL_POLICY,
    },
    P50_RECIPE_POLICY: {
        "development_cell_roles": DEVELOPMENT_CELL_ROLES,
    },
}

# ---- Canonical hashing ----


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def process_v2_chain_self_hash(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 of ``payload`` MINUS its ``contract_sha256``.

    This is the one self-hash equation the whole chain uses, and it is the same
    equation ``scripts/verify_process_v2_hash_chain.py`` uses to DISCOVER a
    self-hash field, so a chain artifact self-identifies to that verifier without
    the verifier knowing this module exists.
    """

    return _canonical_sha256(
        {key: value for key, value in payload.items() if key != SELF_HASH_FIELD}
    )


def _declared_self_hash(payload: Mapping[str, Any]) -> str | None:
    """Return the target's own self-hash, found by the equation, not by name.

    Raises:
        ProcessV2ChainError: if more than one top-level field satisfies it, which
            would make ``semantic_sha256`` ambiguous.
    """

    found = [
        key
        for key, value in payload.items()
        if key.endswith("_sha256")
        and isinstance(value, str)
        and _HEX64.match(value)
        and value
        == _canonical_sha256({k: v for k, v in payload.items() if k != key})
    ]
    if len(found) > 1:
        raise ProcessV2ChainError(
            f"target declares {len(found)} self-hash fields {sorted(found)}; "
            "semantic_sha256 would be ambiguous"
        )
    return payload[found[0]] if found else None


# ---- Pin construction ----


def _read_parent_bytes(repo_root: Path, relative_path: str) -> bytes:
    path = repo_root / relative_path
    try:
        return path.read_bytes()
    except OSError as error:
        raise ProcessV2ChainError(
            f"parent {relative_path} is not readable; build the chain "
            "parents-first with write_process_v2_chain()"
        ) from error


def _semantic_sha256(relative_path: str, raw: bytes) -> str:
    """The semantic hash of a pin target under the rule stated in the module docstring."""

    if not relative_path.endswith(".json"):
        return _sha256_bytes(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProcessV2ChainError(f"parent {relative_path} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ProcessV2ChainError(f"parent {relative_path} is not a JSON object")
    declared = _declared_self_hash(payload)
    return declared if declared is not None else _canonical_sha256(payload)


def _pin(repo_root: Path, relative_path: str) -> dict[str, str]:
    """Build the Gate-0 canonical triple for one pin target."""

    raw = _read_parent_bytes(repo_root, relative_path)
    return {
        "file_sha256": _sha256_bytes(raw),
        "path": relative_path,
        "semantic_sha256": _semantic_sha256(relative_path, raw),
    }


def _parents_block(repo_root: Path, name: str) -> dict[str, dict[str, str]]:
    return {
        role: _pin(repo_root, relative_path)
        for role, relative_path in sorted(_PARENTS[name].items())
    }


def _process_identity_block() -> dict[str, Any]:
    """Bind the LIVE Process-V2 identity; never a stored or V1 value."""

    identity = editing_process_v2_identity()
    return {
        "identity_schema": str(identity["schema"]),
        "identity_schema_version": int(identity["schema_version"]),  # type: ignore[call-overload]
        "module": PROCESS_IDENTITY_MODULE,
        PROCESS_IDENTITY_PIN_FIELD: str(identity[PROCESS_IDENTITY_PIN_FIELD]),
        "process_semantics": str(identity["process_semantics"]),
        "provider": PROCESS_V2_IDENTITY_PROVIDER,
    }


def _admitted_source_block() -> dict[str, Any]:
    """The rebind-run binding slot: always present, null until a run fills it."""

    return {
        "completion_sha256": None,
        "run_identity_sha256": None,
        "schema": ADMITTED_SOURCE_SCHEMA,
    }


def _operator_capability_fingerprint(repo_root: Path) -> str:
    """Read the V2 operator capability expectation from the pinned Gate-0 contract.

    Reading it from the parent rather than restating it means the runtime's
    expectation cannot silently disagree with the contract this artifact pins.
    """

    payload = json.loads(_read_parent_bytes(repo_root, GATE_ZERO_MODEL_PROCESS_V2))
    fingerprint = payload.get("model_identity", {}).get("operator_capability_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ProcessV2ChainError(
            f"{GATE_ZERO_MODEL_PROCESS_V2} declares no "
            "model_identity.operator_capability_fingerprint"
        )
    return fingerprint


# ---- Builders ----


def _envelope(repo_root: Path, name: str) -> dict[str, Any]:
    meta = _ENVELOPE[name]
    payload: dict[str, Any] = {
        "active_families": list(ACTIVE8_FAMILIES),
        "admitted_source": _admitted_source_block(),
        "contract_id": meta["contract_id"],
        "parents": _parents_block(repo_root, name),
        "process_identity": _process_identity_block(),
        "schema": meta["schema"],
        "schema_version": 1,
        "status": meta["status"],
    }
    payload.update({field: False for field in AUTHORITY_FIELDS})
    return payload


def _build_decision_runtime(repo_root: Path) -> dict[str, Any]:
    model = dict(_V1_RUNTIME_MODEL)
    model["operator_capability_fingerprint"] = _operator_capability_fingerprint(repo_root)
    payload = _envelope(repo_root, ACTIVE8_DECISION_RUNTIME)
    payload["model"] = model
    payload["software"] = json.loads(json.dumps(_V1_RUNTIME_SOFTWARE))
    return payload


def _build_capability_cells(repo_root: Path) -> dict[str, Any]:
    payload = _envelope(repo_root, CAPABILITY_CELLS)
    payload["bindings"] = {
        "action_codec_schema_version": _V1_ACTION_CODEC_SCHEMA_VERSION,
        "data_lanes": list(_V1_DATA_LANES),
        "partition_roles": list(_V1_PARTITION_ROLES),
    }
    payload["cell_identity_policy"] = json.loads(json.dumps(_V1_CELL_IDENTITY_POLICY))
    payload["exact_evidence_strata"] = json.loads(json.dumps(_V1_EXACT_EVIDENCE_STRATA))
    payload["fail_closed_policy"] = json.loads(json.dumps(_V1_FAIL_CLOSED_POLICY))
    payload["family_contexts"] = json.loads(json.dumps(_V1_FAMILY_CONTEXTS))
    payload["scope"] = json.loads(json.dumps(_V1_CELL_SCOPE))
    return payload


def _build_development_cell_roles(repo_root: Path) -> dict[str, Any]:
    payload = _envelope(repo_root, DEVELOPMENT_CELL_ROLES)
    payload["conditional_cell_ids"] = list(_V1_CONDITIONAL_CELL_IDS)
    payload["conditional_policy"] = json.loads(json.dumps(_V1_CONDITIONAL_POLICY))
    payload["partition_policy"] = json.loads(json.dumps(_V1_PARTITION_POLICY))
    payload["required_cell_ids"] = list(_V1_REQUIRED_CELL_IDS)
    payload["scope"] = _V1_CELL_ROLE_SCOPE
    payload["separate_lane_cell_ids"] = list(_V1_SEPARATE_LANE_CELL_IDS)
    payload["separate_lane_policy"] = json.loads(json.dumps(_V1_SEPARATE_LANE_POLICY))
    return payload


def _build_gate_zero_structural(repo_root: Path) -> dict[str, Any]:
    payload = _envelope(repo_root, GATE_ZERO_STRUCTURAL)
    payload["decision_policy"] = json.loads(json.dumps(_V1_DECISION_POLICY))
    payload["required_architecture"] = json.loads(json.dumps(_V1_REQUIRED_ARCHITECTURE))
    payload["structural_checks"] = json.loads(json.dumps(_V1_STRUCTURAL_CHECKS))
    return payload


def _build_t1_panel_policy(repo_root: Path) -> dict[str, Any]:
    payload = _envelope(repo_root, T1_PANEL_POLICY)
    payload.update(json.loads(json.dumps(_V1_PANEL_BODY)))
    return payload


def _build_t1_capacity_policy(repo_root: Path) -> dict[str, Any]:
    payload = _envelope(repo_root, T1_CAPACITY_POLICY)
    payload.update(json.loads(json.dumps(_V1_CAPACITY_BODY)))
    return payload


def _build_p50_recipe_policy(repo_root: Path) -> dict[str, Any]:
    payload = _envelope(repo_root, P50_RECIPE_POLICY)
    payload.update(json.loads(json.dumps(_V1_P50_BODY)))
    return payload


_BUILDERS = {
    ACTIVE8_DECISION_RUNTIME: _build_decision_runtime,
    CAPABILITY_CELLS: _build_capability_cells,
    DEVELOPMENT_CELL_ROLES: _build_development_cell_roles,
    GATE_ZERO_STRUCTURAL: _build_gate_zero_structural,
    T1_PANEL_POLICY: _build_t1_panel_policy,
    T1_CAPACITY_POLICY: _build_t1_capacity_policy,
    P50_RECIPE_POLICY: _build_p50_recipe_policy,
}


def _require_known(name: str) -> str:
    if name not in _BUILDERS:
        raise ProcessV2ChainError(
            f"{name!r} is not a Process-V2 chain artifact; "
            f"expected one of {list(PROCESS_V2_CHAIN_ARTIFACTS)}"
        )
    return name


def build_process_v2_chain_artifact(name: str, *, repo_root: Path) -> dict[str, Any]:
    """Build one chain artifact, sealed with its ``contract_sha256``.

    The artifact's parents must already exist on disk: a child pins its parent's
    FINAL bytes, so building a child before its parent is sealed is an error rather
    than a stale pin.

    Raises:
        ProcessV2ChainError: on an unknown name or an unreadable/ambiguous parent.
    """

    payload = _BUILDERS[_require_known(name)](Path(repo_root))
    payload[SELF_HASH_FIELD] = process_v2_chain_self_hash(payload)
    return payload


def serialize_process_v2_chain_artifact(payload: Mapping[str, Any]) -> bytes:
    """Serialize to the chain's byte-stable form: ``indent=2, sort_keys=True`` + ``\\n``."""

    text = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode()


# ---- Validation ----


def _fail(name: str, detail: str) -> None:
    raise ProcessV2ChainError(f"{name}: {detail}")


def _first_difference(observed: object, expected: object, path: str = "") -> str | None:
    """Return a dotted path to the first structural difference, or ``None``."""

    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        for key in sorted(set(expected) | set(observed)):
            if key not in observed:
                return f"{path}.{key} (missing)"
            if key not in expected:
                return f"{path}.{key} (unexpected)"
            found = _first_difference(observed[key], expected[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if (
        isinstance(expected, Sequence)
        and isinstance(observed, Sequence)
        and not isinstance(expected, str)
        and not isinstance(observed, str)
    ):
        if len(expected) != len(observed):
            return f"{path} (length {len(observed)} != {len(expected)})"
        for index, (left, right) in enumerate(zip(observed, expected)):
            found = _first_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if type(observed) is not type(expected) or observed != expected:
        return f"{path} ({observed!r} != {expected!r})"
    return None


def _check_authority(name: str, payload: Mapping[str, Any]) -> None:
    for field in AUTHORITY_FIELDS:
        if field not in payload:
            _fail(name, f"authority flag {field!r} is missing")
        if payload[field] is not False:
            _fail(
                name,
                f"authority flag {field!r} is {payload[field]!r}, not exactly False; "
                "no Process-V2 chain artifact may grant downstream authority",
            )


def _check_active8(name: str, payload: Mapping[str, Any]) -> None:
    observed = payload.get("active_families")
    if not isinstance(observed, list) or tuple(observed) != tuple(ACTIVE8_FAMILIES):
        _fail(
            name,
            f"active_families is {observed!r}, which differs from the Active8 "
            f"operator order {list(ACTIVE8_FAMILIES)}",
        )


def _check_process_identity(name: str, payload: Mapping[str, Any]) -> None:
    block = payload.get("process_identity")
    if not isinstance(block, Mapping) or PROCESS_IDENTITY_PIN_FIELD not in block:
        _fail(name, "process_identity block is missing its process_identity_sha256 pin")
    assert isinstance(block, Mapping)  # narrowed by the guard above
    pinned = block[PROCESS_IDENTITY_PIN_FIELD]
    live_v2 = str(editing_process_v2_identity()[PROCESS_IDENTITY_PIN_FIELD])
    if pinned == live_v2:
        return
    live_v1 = str(editing_v2_process_identity()[PROCESS_IDENTITY_PIN_FIELD])
    if pinned == live_v1:
        _fail(
            name,
            "process_identity.process_identity_sha256 binds the V1 semantic process "
            f"identity {live_v1}; the Process-V2 chain must bind the live Process-V2 "
            f"identity {live_v2}",
        )
    _fail(
        name,
        f"process_identity.process_identity_sha256 is {pinned!r}, which is not the "
        f"live Process-V2 identity {live_v2}",
    )


def _check_parents(name: str, payload: Mapping[str, Any], repo_root: Path) -> None:
    block = payload.get("parents")
    expected_roles = set(_PARENTS[name])
    if not isinstance(block, Mapping):
        _fail(name, "parents block is missing or is not an object")
    assert isinstance(block, Mapping)  # narrowed by the guard above
    if set(block) != expected_roles:
        _fail(
            name,
            f"parent roles {sorted(block)} differ from {sorted(expected_roles)}",
        )
    for role in sorted(expected_roles):
        pin = block[role]
        if not isinstance(pin, Mapping) or set(pin) != {
            "path",
            "file_sha256",
            "semantic_sha256",
        }:
            _fail(
                name,
                f"parent {role!r} is not the canonical triple "
                "{path, file_sha256, semantic_sha256}",
            )
        assert isinstance(pin, Mapping)  # narrowed by the guard above
        path = pin["path"]
        if not isinstance(path, str):
            _fail(name, f"parent {role!r} path is not a string")
        if _V1_STEM.search(Path(str(path)).stem):
            _fail(
                name,
                f"parent {role!r} points at the V1 artifact {path!r}; the Process-V2 "
                "chain must never bind a _v1 config",
            )
        if path != _PARENTS[name][role]:
            _fail(
                name,
                f"parent {role!r} points at {path!r}, not {_PARENTS[name][role]!r}",
            )
        live = _pin(repo_root, _PARENTS[name][role])
        for field in ("file_sha256", "semantic_sha256"):
            if pin[field] != live[field]:
                _fail(
                    name,
                    f"parent {role!r} pins {path} {field} at {pin[field]!r}, "
                    f"but the live value is {live[field]!r}",
                )


def _check_admitted_source(name: str, payload: Mapping[str, Any]) -> None:
    block = payload.get("admitted_source")
    if not isinstance(block, Mapping) or set(block) != set(ADMITTED_SOURCE_FIELDS):
        _fail(
            name,
            "admitted_source must carry exactly "
            f"{sorted(ADMITTED_SOURCE_FIELDS)}; a later run may fill it, never "
            "extend it",
        )
    assert isinstance(block, Mapping)  # narrowed by the guard above
    if block["schema"] != ADMITTED_SOURCE_SCHEMA:
        _fail(name, f"admitted_source.schema must be {ADMITTED_SOURCE_SCHEMA!r}")
    for field in ("completion_sha256", "run_identity_sha256"):
        value = block[field]
        if value is None:
            continue
        if not isinstance(value, str) or not _HEX64.match(value):
            _fail(
                name,
                f"admitted_source.{field} must be null or a 64-hex sha256, "
                f"got {value!r}",
            )


def validate_process_v2_chain_artifact(
    value: object, *, name: str, repo_root: Path
) -> dict[str, Any]:
    """Validate one chain artifact against the live chain and return it.

    Every mixed-V1/V2 defect is reported with a message naming that specific defect
    rather than collapsing into a generic hash mismatch, because a chain that fails
    only at the self-hash tells a reader nothing about WHICH binding drifted.

    Raises:
        ProcessV2ChainError: on any disagreement.
    """

    _require_known(name)
    repo_root = Path(repo_root)
    if not isinstance(value, Mapping):
        _fail(name, f"artifact must be a JSON object, got {type(value).__name__}")
    assert isinstance(value, Mapping)  # narrowed by the guard above
    payload: dict[str, Any] = dict(value)

    expected = build_process_v2_chain_artifact(name, repo_root=repo_root)
    if set(payload) != set(expected):
        missing = sorted(set(expected) - set(payload))
        unexpected = sorted(set(payload) - set(expected))
        _fail(
            name,
            f"field set differs from the deterministic rebuild; missing={missing} "
            f"unexpected={unexpected}",
        )

    status = payload.get("status")
    if not isinstance(status, str) or not status.endswith(STATUS_SUFFIX):
        _fail(name, f"status {status!r} must end in {STATUS_SUFFIX!r}")
    _check_authority(name, payload)
    _check_active8(name, payload)
    _check_process_identity(name, payload)
    _check_parents(name, payload, repo_root)
    _check_admitted_source(name, payload)

    difference = _first_difference(
        {k: v for k, v in payload.items() if k != SELF_HASH_FIELD},
        {k: v for k, v in expected.items() if k != SELF_HASH_FIELD},
    )
    if difference is not None:
        _fail(
            name,
            f"body differs from the deterministic rebuild at {difference.lstrip('.')}; "
            "mirrored V1 policy values are frozen",
        )

    if payload.get(SELF_HASH_FIELD) != process_v2_chain_self_hash(payload):
        _fail(
            name,
            f"{SELF_HASH_FIELD} is {payload.get(SELF_HASH_FIELD)!r}, which disagrees "
            f"with the canonical hash of its own body "
            f"{process_v2_chain_self_hash(payload)!r}",
        )
    return payload


def load_process_v2_chain_artifact(name: str, *, repo_root: Path) -> dict[str, Any]:
    """Read one committed chain artifact from disk and validate it.

    Raises:
        ProcessV2ChainError: if it is absent, unparseable, or inconsistent.
    """

    _require_known(name)
    repo_root = Path(repo_root)
    try:
        raw = (repo_root / name).read_bytes()
    except OSError as error:
        raise ProcessV2ChainError(f"{name}: chain artifact is not readable") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProcessV2ChainError(f"{name}: chain artifact is not valid JSON") from error
    return validate_process_v2_chain_artifact(payload, name=name, repo_root=repo_root)


# ---- Whole-chain build ----


def write_process_v2_chain(repo_root: Path) -> dict[str, str]:
    """Build and write all seven artifacts parents-first; return ``name -> self-hash``.

    Each child is built only after its parent has been written, so every pin
    addresses the parent's FINAL bytes.  Rebuilding over an already-written chain
    reproduces identical bytes.
    """

    repo_root = Path(repo_root)
    sealed: dict[str, str] = {}
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = build_process_v2_chain_artifact(name, repo_root=repo_root)
        path = repo_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(serialize_process_v2_chain_artifact(payload))
        sealed[name] = str(payload[SELF_HASH_FIELD])
    return sealed
