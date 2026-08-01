"""Fail-closed semantic Editing-V2 P50 preimplementation boundary.

The semantic Active8 source identity already exists and can be cross-linked to
Gate0.  A P50-authorizing boundary must additionally reopen the exact source
index and use a semantic trainer runtime.  The semantic P50 recipe, thresholds,
full-corpus successor cache, sampler, and planned stream do not yet exist as a
prospectively frozen artifact.  This module validates only the identity receipt
and the explicit nonauthorizing preimplementation contract.  It cannot
authorize P50 and contains no guessed recipe values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compose_v4.data.editing_v2_semantic_active8_decision_source import (
    INDEX_SCHEMA,
    INDEX_SCHEMA_VERSION,
    INDEX_STATUS,
)
from compose_v4.experiments.editing_v2_semantic_p50_source_inventory import (
    SEMANTIC_P50_SOURCE_INVENTORY_FILENAME,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
)

CONTRACT_RELATIVE_PATH = "configs/editing_v2_semantic_p50_preimplementation_v1.json"
CONTRACT_SCHEMA = "compose.editing_v2.semantic_p50_preimplementation_contract"
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_STATUS = "PREIMPLEMENTATION_RECIPE_AND_STREAM_NOT_FROZEN_NO_AUTHORITY"
CONTRACT_SHA256 = "f3ec91d1995061c0057e10ca6cc3d11470c6538e494bbe87a3e1c505d0243340"
DISABLED_FAMILIES = ("ring_system_delete", "ring_system_grow")

_NO_AUTHORITY = {
    "training_authorized": False,
    "bounded_p50_authorized": False,
    "p500_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_SOURCE_NO_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_SOURCE_REQUIRED_SHA_FIELDS = {
    "migration_completion_file_sha256",
    "migration_completion_sha256",
    "chunk_cache_plan_file_sha256",
    "chunk_cache_plan_sha256",
    "chunk_cache_global_pointer_file_sha256",
    "chunk_cache_global_pointer_sha256",
    "chunk_cache_global_completion_file_sha256",
    "chunk_cache_global_completion_sha256",
    "decision_plan_file_sha256",
    "decision_plan_sha256",
    "decision_run_identity_sha256",
    "decision_completion_file_sha256",
    "decision_completion_sha256",
    "decision_result_inventory_sha256",
    "process_identity_sha256",
    "policy_sha256",
    "model_runtime_identity_sha256",
    "source_identity_sha256",
    "accepted_trace_inventory_sha256",
    "accepted_progress_inventory_sha256",
    "excluded_trace_inventory_sha256",
    "trace_decision_inventory_sha256",
    "decision_lookup_inventory_sha256",
    "exclusion_lookup_inventory_sha256",
    "decision_source_implementation_sha256",
    "inventory_sha256",
}
_REQUIRED_LINEAGE = (
    "semantic_migration_completion",
    "semantic_chunk_cache_plan_and_global_completion",
    "semantic_active8_decision_plan_and_completion",
    "semantic_active8_decision_inventory",
    "semantic_gate_zero_evidence",
    "semantic_t1_capacity_decision",
)
_UNFROZEN_RECIPE_FIELDS = {
    "initialization_regime",
    "optimizer_and_schedule",
    "semantic_source_reopen_contract",
    "semantic_trainer_runtime_contract",
    "semantic_lane_and_cell_sampling_weights",
    "effective_teacher_coefficients",
    "path_position_coefficients",
    "exact_stream_union_successor_cache_and_batch_contract",
    "minimum_gradient_updates_per_required_family",
    "minimum_gradient_updates_per_required_semantic_cell",
    "maximum_required_family_successor_nll_regression",
    "maximum_required_semantic_cell_successor_nll_regression",
    "initialization_specific_retention_threshold",
}
_UNFROZEN_STREAM_FIELDS = {
    "address_stream_derivation_contract_sha256",
    "ordered_address_stream_sha256",
    "ordered_training_stream_sha256",
}
_REQUIRED_RECIPE_BINDINGS = (
    "editing_training_gate_contract_file_sha256",
    "semantic_source_inventory_file_sha256",
    "semantic_source_inventory_sha256",
    "semantic_source_process_identity_sha256",
    "semantic_source_model_runtime_identity_sha256",
    "semantic_source_active8_policy_sha256",
    "semantic_source_operator_capability_fingerprint",
    "semantic_source_decision_source_implementation_sha256",
    "semantic_gate_zero_evidence_file_sha256",
    "semantic_gate_zero_evidence_sha256",
    "semantic_t1_decision_file_sha256",
    "semantic_t1_decision_sha256",
    "semantic_t1_selected_model_state_sha256",
    "p50_stream_union_successor_cache_file_sha256",
    "p50_stream_union_successor_cache_sha256",
    "semantic_trainer_runtime_implementation_sha256",
    "sampling_and_exposure_contract_sha256",
    "execution_environment_contract_sha256",
    "exact_launch_projection_sha256",
    "exact_planned_address_and_training_stream_sha256",
)
_PROPOSAL_STATUS = "PROPOSED_STAGE_A_CAPABILITY_P50_PENDING_STREAM_AND_RUNTIME_NO_AUTHORITY"
_PROPOSED_RECIPE_VALUES = {
    "scientific_scope": "stage_a_capability_pilot_not_production_law_calibration",
    "initialization_regime": "scratch",
    "optimizer_steps": 50,
    "batch_size": 64,
    "scheduled_nonterminal_examples": 3200,
    "resume": False,
    "optimizer": {
        "kind": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler_family": "constant",
        "gradient_clip_norm": 10.0,
    },
    "seed": 31,
    "numerics": {
        "dtype": "float32",
        "mixed_precision": False,
        "deterministic_algorithms_required": True,
    },
    "objective": {
        "name": "balanced_semantic_cell_productive_identity",
        "unit": "productive_embedded_canonical_successor",
        "hazard_included": False,
        "hazard_weight": 0.0,
        "terminal_rows": "excluded",
        "importance_correction": "none",
        "path_position_coefficient": 1.0,
    },
    "sampling": {
        "cell_order": "equal_round_robin_over_declared_nonempty_semantic_cells",
        "within_cell": "deterministic_uniform_cycle",
        "ordered_address_count": 3200,
    },
    "cache_coverage_mode": "complete_trace_closure_of_planned_address_union",
    "source_population": "full_admitted_semantic_active8_corpus",
    "threshold_proposals": {
        "maximum_family_final_minus_baseline_successor_nll_nats": 0.25,
        "family_gradient_floor_rule": ("max_1_ceil_0.8_times_planned_optimizer_step_opportunities"),
        "zero_planned_family_opportunities_allowed": False,
        "scratch_retention_threshold": None,
    },
}
_KNOWN_PROPOSAL_BLOCKERS = (
    "semantic_gate_zero_pass_and_semantic_t1_go_not_yet_materialized",
    "legacy_loader_and_recipe_verifier_do_not_support_stage_a_semantic_stream",
    "exact_row_time_derivation_and_full_training_stream_not_frozen",
    "exact_scratch_model_runtime_and_initial_state_not_frozen",
    "constant_schedule_projection_and_execution_determinism_not_fully_frozen",
    "lane_and_source_group_policy_not_frozen",
    "validation_baseline_and_per_semantic_cell_nll_ceiling_not_frozen",
    "semantic_cell_gradient_attribution_not_implemented",
    "complete_trace_closure_cache_and_exact_coverage_receipt_not_materialized",
    "new_authorizing_semantic_p50_schema_and_runtime_adapter_not_implemented",
)


class SemanticP50PreimplementationError(ValueError):
    """The semantic P50 boundary is malformed or remains nonauthorizing."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticP50PreimplementationError(
            "semantic P50 artifact is not finite canonical JSON"
        ) from error
    return raw + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha(value: object, *, field: str) -> str:
    if not _is_sha(value):
        raise SemanticP50PreimplementationError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _load_canonical(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if not source.is_file():
        raise SemanticP50PreimplementationError(f"{field} is absent: {source}")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticP50PreimplementationError(
            f"{field} is not readable JSON: {source}"
        ) from error
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload, newline=True):
        raise SemanticP50PreimplementationError(
            f"{field} must be canonical newline-terminated JSON"
        )
    return payload, raw


def validate_semantic_p50_source_inventory(
    value: object,
    *,
    source_path: Path,
    gate_zero_evidence: Mapping[str, Any],
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Cross-link a nonauthorizing source identity receipt with Gate0.

    This check binds every upstream semantic hash carried by the identity.  It
    deliberately does not reopen the upstream artifact graph.  A future
    authorizing semantic-P50 verifier must do that before constructing data.
    """

    physical, raw = _load_canonical(source_path, field="semantic P50 source inventory")
    if Path(source_path).name != SEMANTIC_P50_SOURCE_INVENTORY_FILENAME:
        raise SemanticP50PreimplementationError(
            "semantic P50 source inventory has another filename"
        )
    if value != physical:
        raise SemanticP50PreimplementationError(
            "semantic P50 source payload differs from its physical artifact"
        )
    observed_file_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_file_sha256 != _require_sha(
        expected_file_sha256, field="expected semantic source file SHA-256"
    ):
        raise SemanticP50PreimplementationError("semantic P50 source physical hash disagrees")
    expected_identity = gate_zero_evidence.get("decision_source_identity")
    if not isinstance(expected_identity, Mapping) or physical != dict(expected_identity):
        raise SemanticP50PreimplementationError(
            "semantic P50 source differs from Gate0 decision-source identity"
        )
    body = dict(physical)
    supplied_inventory_sha256 = body.pop("inventory_sha256", None)
    if (
        physical.get("schema") != INDEX_SCHEMA
        or physical.get("schema_version") != INDEX_SCHEMA_VERSION
        or physical.get("status") != INDEX_STATUS
        or physical.get("encoding") != "canonical_json_line_stream_v1"
        or supplied_inventory_sha256 != _sha(body)
        or supplied_inventory_sha256 != gate_zero_evidence.get("decision_source_inventory_sha256")
        or any(
            physical.get(name) is not expected for name, expected in _SOURCE_NO_AUTHORITY.items()
        )
    ):
        raise SemanticP50PreimplementationError(
            "semantic P50 source identity, authority, or self-hash disagrees"
        )
    for name in _SOURCE_REQUIRED_SHA_FIELDS:
        _require_sha(physical.get(name), field=f"semantic source.{name}")
    return physical


def validate_semantic_p50_preimplementation_contract(
    value: object,
) -> dict[str, Any]:
    """Validate the exact contract that records why P50 remains blocked."""

    fields = {
        "schema",
        "schema_version",
        "contract_id",
        "status",
        *_NO_AUTHORITY,
        "purpose",
        "proposal_status",
        "proposed_recipe_values",
        "known_proposal_blockers",
        "source_contract",
        "fixed_p50_invariants",
        "unfrozen_recipe_values",
        "unfrozen_stream_values",
        "required_frozen_recipe_bindings",
        "required_next_action",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SemanticP50PreimplementationError(
            "semantic P50 preimplementation contract has missing or unknown fields"
        )
    contract = dict(value)
    body = dict(contract)
    supplied_sha256 = body.pop("contract_sha256")
    if (
        contract["schema"] != CONTRACT_SCHEMA
        or contract["schema_version"] != CONTRACT_SCHEMA_VERSION
        or contract["contract_id"] != "compose-editing-v2-semantic-p50-preimplementation-v1"
        or contract["status"] != CONTRACT_STATUS
        or supplied_sha256 != CONTRACT_SHA256
        or supplied_sha256 != _sha(body)
        or any(contract[name] is not expected for name, expected in _NO_AUTHORITY.items())
        or contract["purpose"]
        != "fail_closed_boundary_until_one_semantic_v2_p50_recipe_and_exact_stream_are_prospectively_frozen"
        or contract["proposal_status"] != _PROPOSAL_STATUS
        or contract["proposed_recipe_values"] != _PROPOSED_RECIPE_VALUES
        or tuple(contract["known_proposal_blockers"]) != _KNOWN_PROPOSAL_BLOCKERS
        or contract["required_next_action"]
        != "prospectively_freeze_nonnull_p50_thresholds_recipe_sampler_and_exact_stream_then_implement_a_new_authorizing_schema_version"
    ):
        raise SemanticP50PreimplementationError(
            "semantic P50 preimplementation identity or authority disagrees"
        )
    source = contract["source_contract"]
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "schema",
            "schema_version",
            "status",
            "must_equal_gate_zero_decision_source_identity",
            "full_physical_reresolution_required_for_authority",
            "required_lineage",
        }
        or source["schema"] != INDEX_SCHEMA
        or source["schema_version"] != INDEX_SCHEMA_VERSION
        or source["status"] != INDEX_STATUS
        or source["must_equal_gate_zero_decision_source_identity"] is not True
        or source["full_physical_reresolution_required_for_authority"] is not True
        or tuple(source["required_lineage"]) != _REQUIRED_LINEAGE
    ):
        raise SemanticP50PreimplementationError(
            "semantic P50 source-contract requirements disagree"
        )
    fixed = contract["fixed_p50_invariants"]
    if (
        not isinstance(fixed, Mapping)
        or set(fixed)
        != {
            "optimizer_steps",
            "resume",
            "objective_unit",
            "hazard_included",
            "required_families",
            "disabled_families",
        }
        or fixed["optimizer_steps"] != 50
        or fixed["resume"] is not False
        or fixed["objective_unit"] != "productive_embedded_canonical_successor"
        or fixed["hazard_included"] is not False
        or tuple(fixed["required_families"]) != RINGCORE_EDITING_FAMILIES
        or tuple(fixed["disabled_families"]) != DISABLED_FAMILIES
    ):
        raise SemanticP50PreimplementationError(
            "semantic P50 fixed invariants disagree with the bounded pilot"
        )
    unresolved = contract["unfrozen_recipe_values"]
    streams = contract["unfrozen_stream_values"]
    if (
        not isinstance(unresolved, Mapping)
        or set(unresolved) != _UNFROZEN_RECIPE_FIELDS
        or any(item is not None for item in unresolved.values())
        or not isinstance(streams, Mapping)
        or set(streams) != _UNFROZEN_STREAM_FIELDS
        or any(item is not None for item in streams.values())
        or tuple(contract["required_frozen_recipe_bindings"]) != _REQUIRED_RECIPE_BINDINGS
    ):
        raise SemanticP50PreimplementationError(
            "semantic P50 unresolved recipe or stream inventory disagrees"
        )
    return contract


def load_semantic_p50_preimplementation_contract(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    return validate_semantic_p50_preimplementation_contract(payload)


def assert_semantic_p50_recipe_frozen(value: object) -> None:
    """Always refuse P50 after validating the exact recorded missing decisions."""

    contract = validate_semantic_p50_preimplementation_contract(value)
    unresolved = sorted(
        [
            *contract["unfrozen_recipe_values"],
            *contract["unfrozen_stream_values"],
        ]
    )
    raise SemanticP50PreimplementationError(
        "semantic P50 recipe is not frozen; unresolved values: " + ", ".join(unresolved)
    )


__all__ = [
    "CONTRACT_RELATIVE_PATH",
    "CONTRACT_SCHEMA",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_SHA256",
    "CONTRACT_STATUS",
    "SemanticP50PreimplementationError",
    "assert_semantic_p50_recipe_frozen",
    "load_semantic_p50_preimplementation_contract",
    "validate_semantic_p50_preimplementation_contract",
    "validate_semantic_p50_source_inventory",
]
