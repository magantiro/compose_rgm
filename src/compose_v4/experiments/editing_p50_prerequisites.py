"""Semantic prerequisite contracts for the bounded COMPOSE P50 pilot.

Physical SHA-256 equality is necessary but not sufficient for launch
authorization.  This module validates the meaning of the frozen Gate-0
evidence, the explicit T1 decision, and the exact P50 recipe, then proves that
all three name the same Active8 corpus and operator contract.

The artifacts authorize only the exactly-50-step development pilot.  They
never authorize P500, P2000, or full scientific training.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from compose_v4.data.active8_trace_inventory import Active8TraceAdmission
from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
from compose_v4.experiments.editing_gate_zero_runtime import (
    EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA,
    EDITING_GATE_ZERO_RUNTIME_EVIDENCE_VERSION,
    EDITING_GATE_ZERO_RUNTIME_STATUS,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
)

T1_P50_DECISION_SCHEMA = "compose.editing.t1_successor_gate_decision"
T1_P50_DECISION_SCHEMA_VERSION = 1
T1_P50_DECISION_PASS_STATUS = "PASS_BOUNDED_P50_PREREQUISITE"
P50_RECIPE_SCHEMA = "compose.editing.p50_launch_recipe"
P50_RECIPE_SCHEMA_VERSION = 1
P50_RECIPE_STATUS = "FROZEN_BOUNDED_P50_RECIPE"
P50_DISABLED_FAMILIES = ("ring_system_delete", "ring_system_grow")

P50_RECIPE_LAUNCH_FIELDS = (
    "optimizer_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "schedule_steps",
    "minimum_learning_rate_fraction",
    "seed",
    "initialization_regime",
    "factorized_training_objective",
    "successor_objective_mode",
    "successor_hazard_weight",
    "hidden_dim",
    "message_passing_steps",
    "mark_dim",
    "rate_factorization",
    "empirical_mark_prior_mode",
    "empirical_mark_prior_smoothing",
    "ring_family_mass_mode",
    "ring_template_factorization",
    "trainable_parameter_scope",
    "late_time_fraction",
    "operational_horizon",
    "progress_stratification_fraction",
    "bond_representation",
    "ring_electronic_mode",
    "teacher_ordering",
    "condition_dropout_probability",
    "property_conditions",
    "use_bf16",
    "denovo_weight",
    "corrupted_prior_mix",
    "cycle_op_mix",
    "disable_ring_grow_macro",
    "organic_vocabulary",
    "evaluation_every",
    "evaluation_batch_size",
    "validation_examples",
    "validation_size",
    "max_atoms",
    "required_families",
    "disabled_families",
    "resume",
    "early_stopping_patience",
    "benchmark_steps",
    "snapshot_checkpoints",
    "source_corpus_inventory_sha256",
    "successor_cache_inventory_sha256",
    "successor_cache_compatibility_sha256",
    "semantic_sidecar_manifest_sha256",
    "unified_packed_manifest_sha256",
    "representability_overlay_sha256",
)

_GATE_ZERO_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "quality_thresholds",
    "training_decision",
    "optimizer_steps",
    "runtime_contract_sha256",
    "source",
    "model",
    "cache_contract",
    "probe",
    "cache_artifacts",
    "support_audit",
    "evidence_sha256",
}
_GATE_ZERO_SOURCE_FIELDS = {
    "semantic_sidecar_manifest_sha256",
    "semantic_sidecar_source_sha256",
    "semantic_census_sha256",
    "unified_packed_manifest_sha256",
    "representability_overlay_sha256",
    "validation_trace_count",
    "validation_progress_row_count",
    "validation_shards",
}
_GATE_ZERO_CACHE_FIELDS = {
    "support_signature",
    "support_signature_sha256",
    "operator_registry_hash",
    "operator_registry_source_sha256",
    "canonicalizer_contract_sha256",
    "tensorization_implementation_hash",
    "fiber_compiler_implementation_hash",
}
_T1_DECISION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "bounded_p50_authorized",
    "full_training_authorized",
    "thresholds_frozen",
    "prerequisites",
    "required_families",
    "disabled_families",
    "arm_results_manifest_sha256",
    "decision_sha256",
}
_PARENT_IDENTITY_FIELDS = {
    "source_corpus_inventory_file_sha256",
    "source_corpus_inventory_sha256",
    "gate_zero_structural_evidence_file_sha256",
    "unified_packed_manifest_sha256",
    "support_contract_sha256",
}
_RECIPE_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "bounded_p50_authorized",
    "full_training_authorized",
    "prerequisites",
    "launch",
    "launch_sha256",
    "planned_stream",
    "recipe_sha256",
}
_PLANNED_STREAM_FIELDS = {
    "ordered_address_stream_sha256",
    "ordered_training_stream_sha256",
}
_RECIPE_PARENT_IDENTITY_FIELDS = {
    *_PARENT_IDENTITY_FIELDS,
    "t1_successor_gate_decision_file_sha256",
}


class EditingP50PrerequisiteError(ValueError):
    """A bounded-pilot prerequisite is malformed or refers to another run."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingP50PrerequisiteError(
            "P50 prerequisite is not finite canonical JSON"
        ) from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise EditingP50PrerequisiteError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return str(value)


def _require_exact_mapping(
    value: object,
    expected: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingP50PrerequisiteError(f"{field} must be an object")
    observed = set(value)
    if observed != expected:
        raise EditingP50PrerequisiteError(
            f"{field} has missing or unknown fields: "
            f"missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _require_positive_family_census(
    value: object,
    *,
    field: str,
) -> None:
    if not isinstance(value, Mapping):
        raise EditingP50PrerequisiteError(f"{field} must be an object")
    for family in RINGCORE_EDITING_FAMILIES:
        count = value.get(family)
        if type(count) is not int or count <= 0:
            raise EditingP50PrerequisiteError(
                f"{field} lacks a positive {family} witness"
            )


def validate_gate_zero_structural_evidence(
    payload: object,
    *,
    source_admission: Active8TraceAdmission,
) -> Mapping[str, Any]:
    """Validate structural evidence and bind it to the Active8 source."""

    evidence = _require_exact_mapping(
        payload,
        _GATE_ZERO_FIELDS,
        field="Gate0 structural evidence",
    )
    body = {
        key: value
        for key, value in evidence.items()
        if key != "evidence_sha256"
    }
    if (
        evidence["schema"] != EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA
        or evidence["schema_version"]
        != EDITING_GATE_ZERO_RUNTIME_EVIDENCE_VERSION
        or evidence["status"] != EDITING_GATE_ZERO_RUNTIME_STATUS
        or evidence["training_authorized"] is not False
        or evidence["quality_thresholds"] is not None
        or evidence["training_decision"] is not None
        or evidence["optimizer_steps"] != 0
        or evidence["evidence_sha256"] != _stable_sha256(body)
    ):
        raise EditingP50PrerequisiteError(
            "Gate0 structural evidence status or self-hash is invalid"
        )
    runtime_contract_sha256 = _require_sha256(
        evidence["runtime_contract_sha256"],
        field="Gate0 runtime_contract_sha256",
    )
    if runtime_contract_sha256 != source_admission.support_contract_sha256:
        raise EditingP50PrerequisiteError(
            "Gate0 evidence and Active8 inventory name different support contracts"
        )

    source = _require_exact_mapping(
        evidence["source"],
        _GATE_ZERO_SOURCE_FIELDS,
        field="Gate0 source",
    )
    if (
        source["unified_packed_manifest_sha256"]
        != source_admission.unified_packed_manifest_sha256
    ):
        raise EditingP50PrerequisiteError(
            "Gate0 evidence and Active8 inventory name different unified corpora"
        )
    for field in (
        "semantic_sidecar_manifest_sha256",
        "semantic_sidecar_source_sha256",
        "semantic_census_sha256",
        "unified_packed_manifest_sha256",
        "representability_overlay_sha256",
    ):
        _require_sha256(source[field], field=f"Gate0 source.{field}")
    for field in ("validation_trace_count", "validation_progress_row_count"):
        if type(source[field]) is not int or source[field] <= 0:
            raise EditingP50PrerequisiteError(
                f"Gate0 source.{field} must be positive"
            )
    if not isinstance(source["validation_shards"], list) or not source[
        "validation_shards"
    ]:
        raise EditingP50PrerequisiteError(
            "Gate0 source must name at least one validation shard"
        )

    cache_contract = _require_exact_mapping(
        evidence["cache_contract"],
        _GATE_ZERO_CACHE_FIELDS,
        field="Gate0 cache contract",
    )
    support_signature = cache_contract["support_signature"]
    if not isinstance(support_signature, Mapping):
        raise EditingP50PrerequisiteError(
            "Gate0 support signature must be an object"
        )
    if (
        cache_contract["support_signature_sha256"]
        != _stable_sha256(support_signature)
    ):
        raise EditingP50PrerequisiteError(
            "Gate0 support-signature self-hash is invalid"
        )
    if (
        support_signature.get("charge_policy") != CHARGE_POLICY_VERSION
        or support_signature.get("enable_ring_restates") is not True
        or support_signature.get("enable_cycle_ops") is not True
        or support_signature.get("enable_ring_opening") is not True
        or support_signature.get("enable_ring_grow_macro") is not False
        or support_signature.get("enable_ring_system_delete") is not False
        or support_signature.get("max_atoms") != 40
        or support_signature.get("atom_insert_arity_support") != [0, 1]
        or support_signature.get("embedded_jump_chain_policy")
        != "productive_canonical_successors"
    ):
        raise EditingP50PrerequisiteError(
            "Gate0 support signature is not the bounded Active8 editing object"
        )
    ordered_families = support_signature.get("ordered_family_vocabulary")
    if not isinstance(ordered_families, list) or not set(
        RINGCORE_EDITING_FAMILIES
    ).issubset(ordered_families):
        raise EditingP50PrerequisiteError(
            "Gate0 support signature lacks the required Active8 family vocabulary"
        )

    support_audit = evidence["support_audit"]
    if not isinstance(support_audit, Mapping):
        raise EditingP50PrerequisiteError("Gate0 support audit must be an object")
    for field in (
        "teacher_examples_by_family",
        "candidate_marks_by_family",
        "productive_successors_by_family",
    ):
        _require_positive_family_census(
            support_audit.get(field),
            field=f"Gate0 support_audit.{field}",
        )
    if not isinstance(evidence["cache_artifacts"], list) or not evidence[
        "cache_artifacts"
    ]:
        raise EditingP50PrerequisiteError(
            "Gate0 evidence must contain immutable cache receipts"
        )
    return evidence


def validate_t1_p50_decision(
    payload: object,
    *,
    source_admission: Active8TraceAdmission,
    source_inventory_file_sha256: str,
    gate_zero_evidence_file_sha256: str,
) -> Mapping[str, Any]:
    """Require an explicit passing T1 decision, never a decision-free arm."""

    decision = _require_exact_mapping(
        payload,
        _T1_DECISION_FIELDS,
        field="T1 P50 decision",
    )
    body = {
        key: value
        for key, value in decision.items()
        if key != "decision_sha256"
    }
    if (
        decision["schema"] != T1_P50_DECISION_SCHEMA
        or decision["schema_version"] != T1_P50_DECISION_SCHEMA_VERSION
        or decision["status"] != T1_P50_DECISION_PASS_STATUS
        or decision["bounded_p50_authorized"] is not True
        or decision["full_training_authorized"] is not False
        or decision["thresholds_frozen"] is not True
        or decision["decision_sha256"] != _stable_sha256(body)
        or tuple(decision["required_families"]) != RINGCORE_EDITING_FAMILIES
        or tuple(decision["disabled_families"]) != P50_DISABLED_FAMILIES
    ):
        raise EditingP50PrerequisiteError(
            "T1 decision is not an exact passing bounded-P50 decision"
        )
    parents = _require_exact_mapping(
        decision["prerequisites"],
        _PARENT_IDENTITY_FIELDS,
        field="T1 decision prerequisites",
    )
    expected = {
        "source_corpus_inventory_file_sha256": source_inventory_file_sha256,
        "source_corpus_inventory_sha256": source_admission.inventory_sha256,
        "gate_zero_structural_evidence_file_sha256": (
            gate_zero_evidence_file_sha256
        ),
        "unified_packed_manifest_sha256": (
            source_admission.unified_packed_manifest_sha256
        ),
        "support_contract_sha256": source_admission.support_contract_sha256,
    }
    if dict(parents) != expected:
        raise EditingP50PrerequisiteError(
            "T1 decision parent identities disagree with Active8 or Gate0"
        )
    _require_sha256(
        decision["arm_results_manifest_sha256"],
        field="T1 arm_results_manifest_sha256",
    )
    return decision


def validate_p50_recipe(
    payload: object,
    *,
    source_admission: Active8TraceAdmission,
    source_inventory_file_sha256: str,
    gate_zero_evidence_file_sha256: str,
    t1_decision_file_sha256: str,
    expected_launch: Mapping[str, object],
) -> Mapping[str, Any]:
    """Validate the frozen recipe and its exact live launcher projection."""

    recipe = _require_exact_mapping(
        payload,
        _RECIPE_FIELDS,
        field="P50 recipe",
    )
    body = {
        key: value
        for key, value in recipe.items()
        if key != "recipe_sha256"
    }
    if (
        recipe["schema"] != P50_RECIPE_SCHEMA
        or recipe["schema_version"] != P50_RECIPE_SCHEMA_VERSION
        or recipe["status"] != P50_RECIPE_STATUS
        or recipe["bounded_p50_authorized"] is not True
        or recipe["full_training_authorized"] is not False
        or recipe["recipe_sha256"] != _stable_sha256(body)
    ):
        raise EditingP50PrerequisiteError(
            "P50 recipe status or self-hash is invalid"
        )
    parents = _require_exact_mapping(
        recipe["prerequisites"],
        _RECIPE_PARENT_IDENTITY_FIELDS,
        field="P50 recipe prerequisites",
    )
    expected_parents = {
        "source_corpus_inventory_file_sha256": source_inventory_file_sha256,
        "source_corpus_inventory_sha256": source_admission.inventory_sha256,
        "gate_zero_structural_evidence_file_sha256": (
            gate_zero_evidence_file_sha256
        ),
        "unified_packed_manifest_sha256": (
            source_admission.unified_packed_manifest_sha256
        ),
        "support_contract_sha256": source_admission.support_contract_sha256,
        "t1_successor_gate_decision_file_sha256": t1_decision_file_sha256,
    }
    if dict(parents) != expected_parents:
        raise EditingP50PrerequisiteError(
            "P50 recipe parent identities disagree with the frozen prerequisites"
        )
    launch = _require_exact_mapping(
        recipe["launch"],
        set(P50_RECIPE_LAUNCH_FIELDS),
        field="P50 recipe launch",
    )
    expected_launch_mapping = _require_exact_mapping(
        expected_launch,
        set(P50_RECIPE_LAUNCH_FIELDS),
        field="live P50 launch projection",
    )
    if recipe["launch_sha256"] != _stable_sha256(launch):
        raise EditingP50PrerequisiteError("P50 recipe launch self-hash is invalid")
    if (
        launch["optimizer_steps"] != 50
        or launch["resume"] is not False
        or launch["max_atoms"] != 40
        or launch["hidden_dim"] != 256
        or launch["message_passing_steps"] != 6
        or launch["mark_dim"] != 32
        or launch["rate_factorization"] != "hierarchical"
        or launch["empirical_mark_prior_mode"] != "none"
        or launch["ring_family_mass_mode"] != "boolean"
        or launch["ring_template_factorization"] != "flat"
        or tuple(launch["required_families"]) != RINGCORE_EDITING_FAMILIES
        or tuple(launch["disabled_families"]) != P50_DISABLED_FAMILIES
        or launch["factorized_training_objective"] != "canonical_successor"
    ):
        raise EditingP50PrerequisiteError(
            "P50 recipe is not the exact non-resumable Active8 successor pilot"
        )
    if dict(launch) != dict(expected_launch_mapping):
        raise EditingP50PrerequisiteError(
            "live P50 launcher arguments differ from the frozen recipe"
        )
    planned_stream = _require_exact_mapping(
        recipe["planned_stream"],
        _PLANNED_STREAM_FIELDS,
        field="P50 recipe planned stream",
    )
    for field in _PLANNED_STREAM_FIELDS:
        _require_sha256(
            planned_stream[field],
            field=f"P50 recipe planned_stream.{field}",
        )
    return recipe


@dataclass(frozen=True)
class VerifiedP50Prerequisites:
    """Cross-linked semantic identities safe to record in P50 metadata."""

    physical_sha256: Mapping[str, str]
    source_inventory_sha256: str
    unified_packed_manifest_sha256: str
    support_contract_sha256: str
    gate_zero_evidence_sha256: str
    t1_decision_sha256: str
    recipe_sha256: str
    launch_sha256: str
    ordered_address_stream_sha256: str
    ordered_training_stream_sha256: str

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "physical_sha256": dict(self.physical_sha256),
            "source_inventory_sha256": self.source_inventory_sha256,
            "unified_packed_manifest_sha256": (
                self.unified_packed_manifest_sha256
            ),
            "support_contract_sha256": self.support_contract_sha256,
            "gate_zero_evidence_sha256": self.gate_zero_evidence_sha256,
            "t1_decision_sha256": self.t1_decision_sha256,
            "recipe_sha256": self.recipe_sha256,
            "launch_sha256": self.launch_sha256,
            "ordered_address_stream_sha256": (
                self.ordered_address_stream_sha256
            ),
            "ordered_training_stream_sha256": (
                self.ordered_training_stream_sha256
            ),
            "bounded_p50_authorized": True,
            "full_training_authorized": False,
        }


__all__ = [
    "P50_DISABLED_FAMILIES",
    "P50_RECIPE_LAUNCH_FIELDS",
    "P50_RECIPE_SCHEMA",
    "P50_RECIPE_SCHEMA_VERSION",
    "P50_RECIPE_STATUS",
    "T1_P50_DECISION_PASS_STATUS",
    "T1_P50_DECISION_SCHEMA",
    "T1_P50_DECISION_SCHEMA_VERSION",
    "EditingP50PrerequisiteError",
    "VerifiedP50Prerequisites",
    "validate_gate_zero_structural_evidence",
    "validate_p50_recipe",
    "validate_t1_p50_decision",
]
