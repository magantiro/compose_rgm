"""Fail-closed semantic capability cells for Editing-V2 Active8 teachers.

The registry deliberately balances only ``(model family, family context)``.
Lane, partition role, evidence roles, candidate-fiber size, and successor-alias
multiplicity remain separate audit dimensions.  This prevents an accidental
Cartesian product from turning sparse audit strata into a training law.

This module classifies only nonterminal progress rows from a whole-trace
Active8 admission decision.  It cannot authorize Gate 0, T1, P50, or training.

The pure action classifier and its complete helper closure now live in
:mod:`compose_v4.data.editing_v2_semantic_capability_cell_classifier`, a LEAF that
imports only chemistry and the Action-V4 codec.  A consumer that needs to name a
transition imports the leaf and acquires nothing else; this module imports the same
objects back so every name it published before the move is still published here,
and is the SAME object.  Everything that remains here is the part that must read
the frozen registry, the corpus contract, the admission decision, and the packed
store -- the dependency cone the leaf exists to keep out of a classifying consumer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_corpus_contract import (
    ACTIVE8_FAMILIES,
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
    EditingCorpusContractError,
    load_editing_corpus_contract,
    validate_evidence_assignment_envelope,
    validate_partition_resolution,
    validate_record_membership_assignment,
)
from compose_v4.data.editing_semantic_sampling_sidecar import (
    EndpointDescriptor,
    MappingDescriptor,
    PathDescriptor,
    SamplingCoefficients,
    SemanticGroups,
    SemanticSamplingProgress,
)
from compose_v4.data.editing_v2_semantic_active8_admission import (
    SemanticActive8TraceDecision,
    SemanticExactCandidateEvidence,
    build_semantic_active8_admission_policy,
    classify_semantic_action,
)
from compose_v4.data.editing_v2_semantic_active8_decision_source import (
    EditingV2SemanticActive8DecisionIndex,
    SemanticActive8AcceptedTransition,
    SemanticActive8DecisionSourceError,
)
from compose_v4.data.editing_v2_semantic_capability_cell_classifier import (
    CountBin,
    SemanticCapabilityCellError,
    _action_audit_axes,
    _component_after_cut,  # noqa: F401 -- re-exported, see the note below
    _cycle_rank,
    _KNOWN_CONTEXTS,
    _minimum_cycle_length_for_edge,  # noqa: F401 -- re-exported, see the note below
    _real_atom_count,
    _ring_system_topology,  # noqa: F401 -- re-exported, see the note below
    _state_graph,  # noqa: F401 -- re-exported, see the note below
    _stratum,
    classify_action_family_context,
)
from compose_v4.data.packed_trace_store import AddressedPackedTrace
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_v2_process_identity,
)
from compose_v4.rewrite.kernel import canonical_state_key

# The classifier import above is a RE-EXPORT: this module keeps naming every
# classifier symbol it named before the classifier moved to its leaf, and it names
# the SAME objects, not copies.  ``SemanticCapabilityCellError`` in particular must
# be one class object, because an exception raised through one class is not
# catchable through another.  Four of them have no remaining use INSIDE this module
# and carry a ``noqa`` for that reason; dropping them would silently break an
# importer that still names them here.
#
# The move also took this module's whole chemistry-primitive import block with it.
# Nothing left here reads bond classes, element indices, or operator payloads: what
# remains loads and validates the frozen registry and assembles assignments, and it
# reaches the exact-state layer only through the leaf.

REGISTRY_SCHEMA = "compose.editing_v2.semantic_capability_cell_registry"
REGISTRY_SCHEMA_VERSION = 1
REGISTRY_STATUS = "DESIGN_NOT_TRAINING_AUTHORIZED"
REGISTRY_RELATIVE_PATH = "configs/editing_v2_semantic_capability_cells_v1.json"
ASSIGNMENT_SCHEMA = "compose.editing_v2.semantic_capability_cell_assignment"
ASSIGNMENT_SCHEMA_VERSION = 1
ASSIGNMENT_STATUS = "CLASSIFIED_NO_GATE_OR_TRAINING_AUTHORITY"
STRUCTURAL_ASSIGNMENT_SCHEMA = (
    "compose.editing_v2.semantic_structural_capability_assignment"
)
STRUCTURAL_ASSIGNMENT_SCHEMA_VERSION = 1
STRUCTURAL_ASSIGNMENT_STATUS = "VERIFIED_STRUCTURAL_NO_GATE_OR_TRAINING_AUTHORITY"

_TOP_LEVEL_FIELDS = {
    "schema",
    "schema_version",
    "registry_id",
    "status",
    "training_authorized",
    "gate_zero_authorized",
    "t1_authorized",
    "bounded_p50_authorized",
    "scope",
    "bindings",
    "cell_identity_policy",
    "family_contexts",
    "exact_evidence_strata",
    "fail_closed_policy",
    "registry_sha256",
}
_SCOPE_FIELDS = {
    "row_unit",
    "generated_object",
    "objective_lane",
    "terminal_rows",
    "maximum_active_atoms",
    "stereochemistry",
    "formal_charge_changes",
}
_BINDING_FIELDS = {
    "semantic_process_contract",
    "semantic_process_identity_sha256",
    "corpus_contract",
    "corpus_contract_file_sha256",
    "classifier_implementation_file",
    "classifier_implementation_sha256",
    "action_codec_schema_version",
    "active_families",
    "data_lanes",
    "partition_roles",
}
_CELL_POLICY_FIELDS = {
    "namespace",
    "balancing_dimensions",
    "separate_nonbalancing_dimensions",
    "rationale",
}
_REQUIRED_SEPARATE_DIMENSIONS = {
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
}
_STRATA_FIELDS = {
    "policy",
    "raw_mark_count",
    "canonical_successor_count",
    "successor_alias_multiplicity",
}
_STRATUM_FIELDS = {"scope", "bins"}
_BIN_FIELDS = {"id", "minimum", "maximum"}
_FAIL_CLOSED_FIELDS = {
    "unknown_family",
    "unknown_family_context",
    "missing_exact_candidate_evidence",
    "unsupported_teacher",
    "terminal_progress_row",
    "lane_or_partition_mismatch",
    "process_identity_drift",
    "registry_tamper",
}


@dataclass(frozen=True, slots=True)
class SemanticCapabilityCellRegistry:
    """Validated non-authorizing capability-cell registry."""

    registry_id: str
    namespace: str
    family_contexts: tuple[tuple[str, tuple[str, ...]], ...]
    raw_mark_bins: tuple[CountBin, ...]
    canonical_successor_bins: tuple[CountBin, ...]
    successor_alias_bins: tuple[CountBin, ...]
    registry_sha256: str
    process_identity_sha256: str
    corpus_contract_file_sha256: str
    classifier_implementation_sha256: str

    @property
    def contexts_by_family(self) -> dict[str, tuple[str, ...]]:
        return dict(self.family_contexts)


@dataclass(frozen=True, slots=True)
class SemanticCapabilityCellAssignment:
    """One deterministic classification retaining provenance and difficulty."""

    trace_id: str
    progress_index: int
    packed_shard_content_sha256: str
    packed_entry_index: int
    candidate_ledger_row_sha256: str
    admitted_record_envelope_sha256: str
    split_assignment_manifest_sha256: str
    record_membership_classifier_identity_sha256: str
    action_sha256: str
    source_state_sha256: str
    target_state_sha256: str
    source_canonical_key: str
    successor_canonical_key: str
    model_family: str
    family_context: str
    capability_cell_id: str
    data_lane: str
    partition_role: str
    record_membership_cells: tuple[str, ...]
    endpoint_evidence_roles: tuple[tuple[str, str], ...]
    path_evidence_roles: tuple[tuple[str, str], ...]
    evidence_profile_id: str
    semantic_groups: SemanticGroups
    endpoint_descriptor: EndpointDescriptor
    path_descriptor: PathDescriptor
    relationship_group_ids: tuple[str, ...]
    mappings: MappingDescriptor
    sampling_coefficients: SamplingCoefficients
    semantic_progress_evidence_sha256: str
    raw_mark_count: int
    raw_mark_count_stratum: str
    canonical_successor_count: int
    canonical_successor_count_stratum: str
    successor_alias_multiplicity: int
    successor_alias_multiplicity_stratum: str
    matching_mark_count: int
    atom_element_transition: tuple[tuple[str, str], ...]
    minimum_edited_cycle_length: int | None
    source_edge_aromatic: bool | None
    successor_edge_aromatic: bool | None
    registry_sha256: str
    process_identity_sha256: str
    corpus_contract_file_sha256: str
    classifier_implementation_sha256: str
    active8_policy_sha256: str
    assignment_sha256: str

    def as_payload(self) -> dict[str, object]:
        """Return a deterministic machine-readable assignment."""

        return {
            "schema": ASSIGNMENT_SCHEMA,
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "status": ASSIGNMENT_STATUS,
            "training_authorized": False,
            "gate_zero_authorized": False,
            "t1_authorized": False,
            "bounded_p50_authorized": False,
            "trace_id": self.trace_id,
            "progress_index": self.progress_index,
            "packed_shard_content_sha256": self.packed_shard_content_sha256,
            "packed_entry_index": self.packed_entry_index,
            "candidate_ledger_row_sha256": self.candidate_ledger_row_sha256,
            "admitted_record_envelope_sha256": self.admitted_record_envelope_sha256,
            "split_assignment_manifest_sha256": self.split_assignment_manifest_sha256,
            "record_membership_classifier_identity_sha256": (
                self.record_membership_classifier_identity_sha256
            ),
            "action_sha256": self.action_sha256,
            "source_state_sha256": self.source_state_sha256,
            "target_state_sha256": self.target_state_sha256,
            "source_canonical_key": self.source_canonical_key,
            "successor_canonical_key": self.successor_canonical_key,
            "model_family": self.model_family,
            "family_context": self.family_context,
            "capability_cell_id": self.capability_cell_id,
            "data_lane": self.data_lane,
            "partition_role": self.partition_role,
            "record_membership_cells": list(self.record_membership_cells),
            "endpoint_evidence_roles": [
                list(item) for item in self.endpoint_evidence_roles
            ],
            "path_evidence_roles": [list(item) for item in self.path_evidence_roles],
            "evidence_profile_id": self.evidence_profile_id,
            "semantic_groups": _groups_payload(self.semantic_groups),
            "endpoint_descriptor": _endpoint_descriptor_payload(
                self.endpoint_descriptor
            ),
            "path_descriptor": _path_descriptor_payload(self.path_descriptor),
            "relationship_group_ids": list(self.relationship_group_ids),
            "mappings": _mappings_payload(self.mappings),
            "sampling_coefficients": _coefficients_payload(self.sampling_coefficients),
            "semantic_progress_evidence_sha256": self.semantic_progress_evidence_sha256,
            "raw_mark_count": self.raw_mark_count,
            "raw_mark_count_stratum": self.raw_mark_count_stratum,
            "canonical_successor_count": self.canonical_successor_count,
            "canonical_successor_count_stratum": self.canonical_successor_count_stratum,
            "successor_alias_multiplicity": self.successor_alias_multiplicity,
            "successor_alias_multiplicity_stratum": (
                self.successor_alias_multiplicity_stratum
            ),
            "matching_mark_count": self.matching_mark_count,
            "atom_element_transition": [
                list(item) for item in self.atom_element_transition
            ],
            "minimum_edited_cycle_length": self.minimum_edited_cycle_length,
            "source_edge_aromatic": self.source_edge_aromatic,
            "successor_edge_aromatic": self.successor_edge_aromatic,
            "registry_sha256": self.registry_sha256,
            "process_identity_sha256": self.process_identity_sha256,
            "corpus_contract_file_sha256": self.corpus_contract_file_sha256,
            "classifier_implementation_sha256": self.classifier_implementation_sha256,
            "active8_policy_sha256": self.active8_policy_sha256,
            "assignment_sha256": self.assignment_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticStructuralCapabilityAssignment:
    """Narrow Gate-0/T1 projection over one verified accepted transition."""

    decision_source_inventory_sha256: str
    decision_sha256: str
    trace_address_sha256: str
    source_progress_address_sha256: str
    successor_progress_address_sha256: str
    trace_id: str
    progress_index: int
    packed_shard_content_sha256: str
    packed_shard_name: str
    packed_entry_index: int
    action_sha256: str
    source_state_sha256: str
    target_state_sha256: str
    source_canonical_key: str
    successor_canonical_key: str
    model_family: str
    family_context: str
    capability_cell_id: str
    data_lane: str
    partition_role: str
    raw_mark_count: int
    raw_mark_count_stratum: str
    canonical_successor_count: int
    canonical_successor_count_stratum: str
    successor_alias_multiplicity: int
    successor_alias_multiplicity_stratum: str
    matching_mark_count: int
    atom_element_transition: tuple[tuple[str, str], ...]
    minimum_edited_cycle_length: int | None
    source_edge_aromatic: bool | None
    successor_edge_aromatic: bool | None
    registry_sha256: str
    process_identity_sha256: str
    corpus_contract_file_sha256: str
    classifier_implementation_sha256: str
    active8_policy_sha256: str
    assignment_sha256: str

    def as_payload(self) -> dict[str, object]:
        """Return the exact, non-authorizing structural evidence row."""

        return {
            "schema": STRUCTURAL_ASSIGNMENT_SCHEMA,
            "schema_version": STRUCTURAL_ASSIGNMENT_SCHEMA_VERSION,
            "status": STRUCTURAL_ASSIGNMENT_STATUS,
            "training_authorized": False,
            "gate_zero_authorized": False,
            "t1_authorized": False,
            "bounded_p50_authorized": False,
            "long_training_authorized": False,
            "checkpoint_selection_authorized": False,
            "final_test_selection_authorized": False,
            "decision_source_inventory_sha256": (self.decision_source_inventory_sha256),
            "decision_sha256": self.decision_sha256,
            "trace_address_sha256": self.trace_address_sha256,
            "source_progress_address_sha256": (self.source_progress_address_sha256),
            "successor_progress_address_sha256": (
                self.successor_progress_address_sha256
            ),
            "trace_id": self.trace_id,
            "progress_index": self.progress_index,
            "packed_shard_content_sha256": self.packed_shard_content_sha256,
            "packed_shard_name": self.packed_shard_name,
            "packed_entry_index": self.packed_entry_index,
            "action_sha256": self.action_sha256,
            "source_state_sha256": self.source_state_sha256,
            "target_state_sha256": self.target_state_sha256,
            "source_canonical_key": self.source_canonical_key,
            "successor_canonical_key": self.successor_canonical_key,
            "model_family": self.model_family,
            "family_context": self.family_context,
            "capability_cell_id": self.capability_cell_id,
            "data_lane": self.data_lane,
            "partition_role": self.partition_role,
            "raw_mark_count": self.raw_mark_count,
            "raw_mark_count_stratum": self.raw_mark_count_stratum,
            "canonical_successor_count": self.canonical_successor_count,
            "canonical_successor_count_stratum": (
                self.canonical_successor_count_stratum
            ),
            "successor_alias_multiplicity": self.successor_alias_multiplicity,
            "successor_alias_multiplicity_stratum": (
                self.successor_alias_multiplicity_stratum
            ),
            "matching_mark_count": self.matching_mark_count,
            "atom_element_transition": [
                list(item) for item in self.atom_element_transition
            ],
            "minimum_edited_cycle_length": self.minimum_edited_cycle_length,
            "source_edge_aromatic": self.source_edge_aromatic,
            "successor_edge_aromatic": self.successor_edge_aromatic,
            "registry_sha256": self.registry_sha256,
            "process_identity_sha256": self.process_identity_sha256,
            "corpus_contract_file_sha256": self.corpus_contract_file_sha256,
            "classifier_implementation_sha256": self.classifier_implementation_sha256,
            "active8_policy_sha256": self.active8_policy_sha256,
            "assignment_sha256": self.assignment_sha256,
        }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SemanticCapabilityCellError(f"{field} must be a lowercase SHA-256")
    return value


def _require_mapping(
    value: object,
    *,
    field: str,
    fields: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticCapabilityCellError(f"{field} must be an object")
    if fields is not None and set(value) != fields:
        raise SemanticCapabilityCellError(
            f"{field} fields disagree; "
            f"missing={sorted(fields - set(value))}, "
            f"unexpected={sorted(set(value) - fields)}"
        )
    return value


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise SemanticCapabilityCellError(f"{field} must be normalized nonempty text")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field=field)


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise SemanticCapabilityCellError(f"{field} must be a nonnegative integer")
    return value


def _require_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise SemanticCapabilityCellError(f"{field} must be an integer")
    return value


def _require_finite_float(
    value: object,
    *,
    field: str,
    positive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticCapabilityCellError(f"{field} must be a finite number")
    normalized = float(value)
    if (
        not math.isfinite(normalized)
        or (positive and normalized <= 0.0)
        or normalized < 0.0
    ):
        qualifier = "positive finite" if positive else "nonnegative finite"
        raise SemanticCapabilityCellError(f"{field} must be {qualifier}")
    return normalized


def _require_sorted_text_tuple(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, tuple):
        raise SemanticCapabilityCellError(f"{field} must be a tuple")
    normalized = tuple(
        _require_text(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if (not allow_empty and not normalized) or len(normalized) != len(set(normalized)):
        raise SemanticCapabilityCellError(
            f"{field} must contain unique normalized values"
        )
    if normalized != tuple(sorted(normalized)):
        raise SemanticCapabilityCellError(f"{field} must be deterministically sorted")
    return normalized


def _slot_mapping_payload(
    value: tuple[tuple[int, int], ...] | None,
    *,
    field: str,
) -> list[list[int]] | None:
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise SemanticCapabilityCellError(f"{field} must be a tuple of slot pairs")
    pairs: list[tuple[int, int]] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise SemanticCapabilityCellError(
                f"{field}[{index}] must be a two-slot tuple"
            )
        pairs.append(
            (
                _require_nonnegative_int(
                    pair[0], field=f"{field}[{index}].source_slot"
                ),
                _require_nonnegative_int(
                    pair[1], field=f"{field}[{index}].target_slot"
                ),
            )
        )
    if pairs != sorted(pairs):
        raise SemanticCapabilityCellError(f"{field} must be deterministically sorted")
    if len({left for left, _ in pairs}) != len(pairs) or len(
        {right for _, right in pairs}
    ) != len(pairs):
        raise SemanticCapabilityCellError(f"{field} must be one-to-one")
    return [[left, right] for left, right in pairs]


def _groups_payload(value: SemanticGroups) -> dict[str, object]:
    if not isinstance(value, SemanticGroups):
        raise SemanticCapabilityCellError(
            "semantic groups must use the typed sidecar schema"
        )
    source = _require_text(value.source_group_id, field="groups.source_group_id")
    scaffold = _require_text(value.scaffold_group_id, field="groups.scaffold_group_id")
    series = _optional_text(value.series_group_id, field="groups.series_group_id")
    return {
        "source_group_id": source,
        "scaffold_group_id": scaffold,
        "series_group_id": series,
        "document_or_source_group_id": _require_text(
            value.document_or_source_group_id,
            field="groups.document_or_source_group_id",
        ),
        "transformation_signature": _optional_text(
            value.transformation_signature,
            field="groups.transformation_signature",
        ),
        "series_scaffold_or_source_group_id": series or scaffold or source,
    }


def _endpoint_descriptor_payload(value: EndpointDescriptor) -> dict[str, object]:
    if not isinstance(value, EndpointDescriptor):
        raise SemanticCapabilityCellError(
            "endpoint descriptor must use the typed sidecar schema"
        )
    return {
        "source_canonical_key": _require_text(
            value.source_canonical_key,
            field="endpoint.source_canonical_key",
        ),
        "target_canonical_key": _require_text(
            value.target_canonical_key,
            field="endpoint.target_canonical_key",
        ),
        "atom_count_delta": _require_int(
            value.atom_count_delta,
            field="endpoint.atom_count_delta",
        ),
        "graph_cycle_rank_delta": _require_int(
            value.graph_cycle_rank_delta,
            field="endpoint.graph_cycle_rank_delta",
        ),
        "source_endpoint_evidence_kind": _require_text(
            value.source_endpoint_evidence_kind,
            field="endpoint.source_endpoint_evidence_kind",
        ),
        "target_endpoint_evidence_kind": _require_text(
            value.target_endpoint_evidence_kind,
            field="endpoint.target_endpoint_evidence_kind",
        ),
        "pair_relationship_evidence_kind": _require_text(
            value.pair_relationship_evidence_kind,
            field="endpoint.pair_relationship_evidence_kind",
        ),
    }


def _path_descriptor_payload(value: PathDescriptor) -> dict[str, object]:
    if not isinstance(value, PathDescriptor):
        raise SemanticCapabilityCellError(
            "path descriptor must use the typed sidecar schema"
        )
    unresolved = _require_sorted_text_tuple(
        value.unresolved_metric_policies,
        field="path.unresolved_metric_policies",
        allow_empty=True,
    )
    if unresolved:
        raise SemanticCapabilityCellError(
            "path descriptor has unresolved metric policies"
        )
    return {
        "path_length": _require_nonnegative_int(
            value.path_length, field="path.path_length"
        ),
        "operator_family_set": list(value.operator_family_set),
        "path_evidence_kind": _require_text(
            value.path_evidence_kind,
            field="path.path_evidence_kind",
        ),
        "intermediate_evidence_kind": _require_text(
            value.intermediate_evidence_kind,
            field="path.intermediate_evidence_kind",
        ),
        "action_sequence_evidence_kind": _require_text(
            value.action_sequence_evidence_kind,
            field="path.action_sequence_evidence_kind",
        ),
        "maximum_intermediate_source_distance": _require_finite_float(
            value.maximum_intermediate_source_distance,
            field="path.maximum_intermediate_source_distance",
            positive=False,
        ),
        "immediate_reversal_count": _require_nonnegative_int(
            value.immediate_reversal_count,
            field="path.immediate_reversal_count",
        ),
        "repeated_state_count": _require_nonnegative_int(
            value.repeated_state_count,
            field="path.repeated_state_count",
        ),
        "metric_policy_sha256": _require_sha256(
            value.metric_policy_sha256,
            field="path.metric_policy_sha256",
        ),
        "unresolved_metric_policies": [],
    }


def _mappings_payload(value: MappingDescriptor) -> dict[str, object]:
    if not isinstance(value, MappingDescriptor):
        raise SemanticCapabilityCellError("mappings must use the typed sidecar schema")
    unresolved = _require_sorted_text_tuple(
        value.unresolved_mapping_policies,
        field="mappings.unresolved_mapping_policies",
        allow_empty=True,
    )
    if unresolved:
        raise SemanticCapabilityCellError("mapping descriptor has unresolved policies")
    return {
        "constant_core_mapping": _slot_mapping_payload(
            value.constant_core_mapping,
            field="mappings.constant_core_mapping",
        ),
        "protected_mapping": _slot_mapping_payload(
            value.protected_mapping,
            field="mappings.protected_mapping",
        ),
        "mapping_policy_sha256": _require_sha256(
            value.mapping_policy_sha256,
            field="mappings.mapping_policy_sha256",
        ),
        "unresolved_mapping_policies": [],
    }


def _coefficients_payload(value: SamplingCoefficients) -> dict[str, object]:
    if not isinstance(value, SamplingCoefficients):
        raise SemanticCapabilityCellError(
            "sampling coefficients must use the typed sidecar schema"
        )
    unresolved = _require_sorted_text_tuple(
        value.unresolved_coefficient_policies,
        field="coefficients.unresolved_coefficient_policies",
        allow_empty=True,
    )
    if unresolved:
        raise SemanticCapabilityCellError(
            "sampling coefficients have unresolved policies"
        )
    return {
        "path_sampling_coefficient": _require_finite_float(
            value.path_sampling_coefficient,
            field="coefficients.path_sampling_coefficient",
            positive=True,
        ),
        "progress_sampling_coefficient": _require_finite_float(
            value.progress_sampling_coefficient,
            field="coefficients.progress_sampling_coefficient",
            positive=True,
        ),
        "effective_teacher_coefficient": _require_finite_float(
            value.effective_teacher_coefficient,
            field="coefficients.effective_teacher_coefficient",
            positive=True,
        ),
        "coefficient_policy_sha256": _require_sha256(
            value.coefficient_policy_sha256,
            field="coefficients.coefficient_policy_sha256",
        ),
        "unresolved_coefficient_policies": [],
    }


def _require_false(value: object, *, field: str) -> None:
    if value is not False:
        raise SemanticCapabilityCellError(f"{field} must remain false")


def _require_string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SemanticCapabilityCellError(f"{field} must be a JSON array")
    items = tuple(
        _require_text(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if len(items) != len(set(items)):
        raise SemanticCapabilityCellError(f"{field} contains duplicates")
    return items


def _validate_bins(value: object, *, field: str) -> tuple[CountBin, ...]:
    payload = _require_mapping(value, field=field, fields=_STRATUM_FIELDS)
    _require_text(payload["scope"], field=f"{field}.scope")
    raw_bins = payload["bins"]
    if not isinstance(raw_bins, list) or not raw_bins:
        raise SemanticCapabilityCellError(f"{field}.bins must be nonempty")
    bins: list[CountBin] = []
    expected_minimum = 1
    for index, raw_bin in enumerate(raw_bins):
        item = _require_mapping(
            raw_bin,
            field=f"{field}.bins[{index}]",
            fields=_BIN_FIELDS,
        )
        identifier = _require_text(item["id"], field=f"{field}.bins[{index}].id")
        minimum = item["minimum"]
        maximum = item["maximum"]
        if type(minimum) is not int or minimum != expected_minimum:
            raise SemanticCapabilityCellError(
                f"{field}.bins[{index}] does not continue the positive-integer partition"
            )
        if maximum is None:
            if index != len(raw_bins) - 1:
                raise SemanticCapabilityCellError(
                    f"{field}.bins[{index}] is open-ended before the final bin"
                )
        elif type(maximum) is not int or maximum < minimum:
            raise SemanticCapabilityCellError(
                f"{field}.bins[{index}].maximum is invalid"
            )
        bins.append(CountBin(identifier, minimum, maximum))
        if maximum is not None:
            expected_minimum = maximum + 1
    if bins[-1].maximum is not None or len({item.id for item in bins}) != len(bins):
        raise SemanticCapabilityCellError(
            f"{field}.bins must end open-ended and use unique IDs"
        )
    return tuple(bins)


def validate_semantic_capability_cell_registry(
    payload: Mapping[str, Any],
) -> SemanticCapabilityCellRegistry:
    """Validate exact structure, self-hash, bindings, and non-authority."""

    root = _require_mapping(payload, field="registry", fields=_TOP_LEVEL_FIELDS)
    if (
        root["schema"] != REGISTRY_SCHEMA
        or root["schema_version"] != REGISTRY_SCHEMA_VERSION
    ):
        raise SemanticCapabilityCellError(
            "semantic capability registry schema is unknown"
        )
    if root["status"] != REGISTRY_STATUS:
        raise SemanticCapabilityCellError(
            "semantic capability registry status has drifted"
        )
    for field in (
        "training_authorized",
        "gate_zero_authorized",
        "t1_authorized",
        "bounded_p50_authorized",
    ):
        _require_false(root[field], field=field)
    registry_id = _require_text(root["registry_id"], field="registry_id")
    expected_sha256 = _canonical_sha256(
        {key: value for key, value in root.items() if key != "registry_sha256"}
    )
    if root["registry_sha256"] != expected_sha256:
        raise SemanticCapabilityCellError(
            "semantic capability registry self-hash disagrees"
        )

    scope = _require_mapping(root["scope"], field="scope", fields=_SCOPE_FIELDS)
    expected_scope = {
        "row_unit": "accepted_nonterminal_progress_row_with_exact_action_v4_teacher",
        "generated_object": "canonical_molecular_successor",
        "objective_lane": "productive_embedded_jump_chain",
        "terminal_rows": "outside_capability_cell_registry_hazard_is_separate",
        "maximum_active_atoms": 40,
        "stereochemistry": "out_of_scope",
        "formal_charge_changes": "out_of_scope_charge_preserving_only",
    }
    if dict(scope) != expected_scope:
        raise SemanticCapabilityCellError(
            "semantic capability registry scope has drifted"
        )

    bindings = _require_mapping(
        root["bindings"], field="bindings", fields=_BINDING_FIELDS
    )
    active_families = _require_string_tuple(
        bindings["active_families"], field="bindings.active_families"
    )
    data_lanes = _require_string_tuple(
        bindings["data_lanes"], field="bindings.data_lanes"
    )
    partition_roles = _require_string_tuple(
        bindings["partition_roles"], field="bindings.partition_roles"
    )
    process_identity = editing_v2_process_identity()
    corpus_path = _repository_root() / str(bindings["corpus_contract"])
    corpus_contract_file_sha256 = _file_sha256(corpus_path)
    classifier_path = _repository_root() / str(
        bindings["classifier_implementation_file"]
    )
    classifier_implementation_sha256 = _file_sha256(classifier_path)
    if (
        bindings["semantic_process_contract"]
        != "configs/editing_v2_semantic_process_v1.json"
        or bindings["semantic_process_identity_sha256"]
        != process_identity["process_identity_sha256"]
        or bindings["corpus_contract"] != "configs/editing_corpus_v2_contract.json"
        or bindings["corpus_contract_file_sha256"] != corpus_contract_file_sha256
        or bindings["classifier_implementation_file"]
        != "src/compose_v4/data/editing_v2_semantic_capability_cells.py"
        or bindings["classifier_implementation_sha256"]
        != classifier_implementation_sha256
        or bindings["action_codec_schema_version"] != action_codec_v4.SCHEMA_VERSION
        or active_families != tuple(ACTIVE8_FAMILIES)
        or data_lanes != tuple(REQUIRED_DATA_LANES)
        or partition_roles != tuple(REQUIRED_PARTITION_ROLES)
    ):
        raise SemanticCapabilityCellError(
            "semantic capability registry bindings have drifted"
        )

    policy = _require_mapping(
        root["cell_identity_policy"],
        field="cell_identity_policy",
        fields=_CELL_POLICY_FIELDS,
    )
    namespace = _require_text(
        policy["namespace"], field="cell_identity_policy.namespace"
    )
    balancing = _require_string_tuple(
        policy["balancing_dimensions"],
        field="cell_identity_policy.balancing_dimensions",
    )
    separate = _require_string_tuple(
        policy["separate_nonbalancing_dimensions"],
        field="cell_identity_policy.separate_nonbalancing_dimensions",
    )
    if balancing != ("model_family", "family_specific_context"):
        raise SemanticCapabilityCellError("balancing dimensions must remain compact")
    if set(separate) != _REQUIRED_SEPARATE_DIMENSIONS:
        raise SemanticCapabilityCellError(
            "nonbalancing audit dimensions are incomplete"
        )
    _require_text(policy["rationale"], field="cell_identity_policy.rationale")

    raw_contexts = _require_mapping(root["family_contexts"], field="family_contexts")
    contexts = {
        family: _require_string_tuple(
            raw_contexts.get(family), field=f"family_contexts.{family}"
        )
        for family in ACTIVE8_FAMILIES
    }
    if set(raw_contexts) != set(ACTIVE8_FAMILIES) or contexts != _KNOWN_CONTEXTS:
        raise SemanticCapabilityCellError(
            "family-specific capability contexts have drifted"
        )

    strata = _require_mapping(
        root["exact_evidence_strata"],
        field="exact_evidence_strata",
        fields=_STRATA_FIELDS,
    )
    if strata["policy"] != (
        "prospective_non_data_derived_logarithmic_engineering_bins_not_balancing_dimensions"
    ):
        raise SemanticCapabilityCellError("exact-evidence stratum policy has drifted")
    raw_bins = _validate_bins(
        strata["raw_mark_count"], field="exact_evidence_strata.raw_mark_count"
    )
    successor_bins = _validate_bins(
        strata["canonical_successor_count"],
        field="exact_evidence_strata.canonical_successor_count",
    )
    alias_bins = _validate_bins(
        strata["successor_alias_multiplicity"],
        field="exact_evidence_strata.successor_alias_multiplicity",
    )

    fail_closed = _require_mapping(
        root["fail_closed_policy"],
        field="fail_closed_policy",
        fields=_FAIL_CLOSED_FIELDS,
    )
    if any(value != "error" for value in fail_closed.values()):
        raise SemanticCapabilityCellError("all unknown or stale cases must fail closed")

    corpus_contract = load_editing_corpus_contract(corpus_path)
    if corpus_contract["training_authorized"] is not False:
        raise SemanticCapabilityCellError(
            "capability registry cannot bind a training-authorizing corpus contract"
        )
    return SemanticCapabilityCellRegistry(
        registry_id=registry_id,
        namespace=namespace,
        family_contexts=tuple(
            (family, contexts[family]) for family in ACTIVE8_FAMILIES
        ),
        raw_mark_bins=raw_bins,
        canonical_successor_bins=successor_bins,
        successor_alias_bins=alias_bins,
        registry_sha256=expected_sha256,
        process_identity_sha256=str(process_identity["process_identity_sha256"]),
        corpus_contract_file_sha256=corpus_contract_file_sha256,
        classifier_implementation_sha256=classifier_implementation_sha256,
    )


@lru_cache(maxsize=1)
def _load_default_registry() -> SemanticCapabilityCellRegistry:
    path = _repository_root() / REGISTRY_RELATIVE_PATH
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticCapabilityCellError(
            f"cannot read semantic capability registry {path}"
        ) from error
    return validate_semantic_capability_cell_registry(payload)


def load_semantic_capability_cell_registry(
    path: str | Path | None = None,
) -> SemanticCapabilityCellRegistry:
    """Load the default registry or validate an explicit registry file."""

    if path is None:
        return _load_default_registry()
    selected = Path(path)
    try:
        payload = json.loads(selected.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticCapabilityCellError(
            f"cannot read semantic capability registry {selected}"
        ) from error
    return validate_semantic_capability_cell_registry(payload)


def _resolve_evidence_profile(
    contract: Mapping[str, Any],
    *,
    data_lane: str,
    endpoint: Mapping[str, object],
    path: Mapping[str, object],
) -> str:
    components = {
        "source_endpoint": str(endpoint["source_endpoint_evidence_kind"]),
        "target_endpoint": str(endpoint["target_endpoint_evidence_kind"]),
        "pair_relationship": str(endpoint["pair_relationship_evidence_kind"]),
        "path": str(path["path_evidence_kind"]),
        "intermediates": str(path["intermediate_evidence_kind"]),
        "action_sequence": str(path["action_sequence_evidence_kind"]),
    }
    matches = tuple(
        str(profile["id"])
        for profile in contract["evidence_component_contract"]["profiles"]
        if profile["components"] == components
    )
    if len(matches) != 1:
        raise SemanticCapabilityCellError(
            "semantic evidence components do not identify exactly one profile"
        )
    try:
        validate_evidence_assignment_envelope(
            contract,
            data_lane=data_lane,
            evidence_profile_id=matches[0],
            evidence_components=components,
        )
    except EditingCorpusContractError as error:
        raise SemanticCapabilityCellError(
            "progress row lane and evidence components are inconsistent"
        ) from error
    return matches[0]


def _validate_directional_memberships(
    memberships: tuple[str, ...],
    *,
    atom_count_delta: int,
    graph_cycle_rank_delta: int,
) -> None:
    selected = set(memberships)
    expected = {
        "cardinality_growth": atom_count_delta > 0,
        "cardinality_shrinkage": atom_count_delta < 0,
        "cycle_rank_increase": graph_cycle_rank_delta > 0,
        "cycle_rank_decrease": graph_cycle_rank_delta < 0,
    }
    disagreements = tuple(
        cell for cell, present in expected.items() if (cell in selected) is not present
    )
    if disagreements:
        raise SemanticCapabilityCellError(
            "directional record memberships disagree with exact endpoint deltas: "
            + ",".join(disagreements)
        )


def _semantic_progress_payload(
    progress: SemanticSamplingProgress,
    *,
    groups: Mapping[str, object],
    endpoint: Mapping[str, object],
    path: Mapping[str, object],
    relationship_group_ids: tuple[str, ...],
    mappings: Mapping[str, object],
    coefficients: Mapping[str, object],
) -> dict[str, object]:
    address = progress.address
    return {
        "address": {
            "packed_shard_content_sha256": address.packed_shard_content_sha256,
            "packed_shard_name": address.packed_shard_name,
            "entry_index": address.entry_index,
            "trace_id": address.trace_id,
            "data_lane": address.layer,
            "partition_role": address.partition,
            "source_key": address.source_key,
            "target_key": address.target_key,
            "path_length": address.path_length,
        },
        "progress_index": progress.progress_index,
        "candidate_ledger_row_sha256": progress.candidate_ledger_row_sha256,
        "partition_resolution": dict(progress.partition_resolution),
        "admitted_record_envelope_sha256": progress.admitted_record_envelope_sha256,
        "groups": dict(groups),
        "record_membership_cells": list(progress.record_membership_cells),
        "record_membership_classifier_identity_sha256": (
            progress.record_membership_classifier_identity_sha256
        ),
        "endpoint_descriptor": dict(endpoint),
        "path_descriptor": dict(path),
        "relationship_group_ids": list(relationship_group_ids),
        "mappings": dict(mappings),
        "sampling_coefficients": dict(coefficients),
    }


def _validate_exact_evidence(
    evidence: SemanticExactCandidateEvidence | None,
    *,
    action_sha256: str,
    source: MolecularGraph,
    successor: MolecularGraph,
) -> SemanticExactCandidateEvidence:
    if evidence is None:
        raise SemanticCapabilityCellError(
            "accepted teacher lacks exact candidate evidence"
        )
    if not evidence.supported or evidence.exclusion_reason is not None:
        raise SemanticCapabilityCellError(
            "unsupported teacher cannot receive a capability cell"
        )
    if evidence.action_sha256 != action_sha256:
        raise SemanticCapabilityCellError(
            "candidate evidence action identity has drifted"
        )
    if evidence.source_state_sha256 != persistent_slot_state_sha256(source):
        raise SemanticCapabilityCellError(
            "candidate evidence source identity has drifted"
        )
    if evidence.target_state_sha256 != persistent_slot_state_sha256(successor):
        raise SemanticCapabilityCellError(
            "candidate evidence target identity has drifted"
        )
    if evidence.canonical_successor_key != canonical_state_key(successor):
        raise SemanticCapabilityCellError(
            "candidate evidence canonical successor has drifted"
        )
    if evidence.matching_mark_count != 1:
        raise SemanticCapabilityCellError(
            "accepted teacher must appear exactly once in the exact Action-V4 fiber"
        )
    if (
        evidence.raw_mark_count < 1
        or evidence.canonical_successor_count < 1
        or evidence.successor_alias_count < 1
        or evidence.raw_mark_count < evidence.canonical_successor_count
        or evidence.successor_alias_count > evidence.raw_mark_count
        or evidence.matching_mark_count > evidence.successor_alias_count
    ):
        raise SemanticCapabilityCellError(
            "candidate-fiber counts are internally inconsistent"
        )
    return evidence


def classify_verified_structural_transition(
    decision_index: EditingV2SemanticActive8DecisionIndex,
    transition: SemanticActive8AcceptedTransition,
    *,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> SemanticStructuralCapabilityAssignment:
    """Classify exact structural evidence without requiring sampling semantics."""

    if not isinstance(decision_index, EditingV2SemanticActive8DecisionIndex):
        raise TypeError(
            "structural capability classification requires the verified decision index"
        )
    if not isinstance(transition, SemanticActive8AcceptedTransition):
        raise TypeError(
            "structural capability classification requires a verified transition"
        )
    selected = registry or load_semantic_capability_cell_registry()
    if not isinstance(selected, SemanticCapabilityCellRegistry):
        raise TypeError("registry must be a SemanticCapabilityCellRegistry")
    try:
        decision_index.validate_accepted_transition(transition)
    except SemanticActive8DecisionSourceError as error:
        raise SemanticCapabilityCellError(
            "structural transition failed decision-source revalidation"
        ) from error
    if (
        transition.decision_source_inventory_sha256 != decision_index.inventory_sha256
        or decision_index.process_identity_sha256 != selected.process_identity_sha256
    ):
        raise SemanticCapabilityCellError(
            "structural transition process or source identity has drifted"
        )

    addressed = transition.addressed_trace
    address = addressed.address
    index = transition.step_index
    step = addressed.trace.steps[index]
    source = addressed.path.state_at(index)
    successor = addressed.path.state_at(index + 1)
    policy = build_semantic_active8_admission_policy()
    current_classification = classify_semantic_action(
        step,
        step_index=index,
        policy=policy,
    )
    action_decision = transition.action_decision
    if (
        transition.decision.trace_id != address.trace_id
        or transition.decision.policy_sha256 != policy.policy_sha256
        or transition.decision.semantic_migration_status != "admitted"
        or transition.decision.semantic_migration_rejection is not None
        or action_decision.classification != current_classification
        or not current_classification.policy_eligible
        or current_classification.model_family not in ACTIVE8_FAMILIES
    ):
        raise SemanticCapabilityCellError(
            "structural teacher is not current accepted Action-V4 evidence"
        )
    evidence = _validate_exact_evidence(
        action_decision.candidate_evidence,
        action_sha256=current_classification.action_sha256,
        source=source,
        successor=successor,
    )
    family, context = classify_action_family_context(source, successor, step)
    if family != current_classification.model_family:
        raise SemanticCapabilityCellError(
            "structural family-context classifier disagrees with Action V4"
        )
    if context not in selected.contexts_by_family.get(family, ()):
        raise SemanticCapabilityCellError(
            "structural family context is absent from the bound registry"
        )

    trace_address_payload = {
        "packed_shard_content_sha256": address.packed_shard_content_sha256,
        "packed_shard_name": address.packed_shard_name,
        "entry_index": address.entry_index,
        "trace_id": address.trace_id,
        "layer": address.layer,
        "partition": address.partition,
        "source_key": address.source_key,
        "target_key": address.target_key,
        "path_length": address.path_length,
    }
    trace_address_sha256 = _canonical_sha256(trace_address_payload)
    source_progress_sha256 = _canonical_sha256(
        transition.source_progress_address.as_payload()
    )
    successor_progress_sha256 = _canonical_sha256(
        transition.successor_progress_address.as_payload()
    )
    raw_stratum = _stratum(
        evidence.raw_mark_count,
        selected.raw_mark_bins,
        field="raw_mark_count",
    )
    successor_stratum = _stratum(
        evidence.canonical_successor_count,
        selected.canonical_successor_bins,
        field="canonical_successor_count",
    )
    alias_stratum = _stratum(
        evidence.successor_alias_count,
        selected.successor_alias_bins,
        field="successor_alias_count",
    )
    (
        atom_element_transition,
        minimum_edited_cycle_length,
        source_edge_aromatic,
        successor_edge_aromatic,
    ) = _action_audit_axes(source, successor, step, family=family)
    capability_cell_id = f"{selected.namespace}:{family}:{context}"
    body = {
        "schema": STRUCTURAL_ASSIGNMENT_SCHEMA,
        "schema_version": STRUCTURAL_ASSIGNMENT_SCHEMA_VERSION,
        "status": STRUCTURAL_ASSIGNMENT_STATUS,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_authorized": False,
        "final_test_selection_authorized": False,
        "decision_source_inventory_sha256": decision_index.inventory_sha256,
        "decision_sha256": transition.decision_sha256,
        "trace_address_sha256": trace_address_sha256,
        "source_progress_address_sha256": source_progress_sha256,
        "successor_progress_address_sha256": successor_progress_sha256,
        "trace_id": address.trace_id,
        "progress_index": index,
        "packed_shard_content_sha256": address.packed_shard_content_sha256,
        "packed_shard_name": address.packed_shard_name,
        "packed_entry_index": address.entry_index,
        "action_sha256": current_classification.action_sha256,
        "source_state_sha256": evidence.source_state_sha256,
        "target_state_sha256": evidence.target_state_sha256,
        "source_canonical_key": canonical_state_key(source),
        "successor_canonical_key": evidence.canonical_successor_key,
        "model_family": family,
        "family_context": context,
        "capability_cell_id": capability_cell_id,
        "data_lane": address.layer,
        "partition_role": address.partition,
        "raw_mark_count": evidence.raw_mark_count,
        "raw_mark_count_stratum": raw_stratum,
        "canonical_successor_count": evidence.canonical_successor_count,
        "canonical_successor_count_stratum": successor_stratum,
        "successor_alias_multiplicity": evidence.successor_alias_count,
        "successor_alias_multiplicity_stratum": alias_stratum,
        "matching_mark_count": evidence.matching_mark_count,
        "atom_element_transition": [list(item) for item in atom_element_transition],
        "minimum_edited_cycle_length": minimum_edited_cycle_length,
        "source_edge_aromatic": source_edge_aromatic,
        "successor_edge_aromatic": successor_edge_aromatic,
        "registry_sha256": selected.registry_sha256,
        "process_identity_sha256": selected.process_identity_sha256,
        "corpus_contract_file_sha256": selected.corpus_contract_file_sha256,
        "classifier_implementation_sha256": selected.classifier_implementation_sha256,
        "active8_policy_sha256": policy.policy_sha256,
    }
    assignment = SemanticStructuralCapabilityAssignment(
        decision_source_inventory_sha256=decision_index.inventory_sha256,
        decision_sha256=transition.decision_sha256,
        trace_address_sha256=trace_address_sha256,
        source_progress_address_sha256=source_progress_sha256,
        successor_progress_address_sha256=successor_progress_sha256,
        trace_id=address.trace_id,
        progress_index=index,
        packed_shard_content_sha256=address.packed_shard_content_sha256,
        packed_shard_name=address.packed_shard_name,
        packed_entry_index=address.entry_index,
        action_sha256=current_classification.action_sha256,
        source_state_sha256=evidence.source_state_sha256,
        target_state_sha256=evidence.target_state_sha256,
        source_canonical_key=canonical_state_key(source),
        successor_canonical_key=evidence.canonical_successor_key,
        model_family=family,
        family_context=context,
        capability_cell_id=capability_cell_id,
        data_lane=address.layer,
        partition_role=address.partition,
        raw_mark_count=evidence.raw_mark_count,
        raw_mark_count_stratum=raw_stratum,
        canonical_successor_count=evidence.canonical_successor_count,
        canonical_successor_count_stratum=successor_stratum,
        successor_alias_multiplicity=evidence.successor_alias_count,
        successor_alias_multiplicity_stratum=alias_stratum,
        matching_mark_count=evidence.matching_mark_count,
        atom_element_transition=atom_element_transition,
        minimum_edited_cycle_length=minimum_edited_cycle_length,
        source_edge_aromatic=source_edge_aromatic,
        successor_edge_aromatic=successor_edge_aromatic,
        registry_sha256=selected.registry_sha256,
        process_identity_sha256=selected.process_identity_sha256,
        corpus_contract_file_sha256=selected.corpus_contract_file_sha256,
        classifier_implementation_sha256=selected.classifier_implementation_sha256,
        active8_policy_sha256=policy.policy_sha256,
        assignment_sha256=_canonical_sha256(body),
    )
    payload = assignment.as_payload()
    if assignment.assignment_sha256 != _canonical_sha256(
        {key: value for key, value in payload.items() if key != "assignment_sha256"}
    ):
        raise SemanticCapabilityCellError(
            "structural capability assignment self-hash disagrees"
        )
    return assignment


def classify_accepted_semantic_progress(
    addressed: AddressedPackedTrace,
    progress: SemanticSamplingProgress,
    trace_decision: SemanticActive8TraceDecision,
    *,
    registry: SemanticCapabilityCellRegistry | None = None,
) -> SemanticCapabilityCellAssignment:
    """Assign one accepted nonterminal progress row without granting authority."""

    selected = registry or load_semantic_capability_cell_registry()
    live_registry = load_semantic_capability_cell_registry()
    if selected != live_registry:
        raise SemanticCapabilityCellError(
            "semantic capability registry is stale or substituted"
        )
    policy = build_semantic_active8_admission_policy()
    if selected.process_identity_sha256 != policy.process_identity_sha256:
        raise SemanticCapabilityCellError(
            "semantic process identity differs from Active8 policy"
        )
    if progress.address != addressed.address:
        raise SemanticCapabilityCellError(
            "progress row and addressed trace identities differ"
        )
    if (
        addressed.address.path_length != len(addressed.trace.steps)
        or addressed.path.path_length != addressed.address.path_length
    ):
        raise SemanticCapabilityCellError("addressed trace path length has drifted")
    if progress.address.layer not in REQUIRED_DATA_LANES:
        raise SemanticCapabilityCellError(
            "progress row data lane is outside the five-lane contract"
        )
    if progress.address.partition not in REQUIRED_PARTITION_ROLES:
        raise SemanticCapabilityCellError(
            "progress row partition role is outside the split contract"
        )
    corpus_contract = load_editing_corpus_contract(
        _repository_root() / "configs/editing_corpus_v2_contract.json"
    )
    try:
        validate_partition_resolution(
            corpus_contract,
            progress.partition_resolution,
            admitted_record=True,
        )
    except (TypeError, EditingCorpusContractError) as error:
        raise SemanticCapabilityCellError(
            "progress row split resolution is invalid"
        ) from error
    try:
        validate_record_membership_assignment(
            corpus_contract,
            progress.record_membership_cells,
        )
    except (TypeError, EditingCorpusContractError) as error:
        raise SemanticCapabilityCellError(
            "progress row record-membership assignment is invalid"
        ) from error
    membership_order = tuple(
        str(cell["id"])
        for cell in corpus_contract["record_membership_contract"]["cells"]
    )
    observed_memberships = tuple(progress.record_membership_cells)
    if observed_memberships != tuple(
        cell for cell in membership_order if cell in observed_memberships
    ):
        raise SemanticCapabilityCellError(
            "progress row record memberships are not in frozen contract order"
        )
    status_by_membership = {
        str(cell["id"]): str(cell["admission_status"])
        for cell in corpus_contract["record_membership_contract"]["cells"]
    }
    unresolved_memberships = {
        cell: status_by_membership[cell]
        for cell in observed_memberships
        if status_by_membership[cell] != "admissible"
    }
    if unresolved_memberships:
        raise SemanticCapabilityCellError(
            f"progress row carries unresolved membership cells: {unresolved_memberships}"
        )
    if progress.partition_resolution["assigned_role"] != progress.address.partition:
        raise SemanticCapabilityCellError(
            "progress row lane or partition resolution has drifted"
        )
    if (
        trace_decision.trace_id != addressed.address.trace_id
        or trace_decision.policy_sha256 != policy.policy_sha256
        or trace_decision.semantic_migration_status != "admitted"
        or trace_decision.active8_status != "accepted"
        or not trace_decision.emits_progress_rows
        or trace_decision.active8_exclusions
    ):
        raise SemanticCapabilityCellError(
            "trace lacks current accepted whole-trace evidence"
        )
    if len(trace_decision.action_decisions) != len(addressed.trace.steps):
        raise SemanticCapabilityCellError(
            "Active8 action evidence does not cover the complete trace"
        )
    trace_families: list[str] = []
    for step_index, (trace_step, decision) in enumerate(
        zip(addressed.trace.steps, trace_decision.action_decisions, strict=True)
    ):
        exact_classification = classify_semantic_action(
            trace_step,
            step_index=step_index,
            policy=policy,
        )
        if (
            not exact_classification.policy_eligible
            or exact_classification.model_family is None
            or decision.classification != exact_classification
        ):
            raise SemanticCapabilityCellError(
                "complete-trace Action-V4 classification has drifted"
            )
        trace_families.append(exact_classification.model_family)
    index = progress.progress_index
    if type(index) is not int or not 0 <= index < len(addressed.trace.steps):
        raise SemanticCapabilityCellError(
            "terminal or out-of-range progress rows have no capability cell"
        )
    step = addressed.trace.steps[index]
    action_decision = trace_decision.action_decisions[index]
    current_classification = classify_semantic_action(
        step,
        step_index=index,
        policy=policy,
    )
    if (
        not current_classification.policy_eligible
        or action_decision.classification != current_classification
        or current_classification.action_sha256 is None
        or current_classification.model_family is None
    ):
        raise SemanticCapabilityCellError(
            "progress teacher is not current Action-V4 evidence"
        )
    source = addressed.path.state_at(index)
    successor = addressed.path.state_at(index + 1)
    evidence = _validate_exact_evidence(
        action_decision.candidate_evidence,
        action_sha256=current_classification.action_sha256,
        source=source,
        successor=successor,
    )
    family, context = classify_action_family_context(source, successor, step)
    if family != current_classification.model_family:
        raise SemanticCapabilityCellError(
            "family-context classifier disagrees with Action V4"
        )
    if context not in selected.contexts_by_family.get(family, ()):
        raise SemanticCapabilityCellError(
            "family context is absent from the bound registry"
        )
    groups_payload = _groups_payload(progress.groups)
    endpoint_payload = _endpoint_descriptor_payload(progress.endpoint_descriptor)
    path_payload = _path_descriptor_payload(progress.path_descriptor)
    mappings_payload = _mappings_payload(progress.mappings)
    coefficients_payload = _coefficients_payload(progress.sampling_coefficients)
    relationship_group_ids = _require_sorted_text_tuple(
        progress.relationship_group_ids,
        field="relationship_group_ids",
        allow_empty=True,
    )
    endpoint_source = addressed.path.state_at(0)
    endpoint_target = addressed.path.state_at(addressed.address.path_length)
    exact_atom_count_delta = _real_atom_count(endpoint_target) - _real_atom_count(
        endpoint_source
    )
    exact_cycle_rank_delta = _cycle_rank(endpoint_target) - _cycle_rank(endpoint_source)
    exact_operator_family_set = tuple(
        candidate for candidate in ACTIVE8_FAMILIES if candidate in set(trace_families)
    )
    if (
        endpoint_payload["source_canonical_key"] != addressed.address.source_key
        or endpoint_payload["target_canonical_key"] != addressed.address.target_key
        or endpoint_payload["atom_count_delta"] != exact_atom_count_delta
        or endpoint_payload["graph_cycle_rank_delta"] != exact_cycle_rank_delta
        or path_payload["path_length"] != addressed.address.path_length
        or tuple(path_payload["operator_family_set"]) != exact_operator_family_set
    ):
        raise SemanticCapabilityCellError(
            "progress semantic descriptors differ from the exact trace"
        )
    _validate_directional_memberships(
        tuple(progress.record_membership_cells),
        atom_count_delta=exact_atom_count_delta,
        graph_cycle_rank_delta=exact_cycle_rank_delta,
    )
    evidence_profile_id = _resolve_evidence_profile(
        corpus_contract,
        data_lane=progress.address.layer,
        endpoint=endpoint_payload,
        path=path_payload,
    )
    semantic_progress_payload = _semantic_progress_payload(
        progress,
        groups=groups_payload,
        endpoint=endpoint_payload,
        path=path_payload,
        relationship_group_ids=relationship_group_ids,
        mappings=mappings_payload,
        coefficients=coefficients_payload,
    )
    semantic_progress_evidence_sha256 = _canonical_sha256(semantic_progress_payload)
    (
        atom_element_transition,
        minimum_edited_cycle_length,
        source_edge_aromatic,
        successor_edge_aromatic,
    ) = _action_audit_axes(source, successor, step, family=family)

    candidate_ledger_row_sha256 = _require_sha256(
        progress.candidate_ledger_row_sha256,
        field="candidate_ledger_row_sha256",
    )
    admitted_record_envelope_sha256 = _require_sha256(
        progress.admitted_record_envelope_sha256,
        field="admitted_record_envelope_sha256",
    )
    record_membership_classifier_identity_sha256 = _require_sha256(
        progress.record_membership_classifier_identity_sha256,
        field="record_membership_classifier_identity_sha256",
    )
    split_assignment_manifest_sha256 = _require_sha256(
        progress.partition_resolution["split_assignment_manifest_sha256"],
        field="split_assignment_manifest_sha256",
    )

    raw_stratum = _stratum(
        evidence.raw_mark_count,
        selected.raw_mark_bins,
        field="raw_mark_count",
    )
    successor_stratum = _stratum(
        evidence.canonical_successor_count,
        selected.canonical_successor_bins,
        field="canonical_successor_count",
    )
    alias_stratum = _stratum(
        evidence.successor_alias_count,
        selected.successor_alias_bins,
        field="successor_alias_count",
    )
    endpoint_roles = (
        ("source_endpoint", progress.endpoint_descriptor.source_endpoint_evidence_kind),
        ("target_endpoint", progress.endpoint_descriptor.target_endpoint_evidence_kind),
        (
            "pair_relationship",
            progress.endpoint_descriptor.pair_relationship_evidence_kind,
        ),
    )
    path_roles = (
        ("path", progress.path_descriptor.path_evidence_kind),
        ("intermediates", progress.path_descriptor.intermediate_evidence_kind),
        ("action_sequence", progress.path_descriptor.action_sequence_evidence_kind),
    )
    capability_cell_id = f"{selected.namespace}:{family}:{context}"
    body = {
        "schema": ASSIGNMENT_SCHEMA,
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "status": ASSIGNMENT_STATUS,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
        "trace_id": addressed.address.trace_id,
        "progress_index": index,
        "packed_shard_content_sha256": addressed.address.packed_shard_content_sha256,
        "packed_entry_index": addressed.address.entry_index,
        "candidate_ledger_row_sha256": candidate_ledger_row_sha256,
        "admitted_record_envelope_sha256": admitted_record_envelope_sha256,
        "split_assignment_manifest_sha256": split_assignment_manifest_sha256,
        "record_membership_classifier_identity_sha256": (
            record_membership_classifier_identity_sha256
        ),
        "action_sha256": current_classification.action_sha256,
        "source_state_sha256": evidence.source_state_sha256,
        "target_state_sha256": evidence.target_state_sha256,
        "source_canonical_key": canonical_state_key(source),
        "successor_canonical_key": evidence.canonical_successor_key,
        "model_family": family,
        "family_context": context,
        "capability_cell_id": capability_cell_id,
        "data_lane": progress.address.layer,
        "partition_role": progress.address.partition,
        "record_membership_cells": list(progress.record_membership_cells),
        "endpoint_evidence_roles": [list(item) for item in endpoint_roles],
        "path_evidence_roles": [list(item) for item in path_roles],
        "evidence_profile_id": evidence_profile_id,
        "semantic_groups": groups_payload,
        "endpoint_descriptor": endpoint_payload,
        "path_descriptor": path_payload,
        "relationship_group_ids": list(relationship_group_ids),
        "mappings": mappings_payload,
        "sampling_coefficients": coefficients_payload,
        "semantic_progress_evidence_sha256": semantic_progress_evidence_sha256,
        "raw_mark_count": evidence.raw_mark_count,
        "raw_mark_count_stratum": raw_stratum,
        "canonical_successor_count": evidence.canonical_successor_count,
        "canonical_successor_count_stratum": successor_stratum,
        "successor_alias_multiplicity": evidence.successor_alias_count,
        "successor_alias_multiplicity_stratum": alias_stratum,
        "matching_mark_count": evidence.matching_mark_count,
        "atom_element_transition": [list(item) for item in atom_element_transition],
        "minimum_edited_cycle_length": minimum_edited_cycle_length,
        "source_edge_aromatic": source_edge_aromatic,
        "successor_edge_aromatic": successor_edge_aromatic,
        "registry_sha256": selected.registry_sha256,
        "process_identity_sha256": selected.process_identity_sha256,
        "corpus_contract_file_sha256": selected.corpus_contract_file_sha256,
        "classifier_implementation_sha256": selected.classifier_implementation_sha256,
        "active8_policy_sha256": policy.policy_sha256,
    }
    assignment = SemanticCapabilityCellAssignment(
        trace_id=addressed.address.trace_id,
        progress_index=index,
        packed_shard_content_sha256=addressed.address.packed_shard_content_sha256,
        packed_entry_index=addressed.address.entry_index,
        candidate_ledger_row_sha256=candidate_ledger_row_sha256,
        admitted_record_envelope_sha256=admitted_record_envelope_sha256,
        split_assignment_manifest_sha256=split_assignment_manifest_sha256,
        record_membership_classifier_identity_sha256=(
            record_membership_classifier_identity_sha256
        ),
        action_sha256=current_classification.action_sha256,
        source_state_sha256=evidence.source_state_sha256,
        target_state_sha256=evidence.target_state_sha256,
        source_canonical_key=canonical_state_key(source),
        successor_canonical_key=evidence.canonical_successor_key,
        model_family=family,
        family_context=context,
        capability_cell_id=capability_cell_id,
        data_lane=progress.address.layer,
        partition_role=progress.address.partition,
        record_membership_cells=tuple(progress.record_membership_cells),
        endpoint_evidence_roles=endpoint_roles,
        path_evidence_roles=path_roles,
        evidence_profile_id=evidence_profile_id,
        semantic_groups=progress.groups,
        endpoint_descriptor=progress.endpoint_descriptor,
        path_descriptor=progress.path_descriptor,
        relationship_group_ids=relationship_group_ids,
        mappings=progress.mappings,
        sampling_coefficients=progress.sampling_coefficients,
        semantic_progress_evidence_sha256=semantic_progress_evidence_sha256,
        raw_mark_count=evidence.raw_mark_count,
        raw_mark_count_stratum=raw_stratum,
        canonical_successor_count=evidence.canonical_successor_count,
        canonical_successor_count_stratum=successor_stratum,
        successor_alias_multiplicity=evidence.successor_alias_count,
        successor_alias_multiplicity_stratum=alias_stratum,
        matching_mark_count=evidence.matching_mark_count,
        atom_element_transition=atom_element_transition,
        minimum_edited_cycle_length=minimum_edited_cycle_length,
        source_edge_aromatic=source_edge_aromatic,
        successor_edge_aromatic=successor_edge_aromatic,
        registry_sha256=selected.registry_sha256,
        process_identity_sha256=selected.process_identity_sha256,
        corpus_contract_file_sha256=selected.corpus_contract_file_sha256,
        classifier_implementation_sha256=selected.classifier_implementation_sha256,
        active8_policy_sha256=policy.policy_sha256,
        assignment_sha256=_canonical_sha256(body),
    )
    payload = assignment.as_payload()
    if assignment.assignment_sha256 != _canonical_sha256(
        {key: value for key, value in payload.items() if key != "assignment_sha256"}
    ):
        raise SemanticCapabilityCellError("capability assignment self-hash disagrees")
    return assignment


__all__ = [
    "ASSIGNMENT_SCHEMA",
    "ASSIGNMENT_SCHEMA_VERSION",
    "ASSIGNMENT_STATUS",
    "REGISTRY_RELATIVE_PATH",
    "REGISTRY_SCHEMA",
    "REGISTRY_SCHEMA_VERSION",
    "REGISTRY_STATUS",
    "STRUCTURAL_ASSIGNMENT_SCHEMA",
    "STRUCTURAL_ASSIGNMENT_SCHEMA_VERSION",
    "STRUCTURAL_ASSIGNMENT_STATUS",
    "CountBin",
    "SemanticCapabilityCellAssignment",
    "SemanticCapabilityCellError",
    "SemanticCapabilityCellRegistry",
    "SemanticStructuralCapabilityAssignment",
    "classify_accepted_semantic_progress",
    "classify_action_family_context",
    "classify_verified_structural_transition",
    "load_semantic_capability_cell_registry",
    "validate_semantic_capability_cell_registry",
]
