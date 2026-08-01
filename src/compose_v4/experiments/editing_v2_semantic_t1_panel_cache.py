"""Train-only semantic Editing-V2 T1 panel and cache-input preparation.

The bounded T1 capacity question is whether the model can fit one molecular
successor per exact source and frozen time.  Corpus repetition and mark aliases
must not change that objective.  This module therefore:

* reads only accepted ``train`` transitions from the exact semantic Active8
  decision source;
* collapses occurrences by ``(source state, time, canonical successor)``;
* gives every selected molecular target one unit coefficient;
* retains corpus teacher-mark multiplicity and production successor-alias
  multiplicity as separate audit fields; and
* emits the exact complete-trace union required for one CPU successor-fiber
  cache build, after which GPU optimization must consume cached coordinates.

Candidate-fiber size is a coverage dimension, not a new balancing cell.  A
bounded deterministic round-robin covers each observed
``(semantic cell, canonical-successor-count bin)`` before adding a second
example from any such stratum.  Raw mark count and alias multiplicity remain
separate audit fields, avoiding a sparse Cartesian product.

The artifact is preparation evidence only.  It grants no T1, P50, training,
checkpoint-selection, or final-test authority, and it contains no hazard term,
optimization threshold, model update, or P50 sampling policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_active8_decision_source import (
    EditingV2SemanticActive8DecisionIndex,
    SemanticActive8AcceptedTrace,
    SemanticActive8AcceptedTransition,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    REGISTRY_RELATIVE_PATH,
    SemanticCapabilityCellRegistry,
    SemanticStructuralCapabilityAssignment,
    classify_verified_structural_transition,
    load_semantic_capability_cell_registry,
)
from compose_v4.data.immutable_artifact import write_bytes_if_absent
from compose_v4.experiments.editing_v2_semantic_gate_zero import (
    DECISION_SCHEMA as GATE_ZERO_DECISION_SCHEMA,
    DECISION_SCHEMA_VERSION as GATE_ZERO_DECISION_SCHEMA_VERSION,
    DECISION_STATUS as GATE_ZERO_DECISION_STATUS,
    EVIDENCE_SCHEMA as GATE_ZERO_EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA_VERSION as GATE_ZERO_EVIDENCE_SCHEMA_VERSION,
    EVIDENCE_STATUS as GATE_ZERO_EVIDENCE_STATUS,
    FrozenSemanticGateZeroStructuralContract,
    _EVIDENCE_FIELDS as GATE_ZERO_EVIDENCE_FIELDS,
    load_semantic_gate_zero_structural_contract,
    structural_decision_from_evidence,
)

PANEL_SCHEMA = "compose.editing_v2.semantic_t1_panel_cache_inputs"
PANEL_SCHEMA_VERSION = 2
PANEL_STATUS = "PREPARED_NO_T1_OR_TRAINING_AUTHORITY"
REQUEST_SCHEMA = "compose.editing_v2.semantic_t1_panel_request"
REQUEST_SCHEMA_VERSION = 2
REQUEST_STATUS = "BOUNDED_REQUEST_NO_T1_OR_TRAINING_AUTHORITY"
SELECTION_RULE = "stable_hash_stratum_round_robin_v1"
UNIQUE_OBJECTIVE_UNIT = "exact_source_frozen_time_canonical_successor"
CACHE_HANDOFF = "complete_train_trace_union_for_cpu_successor_fiber_cache_v1"
GPU_CACHE_POLICY = "precompiled_successor_coordinates_required_no_gpu_chemistry"

_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/editing_v2_semantic_t1_panel_cache.py",
    "src/compose_v4/data/editing_v2_semantic_active8_decision_source.py",
    "src/compose_v4/data/editing_v2_semantic_capability_cells.py",
    "src/compose_v4/experiments/successor_fiber_cache_builder.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
)
_GATE_ZERO_DECISION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *_AUTHORITY,
    "structural_result",
    "evidence_sha256",
    "contract_sha256",
    "decision_source_inventory_sha256",
    "next_authorized_stage",
    "required_next_action",
    "decision_sha256",
}
_GATE_ZERO_COMPLETION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *_AUTHORITY,
    "structural_result",
    "contract_sha256",
    "evidence_file_sha256",
    "evidence_sha256",
    "decision_file_sha256",
    "decision_sha256",
    "completion_sha256",
}


class SemanticT1PanelError(RuntimeError):
    """Exact semantic T1 panel evidence is incomplete or inconsistent."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha(value: object, *, field_name: str) -> str:
    if not _is_sha(value):
        raise SemanticT1PanelError(f"{field_name} must be a lowercase SHA-256")
    return str(value)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_registry(
    registry: SemanticCapabilityCellRegistry | None,
    *,
    repo_root: Path | None,
) -> SemanticCapabilityCellRegistry:
    if registry is not None:
        return registry
    path = (
        None
        if repo_root is None
        else Path(repo_root).resolve() / REGISTRY_RELATIVE_PATH
    )
    return load_semantic_capability_cell_registry(path)


def semantic_t1_panel_implementation_sha256(*, repo_root: Path | None = None) -> str:
    """Hash the exact source files that define panel and cache-input meaning."""

    root = Path(repo_root or _repository_root()).resolve()
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise SemanticT1PanelError(
                f"semantic T1 implementation source is absent: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticT1PanelRequest:
    """Explicit bounded-panel request, deliberately excluding gate thresholds."""

    request_id: str
    source_revision_sha256: str
    support_time_hex: str
    maximum_entries_by_family: tuple[tuple[str, int], ...]
    request_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or self.request_id.strip() != self.request_id
        ):
            raise ValueError("semantic T1 request_id must be normalized text")
        _require_sha(self.source_revision_sha256, field_name="source_revision_sha256")
        try:
            support_time = float.fromhex(self.support_time_hex)
        except (TypeError, ValueError) as error:
            raise ValueError("support_time_hex must be a hexadecimal float") from error
        if support_time.hex() != self.support_time_hex or not 0.0 < support_time < 1.0:
            raise ValueError("semantic T1 support time must be canonical and in (0, 1)")
        limits = self.maximum_entries_by_family
        if (
            tuple(sorted(limits)) != limits
            or tuple(family for family, _ in limits) != tuple(sorted(ACTIVE8_FAMILIES))
            or any(type(limit) is not int or limit <= 0 for _, limit in limits)
        ):
            raise ValueError(
                "maximum_entries_by_family must name every Active8 family once"
            )
        expected = _sha(self.identity_body())
        if self.request_sha256:
            if self.request_sha256 != expected:
                raise ValueError("semantic T1 request self-hash disagrees")
        else:
            object.__setattr__(self, "request_sha256", expected)

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        source_revision_sha256: str,
        support_time: float,
        maximum_entries_by_family: Mapping[str, int],
    ) -> SemanticT1PanelRequest:
        if type(support_time) is not float:
            raise TypeError("semantic T1 support_time must be an explicit float")
        return cls(
            request_id=request_id,
            source_revision_sha256=source_revision_sha256,
            support_time_hex=support_time.hex(),
            maximum_entries_by_family=tuple(sorted(maximum_entries_by_family.items())),
        )

    @property
    def support_time(self) -> float:
        return float.fromhex(self.support_time_hex)

    @property
    def limit_by_family(self) -> dict[str, int]:
        return dict(self.maximum_entries_by_family)

    @property
    def selection_seed_sha256(self) -> str:
        """Bind selection to scientific inputs, excluding the descriptive ID."""

        return _sha(
            {
                "selection_rule": SELECTION_RULE,
                "source_revision_sha256": self.source_revision_sha256,
                "support_time_hex": self.support_time_hex,
                "maximum_entries_by_family": {
                    family: limit for family, limit in self.maximum_entries_by_family
                },
            }
        )

    def identity_body(self) -> dict[str, object]:
        return {
            "schema": REQUEST_SCHEMA,
            "schema_version": REQUEST_SCHEMA_VERSION,
            "status": REQUEST_STATUS,
            **_AUTHORITY,
            "request_id": self.request_id,
            "source_revision_sha256": self.source_revision_sha256,
            "support_time_hex": self.support_time_hex,
            "maximum_entries_by_family": {
                family: limit for family, limit in self.maximum_entries_by_family
            },
            "selection_rule": SELECTION_RULE,
            "selection_seed_sha256": self.selection_seed_sha256,
            "unique_objective_unit": UNIQUE_OBJECTIVE_UNIT,
            "equal_objective_coefficient": 1,
            "hazard_included": False,
            "candidate_size_is_audit_stratum_not_balancing_cell": True,
        }

    def as_payload(self) -> dict[str, object]:
        return {**self.identity_body(), "request_sha256": self.request_sha256}


@dataclass(frozen=True, slots=True)
class SemanticT1GateZeroBinding:
    """Identity-only binding to a fully revalidated structural PASS."""

    contract_sha256: str
    evidence_sha256: str
    evidence_file_sha256: str
    decision_sha256: str
    decision_file_sha256: str
    completion_sha256: str
    decision_source_inventory_sha256: str
    model_runtime_identity_sha256: str
    model_source_revision_sha256: str
    initial_model_state_sha256: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _require_sha(getattr(self, field_name), field_name=field_name)

    def as_payload(self) -> dict[str, str]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


def bind_verified_semantic_gate_zero_pass(
    decision_index: EditingV2SemanticActive8DecisionIndex,
    artifacts: Mapping[str, Any],
    *,
    decision_plan_path: Path,
    contract: FrozenSemanticGateZeroStructuralContract | None = None,
    repo_root: Path | None = None,
) -> SemanticT1GateZeroBinding:
    """Bind output returned by the public Gate-0 revalidating loader.

    This check does not turn Gate 0 into T1 authorization.  It merely prevents
    panel preparation from drifting away from the structural evidence it
    follows.
    """

    if not isinstance(decision_index, EditingV2SemanticActive8DecisionIndex):
        raise TypeError("semantic T1 requires the verified Active8 decision index")
    root = Path(repo_root or _repository_root()).resolve()
    selected_contract = contract or load_semantic_gate_zero_structural_contract(
        repo_root=root
    )
    if set(artifacts) != {"evidence", "decision", "completion"}:
        raise SemanticT1PanelError("Gate-0 artifact bundle fields disagree")
    evidence = artifacts["evidence"]
    decision = artifacts["decision"]
    completion = artifacts["completion"]
    if not all(isinstance(item, Mapping) for item in (evidence, decision, completion)):
        raise SemanticT1PanelError("Gate-0 artifact bundle must contain objects")
    if (
        set(evidence) != GATE_ZERO_EVIDENCE_FIELDS
        or set(decision) != _GATE_ZERO_DECISION_FIELDS
        or set(completion) != _GATE_ZERO_COMPLETION_FIELDS
        or evidence.get("schema") != GATE_ZERO_EVIDENCE_SCHEMA
        or evidence.get("schema_version") != GATE_ZERO_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != GATE_ZERO_EVIDENCE_STATUS
        or decision.get("schema") != GATE_ZERO_DECISION_SCHEMA
        or decision.get("schema_version") != GATE_ZERO_DECISION_SCHEMA_VERSION
        or decision.get("status") != GATE_ZERO_DECISION_STATUS
        or completion.get("schema")
        != "compose.editing.semantic_gate_zero_structural_completion"
        or completion.get("schema_version") != 1
        or completion.get("status")
        != "COMPLETE_STRUCTURAL_ARTIFACTS_NO_DOWNSTREAM_AUTHORITY"
        or evidence.get("structural_result") != "PASS"
        or decision.get("structural_result") != "PASS"
        or completion.get("structural_result") != "PASS"
        or decision.get("next_authorized_stage") is not None
        or any(
            artifact.get(field_name) is not expected
            for artifact in (evidence, decision, completion)
            for field_name, expected in _AUTHORITY.items()
        )
    ):
        raise SemanticT1PanelError(
            "semantic T1 preparation requires a non-authorizing structural PASS"
        )
    runtime = evidence.get("model_runtime_identity")
    if not isinstance(runtime, Mapping):
        raise SemanticT1PanelError("Gate-0 model-runtime identity is absent")
    expected_decision = structural_decision_from_evidence(
        evidence,
        index=decision_index,
        contract=selected_contract,
        decision_plan_path=Path(decision_plan_path),
        repo_root=root,
    )
    if dict(decision) != expected_decision:
        raise SemanticT1PanelError(
            "Gate-0 decision differs from recomputed exact structural evidence"
        )
    evidence_body = dict(evidence)
    evidence_sha256 = evidence_body.pop("evidence_sha256", None)
    decision_body = dict(decision)
    decision_sha256 = decision_body.pop("decision_sha256", None)
    completion_body = dict(completion)
    completion_sha256 = completion_body.pop("completion_sha256", None)
    evidence_bytes = _canonical_bytes(dict(evidence), newline=True)
    decision_bytes = _canonical_bytes(dict(decision), newline=True)
    if (
        evidence_sha256 != _sha(evidence_body)
        or decision_sha256 != _sha(decision_body)
        or completion_sha256 != _sha(completion_body)
        or evidence.get("decision_source_inventory_sha256")
        != decision_index.inventory_sha256
        or decision.get("decision_source_inventory_sha256")
        != decision_index.inventory_sha256
        or decision.get("evidence_sha256") != evidence_sha256
        or completion.get("evidence_sha256") != evidence_sha256
        or completion.get("decision_sha256") != decision_sha256
        or completion.get("contract_sha256") != evidence.get("contract_sha256")
        or decision.get("contract_sha256") != evidence.get("contract_sha256")
        or completion.get("evidence_file_sha256")
        != hashlib.sha256(evidence_bytes).hexdigest()
        or completion.get("decision_file_sha256")
        != hashlib.sha256(decision_bytes).hexdigest()
    ):
        raise SemanticT1PanelError("Gate-0 artifact lineage or self-hash disagrees")
    return SemanticT1GateZeroBinding(
        contract_sha256=_require_sha(
            evidence.get("contract_sha256"), field_name="contract_sha256"
        ),
        evidence_sha256=_require_sha(evidence_sha256, field_name="evidence_sha256"),
        evidence_file_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        decision_sha256=_require_sha(decision_sha256, field_name="decision_sha256"),
        decision_file_sha256=hashlib.sha256(decision_bytes).hexdigest(),
        completion_sha256=_require_sha(
            completion_sha256, field_name="completion_sha256"
        ),
        decision_source_inventory_sha256=decision_index.inventory_sha256,
        model_runtime_identity_sha256=_require_sha(
            runtime.get("identity_sha256"),
            field_name="model_runtime_identity_sha256",
        ),
        model_source_revision_sha256=_require_sha(
            runtime.get("source_revision_sha256"),
            field_name="model_source_revision_sha256",
        ),
        initial_model_state_sha256=_require_sha(
            runtime.get("initial_model_state_sha256"),
            field_name="initial_model_state_sha256",
        ),
    )


@dataclass(frozen=True, slots=True)
class SemanticT1TeacherOccurrence:
    """One exact train teacher before molecular-successor deduplication."""

    decision_source_inventory_sha256: str
    decision_sha256: str
    trace_address_sha256: str
    source_progress_address_sha256: str
    successor_progress_address_sha256: str
    trace_id: str
    packed_shard_content_sha256: str
    packed_shard_name: str
    packed_entry_index: int
    trace_source_key: str
    trace_target_key: str
    path_length: int
    progress_index: int
    data_lane: str
    partition_role: str
    action_sha256: str
    source_state_sha256: str
    target_state_sha256: str
    source_canonical_key: str
    successor_canonical_key: str
    model_family: str
    family_context: str
    capability_cell_id: str
    raw_mark_count: int
    raw_mark_count_stratum: str
    canonical_successor_count: int
    canonical_successor_count_stratum: str
    successor_alias_multiplicity: int
    successor_alias_multiplicity_stratum: str
    matching_mark_count: int
    registry_sha256: str
    process_identity_sha256: str
    corpus_contract_file_sha256: str
    classifier_implementation_sha256: str
    active8_policy_sha256: str
    assignment_sha256: str
    occurrence_sha256: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "decision_source_inventory_sha256",
            "decision_sha256",
            "trace_address_sha256",
            "source_progress_address_sha256",
            "successor_progress_address_sha256",
            "packed_shard_content_sha256",
            "action_sha256",
            "source_state_sha256",
            "target_state_sha256",
            "registry_sha256",
            "process_identity_sha256",
            "corpus_contract_file_sha256",
            "classifier_implementation_sha256",
            "active8_policy_sha256",
            "assignment_sha256",
        ):
            _require_sha(getattr(self, field_name), field_name=field_name)
        if self.partition_role != "train":
            raise ValueError("semantic T1 occurrences must be train-only")
        if self.model_family not in ACTIVE8_FAMILIES:
            raise ValueError("semantic T1 occurrence is outside Active8")
        if not self.capability_cell_id.endswith(
            f":{self.model_family}:{self.family_context}"
        ):
            raise ValueError("semantic T1 capability-cell identity disagrees")
        for field_name in (
            "packed_entry_index",
            "path_length",
            "progress_index",
            "raw_mark_count",
            "canonical_successor_count",
            "successor_alias_multiplicity",
            "matching_mark_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        if (
            self.progress_index >= self.path_length
            or self.raw_mark_count < 1
            or self.canonical_successor_count < 1
            or self.successor_alias_multiplicity < 1
            or self.matching_mark_count != 1
            or not self.trace_id
            or not self.packed_shard_name
            or Path(self.packed_shard_name).name != self.packed_shard_name
            or not self.data_lane
            or not self.trace_source_key
            or not self.trace_target_key
            or not self.source_canonical_key
            or not self.successor_canonical_key
            or self.source_canonical_key == self.successor_canonical_key
        ):
            raise ValueError("semantic T1 occurrence is internally inconsistent")
        expected = _sha(self.identity_body())
        if self.occurrence_sha256:
            if self.occurrence_sha256 != expected:
                raise ValueError("semantic T1 occurrence self-hash disagrees")
        else:
            object.__setattr__(self, "occurrence_sha256", expected)

    @classmethod
    def from_verified_transition(
        cls,
        transition: SemanticActive8AcceptedTransition,
        assignment: SemanticStructuralCapabilityAssignment,
    ) -> SemanticT1TeacherOccurrence:
        address = transition.addressed_trace.address
        if assignment.partition_role != "train" or address.partition != "train":
            raise SemanticT1PanelError("held-out transition reached T1 preparation")
        return cls(
            decision_source_inventory_sha256=(
                assignment.decision_source_inventory_sha256
            ),
            decision_sha256=assignment.decision_sha256,
            trace_address_sha256=assignment.trace_address_sha256,
            source_progress_address_sha256=(assignment.source_progress_address_sha256),
            successor_progress_address_sha256=(
                assignment.successor_progress_address_sha256
            ),
            trace_id=assignment.trace_id,
            packed_shard_content_sha256=assignment.packed_shard_content_sha256,
            packed_shard_name=assignment.packed_shard_name,
            packed_entry_index=assignment.packed_entry_index,
            trace_source_key=address.source_key,
            trace_target_key=address.target_key,
            path_length=address.path_length,
            progress_index=assignment.progress_index,
            data_lane=assignment.data_lane,
            partition_role=assignment.partition_role,
            action_sha256=assignment.action_sha256,
            source_state_sha256=assignment.source_state_sha256,
            target_state_sha256=assignment.target_state_sha256,
            source_canonical_key=assignment.source_canonical_key,
            successor_canonical_key=assignment.successor_canonical_key,
            model_family=assignment.model_family,
            family_context=assignment.family_context,
            capability_cell_id=assignment.capability_cell_id,
            raw_mark_count=assignment.raw_mark_count,
            raw_mark_count_stratum=assignment.raw_mark_count_stratum,
            canonical_successor_count=assignment.canonical_successor_count,
            canonical_successor_count_stratum=(
                assignment.canonical_successor_count_stratum
            ),
            successor_alias_multiplicity=assignment.successor_alias_multiplicity,
            successor_alias_multiplicity_stratum=(
                assignment.successor_alias_multiplicity_stratum
            ),
            matching_mark_count=assignment.matching_mark_count,
            registry_sha256=assignment.registry_sha256,
            process_identity_sha256=assignment.process_identity_sha256,
            corpus_contract_file_sha256=assignment.corpus_contract_file_sha256,
            classifier_implementation_sha256=(
                assignment.classifier_implementation_sha256
            ),
            active8_policy_sha256=assignment.active8_policy_sha256,
            assignment_sha256=assignment.assignment_sha256,
        )

    def identity_body(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
            if field_name != "occurrence_sha256"
        }


def iter_train_semantic_t1_occurrences(
    decision_index: EditingV2SemanticActive8DecisionIndex,
    *,
    registry: SemanticCapabilityCellRegistry,
) -> Iterable[SemanticT1TeacherOccurrence]:
    """Stream classified teachers without opening held-out transition states."""

    if not isinstance(decision_index, EditingV2SemanticActive8DecisionIndex):
        raise TypeError("semantic T1 requires the verified Active8 decision index")
    if not isinstance(registry, SemanticCapabilityCellRegistry):
        raise TypeError("semantic T1 requires the verified capability registry")
    for trace in decision_index.iter_accepted_traces_for_partition("train"):
        for transition in decision_index.accepted_transitions_for(trace):
            assignment = classify_verified_structural_transition(
                decision_index,
                transition,
                registry=registry,
            )
            yield SemanticT1TeacherOccurrence.from_verified_transition(
                transition,
                assignment,
            )


@dataclass(frozen=True, slots=True, order=True)
class SemanticT1CoverageStratum:
    """Observed audit stratum; only family and context define the semantic cell."""

    model_family: str
    capability_cell_id: str
    canonical_successor_count_stratum: str

    def __post_init__(self) -> None:
        if self.model_family not in ACTIVE8_FAMILIES:
            raise ValueError("semantic T1 coverage stratum is outside Active8")
        if not self.capability_cell_id:
            raise ValueError("semantic T1 coverage stratum lacks a semantic cell")

    @classmethod
    def from_occurrence(
        cls, occurrence: SemanticT1TeacherOccurrence
    ) -> SemanticT1CoverageStratum:
        return cls(
            model_family=occurrence.model_family,
            capability_cell_id=occurrence.capability_cell_id,
            canonical_successor_count_stratum=(
                occurrence.canonical_successor_count_stratum
            ),
        )

    @property
    def stratum_sha256(self) -> str:
        return _sha(self.as_payload())

    def as_payload(self) -> dict[str, str]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


def _group_identity(
    occurrence: SemanticT1TeacherOccurrence,
    *,
    support_time_hex: str,
) -> dict[str, str]:
    return {
        "source_state_sha256": occurrence.source_state_sha256,
        "support_time_hex": support_time_hex,
        "canonical_successor_key": occurrence.successor_canonical_key,
    }


@dataclass(slots=True)
class _CandidateAggregate:
    group_sha256: str
    selection_rank_sha256: str
    stratum: SemanticT1CoverageStratum
    representative: SemanticT1TeacherOccurrence
    occurrence_count: int = 0
    teacher_action_counts: Counter[str] = field(default_factory=Counter)
    target_state_counts: Counter[str] = field(default_factory=Counter)
    data_lane_counts: Counter[str] = field(default_factory=Counter)

    @classmethod
    def create(
        cls,
        occurrence: SemanticT1TeacherOccurrence,
        *,
        request: SemanticT1PanelRequest,
    ) -> _CandidateAggregate:
        group_sha256 = _sha(
            _group_identity(occurrence, support_time_hex=request.support_time_hex)
        )
        selected = cls(
            group_sha256=group_sha256,
            selection_rank_sha256=_sha(
                {
                    "selection_rule": SELECTION_RULE,
                    "selection_seed_sha256": request.selection_seed_sha256,
                    "group_sha256": group_sha256,
                }
            ),
            stratum=SemanticT1CoverageStratum.from_occurrence(occurrence),
            representative=occurrence,
        )
        selected.add(occurrence, request=request)
        return selected

    def add(
        self,
        occurrence: SemanticT1TeacherOccurrence,
        *,
        request: SemanticT1PanelRequest,
    ) -> None:
        if (
            _sha(_group_identity(occurrence, support_time_hex=request.support_time_hex))
            != self.group_sha256
        ):
            raise SemanticT1PanelError("semantic T1 candidate group identity drifted")
        expected = self.representative
        invariant_fields = (
            "decision_source_inventory_sha256",
            "source_state_sha256",
            "source_canonical_key",
            "successor_canonical_key",
            "model_family",
            "family_context",
            "capability_cell_id",
            "raw_mark_count",
            "raw_mark_count_stratum",
            "canonical_successor_count",
            "canonical_successor_count_stratum",
            "successor_alias_multiplicity",
            "successor_alias_multiplicity_stratum",
            "matching_mark_count",
            "registry_sha256",
            "process_identity_sha256",
            "corpus_contract_file_sha256",
            "classifier_implementation_sha256",
            "active8_policy_sha256",
        )
        if (
            any(
                getattr(occurrence, field_name) != getattr(expected, field_name)
                for field_name in invariant_fields
            )
            or SemanticT1CoverageStratum.from_occurrence(occurrence) != self.stratum
        ):
            raise SemanticT1PanelError(
                "one molecular successor has inconsistent family or fiber evidence"
            )
        self.occurrence_count += 1
        self.teacher_action_counts[occurrence.action_sha256] += 1
        self.target_state_counts[occurrence.target_state_sha256] += 1
        self.data_lane_counts[occurrence.data_lane] += 1
        if occurrence.occurrence_sha256 < self.representative.occurrence_sha256:
            self.representative = occurrence


@dataclass(frozen=True, slots=True)
class SemanticT1PanelEntry:
    """One unit-weight molecular target plus separate mark/alias evidence."""

    panel_entry_sha256: str
    group_sha256: str
    selection_rank_sha256: str
    support_time_hex: str
    model_family: str
    family_context: str
    capability_cell_id: str
    coverage_stratum: SemanticT1CoverageStratum
    source_state_sha256: str
    source_canonical_key: str
    successor_canonical_key: str
    representative_target_state_sha256: str
    representative_action_sha256: str
    representative_occurrence_sha256: str
    packed_shard_content_sha256: str
    packed_shard_name: str
    packed_entry_index: int
    trace_id: str
    representative_data_lane: str
    trace_source_key: str
    trace_target_key: str
    path_length: int
    progress_index: int
    trace_address_sha256: str
    decision_sha256: str
    source_progress_address_sha256: str
    successor_progress_address_sha256: str
    objective_coefficient: int
    teacher_occurrence_count: int
    teacher_action_occurrence_counts: tuple[tuple[str, int], ...]
    target_state_occurrence_counts: tuple[tuple[str, int], ...]
    data_lane_occurrence_counts: tuple[tuple[str, int], ...]
    raw_mark_count: int
    canonical_successor_count: int
    production_successor_alias_multiplicity: int
    teacher_action_identity_count: int

    @classmethod
    def from_aggregate(cls, aggregate: _CandidateAggregate) -> SemanticT1PanelEntry:
        occurrence = aggregate.representative
        action_counts = tuple(sorted(aggregate.teacher_action_counts.items()))
        target_counts = tuple(sorted(aggregate.target_state_counts.items()))
        lane_counts = tuple(sorted(aggregate.data_lane_counts.items()))
        body: dict[str, object] = {
            "group_sha256": aggregate.group_sha256,
            "selection_rank_sha256": aggregate.selection_rank_sha256,
            "support_time_hex": "",
            "model_family": occurrence.model_family,
            "family_context": occurrence.family_context,
            "capability_cell_id": occurrence.capability_cell_id,
            "coverage_stratum": aggregate.stratum.as_payload(),
            "source_state_sha256": occurrence.source_state_sha256,
            "source_canonical_key": occurrence.source_canonical_key,
            "successor_canonical_key": occurrence.successor_canonical_key,
            "representative_target_state_sha256": occurrence.target_state_sha256,
            "representative_action_sha256": occurrence.action_sha256,
            "representative_occurrence_sha256": occurrence.occurrence_sha256,
            "packed_shard_content_sha256": occurrence.packed_shard_content_sha256,
            "packed_shard_name": occurrence.packed_shard_name,
            "packed_entry_index": occurrence.packed_entry_index,
            "trace_id": occurrence.trace_id,
            "representative_data_lane": occurrence.data_lane,
            "trace_source_key": occurrence.trace_source_key,
            "trace_target_key": occurrence.trace_target_key,
            "path_length": occurrence.path_length,
            "progress_index": occurrence.progress_index,
            "trace_address_sha256": occurrence.trace_address_sha256,
            "decision_sha256": occurrence.decision_sha256,
            "source_progress_address_sha256": (
                occurrence.source_progress_address_sha256
            ),
            "successor_progress_address_sha256": (
                occurrence.successor_progress_address_sha256
            ),
            "objective_coefficient": 1,
            "teacher_occurrence_count": aggregate.occurrence_count,
            "teacher_action_occurrence_counts": [list(item) for item in action_counts],
            "target_state_occurrence_counts": [list(item) for item in target_counts],
            "data_lane_occurrence_counts": [list(item) for item in lane_counts],
            "raw_mark_count": occurrence.raw_mark_count,
            "canonical_successor_count": occurrence.canonical_successor_count,
            "production_successor_alias_multiplicity": (
                occurrence.successor_alias_multiplicity
            ),
            "teacher_action_identity_count": len(action_counts),
        }
        # The caller fills the one frozen time before sealing the entry.
        return cls(
            panel_entry_sha256="",
            support_time_hex="",
            coverage_stratum=aggregate.stratum,
            teacher_action_occurrence_counts=action_counts,
            target_state_occurrence_counts=target_counts,
            data_lane_occurrence_counts=lane_counts,
            **{
                key: value
                for key, value in body.items()
                if key
                not in {
                    "coverage_stratum",
                    "teacher_action_occurrence_counts",
                    "target_state_occurrence_counts",
                    "data_lane_occurrence_counts",
                    "support_time_hex",
                }
            },
        )

    def with_support_time(self, support_time_hex: str) -> SemanticT1PanelEntry:
        values = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
            if field_name != "panel_entry_sha256"
        }
        values["support_time_hex"] = support_time_hex
        body = _panel_entry_body(values)
        return SemanticT1PanelEntry(
            panel_entry_sha256=_sha(body),
            **values,
        )

    def __post_init__(self) -> None:
        for field_name in (
            "group_sha256",
            "selection_rank_sha256",
            "source_state_sha256",
            "representative_target_state_sha256",
            "representative_action_sha256",
            "representative_occurrence_sha256",
            "packed_shard_content_sha256",
            "trace_address_sha256",
            "decision_sha256",
            "source_progress_address_sha256",
            "successor_progress_address_sha256",
        ):
            _require_sha(getattr(self, field_name), field_name=field_name)
        if self.panel_entry_sha256 and not _is_sha(self.panel_entry_sha256):
            raise ValueError("panel_entry_sha256 must be empty or a SHA-256")
        if self.objective_coefficient != 1:
            raise ValueError("every semantic T1 molecular target must have unit weight")
        if self.model_family != self.coverage_stratum.model_family:
            raise ValueError("panel entry family and coverage stratum disagree")
        if self.capability_cell_id != self.coverage_stratum.capability_cell_id:
            raise ValueError("panel entry cell and coverage stratum disagree")
        for counts in (
            self.teacher_action_occurrence_counts,
            self.target_state_occurrence_counts,
            self.data_lane_occurrence_counts,
        ):
            if (
                tuple(sorted(counts)) != counts
                or len(dict(counts)) != len(counts)
                or any(
                    not key or type(count) is not int or count <= 0
                    for key, count in counts
                )
            ):
                raise ValueError("semantic T1 occurrence counts are invalid")
            if sum(count for _, count in counts) != self.teacher_occurrence_count:
                raise ValueError("semantic T1 occurrence count totals disagree")
        if (
            self.teacher_action_identity_count
            != len(self.teacher_action_occurrence_counts)
            or self.teacher_occurrence_count < 1
            or self.production_successor_alias_multiplicity < 1
            or self.teacher_action_identity_count
            > self.production_successor_alias_multiplicity
            or self.progress_index >= self.path_length
            or self.model_family not in ACTIVE8_FAMILIES
            or not self.representative_data_lane
            or not self.trace_source_key
            or not self.trace_target_key
            or self.source_canonical_key == self.successor_canonical_key
            or self.representative_action_sha256
            not in dict(self.teacher_action_occurrence_counts)
            or self.representative_target_state_sha256
            not in dict(self.target_state_occurrence_counts)
            or self.representative_data_lane
            not in dict(self.data_lane_occurrence_counts)
        ):
            raise ValueError("semantic T1 panel entry multiplicity is inconsistent")
        if self.support_time_hex:
            try:
                value = float.fromhex(self.support_time_hex)
            except ValueError as error:
                raise ValueError("panel support time is invalid") from error
            if value.hex() != self.support_time_hex or not 0.0 < value < 1.0:
                raise ValueError("panel support time is outside (0, 1)")
            if self.group_sha256 != _sha(
                {
                    "source_state_sha256": self.source_state_sha256,
                    "support_time_hex": self.support_time_hex,
                    "canonical_successor_key": self.successor_canonical_key,
                }
            ):
                raise ValueError("semantic T1 panel group identity disagrees")
            expected = _sha(_panel_entry_body(self))
            if self.panel_entry_sha256 != expected:
                raise ValueError("semantic T1 panel entry self-hash disagrees")
        elif self.panel_entry_sha256:
            raise ValueError("unsealed panel entry cannot carry a self-hash")

    def as_payload(self) -> dict[str, object]:
        if not self.panel_entry_sha256:
            raise SemanticT1PanelError("semantic T1 panel entry is not sealed")
        return {
            **_panel_entry_body(self),
            "panel_entry_sha256": self.panel_entry_sha256,
        }


def _panel_entry_body(
    value: SemanticT1PanelEntry | Mapping[str, Any],
) -> dict[str, object]:
    get = (
        (lambda name: getattr(value, name))
        if isinstance(value, SemanticT1PanelEntry)
        else (lambda name: value[name])
    )
    stratum = get("coverage_stratum")
    return {
        field_name: (
            stratum.as_payload()
            if field_name == "coverage_stratum"
            else (
                [list(item) for item in get(field_name)]
                if field_name
                in {
                    "teacher_action_occurrence_counts",
                    "target_state_occurrence_counts",
                    "data_lane_occurrence_counts",
                }
                else get(field_name)
            )
        )
        for field_name in SemanticT1PanelEntry.__dataclass_fields__
        if field_name != "panel_entry_sha256"
    }


@dataclass(frozen=True, slots=True)
class SemanticT1CacheTraceInput:
    """One exact accepted train trace to compile once on CPU."""

    packed_shard_content_sha256: str
    packed_shard_name: str
    packed_entry_index: int
    trace_id: str
    data_lane: str
    trace_source_key: str
    trace_target_key: str
    path_length: int
    trace_address_sha256: str
    decision_sha256: str
    selected_progress_indices: tuple[int, ...]
    panel_entry_sha256s: tuple[str, ...]
    cache_input_sha256: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "packed_shard_content_sha256",
            "trace_address_sha256",
            "decision_sha256",
        ):
            _require_sha(getattr(self, field_name), field_name=field_name)
        if (
            not self.trace_id
            or not self.data_lane
            or not self.trace_source_key
            or not self.trace_target_key
            or not self.packed_shard_name
            or Path(self.packed_shard_name).name != self.packed_shard_name
            or type(self.packed_entry_index) is not int
            or self.packed_entry_index < 0
            or type(self.path_length) is not int
            or self.path_length < 1
            or tuple(sorted(set(self.selected_progress_indices)))
            != self.selected_progress_indices
            or any(
                type(index) is not int or not 0 <= index < self.path_length
                for index in self.selected_progress_indices
            )
            or len(set(self.panel_entry_sha256s)) != len(self.panel_entry_sha256s)
            or len(self.selected_progress_indices) != len(self.panel_entry_sha256s)
            or any(not _is_sha(value) for value in self.panel_entry_sha256s)
        ):
            raise ValueError("semantic T1 cache trace input is invalid")
        expected = _sha(self.identity_body())
        if self.cache_input_sha256:
            if self.cache_input_sha256 != expected:
                raise ValueError("semantic T1 cache input self-hash disagrees")
        else:
            object.__setattr__(self, "cache_input_sha256", expected)

    def identity_body(self) -> dict[str, object]:
        return {
            "packed_shard_content_sha256": self.packed_shard_content_sha256,
            "packed_shard_name": self.packed_shard_name,
            "packed_entry_index": self.packed_entry_index,
            "trace_id": self.trace_id,
            "data_lane": self.data_lane,
            "partition_role": "train",
            "trace_source_key": self.trace_source_key,
            "trace_target_key": self.trace_target_key,
            "path_length": self.path_length,
            "trace_address_sha256": self.trace_address_sha256,
            "decision_sha256": self.decision_sha256,
            "selected_progress_indices": list(self.selected_progress_indices),
            "panel_entry_sha256s": list(self.panel_entry_sha256s),
        }

    def as_payload(self) -> dict[str, object]:
        return {**self.identity_body(), "cache_input_sha256": self.cache_input_sha256}


@dataclass(frozen=True, slots=True)
class SemanticT1CoverageReceipt:
    stratum: SemanticT1CoverageStratum
    train_teacher_occurrence_count: int
    retained_candidate_count: int
    selected_candidate_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "train_teacher_occurrence_count",
            "retained_candidate_count",
            "selected_candidate_count",
        ):
            if (
                type(getattr(self, field_name)) is not int
                or getattr(self, field_name) < 0
            ):
                raise ValueError("semantic T1 coverage counts must be nonnegative")
        if (
            self.train_teacher_occurrence_count < 1
            or self.retained_candidate_count < 1
            or not 0 <= self.selected_candidate_count <= self.retained_candidate_count
        ):
            raise ValueError("semantic T1 coverage receipt is inconsistent")

    def as_payload(self) -> dict[str, object]:
        return {
            "stratum": self.stratum.as_payload(),
            "stratum_sha256": self.stratum.stratum_sha256,
            "train_teacher_occurrence_count": self.train_teacher_occurrence_count,
            "retained_candidate_count": self.retained_candidate_count,
            "selected_candidate_count": self.selected_candidate_count,
        }


@dataclass(frozen=True, slots=True)
class SemanticT1PanelArtifact:
    """Complete non-authorizing panel plus one-time CPU cache inputs."""

    request: SemanticT1PanelRequest
    gate_zero_binding: SemanticT1GateZeroBinding
    decision_source_identity: tuple[tuple[str, object], ...]
    registry_sha256: str
    process_identity_sha256: str
    corpus_contract_file_sha256: str
    classifier_implementation_sha256: str
    panel_implementation_sha256: str
    entries: tuple[SemanticT1PanelEntry, ...]
    cache_trace_inputs: tuple[SemanticT1CacheTraceInput, ...]
    coverage_receipts: tuple[SemanticT1CoverageReceipt, ...]
    first_pass_train_teacher_count: int
    second_pass_train_teacher_count: int
    first_pass_train_teacher_stream_sha256: str
    second_pass_train_teacher_stream_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "registry_sha256",
            "process_identity_sha256",
            "corpus_contract_file_sha256",
            "classifier_implementation_sha256",
            "panel_implementation_sha256",
            "first_pass_train_teacher_stream_sha256",
            "second_pass_train_teacher_stream_sha256",
        ):
            _require_sha(getattr(self, field_name), field_name=field_name)
        if (
            self.gate_zero_binding.decision_source_inventory_sha256
            != dict(self.decision_source_identity).get("inventory_sha256")
            or tuple(sorted(self.decision_source_identity))
            != self.decision_source_identity
            or len(dict(self.decision_source_identity))
            != len(self.decision_source_identity)
        ):
            raise ValueError("semantic T1 decision-source identity disagrees")
        if (
            type(self.first_pass_train_teacher_count) is not int
            or self.first_pass_train_teacher_count <= 0
            or self.first_pass_train_teacher_count
            != self.second_pass_train_teacher_count
            or self.first_pass_train_teacher_stream_sha256
            != self.second_pass_train_teacher_stream_sha256
        ):
            raise ValueError("semantic T1 source passes are incomplete or unequal")
        entry_ids = tuple(entry.panel_entry_sha256 for entry in self.entries)
        if (
            tuple(
                sorted(
                    self.entries,
                    key=lambda item: (item.model_family, item.panel_entry_sha256),
                )
            )
            != self.entries
            or len(set(entry_ids)) != len(entry_ids)
            or not self.entries
        ):
            raise ValueError("semantic T1 panel entries are not canonical and unique")
        unique_units = {
            (
                entry.source_state_sha256,
                entry.support_time_hex,
                entry.successor_canonical_key,
            )
            for entry in self.entries
        }
        if len(unique_units) != len(self.entries):
            raise ValueError("semantic T1 panel repeats a molecular objective unit")
        if any(
            entry.support_time_hex != self.request.support_time_hex
            or entry.selection_rank_sha256
            != _sha(
                {
                    "selection_rule": SELECTION_RULE,
                    "selection_seed_sha256": self.request.selection_seed_sha256,
                    "group_sha256": entry.group_sha256,
                }
            )
            for entry in self.entries
        ):
            raise ValueError("semantic T1 entry time or selection rank disagrees")
        if (
            tuple(
                sorted(
                    self.cache_trace_inputs,
                    key=lambda item: (
                        item.packed_shard_content_sha256,
                        item.packed_entry_index,
                        item.trace_id,
                    ),
                )
            )
            != self.cache_trace_inputs
        ):
            raise ValueError("semantic T1 cache inputs are not canonical")
        cache_trace_keys = tuple(
            (
                item.packed_shard_content_sha256,
                item.packed_entry_index,
                item.trace_id,
            )
            for item in self.cache_trace_inputs
        )
        if len(set(cache_trace_keys)) != len(cache_trace_keys):
            raise ValueError("semantic T1 cache trace keys are not unique")
        cached_entries = tuple(
            sorted(
                panel_entry
                for item in self.cache_trace_inputs
                for panel_entry in item.panel_entry_sha256s
            )
        )
        if cached_entries != tuple(sorted(entry_ids)):
            raise ValueError(
                "semantic T1 cache inputs do not cover each panel entry once"
            )
        if (
            tuple(sorted(self.coverage_receipts, key=lambda item: item.stratum))
            != self.coverage_receipts
        ):
            raise ValueError("semantic T1 coverage receipts are not canonical")
        receipt_strata = tuple(item.stratum for item in self.coverage_receipts)
        if len(set(receipt_strata)) != len(receipt_strata):
            raise ValueError("semantic T1 coverage receipt strata are not unique")
        selected_by_stratum = Counter(entry.coverage_stratum for entry in self.entries)
        if {
            receipt.stratum: receipt.selected_candidate_count
            for receipt in self.coverage_receipts
        } != dict(selected_by_stratum):
            raise ValueError("semantic T1 coverage receipts disagree with the panel")
        family_counts = Counter(entry.model_family for entry in self.entries)
        if set(family_counts) != set(ACTIVE8_FAMILIES) or any(
            family_counts[family] > self.request.limit_by_family[family]
            for family in ACTIVE8_FAMILIES
        ):
            raise ValueError("semantic T1 panel family coverage or bound disagrees")
        expected = _sha(self.identity_body())
        if self.artifact_sha256:
            if self.artifact_sha256 != expected:
                raise ValueError("semantic T1 panel artifact self-hash disagrees")
        else:
            object.__setattr__(self, "artifact_sha256", expected)

    def identity_body(self) -> dict[str, object]:
        family_counts = Counter(entry.model_family for entry in self.entries)
        return {
            "schema": PANEL_SCHEMA,
            "schema_version": PANEL_SCHEMA_VERSION,
            "status": PANEL_STATUS,
            **_AUTHORITY,
            "request": self.request.as_payload(),
            "gate_zero_binding": self.gate_zero_binding.as_payload(),
            "decision_source_identity": dict(self.decision_source_identity),
            "registry_sha256": self.registry_sha256,
            "process_identity_sha256": self.process_identity_sha256,
            "corpus_contract_file_sha256": self.corpus_contract_file_sha256,
            "classifier_implementation_sha256": self.classifier_implementation_sha256,
            "panel_implementation_sha256": self.panel_implementation_sha256,
            "objective": {
                "unit": UNIQUE_OBJECTIVE_UNIT,
                "coefficient_per_entry": 1,
                "canonical_successor_aliases_aggregated": True,
                "teacher_mark_multiplicity_is_audit_only": True,
                "hazard_included": False,
            },
            "selection": {
                "rule": SELECTION_RULE,
                "semantic_cell_dimensions": ["model_family", "family_context"],
                "audit_coverage_dimensions": [
                    "canonical_successor_count_stratum",
                ],
                "separate_nonbalancing_audit_dimensions": [
                    "raw_mark_count_stratum",
                    "successor_alias_multiplicity_stratum",
                ],
            },
            "cache_handoff": {
                "kind": CACHE_HANDOFF,
                "compile_device_role": "cpu_once",
                "gpu_policy": GPU_CACHE_POLICY,
                "complete_trace_rows_required": True,
                "exact_persistent_slot_states_reopened_from_bound_source": True,
                "cache_contains_model_scores_or_probabilities": False,
                "hazard_coordinates_included": False,
            },
            "panel_entry_inventory_sha256": _sha(
                [entry.panel_entry_sha256 for entry in self.entries]
            ),
            "cache_input_inventory_sha256": _sha(
                [item.cache_input_sha256 for item in self.cache_trace_inputs]
            ),
            "counts": {
                "first_pass_train_teacher_count": self.first_pass_train_teacher_count,
                "second_pass_train_teacher_count": self.second_pass_train_teacher_count,
                "panel_entry_count": len(self.entries),
                "cache_trace_count": len(self.cache_trace_inputs),
                "coverage_stratum_count": len(self.coverage_receipts),
                "panel_entries_by_family": {
                    family: family_counts[family] for family in sorted(ACTIVE8_FAMILIES)
                },
            },
            "train_teacher_source_passes": {
                "first_pass_stream_sha256": self.first_pass_train_teacher_stream_sha256,
                "second_pass_stream_sha256": self.second_pass_train_teacher_stream_sha256,
                "ordered_source_stream_required": True,
            },
            "entries": [entry.as_payload() for entry in self.entries],
            "cache_trace_inputs": [
                item.as_payload() for item in self.cache_trace_inputs
            ],
            "coverage_receipts": [item.as_payload() for item in self.coverage_receipts],
        }

    def as_payload(self) -> dict[str, object]:
        return {**self.identity_body(), "artifact_sha256": self.artifact_sha256}


def _reservoir_pass(
    occurrence_factory: Callable[[], Iterable[SemanticT1TeacherOccurrence]],
    *,
    request: SemanticT1PanelRequest,
    expected_decision_source_sha256: str,
) -> tuple[
    dict[SemanticT1CoverageStratum, dict[str, _CandidateAggregate]],
    Counter[SemanticT1CoverageStratum],
    int,
    str,
]:
    buckets: dict[SemanticT1CoverageStratum, dict[str, _CandidateAggregate]] = (
        defaultdict(dict)
    )
    occurrence_counts: Counter[SemanticT1CoverageStratum] = Counter()
    total = 0
    stream = hashlib.sha256()
    limits = request.limit_by_family
    for occurrence in occurrence_factory():
        if not isinstance(occurrence, SemanticT1TeacherOccurrence):
            raise TypeError("semantic T1 occurrence factory returned another type")
        if (
            occurrence.decision_source_inventory_sha256
            != expected_decision_source_sha256
        ):
            raise SemanticT1PanelError("semantic T1 occurrence source identity drifted")
        total += 1
        stream.update(occurrence.occurrence_sha256.encode("ascii"))
        stream.update(b"\n")
        stratum = SemanticT1CoverageStratum.from_occurrence(occurrence)
        occurrence_counts[stratum] += 1
        bucket = buckets[stratum]
        group_sha256 = _sha(
            _group_identity(occurrence, support_time_hex=request.support_time_hex)
        )
        retained = bucket.get(group_sha256)
        if retained is not None:
            retained.add(occurrence, request=request)
            continue
        candidate = _CandidateAggregate.create(occurrence, request=request)
        limit = limits[occurrence.model_family]
        if len(bucket) < limit:
            bucket[group_sha256] = candidate
            continue
        worst = max(
            bucket.values(),
            key=lambda item: (item.selection_rank_sha256, item.group_sha256),
        )
        if (candidate.selection_rank_sha256, candidate.group_sha256) < (
            worst.selection_rank_sha256,
            worst.group_sha256,
        ):
            del bucket[worst.group_sha256]
            bucket[group_sha256] = candidate
    if total <= 0:
        raise SemanticT1PanelError("semantic T1 source yielded no train teachers")
    return dict(buckets), occurrence_counts, total, stream.hexdigest()


def _selected_group_ids(
    buckets: Mapping[SemanticT1CoverageStratum, Mapping[str, _CandidateAggregate]],
    *,
    request: SemanticT1PanelRequest,
) -> tuple[str, ...]:
    selected: list[str] = []
    for family in sorted(ACTIVE8_FAMILIES):
        strata = tuple(
            sorted(stratum for stratum in buckets if stratum.model_family == family)
        )
        if not strata:
            raise SemanticT1PanelError(
                f"semantic T1 source has no train stratum for {family}"
            )
        limit = request.limit_by_family[family]
        if limit < len(strata):
            raise SemanticT1PanelError(
                f"semantic T1 bound for {family} cannot cover all observed strata"
            )
        ordered = {
            stratum: tuple(
                sorted(
                    buckets[stratum].values(),
                    key=lambda item: (
                        item.selection_rank_sha256,
                        item.group_sha256,
                    ),
                )
            )
            for stratum in strata
        }
        depth = 0
        family_selected: list[str] = []
        while len(family_selected) < limit:
            added = False
            for stratum in strata:
                candidates = ordered[stratum]
                if depth < len(candidates):
                    family_selected.append(candidates[depth].group_sha256)
                    added = True
                    if len(family_selected) == limit:
                        break
            if not added:
                break
            depth += 1
        selected.extend(family_selected)
    if len(selected) != len(set(selected)):
        raise SemanticT1PanelError(
            "one canonical molecular objective unit was selected in multiple families"
        )
    return tuple(selected)


def _exact_selected_pass(
    occurrence_factory: Callable[[], Iterable[SemanticT1TeacherOccurrence]],
    *,
    request: SemanticT1PanelRequest,
    selected_group_ids: tuple[str, ...],
    expected_decision_source_sha256: str,
) -> tuple[dict[str, _CandidateAggregate], int, str]:
    selected_set = set(selected_group_ids)
    aggregates: dict[str, _CandidateAggregate] = {}
    total = 0
    stream = hashlib.sha256()
    for occurrence in occurrence_factory():
        if not isinstance(occurrence, SemanticT1TeacherOccurrence):
            raise TypeError("semantic T1 occurrence factory returned another type")
        if (
            occurrence.decision_source_inventory_sha256
            != expected_decision_source_sha256
        ):
            raise SemanticT1PanelError("semantic T1 occurrence source identity drifted")
        total += 1
        stream.update(occurrence.occurrence_sha256.encode("ascii"))
        stream.update(b"\n")
        group_sha256 = _sha(
            _group_identity(occurrence, support_time_hex=request.support_time_hex)
        )
        if group_sha256 not in selected_set:
            continue
        aggregate = aggregates.get(group_sha256)
        if aggregate is None:
            aggregates[group_sha256] = _CandidateAggregate.create(
                occurrence, request=request
            )
        else:
            aggregate.add(occurrence, request=request)
    if set(aggregates) != selected_set:
        raise SemanticT1PanelError(
            "semantic T1 second pass did not reproduce every selected target"
        )
    return aggregates, total, stream.hexdigest()


def _cache_trace_inputs(
    entries: tuple[SemanticT1PanelEntry, ...],
) -> tuple[SemanticT1CacheTraceInput, ...]:
    grouped: dict[
        tuple[str, str, int, str, str, str, str, int, str, str],
        list[SemanticT1PanelEntry],
    ] = defaultdict(list)
    for entry in entries:
        grouped[
            (
                entry.packed_shard_content_sha256,
                entry.packed_shard_name,
                entry.packed_entry_index,
                entry.trace_id,
                entry.representative_data_lane,
                entry.trace_source_key,
                entry.trace_target_key,
                entry.path_length,
                entry.trace_address_sha256,
                entry.decision_sha256,
            )
        ].append(entry)
    rows: list[SemanticT1CacheTraceInput] = []
    for key, selected in sorted(grouped.items()):
        by_progress: dict[int, str] = {}
        for entry in selected:
            previous = by_progress.setdefault(
                entry.progress_index, entry.panel_entry_sha256
            )
            if previous != entry.panel_entry_sha256:
                raise SemanticT1PanelError(
                    "one trace progress maps to multiple selected molecular targets"
                )
        progress_and_entries = tuple(sorted(by_progress.items()))
        rows.append(
            SemanticT1CacheTraceInput(
                packed_shard_content_sha256=key[0],
                packed_shard_name=key[1],
                packed_entry_index=key[2],
                trace_id=key[3],
                data_lane=key[4],
                trace_source_key=key[5],
                trace_target_key=key[6],
                path_length=key[7],
                trace_address_sha256=key[8],
                decision_sha256=key[9],
                selected_progress_indices=tuple(
                    progress for progress, _ in progress_and_entries
                ),
                panel_entry_sha256s=tuple(
                    panel_entry for _, panel_entry in progress_and_entries
                ),
            )
        )
    return tuple(rows)


def build_semantic_t1_panel_from_occurrence_factory(
    occurrence_factory: Callable[[], Iterable[SemanticT1TeacherOccurrence]],
    *,
    request: SemanticT1PanelRequest,
    gate_zero_binding: SemanticT1GateZeroBinding,
    decision_source_identity: Mapping[str, object],
    registry: SemanticCapabilityCellRegistry,
    repo_root: Path | None = None,
) -> SemanticT1PanelArtifact:
    """Build a deterministic panel through two independently counted passes.

    The factory must reopen the same exact source on each call.  The second pass
    recomputes every selected group's multiplicity and catches a selected
    canonical successor that changes family, context, or candidate evidence.
    """

    if not callable(occurrence_factory):
        raise TypeError("semantic T1 occurrence_factory must be callable")
    if not isinstance(request, SemanticT1PanelRequest):
        raise TypeError("semantic T1 request has another type")
    if not isinstance(gate_zero_binding, SemanticT1GateZeroBinding):
        raise TypeError("semantic T1 Gate-0 binding has another type")
    if not isinstance(registry, SemanticCapabilityCellRegistry):
        raise TypeError("semantic T1 registry has another type")
    source_identity = dict(decision_source_identity)
    if source_identity.get("inventory_sha256") != (
        gate_zero_binding.decision_source_inventory_sha256
    ) or source_identity.get("process_identity_sha256") != (
        registry.process_identity_sha256
    ):
        raise SemanticT1PanelError(
            "semantic T1 decision-source identity differs from Gate 0"
        )
    if request.source_revision_sha256 != gate_zero_binding.model_source_revision_sha256:
        raise SemanticT1PanelError(
            "semantic T1 request source revision differs from Gate-0 initialization"
        )

    def verified_occurrence_factory() -> Iterable[SemanticT1TeacherOccurrence]:
        for occurrence in occurrence_factory():
            if not isinstance(occurrence, SemanticT1TeacherOccurrence):
                raise TypeError("semantic T1 occurrence factory returned another type")
            expected_cell = f"{registry.namespace}:{occurrence.model_family}:{occurrence.family_context}"
            if (
                occurrence.family_context
                not in registry.contexts_by_family.get(occurrence.model_family, ())
                or occurrence.capability_cell_id != expected_cell
                or occurrence.registry_sha256 != registry.registry_sha256
                or occurrence.process_identity_sha256
                != registry.process_identity_sha256
                or occurrence.corpus_contract_file_sha256
                != registry.corpus_contract_file_sha256
                or occurrence.classifier_implementation_sha256
                != registry.classifier_implementation_sha256
                or occurrence.active8_policy_sha256
                != source_identity.get("policy_sha256")
            ):
                raise SemanticT1PanelError(
                    "semantic T1 occurrence capability-registry identity drifted"
                )
            yield occurrence

    buckets, occurrence_counts, first_total, first_stream_sha256 = _reservoir_pass(
        verified_occurrence_factory,
        request=request,
        expected_decision_source_sha256=(
            gate_zero_binding.decision_source_inventory_sha256
        ),
    )
    selected_ids = _selected_group_ids(buckets, request=request)
    exact, second_total, second_stream_sha256 = _exact_selected_pass(
        verified_occurrence_factory,
        request=request,
        selected_group_ids=selected_ids,
        expected_decision_source_sha256=(
            gate_zero_binding.decision_source_inventory_sha256
        ),
    )
    if first_total != second_total or first_stream_sha256 != second_stream_sha256:
        raise SemanticT1PanelError(
            "semantic T1 source changed between deterministic preparation passes"
        )
    entries = tuple(
        sorted(
            (
                SemanticT1PanelEntry.from_aggregate(
                    exact[group_sha256]
                ).with_support_time(request.support_time_hex)
                for group_sha256 in selected_ids
            ),
            key=lambda item: (item.model_family, item.panel_entry_sha256),
        )
    )
    selected_counts = Counter(entry.coverage_stratum for entry in entries)
    coverage = tuple(
        SemanticT1CoverageReceipt(
            stratum=stratum,
            train_teacher_occurrence_count=occurrence_counts[stratum],
            retained_candidate_count=len(buckets[stratum]),
            selected_candidate_count=selected_counts[stratum],
        )
        for stratum in sorted(buckets)
    )
    return SemanticT1PanelArtifact(
        request=request,
        gate_zero_binding=gate_zero_binding,
        decision_source_identity=tuple(sorted(source_identity.items())),
        registry_sha256=registry.registry_sha256,
        process_identity_sha256=registry.process_identity_sha256,
        corpus_contract_file_sha256=registry.corpus_contract_file_sha256,
        classifier_implementation_sha256=registry.classifier_implementation_sha256,
        panel_implementation_sha256=semantic_t1_panel_implementation_sha256(
            repo_root=repo_root
        ),
        entries=entries,
        cache_trace_inputs=_cache_trace_inputs(entries),
        coverage_receipts=coverage,
        first_pass_train_teacher_count=first_total,
        second_pass_train_teacher_count=second_total,
        first_pass_train_teacher_stream_sha256=first_stream_sha256,
        second_pass_train_teacher_stream_sha256=second_stream_sha256,
    )


def prepare_editing_v2_semantic_t1_panel(
    decision_index: EditingV2SemanticActive8DecisionIndex,
    *,
    gate_zero_artifacts: Mapping[str, Any],
    decision_plan_path: Path,
    request: SemanticT1PanelRequest,
    registry: SemanticCapabilityCellRegistry | None = None,
    gate_zero_contract: FrozenSemanticGateZeroStructuralContract | None = None,
    repo_root: Path | None = None,
) -> SemanticT1PanelArtifact:
    """Prepare the exact train-only panel from the production decision source."""

    selected_registry = _load_registry(registry, repo_root=repo_root)
    binding = bind_verified_semantic_gate_zero_pass(
        decision_index,
        gate_zero_artifacts,
        decision_plan_path=decision_plan_path,
        contract=gate_zero_contract,
        repo_root=repo_root,
    )

    def occurrence_factory() -> Iterable[SemanticT1TeacherOccurrence]:
        return iter_train_semantic_t1_occurrences(
            decision_index,
            registry=selected_registry,
        )

    return build_semantic_t1_panel_from_occurrence_factory(
        occurrence_factory,
        request=request,
        gate_zero_binding=binding,
        decision_source_identity=decision_index.identity_payload(),
        registry=selected_registry,
        repo_root=repo_root,
    )


def resolve_semantic_t1_cache_trace_inputs(
    decision_index: EditingV2SemanticActive8DecisionIndex,
    artifact: SemanticT1PanelArtifact,
    *,
    registry: SemanticCapabilityCellRegistry | None = None,
    repo_root: Path | None = None,
) -> tuple[SemanticActive8AcceptedTrace, ...]:
    """Reopen and revalidate the exact complete traces for CPU compilation.

    This resolver performs no successor enumeration.  Its output is the exact
    bounded trace union a later CPU cache builder must compile once.  GPU work
    must consume the resulting immutable fiber cache rather than call this
    resolver or any chemistry enumerator.
    """

    if not isinstance(decision_index, EditingV2SemanticActive8DecisionIndex):
        raise TypeError("semantic T1 cache resolution requires the decision index")
    if not isinstance(artifact, SemanticT1PanelArtifact):
        raise TypeError("semantic T1 cache resolution requires the typed artifact")
    selected_registry = _load_registry(registry, repo_root=repo_root)
    if (
        decision_index.identity_payload() != dict(artifact.decision_source_identity)
        or artifact.registry_sha256 != selected_registry.registry_sha256
        or artifact.process_identity_sha256 != selected_registry.process_identity_sha256
        or artifact.corpus_contract_file_sha256
        != selected_registry.corpus_contract_file_sha256
        or artifact.classifier_implementation_sha256
        != selected_registry.classifier_implementation_sha256
        or artifact.panel_implementation_sha256
        != semantic_t1_panel_implementation_sha256(repo_root=repo_root)
    ):
        raise SemanticT1PanelError(
            "semantic T1 cache resolution upstream identity drifted"
        )
    entry_by_id = {entry.panel_entry_sha256: entry for entry in artifact.entries}
    wanted = {
        (
            item.packed_shard_content_sha256,
            item.packed_entry_index,
            item.trace_id,
        ): item
        for item in artifact.cache_trace_inputs
    }
    if len(wanted) != len(artifact.cache_trace_inputs):
        raise SemanticT1PanelError("semantic T1 cache trace keys are not unique")
    resolved: dict[tuple[str, int, str], SemanticActive8AcceptedTrace] = {}
    for trace in decision_index.iter_accepted_traces_for_partition("train"):
        address = trace.addressed_trace.address
        key = (
            address.packed_shard_content_sha256,
            address.entry_index,
            address.trace_id,
        )
        cache_input = wanted.get(key)
        if cache_input is None:
            continue
        if (
            address.partition != "train"
            or address.packed_shard_name != cache_input.packed_shard_name
            or address.layer != cache_input.data_lane
            or address.source_key != cache_input.trace_source_key
            or address.target_key != cache_input.trace_target_key
            or address.path_length != cache_input.path_length
            or trace.decision_sha256 != cache_input.decision_sha256
        ):
            raise SemanticT1PanelError(
                "semantic T1 cache trace envelope differs from the selected input"
            )
        transitions = {
            transition.step_index: transition
            for transition in decision_index.accepted_transitions_for(trace)
        }
        for progress_index, panel_entry_sha256 in zip(
            cache_input.selected_progress_indices,
            cache_input.panel_entry_sha256s,
            strict=True,
        ):
            entry = entry_by_id[panel_entry_sha256]
            transition = transitions.get(progress_index)
            if transition is None:
                raise SemanticT1PanelError(
                    "semantic T1 cache input selects an absent progress transition"
                )
            assignment = classify_verified_structural_transition(
                decision_index,
                transition,
                registry=selected_registry,
            )
            if (
                assignment.trace_address_sha256 != cache_input.trace_address_sha256
                or assignment.decision_sha256 != cache_input.decision_sha256
                or assignment.progress_index != entry.progress_index
                or assignment.source_progress_address_sha256
                != entry.source_progress_address_sha256
                or assignment.successor_progress_address_sha256
                != entry.successor_progress_address_sha256
                or assignment.action_sha256 != entry.representative_action_sha256
                or assignment.source_state_sha256 != entry.source_state_sha256
                or assignment.target_state_sha256
                != entry.representative_target_state_sha256
                or assignment.source_canonical_key != entry.source_canonical_key
                or assignment.successor_canonical_key != entry.successor_canonical_key
                or assignment.model_family != entry.model_family
                or assignment.family_context != entry.family_context
                or assignment.capability_cell_id != entry.capability_cell_id
                or assignment.raw_mark_count != entry.raw_mark_count
                or assignment.canonical_successor_count
                != entry.canonical_successor_count
                or assignment.successor_alias_multiplicity
                != entry.production_successor_alias_multiplicity
            ):
                raise SemanticT1PanelError(
                    "semantic T1 selected transition differs from panel evidence"
                )
        resolved[key] = trace
    if set(resolved) != set(wanted):
        raise SemanticT1PanelError(
            "semantic T1 cache trace union is incomplete in the exact source"
        )
    return tuple(resolved[key] for key in sorted(resolved))


def serialize_semantic_t1_panel(artifact: SemanticT1PanelArtifact) -> bytes:
    """Serialize canonical bytes after rechecking the typed artifact."""

    if not isinstance(artifact, SemanticT1PanelArtifact):
        raise TypeError("semantic T1 artifact has another type")
    if artifact.artifact_sha256 != _sha(artifact.identity_body()):
        raise SemanticT1PanelError(
            "semantic T1 artifact changed after construction or has a stale self-hash"
        )
    return _canonical_bytes(artifact.as_payload(), newline=True)


def _parse_count_pairs(
    value: object, *, field_name: str
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise SemanticT1PanelError(f"{field_name} must be a list")
    rows: list[tuple[str, int]] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or type(item[1]) is not int
            or item[1] <= 0
        ):
            raise SemanticT1PanelError(f"{field_name}[{index}] is invalid")
        rows.append((item[0], item[1]))
    return tuple(rows)


def _parse_stratum(value: object, *, field_name: str) -> SemanticT1CoverageStratum:
    if not isinstance(value, dict) or set(value) != set(
        SemanticT1CoverageStratum.__dataclass_fields__
    ):
        raise SemanticT1PanelError(f"{field_name} fields disagree")
    try:
        return SemanticT1CoverageStratum(**value)
    except (TypeError, ValueError) as error:
        raise SemanticT1PanelError(f"{field_name} is invalid") from error


def deserialize_semantic_t1_panel(
    content: bytes,
    *,
    expected_artifact_sha256: str | None = None,
    expected_panel_implementation_sha256: str | None = None,
) -> SemanticT1PanelArtifact:
    """Reconstruct typed cache inputs and reject any semantic reinterpretation."""

    if not isinstance(content, bytes):
        raise TypeError("semantic T1 serialized content must be bytes")
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticT1PanelError("semantic T1 artifact is not valid JSON") from error
    if not isinstance(payload, dict) or content != _canonical_bytes(
        payload, newline=True
    ):
        raise SemanticT1PanelError("semantic T1 artifact is not canonical JSON")
    if (
        payload.get("schema") != PANEL_SCHEMA
        or payload.get("schema_version") != PANEL_SCHEMA_VERSION
        or payload.get("status") != PANEL_STATUS
        or any(
            payload.get(field_name) is not expected
            for field_name, expected in _AUTHORITY.items()
        )
    ):
        raise SemanticT1PanelError("semantic T1 schema or authority disagrees")
    request_payload = payload.get("request")
    if not isinstance(request_payload, dict):
        raise SemanticT1PanelError("semantic T1 request is absent")
    limits = request_payload.get("maximum_entries_by_family")
    if not isinstance(limits, dict):
        raise SemanticT1PanelError("semantic T1 family limits are invalid")
    try:
        request = SemanticT1PanelRequest(
            request_id=request_payload["request_id"],
            source_revision_sha256=request_payload["source_revision_sha256"],
            support_time_hex=request_payload["support_time_hex"],
            maximum_entries_by_family=tuple(sorted(limits.items())),
            request_sha256=request_payload["request_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SemanticT1PanelError("semantic T1 request is invalid") from error
    if request.as_payload() != request_payload:
        raise SemanticT1PanelError("semantic T1 request policy disagrees")

    binding_payload = payload.get("gate_zero_binding")
    if not isinstance(binding_payload, dict) or set(binding_payload) != set(
        SemanticT1GateZeroBinding.__dataclass_fields__
    ):
        raise SemanticT1PanelError("semantic T1 Gate-0 binding fields disagree")
    try:
        binding = SemanticT1GateZeroBinding(**binding_payload)
    except (TypeError, ValueError, SemanticT1PanelError) as error:
        raise SemanticT1PanelError("semantic T1 Gate-0 binding is invalid") from error

    entry_payloads = payload.get("entries")
    if not isinstance(entry_payloads, list):
        raise SemanticT1PanelError("semantic T1 entries must be a list")
    entries: list[SemanticT1PanelEntry] = []
    entry_fields = set(SemanticT1PanelEntry.__dataclass_fields__)
    for index, item in enumerate(entry_payloads):
        if not isinstance(item, dict) or set(item) != entry_fields:
            raise SemanticT1PanelError(f"semantic T1 entry {index} fields disagree")
        values = dict(item)
        values["coverage_stratum"] = _parse_stratum(
            values["coverage_stratum"], field_name=f"entries[{index}].coverage_stratum"
        )
        for field_name in (
            "teacher_action_occurrence_counts",
            "target_state_occurrence_counts",
            "data_lane_occurrence_counts",
        ):
            values[field_name] = _parse_count_pairs(
                values[field_name], field_name=f"entries[{index}].{field_name}"
            )
        try:
            entries.append(SemanticT1PanelEntry(**values))
        except (TypeError, ValueError, SemanticT1PanelError) as error:
            raise SemanticT1PanelError(
                f"semantic T1 entry {index} is invalid"
            ) from error

    cache_payloads = payload.get("cache_trace_inputs")
    if not isinstance(cache_payloads, list):
        raise SemanticT1PanelError("semantic T1 cache inputs must be a list")
    cache_inputs: list[SemanticT1CacheTraceInput] = []
    cache_fields = set(SemanticT1CacheTraceInput.__dataclass_fields__) | {
        "partition_role"
    }
    for index, item in enumerate(cache_payloads):
        if (
            not isinstance(item, dict)
            or set(item) != cache_fields
            or item.get("partition_role") != "train"
        ):
            raise SemanticT1PanelError(
                f"semantic T1 cache input {index} fields or role disagree"
            )
        values = {key: value for key, value in item.items() if key != "partition_role"}
        for field_name in ("selected_progress_indices", "panel_entry_sha256s"):
            if not isinstance(values[field_name], list):
                raise SemanticT1PanelError(
                    f"semantic T1 cache input {index} {field_name} is invalid"
                )
            values[field_name] = tuple(values[field_name])
        try:
            cache_inputs.append(SemanticT1CacheTraceInput(**values))
        except (TypeError, ValueError, SemanticT1PanelError) as error:
            raise SemanticT1PanelError(
                f"semantic T1 cache input {index} is invalid"
            ) from error

    coverage_payloads = payload.get("coverage_receipts")
    if not isinstance(coverage_payloads, list):
        raise SemanticT1PanelError("semantic T1 coverage receipts must be a list")
    coverage: list[SemanticT1CoverageReceipt] = []
    expected_coverage_fields = {
        "stratum",
        "stratum_sha256",
        "train_teacher_occurrence_count",
        "retained_candidate_count",
        "selected_candidate_count",
    }
    for index, item in enumerate(coverage_payloads):
        if not isinstance(item, dict) or set(item) != expected_coverage_fields:
            raise SemanticT1PanelError(
                f"semantic T1 coverage receipt {index} fields disagree"
            )
        stratum = _parse_stratum(
            item["stratum"], field_name=f"coverage_receipts[{index}].stratum"
        )
        if item["stratum_sha256"] != stratum.stratum_sha256:
            raise SemanticT1PanelError(
                f"semantic T1 coverage receipt {index} stratum hash disagrees"
            )
        try:
            coverage.append(
                SemanticT1CoverageReceipt(
                    stratum=stratum,
                    train_teacher_occurrence_count=item[
                        "train_teacher_occurrence_count"
                    ],
                    retained_candidate_count=item["retained_candidate_count"],
                    selected_candidate_count=item["selected_candidate_count"],
                )
            )
        except (TypeError, ValueError) as error:
            raise SemanticT1PanelError(
                f"semantic T1 coverage receipt {index} is invalid"
            ) from error

    source_identity = payload.get("decision_source_identity")
    counts = payload.get("counts")
    source_passes = payload.get("train_teacher_source_passes")
    if (
        not isinstance(source_identity, dict)
        or not isinstance(counts, dict)
        or not isinstance(source_passes, dict)
        or set(source_passes)
        != {
            "first_pass_stream_sha256",
            "second_pass_stream_sha256",
            "ordered_source_stream_required",
        }
        or source_passes.get("ordered_source_stream_required") is not True
    ):
        raise SemanticT1PanelError("semantic T1 source identity or counts are invalid")
    try:
        artifact = SemanticT1PanelArtifact(
            request=request,
            gate_zero_binding=binding,
            decision_source_identity=tuple(sorted(source_identity.items())),
            registry_sha256=payload["registry_sha256"],
            process_identity_sha256=payload["process_identity_sha256"],
            corpus_contract_file_sha256=payload["corpus_contract_file_sha256"],
            classifier_implementation_sha256=payload[
                "classifier_implementation_sha256"
            ],
            panel_implementation_sha256=payload["panel_implementation_sha256"],
            entries=tuple(entries),
            cache_trace_inputs=tuple(cache_inputs),
            coverage_receipts=tuple(coverage),
            first_pass_train_teacher_count=counts["first_pass_train_teacher_count"],
            second_pass_train_teacher_count=counts["second_pass_train_teacher_count"],
            first_pass_train_teacher_stream_sha256=source_passes[
                "first_pass_stream_sha256"
            ],
            second_pass_train_teacher_stream_sha256=source_passes[
                "second_pass_stream_sha256"
            ],
            artifact_sha256=payload["artifact_sha256"],
        )
    except (KeyError, TypeError, ValueError, SemanticT1PanelError) as error:
        raise SemanticT1PanelError("semantic T1 artifact body is invalid") from error
    if artifact.as_payload() != payload:
        raise SemanticT1PanelError(
            "semantic T1 artifact contains unknown or recomputed-field drift"
        )
    if (
        expected_artifact_sha256 is not None
        and artifact.artifact_sha256
        != _require_sha(expected_artifact_sha256, field_name="expected_artifact_sha256")
    ):
        raise SemanticT1PanelError("semantic T1 expected artifact hash disagrees")
    if (
        expected_panel_implementation_sha256 is not None
        and artifact.panel_implementation_sha256
        != _require_sha(
            expected_panel_implementation_sha256,
            field_name="expected_panel_implementation_sha256",
        )
    ):
        raise SemanticT1PanelError("semantic T1 expected implementation hash disagrees")
    return artifact


def read_semantic_t1_panel(
    path: Path,
    *,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    expected_file_bytes: int,
    expected_panel_implementation_sha256: str,
) -> SemanticT1PanelArtifact:
    """Read exact bytes with physical, semantic, and implementation bindings."""

    source = Path(path)
    try:
        content = source.read_bytes()
    except OSError as error:
        raise SemanticT1PanelError(
            f"semantic T1 panel artifact is absent: {source}"
        ) from error
    if type(expected_file_bytes) is not int or expected_file_bytes <= 0:
        raise ValueError("expected_file_bytes must be a positive integer")
    if len(content) != expected_file_bytes:
        raise SemanticT1PanelError("semantic T1 panel byte count disagrees")
    if hashlib.sha256(content).hexdigest() != _require_sha(
        expected_file_sha256, field_name="expected_file_sha256"
    ):
        raise SemanticT1PanelError("semantic T1 panel physical SHA-256 disagrees")
    current_implementation_sha256 = semantic_t1_panel_implementation_sha256()
    if (
        _require_sha(
            expected_panel_implementation_sha256,
            field_name="expected_panel_implementation_sha256",
        )
        != current_implementation_sha256
    ):
        raise SemanticT1PanelError(
            "semantic T1 expected implementation is not the current implementation"
        )
    return deserialize_semantic_t1_panel(
        content,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_panel_implementation_sha256=current_implementation_sha256,
    )


def write_semantic_t1_panel(
    path: Path,
    artifact: SemanticT1PanelArtifact,
) -> bool:
    """Immutably publish panel/cache inputs, allowing byte-identical reuse."""

    return write_bytes_if_absent(path, serialize_semantic_t1_panel(artifact))


__all__ = [
    "CACHE_HANDOFF",
    "GPU_CACHE_POLICY",
    "PANEL_SCHEMA",
    "PANEL_SCHEMA_VERSION",
    "PANEL_STATUS",
    "SELECTION_RULE",
    "SemanticT1CacheTraceInput",
    "SemanticT1CoverageReceipt",
    "SemanticT1CoverageStratum",
    "SemanticT1GateZeroBinding",
    "SemanticT1PanelArtifact",
    "SemanticT1PanelEntry",
    "SemanticT1PanelError",
    "SemanticT1PanelRequest",
    "SemanticT1TeacherOccurrence",
    "bind_verified_semantic_gate_zero_pass",
    "build_semantic_t1_panel_from_occurrence_factory",
    "deserialize_semantic_t1_panel",
    "iter_train_semantic_t1_occurrences",
    "prepare_editing_v2_semantic_t1_panel",
    "read_semantic_t1_panel",
    "resolve_semantic_t1_cache_trace_inputs",
    "semantic_t1_panel_implementation_sha256",
    "serialize_semantic_t1_panel",
    "write_semantic_t1_panel",
]
