"""Process-V2-only Active8 policy and exact candidate evaluation.

The historical semantic Active8 module is a frozen V1 artifact whose own file
bytes participate in its runtime policy identity.  Process V2 therefore owns a
separate adapter rather than widening the V1 policy builder.  This module
reuses the production successor kernel, ActionCodecV4, rewrite executor, and
the stable decision value types; it changes only the process identity and the
atom-delete candidate semantics that the model must implement.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

import torch

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES, DISABLED_FAMILIES
from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256
from compose_v4.data.packed_trace_store import AddressedPackedTrace
from compose_v4.experiments.production_successor_kernel import (
    ProductionSuccessorKernelError,
    SuccessorKernelResult,
    _TABLE_FAMILIES,
    _coordinate_action,
    canonical_successor_result,
)
from compose_v4.experiments.factorized_mark_conditional import (
    operator_capability_batch_kwargs,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedTraceletRateModel,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.action_codec_v4 import ActionCodecV4Error
from compose_v4.rewrite.editing_v2_process_identity import (
    EditingV2ProcessIdentityError,
    editing_process_v2_identity,
)
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace import RewriteStep

POLICY_SCHEMA = "compose.data.editing_v2_process_v2_active8_admission_policy"
POLICY_SCHEMA_VERSION = 2
POLICY_STATUS = "FROZEN_PROCESS_V2_PRIMITIVE_POLICY_NO_ACTIVE8_OR_TRAINING_AUTHORITY"
SEMANTIC_ACTIVE8_FAMILIES = tuple(ACTIVE8_FAMILIES)
SEMANTIC_ACTIVE8_EXECUTOR_RULES = tuple(action_codec_v4.ACTIVE8_EXECUTOR_RULES)
SEMANTIC_ACTIVE8_RULE_TO_FAMILY = tuple(
    (rule, action_codec_v4.canonical_family(rule))
    for rule in SEMANTIC_ACTIVE8_EXECUTOR_RULES
)
EXPLICITLY_DISABLED_RULES = tuple(DISABLED_FAMILIES)

_POLICY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "training_authorized",
        "active8_authorized",
        "selection_unit",
        "active_families",
        "executor_rule_to_family",
        "explicitly_disabled_rules",
        "maximum_existing_neighbors_for_atom_insert",
        "all_actions_require_action_v4",
        "all_teachers_require_exact_mark_membership",
        "all_teachers_require_exact_persistent_successor",
        "all_teachers_require_canonical_successor_membership",
        "excluded_trace_emits_progress_rows",
        "process_identity_sha256",
        "action_codec_schema_version",
        "action_codec_implementation_hash",
        "required_model_modes",
        "candidate_evaluator",
        "implementation_file_sha256",
        "policy_sha256",
    }
)

_REQUIRED_MODEL_MODES: tuple[tuple[str, object], ...] = (
    ("compute_ring_grow_support", False),
    ("compute_ring_restates", True),
    ("compute_cyclic_graft", True),
    ("compute_ring_opening", True),
    ("compute_ring_system_delete", False),
    ("editing_process_semantics", PROCESS_V2_EDITING_PROCESS_SEMANTICS),
    ("atom_restate_action_semantics", SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS),
    ("ring_restate_scorer_mode", SEMANTIC_RING_RESTATE_SCORER_MODE),
    ("cycle_close_action_semantics", SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS),
    ("cycle_open_action_semantics", SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS),
    ("atom_delete_action_semantics", PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS),
    ("enable_cycle_ops", True),
    ("enable_ring_grow_macro", False),
    ("enable_ring_system_delete", False),
    ("use_aromatic_bond_view", True),
)

_CANDIDATE_EVALUATOR = (
    "compose_v4.data.editing_v2_process_v2_active8_admission."
    "ProductionProcessV2BatchedTeacherSupportChecker"
)


class SemanticActive8AdmissionError(RuntimeError):
    """The Process-V2 policy, model, or exact checker contract is invalid."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class SemanticActive8AdmissionPolicy:
    """Self-identifying, non-authorizing Process-V2 Active8 policy."""

    schema: str
    schema_version: int
    status: str
    training_authorized: bool
    active8_authorized: bool
    selection_unit: str
    active_families: tuple[str, ...]
    executor_rule_to_family: tuple[tuple[str, str], ...]
    explicitly_disabled_rules: tuple[str, ...]
    maximum_existing_neighbors_for_atom_insert: int
    all_actions_require_action_v4: bool
    all_teachers_require_exact_mark_membership: bool
    all_teachers_require_exact_persistent_successor: bool
    all_teachers_require_canonical_successor_membership: bool
    excluded_trace_emits_progress_rows: bool
    process_identity_sha256: str
    action_codec_schema_version: int
    action_codec_implementation_hash: str
    required_model_modes: tuple[tuple[str, object], ...]
    candidate_evaluator: str
    implementation_file_sha256: str
    policy_sha256: str

    def as_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible policy payload."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "status": self.status,
            "training_authorized": self.training_authorized,
            "active8_authorized": self.active8_authorized,
            "selection_unit": self.selection_unit,
            "active_families": list(self.active_families),
            "executor_rule_to_family": [
                [rule, family] for rule, family in self.executor_rule_to_family
            ],
            "explicitly_disabled_rules": list(self.explicitly_disabled_rules),
            "maximum_existing_neighbors_for_atom_insert": (
                self.maximum_existing_neighbors_for_atom_insert
            ),
            "all_actions_require_action_v4": self.all_actions_require_action_v4,
            "all_teachers_require_exact_mark_membership": (
                self.all_teachers_require_exact_mark_membership
            ),
            "all_teachers_require_exact_persistent_successor": (
                self.all_teachers_require_exact_persistent_successor
            ),
            "all_teachers_require_canonical_successor_membership": (
                self.all_teachers_require_canonical_successor_membership
            ),
            "excluded_trace_emits_progress_rows": self.excluded_trace_emits_progress_rows,
            "process_identity_sha256": self.process_identity_sha256,
            "action_codec_schema_version": self.action_codec_schema_version,
            "action_codec_implementation_hash": self.action_codec_implementation_hash,
            "required_model_modes": [
                [name, value] for name, value in self.required_model_modes
            ],
            "candidate_evaluator": self.candidate_evaluator,
            "implementation_file_sha256": self.implementation_file_sha256,
            "policy_sha256": self.policy_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticActionClassification:
    """Action-V4 classification before Process-V2 candidate evaluation."""

    step_index: int
    executor_rule: str
    model_family: str | None
    action_sha256: str | None
    policy_eligible: bool
    exclusion_reason: str | None

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("classification step_index must be nonnegative")
        if not self.executor_rule:
            raise ValueError("classification executor_rule must be nonempty")
        if self.policy_eligible:
            if (
                self.model_family is None
                or not _is_sha256(self.action_sha256)
                or self.exclusion_reason is not None
            ):
                raise ValueError("eligible action classification evidence is incomplete")
        elif not self.exclusion_reason or self.action_sha256 is not None:
            raise ValueError("excluded action classification evidence is inconsistent")


@dataclass(frozen=True, slots=True)
class SemanticExactCandidateEvidence:
    """Exact marked-law and quotient evidence for one Process-V2 teacher."""

    supported: bool
    action_sha256: str
    source_state_sha256: str
    target_state_sha256: str
    source_canonical_key: str
    canonical_successor_key: str
    raw_mark_count: int
    canonical_successor_count: int
    matching_mark_count: int
    successor_alias_count: int
    exclusion_reason: str | None

    def __post_init__(self) -> None:
        for field in ("action_sha256", "source_state_sha256", "target_state_sha256"):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if not self.source_canonical_key or not self.canonical_successor_key:
            raise ValueError("teacher-support canonical keys must be nonempty")
        for field in (
            "raw_mark_count",
            "canonical_successor_count",
            "matching_mark_count",
            "successor_alias_count",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be nonnegative")
        if self.supported:
            if (
                self.exclusion_reason is not None
                or self.matching_mark_count == 0
                or self.successor_alias_count == 0
            ):
                raise ValueError("supported candidate evidence is incomplete")
        elif not self.exclusion_reason:
            raise ValueError("unsupported candidate evidence requires an exclusion reason")


@dataclass(frozen=True, slots=True)
class ProcessV2SemanticExactCandidateAudit:
    """Frozen evidence beside the exact persistent-successor mark count."""

    evidence: SemanticExactCandidateEvidence
    exact_successor_mark_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, SemanticExactCandidateEvidence):
            raise ValueError("candidate audit requires exact candidate evidence")
        if type(self.exact_successor_mark_count) is not int:
            raise ValueError("exact_successor_mark_count must be an integer")
        if not 0 <= self.exact_successor_mark_count <= self.evidence.matching_mark_count:
            raise ValueError(
                "exact_successor_mark_count must lie within the matching marks it counts"
            )
        if self.evidence.supported and self.exact_successor_mark_count == 0:
            raise ValueError("a supported teacher reproduces at least one exact successor")


@dataclass(frozen=True, slots=True)
class ProcessV2TeacherSupportEvidence:
    """Exhaustive per-teacher evidence without constructing its whole quotient.

    Active8 admission needs to establish that the exact teacher coordinate is
    in the Process-V2 model support and that executing it produces the stored,
    productive persistent-slot successor.  The number of *other* legal marks,
    canonical successors and aliases describes fiber geometry; it is audited
    by the bounded release sentinel and compiled for T1/P50 panels instead of
    being recomputed for every corpus transition.
    """

    supported: bool
    exclusion_reason: str | None
    action_sha256: str
    source_state_sha256: str
    target_state_sha256: str
    source_canonical_key: str
    canonical_successor_key: str
    teacher_coordinate_legal: bool
    teacher_executes_to_exact_successor: bool
    productive_canonical_successor: bool
    raw_mark_count: int
    matching_mark_count: int
    exact_successor_mark_count: int

    def __post_init__(self) -> None:
        for field in ("action_sha256", "source_state_sha256", "target_state_sha256"):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if not self.source_canonical_key or not self.canonical_successor_key:
            raise ValueError("teacher-support canonical keys must be nonempty")
        flags = (
            self.teacher_coordinate_legal,
            self.teacher_executes_to_exact_successor,
            self.productive_canonical_successor,
        )
        if any(type(value) is not bool for value in flags):
            raise ValueError("teacher-support flags must be booleans")
        for field in (
            "raw_mark_count",
            "matching_mark_count",
            "exact_successor_mark_count",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be nonnegative")
        if not (
            self.exact_successor_mark_count
            <= self.matching_mark_count
            <= self.raw_mark_count
        ):
            raise ValueError("teacher-support mark counts do not aggregate")
        if self.supported:
            if (
                self.exclusion_reason is not None
                or not all(flags)
                or self.matching_mark_count != 1
                or self.exact_successor_mark_count != 1
            ):
                raise ValueError("supported teacher evidence is incomplete")
        elif not self.exclusion_reason:
            raise ValueError("unsupported teacher evidence requires an exclusion reason")


@dataclass(frozen=True, slots=True)
class SemanticActive8Exclusion:
    """One Process-V2 Active8 exclusion."""

    stage: Literal["action_v4_policy", "production_candidate_support"]
    step_index: int
    executor_rule: str
    model_family: str | None
    reason: str

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0 or not self.reason:
            raise ValueError("Active8 exclusion is incomplete")


@dataclass(frozen=True, slots=True)
class SemanticActive8ActionDecision:
    """Static and dynamic evidence for one Process-V2 action."""

    classification: SemanticActionClassification
    candidate_evidence: (
        SemanticExactCandidateEvidence | ProcessV2TeacherSupportEvidence | None
    )

    def __post_init__(self) -> None:
        if not self.classification.policy_eligible and self.candidate_evidence is not None:
            raise ValueError("policy-excluded actions cannot carry candidate evidence")


@dataclass(frozen=True, slots=True)
class SemanticActive8TraceDecision:
    """Whole-trace Process-V2 Active8 result."""

    trace_id: str
    policy_sha256: str
    semantic_migration_status: Literal["admitted"]
    semantic_migration_rejection: None
    active8_status: Literal["accepted", "excluded"]
    emits_progress_rows: bool
    action_decisions: tuple[SemanticActive8ActionDecision, ...]
    active8_exclusions: tuple[SemanticActive8Exclusion, ...]

    def __post_init__(self) -> None:
        if not self.trace_id or not _is_sha256(self.policy_sha256):
            raise ValueError("trace decision identity is incomplete")
        accepted = self.active8_status == "accepted"
        if (
            self.semantic_migration_status != "admitted"
            or self.semantic_migration_rejection is not None
            or self.active8_status not in {"accepted", "excluded"}
            or accepted != (not self.active8_exclusions)
            or self.emits_progress_rows != accepted
        ):
            raise ValueError("whole-trace Active8 decision is inconsistent")


class ProcessV2SemanticExactCandidateChecker(Protocol):
    """Check one complete Process-V2 teacher against production mark support."""

    def __call__(
        self,
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> SemanticExactCandidateEvidence | ProcessV2TeacherSupportEvidence: ...


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _action_sha256(rule_name: str, action: Any) -> tuple[str, str]:
    record = action_codec_v4.encode_action(rule_name, action)
    decoded_rule, decoded_action = action_codec_v4.decode_action(record)
    if decoded_rule != rule_name or decoded_action != action:
        raise SemanticActive8AdmissionError(
            "ActionCodecV4 round trip changed the Process-V2 teacher action"
        )
    return canonical_sha256(record), str(record["model_family"])


@lru_cache(maxsize=1)
def build_process_v2_semantic_active8_admission_policy() -> SemanticActive8AdmissionPolicy:
    """Build the non-authorizing Process-V2 Active8 policy."""

    try:
        process_identity = editing_process_v2_identity()
    except EditingV2ProcessIdentityError as error:
        raise SemanticActive8AdmissionError(
            "cannot establish the Editing Process-V2 identity"
        ) from error
    body: dict[str, object] = {
        "schema": POLICY_SCHEMA,
        "schema_version": POLICY_SCHEMA_VERSION,
        "status": POLICY_STATUS,
        "training_authorized": False,
        "active8_authorized": False,
        "selection_unit": "complete_semantic_trace",
        "active_families": list(SEMANTIC_ACTIVE8_FAMILIES),
        "executor_rule_to_family": [
            [rule, family] for rule, family in SEMANTIC_ACTIVE8_RULE_TO_FAMILY
        ],
        "explicitly_disabled_rules": list(EXPLICITLY_DISABLED_RULES),
        "maximum_existing_neighbors_for_atom_insert": 1,
        "all_actions_require_action_v4": True,
        "all_teachers_require_exact_mark_membership": True,
        "all_teachers_require_exact_persistent_successor": True,
        "all_teachers_require_canonical_successor_membership": True,
        "excluded_trace_emits_progress_rows": False,
        "process_identity_sha256": process_identity["process_identity_sha256"],
        "action_codec_schema_version": action_codec_v4.SCHEMA_VERSION,
        "action_codec_implementation_hash": action_codec_v4.codec_implementation_hash(),
        "required_model_modes": [list(item) for item in _REQUIRED_MODEL_MODES],
        "candidate_evaluator": _CANDIDATE_EVALUATOR,
        "implementation_file_sha256": _file_sha256(Path(__file__)),
    }
    return SemanticActive8AdmissionPolicy(
        schema=POLICY_SCHEMA,
        schema_version=POLICY_SCHEMA_VERSION,
        status=POLICY_STATUS,
        training_authorized=False,
        active8_authorized=False,
        selection_unit="complete_semantic_trace",
        active_families=SEMANTIC_ACTIVE8_FAMILIES,
        executor_rule_to_family=SEMANTIC_ACTIVE8_RULE_TO_FAMILY,
        explicitly_disabled_rules=EXPLICITLY_DISABLED_RULES,
        maximum_existing_neighbors_for_atom_insert=1,
        all_actions_require_action_v4=True,
        all_teachers_require_exact_mark_membership=True,
        all_teachers_require_exact_persistent_successor=True,
        all_teachers_require_canonical_successor_membership=True,
        excluded_trace_emits_progress_rows=False,
        process_identity_sha256=str(process_identity["process_identity_sha256"]),
        action_codec_schema_version=action_codec_v4.SCHEMA_VERSION,
        action_codec_implementation_hash=action_codec_v4.codec_implementation_hash(),
        required_model_modes=_REQUIRED_MODEL_MODES,
        candidate_evaluator=_CANDIDATE_EVALUATOR,
        implementation_file_sha256=str(body["implementation_file_sha256"]),
        policy_sha256=canonical_sha256(body),
    )


@lru_cache(maxsize=8)
def validate_process_v2_semantic_active8_admission_policy(
    policy: SemanticActive8AdmissionPolicy,
) -> SemanticActive8AdmissionPolicy:
    """Require the exact live Process-V2 policy and its self-hash."""

    if not isinstance(policy, SemanticActive8AdmissionPolicy):
        raise SemanticActive8AdmissionError("Process-V2 Active8 policy has another type")
    payload = policy.as_payload()
    body = {key: value for key, value in payload.items() if key != "policy_sha256"}
    if (
        set(payload) != _POLICY_FIELDS
        or policy != build_process_v2_semantic_active8_admission_policy()
        or policy.policy_sha256 != canonical_sha256(body)
    ):
        raise SemanticActive8AdmissionError(
            "Process-V2 Active8 policy is stale, malformed, or self-inconsistent"
        )
    return policy


def classify_process_v2_semantic_action(
    step: RewriteStep,
    *,
    step_index: int,
    policy: SemanticActive8AdmissionPolicy,
) -> SemanticActionClassification:
    """Classify one Process-V2 action through the frozen ActionCodecV4 ontology."""

    validate_process_v2_semantic_active8_admission_policy(policy)
    if type(step_index) is not int or step_index < 0:
        raise SemanticActive8AdmissionError("step_index must be a nonnegative integer")
    rule_name = str(step.rule_name)
    if rule_name in policy.explicitly_disabled_rules:
        return SemanticActionClassification(
            step_index=step_index,
            executor_rule=rule_name,
            model_family=rule_name,
            action_sha256=None,
            policy_eligible=False,
            exclusion_reason="explicitly_disabled_operator_family",
        )
    if (
        rule_name == "atom_insert"
        and isinstance(step.action, AtomInsert)
        and len(tuple(step.action.neighbors)) > policy.maximum_existing_neighbors_for_atom_insert
    ):
        return SemanticActionClassification(
            step_index=step_index,
            executor_rule=rule_name,
            model_family="atom_insert",
            action_sha256=None,
            policy_eligible=False,
            exclusion_reason="unsupported_multi_neighbor_atom_insert",
        )
    try:
        action_sha256, family = _action_sha256(rule_name, step.action)
    except ActionCodecV4Error:
        return SemanticActionClassification(
            step_index=step_index,
            executor_rule=rule_name,
            model_family=None,
            action_sha256=None,
            policy_eligible=False,
            exclusion_reason="outside_action_codec_v4",
        )
    if dict(policy.executor_rule_to_family).get(rule_name) != family:
        raise SemanticActive8AdmissionError(
            "ActionCodecV4 family mapping differs from the Process-V2 Active8 policy"
        )
    if family not in policy.active_families:
        raise SemanticActive8AdmissionError(
            "ActionCodecV4 emitted a family outside the Process-V2 Active8 policy"
        )
    return SemanticActionClassification(
        step_index=step_index,
        executor_rule=rule_name,
        model_family=family,
        action_sha256=action_sha256,
        policy_eligible=True,
        exclusion_reason=None,
    )


def _validate_process_v2_model_modes(
    model: FactorizedTraceletRateModel,
    policy: SemanticActive8AdmissionPolicy,
) -> None:
    validate_process_v2_semantic_active8_admission_policy(policy)
    if model.training:
        raise SemanticActive8AdmissionError(
            "Process-V2 Active8 candidate evaluation requires model.eval()"
        )
    capabilities = model.operator_capabilities
    observed: dict[str, object] = {
        "compute_ring_grow_support": capabilities.compute_ring_grow_support,
        "compute_ring_restates": capabilities.compute_ring_restates,
        "compute_cyclic_graft": capabilities.compute_cyclic_graft,
        "compute_ring_opening": capabilities.compute_ring_opening,
        "compute_ring_system_delete": capabilities.compute_ring_system_delete,
        "editing_process_semantics": capabilities.editing_process_semantics,
        "atom_restate_action_semantics": capabilities.atom_restate_action_semantics,
        "ring_restate_scorer_mode": capabilities.ring_restate_scorer_mode,
        "cycle_close_action_semantics": capabilities.cycle_close_action_semantics,
        "cycle_open_action_semantics": capabilities.cycle_open_action_semantics,
        "atom_delete_action_semantics": capabilities.atom_delete_action_semantics,
        "enable_cycle_ops": model.enable_cycle_ops,
        "enable_ring_grow_macro": model.enable_ring_grow_macro,
        "enable_ring_system_delete": model.enable_ring_system_delete,
        "use_aromatic_bond_view": True,
    }
    expected = dict(policy.required_model_modes)
    if observed != expected:
        mismatches = {
            key: {"expected": expected.get(key), "observed": observed.get(key)}
            for key in sorted(set(expected) | set(observed))
            if expected.get(key) != observed.get(key)
        }
        raise SemanticActive8AdmissionError(
            f"model does not implement the Process-V2 Active8 modes: {mismatches}"
        )
    if tuple(model.atom_vocabulary.classes) != tuple(ORGANIC_VOCABULARY.classes):
        raise SemanticActive8AdmissionError(
            "Process-V2 Active8 requires the broad-organic atom vocabulary"
        )


class ProductionProcessV2SemanticExactCandidateChecker:
    """Exact Process-V2 teacher check through the production canonical quotient."""

    def __init__(
        self,
        model: FactorizedTraceletRateModel,
        *,
        policy: SemanticActive8AdmissionPolicy | None = None,
        cache_size: int = 4096,
        time: float = 0.5,
    ) -> None:
        if type(cache_size) is not int or cache_size <= 0:
            raise ValueError("Process-V2 candidate cache_size must be positive")
        if not 0.0 < float(time) < 1.0:
            raise ValueError("Process-V2 candidate enumeration time must lie in (0, 1)")
        selected = policy or build_process_v2_semantic_active8_admission_policy()
        _validate_process_v2_model_modes(model, selected)
        self.model = model
        self.policy = selected
        self.cache_size = cache_size
        self.time = float(time)
        self.system = editing_v2_semantic_rewrite_system()
        self._successor_rows: OrderedDict[str, SuccessorKernelResult] = OrderedDict()

    def _successor_result(self, state: object) -> SuccessorKernelResult:
        state_sha256 = persistent_slot_state_sha256(state)
        cached = self._successor_rows.get(state_sha256)
        if cached is not None:
            self._successor_rows.move_to_end(state_sha256)
            return cached
        try:
            result = canonical_successor_result(
                self.model,
                state,
                self.time,
                system=self.system,
            )
        except (ProductionSuccessorKernelError, ValueError, RuntimeError) as error:
            raise SemanticActive8AdmissionError(
                "production Process-V2 successor enumeration failed"
            ) from error
        allowed_families = frozenset(self.policy.active_families)
        for mark in result.marked_law.marks:
            if mark.family_name not in allowed_families:
                raise SemanticActive8AdmissionError(
                    "production marked law emitted a disabled Active8 family"
                )
            try:
                _, family = _action_sha256(mark.executor_rule_name, mark.action)
            except ActionCodecV4Error as error:
                raise SemanticActive8AdmissionError(
                    "production marked law emitted an action outside ActionCodecV4"
                ) from error
            if family != mark.family_name:
                raise SemanticActive8AdmissionError(
                    "production mark family differs from ActionCodecV4"
                )
        self._successor_rows[state_sha256] = result
        self._successor_rows.move_to_end(state_sha256)
        while len(self._successor_rows) > self.cache_size:
            self._successor_rows.popitem(last=False)
        return result

    def __call__(
        self,
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> SemanticExactCandidateEvidence:
        return self.evaluate(addressed, step_index).evidence

    def evaluate(
        self,
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> ProcessV2SemanticExactCandidateAudit:
        """Return evidence and the exact-successor mark count computed with it."""

        if type(step_index) is not int or not 0 <= step_index < len(addressed.trace.steps):
            raise SemanticActive8AdmissionError(
                "Process-V2 candidate step_index lies outside the trace"
            )
        step = addressed.trace.steps[step_index]
        classification = classify_process_v2_semantic_action(
            step, step_index=step_index, policy=self.policy
        )
        if not classification.policy_eligible or classification.action_sha256 is None:
            raise SemanticActive8AdmissionError(
                "Process-V2 candidate checker received a policy-excluded action"
            )
        source = addressed.path.state_at(step_index)
        target = addressed.path.state_at(step_index + 1)
        source_sha256 = persistent_slot_state_sha256(source)
        target_sha256 = persistent_slot_state_sha256(target)
        target_key = canonical_state_key(target)
        result = self._successor_result(source)
        matching_marks = []
        for mark in result.marked_law.marks:
            mark_sha256, _ = _action_sha256(mark.executor_rule_name, mark.action)
            if mark_sha256 == classification.action_sha256:
                matching_marks.append(mark)
        matching_exact = 0
        for mark in matching_marks:
            successor = self.system.apply(source, mark.executor_rule_name, mark.action)
            if persistent_slot_state_sha256(successor) == target_sha256:
                matching_exact += 1
        canonical_match = next(
            (successor for successor in result.batch.successors if successor.key == target_key),
            None,
        )
        reason: str | None = None
        if not matching_marks:
            reason = "teacher_action_absent_from_production_marked_law"
        elif matching_exact == 0:
            reason = "teacher_action_does_not_reproduce_exact_successor"
        elif canonical_match is None:
            reason = "teacher_successor_absent_from_canonical_quotient"
        elif target_key == result.marked_law.source_key:
            reason = "teacher_successor_is_virtual_self_transition"
        return ProcessV2SemanticExactCandidateAudit(
            evidence=SemanticExactCandidateEvidence(
                supported=reason is None,
                action_sha256=classification.action_sha256,
                source_state_sha256=source_sha256,
                target_state_sha256=target_sha256,
                source_canonical_key=result.marked_law.source_key,
                canonical_successor_key=target_key,
                raw_mark_count=len(result.marked_law.marks),
                canonical_successor_count=len(result.batch.successors),
                matching_mark_count=len(matching_marks),
                successor_alias_count=(
                    0 if canonical_match is None else canonical_match.alias_count
                ),
                exclusion_reason=reason,
            ),
            exact_successor_mark_count=matching_exact,
        )


class ProductionProcessV2BatchedTeacherSupportChecker:
    """Check complete teacher marks in batches without building full quotients.

    The production model's exact action masks and coordinate decoder define
    mark support.  This checker builds those masks once per batch, decodes only
    the teacher's family, and executes only matching complete marks.  It does
    not use neural scores, hazards, or probabilities, and it never executes the
    other legal marks.  The slow canonical quotient remains the independent
    release-sentinel and T1/P50 oracle.
    """

    def __init__(
        self,
        model: FactorizedTraceletRateModel,
        *,
        policy: SemanticActive8AdmissionPolicy | None = None,
        batch_size: int = 64,
        time: float = 0.5,
        chemistry_feature_cache_size: int = 4096,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Process-V2 teacher-support batch_size must be positive")
        if type(chemistry_feature_cache_size) is not int or chemistry_feature_cache_size <= 0:
            raise ValueError("Process-V2 chemistry feature cache size must be positive")
        if not 0.0 < float(time) < 1.0:
            raise ValueError("Process-V2 teacher-support time must lie in (0, 1)")
        selected = policy or build_process_v2_semantic_active8_admission_policy()
        _validate_process_v2_model_modes(model, selected)
        self.model = model
        self.policy = selected
        self.batch_size = batch_size
        self.time = float(time)
        self.chemistry_feature_cache_size = chemistry_feature_cache_size
        self.system = editing_v2_semantic_rewrite_system()
        self._chemistry_feature_cache: OrderedDict[Any, Any] = OrderedDict()

    def evaluate_many(
        self,
        queries: Sequence[tuple[AddressedPackedTrace, int]],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[ProcessV2TeacherSupportEvidence, ...]:
        """Return evidence in input order, batching only compatible slot shapes."""

        if not queries:
            return ()
        prepared: list[dict[str, Any]] = []
        groups: dict[int, list[int]] = {}
        for query_index, (addressed, step_index) in enumerate(queries):
            if type(step_index) is not int or not 0 <= step_index < len(addressed.trace.steps):
                raise SemanticActive8AdmissionError(
                    "Process-V2 teacher-support step_index lies outside the trace"
                )
            step = addressed.trace.steps[step_index]
            classification = classify_process_v2_semantic_action(
                step, step_index=step_index, policy=self.policy
            )
            if not classification.policy_eligible or classification.action_sha256 is None:
                raise SemanticActive8AdmissionError(
                    "batched teacher-support received a policy-excluded action"
                )
            source = addressed.path.state_at(step_index)
            target = addressed.path.state_at(step_index + 1)
            prepared.append(
                {
                    "classification": classification,
                    "source": source,
                    "target": target,
                    "step": step,
                }
            )
            groups.setdefault(int(source.n_atoms), []).append(query_index)

        outputs: list[ProcessV2TeacherSupportEvidence | None] = [None] * len(queries)
        processed = 0
        for n_slots in sorted(groups):
            indices = groups[n_slots]
            for start in range(0, len(indices), self.batch_size):
                selected_indices = indices[start : start + self.batch_size]
                items = [prepared[index] for index in selected_indices]
                batch = prepare_factorized_mark_batch(
                    tuple(item["source"] for item in items),
                    (self.time,) * len(items),
                    tuple(item["step"].action for item in items),
                    tuple(str(item["step"].rule_name) for item in items),
                    (1.0,) * len(items),
                    use_aromatic_bond_view=True,
                    ring_catalog=self.model.ring_catalog,
                    chemistry_feature_cache=self._chemistry_feature_cache,
                    chemistry_feature_cache_limit=self.chemistry_feature_cache_size,
                    **operator_capability_batch_kwargs(self.model.operator_capabilities),
                ).to(self.model.device)
                dtype = next(self.model.parameters()).dtype
                node = torch.zeros(
                    (batch.batch_size, batch.n_slots, self.model.hidden_dim),
                    dtype=dtype,
                    device=self.model.device,
                )
                global_state = torch.zeros(
                    (batch.batch_size, self.model.hidden_dim),
                    dtype=dtype,
                    device=self.model.device,
                )
                pair = torch.zeros(
                    (
                        batch.batch_size,
                        batch.n_slots,
                        batch.n_slots,
                        self.model.hidden_dim,
                    ),
                    dtype=dtype,
                    device=self.model.device,
                )
                with torch.inference_mode():
                    masks, _logits, _partitions = self.model._action_tables(
                        batch,
                        node,
                        global_state,
                        pair,
                        require_exact_ring_support=False,
                    )
                allowed = frozenset(self.policy.active_families)
                if bool(masks["ring_system_grow"].any()):
                    raise SemanticActive8AdmissionError(
                        "ring_system_grow has legal support despite being disabled in Active8"
                    )
                for candidate_family, table_name in _TABLE_FAMILIES:
                    if candidate_family not in allowed and bool(masks[table_name].any()):
                        raise SemanticActive8AdmissionError(
                            f"disabled family {candidate_family} has legal Process-V2 support"
                        )
                cpu_masks = {name: value.detach().cpu() for name, value in masks.items()}
                for batch_index, query_index in enumerate(selected_indices):
                    item = prepared[query_index]
                    outputs[query_index] = self._evidence_from_masks(
                        item,
                        batch=batch,
                        batch_index=batch_index,
                        masks=cpu_masks,
                    )
                processed += len(selected_indices)
                if progress_callback is not None:
                    progress_callback(processed, len(queries))
        if any(output is None for output in outputs):
            raise SemanticActive8AdmissionError(
                "batched teacher-support failed to produce every requested result"
            )
        return tuple(output for output in outputs if output is not None)

    def evaluate(
        self,
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> ProcessV2TeacherSupportEvidence:
        """Evaluate one teacher through the same batched implementation."""

        return self.evaluate_many(((addressed, step_index),))[0]

    def __call__(
        self,
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> ProcessV2TeacherSupportEvidence:
        """Satisfy the scalar checker protocol through the batched implementation."""

        return self.evaluate(addressed, step_index)

    def _evidence_from_masks(
        self,
        item: dict[str, Any],
        *,
        batch: Any,
        batch_index: int,
        masks: dict[str, torch.Tensor],
    ) -> ProcessV2TeacherSupportEvidence:
        classification = item["classification"]
        source = item["source"]
        target = item["target"]
        action_sha256 = str(classification.action_sha256)
        family = str(classification.model_family)
        source_sha256 = persistent_slot_state_sha256(source)
        target_sha256 = persistent_slot_state_sha256(target)
        target_key = canonical_state_key(target)
        source_key = canonical_state_key(source)
        matching: list[tuple[str, Any]] = []
        raw_mark_count = 0
        one = batch.subbatch(batch_index, batch_index + 1)
        allowed = frozenset(self.policy.active_families)
        for candidate_family, table_name in _TABLE_FAMILIES:
            if candidate_family not in allowed:
                continue
            row_mask = masks[table_name][batch_index]
            raw_mark_count += int(row_mask.sum().item())
            if candidate_family != family:
                continue
            for coordinate_row in torch.nonzero(row_mask, as_tuple=False):
                coordinate = tuple(int(value) for value in coordinate_row)
                executor_rule, action = _coordinate_action(
                    self.model,
                    source,
                    one,
                    family_name=candidate_family,
                    table_name=table_name,
                    coordinate=coordinate,
                )
                candidate_sha256, observed_family = _action_sha256(executor_rule, action)
                if observed_family != candidate_family:
                    raise SemanticActive8AdmissionError(
                        "Process-V2 mask coordinate decodes to another model family"
                    )
                if candidate_sha256 == action_sha256:
                    matching.append((executor_rule, action))

        exact = 0
        for executor_rule, action in matching:
            try:
                successor = self.system.apply(source, executor_rule, action)
            except Exception as error:
                raise SemanticActive8AdmissionError(
                    "a Process-V2 legal teacher coordinate is rejected by the executor"
                ) from error
            if persistent_slot_state_sha256(successor) == target_sha256:
                exact += 1
        coordinate_legal = len(matching) == 1
        executes_exact = exact == 1
        productive = target_key != source_key
        reason: str | None = None
        if not coordinate_legal:
            reason = "teacher_action_not_a_unique_process_v2_coordinate"
        elif not executes_exact:
            reason = "teacher_action_does_not_reproduce_exact_successor"
        elif not productive:
            reason = "teacher_successor_is_virtual_self_transition"
        return ProcessV2TeacherSupportEvidence(
            supported=reason is None,
            exclusion_reason=reason,
            action_sha256=action_sha256,
            source_state_sha256=source_sha256,
            target_state_sha256=target_sha256,
            source_canonical_key=source_key,
            canonical_successor_key=target_key,
            teacher_coordinate_legal=coordinate_legal,
            teacher_executes_to_exact_successor=executes_exact,
            productive_canonical_successor=productive,
            raw_mark_count=raw_mark_count,
            matching_mark_count=len(matching),
            exact_successor_mark_count=exact,
        )


def evaluate_process_v2_semantic_active8_trace(
    addressed: AddressedPackedTrace,
    *,
    exact_candidate_checker: ProcessV2SemanticExactCandidateChecker,
    policy: SemanticActive8AdmissionPolicy | None = None,
) -> SemanticActive8TraceDecision:
    """Apply immutable whole-trace admission under the Process-V2 policy."""

    selected = validate_process_v2_semantic_active8_admission_policy(
        policy or build_process_v2_semantic_active8_admission_policy()
    )
    if addressed.address.path_length != len(addressed.trace.steps):
        raise SemanticActive8AdmissionError("Process-V2 trace address and action count disagree")
    if addressed.path.path_length != addressed.address.path_length:
        raise SemanticActive8AdmissionError(
            "Process-V2 exact-state path and address disagree"
        )
    resolved_classifications = tuple(
        classify_process_v2_semantic_action(step, step_index=index, policy=selected)
        for index, step in enumerate(addressed.trace.steps)
    )
    exclusions = [
        SemanticActive8Exclusion(
            stage="action_v4_policy",
            step_index=item.step_index,
            executor_rule=item.executor_rule,
            model_family=item.model_family,
            reason=str(item.exclusion_reason),
        )
        for item in resolved_classifications
        if not item.policy_eligible
    ]
    action_decisions: list[SemanticActive8ActionDecision] = []
    if exclusions:
        action_decisions.extend(
            SemanticActive8ActionDecision(classification=item, candidate_evidence=None)
            for item in resolved_classifications
        )
    else:
        for classification in resolved_classifications:
            evidence = exact_candidate_checker(addressed, classification.step_index)
            if not isinstance(
                evidence,
                (SemanticExactCandidateEvidence, ProcessV2TeacherSupportEvidence),
            ):
                raise SemanticActive8AdmissionError(
                    "Process-V2 candidate checker returned another result type"
                )
            if evidence.action_sha256 != classification.action_sha256:
                raise SemanticActive8AdmissionError(
                    "candidate evidence action identity differs from ActionCodecV4"
                )
            action_decisions.append(
                SemanticActive8ActionDecision(
                    classification=classification,
                    candidate_evidence=evidence,
                )
            )
            if not evidence.supported:
                exclusions.append(
                    SemanticActive8Exclusion(
                        stage="production_candidate_support",
                        step_index=classification.step_index,
                        executor_rule=classification.executor_rule,
                        model_family=classification.model_family,
                        reason=str(evidence.exclusion_reason),
                    )
                )
    accepted = not exclusions
    return SemanticActive8TraceDecision(
        trace_id=addressed.address.trace_id,
        policy_sha256=selected.policy_sha256,
        semantic_migration_status="admitted",
        semantic_migration_rejection=None,
        active8_status="accepted" if accepted else "excluded",
        emits_progress_rows=accepted,
        action_decisions=tuple(action_decisions),
        active8_exclusions=tuple(exclusions),
    )


__all__ = [
    "POLICY_SCHEMA",
    "POLICY_SCHEMA_VERSION",
    "POLICY_STATUS",
    "ProcessV2SemanticExactCandidateAudit",
    "ProcessV2SemanticExactCandidateChecker",
    "ProcessV2TeacherSupportEvidence",
    "ProductionProcessV2BatchedTeacherSupportChecker",
    "ProductionProcessV2SemanticExactCandidateChecker",
    "SemanticActionClassification",
    "SemanticActive8ActionDecision",
    "SemanticActive8AdmissionError",
    "SemanticActive8AdmissionPolicy",
    "SemanticActive8Exclusion",
    "SemanticActive8TraceDecision",
    "SemanticExactCandidateEvidence",
    "build_process_v2_semantic_active8_admission_policy",
    "classify_process_v2_semantic_action",
    "evaluate_process_v2_semantic_active8_trace",
    "validate_process_v2_semantic_active8_admission_policy",
]
