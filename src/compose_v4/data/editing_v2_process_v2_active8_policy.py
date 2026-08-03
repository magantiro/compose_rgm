"""The Process-V2 Active8 admission policy and its exact candidate checker.

WHAT THIS DECIDES
-----------------
One whole trace, all or nothing.  A trace is *accepted* only when every one of
its actions is inside the eight declared development families, is representable
by the frozen ``ActionCodecV4`` ontology, and -- evaluated through the
**unchanged production model and executor** -- reproduces both its exact
persistent-slot successor and a canonical molecular successor of the model's own
quotient.  A single unsupported middle action excludes the ENTIRE trace, because
a partially admitted trace would hand a later stage a progress row whose
predecessor transition was never learnable.

WHY IT IS A SEPARATE MODULE FROM THE V1 POLICY
----------------------------------------------
Process V2 expands the learned ``atom_delete`` fiber, so the process identity
this policy binds is a different value from the V1 one and the model modes it
requires are a different set: ``editing_process_semantics`` is the V2 spelling
and ``atom_delete_action_semantics`` is the V2 uniform-gated mode, which the V1
policy cannot express and must never accept.  Sharing one policy object between
the two processes would mean one ``policy_sha256`` covering two different
admission laws, which is exactly the conflation the split process identity
exists to prevent.  Nothing here imports a V1 decision module.

WHAT IT DOES NOT DO
-------------------
It reads no corpus, publishes no artifact, opens no file except its own source
for the implementation hash, and chooses no threshold.  Candidate membership is
delegated to the one production successor evaluator; this module never builds a
second chemistry enumerator.  Every authority field it publishes is ``False``.

THE TWO REJECTION CATEGORIES STAY DISTINCT
------------------------------------------
A trace the upstream Process-V2 rebind already refused was never a candidate:
Active8 does not evaluate it and records it under
:data:`~compose_v4.data.editing_v2_process_v2_active8_interfaces.UPSTREAM_REJECTED`.
A trace Active8 evaluated and refused is recorded under
:data:`~compose_v4.data.editing_v2_process_v2_active8_interfaces.ACTIVE8_EXCLUDED`.
Merging them would make "we did not look" indistinguishable from "we looked and
said no", which is a different scientific statement about the same trace.

INVARIANTS MAINTAINED (and tested)
----------------------------------
* the eight declared families are exactly ``ACTIVE8_EXECUTOR_RULES``, and
  ``ring_system_delete``/``ring_system_grow`` are refused as explicitly disabled;
* the policy binds ``editing_process_v2_identity()``, never the V1 identity;
* the checker refuses a model that is not in the exact Process-V2 modes;
* a teacher mark must be present in the production marked law, must replay to
  the exact persistent-slot successor, and its canonical successor must be a row
  of the production quotient;
* raw mark count, canonical successor count, matching mark count and successor
  alias count are all carried through unaggregated;
* one unsupported action excludes the whole trace and it emits no progress rows.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES, DISABLED_FAMILIES
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_EXCLUDED,
    REJECTION_CATEGORIES,
    UPSTREAM_REJECTED,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    require_authority_false,
)
from compose_v4.data.packed_trace_store import AddressedPackedTrace
from compose_v4.experiments.production_successor_kernel import (
    ProductionSuccessorKernelError,
    SuccessorKernelResult,
    canonical_successor_result,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS,
    PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS,
    SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS,
    SEMANTIC_RING_RESTATE_SCORER_MODE,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.action_codec_v4 import ActionCodecV4Error
from compose_v4.rewrite.editing_v2_process_identity import (
    EditingV2ProcessIdentityError,
    editing_process_v2_identity,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace import RewriteStep

# ---- Frozen identity ----

POLICY_SCHEMA = "compose.data.editing_v2_process_v2_active8_policy"
POLICY_SCHEMA_VERSION = 1
POLICY_STATUS = "FROZEN_PROCESS_V2_ACTIVE8_POLICY_NO_DOWNSTREAM_AUTHORITY"
SELECTION_UNIT = "complete_process_v2_trace"

#: The eight declared development families, in executor-rule spelling, read from
#: the frozen codec rather than restated here.
PROCESS_V2_ACTIVE8_EXECUTOR_RULES: tuple[str, ...] = tuple(
    action_codec_v4.ACTIVE8_EXECUTOR_RULES
)
PROCESS_V2_ACTIVE8_FAMILIES: tuple[str, ...] = tuple(ACTIVE8_FAMILIES)
PROCESS_V2_ACTIVE8_RULE_TO_FAMILY: tuple[tuple[str, str], ...] = tuple(
    (rule, action_codec_v4.canonical_family(rule))
    for rule in PROCESS_V2_ACTIVE8_EXECUTOR_RULES
)
#: ``ring_system_delete`` and ``ring_system_grow``: outside the bounded pilot and
#: refused rather than silently absent, so an artifact that carries one names it.
PROCESS_V2_EXPLICITLY_DISABLED_RULES: tuple[str, ...] = tuple(DISABLED_FAMILIES)

MAXIMUM_EXISTING_NEIGHBORS_FOR_ATOM_INSERT = 1

_CANDIDATE_EVALUATOR = (
    "compose_v4.experiments.production_successor_kernel.canonical_successor_result"
)

#: The exact model modes a Process-V2 candidate evaluation requires.  The two
#: that differ from V1 are the process semantics and the atom-delete mode; both
#: are named rather than defaulted, so a V1 model cannot satisfy this policy.
_REQUIRED_MODEL_MODES: tuple[tuple[str, object], ...] = (
    ("compute_ring_grow_support", False),
    ("compute_ring_restates", True),
    ("compute_cyclic_graft", True),
    ("compute_ring_opening", True),
    ("compute_ring_system_delete", False),
    ("editing_process_semantics", PROCESS_V2_EDITING_PROCESS_SEMANTICS),
    ("atom_delete_action_semantics", PROCESS_V2_ATOM_DELETE_ACTION_SEMANTICS),
    ("atom_restate_action_semantics", SEMANTIC_ATOM_RESTATE_ACTION_SEMANTICS),
    ("ring_restate_scorer_mode", SEMANTIC_RING_RESTATE_SCORER_MODE),
    ("cycle_close_action_semantics", SEMANTIC_CYCLE_CLOSE_ACTION_SEMANTICS),
    ("cycle_open_action_semantics", SEMANTIC_CYCLE_OPEN_ACTION_SEMANTICS),
    ("enable_cycle_ops", True),
    ("enable_ring_grow_macro", False),
    ("enable_ring_system_delete", False),
    ("use_aromatic_bond_view", True),
)

# ---- Reason codes ----

#: Static, decided before the model is consulted.
REASON_EXPLICITLY_DISABLED = "explicitly_disabled_operator_family"
REASON_OUTSIDE_ACTION_CODEC = "outside_action_codec_v4"
REASON_MULTI_NEIGHBOR_INSERT = "unsupported_multi_neighbor_atom_insert"
#: Dynamic, decided by the production marked law and its canonical quotient.
REASON_MARK_ABSENT = "teacher_mark_absent_from_process_v2_marked_law"
REASON_MARK_NOT_EXACT = "teacher_mark_does_not_reproduce_exact_successor"
REASON_SUCCESSOR_ABSENT = "teacher_successor_absent_from_canonical_quotient"
REASON_SUCCESSOR_VIRTUAL = "teacher_successor_is_virtual_self_transition"

STATIC_EXCLUSION_STAGE = "action_codec_v4_policy"
DYNAMIC_EXCLUSION_STAGE = "production_candidate_support"
EXCLUSION_STAGES: tuple[str, ...] = (STATIC_EXCLUSION_STAGE, DYNAMIC_EXCLUSION_STAGE)

STATIC_EXCLUSION_REASONS: tuple[str, ...] = (
    REASON_EXPLICITLY_DISABLED,
    REASON_MULTI_NEIGHBOR_INSERT,
    REASON_OUTSIDE_ACTION_CODEC,
)
DYNAMIC_EXCLUSION_REASONS: tuple[str, ...] = (
    REASON_MARK_ABSENT,
    REASON_MARK_NOT_EXACT,
    REASON_SUCCESSOR_ABSENT,
    REASON_SUCCESSOR_VIRTUAL,
)
ACTIVE8_EXCLUSION_REASONS: tuple[str, ...] = tuple(
    sorted({*STATIC_EXCLUSION_REASONS, *DYNAMIC_EXCLUSION_REASONS})
)

_POLICY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "active8_authorized",
        "selection_unit",
        "active_families",
        "executor_rule_to_family",
        "explicitly_disabled_rules",
        "maximum_existing_neighbors_for_atom_insert",
        "all_actions_require_action_codec_v4",
        "all_teachers_require_exact_mark_membership",
        "all_teachers_require_exact_persistent_successor",
        "all_teachers_require_canonical_successor_membership",
        "unsupported_middle_action_excludes_whole_trace",
        "excluded_trace_emits_progress_rows",
        "upstream_rejected_trace_is_evaluated",
        "rejection_categories",
        "active8_exclusion_reasons",
        "process_identity_sha256",
        "action_codec_schema_version",
        "action_codec_implementation_hash",
        "required_model_modes",
        "candidate_evaluator",
        "implementation_file_sha256",
        "policy_sha256",
    }
)


class ProcessV2Active8PolicyError(RuntimeError):
    """The Process-V2 Active8 policy, model or checker contract is invalid."""


# ---- Canonical helpers ----


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _action_identity(rule_name: str, action: Any) -> tuple[str, str]:
    """Return ``(action_sha256, model_family)`` through the frozen V4 codec.

    The round trip is asserted rather than assumed: an action whose encoding
    does not decode back to itself is not addressable by a hash, and admitting
    it would give the teacher an identity the model can never match.
    """

    record = action_codec_v4.encode_action(rule_name, action)
    decoded_rule, decoded_action = action_codec_v4.decode_action(record)
    if decoded_rule != rule_name or decoded_action != action:
        raise ProcessV2Active8PolicyError(
            "ActionCodecV4 round trip changed the Process-V2 teacher action"
        )
    return _canonical_sha256(record), str(record["model_family"])


# ---- Policy ----


@dataclass(frozen=True, slots=True)
class ProcessV2Active8Policy:
    """The self-identifying, non-authorizing Process-V2 Active8 policy."""

    schema: str
    schema_version: int
    status: str
    training_authorized: bool
    gate_zero_authorized: bool
    t1_authorized: bool
    bounded_p50_authorized: bool
    long_training_authorized: bool
    checkpoint_selection_authorized: bool
    final_test_selection_authorized: bool
    active8_authorized: bool
    selection_unit: str
    active_families: tuple[str, ...]
    executor_rule_to_family: tuple[tuple[str, str], ...]
    explicitly_disabled_rules: tuple[str, ...]
    maximum_existing_neighbors_for_atom_insert: int
    rejection_categories: tuple[str, ...]
    active8_exclusion_reasons: tuple[str, ...]
    process_identity_sha256: str
    action_codec_schema_version: int
    action_codec_implementation_hash: str
    required_model_modes: tuple[tuple[str, object], ...]
    candidate_evaluator: str
    implementation_file_sha256: str
    policy_sha256: str

    def as_payload(self) -> dict[str, Any]:
        """A deterministic JSON-compatible copy, including its own self-hash."""

        return {**_policy_body(self), "policy_sha256": self.policy_sha256}


def _policy_body(policy: ProcessV2Active8Policy) -> dict[str, Any]:
    """The exact hashed body, built in one place so writer and reader agree."""

    return {
        "schema": policy.schema,
        "schema_version": policy.schema_version,
        "status": policy.status,
        "training_authorized": policy.training_authorized,
        "gate_zero_authorized": policy.gate_zero_authorized,
        "t1_authorized": policy.t1_authorized,
        "bounded_p50_authorized": policy.bounded_p50_authorized,
        "long_training_authorized": policy.long_training_authorized,
        "checkpoint_selection_authorized": policy.checkpoint_selection_authorized,
        "final_test_selection_authorized": policy.final_test_selection_authorized,
        "active8_authorized": policy.active8_authorized,
        "selection_unit": policy.selection_unit,
        "active_families": list(policy.active_families),
        "executor_rule_to_family": [
            [rule, family] for rule, family in policy.executor_rule_to_family
        ],
        "explicitly_disabled_rules": list(policy.explicitly_disabled_rules),
        "maximum_existing_neighbors_for_atom_insert": (
            policy.maximum_existing_neighbors_for_atom_insert
        ),
        # Every one of these is a statement the checker below actually enforces.
        # A policy field nothing enforces is documentation wearing an identity.
        "all_actions_require_action_codec_v4": True,
        "all_teachers_require_exact_mark_membership": True,
        "all_teachers_require_exact_persistent_successor": True,
        "all_teachers_require_canonical_successor_membership": True,
        "unsupported_middle_action_excludes_whole_trace": True,
        "excluded_trace_emits_progress_rows": False,
        "upstream_rejected_trace_is_evaluated": False,
        "rejection_categories": list(policy.rejection_categories),
        "active8_exclusion_reasons": list(policy.active8_exclusion_reasons),
        "process_identity_sha256": policy.process_identity_sha256,
        "action_codec_schema_version": policy.action_codec_schema_version,
        "action_codec_implementation_hash": policy.action_codec_implementation_hash,
        "required_model_modes": [[name, value] for name, value in policy.required_model_modes],
        "candidate_evaluator": policy.candidate_evaluator,
        "implementation_file_sha256": policy.implementation_file_sha256,
    }


@lru_cache(maxsize=1)
def build_process_v2_active8_policy() -> ProcessV2Active8Policy:
    """Build the current policy, binding the live Process-V2 process identity."""

    try:
        identity = editing_process_v2_identity()
    except EditingV2ProcessIdentityError as error:
        raise ProcessV2Active8PolicyError(
            "cannot establish the Process-V2 editing process identity"
        ) from error
    process_sha256 = identity.get("process_identity_sha256")
    if not _is_sha256(process_sha256):
        raise ProcessV2Active8PolicyError(
            "the Process-V2 process identity lacks a full lowercase SHA-256"
        )
    draft = ProcessV2Active8Policy(
        schema=POLICY_SCHEMA,
        schema_version=POLICY_SCHEMA_VERSION,
        status=POLICY_STATUS,
        **authority_false_block(),
        active8_authorized=False,
        selection_unit=SELECTION_UNIT,
        active_families=PROCESS_V2_ACTIVE8_FAMILIES,
        executor_rule_to_family=PROCESS_V2_ACTIVE8_RULE_TO_FAMILY,
        explicitly_disabled_rules=PROCESS_V2_EXPLICITLY_DISABLED_RULES,
        maximum_existing_neighbors_for_atom_insert=(
            MAXIMUM_EXISTING_NEIGHBORS_FOR_ATOM_INSERT
        ),
        rejection_categories=REJECTION_CATEGORIES,
        active8_exclusion_reasons=ACTIVE8_EXCLUSION_REASONS,
        process_identity_sha256=str(process_sha256),
        action_codec_schema_version=action_codec_v4.SCHEMA_VERSION,
        action_codec_implementation_hash=action_codec_v4.codec_implementation_hash(),
        required_model_modes=_REQUIRED_MODEL_MODES,
        candidate_evaluator=_CANDIDATE_EVALUATOR,
        implementation_file_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        policy_sha256="0" * 64,
    )
    body = _policy_body(draft)
    require_authority_false(body, label="the Process-V2 Active8 policy")
    return ProcessV2Active8Policy(
        **{
            field: getattr(draft, field)
            for field in ProcessV2Active8Policy.__slots__
            if field != "policy_sha256"
        },
        policy_sha256=_canonical_sha256(body),
    )


def validate_process_v2_active8_policy(
    policy: ProcessV2Active8Policy,
) -> ProcessV2Active8Policy:
    """Require exact shape, self-hash, and equality with the live policy."""

    if not isinstance(policy, ProcessV2Active8Policy):
        raise ProcessV2Active8PolicyError(
            "the Process-V2 Active8 policy has another type; a V1 policy object "
            "is not a Process-V2 policy even when its fields look alike"
        )
    payload = policy.as_payload()
    if set(payload) != _POLICY_FIELDS:
        raise ProcessV2Active8PolicyError("Process-V2 Active8 policy fields disagree")
    if (
        not _is_sha256(policy.policy_sha256)
        or policy.policy_sha256 != _canonical_sha256(_policy_body(policy))
        or policy != build_process_v2_active8_policy()
    ):
        raise ProcessV2Active8PolicyError(
            "the Process-V2 Active8 policy is stale, malformed, or self-inconsistent"
        )
    return policy


def validate_process_v2_active8_policy_payload(value: object) -> dict[str, Any]:
    """Validate a serialized policy against the live one, refusing a V1 body."""

    if not isinstance(value, dict) or set(value) != _POLICY_FIELDS:
        raise ProcessV2Active8PolicyError(
            "a serialized Process-V2 Active8 policy must carry exactly its own fields"
        )
    require_authority_false(value, label="the serialized Process-V2 Active8 policy")
    current = build_process_v2_active8_policy().as_payload()
    if value != current:
        raise ProcessV2Active8PolicyError(
            "the serialized Process-V2 Active8 policy differs from the live policy"
        )
    return dict(value)


# ---- Static classification ----


@dataclass(frozen=True, slots=True)
class ProcessV2ActionClassification:
    """One action's ActionCodecV4 classification, before the model is consulted."""

    step_index: int
    executor_rule: str
    model_family: str | None
    action_sha256: str | None
    policy_eligible: bool
    exclusion_reason: str | None

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("classification step_index must be a nonnegative integer")
        if not self.executor_rule:
            raise ValueError("classification executor_rule must be nonempty")
        if self.policy_eligible:
            if (
                self.model_family is None
                or not _is_sha256(self.action_sha256)
                or self.exclusion_reason is not None
            ):
                raise ValueError("eligible classification evidence is incomplete")
        elif (
            self.exclusion_reason not in STATIC_EXCLUSION_REASONS
            or self.action_sha256 is not None
        ):
            raise ValueError("excluded classification evidence is inconsistent")

    def as_payload(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "executor_rule": self.executor_rule,
            "model_family": self.model_family,
            "action_sha256": self.action_sha256,
            "policy_eligible": self.policy_eligible,
            "exclusion_reason": self.exclusion_reason,
        }


def classify_process_v2_action(
    step: RewriteStep,
    *,
    step_index: int,
    policy: ProcessV2Active8Policy,
) -> ProcessV2ActionClassification:
    """Classify one action through the frozen ActionCodecV4 ontology only."""

    validate_process_v2_active8_policy(policy)
    if type(step_index) is not int or step_index < 0:
        raise ProcessV2Active8PolicyError("step_index must be a nonnegative integer")
    rule_name = str(step.rule_name)
    if rule_name in policy.explicitly_disabled_rules:
        return ProcessV2ActionClassification(
            step_index=step_index,
            executor_rule=rule_name,
            model_family=rule_name,
            action_sha256=None,
            policy_eligible=False,
            exclusion_reason=REASON_EXPLICITLY_DISABLED,
        )
    if (
        rule_name == "atom_insert"
        and isinstance(step.action, AtomInsert)
        and len(tuple(step.action.neighbors))
        > policy.maximum_existing_neighbors_for_atom_insert
    ):
        return ProcessV2ActionClassification(
            step_index=step_index,
            executor_rule=rule_name,
            model_family="atom_insert",
            action_sha256=None,
            policy_eligible=False,
            exclusion_reason=REASON_MULTI_NEIGHBOR_INSERT,
        )
    try:
        action_sha256, family = _action_identity(rule_name, step.action)
    except ActionCodecV4Error:
        return ProcessV2ActionClassification(
            step_index=step_index,
            executor_rule=rule_name,
            model_family=None,
            action_sha256=None,
            policy_eligible=False,
            exclusion_reason=REASON_OUTSIDE_ACTION_CODEC,
        )
    mapping = dict(policy.executor_rule_to_family)
    if mapping.get(rule_name) != family or family not in policy.active_families:
        raise ProcessV2Active8PolicyError(
            "ActionCodecV4 family mapping differs from the Process-V2 Active8 policy"
        )
    return ProcessV2ActionClassification(
        step_index=step_index,
        executor_rule=rule_name,
        model_family=family,
        action_sha256=action_sha256,
        policy_eligible=True,
        exclusion_reason=None,
    )


# ---- Exact candidate evidence ----


@dataclass(frozen=True, slots=True)
class ProcessV2CandidateEvidence:
    """Exact marked-law and canonical-quotient evidence for one teacher action.

    Every count is carried unaggregated.  ``raw_mark_count`` and
    ``canonical_successor_count`` describe the fiber the model actually
    enumerated; ``matching_mark_count`` and ``successor_alias_count`` describe
    the teacher's own multiplicity inside it.  Collapsing any of them into a
    single "supported" boolean would discard the alias structure the canonical
    successor law is defined by.
    """

    supported: bool
    action_sha256: str
    source_state_sha256: str
    target_state_sha256: str
    canonical_successor_key: str
    raw_mark_count: int
    canonical_successor_count: int
    matching_mark_count: int
    exact_successor_mark_count: int
    successor_alias_count: int
    exclusion_reason: str | None

    def __post_init__(self) -> None:
        for field in ("action_sha256", "source_state_sha256", "target_state_sha256"):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if not self.canonical_successor_key:
            raise ValueError("canonical_successor_key must be nonempty")
        for field in (
            "raw_mark_count",
            "canonical_successor_count",
            "matching_mark_count",
            "exact_successor_mark_count",
            "successor_alias_count",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.exact_successor_mark_count > self.matching_mark_count:
            raise ValueError("more exact-successor marks than matching marks")
        if self.supported:
            if (
                self.exclusion_reason is not None
                or self.matching_mark_count == 0
                or self.exact_successor_mark_count == 0
                or self.successor_alias_count == 0
                or self.canonical_successor_count == 0
            ):
                raise ValueError("supported candidate evidence is incomplete")
        elif self.exclusion_reason not in DYNAMIC_EXCLUSION_REASONS:
            raise ValueError("unsupported candidate evidence needs a declared reason")

    def as_payload(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "action_sha256": self.action_sha256,
            "source_state_sha256": self.source_state_sha256,
            "target_state_sha256": self.target_state_sha256,
            "canonical_successor_key": self.canonical_successor_key,
            "raw_mark_count": self.raw_mark_count,
            "canonical_successor_count": self.canonical_successor_count,
            "matching_mark_count": self.matching_mark_count,
            "exact_successor_mark_count": self.exact_successor_mark_count,
            "successor_alias_count": self.successor_alias_count,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True, slots=True)
class ProcessV2Active8Exclusion:
    """One Active8 exclusion, addressed to the action that caused it."""

    stage: Literal["action_codec_v4_policy", "production_candidate_support"]
    step_index: int
    executor_rule: str
    model_family: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.stage not in EXCLUSION_STAGES:
            raise ValueError(f"unknown Process-V2 Active8 exclusion stage {self.stage!r}")
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("exclusion step_index must be a nonnegative integer")
        if self.reason not in ACTIVE8_EXCLUSION_REASONS:
            raise ValueError(f"undeclared Process-V2 Active8 reason {self.reason!r}")

    def as_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "step_index": self.step_index,
            "executor_rule": self.executor_rule,
            "model_family": self.model_family,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ProcessV2ActionDecision:
    """Static classification plus, when it was reached, dynamic evidence."""

    classification: ProcessV2ActionClassification
    candidate_evidence: ProcessV2CandidateEvidence | None

    def __post_init__(self) -> None:
        if not self.classification.policy_eligible and self.candidate_evidence is not None:
            raise ValueError(
                "a statically excluded action cannot carry production candidate evidence"
            )

    def as_payload(self) -> dict[str, Any]:
        return {
            **self.classification.as_payload(),
            "candidate_evidence": (
                None
                if self.candidate_evidence is None
                else self.candidate_evidence.as_payload()
            ),
        }


@dataclass(frozen=True, slots=True)
class ProcessV2Active8TraceDecision:
    """One whole-trace decision, with the two rejection categories kept apart."""

    trace_id: str
    policy_sha256: str
    category: str | None
    upstream_rejection_code: str | None
    active8_status: Literal["accepted", "excluded", "not_evaluated"]
    emits_progress_rows: bool
    action_decisions: tuple[ProcessV2ActionDecision, ...]
    active8_exclusions: tuple[ProcessV2Active8Exclusion, ...]

    def __post_init__(self) -> None:
        if not self.trace_id or not _is_sha256(self.policy_sha256):
            raise ValueError("Process-V2 trace decision identity is incomplete")
        if self.category is not None and self.category not in REJECTION_CATEGORIES:
            raise ValueError(f"undeclared rejection category {self.category!r}")
        if self.category == UPSTREAM_REJECTED:
            if (
                not self.upstream_rejection_code
                or self.active8_status != "not_evaluated"
                or self.emits_progress_rows
                or self.action_decisions
                or self.active8_exclusions
            ):
                raise ValueError(
                    "an upstream rebind rejection was reinterpreted as Active8 evidence"
                )
            return
        if self.upstream_rejection_code is not None:
            raise ValueError(
                "an Active8-evaluated trace cannot carry an upstream rejection code"
            )
        accepted = self.active8_status == "accepted"
        if (
            self.active8_status not in {"accepted", "excluded"}
            or accepted != (not self.active8_exclusions)
            or self.emits_progress_rows != accepted
            or (accepted and self.category is not None)
            or (not accepted and self.category != ACTIVE8_EXCLUDED)
        ):
            raise ValueError("the whole-trace Process-V2 decision is internally inconsistent")


class ProcessV2ExactCandidateCheck(Protocol):
    """Check one Process-V2 teacher through the exact production successor law."""

    def __call__(
        self, addressed: AddressedPackedTrace, step_index: int
    ) -> ProcessV2CandidateEvidence: ...


def _require_process_v2_model_modes(
    model: FactorizedTraceletRateModel, policy: ProcessV2Active8Policy
) -> None:
    validate_process_v2_active8_policy(policy)
    if model.training:
        raise ProcessV2Active8PolicyError(
            "Process-V2 candidate evaluation requires model.eval()"
        )
    capabilities = model.operator_capabilities
    observed: dict[str, object] = {
        "compute_ring_grow_support": capabilities.compute_ring_grow_support,
        "compute_ring_restates": capabilities.compute_ring_restates,
        "compute_cyclic_graft": capabilities.compute_cyclic_graft,
        "compute_ring_opening": capabilities.compute_ring_opening,
        "compute_ring_system_delete": capabilities.compute_ring_system_delete,
        "editing_process_semantics": capabilities.editing_process_semantics,
        "atom_delete_action_semantics": capabilities.atom_delete_action_semantics,
        "atom_restate_action_semantics": capabilities.atom_restate_action_semantics,
        "ring_restate_scorer_mode": capabilities.ring_restate_scorer_mode,
        "cycle_close_action_semantics": capabilities.cycle_close_action_semantics,
        "cycle_open_action_semantics": capabilities.cycle_open_action_semantics,
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
        raise ProcessV2Active8PolicyError(
            f"the model does not implement the Process-V2 Active8 modes: {mismatches}"
        )
    if tuple(model.atom_vocabulary.classes) != tuple(ORGANIC_VOCABULARY.classes):
        raise ProcessV2Active8PolicyError(
            "Process-V2 Active8 requires the broad-organic atom vocabulary"
        )


class ProcessV2ProductionCandidateChecker:
    """The exact teacher check, through the unchanged model and executor.

    Enumeration goes through the one production successor evaluator and the one
    production rewrite system.  Nothing here re-derives a fiber, and the marked
    law is audited on the way past: a mark outside the eight declared families,
    or outside ActionCodecV4, is a refusal rather than a silently dropped
    candidate.
    """

    def __init__(
        self,
        model: FactorizedTraceletRateModel,
        *,
        policy: ProcessV2Active8Policy | None = None,
        cache_size: int = 4096,
        time: float = 0.5,
    ) -> None:
        if type(cache_size) is not int or cache_size <= 0:
            raise ValueError("Process-V2 candidate cache_size must be a positive integer")
        if not 0.0 < float(time) < 1.0:
            raise ValueError("Process-V2 candidate enumeration time must lie in (0, 1)")
        selected = validate_process_v2_active8_policy(
            policy or build_process_v2_active8_policy()
        )
        _require_process_v2_model_modes(model, selected)
        self.model = model
        self.policy = selected
        self.cache_size = cache_size
        self.time = float(time)
        self.system = editing_v2_semantic_rewrite_system()
        self._results: OrderedDict[str, SuccessorKernelResult] = OrderedDict()

    def _successor_result(self, state: object) -> SuccessorKernelResult:
        state_sha256 = persistent_slot_state_sha256(state)
        cached = self._results.get(state_sha256)
        if cached is not None:
            self._results.move_to_end(state_sha256)
            return cached
        try:
            result = canonical_successor_result(
                self.model, state, self.time, system=self.system
            )
        except (ProductionSuccessorKernelError, ValueError, RuntimeError) as error:
            raise ProcessV2Active8PolicyError(
                "the production Process-V2 successor enumeration failed"
            ) from error
        allowed = frozenset(self.policy.active_families)
        for mark in result.marked_law.marks:
            if mark.family_name not in allowed:
                raise ProcessV2Active8PolicyError(
                    "the production marked law emitted a family outside the declared "
                    f"Process-V2 development eight: {mark.family_name!r}"
                )
            try:
                _, family = _action_identity(mark.executor_rule_name, mark.action)
            except ActionCodecV4Error as error:
                raise ProcessV2Active8PolicyError(
                    "the production marked law emitted an action outside ActionCodecV4"
                ) from error
            if family != mark.family_name:
                raise ProcessV2Active8PolicyError(
                    "the production mark family differs from ActionCodecV4"
                )
        self._results[state_sha256] = result
        self._results.move_to_end(state_sha256)
        while len(self._results) > self.cache_size:
            self._results.popitem(last=False)
        return result

    def __call__(
        self, addressed: AddressedPackedTrace, step_index: int
    ) -> ProcessV2CandidateEvidence:
        if type(step_index) is not int or not 0 <= step_index < len(addressed.trace.steps):
            raise ProcessV2Active8PolicyError(
                "the Process-V2 candidate step_index lies outside the trace"
            )
        classification = classify_process_v2_action(
            addressed.trace.steps[step_index],
            step_index=step_index,
            policy=self.policy,
        )
        if not classification.policy_eligible or classification.action_sha256 is None:
            raise ProcessV2Active8PolicyError(
                "the production candidate checker received a statically excluded action"
            )
        source = addressed.path.state_at(step_index)
        target = addressed.path.state_at(step_index + 1)
        source_sha256 = persistent_slot_state_sha256(source)
        target_sha256 = persistent_slot_state_sha256(target)
        target_key = canonical_state_key(target)
        result = self._successor_result(source)

        matching = [
            mark
            for mark in result.marked_law.marks
            if _action_identity(mark.executor_rule_name, mark.action)[0]
            == classification.action_sha256
        ]
        exact = sum(
            1
            for mark in matching
            if persistent_slot_state_sha256(
                self.system.apply(source, mark.executor_rule_name, mark.action)
            )
            == target_sha256
        )
        canonical = next(
            (row for row in result.batch.successors if row.key == target_key), None
        )
        reason: str | None = None
        if not matching:
            reason = REASON_MARK_ABSENT
        elif exact == 0:
            reason = REASON_MARK_NOT_EXACT
        elif canonical is None:
            reason = REASON_SUCCESSOR_ABSENT
        elif target_key == result.marked_law.source_key:
            reason = REASON_SUCCESSOR_VIRTUAL
        return ProcessV2CandidateEvidence(
            supported=reason is None,
            action_sha256=classification.action_sha256,
            source_state_sha256=source_sha256,
            target_state_sha256=target_sha256,
            canonical_successor_key=target_key,
            raw_mark_count=len(result.marked_law.marks),
            canonical_successor_count=len(result.batch.successors),
            matching_mark_count=len(matching),
            exact_successor_mark_count=exact,
            successor_alias_count=(0 if canonical is None else canonical.alias_count),
            exclusion_reason=reason,
        )


def evaluate_process_v2_active8_trace(
    addressed: AddressedPackedTrace,
    *,
    candidate_check: ProcessV2ExactCandidateCheck,
    policy: ProcessV2Active8Policy | None = None,
) -> ProcessV2Active8TraceDecision:
    """Classify every action and apply immutable all-or-nothing trace admission.

    A statically excluded action short-circuits the model entirely: the trace is
    already excluded, so evaluating its remaining actions would spend production
    enumeration on a decision that cannot change.  Every action still receives a
    decision row, so the census denominators stay complete.
    """

    selected = validate_process_v2_active8_policy(
        policy or build_process_v2_active8_policy()
    )
    if addressed.address.path_length != len(addressed.trace.steps):
        raise ProcessV2Active8PolicyError(
            "the Process-V2 trace address and action count disagree"
        )
    if addressed.path.path_length != addressed.address.path_length:
        raise ProcessV2Active8PolicyError(
            "the Process-V2 trace exact-state path and address disagree"
        )
    classifications = tuple(
        classify_process_v2_action(step, step_index=index, policy=selected)
        for index, step in enumerate(addressed.trace.steps)
    )
    exclusions: list[ProcessV2Active8Exclusion] = [
        ProcessV2Active8Exclusion(
            stage=STATIC_EXCLUSION_STAGE,
            step_index=item.step_index,
            executor_rule=item.executor_rule,
            model_family=item.model_family,
            reason=str(item.exclusion_reason),
        )
        for item in classifications
        if not item.policy_eligible
    ]
    decisions: list[ProcessV2ActionDecision] = []
    if exclusions:
        decisions.extend(
            ProcessV2ActionDecision(classification=item, candidate_evidence=None)
            for item in classifications
        )
    else:
        for classification in classifications:
            evidence = candidate_check(addressed, classification.step_index)
            if not isinstance(evidence, ProcessV2CandidateEvidence):
                raise ProcessV2Active8PolicyError(
                    "the exact Process-V2 candidate check returned another result type"
                )
            if evidence.action_sha256 != classification.action_sha256:
                raise ProcessV2Active8PolicyError(
                    "candidate evidence action identity differs from ActionCodecV4"
                )
            decisions.append(
                ProcessV2ActionDecision(
                    classification=classification, candidate_evidence=evidence
                )
            )
            if not evidence.supported:
                exclusions.append(
                    ProcessV2Active8Exclusion(
                        stage=DYNAMIC_EXCLUSION_STAGE,
                        step_index=classification.step_index,
                        executor_rule=classification.executor_rule,
                        model_family=classification.model_family,
                        reason=str(evidence.exclusion_reason),
                    )
                )
    accepted = not exclusions
    return ProcessV2Active8TraceDecision(
        trace_id=addressed.address.trace_id,
        policy_sha256=selected.policy_sha256,
        category=None if accepted else ACTIVE8_EXCLUDED,
        upstream_rejection_code=None,
        active8_status="accepted" if accepted else "excluded",
        emits_progress_rows=accepted,
        action_decisions=tuple(decisions),
        active8_exclusions=tuple(exclusions),
    )


def upstream_rejected_trace_decision(
    *,
    trace_id: str,
    rejection_code: str,
    policy: ProcessV2Active8Policy | None = None,
) -> ProcessV2Active8TraceDecision:
    """Carry an upstream rebind rejection through without evaluating anything.

    The trace was never an Active8 candidate.  It is retained so the census can
    account for it, and it produces no exclusion, no action decision and no
    progress row: an unevaluated trace and an evaluated-and-refused trace are
    different statements.
    """

    selected = validate_process_v2_active8_policy(
        policy or build_process_v2_active8_policy()
    )
    if not trace_id or trace_id.strip() != trace_id:
        raise ProcessV2Active8PolicyError("trace_id must be normalized nonempty text")
    if not rejection_code or rejection_code.strip() != rejection_code:
        raise ProcessV2Active8PolicyError(
            "an upstream rebind rejection code must be normalized nonempty text"
        )
    return ProcessV2Active8TraceDecision(
        trace_id=trace_id,
        policy_sha256=selected.policy_sha256,
        category=UPSTREAM_REJECTED,
        upstream_rejection_code=rejection_code,
        active8_status="not_evaluated",
        emits_progress_rows=False,
        action_decisions=(),
        active8_exclusions=(),
    )


__all__ = [
    "ACTIVE8_EXCLUSION_REASONS",
    "DYNAMIC_EXCLUSION_REASONS",
    "DYNAMIC_EXCLUSION_STAGE",
    "EXCLUSION_STAGES",
    "MAXIMUM_EXISTING_NEIGHBORS_FOR_ATOM_INSERT",
    "POLICY_SCHEMA",
    "POLICY_SCHEMA_VERSION",
    "POLICY_STATUS",
    "PROCESS_V2_ACTIVE8_EXECUTOR_RULES",
    "PROCESS_V2_ACTIVE8_FAMILIES",
    "PROCESS_V2_ACTIVE8_RULE_TO_FAMILY",
    "PROCESS_V2_EXPLICITLY_DISABLED_RULES",
    "REASON_EXPLICITLY_DISABLED",
    "REASON_MARK_ABSENT",
    "REASON_MARK_NOT_EXACT",
    "REASON_MULTI_NEIGHBOR_INSERT",
    "REASON_OUTSIDE_ACTION_CODEC",
    "REASON_SUCCESSOR_ABSENT",
    "REASON_SUCCESSOR_VIRTUAL",
    "SELECTION_UNIT",
    "STATIC_EXCLUSION_REASONS",
    "STATIC_EXCLUSION_STAGE",
    "ProcessV2Active8Exclusion",
    "ProcessV2Active8Policy",
    "ProcessV2Active8PolicyError",
    "ProcessV2Active8TraceDecision",
    "ProcessV2ActionDecision",
    "ProcessV2ActionClassification",
    "ProcessV2CandidateEvidence",
    "ProcessV2ExactCandidateCheck",
    "ProcessV2ProductionCandidateChecker",
    "build_process_v2_active8_policy",
    "classify_process_v2_action",
    "evaluate_process_v2_active8_trace",
    "upstream_rejected_trace_decision",
    "validate_process_v2_active8_policy",
    "validate_process_v2_active8_policy_payload",
]
