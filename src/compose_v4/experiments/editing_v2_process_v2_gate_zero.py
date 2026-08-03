"""Process-V2 Gate 0: structural evidence over the Process-V2 Active8 index.

WHY THIS IS A SEPARATE RUNNER
-----------------------------
``editing_v2_semantic_gate_zero`` is the V1 runner.  It is not parameterizable
into this one and must not be: it imports and calls
``resolve_editing_v2_semantic_active8_decision_source``, the V1 concrete decision
source, whose loader routes through ``resolve_editing_v2_semantic_active8_sources``
and revalidates the LIVE V1 process identity.  A Process-V2 payload is sealed
under the V2 identity, so that loader refuses it by construction.  This module
therefore consumes :class:`ProcessV2Active8Index` -- a protocol -- and never a
concrete class, which is the whole reason the seam is a protocol.

What is reused rather than rewritten: the canonical-serialization and self-hash
primitives come from the frozen Process-V2 schema module, the eight families come
from the frozen corpus contract, the executor-to-family alias comes from the
frozen V4 action codec, and every policy value -- the decision-eligible and
sealed partition roles, the required/conditional/separate-lane cell partition,
the strata bins, the failure-receipt limit -- is READ from the Process-V2 chain
contracts.  Nothing is transcribed from the V1 runner's frozen constants.

WHAT GATE 0 PROVES, AND WHAT IT REFUSES TO PROVE
------------------------------------------------
It proves identity and coverage over TRAIN-role traces only.  Validation,
controller-validation and final-test traces are hashed into per-role sealed
inventories and are never candidate-evaluated: no transition of a sealed-role
trace is ever requested from the index, so nothing about them can reach a cell
count, a family count, a threshold, or the PASS/FAIL rule.  The census totals
they participate in are reconciled against the index's OWN census, which counts
them too, so agreement there is a consistency check between two sources rather
than an exposure of held-out content.

A ``PASS`` is structural evidence, not permission.  Every artifact this module
publishes carries the complete Process-V2 authority vocabulary explicitly false,
in the PASS case exactly as in the FAIL case, and ``FAIL`` is a completed result
that is published rather than retried into a pass.

THE CONTRACT IS PINNED BY REBUILD, NOT BY A HARDCODED HASH
----------------------------------------------------------
The V1 runner carries ``FROZEN_CONTRACT_SHA256`` as a module constant.  This one
deliberately does not.  ``load_process_v2_chain_artifact`` rebuilds the contract
deterministically from the frozen policy registry and requires equality plus a
correct self-hash, which is strictly stronger than comparing one constant to
another -- and a constant-versus-constant check is self-consistent by
construction, which is how a stale binding comes to validate.  What is pinned
here is the ``contract_id``, a name that does not move when a policy is
reprojected.

WHAT THE HASH BINDINGS DEFEND AGAINST, AND WHAT THEY DO NOT
------------------------------------------------------------
Content hashes are not signatures.  Nothing here defends against a writer who
can replace every artifact and every trusted root together -- given that, the
whole chain can be made self-consistent at any value.  What the bindings do
defend against is the failure mode that actually occurs: evidence that is
corrupted, stale, produced by a superseded implementation, or sealed under an
identity other than the one it is being read under.  Every claim in this module
should be read at that strength and no higher.

``terminal`` MEANS THE SOURCE PROGRESS STATE
---------------------------------------------
The frozen teacher unit is an ``accepted_nonterminal_action_v4_transition``, and
the progress vocabulary is the Active8 stage's: a progress position is terminal
iff it is the last one on the path (``editing_v2_process_v2_active8_mapreduce``
publishes ``"terminal": index == address.path_length``).  A transition is taught
FROM its source position, so ``terminal`` describes that source and is false for
every accepted action -- including the last action of a trace, whose SUCCESSOR
is terminal.  The index publishes ``successor_is_terminal`` for the successor;
reading it as the assignment's ``terminal`` conflates the two and made the final
action of every accepted trace look like a teacher taught from a terminal state.

WHAT DERIVES THE STRATA, AND WHAT IS PROVEN ABOUT THE BINS
----------------------------------------------------------
The index preserves counts and does not classify; Gate 0 derives each evidence
stratum from those counts exactly ONCE, in :func:`_structural_view`.  A stratum
is therefore trustworthy because its INPUT is verified upstream, not because it
was compared afterwards against a value derived the same way -- that comparison
is what the per-teacher stratum requirement used to be, and it could not fail.
The property that is genuinely checkable without any transition is the bin
declaration itself, so :func:`strata_bin_defects` proves at load time that each
frozen stratum's bins tile ``[1, inf)`` exactly once, and the per-teacher
requirement now validates a SUPPLIED view's declared stratum against the frozen
bin it names rather than recomputing it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_semantic_capability_cells import (
    classify_action_family_context,
)
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.semantic_trace import RewriteStep
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_CENSUS_FIELDS,
    REJECTION_CATEGORIES,
    TRACE_KEY_FIELDS,
    ProcessV2Active8Index,
    ProcessV2TraceKey,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    ProcessV2SchemaError,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
    require_no_granted_authority,
)
from compose_v4.experiments.editing_gate_zero_semantic_contract import (
    PROCESS_V2_CONTRACT_SCHEMA,
    GateZeroSemanticContractError,
    load_gate_zero_semantic_contract,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    ACTIVE8_DECISION_RUNTIME,
    CAPABILITY_CELLS,
    DEVELOPMENT_CELL_ROLES,
    GATE_ZERO_STRUCTURAL,
    ProcessV2ChainError,
    load_process_v2_chain_artifact,
)
from compose_v4.rewrite.action_codec_v4 import (
    ACTIVE8_EXECUTOR_RULES,
    ActionCodecV4Error,
    canonical_family,
)
from compose_v4.rewrite.editing_v2_process_identity import editing_process_v2_identity

# ---- Identity ----

CONTRACT_RELATIVE_PATH = GATE_ZERO_STRUCTURAL
CONTRACT_ID = "editing_v2_process_v2_gate_zero_structural"

EVIDENCE_SCHEMA = "compose.editing_v2.process_v2.gate_zero_structural_evidence"
EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_STATUS = "PROCESS_V2_STRUCTURAL_EVIDENCE_COMPLETE_NO_DOWNSTREAM_AUTHORITY"
DECISION_SCHEMA = "compose.editing_v2.process_v2.gate_zero_structural_decision"
DECISION_SCHEMA_VERSION = 1
DECISION_STATUS = "PROCESS_V2_STRUCTURAL_RESULT_RECORDED_NO_DOWNSTREAM_AUTHORITY"
COMPLETION_SCHEMA = "compose.editing_v2.process_v2.gate_zero_structural_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_STATUS = "PROCESS_V2_COMPLETE_STRUCTURAL_ARTIFACTS_NO_DOWNSTREAM_AUTHORITY"

EVIDENCE_FILENAME = "STRUCTURAL_EVIDENCE.json"
DECISION_FILENAME = "STRUCTURAL_DECISION.json"
COMPLETION_FILENAME = "COMPLETE.json"

#: The fields Gate 0 requires of every row ``iter_resolved_traces()`` yields.
#: A row may carry more; it may not carry fewer.  The first three are the frozen
#: :data:`TRACE_KEY_FIELDS`, asserted below rather than retyped.
TRACE_ROW_FIELDS: tuple[str, ...] = (
    *TRACE_KEY_FIELDS,
    "partition_role",
    "rejection_category",
    "decision_sha256",
    "accepted_transition_count",
)

#: The fields Gate 0 requires of every accepted transition.  Again a lower
#: bound: the index publishes richer provenance and Gate 0 does not constrain it.
TRANSITION_FIELDS: tuple[str, ...] = (
    "assignment_sha256",
    "action_sha256",
    "canonical_successor_count",
    "canonical_successor_count_stratum",
    "capability_cell_id",
    "data_lane",
    "executor_rule",
    "family_context",
    "matching_mark_count",
    "model_family",
    "partition_role",
    "progress_index",
    "raw_mark_count",
    "raw_mark_count_stratum",
    "source_state_sha256",
    "successor_alias_multiplicity",
    "successor_alias_multiplicity_stratum",
    "target_state_sha256",
    "terminal",
)

#: The per-teacher structural requirements, in published order.  Each maps to a
#: check named ``every_decision_eligible_teacher_<name>`` and to a typed receipt.
TEACHER_REQUIREMENTS: tuple[str, ...] = (
    "has_positive_raw_mark_count",
    "has_positive_canonical_successor_count",
    "matches_exactly_one_action_v4_mark",
    "has_exact_successor_support",
    "has_exact_successor_alias_aggregation",
    "declares_the_frozen_evidence_strata",
    "declares_the_frozen_executor_family_alias",
)

#: The index's exact-successor evidence, read by ``has_exact_successor_support``.
#: Deliberately NOT in :data:`TRANSITION_FIELDS`: that tuple is the hard field
#: requirement on every view, and the decision-logic suite supplies stand-in
#: views that predate this field.  The production adapter always carries it --
#: :func:`_structural_view` requires it of the published transition -- so the
#: requirement is live on the path that decides anything.
SUPPORT_EVIDENCE_FIELD = "exact_successor_mark_count"

_STRATUM_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("raw_mark_count", "raw_mark_count_stratum", "raw_mark_strata"),
    (
        "canonical_successor_count",
        "canonical_successor_count_stratum",
        "canonical_successor_strata",
    ),
    (
        "successor_alias_multiplicity",
        "successor_alias_multiplicity_stratum",
        "successor_alias_strata",
    ),
)


class ProcessV2GateZeroError(RuntimeError):
    """A Process-V2 Gate-0 contract, index, or artifact invariant is invalid."""


# ---- Pure calculations ----


def _bytes(value: object, *, newline: bool = False) -> bytes:
    return canonical_bytes(value) + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return canonical_sha256(value)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessV2GateZeroError(f"{field} must be a lowercase SHA-256")
    return value


def _require_count(value: object, *, field: str) -> int:
    """An exact non-negative ``int``.  ``bool`` is an ``int`` and is not a count."""

    if type(value) is not int or value < 0:
        raise ProcessV2GateZeroError(
            f"{field} is {value!r}; a Gate-0 count must be an exact non-negative int"
        )
    return value


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ProcessV2GateZeroError(f"{field} is absent or invalid: {path}") from error
    if not isinstance(value, dict):
        raise ProcessV2GateZeroError(f"{field} must be an object")
    return value, raw


#: The smallest count any frozen evidence stratum may describe.  A stratum
#: describes a POSITIVE count -- a teacher with no raw marks, no canonical
#: successors or no successor aliases is unsupported, which is a per-teacher
#: requirement failure rather than a stratum.
MINIMUM_STRATIFIED_COUNT = 1


def strata_bin_defects(bins: tuple[tuple[str, int, int | None], ...]) -> tuple[str, ...]:
    """Every way one frozen stratum's bins fail to tile ``[1, inf)`` exactly once.

    This is the part of the stratum story that is a real property, checkable
    against nothing but the contract: completeness and disjointness of the bin
    set.  ``_stratum`` in the capability-cell module raises when a value resolves
    to other than one bin, which discovers the same defect one row at a time and
    only for the counts a corpus happens to contain; proving it here makes it a
    property of the declaration, so the derivation in :func:`_structural_view`
    can be a single total lookup.

    Returns the defects rather than raising so a caller can report all of them.
    """

    defects: list[str] = []
    if not bins:
        return ("declares no bins",)
    identifiers = [item[0] for item in bins]
    if len(set(identifiers)) != len(identifiers):
        defects.append("repeats a stratum id")
    ordered = sorted(bins, key=lambda item: item[1])
    if ordered[0][1] != MINIMUM_STRATIFIED_COUNT:
        defects.append(f"does not start at {MINIMUM_STRATIFIED_COUNT}")
    for position, (identifier, minimum, maximum) in enumerate(ordered):
        last = position == len(ordered) - 1
        if maximum is None:
            if not last:
                defects.append(f"{identifier!r} is unbounded above but is not the last bin")
        elif maximum < minimum:
            defects.append(f"{identifier!r} is empty")
        elif last:
            defects.append(f"{identifier!r} is the last bin but leaves counts above it unbinned")
        if not last:
            following = ordered[position + 1]
            if maximum is not None and following[1] != maximum + 1:
                defects.append(
                    f"{identifier!r} and {following[0]!r} do not meet: a count between them "
                    "falls in no bin, or in both"
                )
    return tuple(defects)


def _structural_view(
    transition: Mapping[str, Any], *, contract: FrozenProcessV2GateZeroContract
) -> dict[str, Any]:
    """Derive Gate 0's structural view of one published transition.

    The index publishes RAW evidence -- the exact persistent-slot states, the
    ActionV4 record, and the raw counts -- and deliberately does not classify.
    That split is the handoff's: the Active8 stage preserves counts, and Gate 0
    evaluates every required semantic cell and publishes the strata.  This
    function is that evaluation, and it is the only place it happens.

    It was not always here.  Gate 0 was first written to CONSUME a classified
    transition, so it required `capability_cell_id`, `family_context` and three
    strata that no index publishes.  Both suites passed and the chain did not
    join, because a seam that names a protocol but not the row and the transition
    crossing it has not named them.

    Nothing is recomputed that the index already decided: the family comes from
    the codec's own alias, the counts are carried through, and the exact states
    are the cached arrays rather than anything re-derived.
    """

    rule = str(transition["executor_rule"])
    decoded_rule, action = decode_action(transition["action_record"])
    if decoded_rule != rule:
        raise ProcessV2GateZeroError(
            f"a transition declares executor rule {rule!r} but its ActionV4 record "
            f"decodes to {decoded_rule!r}"
        )
    family, context = classify_action_family_context(
        transition["source_state"],
        transition["successor_state"],
        RewriteStep(rule_name=decoded_rule, action=action),
    )
    if family != str(transition["model_family"]):
        raise ProcessV2GateZeroError(
            f"a transition declares model family {transition['model_family']!r} but its "
            f"action classifies as {family!r}"
        )
    view = {
        key: transition[key]
        for key in (
            "v1_task_identity_sha256",
            "entry_index",
            "trace_id",
            "action_sha256",
            "data_lane",
            "executor_rule",
            "model_family",
            "source_state_sha256",
            "raw_mark_count",
            "canonical_successor_count",
            "matching_mark_count",
            SUPPORT_EVIDENCE_FIELD,
        )
    }
    view["partition_role"] = str(transition["split"])
    view["progress_index"] = int(transition["source_progress_index"])
    view["target_state_sha256"] = str(transition["successor_state_sha256"])
    # `terminal` is the SOURCE progress position, not the successor. A published
    # transition exists exactly because its source has an outgoing step, so no
    # accepted action is ever taught from a terminal position -- including the
    # last action of a trace, for which `successor_is_terminal` is true and is
    # deliberately not read here. Reading it made every accepted trace's final
    # action look like a terminal-source teacher.
    view["terminal"] = False
    view["successor_alias_multiplicity"] = int(transition["successor_alias_count"])
    view["family_context"] = context
    view["capability_cell_id"] = f"{contract.namespace}:{family}:{context}"
    for source_field, stratum_field, _evidence_key in _STRATUM_FIELDS:
        # The ONE derivation of each evidence stratum, and Gate 0 is its author:
        # the index preserves counts and does not classify. It is a single total
        # lookup rather than the raising `_stratum`, because `strata_bin_defects`
        # has already proven at load time that the frozen bins tile [1, inf)
        # exactly once -- so "resolves to exactly one bin" is a property of the
        # declaration rather than something rediscovered per row. A count outside
        # every bin yields None here and fails a per-teacher requirement, instead
        # of raising out of the gate.
        #
        # The contract keys its bins by the SOURCE field; `_STRATUM_FIELDS`'s
        # third element names the evidence block the counts are published under,
        # which is a different vocabulary.
        view[stratum_field] = stratum_for(
            int(view[source_field]), contract.strata_bins[source_field]
        )
    # The assignment identity: what this transition structurally IS, and nothing
    # about when or by which run it was observed. Two runs over the same corpus
    # must agree on it, and two different transitions must not collide.
    view["assignment_sha256"] = canonical_sha256(
        {key: view[key] for key in sorted(view) if key != "assignment_sha256"}
    )
    return view


def _empty_cell_row(cell_id: str, family: str, context: str) -> dict[str, Any]:
    return {
        "capability_cell_id": cell_id,
        "model_family": family,
        "family_context": context,
        "teacher_count": 0,
        "unique_source_state_count": 0,
        "unique_target_state_count": 0,
        "unique_action_count": 0,
        "raw_candidate_mark_sum": 0,
        "canonical_candidate_successor_sum": 0,
        "matching_candidate_mark_sum": 0,
        "successor_alias_sum": 0,
        "raw_mark_strata": {},
        "canonical_successor_strata": {},
        "successor_alias_strata": {},
        "lane_counts": {},
        "executor_rule_counts": {},
    }


def stratum_for(value: int, bins: tuple[tuple[str, int, int | None], ...]) -> str | None:
    """The frozen bin holding ``value``, or ``None`` when no bin does.

    The bins come from the Process-V2 capability-cell contract, so recomputing a
    declared stratum here compares the index against a frozen policy rather than
    against this module -- an expectation rebuilt from the code under test cannot
    fail.
    """

    for identifier, minimum, maximum in bins:
        if value >= minimum and (maximum is None or value <= maximum):
            return identifier
    return None


# ---- The frozen contract and its exact typed parents ----


@dataclass(frozen=True, slots=True)
class FrozenProcessV2GateZeroContract:
    """The validated Process-V2 Gate-0 contract plus every parent it declares."""

    source: Path
    payload: dict[str, Any]
    file_sha256: str
    capability_cells: dict[str, Any]
    development_cell_roles: dict[str, Any]
    decision_runtime: dict[str, Any]
    semantic_model_process: dict[str, Any]
    semantic_process: dict[str, Any]
    namespace: str
    registered_cell_ids: tuple[str, ...]
    family_contexts: tuple[tuple[str, tuple[str, ...]], ...]
    required_cell_ids: tuple[str, ...]
    conditional_cell_ids: tuple[str, ...]
    separate_lane_cell_ids: tuple[str, ...]
    decision_roles: tuple[str, ...]
    sealed_roles: tuple[str, ...]
    data_lanes: tuple[str, ...]
    partition_roles: tuple[str, ...]
    strata_bins: Mapping[str, tuple[tuple[str, int, int | None], ...]]
    failure_receipt_limit: int
    process_identity_sha256: str

    @property
    def sha256(self) -> str:
        return str(self.payload["contract_sha256"])


def _edge(contract_payload: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    parents = contract_payload.get("parents")
    if not isinstance(parents, Mapping) or role not in parents:
        raise ProcessV2GateZeroError(
            f"the Process-V2 Gate-0 contract declares no {role!r} parent edge"
        )
    return parents[role]


def _chain_parent(
    contract_payload: Mapping[str, Any], role: str, *, expected_target: str, repo_root: Path
) -> dict[str, Any]:
    """Load one chain-member parent and require the declared pointers to hold."""

    edge = _edge(contract_payload, role)
    target = str(edge["semantic"]["target"])
    if target != expected_target:
        raise ProcessV2GateZeroError(
            f"Gate-0 parent {role!r} points at {target!r}, not {expected_target!r}"
        )
    try:
        parent = load_process_v2_chain_artifact(target, repo_root=repo_root)
    except ProcessV2ChainError as error:
        raise ProcessV2GateZeroError(
            f"Gate-0 parent {role!r} does not validate as a Process-V2 chain artifact: {error}"
        ) from error
    if parent.get("contract_sha256") != edge["semantic"]["sha256"]:
        raise ProcessV2GateZeroError(f"Gate-0 parent {role!r} semantic identity disagrees")
    if _file_sha(repo_root / target) != edge["physical"]["sha256"]:
        raise ProcessV2GateZeroError(f"Gate-0 parent {role!r} physical bytes disagree")
    return parent


def _external_parent(
    contract_payload: Mapping[str, Any], role: str, *, repo_root: Path
) -> tuple[dict[str, Any], str]:
    edge = _edge(contract_payload, role)
    target = str(edge["semantic"]["target"])
    payload, raw = _load_json(repo_root / target, field=f"Gate-0 parent {role}")
    if hashlib.sha256(raw).hexdigest() != edge["physical"]["sha256"]:
        raise ProcessV2GateZeroError(f"Gate-0 parent {role!r} physical bytes disagree")
    if payload.get("contract_sha256") != edge["semantic"]["sha256"]:
        raise ProcessV2GateZeroError(f"Gate-0 parent {role!r} semantic identity disagrees")
    return payload, target


def _bins(value: object, *, field: str) -> tuple[tuple[str, int, int | None], ...]:
    if not isinstance(value, Mapping) or not isinstance(value.get("bins"), list):
        raise ProcessV2GateZeroError(f"frozen evidence stratum {field!r} declares no bins")
    parsed: list[tuple[str, int, int | None]] = []
    for entry in value["bins"]:
        if not isinstance(entry, Mapping) or set(entry) != {"id", "minimum", "maximum"}:
            raise ProcessV2GateZeroError(f"frozen evidence stratum {field!r} has a malformed bin")
        maximum = entry["maximum"]
        if maximum is not None:
            _require_count(maximum, field=f"{field}.maximum")
        parsed.append(
            (str(entry["id"]), _require_count(entry["minimum"], field=f"{field}.minimum"), maximum)
        )
    if not parsed:
        raise ProcessV2GateZeroError(f"frozen evidence stratum {field!r} declares no bins")
    defects = strata_bin_defects(tuple(parsed))
    if defects:
        raise ProcessV2GateZeroError(
            f"frozen evidence stratum {field!r} does not tile the count range exactly once: "
            + "; ".join(defects)
        )
    return tuple(parsed)


def load_process_v2_gate_zero_contract(
    *, repo_root: Path | None = None
) -> FrozenProcessV2GateZeroContract:
    """Load the schema-3 Process-V2 Gate-0 contract and its exact typed parents.

    Every parent is loaded through its own validating loader and then required to
    match the physical and semantic hash the Gate-0 contract declares for it, so a
    parent that was edited without republishing the chain fails here rather than
    silently changing what Gate 0 requires.
    """

    root = Path(repo_root or _repository_root()).resolve()
    source = root / CONTRACT_RELATIVE_PATH
    try:
        payload = load_process_v2_chain_artifact(CONTRACT_RELATIVE_PATH, repo_root=root)
    except ProcessV2ChainError as error:
        raise ProcessV2GateZeroError(
            f"the Process-V2 Gate-0 structural contract does not validate: {error}"
        ) from error
    if payload.get("contract_id") != CONTRACT_ID:
        raise ProcessV2GateZeroError(
            f"contract_id is {payload.get('contract_id')!r}, not {CONTRACT_ID!r}"
        )
    try:
        require_no_granted_authority(payload, label=CONTRACT_ID)
        require_authority_false(payload, label=CONTRACT_ID)
    except ProcessV2SchemaError as error:
        raise ProcessV2GateZeroError(str(error)) from error

    cells = _chain_parent(
        payload, "capability_cell_registry", expected_target=CAPABILITY_CELLS, repo_root=root
    )
    roles = _chain_parent(
        payload, "development_cell_roles", expected_target=DEVELOPMENT_CELL_ROLES, repo_root=root
    )
    runtime = _chain_parent(
        payload, "decision_runtime", expected_target=ACTIVE8_DECISION_RUNTIME, repo_root=root
    )
    model_process, model_process_target = _external_parent(
        payload, "semantic_model_process", repo_root=root
    )
    semantic_process, _ = _external_parent(payload, "semantic_process", repo_root=root)
    if model_process.get("schema") != PROCESS_V2_CONTRACT_SCHEMA:
        raise ProcessV2GateZeroError(
            "the Gate-0 model/process parent is not the Process-V2 variant; a V1 body "
            "can only express a V1 model identity"
        )
    try:
        load_gate_zero_semantic_contract(root / model_process_target)
    except GateZeroSemanticContractError as error:
        raise ProcessV2GateZeroError(
            f"the Gate-0 model/process parent does not validate: {error}"
        ) from error

    identity = payload["process_identity"]["process_identity_sha256"]
    live = str(editing_process_v2_identity()["process_identity_sha256"])
    if identity != live:
        raise ProcessV2GateZeroError(
            f"the Gate-0 contract binds process identity {identity!r}, not the live "
            f"Process-V2 identity {live!r}"
        )

    checks = payload["structural_checks"]
    decision_roles = tuple(checks["decision_eligible_partition_roles"])
    sealed_roles = tuple(checks["sealed_nondecision_partition_roles"])
    if decision_roles != ("train",) or set(decision_roles) & set(sealed_roles):
        raise ProcessV2GateZeroError("the decision-eligible partition policy disagrees")
    if checks.get("legacy_action_v2_evidence") != "forbidden":
        raise ProcessV2GateZeroError(
            "the Gate-0 contract does not forbid legacy Action-V2 evidence"
        )
    for required in (
        "require_every_active8_family",
        "require_every_teacher_supported",
        "require_positive_raw_mark_count",
        "require_positive_canonical_successor_count",
        "require_positive_successor_alias_count",
        "require_exactly_one_matching_mark",
        "require_one_assignment_per_accepted_action",
        "require_zero_terminal_assignments",
    ):
        if checks.get(required) is not True:
            raise ProcessV2GateZeroError(f"the Gate-0 contract does not require {required!r}")
    policy = payload["decision_policy"]
    granted = sorted(
        key for key, value in policy.items() if key.startswith("pass_grants_") and value
    )
    if granted:
        raise ProcessV2GateZeroError(f"the Gate-0 decision policy grants authority at {granted}")

    namespace = str(cells["cell_identity_policy"]["namespace"])
    family_contexts = tuple(
        (family, tuple(cells["family_contexts"][family])) for family in ACTIVE8_FAMILIES
    )
    if set(cells["family_contexts"]) != set(ACTIVE8_FAMILIES):
        raise ProcessV2GateZeroError("the capability registry does not cover the Active8 families")
    registered = tuple(
        f"{namespace}:{family}:{context}"
        for family, contexts in family_contexts
        for context in contexts
    )
    required_cells = tuple(roles["required_cell_ids"])
    conditional_cells = tuple(roles["conditional_cell_ids"])
    separate_cells = tuple(roles["separate_lane_cell_ids"])
    partition = required_cells + conditional_cells + separate_cells
    if sorted(partition) != sorted(registered) or len(set(partition)) != len(partition):
        raise ProcessV2GateZeroError(
            "the development cell-role policy does not partition the registered cells exactly once"
        )
    return FrozenProcessV2GateZeroContract(
        source=source,
        payload=payload,
        file_sha256=_file_sha(source),
        capability_cells=cells,
        development_cell_roles=roles,
        decision_runtime=runtime,
        semantic_model_process=model_process,
        semantic_process=semantic_process,
        namespace=namespace,
        registered_cell_ids=registered,
        family_contexts=family_contexts,
        required_cell_ids=required_cells,
        conditional_cell_ids=conditional_cells,
        separate_lane_cell_ids=separate_cells,
        decision_roles=decision_roles,
        sealed_roles=sealed_roles,
        data_lanes=tuple(cells["bindings"]["data_lanes"]),
        partition_roles=tuple(cells["bindings"]["partition_roles"]),
        strata_bins={
            field: _bins(cells["exact_evidence_strata"][field], field=field)
            for field, _stratum, _row in _STRATUM_FIELDS
        },
        failure_receipt_limit=_require_count(
            checks["classification_failure_receipt_limit"],
            field="classification_failure_receipt_limit",
        ),
        process_identity_sha256=live,
    )


# ---- Structural evidence ----


def _trace_key(row: Mapping[str, Any]) -> ProcessV2TraceKey:
    missing = [field for field in TRACE_ROW_FIELDS if field not in row]
    if missing:
        raise ProcessV2GateZeroError(f"a resolved trace row omits {sorted(missing)}")
    task = _require_sha(row["v1_task_identity_sha256"], field="v1_task_identity_sha256")
    entry = _require_count(row["entry_index"], field="entry_index")
    trace_id = row["trace_id"]
    if not isinstance(trace_id, str) or not trace_id:
        raise ProcessV2GateZeroError("a resolved trace row carries an empty trace_id")
    return (task, entry, trace_id)


#: The keyword the bulk transition stream must accept, and the whole reason Gate 0
#: probes a signature rather than trusting a method name.
ROLE_FILTER_PARAMETER = "partition_roles"

_BULK_STREAM_REQUIREMENT = (
    "iter_accepted_transitions(*, partition_roles: Collection[str]) -> "
    "Iterator[Mapping[str, Any]], which selects tasks by their partition role "
    "BEFORE reading a cache chunk, streams each selected chunk exactly once, and "
    "validates each transition against the task, row and trace it already holds"
)


def _require_role_filtered_bulk_transitions(index: object) -> Callable[..., Any]:
    """The bulk seam, probed by SIGNATURE rather than by name.

    Gate 0 must never open the cache chunk of a sealed partition role, and a bulk
    stream that cannot be restricted before it opens one can only be filtered
    afterwards -- by which point the held-out molecular states have already been
    decoded.  "Never counted" is not "never opened", and the frozen contract says
    the second.

    A name-only check would pass for an unfiltered stream and then fail at the
    call with a bare ``TypeError``, so the keyword is required explicitly: a
    capability check that cannot distinguish the capability is not one.
    """

    stream = getattr(index, "iter_accepted_transitions", None)
    if not callable(stream):
        raise ProcessV2GateZeroError(
            "the decision index publishes no bulk accepted-transition stream. Gate 0 "
            "requires " + _BULK_STREAM_REQUIREMENT
        )
    try:
        parameters = inspect.signature(stream).parameters
    except (TypeError, ValueError) as error:  # pragma: no cover - exotic callables
        raise ProcessV2GateZeroError(
            "the decision index's accepted-transition stream has no inspectable "
            "signature, so its role filter cannot be established. Gate 0 requires "
            + _BULK_STREAM_REQUIREMENT
        ) from error
    parameter = parameters.get(ROLE_FILTER_PARAMETER)
    if parameter is None or parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise ProcessV2GateZeroError(
            "the decision index's accepted-transition stream takes no keyword-only "
            f"{ROLE_FILTER_PARAMETER!r}, so it cannot be restricted before it opens a "
            "cache chunk, and Gate 0 will not decode a sealed partition role in order "
            "to discard it afterwards. Gate 0 requires " + _BULK_STREAM_REQUIREMENT
        )
    return stream


def _transition_trace_key(transition: Mapping[str, Any]) -> ProcessV2TraceKey:
    """The trace a streamed transition belongs to, from the transition itself.

    The bulk stream is not addressed per trace, so the join back to the resolved
    row is by the transition's own whole key -- never by stream position, and
    never by a bare ``trace_id``, which is unique only within one V1 task.
    """

    missing = [field for field in TRACE_KEY_FIELDS if field not in transition]
    if missing:
        raise ProcessV2GateZeroError(
            f"a streamed accepted transition omits its trace key {sorted(missing)}"
        )
    task = _require_sha(transition["v1_task_identity_sha256"], field="v1_task_identity_sha256")
    entry = _require_count(transition["entry_index"], field="entry_index")
    trace_id = transition["trace_id"]
    if not isinstance(trace_id, str) or not trace_id:
        raise ProcessV2GateZeroError("a streamed accepted transition carries an empty trace_id")
    return (task, entry, trace_id)


def _teacher_failures(
    transition: Mapping[str, Any], *, contract: FrozenProcessV2GateZeroContract
) -> tuple[str, ...]:
    """Every frozen per-teacher requirement this transition violates."""

    failed: list[str] = []
    raw = _require_count(transition["raw_mark_count"], field="raw_mark_count")
    successors = _require_count(
        transition["canonical_successor_count"], field="canonical_successor_count"
    )
    matching = _require_count(transition["matching_mark_count"], field="matching_mark_count")
    aliases = _require_count(
        transition["successor_alias_multiplicity"], field="successor_alias_multiplicity"
    )
    if raw < 1:
        failed.append("has_positive_raw_mark_count")
    if successors < 1:
        failed.append("has_positive_canonical_successor_count")
    if matching != 1:
        failed.append("matches_exactly_one_action_v4_mark")
    # `require_every_teacher_supported`. The frozen support predicate lives in
    # `ProcessV2CandidateEvidence.__post_init__`: supported means no exclusion
    # reason AND positive matching, exact-successor, alias and canonical-
    # successor counts. Four of those five are count-expressible and are decided
    # here. The `supported` boolean and its exclusion reason do NOT cross the
    # seam -- an accepted transition carries neither -- so a row whose boolean
    # was flipped without moving a count is invisible to Gate 0 and must be
    # refused upstream; this enforces the part the published evidence supports
    # rather than claiming the whole predicate.
    exact = transition.get(SUPPORT_EVIDENCE_FIELD)
    supported = successors >= 1 and matching >= 1 and aliases >= 1
    if exact is not None:
        # The distinctive arm: a matching mark that actually produces the
        # teacher's exact successor state. Nothing else Gate 0 reads implies it.
        exact_marks = _require_count(exact, field=SUPPORT_EVIDENCE_FIELD)
        supported = supported and 1 <= exact_marks <= matching
    if not supported:
        failed.append("has_exact_successor_support")
    # Alias aggregation: the teacher's canonical successor fiber is a non-empty
    # subset of the raw marks, the canonical successors are an aggregation of
    # those marks, and the matching mark lies inside the teacher's own fiber.
    if not 1 <= aliases <= raw or successors > raw or matching > aliases:
        failed.append("has_exact_successor_alias_aggregation")
    for count_field, stratum_field, _row_field in _STRATUM_FIELDS:
        # A BOUNDARY check on the supplied view, not corroboration of the count:
        # the declared stratum must NAME a frozen bin of this field and that
        # bin's declared range must contain the count. It is not `stratum_for`
        # recomputed and compared -- that expectation came from the same input as
        # its observation and could not fail. For the production adapter, which
        # derives the stratum itself, this is satisfied by construction; the real
        # property of the bins is proven once, at load, by `strata_bin_defects`.
        observed = _require_count(transition[count_field], field=count_field)
        declared = transition[stratum_field]
        named = {
            identifier: (minimum, maximum)
            for identifier, minimum, maximum in contract.strata_bins[count_field]
        }.get(declared if isinstance(declared, str) else "")
        if named is None or not (
            observed >= named[0] and (named[1] is None or observed <= named[1])
        ):
            failed.append("declares_the_frozen_evidence_strata")
            break
    try:
        aliased = canonical_family(str(transition["executor_rule"]))
    except ActionCodecV4Error:
        aliased = None
    if aliased is None or aliased != transition["model_family"]:
        failed.append("declares_the_frozen_executor_family_alias")
    return tuple(failed)


def _receipt(
    *,
    key: ProcessV2TraceKey,
    decision_sha256: str,
    transition: Mapping[str, Any] | None,
    failure_type: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "v1_task_identity_sha256": key[0],
        "entry_index": key[1],
        "trace_id": key[2],
        "decision_sha256": decision_sha256,
        "progress_index": None if transition is None else transition.get("progress_index"),
        "model_family": None if transition is None else transition.get("model_family"),
        "capability_cell_id": None if transition is None else transition.get("capability_cell_id"),
        "failure_type": failure_type,
        "reason": reason,
    }


def build_process_v2_gate_zero_evidence(
    index: ProcessV2Active8Index,
    *,
    contract: FrozenProcessV2GateZeroContract,
    view: Callable[..., Mapping[str, Any]] = _structural_view,
) -> dict[str, Any]:
    """Read every resolved trace, decode only decision-eligible chunks, seal the rest.

    TWO STREAMS, NOT A LOOKUP PER TRACE
    -----------------------------------
    Resolved decision metadata is read for EVERY partition role, because the
    census and the sealed-role inventories must be complete.  Molecular state is
    decoded for the decision-eligible roles ONLY, and the restriction is handed
    to the index BEFORE it opens a cache chunk, so a sealed role's states are
    never decoded rather than decoded and discarded.

    What this replaced was a per-trace ``accepted_transitions_for`` lookup, whose
    own docstring says a loop over it "would decode one chunk per trace and
    reinstate exactly the triangular rescan the chunk cache exists to remove".
    Measured on the real chain it cost ``T + 2*sum(L)`` chunk decodes for ``T``
    decision-eligible traces of length ``L``: one for the lookup, one for the
    index's internal validation and one for Gate 0's own re-validation, because
    ``validate_accepted_transition`` is ITSELF a point lookup.  Gate 0 therefore
    does not call it from the bulk path -- doing so reinstates the cost the bulk
    path exists to remove.  Validation is not skipped; it moves inside the
    stream, where the task, row and trace are already in hand.

    Gate 0 consumes that evidence and does not re-derive it.  It classifies each
    teacher's action into its capability cell and derives the evidence strata --
    the handoff the contract assigns it -- and never re-enumerates a successor
    fiber, which is the Active8 stage's to own and to have verified.

    Memory is bounded in the TRANSITION dimension: the only per-row state kept is
    the per-cell aggregate, the unique-identity sets the contract requires Gate 0
    to publish, and one binding per decision-eligible TRACE for the join back to
    its resolved row -- the same order as the duplicate-key set that was already
    retained, and independent of trace length.

    ``view`` is the adapter from a PUBLISHED transition to the structural view
    this function decides over, and it defaults to the real one.  It is a
    parameter because the two are separately testable: the adapter needs exact
    molecular states and a real ActionV4 record, so a fixture that wants to
    drive the DECISION logic over controlled inputs supplies views directly
    rather than fabricating chemistry that happens to classify where it wants.
    A test that substitutes it is testing the gate, not the boundary; the
    boundary is proven by the end-to-end chain, and a guard asserts this default
    is the real adapter so substitution cannot become the normal path.
    """

    if not isinstance(index, ProcessV2Active8Index):
        raise ProcessV2GateZeroError(
            "the supplied decision index does not satisfy ProcessV2Active8Index; Gate 0 "
            "consumes the frozen protocol and never a concrete decision source"
        )
    bulk_transitions = _require_role_filtered_bulk_transitions(index)
    decision_roles = set(contract.decision_roles)
    sealed_roles = contract.sealed_roles
    allowed_roles = decision_roles | set(sealed_roles)
    if not allowed_roles.issubset(contract.partition_roles):
        raise ProcessV2GateZeroError("a declared partition role is outside the frozen bindings")

    cell_rows: dict[str, dict[str, Any]] = {}
    cell_sets: dict[str, dict[str, set[str]]] = {}
    for family, contexts in contract.family_contexts:
        for context in contexts:
            cell_id = f"{contract.namespace}:{family}:{context}"
            cell_rows[cell_id] = _empty_cell_row(cell_id, family, context)
            cell_sets[cell_id] = {"sources": set(), "targets": set(), "actions": set()}

    sealed_streams = {role: hashlib.sha256() for role in sealed_roles}
    accepted_trace_stream = hashlib.sha256()
    assignment_stream = hashlib.sha256()
    family_counts: Counter[str] = Counter()
    executor_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    requirement_failures: Counter[str] = Counter()
    receipts: list[dict[str, Any]] = []
    classification_failures = 0
    seen_keys: set[ProcessV2TraceKey] = set()
    assignment_ids: set[str] = set()
    transition_keys: set[tuple[str, int, str, int]] = set()
    unique_sources: set[str] = set()
    unique_targets: set[str] = set()

    resolved_traces = 0
    upstream_rejected = 0
    active8_excluded = 0
    active8_accepted = 0
    decision_traces = 0
    decision_transitions = 0
    terminal_assignments = 0
    declared_transition_total = 0
    #: One binding per decision-eligible ACCEPTED trace: what the resolved row
    #: declared, so a streamed transition can be joined back to it without a
    #: second lookup.  Bounded by the number of eligible traces, not by their
    #: length and not by the corpus -- sealed and rejected traces never enter it.
    eligible: dict[ProcessV2TraceKey, tuple[str, int, str]] = {}
    observed_transitions: Counter[ProcessV2TraceKey] = Counter()

    for row in index.iter_resolved_traces():
        resolved_traces += 1
        key = _trace_key(row)
        if key in seen_keys:
            raise ProcessV2GateZeroError(f"the resolved trace stream repeats the key {key!r}")
        seen_keys.add(key)
        role = row["partition_role"]
        if role not in allowed_roles:
            raise ProcessV2GateZeroError(f"a resolved trace uses the undeclared role {role!r}")
        category = row["rejection_category"]
        if category is not None and category not in REJECTION_CATEGORIES:
            raise ProcessV2GateZeroError(
                f"a resolved trace declares the unknown category {category!r}"
            )
        decision_sha256 = _require_sha(row["decision_sha256"], field="decision_sha256")
        declared = _require_count(
            row["accepted_transition_count"], field="accepted_transition_count"
        )
        if category is not None:
            if declared:
                raise ProcessV2GateZeroError(
                    "a rejected or excluded trace declares accepted transitions; absence of "
                    "transitions is the answer, not an error"
                )
            if category == REJECTION_CATEGORIES[0]:
                upstream_rejected += 1
            else:
                active8_excluded += 1
            continue

        active8_accepted += 1
        stamp = _bytes(
            {"trace_key": list(key), "decision_sha256": decision_sha256}, newline=True
        )
        accepted_trace_stream.update(stamp)
        if role not in decision_roles:
            # Sealed: hashed from decision METADATA, and its molecular state is
            # never decoded.  The role filter below is handed to the index before
            # it opens a cache chunk, so this is "never opened", not "opened and
            # then not counted" -- the two differ exactly when it matters.
            sealed_streams[role].update(stamp)
            continue

        decision_traces += 1
        declared_transition_total += declared
        eligible[key] = (decision_sha256, declared, role)

    for published in bulk_transitions(partition_roles=tuple(contract.decision_roles)):
        key = _transition_trace_key(published)
        binding = eligible.get(key)
        if binding is None:
            # The role filter is asserted against the data, not assumed from the
            # argument: a stream that yielded a sealed or rejected trace's
            # transition would already have decoded a held-out state.
            raise ProcessV2GateZeroError(
                f"the role-filtered transition stream yielded trace {key!r}, which is not a "
                "decision-eligible accepted trace"
            )
        decision_sha256, _declared, role = binding
        observed_transitions[key] += 1
        decision_transitions += 1
        transition = view(published, contract=contract)
        missing = [field for field in TRANSITION_FIELDS if field not in transition]
        if missing:
            raise ProcessV2GateZeroError(f"an accepted transition view omits {sorted(missing)}")
        if transition["partition_role"] != role:
            raise ProcessV2GateZeroError(
                "an accepted transition declares a partition role its trace does not"
            )
        if transition["partition_role"] not in decision_roles:
            raise ProcessV2GateZeroError(
                "an accepted transition of a sealed partition role reached the decision path"
            )
        if transition["terminal"] is True:
            terminal_assignments += 1
        progress_index = _require_count(transition["progress_index"], field="progress_index")
        assignment_sha256 = _require_sha(transition["assignment_sha256"], field="assignment_sha256")
        transition_key = (*key, progress_index)
        if transition_key in transition_keys or assignment_sha256 in assignment_ids:
            raise ProcessV2GateZeroError(
                "the accepted transition stream contains a duplicate structural assignment"
            )
        cell_id = transition["capability_cell_id"]
        row_for_cell = cell_rows.get(cell_id)
        if row_for_cell is None:
            classification_failures += 1
            if len(receipts) < contract.failure_receipt_limit:
                receipts.append(
                    _receipt(
                        key=key,
                        decision_sha256=decision_sha256,
                        transition=transition,
                        failure_type="capability_cell_outside_the_frozen_registry",
                        reason=f"{cell_id!r} is not a registered Process-V2 capability cell",
                    )
                )
            continue
        if transition["data_lane"] not in contract.data_lanes:
            raise ProcessV2GateZeroError(
                f"an accepted transition declares the undeclared lane {transition['data_lane']!r}"
            )
        transition_keys.add(transition_key)
        assignment_ids.add(assignment_sha256)
        assignment_stream.update(
            _bytes(
                {
                    "trace_key": list(key),
                    "progress_index": progress_index,
                    "assignment_sha256": assignment_sha256,
                },
                newline=True,
            )
        )
        for failure in _teacher_failures(transition, contract=contract):
            requirement_failures[failure] += 1
            if len(receipts) < contract.failure_receipt_limit:
                receipts.append(
                    _receipt(
                        key=key,
                        decision_sha256=decision_sha256,
                        transition=transition,
                        failure_type=f"teacher_{failure}",
                        reason=f"the teacher violates the frozen requirement {failure!r}",
                    )
                )
        sets = cell_sets[cell_id]
        row_for_cell["teacher_count"] += 1
        row_for_cell["raw_candidate_mark_sum"] += transition["raw_mark_count"]
        row_for_cell["canonical_candidate_successor_sum"] += transition[
            "canonical_successor_count"
        ]
        row_for_cell["matching_candidate_mark_sum"] += transition["matching_mark_count"]
        row_for_cell["successor_alias_sum"] += transition["successor_alias_multiplicity"]
        for _count_field, stratum_field, row_field in _STRATUM_FIELDS:
            stratum = str(transition[stratum_field])
            row_for_cell[row_field][stratum] = row_for_cell[row_field].get(stratum, 0) + 1
        lane = str(transition["data_lane"])
        executor_rule = str(transition["executor_rule"])
        row_for_cell["lane_counts"][lane] = row_for_cell["lane_counts"].get(lane, 0) + 1
        row_for_cell["executor_rule_counts"][executor_rule] = (
            row_for_cell["executor_rule_counts"].get(executor_rule, 0) + 1
        )
        sets["sources"].add(str(transition["source_state_sha256"]))
        sets["targets"].add(str(transition["target_state_sha256"]))
        sets["actions"].add(str(transition["action_sha256"]))
        unique_sources.add(str(transition["source_state_sha256"]))
        unique_targets.add(str(transition["target_state_sha256"]))
        family_counts[str(transition["model_family"])] += 1
        executor_counts[executor_rule] += 1
        lane_counts[lane] += 1

    for key, (_decision_sha256, declared, _role) in eligible.items():
        if observed_transitions[key] != declared:
            raise ProcessV2GateZeroError(
                f"trace {key!r} declares {declared} accepted transitions and the stream "
                f"yielded {observed_transitions[key]}"
            )

    for cell_id, cell_row in cell_rows.items():
        sets = cell_sets[cell_id]
        cell_row["unique_source_state_count"] = len(sets["sources"])
        cell_row["unique_target_state_count"] = len(sets["targets"])
        cell_row["unique_action_count"] = len(sets["actions"])
        for field in (
            "raw_mark_strata",
            "canonical_successor_strata",
            "successor_alias_strata",
            "lane_counts",
            "executor_rule_counts",
        ):
            cell_row[field] = dict(sorted(cell_row[field].items()))

    # One call: four separate calls could interleave four different answers, and
    # a census reconciled across inconsistent snapshots reconciles nothing.
    published_counts = dict(index.counts())
    census = {
        field: _require_count(published_counts.get(field), field=f"index census {field}")
        for field in ACTIVE8_CENSUS_FIELDS
    }
    live_process_identity = str(editing_process_v2_identity()["process_identity_sha256"])
    empty = {
        role: tuple(
            cell_id for cell_id in ids if cell_rows[cell_id]["teacher_count"] == 0
        )
        for role, ids in (
            ("required", contract.required_cell_ids),
            ("conditional", contract.conditional_cell_ids),
            ("separate_lane", contract.separate_lane_cell_ids),
        )
    }
    checks: dict[str, bool] = {
        "contract_identity_bound": True,
        "capability_registry_identity_bound": True,
        "development_cell_role_policy_bound": True,
        "decision_runtime_identity_bound": True,
        "semantic_model_process_identity_bound": True,
        "process_identity_matches_live_process_v2": (
            index.process_identity_sha256 == live_process_identity
        ),
        "decision_eligible_roles_bound": contract.decision_roles == ("train",),
        "sealed_nondecision_roles_not_disclosed": all(
            role not in decision_roles for role in sealed_roles
        ),
        "active8_census_identity_reconciles": (
            census["source_entries"]
            == census["upstream_rejected_entries"]
            + census["active8_accepted_entries"]
            + census["active8_excluded_entries"]
        ),
        "resolved_trace_census_matches": (
            resolved_traces == census["source_entries"]
            and upstream_rejected == census["upstream_rejected_entries"]
            and active8_accepted == census["active8_accepted_entries"]
            and active8_excluded == census["active8_excluded_entries"]
        ),
        "decision_eligible_transition_census_matches": (
            decision_transitions == declared_transition_total
        ),
        "one_structural_assignment_per_decision_eligible_transition": (
            len(assignment_ids) + classification_failures == decision_transitions
        ),
        "structural_assignment_identities_are_unique": (
            len(assignment_ids)
            == len(transition_keys)
            == decision_transitions - classification_failures
        ),
        "terminal_assignment_count_is_zero": terminal_assignments == 0,
        "all_active8_families_have_decision_eligible_teachers": all(
            family_counts[family] > 0 for family in ACTIVE8_FAMILIES
        ),
        "all_active8_executor_rules_have_decision_eligible_teachers": all(
            executor_counts[rule] > 0 for rule in ACTIVE8_EXECUTOR_RULES
        ),
        "all_required_editing_cells_have_decision_eligible_teachers": not empty["required"],
        "conditional_editing_cells_valid_when_present": all(
            cell_rows[cell_id]["matching_candidate_mark_sum"]
            == cell_rows[cell_id]["teacher_count"]
            for cell_id in contract.conditional_cell_ids
        ),
        "separate_lane_cells_excluded_from_editing_authority": all(
            cell_id not in contract.required_cell_ids
            for cell_id in contract.separate_lane_cell_ids
        ),
        "classification_failure_count_is_zero": classification_failures == 0,
        **{
            f"every_decision_eligible_teacher_{name}": requirement_failures[name] == 0
            for name in TEACHER_REQUIREMENTS
        },
        "legacy_action_v2_evidence_used": False,
    }
    structural_result = (
        "PASS"
        if all(
            value is True
            for key, value in checks.items()
            if key != "legacy_action_v2_evidence_used"
        )
        and checks["legacy_action_v2_evidence_used"] is False
        else "FAIL"
    )
    body: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": EVIDENCE_STATUS,
        **authority_false_block(),
        "structural_result": structural_result,
        "contract_id": CONTRACT_ID,
        "contract_file_sha256": contract.file_sha256,
        "contract_sha256": contract.sha256,
        "implementation_file_sha256": _file_sha(Path(__file__)),
        "process_identity_sha256": contract.process_identity_sha256,
        "decision_index_identity": dict(index.identity()),
        "decision_index_completion_sha256": index.completion_sha256,
        "decision_index_process_identity_sha256": index.process_identity_sha256,
        "active8_census": census,
        "rejected_traces_by_code": dict(sorted(index.rejected_traces_by_code().items())),
        "capability_registry": {
            "contract_id": str(contract.capability_cells["contract_id"]),
            "contract_sha256": str(contract.capability_cells["contract_sha256"]),
            "namespace": contract.namespace,
            "family_contexts": {
                family: list(contexts) for family, contexts in contract.family_contexts
            },
        },
        "development_cell_roles": {
            "contract_id": str(contract.development_cell_roles["contract_id"]),
            "contract_sha256": str(contract.development_cell_roles["contract_sha256"]),
            "required_cell_ids": list(contract.required_cell_ids),
            "conditional_cell_ids": list(contract.conditional_cell_ids),
            "separate_lane_cell_ids": list(contract.separate_lane_cell_ids),
        },
        "decision_runtime_contract_sha256": str(contract.decision_runtime["contract_sha256"]),
        "semantic_model_process_contract_sha256": str(
            contract.semantic_model_process["contract_sha256"]
        ),
        "semantic_process_contract_sha256": str(contract.semantic_process["contract_sha256"]),
        "active8_families": list(ACTIVE8_FAMILIES),
        "active8_executor_rules": list(ACTIVE8_EXECUTOR_RULES),
        # Published so a reader can rebuild every stratum from the counts rather
        # than take Gate 0's word for them. Gate 0 is the sole author of the
        # strata -- the index preserves counts and does not classify -- so the
        # per-teacher stratum requirement is a check on a SUPPLIED view and not
        # independent corroboration of a production assignment; the property that
        # is genuinely proven is that these bins tile the count range once, which
        # `strata_bin_defects` establishes at contract load.
        "evidence_strata_authority": "gate_zero_derives_each_stratum_once_from_index_verified_counts",
        "frozen_evidence_strata": {
            field: [
                {"id": identifier, "minimum": minimum, "maximum": maximum}
                for identifier, minimum, maximum in contract.strata_bins[field]
            ]
            for field, _stratum_field, _row_field in _STRATUM_FIELDS
        },
        "checks": checks,
        "counts": {
            "resolved_traces": resolved_traces,
            "upstream_rejected_traces": upstream_rejected,
            "active8_excluded_traces": active8_excluded,
            "active8_accepted_traces": active8_accepted,
            "decision_eligible_traces": decision_traces,
            "decision_eligible_transitions": decision_transitions,
            "structural_assignments": len(assignment_ids),
            "terminal_assignments": terminal_assignments,
            "classification_failures": classification_failures,
            "registered_capability_cells": len(cell_rows),
            "observed_capability_cells": sum(
                row["teacher_count"] > 0 for row in cell_rows.values()
            ),
            "required_editing_cells": len(contract.required_cell_ids),
            "observed_required_editing_cells": (
                len(contract.required_cell_ids) - len(empty["required"])
            ),
            "conditional_editing_cells": len(contract.conditional_cell_ids),
            "observed_conditional_editing_cells": (
                len(contract.conditional_cell_ids) - len(empty["conditional"])
            ),
            "separate_lane_cells": len(contract.separate_lane_cell_ids),
            "observed_separate_lane_cells": (
                len(contract.separate_lane_cell_ids) - len(empty["separate_lane"])
            ),
            "unique_decision_eligible_source_states": len(unique_sources),
            "unique_decision_eligible_target_states": len(unique_targets),
        },
        "decision_eligible_partition_roles": list(contract.decision_roles),
        "sealed_nondecision_partition_roles": list(sealed_roles),
        "decision_eligible_teacher_counts_by_family": {
            family: family_counts[family] for family in ACTIVE8_FAMILIES
        },
        "decision_eligible_teacher_counts_by_executor_rule": {
            rule: executor_counts[rule] for rule in ACTIVE8_EXECUTOR_RULES
        },
        "decision_eligible_teacher_counts_by_lane": dict(sorted(lane_counts.items())),
        "teacher_requirement_failure_counts": {
            name: requirement_failures[name] for name in TEACHER_REQUIREMENTS
        },
        "sealed_nondecision_role_inventory_sha256": {
            role: sealed_streams[role].hexdigest() for role in sealed_roles
        },
        "classification_failure_count": classification_failures,
        "classification_failure_receipts": receipts,
        "capability_cell_counts": [cell_rows[key] for key in sorted(cell_rows)],
        "empty_capability_cell_ids": [
            key for key in sorted(cell_rows) if cell_rows[key]["teacher_count"] == 0
        ],
        "empty_required_editing_cell_ids": list(empty["required"]),
        "empty_conditional_editing_cell_ids": list(empty["conditional"]),
        "empty_separate_lane_cell_ids": list(empty["separate_lane"]),
        "accepted_trace_stream_sha256": accepted_trace_stream.hexdigest(),
        "structural_assignment_inventory_sha256": assignment_stream.hexdigest(),
    }
    evidence = {**body, "evidence_sha256": _sha(body)}
    _require_nonauthorizing(evidence, label="Process-V2 Gate-0 structural evidence")
    return evidence


# ---- Decision, completion, publication ----


def _require_nonauthorizing(payload: Mapping[str, Any], *, label: str) -> None:
    """Both authority guards, at every depth, on PASS and FAIL alike."""

    try:
        require_no_granted_authority(payload, label=label)
        require_authority_false(payload, label=label)
    except ProcessV2SchemaError as error:
        raise ProcessV2GateZeroError(str(error)) from error


def process_v2_gate_zero_decision(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Record PASS or FAIL without converting structural evidence into authority."""

    result = evidence["structural_result"]
    if result not in {"PASS", "FAIL"}:
        raise ProcessV2GateZeroError(f"structural_result is {result!r}, not PASS or FAIL")
    body: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": DECISION_STATUS,
        **authority_false_block(),
        "structural_result": result,
        "evidence_sha256": _require_sha(evidence["evidence_sha256"], field="evidence_sha256"),
        "contract_sha256": evidence["contract_sha256"],
        "decision_index_completion_sha256": evidence["decision_index_completion_sha256"],
        "next_authorized_stage": None,
        "required_next_action": (
            "record_separate_human_or_registered_gate_zero_authorization"
            if result == "PASS"
            else "repair_missing_structural_coverage_and_rerun_process_v2_gate_zero"
        ),
    }
    decision = {**body, "decision_sha256": _sha(body)}
    _require_nonauthorizing(decision, label="Process-V2 Gate-0 structural decision")
    return decision


def _completion(
    *,
    evidence: Mapping[str, Any],
    decision: Mapping[str, Any],
    evidence_bytes: bytes,
    decision_bytes: bytes,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **authority_false_block(),
        "structural_result": evidence["structural_result"],
        "contract_sha256": evidence["contract_sha256"],
        "evidence_file_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "evidence_sha256": evidence["evidence_sha256"],
        "decision_file_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "decision_sha256": decision["decision_sha256"],
    }
    completion = {**body, "completion_sha256": _sha(body)}
    _require_nonauthorizing(completion, label="Process-V2 Gate-0 structural completion")
    return completion


def _publish(path: Path, content: bytes) -> None:
    """Write once, atomically.  A byte-identical rewrite is a no-op, not a collision."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != content:
            raise ProcessV2GateZeroError(f"immutable Gate-0 artifact collision at {target}")
        return
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def run_process_v2_gate_zero(
    index: ProcessV2Active8Index,
    *,
    repo_root: Path,
    artifact_root: Path,
    output_directory: Path,
    contract: FrozenProcessV2GateZeroContract | None = None,
    view: Callable[..., Mapping[str, Any]] = _structural_view,
) -> dict[str, Any]:
    """Build, decide and publish Process-V2 Gate-0 structural artifacts.

    ``FAIL`` publishes exactly as ``PASS`` does.  A gate that only writes its
    positive result cannot be audited for its negative ones.
    """

    selected = contract or load_process_v2_gate_zero_contract(repo_root=Path(repo_root))
    evidence = build_process_v2_gate_zero_evidence(index, contract=selected, view=view)
    decision = process_v2_gate_zero_decision(evidence)
    output = Path(output_directory).resolve()
    artifact = Path(artifact_root).resolve()
    if not output.is_relative_to(artifact):
        raise ProcessV2GateZeroError("output_directory lies outside artifact_root")
    evidence_bytes = _bytes(evidence, newline=True)
    decision_bytes = _bytes(decision, newline=True)
    completion = _completion(
        evidence=evidence,
        decision=decision,
        evidence_bytes=evidence_bytes,
        decision_bytes=decision_bytes,
    )
    _publish(output / EVIDENCE_FILENAME, evidence_bytes)
    _publish(output / DECISION_FILENAME, decision_bytes)
    _publish(output / COMPLETION_FILENAME, _bytes(completion, newline=True))
    return {"evidence": evidence, "decision": decision, "completion": completion}


def load_process_v2_gate_zero_artifacts(
    *,
    output_directory: Path,
    index: ProcessV2Active8Index,
    contract: FrozenProcessV2GateZeroContract,
) -> dict[str, Any]:
    """Revalidate persisted bytes against a freshly recomputed source snapshot."""

    output = Path(output_directory).resolve()
    evidence, evidence_bytes = _load_json(output / EVIDENCE_FILENAME, field="Gate-0 evidence")
    decision, decision_bytes = _load_json(output / DECISION_FILENAME, field="Gate-0 decision")
    completion, completion_bytes = _load_json(
        output / COMPLETION_FILENAME, field="Gate-0 completion"
    )
    for payload, raw in (
        (evidence, evidence_bytes),
        (decision, decision_bytes),
        (completion, completion_bytes),
    ):
        if raw != _bytes(payload, newline=True):
            raise ProcessV2GateZeroError("a persisted Gate-0 artifact is not canonical JSON")
    expected_evidence = build_process_v2_gate_zero_evidence(index, contract=contract)
    if evidence != expected_evidence:
        raise ProcessV2GateZeroError(
            "persisted Gate-0 evidence differs from the recomputed exact source snapshot"
        )
    expected_decision = process_v2_gate_zero_decision(evidence)
    expected_completion = _completion(
        evidence=evidence,
        decision=expected_decision,
        evidence_bytes=evidence_bytes,
        decision_bytes=decision_bytes,
    )
    if decision != expected_decision or completion != expected_completion:
        raise ProcessV2GateZeroError("a persisted Gate-0 artifact differs from exact evidence")
    for payload, label in (
        (evidence, "evidence"),
        (decision, "decision"),
        (completion, "completion"),
    ):
        _require_nonauthorizing(payload, label=f"persisted Process-V2 Gate-0 {label}")
    return {"evidence": evidence, "decision": decision, "completion": completion}


__all__ = [
    "AUTHORITY_FIELDS",
    "COMPLETION_FILENAME",
    "CONTRACT_ID",
    "CONTRACT_RELATIVE_PATH",
    "DECISION_FILENAME",
    "EVIDENCE_FILENAME",
    "MINIMUM_STRATIFIED_COUNT",
    "ROLE_FILTER_PARAMETER",
    "SUPPORT_EVIDENCE_FIELD",
    "TEACHER_REQUIREMENTS",
    "TRACE_ROW_FIELDS",
    "TRANSITION_FIELDS",
    "FrozenProcessV2GateZeroContract",
    "ProcessV2GateZeroError",
    "build_process_v2_gate_zero_evidence",
    "load_process_v2_gate_zero_artifacts",
    "load_process_v2_gate_zero_contract",
    "process_v2_gate_zero_decision",
    "run_process_v2_gate_zero",
    "strata_bin_defects",
    "stratum_for",
]
