"""The Active8 map task: one source chunk in, published evidence out.

This is the only expensive pass over the corpus.  Nothing downstream opens a
molecular chunk, reconstructs a model or re-enumerates a successor fiber, so
everything a later stage needs is decided and published HERE, while the exact
source state, the action, the candidate fiber and the executor result are still
in memory.

WHAT THE TASK DECIDES, AND WHAT IT MERELY ANNOTATES
---------------------------------------------------
Admission is a function of candidate evidence alone.  A trace is evaluated
through the production evaluator, and the whole-trace all-or-nothing rule the
production admission module already owns decides ``accepted`` or ``excluded``.
A trace the rebind already refused is never candidate-evaluated at all: it is
published as ``not_evaluated`` under the ``upstream_rebind_rejected`` category
with the upstream code it was refused for, so "we did not look" stays
distinguishable from "we looked and said no".

The capability cell is an ANNOTATION.  It is assigned at write time because
that is the only moment the fiber is in memory, but it never participates in
the decision: the classifier is called AFTER admission is fixed, its failure is
recorded as an absent cell rather than raised, and
``classification_affects_admission`` is published as ``False`` on every action.
``tests/test_editing_v2_process_v2_active8_pipeline.py`` proves the
independence by stubbing the classifier to a constant and requiring the
admission half of every row to be byte-identical.

The RAW STRUCTURAL AXES travel beside the cell -- ``audit_axes``,
``audit_cycle_rank_delta``, ``audit_touches_ring_system`` and
``audit_is_terminal_source`` -- so a consumer can audit the assignment without
importing the classifier.  Two of them are derived from the exact source and
successor states alone and mention no action payload at all.

AGGREGATES ARE DERIVED, NEVER CARRIED
-------------------------------------
``candidate_totals`` and ``action_family_histogram`` are recomputed from the
published per-action evidence every time they are read, by
:func:`derive_candidate_totals` and :func:`derive_action_family_histogram`.  A
row that stores a total disagreeing with its own actions is refused.  Nothing
in this pipeline trusts a number it did not recompute from evidence.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import operator
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.data.editing_process_v2_rebind import (
    MANIFEST_FILENAME as REBIND_MANIFEST_FILENAME,
)
from compose_v4.data.editing_process_v2_rebind import (
    PROOF_FILENAME as REBIND_PROOF_FILENAME,
)
from compose_v4.data.editing_process_v2_rebind import (
    ProcessV2RebindError,
    mounted_process_v2_artifact_path,
    validate_process_v2_rebind_task_result,
)
from compose_v4.data.editing_v2_process_v2_active8_plan import (
    model_runtime_descriptor,
    task_by_identity,
    validate_process_v2_active8_plan,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    ProcessV2ChunkTarget,
    read_process_v2_chunk_target,
)
from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    ACCEPTED_EVIDENCE_INVARIANTS,
    ACCEPTED_TRANSITION_FIELDS,
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_DECISION_SHARD_FILENAME,
    ACTIVE8_RECEIPT_FIELDS,
    ACTIVE8_RECEIPT_FILENAME,
    ACTIVE8_ROW_SHARD_FILENAME,
    ACTIVE8_EXCLUDED,
    ACTIVE8_ROW_FIELDS,
    ACTIVE8_TASK_SCHEMA,
    ACTIVE8_TASK_SCHEMA_VERSION,
    CANDIDATE_EVIDENCE_FIELDS,
    CANDIDATE_TOTAL_FIELDS,
    CLASSIFICATION_FIELDS,
    PIPELINE_STATUS_NO_AUTHORITY,
    UPSTREAM_REJECTED,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    self_hashed,
    verify_self_hash,
)
from compose_v4.data.editing_v2_semantic_active8_admission import (
    ProductionSemanticExactCandidateChecker,
    SemanticActive8AdmissionError,
    SemanticExactCandidateAudit,
    build_semantic_active8_admission_policy,
    evaluate_semantic_active8_trace,
    validate_semantic_active8_admission_policy,
)
from compose_v4.data.editing_v2_semantic_capability_cells import (
    SemanticCapabilityCellError,
    _action_audit_axes,
    _cycle_rank,
    _state_graph,
    classify_action_family_context,
    load_semantic_capability_cell_registry,
)
from compose_v4.data.packed_trace_store import AddressedPackedTrace

# The published layout, which the Gate-0 reducer pins as module constants.
# METADATA AND ROWS ARE SEPARATE FILES ON PURPOSE.  Gate 0 reads every role's
# receipt to build the census, and constructs the transitions path only inside
# its eligible-role loop, so a held-out role's molecular rows are never opened.
# Folding the transitions into the receipt would destroy that guarantee by
# making the census read them.
ROWS_FILENAME = ACTIVE8_ROW_SHARD_FILENAME
TRANSITIONS_FILENAME = ACTIVE8_DECISION_SHARD_FILENAME
RECEIPT_FILENAME = ACTIVE8_RECEIPT_FILENAME
SUMMARY_FILENAME = "TASK_SUMMARY.json"
SUMMARY_SCHEMA = f"{ACTIVE8_TASK_SCHEMA}.summary"
SUMMARY_SCHEMA_VERSION = 1
TASK_STATUS = PIPELINE_STATUS_NO_AUTHORITY

ACCEPTED = "accepted"
EXCLUDED = "excluded"
NOT_EVALUATED = "not_evaluated"
ADMISSION_STATUSES: tuple[str, ...] = (ACCEPTED, EXCLUDED, NOT_EVALUATED)

#: One published action inside a row.  Every name is seam vocabulary: the two
#: addressing names come from ``ACTION_KEY_FIELDS`` and
#: ``ACCEPTED_TRANSITION_FIELDS``, the evidence is the seam's evidence block and
#: the rest is the seam's classification block, inlined exactly as the accepted
#: transition inlines it.
ACTION_FIELDS: tuple[str, ...] = (
    "step_index",
    "executor_rule",
    "candidate_evidence",
    *CLASSIFICATION_FIELDS,
)

#: Everything the run needs that the frozen receipt deliberately does not
#: carry: the run and plan binding, the row-file digests, and the derived
#: histograms.  It is a SIBLING file, so the receipt stays exactly
#: ``ACTIVE8_RECEIPT_FIELDS`` and a gate reading the census still opens neither
#: the rows nor this.
SUMMARY_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "action_family_histogram",
        "active8_exclusion_histogram",
        "binding_sha256",
        "candidate_totals",
        "capability_cell_histogram",
        "chunk_index",
        "classification_affects_admission",
        "entry_start",
        "entry_stop",
        "plan_sha256",
        "receipt_sha256",
        "row_inventory_sha256",
        "rows_file_sha256",
        "rows_stream_sha256",
        "run_identity_sha256",
        "task_identity_sha256",
        "transitions_stream_sha256",
        "upstream_rejection_histogram",
        "v1_task_identity_sha256",
        "summary_sha256",
    }
)


class ProcessV2Active8MapError(RuntimeError):
    """The Active8 map task cannot decide or publish its chunk."""


# ---- Raw structural axes -------------------------------------------------------


def _cycle_slots(state: MolecularGraph) -> frozenset[int]:
    """Every real slot lying on a cycle: incident to at least one non-bridge edge."""

    graph = _state_graph(state)
    bridges = {frozenset((int(a), int(b))) for a, b in nx.bridges(graph)}
    return frozenset(
        int(node)
        for node in graph.nodes()
        if any(
            frozenset((int(node), int(other))) not in bridges
            for other in graph.neighbors(node)
        )
    )


def _touched_slots(source: MolecularGraph, successor: MolecularGraph) -> frozenset[int]:
    """Slots the transition changed, read from the two exact states alone.

    Deliberately action-agnostic: it compares atom types and bond orders rather
    than interpreting an action payload, so a consumer holding the two states
    can reproduce it with no knowledge of the operator vocabulary.
    """

    changed = {
        int(slot)
        for slot in np.flatnonzero(
            np.asarray(source.atom_types) != np.asarray(successor.atom_types)
        )
    }
    left, right = np.nonzero(np.asarray(source.bonds) != np.asarray(successor.bonds))
    changed.update(int(slot) for slot in left)
    changed.update(int(slot) for slot in right)
    real = set(np.flatnonzero(is_element(source.atom_types)).tolist())
    real.update(int(slot) for slot in np.flatnonzero(is_element(successor.atom_types)))
    return frozenset(slot for slot in changed if slot in real)


def structural_audit_axes(
    source: MolecularGraph,
    successor: MolecularGraph,
    *,
    step_index: int,
    path_length: int,
) -> dict[str, Any]:
    """The three classifier-free axes that travel beside the capability cell."""

    touched = _touched_slots(source, successor)
    on_cycle = _cycle_slots(source) | _cycle_slots(successor)
    return {
        "audit_cycle_rank_delta": _cycle_rank(successor) - _cycle_rank(source),
        "audit_touches_ring_system": bool(touched & on_cycle),
        # The teacher's SOURCE progress position, never the successor's: a
        # transition exists precisely because its source has an outgoing step.
        "audit_is_terminal_source": bool(int(step_index) == int(path_length)),
    }


def _family_audit_axes(
    source: MolecularGraph,
    successor: MolecularGraph,
    step: Any,
    *,
    family: str,
) -> list[list[Any]]:
    """The production family audit axes as sorted ``[name, value]`` pairs."""

    (
        element_transition,
        minimum_edited_cycle_length,
        source_edge_aromatic,
        successor_edge_aromatic,
    ) = _action_audit_axes(source, successor, step, family=family)
    axes: dict[str, Any] = {name: value for name, value in element_transition}
    axes["minimum_edited_cycle_length"] = minimum_edited_cycle_length
    axes["source_edge_aromatic"] = source_edge_aromatic
    axes["successor_edge_aromatic"] = successor_edge_aromatic
    return [[name, axes[name]] for name in sorted(axes)]


def _classification_block(
    source: MolecularGraph,
    successor: MolecularGraph,
    step: Any,
    *,
    family: str | None,
    supported: bool,
    step_index: int,
    path_length: int,
    namespace: str,
) -> dict[str, Any]:
    """Assign the cell and the axes; never raise, never touch admission.

    Called only after the admission decision is fixed.  A classifier refusal is
    recorded as an absent cell and an absent context, because raising here would
    let the annotation decide whether a row exists at all -- which is exactly
    the independence this block is required not to have.
    """

    block: dict[str, Any] = {
        "capability_cell_id": None,
        "model_family": family,
        "family_context": None,
        "audit_axes": [],
        **structural_audit_axes(
            source, successor, step_index=step_index, path_length=path_length
        ),
        "classification_affects_admission": False,
    }
    if not supported or family is None:
        return block
    try:
        observed_family, context = classify_action_family_context(source, successor, step)
        axes = _family_audit_axes(source, successor, step, family=observed_family)
    except (SemanticCapabilityCellError, ValueError, KeyError, IndexError):
        return block
    if observed_family != family:
        return block
    block["family_context"] = context
    block["capability_cell_id"] = f"{namespace}:{family}:{context}"
    block["audit_axes"] = axes
    return block


# ---- Evidence, derived aggregates ----------------------------------------------


def _evidence_payload(audit: SemanticExactCandidateAudit) -> dict[str, Any]:
    evidence = audit.evidence
    payload = {
        "supported": bool(evidence.supported),
        "exclusion_reason": evidence.exclusion_reason,
        "action_sha256": evidence.action_sha256,
        "source_state_sha256": evidence.source_state_sha256,
        "target_state_sha256": evidence.target_state_sha256,
        "canonical_successor_key": evidence.canonical_successor_key,
        "raw_mark_count": int(evidence.raw_mark_count),
        "canonical_successor_count": int(evidence.canonical_successor_count),
        "matching_mark_count": int(evidence.matching_mark_count),
        "exact_successor_mark_count": int(audit.exact_successor_mark_count),
        "successor_alias_count": int(evidence.successor_alias_count),
    }
    if tuple(payload) != CANDIDATE_EVIDENCE_FIELDS:
        raise ProcessV2Active8MapError("the published candidate-evidence fields disagree")
    return payload


#: Longest first, so ``<=`` is never read as ``<``.
_COMPARATORS: tuple[tuple[str, Any], ...] = (
    ("==", operator.eq),
    ("!=", operator.ne),
    ("<=", operator.le),
    (">=", operator.ge),
    ("<", operator.lt),
    (">", operator.gt),
)


def _invariant_term(token: str, evidence: Mapping[str, Any]) -> int:
    name = token.strip()
    if name in CANDIDATE_EVIDENCE_FIELDS:
        value = evidence[name]
        if type(value) is not int:
            raise ProcessV2Active8MapError(
                f"published candidate evidence {name!r} is not an integer count"
            )
        return value
    try:
        return int(name)
    except ValueError as error:
        raise ProcessV2Active8MapError(
            f"the frozen evidence invariant names {name!r}, which is neither a "
            "published count nor a literal"
        ) from error


def require_accepted_evidence_invariants(evidence: Mapping[str, Any]) -> None:
    """The seam's accepted-evidence arithmetic, evaluated from the seam itself.

    ``ACCEPTED_EVIDENCE_INVARIANTS`` states the arithmetic as data precisely so
    the writer, the reader and the sentinel check ONE list rather than three
    transcriptions of it.  Transcribing it here would reintroduce the third
    copy, so the statements are parsed and evaluated instead: each is one
    comparison between two published counts or literals, and an invariant
    naming anything else is a refusal rather than a skipped check.
    """

    for invariant in ACCEPTED_EVIDENCE_INVARIANTS:
        for symbol, compare in _COMPARATORS:
            left, separator, right = invariant.partition(f" {symbol} ")
            if not separator:
                continue
            if not compare(
                _invariant_term(left, evidence), _invariant_term(right, evidence)
            ):
                raise ProcessV2Active8MapError(
                    "accepted candidate evidence violates the frozen invariant "
                    f"{invariant!r}"
                )
            break
        else:
            raise ProcessV2Active8MapError(
                f"the frozen evidence invariant {invariant!r} is not one comparison"
            )


def derive_candidate_totals(actions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Recompute the row totals from the published action evidence."""

    totals = dict.fromkeys(CANDIDATE_TOTAL_FIELDS, 0)
    for action in actions:
        evidence = action.get("candidate_evidence")
        if evidence is None:
            continue
        totals["raw_candidate_marks"] += int(evidence["raw_mark_count"])
        totals["canonical_candidate_successors"] += int(evidence["canonical_successor_count"])
        totals["matching_candidate_marks"] += int(evidence["matching_mark_count"])
        totals["exact_successor_marks"] += int(evidence["exact_successor_mark_count"])
        totals["successor_aliases"] += int(evidence["successor_alias_count"])
    return totals


def derive_action_family_histogram(actions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Recompute the row family histogram from the published actions."""

    histogram: dict[str, int] = {}
    for action in actions:
        family = action.get("model_family")
        if family is None:
            continue
        histogram[str(family)] = histogram.get(str(family), 0) + 1
    return dict(sorted(histogram.items()))


def require_row_aggregates_derived(row: Mapping[str, Any]) -> None:
    """Refuse a row whose stored aggregates disagree with its own actions."""

    actions = row["actions"]
    if row["candidate_totals"] != derive_candidate_totals(actions):
        raise ProcessV2Active8MapError(
            "the row candidate totals disagree with the evidence they are derived from"
        )
    if row["action_family_histogram"] != derive_action_family_histogram(actions):
        raise ProcessV2Active8MapError(
            "the row family histogram disagrees with the actions it is derived from"
        )


# ---- Rows and transitions -------------------------------------------------------


def _sealed_row(body: Mapping[str, Any]) -> dict[str, Any]:
    row = self_hashed(dict(body), field="row_sha256")
    if tuple(row) != ACTIVE8_ROW_FIELDS:
        raise ProcessV2Active8MapError("the published Active8 row fields disagree")
    require_row_aggregates_derived(row)
    return row


def upstream_rejected_row(
    *,
    v1_task_identity_sha256: str,
    entry_index: int,
    trace_id: str,
    task_identity_sha256: str,
    data_lane: str,
    partition_role: str,
    path_length: int,
    upstream_rejection_code: str,
) -> dict[str, Any]:
    """A trace the rebind refused: present for the census, never evaluated."""

    return _sealed_row(
        {
            "v1_task_identity_sha256": v1_task_identity_sha256,
            "entry_index": int(entry_index),
            "trace_id": trace_id,
            "task_identity_sha256": task_identity_sha256,
            "data_lane": data_lane,
            "partition_role": partition_role,
            "admission_status": NOT_EVALUATED,
            "rejection_category": UPSTREAM_REJECTED,
            "upstream_rejection_code": str(upstream_rejection_code),
            "path_length": int(path_length),
            "actions": [],
            "candidate_totals": dict.fromkeys(CANDIDATE_TOTAL_FIELDS, 0),
            "action_family_histogram": {},
        }
    )


def evaluated_row(
    addressed: AddressedPackedTrace,
    *,
    checker: ProductionSemanticExactCandidateChecker,
    v1_task_identity_sha256: str,
    task_identity_sha256: str,
    namespace: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one admitted trace and publish its row and its transitions.

    The production whole-trace evaluator decides admission; this function then
    re-reads the SAME per-action audits the evaluator consumed, so the published
    evidence is the evidence the decision was taken on rather than a second
    opinion about it.
    """

    audits: dict[int, SemanticExactCandidateAudit] = {}

    def _recording_checker(trace: AddressedPackedTrace, step_index: int):
        audit = checker.evaluate(trace, step_index)
        audits[int(step_index)] = audit
        return audit.evidence

    decision = evaluate_semantic_active8_trace(
        addressed,
        exact_candidate_checker=_recording_checker,
        policy=checker.policy,
    )
    address = addressed.address
    path_length = int(address.path_length)
    actions: list[dict[str, Any]] = []
    for action_decision in decision.action_decisions:
        classification = action_decision.classification
        step_index = int(classification.step_index)
        evidence = action_decision.candidate_evidence
        payload = None
        supported = False
        if evidence is not None:
            audit = audits[step_index]
            if audit.evidence != evidence:
                raise ProcessV2Active8MapError(
                    "the recorded candidate audit differs from the evidence admission used"
                )
            payload = _evidence_payload(audit)
            supported = bool(evidence.supported)
        step = addressed.trace.steps[step_index]
        block = _classification_block(
            addressed.path.state_at(step_index),
            addressed.path.state_at(step_index + 1),
            step,
            family=classification.model_family,
            supported=supported,
            step_index=step_index,
            path_length=path_length,
            namespace=namespace,
        )
        action = {
            "step_index": step_index,
            "executor_rule": str(classification.executor_rule),
            "candidate_evidence": payload,
            **block,
        }
        if tuple(action) != ACTION_FIELDS:
            raise ProcessV2Active8MapError("the published Active8 action fields disagree")
        actions.append(action)
    accepted = decision.active8_status == ACCEPTED
    row = _sealed_row(
        {
            "v1_task_identity_sha256": v1_task_identity_sha256,
            "entry_index": int(address.entry_index),
            "trace_id": str(address.trace_id),
            "task_identity_sha256": task_identity_sha256,
            "data_lane": str(address.layer),
            "partition_role": str(address.partition),
            "admission_status": ACCEPTED if accepted else EXCLUDED,
            "rejection_category": None if accepted else ACTIVE8_EXCLUDED,
            "upstream_rejection_code": None,
            "path_length": path_length,
            "actions": actions,
            "candidate_totals": derive_candidate_totals(actions),
            "action_family_histogram": derive_action_family_histogram(actions),
        }
    )
    if not accepted:
        return row, []
    return row, [accepted_transition(row, action) for action in actions]


def accepted_transition(
    row: Mapping[str, Any], action: Mapping[str, Any]
) -> dict[str, Any]:
    """The exact object Gate 0 aggregates, with nothing left for it to compute."""

    evidence = action["candidate_evidence"]
    if evidence is None or not evidence["supported"]:
        raise ProcessV2Active8MapError("an accepted transition requires supported evidence")
    require_accepted_evidence_invariants(evidence)
    body = {
        "v1_task_identity_sha256": row["v1_task_identity_sha256"],
        "entry_index": int(row["entry_index"]),
        "trace_id": row["trace_id"],
        "step_index": int(action["step_index"]),
        "task_identity_sha256": row["task_identity_sha256"],
        "data_lane": row["data_lane"],
        "partition_role": row["partition_role"],
        "executor_rule": action["executor_rule"],
        "progress_index": int(action["step_index"]),
        "terminal": bool(action["audit_is_terminal_source"]),
        "capability_cell_id": action["capability_cell_id"],
        "model_family": action["model_family"],
        "family_context": action["family_context"],
        "audit_axes": action["audit_axes"],
        "candidate_evidence": dict(evidence),
    }
    transition = self_hashed(body, field="assignment_sha256")
    if tuple(transition) != ACCEPTED_TRANSITION_FIELDS:
        raise ProcessV2Active8MapError("the published accepted-transition fields disagree")
    return transition


# ---- Reading the upstream decision for exactly one chunk ------------------------


def read_rebind_chunk_decisions(
    output: Path,
    *,
    entry_start: int,
    entry_stop: int,
    task_identity_sha256: str,
    chunk_file_sha256: str,
) -> dict[int, dict[str, Any]]:
    """Every rebind decision of one chunk, as ``entry_index -> decision``.

    The rebind's own validator is the authority on the artifact; this reads the
    two published streams it already reconciled and refuses anything that does
    not decide THIS chunk -- the same rebind task the plan names, over the same
    chunk bytes, covering the exact entry range exactly once.
    """

    try:
        receipt = validate_process_v2_rebind_task_result(output)
    except ProcessV2RebindError as error:
        raise ProcessV2Active8MapError(
            f"the upstream rebind task result is invalid: {output}"
        ) from error
    if str(receipt["task_identity_sha256"]) != str(task_identity_sha256):
        raise ProcessV2Active8MapError(
            "the upstream rebind result is not the task the Active8 plan names"
        )
    if str(receipt["task_source_binding"]["chunk_file_sha256"]) != str(chunk_file_sha256):
        raise ProcessV2Active8MapError(
            "the upstream rebind decided another chunk's bytes than this task reads"
        )
    if int(receipt["entry_start"]) != int(entry_start) or int(receipt["entry_stop"]) != int(
        entry_stop
    ):
        raise ProcessV2Active8MapError(
            "the upstream rebind task decided another entry range than this chunk"
        )
    manifest = json.loads((output / REBIND_MANIFEST_FILENAME).read_bytes())
    decisions: dict[int, dict[str, Any]] = {}
    with gzip.open(output / REBIND_PROOF_FILENAME, "rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            proof = json.loads(raw_line)
            decisions[int(proof["entry_index"])] = {"admitted": True, "proof": proof}
    for rejection in manifest["rejected_traces"]:
        entry_index = int(rejection["entry_index"])
        if entry_index in decisions:
            raise ProcessV2Active8MapError(
                f"the upstream rebind decides entry {entry_index} twice"
            )
        decisions[entry_index] = {"admitted": False, "rejection": rejection}
    if sorted(decisions) != list(range(int(entry_start), int(entry_stop))):
        raise ProcessV2Active8MapError(
            "the upstream rebind does not decide the chunk's exact entry range"
        )
    return decisions


def _chunk_target(task: Mapping[str, Any]):
    """Rebuild the exact chunk target one Active8 task binds.

    The pinned identity is the V1 PAYLOAD identity the cache was written under,
    which the plan carries per task.  Substituting the Process-V2 identity here
    would bind a target no published manifest can satisfy.
    """

    return ProcessV2ChunkTarget(
        source_artifact_path=str(task["cache_source_artifact_path"]),
        cache_source_task_identity_sha256=str(task["cache_source_task_identity_sha256"]),
        cache_source_manifest_sha256=str(task["cache_source_manifest_sha256"]),
        cache_semantic_identity_sha256=str(task["cache_semantic_identity_sha256"]),
        cache_physical_identity_sha256=str(task["cache_physical_identity_sha256"]),
        v1_task_identity_sha256=str(task["v1_task_identity_sha256"]),
        data_lane=str(task["data_lane"]),
        split=str(task["split"]),
        semantic_shard_sha256=str(task["v1_semantic_shard_sha256"]),
        pinned_process_identity_sha256=str(task["pinned_process_identity_sha256"]),
        chunk_index=int(task["chunk_index"]),
        chunk_filename=str(task["chunk_filename"]),
        chunk_file_sha256=str(task["chunk_file_sha256"]),
        chunk_uncompressed_sha256=str(task["chunk_uncompressed_sha256"]),
        entry_start=int(task["entry_start"]),
        entry_stop=int(task["entry_stop"]),
        row_count=int(task["chunk_row_count"]),
    )


# ---- Publication ----------------------------------------------------------------


def _write_stream(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Write canonical gzip JSONL; return ``(file_sha256, stream_sha256)``."""

    stream = hashlib.sha256()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in rows:
                line = canonical_bytes(row) + b"\n"
                stream.update(line)
                handle.write(line)
    return hashlib.sha256(path.read_bytes()).hexdigest(), stream.hexdigest()


def read_task_rows(output: Path) -> list[dict[str, Any]]:
    """Read the published rows of one task, proving each row's own hashes."""

    rows: list[dict[str, Any]] = []
    with gzip.open(output / ROWS_FILENAME, "rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if canonical_bytes(row) + b"\n" != raw_line:
                raise ProcessV2Active8MapError("an Active8 row is not canonical JSONL")
            # Set comparison, not tuple: the canonical encoding sorts keys, so a
            # round-tripped row carries the seam's fields in sorted order.  The
            # WRITE side compares the tuple, which is where field order and
            # completeness are actually established.
            if set(row) != set(ACTIVE8_ROW_FIELDS):
                raise ProcessV2Active8MapError("an Active8 row field set disagrees")
            verify_self_hash(row, field="row_sha256", label="the Active8 row")
            require_row_aggregates_derived(row)
            rows.append(row)
    return rows


def read_task_transitions(output: Path) -> list[dict[str, Any]]:
    """Read the published accepted transitions of one task."""

    transitions: list[dict[str, Any]] = []
    with gzip.open(output / TRANSITIONS_FILENAME, "rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            transition = json.loads(raw_line)
            if canonical_bytes(transition) + b"\n" != raw_line:
                raise ProcessV2Active8MapError(
                    "an Active8 transition is not canonical JSONL"
                )
            if set(transition) != set(ACCEPTED_TRANSITION_FIELDS):
                raise ProcessV2Active8MapError("an Active8 transition field set disagrees")
            verify_self_hash(
                transition, field="assignment_sha256", label="the Active8 transition"
            )
            require_accepted_evidence_invariants(transition["candidate_evidence"])
            transitions.append(transition)
    return transitions


def validate_process_v2_active8_task_result(
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconcile one published task from its own bytes.

    Returns ``(receipt, summary)``.  The receipt is the frozen seam artifact a
    gate reads; the summary is this stage's own sibling, carrying the run
    binding and the derived histograms the receipt deliberately excludes.
    """

    output = Path(output)
    if not output.is_dir() or {path.name for path in output.iterdir()} != {
        ROWS_FILENAME,
        TRANSITIONS_FILENAME,
        RECEIPT_FILENAME,
        SUMMARY_FILENAME,
    }:
        raise ProcessV2Active8MapError(
            f"the Active8 task output inventory is incomplete or unexpected: {output}"
        )
    receipt = json.loads((output / RECEIPT_FILENAME).read_bytes())
    if not isinstance(receipt, dict) or set(receipt) != set(ACTIVE8_RECEIPT_FIELDS):
        raise ProcessV2Active8MapError("the Active8 receipt field set disagrees")
    verify_self_hash(receipt, field="receipt_sha256", label="the Active8 receipt")
    if (
        receipt["schema"] != ACTIVE8_TASK_SCHEMA
        or receipt["schema_version"] != ACTIVE8_TASK_SCHEMA_VERSION
        or receipt["status"] != TASK_STATUS
    ):
        raise ProcessV2Active8MapError("the Active8 receipt contract disagrees")
    summary = json.loads((output / SUMMARY_FILENAME).read_bytes())
    if not isinstance(summary, dict) or set(summary) != SUMMARY_FIELDS:
        raise ProcessV2Active8MapError("the Active8 task summary field set disagrees")
    verify_self_hash(summary, field="summary_sha256", label="the Active8 task summary")
    if (
        summary["schema"] != SUMMARY_SCHEMA
        or summary["schema_version"] != SUMMARY_SCHEMA_VERSION
        or summary["status"] != TASK_STATUS
        or summary["classification_affects_admission"] is not False
        or summary["receipt_sha256"] != receipt["receipt_sha256"]
        or summary["task_identity_sha256"] != receipt["task_identity_sha256"]
    ):
        raise ProcessV2Active8MapError("the Active8 task summary does not bind its receipt")
    rows_path = output / ROWS_FILENAME
    transitions_path = output / TRANSITIONS_FILENAME
    if hashlib.sha256(rows_path.read_bytes()).hexdigest() != summary["rows_file_sha256"]:
        raise ProcessV2Active8MapError("the Active8 rows file hash disagrees")
    if (
        hashlib.sha256(transitions_path.read_bytes()).hexdigest()
        != receipt["decision_shard_sha256"]
    ):
        raise ProcessV2Active8MapError("the Active8 decision shard hash disagrees")
    rows = read_task_rows(output)
    transitions = read_task_transitions(output)
    measured = summarize_task(
        rows,
        transitions,
        entry_start=int(summary["entry_start"]),
        entry_stop=int(summary["entry_stop"]),
        task_identity_sha256=str(receipt["task_identity_sha256"]),
    )
    for field in (*ACTIVE8_CENSUS_FIELDS, "transition_count"):
        if receipt[field] != measured[field]:
            raise ProcessV2Active8MapError(
                f"the Active8 receipt {field} disagrees with the rows it summarizes"
            )
    for field, value in measured.items():
        if field in SUMMARY_FIELDS and summary[field] != value:
            raise ProcessV2Active8MapError(
                f"the Active8 task summary {field} disagrees with the rows it summarizes"
            )
    return receipt, summary


def summarize_task(
    rows: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    *,
    entry_start: int,
    entry_stop: int,
    task_identity_sha256: str,
) -> dict[str, Any]:
    """Every summary a receipt publishes, recomputed from rows and transitions."""

    if [int(row["entry_index"]) for row in rows] != list(range(entry_start, entry_stop)):
        raise ProcessV2Active8MapError(
            "the Active8 rows do not cover the chunk's exact entry range in order"
        )
    census: dict[str, int] = dict.fromkeys(ACTIVE8_CENSUS_FIELDS, 0)
    census["source_entries"] = len(rows)
    totals = dict.fromkeys(CANDIDATE_TOTAL_FIELDS, 0)
    families: dict[str, int] = {}
    cells: dict[str, int] = {}
    upstream: dict[str, int] = {}
    exclusions: dict[str, int] = {}
    for row in rows:
        if str(row["task_identity_sha256"]) != task_identity_sha256:
            raise ProcessV2Active8MapError("an Active8 row names another task")
        status = str(row["admission_status"])
        if status == ACCEPTED:
            census["active8_accepted_entries"] += 1
        elif status == EXCLUDED:
            census["active8_excluded_entries"] += 1
        elif status == NOT_EVALUATED:
            census["upstream_rejected_entries"] += 1
            upstream[str(row["upstream_rejection_code"])] = (
                upstream.get(str(row["upstream_rejection_code"]), 0) + 1
            )
        else:
            raise ProcessV2Active8MapError(f"unknown Active8 admission status: {status!r}")
        for name, value in derive_candidate_totals(row["actions"]).items():
            totals[name] += value
        for name, value in derive_action_family_histogram(row["actions"]).items():
            families[name] = families.get(name, 0) + value
        for action in row["actions"]:
            evidence = action["candidate_evidence"]
            if evidence is not None and not evidence["supported"]:
                reason = str(evidence["exclusion_reason"])
                exclusions[reason] = exclusions.get(reason, 0) + 1
    for transition in transitions:
        cell = transition["capability_cell_id"]
        if cell is not None:
            cells[str(cell)] = cells.get(str(cell), 0) + 1
    if (
        census["source_entries"]
        != census["upstream_rejected_entries"]
        + census["active8_accepted_entries"]
        + census["active8_excluded_entries"]
    ):
        raise ProcessV2Active8MapError("the Active8 chunk census does not reconcile")
    return {
        **census,
        "transition_count": len(transitions),
        "action_family_histogram": dict(sorted(families.items())),
        "active8_exclusion_histogram": dict(sorted(exclusions.items())),
        "candidate_totals": totals,
        "capability_cell_histogram": dict(sorted(cells.items())),
        "row_inventory_sha256": canonical_sha256(
            [[int(row["entry_index"]), str(row["row_sha256"])] for row in rows]
        ),
        "upstream_rejection_histogram": dict(sorted(upstream.items())),
    }


def _publish(
    output: Path, *, rows, transitions, receipt_body, summary_body
) -> tuple[dict[str, Any], dict[str, Any]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    rows_file, rows_stream = _write_stream(staging / ROWS_FILENAME, rows)
    shard_file, shard_stream = _write_stream(staging / TRANSITIONS_FILENAME, transitions)
    receipt = self_hashed(
        {**receipt_body, "decision_shard_sha256": shard_file},
        field="receipt_sha256",
    )
    if tuple(receipt) != ACTIVE8_RECEIPT_FIELDS:
        raise ProcessV2Active8MapError("the Active8 receipt field set disagrees")
    summary = self_hashed(
        {
            **summary_body,
            "receipt_sha256": receipt["receipt_sha256"],
            "rows_file_sha256": rows_file,
            "rows_stream_sha256": rows_stream,
            "transitions_stream_sha256": shard_stream,
        },
        field="summary_sha256",
    )
    if set(summary) != SUMMARY_FIELDS:
        raise ProcessV2Active8MapError("the Active8 task summary field set disagrees")
    (staging / RECEIPT_FILENAME).write_bytes(canonical_bytes(receipt) + b"\n")
    (staging / SUMMARY_FILENAME).write_bytes(canonical_bytes(summary) + b"\n")
    if output.exists():
        shutil.rmtree(output)
    os.replace(staging, output)
    return receipt, summary


def execute_process_v2_active8_task(
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    *,
    runtime: Any,
    artifact_root: Path,
    repo_root: Path,
    reuse: bool = True,
) -> dict[str, Any]:
    """Decide and publish exactly one source chunk.

    ``reuse`` is restart safety, not caching: an existing output is reused ONLY
    when it reconciles completely from its own bytes and names this exact plan,
    run and task.  Anything else -- a partial write, a stale namespace, a
    receipt whose summary disagrees with its rows -- is redone from the chunk.
    """

    validated = validate_process_v2_active8_plan(plan, repo_root=Path(repo_root))
    task = task_by_identity(validated, task_identity_sha256)
    binding = validated["binding"]
    if model_runtime_descriptor(runtime) != dict(binding["model_runtime"]):
        raise ProcessV2Active8MapError(
            "the supplied model runtime is not the one the Active8 plan binds"
        )
    policy = validate_semantic_active8_admission_policy(
        build_semantic_active8_admission_policy(process_v2=True)
    )
    if policy.policy_sha256 != binding["active8_policy_sha256"]:
        raise ProcessV2Active8MapError("the live admission policy is not the one planned")
    output = mounted_process_v2_artifact_path(
        str(task["output_artifact_path"]),
        artifact_root=Path(artifact_root),
        field="task.output_artifact_path",
    )
    if reuse and output.is_dir():
        try:
            reused, summary = validate_process_v2_active8_task_result(output)
        except (ProcessV2Active8MapError, OSError, ValueError):
            reused = None
        if reused is not None and (
            reused["task_identity_sha256"] == str(task["task_identity_sha256"])
            and summary["run_identity_sha256"] == str(validated["run_identity_sha256"])
            and summary["plan_sha256"] == str(validated["plan_sha256"])
            and summary["binding_sha256"] == str(binding["binding_sha256"])
        ):
            return reused

    registry = load_semantic_capability_cell_registry()
    if registry.registry_sha256 != binding["capability_cell_registry_sha256"]:
        raise ProcessV2Active8MapError("the live capability-cell registry is not the one planned")
    try:
        checker = ProductionSemanticExactCandidateChecker(runtime.model, policy=policy)
    except SemanticActive8AdmissionError as error:
        raise ProcessV2Active8MapError(
            "the supplied model does not implement the semantic Active8 modes"
        ) from error

    rebind_output = mounted_process_v2_artifact_path(
        str(task["rebind_task_artifact_path"]),
        artifact_root=Path(artifact_root),
        field="task.rebind_task_artifact_path",
    )
    decisions = read_rebind_chunk_decisions(
        rebind_output,
        entry_start=int(task["entry_start"]),
        entry_stop=int(task["entry_stop"]),
        task_identity_sha256=str(task["rebind_task_identity_sha256"]),
        chunk_file_sha256=str(task["chunk_file_sha256"]),
    )
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for row, produced in _decide_chunk(
        task,
        decisions=decisions,
        checker=checker,
        namespace=registry.namespace,
        artifact_root=Path(artifact_root),
        repo_root=Path(repo_root),
    ):
        rows.append(row)
        transitions.extend(produced)
    measured = summarize_task(
        rows,
        transitions,
        entry_start=int(task["entry_start"]),
        entry_stop=int(task["entry_stop"]),
        task_identity_sha256=str(task["task_identity_sha256"]),
    )
    receipt_body = {
        "schema": ACTIVE8_TASK_SCHEMA,
        "schema_version": ACTIVE8_TASK_SCHEMA_VERSION,
        "status": TASK_STATUS,
        "task_identity_sha256": str(task["task_identity_sha256"]),
        # The policy vocabulary, not the data path's `split`: this artifact is
        # exactly what a gate reads to decide eligibility.
        "partition_role": str(task["split"]),
        "data_lane": str(task["data_lane"]),
        "source_chunk_identity_sha256": str(task["chunk_file_sha256"]),
        **{field: measured[field] for field in ACTIVE8_CENSUS_FIELDS},
        "transition_count": int(measured["transition_count"]),
    }
    summary_body = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": TASK_STATUS,
        **authority_false_block(),
        "action_family_histogram": measured["action_family_histogram"],
        "active8_exclusion_histogram": measured["active8_exclusion_histogram"],
        "binding_sha256": str(binding["binding_sha256"]),
        "candidate_totals": measured["candidate_totals"],
        "capability_cell_histogram": measured["capability_cell_histogram"],
        "chunk_index": int(task["chunk_index"]),
        "classification_affects_admission": False,
        "entry_start": int(task["entry_start"]),
        "entry_stop": int(task["entry_stop"]),
        "plan_sha256": str(validated["plan_sha256"]),
        "row_inventory_sha256": measured["row_inventory_sha256"],
        "run_identity_sha256": str(validated["run_identity_sha256"]),
        "task_identity_sha256": str(task["task_identity_sha256"]),
        "upstream_rejection_histogram": measured["upstream_rejection_histogram"],
        "v1_task_identity_sha256": str(task["v1_task_identity_sha256"]),
    }
    return _publish(
        output,
        rows=rows,
        transitions=transitions,
        receipt_body=receipt_body,
        summary_body=summary_body,
    )[0]


def _decide_chunk(
    task: Mapping[str, Any],
    *,
    decisions: Mapping[int, Mapping[str, Any]],
    checker: ProductionSemanticExactCandidateChecker,
    namespace: str,
    artifact_root: Path,
    repo_root: Path,
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    target = _chunk_target(task)
    source_output = mounted_process_v2_artifact_path(
        target.source_artifact_path,
        artifact_root=artifact_root,
        field="task.cache_source_artifact_path",
    )
    v1_task_identity = str(task["v1_task_identity_sha256"])
    task_identity = str(task["task_identity_sha256"])
    for read in read_process_v2_chunk_target(
        source_output,
        target=target,
        sentinel_replay_entries=0,
        recover_row_errors=False,
        repo_root=repo_root,
    ):
        entry_index = int(read.entry_index)
        decision = decisions[entry_index]
        record = read.record
        if str(record["trace_id"]) != str(decision_trace_id(decision)):
            raise ProcessV2Active8MapError(
                f"the cached row and the upstream decision name different traces at {entry_index}"
            )
        if not decision["admitted"]:
            yield (
                upstream_rejected_row(
                    v1_task_identity_sha256=v1_task_identity,
                    entry_index=entry_index,
                    trace_id=str(record["trace_id"]),
                    task_identity_sha256=task_identity,
                    data_lane=str(record["data_lane"]),
                    partition_role=str(record["split"]),
                    path_length=int(record["path_length"]),
                    upstream_rejection_code=str(decision["rejection"]["exclusion_code"]),
                ),
                [],
            )
            continue
        if read.addressed is None:
            raise ProcessV2Active8MapError(
                f"an admitted cached row has no decoded trace at entry {entry_index}"
            )
        yield evaluated_row(
            read.addressed,
            checker=checker,
            v1_task_identity_sha256=v1_task_identity,
            task_identity_sha256=task_identity,
            namespace=namespace,
        )


def decision_trace_id(decision: Mapping[str, Any]) -> str:
    """The trace id one upstream decision names, whichever side decided it."""

    side = decision["proof"] if decision["admitted"] else decision["rejection"]
    return str(side["trace_id"])


__all__ = [
    "ACCEPTED",
    "ACTION_FIELDS",
    "ADMISSION_STATUSES",
    "EXCLUDED",
    "NOT_EVALUATED",
    "RECEIPT_FILENAME",
    "SUMMARY_FIELDS",
    "SUMMARY_FILENAME",
    "ROWS_FILENAME",
    "TRANSITIONS_FILENAME",
    "ProcessV2Active8MapError",
    "accepted_transition",
    "derive_action_family_histogram",
    "derive_candidate_totals",
    "evaluated_row",
    "execute_process_v2_active8_task",
    "read_rebind_chunk_decisions",
    "read_task_rows",
    "read_task_transitions",
    "require_accepted_evidence_invariants",
    "require_row_aggregates_derived",
    "structural_audit_axes",
    "summarize_task",
    "upstream_rejected_row",
    "validate_process_v2_active8_task_result",
]
