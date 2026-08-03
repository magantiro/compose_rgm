"""The verified read-only index over a completed Process-V2 Active8 run.

WHAT GATE 0 IS HANDED, AND WHY IT IS NOT A CACHED OBJECT
--------------------------------------------------------
:func:`resolve_process_v2_active8_decision_index` reopens the run: the exact
published plan bytes, the committed cache chunks, the upstream rebind task
results, every Active8 receipt and the completion.  Nothing is taken from a
Python object that happened to be produced earlier in the same process.  That is
the whole point of a resolved index: an object held in memory and an artifact on
disk can disagree, and the one a later stage acts on must be the artifact.

NO V1 CLASS IS IN THE MRO
-------------------------
:class:`ProcessV2Active8DecisionIndex` inherits from nothing.  It satisfies
:class:`~compose_v4.data.editing_v2_process_v2_active8_interfaces.ProcessV2Active8Index`
structurally, which is exactly what that protocol was declared for: inheriting a
V1 decision index would drag in live-V1 schema and identity revalidation, and the
historical payload this run is built on is sealed under a superseded V1 process
identity, so a V1 base would refuse it by construction.

A TRACE IS ADDRESSED BY ITS WHOLE KEY
-------------------------------------
Every lookup is keyed by ``(v1_task_identity_sha256, entry_index, trace_id)``.  A
trace id is unique within a V1 task and nothing guarantees it across tasks, so a
bare-id lookup can merge two different traces, and the merged answer does not
look wrong: it looks like a trace with more transitions than it has.

EXACT STATES, EXACT ACTIONS
---------------------------
A transition carries the exact persistent-slot states, read through the
production chunk reader from the cached record's own arrays, and the ActionV4
record its teacher action encodes to.  No state is reconstructed from a canonical
SMILES key, and no action identity is re-derived from anything but the codec.

WHAT IT REFUSES
---------------
A V1 decision plan, a V1 process identity, a V1 admission policy and a V1
model/process contract are each refused by name rather than by accident: the
plan validator requires the live Process-V2 identity and the exact Process-V2
policy body, and the runtime validator requires the Process-V2 editing process
and atom-delete semantics.

It grants nothing.  Every authority field it publishes is ``False``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_EXCLUDED,
    JOIN_KEY_FIELDS,
    REJECTION_CATEGORIES,
    TRACE_KEY_FIELDS,
    UPSTREAM_REJECTED,
    ProcessV2TraceKey,
)
from compose_v4.data.editing_v2_process_v2_active8_mapreduce import (
    COMPLETION_FILENAME,
    COMPLETION_SCHEMA,
    COMPLETION_SCHEMA_VERSION,
    COMPLETION_STATUS,
    COUNT_FIELDS,
    DECISION_FILENAME,
    PLAN_FILENAME,
    ProcessV2Active8Incomplete,
    ProcessV2Active8MapReduceError,
    completed_process_v2_active8_task_ids,
    read_process_v2_active8_decision_rows,
    reduction_order,
    structural_census,
    validate_process_v2_active8_plan,
    validate_process_v2_active8_task_result,
)
from compose_v4.data.editing_v2_process_v2_active8_runtime import (
    ProcessV2Active8RuntimeError,
    validate_process_v2_active8_model_runtime_identity,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    ProcessV2ChunkCacheError,
    ProcessV2ChunkTarget,
    mount_process_v2_artifact_path,
    read_process_v2_chunk_target,
    require_process_v2_artifact_path,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
    require_census_reconciles,
)
from compose_v4.rewrite import action_codec_v4

INDEX_SCHEMA = "compose.data.editing_v2_process_v2_active8_decision_index"
INDEX_SCHEMA_VERSION = 1
INDEX_STATUS = "VERIFIED_PROCESS_V2_ACTIVE8_INDEX_NO_DOWNSTREAM_AUTHORITY"

#: Every field a resolved-trace row carries.  A consumer reads this rather than
#: guessing, and the row always exposes the whole trace key.
RESOLVED_TRACE_FIELDS: tuple[str, ...] = (
    *TRACE_KEY_FIELDS,
    "data_lane",
    "split",
    "task_identity_sha256",
    "category",
    "active8_status",
    "upstream_rejection_code",
    "path_length",
    "accepted_transition_count",
    "decision_sha256",
)

#: Every field one accepted transition carries.
ACCEPTED_TRANSITION_FIELDS: tuple[str, ...] = (
    *TRACE_KEY_FIELDS,
    "index_identity_sha256",
    "task_identity_sha256",
    "data_lane",
    "split",
    "step_index",
    "executor_rule",
    "model_family",
    "action_sha256",
    "action_record",
    "source_progress_index",
    "successor_progress_index",
    "source_state_sha256",
    "successor_state_sha256",
    "successor_is_terminal",
    "canonical_successor_key",
    "raw_mark_count",
    "canonical_successor_count",
    "matching_mark_count",
    "exact_successor_mark_count",
    "successor_alias_count",
    "source_state",
    "successor_state",
)

_STATE_FIELDS: tuple[str, ...] = ("source_state", "successor_state")


class ProcessV2Active8DecisionIndexError(RuntimeError):
    """A supplied Process-V2 Active8 run is incomplete or inconsistent."""


def _mounted(value: object, *, artifact_root: Path, field: str) -> Path:
    return mount_process_v2_artifact_path(
        require_process_v2_artifact_path(value, field=field),
        artifact_root=artifact_root,
        field=field,
    )


def _chunk_target(task: Mapping[str, Any]) -> ProcessV2ChunkTarget:
    return ProcessV2ChunkTarget(
        source_artifact_path=str(task["cache_source_artifact_path"]),
        cache_source_task_identity_sha256=str(task["cache_source_task_identity_sha256"]),
        cache_source_manifest_sha256=str(task["cache_source_manifest_sha256"]),
        cache_semantic_identity_sha256=str(task["cache_semantic_identity_sha256"]),
        cache_physical_identity_sha256=str(task["cache_physical_identity_sha256"]),
        v1_task_identity_sha256=str(task["v1_task_identity_sha256"]),
        data_lane=str(task["data_lane"]),
        split=str(task["split"]),
        semantic_shard_sha256=str(task["semantic_shard_sha256"]),
        pinned_process_identity_sha256=str(task["pinned_process_identity_sha256"]),
        chunk_index=int(task["chunk_index"]),
        chunk_filename=str(task["chunk_filename"]),
        chunk_file_sha256=str(task["chunk_file_sha256"]),
        chunk_uncompressed_sha256=str(task["chunk_uncompressed_sha256"]),
        entry_start=int(task["entry_start"]),
        entry_stop=int(task["entry_stop"]),
        row_count=int(task["chunk_row_count"]),
    )


class ProcessV2Active8DecisionIndex:
    """A verified, authority-free index over one completed Active8 run.

    Deliberately a plain class: its ancestry is itself and ``object``, so it
    satisfies the frozen protocol structurally and carries no V1 revalidation.
    Instances are built by :func:`resolve_process_v2_active8_decision_index`,
    which is the only place the artifacts are read and checked.
    """

    def __init__(
        self,
        *,
        plan: Mapping[str, Any],
        completion: Mapping[str, Any],
        artifact_root: Path,
        repo_root: Path,
        plan_file_sha256: str,
        completion_file_sha256: str,
        counts: Mapping[str, int],
        rejection_census: Mapping[str, int],
        exclusion_reason_histogram: Mapping[str, int],
        upstream_reason_histogram: Mapping[str, int],
        action_family_histogram: Mapping[str, int],
        trace_lookup: Mapping[ProcessV2TraceKey, tuple[str, str | None, str, int]],
        index_identity_sha256: str,
    ) -> None:
        self._plan = dict(plan)
        self._completion = dict(completion)
        self._artifact_root = Path(artifact_root).resolve()
        self._repo_root = Path(repo_root)
        self._plan_file_sha256 = plan_file_sha256
        self._completion_file_sha256 = completion_file_sha256
        self._counts = MappingProxyType(dict(counts))
        self._rejection_census = MappingProxyType(dict(rejection_census))
        self._exclusion_reason_histogram = MappingProxyType(
            dict(exclusion_reason_histogram)
        )
        self._upstream_reason_histogram = MappingProxyType(dict(upstream_reason_histogram))
        self._action_family_histogram = MappingProxyType(dict(action_family_histogram))
        self._trace_lookup = MappingProxyType(dict(trace_lookup))
        self._tasks = MappingProxyType(
            {str(task["task_identity_sha256"]): dict(task) for task in self._plan["tasks"]}
        )
        self._index_identity_sha256 = index_identity_sha256

    # ---- Authority: false by construction ----

    @property
    def training_authorized(self) -> bool:
        return False

    @property
    def gate_zero_authorized(self) -> bool:
        return False

    @property
    def t1_authorized(self) -> bool:
        return False

    @property
    def bounded_p50_authorized(self) -> bool:
        return False

    @property
    def long_training_authorized(self) -> bool:
        return False

    @property
    def checkpoint_selection_authorized(self) -> bool:
        return False

    @property
    def final_test_selection_authorized(self) -> bool:
        return False

    # ---- The structural minimum ----

    @property
    def process_identity_sha256(self) -> str:
        """The live Process-V2 identity the decisions were computed under."""

        return str(self._plan["process_v2_identity"]["process_identity_sha256"])

    @property
    def pinned_payload_process_identity_sha256(self) -> str:
        """The superseded V1 identity the chemistry was sealed under.

        Exposed separately and never as ``process_identity_sha256``: collapsing
        the two is exactly how a V1 artifact comes to be read as a V2 one.
        """

        return str(self._plan["pinned_process_identity"]["process_identity_sha256"])

    @property
    def completion_sha256(self) -> str:
        return str(self._completion["completion_sha256"])

    @property
    def index_identity_sha256(self) -> str:
        return self._index_identity_sha256

    @property
    def policy_sha256(self) -> str:
        return str(self._plan["policy"]["policy_sha256"])

    @property
    def model_runtime_identity_sha256(self) -> str:
        return str(self._plan["model_runtime_identity"]["identity_sha256"])

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    def counts(self) -> Mapping[str, int]:
        """The Active8 census, plus the structural three the protocol declares."""

        return MappingProxyType(
            {**dict(self._counts), **structural_census(self._counts)}
        )

    def active8_counts(self) -> Mapping[str, int]:
        return self._counts

    def rejected_traces_by_code(self) -> Mapping[str, int]:
        """The reason-coded rejection census, categories kept apart.

        A reason code is namespaced by the category that produced it, because
        the two categories mean different things and a consumer summing them
        without knowing which is which would report a corpus narrowed for a
        reason it was not narrowed for.
        """

        return self._rejection_census

    def identity(self) -> Mapping[str, Any]:
        """The deterministic descriptor a consumer records as provenance."""

        return MappingProxyType(dict(self._identity_body()))

    def _identity_body(self) -> dict[str, Any]:
        return _index_identity(
            plan=self._plan,
            plan_file_sha256=self._plan_file_sha256,
            completion_file_sha256=self._completion_file_sha256,
            completion_sha256=self.completion_sha256,
            counts=self.counts(),
            rejection_census=self._rejection_census,
            exclusion_reason_histogram=self._exclusion_reason_histogram,
            upstream_reason_histogram=self._upstream_reason_histogram,
            action_family_histogram=self._action_family_histogram,
        )

    # ---- Streaming ----

    def iter_resolved_traces(self) -> Iterator[Mapping[str, Any]]:
        """Every resolved trace once, in the deterministic reduction order.

        Upstream-rejected traces are included: they are part of the census and
        must be visible, and their rows carry their category so a consumer can
        never candidate-evaluate one by accident.
        """

        for task, row in self._iter_rows():
            yield MappingProxyType(_resolved_row(task, row))

    def resolved_trace(self, trace_key: ProcessV2TraceKey) -> Mapping[str, Any]:
        """One resolved trace, addressed by its whole key."""

        task, row = self._row_for(trace_key)
        return MappingProxyType(_resolved_row(task, row))

    def accepted_transitions_for(
        self, trace_key: ProcessV2TraceKey
    ) -> Iterator[Mapping[str, Any]]:
        """The accepted transitions of one trace, in trace order.

        Empty for an excluded or upstream-rejected trace: absence of transitions
        is the answer, not an error.  The exact slot-addressed states come from
        the cached record's own arrays through the production chunk reader.
        """

        task, row = self._row_for(trace_key)
        if row["category"] is not None:
            return
        addressed = self._addressed_trace(task, int(row["entry_index"]), str(row["trace_id"]))
        for transition in _transitions(
            task=task,
            row=row,
            addressed=addressed,
            index_identity_sha256=self._index_identity_sha256,
        ):
            self.validate_accepted_transition(transition)
            yield MappingProxyType(transition)

    def validate_accepted_transition(self, transition: Mapping[str, Any]) -> None:
        """Raise unless this is exactly what the index published.

        Re-derived from the artifacts rather than compared against a cached copy:
        an expectation recomputed from the object under test cannot fail.
        """

        if not isinstance(transition, Mapping) or set(transition) != set(
            ACCEPTED_TRANSITION_FIELDS
        ):
            raise ProcessV2Active8DecisionIndexError(
                "a Process-V2 Active8 transition must carry exactly its declared fields"
            )
        if transition["index_identity_sha256"] != self._index_identity_sha256:
            raise ProcessV2Active8DecisionIndexError(
                "the transition was published by another Process-V2 Active8 index"
            )
        key = tuple(transition[field] for field in TRACE_KEY_FIELDS)
        task, row = self._row_for(key)
        if str(task["task_identity_sha256"]) != transition["task_identity_sha256"]:
            raise ProcessV2Active8DecisionIndexError(
                "the transition names another Process-V2 Active8 task"
            )
        if row["category"] is not None:
            raise ProcessV2Active8DecisionIndexError(
                "a trace that was not accepted has no accepted transitions"
            )
        addressed = self._addressed_trace(task, int(row["entry_index"]), str(row["trace_id"]))
        expected = _transitions(
            task=task,
            row=row,
            addressed=addressed,
            index_identity_sha256=self._index_identity_sha256,
        )
        step_index = transition["step_index"]
        if type(step_index) is not int or not 0 <= step_index < len(expected):
            raise ProcessV2Active8DecisionIndexError(
                "the transition step index lies outside its trace"
            )
        candidate = expected[step_index]
        # The exact states are compared by their persistent-slot hashes, which is
        # what identifies a slot-addressed state; the arrays themselves are then
        # required to be the very objects the reader produced.
        comparable = {
            field: value
            for field, value in transition.items()
            if field not in _STATE_FIELDS
        }
        if comparable != {
            field: value for field, value in candidate.items() if field not in _STATE_FIELDS
        }:
            raise ProcessV2Active8DecisionIndexError(
                "the transition differs from the one the index publishes for its address"
            )
        for field in _STATE_FIELDS:
            if persistent_slot_state_sha256(transition[field]) != persistent_slot_state_sha256(
                candidate[field]
            ):
                raise ProcessV2Active8DecisionIndexError(
                    f"the transition {field} is not the exact persistent-slot state"
                )

    # ---- Internals ----

    def _task_root(self, task: Mapping[str, Any]) -> Path:
        return _mounted(
            task["output_artifact_path"],
            artifact_root=self._artifact_root,
            field="task.output_artifact_path",
        )

    def _iter_rows(self) -> Iterator[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        for task in reduction_order(self._plan["tasks"]):
            bound = self._tasks[str(task["task_identity_sha256"])]
            output = self._task_root(bound)
            validate_process_v2_active8_task_result(
                output, plan=self._plan, task=bound
            )
            raw = gzip.decompress((output / DECISION_FILENAME).read_bytes())
            for row in read_process_v2_active8_decision_rows(
                raw, task=bound, plan=self._plan
            ):
                yield bound, row

    def _row_for(
        self, trace_key: ProcessV2TraceKey
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        key = _require_trace_key(trace_key)
        entry = self._trace_lookup.get(key)
        if entry is None:
            raise ProcessV2Active8DecisionIndexError(
                f"the trace key {key} is absent from the Process-V2 Active8 index"
            )
        task_identity_sha256, _category, decision_sha256, _path_length = entry
        task = self._tasks[task_identity_sha256]
        output = self._task_root(task)
        validate_process_v2_active8_task_result(output, plan=self._plan, task=task)
        raw = gzip.decompress((output / DECISION_FILENAME).read_bytes())
        for row in read_process_v2_active8_decision_rows(raw, task=task, plan=self._plan):
            if (
                str(row["v1_task_identity_sha256"]),
                int(row["entry_index"]),
                str(row["trace_id"]),
            ) == key:
                if row["decision_sha256"] != decision_sha256:
                    raise ProcessV2Active8DecisionIndexError(
                        f"the published decision for {key} moved since the index resolved"
                    )
                return task, row
        raise ProcessV2Active8DecisionIndexError(
            f"the indexed trace {key} is absent from the task that claims it"
        )

    def _addressed_trace(
        self, task: Mapping[str, Any], entry_index: int, trace_id: str
    ) -> Any:
        target = _chunk_target(task)
        source_output = _mounted(
            target.source_artifact_path,
            artifact_root=self._artifact_root,
            field="task.cache_source_artifact_path",
        )
        try:
            for read in read_process_v2_chunk_target(
                source_output,
                target=target,
                expected_process_identity=self._plan["pinned_process_identity"],
                sentinel_replay_entries=0,
                recover_row_errors=False,
                repo_root=self._repo_root,
            ):
                if read.entry_index != entry_index:
                    continue
                if read.addressed is None or read.addressed.address.trace_id != trace_id:
                    raise ProcessV2Active8DecisionIndexError(
                        f"the cached row at entry {entry_index} is not the indexed trace"
                    )
                return read.addressed
        except ProcessV2ChunkCacheError as error:
            raise ProcessV2Active8DecisionIndexError(
                f"the cached chunk backing entry {entry_index} is unreadable"
            ) from error
        raise ProcessV2Active8DecisionIndexError(
            f"the cached chunk does not hold entry {entry_index}"
        )


def _require_trace_key(value: object) -> ProcessV2TraceKey:
    if (
        not isinstance(value, tuple)
        or len(value) != len(TRACE_KEY_FIELDS)
        or not isinstance(value[0], str)
        or type(value[1]) is not int
        or not isinstance(value[2], str)
    ):
        raise ProcessV2Active8DecisionIndexError(
            "a Process-V2 Active8 trace key is "
            f"{tuple(TRACE_KEY_FIELDS)}; a bare trace id is not an address, because "
            "an id is unique only within one V1 task"
        )
    return (str(value[0]), int(value[1]), str(value[2]))


def _resolved_row(task: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "v1_task_identity_sha256": str(row["v1_task_identity_sha256"]),
        "entry_index": int(row["entry_index"]),
        "trace_id": str(row["trace_id"]),
        "data_lane": str(row["data_lane"]),
        "split": str(row["split"]),
        "task_identity_sha256": str(task["task_identity_sha256"]),
        "category": row["category"],
        "active8_status": str(row["active8_status"]),
        "upstream_rejection_code": row["upstream_rejection_code"],
        "path_length": int(row["address"]["path_length"]),
        "accepted_transition_count": (
            int(row["address"]["path_length"]) if row["category"] is None else 0
        ),
        "decision_sha256": str(row["decision_sha256"]),
    }


def _transitions(
    *,
    task: Mapping[str, Any],
    row: Mapping[str, Any],
    addressed: Any,
    index_identity_sha256: str,
) -> list[dict[str, Any]]:
    """Project one accepted decision row onto its exact teacher transitions."""

    path_length = int(row["address"]["path_length"])
    if (
        len(row["actions"]) != path_length
        or len(addressed.trace.steps) != path_length
        or addressed.path.path_length != path_length
    ):
        raise ProcessV2Active8DecisionIndexError(
            "the accepted decision row and its cached trace disagree on path length"
        )
    transitions: list[dict[str, Any]] = []
    for step_index, action in enumerate(row["actions"]):
        evidence = action["candidate_evidence"]
        if evidence is None:
            raise ProcessV2Active8DecisionIndexError(
                "an accepted transition carries no production candidate evidence"
            )
        step = addressed.trace.steps[step_index]
        record = action_codec_v4.encode_action(str(step.rule_name), step.action)
        if canonical_sha256(record) != action["action_sha256"]:
            raise ProcessV2Active8DecisionIndexError(
                "the cached teacher action does not encode to its published identity"
            )
        source = addressed.path.state_at(step_index)
        successor = addressed.path.state_at(step_index + 1)
        source_sha256 = persistent_slot_state_sha256(source)
        successor_sha256 = persistent_slot_state_sha256(successor)
        if (
            source_sha256 != evidence["source_state_sha256"]
            or successor_sha256 != evidence["target_state_sha256"]
        ):
            raise ProcessV2Active8DecisionIndexError(
                "the cached exact states differ from the published candidate evidence"
            )
        transitions.append(
            {
                "v1_task_identity_sha256": str(row["v1_task_identity_sha256"]),
                "entry_index": int(row["entry_index"]),
                "trace_id": str(row["trace_id"]),
                "index_identity_sha256": index_identity_sha256,
                "task_identity_sha256": str(task["task_identity_sha256"]),
                "data_lane": str(row["data_lane"]),
                "split": str(row["split"]),
                "step_index": step_index,
                "executor_rule": str(action["executor_rule"]),
                "model_family": action["model_family"],
                "action_sha256": str(action["action_sha256"]),
                "action_record": dict(record),
                "source_progress_index": step_index,
                "successor_progress_index": step_index + 1,
                "source_state_sha256": source_sha256,
                "successor_state_sha256": successor_sha256,
                "successor_is_terminal": step_index + 1 == path_length,
                "canonical_successor_key": str(evidence["canonical_successor_key"]),
                "raw_mark_count": int(evidence["raw_mark_count"]),
                "canonical_successor_count": int(evidence["canonical_successor_count"]),
                "matching_mark_count": int(evidence["matching_mark_count"]),
                "exact_successor_mark_count": int(evidence["exact_successor_mark_count"]),
                "successor_alias_count": int(evidence["successor_alias_count"]),
                "source_state": source,
                "successor_state": successor,
            }
        )
    return transitions


def _index_identity(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    completion_file_sha256: str,
    completion_sha256: str,
    counts: Mapping[str, int],
    rejection_census: Mapping[str, int],
    exclusion_reason_histogram: Mapping[str, int],
    upstream_reason_histogram: Mapping[str, int],
    action_family_histogram: Mapping[str, int],
) -> dict[str, Any]:
    """The one deterministic identity body, built from values rather than an object.

    Taking the pieces rather than the index means the resolver can compute the
    identity before the object exists, so the object is never constructed with a
    placeholder identity it later overwrites.
    """

    body = {
        "schema": INDEX_SCHEMA,
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": INDEX_STATUS,
        **authority_false_block(),
        "plan_file_sha256": plan_file_sha256,
        "plan_sha256": str(plan["plan_sha256"]),
        "run_identity_sha256": str(plan["run_identity_sha256"]),
        "completion_file_sha256": completion_file_sha256,
        "completion_sha256": completion_sha256,
        "implementation_sha256": str(plan["implementation_sha256"]),
        "process_identity_sha256": str(
            plan["process_v2_identity"]["process_identity_sha256"]
        ),
        "pinned_payload_process_identity_sha256": str(
            plan["pinned_process_identity"]["process_identity_sha256"]
        ),
        "policy_sha256": str(plan["policy"]["policy_sha256"]),
        "model_runtime_identity_sha256": str(
            plan["model_runtime_identity"]["identity_sha256"]
        ),
        "admitted_source_sha256": str(plan["admitted_source_sha256"]),
        "cache_binding": dict(plan["cache_binding"]),
        "rebind_binding": dict(plan["rebind_binding"]),
        "counts": dict(counts),
        "rejected_traces_by_code": dict(rejection_census),
        "active8_exclusion_reason_histogram": dict(exclusion_reason_histogram),
        "upstream_rejection_reason_histogram": dict(upstream_reason_histogram),
        "action_family_histogram": dict(action_family_histogram),
    }
    return {**body, "index_identity_sha256": canonical_sha256(body)}


def resolve_process_v2_active8_decision_index(
    *,
    plan_path: Path,
    completion_path: Path,
    artifact_root: Path,
    repo_root: Path,
) -> ProcessV2Active8DecisionIndex:
    """Reopen a completed run from disk and build the verified index.

    Every artifact is read from the volume, not taken from a caller's object:
    the published plan bytes, the completion bytes, every receipt, every decision
    shard, and the cache chunks the transitions come from.
    """

    root = Path(artifact_root).resolve()
    plan_file = Path(plan_path)
    completion_file = Path(completion_path)
    try:
        plan_bytes = plan_file.read_bytes()
        completion_bytes = completion_file.read_bytes()
    except OSError as error:
        raise ProcessV2Active8Incomplete(
            f"the Process-V2 Active8 run is not readable at {plan_file}"
        ) from error
    try:
        plan = validate_process_v2_active8_plan(json.loads(plan_bytes), repo_root=repo_root)
    except (json.JSONDecodeError, ProcessV2Active8MapReduceError) as error:
        raise ProcessV2Active8DecisionIndexError(
            "the supplied Process-V2 Active8 plan failed revalidation; a V1 decision "
            "plan, a V1 process identity or a V1 admission policy is refused here"
        ) from error
    if canonical_bytes(plan) + b"\n" != plan_bytes:
        raise ProcessV2Active8DecisionIndexError(
            "the published Process-V2 Active8 plan is not canonical bytes"
        )
    try:
        validate_process_v2_active8_model_runtime_identity(plan["model_runtime_identity"])
    except ProcessV2Active8RuntimeError as error:
        raise ProcessV2Active8DecisionIndexError(
            "the run's model runtime identity does not describe a Process-V2 model"
        ) from error

    run_root = _mounted(
        plan["run_artifact_root"], artifact_root=root, field="plan.run_artifact_root"
    )
    if plan_file.resolve() != (run_root / PLAN_FILENAME).resolve():
        raise ProcessV2Active8DecisionIndexError(
            "the supplied plan path is not the one this run published"
        )
    if completion_file.resolve() != (run_root / COMPLETION_FILENAME).resolve():
        raise ProcessV2Active8DecisionIndexError(
            "the supplied completion path is not the one this run published"
        )

    completion = json.loads(completion_bytes)
    if not isinstance(completion, dict):
        raise ProcessV2Active8DecisionIndexError(
            "the Process-V2 Active8 completion is not a JSON object"
        )
    body = {key: item for key, item in completion.items() if key != "completion_sha256"}
    try:
        require_authority_false(completion, label="the Process-V2 Active8 completion")
    except ProcessV2SchemaError as error:
        raise ProcessV2Active8DecisionIndexError(str(error)) from error
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("schema_version") != COMPLETION_SCHEMA_VERSION
        or completion.get("status") != COMPLETION_STATUS
        or completion.get("completion_sha256") != canonical_sha256(body)
        or canonical_bytes(completion) + b"\n" != completion_bytes
    ):
        raise ProcessV2Active8DecisionIndexError(
            "the Process-V2 Active8 completion is malformed or not canonical"
        )
    if (
        completion.get("run_identity_sha256") != plan["run_identity_sha256"]
        or completion.get("plan_sha256") != plan["plan_sha256"]
        or completion.get("implementation_sha256") != plan["implementation_sha256"]
        or completion.get("policy_sha256") != plan["policy"]["policy_sha256"]
        or completion.get("model_runtime_identity_sha256")
        != plan["model_runtime_identity"]["identity_sha256"]
        or completion.get("process_v2_identity_sha256")
        != plan["process_v2_identity"]["process_identity_sha256"]
    ):
        raise ProcessV2Active8DecisionIndexError(
            "the Process-V2 Active8 completion does not seal the plan it was published beside"
        )

    complete = completed_process_v2_active8_task_ids(
        plan, artifact_root=root, repo_root=repo_root
    )
    missing = {task["task_identity_sha256"] for task in plan["tasks"]} - complete
    if missing:
        raise ProcessV2Active8Incomplete(
            f"the Process-V2 Active8 run is missing {len(missing)} planned task results"
        )

    counts: dict[str, int] = dict.fromkeys(COUNT_FIELDS, 0)
    families: dict[str, int] = {}
    reasons: dict[str, int] = {}
    upstream_reasons: dict[str, int] = {}
    lookup: dict[ProcessV2TraceKey, tuple[str, str | None, str, int]] = {}
    tasks_by_id = {str(task["task_identity_sha256"]): dict(task) for task in plan["tasks"]}
    for task in reduction_order(plan["tasks"]):
        bound = tasks_by_id[str(task["task_identity_sha256"])]
        output = _mounted(
            bound["output_artifact_path"],
            artifact_root=root,
            field="task.output_artifact_path",
        )
        receipt = validate_process_v2_active8_task_result(output, plan=plan, task=bound)
        raw = gzip.decompress((output / DECISION_FILENAME).read_bytes())
        for row in read_process_v2_active8_decision_rows(raw, task=bound, plan=plan):
            key = (
                str(row["v1_task_identity_sha256"]),
                int(row["entry_index"]),
                str(row["trace_id"]),
            )
            if key in lookup:
                raise ProcessV2Active8DecisionIndexError(
                    f"the Process-V2 Active8 run decides the trace key {key} twice"
                )
            lookup[key] = (
                str(bound["task_identity_sha256"]),
                row["category"],
                str(row["decision_sha256"]),
                int(row["address"]["path_length"]),
            )
            category = row["category"]
            if category is not None and category not in REJECTION_CATEGORIES:
                raise ProcessV2Active8DecisionIndexError(
                    f"a decision row names an undeclared rejection category {category!r}"
                )
            if category == UPSTREAM_REJECTED:
                code = str(row["upstream_rejection_code"])
                upstream_reasons[code] = upstream_reasons.get(code, 0) + 1
            for exclusion in row["active8_exclusions"]:
                reason = str(exclusion["reason"])
                reasons[reason] = reasons.get(reason, 0) + 1
            for family, value in row["action_family_histogram"].items():
                families[str(family)] = families.get(str(family), 0) + int(value)
        for field in COUNT_FIELDS:
            counts[field] += int(receipt["counts"][field])

    if counts != completion["active8_counts"]:
        raise ProcessV2Active8DecisionIndexError(
            "the reopened Process-V2 Active8 census differs from the published completion"
        )
    if (
        dict(sorted(families.items())) != completion["active8_action_family_histogram"]
        or dict(sorted(reasons.items()))
        != completion["active8_exclusion_reason_histogram"]
        or dict(sorted(upstream_reasons.items()))
        != completion["upstream_rejection_reason_histogram"]
    ):
        raise ProcessV2Active8DecisionIndexError(
            "the reopened Process-V2 Active8 histograms differ from the published completion"
        )
    for field in ACTIVE8_CENSUS_FIELDS:
        if field not in counts:
            raise ProcessV2Active8DecisionIndexError(
                f"the Process-V2 Active8 census omits {field!r}"
            )
    if counts["source_entries"] != (
        counts["upstream_rejected_entries"]
        + counts["active8_accepted_entries"]
        + counts["active8_excluded_entries"]
    ):
        raise ProcessV2Active8DecisionIndexError(
            "the reopened Process-V2 Active8 census does not reconcile"
        )
    try:
        require_census_reconciles(
            structural_census(counts), label="the Process-V2 Active8 index"
        )
    except ProcessV2SchemaError as error:
        raise ProcessV2Active8DecisionIndexError(str(error)) from error
    if len(lookup) != counts["source_entries"]:
        raise ProcessV2Active8DecisionIndexError(
            "the Process-V2 Active8 trace lookup does not cover the exact census"
        )

    # The two categories keep their own namespaces, so a consumer summing the
    # reason census cannot merge "never evaluated" with "evaluated and refused".
    rejection_census = {
        **{f"{UPSTREAM_REJECTED}:{code}": value for code, value in sorted(upstream_reasons.items())},
        **{f"{ACTIVE8_EXCLUDED}:{reason}": value for reason, value in sorted(reasons.items())},
    }
    plan_file_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    completion_file_sha256 = hashlib.sha256(completion_bytes).hexdigest()
    resolved_counts = {**counts, **structural_census(counts)}
    identity = _index_identity(
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        completion_file_sha256=completion_file_sha256,
        completion_sha256=str(completion["completion_sha256"]),
        counts=resolved_counts,
        rejection_census=rejection_census,
        exclusion_reason_histogram=dict(sorted(reasons.items())),
        upstream_reason_histogram=dict(sorted(upstream_reasons.items())),
        action_family_histogram=dict(sorted(families.items())),
    )
    return ProcessV2Active8DecisionIndex(
        plan=plan,
        completion=completion,
        artifact_root=root,
        repo_root=Path(repo_root),
        plan_file_sha256=plan_file_sha256,
        completion_file_sha256=completion_file_sha256,
        counts=counts,
        rejection_census=rejection_census,
        exclusion_reason_histogram=dict(sorted(reasons.items())),
        upstream_reason_histogram=dict(sorted(upstream_reasons.items())),
        action_family_histogram=dict(sorted(families.items())),
        trace_lookup=lookup,
        index_identity_sha256=str(identity["index_identity_sha256"]),
    )


__all__ = [
    "ACCEPTED_TRANSITION_FIELDS",
    "INDEX_SCHEMA",
    "INDEX_SCHEMA_VERSION",
    "INDEX_STATUS",
    "JOIN_KEY_FIELDS",
    "RESOLVED_TRACE_FIELDS",
    "ProcessV2Active8DecisionIndex",
    "ProcessV2Active8DecisionIndexError",
    "resolve_process_v2_active8_decision_index",
]
