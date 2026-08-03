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

WHERE THE CHEMISTRY IS DERIVED, AND WHY NOT HERE
------------------------------------------------
Candidate enumeration belongs to the Active8 MAP TASK, which holds the model and
the production checker and is already enumerating: it derives the evidence, and
the row it publishes is bound to its receipt by exact decision bytes.  This index
does NOT reconstruct a model and does NOT rerun the production candidate checker
-- that is a prohibition, not a performance preference, because Gate 0 fans this
index out over up to forty chunk-parallel map tasks and none of them may carry a
model.  Independent re-enumeration lives in bounded tests and sentinels.

What the read side owes is therefore verification of the BINDING plus everything
derivable without enumeration, and it takes all of it:

* the action identity and family, from the frozen codec applied to the CACHED
  step -- never read off the row;
* both exact persistent-slot state hashes, from the cached path;
* ``canonical_successor_key``, from the production canonicalizer applied to the
  cached successor state.  That is one canonicalization of a state already in
  hand, not an enumeration, and it is the same function the policy used;
* the structural contradictions: an accepted trace whose evidence says
  ``supported == false`` or carries an exclusion reason, and evidence whose
  counts are not the positive alias geometry an accepted transition has;
* the completion inventory, against the receipts the tasks still hold.

What is published is the DERIVED value in every case above, so a consumer
downstream reads a recomputed identity rather than a reported one.

THE COUNTS ARE THE RESIDUAL, AND IT IS STATED
---------------------------------------------
The five per-action candidate counts cannot be re-derived without enumerating,
so on this path they are checked structurally: every row-level total is
recomputed from the per-action evidence it summarises, the arithmetic bounds are
required, and an accepted transition must match exactly one teacher mark with a
positive alias geometry.  A count moved WITHIN those bounds and propagated
consistently through the row total, the receipt and the completion is not
detectable here; it is detectable only by re-enumeration, which is what the map
task does at write time and what the bounded sentinel does in tests.

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
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_EXCLUDED,
    JOIN_KEY_FIELDS,
    REJECTION_CATEGORIES,
    RESOLVED_TRACE_ROW_FIELDS,
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
    result_inventory_row,
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
from compose_v4.rewrite.action_codec_v4 import ActionCodecV4Error
from compose_v4.rewrite.kernel import canonical_state_key

INDEX_SCHEMA = "compose.data.editing_v2_process_v2_active8_decision_index"
INDEX_SCHEMA_VERSION = 1
INDEX_STATUS = "VERIFIED_PROCESS_V2_ACTIVE8_INDEX_NO_DOWNSTREAM_AUTHORITY"

#: The published row: exactly the seam's frozen minimum, plus this index's own
#: provenance.  Built FROM `RESOLVED_TRACE_ROW_FIELDS` rather than alongside it,
#: so the two cannot drift -- which they did once, when this module spelled the
#: partition role `split` and the rejection category `category` while Gate 0
#: required the seam's names.  Both suites passed and the chain did not join.
RESOLVED_TRACE_FIELDS: tuple[str, ...] = (
    *RESOLVED_TRACE_ROW_FIELDS,
    "data_lane",
    "task_identity_sha256",
    "active8_status",
    "upstream_rejection_code",
    "path_length",
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

#: The ONLY partition roles whose molecular states this index will decode.
#:
#: Held-out sealing is a property of what was never READ, not of what was never
#: counted.  Resolved-trace METADATA -- the census, the categories, the decision
#: hashes -- stays available for every role, because a census that omitted the
#: held-out partitions would not reconcile; but no cache chunk of a held-out role
#: is ever opened, so no validation, controller-validation or final-test molecule
#: enters this process.  Bound to ``("train",)``, which is the same bound the
#: Gate-0 contract asserts on its own decision-eligible roles.
DECISION_ELIGIBLE_PARTITION_ROLES: tuple[str, ...] = ("train",)


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
        result_inventory: Sequence[Mapping[str, Any]],
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
        self._result_inventory = MappingProxyType(
            {
                str(item["task_identity_sha256"]): MappingProxyType(dict(item))
                for item in result_inventory
            }
        )

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
        the cached record's own arrays through the production chunk reader, and
        every transition is produced by the one shared in-memory validator, which
        re-derives its whole candidate-evidence payload.

        This is a POINT lookup and it costs one chunk decode, because a chunk is
        the smallest thing the cache addresses.  A consumer that wants every
        accepted transition must use :meth:`iter_accepted_trace_transitions`,
        which decodes each SELECTED chunk exactly once: calling this in a loop
        over the corpus would decode one chunk per trace and reinstate exactly
        the triangular rescan the chunk cache exists to remove.
        """

        task, row = self._row_for(trace_key)
        if row["category"] is not None:
            return
        self._require_decision_eligible(str(row["split"]))
        addressed = self._addressed_trace(task, int(row["entry_index"]), str(row["trace_id"]))
        for transition in self._validated_transitions(
            task=task, row=row, addressed=addressed
        ):
            yield MappingProxyType(transition)

    def result_inventory(self) -> Mapping[str, Mapping[str, Any]]:
        """The reconciled completion inventory, addressed by task identity.

        Every row here was required at resolution to equal the row rebuilt from
        that task's LIVE receipt, so a parallel consumer that owns one task can
        check the receipt it opens against this one row instead of re-reading the
        run.  Published as the fan-out primitive for exactly that reason.
        """

        return self._result_inventory

    def decision_eligible_task_identities(
        self, *, partition_roles: Sequence[str] | None = None
    ) -> tuple[str, ...]:
        """The task identities a role-filtered pass would open, in reduction order.

        The unit a chunk-parallel consumer should shard on: one task is one cache
        chunk, so a map task that takes one identity from here decodes exactly one
        chunk and no held-out chunk is in the list at all.
        """

        roles = self._selected_roles(partition_roles)
        return tuple(
            str(task["task_identity_sha256"])
            for task in reduction_order(self._plan["tasks"])
            if str(self._tasks[str(task["task_identity_sha256"])]["split"]) in roles
        )

    def iter_accepted_trace_transitions(
        self,
        *,
        partition_roles: Sequence[str] | None = None,
        task_identity_sha256: str | None = None,
    ) -> Iterator[tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]]:
        """Every decision-eligible accepted trace with its validated transitions.

        The bulk execution path, and the only one a whole-corpus consumer should
        use.  Three properties it has and the point lookup does not:

        * it is ROLE FILTERED BEFORE ANY CHUNK IS OPENED.  A task whose partition
          role is not selected is skipped at the task level, so no held-out
          molecular state is decoded -- held-out data stays sealed because it was
          never read, not because it was read and then not counted;
        * each selected chunk and each selected decision shard is streamed
          exactly once, and the two are joined in memory by ascending entry
          index;
        * every transition is validated by the same shared in-memory validator
          the point API uses, against the task, row and trace already in hand.
          The bulk path never calls :meth:`validate_accepted_transition`, whose
          own lookup would reinstate the per-transition rescan.

        ``task_identity_sha256`` narrows the pass to ONE task, which is one cache
        chunk: the shard unit for a chunk-parallel consumer.  A map task that
        passes its own identity opens its own chunk and nothing else, and
        validates it from that chunk's receipt plus this index's reconciled
        completion inventory -- no global pass per container.
        """

        roles = self._selected_roles(partition_roles)
        if task_identity_sha256 is not None and task_identity_sha256 not in self._tasks:
            raise ProcessV2Active8DecisionIndexError(
                f"the task {task_identity_sha256!r} is not one this run planned"
            )
        for task in reduction_order(self._plan["tasks"]):
            bound = self._tasks[str(task["task_identity_sha256"])]
            if task_identity_sha256 is not None and (
                str(bound["task_identity_sha256"]) != task_identity_sha256
            ):
                continue
            if str(bound["split"]) not in roles:
                if task_identity_sha256 is not None:
                    self._require_decision_eligible(str(bound["split"]))
                continue
            output = self._task_root(bound)
            validate_process_v2_active8_task_result(output, plan=self._plan, task=bound)
            raw = gzip.decompress((output / DECISION_FILENAME).read_bytes())
            rows = {
                int(row["entry_index"]): row
                for row in read_process_v2_active8_decision_rows(
                    raw, task=bound, plan=self._plan
                )
                if row["category"] is None
            }
            if not rows:
                continue
            for read in self._iter_chunk_rows(bound):
                row = rows.get(read.entry_index)
                if row is None:
                    continue
                if read.addressed is None or read.addressed.address.trace_id != str(
                    row["trace_id"]
                ):
                    raise ProcessV2Active8DecisionIndexError(
                        f"the cached row at entry {read.entry_index} is not the indexed trace"
                    )
                transitions = tuple(
                    MappingProxyType(transition)
                    for transition in self._validated_transitions(
                        task=bound, row=row, addressed=read.addressed
                    )
                )
                yield MappingProxyType(_resolved_row(bound, row)), transitions

    def iter_accepted_transitions(
        self,
        *,
        partition_roles: Sequence[str] | None = None,
        task_identity_sha256: str | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Every decision-eligible accepted transition, in reduction order.

        The flat projection of :meth:`iter_accepted_trace_transitions`; it holds
        every one of that method's properties, including the role filter.
        """

        for _row, transitions in self.iter_accepted_trace_transitions(
            partition_roles=partition_roles, task_identity_sha256=task_identity_sha256
        ):
            yield from transitions

    def validate_accepted_transition(self, transition: Mapping[str, Any]) -> None:
        """Raise unless this is exactly what the index published.

        The POINT form: it performs its own lookup and then calls the shared
        in-memory validator.  A whole-corpus consumer must not call this in a
        loop -- the lookup, not the validation, is what makes it expensive, and
        :meth:`iter_accepted_trace_transitions` validates just as completely with
        the row and trace already in hand.
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
        self._require_decision_eligible(str(row["split"]))
        addressed = self._addressed_trace(task, int(row["entry_index"]), str(row["trace_id"]))
        self._require_published_transition(
            transition,
            published=self._validated_transitions(task=task, row=row, addressed=addressed),
        )

    # ---- Internals ----

    def _selected_roles(self, partition_roles: Sequence[str] | None) -> frozenset[str]:
        eligible = frozenset(DECISION_ELIGIBLE_PARTITION_ROLES)
        if partition_roles is None:
            return eligible
        roles = frozenset(str(role) for role in partition_roles)
        if not roles or not roles <= eligible:
            raise ProcessV2Active8DecisionIndexError(
                "this index decodes molecular states only for the decision-eligible "
                f"partition roles {list(DECISION_ELIGIBLE_PARTITION_ROLES)}; "
                f"{sorted(roles - eligible)} is held out and must stay unread"
            )
        return roles

    def _require_decision_eligible(self, partition_role: str) -> None:
        if partition_role not in DECISION_ELIGIBLE_PARTITION_ROLES:
            raise ProcessV2Active8DecisionIndexError(
                f"the partition role {partition_role!r} is held out; its molecular states "
                "are sealed and this index will not decode them"
            )

    def _validated_transitions(
        self,
        *,
        task: Mapping[str, Any],
        row: Mapping[str, Any],
        addressed: Any,
    ) -> list[dict[str, Any]]:
        """The ONE in-memory validator, over objects the caller already holds.

        Both the point API and the bulk API produce transitions through here and
        nowhere else, so there is exactly one definition of what an accepted
        transition is and of what has to be re-derived before one is published.
        No model is constructed: everything derivable without enumeration is
        derived, and the enumeration itself was done by the map task.
        """

        return _transitions(
            task=task,
            row=row,
            addressed=addressed,
            index_identity_sha256=self._index_identity_sha256,
        )

    def _require_published_transition(
        self, transition: Mapping[str, Any], *, published: Sequence[Mapping[str, Any]]
    ) -> None:
        step_index = transition["step_index"]
        if type(step_index) is not int or not 0 <= step_index < len(published):
            raise ProcessV2Active8DecisionIndexError(
                "the transition step index lies outside its trace"
            )
        candidate = published[step_index]
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

    def _iter_chunk_rows(self, task: Mapping[str, Any]) -> Iterator[Any]:
        """Decode one task's cache chunk once, through the production reader.

        The one place a chunk is opened, so the point lookup and the bulk stream
        cannot drift into two different readings of the same bytes.
        """

        target = _chunk_target(task)
        source_output = _mounted(
            target.source_artifact_path,
            artifact_root=self._artifact_root,
            field="task.cache_source_artifact_path",
        )
        try:
            yield from read_process_v2_chunk_target(
                source_output,
                target=target,
                expected_process_identity=self._plan["pinned_process_identity"],
                sentinel_replay_entries=0,
                recover_row_errors=False,
                repo_root=self._repo_root,
            )
        except ProcessV2ChunkCacheError as error:
            raise ProcessV2Active8DecisionIndexError(
                f"the cached chunk of task {task['task_identity_sha256']} is unreadable"
            ) from error

    def _addressed_trace(
        self, task: Mapping[str, Any], entry_index: int, trace_id: str
    ) -> Any:
        for read in self._iter_chunk_rows(task):
            if read.entry_index != entry_index:
                continue
            if read.addressed is None or read.addressed.address.trace_id != trace_id:
                raise ProcessV2Active8DecisionIndexError(
                    f"the cached row at entry {entry_index} is not the indexed trace"
                )
            return read.addressed
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
        "partition_role": str(row["split"]),
        "task_identity_sha256": str(task["task_identity_sha256"]),
        "rejection_category": row["category"],
        "active8_status": str(row["active8_status"]),
        "upstream_rejection_code": row["upstream_rejection_code"],
        "path_length": int(row["address"]["path_length"]),
        "accepted_transition_count": (
            int(row["address"]["path_length"]) if row["category"] is None else 0
        ),
        "decision_sha256": str(row["decision_sha256"]),
    }


def _require_accepted_evidence(payload: Mapping[str, Any], *, step_index: int) -> None:
    """What evidence must SAY for its transition to be an accepted one.

    Structural, so it holds on the read path where no model exists.  The
    contradiction cases are the load-bearing ones: an accepted trace whose own
    evidence refuses the action, and counts that are not the positive alias
    geometry an admitted teacher necessarily has.
    """

    if payload["supported"] is not True or payload["exclusion_reason"] is not None:
        raise ProcessV2Active8DecisionIndexError(
            f"the evidence at step {step_index} does not support the teacher action, so "
            "this transition is not an accepted one; an accepted trace cannot carry "
            "unsupported evidence or a populated exclusion reason"
        )
    if payload["matching_mark_count"] != 1:
        raise ProcessV2Active8DecisionIndexError(
            f"the teacher action at step {step_index} matches "
            f"{payload['matching_mark_count']} marks of the production marked law; an "
            "accepted transition matches exactly one"
        )
    if (
        payload["exact_successor_mark_count"] != 1
        or payload["successor_alias_count"] < 1
        or payload["canonical_successor_count"] < 1
        or payload["canonical_successor_count"] > payload["raw_mark_count"]
        or payload["successor_alias_count"] > payload["raw_mark_count"]
        or payload["exact_successor_mark_count"] > payload["successor_alias_count"]
    ):
        raise ProcessV2Active8DecisionIndexError(
            f"the candidate counts at step {step_index} are not the positive, consistent "
            "alias geometry an accepted transition requires"
        )


def _transitions(
    *,
    task: Mapping[str, Any],
    row: Mapping[str, Any],
    addressed: Any,
    index_identity_sha256: str,
) -> list[dict[str, Any]]:
    """Project one accepted decision row onto its exact teacher transitions.

    Every identity here is RE-DERIVED and the derived value is what is published:
    the action identity and family from the frozen codec applied to the CACHED
    step, both exact states from the cached path, and the canonical successor key
    from the production canonicalizer applied to the cached successor state.  The
    stored evidence is then required to agree with each.  No model is
    constructed and nothing is enumerated -- the enumeration was the map task's
    job and its result is bound to the receipt by exact decision bytes.
    """

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
        stored = action["candidate_evidence"]
        if not isinstance(stored, Mapping):
            raise ProcessV2Active8DecisionIndexError(
                "an accepted transition carries no production candidate evidence"
            )
        step = addressed.trace.steps[step_index]
        executor_rule = str(step.rule_name)
        try:
            record = action_codec_v4.encode_action(executor_rule, step.action)
            model_family = action_codec_v4.canonical_family(executor_rule)
        except ActionCodecV4Error as error:
            raise ProcessV2Active8DecisionIndexError(
                "the cached teacher action is outside ActionCodecV4"
            ) from error
        action_sha256 = canonical_sha256(record)
        if (
            action_sha256 != action["action_sha256"]
            or executor_rule != action["executor_rule"]
            or model_family != action["model_family"]
        ):
            raise ProcessV2Active8DecisionIndexError(
                "the cached teacher action does not encode to its published identity"
            )
        source = addressed.path.state_at(step_index)
        successor = addressed.path.state_at(step_index + 1)
        source_sha256 = persistent_slot_state_sha256(source)
        successor_sha256 = persistent_slot_state_sha256(successor)
        if (
            source_sha256 != stored["source_state_sha256"]
            or successor_sha256 != stored["target_state_sha256"]
        ):
            raise ProcessV2Active8DecisionIndexError(
                "the cached exact states differ from the published candidate evidence"
            )
        # The canonical successor key is a pure function of the successor STATE,
        # which is in hand, so it is re-derived rather than read: one
        # canonicalization through the production canonicalizer the policy itself
        # used, not an enumeration and not a second evaluator.
        canonical_key = canonical_state_key(successor)
        if canonical_key != stored["canonical_successor_key"]:
            raise ProcessV2Active8DecisionIndexError(
                "the cached successor does not canonicalize to the published candidate "
                f"evidence key at step {step_index}"
            )
        payload = dict(stored)
        _require_accepted_evidence(payload, step_index=step_index)
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
                "executor_rule": executor_rule,
                "model_family": model_family,
                "action_sha256": action_sha256,
                "action_record": dict(record),
                "source_progress_index": step_index,
                "successor_progress_index": step_index + 1,
                "source_state_sha256": source_sha256,
                "successor_state_sha256": successor_sha256,
                "successor_is_terminal": step_index + 1 == path_length,
                "canonical_successor_key": canonical_key,
                "raw_mark_count": int(payload["raw_mark_count"]),
                "canonical_successor_count": int(payload["canonical_successor_count"]),
                "matching_mark_count": int(payload["matching_mark_count"]),
                "exact_successor_mark_count": int(payload["exact_successor_mark_count"]),
                "successor_alias_count": int(payload["successor_alias_count"]),
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
    inventory: list[dict[str, Any]] = []
    tasks_by_id = {str(task["task_identity_sha256"]): dict(task) for task in plan["tasks"]}
    for task in reduction_order(plan["tasks"]):
        bound = tasks_by_id[str(task["task_identity_sha256"])]
        output = _mounted(
            bound["output_artifact_path"],
            artifact_root=root,
            field="task.output_artifact_path",
        )
        receipt = validate_process_v2_active8_task_result(output, plan=plan, task=bound)
        inventory.append(result_inventory_row(bound, receipt))
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

    # ---- The completion inventory, against the receipts still on the volume ----
    #
    # A completion is a pointer document: it names one receipt and one decision
    # file per task and reports their counts.  Resolving without following those
    # pointers would let a stale inventory -- a receipt identity or a decision
    # file hash from a superseded execution of the same task -- survive under a
    # correctly resealed completion, and a chunk-parallel consumer that trusts one
    # inventory row instead of re-reading the run would then trust it.  So each
    # row is rebuilt from the plan task plus the LIVE receipt and required to be
    # equal, in reduction order.
    published_inventory = completion.get("result_inventory")
    if (
        not isinstance(published_inventory, list)
        or completion.get("result_inventory_sha256") != canonical_sha256(published_inventory)
        or completion.get("task_count") != len(inventory)
    ):
        raise ProcessV2Active8DecisionIndexError(
            "the Process-V2 Active8 completion does not seal its own result inventory"
        )
    if published_inventory != inventory:
        disagree = sorted(
            {
                str(row.get("task_identity_sha256"))
                for row in published_inventory
                if row not in inventory
            }
            | {
                str(row["task_identity_sha256"])
                for row in inventory
                if row not in published_inventory
            }
        )
        raise ProcessV2Active8DecisionIndexError(
            "the published Process-V2 Active8 result inventory does not match the receipts "
            f"the tasks actually hold: {disagree}"
        )
    inventory_counts = {
        field: sum(int(row["counts"][field]) for row in inventory) for field in COUNT_FIELDS
    }
    if inventory_counts != counts or counts != completion["active8_counts"]:
        raise ProcessV2Active8DecisionIndexError(
            "the reopened Process-V2 Active8 census differs from the published completion"
        )
    if completion.get("structural_counts") != structural_census(counts):
        raise ProcessV2Active8DecisionIndexError(
            "the published Process-V2 Active8 structural census does not project its counts"
        )
    if (
        completion.get("cache_binding") != plan["cache_binding"]
        or completion.get("rebind_binding") != plan["rebind_binding"]
        or completion.get("admitted_source_sha256") != plan["admitted_source_sha256"]
    ):
        raise ProcessV2Active8DecisionIndexError(
            "the Process-V2 Active8 completion binds different upstream sources than its plan"
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
        result_inventory=inventory,
        index_identity_sha256=str(identity["index_identity_sha256"]),
    )


__all__ = [
    "ACCEPTED_TRANSITION_FIELDS",
    "DECISION_ELIGIBLE_PARTITION_ROLES",
    "INDEX_SCHEMA",
    "INDEX_SCHEMA_VERSION",
    "INDEX_STATUS",
    "JOIN_KEY_FIELDS",
    "RESOLVED_TRACE_FIELDS",
    "ProcessV2Active8DecisionIndex",
    "ProcessV2Active8DecisionIndexError",
    "resolve_process_v2_active8_decision_index",
]
