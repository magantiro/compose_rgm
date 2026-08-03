"""Fail-closed resolution of the Process-V2 admitted editing source.

The completed V1 migration payload holds immutable *chemical* data.  Process V2
decides, separately and later, which of those traces its prospectively frozen
fiber can represent.  This module joins the two without ever relabelling one as
the other:

* a V1 row keeps its **V1 identity**: its V1 schema, its V1 process semantics,
  its superseded V1 process-identity hash and its own record hash;
* the **V2 admission overlay** is a separate object carrying the live Process-V2
  identity, the run and task that proved the row, and the proof hash;
* :class:`ProcessV2AdmittedRecord` holds both side by side, so a consumer can
  never read a V1 payload as if it had been produced under Process V2.

What "fail-closed" means here
-----------------------------

Resolution refuses, rather than degrades, when the overlay is

* **missing** - the run completion or any planned task result is absent;
* **incomplete** - the admitted and rejected entries of a V1 task do not cover
  its exact entry census, so some rows have no recorded decision;
* **stale** - the plan's serialized implementation, the live Process-V2 identity
  or the published plan bytes have moved since the overlay was written;
* **in disagreement with the payload** - a proof names a record hash, trace
  identity, lane, split, path length or canonical key sequence that the V1 row
  does not carry.

There is no partial mode.  A resolver that yielded the rows it happened to be
able to match would silently redefine the corpus.

What this is not
----------------

This is a validated *source*: an iterator, its identities and its measured
counts.  It is deliberately not an Active8 builder, and it grants no authority.
Every field of the frozen authority vocabulary is ``False``; Gate 0, T1 and P50
authority are not conferred, not implied, and not obtainable from this object.  A
later builder must record :meth:`ProcessV2AdmittedSource.identity` as its input
provenance and obtain its own authorization.

The identity is one exact production shape
------------------------------------------

Schema version 3 replaces two arrangements that let a consumer read a descriptor
this module never emits.

* **Authority was a set of mutable constructor arguments whose values the
  descriptor ignored.**  ``ProcessV2AdmittedSource(..., p50_authorized=True)``
  constructed cleanly and still published ``False``, so the object and its own
  artifact disagreed, and the field carried a spelling outside the frozen
  vocabulary.  The seven canonical fields are now read-only properties and the
  descriptor is built from :func:`authority_false_block`, so the object and the
  artifact are the same statement and there is one spelling.
* **No owning validator existed.**  Every consumer re-derived which fields it
  believed this identity carried, and the one that did so
  (``editing_v2_process_v2_evidence_binding``) required three names -- a physical
  inventory hash, a process identity hash and a semantic evidence hash -- that
  this adapter has never emitted under any version.  Nothing caught it because
  that consumer was only ever exercised against a hand-written fixture inventing
  those names.  :func:`validate_process_v2_admitted_source_identity` is now the
  single authority on the shape, and a consumer that wants to know what an
  admitted source looks like calls it instead of restating it.

Version 2 is refused rather than upgraded.  It spelled the bounded-P50 field
under the retired name, omitted three authority fields entirely, and declared no
version on its nested physical-execution body, so a converter would have to
invent the values it lacks.

Every mapping in the descriptor is emitted in sorted key order, at every depth.
That is what makes the artifact byte-stable through any canonical serializer:
a consumer that embeds it verbatim and re-serializes with ``sort_keys=True``
reproduces it exactly rather than reordering it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_FILENAME,
    COMPLETION_SCHEMA,
    COMPLETION_SCHEMA_VERSION,
    COMPLETION_STATUS,
    MANIFEST_FILENAME,
    PLAN_FILENAME,
    PROOF_FILENAME,
    ProcessV2RebindError,
    ProcessV2RebindExclusionCode,
    completed_process_v2_rebind_task_ids,
    editing_process_v2_identity,
    mounted_process_v2_artifact_path,
    validate_process_v2_rebind_plan,
    validate_process_v2_rebind_task_result,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    REJECTION_LEDGER_SCHEMA,
    REJECTION_LEDGER_SCHEMA_VERSION,
    ProcessV2SchemaError,
    authority_false_block,
    require_authority_false,
    require_no_granted_authority,
    verify_self_hash,
)
from compose_v4.data.packed_trace_store import AddressedPackedTrace
from compose_v4.data.semantic_packed_trace_store import (
    SemanticPackedStoreError,
    read_semantic_packed_artifact_range_rows,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    SEMANTIC_ARTIFACT_DIRNAME,
)

ADMITTED_SOURCE_SCHEMA = "compose.data.editing_process_v2_admitted_source"
# Version 2 added the streaming join's outputs: `decision_semantic_sha256`,
# `admitted_evidence_sha256`, `rejection_ledger` and `physical_execution_identity`.
# Version 3 makes the descriptor one exact shape with one owning validator: the
# seven canonical authority fields replace the four ad-hoc ones (which included a
# spelling outside the frozen vocabulary), every mapping is sorted at every depth,
# and the nested physical-execution body carries its own version instead of
# borrowing this one. None of the versions is a superset of the one before, so a
# consumer must import this constant instead of assuming a shape.
ADMITTED_SOURCE_SCHEMA_VERSION = 3
ADMITTED_SOURCE_STATUS = "V2_ADMISSION_OVERLAY_RESOLVED_NO_TRAINING_AUTHORITY"
ADMITTED_SOURCE_SELF_HASH_FIELD = "admitted_source_sha256"
V1_PAYLOAD_IDENTITY_SCHEMA = "compose.data.editing_process_v2_v1_payload_identity"
V1_PAYLOAD_IDENTITY_SCHEMA_VERSION = 1
ADMISSION_OVERLAY_SCHEMA = "compose.data.editing_process_v2_admission_overlay"
ADMISSION_OVERLAY_SCHEMA_VERSION = 1
REFUSAL_SCHEMA = "compose.data.editing_process_v2_admitted_source_refusal"
REFUSAL_SCHEMA_VERSION = 1
REFUSAL_STATUS = "INTEGRITY_REFUSAL_NOTHING_PUBLISHED_UNDER_THE_RUN_NAMESPACE"
DEFAULT_REJECTION_LEDGER_FILENAME = "PROCESS_V2_REJECTION_LEDGER.jsonl.gz"

_ADAPTER_SOURCE_FILE = "src/compose_v4/data/editing_process_v2_admitted_source.py"

# The nested operational replay address. It carries its OWN version: under
# version 2 it echoed `ADMITTED_SOURCE_SCHEMA_VERSION`, so a reader could not
# tell whether the block had changed or merely its parent had.
PHYSICAL_EXECUTION_IDENTITY_SCHEMA = f"{ADMITTED_SOURCE_SCHEMA}.physical_execution_identity"
PHYSICAL_EXECUTION_IDENTITY_SCHEMA_VERSION = 1
PHYSICAL_EXECUTION_SELF_HASH_FIELD = "physical_execution_sha256"

REJECTION_LEDGER_SELF_HASH_FIELD = "rejection_ledger_sha256"
REJECTION_LEDGER_SORT_ORDER = "v1_task_identity_sha256_then_entry_index"

#: The declared census. `source == admitted + rejected` and
#: `admitted_states == admitted_transitions + admitted_entries` both hold over
#: it; the second because an admitted entry is proved only when every one of its
#: persisted steps replays, so its state count is exactly its transition count
#: plus the source state.
_COUNT_FIELDS = (
    "source_entries",
    "admitted_entries",
    "rejected_entries",
    "admitted_states",
    "admitted_transitions",
)

#: Every measured SHA-256 the descriptor pins, and the complete set of them.
#: Named here rather than in each consumer: the consumer that named its own set
#: required three hashes this adapter has never emitted, and nothing failed
#: because that consumer was exercised only against a fixture inventing them.
_IDENTITY_SHA256_FIELDS = (
    "adapter_implementation_sha256",
    "admitted_evidence_sha256",
    "completion_sha256",
    "decision_semantic_sha256",
    "pinned_builder_identity_sha256",
    "pinned_process_identity_sha256",
    "plan_sha256",
    "process_v2_identity_sha256",
    "run_identity_sha256",
    "v1_payload_binding_sha256",
)

#: The exact field set of the nested physical-execution identity.
PHYSICAL_EXECUTION_IDENTITY_FIELDS: tuple[str, ...] = (
    "entries_per_task",
    "physical_execution_sha256",
    "plan_sha256",
    "range_task_count",
    "range_task_inventory_sha256",
    "run_identity_sha256",
    "schema",
    "schema_version",
)

#: The exact field set of the nested rejection ledger.
REJECTION_LEDGER_FIELDS: tuple[str, ...] = (
    "ledger_artifact_path",
    "ledger_file_sha256",
    "rejected_entries",
    "rejected_traces_by_code",
    "rejection_ledger_sha256",
    "rejection_semantic_sha256",
    "rejection_stream_sha256",
    "schema",
    "schema_version",
    "sort_order",
)

#: The exact field set of :meth:`ProcessV2AdmittedSource.identity`, sorted
#: because the descriptor itself is sorted at every depth.
ADMITTED_SOURCE_IDENTITY_FIELDS: tuple[str, ...] = tuple(
    sorted(
        (
            "schema",
            "schema_version",
            "status",
            *AUTHORITY_FIELDS,
            *_IDENTITY_SHA256_FIELDS,
            "physical_execution_identity",
            "rejection_ledger",
            "counts",
            "rejected_traces_by_code",
            ADMITTED_SOURCE_SELF_HASH_FIELD,
        )
    )
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ProcessV2AdmittedSourceError(RuntimeError):
    """The V2 admission overlay cannot be joined to the V1 payload."""


class ProcessV2AdmittedSourceIncomplete(ProcessV2AdmittedSourceError):
    """The overlay is absent or does not decide every V1 record."""


class ProcessV2AdmittedSourceIdentityError(ProcessV2AdmittedSourceError):
    """An admitted-source identity descriptor is malformed, stale, or claims authority.

    Distinct from the resolution errors above so a caller can tell "the overlay
    on the volume is unreadable" from "this descriptor does not describe an
    admitted source", which are different problems with different owners.
    """


# ---- Deterministic serialization ---------------------------------------------


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


def _self_hash(value: Mapping[str, object], *, field: str) -> str:
    return _canonical_sha256({key: item for key, item in value.items() if key != field})


def _sorted_deep(value: Any) -> Any:
    """Rebuild ``value`` with every mapping in sorted key order, at every depth.

    List order is preserved: a sequence's order is data, a mapping's is not.  The
    point is byte stability rather than aesthetics -- a consumer that embeds this
    descriptor verbatim and re-serializes it under ``sort_keys=True`` must
    reproduce it, not reorder it.
    """

    if isinstance(value, Mapping):
        return {key: _sorted_deep(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_sorted_deep(item) for item in value]
    return value


def _adapter_implementation_sha256(repo_root: Path) -> str:
    source = Path(repo_root) / _ADAPTER_SOURCE_FILE
    if not source.is_file():
        raise ProcessV2AdmittedSourceError(
            f"the admitted-source adapter implementation is absent: {source}"
        )
    return hashlib.sha256(source.read_bytes()).hexdigest()


# ---- Record-level identities --------------------------------------------------


@dataclass(frozen=True)
class ProcessV2V1PayloadIdentity:
    """The immutable V1 identity of one chemical record. Never relabelled."""

    data_lane: str
    split: str
    trace_id: str
    entry_index: int
    path_length: int
    record_sha256: str
    process_semantics: str
    process_identity_sha256: str
    process_contract_sha256: str
    trace_schema: str
    trace_schema_version: int
    v1_task_identity_sha256: str
    v1_task_artifact_path: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema": V1_PAYLOAD_IDENTITY_SCHEMA,
            "schema_version": V1_PAYLOAD_IDENTITY_SCHEMA_VERSION,
            "data_lane": self.data_lane,
            "split": self.split,
            "trace_id": self.trace_id,
            "entry_index": self.entry_index,
            "path_length": self.path_length,
            "record_sha256": self.record_sha256,
            "process_semantics": self.process_semantics,
            "process_identity_sha256": self.process_identity_sha256,
            "process_contract_sha256": self.process_contract_sha256,
            "trace_schema": self.trace_schema,
            "trace_schema_version": self.trace_schema_version,
            "v1_task_identity_sha256": self.v1_task_identity_sha256,
            "v1_task_artifact_path": self.v1_task_artifact_path,
        }


@dataclass(frozen=True)
class ProcessV2AdmissionOverlay:
    """The separate V2 decision about one V1 record, with its own identity."""

    admitted: bool
    run_identity_sha256: str
    task_identity_sha256: str
    process_v2_identity_sha256: str
    proof_sha256: str
    replayed_transitions: int
    atom_delete_teachers_by_candidate_source: Mapping[str, int]
    process_v2_atom_delete_candidates: Sequence[Sequence[int]]

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema": ADMISSION_OVERLAY_SCHEMA,
            "schema_version": ADMISSION_OVERLAY_SCHEMA_VERSION,
            "admitted": self.admitted,
            "run_identity_sha256": self.run_identity_sha256,
            "task_identity_sha256": self.task_identity_sha256,
            "process_v2_identity_sha256": self.process_v2_identity_sha256,
            "proof_sha256": self.proof_sha256,
            "replayed_transitions": self.replayed_transitions,
            "atom_delete_teachers_by_candidate_source": dict(
                self.atom_delete_teachers_by_candidate_source
            ),
            "process_v2_atom_delete_candidates": [
                list(slots) for slots in self.process_v2_atom_delete_candidates
            ],
        }


@dataclass(frozen=True)
class ProcessV2AdmittedRecord:
    """One admitted record: V1 chemistry, V1 identity, V2 admission overlay."""

    v1_identity: ProcessV2V1PayloadIdentity
    v2_admission: ProcessV2AdmissionOverlay
    v1_record: Mapping[str, Any]
    addressed: AddressedPackedTrace


# ---- Overlay index ------------------------------------------------------------


@dataclass(frozen=True)
class _EntryDecision:
    admitted: bool
    task_identity_sha256: str
    proof: Mapping[str, Any] | None
    rejection: Mapping[str, Any] | None


def _read_proof_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            rows.append(json.loads(raw_line))
    return rows


def _range_tasks_by_v1_task(plan: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group range tasks under their V1 task, in ascending entry order."""

    grouped: dict[str, list[dict[str, Any]]] = {
        str(task["v1_task_identity_sha256"]): []
        for task in plan["v1_payload_binding"]["v1_tasks"]
    }
    for task in plan["tasks"]:
        identity = str(task["v1_task_identity_sha256"])
        if identity not in grouped:
            raise ProcessV2AdmittedSourceError(
                f"the plan names a range task for an unbound V1 task: {identity}"
            )
        grouped[identity].append(dict(task))
    for ranges in grouped.values():
        ranges.sort(key=lambda task: (int(task["entry_start"]), int(task["entry_stop"])))
    return grouped


def _read_range_decisions(
    task: Mapping[str, Any], *, artifact_root: Path
) -> dict[int, _EntryDecision]:
    """Read the decisions of exactly one range task and nothing else.

    This is the unit that bounds memory: a range holds ``entries_per_task``
    decisions, not the corpus.  The previous resolver built one dictionary over
    every proof row of every task, then rebuilt it a second time to iterate.
    """

    output = mounted_process_v2_artifact_path(
        task["output_artifact_path"],
        artifact_root=artifact_root,
        field="task.output_artifact_path",
    )
    try:
        validate_process_v2_rebind_task_result(output)
    except ProcessV2RebindError as error:
        raise ProcessV2AdmittedSourceError(
            f"the V2 admission overlay task {task['task_identity_sha256']} is invalid"
        ) from error
    manifest = json.loads((output / MANIFEST_FILENAME).read_bytes())
    decisions: dict[int, _EntryDecision] = {}
    identity = str(task["v1_task_identity_sha256"])
    for proof in _read_proof_rows(output / PROOF_FILENAME):
        entry_index = int(proof["entry_index"])
        if entry_index in decisions:
            raise ProcessV2AdmittedSourceError(
                f"the V2 admission overlay decides one V1 entry twice: {identity}[{entry_index}]"
            )
        decisions[entry_index] = _EntryDecision(
            admitted=True,
            task_identity_sha256=str(task["task_identity_sha256"]),
            proof=proof,
            rejection=None,
        )
    for rejection in manifest["rejected_traces"]:
        entry_index = int(rejection["entry_index"])
        if entry_index in decisions:
            raise ProcessV2AdmittedSourceError(
                f"the V2 admission overlay decides one V1 entry twice: {identity}[{entry_index}]"
            )
        decisions[entry_index] = _EntryDecision(
            admitted=False,
            task_identity_sha256=str(task["task_identity_sha256"]),
            proof=None,
            rejection=rejection,
        )
    return decisions


class _LazyTaskDecisions(Mapping):
    """The decisions of one V1 task, read one range at a time.

    Duck-compatible with the ``dict[int, _EntryDecision]`` this used to be, so a
    caller (or a test) that indexes, iterates or measures it sees no difference
    except that the corpus is never resident.
    """

    def __init__(self, ranges: Sequence[Mapping[str, Any]], *, artifact_root: Path) -> None:
        self._ranges = list(ranges)
        self._artifact_root = artifact_root
        self._cursor: int | None = None
        self._window: dict[int, _EntryDecision] = {}

    def _range_holding(self, entry_index: int) -> int | None:
        for position, task in enumerate(self._ranges):
            if int(task["entry_start"]) <= entry_index < int(task["entry_stop"]):
                return position
        return None

    def _load(self, position: int) -> dict[int, _EntryDecision]:
        if self._cursor != position:
            self._window = _read_range_decisions(
                self._ranges[position], artifact_root=self._artifact_root
            )
            self._cursor = position
        return self._window

    def __getitem__(self, entry_index: int) -> _EntryDecision:
        position = self._range_holding(int(entry_index))
        if position is None:
            raise KeyError(entry_index)
        window = self._load(position)
        return window[int(entry_index)]

    def __iter__(self) -> Iterator[int]:
        for position in range(len(self._ranges)):
            yield from sorted(self._load(position))

    def __len__(self) -> int:
        return sum(len(self._load(position)) for position in range(len(self._ranges)))

    def stream(self) -> Iterator[tuple[int, _EntryDecision]]:
        """Yield ``(entry_index, decision)`` in ascending order, one range live."""

        for position in range(len(self._ranges)):
            window = self._load(position)
            for entry_index in sorted(window):
                yield entry_index, window[entry_index]


class _LazyOverlayIndex(MutableMapping):
    """``{v1_task_identity: decisions}`` where the values are read on demand."""

    def __init__(self, plan: Mapping[str, Any], *, artifact_root: Path) -> None:
        self._grouped = _range_tasks_by_v1_task(plan)
        self._artifact_root = artifact_root
        self._overrides: dict[str, Mapping[int, _EntryDecision]] = {}

    def __getitem__(self, identity: str) -> Mapping[int, _EntryDecision]:
        if identity in self._overrides:
            return self._overrides[identity]
        if identity not in self._grouped:
            raise KeyError(identity)
        return _LazyTaskDecisions(self._grouped[identity], artifact_root=self._artifact_root)

    def __setitem__(self, identity: str, value: Mapping[int, _EntryDecision]) -> None:
        if identity not in self._grouped:
            raise KeyError(identity)
        self._overrides[identity] = value

    def __delitem__(self, identity: str) -> None:
        raise TypeError("the V2 admission overlay index is not deletable")

    def __iter__(self) -> Iterator[str]:
        return iter(self._grouped)

    def __len__(self) -> int:
        return len(self._grouped)


def _overlay_index(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> MutableMapping:
    """Index every recorded V2 decision by V1 task identity and entry index.

    The mapping is lazy: indexing a V1 task streams its range results and keeps
    at most one range resident.  The signature and the observable behaviour are
    unchanged, so the guards written against the eager version still hold.
    """

    return _LazyOverlayIndex(plan, artifact_root=artifact_root)


def _decisions_for(
    index: MutableMapping, identity: str
) -> Iterator[tuple[int, _EntryDecision]]:
    """Stream one V1 task's decisions, whether the value is lazy or a plain dict."""

    decisions = index[identity]
    if isinstance(decisions, _LazyTaskDecisions):
        yield from decisions.stream()
        return
    for entry_index in sorted(decisions):
        yield entry_index, decisions[entry_index]


# ---- Resolved source ----------------------------------------------------------


@dataclass(frozen=True)
class ProcessV2AdmittedSource:
    """A validated, complete, authority-free admitted editing source.

    ``iter_records`` re-reads the immutable V1 payload through the same pinned
    reader the proof used and yields only records the overlay admitted, after
    proving that the overlay row and the payload row describe the same record.
    """

    plan: Mapping[str, Any]
    artifact_root: Path
    repo_root: Path
    completion: Mapping[str, Any]
    counts: Mapping[str, int]
    rejected_traces_by_code: Mapping[str, int]
    adapter_implementation_sha256: str
    # Schedule-invariant digests. Both are folded over
    # ``(v1_task_identity, entry_index, ...)`` and deliberately exclude the
    # range-task execution address, so resharding the same payload the same way
    # reproduces them exactly while relocating the run.
    decision_semantic_sha256: str = ""
    admitted_evidence_sha256: str = ""
    rejection_ledger: Mapping[str, Any] = field(default_factory=dict)
    # The operational replay address, kept apart from everything above.
    physical_execution_identity: Mapping[str, Any] = field(default_factory=dict)

    # ---- Authority: false by construction, never by value ----
    #
    # These were constructor arguments, and `identity()` ignored what they were
    # set to. `ProcessV2AdmittedSource(..., p50_authorized=True)` therefore built
    # cleanly and published `False`, so the object contradicted its own artifact,
    # and one of the four names was outside the frozen vocabulary. Read-only
    # properties over the frozen vocabulary make the object and the artifact the
    # same statement, and there is nothing left to set. `test_the_authority
    # _surface_is_exactly_the_frozen_vocabulary` pins the set to AUTHORITY_FIELDS
    # so this block cannot drift away from it.

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

    def identity(self) -> dict[str, Any]:
        """Return the deterministic descriptor a consumer must record.

        Sorted at every depth, so embedding it verbatim and re-serializing under
        ``sort_keys=True`` reproduces it byte for byte.
        :func:`validate_process_v2_admitted_source_identity` is the owning
        authority on this shape; a consumer checks it there rather than
        restating which fields it believes are present.
        """

        body: dict[str, Any] = _sorted_deep(
            {
                "schema": ADMITTED_SOURCE_SCHEMA,
                "schema_version": ADMITTED_SOURCE_SCHEMA_VERSION,
                "status": ADMITTED_SOURCE_STATUS,
                **authority_false_block(),
                "adapter_implementation_sha256": self.adapter_implementation_sha256,
                "admitted_evidence_sha256": self.admitted_evidence_sha256,
                "completion_sha256": str(self.completion["completion_sha256"]),
                "decision_semantic_sha256": self.decision_semantic_sha256,
                "pinned_builder_identity_sha256": str(
                    self.plan["pinned_builder_identity"]["identity_sha256"]
                ),
                "pinned_process_identity_sha256": str(
                    self.plan["pinned_process_identity"]["process_identity_sha256"]
                ),
                "plan_sha256": str(self.plan["plan_sha256"]),
                "process_v2_identity_sha256": str(
                    self.plan["process_v2_identity"]["process_identity_sha256"]
                ),
                "run_identity_sha256": str(self.plan["run_identity_sha256"]),
                "v1_payload_binding_sha256": str(self.plan["v1_payload_binding_sha256"]),
                "physical_execution_identity": dict(self.physical_execution_identity),
                "rejection_ledger": dict(self.rejection_ledger),
                "counts": {name: int(self.counts[name]) for name in _COUNT_FIELDS},
                "rejected_traces_by_code": dict(self.rejected_traces_by_code),
            }
        )
        return _sorted_deep(
            {**body, ADMITTED_SOURCE_SELF_HASH_FIELD: _canonical_sha256(body)}
        )

    def iter_records(self) -> Iterator[ProcessV2AdmittedRecord]:
        """Yield every admitted record, refusing on any overlay disagreement."""

        index = _overlay_index(self.plan, artifact_root=self.artifact_root)
        pinned_process_identity = self.plan["pinned_process_identity"]
        pinned_builder_identity = self.plan["pinned_builder_identity"]
        admitted = 0
        for v1_task in self.plan["v1_payload_binding"]["v1_tasks"]:
            identity = str(v1_task["v1_task_identity_sha256"])
            entries = int(v1_task["v1_entries"])
            semantic_dir = (
                mounted_process_v2_artifact_path(
                    v1_task["v1_task_artifact_path"],
                    artifact_root=self.artifact_root,
                    field="v1_tasks[].v1_task_artifact_path",
                )
                / SEMANTIC_ARTIFACT_DIRNAME
            )
            try:
                rows = read_semantic_packed_artifact_range_rows(
                    semantic_dir,
                    expected_shard_sha256=v1_task["v1_semantic_shard_sha256"],
                    expected_manifest_sha256=v1_task["v1_semantic_manifest_sha256"],
                    expected_source_binding=v1_task["v1_source_binding"],
                    expected_process_identity=pinned_process_identity,
                    expected_builder_identity=pinned_builder_identity,
                    entry_start=0,
                    entry_stop=entries,
                    sentinel_replay_entries=0,
                    recover_row_errors=False,
                )
                # Both sides are ascending in entry index, so they are walked in
                # lockstep. Random access into the overlay would rescan a range
                # index per row, which is how an O(N) join becomes O(N * ranges).
                stream = _decisions_for(index, identity)
                pending: tuple[int, _EntryDecision] | None = next(stream, None)
                for row in rows:
                    while pending is not None and pending[0] < row.entry_index:
                        pending = next(stream, None)
                    if pending is None or pending[0] != row.entry_index:
                        raise ProcessV2AdmittedSourceIncomplete(
                            "the V2 admission overlay has no decision for "
                            f"{identity}[{row.entry_index}]"
                        )
                    decision = pending[1]
                    pending = next(stream, None)
                    if not decision.admitted:
                        continue
                    yield _admitted_record(
                        v1_task=v1_task,
                        decision=decision,
                        row=row,
                        plan=self.plan,
                    )
                    admitted += 1
            except SemanticPackedStoreError as error:
                raise ProcessV2AdmittedSourceError(
                    f"the immutable V1 payload of task {identity} is unreadable under its "
                    "pinned identities"
                ) from error
        if admitted != int(self.counts["admitted_entries"]):
            raise ProcessV2AdmittedSourceError(
                "the admitted source yielded a different census than it resolved: "
                f"{admitted} != {self.counts['admitted_entries']}"
            )


def _admitted_record(
    *,
    v1_task: Mapping[str, Any],
    decision: _EntryDecision,
    row: Any,
    plan: Mapping[str, Any],
) -> ProcessV2AdmittedRecord:
    """Bind one V1 payload row to its overlay after proving they agree."""

    proof = decision.proof
    record = row.record
    if proof is None or record is None or row.addressed is None:
        raise ProcessV2AdmittedSourceError(
            "an admitted V2 overlay row has no readable V1 payload record"
        )
    disagreements = [
        field
        for field, overlay_value, payload_value in (
            ("trace_id", proof["trace_id"], record.get("trace_id")),
            ("data_lane", proof["data_lane"], record.get("data_lane")),
            ("split", proof["split"], record.get("split")),
            ("path_length", proof["path_length"], record.get("path_length")),
            ("v1_record_sha256", proof["v1_record_sha256"], record.get("record_sha256")),
            (
                "canonical_state_keys",
                proof["canonical_state_keys"],
                record.get("canonical_state_keys"),
            ),
            ("source_address", proof["source_address"], record.get("source_address")),
            ("lineage", proof["lineage"], record.get("lineage")),
            ("family_histogram", proof["family_histogram"], record.get("family_histogram")),
        )
        if overlay_value != payload_value
    ]
    if disagreements:
        raise ProcessV2AdmittedSourceError(
            "the V2 admission overlay disagrees with its V1 payload record on "
            f"{sorted(disagreements)} at entry {row.entry_index}"
        )
    if (
        record.get("data_lane") != v1_task["data_lane"]
        or record.get("split") != v1_task["split"]
    ):
        raise ProcessV2AdmittedSourceError(
            f"V1 payload record {row.entry_index} leaves the lane or split it was bound to"
        )
    return ProcessV2AdmittedRecord(
        v1_identity=ProcessV2V1PayloadIdentity(
            data_lane=str(record["data_lane"]),
            split=str(record["split"]),
            trace_id=str(record["trace_id"]),
            entry_index=int(row.entry_index),
            path_length=int(record["path_length"]),
            record_sha256=str(record["record_sha256"]),
            process_semantics=str(record["process_semantics"]),
            process_identity_sha256=str(record["process_identity_sha256"]),
            process_contract_sha256=str(record["process_contract_sha256"]),
            trace_schema=str(record["schema"]),
            trace_schema_version=int(record["schema_version"]),
            v1_task_identity_sha256=str(v1_task["v1_task_identity_sha256"]),
            v1_task_artifact_path=str(v1_task["v1_task_artifact_path"]),
        ),
        v2_admission=ProcessV2AdmissionOverlay(
            admitted=True,
            run_identity_sha256=str(plan["run_identity_sha256"]),
            task_identity_sha256=decision.task_identity_sha256,
            process_v2_identity_sha256=str(
                plan["process_v2_identity"]["process_identity_sha256"]
            ),
            proof_sha256=str(proof["proof_sha256"]),
            replayed_transitions=int(proof["replayed_transitions"]),
            atom_delete_teachers_by_candidate_source=dict(
                proof["atom_delete_teachers_by_candidate_source"]
            ),
            process_v2_atom_delete_candidates=[
                list(slots) for slots in proof["process_v2_atom_delete_candidates"]
            ],
        ),
        v1_record=record,
        addressed=row.addressed,
    )


def resolve_process_v2_admitted_source(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
    rejection_ledger_artifact_path: str | None = None,
) -> ProcessV2AdmittedSource:
    """Resolve the admitted source, or refuse and say exactly why.

    ``rejection_ledger_artifact_path`` publishes the deterministic compressed
    global rejection ledger during the same single pass.  Omitting it still
    computes the ledger's physical and semantic hashes, so a caller can verify a
    previously published ledger without rewriting it.

    The plan is revalidated against the *live* serialized implementation and the
    *live* Process-V2 identity, so a stale overlay cannot resolve.  The
    published plan bytes, the run completion, every planned task result and the
    complete entry coverage of every V1 task are all required.
    """

    try:
        validated = validate_process_v2_rebind_plan(plan, repo_root=repo_root)
    except ProcessV2RebindError as error:
        raise ProcessV2AdmittedSourceError(
            "the Process-V2 rebind plan is stale or invalid, so no admitted source "
            "can be resolved from it"
        ) from error
    run_root = mounted_process_v2_artifact_path(
        validated["run_artifact_root"],
        artifact_root=artifact_root,
        field="plan.run_artifact_root",
    )
    published_plan = run_root / PLAN_FILENAME
    if (
        not published_plan.is_file()
        or published_plan.read_bytes() != _canonical_json_bytes(validated) + b"\n"
    ):
        raise ProcessV2AdmittedSourceIncomplete(
            "the admitted source requires the exact published Process-V2 rebind plan bytes"
        )
    completion_path = run_root / COMPLETION_FILENAME
    if not completion_path.is_file():
        raise ProcessV2AdmittedSourceIncomplete(
            f"the Process-V2 rebind completion is absent: {completion_path}"
        )
    completion = json.loads(completion_path.read_bytes())
    if not isinstance(completion, dict):
        raise ProcessV2AdmittedSourceError("the Process-V2 rebind completion must be an object")
    live_v2_identity = editing_process_v2_identity()
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("schema_version") != COMPLETION_SCHEMA_VERSION
        or completion.get("status") != COMPLETION_STATUS
        or completion.get("training_authorized") is not False
        or completion.get("completion_sha256") != _self_hash(completion, field="completion_sha256")
    ):
        raise ProcessV2AdmittedSourceError("the Process-V2 rebind completion is malformed")
    if (
        completion.get("run_identity_sha256") != validated["run_identity_sha256"]
        or completion.get("plan_sha256") != validated["plan_sha256"]
        or completion.get("task_inventory_sha256") != validated["task_inventory_sha256"]
        or completion.get("v1_payload_binding_sha256") != validated["v1_payload_binding_sha256"]
        or completion.get("process_v2_identity_sha256")
        != live_v2_identity["process_identity_sha256"]
    ):
        raise ProcessV2AdmittedSourceError(
            "the Process-V2 rebind completion does not bind this plan and live identity"
        )

    expected_tasks = {task["task_identity_sha256"] for task in validated["tasks"]}
    try:
        complete = completed_process_v2_rebind_task_ids(
            validated,
            artifact_root=artifact_root,
            repo_root=repo_root,
        )
    except ProcessV2RebindError as error:
        raise ProcessV2AdmittedSourceError(
            "the V2 admission overlay task namespace is invalid"
        ) from error
    missing = expected_tasks - complete
    if missing:
        raise ProcessV2AdmittedSourceIncomplete(
            f"the V2 admission overlay is missing {len(missing)} planned task results"
        )

    index = _overlay_index(validated, artifact_root=artifact_root)
    join = _stream_admission_join(
        validated,
        index=index,
        ledger_destination=rejection_ledger_artifact_path,
        artifact_root=artifact_root,
    )
    counts = join["counts"]
    rejected_by_code = join["rejected_traces_by_code"]
    if completion.get("counts") != counts:
        raise ProcessV2AdmittedSourceError(
            "the resolved admitted census disagrees with the published completion"
        )
    if completion.get("rejected_traces_by_code") != rejected_by_code:
        raise ProcessV2AdmittedSourceError(
            "the resolved rejection census disagrees with the published completion"
        )
    return ProcessV2AdmittedSource(
        plan=validated,
        artifact_root=Path(artifact_root),
        repo_root=Path(repo_root),
        completion=completion,
        counts=counts,
        rejected_traces_by_code=rejected_by_code,
        adapter_implementation_sha256=_adapter_implementation_sha256(repo_root),
        decision_semantic_sha256=join["decision_semantic_sha256"],
        admitted_evidence_sha256=join["admitted_evidence_sha256"],
        rejection_ledger=join["rejection_ledger"],
        physical_execution_identity=_physical_execution_identity(validated),
    )


# ---- The streaming join --------------------------------------------------------


def _physical_execution_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The operational replay address, held apart from every semantic digest.

    Task and run identities are separate fields, and the range-task inventory is
    named here and only here, so a reshard moves this object without moving the
    evidence digests beside it.
    """

    ranges = [
        {
            "task_identity_sha256": str(task["task_identity_sha256"]),
            "v1_task_identity_sha256": str(task["v1_task_identity_sha256"]),
            "entry_start": int(task["entry_start"]),
            "entry_stop": int(task["entry_stop"]),
        }
        for task in plan["tasks"]
    ]
    body = {
        "schema": PHYSICAL_EXECUTION_IDENTITY_SCHEMA,
        "schema_version": PHYSICAL_EXECUTION_IDENTITY_SCHEMA_VERSION,
        "run_identity_sha256": str(plan["run_identity_sha256"]),
        "plan_sha256": str(plan["plan_sha256"]),
        "entries_per_task": int(plan["entries_per_task"]),
        "range_task_count": len(ranges),
        "range_task_inventory_sha256": _canonical_sha256(ranges),
    }
    return {**body, PHYSICAL_EXECUTION_SELF_HASH_FIELD: _canonical_sha256(body)}


def _rejection_ledger_row(
    *, v1_task_identity_sha256: str, entry_index: int, rejection: Mapping[str, Any]
) -> dict[str, Any]:
    """One complete, reason-coded rejected record. Not a hash of one."""

    return {
        "schema": REJECTION_LEDGER_SCHEMA,
        "schema_version": REJECTION_LEDGER_SCHEMA_VERSION,
        "v1_task_identity_sha256": v1_task_identity_sha256,
        "entry_index": entry_index,
        **{key: item for key, item in rejection.items() if key not in {"schema", "schema_version"}},
    }


def _stream_admission_join(
    plan: Mapping[str, Any],
    *,
    index: MutableMapping,
    ledger_destination: str | None,
    artifact_root: Path,
) -> dict[str, Any]:
    """Join overlay decisions to the payload census one V1 task at a time.

    No global proof dictionary exists at any point.  The census, the two
    schedule-invariant digests and the rejection ledger are all folded in a
    single ordered pass, so peak memory is one range task's decisions rather
    than every proof row of the corpus.
    """

    counts: dict[str, int] = {name: 0 for name in _COUNT_FIELDS}
    rejected_by_code: dict[str, int] = {}
    allowed_codes = {code.value for code in ProcessV2RebindExclusionCode}
    decision_digest = hashlib.sha256()
    evidence_digest = hashlib.sha256()
    rejection_stream_digest = hashlib.sha256()
    rejection_semantic_digest = hashlib.sha256()

    ledger_path: Path | None = None
    staging: str | None = None
    handle = None
    if ledger_destination is not None:
        ledger_path = mounted_process_v2_artifact_path(
            ledger_destination,
            artifact_root=artifact_root,
            field="rejection_ledger_artifact_path",
        )
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        raw = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=ledger_path.parent,
            prefix=f".{ledger_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        staging = raw.name
        handle = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)

    try:
        # V1 tasks in sorted identity order, entries ascending inside each: the
        # stable `(v1_task_identity, entry_index)` order the ledger declares.
        for v1_task in sorted(
            plan["v1_payload_binding"]["v1_tasks"],
            key=lambda task: str(task["v1_task_identity_sha256"]),
        ):
            identity = str(v1_task["v1_task_identity_sha256"])
            entries = int(v1_task["v1_entries"])
            counts["source_entries"] += entries
            # Coverage is checked *during* the stream, not by materialising the
            # index first: a separate `sorted(decisions)` pass would read every
            # range result twice, and reading a range result is not cheap.
            observed = 0
            contiguous = True
            for entry_index, decision in _decisions_for(index, identity):
                contiguous = contiguous and entry_index == observed
                observed += 1
                if decision.admitted and decision.proof is not None:
                    proof = decision.proof
                    counts["admitted_entries"] += 1
                    counts["admitted_states"] += len(proof["canonical_state_keys"])
                    counts["admitted_transitions"] += int(proof["replayed_transitions"])
                    decision_digest.update(
                        _canonical_json_bytes([identity, entry_index, True, proof["proof_sha256"]])
                    )
                    evidence_digest.update(
                        _canonical_json_bytes(
                            [
                                identity,
                                entry_index,
                                proof["v1_record_sha256"],
                                proof["proof_sha256"],
                            ]
                        )
                    )
                    continue
                rejection = decision.rejection
                if rejection is None:
                    raise ProcessV2AdmittedSourceError(
                        "a V2 overlay decision is neither an admission nor a rejection"
                    )
                code = str(rejection["exclusion_code"])
                if code not in allowed_codes:
                    raise ProcessV2AdmittedSourceError(
                        f"the V2 admission overlay names an unknown exclusion code {code!r}"
                    )
                counts["rejected_entries"] += 1
                rejected_by_code[code] = rejected_by_code.get(code, 0) + 1
                decision_digest.update(
                    _canonical_json_bytes([identity, entry_index, False, code])
                )
                rejection_semantic_digest.update(
                    _canonical_json_bytes([identity, entry_index, code])
                )
                row = _canonical_json_bytes(
                    _rejection_ledger_row(
                        v1_task_identity_sha256=identity,
                        entry_index=entry_index,
                        rejection=rejection,
                    )
                ) + b"\n"
                rejection_stream_digest.update(row)
                if handle is not None:
                    handle.write(row)
            if observed != entries or not contiguous:
                raise ProcessV2AdmittedSourceIncomplete(
                    f"the V2 admission overlay decides {observed} of {entries} records "
                    f"for V1 task {identity}"
                )
        if handle is not None:
            handle.close()
            handle = None
        ledger_file_sha256: str | None = None
        if ledger_path is not None and staging is not None:
            os.replace(staging, ledger_path)
            staging = None
            ledger_file_sha256 = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    finally:
        if handle is not None:
            handle.close()
        if staging is not None:
            Path(staging).unlink(missing_ok=True)

    ledger_body: dict[str, Any] = {
        "schema": REJECTION_LEDGER_SCHEMA,
        "schema_version": REJECTION_LEDGER_SCHEMA_VERSION,
        "sort_order": REJECTION_LEDGER_SORT_ORDER,
        "rejected_entries": counts["rejected_entries"],
        "rejected_traces_by_code": dict(sorted(rejected_by_code.items())),
        # Physical: the exact bytes of the ordered ledger content.
        "rejection_stream_sha256": rejection_stream_digest.hexdigest(),
        # Semantic: the reason-coded decision set, free of any execution address.
        "rejection_semantic_sha256": rejection_semantic_digest.hexdigest(),
        "ledger_artifact_path": ledger_destination,
        "ledger_file_sha256": ledger_file_sha256,
    }
    return {
        "counts": counts,
        "rejected_traces_by_code": dict(sorted(rejected_by_code.items())),
        "decision_semantic_sha256": decision_digest.hexdigest(),
        "admitted_evidence_sha256": evidence_digest.hexdigest(),
        "rejection_ledger": {
            **ledger_body,
            REJECTION_LEDGER_SELF_HASH_FIELD: _canonical_sha256(ledger_body),
        },
    }


# ---- The owning identity validator ----------------------------------------------


def _identity_fail(detail: str) -> None:
    raise ProcessV2AdmittedSourceIdentityError(detail)


def _exact_str(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if type(value) is not str:
        _identity_fail(
            f"{label}.{key} is {type(value).__name__}, not str; an admitted-source "
            "identity is type-checked rather than coerced, because a coerced field is "
            "not the field the producer emitted"
        )
    assert isinstance(value, str)  # narrowed by the guard above
    return value


def _exact_sha256(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = _exact_str(payload, key, label=label)
    if _SHA256_RE.fullmatch(value) is None:
        _identity_fail(
            f"{label}.{key} is {value!r}, not exactly 64 lowercase hex characters; an "
            "uppercase variant of the same digest is a different string and hashes to a "
            "different descriptor"
        )
    return value


def _exact_count(payload: Mapping[str, Any], key: str, *, label: str) -> int:
    value = payload.get(key)
    # `bool` is excluded explicitly: it is an `int` subclass, so `True` would
    # otherwise be accepted as the count 1.
    if type(value) is not int:
        _identity_fail(
            f"{label}.{key} is {type(value).__name__}, not an exact int; a coerced "
            "count is not a count"
        )
    assert isinstance(value, int)  # narrowed by the guard above
    if value < 0:
        _identity_fail(
            f"{label}.{key} is {value}; a negative count can reconcile an inflated "
            "total and hide lost support"
        )
    return value


def _require_exact_fields(
    payload: Mapping[str, Any], expected: tuple[str, ...], *, label: str
) -> None:
    """Exact field set AND sorted key order, both required.

    Order is checked because the descriptor is emitted sorted at every depth, and
    that is what lets a consumer embed it verbatim and re-serialize it without
    changing it.  A reordered mapping still self-hashes -- the canonical hash
    sorts -- so nothing else would catch it.
    """

    observed = tuple(payload)
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        _identity_fail(
            f"{label} field set differs from the declared shape; missing={missing} "
            f"unexpected={unexpected}"
        )
    if observed != expected:
        _identity_fail(
            f"{label} keys are not in sorted order; a Process-V2 identity descriptor is "
            "sorted at every depth so that embedding it verbatim is byte-stable"
        )


def _require_sorted_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _identity_fail(f"{label} is {type(value).__name__}, not an object")
    assert isinstance(value, Mapping)  # narrowed by the guard above
    for key in value:
        if type(key) is not str:
            _identity_fail(f"{label} has a non-string key {key!r}")
    if list(value) != sorted(value):
        _identity_fail(f"{label} is not in sorted key order")
    return value


def _validate_physical_execution_identity(
    value: object, *, top_level: Mapping[str, Any]
) -> None:
    """Validate the nested replay address through its OWN owning schema."""

    label = "physical_execution_identity"
    block = _require_sorted_mapping(value, label=label)
    _require_exact_fields(block, PHYSICAL_EXECUTION_IDENTITY_FIELDS, label=label)
    if _exact_str(block, "schema", label=label) != PHYSICAL_EXECUTION_IDENTITY_SCHEMA:
        _identity_fail(
            f"{label}.schema is {block['schema']!r}, not "
            f"{PHYSICAL_EXECUTION_IDENTITY_SCHEMA!r}"
        )
    version = block.get("schema_version")
    if type(version) is not int or version != PHYSICAL_EXECUTION_IDENTITY_SCHEMA_VERSION:
        _identity_fail(
            f"{label}.schema_version is {version!r}, not "
            f"{PHYSICAL_EXECUTION_IDENTITY_SCHEMA_VERSION}"
        )
    for key in ("physical_execution_sha256", "plan_sha256", "range_task_inventory_sha256"):
        _exact_sha256(block, key, label=label)
    for key in ("entries_per_task", "range_task_count"):
        _exact_count(block, key, label=label)
    _exact_sha256(block, "run_identity_sha256", label=label)
    # The nested run and plan must be the SAME run and plan the descriptor names.
    # Without this the replay address could point at a different execution of a
    # different payload and every other field would still check out.
    for key in ("run_identity_sha256", "plan_sha256"):
        if block[key] != top_level[key]:
            _identity_fail(
                f"{label}.{key} is {block[key]!r}, but the admitted source names "
                f"{top_level[key]!r}; the replay address must address this run"
            )
    try:
        verify_self_hash(block, field=PHYSICAL_EXECUTION_SELF_HASH_FIELD, label=label)
    except ProcessV2SchemaError as error:
        raise ProcessV2AdmittedSourceIdentityError(str(error)) from error


def _validate_rejection_ledger(value: object, *, top_level: Mapping[str, Any]) -> None:
    """Validate the nested rejection ledger through its OWN owning schema."""

    label = "rejection_ledger"
    block = _require_sorted_mapping(value, label=label)
    _require_exact_fields(block, REJECTION_LEDGER_FIELDS, label=label)
    if _exact_str(block, "schema", label=label) != REJECTION_LEDGER_SCHEMA:
        _identity_fail(f"{label}.schema is {block['schema']!r}, not {REJECTION_LEDGER_SCHEMA!r}")
    version = block.get("schema_version")
    if type(version) is not int or version != REJECTION_LEDGER_SCHEMA_VERSION:
        _identity_fail(
            f"{label}.schema_version is {version!r}, not {REJECTION_LEDGER_SCHEMA_VERSION}"
        )
    if _exact_str(block, "sort_order", label=label) != REJECTION_LEDGER_SORT_ORDER:
        _identity_fail(
            f"{label}.sort_order is {block['sort_order']!r}, not "
            f"{REJECTION_LEDGER_SORT_ORDER!r}"
        )
    for key in (
        "rejection_ledger_sha256",
        "rejection_semantic_sha256",
        "rejection_stream_sha256",
    ):
        _exact_sha256(block, key, label=label)
    for key in ("ledger_artifact_path", "ledger_file_sha256"):
        if block[key] is not None and type(block[key]) is not str:
            _identity_fail(
                f"{label}.{key} is {type(block[key]).__name__}, not str or null"
            )
    if block["ledger_file_sha256"] is not None:
        _exact_sha256(block, "ledger_file_sha256", label=label)
    _exact_count(block, "rejected_entries", label=label)
    by_code = _validate_rejection_census(
        block["rejected_traces_by_code"],
        rejected_entries=int(block["rejected_entries"]),
        label=f"{label}.rejected_traces_by_code",
    )
    # The ledger and the descriptor are two publications of one decision set.
    if int(block["rejected_entries"]) != int(top_level["counts"]["rejected_entries"]):
        _identity_fail(
            f"{label}.rejected_entries is {block['rejected_entries']}, but the census "
            f"records {top_level['counts']['rejected_entries']}"
        )
    if by_code != dict(top_level["rejected_traces_by_code"]):
        _identity_fail(
            f"{label}.rejected_traces_by_code disagrees with the descriptor's own "
            "reason-coded census"
        )
    try:
        verify_self_hash(block, field=REJECTION_LEDGER_SELF_HASH_FIELD, label=label)
    except ProcessV2SchemaError as error:
        raise ProcessV2AdmittedSourceIdentityError(str(error)) from error


def _validate_rejection_census(
    value: object, *, rejected_entries: int, label: str
) -> dict[str, int]:
    census = _require_sorted_mapping(value, label=label)
    resolved = {str(code): _exact_count(census, str(code), label=label) for code in census}
    allowed = {code.value for code in ProcessV2RebindExclusionCode}
    unknown = sorted(set(resolved) - allowed)
    if unknown:
        _identity_fail(
            f"{label} names exclusion codes {unknown} that the rebind does not define"
        )
    total = sum(resolved.values())
    if total != rejected_entries:
        _identity_fail(
            f"{label} sums to {total}, which disagrees with rejected_entries="
            f"{rejected_entries}"
        )
    return resolved


def _validate_counts(value: object) -> dict[str, int]:
    label = "counts"
    census = _require_sorted_mapping(value, label=label)
    _require_exact_fields(census, tuple(sorted(_COUNT_FIELDS)), label=label)
    resolved = {name: _exact_count(census, name, label=label) for name in _COUNT_FIELDS}
    if resolved["admitted_entries"] + resolved["rejected_entries"] != (
        resolved["source_entries"]
    ):
        _identity_fail(
            f"{label} does not reconcile: {resolved['admitted_entries']} admitted + "
            f"{resolved['rejected_entries']} rejected != {resolved['source_entries']} source"
        )
    # An admitted entry is proved only when every one of its persisted steps
    # replays, so its state count is exactly its transition count plus its source
    # state. Summed, that is the identity below, and it is the one check that
    # relates the two evidence counts to the census rather than carrying them
    # beside it: a lost proof row moves all three together, an inflated count
    # moves only one.
    if resolved["admitted_transitions"] + resolved["admitted_entries"] != (
        resolved["admitted_states"]
    ):
        _identity_fail(
            f"{label} evidence does not reconcile: {resolved['admitted_transitions']} "
            f"transitions + {resolved['admitted_entries']} admitted entries != "
            f"{resolved['admitted_states']} states"
        )
    return resolved


def validate_process_v2_admitted_source_identity(
    value: object, *, repo_root: Path
) -> dict[str, Any]:
    """Validate one admitted-source identity descriptor. The owning authority.

    Every consumer of this descriptor calls this rather than restating the shape.
    The consumer that restated it required three hashes this adapter has never
    emitted, under any version, and nothing failed because that consumer was
    exercised only against a hand-written fixture that invented them.

    Checked, in this order so that the most dangerous defect is reported first:
    schema and version, with version 2 refused explicitly rather than read as a
    subset of version 3; a granted authority field at any depth under any
    spelling; the frozen authority vocabulary, complete and false; the exact field
    set and sorted key order at every depth; exact types with no coercion and
    lowercase SHA-256 syntax; the live adapter implementation hash; the live
    Process-V2 identity; the census, its reason-coded breakdown, and the
    state/transition identity; the two nested bodies through their own owning
    schemas, including that they name this run and this plan; and the self-hash.

    Raises:
        ProcessV2AdmittedSourceIdentityError: naming the specific defect.
    """

    if not isinstance(value, Mapping):
        _identity_fail(
            f"an admitted-source identity must be an object, got {type(value).__name__}"
        )
    assert isinstance(value, Mapping)  # narrowed by the guard above
    payload: dict[str, Any] = dict(value)
    label = "the admitted-source identity"

    if payload.get("schema") != ADMITTED_SOURCE_SCHEMA:
        _identity_fail(
            f"{label} declares schema {payload.get('schema')!r}, not "
            f"{ADMITTED_SOURCE_SCHEMA!r}"
        )
    version = payload.get("schema_version")
    if type(version) is not int:
        _identity_fail(
            f"{label} declares schema_version {version!r}, which is not an exact int"
        )
    if version != ADMITTED_SOURCE_SCHEMA_VERSION:
        _identity_fail(
            f"{label} declares schema_version {version}, which is incompatible with "
            f"{ADMITTED_SOURCE_SCHEMA_VERSION}. No version is a superset of the one "
            "before it: version 2 spelled the bounded-P50 field under the retired name, "
            "omitted three authority fields entirely, and declared no independent "
            "version on its nested physical-execution body, so a converter would have "
            "to invent the values it lacks rather than translate them"
        )

    # Authority first: a grant is the defect that must never be read past, and the
    # vocabulary-free walk reaches mappings nested inside lists, where a field-set
    # check would have reported only an unexpected key.
    try:
        require_no_granted_authority(payload, label=label)
        require_authority_false(payload, label=label)
    except ProcessV2SchemaError as error:
        raise ProcessV2AdmittedSourceIdentityError(str(error)) from error

    _require_sorted_mapping(payload, label=label)
    _require_exact_fields(payload, ADMITTED_SOURCE_IDENTITY_FIELDS, label=label)

    if _exact_str(payload, "status", label=label) != ADMITTED_SOURCE_STATUS:
        _identity_fail(
            f"{label} declares status {payload['status']!r}, not {ADMITTED_SOURCE_STATUS!r}"
        )
    for key in _IDENTITY_SHA256_FIELDS:
        _exact_sha256(payload, key, label=label)

    live_adapter = _adapter_implementation_sha256(Path(repo_root))
    if payload["adapter_implementation_sha256"] != live_adapter:
        _identity_fail(
            f"{label} pins adapter implementation "
            f"{payload['adapter_implementation_sha256']}, but the live implementation "
            f"hashes to {live_adapter}; the descriptor was produced by a different "
            "adapter than the one reading it"
        )
    live_v2 = str(editing_process_v2_identity()["process_identity_sha256"])
    if payload["process_v2_identity_sha256"] != live_v2:
        _identity_fail(
            f"{label} binds Process-V2 identity {payload['process_v2_identity_sha256']}, "
            f"but the live Process-V2 identity is {live_v2}; evidence produced under a "
            "superseded identity stays invalid rather than being revalidated"
        )

    counts = _validate_counts(payload["counts"])
    _validate_rejection_census(
        payload["rejected_traces_by_code"],
        rejected_entries=counts["rejected_entries"],
        label="rejected_traces_by_code",
    )
    _validate_physical_execution_identity(
        payload["physical_execution_identity"], top_level=payload
    )
    _validate_rejection_ledger(payload["rejection_ledger"], top_level=payload)

    try:
        verify_self_hash(payload, field=ADMITTED_SOURCE_SELF_HASH_FIELD, label=label)
    except ProcessV2SchemaError as error:
        raise ProcessV2AdmittedSourceIdentityError(str(error)) from error
    return payload


def write_admitted_source_refusal_report(
    *,
    diagnostic_root: Path,
    plan: Mapping[str, Any] | None,
    stage: str,
    error: BaseException,
    detail: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a refusal outside the scientific run namespace.

    An integrity refusal publishes nothing under the run root.  The report is
    content addressed in its own diagnostic namespace so an operator can read
    why without a partial artifact acquiring authority by proximity.
    """

    body: dict[str, Any] = {
        "schema": REFUSAL_SCHEMA,
        "schema_version": REFUSAL_SCHEMA_VERSION,
        "status": REFUSAL_STATUS,
        # The complete frozen vocabulary, not the four fields this used to list:
        # an incomplete authority block is refused by `require_authority_false`,
        # so a partial one would make the refusal report itself unreadable by the
        # guard every other artifact in this module passes.
        **authority_false_block(),
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "run_identity_sha256": (str(plan["run_identity_sha256"]) if plan is not None else None),
        "plan_sha256": (str(plan["plan_sha256"]) if plan is not None else None),
        "detail": dict(detail or {}),
    }
    report = {**body, "refusal_sha256": _canonical_sha256(body)}
    root = Path(diagnostic_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{report['refusal_sha256']}.json"
    if target.exists():
        return target
    staging = target.with_name(f".{target.name}.tmp")
    staging.write_bytes(_canonical_json_bytes(report) + b"\n")
    os.replace(staging, target)
    return target


__all__ = [
    "ADMISSION_OVERLAY_SCHEMA",
    "ADMISSION_OVERLAY_SCHEMA_VERSION",
    "ADMITTED_SOURCE_IDENTITY_FIELDS",
    "ADMITTED_SOURCE_SCHEMA",
    "ADMITTED_SOURCE_SCHEMA_VERSION",
    "ADMITTED_SOURCE_SELF_HASH_FIELD",
    "ADMITTED_SOURCE_STATUS",
    "DEFAULT_REJECTION_LEDGER_FILENAME",
    "PHYSICAL_EXECUTION_IDENTITY_FIELDS",
    "PHYSICAL_EXECUTION_IDENTITY_SCHEMA",
    "PHYSICAL_EXECUTION_IDENTITY_SCHEMA_VERSION",
    "PHYSICAL_EXECUTION_SELF_HASH_FIELD",
    "REFUSAL_SCHEMA",
    "REFUSAL_SCHEMA_VERSION",
    "REFUSAL_STATUS",
    "REJECTION_LEDGER_FIELDS",
    "REJECTION_LEDGER_SELF_HASH_FIELD",
    "REJECTION_LEDGER_SORT_ORDER",
    "V1_PAYLOAD_IDENTITY_SCHEMA",
    "V1_PAYLOAD_IDENTITY_SCHEMA_VERSION",
    "ProcessV2AdmissionOverlay",
    "ProcessV2AdmittedRecord",
    "ProcessV2AdmittedSource",
    "ProcessV2AdmittedSourceError",
    "ProcessV2AdmittedSourceIdentityError",
    "ProcessV2AdmittedSourceIncomplete",
    "ProcessV2V1PayloadIdentity",
    "resolve_process_v2_admitted_source",
    "validate_process_v2_admitted_source_identity",
    "write_admitted_source_refusal_report",
]
