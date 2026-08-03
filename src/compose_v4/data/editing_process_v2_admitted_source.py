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
``training_authorized`` is ``False``; Gate 0, T1 and P50 authority are not
conferred, not implied, and not obtainable from this object.  A later builder
must record :meth:`ProcessV2AdmittedSource.identity` as its input provenance and
obtain its own authorization.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
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
    REJECTION_LEDGER_SCHEMA,
    REJECTION_LEDGER_SCHEMA_VERSION,
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
# Version 2 adds the streaming join's outputs: `decision_semantic_sha256`,
# `admitted_evidence_sha256`, `rejection_ledger` and `physical_execution_identity`.
# Version 1 carried none of them and could not distinguish a reshard from a
# different corpus, so the versions are incompatible rather than additive; a
# consumer must import this constant instead of assuming a shape.
ADMITTED_SOURCE_SCHEMA_VERSION = 2
ADMITTED_SOURCE_STATUS = "V2_ADMISSION_OVERLAY_RESOLVED_NO_TRAINING_AUTHORITY"
V1_PAYLOAD_IDENTITY_SCHEMA = "compose.data.editing_process_v2_v1_payload_identity"
V1_PAYLOAD_IDENTITY_SCHEMA_VERSION = 1
ADMISSION_OVERLAY_SCHEMA = "compose.data.editing_process_v2_admission_overlay"
ADMISSION_OVERLAY_SCHEMA_VERSION = 1
REFUSAL_SCHEMA = "compose.data.editing_process_v2_admitted_source_refusal"
REFUSAL_SCHEMA_VERSION = 1
REFUSAL_STATUS = "INTEGRITY_REFUSAL_NOTHING_PUBLISHED_UNDER_THE_RUN_NAMESPACE"
DEFAULT_REJECTION_LEDGER_FILENAME = "PROCESS_V2_REJECTION_LEDGER.jsonl.gz"

_ADAPTER_SOURCE_FILE = "src/compose_v4/data/editing_process_v2_admitted_source.py"

_COUNT_FIELDS = (
    "source_entries",
    "admitted_entries",
    "rejected_entries",
    "admitted_states",
    "admitted_transitions",
)


class ProcessV2AdmittedSourceError(RuntimeError):
    """The V2 admission overlay cannot be joined to the V1 payload."""


class ProcessV2AdmittedSourceIncomplete(ProcessV2AdmittedSourceError):
    """The overlay is absent or does not decide every V1 record."""


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
    training_authorized: bool = False
    gate_zero_authorized: bool = False
    t1_authorized: bool = False
    p50_authorized: bool = False

    def identity(self) -> dict[str, Any]:
        """Return the deterministic descriptor a consumer must record."""

        body: dict[str, Any] = {
            "schema": ADMITTED_SOURCE_SCHEMA,
            "schema_version": ADMITTED_SOURCE_SCHEMA_VERSION,
            "status": ADMITTED_SOURCE_STATUS,
            "training_authorized": False,
            "gate_zero_authorized": False,
            "t1_authorized": False,
            "p50_authorized": False,
            "decision_semantic_sha256": self.decision_semantic_sha256,
            "admitted_evidence_sha256": self.admitted_evidence_sha256,
            "rejection_ledger": dict(self.rejection_ledger),
            "physical_execution_identity": dict(self.physical_execution_identity),
            "run_identity_sha256": str(self.plan["run_identity_sha256"]),
            "plan_sha256": str(self.plan["plan_sha256"]),
            "completion_sha256": str(self.completion["completion_sha256"]),
            "v1_payload_binding_sha256": str(self.plan["v1_payload_binding_sha256"]),
            "pinned_process_identity_sha256": str(
                self.plan["pinned_process_identity"]["process_identity_sha256"]
            ),
            "pinned_builder_identity_sha256": str(
                self.plan["pinned_builder_identity"]["identity_sha256"]
            ),
            "process_v2_identity_sha256": str(
                self.plan["process_v2_identity"]["process_identity_sha256"]
            ),
            "adapter_implementation_sha256": self.adapter_implementation_sha256,
            "counts": {field: int(self.counts[field]) for field in _COUNT_FIELDS},
            "rejected_traces_by_code": dict(sorted(self.rejected_traces_by_code.items())),
        }
        return {**body, "admitted_source_sha256": _canonical_sha256(body)}

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
        "schema": f"{ADMITTED_SOURCE_SCHEMA}.physical_execution_identity",
        "schema_version": ADMITTED_SOURCE_SCHEMA_VERSION,
        "run_identity_sha256": str(plan["run_identity_sha256"]),
        "plan_sha256": str(plan["plan_sha256"]),
        "entries_per_task": int(plan["entries_per_task"]),
        "range_task_count": len(ranges),
        "range_task_inventory_sha256": _canonical_sha256(ranges),
    }
    return {**body, "physical_execution_sha256": _canonical_sha256(body)}


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
        "sort_order": "v1_task_identity_sha256_then_entry_index",
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
            "rejection_ledger_sha256": _canonical_sha256(ledger_body),
        },
    }


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
        "training_authorized": False,
        "gate_zero_authorized": False,
        "t1_authorized": False,
        "bounded_p50_authorized": False,
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
    "ADMITTED_SOURCE_SCHEMA",
    "ADMITTED_SOURCE_SCHEMA_VERSION",
    "ADMITTED_SOURCE_STATUS",
    "DEFAULT_REJECTION_LEDGER_FILENAME",
    "REFUSAL_SCHEMA",
    "REFUSAL_SCHEMA_VERSION",
    "REFUSAL_STATUS",
    "V1_PAYLOAD_IDENTITY_SCHEMA",
    "V1_PAYLOAD_IDENTITY_SCHEMA_VERSION",
    "ProcessV2AdmissionOverlay",
    "ProcessV2AdmittedRecord",
    "ProcessV2AdmittedSource",
    "ProcessV2AdmittedSourceError",
    "ProcessV2AdmittedSourceIncomplete",
    "ProcessV2V1PayloadIdentity",
    "resolve_process_v2_admitted_source",
    "write_admitted_source_refusal_report",
]
