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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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
from compose_v4.data.packed_trace_store import AddressedPackedTrace
from compose_v4.data.semantic_packed_trace_store import (
    SemanticPackedStoreError,
    read_semantic_packed_artifact_range_rows,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    SEMANTIC_ARTIFACT_DIRNAME,
)

ADMITTED_SOURCE_SCHEMA = "compose.data.editing_process_v2_admitted_source"
ADMITTED_SOURCE_SCHEMA_VERSION = 1
ADMITTED_SOURCE_STATUS = "V2_ADMISSION_OVERLAY_RESOLVED_NO_TRAINING_AUTHORITY"
V1_PAYLOAD_IDENTITY_SCHEMA = "compose.data.editing_process_v2_v1_payload_identity"
V1_PAYLOAD_IDENTITY_SCHEMA_VERSION = 1
ADMISSION_OVERLAY_SCHEMA = "compose.data.editing_process_v2_admission_overlay"
ADMISSION_OVERLAY_SCHEMA_VERSION = 1

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


def _overlay_index(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, dict[int, _EntryDecision]]:
    """Index every recorded V2 decision by V1 task identity and entry index."""

    index: dict[str, dict[int, _EntryDecision]] = {
        task["v1_task_identity_sha256"]: {}
        for task in plan["v1_payload_binding"]["v1_tasks"]
    }
    for task in plan["tasks"]:
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
        decisions = index[task["v1_task_identity_sha256"]]
        for proof in _read_proof_rows(output / PROOF_FILENAME):
            entry_index = int(proof["entry_index"])
            if entry_index in decisions:
                raise ProcessV2AdmittedSourceError(
                    "the V2 admission overlay decides one V1 entry twice: "
                    f"{task['v1_task_identity_sha256']}[{entry_index}]"
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
                    "the V2 admission overlay decides one V1 entry twice: "
                    f"{task['v1_task_identity_sha256']}[{entry_index}]"
                )
            decisions[entry_index] = _EntryDecision(
                admitted=False,
                task_identity_sha256=str(task["task_identity_sha256"]),
                proof=None,
                rejection=rejection,
            )
    return index


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
            decisions = index[identity]
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
                for row in rows:
                    decision = decisions.get(row.entry_index)
                    if decision is None:
                        raise ProcessV2AdmittedSourceIncomplete(
                            "the V2 admission overlay has no decision for "
                            f"{identity}[{row.entry_index}]"
                        )
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
) -> ProcessV2AdmittedSource:
    """Resolve the admitted source, or refuse and say exactly why.

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
    counts: dict[str, int] = {field: 0 for field in _COUNT_FIELDS}
    rejected_by_code: dict[str, int] = {}
    allowed_codes = {code.value for code in ProcessV2RebindExclusionCode}
    for v1_task in validated["v1_payload_binding"]["v1_tasks"]:
        identity = str(v1_task["v1_task_identity_sha256"])
        entries = int(v1_task["v1_entries"])
        decisions = index[identity]
        if sorted(decisions) != list(range(entries)):
            raise ProcessV2AdmittedSourceIncomplete(
                f"the V2 admission overlay decides {len(decisions)} of {entries} records "
                f"for V1 task {identity}"
            )
        counts["source_entries"] += entries
        for decision in decisions.values():
            if decision.admitted and decision.proof is not None:
                counts["admitted_entries"] += 1
                counts["admitted_states"] += len(decision.proof["canonical_state_keys"])
                counts["admitted_transitions"] += int(decision.proof["replayed_transitions"])
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
    if completion.get("counts") != counts:
        raise ProcessV2AdmittedSourceError(
            "the resolved admitted census disagrees with the published completion"
        )
    if completion.get("rejected_traces_by_code") != dict(sorted(rejected_by_code.items())):
        raise ProcessV2AdmittedSourceError(
            "the resolved rejection census disagrees with the published completion"
        )
    return ProcessV2AdmittedSource(
        plan=validated,
        artifact_root=Path(artifact_root),
        repo_root=Path(repo_root),
        completion=completion,
        counts=counts,
        rejected_traces_by_code=dict(sorted(rejected_by_code.items())),
        adapter_implementation_sha256=_adapter_implementation_sha256(repo_root),
    )


__all__ = [
    "ADMISSION_OVERLAY_SCHEMA",
    "ADMISSION_OVERLAY_SCHEMA_VERSION",
    "ADMITTED_SOURCE_SCHEMA",
    "ADMITTED_SOURCE_SCHEMA_VERSION",
    "ADMITTED_SOURCE_STATUS",
    "V1_PAYLOAD_IDENTITY_SCHEMA",
    "V1_PAYLOAD_IDENTITY_SCHEMA_VERSION",
    "ProcessV2AdmissionOverlay",
    "ProcessV2AdmittedRecord",
    "ProcessV2AdmittedSource",
    "ProcessV2AdmittedSourceError",
    "ProcessV2AdmittedSourceIncomplete",
    "ProcessV2V1PayloadIdentity",
    "resolve_process_v2_admitted_source",
]
