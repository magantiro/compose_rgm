"""Bind the exact semantic-migration completion, never a directory subset.

The Process-V2 chain consumes one completed V1 semantic migration as immutable
chemical data.  The rebind's own :func:`bind_v1_semantic_payload` discovers that
payload by listing ``<payload_root>/tasks``, which is correct for *validating*
whatever is there and wrong for *deciding what must be there*: nineteen visible
task directories bind cleanly and silently reduce the corpus by one lane/role
cell.  This module supplies the missing authority.  The exact
``SEMANTIC_MIGRATION_COMPLETE.json`` names the twenty tasks, their receipts,
their packed shards and their census; the payload root and the source inventory
are derived **from that completion**, and a payload that disagrees with it in
any direction (missing, extra, duplicated, re-hashed, re-counted) is refused.

Three properties this module exists to hold
-------------------------------------------

**The completion is historical-pinned input, not a live assertion.**  The
migration ran under the superseded V1 process identity
(``6b98ee21…``); today's live V1 identity is ``6c4721f0…``.  Every existing
loader for this artifact
(``editing_v2_semantic_active8_source_adapter.resolve_editing_v2_semantic_active8_sources``)
revalidates against the *live* identity and a clean Git worktree, so it refuses
the historical payload by construction.  Binding here is therefore pinned:
the identities the payload actually carries are validated for self-consistency
and required to equal the *declared expected* values, and no currency is
claimed for them.

**The rebind consumes the admitted records, not the migration input.**  The
completion's ``counts`` are ``source``, ``admitted`` and ``rejected`` for the
*migration*; each task's packed shard holds only its admitted rows
(``semantic_packed_trace_store`` requires ``manifest["entries"] ==
decision_binding["admitted_count"]``).  So the corpus the Process-V2 rebind
reads is ``counts["admitted"]``, and
:func:`bind_exact_semantic_migration_completion` asserts exactly that against
the bound payload's entry census.  For the current production completion those
are 646,779 admitted of 695,638 source; a chain that plans 695,638 range tasks
is reading the wrong number.

**Nothing here carries authority.**  A binding proves identity and provenance.
It does not authorize the cache, the rebind, Active8, Gate 0, T1 or P50.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.editing_process_v2_rebind import (
    TASK_DIRNAME,
    ProcessV2RebindError,
    bind_v1_semantic_payload,
    mounted_process_v2_artifact_path,
    validate_pinned_builder_identity,
    validate_pinned_process_identity,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    EXACT_COMPLETION_BINDING_SCHEMA,
    EXACT_COMPLETION_BINDING_SCHEMA_VERSION,
    IdentityRole,
    PointerKind,
    ProcessV2SchemaError,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
    require_census_reconciles,
    self_hashed,
    typed_pointer,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_FILENAME as MIGRATION_COMPLETION_FILENAME,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_SCHEMA as MIGRATION_COMPLETION_SCHEMA,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_SCHEMA_VERSION as MIGRATION_COMPLETION_SCHEMA_VERSION,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_STATUS as MIGRATION_COMPLETION_STATUS,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    EXPECTED_TASK_COUNT,
    PLAN_FILENAME as MIGRATION_PLAN_FILENAME,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
)

# ---- Frozen artifact identity ------------------------------------------------

BINDING_SCHEMA = EXACT_COMPLETION_BINDING_SCHEMA
BINDING_SCHEMA_VERSION = EXACT_COMPLETION_BINDING_SCHEMA_VERSION
BINDING_STATUS = "EXACT_COMPLETION_BOUND_HISTORICAL_INPUT_NO_DOWNSTREAM_AUTHORITY"

# The exact key set ``reduce_semantic_trace_migration`` seals.  Mirrored rather
# than imported because that module publishes no completion validator and no
# field-set constant; ``test_editing_v2_process_v2_completion_binder`` reads the
# sealing site's own source and fails if these drift apart, so the mirror cannot
# rot silently.
COMPLETION_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "training_authorized",
        "gate_zero_run",
        "run_identity_sha256",
        "plan_sha256",
        "plan_file_sha256",
        "source_revision_sha256",
        "source_inventory_sha256",
        "process_identity_sha256",
        "builder_identity_sha256",
        "task_inventory_sha256",
        "task_count",
        "result_inventory",
        "result_inventory_sha256",
        "counts",
        "rejections_by_code",
        "accepted_family_histogram",
        "completion_sha256",
    }
)
RESULT_FIELDS: frozenset[str] = frozenset(
    {
        "task_identity_sha256",
        "data_lane",
        "split",
        "source_artifact_path",
        "output_artifact_path",
        "receipt_sha256",
        "semantic_shard_sha256",
        "semantic_manifest_sha256",
        "counts",
    }
)
COUNT_FIELDS: tuple[str, ...] = ("source", "admitted", "rejected")
RUN_ROOT_INVENTORY: frozenset[str] = frozenset(
    {MIGRATION_PLAN_FILENAME, MIGRATION_COMPLETION_FILENAME, TASK_DIRNAME}
)

# The complete lane/role grid, in the migration's own canonical order.  Read
# from the corpus contract rather than retyped, so a lane or role change is a
# contract change and not a silent twenty-entry literal here.
EXPECTED_LANE_ROLE_GRID: tuple[tuple[str, str], ...] = tuple(
    (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
)


class ProcessV2CompletionBindingError(RuntimeError):
    """The exact semantic-migration completion cannot be bound."""


class ProcessV2CompletionIncomplete(ProcessV2CompletionBindingError):
    """The payload does not present exactly the completion's declared inventory."""


# ---- Declared expectations ----------------------------------------------------


@dataclass(frozen=True)
class ProcessV2CompletionExpectation:
    """Everything a caller must state *before* opening the completion.

    There is deliberately no default and no "expect whatever is there" mode: a
    binder that derives its expectation from the artifact it is checking proves
    only that the artifact agrees with itself.
    """

    completion_artifact_path: str
    run_identity_sha256: str
    completion_sha256: str
    process_identity_sha256: str
    builder_identity_sha256: str
    result_inventory_sha256: str
    task_count: int
    source_traces: int
    admitted_traces: int
    rejected_traces: int

    def counts(self) -> dict[str, int]:
        return {
            "source": int(self.source_traces),
            "admitted": int(self.admitted_traces),
            "rejected": int(self.rejected_traces),
        }


# The current production completion.  Volume-relative identity and container
# mount path are kept separate: the artifact lives at
# ``/editing_v2/semantic_v4_migration/…`` on the ``compose-v4-artifacts``
# volume, and a job that mounts it at ``/artifacts`` reads it one prefix down.
#
# These values are *declared expectations* transcribed from the authoritative
# handoff.  They have not been reopened locally; every one of them is asserted
# fail-closed at bind time, and a later authorized read-only source audit is
# what turns them from declared into measured.
PRODUCTION_COMPLETION_VOLUME_PATH = (
    "/editing_v2/semantic_v4_migration"
    "/da6f82845b5998819d59e10d364a34aa6b7a0e76fb6451dcf315b5b1b7e55cc2"
    "/semantic_mapreduce"
    "/2c4ebb5ea8dc55c942978c1548d99f6a7b34bc8a3b7d6a425c814cc0de27dcc7"
    f"/{MIGRATION_COMPLETION_FILENAME}"
)
PRODUCTION_COMPLETION_ARTIFACT_PATH = f"/artifacts{PRODUCTION_COMPLETION_VOLUME_PATH}"

PRODUCTION_COMPLETION_EXPECTATION = ProcessV2CompletionExpectation(
    completion_artifact_path=PRODUCTION_COMPLETION_ARTIFACT_PATH,
    run_identity_sha256="2c4ebb5ea8dc55c942978c1548d99f6a7b34bc8a3b7d6a425c814cc0de27dcc7",
    completion_sha256="a7698306bf1977a6a4cf3d7003948084729a428dd609f5b44058b9ace160ce24",
    # The payload was built under the superseded V1 process identity. Binding
    # the live one here would refuse the only payload that exists.
    process_identity_sha256=SUPERSEDED_V1_PROCESS_IDENTITY_SHA256,
    builder_identity_sha256="5004b099733c81c3b7954576b4c8233f971a8dacf91d10ad42632ba52f0c3a8d",
    result_inventory_sha256="80d12064ac078cc105f142f7f0c7b9aabb27a7c1bed9d1c18e97a3df25a3c911",
    task_count=EXPECTED_TASK_COUNT,
    source_traces=695_638,
    admitted_traces=646_779,
    rejected_traces=48_859,
)


# ---- The binding --------------------------------------------------------------


@dataclass(frozen=True)
class ProcessV2ExactCompletionBinding:
    """One completion, its payload, and the exact inventory joining them."""

    completion_artifact_path: str
    payload_root_artifact_path: str
    completion: Mapping[str, Any]
    completion_file_sha256: str
    v1_payload_binding: Mapping[str, Any]
    source_inventory: tuple[Mapping[str, Any], ...]
    counts: Mapping[str, int]

    @property
    def rebind_source_entries(self) -> int:
        """The record count the Process-V2 rebind actually reads."""

        return int(self.counts["admitted"])

    def as_payload(self) -> dict[str, Any]:
        """The deterministic, authority-free descriptor a consumer records."""

        body: dict[str, Any] = {
            "schema": BINDING_SCHEMA,
            "schema_version": BINDING_SCHEMA_VERSION,
            "status": BINDING_STATUS,
            **authority_false_block(),
            "completion_pointer": typed_pointer(
                kind=PointerKind.REMOTE_ARTIFACT,
                provider="compose-v4-artifacts",
                target=self.completion_artifact_path,
                target_schema=MIGRATION_COMPLETION_SCHEMA,
                identity_role=IdentityRole.PHYSICAL,
                sha256=self.completion_file_sha256,
            ),
            "completion_semantic_pointer": typed_pointer(
                kind=PointerKind.REMOTE_ARTIFACT,
                provider="compose-v4-artifacts",
                target=self.completion_artifact_path,
                target_schema=MIGRATION_COMPLETION_SCHEMA,
                identity_role=IdentityRole.SEMANTIC,
                hash_algorithm="self_hash_field_v1",
                sha256=str(self.completion["completion_sha256"]),
            ),
            "payload_root_artifact_path": self.payload_root_artifact_path,
            "run_identity_sha256": str(self.completion["run_identity_sha256"]),
            "pinned_process_identity_sha256": str(self.completion["process_identity_sha256"]),
            "pinned_builder_identity_sha256": str(self.completion["builder_identity_sha256"]),
            "result_inventory_sha256": str(self.completion["result_inventory_sha256"]),
            "v1_payload_binding_sha256": str(self.v1_payload_binding["binding_sha256"]),
            "task_count": int(self.completion["task_count"]),
            "source_inventory": [dict(entry) for entry in self.source_inventory],
            "source_inventory_sha256": canonical_sha256(
                [dict(entry) for entry in self.source_inventory]
            ),
            "counts": {
                **{field: int(self.counts[field]) for field in COUNT_FIELDS},
                "rebind_source_entries": self.rebind_source_entries,
            },
        }
        return self_hashed(body, field="binding_sha256")


def validate_exact_completion_binding_payload(value: object) -> dict[str, Any]:
    """Validate a published binding payload without reopening the payload."""

    if not isinstance(value, Mapping):
        raise ProcessV2CompletionBindingError("an exact completion binding must be an object")
    payload = dict(value)
    try:
        require_authority_false(payload, label="the exact completion binding")
        if (
            payload.get("schema") != BINDING_SCHEMA
            or payload.get("schema_version") != BINDING_SCHEMA_VERSION
            or payload.get("status") != BINDING_STATUS
        ):
            raise ProcessV2SchemaError("exact completion binding schema or status disagrees")
        body = {key: item for key, item in payload.items() if key != "binding_sha256"}
        if payload.get("binding_sha256") != canonical_sha256(body):
            raise ProcessV2SchemaError("exact completion binding self-hash disagrees")
    except ProcessV2SchemaError as error:
        raise ProcessV2CompletionBindingError(str(error)) from error
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise ProcessV2CompletionBindingError("exact completion binding counts must be an object")
    if int(counts.get("rebind_source_entries", -1)) != int(counts.get("admitted", -2)):
        raise ProcessV2CompletionBindingError(
            "the rebind source census must equal the admitted census; the migration "
            "input count is not the rebind input count"
        )
    return payload


# ---- Reading and validating the completion ------------------------------------


def _load_canonical_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ProcessV2CompletionIncomplete(f"{label} is absent or unreadable: {path}") from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2CompletionBindingError(f"{label} is not JSON: {path}") from error
    if not isinstance(value, dict):
        raise ProcessV2CompletionBindingError(f"{label} must be an object: {path}")
    if raw != canonical_bytes(value) + b"\n":
        raise ProcessV2CompletionBindingError(f"{label} is not canonically serialized: {path}")
    return value, raw


def _require_count_object(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(COUNT_FIELDS):
        raise ProcessV2CompletionBindingError(f"{label} must carry exactly {sorted(COUNT_FIELDS)}")
    counts: dict[str, int] = {}
    for field in COUNT_FIELDS:
        count = value[field]
        if type(count) is not int or count < 0:
            raise ProcessV2CompletionBindingError(f"{label}.{field} must be a nonnegative integer")
        counts[field] = count
    if counts["admitted"] + counts["rejected"] != counts["source"]:
        raise ProcessV2CompletionBindingError(
            f"{label} does not reconcile: {counts['admitted']} admitted + "
            f"{counts['rejected']} rejected != {counts['source']} source"
        )
    return counts


def validate_semantic_migration_completion(
    value: object,
    *,
    expectation: ProcessV2CompletionExpectation,
) -> dict[str, Any]:
    """Validate the completion document itself against a declared expectation."""

    if not isinstance(value, Mapping):
        raise ProcessV2CompletionBindingError("the semantic migration completion must be an object")
    completion = dict(value)
    if set(completion) != COMPLETION_FIELDS:
        raise ProcessV2CompletionBindingError(
            "semantic migration completion fields disagree; "
            f"missing={sorted(COMPLETION_FIELDS - set(completion))}, "
            f"extras={sorted(set(completion) - COMPLETION_FIELDS)}"
        )
    body = {key: item for key, item in completion.items() if key != "completion_sha256"}
    if (
        completion["schema"] != MIGRATION_COMPLETION_SCHEMA
        or completion["schema_version"] != MIGRATION_COMPLETION_SCHEMA_VERSION
        or completion["status"] != MIGRATION_COMPLETION_STATUS
        or completion["training_authorized"] is not False
        or completion["gate_zero_run"] is not False
        or completion["completion_sha256"] != canonical_sha256(body)
    ):
        raise ProcessV2CompletionBindingError(
            "semantic migration completion schema, authority or self-hash disagrees"
        )

    results = completion["result_inventory"]
    if not isinstance(results, list):
        raise ProcessV2CompletionBindingError("the completion result inventory must be a list")
    if completion["result_inventory_sha256"] != canonical_sha256(results):
        raise ProcessV2CompletionBindingError(
            "the completion result-inventory hash does not address its own inventory"
        )
    if completion["task_count"] != len(results):
        raise ProcessV2CompletionBindingError(
            f"the completion declares {completion['task_count']} tasks and lists {len(results)}"
        )
    if len(results) != int(expectation.task_count):
        raise ProcessV2CompletionIncomplete(
            f"the completion carries {len(results)} of {expectation.task_count} expected "
            "lane/role tasks; a smaller valid corpus is never inferred from what is present"
        )

    counts = _require_count_object(completion["counts"], label="completion counts")
    require_census_reconciles(
        {
            "source_entries": counts["source"],
            "admitted_entries": counts["admitted"],
            "rejected_entries": counts["rejected"],
        },
        label="the semantic migration completion",
    )

    mismatched = [
        field
        for field, observed, expected in (
            ("run_identity_sha256", completion["run_identity_sha256"], expectation.run_identity_sha256),
            ("completion_sha256", completion["completion_sha256"], expectation.completion_sha256),
            (
                "process_identity_sha256",
                completion["process_identity_sha256"],
                expectation.process_identity_sha256,
            ),
            (
                "builder_identity_sha256",
                completion["builder_identity_sha256"],
                expectation.builder_identity_sha256,
            ),
            (
                "result_inventory_sha256",
                completion["result_inventory_sha256"],
                expectation.result_inventory_sha256,
            ),
            ("counts.source", counts["source"], int(expectation.source_traces)),
            ("counts.admitted", counts["admitted"], int(expectation.admitted_traces)),
            ("counts.rejected", counts["rejected"], int(expectation.rejected_traces)),
        )
        if observed != expected
    ]
    if mismatched:
        raise ProcessV2CompletionBindingError(
            f"the semantic migration completion disagrees with its declared expectation on "
            f"{sorted(mismatched)}"
        )
    return completion


def _validate_result_inventory(
    results: Sequence[Mapping[str, Any]],
    *,
    payload_root_artifact_path: str,
) -> tuple[dict[str, Any], ...]:
    """Require the exact lane/role grid, once each, with unique task identities."""

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, result in enumerate(results):
        if not isinstance(result, Mapping) or set(result) != RESULT_FIELDS:
            raise ProcessV2CompletionBindingError(f"completion result {index} fields disagree")
        entry = dict(result)
        identity = entry["task_identity_sha256"]
        if not isinstance(identity, str) or len(identity) != 64:
            raise ProcessV2CompletionBindingError(
                f"completion result {index} task identity is malformed"
            )
        if identity in seen:
            raise ProcessV2CompletionBindingError(
                f"the completion repeats task identity {identity}"
            )
        seen.add(identity)
        expected_output = f"{payload_root_artifact_path}/{TASK_DIRNAME}/{identity}"
        if entry["output_artifact_path"] != expected_output:
            raise ProcessV2CompletionBindingError(
                f"completion result {index} output path is not under the derived payload root"
            )
        _require_count_object(entry["counts"], label=f"completion result {index} counts")
        ordered.append(entry)

    observed_grid = tuple((str(entry["data_lane"]), str(entry["split"])) for entry in ordered)
    if sorted(observed_grid) != sorted(EXPECTED_LANE_ROLE_GRID):
        missing = sorted(set(EXPECTED_LANE_ROLE_GRID) - set(observed_grid))
        extra = sorted(set(observed_grid) - set(EXPECTED_LANE_ROLE_GRID))
        raise ProcessV2CompletionIncomplete(
            f"the completion lane/role grid disagrees; missing={missing}, extras={extra}"
        )
    if len(set(observed_grid)) != len(observed_grid):
        raise ProcessV2CompletionBindingError("the completion repeats a lane/role cell")
    return tuple(ordered)


# ---- The public entry point ---------------------------------------------------


def bind_exact_semantic_migration_completion(
    *,
    completion_artifact_path: str,
    artifact_root: Path,
    pinned_process_identity: Mapping[str, Any],
    pinned_builder_identity: Mapping[str, Any],
    expectation: ProcessV2CompletionExpectation,
) -> ProcessV2ExactCompletionBinding:
    """Bind the exact completion and the payload it declares, or refuse.

    The payload root is *derived* from the completion path, the task inventory
    is *derived* from the completion's result inventory, and the payload is
    required to present exactly that inventory.  Nothing is inferred from the
    set of directories that happen to be visible.
    """

    if completion_artifact_path != expectation.completion_artifact_path:
        raise ProcessV2CompletionBindingError(
            "the completion path does not equal the declared expected path"
        )
    posix = PurePosixPath(completion_artifact_path)
    if posix.name != MIGRATION_COMPLETION_FILENAME:
        raise ProcessV2CompletionBindingError(
            f"the exact completion must be named {MIGRATION_COMPLETION_FILENAME}"
        )
    payload_root_artifact_path = str(posix.parent)

    try:
        completion_path = mounted_process_v2_artifact_path(
            completion_artifact_path,
            artifact_root=artifact_root,
            field="completion_artifact_path",
        )
        run_root = mounted_process_v2_artifact_path(
            payload_root_artifact_path,
            artifact_root=artifact_root,
            field="payload_root_artifact_path",
        )
    except ProcessV2RebindError as error:
        raise ProcessV2CompletionBindingError(str(error)) from error

    if not run_root.is_dir():
        raise ProcessV2CompletionIncomplete(f"the migration run root is absent: {run_root}")
    observed_root = {entry.name for entry in run_root.iterdir()}
    if observed_root != set(RUN_ROOT_INVENTORY):
        raise ProcessV2CompletionIncomplete(
            "the migration run root inventory disagrees; "
            f"missing={sorted(set(RUN_ROOT_INVENTORY) - observed_root)}, "
            f"extras={sorted(observed_root - set(RUN_ROOT_INVENTORY))}"
        )

    raw_completion, raw_bytes = _load_canonical_json_object(
        completion_path, label="the semantic migration completion"
    )
    completion = validate_semantic_migration_completion(raw_completion, expectation=expectation)
    source_inventory = _validate_result_inventory(
        completion["result_inventory"],
        payload_root_artifact_path=payload_root_artifact_path,
    )

    declared_tasks = {str(entry["task_identity_sha256"]) for entry in source_inventory}
    task_root = run_root / TASK_DIRNAME
    if not task_root.is_dir():
        raise ProcessV2CompletionIncomplete(f"the migration task namespace is absent: {task_root}")
    observed_tasks = {entry.name for entry in task_root.iterdir()}
    if observed_tasks != declared_tasks:
        raise ProcessV2CompletionIncomplete(
            "the payload task namespace is not exactly the completion's inventory; "
            f"missing={sorted(declared_tasks - observed_tasks)}, "
            f"extras={sorted(observed_tasks - declared_tasks)}"
        )

    # Every receipt, decision ledger, packed manifest, packed shard and their
    # physical and semantic hashes are proven here, under the *pinned*
    # identities. The rebind's binder is reused rather than reimplemented; this
    # module supplies only the exactness the binder cannot know by itself.
    process_identity = validate_pinned_process_identity(pinned_process_identity)
    builder_identity = validate_pinned_builder_identity(pinned_builder_identity)
    if (
        process_identity["process_identity_sha256"] != completion["process_identity_sha256"]
        or builder_identity["identity_sha256"] != completion["builder_identity_sha256"]
    ):
        raise ProcessV2CompletionBindingError(
            "the pinned identities do not equal the identities the completion declares"
        )
    try:
        binding = bind_v1_semantic_payload(
            payload_root_artifact_path=payload_root_artifact_path,
            artifact_root=artifact_root,
            pinned_process_identity=process_identity,
            pinned_builder_identity=builder_identity,
        )
    except ProcessV2RebindError as error:
        raise ProcessV2CompletionBindingError(
            "the declared payload is invalid under the pinned identities"
        ) from error

    bound = {str(task["v1_task_identity_sha256"]): task for task in binding["v1_tasks"]}
    if set(bound) != declared_tasks:
        raise ProcessV2CompletionIncomplete(
            "the bound payload task set is not the completion's declared task set"
        )
    for entry in source_inventory:
        identity = str(entry["task_identity_sha256"])
        task = bound[identity]
        disagreements = [
            field
            for field, declared, observed in (
                ("data_lane", entry["data_lane"], task["data_lane"]),
                ("split", entry["split"], task["split"]),
                ("receipt_sha256", entry["receipt_sha256"], task["v1_receipt_sha256"]),
                (
                    "semantic_shard_sha256",
                    entry["semantic_shard_sha256"],
                    task["v1_semantic_shard_sha256"],
                ),
                (
                    "semantic_manifest_sha256",
                    entry["semantic_manifest_sha256"],
                    task["v1_semantic_manifest_sha256"],
                ),
                ("entries", int(entry["counts"]["admitted"]), int(task["v1_entries"])),
            )
            if declared != observed
        ]
        if disagreements:
            raise ProcessV2CompletionBindingError(
                f"payload task {identity} disagrees with its completion entry on "
                f"{sorted(disagreements)}"
            )

    counts = _require_count_object(completion["counts"], label="completion counts")
    summed = {
        field: sum(int(entry["counts"][field]) for entry in source_inventory)
        for field in COUNT_FIELDS
    }
    if summed != counts:
        raise ProcessV2CompletionBindingError(
            f"the completion census {counts} is not the sum of its twenty task censuses {summed}"
        )
    if int(binding["v1_entry_count"]) != counts["admitted"]:
        raise ProcessV2CompletionBindingError(
            "the bound payload census is not the admitted census: "
            f"{binding['v1_entry_count']} bound entries != {counts['admitted']} admitted. "
            "The rebind reads the admitted records, never the migration's source count "
            f"of {counts['source']}."
        )

    return ProcessV2ExactCompletionBinding(
        completion_artifact_path=completion_artifact_path,
        payload_root_artifact_path=payload_root_artifact_path,
        completion=completion,
        completion_file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        v1_payload_binding=binding,
        source_inventory=source_inventory,
        counts=counts,
    )


__all__ = [
    "BINDING_SCHEMA",
    "BINDING_SCHEMA_VERSION",
    "BINDING_STATUS",
    "COMPLETION_FIELDS",
    "COUNT_FIELDS",
    "EXPECTED_LANE_ROLE_GRID",
    "PRODUCTION_COMPLETION_ARTIFACT_PATH",
    "PRODUCTION_COMPLETION_EXPECTATION",
    "PRODUCTION_COMPLETION_VOLUME_PATH",
    "RESULT_FIELDS",
    "RUN_ROOT_INVENTORY",
    "ProcessV2CompletionBindingError",
    "ProcessV2CompletionExpectation",
    "ProcessV2CompletionIncomplete",
    "ProcessV2ExactCompletionBinding",
    "bind_exact_semantic_migration_completion",
    "validate_exact_completion_binding_payload",
    "validate_semantic_migration_completion",
]
