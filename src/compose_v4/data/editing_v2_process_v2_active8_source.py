"""The join between the Process-V2 rebind and Active8 materialization.

Active8 consumes a lane/role inventory of exact semantic packed artifacts.  The
Process-V2 rebind does not rewrite those artifacts: it publishes an **overlay**
of per-entry admission decisions over the same immutable V1 payload.  So the
Process-V2 Active8 source is a join, not a new corpus, and this module is the one
place that performs it.

Two identities are carried side by side and never merged:

* the **V1 payload identity** the chemistry was built under, which is superseded
  and stays superseded; and
* the **live Process-V2 identity** the admission decision was made under.

A consumer that wants one of them must say which.  Collapsing them into a single
``process_identity_sha256`` is exactly how a V1 artifact would come to be read as
a V2 one, so the inventory exposes both under distinct names and refuses when the
V1 inventory and the admitted source disagree about which payload they describe.

What this module deliberately does NOT do:

* it does not re-derive admission.  ``resolve_process_v2_admitted_source`` is the
  authority, and this module calls it;
* it does not re-read the packed shards.  ``resolve_editing_v2_semantic_active8_sources``
  is the authority for the V1 lane/role inventory, and this module calls it;
* it authorizes nothing.  Every authority flag it publishes is false, and a
  complete Process-V2 source is not permission to materialize Active8, to run
  Gate 0, or to train.

Schema version 2: the interface, not the decision
-------------------------------------------------

Version 2 changes what this module PUBLISHES and how a consumer checks it.  It
changes nothing about how the join is decided: the same two authorities are
called, the same cross-checks refuse the same inputs, and no admission or census
is re-derived here.

* the embedded admitted-source descriptor is the version-3 one, validated
  through its owning validator rather than trusted;
* all seven frozen authority fields are published, from
  :func:`authority_false_block`.  Version 1 published four, so the three it
  omitted could not be read as false by a consumer that required them;
* every mapping is sorted at every depth, matching the descriptor it embeds, so
  the identity survives canonical re-serialization byte for byte;
* :func:`validate_process_v2_active8_source_identity` is the owning validator.
  Version 1 had none, so the join's guarantees held at resolution time and
  evaporated the moment the identity was written to a file: a consumer reading
  that file back had nothing to check it against.

What the validator adds over the resolver's own checks is the joint statement
the two halves make TOGETHER.  The resolver proves each half internally
consistent; the identity claims that the top-level Process-V2 identity, V1
payload identity, census and rejection census are the SAME values the embedded
descriptor carries.  Nothing checked that, and a descriptor swapped for another
run's would satisfy every per-half check while the identity above it described a
different corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from compose_v4.data.editing_process_v2_admitted_source import (
    ProcessV2AdmittedSource,
    ProcessV2AdmittedSourceIdentityError,
    resolve_process_v2_admitted_source,
    validate_process_v2_admitted_source_identity,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    ProcessV2SchemaError,
    authority_false_block,
    require_authority_false,
    require_no_granted_authority,
    verify_self_hash,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    EditingV2SemanticActive8SourceInventory,
    resolve_editing_v2_semantic_active8_sources,
)

# ---- Frozen identity ----

SOURCE_SCHEMA = "compose.data.editing_v2_process_v2_active8_source"
# Version 2 embeds the admitted-source version-3 descriptor, publishes the
# complete seven-field authority vocabulary instead of four, sorts at every
# depth, and is checkable by an owning validator. Version 1 had none of these, so
# the versions are incompatible rather than additive.
SOURCE_SCHEMA_VERSION = 2
SOURCE_STATUS = "VERIFIED_PROCESS_V2_ACTIVE8_SOURCE_NO_DOWNSTREAM_AUTHORITY"
SOURCE_SELF_HASH_FIELD = "process_v2_active8_source_sha256"

_COUNT_FIELDS = (
    "source_entries",
    "admitted_entries",
    "rejected_entries",
    "admitted_states",
    "admitted_transitions",
)

#: The exact field set of one per-lane/role source row.
ACTIVE8_SOURCE_ROW_FIELDS: tuple[str, ...] = (
    "admitted_entry_count",
    "data_lane",
    "partition_role",
    "rejected_entry_count",
    "semantic_manifest_sha256",
    "semantic_shard_sha256",
    "task_identity_sha256",
    "v1_entry_count",
)

#: The exact field set of the inventory identity, sorted.
ACTIVE8_SOURCE_IDENTITY_FIELDS: tuple[str, ...] = tuple(
    sorted(
        (
            "schema",
            "schema_version",
            "status",
            *AUTHORITY_FIELDS,
            "admitted_source_identity",
            "counts",
            "process_v2_identity_sha256",
            "rejected_traces_by_code",
            "sources",
            "v1_migration_completion_sha256",
            "v1_payload_process_identity_sha256",
            SOURCE_SELF_HASH_FIELD,
        )
    )
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ProcessV2Active8SourceError(RuntimeError):
    """The Process-V2 Active8 source cannot be resolved from these inputs."""


class ProcessV2Active8SourceIdentityError(ProcessV2Active8SourceError):
    """A published Process-V2 Active8 source identity is malformed or inconsistent."""


# ---- Deterministic hashing ----


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sorted_deep(value: Any) -> Any:
    """Every mapping in sorted key order, at every depth. List order preserved."""

    if isinstance(value, Mapping):
        return {key: _sorted_deep(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_sorted_deep(item) for item in value]
    return value


# ---- The resolved source ----


@dataclass(frozen=True)
class ProcessV2Active8Source:
    """One lane/role artifact with its Process-V2 admitted census."""

    data_lane: str
    partition_role: str
    task_identity_sha256: str
    semantic_shard_sha256: str
    semantic_manifest_sha256: str
    v1_entry_count: int
    admitted_entry_count: int
    rejected_entry_count: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "data_lane": self.data_lane,
            "partition_role": self.partition_role,
            "task_identity_sha256": self.task_identity_sha256,
            "semantic_shard_sha256": self.semantic_shard_sha256,
            "semantic_manifest_sha256": self.semantic_manifest_sha256,
            "v1_entry_count": self.v1_entry_count,
            "admitted_entry_count": self.admitted_entry_count,
            "rejected_entry_count": self.rejected_entry_count,
        }


@dataclass(frozen=True)
class ProcessV2Active8SourceInventory:
    """The joined inventory. It grants nothing."""

    v1_inventory: EditingV2SemanticActive8SourceInventory
    admitted_source: ProcessV2AdmittedSource
    sources: tuple[ProcessV2Active8Source, ...]
    counts: Mapping[str, int]
    rejected_traces_by_code: Mapping[str, int]

    # Named so neither identity can be mistaken for the other.
    @property
    def v1_payload_process_identity_sha256(self) -> str:
        return str(
            self.admitted_source.plan["pinned_process_identity"]["process_identity_sha256"]
        )

    @property
    def process_v2_identity_sha256(self) -> str:
        return str(
            self.admitted_source.plan["process_v2_identity"]["process_identity_sha256"]
        )

    def identity(self) -> dict[str, Any]:
        """The descriptor a downstream materializer must record verbatim.

        Sorted at every depth, matching the admitted-source descriptor it embeds,
        so the whole object survives canonical re-serialization unchanged.
        :func:`validate_process_v2_active8_source_identity` is the owning
        authority on this shape.
        """

        body: dict[str, Any] = _sorted_deep(
            {
                "schema": SOURCE_SCHEMA,
                "schema_version": SOURCE_SCHEMA_VERSION,
                "status": SOURCE_STATUS,
                **authority_false_block(),
                "v1_migration_completion_sha256": (
                    self.v1_inventory.migration_completion_sha256
                ),
                "v1_payload_process_identity_sha256": (
                    self.v1_payload_process_identity_sha256
                ),
                "process_v2_identity_sha256": self.process_v2_identity_sha256,
                "admitted_source_identity": self.admitted_source.identity(),
                "counts": {field: int(self.counts[field]) for field in _COUNT_FIELDS},
                "rejected_traces_by_code": dict(self.rejected_traces_by_code),
                "sources": [source.as_payload() for source in self.sources],
            }
        )
        return _sorted_deep({**body, SOURCE_SELF_HASH_FIELD: _canonical_sha256(body)})


# ---- Resolution ----


def resolve_process_v2_active8_source_inventory(
    plan: Mapping[str, Any],
    *,
    v1_migration_completion_path: Path,
    artifact_root: Path,
    repo_root: Path,
) -> ProcessV2Active8SourceInventory:
    """Join the V1 lane/role inventory to the Process-V2 admission overlay.

    Both halves are resolved by their own authority and then cross-checked. The
    check that matters is that they describe the SAME payload: an admitted
    overlay proved against one migration silently paired with a different
    migration's shards would produce a corpus whose chemistry and whose
    admission decisions came from different runs.
    """

    admitted = resolve_process_v2_admitted_source(
        plan, artifact_root=Path(artifact_root), repo_root=Path(repo_root)
    )
    v1_inventory = resolve_editing_v2_semantic_active8_sources(
        Path(v1_migration_completion_path),
        artifact_root=Path(artifact_root),
        repo_root=Path(repo_root),
    )

    # The overlay names its V1 tasks; the inventory names the same tasks. If the
    # two task sets differ, they are not the same payload.
    binding = plan["v1_payload_binding"]
    overlay_tasks = {
        str(task["v1_task_identity_sha256"]): task for task in binding["v1_tasks"]
    }
    inventory_tasks = {
        str(source.task_identity_sha256): source for source in v1_inventory.sources
    }
    if set(overlay_tasks) != set(inventory_tasks):
        raise ProcessV2Active8SourceError(
            "the Process-V2 overlay and the V1 migration inventory name different "
            f"task sets: overlay-only={sorted(set(overlay_tasks) - set(inventory_tasks))} "
            f"inventory-only={sorted(set(inventory_tasks) - set(overlay_tasks))}"
        )
    for task_id, overlay_task in overlay_tasks.items():
        source = inventory_tasks[task_id]
        if str(overlay_task["v1_semantic_shard_sha256"]) != source.semantic_shard_sha256:
            raise ProcessV2Active8SourceError(
                f"task {task_id} names a different semantic shard in the overlay than "
                "in the V1 migration inventory"
            )
        if str(overlay_task["data_lane"]) != source.data_lane or str(
            overlay_task["split"]
        ) != source.partition_role:
            raise ProcessV2Active8SourceError(
                f"task {task_id} lane or split disagrees between the overlay and the "
                "V1 migration inventory"
            )

    # Per-task admitted census, taken from the published task results rather
    # than recomputed, so this module never becomes a second census authority.
    admitted_by_task: Counter[str] = Counter()
    rejected_by_task: Counter[str] = Counter()
    for result in admitted.completion["result_inventory"]:
        v1_task = str(result["v1_task_identity_sha256"])
        admitted_by_task[v1_task] += int(result["counts"]["admitted_entries"])
        rejected_by_task[v1_task] += int(result["counts"]["rejected_entries"])

    sources = tuple(
        ProcessV2Active8Source(
            data_lane=source.data_lane,
            partition_role=source.partition_role,
            task_identity_sha256=source.task_identity_sha256,
            semantic_shard_sha256=source.semantic_shard_sha256,
            semantic_manifest_sha256=source.semantic_manifest_sha256,
            v1_entry_count=int(overlay_tasks[source.task_identity_sha256]["v1_entries"]),
            admitted_entry_count=int(admitted_by_task[source.task_identity_sha256]),
            rejected_entry_count=int(rejected_by_task[source.task_identity_sha256]),
        )
        for source in sorted(
            v1_inventory.sources, key=lambda item: item.task_identity_sha256
        )
    )

    # Source == admitted + rejected, per task and in total. The library already
    # enforces this run-wide; asserting it again here is what makes the JOIN
    # trustworthy rather than merely the reduction.
    for source in sources:
        if source.admitted_entry_count + source.rejected_entry_count != (
            source.v1_entry_count
        ):
            raise ProcessV2Active8SourceError(
                f"task {source.task_identity_sha256} admits "
                f"{source.admitted_entry_count} and rejects "
                f"{source.rejected_entry_count} of {source.v1_entry_count} entries"
            )
    counts = {field: int(admitted.counts[field]) for field in _COUNT_FIELDS}
    if sum(source.v1_entry_count for source in sources) != counts["source_entries"]:
        raise ProcessV2Active8SourceError(
            "the joined per-task census does not account for every source entry"
        )

    return ProcessV2Active8SourceInventory(
        v1_inventory=v1_inventory,
        admitted_source=admitted,
        sources=sources,
        counts=counts,
        rejected_traces_by_code=dict(admitted.rejected_traces_by_code),
    )


# ---- The owning identity validator ----


def _identity_fail(detail: str) -> None:
    raise ProcessV2Active8SourceIdentityError(detail)


def _sorted_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _identity_fail(f"{label} is {type(value).__name__}, not an object")
    assert isinstance(value, Mapping)  # narrowed by the guard above
    if any(type(key) is not str for key in value):
        _identity_fail(f"{label} has a non-string key")
    if list(value) != sorted(value):
        _identity_fail(f"{label} is not in sorted key order")
    return value


def _exact_fields(
    payload: Mapping[str, Any], expected: tuple[str, ...], *, label: str
) -> None:
    observed = tuple(payload)
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        _identity_fail(
            f"{label} field set differs from the declared shape; missing={missing} "
            f"unexpected={unexpected}"
        )
    if observed != expected:
        _identity_fail(f"{label} keys are not in sorted order")


def _exact_sha256(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _identity_fail(
            f"{label}.{key} is {value!r}, not exactly 64 lowercase hex characters"
        )
    assert isinstance(value, str)  # narrowed by the guard above
    return value


def _exact_count(payload: Mapping[str, Any], key: str, *, label: str) -> int:
    value = payload.get(key)
    # `bool` is an `int` subclass, so it is excluded explicitly.
    if type(value) is not int:
        _identity_fail(
            f"{label}.{key} is {type(value).__name__}, not an exact int; a coerced "
            "count is not a count"
        )
    assert isinstance(value, int)  # narrowed by the guard above
    if value < 0:
        _identity_fail(f"{label}.{key} is negative ({value})")
    return value


def validate_process_v2_active8_source_identity(
    value: object, *, repo_root: Path
) -> dict[str, Any]:
    """Validate one published Process-V2 Active8 source identity. The owning authority.

    The resolver proves each half of the join internally consistent, but it does
    so at resolution time; once the identity is a file, a consumer reading it back
    has only this.  What is checked here that nothing checked before is the JOINT
    statement the two halves make: that the Process-V2 identity, the V1 payload
    identity, the census and the reason-coded rejection census at the top level
    are the SAME values the embedded descriptor carries.  A descriptor swapped for
    another run's satisfies every per-half check while the identity above it
    describes a different corpus.

    Raises:
        ProcessV2Active8SourceIdentityError: naming the specific defect.
    """

    label = "the Process-V2 Active8 source identity"
    if not isinstance(value, Mapping):
        _identity_fail(f"{label} must be an object, got {type(value).__name__}")
    assert isinstance(value, Mapping)  # narrowed by the guard above
    payload: dict[str, Any] = dict(value)

    if payload.get("schema") != SOURCE_SCHEMA:
        _identity_fail(f"{label} declares schema {payload.get('schema')!r}, not {SOURCE_SCHEMA!r}")
    version = payload.get("schema_version")
    if type(version) is not int:
        _identity_fail(f"{label} declares schema_version {version!r}, not an exact int")
    if version != SOURCE_SCHEMA_VERSION:
        _identity_fail(
            f"{label} declares schema_version {version}, which is incompatible with "
            f"{SOURCE_SCHEMA_VERSION}. Version 1 published four authority fields rather "
            "than seven, embedded an admitted-source descriptor of a schema that has "
            "since been refused, and declared no sort order, so it cannot be read as a "
            "subset of this one"
        )

    # Authority first, and at every depth: the embedded descriptor and the source
    # rows are both nested, so a top-level scan would be checking the one place a
    # grant is least likely to be written.
    try:
        require_no_granted_authority(payload, label=label)
        require_authority_false(payload, label=label)
    except ProcessV2SchemaError as error:
        raise ProcessV2Active8SourceIdentityError(str(error)) from error

    _sorted_mapping(payload, label=label)
    _exact_fields(payload, ACTIVE8_SOURCE_IDENTITY_FIELDS, label=label)
    if payload.get("status") != SOURCE_STATUS:
        _identity_fail(f"{label} declares status {payload.get('status')!r}, not {SOURCE_STATUS!r}")
    for key in (
        "process_v2_identity_sha256",
        "v1_migration_completion_sha256",
        "v1_payload_process_identity_sha256",
        SOURCE_SELF_HASH_FIELD,
    ):
        _exact_sha256(payload, key, label=label)

    # The embedded descriptor is checked by ITS owner, never by a second opinion.
    try:
        source = validate_process_v2_admitted_source_identity(
            payload["admitted_source_identity"], repo_root=Path(repo_root)
        )
    except ProcessV2AdmittedSourceIdentityError as error:
        raise ProcessV2Active8SourceIdentityError(
            f"{label} embeds an admitted-source identity that is not valid: {error}"
        ) from error

    # ---- The joint statement ----
    if payload["process_v2_identity_sha256"] != source["process_v2_identity_sha256"]:
        _identity_fail(
            f"{label} names Process-V2 identity {payload['process_v2_identity_sha256']}, "
            f"but its embedded descriptor was resolved under "
            f"{source['process_v2_identity_sha256']}"
        )
    if payload["v1_payload_process_identity_sha256"] != (
        source["pinned_process_identity_sha256"]
    ):
        _identity_fail(
            f"{label} names V1 payload identity "
            f"{payload['v1_payload_process_identity_sha256']}, but its embedded "
            f"descriptor pins {source['pinned_process_identity_sha256']}"
        )
    if payload["v1_payload_process_identity_sha256"] == (
        payload["process_v2_identity_sha256"]
    ):
        _identity_fail(
            f"{label} carries one value under both identity names; collapsing them is "
            "exactly how a V1 artifact comes to be read as a V2 one"
        )

    counts = _sorted_mapping(payload["counts"], label="counts")
    _exact_fields(counts, tuple(sorted(_COUNT_FIELDS)), label="counts")
    resolved = {name: _exact_count(counts, name, label="counts") for name in _COUNT_FIELDS}
    if dict(counts) != dict(source["counts"]):
        _identity_fail(
            f"{label} publishes a census its embedded descriptor does not carry; the "
            "join reports the source's census rather than a second one"
        )
    rejected = _sorted_mapping(
        payload["rejected_traces_by_code"], label="rejected_traces_by_code"
    )
    for code in rejected:
        _exact_count(rejected, code, label="rejected_traces_by_code")
    if dict(rejected) != dict(source["rejected_traces_by_code"]):
        _identity_fail(
            f"{label} publishes a rejection census its embedded descriptor does not carry"
        )

    # ---- Per-source rows reconcile, individually and in total ----
    rows = payload["sources"]
    if not isinstance(rows, list):
        _identity_fail("sources is not a list")
    totals = {"v1_entry_count": 0, "admitted_entry_count": 0, "rejected_entry_count": 0}
    seen: set[str] = set()
    previous = ""
    for index, row in enumerate(rows):
        where = f"sources[{index}]"
        entry = _sorted_mapping(row, label=where)
        _exact_fields(entry, ACTIVE8_SOURCE_ROW_FIELDS, label=where)
        for key in (
            "semantic_manifest_sha256",
            "semantic_shard_sha256",
            "task_identity_sha256",
        ):
            _exact_sha256(entry, key, label=where)
        for key in ("data_lane", "partition_role"):
            if type(entry[key]) is not str or not entry[key]:
                _identity_fail(f"{where}.{key} is not a non-empty string")
        task = str(entry["task_identity_sha256"])
        if task in seen:
            _identity_fail(f"{where} repeats task {task}; one task is one row")
        seen.add(task)
        if task < previous:
            _identity_fail(
                f"{where} is out of order; rows are sorted by task identity so the "
                "inventory is byte-stable across runs"
            )
        previous = task
        row_counts = {
            key: _exact_count(entry, key, label=where)
            for key in ("admitted_entry_count", "rejected_entry_count", "v1_entry_count")
        }
        if row_counts["admitted_entry_count"] + row_counts["rejected_entry_count"] != (
            row_counts["v1_entry_count"]
        ):
            _identity_fail(
                f"{where} admits {row_counts['admitted_entry_count']} and rejects "
                f"{row_counts['rejected_entry_count']} of {row_counts['v1_entry_count']} "
                "entries"
            )
        for key, total in totals.items():
            totals[key] = total + row_counts[key]
    for row_key, census_key in (
        ("v1_entry_count", "source_entries"),
        ("admitted_entry_count", "admitted_entries"),
        ("rejected_entry_count", "rejected_entries"),
    ):
        if totals[row_key] != resolved[census_key]:
            _identity_fail(
                f"the per-source {row_key} totals {totals[row_key]}, which disagrees "
                f"with {census_key}={resolved[census_key]}; a run-wide census can "
                "reconcile while the rows it is supposed to summarise do not"
            )

    try:
        verify_self_hash(payload, field=SOURCE_SELF_HASH_FIELD, label=label)
    except ProcessV2SchemaError as error:
        raise ProcessV2Active8SourceIdentityError(str(error)) from error
    return payload


__all__ = [
    "ACTIVE8_SOURCE_IDENTITY_FIELDS",
    "ACTIVE8_SOURCE_ROW_FIELDS",
    "SOURCE_SCHEMA",
    "SOURCE_SCHEMA_VERSION",
    "SOURCE_SELF_HASH_FIELD",
    "SOURCE_STATUS",
    "ProcessV2Active8Source",
    "ProcessV2Active8SourceError",
    "ProcessV2Active8SourceIdentityError",
    "ProcessV2Active8SourceInventory",
    "resolve_process_v2_active8_source_inventory",
    "validate_process_v2_active8_source_identity",
]
