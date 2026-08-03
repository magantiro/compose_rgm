"""The cache-fed Process-V2 Active8 source: a streaming join, not a corpus.

Active8 consumes a lane/role inventory of exact semantic rows.  The Process-V2
rebind does not rewrite those rows: it publishes an **overlay** of per-entry
admission decisions over the same immutable V1 payload, and the chunk cache
publishes those rows verbatim in a committed, content-addressed generation.  So
the Process-V2 Active8 source is a join over two artifacts that already exist,
and this module is the one place that performs it.

Three inputs, joined by one key
-------------------------------

* the **committed chunk-cache generation** the rebind plan binds, which holds
  every source row byte for byte;
* the **committed rebind task decisions** -- one proof row or one reason-coded
  rejection row per entry;
* the **admitted-source identity**, which is the descriptor the decision run
  published about itself.

The key is :data:`JOIN_KEY_FIELDS` -- the V1 task identity plus the GLOBAL entry
index -- and never chunk position.  Chunk position is an artifact of the chunk
size, which is a partitioning choice: two caches of the same corpus at different
chunk sizes hold the same rows at different offsets, so a positional join would
pair a decision with the wrong trace the moment the chunk size moved.

No V1 payload read, on any path
-------------------------------

The V1 packed shards are bound once, by hash, at *plan* time.  After that the
corpus is the cache: this module opens the committed cache generation the plan
names, streams its verified chunks, and never opens ``traces.jsonl.gz``.  There
is no raw-shard fallback -- an oracle-geometry plan carries no cache binding and
:func:`require_production_source_geometry` refuses it, so the fallback is absent
by construction rather than by discipline.  The consequence that matters
operationally: deleting the V1 payload after the cache is published does not
break this path.

That is also why the historical V1 loader is gone.  It revalidates the *live*
V1 process identity, so it refuses the exact historical payload by construction;
routing the pinned payload through it would have failed only against the real
artifact, at the point where a remote stage was already running.

Two identities, deliberately never merged
-----------------------------------------

* the **V1 payload identity** the chemistry was built under, which is superseded
  and stays superseded, carried as *payload* provenance; and
* the **live Process-V2 identity** the admission decision was made under,
  carried as *decision* provenance.

A consumer that wants one of them must say which.  Collapsing them into a single
``process_identity_sha256`` is exactly how a V1 artifact would come to be read as
a V2 one, so both the inventory and its published identity expose them under
distinct names and the validator refuses one value carried under both.

Two rejection categories, also never merged
-------------------------------------------

An entry the rebind refused is :data:`UPSTREAM_REJECTED`.  It is yielded, so the
census can account for it, and it carries no decoded chemistry at all: the
stream hands out ``addressed``/``v1_record`` only for admitted entries, so an
upstream-rejected trace cannot be candidate-evaluated by a consumer that forgot
to check the category.  Active8's own exclusions are a different category and
belong to the Active8 stage, not here.

Schema version 3: the source of the inventory moved
---------------------------------------------------

Version 2 built its lane/role rows from the live V1 migration loader and
published a ``task_identity_sha256`` that was that loader's task identity.
Version 3 builds the same rows from the committed cache generation, names the
key ``v1_task_identity_sha256`` after the frozen join key, and publishes the
three cache-generation digests that say which committed generation the stream
and the identity both reopened.  Neither version is a subset of the other, so a
consumer must import :data:`SOURCE_SCHEMA_VERSION` rather than assume a shape.

What this module still does NOT do: it does not re-derive admission
(``resolve_process_v2_admitted_source`` is the authority, and this module calls
it), it does not re-decide anything, and it authorizes nothing.  Every authority
flag it publishes is false, and a complete Process-V2 source is not permission to
materialize Active8, to run Gate 0, or to train.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from compose_v4.data.editing_process_v2_admitted_source import (
    ProcessV2AdmissionOverlay,
    ProcessV2AdmittedSource,
    ProcessV2AdmittedSourceIdentityError,
    ProcessV2V1PayloadIdentity,
    resolve_process_v2_admitted_source,
    validate_process_v2_admitted_source_identity,
)
from compose_v4.data.editing_process_v2_admitted_source import (
    _decisions_for as _recorded_decisions_for,
)
from compose_v4.data.editing_process_v2_admitted_source import (
    _overlay_index as _recorded_decision_index,
)
from compose_v4.data.editing_process_v2_rebind import (
    ProcessV2RebindError,
    require_production_source_geometry,
    validate_process_v2_rebind_cache_binding,
)
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    JOIN_KEY_FIELDS,
    UPSTREAM_REJECTED,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    ProcessV2ChunkCacheError,
    ProcessV2ChunkCacheGeneration,
    chunk_targets_for_source,
    load_committed_process_v2_chunk_cache_completion,
    mount_process_v2_artifact_path,
    open_process_v2_chunk_cache,
    read_process_v2_chunk_target,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    ProcessV2SchemaError,
    authority_false_block,
    require_authority_false,
    require_no_granted_authority,
    verify_self_hash,
)
from compose_v4.data.packed_trace_store import AddressedPackedTrace

# ---- Frozen identity ----

SOURCE_SCHEMA = "compose.data.editing_v2_process_v2_active8_source"
# Version 3 takes the lane/role inventory from the committed chunk-cache
# generation instead of the live V1 migration loader, keys each row by
# `v1_task_identity_sha256` (the frozen join key) instead of a loader-local task
# identity, and publishes the three cache-generation digests. Version 2 carried
# none of these, so the versions are incompatible rather than additive.
SOURCE_SCHEMA_VERSION = 3
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
    "cache_source_task_identity_sha256",
    "data_lane",
    "partition_role",
    "rejected_entry_count",
    "semantic_manifest_sha256",
    "semantic_shard_sha256",
    "v1_entry_count",
    "v1_task_identity_sha256",
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
            "cache_completion_sha256",
            "cache_physical_identity_sha256",
            "cache_semantic_identity_sha256",
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
    """One lane/role cached source with its Process-V2 admitted census."""

    data_lane: str
    partition_role: str
    v1_task_identity_sha256: str
    cache_source_task_identity_sha256: str
    cache_source_artifact_path: str
    semantic_shard_sha256: str
    semantic_manifest_sha256: str
    v1_entry_count: int
    admitted_entry_count: int
    rejected_entry_count: int

    def as_payload(self) -> dict[str, Any]:
        """The published row. The cache source PATH is deliberately absent.

        A path is where a generation happens to be mounted; the generation's own
        digests, published once at the top level, are what identify it. Naming
        the mount point per row would put twenty copies of an address into an
        identity that is supposed to be relocation-invariant.
        """

        return {
            "data_lane": self.data_lane,
            "partition_role": self.partition_role,
            "v1_task_identity_sha256": self.v1_task_identity_sha256,
            "cache_source_task_identity_sha256": self.cache_source_task_identity_sha256,
            "semantic_shard_sha256": self.semantic_shard_sha256,
            "semantic_manifest_sha256": self.semantic_manifest_sha256,
            "v1_entry_count": self.v1_entry_count,
            "admitted_entry_count": self.admitted_entry_count,
            "rejected_entry_count": self.rejected_entry_count,
        }


@dataclass(frozen=True)
class ProcessV2SourceEntry:
    """One V1 entry joined to the decision the rebind recorded for it.

    The two provenances stay apart and stay named: ``v1_identity`` is the
    superseded V1 payload identity of the chemistry, ``v2_admission`` is the live
    Process-V2 identity of the decision.

    An upstream-rejected entry carries its category and its reason-coded
    rejection row and **no decoded chemistry at all**.  That is structural, not a
    convention: a consumer cannot candidate-evaluate a trace it was never handed,
    so "present in the census, never evaluated" cannot be violated by forgetting
    to check a flag.
    """

    v1_identity: ProcessV2V1PayloadIdentity
    rejection_category: str | None
    v2_admission: ProcessV2AdmissionOverlay | None
    rejection: Mapping[str, Any] | None
    v1_record: Mapping[str, Any] | None
    addressed: AddressedPackedTrace | None

    def __post_init__(self) -> None:
        evaluable = (self.v2_admission, self.v1_record, self.addressed)
        if self.rejection_category is None:
            if self.rejection is not None or any(item is None for item in evaluable):
                raise ProcessV2Active8SourceError(
                    "an admitted source entry must carry its decoded record, its addressed "
                    "trace and its V2 admission overlay, and no rejection"
                )
        else:
            if self.rejection is None or any(item is not None for item in evaluable):
                raise ProcessV2Active8SourceError(
                    f"a {self.rejection_category} source entry must carry its rejection row "
                    "and nothing a candidate evaluator could consume"
                )

    @property
    def admitted(self) -> bool:
        return self.rejection_category is None

    @property
    def join_key(self) -> tuple[str, int]:
        """The frozen :data:`JOIN_KEY_FIELDS` key, in the frozen order."""

        return (
            self.v1_identity.v1_task_identity_sha256,
            self.v1_identity.entry_index,
        )


@dataclass(frozen=True)
class ProcessV2Active8SourceInventory:
    """The joined inventory and its stream. It grants nothing."""

    admitted_source: ProcessV2AdmittedSource
    cache_generation: ProcessV2ChunkCacheGeneration
    artifact_root: Path
    repo_root: Path
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

    @property
    def v1_migration_completion_sha256(self) -> str:
        """The historical V1 migration the cached rows came from.

        Read off the committed cache generation's semantic identity, which
        recorded it when the rows were cached, so the payload provenance survives
        the payload itself.
        """

        return str(
            self.cache_generation.completion["cache_semantic_identity"][
                "migration_completion_sha256"
            ]
        )

    @property
    def cache_completion_sha256(self) -> str:
        return str(self.cache_generation.completion["completion_sha256"])

    @property
    def cache_semantic_identity_sha256(self) -> str:
        return str(self.cache_generation.completion["cache_semantic_identity_sha256"])

    @property
    def cache_physical_identity_sha256(self) -> str:
        return str(self.cache_generation.completion["cache_physical_identity_sha256"])

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
                "v1_migration_completion_sha256": self.v1_migration_completion_sha256,
                "v1_payload_process_identity_sha256": (
                    self.v1_payload_process_identity_sha256
                ),
                "process_v2_identity_sha256": self.process_v2_identity_sha256,
                "cache_completion_sha256": self.cache_completion_sha256,
                "cache_semantic_identity_sha256": self.cache_semantic_identity_sha256,
                "cache_physical_identity_sha256": self.cache_physical_identity_sha256,
                "admitted_source_identity": self.admitted_source.identity(),
                "counts": {field: int(self.counts[field]) for field in _COUNT_FIELDS},
                "rejected_traces_by_code": dict(self.rejected_traces_by_code),
                "sources": [source.as_payload() for source in self.sources],
            }
        )
        return _sorted_deep({**body, SOURCE_SELF_HASH_FIELD: _canonical_sha256(body)})

    # ---- The production stream ----

    def iter_source_entries(self) -> Iterator[ProcessV2SourceEntry]:
        """Stream every source entry, in ``(v1_task_identity, entry_index)`` order.

        The rows come from the same committed cache generation
        :meth:`identity` names, and the decisions from the same committed rebind
        run the embedded admitted-source descriptor names, so the stream and the
        identity cannot describe different generations.

        Nothing is re-decided here.  Each row is paired with the decision already
        recorded for its join key, and the pairing is proven: the decision's
        ``v1_record_sha256`` must be the cached record's own self-hash, which the
        decoder verified, so agreement on that one digest is agreement on every
        field of the record rather than on a sample of them.

        Fail-closed throughout.  A row the overlay did not decide, a decision for
        a row the cache does not hold, a lane or split disagreement, and a
        truncated stream are all refusals; there is no partial mode, because a
        stream that yielded what it could match would silently redefine the
        corpus.
        """

        plan = self.admitted_source.plan
        pinned_process_identity = plan["pinned_process_identity"]
        v1_tasks = {
            str(task["v1_task_identity_sha256"]): task
            for task in plan["v1_payload_binding"]["v1_tasks"]
        }
        recorded = _recorded_decision_index(plan, artifact_root=self.artifact_root)
        manifests = self._manifests_by_v1_task()
        observed: Counter[str] = Counter()
        for source in self.sources:
            identity = source.v1_task_identity_sha256
            artifact_path, manifest = manifests[identity]
            output = self._mounted(artifact_path)
            decisions = _recorded_decisions_for(recorded, identity)
            pending = next(decisions, None)
            for target in chunk_targets_for_source(
                manifest, source_artifact_path=artifact_path
            ):
                try:
                    rows = read_process_v2_chunk_target(
                        output,
                        target=target,
                        expected_process_identity=pinned_process_identity,
                        sentinel_replay_entries=0,
                        # A cached row that will not decode is an integrity
                        # failure of an artifact the cache already certified, not
                        # a per-row reason code to carry downstream.
                        recover_row_errors=False,
                        repo_root=self.repo_root,
                    )
                    for row in rows:
                        # Defence in depth, and known to be unreachable while its
                        # two neighbours hold: the admitted source proves the
                        # overlay decides every entry of every V1 task
                        # contiguously, and resolution requires each cached
                        # source's row count to equal the payload binding's. Kept
                        # so that weakening either one surfaces here instead of
                        # silently streaming a subset.
                        if pending is None or pending[0] != row.entry_index:
                            raise ProcessV2Active8SourceError(
                                "the committed rebind recorded no decision at "
                                f"{JOIN_KEY_FIELDS} = ({identity}, {row.entry_index})"
                            )
                        entry_index, decision = pending
                        pending = next(decisions, None)
                        observed["source_entries"] += 1
                        observed[
                            "admitted_entries" if decision.admitted else "rejected_entries"
                        ] += 1
                        yield self._joined_entry(
                            v1_task=v1_tasks[identity],
                            source=source,
                            entry_index=entry_index,
                            decision=decision,
                            row=row,
                        )
                except ProcessV2ChunkCacheError as error:
                    raise ProcessV2Active8SourceError(
                        f"the committed cache chunk {target.chunk_filename} of source "
                        f"{identity} is unreadable: {error}"
                    ) from error
            if pending is not None:
                raise ProcessV2Active8SourceError(
                    f"the committed rebind decided entry {pending[0]} of V1 task "
                    f"{identity}, which the committed cache does not hold"
                )
        for field in ("source_entries", "admitted_entries", "rejected_entries"):
            if observed[field] != int(self.counts[field]):
                raise ProcessV2Active8SourceError(
                    f"the source stream yielded {observed[field]} {field} and the resolved "
                    f"census declares {self.counts[field]}"
                )

    def _mounted(self, artifact_path: str) -> Path:
        try:
            return mount_process_v2_artifact_path(
                artifact_path,
                artifact_root=self.artifact_root,
                field="cache source artifact path",
            )
        except ProcessV2ChunkCacheError as error:
            raise ProcessV2Active8SourceError(str(error)) from error

    def _manifests_by_v1_task(self) -> dict[str, tuple[str, Mapping[str, Any]]]:
        return {
            str(manifest["v1_task_identity_sha256"]): (path, manifest)
            for path, manifest in zip(
                self.cache_generation.source_artifact_paths,
                self.cache_generation.source_manifests,
                strict=True,
            )
        }

    def _joined_entry(
        self,
        *,
        v1_task: Mapping[str, Any],
        source: ProcessV2Active8Source,
        entry_index: int,
        decision: Any,
        row: Any,
    ) -> ProcessV2SourceEntry:
        """Pair one cached row with its recorded decision, or refuse."""

        plan = self.admitted_source.plan
        record = row.record
        overlay = decision.proof if decision.admitted else decision.rejection
        if overlay is None:
            raise ProcessV2Active8SourceError(
                "a recorded Process-V2 decision is neither an admission nor a rejection at "
                f"{JOIN_KEY_FIELDS} = ({source.v1_task_identity_sha256}, {entry_index})"
            )
        disagreements = [
            field
            for field, decided, cached in (
                ("trace_id", overlay["trace_id"], record.get("trace_id")),
                # The record's own self-hash, verified by the decoder: equality
                # here is equality of the whole record, not of a sample of it.
                ("v1_record_sha256", overlay["v1_record_sha256"], record.get("record_sha256")),
                ("data_lane", overlay["data_lane"], source.data_lane),
                ("split", overlay["split"], source.partition_role),
            )
            if decided != cached
        ]
        if disagreements:
            # Also defence in depth today: a cache of another payload names a
            # different `semantic_shard_sha256`, which resolution refuses, and a
            # chunk read from the wrong source is refused by the cache reader,
            # which binds every target to its own on-disk manifest. What this
            # states, and they do not, is that the row and the decision are the
            # SAME record rather than merely the same position.
            raise ProcessV2Active8SourceError(
                "the committed cache row and the committed rebind decision disagree on "
                f"{sorted(disagreements)} at {JOIN_KEY_FIELDS} = "
                f"({source.v1_task_identity_sha256}, {entry_index})"
            )
        v1_identity = ProcessV2V1PayloadIdentity(
            data_lane=str(record["data_lane"]),
            split=str(record["split"]),
            trace_id=str(record["trace_id"]),
            entry_index=int(entry_index),
            path_length=int(record["path_length"]),
            record_sha256=str(record["record_sha256"]),
            process_semantics=str(record["process_semantics"]),
            process_identity_sha256=str(record["process_identity_sha256"]),
            process_contract_sha256=str(record["process_contract_sha256"]),
            trace_schema=str(record["schema"]),
            trace_schema_version=int(record["schema_version"]),
            v1_task_identity_sha256=str(v1_task["v1_task_identity_sha256"]),
            v1_task_artifact_path=str(v1_task["v1_task_artifact_path"]),
        )
        if not decision.admitted:
            return ProcessV2SourceEntry(
                v1_identity=v1_identity,
                rejection_category=UPSTREAM_REJECTED,
                v2_admission=None,
                rejection=dict(overlay),
                v1_record=None,
                addressed=None,
            )
        return ProcessV2SourceEntry(
            v1_identity=v1_identity,
            rejection_category=None,
            v2_admission=ProcessV2AdmissionOverlay(
                admitted=True,
                run_identity_sha256=str(plan["run_identity_sha256"]),
                task_identity_sha256=str(decision.task_identity_sha256),
                process_v2_identity_sha256=self.process_v2_identity_sha256,
                proof_sha256=str(overlay["proof_sha256"]),
                replayed_transitions=int(overlay["replayed_transitions"]),
                atom_delete_teachers_by_candidate_source=dict(
                    overlay["atom_delete_teachers_by_candidate_source"]
                ),
                process_v2_atom_delete_candidates=[
                    list(slots) for slots in overlay["process_v2_atom_delete_candidates"]
                ],
            ),
            rejection=None,
            v1_record=record,
            addressed=row.addressed,
        )


# ---- Resolution ----


def _open_bound_cache_generation(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
    verify_chunk_bytes: bool,
) -> ProcessV2ChunkCacheGeneration:
    """Reopen the exact committed cache generation the plan binds.

    The generation is named by the plan rather than by the caller, so a source
    cannot be pointed at a different generation than the one the decisions were
    made against.  Every digest the binding froze is re-required against the
    completion that is actually on disk, which is what makes "the same committed
    generation" a checked statement instead of an assumption.
    """

    try:
        require_production_source_geometry(plan, label="the Process-V2 rebind plan")
        binding = validate_process_v2_rebind_cache_binding(plan["cache_binding"])
    except ProcessV2RebindError as error:
        raise ProcessV2Active8SourceError(
            f"the Process-V2 rebind plan does not bind a committed chunk cache: {error}"
        ) from error
    try:
        completion = load_committed_process_v2_chunk_cache_completion(
            str(binding["cache_run_artifact_root"]),
            artifact_root=Path(artifact_root),
            repo_root=Path(repo_root),
        )
        disagreements = [
            field
            for field, bound, observed in (
                (
                    "cache_completion_sha256",
                    binding["cache_completion_sha256"],
                    completion["completion_sha256"],
                ),
                (
                    "cache_semantic_identity_sha256",
                    binding["cache_semantic_identity_sha256"],
                    completion["cache_semantic_identity_sha256"],
                ),
                (
                    "cache_planned_semantic_identity_sha256",
                    binding["cache_planned_semantic_identity_sha256"],
                    completion["planned_semantic_identity_sha256"],
                ),
                (
                    "cache_physical_identity_sha256",
                    binding["cache_physical_identity_sha256"],
                    completion["cache_physical_identity_sha256"],
                ),
                (
                    "cache_source_inventory_sha256",
                    binding["cache_source_inventory_sha256"],
                    completion["source_inventory_sha256"],
                ),
                ("records_per_chunk", int(binding["records_per_chunk"]),
                 int(completion["records_per_chunk"])),
                ("source_count", int(binding["source_count"]), int(completion["source_count"])),
                ("entries", int(binding["entries"]), int(completion["entries"])),
                ("chunk_count", int(binding["chunk_count"]), int(completion["chunk_count"])),
            )
            if bound != observed
        ]
        if disagreements:
            raise ProcessV2Active8SourceError(
                "the committed chunk-cache generation on disk is not the one the rebind plan "
                f"bound; it disagrees on {sorted(disagreements)}"
            )
        return open_process_v2_chunk_cache(
            completion,
            artifact_root=Path(artifact_root),
            repo_root=Path(repo_root),
            verify_chunk_bytes=verify_chunk_bytes,
        )
    except ProcessV2ChunkCacheError as error:
        raise ProcessV2Active8SourceError(
            f"the committed Process-V2 chunk cache cannot be opened: {error}"
        ) from error


def resolve_process_v2_active8_source_inventory(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path,
    verify_chunk_bytes: bool = True,
) -> ProcessV2Active8SourceInventory:
    """Join the committed cache generation to the Process-V2 admission overlay.

    Both halves are resolved by their own authority and then cross-checked. The
    check that matters is that they describe the SAME payload: an admitted
    overlay proved against one migration silently paired with a different
    migration's cached rows would produce a corpus whose chemistry and whose
    admission decisions came from different runs.

    ``verify_chunk_bytes`` is the chunk cache's own whole-generation guarantee and
    defaults to on.  Turning it off checks every manifest and every declared
    identity but not the chunk payloads; the stream hashes each chunk it reads
    anyway, so the difference is exactly the chunks nobody reads -- and the
    identity is published whether or not anybody streams.
    """

    admitted = resolve_process_v2_admitted_source(
        plan, artifact_root=Path(artifact_root), repo_root=Path(repo_root)
    )
    generation = _open_bound_cache_generation(
        admitted.plan,
        artifact_root=artifact_root,
        repo_root=repo_root,
        verify_chunk_bytes=verify_chunk_bytes,
    )

    # The overlay names its V1 tasks; the cache names the same tasks. If the two
    # task sets differ, they are not the same payload.
    binding = admitted.plan["v1_payload_binding"]
    overlay_tasks = {
        str(task["v1_task_identity_sha256"]): task for task in binding["v1_tasks"]
    }
    cached_tasks: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for path, manifest in zip(
        generation.source_artifact_paths, generation.source_manifests, strict=True
    ):
        cached_tasks[str(manifest["v1_task_identity_sha256"])] = (path, manifest)
    if set(overlay_tasks) != set(cached_tasks):
        raise ProcessV2Active8SourceError(
            "the Process-V2 overlay and the committed chunk cache name different "
            f"task sets: overlay-only={sorted(set(overlay_tasks) - set(cached_tasks))} "
            f"cache-only={sorted(set(cached_tasks) - set(overlay_tasks))}"
        )
    for task_id, overlay_task in overlay_tasks.items():
        _path, manifest = cached_tasks[task_id]
        if str(overlay_task["v1_semantic_shard_sha256"]) != str(
            manifest["semantic_shard_sha256"]
        ):
            raise ProcessV2Active8SourceError(
                f"task {task_id} names a different semantic shard in the overlay than "
                "in the committed chunk cache"
            )
        if str(overlay_task["data_lane"]) != str(manifest["data_lane"]) or str(
            overlay_task["split"]
        ) != str(manifest["split"]):
            raise ProcessV2Active8SourceError(
                f"task {task_id} lane or split disagrees between the overlay and the "
                "committed chunk cache"
            )
        if int(overlay_task["v1_entries"]) != int(manifest["entries"]):
            raise ProcessV2Active8SourceError(
                f"task {task_id} binds {overlay_task['v1_entries']} V1 entries and the "
                f"committed chunk cache holds {manifest['entries']}"
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
            data_lane=str(cached_tasks[task_id][1]["data_lane"]),
            partition_role=str(cached_tasks[task_id][1]["split"]),
            v1_task_identity_sha256=task_id,
            cache_source_task_identity_sha256=str(
                cached_tasks[task_id][1]["task_identity_sha256"]
            ),
            cache_source_artifact_path=str(cached_tasks[task_id][0]),
            semantic_shard_sha256=str(cached_tasks[task_id][1]["semantic_shard_sha256"]),
            semantic_manifest_sha256=str(cached_tasks[task_id][1]["semantic_manifest_sha256"]),
            v1_entry_count=int(overlay_tasks[task_id]["v1_entries"]),
            admitted_entry_count=int(admitted_by_task[task_id]),
            rejected_entry_count=int(rejected_by_task[task_id]),
        )
        # Sorted by the first frozen join-key field, so the published rows and
        # the stream are in one order rather than two.
        for task_id in sorted(cached_tasks)
    )

    # Source == admitted + rejected, per task and in total. The library already
    # enforces this run-wide; asserting it again here is what makes the JOIN
    # trustworthy rather than merely the reduction.
    for source in sources:
        if source.admitted_entry_count + source.rejected_entry_count != (
            source.v1_entry_count
        ):
            raise ProcessV2Active8SourceError(
                f"task {source.v1_task_identity_sha256} admits "
                f"{source.admitted_entry_count} and rejects "
                f"{source.rejected_entry_count} of {source.v1_entry_count} entries"
            )
    counts = {field: int(admitted.counts[field]) for field in _COUNT_FIELDS}
    if sum(source.v1_entry_count for source in sources) != counts["source_entries"]:
        raise ProcessV2Active8SourceError(
            "the joined per-task census does not account for every source entry"
        )

    return ProcessV2Active8SourceInventory(
        admitted_source=admitted,
        cache_generation=generation,
        artifact_root=Path(artifact_root),
        repo_root=Path(repo_root),
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
    """The exact field set. Order has one authority, :func:`_sorted_mapping`.

    Every caller sorts-checks the same object first, so a second order comparison
    here could never fire; an unreachable guard reads as defence in depth and is
    not.
    """

    observed = set(payload)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        unexpected = sorted(observed - set(expected))
        _identity_fail(
            f"{label} field set differs from the declared shape; missing={missing} "
            f"unexpected={unexpected}"
        )


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
            f"{SOURCE_SCHEMA_VERSION}. Earlier versions took their lane/role rows from the "
            "live V1 migration loader, keyed each row by that loader's task identity, and "
            "named no committed cache generation, so they cannot be read as a subset of "
            "this one"
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
        "cache_completion_sha256",
        "cache_physical_identity_sha256",
        "cache_semantic_identity_sha256",
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
            "cache_source_task_identity_sha256",
            "semantic_manifest_sha256",
            "semantic_shard_sha256",
            "v1_task_identity_sha256",
        ):
            _exact_sha256(entry, key, label=where)
        for key in ("data_lane", "partition_role"):
            if type(entry[key]) is not str or not entry[key]:
                _identity_fail(f"{where}.{key} is not a non-empty string")
        task = str(entry["v1_task_identity_sha256"])
        if task in seen:
            _identity_fail(f"{where} repeats task {task}; one task is one row")
        seen.add(task)
        if task < previous:
            _identity_fail(
                f"{where} is out of order; rows are sorted by V1 task identity so the "
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
    "ProcessV2SourceEntry",
    "resolve_process_v2_active8_source_inventory",
    "validate_process_v2_active8_source_identity",
]
