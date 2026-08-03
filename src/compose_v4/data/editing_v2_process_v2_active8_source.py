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
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from compose_v4.data.editing_process_v2_admitted_source import (
    ProcessV2AdmittedSource,
    resolve_process_v2_admitted_source,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    EditingV2SemanticActive8SourceInventory,
    resolve_editing_v2_semantic_active8_sources,
)

# ---- Frozen identity ----

SOURCE_SCHEMA = "compose.data.editing_v2_process_v2_active8_source"
SOURCE_SCHEMA_VERSION = 1
SOURCE_STATUS = "VERIFIED_PROCESS_V2_ACTIVE8_SOURCE_NO_DOWNSTREAM_AUTHORITY"

_COUNT_FIELDS = (
    "source_entries",
    "admitted_entries",
    "rejected_entries",
    "admitted_states",
    "admitted_transitions",
)


class ProcessV2Active8SourceError(RuntimeError):
    """The Process-V2 Active8 source cannot be resolved from these inputs."""


# ---- Deterministic hashing ----


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
        """The descriptor a downstream materializer must record verbatim."""

        body: dict[str, Any] = {
            "schema": SOURCE_SCHEMA,
            "schema_version": SOURCE_SCHEMA_VERSION,
            "status": SOURCE_STATUS,
            "training_authorized": False,
            "gate_zero_authorized": False,
            "t1_authorized": False,
            "bounded_p50_authorized": False,
            "v1_migration_completion_sha256": (
                self.v1_inventory.migration_completion_sha256
            ),
            "v1_payload_process_identity_sha256": self.v1_payload_process_identity_sha256,
            "process_v2_identity_sha256": self.process_v2_identity_sha256,
            "admitted_source_identity": self.admitted_source.identity(),
            "counts": {field: int(self.counts[field]) for field in _COUNT_FIELDS},
            "rejected_traces_by_code": dict(sorted(self.rejected_traces_by_code.items())),
            "sources": [source.as_payload() for source in self.sources],
        }
        return {**body, "process_v2_active8_source_sha256": _canonical_sha256(body)}


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


__all__ = [
    "SOURCE_SCHEMA",
    "SOURCE_SCHEMA_VERSION",
    "SOURCE_STATUS",
    "ProcessV2Active8Source",
    "ProcessV2Active8SourceError",
    "ProcessV2Active8SourceInventory",
    "resolve_process_v2_active8_source_inventory",
]
