"""Resumable map/reduce publication for the active-8 trace inventory.

The scientific admission predicate lives in :mod:`active8_trace_inventory`.
This module only supplies bounded-distribution mechanics:

* deterministic entry-range tasks over each exact packed source shard;
* content-addressed, immutable publication with byte-collision checks;
* immutable task receipts, so interrupted maps resume without recomputation;
* a reducer that refuses gaps, overlaps, extras, or an incomplete receipt set;
* one final deterministic decision object per original packed source shard;
* a unified inventory manifest compatible with the local inventory readers.

No function in this module authorizes or launches training.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from compose_v4.chem.persistent_state_identity import (
    PERSISTENT_STATE_DIGEST_SCHEMA,
    PERSISTENT_STATE_DIGEST_VERSION,
)
from compose_v4.data.active8_trace_inventory import (
    ACTIVE8_FAMILIES,
    ACTIVE8_TRACE_DECISION_SCHEMA,
    ACTIVE8_TRACE_DECISION_SCHEMA_VERSION,
    ACTIVE8_TRACE_INVENTORY_SCHEMA,
    ACTIVE8_TRACE_INVENTORY_SCHEMA_VERSION,
    Active8SourceShard,
    ExactCandidateChecker,
    inventory_record_for_trace,
)
from compose_v4.data.active8_trace_inventory import (
    implementation_identity as active8_implementation_identity,
)
from compose_v4.data.packed_trace_store import (
    PACKED_STORE_SCHEMA,
    PACKED_STORE_SCHEMA_VERSION,
    manifest_path_for,
    read_frozen_source_addressed_packed_shard,
)
from compose_v4.data.provenance_overlay import overlay_path_for

ACTIVE8_MAPREDUCE_PLAN_SCHEMA = "compose.data.active8_inventory_mapreduce_plan"
ACTIVE8_MAPREDUCE_PLAN_SCHEMA_VERSION = 2
ACTIVE8_MAP_RECEIPT_SCHEMA = "compose.data.active8_inventory_map_receipt"
ACTIVE8_MAP_RECEIPT_SCHEMA_VERSION = 2
ACTIVE8_MAPREDUCE_COMPLETE_SCHEMA = "compose.data.active8_inventory_mapreduce_complete"
ACTIVE8_MAPREDUCE_COMPLETE_SCHEMA_VERSION = 2

ENTRY_RANGE_ALGORITHM = "fixed_target_entries_contiguous_v1"
DEFAULT_TARGET_ENTRIES_PER_RANGE = 500
DEFAULT_WORKER_RESOURCES = {
    "cpu": 2.0,
    "memory_mb": 8192,
    "max_containers": 64,
    "omp_num_threads": 1,
}

_MAPREDUCE_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/data/active8_inventory_mapreduce.py",
    "src/compose_v4/data/active8_trace_inventory.py",
    "modal_apps/build_active8_trace_inventory_app.py",
)


class Active8MapReduceError(RuntimeError):
    """The distributed inventory cannot prove immutable complete coverage."""


class Active8MapReduceIncomplete(Active8MapReduceError):
    """At least one expected source decision is absent."""


@dataclass(frozen=True)
class Active8MapTask:
    """Content-addressed request for one immutable source-entry range."""

    run_identity_sha256: str
    task_identity_sha256: str
    manifest_layer: str
    envelope_layer: str
    partition: str
    relative_path: str
    source_path: str
    packed_shard_name: str
    packed_shard_content_sha256: str
    packed_manifest_sha256: str
    packed_provenance_overlay_sha256: str | None
    packed_manifest_entries: int
    range_index: int
    range_count: int
    entry_start: int
    entry_stop: int
    range_policy: dict[str, object]
    range_policy_sha256: str
    worker_resources: dict[str, object]
    worker_resources_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "run_identity_sha256",
            "task_identity_sha256",
            "packed_shard_content_sha256",
            "packed_manifest_sha256",
            "range_policy_sha256",
            "worker_resources_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        overlay = self.packed_provenance_overlay_sha256
        if overlay is not None and not _is_sha256(overlay):
            raise ValueError("packed_provenance_overlay_sha256 is malformed")
        for name in (
            "packed_manifest_entries",
            "range_index",
            "entry_start",
            "entry_stop",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.range_count) is not int or self.range_count <= 0:
            raise ValueError("range_count must be a positive integer")
        if (
            self.range_index >= self.range_count
            or self.entry_start > self.entry_stop
            or self.entry_stop > self.packed_manifest_entries
        ):
            raise ValueError("map task entry range is outside its source census")
        if (
            _normalize_worker_resources(self.worker_resources) != self.worker_resources
            or _sha256_payload(self.worker_resources) != self.worker_resources_sha256
        ):
            raise ValueError("map task worker resources are malformed")
        if (
            set(self.range_policy) != {"algorithm", "target_entries_per_range"}
            or self.range_policy.get("algorithm") != ENTRY_RANGE_ALGORITHM
            or type(self.range_policy.get("target_entries_per_range")) is not int
            or int(self.range_policy["target_entries_per_range"]) <= 0
            or _sha256_payload(self.range_policy) != self.range_policy_sha256
        ):
            raise ValueError("map task range policy is malformed")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.manifest_layer,
                self.envelope_layer,
                self.partition,
                self.relative_path,
                self.source_path,
                self.packed_shard_name,
            )
        ):
            raise ValueError("map task text fields must be non-empty")

    @property
    def source_binding(self) -> dict[str, object]:
        return {
            "manifest_layer": self.manifest_layer,
            "envelope_layer": self.envelope_layer,
            "partition": self.partition,
            "relative_path": self.relative_path,
            "packed_shard_name": self.packed_shard_name,
            "packed_shard_content_sha256": self.packed_shard_content_sha256,
            "packed_manifest_sha256": self.packed_manifest_sha256,
            "packed_provenance_overlay_sha256": (self.packed_provenance_overlay_sha256),
            "packed_manifest_entries": self.packed_manifest_entries,
        }

    @property
    def range_binding(self) -> dict[str, int]:
        return {
            "range_index": self.range_index,
            "range_count": self.range_count,
            "entry_start": self.entry_start,
            "entry_stop": self.entry_stop,
        }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _normalize_worker_resources(
    value: Mapping[str, object] | None,
) -> dict[str, object]:
    resources = dict(DEFAULT_WORKER_RESOURCES if value is None else value)
    required = {"cpu", "memory_mb", "max_containers", "omp_num_threads"}
    if set(resources) != required:
        raise ValueError(
            "worker_resources must contain exactly cpu, memory_mb, "
            "max_containers, and omp_num_threads"
        )
    cpu = resources["cpu"]
    if isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or float(cpu) <= 0:
        raise ValueError("worker_resources.cpu must be positive")
    normalized: dict[str, object] = {"cpu": float(cpu)}
    for field in ("memory_mb", "max_containers", "omp_num_threads"):
        item = resources[field]
        if type(item) is not int or item <= 0:
            raise ValueError(f"worker_resources.{field} must be a positive integer")
        normalized[field] = item
    if int(normalized["max_containers"]) > 64:
        raise ValueError("worker_resources.max_containers may not exceed 64")
    return normalized


def _packed_manifest_entry_count(path: Path) -> int:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Active8MapReduceError(f"packed manifest is unreadable: {path}") from error
    entries = payload.get("entries") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != PACKED_STORE_SCHEMA
        or payload.get("schema_version") != PACKED_STORE_SCHEMA_VERSION
        or type(entries) is not int
        or entries < 0
    ):
        raise Active8MapReduceError(
            f"packed manifest has unsupported schema or entry census: {path}"
        )
    return entries


def _entry_range_bindings(
    entries: int,
    *,
    target_entries_per_range: int,
) -> tuple[dict[str, int], ...]:
    if type(entries) is not int or entries < 0:
        raise ValueError("entries must be a nonnegative integer")
    if type(target_entries_per_range) is not int or target_entries_per_range <= 0:
        raise ValueError("target_entries_per_range must be a positive integer")
    bounds = (
        [(0, 0)]
        if entries == 0
        else [
            (start, min(start + target_entries_per_range, entries))
            for start in range(0, entries, target_entries_per_range)
        ]
    )
    count = len(bounds)
    return tuple(
        {
            "range_index": index,
            "range_count": count,
            "entry_start": start,
            "entry_stop": stop,
        }
        for index, (start, stop) in enumerate(bounds)
    )


def _implementation_identity(*, repo_root: Path | None = None) -> dict[str, object]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    sources: dict[str, str] = {}
    for relative in _MAPREDUCE_IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise Active8MapReduceError(f"map/reduce implementation source is absent: {path}")
        sources[relative] = _sha256_file(path)
    return {
        "sources": sources,
        "implementation_sha256": _sha256_payload(sources),
    }


def _task_identity_payload(
    *,
    run_identity_sha256: str,
    source_binding: Mapping[str, object],
    range_binding: Mapping[str, object],
    range_policy_sha256: str,
    worker_resources_sha256: str,
) -> dict[str, object]:
    return {
        "schema": ACTIVE8_MAP_RECEIPT_SCHEMA,
        "schema_version": ACTIVE8_MAP_RECEIPT_SCHEMA_VERSION,
        "run_identity_sha256": run_identity_sha256,
        "source_binding": dict(source_binding),
        "range_binding": dict(range_binding),
        "range_policy_sha256": range_policy_sha256,
        "worker_resources_sha256": worker_resources_sha256,
    }


def plan_active8_mapreduce(
    shards: Iterable[Active8SourceShard],
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, object],
    support_contract_sha256: str,
    repo_root: Path | None = None,
    target_entries_per_range: int = DEFAULT_TARGET_ENTRIES_PER_RANGE,
    worker_resources: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Freeze exact sources, entry ranges, resources, and stable map identities."""

    if not _is_sha256(support_contract_sha256):
        raise ValueError("support_contract_sha256 must be a lowercase SHA-256")
    if type(target_entries_per_range) is not int or target_entries_per_range <= 0:
        raise ValueError("target_entries_per_range must be a positive integer")
    normalized_resources = _normalize_worker_resources(worker_resources)
    worker_resources_sha256 = _sha256_payload(normalized_resources)
    source_manifest_path = Path(source_manifest_path)
    if not source_manifest_path.is_file():
        raise Active8MapReduceError("unified packed manifest is absent")
    decoded = json.loads(source_manifest_path.read_text())
    if _sha256_payload(decoded) != _sha256_payload(source_manifest):
        raise Active8MapReduceError("decoded unified manifest differs from the exact source file")
    ordered = tuple(
        sorted(
            shards,
            key=lambda item: (
                item.manifest_layer,
                item.partition,
                item.relative_path,
            ),
        )
    )
    if not ordered:
        raise Active8MapReduceError("at least one packed source shard is required")

    active8_identity = active8_implementation_identity(repo_root=repo_root)
    mapreduce_identity = _implementation_identity(repo_root=repo_root)
    source_manifest_sha256 = _sha256_file(source_manifest_path)
    preliminary_bindings: list[dict[str, object]] = []
    runtime_paths: list[str] = []
    seen_physical_digests: set[str] = set()
    for shard in ordered:
        source = Path(shard.path)
        packed_manifest = manifest_path_for(source)
        packed_overlay = overlay_path_for(source)
        if not source.is_file() or not packed_manifest.is_file():
            raise Active8MapReduceError(f"packed shard or its manifest is absent: {source}")
        physical_digest = _sha256_file(source)
        if physical_digest in seen_physical_digests:
            raise Active8MapReduceError("one physical packed source is declared more than once")
        seen_physical_digests.add(physical_digest)
        binding = {
            "manifest_layer": shard.manifest_layer,
            "envelope_layer": shard.envelope_layer,
            "partition": shard.partition,
            "relative_path": shard.relative_path,
            "packed_shard_name": source.name,
            "packed_shard_content_sha256": physical_digest,
            "packed_manifest_sha256": _sha256_file(packed_manifest),
            "packed_provenance_overlay_sha256": (
                _sha256_file(packed_overlay) if packed_overlay.is_file() else None
            ),
            "packed_manifest_entries": _packed_manifest_entry_count(packed_manifest),
        }
        preliminary_bindings.append(binding)
        runtime_paths.append(str(source))

    range_policy = {
        "algorithm": ENTRY_RANGE_ALGORITHM,
        "target_entries_per_range": target_entries_per_range,
    }
    range_policy_sha256 = _sha256_payload(range_policy)
    source_range_partitions = [
        {
            "source_binding_sha256": _sha256_payload(binding),
            "packed_manifest_entries": binding["packed_manifest_entries"],
            "ranges": list(
                _entry_range_bindings(
                    int(binding["packed_manifest_entries"]),
                    target_entries_per_range=target_entries_per_range,
                )
            ),
        }
        for binding in preliminary_bindings
    ]
    run_payload = {
        "schema": ACTIVE8_MAPREDUCE_PLAN_SCHEMA,
        "schema_version": ACTIVE8_MAPREDUCE_PLAN_SCHEMA_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_semantic_sha256": _sha256_payload(source_manifest),
        "support_contract_sha256": support_contract_sha256,
        "active8_families": list(ACTIVE8_FAMILIES),
        "active8_implementation_sha256": active8_identity["implementation_sha256"],
        "mapreduce_implementation_sha256": mapreduce_identity["implementation_sha256"],
        "source_bindings": preliminary_bindings,
        "range_policy": range_policy,
        "range_policy_sha256": range_policy_sha256,
        "source_range_partitions": source_range_partitions,
        "worker_resources": normalized_resources,
        "worker_resources_sha256": worker_resources_sha256,
    }
    run_identity_sha256 = _sha256_payload(run_payload)
    tasks: list[Active8MapTask] = []
    for binding, runtime_path, partition in zip(
        preliminary_bindings,
        runtime_paths,
        source_range_partitions,
        strict=True,
    ):
        for range_binding in partition["ranges"]:
            task_identity_sha256 = _sha256_payload(
                _task_identity_payload(
                    run_identity_sha256=run_identity_sha256,
                    source_binding=binding,
                    range_binding=range_binding,
                    range_policy_sha256=range_policy_sha256,
                    worker_resources_sha256=worker_resources_sha256,
                )
            )
            tasks.append(
                Active8MapTask(
                    run_identity_sha256=run_identity_sha256,
                    task_identity_sha256=task_identity_sha256,
                    source_path=runtime_path,
                    range_policy=range_policy,
                    range_policy_sha256=range_policy_sha256,
                    worker_resources=normalized_resources,
                    worker_resources_sha256=worker_resources_sha256,
                    **binding,
                    **range_binding,
                )
            )
    return {
        **run_payload,
        "run_identity_sha256": run_identity_sha256,
        "expected_source_decisions": len(preliminary_bindings),
        "expected_map_tasks": len(tasks),
        "tasks": [asdict(task) for task in tasks],
        "active8_implementation_identity": active8_identity,
        "mapreduce_implementation_identity": mapreduce_identity,
    }


def _task_from_mapping(raw: Mapping[str, object]) -> Active8MapTask:
    try:
        return Active8MapTask(**dict(raw))
    except (TypeError, ValueError) as error:
        raise Active8MapReduceError("active-8 map task is malformed") from error


def _run_root(output_root: Path, run_identity_sha256: str) -> Path:
    return Path(output_root) / "runs" / run_identity_sha256


def _receipt_path(output_root: Path, task: Active8MapTask) -> Path:
    return (
        _run_root(output_root, task.run_identity_sha256)
        / "receipts"
        / f"{task.task_identity_sha256}.json"
    )


def _publish_bytes_immutable(path: Path, content: bytes) -> bool:
    """Atomically publish bytes; an existing content object must be identical.

    Modal Volumes do not support POSIX hard links. The temporary file is
    therefore created in the destination directory and atomically renamed.
    Concurrent writers can only target the same content-addressed path when
    their bytes have the same SHA-256; the destination is verified after the
    rename as well as before reuse.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_sha256 = hashlib.sha256(content).hexdigest()
    if path.exists():
        if _sha256_file(path) != expected_sha256 or path.read_bytes() != content:
            raise Active8MapReduceError(f"immutable artifact collision at {path}")
        return True
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        if _sha256_file(path) != expected_sha256 or path.read_bytes() != content:
            raise Active8MapReduceError(f"immutable artifact collision at {path}")
        return False
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _publish_file_immutable(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> bool:
    """Atomically publish one potentially large content-addressed object."""

    source = Path(source)
    destination = Path(destination)
    if _sha256_file(source) != expected_sha256:
        raise Active8MapReduceError("temporary artifact digest changed before publish")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise Active8MapReduceError(f"immutable content-object collision at {destination}")
        return True
    os.replace(source, destination)
    if _sha256_file(destination) != expected_sha256:
        raise Active8MapReduceError(f"immutable content-object collision at {destination}")
    return False


def _receipt_self_hash(payload: Mapping[str, object]) -> str:
    body = dict(payload)
    body.pop("receipt_sha256", None)
    return _sha256_payload(body)


def _load_receipt(
    path: Path,
    *,
    task: Active8MapTask,
    output_root: Path,
) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Active8MapReduceError(f"map receipt is unreadable: {path}") from error
    if (
        payload.get("schema") != ACTIVE8_MAP_RECEIPT_SCHEMA
        or payload.get("schema_version") != ACTIVE8_MAP_RECEIPT_SCHEMA_VERSION
        or payload.get("run_identity_sha256") != task.run_identity_sha256
        or payload.get("task_identity_sha256") != task.task_identity_sha256
        or payload.get("source_binding") != task.source_binding
        or payload.get("range_binding") != task.range_binding
        or payload.get("range_policy") != task.range_policy
        or payload.get("range_policy_sha256") != task.range_policy_sha256
        or payload.get("worker_resources") != task.worker_resources
        or payload.get("worker_resources_sha256") != task.worker_resources_sha256
    ):
        raise Active8MapReduceError("map receipt disagrees with its frozen task")
    if payload.get("receipt_sha256") != _receipt_self_hash(payload):
        raise Active8MapReduceError("map receipt self-hash mismatch")
    object_path = Path(output_root) / str(payload["decision_object"])
    if not object_path.is_file() or _sha256_file(object_path) != payload["decision_object_sha256"]:
        raise Active8MapReduceError("map receipt decision object is absent or hash-mismatched")
    return payload


def map_active8_source_decisions(
    raw_task: Mapping[str, object],
    *,
    exact_candidate_checker: ExactCandidateChecker,
    output_root: Path,
) -> dict[str, object]:
    """Map one immutable source-entry range, reusing its receipt on resume."""

    task = _task_from_mapping(raw_task)
    output_root = Path(output_root)
    receipt_path = _receipt_path(output_root, task)
    if receipt_path.exists():
        return {
            **_load_receipt(
                receipt_path,
                task=task,
                output_root=output_root,
            ),
            "reused": True,
        }
    source = Path(task.source_path)
    if not source.is_file():
        raise Active8MapReduceError("map source is absent from its frozen runtime path")
    # The range reader below performs the authoritative full shard, manifest,
    # and overlay byte validation. Do not hash the same large source twice
    # before scoring a range.

    counts = Counter(
        traces=0,
        accepted_traces=0,
        excluded_traces=0,
        accepted_progress_rows=0,
        accepted_nonterminal_rows=0,
        accepted_terminal_rows=0,
    )
    exclusions_by_reason: Counter[str] = Counter()
    family_rows: Counter[str] = Counter({family: 0 for family in ACTIVE8_FAMILIES})
    scratch_root = _run_root(output_root, task.run_identity_sha256) / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    observed_entries: list[int] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=scratch_root,
            prefix=f".{task.task_identity_sha256}.",
            suffix=".jsonl.gz.tmp",
            delete=False,
        ) as raw:
            temporary_name = raw.name
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed:
                for addressed in read_frozen_source_addressed_packed_shard(
                    source,
                    expected_shard_sha256=task.packed_shard_content_sha256,
                    expected_manifest_sha256=task.packed_manifest_sha256,
                    expected_overlay_sha256=(task.packed_provenance_overlay_sha256),
                    verify_fraction=0.0,
                    entry_start=task.entry_start,
                    entry_stop=task.entry_stop,
                ):
                    address = addressed.address
                    if (
                        address.packed_shard_content_sha256 != task.packed_shard_content_sha256
                        or address.layer != task.envelope_layer
                        or address.partition != task.partition
                    ):
                        raise Active8MapReduceError(
                            "addressed trace disagrees with its frozen source task"
                        )
                    observed_entries.append(address.entry_index)
                    record = inventory_record_for_trace(
                        addressed,
                        exact_candidate_checker=exact_candidate_checker,
                    )
                    compressed.write(_canonical_json_bytes(record) + b"\n")
                    counts["traces"] += 1
                    if record["decision"] == "accepted":
                        rows = record["progress_rows"]
                        counts["accepted_traces"] += 1
                        counts["accepted_progress_rows"] += len(rows)
                        counts["accepted_nonterminal_rows"] += address.path_length
                        counts["accepted_terminal_rows"] += 1
                        for row in rows:
                            family = row["teacher_family"]
                            if family is not None:
                                family_rows[str(family)] += 1
                    else:
                        counts["excluded_traces"] += 1
                        for exclusion in record["exclusions"]:
                            exclusions_by_reason[str(exclusion["reason"])] += 1
            raw.flush()
            os.fsync(raw.fileno())
        if observed_entries != list(range(task.entry_start, task.entry_stop)):
            raise Active8MapReduceError(
                "packed source did not yield its complete ordered entry range"
            )
        temporary = Path(temporary_name)
        object_sha256 = _sha256_file(temporary)
        object_relative = (
            Path("objects") / "decisions" / object_sha256[:2] / f"{object_sha256}.jsonl.gz"
        )
        object_reused = _publish_file_immutable(
            temporary,
            output_root / object_relative,
            expected_sha256=object_sha256,
        )
        receipt: dict[str, object] = {
            "schema": ACTIVE8_MAP_RECEIPT_SCHEMA,
            "schema_version": ACTIVE8_MAP_RECEIPT_SCHEMA_VERSION,
            "run_identity_sha256": task.run_identity_sha256,
            "task_identity_sha256": task.task_identity_sha256,
            "source_binding": task.source_binding,
            "range_binding": task.range_binding,
            "range_policy": task.range_policy,
            "range_policy_sha256": task.range_policy_sha256,
            "worker_resources": task.worker_resources,
            "worker_resources_sha256": task.worker_resources_sha256,
            "decision_object": object_relative.as_posix(),
            "decision_object_sha256": object_sha256,
            "counts": dict(counts),
            "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
            "accepted_nonterminal_rows_by_family": {
                family: family_rows[family] for family in ACTIVE8_FAMILIES
            },
        }
        receipt["receipt_sha256"] = _receipt_self_hash(receipt)
        receipt_reused = _publish_bytes_immutable(
            receipt_path,
            json.dumps(
                receipt,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n",
        )
        return {
            **receipt,
            "reused": receipt_reused and object_reused,
        }
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _iter_decision_records(
    object_path: Path,
    *,
    source_digest: str,
    entry_start: int,
    entry_stop: int,
) -> Iterable[dict[str, object]]:
    expected_entry = entry_start
    seen_trace_ids: set[str] = set()
    with gzip.open(object_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise Active8MapReduceError(
                    f"decision object contains invalid JSON: {object_path}"
                ) from error
            if (
                not isinstance(record, dict)
                or record.get("schema") != ACTIVE8_TRACE_DECISION_SCHEMA
                or record.get("schema_version") != ACTIVE8_TRACE_DECISION_SCHEMA_VERSION
            ):
                raise Active8MapReduceError("decision object contains another schema")
            key = record.get("trace_key") or {}
            entry_index = key.get("entry_index")
            trace_id = key.get("trace_id")
            if (
                key.get("packed_shard_content_sha256") != source_digest
                or type(entry_index) is not int
                or entry_index != expected_entry
                or not isinstance(trace_id, str)
                or not trace_id
            ):
                raise Active8MapReduceError(
                    "decision object lost or reordered physical source entries"
                )
            if trace_id in seen_trace_ids:
                raise Active8MapReduceError("decision object repeats a trace identity")
            seen_trace_ids.add(trace_id)
            expected_entry += 1
            yield record
    if expected_entry != entry_stop:
        raise Active8MapReduceError(
            f"decision object covers [{entry_start}, {expected_entry}) instead "
            f"of [{entry_start}, {entry_stop})"
        )


def _census_decision_object(
    object_path: Path,
    *,
    source_digest: str,
    entry_start: int,
    entry_stop: int,
) -> dict[str, object]:
    counts = Counter(
        traces=0,
        accepted_traces=0,
        excluded_traces=0,
        accepted_progress_rows=0,
        accepted_nonterminal_rows=0,
        accepted_terminal_rows=0,
    )
    exclusions_by_reason: Counter[str] = Counter()
    family_rows: Counter[str] = Counter({family: 0 for family in ACTIVE8_FAMILIES})
    for record in _iter_decision_records(
        object_path,
        source_digest=source_digest,
        entry_start=entry_start,
        entry_stop=entry_stop,
    ):
        counts["traces"] += 1
        if record.get("decision") == "accepted":
            rows = record.get("progress_rows")
            path_length = record.get("path_length")
            if (
                type(path_length) is not int
                or path_length < 0
                or not isinstance(rows, list)
                or len(rows) != path_length + 1
                or not rows[-1].get("is_terminal")
                or record.get("exclusions") != []
            ):
                raise Active8MapReduceError("accepted decision does not retain its complete path")
            counts["accepted_traces"] += 1
            counts["accepted_progress_rows"] += len(rows)
            counts["accepted_nonterminal_rows"] += path_length
            counts["accepted_terminal_rows"] += 1
            for row in rows:
                family = row.get("teacher_family")
                if family is not None:
                    if family not in ACTIVE8_FAMILIES:
                        raise Active8MapReduceError(
                            "accepted decision contains a non-active family"
                        )
                    family_rows[str(family)] += 1
        elif record.get("decision") == "excluded":
            if record.get("progress_rows") != [] or not record.get("exclusions"):
                raise Active8MapReduceError("excluded decision leaked rows or lacks a reason")
            counts["excluded_traces"] += 1
            for exclusion in record["exclusions"]:
                exclusions_by_reason[str(exclusion["reason"])] += 1
        else:
            raise Active8MapReduceError("decision object has an unknown verdict")
    if counts["traces"] != entry_stop - entry_start:
        raise Active8MapReduceError("decision object trace census disagrees with its entry range")
    return {
        "counts": dict(counts),
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
        "accepted_nonterminal_rows_by_family": {
            family: family_rows[family] for family in ACTIVE8_FAMILIES
        },
    }


def _assert_plan_identity(plan: Mapping[str, object]) -> tuple[Active8MapTask, ...]:
    if (
        plan.get("schema") != ACTIVE8_MAPREDUCE_PLAN_SCHEMA
        or plan.get("schema_version") != ACTIVE8_MAPREDUCE_PLAN_SCHEMA_VERSION
    ):
        raise Active8MapReduceError("active-8 map/reduce plan has another schema")
    body = {
        key: plan[key]
        for key in (
            "schema",
            "schema_version",
            "source_manifest_sha256",
            "source_manifest_semantic_sha256",
            "support_contract_sha256",
            "active8_families",
            "active8_implementation_sha256",
            "mapreduce_implementation_sha256",
            "source_bindings",
            "range_policy",
            "range_policy_sha256",
            "source_range_partitions",
            "worker_resources",
            "worker_resources_sha256",
        )
    }
    range_policy = plan.get("range_policy")
    worker_resources = plan.get("worker_resources")
    try:
        normalized_resources = _normalize_worker_resources(
            worker_resources if isinstance(worker_resources, Mapping) else None
        )
    except ValueError as error:
        raise Active8MapReduceError("active-8 worker resources are malformed") from error
    if (
        plan.get("run_identity_sha256") != _sha256_payload(body)
        or plan.get("active8_families") != list(ACTIVE8_FAMILIES)
        or not isinstance(range_policy, Mapping)
        or set(range_policy) != {"algorithm", "target_entries_per_range"}
        or range_policy.get("algorithm") != ENTRY_RANGE_ALGORITHM
        or type(range_policy.get("target_entries_per_range")) is not int
        or int(range_policy["target_entries_per_range"]) <= 0
        or plan.get("range_policy_sha256") != _sha256_payload(range_policy)
        or normalized_resources != worker_resources
        or plan.get("worker_resources_sha256") != _sha256_payload(normalized_resources)
        or not isinstance(plan.get("active8_implementation_identity"), Mapping)
        or plan["active8_implementation_identity"].get("implementation_sha256")
        != plan.get("active8_implementation_sha256")
        or not isinstance(plan.get("mapreduce_implementation_identity"), Mapping)
        or plan["mapreduce_implementation_identity"].get("implementation_sha256")
        != plan.get("mapreduce_implementation_sha256")
    ):
        raise Active8MapReduceError("active-8 map/reduce plan identity mismatch")
    source_bindings = plan.get("source_bindings")
    partitions = plan.get("source_range_partitions")
    if (
        not isinstance(source_bindings, list)
        or not source_bindings
        or not isinstance(partitions, list)
        or len(partitions) != len(source_bindings)
        or plan.get("expected_source_decisions") != len(source_bindings)
    ):
        raise Active8MapReduceError("active-8 source or range-partition census drifted")
    expected_specs: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    target_entries = int(range_policy["target_entries_per_range"])
    for source_binding, partition in zip(
        source_bindings,
        partitions,
        strict=True,
    ):
        if not isinstance(source_binding, Mapping) or not isinstance(partition, Mapping):
            raise Active8MapReduceError("active-8 source range partition is malformed")
        entries = source_binding.get("packed_manifest_entries")
        expected_ranges = (
            list(
                _entry_range_bindings(
                    entries,
                    target_entries_per_range=target_entries,
                )
            )
            if type(entries) is int and entries >= 0
            else None
        )
        if (
            expected_ranges is None
            or partition.get("source_binding_sha256") != _sha256_payload(source_binding)
            or partition.get("packed_manifest_entries") != entries
            or partition.get("ranges") != expected_ranges
        ):
            raise Active8MapReduceError(
                "active-8 ranges are not deterministic contiguous full coverage"
            )
        expected_specs.extend((source_binding, range_binding) for range_binding in expected_ranges)
    tasks = tuple(_task_from_mapping(raw) for raw in plan.get("tasks", ()))
    if (
        len(tasks) != plan.get("expected_map_tasks")
        or len(tasks) != len(expected_specs)
        or any(task.run_identity_sha256 != plan["run_identity_sha256"] for task in tasks)
        or any(
            task.source_binding != source_binding
            or task.range_binding != range_binding
            or task.range_policy != plan["range_policy"]
            or task.range_policy_sha256 != plan["range_policy_sha256"]
            or task.worker_resources_sha256 != plan["worker_resources_sha256"]
            or task.worker_resources != plan["worker_resources"]
            for task, (source_binding, range_binding) in zip(
                tasks,
                expected_specs,
                strict=True,
            )
        )
        or any(
            task.task_identity_sha256
            != _sha256_payload(
                _task_identity_payload(
                    run_identity_sha256=task.run_identity_sha256,
                    source_binding=task.source_binding,
                    range_binding=task.range_binding,
                    range_policy_sha256=task.range_policy_sha256,
                    worker_resources_sha256=(task.worker_resources_sha256),
                )
            )
            for task in tasks
        )
        or len({task.task_identity_sha256 for task in tasks}) != len(tasks)
    ):
        raise Active8MapReduceError("active-8 map/reduce task census drifted")
    return tasks


def verified_completed_task_identities(
    plan: Mapping[str, object],
    *,
    output_root: Path,
) -> frozenset[str]:
    """Return only byte-verified receipts; malformed existing receipts fail."""

    tasks = _assert_plan_identity(plan)
    output_root = Path(output_root)
    completed: set[str] = set()
    for task in tasks:
        receipt = _receipt_path(output_root, task)
        if not receipt.exists():
            continue
        _load_receipt(
            receipt,
            task=task,
            output_root=output_root,
        )
        completed.add(task.task_identity_sha256)
    return frozenset(completed)


def _source_task_groups(
    plan: Mapping[str, object],
    tasks: tuple[Active8MapTask, ...],
) -> tuple[tuple[Mapping[str, object], tuple[Active8MapTask, ...]], ...]:
    groups: list[tuple[Mapping[str, object], tuple[Active8MapTask, ...]]] = []
    cursor = 0
    for source_binding, partition in zip(
        plan["source_bindings"],
        plan["source_range_partitions"],
        strict=True,
    ):
        ranges = partition["ranges"]
        stop = cursor + len(ranges)
        group = tasks[cursor:stop]
        if (
            len(group) != len(ranges)
            or [task.source_binding for task in group] != [source_binding] * len(ranges)
            or [task.range_binding for task in group] != ranges
        ):
            raise Active8MapReduceError("map tasks do not form exact ordered source-range groups")
        groups.append((source_binding, group))
        cursor = stop
    if cursor != len(tasks):
        raise Active8MapReduceError("map task set contains extra source ranges")
    return tuple(groups)


def _publish_logical_source_decisions(
    *,
    tasks: tuple[Active8MapTask, ...],
    receipts: tuple[Mapping[str, object], ...],
    output_root: Path,
) -> tuple[str, str, dict[str, object]]:
    if not tasks or len(tasks) != len(receipts):
        raise Active8MapReduceError("logical source reduction requires every ordered range receipt")
    source_digest = tasks[0].packed_shard_content_sha256
    source_entries = tasks[0].packed_manifest_entries
    scratch_root = _run_root(output_root, tasks[0].run_identity_sha256) / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    seen_trace_ids: set[str] = set()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=scratch_root,
            prefix=f".source-{source_digest[:16]}.",
            suffix=".jsonl.gz.tmp",
            delete=False,
        ) as raw:
            temporary_name = raw.name
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed:
                for task, receipt in zip(tasks, receipts, strict=True):
                    object_path = output_root / str(receipt["decision_object"])
                    for record in _iter_decision_records(
                        object_path,
                        source_digest=source_digest,
                        entry_start=task.entry_start,
                        entry_stop=task.entry_stop,
                    ):
                        trace_id = str(record["trace_key"]["trace_id"])
                        if trace_id in seen_trace_ids:
                            raise Active8MapReduceError(
                                "range objects repeat a source trace identity"
                            )
                        seen_trace_ids.add(trace_id)
                        compressed.write(_canonical_json_bytes(record) + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        temporary = Path(temporary_name)
        object_sha256 = _sha256_file(temporary)
        object_relative = (
            Path("objects") / "decisions" / object_sha256[:2] / f"{object_sha256}.jsonl.gz"
        )
        _publish_file_immutable(
            temporary,
            output_root / object_relative,
            expected_sha256=object_sha256,
        )
        census = _census_decision_object(
            output_root / object_relative,
            source_digest=source_digest,
            entry_start=0,
            entry_stop=source_entries,
        )
        return object_relative.as_posix(), object_sha256, census
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def reduce_active8_mapreduce(
    plan: Mapping[str, object],
    *,
    output_root: Path,
) -> dict[str, object]:
    """Publish a unified inventory only after every exact map receipt exists."""

    tasks = _assert_plan_identity(plan)
    output_root = Path(output_root)
    run_root = _run_root(output_root, str(plan["run_identity_sha256"]))
    receipts_dir = run_root / "receipts"
    expected_receipt_names = {f"{task.task_identity_sha256}.json" for task in tasks}
    observed_receipt_names = (
        {path.name for path in receipts_dir.glob("*.json")} if receipts_dir.is_dir() else set()
    )
    missing = sorted(expected_receipt_names - observed_receipt_names)
    unexpected = sorted(observed_receipt_names - expected_receipt_names)
    if missing:
        raise Active8MapReduceIncomplete(
            f"{len(missing)} expected source-range decision receipt(s) are absent"
        )
    if unexpected:
        raise Active8MapReduceError("run receipt namespace contains unexpected task identities")

    totals = Counter(
        source_shards=0,
        traces=0,
        accepted_traces=0,
        excluded_traces=0,
        accepted_progress_rows=0,
        accepted_nonterminal_rows=0,
        accepted_terminal_rows=0,
    )
    exclusions_by_reason: Counter[str] = Counter()
    family_rows: Counter[str] = Counter({family: 0 for family in ACTIVE8_FAMILIES})
    shard_reports: list[dict[str, object]] = []
    for source_binding, source_tasks in _source_task_groups(
        plan,
        tasks,
    ):
        source_counts: Counter[str] = Counter()
        source_exclusions: Counter[str] = Counter()
        source_family_rows: Counter[str] = Counter({family: 0 for family in ACTIVE8_FAMILIES})
        source_receipts: list[Mapping[str, object]] = []
        for task in source_tasks:
            receipt = _load_receipt(
                _receipt_path(output_root, task),
                task=task,
                output_root=output_root,
            )
            object_path = output_root / str(receipt["decision_object"])
            census = _census_decision_object(
                object_path,
                source_digest=task.packed_shard_content_sha256,
                entry_start=task.entry_start,
                entry_stop=task.entry_stop,
            )
            if any(
                census[field] != receipt[field]
                for field in (
                    "counts",
                    "exclusions_by_reason",
                    "accepted_nonterminal_rows_by_family",
                )
            ):
                raise Active8MapReduceError(
                    "map receipt census disagrees with its range decision object"
                )
            source_counts.update(receipt["counts"])
            source_exclusions.update(receipt["exclusions_by_reason"])
            source_family_rows.update(receipt["accepted_nonterminal_rows_by_family"])
            source_receipts.append(receipt)
        (
            _logical_object,
            logical_object_sha256,
            logical_census,
        ) = _publish_logical_source_decisions(
            tasks=source_tasks,
            receipts=tuple(source_receipts),
            output_root=output_root,
        )
        expected_source_census = {
            "counts": dict(source_counts),
            "exclusions_by_reason": dict(sorted(source_exclusions.items())),
            "accepted_nonterminal_rows_by_family": {
                family: source_family_rows[family] for family in ACTIVE8_FAMILIES
            },
        }
        if logical_census != expected_source_census:
            raise Active8MapReduceError(
                "concatenated source decision census disagrees with its ranges"
            )
        totals["source_shards"] += 1
        totals.update(source_counts)
        exclusions_by_reason.update(source_exclusions)
        family_rows.update(source_family_rows)
        range_task_ids = [task.task_identity_sha256 for task in source_tasks]
        range_receipt_ids = [str(receipt["receipt_sha256"]) for receipt in source_receipts]
        shard_reports.append(
            {
                **source_binding,
                "inventory_shard": (
                    Path("..")
                    / "decisions"
                    / logical_object_sha256[:2]
                    / f"{logical_object_sha256}.jsonl.gz"
                ).as_posix(),
                "inventory_shard_sha256": logical_object_sha256,
                "counts": expected_source_census["counts"],
                "exclusions_by_reason": expected_source_census["exclusions_by_reason"],
                "accepted_nonterminal_rows_by_family": (
                    expected_source_census["accepted_nonterminal_rows_by_family"]
                ),
                "map_range_count": len(source_tasks),
                "map_range_task_identity_sha256s": range_task_ids,
                "map_range_receipt_sha256s": range_receipt_ids,
                "map_task_set_sha256": _sha256_payload(range_task_ids),
                "map_receipt_set_sha256": _sha256_payload(range_receipt_ids),
            }
        )
    if totals["traces"] != totals["accepted_traces"] + totals["excluded_traces"]:
        raise Active8MapReduceError("reduced trace census is not exhaustive")
    if totals["accepted_progress_rows"] != (
        totals["accepted_nonterminal_rows"] + totals["accepted_terminal_rows"]
    ):
        raise Active8MapReduceError("reduced progress-row census is inconsistent")

    source_identity = {
        "unified_packed_manifest_sha256": plan["source_manifest_sha256"],
        "unified_packed_manifest_semantic_sha256": plan["source_manifest_semantic_sha256"],
        "support_contract_sha256": plan["support_contract_sha256"],
        "physical_shard_binding_sha256": _sha256_payload(plan["source_bindings"]),
    }
    source_identity["effective_source_corpus_cache_sha256"] = _sha256_payload(source_identity)
    inventory: dict[str, object] = {
        "schema": ACTIVE8_TRACE_INVENTORY_SCHEMA,
        "schema_version": ACTIVE8_TRACE_INVENTORY_SCHEMA_VERSION,
        "status": "IMMUTABLE_WHOLE_TRACE_ACTIVE8_BOUNDARY",
        "training_authorized": False,
        "selection_policy": {
            "unit": "complete_packed_trace",
            "active_families": list(ACTIVE8_FAMILIES),
            "all_nonterminal_teachers_must_match_exact_candidates": True,
            "excluded_trace_emits_progress_rows": False,
            "accepted_trace_retains_terminal_row": True,
            "multi_neighbor_atom_insert_supported": False,
        },
        "persistent_state_identity": {
            "schema": PERSISTENT_STATE_DIGEST_SCHEMA,
            "schema_version": PERSISTENT_STATE_DIGEST_VERSION,
        },
        "source_identity": source_identity,
        "implementation_identity": plan["active8_implementation_identity"],
        "mapreduce_identity": {
            "run_identity_sha256": plan["run_identity_sha256"],
            "implementation_identity": plan["mapreduce_implementation_identity"],
            "range_policy": plan["range_policy"],
            "worker_resources": plan["worker_resources"],
            "worker_resources_sha256": plan["worker_resources_sha256"],
        },
        "counts": dict(totals),
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
        "accepted_nonterminal_rows_by_family": {
            family: family_rows[family] for family in ACTIVE8_FAMILIES
        },
        "shards": shard_reports,
    }
    inventory["inventory_sha256"] = _sha256_payload(inventory)
    inventory_bytes = (
        json.dumps(
            inventory,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    inventory_object_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    inventory_relative = Path("objects") / "manifests" / f"{inventory_object_sha256}.json"
    inventory_reused = _publish_bytes_immutable(
        output_root / inventory_relative,
        inventory_bytes,
    )
    complete: dict[str, object] = {
        "schema": ACTIVE8_MAPREDUCE_COMPLETE_SCHEMA,
        "schema_version": ACTIVE8_MAPREDUCE_COMPLETE_SCHEMA_VERSION,
        "status": "COMPLETE",
        "training_authorized": False,
        "run_identity_sha256": plan["run_identity_sha256"],
        "expected_source_decisions": len(plan["source_bindings"]),
        "expected_map_tasks": len(tasks),
        "inventory_manifest": inventory_relative.as_posix(),
        "inventory_manifest_file_sha256": inventory_object_sha256,
        "inventory_sha256": inventory["inventory_sha256"],
        "effective_source_corpus_cache_sha256": source_identity[
            "effective_source_corpus_cache_sha256"
        ],
        "counts": dict(totals),
        "exclusions_by_reason": dict(sorted(exclusions_by_reason.items())),
        "accepted_nonterminal_rows_by_family": {
            family: family_rows[family] for family in ACTIVE8_FAMILIES
        },
    }
    complete["complete_sha256"] = _sha256_payload(complete)
    complete_reused = _publish_bytes_immutable(
        run_root / "COMPLETE.json",
        json.dumps(
            complete,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )
    return {
        **complete,
        "reused": inventory_reused and complete_reused,
    }


__all__ = [
    "ACTIVE8_MAPREDUCE_COMPLETE_SCHEMA",
    "ACTIVE8_MAPREDUCE_PLAN_SCHEMA",
    "ACTIVE8_MAP_RECEIPT_SCHEMA",
    "Active8MapReduceError",
    "Active8MapReduceIncomplete",
    "Active8MapTask",
    "map_active8_source_decisions",
    "plan_active8_mapreduce",
    "reduce_active8_mapreduce",
    "verified_completed_task_identities",
]
