"""Resumable map/reduce publication for the active-8 trace inventory.

The scientific admission predicate lives in :mod:`active8_trace_inventory`.
This module only supplies bounded-distribution mechanics:

* one deterministic decision object per exact packed source shard;
* content-addressed, immutable publication with byte-collision checks;
* immutable task receipts, so interrupted maps resume without recomputation;
* a reducer that refuses to publish until the expected receipt set is exact;
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
    manifest_path_for,
    read_addressed_packed_shard,
)
from compose_v4.data.provenance_overlay import overlay_path_for

ACTIVE8_MAPREDUCE_PLAN_SCHEMA = "compose.data.active8_inventory_mapreduce_plan"
ACTIVE8_MAPREDUCE_PLAN_SCHEMA_VERSION = 1
ACTIVE8_MAP_RECEIPT_SCHEMA = "compose.data.active8_inventory_map_receipt"
ACTIVE8_MAP_RECEIPT_SCHEMA_VERSION = 1
ACTIVE8_MAPREDUCE_COMPLETE_SCHEMA = "compose.data.active8_inventory_mapreduce_complete"
ACTIVE8_MAPREDUCE_COMPLETE_SCHEMA_VERSION = 1

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
    """Content-addressed request for one exact packed source shard."""

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

    def __post_init__(self) -> None:
        for name in (
            "run_identity_sha256",
            "task_identity_sha256",
            "packed_shard_content_sha256",
            "packed_manifest_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        overlay = self.packed_provenance_overlay_sha256
        if overlay is not None and not _is_sha256(overlay):
            raise ValueError("packed_provenance_overlay_sha256 is malformed")
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
            "packed_provenance_overlay_sha256": (
                self.packed_provenance_overlay_sha256
            ),
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


def _implementation_identity(*, repo_root: Path | None = None) -> dict[str, object]:
    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    sources: dict[str, str] = {}
    for relative in _MAPREDUCE_IMPLEMENTATION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise Active8MapReduceError(
                f"map/reduce implementation source is absent: {path}"
            )
        sources[relative] = _sha256_file(path)
    return {
        "sources": sources,
        "implementation_sha256": _sha256_payload(sources),
    }


def _task_identity_payload(
    *,
    run_identity_sha256: str,
    source_binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": ACTIVE8_MAP_RECEIPT_SCHEMA,
        "schema_version": ACTIVE8_MAP_RECEIPT_SCHEMA_VERSION,
        "run_identity_sha256": run_identity_sha256,
        "source_binding": dict(source_binding),
    }


def plan_active8_mapreduce(
    shards: Iterable[Active8SourceShard],
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, object],
    support_contract_sha256: str,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Freeze the exact expected source set and derive stable map identities."""

    if not _is_sha256(support_contract_sha256):
        raise ValueError("support_contract_sha256 must be a lowercase SHA-256")
    source_manifest_path = Path(source_manifest_path)
    if not source_manifest_path.is_file():
        raise Active8MapReduceError("unified packed manifest is absent")
    decoded = json.loads(source_manifest_path.read_text())
    if _sha256_payload(decoded) != _sha256_payload(source_manifest):
        raise Active8MapReduceError(
            "decoded unified manifest differs from the exact source file"
        )
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
            raise Active8MapReduceError(
                f"packed shard or its manifest is absent: {source}"
            )
        physical_digest = _sha256_file(source)
        if physical_digest in seen_physical_digests:
            raise Active8MapReduceError(
                "one physical packed source is declared more than once"
            )
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
                _sha256_file(packed_overlay)
                if packed_overlay.is_file()
                else None
            ),
        }
        preliminary_bindings.append(binding)
        runtime_paths.append(str(source))

    run_payload = {
        "schema": ACTIVE8_MAPREDUCE_PLAN_SCHEMA,
        "schema_version": ACTIVE8_MAPREDUCE_PLAN_SCHEMA_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_semantic_sha256": _sha256_payload(source_manifest),
        "support_contract_sha256": support_contract_sha256,
        "active8_families": list(ACTIVE8_FAMILIES),
        "active8_implementation_sha256": active8_identity[
            "implementation_sha256"
        ],
        "mapreduce_implementation_sha256": mapreduce_identity[
            "implementation_sha256"
        ],
        "source_bindings": preliminary_bindings,
    }
    run_identity_sha256 = _sha256_payload(run_payload)
    tasks: list[Active8MapTask] = []
    for binding, runtime_path in zip(
        preliminary_bindings,
        runtime_paths,
        strict=True,
    ):
        task_identity_sha256 = _sha256_payload(
            _task_identity_payload(
                run_identity_sha256=run_identity_sha256,
                source_binding=binding,
            )
        )
        tasks.append(
            Active8MapTask(
                run_identity_sha256=run_identity_sha256,
                task_identity_sha256=task_identity_sha256,
                source_path=runtime_path,
                **binding,
            )
        )
    return {
        **run_payload,
        "run_identity_sha256": run_identity_sha256,
        "expected_source_decisions": len(tasks),
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
    """Publish bytes exactly once; existing bytes must be identical."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_sha256 = hashlib.sha256(content).hexdigest()
    if path.exists():
        if _sha256_file(path) != expected_sha256 or path.read_bytes() != content:
            raise Active8MapReduceError(
                f"immutable artifact collision at {path}"
            )
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
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if _sha256_file(path) != expected_sha256 or path.read_bytes() != content:
                raise Active8MapReduceError(
                    f"immutable artifact collision at {path}"
                )
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
    """Atomically publish one potentially large object without loading it."""

    source = Path(source)
    destination = Path(destination)
    if _sha256_file(source) != expected_sha256:
        raise Active8MapReduceError("temporary artifact digest changed before publish")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise Active8MapReduceError(
                f"immutable content-object collision at {destination}"
            )
        return True
    try:
        os.link(source, destination)
    except FileExistsError:
        if _sha256_file(destination) != expected_sha256:
            raise Active8MapReduceError(
                f"immutable content-object collision at {destination}"
            )
        return True
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
    ):
        raise Active8MapReduceError("map receipt disagrees with its frozen task")
    if payload.get("receipt_sha256") != _receipt_self_hash(payload):
        raise Active8MapReduceError("map receipt self-hash mismatch")
    object_path = Path(output_root) / str(payload["decision_object"])
    if (
        not object_path.is_file()
        or _sha256_file(object_path) != payload["decision_object_sha256"]
    ):
        raise Active8MapReduceError(
            "map receipt decision object is absent or hash-mismatched"
        )
    return payload


def map_active8_source_decisions(
    raw_task: Mapping[str, object],
    *,
    exact_candidate_checker: ExactCandidateChecker,
    output_root: Path,
) -> dict[str, object]:
    """Map one source shard, reusing a verified immutable receipt on resume."""

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
    if (
        not source.is_file()
        or _sha256_file(source) != task.packed_shard_content_sha256
        or _sha256_file(manifest_path_for(source))
        != task.packed_manifest_sha256
    ):
        raise Active8MapReduceError(
            "map source bytes disagree with the frozen task"
        )
    overlay_path = overlay_path_for(source)
    if task.packed_provenance_overlay_sha256 is None:
        if overlay_path.exists():
            raise Active8MapReduceError(
                "an undeclared provenance overlay appeared after planning"
            )
    elif (
        not overlay_path.is_file()
        or _sha256_file(overlay_path)
        != task.packed_provenance_overlay_sha256
    ):
        raise Active8MapReduceError(
            "packed provenance overlay disagrees with the frozen task"
        )

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
                for addressed in read_addressed_packed_shard(
                    source,
                    verify_fraction=0.0,
                ):
                    address = addressed.address
                    if (
                        address.packed_shard_content_sha256
                        != task.packed_shard_content_sha256
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
        if observed_entries != list(range(len(observed_entries))):
            raise Active8MapReduceError(
                "packed source did not yield a complete ordered entry census"
            )
        temporary = Path(temporary_name)
        object_sha256 = _sha256_file(temporary)
        object_relative = (
            Path("objects")
            / "decisions"
            / object_sha256[:2]
            / f"{object_sha256}.jsonl.gz"
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


def _census_decision_object(
    object_path: Path,
    *,
    source_digest: str,
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
    last_entry = -1
    with gzip.open(object_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (
                record.get("schema") != ACTIVE8_TRACE_DECISION_SCHEMA
                or record.get("schema_version")
                != ACTIVE8_TRACE_DECISION_SCHEMA_VERSION
            ):
                raise Active8MapReduceError("decision object contains another schema")
            key = record.get("trace_key") or {}
            entry_index = key.get("entry_index")
            if (
                key.get("packed_shard_content_sha256") != source_digest
                or type(entry_index) is not int
                or entry_index != last_entry + 1
            ):
                raise Active8MapReduceError(
                    "decision object lost or reordered physical source entries"
                )
            last_entry = entry_index
            counts["traces"] += 1
            if record.get("decision") == "accepted":
                rows = record.get("progress_rows")
                path_length = record.get("path_length")
                if (
                    type(path_length) is not int
                    or not isinstance(rows, list)
                    or len(rows) != path_length + 1
                    or not rows[-1].get("is_terminal")
                    or record.get("exclusions") != []
                ):
                    raise Active8MapReduceError(
                        "accepted decision does not retain its complete path"
                    )
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
                    raise Active8MapReduceError(
                        "excluded decision leaked rows or lacks a reason"
                    )
                counts["excluded_traces"] += 1
                for exclusion in record["exclusions"]:
                    exclusions_by_reason[str(exclusion["reason"])] += 1
            else:
                raise Active8MapReduceError("decision object has an unknown verdict")
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
        )
    }
    if (
        plan.get("run_identity_sha256") != _sha256_payload(body)
        or plan.get("active8_families") != list(ACTIVE8_FAMILIES)
        or not isinstance(plan.get("active8_implementation_identity"), Mapping)
        or plan["active8_implementation_identity"].get(
            "implementation_sha256"
        )
        != plan.get("active8_implementation_sha256")
        or not isinstance(plan.get("mapreduce_implementation_identity"), Mapping)
        or plan["mapreduce_implementation_identity"].get(
            "implementation_sha256"
        )
        != plan.get("mapreduce_implementation_sha256")
    ):
        raise Active8MapReduceError("active-8 map/reduce plan identity mismatch")
    tasks = tuple(_task_from_mapping(raw) for raw in plan.get("tasks", ()))
    if (
        len(tasks) != plan.get("expected_source_decisions")
        or [task.source_binding for task in tasks] != plan.get("source_bindings")
        or any(
            task.run_identity_sha256 != plan["run_identity_sha256"]
            for task in tasks
        )
        or any(
            task.task_identity_sha256
            != _sha256_payload(
                _task_identity_payload(
                    run_identity_sha256=task.run_identity_sha256,
                    source_binding=task.source_binding,
                )
            )
            for task in tasks
        )
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
    expected_receipt_names = {
        f"{task.task_identity_sha256}.json" for task in tasks
    }
    observed_receipt_names = (
        {path.name for path in receipts_dir.glob("*.json")}
        if receipts_dir.is_dir()
        else set()
    )
    missing = sorted(expected_receipt_names - observed_receipt_names)
    unexpected = sorted(observed_receipt_names - expected_receipt_names)
    if missing:
        raise Active8MapReduceIncomplete(
            f"{len(missing)} expected source decision receipt(s) are absent"
        )
    if unexpected:
        raise Active8MapReduceError(
            "run receipt namespace contains unexpected task identities"
        )

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
    for task in tasks:
        receipt = _load_receipt(
            _receipt_path(output_root, task),
            task=task,
            output_root=output_root,
        )
        object_path = output_root / str(receipt["decision_object"])
        census = _census_decision_object(
            object_path,
            source_digest=task.packed_shard_content_sha256,
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
                "map receipt census disagrees with its decision object"
            )
        totals["source_shards"] += 1
        totals.update(receipt["counts"])
        exclusions_by_reason.update(receipt["exclusions_by_reason"])
        family_rows.update(receipt["accepted_nonterminal_rows_by_family"])
        shard_reports.append(
            {
                **task.source_binding,
                "inventory_shard": (
                    Path("..")
                    / "decisions"
                    / str(receipt["decision_object_sha256"])[:2]
                    / f"{receipt['decision_object_sha256']}.jsonl.gz"
                ).as_posix(),
                "inventory_shard_sha256": receipt["decision_object_sha256"],
                "counts": receipt["counts"],
                "exclusions_by_reason": receipt["exclusions_by_reason"],
                "accepted_nonterminal_rows_by_family": receipt[
                    "accepted_nonterminal_rows_by_family"
                ],
                "map_task_identity_sha256": task.task_identity_sha256,
                "map_receipt_sha256": receipt["receipt_sha256"],
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
        "unified_packed_manifest_semantic_sha256": plan[
            "source_manifest_semantic_sha256"
        ],
        "support_contract_sha256": plan["support_contract_sha256"],
        "physical_shard_binding_sha256": _sha256_payload(
            plan["source_bindings"]
        ),
    }
    source_identity["effective_source_corpus_cache_sha256"] = _sha256_payload(
        source_identity
    )
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
            "implementation_identity": plan[
                "mapreduce_implementation_identity"
            ],
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
    inventory_relative = (
        Path("objects")
        / "manifests"
        / f"{inventory_object_sha256}.json"
    )
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
        "expected_source_decisions": len(tasks),
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
