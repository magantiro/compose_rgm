"""Validate semantic Editing-V2 migration outputs as pre-Active8 sources.

The semantic migration reducer publishes one content-addressed task for every
five-lane by four-role cell.  This module is the read-only boundary between
that completed migration and a future ranged Active8 map/reduce.  It validates
the exact published plan, completion, task receipts, and semantic packed bytes,
then returns an immutable typed inventory.

The inventory records candidate-to-migration-to-semantic count flow.  It does
not enumerate an Active8 fiber, make an admission decision, declare a threshold,
or grant Gate 0 or training authority.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
)
from compose_v4.data.semantic_packed_trace_store import (
    COMPLETION_FILENAME as SEMANTIC_COMPLETION_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    MANIFEST_FILENAME as SEMANTIC_MANIFEST_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    SHARD_FILENAME as SEMANTIC_SHARD_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    SemanticPackedStoreError,
    load_semantic_packed_manifest,
)
from compose_v4.data.semantic_trace_migration_mapreduce import (
    COMPLETION_FILENAME,
    COMPLETION_SCHEMA,
    COMPLETION_SCHEMA_VERSION,
    COMPLETION_STATUS,
    EXPECTED_TASK_COUNT,
    PLAN_FILENAME,
    SemanticTraceMigrationMapReduceError,
    load_semantic_trace_migration_plan,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    RECEIPT_FILENAME,
    SEMANTIC_ARTIFACT_DIRNAME,
    SemanticTraceMaterializationError,
    validate_semantic_trace_materialization,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    EditingV2ProcessIdentityError,
    editing_v2_process_identity,
)

SOURCE_INVENTORY_STATUS = "COMPLETE_SEMANTIC_SOURCES_ACTIVE8_NOT_RUN_NO_TRAINING_AUTHORITY"
ACTIVE8_ADMISSION_STATUS = "NOT_RUN"

_SHA256_HEX = frozenset("0123456789abcdef")
_COMPLETION_FIELDS = {
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
_RESULT_FIELDS = {
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
_COUNT_FIELDS = {"source", "admitted", "rejected"}
_RUN_ROOT_INVENTORY = {PLAN_FILENAME, COMPLETION_FILENAME, "tasks"}
_TASK_INVENTORY = {"decisions.jsonl.gz", RECEIPT_FILENAME, SEMANTIC_ARTIFACT_DIRNAME}
_SEMANTIC_INVENTORY = {
    SEMANTIC_SHARD_FILENAME,
    SEMANTIC_MANIFEST_FILENAME,
    SEMANTIC_COMPLETION_FILENAME,
}


class EditingV2SemanticActive8SourceError(RuntimeError):
    """The completed semantic migration is not an exact Active8 source set."""


@dataclass(frozen=True, slots=True)
class SemanticActive8CountFlow:
    """Candidate, migration-decision, and semantic-row counts for one scope."""

    candidate_traces: int
    migration_source_traces: int
    migration_admitted_traces: int
    migration_rejected_traces: int
    semantic_entries: int
    semantic_states: int
    semantic_actions: int


@dataclass(frozen=True, slots=True)
class SemanticActive8SourceBinding:
    """Immutable source lineage required by the semantic ranged reader."""

    source_shard_name: str
    source_shard_sha256: str
    source_manifest_sha256: str
    source_overlay_sha256: str | None
    source_unified_manifest_sha256: str
    source_entry_count: int

    def as_mapping(self) -> dict[str, object]:
        """Return a fresh reader-compatible mapping without exposing state."""

        return {
            "source_shard_name": self.source_shard_name,
            "source_shard_sha256": self.source_shard_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_overlay_sha256": self.source_overlay_sha256,
            "source_unified_manifest_sha256": self.source_unified_manifest_sha256,
            "source_entry_count": self.source_entry_count,
        }


@dataclass(frozen=True, slots=True)
class SemanticActive8Source:
    """One exact lane/role semantic packed artifact awaiting Active8."""

    data_lane: str
    partition_role: str
    task_identity_sha256: str
    task_output_artifact_path: str
    task_output_directory: Path
    task_receipt_path: Path
    task_receipt_file_sha256: str
    task_receipt_sha256: str
    semantic_artifact_directory: Path
    semantic_shard_path: Path
    semantic_shard_sha256: str
    semantic_manifest_path: Path
    semantic_manifest_file_sha256: str
    semantic_manifest_sha256: str
    semantic_completion_path: Path
    semantic_completion_file_sha256: str
    semantic_completion_sha256: str
    semantic_record_stream_sha256: str
    process_identity_sha256: str
    action_codec_schema_version: int
    action_codec_implementation_hash: str
    source_binding: SemanticActive8SourceBinding
    family_histogram: tuple[tuple[str, int], ...]
    counts: SemanticActive8CountFlow

    @property
    def entry_count(self) -> int:
        """Return the exact range upper bound for later ranged Active8 work."""

        return self.counts.semantic_entries


@dataclass(frozen=True, slots=True)
class EditingV2SemanticActive8SourceInventory:
    """Exact immutable 20-cell source inventory with no admission authority."""

    status: str
    training_authorized: bool
    active8_admission_status: str
    migration_completion_path: Path
    migration_completion_file_sha256: str
    migration_completion_sha256: str
    migration_plan_path: Path
    migration_plan_file_sha256: str
    migration_plan_sha256: str
    run_identity_sha256: str
    source_revision_sha256: str
    source_inventory_sha256: str
    task_inventory_sha256: str
    result_inventory_sha256: str
    process_identity_sha256: str
    builder_identity_sha256: str
    candidate_materialization_manifest_sha256: str
    candidate_provenance_source_stream_sha256: str
    split_assignment_sha256: str
    lane_registry_sha256: str
    membership_receipt_file_sha256: str
    membership_receipt_sha256: str
    family_histogram: tuple[tuple[str, int], ...]
    rejection_histogram: tuple[tuple[str, int], ...]
    counts: SemanticActive8CountFlow
    sources: tuple[SemanticActive8Source, ...]


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _SHA256_HEX for character in digest):
        raise EditingV2SemanticActive8SourceError(f"{field} must be a full lowercase SHA-256")
    return digest


def _require_count(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise EditingV2SemanticActive8SourceError(f"{field} must be a nonnegative integer")
    return value


def _require_counter(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise EditingV2SemanticActive8SourceError(f"{field} must be an object")
    counter: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key or key.strip() != key:
            raise EditingV2SemanticActive8SourceError(
                f"{field} keys must be nonempty normalized strings"
            )
        counter[key] = _require_count(count, field=f"{field}.{key}")
    if list(value) != sorted(value):
        raise EditingV2SemanticActive8SourceError(
            f"{field} must use deterministic sorted key order"
        )
    return counter


def _require_counts(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _COUNT_FIELDS:
        raise EditingV2SemanticActive8SourceError(
            f"{field} must contain exactly source, admitted, and rejected"
        )
    counts = {
        key: _require_count(value[key], field=f"{field}.{key}")
        for key in ("source", "admitted", "rejected")
    }
    if counts["source"] != counts["admitted"] + counts["rejected"]:
        raise EditingV2SemanticActive8SourceError(
            f"{field} does not account for every source trace"
        )
    return counts


def _typed_source_binding(value: object, *, field: str) -> SemanticActive8SourceBinding:
    expected = {
        "source_shard_name",
        "source_shard_sha256",
        "source_manifest_sha256",
        "source_overlay_sha256",
        "source_unified_manifest_sha256",
        "source_entry_count",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise EditingV2SemanticActive8SourceError(f"{field} fields disagree")
    name = value["source_shard_name"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise EditingV2SemanticActive8SourceError(
            f"{field}.source_shard_name must be one nonempty basename"
        )
    overlay = value["source_overlay_sha256"]
    if overlay is not None:
        overlay = _require_sha256(overlay, field=f"{field}.source_overlay_sha256")
    return SemanticActive8SourceBinding(
        source_shard_name=name,
        source_shard_sha256=_require_sha256(
            value["source_shard_sha256"], field=f"{field}.source_shard_sha256"
        ),
        source_manifest_sha256=_require_sha256(
            value["source_manifest_sha256"],
            field=f"{field}.source_manifest_sha256",
        ),
        source_overlay_sha256=overlay,
        source_unified_manifest_sha256=_require_sha256(
            value["source_unified_manifest_sha256"],
            field=f"{field}.source_unified_manifest_sha256",
        ),
        source_entry_count=_require_count(
            value["source_entry_count"], field=f"{field}.source_entry_count"
        ),
    )


def _load_json_object(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2SemanticActive8SourceError(f"{field} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise EditingV2SemanticActive8SourceError(f"{field} must be an object")
    if raw != _canonical_json_bytes(value, newline=True):
        raise EditingV2SemanticActive8SourceError(f"{field} bytes are not canonical JSON")
    return value, raw


def _require_artifact_path(value: object, *, field: str) -> str:
    raw = value if isinstance(value, str) else ""
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or not path.is_absolute()
        or len(path.parts) < 3
        or path.parts[1] != "artifacts"
        or ".." in path.parts
        or str(path) != raw
        or raw.endswith("/")
    ):
        raise EditingV2SemanticActive8SourceError(
            f"{field} must be a normalized path below /artifacts"
        )
    return raw


def _mounted_artifact_path(
    artifact_path: object,
    *,
    artifact_root: Path,
    field: str,
) -> Path:
    normalized = _require_artifact_path(artifact_path, field=field)
    root = Path(artifact_root).resolve()
    resolved = (root / PurePosixPath(normalized).relative_to("/artifacts")).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EditingV2SemanticActive8SourceError(
            f"{field} resolves outside the artifact root"
        ) from error
    return resolved


def _require_exact_directory(path: Path, expected: set[str], *, field: str) -> None:
    if not path.is_dir():
        raise EditingV2SemanticActive8SourceError(f"{field} is absent: {path}")
    observed = {child.name for child in path.iterdir()}
    if observed != expected:
        raise EditingV2SemanticActive8SourceError(
            f"{field} inventory disagrees; "
            f"missing={sorted(expected - observed)}, extras={sorted(observed - expected)}"
        )


def _load_completion(path: Path) -> tuple[dict[str, Any], bytes]:
    completion, raw = _load_json_object(path, field="semantic migration completion")
    if set(completion) != _COMPLETION_FIELDS:
        raise EditingV2SemanticActive8SourceError("semantic migration completion fields disagree")
    body = {key: item for key, item in completion.items() if key != "completion_sha256"}
    if (
        completion["schema"] != COMPLETION_SCHEMA
        or completion["schema_version"] != COMPLETION_SCHEMA_VERSION
        or completion["status"] != COMPLETION_STATUS
        or completion["training_authorized"] is not False
        or completion["gate_zero_run"] is not False
        or completion["completion_sha256"] != _canonical_sha256(body)
    ):
        raise EditingV2SemanticActive8SourceError(
            "semantic migration completion identity or authority disagrees"
        )
    return completion, raw


def _add_counter(target: Counter[str], value: Mapping[str, int]) -> None:
    for key, count in value.items():
        target[key] += count


def resolve_editing_v2_semantic_active8_sources(
    completion_path: Path,
    *,
    artifact_root: Path,
    repo_root: Path,
) -> EditingV2SemanticActive8SourceInventory:
    """Validate one exact migration completion and return its 20 sources.

    ``completion_path`` is supplied by the caller.  No run identity is embedded
    in this adapter; the supplied completion must instead bind its colocated
    plan and every published task result.
    """

    root = Path(artifact_root).resolve()
    supplied_completion = Path(completion_path)
    if supplied_completion.name != COMPLETION_FILENAME:
        raise EditingV2SemanticActive8SourceError(
            f"completion_path must name {COMPLETION_FILENAME}"
        )
    resolved_completion = supplied_completion.resolve()
    try:
        resolved_completion.relative_to(root)
    except ValueError as error:
        raise EditingV2SemanticActive8SourceError(
            "completion_path resolves outside the artifact root"
        ) from error
    run_root = resolved_completion.parent
    _require_exact_directory(
        run_root,
        _RUN_ROOT_INVENTORY,
        field="semantic migration run root",
    )
    completion, completion_bytes = _load_completion(resolved_completion)

    plan_path = run_root / PLAN_FILENAME
    try:
        plan = load_semantic_trace_migration_plan(plan_path, repo_root=Path(repo_root))
    except SemanticTraceMigrationMapReduceError as error:
        raise EditingV2SemanticActive8SourceError(
            "semantic migration plan is stale or invalid"
        ) from error
    plan_bytes = plan_path.read_bytes()
    if plan_bytes != _canonical_json_bytes(plan, newline=True):
        raise EditingV2SemanticActive8SourceError(
            "semantic migration plan bytes are not canonical JSON"
        )
    plan_file_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    expected_run_root = _mounted_artifact_path(
        plan["run_artifact_root"],
        artifact_root=root,
        field="plan.run_artifact_root",
    )
    if expected_run_root != run_root:
        raise EditingV2SemanticActive8SourceError(
            "supplied completion path does not belong to its frozen plan"
        )
    try:
        live_process_identity = editing_v2_process_identity()
    except EditingV2ProcessIdentityError as error:
        raise EditingV2SemanticActive8SourceError(
            "live Editing-V2 ActionCodecV4 process identity is invalid"
        ) from error
    for field in (
        "run_identity_sha256",
        "plan_sha256",
        "source_revision_sha256",
        "source_inventory_sha256",
        "process_identity_sha256",
        "builder_identity_sha256",
        "task_inventory_sha256",
        "result_inventory_sha256",
        "completion_sha256",
    ):
        _require_sha256(completion[field], field=f"completion.{field}")
    if (
        completion["run_identity_sha256"] != plan["run_identity_sha256"]
        or completion["plan_sha256"] != plan["plan_sha256"]
        or completion["plan_file_sha256"] != plan_file_sha256
        or completion["source_revision_sha256"] != plan["source_revision"]["source_revision_sha256"]
        or completion["source_inventory_sha256"] != plan["source_inventory_sha256"]
        or completion["process_identity_sha256"] != live_process_identity["process_identity_sha256"]
        or completion["process_identity_sha256"]
        != plan["process_identity"]["process_identity_sha256"]
        or completion["builder_identity_sha256"] != plan["builder_identity"]["identity_sha256"]
        or completion["task_inventory_sha256"] != plan["task_inventory_sha256"]
    ):
        raise EditingV2SemanticActive8SourceError(
            "completion, plan, process, builder, or source identity disagrees"
        )

    results = completion["result_inventory"]
    if not isinstance(results, list):
        raise EditingV2SemanticActive8SourceError("completion.result_inventory must be a list")
    if completion["result_inventory_sha256"] != _canonical_sha256(results):
        raise EditingV2SemanticActive8SourceError("completion result inventory SHA-256 disagrees")
    if (
        completion["task_count"] != EXPECTED_TASK_COUNT
        or len(results) != EXPECTED_TASK_COUNT
        or len(plan["tasks"]) != EXPECTED_TASK_COUNT
    ):
        raise EditingV2SemanticActive8SourceError(
            "semantic Active8 source adapter requires exactly 20 task results"
        )

    task_root = run_root / "tasks"
    expected_task_ids = [task["task_identity_sha256"] for task in plan["tasks"]]
    if len(set(expected_task_ids)) != EXPECTED_TASK_COUNT:
        raise EditingV2SemanticActive8SourceError(
            "migration plan contains duplicate task identities"
        )
    _require_exact_directory(
        task_root,
        set(expected_task_ids),
        field="semantic migration task root",
    )

    expected_cells = tuple(
        (lane, role) for lane in REQUIRED_DATA_LANES for role in REQUIRED_PARTITION_ROLES
    )
    observed_cells: list[tuple[str, str]] = []
    observed_task_ids: set[str] = set()
    observed_output_paths: set[str] = set()
    total_counts: Counter[str] = Counter()
    total_rejections: Counter[str] = Counter()
    total_families: Counter[str] = Counter()
    total_states = 0
    total_actions = 0
    sources: list[SemanticActive8Source] = []

    for index, (task, result) in enumerate(zip(plan["tasks"], results, strict=True)):
        if not isinstance(result, Mapping) or set(result) != _RESULT_FIELDS:
            raise EditingV2SemanticActive8SourceError(f"completion result {index} fields disagree")
        result = dict(result)
        task_id = _require_sha256(
            result["task_identity_sha256"],
            field=f"result_inventory[{index}].task_identity_sha256",
        )
        output_artifact_path = _require_artifact_path(
            result["output_artifact_path"],
            field=f"result_inventory[{index}].output_artifact_path",
        )
        cell = (result["data_lane"], result["split"])
        if (
            task_id != task["task_identity_sha256"]
            or cell != (task["data_lane"], task["split"])
            or result["source_artifact_path"] != task["source_artifact_path"]
            or output_artifact_path != task["output_artifact_path"]
        ):
            raise EditingV2SemanticActive8SourceError(
                f"completion result {index} does not match its planned task"
            )
        if task_id in observed_task_ids or output_artifact_path in observed_output_paths:
            raise EditingV2SemanticActive8SourceError(
                "completion results contain duplicate task or output identities"
            )
        observed_task_ids.add(task_id)
        observed_output_paths.add(output_artifact_path)
        observed_cells.append(cell)

        output = _mounted_artifact_path(
            output_artifact_path,
            artifact_root=root,
            field=f"result_inventory[{index}].output_artifact_path",
        )
        if output.parent != task_root or output.name != task_id:
            raise EditingV2SemanticActive8SourceError(
                f"task result {task_id} is not at its exact task namespace path"
            )
        _require_exact_directory(
            output,
            _TASK_INVENTORY,
            field=f"semantic migration task {task_id}",
        )
        try:
            receipt = validate_semantic_trace_materialization(output)
        except (SemanticTraceMaterializationError, SemanticPackedStoreError) as error:
            raise EditingV2SemanticActive8SourceError(
                f"semantic migration task receipt is invalid: {task_id}"
            ) from error

        receipt_path = output / RECEIPT_FILENAME
        receipt_payload, receipt_bytes = _load_json_object(
            receipt_path,
            field=f"task receipt {task_id}",
        )
        if receipt_payload != receipt:
            raise EditingV2SemanticActive8SourceError(
                f"task receipt {task_id} changed during validation"
            )
        result_counts = _require_counts(
            result["counts"],
            field=f"result_inventory[{index}].counts",
        )
        receipt_counts = _require_counts(
            receipt["counts"],
            field=f"task receipt {task_id}.counts",
        )
        candidate_count = _require_count(
            task["source_entry_count"],
            field=f"task {task_id}.source_entry_count",
        )
        source_binding = _typed_source_binding(
            receipt["source_binding"],
            field=f"task receipt {task_id}.source_binding",
        )
        if (
            result["receipt_sha256"] != receipt["receipt_sha256"]
            or result["semantic_shard_sha256"] != receipt["semantic_shard_sha256"]
            or result["semantic_manifest_sha256"] != receipt["semantic_manifest_sha256"]
            or result_counts != receipt_counts
            or candidate_count != receipt_counts["source"]
            or source_binding.source_entry_count != candidate_count
        ):
            raise EditingV2SemanticActive8SourceError(
                f"task result, receipt, and candidate census disagree: {task_id}"
            )

        semantic_dir = output / SEMANTIC_ARTIFACT_DIRNAME
        _require_exact_directory(
            semantic_dir,
            _SEMANTIC_INVENTORY,
            field=f"semantic packed artifact {task_id}",
        )
        try:
            manifest = load_semantic_packed_manifest(
                semantic_dir,
                expected_shard_sha256=receipt["semantic_shard_sha256"],
                expected_manifest_sha256=receipt["semantic_manifest_sha256"],
                expected_source_binding=receipt["source_binding"],
            )
        except SemanticPackedStoreError as error:
            raise EditingV2SemanticActive8SourceError(
                f"semantic packed artifact is invalid: {task_id}"
            ) from error
        semantic_completion_path = semantic_dir / SEMANTIC_COMPLETION_FILENAME
        semantic_completion, semantic_completion_bytes = _load_json_object(
            semantic_completion_path,
            field=f"semantic completion {task_id}",
        )
        semantic_entries = _require_count(
            manifest["entries"], field=f"semantic manifest {task_id}.entries"
        )
        semantic_states = _require_count(
            manifest["states"], field=f"semantic manifest {task_id}.states"
        )
        semantic_actions = _require_count(
            manifest["actions"], field=f"semantic manifest {task_id}.actions"
        )
        family_histogram = _require_counter(
            manifest["family_histogram"],
            field=f"semantic manifest {task_id}.family_histogram",
        )
        rejection_histogram = _require_counter(
            receipt["rejections_by_code"],
            field=f"task receipt {task_id}.rejections_by_code",
        )
        receipt_families = _require_counter(
            receipt["accepted_family_histogram"],
            field=f"task receipt {task_id}.accepted_family_histogram",
        )
        if (
            manifest["data_lane"] != cell[0]
            or manifest["split"] != cell[1]
            or manifest["process_identity_sha256"]
            != live_process_identity["process_identity_sha256"]
            or manifest["action_codec_schema_version"]
            != live_process_identity["action_codec_schema_version"]
            or manifest["action_codec_implementation_hash"]
            != live_process_identity["action_codec_implementation_hash"]
            or receipt["process_identity"] != live_process_identity
            or semantic_entries != receipt_counts["admitted"]
            or family_histogram != receipt_families
        ):
            raise EditingV2SemanticActive8SourceError(
                f"semantic artifact identity or census disagrees: {task_id}"
            )

        _add_counter(total_counts, receipt_counts)
        _add_counter(total_rejections, rejection_histogram)
        _add_counter(total_families, family_histogram)
        total_states += semantic_states
        total_actions += semantic_actions
        semantic_manifest_path = semantic_dir / SEMANTIC_MANIFEST_FILENAME
        sources.append(
            SemanticActive8Source(
                data_lane=cell[0],
                partition_role=cell[1],
                task_identity_sha256=task_id,
                task_output_artifact_path=output_artifact_path,
                task_output_directory=output,
                task_receipt_path=receipt_path,
                task_receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                task_receipt_sha256=receipt["receipt_sha256"],
                semantic_artifact_directory=semantic_dir,
                semantic_shard_path=semantic_dir / SEMANTIC_SHARD_FILENAME,
                semantic_shard_sha256=receipt["semantic_shard_sha256"],
                semantic_manifest_path=semantic_manifest_path,
                semantic_manifest_file_sha256=_file_sha256(semantic_manifest_path),
                semantic_manifest_sha256=manifest["manifest_sha256"],
                semantic_completion_path=semantic_completion_path,
                semantic_completion_file_sha256=hashlib.sha256(
                    semantic_completion_bytes
                ).hexdigest(),
                semantic_completion_sha256=semantic_completion["completion_sha256"],
                semantic_record_stream_sha256=_require_sha256(
                    manifest["record_stream_sha256"],
                    field=f"semantic manifest {task_id}.record_stream_sha256",
                ),
                process_identity_sha256=manifest["process_identity_sha256"],
                action_codec_schema_version=manifest["action_codec_schema_version"],
                action_codec_implementation_hash=manifest["action_codec_implementation_hash"],
                source_binding=source_binding,
                family_histogram=tuple(family_histogram.items()),
                counts=SemanticActive8CountFlow(
                    candidate_traces=candidate_count,
                    migration_source_traces=receipt_counts["source"],
                    migration_admitted_traces=receipt_counts["admitted"],
                    migration_rejected_traces=receipt_counts["rejected"],
                    semantic_entries=semantic_entries,
                    semantic_states=semantic_states,
                    semantic_actions=semantic_actions,
                ),
            )
        )

    if tuple(observed_cells) != expected_cells:
        raise EditingV2SemanticActive8SourceError(
            "semantic result inventory is not the exact ordered five-lane by four-role grid"
        )
    completion_counts = _require_counts(completion["counts"], field="completion.counts")
    completion_rejections = _require_counter(
        completion["rejections_by_code"], field="completion.rejections_by_code"
    )
    completion_families = _require_counter(
        completion["accepted_family_histogram"],
        field="completion.accepted_family_histogram",
    )
    candidate_count = _require_count(
        plan["expected_source_entry_count"],
        field="plan.expected_source_entry_count",
    )
    semantic_entries = sum(source.counts.semantic_entries for source in sources)
    if (
        candidate_count != total_counts["source"]
        or completion_counts != dict(total_counts)
        or completion_rejections != dict(sorted(total_rejections.items()))
        or completion_families != dict(sorted(total_families.items()))
        or semantic_entries != total_counts["admitted"]
    ):
        raise EditingV2SemanticActive8SourceError(
            "candidate, migration, completion, and semantic aggregate counts disagree"
        )

    upstream = plan["upstream"]
    source_stream = upstream["candidate_provenance_source_stream"]
    return EditingV2SemanticActive8SourceInventory(
        status=SOURCE_INVENTORY_STATUS,
        training_authorized=False,
        active8_admission_status=ACTIVE8_ADMISSION_STATUS,
        migration_completion_path=resolved_completion,
        migration_completion_file_sha256=hashlib.sha256(completion_bytes).hexdigest(),
        migration_completion_sha256=completion["completion_sha256"],
        migration_plan_path=plan_path,
        migration_plan_file_sha256=plan_file_sha256,
        migration_plan_sha256=plan["plan_sha256"],
        run_identity_sha256=plan["run_identity_sha256"],
        source_revision_sha256=plan["source_revision"]["source_revision_sha256"],
        source_inventory_sha256=plan["source_inventory_sha256"],
        task_inventory_sha256=plan["task_inventory_sha256"],
        result_inventory_sha256=completion["result_inventory_sha256"],
        process_identity_sha256=completion["process_identity_sha256"],
        builder_identity_sha256=completion["builder_identity_sha256"],
        candidate_materialization_manifest_sha256=upstream[
            "candidate_materialization_manifest_sha256"
        ],
        candidate_provenance_source_stream_sha256=source_stream["source_stream_sha256"],
        split_assignment_sha256=upstream["split_assignment_sha256"],
        lane_registry_sha256=upstream["lane_registry_sha256"],
        membership_receipt_file_sha256=upstream["membership_receipt_file_sha256"],
        membership_receipt_sha256=upstream["membership_receipt_sha256"],
        family_histogram=tuple(completion_families.items()),
        rejection_histogram=tuple(completion_rejections.items()),
        counts=SemanticActive8CountFlow(
            candidate_traces=candidate_count,
            migration_source_traces=completion_counts["source"],
            migration_admitted_traces=completion_counts["admitted"],
            migration_rejected_traces=completion_counts["rejected"],
            semantic_entries=semantic_entries,
            semantic_states=total_states,
            semantic_actions=total_actions,
        ),
        sources=tuple(sources),
    )


__all__ = [
    "ACTIVE8_ADMISSION_STATUS",
    "SOURCE_INVENTORY_STATUS",
    "EditingV2SemanticActive8SourceError",
    "EditingV2SemanticActive8SourceInventory",
    "SemanticActive8CountFlow",
    "SemanticActive8Source",
    "SemanticActive8SourceBinding",
    "resolve_editing_v2_semantic_active8_sources",
]
