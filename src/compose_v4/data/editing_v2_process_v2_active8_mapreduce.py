"""Deterministic map/reduce for Process-V2 Active8 whole-trace decisions.

THE UNIT OF WORK IS ONE VERIFIED CACHE CHUNK
--------------------------------------------
The Process-V2 chunk cache already partitions the immutable payload into
content-addressed chunks, and the Process-V2 rebind already publishes exactly
one proof task per chunk.  Active8 adopts the same partition rather than
inventing a third one, so every task is a three-way join on ONE address: the
cache chunk it reads, the rebind task that decided the same entries, and the
Active8 result it publishes.  A task that had to re-partition would have to
re-derive the join, and a re-derived join is a place two partitions can disagree.

WHAT A TASK PROVES, PER ENTRY
-----------------------------
1. the row is the exact cached row, decoded through the production chunk reader
   with its persistent-slot states and its ActionV4 trace -- never rebuilt from
   a canonical SMILES key;
2. the row is joined to its rebind decision by
   ``(v1_task_identity_sha256, entry_index)``, the frozen join key, and never by
   chunk position;
3. an upstream-rejected entry is carried through under
   ``upstream_rebind_rejected`` and is NOT candidate-evaluated;
4. every other entry is evaluated whole through the frozen Process-V2 Active8
   policy against the unchanged production model and executor;
5. the four candidate counts -- raw marks, canonical successors, matching marks
   and successor aliases -- survive to the published row unaggregated.

THE CENSUS IS THE SEAM'S, EXACTLY
---------------------------------
``source_entries == upstream_rejected_entries + active8_accepted_entries +
active8_excluded_entries`` at the task level and again, summed, at the run
level.  The two rejection categories stay distinct all the way to the
completion: a trace the rebind refused was never a candidate, and merging it
with one Active8 evaluated and refused would report a corpus that was narrowed
for a reason it was not narrowed for.

WHAT IS PUBLISHED, AND WHEN
---------------------------
A task publishes by directory rename, so its two objects appear together or not
at all; a partially written task is a hidden staging sibling, which is not a
result and does not block the retry.  The reducer requires every planned task,
refuses any object the plan did not name, recomputes the whole census from the
decision rows rather than trusting a receipt, and only then writes the
completion.  An incomplete, duplicated, extra, mismatched or corrupted receipt
therefore publishes nothing at all.

Neither a task nor a completion grants Gate 0, T1, P50, checkpoint-selection or
training authority.  Every authority field is published ``False``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_process_v2_rebind import (
    MANIFEST_FILENAME as REBIND_MANIFEST_FILENAME,
)
from compose_v4.data.editing_process_v2_rebind import (
    PROOF_FILENAME as REBIND_PROOF_FILENAME,
)
from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_SCHEMA as REBIND_COMPLETION_SCHEMA,
)
from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_SCHEMA_VERSION as REBIND_COMPLETION_SCHEMA_VERSION,
)
from compose_v4.data.editing_process_v2_rebind import (
    COMPLETION_STATUS as REBIND_COMPLETION_STATUS,
)
from compose_v4.data.editing_process_v2_rebind import (
    ProcessV2RebindError,
    ProcessV2RebindExclusionCode,
    editing_process_v2_identity,
    require_production_source_geometry,
    validate_pinned_process_identity,
    validate_process_v2_rebind_plan,
    validate_process_v2_rebind_task_result,
)
from compose_v4.data.editing_v2_process_v2_active8_interfaces import (
    ACTIVE8_CENSUS_FIELDS,
    ACTIVE8_EXCLUDED,
    JOIN_KEY_FIELDS,
    TRACE_KEY_FIELDS,
    UPSTREAM_REJECTED,
)
from compose_v4.data.editing_v2_process_v2_active8_policy import (
    ProcessV2Active8Policy,
    ProcessV2Active8PolicyError,
    ProcessV2Active8TraceDecision,
    ProcessV2ProductionCandidateChecker,
    build_process_v2_active8_policy,
    evaluate_process_v2_active8_trace,
    upstream_rejected_trace_decision,
    validate_process_v2_active8_policy,
    validate_process_v2_active8_policy_payload,
)
from compose_v4.data.editing_v2_process_v2_active8_runtime import (
    ProcessV2Active8RuntimeError,
    validate_process_v2_active8_model_runtime_identity,
)
from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    ProcessV2ArtifactPathError,
    ProcessV2ChunkCacheError,
    ProcessV2ChunkTarget,
    load_committed_process_v2_chunk_cache_completion,
    mount_process_v2_artifact_path,
    open_process_v2_chunk_cache,
    read_process_v2_chunk_target,
    require_process_v2_artifact_path,
    run_bounded_map,
    validate_map_container_bound,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    ProcessV2SchemaError,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
    require_census_reconciles,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel

# ---- Frozen identity ----

PLAN_SCHEMA = "compose.data.editing_v2_process_v2_active8_decision_plan"
PLAN_SCHEMA_VERSION = 1
PLAN_STATUS = "FROZEN_PROCESS_V2_ACTIVE8_TASKS_NO_DOWNSTREAM_AUTHORITY"
DECISION_SCHEMA = "compose.data.editing_v2_process_v2_active8_trace_decision"
DECISION_SCHEMA_VERSION = 1
RECEIPT_SCHEMA = "compose.data.editing_v2_process_v2_active8_receipt"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_STATUS = "COMPLETE_PROCESS_V2_ACTIVE8_CHUNK_NO_DOWNSTREAM_AUTHORITY"
COMPLETION_SCHEMA = "compose.data.editing_v2_process_v2_active8_completion"
COMPLETION_SCHEMA_VERSION = 1
COMPLETION_STATUS = "COMPLETE_PROCESS_V2_ACTIVE8_CENSUS_NO_DOWNSTREAM_AUTHORITY"
IMPLEMENTATION_REVISION_SCHEMA = (
    "compose.data.editing_v2_process_v2_active8_implementation_revision"
)
IMPLEMENTATION_REVISION_SCHEMA_VERSION = 1

PLAN_FILENAME = "PROCESS_V2_ACTIVE8_PLAN.json"
DECISION_FILENAME = "decisions.jsonl.gz"
RECEIPT_FILENAME = "RECEIPT.json"
COMPLETION_FILENAME = "PROCESS_V2_ACTIVE8_COMPLETE.json"
TASK_DIRNAME = "tasks"

DEFAULT_OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_active8"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The modules whose contents change what an Active8 decision IS.  Narrow on
#: purpose: a commit that touches nothing here must not relocate the run.  The
#: policy and this driver decide; the codec, the successor evaluator, the
#: executor and the process identity define what they decide against.
IMPLEMENTATION_FILES: tuple[str, ...] = (
    "src/compose_v4/data/editing_v2_process_v2_active8_interfaces.py",
    "src/compose_v4/data/editing_v2_process_v2_active8_mapreduce.py",
    "src/compose_v4/data/editing_v2_process_v2_active8_policy.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/rewrite/action_codec_v4.py",
    "src/compose_v4/rewrite/editing_v2_process_identity.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/process_v2_atom_delete.py",
)

#: Every count a task and the run publish.  The first four are the seam's census
#: identity; the rest are the unaggregated candidate geometry.
COUNT_FIELDS: tuple[str, ...] = (
    *ACTIVE8_CENSUS_FIELDS,
    "evaluated_entries",
    "accepted_actions",
    "evaluated_actions",
    "progress_rows",
    "raw_candidate_marks",
    "canonical_candidate_successors",
    "matching_candidate_marks",
    "exact_successor_candidate_marks",
    "successor_alias_multiplicity",
)
_CANDIDATE_TOTAL_FIELDS: tuple[str, ...] = COUNT_FIELDS[-5:]

_ADDRESS_FIELDS: tuple[str, ...] = (
    "packed_shard_content_sha256",
    "packed_shard_name",
    "entry_index",
    "trace_id",
    "layer",
    "partition",
    "source_key",
    "target_key",
    "path_length",
)
_CACHE_ADDRESS_FIELDS: tuple[str, ...] = (
    "cache_source_artifact_path",
    "cache_source_task_identity_sha256",
    "cache_source_manifest_sha256",
    "cache_semantic_identity_sha256",
    "cache_physical_identity_sha256",
    "chunk_index",
    "chunk_filename",
    "chunk_file_sha256",
    "chunk_uncompressed_sha256",
    "chunk_row_count",
)
_REBIND_ADDRESS_FIELDS: tuple[str, ...] = (
    "rebind_task_identity_sha256",
    "rebind_output_artifact_path",
    "rebind_receipt_sha256",
    "rebind_manifest_sha256",
    "rebind_proof_ledger_sha256",
)
_TASK_FIELDS = frozenset(
    {
        "task_identity_sha256",
        "output_artifact_path",
        "v1_task_identity_sha256",
        "data_lane",
        "split",
        "entry_start",
        "entry_stop",
        "semantic_shard_sha256",
        "pinned_process_identity_sha256",
        *_CACHE_ADDRESS_FIELDS,
        *_REBIND_ADDRESS_FIELDS,
    }
)
_CACHE_BINDING_FIELDS = frozenset(
    {
        "cache_completion_sha256",
        "cache_implementation_sha256",
        "cache_semantic_identity_sha256",
        "cache_planned_semantic_identity_sha256",
        "cache_physical_identity_sha256",
        "cache_run_artifact_root",
        "records_per_chunk",
        "source_count",
        "entries",
        "chunk_count",
    }
)
_REBIND_BINDING_FIELDS = frozenset(
    {
        "rebind_completion_sha256",
        "rebind_run_identity_sha256",
        "rebind_plan_sha256",
        "rebind_source_geometry",
        "rebind_result_inventory_sha256",
        "rebind_task_count",
        "rebind_source_entries",
        "rebind_admitted_entries",
        "rebind_rejected_entries",
        "rebind_rejected_traces_by_code",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "implementation_revision",
        "implementation_sha256",
        "process_v2_identity",
        "pinned_process_identity",
        "policy",
        "model_runtime_identity",
        "cache_binding",
        "rebind_binding",
        "admitted_source_identity",
        "admitted_source_sha256",
        "run_identity_sha256",
        "run_artifact_root",
        "output_artifact_prefix",
        "expected_task_count",
        "expected_entry_count",
        "tasks",
        "task_inventory_sha256",
        "plan_sha256",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "run_identity_sha256",
        "plan_sha256",
        "implementation_sha256",
        "process_v2_identity_sha256",
        "policy_sha256",
        "model_runtime_identity_sha256",
        "task",
        "decision_file_sha256",
        "decision_stream_sha256",
        "counts",
        "action_family_histogram",
        "active8_exclusion_reason_histogram",
        "upstream_rejection_reason_histogram",
        "receipt_sha256",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        *TRACE_KEY_FIELDS,
        "data_lane",
        "split",
        "address",
        "policy_sha256",
        "category",
        "upstream_rejection_code",
        "active8_status",
        "emits_progress_rows",
        "actions",
        "active8_exclusions",
        "progress_rows",
        "candidate_totals",
        "action_family_histogram",
        "active8_exclusion_reason_histogram",
        "decision_sha256",
    }
)

_UPSTREAM_REJECTION_CODES: frozenset[str] = frozenset(
    code.value for code in ProcessV2RebindExclusionCode
)


class ProcessV2Active8MapReduceError(RuntimeError):
    """The Process-V2 Active8 plan, task result, or reduction is inconsistent."""


class ProcessV2Active8Incomplete(ProcessV2Active8MapReduceError):
    """At least one planned Process-V2 Active8 task has not published."""


# ---- Small validated primitives ----


def _sha(value: object) -> str:
    return canonical_sha256(value)


def _line(value: object) -> bytes:
    return canonical_bytes(value) + b"\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessV2Active8MapReduceError(f"{field} must be a full lowercase SHA-256")
    return value


def _require_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ProcessV2Active8MapReduceError(f"{field} must be a nonnegative integer")
    return value


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ProcessV2Active8MapReduceError(f"{field} must be normalized nonempty text")
    return value


def _mounted(value: object, *, artifact_root: Path, field: str) -> Path:
    try:
        return mount_process_v2_artifact_path(
            require_process_v2_artifact_path(value, field=field),
            artifact_root=artifact_root,
            field=field,
        )
    except ProcessV2ArtifactPathError as error:
        raise ProcessV2Active8MapReduceError(
            f"{field} is not a mountable artifact path"
        ) from error


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2Active8MapReduceError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ProcessV2Active8MapReduceError(f"{label} is not a JSON object: {path}")
    return value


def _publish_bytes(target: Path, payload: bytes, *, label: str) -> None:
    """Write one immutable object, or prove the existing one is byte-identical."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise ProcessV2Active8MapReduceError(f"immutable {label} collision at {target}")
        return
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _deterministic_gzip(raw: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


# ---- The narrow implementation revision ----


def build_process_v2_active8_implementation_revision(
    *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Hash exactly the modules that decide, or define what is decided against."""

    root = Path(repo_root or DEFAULT_REPO_ROOT)
    files: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise ProcessV2Active8MapReduceError(
                f"the Process-V2 Active8 implementation file is absent: {relative}"
            )
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    body = {
        "schema": IMPLEMENTATION_REVISION_SCHEMA,
        "schema_version": IMPLEMENTATION_REVISION_SCHEMA_VERSION,
        "implementation_files": dict(sorted(files.items())),
    }
    return {**body, "implementation_sha256": _sha(body)}


def validate_process_v2_active8_implementation_revision(
    value: object, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Refuse any revision that is not the one recomputed from the live tree.

    Owner-computed rather than caller-supplied: a revision a caller can choose is
    provenance at most, so the only thing a supplied one adds is the requirement
    that the caller's belief matches.
    """

    current = build_process_v2_active8_implementation_revision(repo_root=repo_root)
    if value != current:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 implementation revision differs from the live tree"
        )
    return dict(current)


# ---- Model runtime identity ----


def validate_model_runtime_identity(value: object) -> dict[str, Any]:
    """Require a self-hashed, nonempty, Process-V2 model runtime descriptor."""

    if not isinstance(value, Mapping):
        raise ProcessV2Active8MapReduceError("the model runtime identity must be an object")
    identity = dict(value)
    declared = _require_sha256(
        identity.get("identity_sha256"), field="model_runtime_identity.identity_sha256"
    )
    body = {key: item for key, item in identity.items() if key != "identity_sha256"}
    if not body or declared != _sha(body):
        raise ProcessV2Active8MapReduceError(
            "the model runtime identity is empty or its self-hash disagrees"
        )
    # Intact is not the same as correct: a Process-V1 runtime descriptor is
    # perfectly self-consistent, so the owning validator is asked whether this
    # one describes a Process-V2 model at all.
    try:
        validate_process_v2_active8_model_runtime_identity(identity)
    except ProcessV2Active8RuntimeError as error:
        raise ProcessV2Active8MapReduceError(str(error)) from error
    return identity


# ---- Binding the two upstream stages ----


def bind_process_v2_active8_sources(
    *,
    cache_run_artifact_root: str,
    rebind_plan: Mapping[str, Any],
    rebind_completion: Mapping[str, Any],
    admitted_source_identity: Mapping[str, Any],
    artifact_root: Path,
    repo_root: Path,
    verify_chunk_bytes: bool = True,
) -> dict[str, Any]:
    """Bind one committed cache generation to one complete rebind run.

    Every binding here is a refusal surface, not a note.  The cache must be
    committed, the rebind must be complete and produced under the production
    chunk-cache geometry, the rebind must have read *this* cache generation, and
    the admitted-source descriptor must name the same rebind run.  A bounded
    raw-oracle rebind is refused outright: it is a correctness comparison, and
    an Active8 decision built on one would be evidence resting on a control.
    """

    try:
        cache_completion = load_committed_process_v2_chunk_cache_completion(
            cache_run_artifact_root, artifact_root=artifact_root, repo_root=repo_root
        )
        generation = open_process_v2_chunk_cache(
            cache_completion,
            artifact_root=artifact_root,
            repo_root=repo_root,
            verify_chunk_bytes=verify_chunk_bytes,
        )
    except ProcessV2ChunkCacheError as error:
        raise ProcessV2Active8MapReduceError(
            f"the Process-V2 chunk cache at {cache_run_artifact_root} cannot be bound: {error}"
        ) from error

    try:
        plan = validate_process_v2_rebind_plan(rebind_plan, repo_root=repo_root)
        require_production_source_geometry(plan, label="the Process-V2 rebind plan")
        require_production_source_geometry(
            rebind_completion, label="the Process-V2 rebind completion"
        )
    except ProcessV2RebindError as error:
        raise ProcessV2Active8MapReduceError(
            f"the Process-V2 rebind plan is not a production source for Active8: {error}"
        ) from error
    completion = dict(rebind_completion)
    body = {key: item for key, item in completion.items() if key != "completion_sha256"}
    if (
        completion.get("schema") != REBIND_COMPLETION_SCHEMA
        or completion.get("schema_version") != REBIND_COMPLETION_SCHEMA_VERSION
        or completion.get("status") != REBIND_COMPLETION_STATUS
        or completion.get("training_authorized") is not False
        or completion.get("completion_sha256") != _sha(body)
        or completion.get("run_identity_sha256") != plan["run_identity_sha256"]
        or completion.get("plan_sha256") != plan["plan_sha256"]
        or completion.get("task_inventory_sha256") != plan["task_inventory_sha256"]
    ):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 rebind completion does not seal the plan it is bound with"
        )
    live_identity = editing_process_v2_identity()
    if (
        completion.get("process_v2_identity_sha256")
        != live_identity["process_identity_sha256"]
    ):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 rebind completion names another Process-V2 identity"
        )
    rebind_cache = plan.get("cache_binding")
    if not isinstance(rebind_cache, Mapping) or rebind_cache.get(
        "cache_physical_identity_sha256"
    ) != cache_completion.get("cache_physical_identity_sha256"):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 rebind read a different chunk-cache generation than the one bound"
        )

    descriptor = dict(admitted_source_identity)
    try:
        require_authority_false(descriptor, label="the admitted-source identity")
    except ProcessV2SchemaError as error:
        raise ProcessV2Active8MapReduceError(
            "the admitted-source identity does not publish every authority field false"
        ) from error
    if (
        descriptor.get("run_identity_sha256") != plan["run_identity_sha256"]
        or descriptor.get("plan_sha256") != plan["plan_sha256"]
        or descriptor.get("completion_sha256") != completion["completion_sha256"]
    ):
        raise ProcessV2Active8MapReduceError(
            "the admitted-source identity describes another Process-V2 rebind run"
        )
    admitted_sha256 = _require_sha256(
        descriptor.get("admitted_source_sha256"),
        field="admitted_source_identity.admitted_source_sha256",
    )

    cache_binding = {
        "cache_completion_sha256": str(cache_completion["completion_sha256"]),
        "cache_implementation_sha256": str(cache_completion["cache_implementation_sha256"]),
        # Two values, and both are needed. The MEASURED identity is what the
        # independent fused passes produced and the reducer sealed; the PLANNED
        # one is all a worker could know when it wrote its manifest, so it is the
        # value a chunk target -- and therefore an Active8 task -- carries.
        # Binding only the measured one would refuse every correct task.
        "cache_semantic_identity_sha256": str(
            cache_completion["cache_semantic_identity_sha256"]
        ),
        "cache_planned_semantic_identity_sha256": str(
            cache_completion["planned_semantic_identity_sha256"]
        ),
        "cache_physical_identity_sha256": str(
            cache_completion["cache_physical_identity_sha256"]
        ),
        "cache_run_artifact_root": str(cache_completion["run_artifact_root"]),
        "records_per_chunk": int(cache_completion["records_per_chunk"]),
        "source_count": int(cache_completion["source_count"]),
        "entries": int(cache_completion["entries"]),
        "chunk_count": int(cache_completion["chunk_count"]),
    }
    rebind_binding = {
        "rebind_completion_sha256": str(completion["completion_sha256"]),
        "rebind_run_identity_sha256": str(completion["run_identity_sha256"]),
        "rebind_plan_sha256": str(completion["plan_sha256"]),
        "rebind_source_geometry": str(completion["source_geometry"]),
        "rebind_result_inventory_sha256": str(completion["result_inventory_sha256"]),
        "rebind_task_count": int(completion["task_count"]),
        "rebind_source_entries": int(completion["counts"]["source_entries"]),
        "rebind_admitted_entries": int(completion["counts"]["admitted_entries"]),
        "rebind_rejected_entries": int(completion["counts"]["rejected_entries"]),
        "rebind_rejected_traces_by_code": dict(completion["rejected_traces_by_code"]),
    }
    return {
        "pinned_process_identity": dict(plan["pinned_process_identity"]),
        "cache_binding": cache_binding,
        "rebind_binding": rebind_binding,
        "admitted_source_identity": dict(descriptor),
        "admitted_source_sha256": admitted_sha256,
        "targets": generation.targets(),
        "rebind_results": tuple(dict(row) for row in completion["result_inventory"]),
    }


def _task_bodies(
    *,
    targets: Sequence[ProcessV2ChunkTarget],
    rebind_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One task body per verified cache chunk, joined to its rebind result.

    The join is required to be a bijection on ``(v1_task_identity_sha256,
    entry_start, entry_stop)``.  A cache chunk with no rebind decision, or a
    rebind range the cache never published, means the two stages partitioned the
    same payload differently, and a plan built on that would silently decide a
    subset.
    """

    by_range: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rebind_results:
        key = (
            str(row["v1_task_identity_sha256"]),
            int(row["entry_start"]),
            int(row["entry_stop"]),
        )
        if key in by_range:
            raise ProcessV2Active8MapReduceError(
                f"the Process-V2 rebind published two results for the same range {key}"
            )
        by_range[key] = row
    bodies: list[dict[str, Any]] = []
    consumed: set[tuple[str, int, int]] = set()
    for target in targets:
        key = (target.v1_task_identity_sha256, target.entry_start, target.entry_stop)
        rebind = by_range.get(key)
        if rebind is None:
            raise ProcessV2Active8MapReduceError(
                f"the Process-V2 rebind decided no entries for cache chunk {key}"
            )
        if str(rebind["data_lane"]) != target.data_lane or str(rebind["split"]) != target.split:
            raise ProcessV2Active8MapReduceError(
                f"cache chunk {key} and its rebind result name different lane/split cells"
            )
        consumed.add(key)
        bodies.append(
            {
                "v1_task_identity_sha256": target.v1_task_identity_sha256,
                "data_lane": target.data_lane,
                "split": target.split,
                "entry_start": target.entry_start,
                "entry_stop": target.entry_stop,
                "semantic_shard_sha256": target.semantic_shard_sha256,
                "pinned_process_identity_sha256": target.pinned_process_identity_sha256,
                **target.as_payload(),
                "rebind_task_identity_sha256": str(rebind["task_identity_sha256"]),
                "rebind_output_artifact_path": str(rebind["output_artifact_path"]),
                "rebind_receipt_sha256": str(rebind["receipt_sha256"]),
                "rebind_manifest_sha256": str(rebind["manifest_sha256"]),
                "rebind_proof_ledger_sha256": str(rebind["proof_ledger_sha256"]),
            }
        )
    unconsumed = sorted(set(by_range) - consumed)
    if unconsumed:
        raise ProcessV2Active8MapReduceError(
            f"the chunk cache published no chunk for rebind ranges {unconsumed}"
        )
    return bodies


def plan_process_v2_active8_decisions(
    *,
    cache_run_artifact_root: str,
    rebind_plan: Mapping[str, Any],
    rebind_completion: Mapping[str, Any],
    admitted_source_identity: Mapping[str, Any],
    model_runtime_identity: Mapping[str, Any],
    artifact_root: Path,
    repo_root: Path,
    policy: ProcessV2Active8Policy | None = None,
    output_artifact_prefix: str = DEFAULT_OUTPUT_ARTIFACT_PREFIX,
    verify_chunk_bytes: bool = True,
) -> dict[str, Any]:
    """Freeze one deterministic task per verified cache chunk."""

    require_process_v2_artifact_path(output_artifact_prefix, field="output_artifact_prefix")
    selected = validate_process_v2_active8_policy(
        policy or build_process_v2_active8_policy()
    )
    revision = build_process_v2_active8_implementation_revision(repo_root=repo_root)
    runtime = validate_model_runtime_identity(model_runtime_identity)
    bound = bind_process_v2_active8_sources(
        cache_run_artifact_root=cache_run_artifact_root,
        rebind_plan=rebind_plan,
        rebind_completion=rebind_completion,
        admitted_source_identity=admitted_source_identity,
        artifact_root=artifact_root,
        repo_root=repo_root,
        verify_chunk_bytes=verify_chunk_bytes,
    )
    identity = editing_process_v2_identity()
    if selected.process_identity_sha256 != identity["process_identity_sha256"]:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 policy binds another process identity"
        )
    bodies = _task_bodies(
        targets=bound["targets"], rebind_results=bound["rebind_results"]
    )
    run_body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "implementation_sha256": str(revision["implementation_sha256"]),
        "process_identity_sha256": str(identity["process_identity_sha256"]),
        "pinned_process_identity_sha256": str(
            bound["pinned_process_identity"]["process_identity_sha256"]
        ),
        "policy_sha256": selected.policy_sha256,
        "model_runtime_identity_sha256": str(runtime["identity_sha256"]),
        "cache_binding": bound["cache_binding"],
        "rebind_binding": bound["rebind_binding"],
        "admitted_source_sha256": bound["admitted_source_sha256"],
        "output_artifact_prefix": output_artifact_prefix,
        "task_bodies_sha256": _sha(bodies),
    }
    run_identity_sha256 = _sha(run_body)
    run_artifact_root = f"{output_artifact_prefix}/runs/{run_identity_sha256}"
    tasks: list[dict[str, Any]] = []
    for body in bodies:
        task_identity_sha256 = _sha(
            {"run_identity_sha256": run_identity_sha256, "task": body}
        )
        tasks.append(
            {
                **body,
                "task_identity_sha256": task_identity_sha256,
                "output_artifact_path": (
                    f"{run_artifact_root}/{TASK_DIRNAME}/{task_identity_sha256}"
                ),
            }
        )
    plan_body: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PLAN_STATUS,
        **authority_false_block(),
        "implementation_revision": dict(revision),
        "implementation_sha256": str(revision["implementation_sha256"]),
        "process_v2_identity": dict(identity),
        "pinned_process_identity": dict(bound["pinned_process_identity"]),
        "policy": selected.as_payload(),
        "model_runtime_identity": dict(runtime),
        "cache_binding": bound["cache_binding"],
        "rebind_binding": bound["rebind_binding"],
        "admitted_source_identity": bound["admitted_source_identity"],
        "admitted_source_sha256": bound["admitted_source_sha256"],
        "run_identity_sha256": run_identity_sha256,
        "run_artifact_root": run_artifact_root,
        "output_artifact_prefix": output_artifact_prefix,
        "expected_task_count": len(tasks),
        "expected_entry_count": int(bound["cache_binding"]["entries"]),
        "tasks": tasks,
        "task_inventory_sha256": _sha(tasks),
    }
    plan = {**plan_body, "plan_sha256": _sha(plan_body)}
    return validate_process_v2_active8_plan(plan, repo_root=repo_root)


def validate_process_v2_active8_plan(
    value: object, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate plan identity, task addressing and census; read no artifact."""

    if not isinstance(value, Mapping):
        raise ProcessV2Active8MapReduceError("the Process-V2 Active8 plan must be an object")
    plan = dict(value)
    if set(plan) != _PLAN_FIELDS:
        raise ProcessV2Active8MapReduceError(
            "Process-V2 Active8 plan fields disagree; "
            f"missing={sorted(_PLAN_FIELDS - set(plan))}, "
            f"unexpected={sorted(set(plan) - _PLAN_FIELDS)}"
        )
    require_authority_false(plan, label="the Process-V2 Active8 plan")
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["schema_version"] != PLAN_SCHEMA_VERSION
        or plan["status"] != PLAN_STATUS
    ):
        raise ProcessV2Active8MapReduceError("the Process-V2 Active8 plan contract disagrees")
    revision = validate_process_v2_active8_implementation_revision(
        plan["implementation_revision"], repo_root=repo_root
    )
    if plan["implementation_sha256"] != revision["implementation_sha256"]:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 plan implementation hash disagrees with its revision"
        )
    identity = editing_process_v2_identity()
    if plan["process_v2_identity"] != identity:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 plan names another Process-V2 process identity"
        )
    try:
        pinned = validate_pinned_process_identity(plan["pinned_process_identity"])
    except ProcessV2RebindError as error:
        raise ProcessV2Active8MapReduceError(
            "the pinned V1 process identity the Active8 plan carries is not self-consistent"
        ) from error
    if pinned["process_identity_sha256"] == identity["process_identity_sha256"]:
        raise ProcessV2Active8MapReduceError(
            "the pinned payload identity and the live Process-V2 identity are the same value; "
            "the historical payload is superseded by construction and the two must stay distinct"
        )
    try:
        policy_payload = validate_process_v2_active8_policy_payload(plan["policy"])
    except ProcessV2Active8PolicyError as error:
        raise ProcessV2Active8MapReduceError(
            f"the Process-V2 Active8 plan does not carry the live Active8 policy: {error}"
        ) from error
    runtime = validate_model_runtime_identity(plan["model_runtime_identity"])
    for field_set, block, label in (
        (_CACHE_BINDING_FIELDS, plan["cache_binding"], "cache_binding"),
        (_REBIND_BINDING_FIELDS, plan["rebind_binding"], "rebind_binding"),
    ):
        if not isinstance(block, Mapping) or set(block) != field_set:
            raise ProcessV2Active8MapReduceError(f"Process-V2 Active8 plan {label} fields disagree")
    try:
        require_production_source_geometry(
            {"source_geometry": plan["rebind_binding"]["rebind_source_geometry"]},
            label="the bound Process-V2 rebind",
        )
    except ProcessV2RebindError as error:
        raise ProcessV2Active8MapReduceError(
            "the bound Process-V2 rebind geometry is not the production one"
        ) from error
    descriptor = plan["admitted_source_identity"]
    if not isinstance(descriptor, Mapping):
        raise ProcessV2Active8MapReduceError("the admitted-source identity must be an object")
    require_authority_false(descriptor, label="the bound admitted-source identity")
    if descriptor.get("admitted_source_sha256") != plan["admitted_source_sha256"]:
        raise ProcessV2Active8MapReduceError(
            "the plan's admitted-source hash disagrees with the descriptor it embeds"
        )

    tasks = plan["tasks"]
    if not isinstance(tasks, list) or len(tasks) != plan["expected_task_count"]:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 plan task count disagrees with its inventory"
        )
    bodies: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    entries = 0
    for task in tasks:
        if not isinstance(task, Mapping) or set(task) != _TASK_FIELDS:
            raise ProcessV2Active8MapReduceError("Process-V2 Active8 plan task fields disagree")
        body = {
            key: item
            for key, item in task.items()
            if key not in {"task_identity_sha256", "output_artifact_path"}
        }
        expected_identity = _sha(
            {"run_identity_sha256": plan["run_identity_sha256"], "task": body}
        )
        expected_path = (
            f"{plan['run_artifact_root']}/{TASK_DIRNAME}/{expected_identity}"
        )
        if (
            task["task_identity_sha256"] != expected_identity
            or task["output_artifact_path"] != expected_path
        ):
            raise ProcessV2Active8MapReduceError(
                "a Process-V2 Active8 task is not addressed by its own content"
            )
        start = _require_int(task["entry_start"], field="task.entry_start")
        stop = _require_int(task["entry_stop"], field="task.entry_stop")
        rows = _require_int(task["chunk_row_count"], field="task.chunk_row_count")
        if stop - start != rows:
            raise ProcessV2Active8MapReduceError(
                "a Process-V2 Active8 task entry range disagrees with its chunk row count"
            )
        key = (str(task["v1_task_identity_sha256"]), start, stop)
        if key in seen:
            raise ProcessV2Active8MapReduceError(
                f"the Process-V2 Active8 plan names the entry range {key} twice"
            )
        seen.add(key)
        entries += rows
        if task["pinned_process_identity_sha256"] != pinned["process_identity_sha256"]:
            raise ProcessV2Active8MapReduceError(
                "a Process-V2 Active8 task reads a chunk pinned to another payload identity"
            )
        if (
            task["cache_physical_identity_sha256"]
            != plan["cache_binding"]["cache_physical_identity_sha256"]
            or task["cache_semantic_identity_sha256"]
            != plan["cache_binding"]["cache_planned_semantic_identity_sha256"]
        ):
            raise ProcessV2Active8MapReduceError(
                "a Process-V2 Active8 task reads a chunk of another cache generation"
            )
        bodies.append(body)
    if entries != plan["expected_entry_count"]:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 plan does not cover the exact cached entry census"
        )
    if plan["task_inventory_sha256"] != _sha(list(tasks)):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 plan task inventory hash disagrees"
        )
    run_body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "implementation_sha256": plan["implementation_sha256"],
        "process_identity_sha256": str(identity["process_identity_sha256"]),
        "pinned_process_identity_sha256": str(pinned["process_identity_sha256"]),
        "policy_sha256": str(policy_payload["policy_sha256"]),
        "model_runtime_identity_sha256": str(runtime["identity_sha256"]),
        "cache_binding": dict(plan["cache_binding"]),
        "rebind_binding": dict(plan["rebind_binding"]),
        "admitted_source_sha256": plan["admitted_source_sha256"],
        "output_artifact_prefix": plan["output_artifact_prefix"],
        "task_bodies_sha256": _sha(bodies),
    }
    if plan["run_identity_sha256"] != _sha(run_body):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 run identity is not the hash of what it binds"
        )
    if plan["run_artifact_root"] != (
        f"{plan['output_artifact_prefix']}/runs/{plan['run_identity_sha256']}"
    ):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 run root is not addressed by the run identity"
        )
    body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    if plan["plan_sha256"] != _sha(body):
        raise ProcessV2Active8MapReduceError("the Process-V2 Active8 plan self-hash disagrees")
    return plan


def write_process_v2_active8_plan(
    plan: Mapping[str, Any], *, artifact_root: Path, repo_root: Path
) -> Path:
    """Publish the exact plan bytes the reducer will later require."""

    validated = validate_process_v2_active8_plan(plan, repo_root=repo_root)
    run_root = _mounted(
        validated["run_artifact_root"], artifact_root=artifact_root, field="plan.run_artifact_root"
    )
    target = run_root / PLAN_FILENAME
    _publish_bytes(target, _line(validated), label="Process-V2 Active8 plan")
    return target


def load_process_v2_active8_plan(path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Read a published plan and revalidate it whole."""

    return validate_process_v2_active8_plan(
        _load_json_object(Path(path), label="Process-V2 Active8 plan"), repo_root=repo_root
    )


def _task_by_identity(plan: Mapping[str, Any], task_identity_sha256: str) -> dict[str, Any]:
    _require_sha256(task_identity_sha256, field="task_identity_sha256")
    matches = [
        task for task in plan["tasks"] if task["task_identity_sha256"] == task_identity_sha256
    ]
    if len(matches) != 1:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 task identity is absent or duplicated in the frozen plan"
        )
    return dict(matches[0])


def reduction_order(tasks: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Order tasks by ``(data_lane, split, entry_start)`` and prove it total.

    Stated rather than assumed.  Plan order is deterministic but keyed by content
    hashes, so it is deterministic in an order nobody can read; the reduction
    sorts by the address the artifact is actually published under and refuses a
    duplicate key rather than breaking the tie silently.
    """

    ordered = sorted(
        (dict(task) for task in tasks),
        key=lambda task: (
            str(task["data_lane"]),
            str(task["split"]),
            int(task["entry_start"]),
        ),
    )
    keys = [(task["data_lane"], task["split"], int(task["entry_start"])) for task in ordered]
    if len(set(keys)) != len(keys):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 reduction order is ambiguous: two tasks share one "
            "(data_lane, split, entry_start) address"
        )
    return tuple(ordered)


# ---- Reading the upstream rebind decision of one range ----


def _chunk_target_for_task(task: Mapping[str, Any]) -> ProcessV2ChunkTarget:
    """Rebuild the exact chunk target one Active8 task binds."""

    return ProcessV2ChunkTarget(
        source_artifact_path=str(task["cache_source_artifact_path"]),
        cache_source_task_identity_sha256=str(task["cache_source_task_identity_sha256"]),
        cache_source_manifest_sha256=str(task["cache_source_manifest_sha256"]),
        cache_semantic_identity_sha256=str(task["cache_semantic_identity_sha256"]),
        cache_physical_identity_sha256=str(task["cache_physical_identity_sha256"]),
        v1_task_identity_sha256=str(task["v1_task_identity_sha256"]),
        data_lane=str(task["data_lane"]),
        split=str(task["split"]),
        semantic_shard_sha256=str(task["semantic_shard_sha256"]),
        pinned_process_identity_sha256=str(task["pinned_process_identity_sha256"]),
        chunk_index=int(task["chunk_index"]),
        chunk_filename=str(task["chunk_filename"]),
        chunk_file_sha256=str(task["chunk_file_sha256"]),
        chunk_uncompressed_sha256=str(task["chunk_uncompressed_sha256"]),
        entry_start=int(task["entry_start"]),
        entry_stop=int(task["entry_stop"]),
        row_count=int(task["chunk_row_count"]),
    )


def read_upstream_rebind_decisions(
    task: Mapping[str, Any], *, artifact_root: Path
) -> dict[int, tuple[bool, dict[str, Any]]]:
    """Return ``{entry_index: (admitted, row)}`` for exactly one rebind task.

    The proof row does not carry the V1 task identity -- it carries the entry
    index -- so the identity comes from the sibling receipt in the same output
    directory, which is exactly where the frozen join key's first component
    lives.  Reading it any other way would reconstruct the key rather than read
    it.
    """

    output = _mounted(
        task["rebind_output_artifact_path"],
        artifact_root=artifact_root,
        field="task.rebind_output_artifact_path",
    )
    try:
        receipt = validate_process_v2_rebind_task_result(output)
    except ProcessV2RebindError as error:
        raise ProcessV2Active8MapReduceError(
            f"the bound Process-V2 rebind task result is invalid: {output}"
        ) from error
    manifest = _load_json_object(
        output / REBIND_MANIFEST_FILENAME, label="Process-V2 rebind manifest"
    )
    disagreements = [
        field
        for field, declared, observed in (
            ("receipt_sha256", task["rebind_receipt_sha256"], receipt["receipt_sha256"]),
            ("manifest_sha256", task["rebind_manifest_sha256"], receipt["manifest_sha256"]),
            (
                "proof_ledger_sha256",
                task["rebind_proof_ledger_sha256"],
                receipt["proof_ledger_sha256"],
            ),
            (
                "task_identity_sha256",
                task["rebind_task_identity_sha256"],
                receipt["task_identity_sha256"],
            ),
            (
                "v1_task_identity_sha256",
                task["v1_task_identity_sha256"],
                receipt["v1_task_identity_sha256"],
            ),
            ("data_lane", task["data_lane"], receipt["data_lane"]),
            ("split", task["split"], receipt["split"]),
            ("entry_start", int(task["entry_start"]), int(receipt["entry_start"])),
            ("entry_stop", int(task["entry_stop"]), int(receipt["entry_stop"])),
        )
        if declared != observed
    ]
    if disagreements:
        raise ProcessV2Active8MapReduceError(
            "the bound Process-V2 rebind task disagrees with the Active8 plan on "
            f"{sorted(disagreements)}"
        )
    decisions: dict[int, tuple[bool, dict[str, Any]]] = {}
    with gzip.open(output / REBIND_PROOF_FILENAME, "rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            proof = json.loads(raw_line)
            entry_index = int(proof["entry_index"])
            if entry_index in decisions:
                raise ProcessV2Active8MapReduceError(
                    f"the Process-V2 rebind decided entry {entry_index} twice"
                )
            decisions[entry_index] = (True, dict(proof))
    for rejection in manifest["rejected_traces"]:
        entry_index = int(rejection["entry_index"])
        if entry_index in decisions:
            raise ProcessV2Active8MapReduceError(
                f"the Process-V2 rebind decided entry {entry_index} twice"
            )
        decisions[entry_index] = (False, dict(rejection))
    expected = set(range(int(task["entry_start"]), int(task["entry_stop"])))
    if set(decisions) != expected:
        raise ProcessV2Active8MapReduceError(
            "the bound Process-V2 rebind task does not decide the exact entry range"
        )
    return decisions


# ---- One published decision row ----


def _address_payload(address: Any) -> dict[str, Any]:
    return {
        "packed_shard_content_sha256": address.packed_shard_content_sha256,
        "packed_shard_name": address.packed_shard_name,
        "entry_index": address.entry_index,
        "trace_id": address.trace_id,
        "layer": address.layer,
        "partition": address.partition,
        "source_key": address.source_key,
        "target_key": address.target_key,
        "path_length": address.path_length,
    }


def _decision_row(
    *,
    task: Mapping[str, Any],
    addressed: Any,
    decision: ProcessV2Active8TraceDecision,
) -> dict[str, Any]:
    """Build the one published row, self-hashed, with every count intact."""

    address = addressed.address
    actions = [item.as_payload() for item in decision.action_decisions]
    families: Counter[str] = Counter(
        str(item["model_family"]) for item in actions if item["model_family"] is not None
    )
    reasons: Counter[str] = Counter(
        item.reason for item in decision.active8_exclusions
    )
    totals: Counter[str] = Counter()
    for item in actions:
        evidence = item["candidate_evidence"]
        if evidence is None:
            continue
        totals["raw_candidate_marks"] += int(evidence["raw_mark_count"])
        totals["canonical_candidate_successors"] += int(evidence["canonical_successor_count"])
        totals["matching_candidate_marks"] += int(evidence["matching_mark_count"])
        totals["exact_successor_candidate_marks"] += int(
            evidence["exact_successor_mark_count"]
        )
        totals["successor_alias_multiplicity"] += int(evidence["successor_alias_count"])
    progress = (
        [
            {
                "progress_index": index,
                "terminal": index == address.path_length,
                "state_sha256": persistent_slot_state_sha256(addressed.path.state_at(index)),
            }
            for index in range(address.path_length + 1)
        ]
        if decision.emits_progress_rows
        else []
    )
    body: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "schema_version": DECISION_SCHEMA_VERSION,
        "v1_task_identity_sha256": str(task["v1_task_identity_sha256"]),
        "entry_index": int(address.entry_index),
        "trace_id": str(address.trace_id),
        "data_lane": str(task["data_lane"]),
        "split": str(task["split"]),
        "address": _address_payload(address),
        "policy_sha256": decision.policy_sha256,
        "category": decision.category,
        "upstream_rejection_code": decision.upstream_rejection_code,
        "active8_status": decision.active8_status,
        "emits_progress_rows": decision.emits_progress_rows,
        "actions": actions,
        "active8_exclusions": [item.as_payload() for item in decision.active8_exclusions],
        "progress_rows": progress,
        "candidate_totals": {field: totals[field] for field in _CANDIDATE_TOTAL_FIELDS},
        "action_family_histogram": dict(sorted(families.items())),
        "active8_exclusion_reason_histogram": dict(sorted(reasons.items())),
    }
    return {**body, "decision_sha256": _sha(body)}


def _row_counts(row: Mapping[str, Any]) -> Counter[str]:
    """The census one decision row contributes, derived from the row itself."""

    counts: Counter[str] = Counter()
    counts["source_entries"] += 1
    category = row["category"]
    if category == UPSTREAM_REJECTED:
        counts["upstream_rejected_entries"] += 1
    elif category == ACTIVE8_EXCLUDED:
        counts["active8_excluded_entries"] += 1
        counts["evaluated_entries"] += 1
    elif category is None:
        counts["active8_accepted_entries"] += 1
        counts["evaluated_entries"] += 1
        counts["accepted_actions"] += len(row["actions"])
    else:
        raise ProcessV2Active8MapReduceError(
            f"a Process-V2 Active8 decision row names an undeclared category {category!r}"
        )
    if category != UPSTREAM_REJECTED:
        counts["evaluated_actions"] += len(row["actions"])
    counts["progress_rows"] += len(row["progress_rows"])
    counts.update(row["candidate_totals"])
    return counts


# ---- Task execution ----


def execute_process_v2_active8_decision_task(
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    *,
    artifact_root: Path,
    repo_root: Path,
    model: FactorizedTraceletRateModel,
    candidate_checker: ProcessV2ProductionCandidateChecker,
    model_runtime_identity_resolver: Callable[
        [FactorizedTraceletRateModel], Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """Decide exactly one cache chunk, or reuse its already published result."""

    validated = validate_process_v2_active8_plan(plan, repo_root=repo_root)
    task = _task_by_identity(validated, task_identity_sha256)
    if (
        not isinstance(candidate_checker, ProcessV2ProductionCandidateChecker)
        or candidate_checker.model is not model
        or candidate_checker.policy.policy_sha256 != validated["policy"]["policy_sha256"]
    ):
        raise ProcessV2Active8MapReduceError(
            "a Process-V2 Active8 worker requires the exact production model/checker pair"
        )
    observed_runtime = validate_model_runtime_identity(
        model_runtime_identity_resolver(model)
    )
    if observed_runtime != validated["model_runtime_identity"]:
        raise ProcessV2Active8MapReduceError(
            "the live model runtime identity differs from the frozen Active8 plan"
        )
    output = _mounted(
        task["output_artifact_path"],
        artifact_root=artifact_root,
        field="task.output_artifact_path",
    )
    if output.exists():
        receipt = validate_process_v2_active8_task_result(
            output, plan=validated, task=task
        )
        return {**receipt, "reused": True}

    upstream = read_upstream_rebind_decisions(task, artifact_root=artifact_root)
    target = _chunk_target_for_task(task)
    source_output = _mounted(
        target.source_artifact_path,
        artifact_root=artifact_root,
        field="task.cache_source_artifact_path",
    )
    rows: list[dict[str, Any]] = []
    observed_indices: list[int] = []
    try:
        for read in read_process_v2_chunk_target(
            source_output,
            target=target,
            expected_process_identity=validated["pinned_process_identity"],
            sentinel_replay_entries=0,
            recover_row_errors=False,
            repo_root=repo_root,
        ):
            observed_indices.append(read.entry_index)
            addressed = read.addressed
            if read.error is not None or addressed is None:
                raise ProcessV2Active8MapReduceError(
                    f"the cached row {read.entry_index} did not decode: {read.error}"
                )
            admitted, upstream_row = upstream[read.entry_index]
            if str(upstream_row["trace_id"]) != addressed.address.trace_id:
                raise ProcessV2Active8MapReduceError(
                    "the upstream rebind decision and the cached row name different traces "
                    f"at entry {read.entry_index}"
                )
            if admitted:
                if int(upstream_row["path_length"]) != addressed.address.path_length:
                    raise ProcessV2Active8MapReduceError(
                        "the upstream rebind proof and the cached row disagree on path length "
                        f"at entry {read.entry_index}"
                    )
                decision = evaluate_process_v2_active8_trace(
                    addressed,
                    candidate_check=candidate_checker,
                    policy=candidate_checker.policy,
                )
            else:
                code = str(upstream_row["exclusion_code"])
                if code not in _UPSTREAM_REJECTION_CODES:
                    raise ProcessV2Active8MapReduceError(
                        f"the upstream rebind rejection code {code!r} is undeclared"
                    )
                decision = upstream_rejected_trace_decision(
                    trace_id=addressed.address.trace_id,
                    rejection_code=code,
                    policy=candidate_checker.policy,
                )
            rows.append(_decision_row(task=task, addressed=addressed, decision=decision))
    except (ProcessV2ChunkCacheError, ProcessV2Active8PolicyError) as error:
        raise ProcessV2Active8MapReduceError(
            f"the Process-V2 Active8 task {task_identity_sha256} could not decide its chunk: "
            f"{error}"
        ) from error
    if observed_indices != list(range(int(task["entry_start"]), int(task["entry_stop"]))):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 task did not read its exact physical entry indices"
        )

    raw = b"".join(_line(row) for row in rows)
    decision_bytes = _deterministic_gzip(raw)
    totals: Counter[str] = Counter()
    families: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    upstream_reasons: Counter[str] = Counter()
    for row in rows:
        totals.update(_row_counts(row))
        families.update(row["action_family_histogram"])
        reasons.update(row["active8_exclusion_reason_histogram"])
        if row["category"] == UPSTREAM_REJECTED:
            upstream_reasons[str(row["upstream_rejection_code"])] += 1
    counts = {field: totals[field] for field in COUNT_FIELDS}
    _require_active8_census(counts, label=f"Process-V2 Active8 task {task_identity_sha256}")
    receipt_body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        **authority_false_block(),
        "run_identity_sha256": validated["run_identity_sha256"],
        "plan_sha256": validated["plan_sha256"],
        "implementation_sha256": validated["implementation_sha256"],
        "process_v2_identity_sha256": str(
            validated["process_v2_identity"]["process_identity_sha256"]
        ),
        "policy_sha256": str(validated["policy"]["policy_sha256"]),
        "model_runtime_identity_sha256": str(
            validated["model_runtime_identity"]["identity_sha256"]
        ),
        "task": dict(task),
        "decision_file_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "decision_stream_sha256": hashlib.sha256(raw).hexdigest(),
        "counts": counts,
        "action_family_histogram": dict(sorted(families.items())),
        "active8_exclusion_reason_histogram": dict(sorted(reasons.items())),
        "upstream_rejection_reason_histogram": dict(sorted(upstream_reasons.items())),
    }
    receipt = {**receipt_body, "receipt_sha256": _sha(receipt_body)}
    _publish_task_atomically(
        output, receipt=receipt, decision_bytes=decision_bytes, plan=validated, task=task
    )
    return {**receipt, "reused": False}


def _publish_task_atomically(
    output: Path,
    *,
    receipt: Mapping[str, Any],
    decision_bytes: bytes,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
) -> None:
    """Stage both objects, validate them, then rename the whole directory.

    Two files published independently can be observed half-written; a directory
    rename cannot.  A hard termination leaves only the hidden staging sibling,
    which is not a result and does not block the retry.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.", suffix=".staging")
    )
    try:
        (staging / DECISION_FILENAME).write_bytes(decision_bytes)
        (staging / RECEIPT_FILENAME).write_bytes(_line(receipt))
        validate_process_v2_active8_task_result(staging, plan=plan, task=task)
        if output.exists():
            validate_process_v2_active8_task_result(output, plan=plan, task=task)
            return
        try:
            os.rename(staging, output)
        except OSError:
            if not output.exists():
                raise
            validate_process_v2_active8_task_result(output, plan=plan, task=task)
            return
        staging = output
    finally:
        if staging != output and staging.exists():
            shutil.rmtree(staging)


def validate_process_v2_active8_task_result(
    output_dir: Path,
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile one published task from its own bytes, against its plan task."""

    output = Path(output_dir)
    if not output.is_dir():
        raise ProcessV2Active8Incomplete(f"the Process-V2 Active8 task is absent: {output}")
    observed = {path.name for path in output.iterdir()}
    if observed != {DECISION_FILENAME, RECEIPT_FILENAME}:
        raise ProcessV2Active8MapReduceError(
            f"the Process-V2 Active8 task inventory is incomplete or unexpected: {output}"
        )
    receipt_bytes = (output / RECEIPT_FILENAME).read_bytes()
    receipt = _load_json_object(output / RECEIPT_FILENAME, label="Process-V2 Active8 receipt")
    if set(receipt) != _RECEIPT_FIELDS or _line(receipt) != receipt_bytes:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 receipt fields or encoding disagree"
        )
    require_authority_false(receipt, label="the Process-V2 Active8 receipt")
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
        or receipt["status"] != RECEIPT_STATUS
        or receipt["receipt_sha256"] != _sha(body)
        or receipt["task"] != dict(task)
        or receipt["run_identity_sha256"] != plan["run_identity_sha256"]
        or receipt["plan_sha256"] != plan["plan_sha256"]
        or receipt["implementation_sha256"] != plan["implementation_sha256"]
        or receipt["process_v2_identity_sha256"]
        != plan["process_v2_identity"]["process_identity_sha256"]
        or receipt["policy_sha256"] != plan["policy"]["policy_sha256"]
        or receipt["model_runtime_identity_sha256"]
        != plan["model_runtime_identity"]["identity_sha256"]
        or set(receipt["counts"]) != set(COUNT_FIELDS)
    ):
        raise ProcessV2Active8MapReduceError("the Process-V2 Active8 receipt identity disagrees")

    decision_bytes = (output / DECISION_FILENAME).read_bytes()
    if hashlib.sha256(decision_bytes).hexdigest() != receipt["decision_file_sha256"]:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 decision bytes disagree with their receipt"
        )
    try:
        raw = gzip.decompress(decision_bytes)
    except (OSError, EOFError) as error:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 decision file is not gzip"
        ) from error
    if hashlib.sha256(raw).hexdigest() != receipt["decision_stream_sha256"]:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 decision stream disagrees with its receipt"
        )
    if _deterministic_gzip(raw) != decision_bytes:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 decision file is not deterministic gzip"
        )
    rows = read_process_v2_active8_decision_rows(raw, task=task, plan=plan)
    totals: Counter[str] = Counter()
    families: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    upstream_reasons: Counter[str] = Counter()
    for row in rows:
        totals.update(_row_counts(row))
        families.update(row["action_family_histogram"])
        reasons.update(row["active8_exclusion_reason_histogram"])
        if row["category"] == UPSTREAM_REJECTED:
            upstream_reasons[str(row["upstream_rejection_code"])] += 1
    counts = {field: totals[field] for field in COUNT_FIELDS}
    if (
        counts != receipt["counts"]
        or dict(sorted(families.items())) != receipt["action_family_histogram"]
        or dict(sorted(reasons.items())) != receipt["active8_exclusion_reason_histogram"]
        or dict(sorted(upstream_reasons.items()))
        != receipt["upstream_rejection_reason_histogram"]
    ):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 receipt census disagrees with its own decision rows"
        )
    _require_active8_census(counts, label="the Process-V2 Active8 task receipt")
    return receipt


def read_process_v2_active8_decision_rows(
    raw: bytes, *, task: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Decode and structurally revalidate one task's decision stream."""

    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProcessV2Active8MapReduceError(
                f"Process-V2 Active8 decision row {index} is malformed"
            ) from error
        if not isinstance(row, dict) or set(row) != _DECISION_FIELDS or _line(row) != line:
            raise ProcessV2Active8MapReduceError(
                f"Process-V2 Active8 decision row {index} fields or encoding disagree"
            )
        body = {key: item for key, item in row.items() if key != "decision_sha256"}
        if (
            row["schema"] != DECISION_SCHEMA
            or row["schema_version"] != DECISION_SCHEMA_VERSION
            or row["decision_sha256"] != _sha(body)
            or row["policy_sha256"] != plan["policy"]["policy_sha256"]
            or row["v1_task_identity_sha256"] != task["v1_task_identity_sha256"]
            or row["data_lane"] != task["data_lane"]
            or row["split"] != task["split"]
            or not isinstance(row["address"], dict)
            or set(row["address"]) != set(_ADDRESS_FIELDS)
            or row["address"]["entry_index"] != row["entry_index"]
            or row["address"]["trace_id"] != row["trace_id"]
            or row["address"]["layer"] != task["data_lane"]
            or row["address"]["partition"] != task["split"]
            or row["address"]["packed_shard_content_sha256"] != task["semantic_shard_sha256"]
        ):
            raise ProcessV2Active8MapReduceError(
                f"Process-V2 Active8 decision row {index} identity disagrees"
            )
        accepted = row["category"] is None
        expected_progress = row["address"]["path_length"] + 1 if accepted else 0
        if (
            row["emits_progress_rows"] is not accepted
            or len(row["progress_rows"]) != expected_progress
            or (accepted and row["active8_status"] != "accepted")
            or (row["category"] == ACTIVE8_EXCLUDED and row["active8_status"] != "excluded")
            or (row["category"] == UPSTREAM_REJECTED and row["active8_status"] != "not_evaluated")
            or (accepted and row["active8_exclusions"])
            or (row["category"] == ACTIVE8_EXCLUDED and not row["active8_exclusions"])
            or (
                row["category"] == UPSTREAM_REJECTED
                and (row["actions"] or row["active8_exclusions"])
            )
            or (
                row["category"] != UPSTREAM_REJECTED
                and len(row["actions"]) != row["address"]["path_length"]
            )
            or (row["category"] == UPSTREAM_REJECTED)
            != (row["upstream_rejection_code"] is not None)
        ):
            raise ProcessV2Active8MapReduceError(
                f"Process-V2 Active8 decision row {index} crosses its own admission boundary"
            )
        rows.append(row)
    observed = [int(row["entry_index"]) for row in rows]
    if observed != list(range(int(task["entry_start"]), int(task["entry_stop"]))):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 decision rows are not exact task-address coverage"
        )
    return rows


def _require_active8_census(counts: Mapping[str, int], *, label: str) -> None:
    """The seam's identity, plus the structural census the protocol declares."""

    for field in ACTIVE8_CENSUS_FIELDS:
        _require_int(counts.get(field), field=f"{label} {field}")
    source = int(counts["source_entries"])
    partitioned = (
        int(counts["upstream_rejected_entries"])
        + int(counts["active8_accepted_entries"])
        + int(counts["active8_excluded_entries"])
    )
    if source != partitioned:
        raise ProcessV2Active8MapReduceError(
            f"{label} census does not reconcile: {partitioned} partitioned != {source} source"
        )
    try:
        require_census_reconciles(structural_census(counts), label=label)
    except ProcessV2SchemaError as error:
        raise ProcessV2Active8MapReduceError(str(error)) from error


def structural_census(counts: Mapping[str, int]) -> dict[str, int]:
    """Project the Active8 census onto the structural three-field census.

    ``admitted`` is exactly the Active8-accepted set.  Both rejection categories
    land in ``rejected`` at this level and stay separately named above it, so a
    consumer of the smaller protocol sees a reconciling census without the two
    categories being merged anywhere they are actually reported.
    """

    return {
        "source_entries": int(counts["source_entries"]),
        "admitted_entries": int(counts["active8_accepted_entries"]),
        "rejected_entries": int(counts["upstream_rejected_entries"])
        + int(counts["active8_excluded_entries"]),
    }


def completed_process_v2_active8_task_ids(
    plan: Mapping[str, Any], *, artifact_root: Path, repo_root: Path
) -> frozenset[str]:
    """Return only fully verified durable results, refusing unexpected objects."""

    validated = validate_process_v2_active8_plan(plan, repo_root=repo_root)
    task_root = _mounted(
        f"{validated['run_artifact_root']}/{TASK_DIRNAME}",
        artifact_root=artifact_root,
        field="plan task root",
    )
    expected = {task["task_identity_sha256"] for task in validated["tasks"]}
    if not task_root.exists():
        return frozenset()
    observed = {path.name for path in task_root.iterdir()}
    staging = {
        name
        for name in observed
        if any(name.startswith(f".{task_id}.") and name.endswith(".staging") for task_id in expected)
    }
    unexpected = (observed - staging) - expected
    if unexpected:
        raise ProcessV2Active8MapReduceError(
            f"the Process-V2 Active8 task namespace holds unexpected objects: {sorted(unexpected)}"
        )
    by_id = {task["task_identity_sha256"]: task for task in validated["tasks"]}
    complete: set[str] = set()
    for task_id in sorted(observed - staging):
        validate_process_v2_active8_task_result(
            task_root / task_id, plan=validated, task=by_id[task_id]
        )
        complete.add(task_id)
    return frozenset(complete)


def reduce_process_v2_active8_decisions(
    plan: Mapping[str, Any], *, artifact_root: Path, repo_root: Path
) -> dict[str, Any]:
    """Prove exact chunk coverage and publish the reconciled Active8 census.

    Order-independent by construction: the reduction walks
    :func:`reduction_order`, recomputes every count from the published decision
    rows rather than from a receipt, and requires the seam's census identity at
    the task level and again at the run level.
    """

    validated = validate_process_v2_active8_plan(plan, repo_root=repo_root)
    run_root = _mounted(
        validated["run_artifact_root"],
        artifact_root=artifact_root,
        field="plan.run_artifact_root",
    )
    published_plan = run_root / PLAN_FILENAME
    if not published_plan.is_file() or published_plan.read_bytes() != _line(validated):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 reduction requires the exact published plan bytes"
        )
    complete = completed_process_v2_active8_task_ids(
        validated, artifact_root=artifact_root, repo_root=repo_root
    )
    missing = {task["task_identity_sha256"] for task in validated["tasks"]} - complete
    if missing:
        raise ProcessV2Active8Incomplete(
            f"the Process-V2 Active8 run is missing {len(missing)} planned task results"
        )
    totals: Counter[str] = Counter()
    families: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    upstream_reasons: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    by_id = {task["task_identity_sha256"]: task for task in validated["tasks"]}
    for task in reduction_order(validated["tasks"]):
        output = _mounted(
            task["output_artifact_path"],
            artifact_root=artifact_root,
            field="task.output_artifact_path",
        )
        receipt = validate_process_v2_active8_task_result(
            output, plan=validated, task=by_id[task["task_identity_sha256"]]
        )
        raw = gzip.decompress((output / DECISION_FILENAME).read_bytes())
        for row in read_process_v2_active8_decision_rows(raw, task=task, plan=validated):
            key = (str(row["v1_task_identity_sha256"]), int(row["entry_index"]))
            if key in seen:
                raise ProcessV2Active8MapReduceError(
                    f"the Process-V2 Active8 run decides the join key {key} twice"
                )
            seen.add(key)
        totals.update(receipt["counts"])
        families.update(receipt["action_family_histogram"])
        reasons.update(receipt["active8_exclusion_reason_histogram"])
        upstream_reasons.update(receipt["upstream_rejection_reason_histogram"])
        results.append(
            {
                "task_identity_sha256": task["task_identity_sha256"],
                "v1_task_identity_sha256": task["v1_task_identity_sha256"],
                "data_lane": task["data_lane"],
                "split": task["split"],
                "entry_start": task["entry_start"],
                "entry_stop": task["entry_stop"],
                "output_artifact_path": task["output_artifact_path"],
                "receipt_sha256": receipt["receipt_sha256"],
                "decision_file_sha256": receipt["decision_file_sha256"],
                "counts": dict(receipt["counts"]),
            }
        )
    counts = {field: int(totals[field]) for field in COUNT_FIELDS}
    _require_active8_census(counts, label="the Process-V2 Active8 run")
    if counts["source_entries"] != int(validated["expected_entry_count"]):
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 reduction does not account for every cached entry"
        )
    if len(seen) != counts["source_entries"]:
        raise ProcessV2Active8MapReduceError(
            "the Process-V2 Active8 reduction join keys do not cover the exact census"
        )
    upstream_total = int(validated["rebind_binding"]["rebind_rejected_entries"])
    if counts["upstream_rejected_entries"] != upstream_total:
        raise ProcessV2Active8MapReduceError(
            f"the Process-V2 Active8 run carried {counts['upstream_rejected_entries']} upstream "
            f"rejections, the bound rebind published {upstream_total}"
        )
    completion_body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **authority_false_block(),
        "run_identity_sha256": validated["run_identity_sha256"],
        "plan_sha256": validated["plan_sha256"],
        "implementation_sha256": validated["implementation_sha256"],
        "process_v2_identity_sha256": str(
            validated["process_v2_identity"]["process_identity_sha256"]
        ),
        "policy_sha256": str(validated["policy"]["policy_sha256"]),
        "model_runtime_identity_sha256": str(
            validated["model_runtime_identity"]["identity_sha256"]
        ),
        "cache_binding": dict(validated["cache_binding"]),
        "rebind_binding": dict(validated["rebind_binding"]),
        "admitted_source_sha256": validated["admitted_source_sha256"],
        "task_count": len(results),
        "result_inventory": results,
        "result_inventory_sha256": _sha(results),
        "active8_counts": counts,
        "structural_counts": structural_census(counts),
        "active8_action_family_histogram": dict(sorted(families.items())),
        "active8_exclusion_reason_histogram": dict(sorted(reasons.items())),
        "upstream_rejection_reason_histogram": dict(sorted(upstream_reasons.items())),
    }
    completion = {**completion_body, "completion_sha256": _sha(completion_body)}
    _publish_bytes(
        run_root / COMPLETION_FILENAME, _line(completion), label="Process-V2 Active8 completion"
    )
    return completion


def run_process_v2_active8_map(
    plan: Mapping[str, Any],
    *,
    submit: Callable[[tuple[str, ...]], Iterable[Any]],
    max_map_containers: int,
    repo_root: Path | None = None,
    set_autoscaler: Callable[..., Any] | None = None,
) -> list[Any]:
    """Submit every planned task under the shared bounded wave schedule.

    The wave planner lives in the chunk-cache module and is shared by every
    Process-V2 launcher, so the platform container ceiling has one home and
    cannot drift between two apps.
    """

    validated = validate_process_v2_active8_plan(plan, repo_root=repo_root)
    validate_map_container_bound(max_map_containers)
    identities = [task["task_identity_sha256"] for task in validated["tasks"]]
    return run_bounded_map(
        identities,
        max_map_containers=max_map_containers,
        submit=submit,
        set_autoscaler=set_autoscaler,
    )


def iter_process_v2_active8_decision_rows(
    plan: Mapping[str, Any], *, artifact_root: Path, repo_root: Path
) -> Iterable[dict[str, Any]]:
    """Stream every published decision row once, in the reduction order."""

    validated = validate_process_v2_active8_plan(plan, repo_root=repo_root)
    by_id = {task["task_identity_sha256"]: task for task in validated["tasks"]}
    for task in reduction_order(validated["tasks"]):
        output = _mounted(
            task["output_artifact_path"],
            artifact_root=artifact_root,
            field="task.output_artifact_path",
        )
        validate_process_v2_active8_task_result(
            output, plan=validated, task=by_id[task["task_identity_sha256"]]
        )
        raw = gzip.decompress((output / DECISION_FILENAME).read_bytes())
        yield from read_process_v2_active8_decision_rows(raw, task=task, plan=validated)


__all__ = [
    "COMPLETION_FILENAME",
    "COMPLETION_SCHEMA",
    "COMPLETION_SCHEMA_VERSION",
    "COMPLETION_STATUS",
    "COUNT_FIELDS",
    "DECISION_FILENAME",
    "DECISION_SCHEMA",
    "DECISION_SCHEMA_VERSION",
    "DEFAULT_OUTPUT_ARTIFACT_PREFIX",
    "DEFAULT_REPO_ROOT",
    "IMPLEMENTATION_FILES",
    "IMPLEMENTATION_REVISION_SCHEMA",
    "IMPLEMENTATION_REVISION_SCHEMA_VERSION",
    "JOIN_KEY_FIELDS",
    "PLAN_FILENAME",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "PLAN_STATUS",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "RECEIPT_SCHEMA_VERSION",
    "RECEIPT_STATUS",
    "TASK_DIRNAME",
    "ProcessV2Active8Incomplete",
    "ProcessV2Active8MapReduceError",
    "bind_process_v2_active8_sources",
    "build_process_v2_active8_implementation_revision",
    "completed_process_v2_active8_task_ids",
    "execute_process_v2_active8_decision_task",
    "iter_process_v2_active8_decision_rows",
    "load_process_v2_active8_plan",
    "plan_process_v2_active8_decisions",
    "read_process_v2_active8_decision_rows",
    "read_upstream_rebind_decisions",
    "reduce_process_v2_active8_decisions",
    "reduction_order",
    "run_process_v2_active8_map",
    "structural_census",
    "validate_model_runtime_identity",
    "validate_process_v2_active8_implementation_revision",
    "validate_process_v2_active8_plan",
    "validate_process_v2_active8_task_result",
    "write_process_v2_active8_plan",
]
