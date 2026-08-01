"""Verified read-only index over completed semantic Active8 decisions.

This module is the boundary between the non-authorizing Active8 decision
map/reduce and future Gate 0 or T1 consumers.  It revalidates the semantic
migration, mechanical chunk cache, decision plan, every decision receipt, and
the exact persistent-slot source rows.  It then exposes a compact O(1)
whole-trace admission index and bounded streaming readers for accepted and
excluded traces.

No artifact produced or returned here grants Gate 0, T1, P50, training,
checkpoint-selection, or final-test authority.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data import (
    editing_v2_semantic_active8_decision_mapreduce as decision_mr,
)
from compose_v4.data.editing_v2_semantic_active8_admission import (
    SemanticActionClassification,
    SemanticActive8ActionDecision,
    SemanticActive8Exclusion,
    SemanticActive8TraceDecision,
    SemanticExactCandidateEvidence,
    build_semantic_active8_admission_policy,
    classify_semantic_action,
)
from compose_v4.data.editing_v2_semantic_active8_source_adapter import (
    resolve_editing_v2_semantic_active8_sources,
)
from compose_v4.data.packed_trace_store import AddressedPackedTrace, PackedTraceAddress
from compose_v4.data.semantic_active8_chunk_cache import (
    SemanticActive8ChunkCacheError,
    read_semantic_active8_chunk,
)
from compose_v4.data.semantic_active8_chunk_cache_mapreduce import (
    GLOBAL_COMPLETION_FILENAME as CHUNK_GLOBAL_COMPLETION_FILENAME,
)
from compose_v4.data.semantic_active8_chunk_cache_mapreduce import (
    SemanticActive8ChunkCacheMapReduceError,
    reduce_semantic_active8_chunk_caches,
)
from compose_v4.rewrite.kernel import canonical_state_key

INDEX_SCHEMA = "compose.data.editing_v2_semantic_active8_decision_source_index"
INDEX_SCHEMA_VERSION = 1
INDEX_STATUS = "VERIFIED_ACTIVE8_DECISION_SOURCE_NO_DOWNSTREAM_AUTHORITY"
INDEX_ENCODING = "canonical_json_line_stream_v1"
MAX_DECISION_SOURCE_CHUNK_ROWS = 1_000
MAX_DECISION_SOURCE_CHUNK_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DECISION_SOURCE_CHUNK_UNCOMPRESSED_BYTES = 256 * 1024 * 1024

_AUTHORITY = {
    "training_authorized": False,
    "gate_zero_authorized": False,
    "t1_authorized": False,
    "bounded_p50_authorized": False,
    "long_training_authorized": False,
    "checkpoint_selection_authorized": False,
    "final_test_selection_authorized": False,
}
_DECISION_ROW_FIELDS = {
    "schema",
    "schema_version",
    "address",
    "policy_sha256",
    "semantic_migration_status",
    "semantic_migration_rejection",
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
_ACTION_FIELDS = {
    "step_index",
    "executor_rule",
    "model_family",
    "action_sha256",
    "policy_eligible",
    "exclusion_reason",
    "candidate_evidence",
}
_CANDIDATE_FIELDS = {
    "supported",
    "action_sha256",
    "source_state_sha256",
    "target_state_sha256",
    "canonical_successor_key",
    "raw_mark_count",
    "canonical_successor_count",
    "matching_mark_count",
    "successor_alias_count",
    "exclusion_reason",
}
_EXCLUSION_FIELDS = {
    "stage",
    "step_index",
    "executor_rule",
    "model_family",
    "reason",
}
_PROGRESS_FIELDS = {"progress_index", "terminal", "state_sha256"}

TraceKey = tuple[str, int, str]
ProgressKey = tuple[str, int, str, int]


class SemanticActive8DecisionSourceError(RuntimeError):
    """A supplied Active8 decision source is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class SemanticActive8ProgressAddress:
    """Exact persistent-slot progress identity inside one semantic trace."""

    packed_shard_content_sha256: str
    packed_shard_name: str
    entry_index: int
    trace_id: str
    data_lane: str
    partition_role: str
    progress_index: int
    terminal: bool
    state_sha256: str

    @property
    def trace_key(self) -> TraceKey:
        return (
            self.packed_shard_content_sha256,
            self.entry_index,
            self.trace_id,
        )

    @property
    def key(self) -> ProgressKey:
        return (*self.trace_key, self.progress_index)

    def as_payload(self) -> dict[str, object]:
        return {
            "packed_shard_content_sha256": self.packed_shard_content_sha256,
            "packed_shard_name": self.packed_shard_name,
            "entry_index": self.entry_index,
            "trace_id": self.trace_id,
            "data_lane": self.data_lane,
            "partition_role": self.partition_role,
            "progress_index": self.progress_index,
            "terminal": self.terminal,
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticActive8AcceptedTrace:
    """One exact admitted trace and all progress addresses it may expose."""

    addressed_trace: AddressedPackedTrace
    decision_sha256: str
    decision: SemanticActive8TraceDecision
    action_decisions: tuple[SemanticActive8ActionDecision, ...]
    progress_addresses: tuple[SemanticActive8ProgressAddress, ...]


@dataclass(frozen=True, slots=True)
class SemanticActive8AcceptedTransition:
    """One exact nonterminal teacher transition from an admitted trace."""

    decision_source_inventory_sha256: str
    addressed_trace: AddressedPackedTrace
    decision_sha256: str
    decision: SemanticActive8TraceDecision
    step_index: int
    action_decision: SemanticActive8ActionDecision
    source_progress_address: SemanticActive8ProgressAddress
    successor_progress_address: SemanticActive8ProgressAddress


@dataclass(frozen=True, slots=True)
class SemanticActive8ExcludedTrace:
    """One exact excluded trace, retained separately with its reasons."""

    addressed_trace: AddressedPackedTrace
    decision_sha256: str
    decision: SemanticActive8TraceDecision
    action_decisions: tuple[SemanticActive8ActionDecision, ...]
    exclusions: tuple[SemanticActive8Exclusion, ...]


@dataclass(frozen=True, slots=True)
class EditingV2SemanticActive8DecisionIndex:
    """Compact admission index with bounded exact-state streaming readers."""

    status: str
    training_authorized: bool
    gate_zero_authorized: bool
    t1_authorized: bool
    bounded_p50_authorized: bool
    long_training_authorized: bool
    checkpoint_selection_authorized: bool
    final_test_selection_authorized: bool
    artifact_root: Path
    migration_completion_file_sha256: str
    migration_completion_sha256: str
    chunk_cache_plan_file_sha256: str
    chunk_cache_plan_sha256: str
    chunk_cache_global_pointer_file_sha256: str
    chunk_cache_global_pointer_sha256: str
    chunk_cache_global_completion_file_sha256: str
    chunk_cache_global_completion_sha256: str
    decision_plan_file_sha256: str
    decision_plan_sha256: str
    decision_run_identity_sha256: str
    decision_completion_file_sha256: str
    decision_completion_sha256: str
    decision_result_inventory_sha256: str
    process_identity_sha256: str
    policy_sha256: str
    model_runtime_identity_sha256: str
    source_identity_sha256: str
    accepted_trace_inventory_sha256: str
    accepted_progress_inventory_sha256: str
    excluded_trace_inventory_sha256: str
    trace_decision_inventory_sha256: str
    decision_lookup_inventory_sha256: str
    exclusion_lookup_inventory_sha256: str
    decision_source_implementation_sha256: str
    counts: tuple[tuple[str, int], ...]
    action_family_histogram: tuple[tuple[str, int], ...]
    active8_exclusion_reason_histogram: tuple[tuple[str, int], ...]
    migration_rejection_histogram: tuple[tuple[str, int], ...]
    inventory_sha256: str
    _decision_plan_bytes: bytes = field(repr=False, compare=False)
    _decisions_by_shard: Mapping[
        str,
        tuple[tuple[str, bool, str, str], ...],
    ] = field(
        repr=False,
        compare=False,
    )
    _exclusions_by_key: Mapping[TraceKey, tuple[SemanticActive8Exclusion, ...]] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.status != INDEX_STATUS or any(
            getattr(self, field_name) is not expected
            for field_name, expected in _AUTHORITY.items()
        ):
            raise ValueError(
                "semantic Active8 decision index crosses its authority boundary"
            )
        object.__setattr__(self, "artifact_root", Path(self.artifact_root).resolve())
        object.__setattr__(
            self,
            "_decisions_by_shard",
            MappingProxyType(dict(self._decisions_by_shard)),
        )
        object.__setattr__(
            self,
            "_exclusions_by_key",
            MappingProxyType(dict(self._exclusions_by_key)),
        )
        sha_fields = (
            "migration_completion_file_sha256",
            "migration_completion_sha256",
            "chunk_cache_plan_file_sha256",
            "chunk_cache_plan_sha256",
            "chunk_cache_global_pointer_file_sha256",
            "chunk_cache_global_pointer_sha256",
            "chunk_cache_global_completion_file_sha256",
            "chunk_cache_global_completion_sha256",
            "decision_plan_file_sha256",
            "decision_plan_sha256",
            "decision_run_identity_sha256",
            "decision_completion_file_sha256",
            "decision_completion_sha256",
            "decision_result_inventory_sha256",
            "process_identity_sha256",
            "policy_sha256",
            "model_runtime_identity_sha256",
            "source_identity_sha256",
            "accepted_trace_inventory_sha256",
            "accepted_progress_inventory_sha256",
            "excluded_trace_inventory_sha256",
            "trace_decision_inventory_sha256",
            "decision_lookup_inventory_sha256",
            "exclusion_lookup_inventory_sha256",
            "decision_source_implementation_sha256",
            "inventory_sha256",
        )
        if any(not _is_sha256(getattr(self, name)) for name in sha_fields):
            raise ValueError("semantic Active8 decision index has an invalid SHA-256")
        if not isinstance(self._decision_plan_bytes, bytes):
            raise TypeError("semantic Active8 cached decision plan must be bytes")
        try:
            cached_plan = json.loads(self._decision_plan_bytes)
            validated_cached_plan = decision_mr._validate_plan(cached_plan)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            decision_mr.SemanticActive8DecisionMapReduceError,
        ) as error:
            raise ValueError(
                "semantic Active8 cached decision plan is invalid"
            ) from error
        if (
            self._decision_plan_bytes
            != _canonical_bytes(validated_cached_plan, newline=True)
            or hashlib.sha256(self._decision_plan_bytes).hexdigest()
            != self.decision_plan_file_sha256
            or validated_cached_plan["plan_sha256"] != self.decision_plan_sha256
            or validated_cached_plan["run_identity_sha256"]
            != self.decision_run_identity_sha256
        ):
            raise ValueError("semantic Active8 cached decision plan identity disagrees")
        for field_name in (
            "counts",
            "action_family_histogram",
            "active8_exclusion_reason_histogram",
            "migration_rejection_histogram",
        ):
            items = getattr(self, field_name)
            if (
                tuple(sorted(items)) != items
                or len(dict(items)) != len(items)
                or any(
                    not isinstance(key, str)
                    or not key
                    or type(value) is not int
                    or value < 0
                    for key, value in items
                )
            ):
                raise ValueError(
                    f"semantic Active8 decision index {field_name} is invalid"
                )
        count_map = dict(self.counts)
        accepted = 0
        excluded_keys: set[TraceKey] = set()
        trace_count = 0
        for shard_sha, entries in self._decisions_by_shard.items():
            if not _is_sha256(shard_sha) or not isinstance(entries, tuple):
                raise ValueError("semantic Active8 decision lookup shard is invalid")
            for entry_index, entry in enumerate(entries):
                if (
                    not isinstance(entry, tuple)
                    or len(entry) != 4
                    or not isinstance(entry[0], str)
                    or not entry[0]
                    or type(entry[1]) is not bool
                    or not _is_sha256(entry[2])
                    or not _is_sha256(entry[3])
                ):
                    raise ValueError(
                        "semantic Active8 decision lookup entry is invalid"
                    )
                trace_count += 1
                if entry[1]:
                    accepted += 1
                else:
                    excluded_keys.add((shard_sha, entry_index, entry[0]))
        if (
            trace_count != count_map.get("traces")
            or accepted != count_map.get("accepted_traces")
            or len(excluded_keys) != count_map.get("excluded_traces")
            or accepted + len(excluded_keys) != trace_count
            or set(self._exclusions_by_key) != excluded_keys
            or any(
                not isinstance(exclusions, tuple)
                or not exclusions
                or any(
                    not isinstance(exclusion, SemanticActive8Exclusion)
                    for exclusion in exclusions
                )
                for exclusions in self._exclusions_by_key.values()
            )
        ):
            raise ValueError("semantic Active8 decision lookup census is inconsistent")
        if (
            _decision_lookup_sha256(self._decisions_by_shard)
            != self.decision_lookup_inventory_sha256
        ):
            raise ValueError("semantic Active8 decision lookup hash disagrees")
        if (
            _exclusion_lookup_sha256(self._exclusions_by_key)
            != self.exclusion_lookup_inventory_sha256
        ):
            raise ValueError("semantic Active8 exclusion lookup hash disagrees")
        if self.inventory_sha256 != _sha(_index_identity_body(self)):
            raise ValueError("semantic Active8 decision index self-hash disagrees")

    @property
    def count_map(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.counts))

    def identity_payload(self) -> dict[str, object]:
        """Return the deterministic, non-authorizing downstream source identity."""

        body = _index_identity_body(self)
        return {**body, "inventory_sha256": self.inventory_sha256}

    def _decision_entry(
        self,
        address: PackedTraceAddress,
    ) -> tuple[bool, str]:
        entries = self._decisions_by_shard.get(address.packed_shard_content_sha256)
        if entries is None or not 0 <= address.entry_index < len(entries):
            raise SemanticActive8DecisionSourceError(
                "trace address is absent from the semantic Active8 decision index"
            )
        trace_id, accepted, address_sha256, decision_sha256 = entries[
            address.entry_index
        ]
        if trace_id != address.trace_id or address_sha256 != _sha(
            _address_payload(address)
        ):
            raise SemanticActive8DecisionSourceError(
                "trace identity differs at the indexed semantic source address"
            )
        return accepted, decision_sha256

    def is_accepted(self, address: PackedTraceAddress) -> bool:
        """Return the exact whole-trace decision, rejecting identity drift."""

        accepted, _ = self._decision_entry(address)
        return accepted

    def decision_sha256_for(self, address: PackedTraceAddress) -> str:
        """Return the self-hash of the exactly indexed decision receipt."""

        _, decision_sha256 = self._decision_entry(address)
        return decision_sha256

    def exclusions_for(
        self,
        address: PackedTraceAddress,
    ) -> tuple[SemanticActive8Exclusion, ...]:
        """Return exclusions only for an exactly indexed excluded trace."""

        accepted = self.is_accepted(address)
        key = _trace_key(address)
        exclusions = self._exclusions_by_key.get(key, ())
        if accepted and exclusions:
            raise SemanticActive8DecisionSourceError(
                "accepted trace carries indexed Active8 exclusions"
            )
        if not accepted and not exclusions:
            raise SemanticActive8DecisionSourceError(
                "excluded trace is missing its Active8 reasons"
            )
        return exclusions

    def iter_accepted_traces(self) -> Iterable[SemanticActive8AcceptedTrace]:
        """Stream exact admitted traces without SMILES reconstruction."""

        for accepted, resolved in _iter_resolved_traces(self):
            if accepted:
                if not isinstance(resolved, SemanticActive8AcceptedTrace):
                    raise SemanticActive8DecisionSourceError(
                        "accepted trace stream returned an excluded record"
                    )
                yield resolved

    def iter_excluded_traces(self) -> Iterable[SemanticActive8ExcludedTrace]:
        """Stream exact excluded traces and their reasons separately."""

        for accepted, resolved in _iter_resolved_traces(self):
            if not accepted:
                if not isinstance(resolved, SemanticActive8ExcludedTrace):
                    raise SemanticActive8DecisionSourceError(
                        "excluded trace stream returned an accepted record"
                    )
                yield resolved

    def iter_accepted_progress_addresses(
        self,
    ) -> Iterable[SemanticActive8ProgressAddress]:
        """Stream every exact admitted progress address in source order."""

        for trace in self.iter_accepted_traces():
            yield from trace.progress_addresses

    def iter_accepted_nonterminal_transitions(
        self,
    ) -> Iterable[SemanticActive8AcceptedTransition]:
        """Stream one verified teacher transition per accepted action."""

        for trace in self.iter_accepted_traces():
            path_length = trace.addressed_trace.address.path_length
            if (
                len(trace.addressed_trace.trace.steps) != path_length
                or len(trace.action_decisions) != path_length
                or len(trace.progress_addresses) != path_length + 1
                or self.decision_sha256_for(trace.addressed_trace.address)
                != trace.decision_sha256
            ):
                raise SemanticActive8DecisionSourceError(
                    "accepted trace cannot produce a complete transition stream"
                )
            for step_index, action_decision in enumerate(trace.action_decisions):
                source = trace.progress_addresses[step_index]
                successor = trace.progress_addresses[step_index + 1]
                if (
                    source.progress_index != step_index
                    or source.terminal
                    or successor.progress_index != step_index + 1
                    or source.trace_key != successor.trace_key
                ):
                    raise SemanticActive8DecisionSourceError(
                        "accepted transition progress identities are inconsistent"
                    )
                yield SemanticActive8AcceptedTransition(
                    decision_source_inventory_sha256=self.inventory_sha256,
                    addressed_trace=trace.addressed_trace,
                    decision_sha256=trace.decision_sha256,
                    decision=trace.decision,
                    step_index=step_index,
                    action_decision=action_decision,
                    source_progress_address=source,
                    successor_progress_address=successor,
                )


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _decision_lookup_sha256(
    decisions_by_shard: Mapping[str, tuple[tuple[str, bool, str, str], ...]],
) -> str:
    payload = [
        {
            "packed_shard_content_sha256": shard_sha,
            "entries": [
                {
                    "entry_index": entry_index,
                    "trace_id": trace_id,
                    "accepted": accepted,
                    "address_sha256": address_sha256,
                    "decision_sha256": decision_sha256,
                }
                for entry_index, (
                    trace_id,
                    accepted,
                    address_sha256,
                    decision_sha256,
                ) in enumerate(entries)
            ],
        }
        for shard_sha, entries in sorted(decisions_by_shard.items())
    ]
    return _sha(payload)


def _exclusion_lookup_sha256(
    exclusions_by_key: Mapping[TraceKey, tuple[SemanticActive8Exclusion, ...]],
) -> str:
    payload = [
        {
            "packed_shard_content_sha256": shard_sha,
            "entry_index": entry_index,
            "trace_id": trace_id,
            "exclusions": [
                {
                    "stage": exclusion.stage,
                    "step_index": exclusion.step_index,
                    "executor_rule": exclusion.executor_rule,
                    "model_family": exclusion.model_family,
                    "reason": exclusion.reason,
                }
                for exclusion in exclusions
            ],
        }
        for (shard_sha, entry_index, trace_id), exclusions in sorted(
            exclusions_by_key.items()
        )
    ]
    return _sha(payload)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_canonical_object(
    path: Path, *, field_name: str
) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticActive8DecisionSourceError(
            f"{field_name} is unreadable: {path}"
        ) from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value, newline=True):
        raise SemanticActive8DecisionSourceError(
            f"{field_name} is not a canonical JSON object"
        )
    return value, raw


def _require_inside(path: Path, *, artifact_root: Path, field_name: str) -> Path:
    root = Path(artifact_root).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise SemanticActive8DecisionSourceError(
            f"{field_name} resolves outside artifact_root"
        )
    return resolved


def _mounted_artifact_path(
    value: object,
    *,
    artifact_root: Path,
    field_name: str,
) -> Path:
    if not isinstance(value, str):
        raise SemanticActive8DecisionSourceError(
            f"{field_name} must be a normalized /artifacts path"
        )
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or len(pure.parts) < 3
        or pure.parts[1] != "artifacts"
        or ".." in pure.parts
        or "\\" in value
        or str(pure) != value
        or value.endswith("/")
    ):
        raise SemanticActive8DecisionSourceError(
            f"{field_name} must be a normalized /artifacts path"
        )
    return _require_inside(
        Path(artifact_root) / Path(*pure.parts[2:]),
        artifact_root=artifact_root,
        field_name=field_name,
    )


def _trace_key(address: PackedTraceAddress) -> TraceKey:
    return (
        address.packed_shard_content_sha256,
        address.entry_index,
        address.trace_id,
    )


def _address_payload(address: PackedTraceAddress) -> dict[str, object]:
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


def _stream_update(digest: Any, payload: Mapping[str, object]) -> None:
    digest.update(_canonical_bytes(dict(payload), newline=True))


def _expected_input_path(path: Path, expected: Path, *, field_name: str) -> None:
    if path != expected.resolve():
        raise SemanticActive8DecisionSourceError(
            f"{field_name} is not the exact published artifact for its plan"
        )


def _load_verified_chunk_completion(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    global_pointer_path: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    output_root = _mounted_artifact_path(
        plan.get("output_artifact_root"),
        artifact_root=artifact_root,
        field_name="chunk_cache_plan.output_artifact_root",
    )
    expected_run_root = (
        output_root / "global_runs" / str(plan.get("run_identity_sha256"))
    )
    _expected_input_path(
        plan_path,
        expected_run_root / "PLAN.json",
        field_name="chunk_cache_plan_path",
    )
    _expected_input_path(
        global_pointer_path,
        expected_run_root / CHUNK_GLOBAL_COMPLETION_FILENAME,
        field_name="chunk_cache_global_completion_path",
    )
    pointer, pointer_bytes = _load_canonical_object(
        global_pointer_path,
        field_name="chunk-cache global completion pointer",
    )
    relative = pointer.get("global_completion_object_path")
    if not isinstance(relative, str):
        raise SemanticActive8DecisionSourceError(
            "chunk-cache pointer lacks its completion object path"
        )
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
        raise SemanticActive8DecisionSourceError(
            "chunk-cache completion object path is not normalized"
        )
    completion_path = _require_inside(
        output_root / Path(*pure.parts),
        artifact_root=artifact_root,
        field_name="chunk-cache completion object",
    )
    completion, completion_bytes = _load_canonical_object(
        completion_path,
        field_name="chunk-cache global completion object",
    )
    if _file_sha(completion_path) != pointer.get(
        "global_completion_file_sha256"
    ) or completion.get("completion_sha256") != pointer.get("global_completion_sha256"):
        raise SemanticActive8DecisionSourceError(
            "chunk-cache completion bytes or identity disagree"
        )
    # Only after every supplied completion byte is present do we invoke the
    # idempotent reducer as an independent receipt and lineage validator.  This
    # prevents the read boundary from repairing a missing completion object.
    reduced = reduce_semantic_active8_chunk_caches(plan, artifact_root=artifact_root)
    reduced_pointer = {
        key: value for key, value in reduced.items() if key != "completion"
    }
    if pointer != reduced_pointer or completion != reduced["completion"]:
        raise SemanticActive8DecisionSourceError(
            "supplied chunk-cache completion differs from verified reduction"
        )
    return pointer, pointer_bytes, completion, completion_bytes


def _candidate_evidence(
    value: object,
    *,
    classification: SemanticActionClassification,
    addressed: AddressedPackedTrace,
) -> SemanticExactCandidateEvidence | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _CANDIDATE_FIELDS:
        raise SemanticActive8DecisionSourceError(
            "decision candidate-evidence fields disagree"
        )
    try:
        evidence = SemanticExactCandidateEvidence(**dict(value))
    except (TypeError, ValueError) as error:
        raise SemanticActive8DecisionSourceError(
            "decision candidate evidence is invalid"
        ) from error
    source_sha = persistent_slot_state_sha256(
        addressed.path.state_at(classification.step_index)
    )
    target_sha = persistent_slot_state_sha256(
        addressed.path.state_at(classification.step_index + 1)
    )
    if (
        evidence.action_sha256 != classification.action_sha256
        or evidence.source_state_sha256 != source_sha
        or evidence.target_state_sha256 != target_sha
        or evidence.canonical_successor_key
        != canonical_state_key(addressed.path.state_at(classification.step_index + 1))
    ):
        raise SemanticActive8DecisionSourceError(
            "candidate evidence differs from its exact action or successor"
        )
    if (
        evidence.raw_mark_count < evidence.canonical_successor_count
        or evidence.raw_mark_count < evidence.matching_mark_count
        or evidence.raw_mark_count < evidence.successor_alias_count
        or (
            evidence.supported
            and evidence.successor_alias_count < evidence.matching_mark_count
        )
        or (evidence.supported and evidence.canonical_successor_count == 0)
    ):
        raise SemanticActive8DecisionSourceError(
            "candidate evidence counts are internally inconsistent"
        )
    return evidence


def _typed_decision(
    row: Mapping[str, Any],
    addressed: AddressedPackedTrace,
) -> tuple[
    SemanticActive8TraceDecision,
    tuple[SemanticActive8ProgressAddress, ...],
]:
    if set(row) != _DECISION_ROW_FIELDS:
        raise SemanticActive8DecisionSourceError("decision row fields disagree")
    row_body = {key: value for key, value in row.items() if key != "decision_sha256"}
    if row.get("decision_sha256") != _sha(row_body):
        raise SemanticActive8DecisionSourceError("decision row self-hash disagrees")
    if row.get("address") != _address_payload(addressed.address):
        raise SemanticActive8DecisionSourceError(
            "decision row differs from its exact original source address"
        )
    actions = row.get("actions")
    if not isinstance(actions, list) or len(actions) != addressed.address.path_length:
        raise SemanticActive8DecisionSourceError(
            "decision action list does not cover the exact trace"
        )
    policy = build_semantic_active8_admission_policy()
    action_decisions: list[SemanticActive8ActionDecision] = []
    classifications: list[SemanticActionClassification] = []
    for index, (value, step) in enumerate(
        zip(actions, addressed.trace.steps, strict=True)
    ):
        if not isinstance(value, Mapping) or set(value) != _ACTION_FIELDS:
            raise SemanticActive8DecisionSourceError("decision action fields disagree")
        expected = classify_semantic_action(step, step_index=index, policy=policy)
        observed = {
            "step_index": value["step_index"],
            "executor_rule": value["executor_rule"],
            "model_family": value["model_family"],
            "action_sha256": value["action_sha256"],
            "policy_eligible": value["policy_eligible"],
            "exclusion_reason": value["exclusion_reason"],
        }
        expected_payload = {
            "step_index": expected.step_index,
            "executor_rule": expected.executor_rule,
            "model_family": expected.model_family,
            "action_sha256": expected.action_sha256,
            "policy_eligible": expected.policy_eligible,
            "exclusion_reason": expected.exclusion_reason,
        }
        if observed != expected_payload:
            raise SemanticActive8DecisionSourceError(
                "decision action classification differs from ActionCodecV4"
            )
        evidence = _candidate_evidence(
            value["candidate_evidence"],
            classification=expected,
            addressed=addressed,
        )
        classifications.append(expected)
        action_decisions.append(
            SemanticActive8ActionDecision(
                classification=expected,
                candidate_evidence=evidence,
            )
        )
    any_policy_exclusion = any(not item.policy_eligible for item in classifications)
    if any_policy_exclusion:
        if any(item.candidate_evidence is not None for item in action_decisions):
            raise SemanticActive8DecisionSourceError(
                "policy-excluded trace carries production candidate evidence"
            )
    elif any(item.candidate_evidence is None for item in action_decisions):
        raise SemanticActive8DecisionSourceError(
            "eligible trace is missing production candidate evidence"
        )

    exclusions_payload = row.get("active8_exclusions")
    if not isinstance(exclusions_payload, list):
        raise SemanticActive8DecisionSourceError("Active8 exclusions must be an array")
    exclusions: list[SemanticActive8Exclusion] = []
    for value in exclusions_payload:
        if not isinstance(value, Mapping) or set(value) != _EXCLUSION_FIELDS:
            raise SemanticActive8DecisionSourceError(
                "Active8 exclusion fields disagree"
            )
        try:
            exclusions.append(SemanticActive8Exclusion(**dict(value)))
        except (TypeError, ValueError) as error:
            raise SemanticActive8DecisionSourceError(
                "Active8 exclusion is invalid"
            ) from error
    expected_exclusions: list[SemanticActive8Exclusion] = []
    if any_policy_exclusion:
        expected_exclusions.extend(
            SemanticActive8Exclusion(
                stage="action_v4_policy",
                step_index=item.step_index,
                executor_rule=item.executor_rule,
                model_family=item.model_family,
                reason=str(item.exclusion_reason),
            )
            for item in classifications
            if not item.policy_eligible
        )
    else:
        expected_exclusions.extend(
            SemanticActive8Exclusion(
                stage="production_candidate_support",
                step_index=item.classification.step_index,
                executor_rule=item.classification.executor_rule,
                model_family=item.classification.model_family,
                reason=str(item.candidate_evidence.exclusion_reason),
            )
            for item in action_decisions
            if item.candidate_evidence is not None
            and not item.candidate_evidence.supported
        )
    if exclusions != expected_exclusions:
        raise SemanticActive8DecisionSourceError(
            "Active8 exclusions differ from action and candidate evidence"
        )
    try:
        decision = SemanticActive8TraceDecision(
            trace_id=addressed.address.trace_id,
            policy_sha256=str(row.get("policy_sha256")),
            semantic_migration_status=row.get("semantic_migration_status"),
            semantic_migration_rejection=row.get("semantic_migration_rejection"),
            active8_status=row.get("active8_status"),
            emits_progress_rows=row.get("emits_progress_rows"),
            action_decisions=tuple(action_decisions),
            active8_exclusions=tuple(exclusions),
        )
    except (TypeError, ValueError) as error:
        raise SemanticActive8DecisionSourceError(
            "whole-trace Active8 decision is invalid"
        ) from error
    if decision.policy_sha256 != policy.policy_sha256:
        raise SemanticActive8DecisionSourceError(
            "decision row names another semantic Active8 policy"
        )

    progress_payload = row.get("progress_rows")
    if not isinstance(progress_payload, list):
        raise SemanticActive8DecisionSourceError("progress rows must be an array")
    progress: list[SemanticActive8ProgressAddress] = []
    for index, value in enumerate(progress_payload):
        if not isinstance(value, Mapping) or set(value) != _PROGRESS_FIELDS:
            raise SemanticActive8DecisionSourceError("progress-row fields disagree")
        state_sha = persistent_slot_state_sha256(addressed.path.state_at(index))
        terminal = index == addressed.address.path_length
        if (
            value.get("progress_index") != index
            or value.get("terminal") is not terminal
            or value.get("state_sha256") != state_sha
        ):
            raise SemanticActive8DecisionSourceError(
                "progress row differs from its exact persistent-slot state"
            )
        progress.append(
            SemanticActive8ProgressAddress(
                packed_shard_content_sha256=(
                    addressed.address.packed_shard_content_sha256
                ),
                packed_shard_name=addressed.address.packed_shard_name,
                entry_index=addressed.address.entry_index,
                trace_id=addressed.address.trace_id,
                data_lane=addressed.address.layer,
                partition_role=addressed.address.partition,
                progress_index=index,
                terminal=terminal,
                state_sha256=state_sha,
            )
        )
    expected_progress_count = (
        addressed.address.path_length + 1
        if decision.active8_status == "accepted"
        else 0
    )
    if len(progress) != expected_progress_count:
        raise SemanticActive8DecisionSourceError(
            "whole-trace decision exposes the wrong progress-row count"
        )
    return decision, tuple(progress)


def _iter_plan_rows(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> Iterable[tuple[AddressedPackedTrace, Mapping[str, Any], Mapping[str, Any]]]:
    for task in plan["tasks"]:
        if (
            type(task.get("row_count")) is not int
            or not 0 <= task["row_count"] <= MAX_DECISION_SOURCE_CHUNK_ROWS
        ):
            raise SemanticActive8DecisionSourceError(
                "decision task exceeds the bounded source chunk size"
            )
        try:
            receipt, rows = decision_mr._load_task_result(
                plan,
                task,
                artifact_root=artifact_root,
            )
        except decision_mr.SemanticActive8DecisionMapReduceError as error:
            raise SemanticActive8DecisionSourceError(
                "decision task receipt failed revalidation"
            ) from error
        source_rows = read_semantic_active8_chunk(
            _mounted_artifact_path(
                task["cache_root"],
                artifact_root=artifact_root,
                field_name="decision task cache_root",
            ),
            receipt_object_path=task["chunk_receipt_object_path"],
            expected_receipt_file_sha256=task["chunk_receipt_file_sha256"],
            expected_source_shard_sha256=task["source_shard_sha256"],
            expected_source_manifest_sha256=task["source_manifest_file_sha256"],
        )
        observed = 0
        try:
            for addressed, row in zip(source_rows, rows, strict=True):
                observed += 1
                yield addressed, row, receipt
        except (SemanticActive8ChunkCacheError, ValueError) as error:
            raise SemanticActive8DecisionSourceError(
                "decision rows and exact source chunk have different lengths"
            ) from error
        if observed != task["row_count"]:
            raise SemanticActive8DecisionSourceError(
                "decision task does not cover its exact chunk row count"
            )


def _validate_bounded_gzip_artifact(path: Path, *, field_name: str) -> None:
    try:
        compressed_size = path.stat().st_size
    except OSError as error:
        raise SemanticActive8DecisionSourceError(
            f"{field_name} is unreadable: {path}"
        ) from error
    if not path.is_file():
        raise SemanticActive8DecisionSourceError(f"{field_name} is not a file: {path}")
    if compressed_size > MAX_DECISION_SOURCE_CHUNK_COMPRESSED_BYTES:
        raise SemanticActive8DecisionSourceError(
            f"{field_name} exceeds the bounded compressed byte size"
        )
    uncompressed_size = 0
    try:
        with gzip.open(path, "rb") as handle:
            while block := handle.read(1 << 20):
                uncompressed_size += len(block)
                if uncompressed_size > MAX_DECISION_SOURCE_CHUNK_UNCOMPRESSED_BYTES:
                    raise SemanticActive8DecisionSourceError(
                        f"{field_name} exceeds the bounded uncompressed byte size"
                    )
    except SemanticActive8DecisionSourceError:
        raise
    except (EOFError, OSError) as error:
        raise SemanticActive8DecisionSourceError(
            f"{field_name} failed revalidation because it is not readable gzip"
        ) from error


def _validate_bounded_decision_plan(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    try:
        validated = decision_mr._validate_plan(plan)
    except decision_mr.SemanticActive8DecisionMapReduceError as error:
        raise SemanticActive8DecisionSourceError(
            "semantic Active8 decision plan failed revalidation"
        ) from error
    for task in validated["tasks"]:
        if task["row_count"] > MAX_DECISION_SOURCE_CHUNK_ROWS:
            raise SemanticActive8DecisionSourceError(
                "decision task exceeds the bounded source chunk size"
            )
        cache_root = _mounted_artifact_path(
            task["cache_root"],
            artifact_root=artifact_root,
            field_name="decision task cache_root",
        )
        source_chunk_path = _require_inside(
            cache_root / task["chunk_object_path"],
            artifact_root=artifact_root,
            field_name="semantic source chunk",
        )
        decision_chunk_path = _require_inside(
            decision_mr._task_root(
                validated,
                task,
                artifact_root=artifact_root,
            )
            / decision_mr.DECISION_FILENAME,
            artifact_root=artifact_root,
            field_name="semantic decision chunk",
        )
        _validate_bounded_gzip_artifact(
            source_chunk_path,
            field_name="semantic source chunk",
        )
        _validate_bounded_gzip_artifact(
            decision_chunk_path,
            field_name="semantic decision chunk",
        )
    return validated


def _resolved_trace(
    addressed: AddressedPackedTrace,
    row: Mapping[str, Any],
) -> tuple[
    bool,
    SemanticActive8AcceptedTrace | SemanticActive8ExcludedTrace,
]:
    decision, progress = _typed_decision(row, addressed)
    accepted = decision.active8_status == "accepted"
    if accepted:
        return True, SemanticActive8AcceptedTrace(
            addressed_trace=addressed,
            decision_sha256=str(row["decision_sha256"]),
            decision=decision,
            action_decisions=decision.action_decisions,
            progress_addresses=progress,
        )
    return False, SemanticActive8ExcludedTrace(
        addressed_trace=addressed,
        decision_sha256=str(row["decision_sha256"]),
        decision=decision,
        action_decisions=decision.action_decisions,
        exclusions=decision.active8_exclusions,
    )


def _iter_resolved_traces(
    index: EditingV2SemanticActive8DecisionIndex,
) -> Iterable[tuple[bool, SemanticActive8AcceptedTrace | SemanticActive8ExcludedTrace]]:
    plan = json.loads(index._decision_plan_bytes)
    for addressed, row, _ in _iter_plan_rows(
        plan,
        artifact_root=index.artifact_root,
    ):
        accepted, resolved = _resolved_trace(addressed, row)
        if index.is_accepted(addressed.address) is not accepted:
            raise SemanticActive8DecisionSourceError(
                "streamed decision differs from the frozen admission index"
            )
        if (
            not accepted
            and index.exclusions_for(addressed.address) != resolved.exclusions
        ):
            raise SemanticActive8DecisionSourceError(
                "streamed exclusion reasons differ from the frozen index"
            )
        yield accepted, resolved


def _index_identity_body(
    index: EditingV2SemanticActive8DecisionIndex | Mapping[str, Any],
) -> dict[str, object]:
    def value(field_name: str) -> Any:
        if isinstance(index, Mapping):
            return index[field_name]
        return getattr(index, field_name)

    return {
        "schema": INDEX_SCHEMA,
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": value("status"),
        **_AUTHORITY,
        "encoding": INDEX_ENCODING,
        "migration_completion_file_sha256": value("migration_completion_file_sha256"),
        "migration_completion_sha256": value("migration_completion_sha256"),
        "chunk_cache_plan_file_sha256": value("chunk_cache_plan_file_sha256"),
        "chunk_cache_plan_sha256": value("chunk_cache_plan_sha256"),
        "chunk_cache_global_pointer_file_sha256": (
            value("chunk_cache_global_pointer_file_sha256")
        ),
        "chunk_cache_global_pointer_sha256": value("chunk_cache_global_pointer_sha256"),
        "chunk_cache_global_completion_file_sha256": (
            value("chunk_cache_global_completion_file_sha256")
        ),
        "chunk_cache_global_completion_sha256": (
            value("chunk_cache_global_completion_sha256")
        ),
        "decision_plan_file_sha256": value("decision_plan_file_sha256"),
        "decision_plan_sha256": value("decision_plan_sha256"),
        "decision_run_identity_sha256": value("decision_run_identity_sha256"),
        "decision_completion_file_sha256": value("decision_completion_file_sha256"),
        "decision_completion_sha256": value("decision_completion_sha256"),
        "decision_result_inventory_sha256": value("decision_result_inventory_sha256"),
        "process_identity_sha256": value("process_identity_sha256"),
        "policy_sha256": value("policy_sha256"),
        "model_runtime_identity_sha256": value("model_runtime_identity_sha256"),
        "source_identity_sha256": value("source_identity_sha256"),
        "accepted_trace_inventory_sha256": value("accepted_trace_inventory_sha256"),
        "accepted_progress_inventory_sha256": value(
            "accepted_progress_inventory_sha256"
        ),
        "excluded_trace_inventory_sha256": value("excluded_trace_inventory_sha256"),
        "trace_decision_inventory_sha256": value("trace_decision_inventory_sha256"),
        "decision_lookup_inventory_sha256": value("decision_lookup_inventory_sha256"),
        "exclusion_lookup_inventory_sha256": value("exclusion_lookup_inventory_sha256"),
        "decision_source_implementation_sha256": value(
            "decision_source_implementation_sha256"
        ),
        "counts": dict(value("counts")),
        "action_family_histogram": dict(value("action_family_histogram")),
        "active8_exclusion_reason_histogram": dict(
            value("active8_exclusion_reason_histogram")
        ),
        "migration_rejection_histogram": dict(value("migration_rejection_histogram")),
    }


def resolve_editing_v2_semantic_active8_decision_source(
    *,
    migration_completion_path: Path,
    chunk_cache_plan_path: Path,
    chunk_cache_global_completion_path: Path,
    decision_plan_path: Path,
    decision_completion_path: Path,
    artifact_root: Path,
    repo_root: Path,
) -> EditingV2SemanticActive8DecisionIndex:
    """Revalidate all upstream bytes and build a non-authorizing source index."""

    root = Path(artifact_root).resolve()
    migration_path = _require_inside(
        migration_completion_path,
        artifact_root=root,
        field_name="migration_completion_path",
    )
    chunk_plan_path = _require_inside(
        chunk_cache_plan_path,
        artifact_root=root,
        field_name="chunk_cache_plan_path",
    )
    chunk_completion_path = _require_inside(
        chunk_cache_global_completion_path,
        artifact_root=root,
        field_name="chunk_cache_global_completion_path",
    )
    active8_plan_path = _require_inside(
        decision_plan_path,
        artifact_root=root,
        field_name="decision_plan_path",
    )
    active8_completion_path = _require_inside(
        decision_completion_path,
        artifact_root=root,
        field_name="decision_completion_path",
    )
    source_inventory = resolve_editing_v2_semantic_active8_sources(
        migration_path,
        artifact_root=root,
        repo_root=Path(repo_root),
    )

    chunk_plan, chunk_plan_bytes = _load_canonical_object(
        chunk_plan_path,
        field_name="semantic Active8 chunk-cache plan",
    )
    try:
        pointer, pointer_bytes, chunk_completion, chunk_completion_bytes = (
            _load_verified_chunk_completion(
                plan=chunk_plan,
                plan_path=chunk_plan_path,
                global_pointer_path=chunk_completion_path,
                artifact_root=root,
            )
        )
    except SemanticActive8ChunkCacheMapReduceError as error:
        raise SemanticActive8DecisionSourceError(
            "semantic Active8 chunk-cache reduction failed revalidation"
        ) from error

    decision_plan, decision_plan_bytes = _load_canonical_object(
        active8_plan_path,
        field_name="semantic Active8 decision plan",
    )
    decision_plan = _validate_bounded_decision_plan(
        decision_plan,
        artifact_root=root,
    )
    decision_output_root = _mounted_artifact_path(
        decision_plan.get("output_artifact_root"),
        artifact_root=root,
        field_name="decision_plan.output_artifact_root",
    )
    expected_decision_root = (
        decision_output_root / "runs" / str(decision_plan.get("run_identity_sha256"))
    )
    _expected_input_path(
        active8_plan_path,
        expected_decision_root / decision_mr.PLAN_FILENAME,
        field_name="decision_plan_path",
    )
    _expected_input_path(
        active8_completion_path,
        expected_decision_root / decision_mr.COMPLETION_FILENAME,
        field_name="decision_completion_path",
    )
    supplied_completion, decision_completion_bytes = _load_canonical_object(
        active8_completion_path,
        field_name="semantic Active8 decision completion",
    )
    try:
        reduced_completion = decision_mr.reduce_semantic_active8_decisions(
            decision_plan,
            inventory=source_inventory,
            artifact_root=root,
        )
    except decision_mr.SemanticActive8DecisionMapReduceError as error:
        raise SemanticActive8DecisionSourceError(
            "semantic Active8 decision reduction failed revalidation"
        ) from error
    if supplied_completion != reduced_completion:
        raise SemanticActive8DecisionSourceError(
            "supplied Active8 completion differs from verified reduction"
        )
    if decision_plan.get(
        "chunk_cache_global_completion_sha256"
    ) != chunk_completion.get("completion_sha256") or decision_plan.get(
        "source_identity"
    ) != chunk_completion.get(
        "migration_identity"
    ):
        raise SemanticActive8DecisionSourceError(
            "decision plan, chunk cache, and migration name different sources"
        )

    accepted_trace_stream = hashlib.sha256()
    accepted_progress_stream = hashlib.sha256()
    excluded_trace_stream = hashlib.sha256()
    decision_stream = hashlib.sha256()
    decisions_by_shard: dict[str, list[tuple[str, bool, str, str]]] = {}
    exclusions_by_key: dict[TraceKey, tuple[SemanticActive8Exclusion, ...]] = {}
    observed = Counter()
    observed_families: Counter[str] = Counter()
    observed_reasons: Counter[str] = Counter()
    for addressed, row, _ in _iter_plan_rows(decision_plan, artifact_root=root):
        accepted, resolved = _resolved_trace(addressed, row)
        address = addressed.address
        key = _trace_key(address)
        shard = decisions_by_shard.setdefault(address.packed_shard_content_sha256, [])
        if address.entry_index != len(shard):
            raise SemanticActive8DecisionSourceError(
                "semantic Active8 index source addresses are not position-complete"
            )
        shard.append(
            (
                address.trace_id,
                accepted,
                _sha(_address_payload(address)),
                str(row["decision_sha256"]),
            )
        )
        trace_payload = {
            "address": _address_payload(address),
            "decision_sha256": row["decision_sha256"],
        }
        _stream_update(decision_stream, {**trace_payload, "accepted": accepted})
        observed["traces"] += 1
        observed["actions"] += len(row["actions"])
        observed_families.update(row["action_family_histogram"])
        if accepted:
            if not isinstance(resolved, SemanticActive8AcceptedTrace):
                raise SemanticActive8DecisionSourceError(
                    "accepted decision resolved to an excluded trace"
                )
            observed["accepted_traces"] += 1
            observed["progress_rows"] += len(resolved.progress_addresses)
            _stream_update(accepted_trace_stream, trace_payload)
            for progress in resolved.progress_addresses:
                _stream_update(accepted_progress_stream, progress.as_payload())
        else:
            if not isinstance(resolved, SemanticActive8ExcludedTrace):
                raise SemanticActive8DecisionSourceError(
                    "excluded decision resolved to an accepted trace"
                )
            observed["excluded_traces"] += 1
            exclusions_by_key[key] = resolved.exclusions
            reason_payloads = [
                {
                    "stage": item.stage,
                    "step_index": item.step_index,
                    "executor_rule": item.executor_rule,
                    "model_family": item.model_family,
                    "reason": item.reason,
                }
                for item in resolved.exclusions
            ]
            _stream_update(
                excluded_trace_stream,
                {**trace_payload, "active8_exclusions": reason_payloads},
            )
            observed_reasons.update(item.reason for item in resolved.exclusions)
        observed.update(row["candidate_totals"])

    expected_counts = supplied_completion["active8_counts"]
    normalized_counts = {key: observed[key] for key in expected_counts}
    if (
        normalized_counts != expected_counts
        or dict(sorted(observed_families.items()))
        != supplied_completion["active8_action_family_histogram"]
        or dict(sorted(observed_reasons.items()))
        != supplied_completion["active8_exclusion_reason_histogram"]
    ):
        raise SemanticActive8DecisionSourceError(
            "semantic Active8 index census differs from completion"
        )

    source_identity_sha256 = _sha(decision_plan["source_identity"])
    frozen_decisions_by_shard = {
        key: tuple(value) for key, value in decisions_by_shard.items()
    }
    frozen_exclusions_by_key = dict(exclusions_by_key)
    index_kwargs: dict[str, Any] = {
        "status": INDEX_STATUS,
        **_AUTHORITY,
        "artifact_root": root,
        "migration_completion_file_sha256": (
            source_inventory.migration_completion_file_sha256
        ),
        "migration_completion_sha256": source_inventory.migration_completion_sha256,
        "chunk_cache_plan_file_sha256": hashlib.sha256(chunk_plan_bytes).hexdigest(),
        "chunk_cache_plan_sha256": chunk_plan["plan_sha256"],
        "chunk_cache_global_pointer_file_sha256": hashlib.sha256(
            pointer_bytes
        ).hexdigest(),
        "chunk_cache_global_pointer_sha256": pointer["pointer_sha256"],
        "chunk_cache_global_completion_file_sha256": hashlib.sha256(
            chunk_completion_bytes
        ).hexdigest(),
        "chunk_cache_global_completion_sha256": chunk_completion["completion_sha256"],
        "decision_plan_file_sha256": hashlib.sha256(decision_plan_bytes).hexdigest(),
        "decision_plan_sha256": decision_plan["plan_sha256"],
        "decision_run_identity_sha256": decision_plan["run_identity_sha256"],
        "decision_completion_file_sha256": hashlib.sha256(
            decision_completion_bytes
        ).hexdigest(),
        "decision_completion_sha256": supplied_completion["completion_sha256"],
        "decision_result_inventory_sha256": supplied_completion[
            "result_inventory_sha256"
        ],
        "process_identity_sha256": source_inventory.process_identity_sha256,
        "policy_sha256": supplied_completion["policy_sha256"],
        "model_runtime_identity_sha256": supplied_completion[
            "model_runtime_identity_sha256"
        ],
        "source_identity_sha256": source_identity_sha256,
        "accepted_trace_inventory_sha256": accepted_trace_stream.hexdigest(),
        "accepted_progress_inventory_sha256": accepted_progress_stream.hexdigest(),
        "excluded_trace_inventory_sha256": excluded_trace_stream.hexdigest(),
        "trace_decision_inventory_sha256": decision_stream.hexdigest(),
        "decision_lookup_inventory_sha256": _decision_lookup_sha256(
            frozen_decisions_by_shard
        ),
        "exclusion_lookup_inventory_sha256": _exclusion_lookup_sha256(
            frozen_exclusions_by_key
        ),
        "decision_source_implementation_sha256": _file_sha(Path(__file__)),
        "counts": tuple(sorted(normalized_counts.items())),
        "action_family_histogram": tuple(
            supplied_completion["active8_action_family_histogram"].items()
        ),
        "active8_exclusion_reason_histogram": tuple(
            supplied_completion["active8_exclusion_reason_histogram"].items()
        ),
        "migration_rejection_histogram": tuple(
            supplied_completion["migration_rejection_census"][
                "rejections_by_code"
            ].items()
        ),
        "_decision_plan_bytes": decision_plan_bytes,
        "_decisions_by_shard": frozen_decisions_by_shard,
        "_exclusions_by_key": frozen_exclusions_by_key,
    }
    inventory_sha256 = _sha(_index_identity_body(index_kwargs))
    return EditingV2SemanticActive8DecisionIndex(
        **index_kwargs,
        inventory_sha256=inventory_sha256,
    )


__all__ = [
    "INDEX_ENCODING",
    "INDEX_SCHEMA",
    "INDEX_SCHEMA_VERSION",
    "INDEX_STATUS",
    "MAX_DECISION_SOURCE_CHUNK_ROWS",
    "EditingV2SemanticActive8DecisionIndex",
    "SemanticActive8AcceptedTrace",
    "SemanticActive8AcceptedTransition",
    "SemanticActive8DecisionSourceError",
    "SemanticActive8ExcludedTrace",
    "SemanticActive8ProgressAddress",
    "resolve_editing_v2_semantic_active8_decision_source",
]
