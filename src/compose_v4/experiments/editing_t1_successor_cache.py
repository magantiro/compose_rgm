"""Immutable complete-trace successor caches for the bounded editing T1 panel.

The general sharded successor-cache inventory represents a complete packed
shard derivative.  A T1 panel is deliberately a selected trace union, so this
module gives that narrower coverage its own explicit, self-hashed manifest.
Unselected packed entries are never mislabeled as support exclusions.

Leaf artifacts reuse the production canonical-JSON successor cache.  Each leaf
therefore contains complete selected traces, including terminal rows, and is
bound to exact packed-corpus and implementation provenance.  The manifest is
published only after every immutable leaf exists and records the selected entry
indices separately from any corpus admission decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from rdkit import rdBase

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.immutable_artifact import (
    ImmutableArtifactError,
    write_bytes_if_absent,
)
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCache,
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheError,
    SuccessorFiberCacheProvenance,
    SuccessorFiberCacheRecord,
    read_successor_fiber_cache,
    serialize_successor_fiber_cache,
    write_successor_fiber_cache,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_p50_gate import (
    P50GateError,
    state_dict_semantic_sha256,
)
from compose_v4.experiments.successor_fiber_cache_builder import (
    SuccessorFiberCacheBuildError,
    compile_successor_fiber_trace_union,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import RewriteSystem

EDITING_T1_SUCCESSOR_CACHE_MANIFEST_SCHEMA = (
    "compose.editing.t1_successor_trace_union_cache_manifest"
)
EDITING_T1_SUCCESSOR_CACHE_MANIFEST_VERSION = 1
EDITING_T1_SUCCESSOR_CACHE_MANIFEST_STATUS = (
    "FROZEN_BOUNDED_T1_TRACE_UNION_NO_TRAINING_AUTHORITY"
)
EDITING_T1_SUCCESSOR_CACHE_COVERAGE_MODE = "frozen_t1_panel_trace_union_complete_traces"
MAX_T1_SUCCESSOR_CACHE_MANIFEST_BYTES = 32 << 20

_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "gate_decision",
    "coverage_mode",
    "identity",
    "selected_trace_count",
    "selected_trace_set_sha256",
    "selected_traces",
    "entry_count",
    "entries",
    "manifest_sha256",
}
_IDENTITY_FIELDS = {
    "t1_runtime_contract_sha256",
    "t1_implementation_sha256",
    "gate_zero_runtime_contract_sha256",
    "panel_artifact_sha256",
    "panel_selection_sha256",
    "panel_census_sha256",
    "panel_capacity_strata_sha256",
    "forensics_file_sha256",
    "charge_policy_audit_file_sha256",
    "charge_policy_exclusions_file_sha256",
    "charge_policy_exclusion_payload_sha256",
    "charge_policy_source_input_inventory_sha256",
    "active8_inventory_manifest_file_sha256",
    "active8_inventory_sha256",
    "active8_effective_source_corpus_cache_sha256",
    "active8_unified_packed_manifest_sha256",
    "active8_support_contract_sha256",
    "unified_packed_manifest_sha256",
    "representability_overlay_sha256",
    "support_time",
    "scratch_seed",
    "initial_model_state_sha256",
    "compiler_device",
    "compiler_dtype",
    "torch_version",
    "cuda_version",
    "rdkit_version",
}
_TRACE_FIELDS = {
    "packed_shard_content_sha256",
    "packed_shard_name",
    "entry_index",
    "layer",
    "partition",
    "trace_id",
    "trace_source_key",
    "trace_target_key",
    "path_length",
}
_ENTRY_FIELDS = {
    "packed_shard_content_sha256",
    "packed_shard_name",
    "layer",
    "partition",
    "packed_manifest_sha256",
    "packed_provenance_overlay_sha256",
    "included_entry_indices",
    "included_trace_set_sha256",
    "cache_relative_path",
    "cache_content_sha256",
    "cache_file_sha256",
    "cache_file_bytes",
    "active_trace_count",
    "progress_record_count",
    "jump_record_count",
    "terminal_record_count",
    "teacher_alias_count",
    "virtual_alias_count",
    "maximum_alias_count",
    "provenance",
}


class EditingT1SuccessorCacheError(RuntimeError):
    """A T1 trace-union cache is absent, stale, malformed, or incomplete."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache metadata is not finite canonical JSON"
        ) from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be nonempty text")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure == PurePosixPath(".")
        or ".." in pure.parts
        or "\\" in value
        or any(character in value for character in "*?[]")
    ):
        raise ValueError(f"{field} must be a safe POSIX-relative path")
    return pure.as_posix()


def _require_exact_mapping(
    value: object,
    fields: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingT1SuccessorCacheError(f"{name} must be an object")
    if set(value) != fields:
        raise EditingT1SuccessorCacheError(
            f"{name} fields disagree; missing={sorted(fields - set(value))}, "
            f"unexpected={sorted(set(value) - fields)}"
        )
    return value


def _write_bytes_if_absent(path: Path, content: bytes) -> None:
    """Atomically publish exact bytes without replacing another artifact."""

    destination = Path(path)
    try:
        write_bytes_if_absent(destination, content)
    except ImmutableArtifactError as error:
        raise FileExistsError(
            "immutable T1 successor-cache artifact already differs: "
            f"{destination}"
        ) from error


@dataclass(frozen=True)
class T1SuccessorCacheIdentity:
    """Frozen scientific and runtime identity for one compiled T1 trace union."""

    t1_runtime_contract_sha256: str
    t1_implementation_sha256: str
    gate_zero_runtime_contract_sha256: str
    panel_artifact_sha256: str
    panel_selection_sha256: str
    panel_census_sha256: str
    panel_capacity_strata_sha256: str
    forensics_file_sha256: str
    charge_policy_audit_file_sha256: str
    charge_policy_exclusions_file_sha256: str
    charge_policy_exclusion_payload_sha256: str
    charge_policy_source_input_inventory_sha256: str
    active8_inventory_manifest_file_sha256: str
    active8_inventory_sha256: str
    active8_effective_source_corpus_cache_sha256: str
    active8_unified_packed_manifest_sha256: str
    active8_support_contract_sha256: str
    unified_packed_manifest_sha256: str
    representability_overlay_sha256: str
    support_time: float
    scratch_seed: int
    initial_model_state_sha256: str
    compiler_device: str
    compiler_dtype: str
    torch_version: str
    cuda_version: str | None
    rdkit_version: str

    def __post_init__(self) -> None:
        digest_fields = _IDENTITY_FIELDS - {
            "support_time",
            "scratch_seed",
            "compiler_device",
            "compiler_dtype",
            "torch_version",
            "cuda_version",
            "rdkit_version",
        }
        for field in digest_fields:
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if (
            isinstance(self.support_time, bool)
            or not isinstance(self.support_time, (int, float))
            or not math.isfinite(float(self.support_time))
            or not 0.0 < float(self.support_time) < 1.0
        ):
            raise ValueError("support_time must be finite and strictly inside (0, 1)")
        if type(self.scratch_seed) is not int or self.scratch_seed < 0:
            raise ValueError("scratch_seed must be a nonnegative integer")
        for field in (
            "compiler_device",
            "compiler_dtype",
            "torch_version",
            "rdkit_version",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be nonempty text")
        if self.cuda_version is not None and (
            not isinstance(self.cuda_version, str) or not self.cuda_version
        ):
            raise ValueError("cuda_version must be null or nonempty text")
        if (
            self.active8_unified_packed_manifest_sha256
            != self.unified_packed_manifest_sha256
        ):
            raise ValueError(
                "Active8 and T1 cache identities name different unified manifests"
            )
        if (
            self.active8_support_contract_sha256
            != self.gate_zero_runtime_contract_sha256
        ):
            raise ValueError(
                "Active8 and T1 cache identities name different support contracts"
            )


@dataclass(frozen=True, order=True)
class T1SuccessorCacheTraceSpec:
    """Full immutable packed identity for one selected complete trace."""

    packed_shard_content_sha256: str
    packed_shard_name: str
    entry_index: int
    layer: str
    partition: str
    trace_id: str
    trace_source_key: str
    trace_target_key: str
    path_length: int

    def __post_init__(self) -> None:
        if not _is_sha256(self.packed_shard_content_sha256):
            raise ValueError("packed_shard_content_sha256 must be a lowercase SHA-256")
        if (
            not isinstance(self.packed_shard_name, str)
            or not self.packed_shard_name
            or Path(self.packed_shard_name).name != self.packed_shard_name
        ):
            raise ValueError("packed_shard_name must be one nonempty basename")
        if type(self.entry_index) is not int or self.entry_index < 0:
            raise ValueError("entry_index must be a nonnegative integer")
        if type(self.path_length) is not int or self.path_length < 0:
            raise ValueError("path_length must be a nonnegative integer")
        for field in (
            "layer",
            "partition",
            "trace_id",
            "trace_source_key",
            "trace_target_key",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be nonempty text")

    @classmethod
    def from_packed_address(
        cls,
        address: PackedTraceAddress,
    ) -> T1SuccessorCacheTraceSpec:
        return cls(
            packed_shard_content_sha256=address.packed_shard_content_sha256,
            packed_shard_name=address.packed_shard_name,
            entry_index=address.entry_index,
            layer=address.layer,
            partition=address.partition,
            trace_id=address.trace_id,
            trace_source_key=address.source_key,
            trace_target_key=address.target_key,
            path_length=address.path_length,
        )

    @classmethod
    def from_cache_address(
        cls,
        address: SuccessorFiberCacheAddress,
    ) -> T1SuccessorCacheTraceSpec:
        return cls(
            packed_shard_content_sha256=address.packed_shard_content_sha256,
            packed_shard_name=address.packed_shard_name,
            entry_index=address.entry_index,
            layer=address.layer,
            partition=address.partition,
            trace_id=address.trace_id,
            trace_source_key=address.trace_source_key,
            trace_target_key=address.trace_target_key,
            path_length=address.path_length,
        )


def _trace_set_sha256(
    traces: Iterable[T1SuccessorCacheTraceSpec],
) -> str:
    return _stable_sha256([asdict(trace) for trace in sorted(traces)])


@dataclass(frozen=True, order=True)
class T1SuccessorCacheShardSpec:
    """Immutable leaf identity and selected-entry census for one packed shard."""

    packed_shard_content_sha256: str
    packed_shard_name: str
    layer: str
    partition: str
    packed_manifest_sha256: str
    packed_provenance_overlay_sha256: str | None
    included_entry_indices: tuple[int, ...]
    included_trace_set_sha256: str
    cache_relative_path: str
    cache_content_sha256: str
    cache_file_sha256: str
    cache_file_bytes: int
    active_trace_count: int
    progress_record_count: int
    jump_record_count: int
    terminal_record_count: int
    teacher_alias_count: int
    virtual_alias_count: int
    maximum_alias_count: int
    provenance: SuccessorFiberCacheProvenance

    def __post_init__(self) -> None:
        if not _is_sha256(self.packed_shard_content_sha256):
            raise ValueError("packed_shard_content_sha256 must be a lowercase SHA-256")
        if (
            not isinstance(self.packed_shard_name, str)
            or not self.packed_shard_name
            or Path(self.packed_shard_name).name != self.packed_shard_name
        ):
            raise ValueError("packed_shard_name must be one nonempty basename")
        for field in ("layer", "partition"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be nonempty text")
        for field in (
            "packed_manifest_sha256",
            "included_trace_set_sha256",
            "cache_content_sha256",
            "cache_file_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if self.packed_provenance_overlay_sha256 is not None and not _is_sha256(
            self.packed_provenance_overlay_sha256
        ):
            raise ValueError(
                "packed_provenance_overlay_sha256 must be null or a SHA-256"
            )
        normalized_path = _safe_relative_path(
            self.cache_relative_path,
            field="cache_relative_path",
        )
        if normalized_path != self.cache_relative_path:
            raise ValueError("cache_relative_path is not normalized")
        if (
            not isinstance(self.included_entry_indices, tuple)
            or not self.included_entry_indices
            or self.included_entry_indices
            != tuple(sorted(set(self.included_entry_indices)))
            or any(
                type(index) is not int or index < 0
                for index in self.included_entry_indices
            )
        ):
            raise ValueError(
                "included_entry_indices must be nonempty sorted unique "
                "nonnegative integers"
            )
        for field in (
            "cache_file_bytes",
            "active_trace_count",
            "progress_record_count",
            "jump_record_count",
            "terminal_record_count",
            "teacher_alias_count",
            "virtual_alias_count",
            "maximum_alias_count",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.cache_file_bytes <= 0:
            raise ValueError("cache_file_bytes must be positive")
        if self.active_trace_count != len(self.included_entry_indices):
            raise ValueError(
                "active_trace_count must equal the selected entry-index count"
            )
        if self.terminal_record_count != self.active_trace_count:
            raise ValueError("each selected complete trace must have one terminal row")
        if (
            self.jump_record_count + self.terminal_record_count
            != self.progress_record_count
        ):
            raise ValueError("jump plus terminal rows must equal progress rows")
        if self.teacher_alias_count < self.jump_record_count:
            raise ValueError("teacher alias count is smaller than jump count")
        if self.maximum_alias_count > (
            self.teacher_alias_count + self.virtual_alias_count
        ):
            raise ValueError("maximum alias count exceeds the leaf alias total")
        if (
            self.provenance.packed_shard_content_sha256
            != self.packed_shard_content_sha256
            or self.provenance.packed_manifest_sha256 != self.packed_manifest_sha256
            or self.provenance.packed_provenance_overlay_sha256
            != self.packed_provenance_overlay_sha256
        ):
            raise ValueError("leaf provenance disagrees with its packed identity")


@dataclass(frozen=True)
class T1SuccessorCacheManifest:
    """Self-hashed selected-trace inventory published after immutable leaves."""

    identity: T1SuccessorCacheIdentity
    selected_traces: tuple[T1SuccessorCacheTraceSpec, ...]
    selected_trace_set_sha256: str
    entries: tuple[T1SuccessorCacheShardSpec, ...]
    manifest_sha256: str
    schema: str = EDITING_T1_SUCCESSOR_CACHE_MANIFEST_SCHEMA
    schema_version: int = EDITING_T1_SUCCESSOR_CACHE_MANIFEST_VERSION
    status: str = EDITING_T1_SUCCESSOR_CACHE_MANIFEST_STATUS
    training_authorized: bool = False
    gate_decision: None = None
    coverage_mode: str = EDITING_T1_SUCCESSOR_CACHE_COVERAGE_MODE

    def __post_init__(self) -> None:
        if self.schema != EDITING_T1_SUCCESSOR_CACHE_MANIFEST_SCHEMA:
            raise ValueError("T1 successor-cache manifest schema drifted")
        if self.schema_version != EDITING_T1_SUCCESSOR_CACHE_MANIFEST_VERSION:
            raise ValueError("T1 successor-cache manifest version drifted")
        if self.status != EDITING_T1_SUCCESSOR_CACHE_MANIFEST_STATUS:
            raise ValueError("T1 successor-cache manifest status drifted")
        if self.training_authorized is not False or self.gate_decision is not None:
            raise ValueError("T1 successor cache cannot carry training authority")
        if self.coverage_mode != EDITING_T1_SUCCESSOR_CACHE_COVERAGE_MODE:
            raise ValueError("T1 successor-cache coverage mode drifted")
        if not isinstance(self.identity, T1SuccessorCacheIdentity):
            raise TypeError("identity must be T1SuccessorCacheIdentity")
        if not self.selected_traces or self.selected_traces != tuple(
            sorted(self.selected_traces)
        ):
            raise ValueError("selected traces must be nonempty and canonical")
        exact_keys = tuple(
            (trace.packed_shard_content_sha256, trace.entry_index)
            for trace in self.selected_traces
        )
        if len(exact_keys) != len(set(exact_keys)):
            raise ValueError("selected traces repeat an exact packed entry")
        if self.selected_trace_set_sha256 != _trace_set_sha256(self.selected_traces):
            raise ValueError("selected trace-set SHA-256 does not match")
        if not self.entries or self.entries != tuple(
            sorted(
                self.entries,
                key=lambda entry: entry.packed_shard_content_sha256,
            )
        ):
            raise ValueError("cache entries must be nonempty and canonical")
        entry_digests = tuple(
            entry.packed_shard_content_sha256 for entry in self.entries
        )
        if len(entry_digests) != len(set(entry_digests)):
            raise ValueError("cache manifest repeats a packed shard")
        paths = tuple(entry.cache_relative_path for entry in self.entries)
        if len(paths) != len(set(paths)):
            raise ValueError("cache manifest repeats a leaf path")

        traces_by_shard: dict[str, list[T1SuccessorCacheTraceSpec]] = defaultdict(list)
        for trace in self.selected_traces:
            traces_by_shard[trace.packed_shard_content_sha256].append(trace)
        if set(traces_by_shard) != set(entry_digests):
            raise ValueError("selected traces and cache entries name different shards")
        for entry in self.entries:
            traces = tuple(sorted(traces_by_shard[entry.packed_shard_content_sha256]))
            expected_indices = tuple(trace.entry_index for trace in traces)
            if entry.included_entry_indices != expected_indices:
                raise ValueError(
                    "cache entry indices disagree with the selected trace union"
                )
            if entry.included_trace_set_sha256 != _trace_set_sha256(traces):
                raise ValueError("cache entry trace-set SHA-256 does not match")
            envelopes = {
                (trace.packed_shard_name, trace.layer, trace.partition)
                for trace in traces
            }
            if envelopes != {(entry.packed_shard_name, entry.layer, entry.partition)}:
                raise ValueError("selected trace envelope disagrees with cache entry")
            provenance = entry.provenance
            if (
                provenance.unified_packed_manifest_sha256
                != self.identity.unified_packed_manifest_sha256
                or provenance.representability_overlay_sha256
                != self.identity.representability_overlay_sha256
            ):
                raise ValueError(
                    "cache provenance disagrees with manifest-wide source identity"
                )
        if not _is_sha256(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256")
        if self.manifest_sha256 != _stable_sha256(_manifest_body(self)):
            raise ValueError("manifest_sha256 does not match the manifest body")

    @property
    def by_packed_digest(self) -> Mapping[str, T1SuccessorCacheShardSpec]:
        return {entry.packed_shard_content_sha256: entry for entry in self.entries}


@dataclass(frozen=True)
class T1SuccessorCacheManifestReceipt:
    """Physical and semantic identity passed independently to cache loaders."""

    manifest_relative_path: str
    manifest_sha256: str
    manifest_file_sha256: str
    manifest_file_bytes: int
    selected_trace_set_sha256: str
    initial_model_state_sha256: str

    def __post_init__(self) -> None:
        normalized = _safe_relative_path(
            self.manifest_relative_path,
            field="manifest_relative_path",
        )
        if normalized != self.manifest_relative_path:
            raise ValueError("manifest_relative_path is not normalized")
        for field in (
            "manifest_sha256",
            "manifest_file_sha256",
            "selected_trace_set_sha256",
            "initial_model_state_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if type(self.manifest_file_bytes) is not int or self.manifest_file_bytes <= 0:
            raise ValueError("manifest_file_bytes must be a positive integer")


@dataclass(frozen=True, order=True)
class T1SuccessorCacheShardReceipt:
    """Runtime-compatible identity of one validated cache leaf."""

    packed_shard_content_sha256: str
    cache_content_sha256: str
    encoded_sha256: str
    record_count: int

    def __post_init__(self) -> None:
        for field in (
            "packed_shard_content_sha256",
            "cache_content_sha256",
            "encoded_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if type(self.record_count) is not int or self.record_count <= 0:
            raise ValueError("record_count must be a positive integer")


def _entry_payload(entry: T1SuccessorCacheShardSpec) -> dict[str, Any]:
    return {
        **{
            field: getattr(entry, field)
            for field in _ENTRY_FIELDS
            - {
                "included_entry_indices",
                "provenance",
            }
        },
        "included_entry_indices": list(entry.included_entry_indices),
        "provenance": asdict(entry.provenance),
    }


def _manifest_body(manifest: T1SuccessorCacheManifest) -> dict[str, Any]:
    return {
        "schema": manifest.schema,
        "schema_version": manifest.schema_version,
        "status": manifest.status,
        "training_authorized": manifest.training_authorized,
        "gate_decision": manifest.gate_decision,
        "coverage_mode": manifest.coverage_mode,
        "identity": asdict(manifest.identity),
        "selected_trace_count": len(manifest.selected_traces),
        "selected_trace_set_sha256": manifest.selected_trace_set_sha256,
        "selected_traces": [asdict(trace) for trace in manifest.selected_traces],
        "entry_count": len(manifest.entries),
        "entries": [_entry_payload(entry) for entry in manifest.entries],
    }


def seal_t1_successor_cache_manifest(
    *,
    identity: T1SuccessorCacheIdentity,
    selected_traces: Iterable[T1SuccessorCacheTraceSpec],
    entries: Iterable[T1SuccessorCacheShardSpec],
) -> tuple[bytes, T1SuccessorCacheManifest]:
    """Create deterministic self-hashed bytes for one T1 cache manifest."""

    ordered_traces = tuple(sorted(selected_traces))
    ordered_entries = tuple(
        sorted(entries, key=lambda entry: entry.packed_shard_content_sha256)
    )
    provisional = {
        "schema": EDITING_T1_SUCCESSOR_CACHE_MANIFEST_SCHEMA,
        "schema_version": EDITING_T1_SUCCESSOR_CACHE_MANIFEST_VERSION,
        "status": EDITING_T1_SUCCESSOR_CACHE_MANIFEST_STATUS,
        "training_authorized": False,
        "gate_decision": None,
        "coverage_mode": EDITING_T1_SUCCESSOR_CACHE_COVERAGE_MODE,
        "identity": asdict(identity),
        "selected_trace_count": len(ordered_traces),
        "selected_trace_set_sha256": _trace_set_sha256(ordered_traces),
        "selected_traces": [asdict(trace) for trace in ordered_traces],
        "entry_count": len(ordered_entries),
        "entries": [_entry_payload(entry) for entry in ordered_entries],
    }
    manifest = T1SuccessorCacheManifest(
        identity=identity,
        selected_traces=ordered_traces,
        selected_trace_set_sha256=provisional["selected_trace_set_sha256"],
        entries=ordered_entries,
        manifest_sha256=_stable_sha256(provisional),
    )
    encoded = (
        _canonical_json_bytes(
            {**_manifest_body(manifest), "manifest_sha256": manifest.manifest_sha256}
        )
        + b"\n"
    )
    if len(encoded) > MAX_T1_SUCCESSOR_CACHE_MANIFEST_BYTES:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache manifest exceeds its size bound"
        )
    return encoded, manifest


def deserialize_t1_successor_cache_manifest(
    encoded: bytes,
    *,
    expected_manifest_sha256: str,
    expected_identity: T1SuccessorCacheIdentity,
) -> T1SuccessorCacheManifest:
    """Decode exact canonical bytes under independent manifest and T1 identities."""

    if len(encoded) > MAX_T1_SUCCESSOR_CACHE_MANIFEST_BYTES:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache manifest exceeds its size bound"
        )
    try:
        value = json.loads(
            encoded,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache manifest is not valid finite JSON"
        ) from error
    payload = _require_exact_mapping(
        value,
        _MANIFEST_FIELDS,
        name="T1 successor-cache manifest",
    )
    if not _is_sha256(expected_manifest_sha256):
        raise EditingT1SuccessorCacheError(
            "expected_manifest_sha256 must be a lowercase SHA-256"
        )
    if payload["manifest_sha256"] != expected_manifest_sha256:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache manifest does not match the expected identity"
        )
    identity_payload = _require_exact_mapping(
        payload["identity"],
        _IDENTITY_FIELDS,
        name="T1 successor-cache identity",
    )
    try:
        identity = T1SuccessorCacheIdentity(**dict(identity_payload))
    except (TypeError, ValueError) as error:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache identity is invalid"
        ) from error
    if identity != expected_identity:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache manifest names another frozen identity"
        )
    raw_traces = payload["selected_traces"]
    raw_entries = payload["entries"]
    if not isinstance(raw_traces, list) or not isinstance(raw_entries, list):
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache traces and entries must be arrays"
        )
    try:
        traces = tuple(
            T1SuccessorCacheTraceSpec(
                **dict(
                    _require_exact_mapping(
                        item,
                        _TRACE_FIELDS,
                        name="T1 selected trace",
                    )
                )
            )
            for item in raw_traces
        )
        entries: list[T1SuccessorCacheShardSpec] = []
        for item in raw_entries:
            entry_payload = dict(
                _require_exact_mapping(
                    item,
                    _ENTRY_FIELDS,
                    name="T1 cache entry",
                )
            )
            raw_indices = entry_payload["included_entry_indices"]
            if not isinstance(raw_indices, list):
                raise EditingT1SuccessorCacheError(
                    "included_entry_indices must be an array"
                )
            provenance_payload = entry_payload["provenance"]
            if not isinstance(provenance_payload, Mapping):
                raise EditingT1SuccessorCacheError(
                    "T1 cache entry provenance must be an object"
                )
            entry_payload["included_entry_indices"] = tuple(raw_indices)
            entry_payload["provenance"] = SuccessorFiberCacheProvenance(
                **dict(provenance_payload)
            )
            entries.append(T1SuccessorCacheShardSpec(**entry_payload))
        manifest = T1SuccessorCacheManifest(
            identity=identity,
            selected_traces=traces,
            selected_trace_set_sha256=payload["selected_trace_set_sha256"],
            entries=tuple(entries),
            manifest_sha256=payload["manifest_sha256"],
            schema=payload["schema"],
            schema_version=payload["schema_version"],
            status=payload["status"],
            training_authorized=payload["training_authorized"],
            gate_decision=payload["gate_decision"],
            coverage_mode=payload["coverage_mode"],
        )
    except EditingT1SuccessorCacheError:
        raise
    except (TypeError, ValueError) as error:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache manifest contract is invalid"
        ) from error
    if payload["selected_trace_count"] != len(manifest.selected_traces):
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache selected trace count disagrees"
        )
    if payload["entry_count"] != len(manifest.entries):
        raise EditingT1SuccessorCacheError("T1 successor-cache entry count disagrees")
    canonical, _ = seal_t1_successor_cache_manifest(
        identity=manifest.identity,
        selected_traces=manifest.selected_traces,
        entries=manifest.entries,
    )
    if canonical != encoded:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache manifest bytes are not canonical"
        )
    return manifest


def read_t1_successor_cache_manifest(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_file_sha256: str,
    expected_file_bytes: int,
    expected_identity: T1SuccessorCacheIdentity,
) -> T1SuccessorCacheManifest:
    """Read one exact physical manifest and verify all independent identities."""

    source = Path(path)
    if not source.is_file():
        raise EditingT1SuccessorCacheError(
            f"T1 successor-cache manifest does not exist: {source}"
        )
    if type(expected_file_bytes) is not int or expected_file_bytes <= 0:
        raise EditingT1SuccessorCacheError(
            "expected_file_bytes must be a positive integer"
        )
    if source.stat().st_size != expected_file_bytes:
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache manifest byte count disagrees"
        )
    if not _is_sha256(expected_file_sha256) or (
        _sha256_file(source) != expected_file_sha256
    ):
        raise EditingT1SuccessorCacheError(
            "T1 successor-cache manifest physical SHA-256 disagrees"
        )
    return deserialize_t1_successor_cache_manifest(
        source.read_bytes(),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_identity=expected_identity,
    )


def _summarize_cache(cache: SuccessorFiberCache) -> dict[str, int]:
    records = cache.records
    terminal_count = sum(record.address.is_terminal for record in records)
    teacher_alias_counts = tuple(
        0 if record.teacher_fiber is None else len(record.teacher_fiber.aliases)
        for record in records
    )
    virtual_alias_counts = tuple(
        len(record.state_support.virtual_aliases) for record in records
    )
    alias_counts = tuple(
        teacher + virtual
        for teacher, virtual in zip(
            teacher_alias_counts,
            virtual_alias_counts,
            strict=True,
        )
    )
    return {
        "active_trace_count": len({record.address.trace_key for record in records}),
        "progress_record_count": len(records),
        "jump_record_count": len(records) - terminal_count,
        "terminal_record_count": terminal_count,
        "teacher_alias_count": sum(teacher_alias_counts),
        "virtual_alias_count": sum(virtual_alias_counts),
        "maximum_alias_count": max(alias_counts, default=0),
    }


def _selected_trace_specs(
    records: Iterable[PathRecord],
) -> tuple[tuple[PathRecord, ...], tuple[T1SuccessorCacheTraceSpec, ...]]:
    addressed: list[tuple[T1SuccessorCacheTraceSpec, PathRecord]] = []
    for record in records:
        address = record.corpus_address
        if address is None:
            raise EditingT1SuccessorCacheError(
                "T1 cache build selected an unaddressed trace"
            )
        addressed.append(
            (T1SuccessorCacheTraceSpec.from_packed_address(address), record)
        )
    if not addressed:
        raise EditingT1SuccessorCacheError("T1 cache build selected no trace")
    addressed.sort(key=lambda item: item[0])
    keys = tuple(
        (spec.packed_shard_content_sha256, spec.entry_index) for spec, _ in addressed
    )
    if len(keys) != len(set(keys)):
        raise EditingT1SuccessorCacheError(
            "T1 cache build repeats an exact packed trace"
        )
    envelopes_by_shard: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for spec, _ in addressed:
        envelopes_by_shard[spec.packed_shard_content_sha256].add(
            (spec.packed_shard_name, spec.layer, spec.partition)
        )
    if any(len(envelopes) != 1 for envelopes in envelopes_by_shard.values()):
        raise EditingT1SuccessorCacheError(
            "one packed-shard digest maps to multiple T1 trace envelopes"
        )
    return (
        tuple(record for _, record in addressed),
        tuple(spec for spec, _ in addressed),
    )


def t1_selected_trace_set_sha256(records: Iterable[PathRecord]) -> str:
    """Hash the exact complete-trace selection for one frozen T1 panel view."""

    _, selected_traces = _selected_trace_specs(records)
    return _trace_set_sha256(selected_traces)


def _validate_model_build_identity(
    model: FactorizedTraceletRateModel,
    identity: T1SuccessorCacheIdentity,
) -> None:
    """Require the compiled law to use the declared scratch/runtime identity."""

    try:
        state_dict = model.state_dict()
        observed_state_sha256 = state_dict_semantic_sha256(state_dict)
    except (AttributeError, P50GateError) as error:
        raise EditingT1SuccessorCacheError(
            "T1 cache model state cannot be hashed exactly"
        ) from error
    if observed_state_sha256 != identity.initial_model_state_sha256:
        raise EditingT1SuccessorCacheError(
            "T1 cache model state differs from the declared scratch identity"
        )
    tensors = tuple(state_dict.values())
    devices = {str(tensor.device) for tensor in tensors}
    floating_dtypes = {
        str(tensor.dtype)
        for tensor in tensors
        if tensor.is_floating_point() or tensor.is_complex()
    }
    if devices != {identity.compiler_device}:
        raise EditingT1SuccessorCacheError(
            "T1 cache compiler device differs from the model state"
        )
    if floating_dtypes != {identity.compiler_dtype}:
        raise EditingT1SuccessorCacheError(
            "T1 cache compiler dtype differs from the model state"
        )
    if str(torch.__version__) != identity.torch_version:
        raise EditingT1SuccessorCacheError(
            "T1 cache Torch version differs from the declared compiler runtime"
        )
    if torch.version.cuda != identity.cuda_version:
        raise EditingT1SuccessorCacheError(
            "T1 cache CUDA version differs from the declared compiler runtime"
        )
    if rdBase.rdkitVersion != identity.rdkit_version:
        raise EditingT1SuccessorCacheError(
            "T1 cache RDKit version differs from the declared compiler runtime"
        )


def build_t1_successor_cache(
    cache_root: Path,
    model: FactorizedTraceletRateModel,
    records: Iterable[PathRecord],
    *,
    provenance_by_shard: Mapping[str, SuccessorFiberCacheProvenance],
    identity: T1SuccessorCacheIdentity,
    system: RewriteSystem | None = None,
) -> tuple[T1SuccessorCacheManifest, T1SuccessorCacheManifestReceipt]:
    """Compile, validate, and immutably publish one selected T1 trace union."""

    _validate_model_build_identity(model, identity)
    ordered_records, selected_traces = _selected_trace_specs(records)
    selected_shards = {trace.packed_shard_content_sha256 for trace in selected_traces}
    if set(provenance_by_shard) != selected_shards:
        raise EditingT1SuccessorCacheError(
            "T1 cache provenance must exactly cover the selected packed shards"
        )
    try:
        cache_records = compile_successor_fiber_trace_union(
            model,
            ordered_records,
            time=float(identity.support_time),
            system=system,
        )
    except (SuccessorFiberCacheBuildError, ValueError) as error:
        raise EditingT1SuccessorCacheError(
            "T1 complete-trace union compilation failed"
        ) from error

    records_by_shard: dict[str, list[SuccessorFiberCacheRecord]] = defaultdict(list)
    for record in cache_records:
        records_by_shard[record.address.packed_shard_content_sha256].append(record)
    if set(records_by_shard) != selected_shards:
        raise EditingT1SuccessorCacheError(
            "T1 compiler output names another packed-shard set"
        )

    root = Path(cache_root)
    entries: list[T1SuccessorCacheShardSpec] = []
    for shard_digest in sorted(selected_shards):
        provenance = provenance_by_shard[shard_digest]
        try:
            encoded, cache = serialize_successor_fiber_cache(
                records_by_shard[shard_digest],
                provenance=provenance,
            )
        except SuccessorFiberCacheError as error:
            raise EditingT1SuccessorCacheError(
                "T1 compiler output is not a complete immutable trace leaf"
            ) from error
        relative_path = (
            PurePosixPath("shards")
            / (f"{shard_digest}.{cache.content_sha256}." "successor_fiber_cache.json")
        ).as_posix()
        leaf_path = root / relative_path
        written = write_successor_fiber_cache(
            leaf_path,
            cache.records,
            provenance=provenance,
        )
        if written != cache or leaf_path.read_bytes() != encoded:
            raise EditingT1SuccessorCacheError(
                "published T1 cache leaf differs from canonical compilation"
            )
        traces = tuple(
            trace
            for trace in selected_traces
            if trace.packed_shard_content_sha256 == shard_digest
        )
        summary = _summarize_cache(cache)
        first = traces[0]
        entries.append(
            T1SuccessorCacheShardSpec(
                packed_shard_content_sha256=shard_digest,
                packed_shard_name=first.packed_shard_name,
                layer=first.layer,
                partition=first.partition,
                packed_manifest_sha256=provenance.packed_manifest_sha256,
                packed_provenance_overlay_sha256=(
                    provenance.packed_provenance_overlay_sha256
                ),
                included_entry_indices=tuple(trace.entry_index for trace in traces),
                included_trace_set_sha256=_trace_set_sha256(traces),
                cache_relative_path=relative_path,
                cache_content_sha256=cache.content_sha256,
                cache_file_sha256=hashlib.sha256(encoded).hexdigest(),
                cache_file_bytes=len(encoded),
                provenance=provenance,
                **summary,
            )
        )

    manifest_bytes, manifest = seal_t1_successor_cache_manifest(
        identity=identity,
        selected_traces=selected_traces,
        entries=entries,
    )
    manifest_relative_path = (
        PurePosixPath("manifests") / f"{manifest.manifest_sha256}.json"
    ).as_posix()
    manifest_path = root / manifest_relative_path
    _write_bytes_if_absent(manifest_path, manifest_bytes)
    receipt = T1SuccessorCacheManifestReceipt(
        manifest_relative_path=manifest_relative_path,
        manifest_sha256=manifest.manifest_sha256,
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_file_bytes=len(manifest_bytes),
        selected_trace_set_sha256=manifest.selected_trace_set_sha256,
        initial_model_state_sha256=identity.initial_model_state_sha256,
    )
    read_t1_successor_cache_manifest(
        manifest_path,
        expected_manifest_sha256=receipt.manifest_sha256,
        expected_file_sha256=receipt.manifest_file_sha256,
        expected_file_bytes=receipt.manifest_file_bytes,
        expected_identity=identity,
    )
    return manifest, receipt


class T1SuccessorFiberCache:
    """Lazy exact-address lookup over a validated selected-trace manifest."""

    def __init__(
        self,
        cache_root: Path,
        manifest: T1SuccessorCacheManifest,
        receipt: T1SuccessorCacheManifestReceipt,
        *,
        max_open_shards: int = 2,
    ) -> None:
        if type(max_open_shards) is not int or max_open_shards <= 0:
            raise ValueError("max_open_shards must be a positive integer")
        if (
            receipt.manifest_sha256 != manifest.manifest_sha256
            or receipt.selected_trace_set_sha256 != manifest.selected_trace_set_sha256
            or receipt.initial_model_state_sha256
            != manifest.identity.initial_model_state_sha256
        ):
            raise EditingT1SuccessorCacheError(
                "T1 cache receipt disagrees with the loaded manifest"
            )
        self.cache_root = Path(cache_root)
        self.manifest = manifest
        self.receipt = receipt
        self.max_open_shards = max_open_shards
        self._specs = dict(manifest.by_packed_digest)
        self._traces = {
            (trace.packed_shard_content_sha256, trace.entry_index): trace
            for trace in manifest.selected_traces
        }
        self._loaded: OrderedDict[str, SuccessorFiberCache] = OrderedDict()

    @property
    def loaded_shard_digests(self) -> tuple[str, ...]:
        return tuple(self._loaded)

    def _path_for(self, spec: T1SuccessorCacheShardSpec) -> Path:
        root = self.cache_root.resolve()
        candidate = (root / spec.cache_relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise EditingT1SuccessorCacheError(
                "T1 cache leaf path escapes its declared root"
            ) from error
        return candidate

    def _load(self, spec: T1SuccessorCacheShardSpec) -> SuccessorFiberCache:
        digest = spec.packed_shard_content_sha256
        cached = self._loaded.get(digest)
        if cached is not None:
            self._loaded.move_to_end(digest)
            return cached
        path = self._path_for(spec)
        if not path.is_file():
            raise EditingT1SuccessorCacheError(
                f"declared T1 cache leaf is absent: {path}"
            )
        if path.stat().st_size != spec.cache_file_bytes:
            raise EditingT1SuccessorCacheError(
                "T1 cache leaf byte count disagrees with its manifest"
            )
        if _sha256_file(path) != spec.cache_file_sha256:
            raise EditingT1SuccessorCacheError(
                "T1 cache leaf physical SHA-256 disagrees with its manifest"
            )
        cache = read_successor_fiber_cache(
            path,
            expected_provenance=spec.provenance,
            expected_content_sha256=spec.cache_content_sha256,
        )
        summary = _summarize_cache(cache)
        expected_summary = {
            field: getattr(spec, field)
            for field in (
                "active_trace_count",
                "progress_record_count",
                "jump_record_count",
                "terminal_record_count",
                "teacher_alias_count",
                "virtual_alias_count",
                "maximum_alias_count",
            )
        }
        if summary != expected_summary:
            raise EditingT1SuccessorCacheError(
                "T1 cache leaf census disagrees with its manifest"
            )
        observed_traces = tuple(
            sorted(
                {
                    T1SuccessorCacheTraceSpec.from_cache_address(record.address)
                    for record in cache.records
                }
            )
        )
        expected_traces = tuple(
            trace
            for trace in self.manifest.selected_traces
            if trace.packed_shard_content_sha256 == digest
        )
        if observed_traces != expected_traces:
            raise EditingT1SuccessorCacheError(
                "T1 cache leaf omits or invents selected complete traces"
            )
        self._loaded[digest] = cache
        self._loaded.move_to_end(digest)
        while len(self._loaded) > self.max_open_shards:
            self._loaded.popitem(last=False)
        return cache

    def require(
        self,
        address: SuccessorFiberCacheAddress,
        *,
        source_state: MolecularGraph | None = None,
    ) -> SuccessorFiberCacheRecord:
        """Return one exact cached row; never compile or enumerate on a miss."""

        if not isinstance(address, SuccessorFiberCacheAddress):
            raise TypeError("address must be SuccessorFiberCacheAddress")
        try:
            spec = self._specs[address.packed_shard_content_sha256]
        except KeyError:
            raise EditingT1SuccessorCacheError(
                "packed shard is absent from the frozen T1 cache manifest"
            ) from None
        if address.entry_index not in spec.included_entry_indices:
            raise EditingT1SuccessorCacheError(
                "exact packed trace is absent from the selected T1 trace union"
            )
        packed_address = PackedTraceAddress(
            packed_shard_content_sha256=address.packed_shard_content_sha256,
            packed_shard_name=address.packed_shard_name,
            entry_index=address.entry_index,
            trace_id=address.trace_id,
            layer=address.layer,
            partition=address.partition,
            source_key=address.trace_source_key,
            target_key=address.trace_target_key,
            path_length=address.path_length,
        )
        cache = self._load(spec)
        try:
            record = cache.require_packed_address(
                packed_address,
                progress_index=address.progress_index,
            )
        except KeyError as error:
            raise EditingT1SuccessorCacheError(
                "T1 cache has no exact selected trace-progress row"
            ) from error
        if record.address != address:
            raise EditingT1SuccessorCacheError(
                "T1 cache record disagrees with the full requested address"
            )
        if source_state is not None and (
            persistent_slot_state_sha256(source_state) != record.source_state_sha256
        ):
            raise EditingT1SuccessorCacheError(
                "requested exact state disagrees with its T1 cache row"
            )
        return record

    def require_packed_address(
        self,
        address: PackedTraceAddress,
        *,
        progress_index: int,
        source_state: MolecularGraph | None = None,
    ) -> SuccessorFiberCacheRecord:
        """Resolve a packed trace-progress address with full envelope checking."""

        if not isinstance(address, PackedTraceAddress):
            raise TypeError("address must be PackedTraceAddress")
        try:
            cache_address = SuccessorFiberCacheAddress.from_packed_trace(
                address,
                progress_index=progress_index,
            )
        except ValueError as error:
            raise EditingT1SuccessorCacheError(
                "requested progress index lies outside the packed trace"
            ) from error
        return self.require(cache_address, source_state=source_state)

    def receipts_for(
        self,
        addresses: Iterable[SuccessorFiberCacheAddress | PackedTraceAddress],
    ) -> tuple[T1SuccessorCacheShardReceipt, ...]:
        """Return canonical runtime-compatible receipts for referenced shards."""

        digests: set[str] = set()
        for address in addresses:
            if not isinstance(
                address,
                (SuccessorFiberCacheAddress, PackedTraceAddress),
            ):
                raise TypeError(
                    "receipt addresses must use packed or successor-cache addresses"
                )
            digest = address.packed_shard_content_sha256
            if digest not in self._specs:
                raise EditingT1SuccessorCacheError(
                    "receipt address names a shard outside the T1 cache manifest"
                )
            key = (digest, address.entry_index)
            try:
                expected_trace = self._traces[key]
            except KeyError:
                raise EditingT1SuccessorCacheError(
                    "receipt address names a trace outside the selected T1 union"
                ) from None
            observed_trace = (
                T1SuccessorCacheTraceSpec.from_cache_address(address)
                if isinstance(address, SuccessorFiberCacheAddress)
                else T1SuccessorCacheTraceSpec.from_packed_address(address)
            )
            if observed_trace != expected_trace:
                raise EditingT1SuccessorCacheError(
                    "receipt address disagrees with the selected trace identity"
                )
            digests.add(digest)
        if not digests:
            raise EditingT1SuccessorCacheError(
                "cannot derive cache receipts for an empty address set"
            )
        return tuple(
            T1SuccessorCacheShardReceipt(
                packed_shard_content_sha256=digest,
                cache_content_sha256=self._specs[digest].cache_content_sha256,
                encoded_sha256=self._specs[digest].cache_file_sha256,
                record_count=self._specs[digest].progress_record_count,
            )
            for digest in sorted(digests)
        )

    def validate_all_leaves(self) -> tuple[T1SuccessorCacheShardReceipt, ...]:
        """Read and validate every declared leaf before scarce compute is used."""

        receipts: list[T1SuccessorCacheShardReceipt] = []
        for digest in sorted(self._specs):
            spec = self._specs[digest]
            self._load(spec)
            receipts.append(
                T1SuccessorCacheShardReceipt(
                    packed_shard_content_sha256=digest,
                    cache_content_sha256=spec.cache_content_sha256,
                    encoded_sha256=spec.cache_file_sha256,
                    record_count=spec.progress_record_count,
                )
            )
        return tuple(receipts)


def load_t1_successor_cache(
    cache_root: Path,
    receipt: T1SuccessorCacheManifestReceipt,
    *,
    expected_identity: T1SuccessorCacheIdentity,
    max_open_shards: int = 2,
) -> T1SuccessorFiberCache:
    """Open one exact manifest-backed cache with no enumeration fallback."""

    if not isinstance(receipt, T1SuccessorCacheManifestReceipt):
        raise TypeError("receipt must be T1SuccessorCacheManifestReceipt")
    root = Path(cache_root).resolve()
    manifest_path = (root / receipt.manifest_relative_path).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise EditingT1SuccessorCacheError(
            "T1 cache manifest path escapes its declared root"
        ) from error
    manifest = read_t1_successor_cache_manifest(
        manifest_path,
        expected_manifest_sha256=receipt.manifest_sha256,
        expected_file_sha256=receipt.manifest_file_sha256,
        expected_file_bytes=receipt.manifest_file_bytes,
        expected_identity=expected_identity,
    )
    if (
        manifest.selected_trace_set_sha256 != receipt.selected_trace_set_sha256
        or manifest.identity.initial_model_state_sha256
        != receipt.initial_model_state_sha256
    ):
        raise EditingT1SuccessorCacheError(
            "T1 cache receipt selection or scratch identity disagrees"
        )
    return T1SuccessorFiberCache(
        root,
        manifest,
        receipt,
        max_open_shards=max_open_shards,
    )


__all__ = [
    "EDITING_T1_SUCCESSOR_CACHE_COVERAGE_MODE",
    "EDITING_T1_SUCCESSOR_CACHE_MANIFEST_SCHEMA",
    "EDITING_T1_SUCCESSOR_CACHE_MANIFEST_STATUS",
    "EDITING_T1_SUCCESSOR_CACHE_MANIFEST_VERSION",
    "EditingT1SuccessorCacheError",
    "MAX_T1_SUCCESSOR_CACHE_MANIFEST_BYTES",
    "T1SuccessorCacheIdentity",
    "T1SuccessorCacheManifest",
    "T1SuccessorCacheManifestReceipt",
    "T1SuccessorCacheShardReceipt",
    "T1SuccessorCacheShardSpec",
    "t1_selected_trace_set_sha256",
    "T1SuccessorCacheTraceSpec",
    "T1SuccessorFiberCache",
    "build_t1_successor_cache",
    "deserialize_t1_successor_cache_manifest",
    "load_t1_successor_cache",
    "read_t1_successor_cache_manifest",
    "seal_t1_successor_cache_manifest",
]
