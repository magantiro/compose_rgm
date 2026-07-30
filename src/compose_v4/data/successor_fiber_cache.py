"""Deterministic, provenance-bound storage for canonical-successor fibers.

Successor supervision needs chemistry only once: offline compilation enumerates
the production marks, executes them and groups every coordinate that reaches the
teacher molecule.  The training hot path then indexes those coordinates in the
model's differentiable action tables.  This module stores that precompiled
support without storing any model-dependent weight, logit, rate or probability.

One cache record corresponds to one ``(layer, partition, trace, progress)`` row.
A complete K-step trace therefore has K jump records followed by one terminal
record.  Jump records carry a :class:`TeacherSuccessorFiber`; terminal records
carry only :class:`StateProductiveSupport`.  The writer rejects incomplete
traces and verifies that each jump target is exactly the next progress source.

The artifact hash covers the schema, complete provenance and every support
coordinate.  Loading always requires the caller's expected provenance and fails
closed on drift, malformed structure or content tampering.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)

SUCCESSOR_FIBER_CACHE_SCHEMA = "compose.data.successor_fiber_cache"
SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION = 3

_PROVENANCE_FIELDS = (
    "operator_registry_hash",
    "capability_hash",
    "support_signature_sha256",
    "canonicalization_version",
    "canonicalizer_contract_sha256",
    "packed_corpus_schema",
    "packed_corpus_schema_version",
    "packed_shard_content_sha256",
    "packed_manifest_sha256",
    "packed_provenance_overlay_sha256",
    "unified_packed_manifest_sha256",
    "representability_overlay_sha256",
    "coordinate_schema_version",
    "tensorization_implementation_hash",
    "fiber_compiler_implementation_hash",
)
_ROOT_FIELDS = (
    "schema",
    "schema_version",
    "content_sha256",
    "provenance",
    "records",
)
_RECORD_FIELDS = (
    "packed_shard_content_sha256",
    "packed_shard_name",
    "entry_index",
    "layer",
    "partition",
    "trace_id",
    "trace_source_key",
    "trace_target_key",
    "progress_index",
    "path_length",
    "source_key",
    "source_state_sha256",
    "target_key",
    "target_state_sha256",
    "aliases",
    "virtual_aliases",
)
_ALIAS_FIELDS = ("family_name", "table_name", "coordinate")

_COMPILER_IMPLEMENTATION_SOURCES = (
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "src/compose_v4/chem/persistent_state_identity.py",
)


class SuccessorFiberCacheError(RuntimeError):
    """The cache is malformed, tampered with or built under another contract."""


@dataclass(frozen=True)
class SuccessorFiberCacheLimits:
    """Explicit resource bounds for one cache shard.

    A production corpus should use multiple cache shards rather than disabling
    these limits.  Tests and small diagnostic panels can pass tighter limits.
    """

    max_file_bytes: int = 1 << 30
    max_records: int = 1_000_000
    max_total_aliases: int = 50_000_000
    max_aliases_per_record: int = 1_000_000
    max_coordinate_rank: int = 8
    max_coordinate_value: int = (1 << 31) - 1
    max_key_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS = SuccessorFiberCacheLimits()


@dataclass(frozen=True)
class SuccessorFiberCacheProvenance:
    """Scientific identities that determine the meaning of cached coordinates."""

    operator_registry_hash: str
    capability_hash: str
    support_signature_sha256: str
    canonicalization_version: str
    canonicalizer_contract_sha256: str
    packed_corpus_schema: str
    packed_corpus_schema_version: int
    packed_shard_content_sha256: str
    packed_manifest_sha256: str
    packed_provenance_overlay_sha256: str | None
    unified_packed_manifest_sha256: str
    representability_overlay_sha256: str
    coordinate_schema_version: int
    tensorization_implementation_hash: str
    fiber_compiler_implementation_hash: str

    def __post_init__(self) -> None:
        for name in (
            "operator_registry_hash",
            "capability_hash",
            "canonicalization_version",
            "packed_corpus_schema",
            "tensorization_implementation_hash",
            "fiber_compiler_implementation_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if (
            type(self.packed_corpus_schema_version) is not int
            or self.packed_corpus_schema_version < 0
        ):
            raise ValueError(
                "packed_corpus_schema_version must be a nonnegative integer"
            )
        if (
            type(self.coordinate_schema_version) is not int
            or self.coordinate_schema_version <= 0
        ):
            raise ValueError("coordinate_schema_version must be a positive integer")
        required_digests = (
            "support_signature_sha256",
            "canonicalizer_contract_sha256",
            "packed_shard_content_sha256",
            "packed_manifest_sha256",
            "unified_packed_manifest_sha256",
            "representability_overlay_sha256",
            "fiber_compiler_implementation_hash",
        )
        for name in required_digests:
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        overlay_digest = self.packed_provenance_overlay_sha256
        if overlay_digest is not None and (
            not isinstance(overlay_digest, str)
            or len(overlay_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in overlay_digest
            )
        ):
            raise ValueError(
                "packed_provenance_overlay_sha256 must be null or a lowercase SHA-256"
            )


@dataclass(frozen=True, order=True)
class SuccessorFiberCacheAddress:
    """Stable address of one trace-progress training row."""

    packed_shard_content_sha256: str
    packed_shard_name: str
    entry_index: int
    layer: str
    partition: str
    trace_id: str
    trace_source_key: str
    trace_target_key: str
    progress_index: int
    path_length: int

    def __post_init__(self) -> None:
        digest = self.packed_shard_content_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "packed_shard_content_sha256 must be a lowercase SHA-256"
            )
        if (
            not isinstance(self.packed_shard_name, str)
            or not self.packed_shard_name
            or Path(self.packed_shard_name).name != self.packed_shard_name
        ):
            raise ValueError("packed_shard_name must be one nonempty basename")
        if type(self.entry_index) is not int or self.entry_index < 0:
            raise ValueError("entry_index must be a nonnegative integer")
        for name in (
            "layer",
            "partition",
            "trace_id",
            "trace_source_key",
            "trace_target_key",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.progress_index) is not int or self.progress_index < 0:
            raise ValueError("progress_index must be a nonnegative integer")
        if type(self.path_length) is not int or self.path_length < 0:
            raise ValueError("path_length must be a nonnegative integer")
        if self.progress_index > self.path_length:
            raise ValueError("progress_index cannot exceed path_length")

    @property
    def is_terminal(self) -> bool:
        return self.progress_index == self.path_length

    @property
    def trace_key(
        self,
    ) -> tuple[str, str, int, str, str, str, str, str, int]:
        return (
            self.packed_shard_content_sha256,
            self.packed_shard_name,
            self.entry_index,
            self.layer,
            self.partition,
            self.trace_id,
            self.trace_source_key,
            self.trace_target_key,
            self.path_length,
        )

    @classmethod
    def from_packed_trace(
        cls,
        address: PackedTraceAddress,
        *,
        progress_index: int,
    ) -> SuccessorFiberCacheAddress:
        return cls(
            packed_shard_content_sha256=address.packed_shard_content_sha256,
            packed_shard_name=address.packed_shard_name,
            entry_index=address.entry_index,
            layer=address.layer,
            partition=address.partition,
            trace_id=address.trace_id,
            trace_source_key=address.source_key,
            trace_target_key=address.target_key,
            progress_index=progress_index,
            path_length=address.path_length,
        )


@dataclass(frozen=True)
class SuccessorFiberCacheRecord:
    """One precompiled jump fiber or terminal productive-support row."""

    address: SuccessorFiberCacheAddress
    state_support: StateProductiveSupport
    teacher_fiber: TeacherSuccessorFiber | None

    def __post_init__(self) -> None:
        if self.address.is_terminal:
            if self.teacher_fiber is not None:
                raise ValueError(
                    "a terminal progress row cannot carry a teacher-successor fiber"
                )
        elif self.teacher_fiber is None:
            raise ValueError(
                "a nonterminal progress row must carry a teacher-successor fiber"
            )
        if self.teacher_fiber is not None:
            if self.teacher_fiber.state_support != self.state_support:
                raise ValueError(
                    "record support disagrees with the teacher fiber's support"
                )
            source_key = self.teacher_fiber.source_key
        else:
            source_key = self.state_support.source_key
        if source_key != self.state_support.source_key:
            raise ValueError("record source keys disagree")

    @property
    def source_key(self) -> str:
        return self.state_support.source_key

    @property
    def target_key(self) -> str | None:
        return (
            None
            if self.teacher_fiber is None
            else self.teacher_fiber.target_key
        )

    @property
    def source_state_sha256(self) -> str:
        return self.state_support.source_state_sha256

    @property
    def target_state_sha256(self) -> str | None:
        return (
            None
            if self.teacher_fiber is None
            else self.teacher_fiber.target_state_sha256
        )


@dataclass(frozen=True)
class SuccessorFiberCache:
    """Validated cache artifact returned by the reader and writer."""

    provenance: SuccessorFiberCacheProvenance
    records: tuple[SuccessorFiberCacheRecord, ...]
    content_sha256: str
    _exact_record_index: Mapping[
        tuple[str, int, int],
        SuccessorFiberCacheRecord,
    ] = field(init=False, repr=False, compare=False)
    _legacy_record_index: Mapping[
        tuple[str, str, str, int],
        SuccessorFiberCacheRecord,
    ] = field(init=False, repr=False, compare=False)
    _legacy_trace_index: Mapping[
        tuple[str, str, str],
        tuple[SuccessorFiberCacheRecord, ...],
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        exact_record_index: dict[
            tuple[str, int, int],
            SuccessorFiberCacheRecord,
        ] = {}
        legacy_record_index: dict[
            tuple[str, str, str, int],
            SuccessorFiberCacheRecord,
        ] = {}
        legacy_trace_index: dict[
            tuple[str, str, str],
            list[SuccessorFiberCacheRecord],
        ] = defaultdict(list)
        for record in self.records:
            exact_key = (
                record.address.packed_shard_content_sha256,
                record.address.entry_index,
                record.address.progress_index,
            )
            if exact_key in exact_record_index:
                raise ValueError(f"duplicate exact cache lookup key {exact_key!r}")
            exact_record_index[exact_key] = record
            legacy_key = (
                record.address.layer,
                record.address.partition,
                record.address.trace_id,
                record.address.progress_index,
            )
            if legacy_key in legacy_record_index:
                raise ValueError(
                    f"duplicate legacy cache lookup key {legacy_key!r}"
                )
            legacy_record_index[legacy_key] = record
            legacy_trace_index[legacy_key[:3]].append(record)
        object.__setattr__(
            self,
            "_exact_record_index",
            MappingProxyType(exact_record_index),
        )
        object.__setattr__(
            self,
            "_legacy_record_index",
            MappingProxyType(legacy_record_index),
        )
        object.__setattr__(
            self,
            "_legacy_trace_index",
            MappingProxyType(
                {
                    key: tuple(records)
                    for key, records in legacy_trace_index.items()
                }
            ),
        )

    def record_at(
        self,
        *,
        layer: str,
        partition: str,
        trace_id: str,
        progress_index: int,
    ) -> SuccessorFiberCacheRecord:
        """Return one exact trace-progress record, refusing ambiguous absence."""

        key = (layer, partition, trace_id, progress_index)
        try:
            return self._legacy_record_index[key]
        except KeyError:
            raise KeyError(
                "cache has no unique record for "
                f"{key!r}"
            ) from None

    def records_for_trace(
        self,
        *,
        layer: str,
        partition: str,
        trace_id: str,
    ) -> tuple[SuccessorFiberCacheRecord, ...]:
        """Return a complete trace in progress order."""

        key = (layer, partition, trace_id)
        try:
            return self._legacy_trace_index[key]
        except KeyError:
            raise KeyError(
                f"cache has no trace {key!r}"
            ) from None

    def require_packed_address(
        self,
        address: PackedTraceAddress,
        *,
        progress_index: int,
    ) -> SuccessorFiberCacheRecord:
        """Return one O(1) row and require its full immutable packed identity."""

        exact_key = (
            address.packed_shard_content_sha256,
            address.entry_index,
            progress_index,
        )
        try:
            record = self._exact_record_index[exact_key]
        except KeyError:
            raise KeyError(
                f"cache has no exact packed row {exact_key!r}"
            ) from None
        expected = SuccessorFiberCacheAddress.from_packed_trace(
            address,
            progress_index=progress_index,
        )
        if record.address != expected:
            raise KeyError(
                "cache record does not match the exact packed trace address"
            )
        return record


def fiber_compiler_implementation_hash() -> str:
    """Hash the code that maps production support into table coordinates."""

    repository = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for relative in _COMPILER_IMPLEMENTATION_SOURCES:
        source = repository / relative
        digest.update(relative.encode("utf-8"))
        digest.update(source.read_bytes() if source.exists() else b"<MISSING>")
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SuccessorFiberCacheError(
            "cache payload is not canonical-JSON serializable"
        ) from error
    return text.encode("utf-8")


def _content_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _provenance_payload(
    provenance: SuccessorFiberCacheProvenance,
) -> dict[str, Any]:
    return {
        name: getattr(provenance, name)
        for name in _PROVENANCE_FIELDS
    }


def _validate_text(value: str, *, name: str, limits: SuccessorFiberCacheLimits) -> None:
    if not isinstance(value, str) or not value:
        raise SuccessorFiberCacheError(f"{name} must be a nonempty string")
    if len(value.encode("utf-8")) > limits.max_key_bytes:
        raise SuccessorFiberCacheError(f"{name} exceeds the cache key-size bound")


def _validate_alias(
    alias: TeacherSuccessorAlias,
    *,
    limits: SuccessorFiberCacheLimits,
) -> None:
    _validate_text(alias.family_name, name="alias family_name", limits=limits)
    _validate_text(alias.table_name, name="alias table_name", limits=limits)
    coordinate = alias.coordinate
    if not isinstance(coordinate, tuple) or not coordinate:
        raise SuccessorFiberCacheError(
            "alias coordinate must be a nonempty tuple"
        )
    if len(coordinate) > limits.max_coordinate_rank:
        raise SuccessorFiberCacheError("alias coordinate rank exceeds the bound")
    for value in coordinate:
        if (
            type(value) is not int
            or value < 0
            or value > limits.max_coordinate_value
        ):
            raise SuccessorFiberCacheError(
                "alias coordinates must be bounded nonnegative integers"
            )


def _validate_aliases(
    aliases: tuple[TeacherSuccessorAlias, ...],
    *,
    name: str,
    limits: SuccessorFiberCacheLimits,
) -> None:
    if aliases != tuple(sorted(aliases)):
        raise SuccessorFiberCacheError(
            f"{name} must be in canonical lexicographic order"
        )
    if len(aliases) != len(set(aliases)):
        raise SuccessorFiberCacheError(f"{name} contains duplicate coordinates")
    for alias in aliases:
        _validate_alias(alias, limits=limits)


def _validate_record(
    record: SuccessorFiberCacheRecord,
    *,
    provenance: SuccessorFiberCacheProvenance,
    limits: SuccessorFiberCacheLimits,
) -> int:
    address = record.address
    if (
        address.packed_shard_content_sha256
        != provenance.packed_shard_content_sha256
    ):
        raise SuccessorFiberCacheError(
            "record packed-shard digest disagrees with cache provenance"
        )
    for name, value in (
        ("packed_shard_name", address.packed_shard_name),
        ("layer", address.layer),
        ("partition", address.partition),
        ("trace_id", address.trace_id),
        ("trace_source_key", address.trace_source_key),
        ("trace_target_key", address.trace_target_key),
        ("source_key", record.source_key),
        ("source_state_sha256", record.source_state_sha256),
    ):
        _validate_text(value, name=name, limits=limits)
    _validate_aliases(
        record.state_support.virtual_aliases,
        name="virtual_aliases",
        limits=limits,
    )
    alias_count = len(record.state_support.virtual_aliases)
    if record.teacher_fiber is not None:
        _validate_text(
            record.teacher_fiber.target_key,
            name="target_key",
            limits=limits,
        )
        _validate_text(
            record.teacher_fiber.target_state_sha256,
            name="target_state_sha256",
            limits=limits,
        )
        _validate_aliases(
            record.teacher_fiber.aliases,
            name="teacher aliases",
            limits=limits,
        )
        alias_count += len(record.teacher_fiber.aliases)
    if alias_count > limits.max_aliases_per_record:
        raise SuccessorFiberCacheError(
            "record alias count exceeds the per-record bound"
        )
    return alias_count


def validate_successor_fiber_cache_record(
    record: SuccessorFiberCacheRecord,
    *,
    provenance: SuccessorFiberCacheProvenance,
    limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
) -> int:
    """Validate one decoded record under the shared backend contract."""

    return _validate_record(
        record,
        provenance=provenance,
        limits=limits,
    )


def _canonical_records(
    records: Iterable[SuccessorFiberCacheRecord],
    *,
    provenance: SuccessorFiberCacheProvenance,
    limits: SuccessorFiberCacheLimits,
) -> tuple[SuccessorFiberCacheRecord, ...]:
    ordered = tuple(sorted(records, key=lambda record: record.address))
    if len(ordered) > limits.max_records:
        raise SuccessorFiberCacheError("record count exceeds the cache bound")
    identities = tuple(record.address for record in ordered)
    if len(identities) != len(set(identities)):
        raise SuccessorFiberCacheError(
            "cache contains duplicate trace-progress addresses"
        )

    total_aliases = sum(
        _validate_record(
            record,
            provenance=provenance,
            limits=limits,
        )
        for record in ordered
    )
    if total_aliases > limits.max_total_aliases:
        raise SuccessorFiberCacheError(
            "total alias count exceeds the cache bound"
        )

    grouped: dict[
        tuple[str, str, int, str, str, str, str, str, int],
        list[SuccessorFiberCacheRecord],
    ] = defaultdict(list)
    for record in ordered:
        grouped[record.address.trace_key].append(record)
    for trace_key, trace_records in grouped.items():
        path_lengths = {record.address.path_length for record in trace_records}
        if len(path_lengths) != 1:
            raise SuccessorFiberCacheError(
                f"trace {trace_key!r} has inconsistent path lengths"
            )
        path_length = next(iter(path_lengths))
        progresses = [record.address.progress_index for record in trace_records]
        if progresses != list(range(path_length + 1)):
            raise SuccessorFiberCacheError(
                f"trace {trace_key!r} does not contain every progress row "
                f"0..{path_length}"
            )
        for current, following in pairwise(trace_records):
            if (
                current.target_key != following.source_key
                or current.target_state_sha256
                != following.source_state_sha256
            ):
                raise SuccessorFiberCacheError(
                    f"trace {trace_key!r} target/source chain breaks between "
                    f"progress {current.address.progress_index} and "
                    f"{following.address.progress_index}"
                )
        if (
            trace_records[0].source_key
            != trace_records[0].address.trace_source_key
        ):
            raise SuccessorFiberCacheError(
                f"trace {trace_key!r} first state disagrees with its packed source"
            )
        if (
            trace_records[-1].source_key
            != trace_records[-1].address.trace_target_key
        ):
            raise SuccessorFiberCacheError(
                f"trace {trace_key!r} terminal state disagrees with its packed target"
            )
    return ordered


def canonical_successor_fiber_records(
    records: Iterable[SuccessorFiberCacheRecord],
    *,
    provenance: SuccessorFiberCacheProvenance,
    limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
) -> tuple[SuccessorFiberCacheRecord, ...]:
    """Validate and canonically order records for any immutable backend."""

    return _canonical_records(
        records,
        provenance=provenance,
        limits=limits,
    )


def _alias_payload(alias: TeacherSuccessorAlias) -> dict[str, Any]:
    return {
        "family_name": alias.family_name,
        "table_name": alias.table_name,
        "coordinate": list(alias.coordinate),
    }


def _record_payload(record: SuccessorFiberCacheRecord) -> dict[str, Any]:
    aliases = (
        ()
        if record.teacher_fiber is None
        else record.teacher_fiber.aliases
    )
    return {
        "packed_shard_content_sha256": (
            record.address.packed_shard_content_sha256
        ),
        "packed_shard_name": record.address.packed_shard_name,
        "entry_index": record.address.entry_index,
        "layer": record.address.layer,
        "partition": record.address.partition,
        "trace_id": record.address.trace_id,
        "trace_source_key": record.address.trace_source_key,
        "trace_target_key": record.address.trace_target_key,
        "progress_index": record.address.progress_index,
        "path_length": record.address.path_length,
        "source_key": record.source_key,
        "source_state_sha256": record.source_state_sha256,
        "target_key": record.target_key,
        "target_state_sha256": record.target_state_sha256,
        "aliases": [_alias_payload(alias) for alias in aliases],
        "virtual_aliases": [
            _alias_payload(alias)
            for alias in record.state_support.virtual_aliases
        ],
    }


def serialize_successor_fiber_cache(
    records: Iterable[SuccessorFiberCacheRecord],
    *,
    provenance: SuccessorFiberCacheProvenance,
    limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
) -> tuple[bytes, SuccessorFiberCache]:
    """Return canonical artifact bytes and the validated in-memory cache."""

    ordered = _canonical_records(
        records,
        provenance=provenance,
        limits=limits,
    )
    payload = {
        "schema": SUCCESSOR_FIBER_CACHE_SCHEMA,
        "schema_version": SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION,
        "provenance": _provenance_payload(provenance),
        "records": [_record_payload(record) for record in ordered],
    }
    content_sha256 = _content_sha256(payload)
    envelope = {
        **payload,
        "content_sha256": content_sha256,
    }
    encoded = _canonical_json_bytes(envelope) + b"\n"
    if len(encoded) > limits.max_file_bytes:
        raise SuccessorFiberCacheError("serialized cache exceeds the file bound")
    return encoded, SuccessorFiberCache(
        provenance=provenance,
        records=ordered,
        content_sha256=content_sha256,
    )


def _require_exact_fields(
    payload: dict[str, Any],
    fields: tuple[str, ...],
    *,
    name: str,
) -> None:
    if not isinstance(payload, dict):
        raise SuccessorFiberCacheError(f"{name} must be an object")
    expected, observed = set(fields), set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise SuccessorFiberCacheError(
            f"{name} fields disagree with the schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _decode_alias(payload: Any) -> TeacherSuccessorAlias:
    _require_exact_fields(payload, _ALIAS_FIELDS, name="alias")
    coordinate = payload["coordinate"]
    if not isinstance(coordinate, list):
        raise SuccessorFiberCacheError("alias coordinate must be an array")
    try:
        return TeacherSuccessorAlias(
            family_name=payload["family_name"],
            table_name=payload["table_name"],
            coordinate=tuple(coordinate),
        )
    except (TypeError, ValueError) as error:
        raise SuccessorFiberCacheError("invalid alias payload") from error


def _decode_record(payload: Any) -> SuccessorFiberCacheRecord:
    _require_exact_fields(payload, _RECORD_FIELDS, name="record")
    aliases_payload = payload["aliases"]
    virtual_payload = payload["virtual_aliases"]
    if not isinstance(aliases_payload, list) or not isinstance(
        virtual_payload, list
    ):
        raise SuccessorFiberCacheError("record alias fields must be arrays")
    try:
        address = SuccessorFiberCacheAddress(
            packed_shard_content_sha256=payload[
                "packed_shard_content_sha256"
            ],
            packed_shard_name=payload["packed_shard_name"],
            entry_index=payload["entry_index"],
            layer=payload["layer"],
            partition=payload["partition"],
            trace_id=payload["trace_id"],
            trace_source_key=payload["trace_source_key"],
            trace_target_key=payload["trace_target_key"],
            progress_index=payload["progress_index"],
            path_length=payload["path_length"],
        )
        virtual_aliases = tuple(_decode_alias(item) for item in virtual_payload)
        support = StateProductiveSupport(
            source_key=payload["source_key"],
            source_state_sha256=payload["source_state_sha256"],
            virtual_aliases=virtual_aliases,
        )
        target_key = payload["target_key"]
        target_state_sha256 = payload["target_state_sha256"]
        aliases = tuple(_decode_alias(item) for item in aliases_payload)
        if target_key is None:
            if aliases or target_state_sha256 is not None:
                raise SuccessorFiberCacheError(
                    "terminal record cannot carry teacher aliases or target state"
                )
            fiber = None
        else:
            if not isinstance(target_key, str) or not isinstance(
                target_state_sha256,
                str,
            ):
                raise SuccessorFiberCacheError(
                    "jump record target identity must use strings"
                )
            fiber = TeacherSuccessorFiber(
                source_key=payload["source_key"],
                target_key=target_key,
                target_state_sha256=target_state_sha256,
                aliases=aliases,
                state_support=support,
            )
        return SuccessorFiberCacheRecord(
            address=address,
            state_support=support,
            teacher_fiber=fiber,
        )
    except SuccessorFiberCacheError:
        raise
    except (TypeError, ValueError) as error:
        raise SuccessorFiberCacheError("invalid cache record") from error


def _decode_provenance(payload: Any) -> SuccessorFiberCacheProvenance:
    _require_exact_fields(payload, _PROVENANCE_FIELDS, name="provenance")
    try:
        return SuccessorFiberCacheProvenance(
            **{name: payload[name] for name in _PROVENANCE_FIELDS}
        )
    except (TypeError, ValueError) as error:
        raise SuccessorFiberCacheError("invalid cache provenance") from error


def deserialize_successor_fiber_cache(
    encoded: bytes,
    *,
    expected_provenance: SuccessorFiberCacheProvenance,
    expected_content_sha256: str,
    limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
) -> SuccessorFiberCache:
    """Validate and decode an artifact under an exact expected contract."""

    if len(encoded) > limits.max_file_bytes:
        raise SuccessorFiberCacheError("cache exceeds the file bound")
    try:
        envelope = json.loads(
            encoded,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SuccessorFiberCacheError("cache is not valid finite JSON") from error
    _require_exact_fields(envelope, _ROOT_FIELDS, name="cache root")
    if envelope["schema"] != SUCCESSOR_FIBER_CACHE_SCHEMA:
        raise SuccessorFiberCacheError(
            f"cache schema {envelope['schema']!r} is not supported"
        )
    if envelope["schema_version"] != SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION:
        raise SuccessorFiberCacheError(
            f"cache schema version {envelope['schema_version']!r} is not supported"
        )
    recorded_hash = envelope["content_sha256"]
    if (
        not isinstance(recorded_hash, str)
        or len(recorded_hash) != 64
        or any(character not in "0123456789abcdef" for character in recorded_hash)
    ):
        raise SuccessorFiberCacheError(
            "cache content_sha256 is not a lowercase SHA-256 digest"
        )
    if (
        not isinstance(expected_content_sha256, str)
        or len(expected_content_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_content_sha256
        )
    ):
        raise SuccessorFiberCacheError(
            "expected_content_sha256 is not a lowercase SHA-256 digest"
        )
    payload = {
        key: envelope[key]
        for key in ("schema", "schema_version", "provenance", "records")
    }
    observed_hash = _content_sha256(payload)
    if observed_hash != recorded_hash:
        raise SuccessorFiberCacheError(
            "cache content hash mismatch; artifact was modified or truncated"
        )
    if recorded_hash != expected_content_sha256:
        raise SuccessorFiberCacheError(
            "cache content hash does not match the frozen expected artifact"
        )
    provenance = _decode_provenance(envelope["provenance"])
    if provenance != expected_provenance:
        disagreements = {
            name: {
                "artifact": getattr(provenance, name),
                "expected": getattr(expected_provenance, name),
            }
            for name in _PROVENANCE_FIELDS
            if getattr(provenance, name) != getattr(expected_provenance, name)
        }
        raise SuccessorFiberCacheError(
            f"cache provenance mismatch: {disagreements}"
        )
    records_payload = envelope["records"]
    if not isinstance(records_payload, list):
        raise SuccessorFiberCacheError("cache records must be an array")
    if len(records_payload) > limits.max_records:
        raise SuccessorFiberCacheError("record count exceeds the cache bound")
    records = tuple(_decode_record(record) for record in records_payload)
    ordered = _canonical_records(
        records,
        provenance=provenance,
        limits=limits,
    )
    if records != ordered:
        raise SuccessorFiberCacheError(
            "cache records are not in canonical address order"
        )
    return SuccessorFiberCache(
        provenance=provenance,
        records=ordered,
        content_sha256=recorded_hash,
    )


def write_successor_fiber_cache(
    path: Path,
    records: Iterable[SuccessorFiberCacheRecord],
    *,
    provenance: SuccessorFiberCacheProvenance,
    limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
) -> SuccessorFiberCache:
    """Atomically freeze one deterministic cache shard without overwriting."""

    encoded, cache = serialize_successor_fiber_cache(
        records,
        provenance=provenance,
        limits=limits,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if destination.read_bytes() != encoded:
                raise FileExistsError(
                    "successor-fiber cache already exists with different "
                    f"content: {destination}"
                ) from None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return cache


def read_successor_fiber_cache(
    path: Path,
    *,
    expected_provenance: SuccessorFiberCacheProvenance,
    expected_content_sha256: str,
    limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
) -> SuccessorFiberCache:
    """Read one cache shard after size, integrity and provenance checks."""

    source = Path(path)
    if not source.is_file():
        raise SuccessorFiberCacheError(f"cache file does not exist: {source}")
    if source.stat().st_size > limits.max_file_bytes:
        raise SuccessorFiberCacheError("cache exceeds the file bound")
    return deserialize_successor_fiber_cache(
        source.read_bytes(),
        expected_provenance=expected_provenance,
        expected_content_sha256=expected_content_sha256,
        limits=limits,
    )
