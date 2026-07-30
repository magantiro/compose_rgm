"""Deterministic construction of a bounded Gate-0 successor probe.

The probe is a validation artifact, not a training cache and not a gate
decision.  It selects complete immutable ``PathRecord`` traces whose
nonterminal rows cover an explicit set of model families and semantic cells.
Every progress row of each selected trace is then compiled through the
production successor-fiber builder and serialized through the existing cache
schema.

No numeric quality threshold lives here.  The only selection criterion is
set coverage.  The resulting object is structurally incapable of authorizing
training.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
    SuccessorFiberCache,
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheLimits,
    SuccessorFiberCacheProvenance,
    SuccessorFiberCacheRecord,
    deserialize_successor_fiber_cache,
    fiber_compiler_implementation_hash,
    serialize_successor_fiber_cache,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_p50_gate import (
    state_dict_semantic_sha256,
)
from compose_v4.experiments.factorized_successor_data import (
    SuccessorSemanticCellKey,
)
from compose_v4.experiments.factorized_successor_training import (
    rewrite_action_codec_sha256,
)
from compose_v4.experiments.successor_fiber_cache_builder import (
    SuccessorFiberCacheBuildError,
    compile_successor_fiber_trace,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_NAMES,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system

GATE_ZERO_SUCCESSOR_PROBE_SCHEMA = "compose.editing.gate_zero_successor_probe"
GATE_ZERO_SUCCESSOR_PROBE_SCHEMA_VERSION = 1
GATE_ZERO_SUCCESSOR_PROBE_STATUS = "BOUNDED_VALIDATION_PROBE_ONLY"
GATE_ZERO_SUCCESSOR_PROBE_PARTITION = "validation"
GATE_ZERO_SUCCESSOR_SUPPORT_TIME = 0.5
GATE_ZERO_SUCCESSOR_SELECTION_ALGORITHM = "deterministic_complete_trace_greedy_set_cover_v1"

_DISABLED_PRODUCTION_FAMILIES = frozenset({"ring_system_grow"})
_PRODUCTION_FAMILIES = frozenset(MARK_RULE_NAMES) - _DISABLED_PRODUCTION_FAMILIES


class GateZeroSuccessorProbeError(RuntimeError):
    """A bounded Gate-0 probe cannot be built under the exact contract."""


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
        raise GateZeroSuccessorProbeError(
            "Gate-0 probe metadata is not finite canonical JSON"
        ) from error


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalize_required_text(
    values: tuple[str, ...],
    *,
    field: str,
) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field} must be a nonempty tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must contain nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} cannot contain duplicates")
    return tuple(sorted(values))


@dataclass(frozen=True)
class GateZeroSuccessorProbeConfig:
    """Threshold-free coverage obligations for one validation probe."""

    required_families: tuple[str, ...]
    required_semantic_cells: tuple[str, ...]

    def __post_init__(self) -> None:
        families = _normalize_required_text(
            self.required_families,
            field="required_families",
        )
        semantic_cells = _normalize_required_text(
            self.required_semantic_cells,
            field="required_semantic_cells",
        )
        unknown = tuple(sorted(set(families) - _PRODUCTION_FAMILIES))
        if unknown:
            raise ValueError(
                f"required_families contains disabled or unknown production families: {unknown!r}"
            )
        object.__setattr__(self, "required_families", families)
        object.__setattr__(
            self,
            "required_semantic_cells",
            semantic_cells,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "partition": GATE_ZERO_SUCCESSOR_PROBE_PARTITION,
            "required_families": list(self.required_families),
            "required_semantic_cells": list(self.required_semantic_cells),
            "selection_algorithm": GATE_ZERO_SUCCESSOR_SELECTION_ALGORITHM,
            "support_time": GATE_ZERO_SUCCESSOR_SUPPORT_TIME,
        }

    @property
    def sha256(self) -> str:
        return _stable_sha256(self.payload())


@dataclass(frozen=True)
class GateZeroSuccessorProbeRow:
    """Semantic annotation for one retained cache row."""

    address: SuccessorFiberCacheAddress
    family_name: str | None
    semantic_cell_id: str | None
    teacher_action_sha256: str | None

    def __post_init__(self) -> None:
        if self.address.is_terminal:
            if (
                self.family_name is not None
                or self.semantic_cell_id is not None
                or self.teacher_action_sha256 is not None
            ):
                raise ValueError("a terminal probe row cannot carry teacher annotations")
            return
        if (
            self.family_name not in _PRODUCTION_FAMILIES
            or not isinstance(self.semantic_cell_id, str)
            or not self.semantic_cell_id
            or not _is_sha256(self.teacher_action_sha256)
        ):
            raise ValueError(
                "a nonterminal probe row requires a production family, "
                "semantic cell, and teacher-action SHA-256"
            )

    @property
    def exact_key(self) -> SuccessorSemanticCellKey:
        return (
            self.address.packed_shard_content_sha256,
            self.address.entry_index,
            self.address.progress_index,
        )


@dataclass(frozen=True)
class GateZeroSuccessorProbeCacheShard:
    """One ordinary successor-cache artifact retained by the probe."""

    encoded: bytes
    cache: SuccessorFiberCache

    def __post_init__(self) -> None:
        if not isinstance(self.encoded, bytes) or not self.encoded:
            raise ValueError("encoded cache shard must be nonempty bytes")
        if not isinstance(self.cache, SuccessorFiberCache):
            raise TypeError("cache must use the production cache artifact type")
        try:
            decoded = deserialize_successor_fiber_cache(
                self.encoded,
                expected_provenance=self.cache.provenance,
                expected_content_sha256=self.cache.content_sha256,
            )
        except Exception as error:
            raise ValueError(
                "encoded cache shard does not satisfy the production cache schema"
            ) from error
        if decoded != self.cache:
            raise ValueError("encoded cache shard disagrees with its decoded cache object")

    @property
    def packed_shard_content_sha256(self) -> str:
        return self.cache.provenance.packed_shard_content_sha256

    @property
    def content_sha256(self) -> str:
        return self.cache.content_sha256

    @property
    def encoded_sha256(self) -> str:
        """Exact serialized-byte identity of this ordinary cache artifact."""

        return hashlib.sha256(self.encoded).hexdigest()


@dataclass(frozen=True)
class GateZeroSuccessorProbe:
    """Self-hashed bounded evidence bundle with no training authority."""

    config: GateZeroSuccessorProbeConfig
    config_sha256: str
    source_sha256: str
    semantic_cell_sidecar_sha256: str
    compiler_model_state_sha256: str
    selected_path_records: tuple[PathRecord, ...]
    rows: tuple[GateZeroSuccessorProbeRow, ...]
    cache_shards: tuple[GateZeroSuccessorProbeCacheShard, ...]
    family_witnesses: Mapping[str, tuple[SuccessorFiberCacheAddress, ...]]
    semantic_cell_witnesses: Mapping[
        str,
        tuple[SuccessorFiberCacheAddress, ...],
    ]
    probe_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "config_sha256",
            "source_sha256",
            "semantic_cell_sidecar_sha256",
            "compiler_model_state_sha256",
            "probe_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.config_sha256 != self.config.sha256:
            raise ValueError("probe config hash disagrees with its config")
        if not self.selected_path_records or not self.rows or not self.cache_shards:
            raise ValueError("a Gate-0 successor probe cannot be empty")
        selected_addresses = tuple(
            _require_address(record) for record in self.selected_path_records
        )
        if selected_addresses != tuple(sorted(selected_addresses, key=_address_sort_key)):
            raise ValueError("selected PathRecords are not in canonical order")
        if any(
            address.partition != GATE_ZERO_SUCCESSOR_PROBE_PARTITION
            for address in selected_addresses
        ):
            raise ValueError("selected PathRecords are not validation records")
        expected_row_addresses = tuple(
            sorted(
                SuccessorFiberCacheAddress.from_packed_trace(
                    address,
                    progress_index=progress_index,
                )
                for address in selected_addresses
                for progress_index in range(address.path_length + 1)
            )
        )
        row_addresses = tuple(row.address for row in self.rows)
        if row_addresses != expected_row_addresses:
            raise ValueError("probe rows do not exactly retain every selected trace progress")
        cache_shard_keys = tuple(shard.packed_shard_content_sha256 for shard in self.cache_shards)
        if cache_shard_keys != tuple(sorted(set(cache_shard_keys))):
            raise ValueError("probe cache shards are not canonical and unique")
        cache_addresses = tuple(
            sorted(record.address for shard in self.cache_shards for record in shard.cache.records)
        )
        if cache_addresses != row_addresses:
            raise ValueError("probe annotations and production cache rows are not aligned")
        family_witnesses = {
            name: tuple(addresses) for name, addresses in self.family_witnesses.items()
        }
        cell_witnesses = {
            name: tuple(addresses) for name, addresses in self.semantic_cell_witnesses.items()
        }
        if set(family_witnesses) != set(self.config.required_families):
            raise ValueError("family witnesses must exactly cover required families")
        if set(cell_witnesses) != set(self.config.required_semantic_cells):
            raise ValueError("semantic-cell witnesses must exactly cover required cells")
        expected_family_witnesses = {
            family: tuple(row.address for row in self.rows if row.family_name == family)
            for family in self.config.required_families
        }
        expected_cell_witnesses = {
            cell: tuple(row.address for row in self.rows if row.semantic_cell_id == cell)
            for cell in self.config.required_semantic_cells
        }
        if family_witnesses != expected_family_witnesses or any(
            not addresses for addresses in family_witnesses.values()
        ):
            raise ValueError("required-family witnesses disagree with probe rows")
        if cell_witnesses != expected_cell_witnesses or any(
            not addresses for addresses in cell_witnesses.values()
        ):
            raise ValueError("required semantic-cell witnesses disagree with probe rows")
        object.__setattr__(
            self,
            "family_witnesses",
            MappingProxyType(family_witnesses),
        )
        object.__setattr__(
            self,
            "semantic_cell_witnesses",
            MappingProxyType(cell_witnesses),
        )
        if self.probe_sha256 != _stable_sha256(self._probe_payload()):
            raise ValueError("probe hash disagrees with its exact contents")

    @property
    def training_authorized(self) -> bool:
        """This diagnostic artifact can never authorize optimization."""

        return False

    def assert_training_authorized(self) -> None:
        raise GateZeroSuccessorProbeError(
            "a bounded Gate-0 successor probe is diagnostic validation "
            "evidence and never authorizes training"
        )

    @property
    def semantic_cell_ids(
        self,
    ) -> Mapping[SuccessorSemanticCellKey, str | None]:
        return MappingProxyType({row.exact_key: row.semantic_cell_id for row in self.rows})

    @property
    def cache_records(self) -> tuple[SuccessorFiberCacheRecord, ...]:
        return tuple(record for shard in self.cache_shards for record in shard.cache.records)

    def _probe_payload(self) -> dict[str, Any]:
        return _gate_zero_probe_payload(
            config_sha256=self.config_sha256,
            source_sha256=self.source_sha256,
            semantic_cell_sidecar_sha256=self.semantic_cell_sidecar_sha256,
            compiler_model_state_sha256=self.compiler_model_state_sha256,
            selected_path_records=self.selected_path_records,
            rows=self.rows,
            cache_shards=self.cache_shards,
            family_witnesses=self.family_witnesses,
            semantic_cell_witnesses=self.semantic_cell_witnesses,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            **self._probe_payload(),
            "config": self.config.payload(),
            "probe_sha256": self.probe_sha256,
        }


@dataclass(frozen=True)
class _ScannedRow:
    address: SuccessorFiberCacheAddress
    source_state_sha256: str
    target_state_sha256: str | None
    family_name: str | None
    semantic_cell_id: str | None
    teacher_action_sha256: str | None

    @property
    def obligation_keys(self) -> frozenset[tuple[str, str]]:
        obligations: set[tuple[str, str]] = set()
        if self.family_name is not None:
            obligations.add(("family", self.family_name))
        if self.semantic_cell_id is not None:
            obligations.add(("semantic_cell", self.semantic_cell_id))
        return frozenset(obligations)


@dataclass(frozen=True)
class _ScannedPath:
    record: PathRecord
    address: PackedTraceAddress
    rows: tuple[_ScannedRow, ...]

    @property
    def obligations(self) -> frozenset[tuple[str, str]]:
        return frozenset(obligation for row in self.rows for obligation in row.obligation_keys)


def _require_address(record: PathRecord) -> PackedTraceAddress:
    address = record.corpus_address
    if address is None:
        raise GateZeroSuccessorProbeError(
            "Gate-0 successor probes require immutable packed addresses"
        )
    return address


def _address_sort_key(address: PackedTraceAddress) -> tuple[Any, ...]:
    return (
        address.packed_shard_content_sha256,
        address.packed_shard_name,
        address.entry_index,
        address.layer,
        address.partition,
        address.trace_id,
        address.source_key,
        address.target_key,
        address.path_length,
    )


def _probe_row_payload(row: GateZeroSuccessorProbeRow) -> dict[str, Any]:
    return {
        "address": asdict(row.address),
        "family_name": row.family_name,
        "semantic_cell_id": row.semantic_cell_id,
        "teacher_action_sha256": row.teacher_action_sha256,
    }


def _gate_zero_probe_payload(
    *,
    config_sha256: str,
    source_sha256: str,
    semantic_cell_sidecar_sha256: str,
    compiler_model_state_sha256: str,
    selected_path_records: tuple[PathRecord, ...],
    rows: tuple[GateZeroSuccessorProbeRow, ...],
    cache_shards: tuple[GateZeroSuccessorProbeCacheShard, ...],
    family_witnesses: Mapping[
        str,
        tuple[SuccessorFiberCacheAddress, ...],
    ],
    semantic_cell_witnesses: Mapping[
        str,
        tuple[SuccessorFiberCacheAddress, ...],
    ],
) -> dict[str, Any]:
    return {
        "schema": GATE_ZERO_SUCCESSOR_PROBE_SCHEMA,
        "schema_version": GATE_ZERO_SUCCESSOR_PROBE_SCHEMA_VERSION,
        "status": GATE_ZERO_SUCCESSOR_PROBE_STATUS,
        "training_authorized": False,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "semantic_cell_sidecar_sha256": semantic_cell_sidecar_sha256,
        "compiler_model_state_sha256": compiler_model_state_sha256,
        "selected_path_addresses": [
            asdict(_require_address(record)) for record in selected_path_records
        ],
        "rows": [_probe_row_payload(row) for row in rows],
        "cache_shards": [
            {
                "packed_shard_content_sha256": (shard.packed_shard_content_sha256),
                "cache_content_sha256": shard.content_sha256,
                "cache_encoded_sha256": shard.encoded_sha256,
                "record_count": len(shard.cache.records),
            }
            for shard in cache_shards
        ],
        "family_witnesses": {
            family: [asdict(address) for address in addresses]
            for family, addresses in sorted(family_witnesses.items())
        },
        "semantic_cell_witnesses": {
            cell: [asdict(address) for address in addresses]
            for cell, addresses in sorted(semantic_cell_witnesses.items())
        },
    }


def _validate_semantic_cell_sidecar(
    semantic_cell_ids: Mapping[
        SuccessorSemanticCellKey,
        str | None,
    ],
) -> None:
    for key, value in semantic_cell_ids.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or not _is_sha256(key[0])
            or type(key[1]) is not int
            or key[1] < 0
            or type(key[2]) is not int
            or key[2] < 0
            or (value is not None and (not isinstance(value, str) or not value))
        ):
            raise GateZeroSuccessorProbeError(
                "semantic-cell sidecar contains a malformed exact row"
            )


def _scan_source_records(
    records: Sequence[PathRecord],
    semantic_cell_ids: Mapping[
        SuccessorSemanticCellKey,
        str | None,
    ],
) -> tuple[tuple[_ScannedPath, ...], str, str]:
    if not records:
        raise GateZeroSuccessorProbeError("Gate-0 successor probe source cannot be empty")
    if not isinstance(semantic_cell_ids, Mapping):
        raise TypeError("semantic_cell_ids must be an exact-address mapping")
    _validate_semantic_cell_sidecar(semantic_cell_ids)

    materialized = tuple(records)
    if any(not isinstance(record, PathRecord) for record in materialized):
        raise TypeError("records must contain PathRecord values only")
    addresses = tuple(_require_address(record) for record in materialized)
    if any(address.partition != GATE_ZERO_SUCCESSOR_PROBE_PARTITION for address in addresses):
        raise GateZeroSuccessorProbeError(
            "Gate-0 successor probes accept validation PathRecords only"
        )
    exact_trace_keys = tuple(
        (address.packed_shard_content_sha256, address.entry_index) for address in addresses
    )
    if len(exact_trace_keys) != len(set(exact_trace_keys)):
        raise GateZeroSuccessorProbeError(
            "Gate-0 successor probe source repeats an immutable packed entry"
        )

    ordered = tuple(
        record
        for _address, record in sorted(
            zip(addresses, materialized, strict=True),
            key=lambda item: _address_sort_key(item[0]),
        )
    )
    expected_cell_keys: set[SuccessorSemanticCellKey] = set()
    scanned_paths: list[_ScannedPath] = []
    runtime = de_novo_rewrite_system()
    for record in ordered:
        address = _require_address(record)
        path = record.path
        if path.path_length != address.path_length or len(path.trace.steps) != address.path_length:
            raise GateZeroSuccessorProbeError("packed address and trace path lengths disagree")
        if record.target_key != address.target_key:
            raise GateZeroSuccessorProbeError(
                "PathRecord target key disagrees with its immutable address"
            )
        states = tuple(
            path.state_at(progress_index) for progress_index in range(address.path_length + 1)
        )
        state_digests = tuple(persistent_slot_state_sha256(state) for state in states)
        if (
            persistent_slot_state_sha256(path.trace.source) != state_digests[0]
            or persistent_slot_state_sha256(path.trace.target) != state_digests[-1]
        ):
            raise GateZeroSuccessorProbeError(
                "trace endpoints disagree with stored progress states"
            )
        try:
            source_key = canonical_state_key(states[0])
            target_key = canonical_state_key(states[-1])
        except Exception as error:
            raise GateZeroSuccessorProbeError(
                "trace endpoint is not a canonical production state"
            ) from error
        if source_key != address.source_key or target_key != address.target_key:
            raise GateZeroSuccessorProbeError(
                "trace endpoint identity disagrees with its immutable address"
            )

        scanned_rows: list[_ScannedRow] = []
        for progress_index, state in enumerate(states):
            key = (
                address.packed_shard_content_sha256,
                address.entry_index,
                progress_index,
            )
            expected_cell_keys.add(key)
            cache_address = SuccessorFiberCacheAddress.from_packed_trace(
                address,
                progress_index=progress_index,
            )
            if progress_index == address.path_length:
                family = None
                action_sha256 = None
                target_state_sha256 = None
            else:
                step = path.trace.steps[progress_index]
                try:
                    family = canonical_family(step.rule_name)
                    action_sha256 = rewrite_action_codec_sha256(
                        step.rule_name,
                        step.action,
                    )
                except Exception as error:
                    raise GateZeroSuccessorProbeError(
                        "a source teacher has no canonical production action identity"
                    ) from error
                try:
                    replayed = runtime.apply(
                        state,
                        step.rule_name,
                        step.action,
                    )
                except Exception as error:
                    raise GateZeroSuccessorProbeError(
                        "a source teacher is not an executable production rewrite"
                    ) from error
                target_state_sha256 = state_digests[progress_index + 1]
                if persistent_slot_state_sha256(replayed) != target_state_sha256:
                    raise GateZeroSuccessorProbeError(
                        "source teacher does not execute to its stored next state"
                    )

            if key not in semantic_cell_ids:
                raise GateZeroSuccessorProbeError(
                    f"semantic-cell sidecar is missing exact row {key!r}"
                )
            semantic_cell = semantic_cell_ids[key]
            if progress_index == address.path_length:
                if semantic_cell is not None:
                    raise GateZeroSuccessorProbeError(
                        "terminal semantic-cell assignment must be null"
                    )
            elif not isinstance(semantic_cell, str) or not semantic_cell:
                raise GateZeroSuccessorProbeError(
                    "nonterminal semantic-cell assignment must be nonempty text"
                )
            scanned_rows.append(
                _ScannedRow(
                    address=cache_address,
                    source_state_sha256=state_digests[progress_index],
                    target_state_sha256=target_state_sha256,
                    family_name=family,
                    semantic_cell_id=semantic_cell,
                    teacher_action_sha256=action_sha256,
                )
            )
        scanned_paths.append(
            _ScannedPath(
                record=record,
                address=address,
                rows=tuple(scanned_rows),
            )
        )

    observed_cell_keys = set(semantic_cell_ids)
    if observed_cell_keys != expected_cell_keys:
        missing = sorted(expected_cell_keys - observed_cell_keys)
        unexpected = sorted(observed_cell_keys - expected_cell_keys)
        raise GateZeroSuccessorProbeError(
            "semantic-cell sidecar must be an exact census of the supplied "
            f"validation rows; missing={missing[:20]!r}, "
            f"unexpected={unexpected[:20]!r}"
        )
    source_payload = [
        {
            "address": asdict(scanned.address),
            "rows": [
                {
                    "address": asdict(row.address),
                    "source_state_sha256": row.source_state_sha256,
                    "target_state_sha256": row.target_state_sha256,
                    "family_name": row.family_name,
                    "teacher_action_sha256": row.teacher_action_sha256,
                }
                for row in scanned.rows
            ],
        }
        for scanned in scanned_paths
    ]
    sidecar_payload = [
        {
            "packed_shard_content_sha256": key[0],
            "entry_index": key[1],
            "progress_index": key[2],
            "semantic_cell_id": semantic_cell_ids[key],
        }
        for key in sorted(semantic_cell_ids)
    ]
    return (
        tuple(scanned_paths),
        _stable_sha256(source_payload),
        _stable_sha256(sidecar_payload),
    )


def _select_complete_paths(
    scanned_paths: tuple[_ScannedPath, ...],
    config: GateZeroSuccessorProbeConfig,
) -> tuple[_ScannedPath, ...]:
    uncovered = {
        *(("family", family) for family in config.required_families),
        *(("semantic_cell", cell) for cell in config.required_semantic_cells),
    }
    selected: list[_ScannedPath] = []
    remaining = list(scanned_paths)
    while uncovered:
        ranked = sorted(
            (
                (
                    len(path.obligations & uncovered),
                    _address_sort_key(path.address),
                    path,
                )
                for path in remaining
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not ranked or ranked[0][0] == 0:
            missing_families = tuple(value for kind, value in sorted(uncovered) if kind == "family")
            missing_cells = tuple(
                value for kind, value in sorted(uncovered) if kind == "semantic_cell"
            )
            raise GateZeroSuccessorProbeError(
                "validation source cannot cover the explicit Gate-0 "
                f"obligations; missing_families={missing_families!r}, "
                f"missing_semantic_cells={missing_cells!r}"
            )
        chosen = ranked[0][2]
        selected.append(chosen)
        uncovered -= chosen.obligations
        remaining.remove(chosen)
    return tuple(sorted(selected, key=lambda path: _address_sort_key(path.address)))


def _validate_provenance(
    model: FactorizedTraceletRateModel,
    selected_paths: tuple[_ScannedPath, ...],
    provenance_by_shard: Mapping[str, SuccessorFiberCacheProvenance],
) -> None:
    if not isinstance(provenance_by_shard, Mapping):
        raise TypeError("provenance_by_shard must be a mapping")
    required_shards = {path.address.packed_shard_content_sha256 for path in selected_paths}
    if set(provenance_by_shard) != required_shards:
        raise GateZeroSuccessorProbeError(
            "cache provenance keys must exactly match selected packed shards"
        )
    expected_compiler = fiber_compiler_implementation_hash()
    expected_capability = model.operator_capabilities.fingerprint()
    for shard_sha256, provenance in provenance_by_shard.items():
        if not isinstance(provenance, SuccessorFiberCacheProvenance):
            raise TypeError("each cache provenance must use SuccessorFiberCacheProvenance")
        if provenance.packed_shard_content_sha256 != shard_sha256:
            raise GateZeroSuccessorProbeError("cache provenance is bound to another packed shard")
        if provenance.fiber_compiler_implementation_hash != expected_compiler:
            raise GateZeroSuccessorProbeError(
                "cache provenance does not bind the current production successor compiler"
            )
        if provenance.capability_hash != expected_capability:
            raise GateZeroSuccessorProbeError(
                "cache provenance capability hash disagrees with the model"
            )


def _validate_compiled_trace(
    scanned_path: _ScannedPath,
    compiled: tuple[SuccessorFiberCacheRecord, ...],
) -> None:
    expected_addresses = tuple(row.address for row in scanned_path.rows)
    observed_addresses = tuple(record.address for record in compiled)
    if observed_addresses != expected_addresses:
        raise GateZeroSuccessorProbeError(
            "production successor compiler did not retain every exact "
            "progress address of the selected trace"
        )
    for scanned_row, cache_record in zip(
        scanned_path.rows,
        compiled,
        strict=True,
    ):
        if cache_record.source_state_sha256 != scanned_row.source_state_sha256:
            raise GateZeroSuccessorProbeError(
                "production successor compiler returned support for another exact source state"
            )
        if cache_record.target_state_sha256 != scanned_row.target_state_sha256:
            raise GateZeroSuccessorProbeError(
                "production successor compiler returned the wrong exact teacher successor"
            )


def build_gate_zero_successor_probe(
    model: FactorizedTraceletRateModel,
    records: Sequence[PathRecord],
    *,
    semantic_cell_ids: Mapping[
        SuccessorSemanticCellKey,
        str | None,
    ],
    config: GateZeroSuccessorProbeConfig,
    provenance_by_shard: Mapping[str, SuccessorFiberCacheProvenance],
    limits: SuccessorFiberCacheLimits = (DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS),
) -> GateZeroSuccessorProbe:
    """Build one deterministic, complete-trace Gate-0 validation probe.

    This function never writes an artifact, chooses a quality threshold, or
    returns a training decision.  Callers must separately freeze the returned
    manifest and ordinary encoded cache shards if they want durable evidence.
    """

    if not isinstance(model, FactorizedTraceletRateModel):
        raise TypeError("Gate-0 successor probe requires FactorizedTraceletRateModel")
    if not isinstance(config, GateZeroSuccessorProbeConfig):
        raise TypeError("config must be GateZeroSuccessorProbeConfig")
    scanned_paths, source_sha256, sidecar_sha256 = _scan_source_records(
        records,
        semantic_cell_ids,
    )
    selected = _select_complete_paths(scanned_paths, config)
    _validate_provenance(model, selected, provenance_by_shard)

    compiled_by_shard: dict[str, list[SuccessorFiberCacheRecord]] = {}
    annotations: list[GateZeroSuccessorProbeRow] = []
    for path in selected:
        try:
            compiled = compile_successor_fiber_trace(
                model,
                path.record,
                time=GATE_ZERO_SUCCESSOR_SUPPORT_TIME,
            )
        except SuccessorFiberCacheBuildError as error:
            raise GateZeroSuccessorProbeError(
                "production successor-fiber compilation failed for a selected Gate-0 trace"
            ) from error
        _validate_compiled_trace(path, compiled)
        shard_sha256 = path.address.packed_shard_content_sha256
        compiled_by_shard.setdefault(shard_sha256, []).extend(compiled)
        annotations.extend(
            GateZeroSuccessorProbeRow(
                address=row.address,
                family_name=row.family_name,
                semantic_cell_id=row.semantic_cell_id,
                teacher_action_sha256=row.teacher_action_sha256,
            )
            for row in path.rows
        )

    cache_shards: list[GateZeroSuccessorProbeCacheShard] = []
    for shard_sha256 in sorted(compiled_by_shard):
        try:
            encoded, cache = serialize_successor_fiber_cache(
                compiled_by_shard[shard_sha256],
                provenance=provenance_by_shard[shard_sha256],
                limits=limits,
            )
        except Exception as error:
            raise GateZeroSuccessorProbeError(
                "selected support failed the production successor-cache schema"
            ) from error
        cache_shards.append(
            GateZeroSuccessorProbeCacheShard(
                encoded=encoded,
                cache=cache,
            )
        )

    rows = tuple(sorted(annotations, key=lambda row: row.address))
    family_witnesses = {
        family: tuple(row.address for row in rows if row.family_name == family)
        for family in config.required_families
    }
    semantic_cell_witnesses = {
        cell: tuple(row.address for row in rows if row.semantic_cell_id == cell)
        for cell in config.required_semantic_cells
    }
    model_sha256 = state_dict_semantic_sha256(model.state_dict())
    selected_records = tuple(path.record for path in selected)
    frozen_cache_shards = tuple(cache_shards)
    probe_sha256 = _stable_sha256(
        _gate_zero_probe_payload(
            config_sha256=config.sha256,
            source_sha256=source_sha256,
            semantic_cell_sidecar_sha256=sidecar_sha256,
            compiler_model_state_sha256=model_sha256,
            selected_path_records=selected_records,
            rows=rows,
            cache_shards=frozen_cache_shards,
            family_witnesses=family_witnesses,
            semantic_cell_witnesses=semantic_cell_witnesses,
        )
    )
    return GateZeroSuccessorProbe(
        config=config,
        config_sha256=config.sha256,
        source_sha256=source_sha256,
        semantic_cell_sidecar_sha256=sidecar_sha256,
        compiler_model_state_sha256=model_sha256,
        selected_path_records=selected_records,
        rows=rows,
        cache_shards=frozen_cache_shards,
        family_witnesses=family_witnesses,
        semantic_cell_witnesses=semantic_cell_witnesses,
        probe_sha256=probe_sha256,
    )


__all__ = [
    "GATE_ZERO_SUCCESSOR_PROBE_PARTITION",
    "GATE_ZERO_SUCCESSOR_PROBE_SCHEMA",
    "GATE_ZERO_SUCCESSOR_PROBE_SCHEMA_VERSION",
    "GATE_ZERO_SUCCESSOR_PROBE_STATUS",
    "GATE_ZERO_SUCCESSOR_SELECTION_ALGORITHM",
    "GATE_ZERO_SUCCESSOR_SUPPORT_TIME",
    "GateZeroSuccessorProbe",
    "GateZeroSuccessorProbeCacheShard",
    "GateZeroSuccessorProbeConfig",
    "GateZeroSuccessorProbeError",
    "GateZeroSuccessorProbeRow",
    "build_gate_zero_successor_probe",
]
