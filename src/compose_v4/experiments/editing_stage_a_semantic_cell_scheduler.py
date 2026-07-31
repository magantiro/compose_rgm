"""Deterministic equal-cell scheduling for Stage-A capability acquisition.

This module only chooses frozen nonterminal example addresses.  It does not
derive probabilities, importance corrections, scientific weights, or loss
coefficients.  The caller remains responsible for constructing the declared
semantic-cell mapping from authoritative frozen metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from compose_v4.data.successor_fiber_cache import SuccessorFiberCacheAddress

STAGE_A_SEMANTIC_CELL_SCHEDULER_SCHEMA = "compose.editing.stage_a_semantic_cell_scheduler"
STAGE_A_SEMANTIC_CELL_CURSOR_SCHEMA = "compose.editing.stage_a_semantic_cell_scheduler_cursor"
STAGE_A_SEMANTIC_CELL_SCHEDULER_VERSION = 1

_ALGORITHM = "sorted_cell_round_robin__seeded_per_cell_cycle_hash_shuffle_v1"
_HEX64 = frozenset("0123456789abcdef")
_CURSOR_FIELDS = {
    "schema",
    "schema_version",
    "scheduler_sha256",
    "next_stream_index",
    "cursor_sha256",
}


class StageASemanticCellSchedulerError(ValueError):
    """The declared cells, frozen addresses, or resume cursor are invalid."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX64 for character in value)
    )


@dataclass(frozen=True)
class StageASemanticCellCursor:
    """Content-verifiable position in one immutable scheduler stream."""

    scheduler_sha256: str
    next_stream_index: int

    def __post_init__(self) -> None:
        if not _is_sha256(self.scheduler_sha256):
            raise StageASemanticCellSchedulerError(
                "cursor scheduler_sha256 must be a lowercase SHA-256"
            )
        if type(self.next_stream_index) is not int or self.next_stream_index < 0:
            raise StageASemanticCellSchedulerError(
                "cursor next_stream_index must be a nonnegative integer"
            )

    def _body(self) -> dict[str, object]:
        return {
            "schema": STAGE_A_SEMANTIC_CELL_CURSOR_SCHEMA,
            "schema_version": STAGE_A_SEMANTIC_CELL_SCHEDULER_VERSION,
            "scheduler_sha256": self.scheduler_sha256,
            "next_stream_index": self.next_stream_index,
        }

    @property
    def cursor_sha256(self) -> str:
        return _stable_sha256(self._body())

    def to_payload(self) -> dict[str, object]:
        body = self._body()
        return {**body, "cursor_sha256": _stable_sha256(body)}

    def to_json_bytes(self) -> bytes:
        """Return deterministic, self-hashed cursor bytes."""

        return _canonical_json_bytes(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> StageASemanticCellCursor:
        if not isinstance(payload, Mapping) or set(payload) != _CURSOR_FIELDS:
            raise StageASemanticCellSchedulerError(
                "serialized Stage-A cursor has unexpected fields"
            )
        body = {key: payload[key] for key in _CURSOR_FIELDS if key != "cursor_sha256"}
        if (
            payload.get("schema") != STAGE_A_SEMANTIC_CELL_CURSOR_SCHEMA
            or payload.get("schema_version") != STAGE_A_SEMANTIC_CELL_SCHEDULER_VERSION
            or payload.get("cursor_sha256") != _stable_sha256(body)
        ):
            raise StageASemanticCellSchedulerError("serialized Stage-A cursor identity is invalid")
        scheduler_sha256 = payload["scheduler_sha256"]
        next_stream_index = payload["next_stream_index"]
        if not isinstance(scheduler_sha256, str) or type(next_stream_index) is not int:
            raise StageASemanticCellSchedulerError(
                "serialized Stage-A cursor values have invalid types"
            )
        return cls(
            scheduler_sha256=scheduler_sha256,
            next_stream_index=next_stream_index,
        )

    @classmethod
    def from_json_bytes(cls, encoded: bytes) -> StageASemanticCellCursor:
        if not isinstance(encoded, bytes):
            raise TypeError("serialized Stage-A cursor must be bytes")
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StageASemanticCellSchedulerError(
                "serialized Stage-A cursor is not valid JSON"
            ) from error
        return cls.from_payload(payload)


@dataclass(frozen=True)
class StageAScheduledExample:
    """One unweighted scheduler decision."""

    stream_index: int
    semantic_cell_id: str
    address: SuccessorFiberCacheAddress


@dataclass(frozen=True)
class StageAScheduledBatch:
    """One ordered batch with exact, unweighted exposure metadata."""

    examples: tuple[StageAScheduledExample, ...]

    @property
    def addresses(self) -> tuple[SuccessorFiberCacheAddress, ...]:
        return tuple(example.address for example in self.examples)

    @property
    def semantic_cell_multiplicities(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(example.semantic_cell_id for example in self.examples)
        return tuple(sorted(counts.items()))


class StageASemanticCellScheduler:
    """Infinite deterministic stream with equal semantic-cell exposure."""

    def __init__(
        self,
        *,
        declared_cells: Sequence[str],
        examples_by_cell: Mapping[str, Sequence[SuccessorFiberCacheAddress]],
        seed: int,
    ) -> None:
        if isinstance(declared_cells, (str, bytes)) or not isinstance(declared_cells, Sequence):
            raise TypeError("declared_cells must be a sequence of semantic-cell IDs")
        if not isinstance(examples_by_cell, Mapping):
            raise TypeError("examples_by_cell must be a semantic-cell mapping")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise StageASemanticCellSchedulerError(
                "Stage-A scheduler seed must be an unsigned 64-bit integer"
            )

        declared = tuple(declared_cells)
        if not declared:
            raise StageASemanticCellSchedulerError(
                "Stage-A scheduler requires at least one declared semantic cell"
            )
        invalid_declared = tuple(
            repr(cell) for cell in declared if not isinstance(cell, str) or not cell
        )
        if invalid_declared:
            raise StageASemanticCellSchedulerError(
                f"declared semantic-cell IDs must be nonempty strings: {invalid_declared}"
            )
        if len(set(declared)) != len(declared):
            raise StageASemanticCellSchedulerError("declared semantic-cell IDs must be unique")
        invalid_keys = tuple(
            repr(cell) for cell in examples_by_cell if not isinstance(cell, str) or not cell
        )
        if invalid_keys:
            raise StageASemanticCellSchedulerError(
                f"example semantic-cell IDs must be nonempty strings: {invalid_keys}"
            )

        declared_set = set(declared)
        observed_set = set(examples_by_cell)
        missing = tuple(sorted(declared_set - observed_set))
        unknown = tuple(sorted(observed_set - declared_set))
        if missing:
            raise StageASemanticCellSchedulerError(
                f"declared semantic cells lack examples: {missing}"
            )
        if unknown:
            raise StageASemanticCellSchedulerError(
                f"examples contain unknown semantic cells: {unknown}"
            )

        normalized: dict[str, tuple[SuccessorFiberCacheAddress, ...]] = {}
        all_addresses: set[SuccessorFiberCacheAddress] = set()
        for cell_id in sorted(declared):
            raw_addresses = examples_by_cell[cell_id]
            if isinstance(raw_addresses, (str, bytes)) or not isinstance(raw_addresses, Sequence):
                raise TypeError(f"examples for semantic cell {cell_id!r} must be a sequence")
            addresses = tuple(raw_addresses)
            if not addresses:
                raise StageASemanticCellSchedulerError(f"semantic cell {cell_id!r} has no examples")
            if any(not isinstance(address, SuccessorFiberCacheAddress) for address in addresses):
                raise TypeError(f"semantic cell {cell_id!r} contains a non-frozen address")
            terminal = tuple(address for address in addresses if address.is_terminal)
            if terminal:
                raise StageASemanticCellSchedulerError(
                    f"semantic cell {cell_id!r} contains terminal addresses"
                )
            if len(set(addresses)) != len(addresses):
                raise StageASemanticCellSchedulerError(
                    f"semantic cell {cell_id!r} contains duplicate addresses"
                )
            overlap = tuple(sorted(set(addresses) & all_addresses))
            if overlap:
                raise StageASemanticCellSchedulerError(
                    "one frozen example address is assigned to multiple semantic cells"
                )
            ordered = tuple(sorted(addresses))
            normalized[cell_id] = ordered
            all_addresses.update(ordered)

        inventory = {
            cell_id: [asdict(address) for address in normalized[cell_id]]
            for cell_id in sorted(normalized)
        }
        inventory_sha256 = _stable_sha256(inventory)
        identity_body: dict[str, object] = {
            "schema": STAGE_A_SEMANTIC_CELL_SCHEDULER_SCHEMA,
            "schema_version": STAGE_A_SEMANTIC_CELL_SCHEDULER_VERSION,
            "algorithm": _ALGORITHM,
            "seed": seed,
            "declared_cells": list(sorted(declared)),
            "example_counts_by_cell": {
                cell_id: len(normalized[cell_id]) for cell_id in sorted(normalized)
            },
            "address_inventory_sha256": inventory_sha256,
        }
        self._declared_cells = tuple(sorted(declared))
        self._examples_by_cell = normalized
        self._seed = seed
        self._identity_body = identity_body
        self._scheduler_sha256 = _stable_sha256(identity_body)
        self._last_permutation_by_cell: dict[
            str,
            tuple[int, tuple[SuccessorFiberCacheAddress, ...]],
        ] = {}

    @property
    def declared_cells(self) -> tuple[str, ...]:
        return self._declared_cells

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def scheduler_sha256(self) -> str:
        return self._scheduler_sha256

    def identity_payload(self) -> dict[str, object]:
        return json.loads(
            _canonical_json_bytes(
                {
                    **self._identity_body,
                    "scheduler_sha256": self.scheduler_sha256,
                }
            )
        )

    def initial_cursor(self) -> StageASemanticCellCursor:
        return StageASemanticCellCursor(
            scheduler_sha256=self.scheduler_sha256,
            next_stream_index=0,
        )

    def _cycle_permutation(
        self,
        cell_id: str,
        cycle_index: int,
    ) -> tuple[SuccessorFiberCacheAddress, ...]:
        def shuffle_key(
            address: SuccessorFiberCacheAddress,
        ) -> tuple[str, SuccessorFiberCacheAddress]:
            digest = _stable_sha256(
                {
                    "scheduler_sha256": self.scheduler_sha256,
                    "semantic_cell_id": cell_id,
                    "cycle_index": cycle_index,
                    "address": asdict(address),
                }
            )
            return digest, address

        return tuple(sorted(self._examples_by_cell[cell_id], key=shuffle_key))

    def _cached_cycle_permutation(
        self,
        cell_id: str,
        cycle_index: int,
    ) -> tuple[SuccessorFiberCacheAddress, ...]:
        cached = self._last_permutation_by_cell.get(cell_id)
        if cached is not None and cached[0] == cycle_index:
            return cached[1]
        permutation = self._cycle_permutation(cell_id, cycle_index)
        self._last_permutation_by_cell[cell_id] = (cycle_index, permutation)
        return permutation

    def take(
        self,
        cursor: StageASemanticCellCursor,
        count: int,
    ) -> tuple[StageAScheduledBatch, StageASemanticCellCursor]:
        """Take a balanced contiguous stream segment and return its next cursor."""

        if not isinstance(cursor, StageASemanticCellCursor):
            raise TypeError("Stage-A scheduler requires StageASemanticCellCursor")
        if cursor.scheduler_sha256 != self.scheduler_sha256:
            raise StageASemanticCellSchedulerError(
                "Stage-A cursor belongs to another scheduler identity"
            )
        if type(count) is not int or count < 0:
            raise StageASemanticCellSchedulerError(
                "Stage-A scheduler count must be a nonnegative integer"
            )

        scheduled: list[StageAScheduledExample] = []
        cell_count = len(self.declared_cells)
        for stream_index in range(
            cursor.next_stream_index,
            cursor.next_stream_index + count,
        ):
            cell_id = self.declared_cells[stream_index % cell_count]
            cell_occurrence = stream_index // cell_count
            addresses = self._examples_by_cell[cell_id]
            cycle_index, offset = divmod(cell_occurrence, len(addresses))
            permutation = self._cached_cycle_permutation(cell_id, cycle_index)
            scheduled.append(
                StageAScheduledExample(
                    stream_index=stream_index,
                    semantic_cell_id=cell_id,
                    address=permutation[offset],
                )
            )
        next_cursor = StageASemanticCellCursor(
            scheduler_sha256=self.scheduler_sha256,
            next_stream_index=cursor.next_stream_index + count,
        )
        return StageAScheduledBatch(examples=tuple(scheduled)), next_cursor


__all__ = [
    "STAGE_A_SEMANTIC_CELL_CURSOR_SCHEMA",
    "STAGE_A_SEMANTIC_CELL_SCHEDULER_SCHEMA",
    "STAGE_A_SEMANTIC_CELL_SCHEDULER_VERSION",
    "StageAScheduledBatch",
    "StageAScheduledExample",
    "StageASemanticCellCursor",
    "StageASemanticCellScheduler",
    "StageASemanticCellSchedulerError",
]
