"""Bounded offline compilation of exact trace-progress successor fibers.

This module is intentionally a correctness builder for development panels. The
current production enumerator obtains legal masks through a scored model law;
therefore a full corpus build remains blocked until a support-only enumerator
or a stratified exhaustive invariance gate is frozen.
"""

from __future__ import annotations

from collections.abc import Iterable

from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_successor_training import (
    compile_state_productive_support,
    compile_teacher_successor_fiber,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import RewriteSystem


class SuccessorFiberCacheBuildError(RuntimeError):
    """A development trace cannot produce one exact complete cache chain."""


def compile_successor_fiber_trace(
    model: FactorizedTraceletRateModel,
    record: PathRecord,
    *,
    time: float = 0.5,
    system: RewriteSystem | None = None,
) -> tuple[SuccessorFiberCacheRecord, ...]:
    """Compile every progress state of one immutable addressed trace."""

    address = record.corpus_address
    if address is None:
        raise SuccessorFiberCacheBuildError(
            "successor-fiber compilation requires an immutable packed address"
        )
    path = record.path
    if path.path_length != address.path_length:
        raise SuccessorFiberCacheBuildError(
            "packed address and trace path lengths disagree"
        )
    if not 0.0 < float(time) < 1.0:
        raise ValueError("support compilation time must lie strictly in (0, 1)")

    rows: list[SuccessorFiberCacheRecord] = []
    for progress_index in range(path.path_length + 1):
        source = path.state_at(progress_index)
        cache_address = SuccessorFiberCacheAddress.from_packed_trace(
            address,
            progress_index=progress_index,
        )
        if progress_index == path.path_length:
            support = compile_state_productive_support(
                model,
                source,
                time=time,
                system=system,
            )
            fiber = None
        else:
            target = path.state_at(progress_index + 1)
            fiber = compile_teacher_successor_fiber(
                model,
                source,
                target,
                time=time,
                system=system,
            )
            support = fiber.state_support
        rows.append(
            SuccessorFiberCacheRecord(
                address=cache_address,
                state_support=support,
                teacher_fiber=fiber,
            )
        )
    return tuple(rows)


def compile_successor_fiber_shard(
    model: FactorizedTraceletRateModel,
    records: Iterable[PathRecord],
    *,
    expected_packed_entry_count: int,
    excluded_entry_indices: tuple[int, ...] = (),
    time: float = 0.5,
    system: RewriteSystem | None = None,
) -> tuple[SuccessorFiberCacheRecord, ...]:
    """Compile a complete bounded packed-shard derivative, never a subset."""

    materialized = tuple(records)
    if not materialized:
        raise SuccessorFiberCacheBuildError(
            "successor-fiber shard cannot be empty"
        )
    if (
        type(expected_packed_entry_count) is not int
        or expected_packed_entry_count <= 0
    ):
        raise ValueError("expected_packed_entry_count must be positive")
    if (
        not isinstance(excluded_entry_indices, tuple)
        or excluded_entry_indices
        != tuple(sorted(set(excluded_entry_indices)))
        or any(
            type(index) is not int
            or not 0 <= index < expected_packed_entry_count
            for index in excluded_entry_indices
        )
    ):
        raise ValueError(
            "excluded_entry_indices must be sorted unique in-range integers"
        )
    addresses = tuple(record.corpus_address for record in materialized)
    if any(address is None for address in addresses):
        raise SuccessorFiberCacheBuildError(
            "successor-fiber shard contains an unaddressed record"
        )
    resolved = tuple(address for address in addresses if address is not None)
    envelopes = {
        (
            address.packed_shard_content_sha256,
            address.packed_shard_name,
            address.layer,
            address.partition,
        )
        for address in resolved
    }
    if len(envelopes) != 1:
        raise SuccessorFiberCacheBuildError(
            "one successor cache artifact must derive from exactly one packed shard"
        )
    entry_indices = tuple(address.entry_index for address in resolved)
    if len(entry_indices) != len(set(entry_indices)):
        raise SuccessorFiberCacheBuildError(
            "successor-fiber shard repeats a packed entry index"
        )
    expected_active_indices = set(range(expected_packed_entry_count)) - set(
        excluded_entry_indices
    )
    if set(entry_indices) != expected_active_indices:
        missing = sorted(expected_active_indices - set(entry_indices))
        unexpected = sorted(set(entry_indices) - expected_active_indices)
        raise SuccessorFiberCacheBuildError(
            "successor-fiber shard input is not the complete declared packed "
            f"census; missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    rows: list[SuccessorFiberCacheRecord] = []
    for record in materialized:
        rows.extend(
            compile_successor_fiber_trace(
                model,
                record,
                time=time,
                system=system,
            )
        )
    return tuple(rows)


__all__ = [
    "SuccessorFiberCacheBuildError",
    "compile_successor_fiber_shard",
    "compile_successor_fiber_trace",
]
