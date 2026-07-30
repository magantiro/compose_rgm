"""Bounded offline compilation of exact trace-progress successor fibers.

This module is intentionally a correctness builder for development panels. The
current production enumerator obtains legal masks through a scored model law;
therefore a full corpus build remains blocked until a support-only enumerator
or a stratified exhaustive invariance gate is frozen.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheRecord,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_successor_training import (
    SuccessorTrainingError,
    compile_state_successor_map,
    require_exact_successor_action_identity,
    rewrite_action_codec_sha256,
    teacher_successor_fiber_from_exact_digest,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import RewriteSystem, de_novo_rewrite_system
from compose_v4.rewrite.trace import RewriteStep


class SuccessorFiberCacheBuildError(RuntimeError):
    """A development trace cannot produce one exact complete cache chain."""


@dataclass(frozen=True)
class _ProgressOccurrence:
    """One output row request grouped by exact source-state identity."""

    output_index: int
    source: MolecularGraph
    source_state_sha256: str
    target_state_sha256: str | None
    teacher_step: RewriteStep | None
    cache_address: SuccessorFiberCacheAddress
    packed_source_key: str | None
    packed_target_key: str | None


def _record_occurrences(
    record: PathRecord,
    *,
    output_offset: int,
) -> tuple[_ProgressOccurrence, ...]:
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
    if len(path.trace.steps) != path.path_length:
        raise SuccessorFiberCacheBuildError(
            "stored trace steps and packed progress path lengths disagree"
        )
    if record.target_key != address.target_key:
        raise SuccessorFiberCacheBuildError(
            "path record and packed address target keys disagree"
        )

    states = tuple(
        path.state_at(progress_index)
        for progress_index in range(path.path_length + 1)
    )
    digests = tuple(persistent_slot_state_sha256(state) for state in states)
    if persistent_slot_state_sha256(path.trace.source) != digests[0]:
        raise SuccessorFiberCacheBuildError(
            "stored trace source and packed progress source disagree"
        )
    if persistent_slot_state_sha256(path.trace.target) != digests[-1]:
        raise SuccessorFiberCacheBuildError(
            "stored trace target and packed progress endpoint disagree"
        )

    return tuple(
        _ProgressOccurrence(
            output_index=output_offset + progress_index,
            source=source,
            source_state_sha256=digests[progress_index],
            target_state_sha256=(
                None
                if progress_index == path.path_length
                else digests[progress_index + 1]
            ),
            teacher_step=(
                None
                if progress_index == path.path_length
                else path.trace.steps[progress_index]
            ),
            cache_address=SuccessorFiberCacheAddress.from_packed_trace(
                address,
                progress_index=progress_index,
            ),
            packed_source_key=(
                address.source_key if progress_index == 0 else None
            ),
            packed_target_key=(
                address.target_key
                if progress_index == path.path_length
                else None
            ),
        )
        for progress_index, source in enumerate(states)
    )


def _compile_occurrences(
    model: FactorizedTraceletRateModel,
    occurrences: tuple[_ProgressOccurrence, ...],
    *,
    time: float,
    system: RewriteSystem | None,
) -> tuple[SuccessorFiberCacheRecord, ...]:
    """Compile each exact source once and discard its complete map after use."""

    runtime = system or de_novo_rewrite_system()
    grouped: dict[str, list[_ProgressOccurrence]] = {}
    for occurrence in occurrences:
        grouped.setdefault(occurrence.source_state_sha256, []).append(
            occurrence
        )

    rows: list[SuccessorFiberCacheRecord | None] = [None] * len(occurrences)
    for source_digest, state_occurrences in grouped.items():
        representative = state_occurrences[0]
        try:
            compiled = compile_state_successor_map(
                model,
                representative.source,
                time=time,
                system=runtime,
            )
        except SuccessorTrainingError as error:
            raise SuccessorFiberCacheBuildError(
                "exact source-state successor compilation failed"
            ) from error
        if compiled.source_state_sha256 != source_digest:
            raise SuccessorFiberCacheBuildError(
                "compiled support does not match its exact source-state group"
            )

        for occurrence in state_occurrences:
            if (
                occurrence.packed_source_key is not None
                and compiled.source_key != occurrence.packed_source_key
            ):
                raise SuccessorFiberCacheBuildError(
                    "packed address source key disagrees with its trace source"
                )
            if (
                occurrence.packed_target_key is not None
                and compiled.source_key != occurrence.packed_target_key
            ):
                raise SuccessorFiberCacheBuildError(
                    "packed address target key disagrees with its trace endpoint"
                )
            if occurrence.target_state_sha256 is None:
                if occurrence.teacher_step is not None:
                    raise SuccessorFiberCacheBuildError(
                        "terminal occurrence unexpectedly carries a teacher step"
                    )
                support = compiled.state_support
                fiber = None
            else:
                if occurrence.teacher_step is None:
                    raise SuccessorFiberCacheBuildError(
                        "nonterminal occurrence is missing its teacher step"
                    )
                try:
                    replayed_target = runtime.apply(
                        occurrence.source,
                        occurrence.teacher_step.rule_name,
                        occurrence.teacher_step.action,
                    )
                except Exception as error:
                    raise SuccessorFiberCacheBuildError(
                        "stored teacher step failed under the production executor"
                    ) from error
                if (
                    persistent_slot_state_sha256(replayed_target)
                    != occurrence.target_state_sha256
                ):
                    raise SuccessorFiberCacheBuildError(
                        "stored teacher step does not execute to the stored exact "
                        "next state"
                    )
                try:
                    fiber = teacher_successor_fiber_from_exact_digest(
                        compiled,
                        occurrence.target_state_sha256,
                    )
                except SuccessorTrainingError as error:
                    address = occurrence.cache_address
                    raise SuccessorFiberCacheBuildError(
                        "stored exact trace target is absent from source support: "
                        f"shard={address.packed_shard_content_sha256}, "
                        f"entry_index={address.entry_index}, "
                        f"progress_index={address.progress_index}, "
                        f"trace_id={address.trace_id!r}, "
                        f"teacher_rule={occurrence.teacher_step.rule_name!r}, "
                        f"source_state_sha256={occurrence.source_state_sha256}, "
                        "target_state_sha256="
                        f"{occurrence.target_state_sha256}"
                    ) from error
                try:
                    teacher_action_sha256 = rewrite_action_codec_sha256(
                        occurrence.teacher_step.rule_name,
                        occurrence.teacher_step.action,
                    )
                except SuccessorTrainingError as error:
                    raise SuccessorFiberCacheBuildError(
                        "stored teacher action has no canonical production identity"
                    ) from error
                try:
                    require_exact_successor_action_identity(
                        compiled,
                        target_state_sha256=occurrence.target_state_sha256,
                        action_sha256=teacher_action_sha256,
                    )
                except SuccessorTrainingError as error:
                    raise SuccessorFiberCacheBuildError(
                        "stored teacher action is absent from the exact target "
                        "marked support"
                    ) from error
                support = fiber.state_support
            rows[occurrence.output_index] = SuccessorFiberCacheRecord(
                address=occurrence.cache_address,
                state_support=support,
                teacher_fiber=fiber,
            )

        # Do not retain the potentially large all-successor grouping after all
        # occurrences of this exact state have been projected to compact rows.
        del compiled

    if any(row is None for row in rows):
        raise SuccessorFiberCacheBuildError(
            "state-centric compiler did not produce every requested cache row"
        )
    return tuple(row for row in rows if row is not None)


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

    return _compile_occurrences(
        model,
        _record_occurrences(record, output_offset=0),
        time=float(time),
        system=system,
    )


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
    if not 0.0 < float(time) < 1.0:
        raise ValueError("support compilation time must lie strictly in (0, 1)")

    occurrences: list[_ProgressOccurrence] = []
    output_offset = 0
    for record in materialized:
        record_occurrences = _record_occurrences(
            record,
            output_offset=output_offset,
        )
        occurrences.extend(record_occurrences)
        output_offset += len(record_occurrences)
    return _compile_occurrences(
        model,
        tuple(occurrences),
        time=float(time),
        system=system,
    )


__all__ = [
    "SuccessorFiberCacheBuildError",
    "compile_successor_fiber_shard",
    "compile_successor_fiber_trace",
]
