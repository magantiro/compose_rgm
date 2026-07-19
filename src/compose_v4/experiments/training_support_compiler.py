"""Parallel deterministic compilation of sparse training-support shards."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_mark_conditional import FactorizedMarkDataset
from compose_v4.experiments.training_support_cache import (
    ShardedTrainingSupportCache,
    TrainingSupportRow,
    TrainingSupportShard,
    load_training_support_shard,
    save_training_support_shard,
    training_support_shard_path,
)
from compose_v4.rewrite.ring_system_fiber import warm_ring_system_candidate_indices
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


@dataclass(frozen=True)
class IndexedTrainingSupportRow:
    absolute_index: int
    support: TrainingSupportRow


class TrainingSupportCompilationDataset(Dataset[IndexedTrainingSupportRow]):
    """Small-output wrapper around the exact deterministic training dataset."""

    def __init__(
        self,
        records: tuple[PathRecord, ...],
        *,
        start_index: int,
        length: int,
        seed: int,
        late_time_fraction: float,
        operational_horizon: float,
        progress_stratification_fraction: float,
        ring_catalog: TypedRingCatalog,
        ring_electronic_mode: str,
    ) -> None:
        self.start_index = int(start_index)
        self.dataset = FactorizedMarkDataset(
            records,
            start_index=start_index,
            length=length,
            seed=seed,
            late_time_fraction=late_time_fraction,
            operational_horizon=operational_horizon,
            progress_stratification_fraction=progress_stratification_fraction,
            ring_catalog=ring_catalog,
            ring_electronic_mode=ring_electronic_mode,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> IndexedTrainingSupportRow:
        example = self.dataset[index]
        return IndexedTrainingSupportRow(
            absolute_index=self.start_index + int(index),
            support=TrainingSupportRow(
                indices=example.ring_grow_support_indices or (),
                width=int(example.ring_grow_support_width),
                support_is_exact=bool(example.ring_grow_support_is_exact),
                enablement_is_exact=bool(example.ring_grow_enablement_is_exact),
            ),
        )


def _collate_support_rows(
    rows: list[IndexedTrainingSupportRow],
) -> tuple[IndexedTrainingSupportRow, ...]:
    return tuple(rows)


def _initialize_support_worker(_: int) -> None:
    # Chemistry workers are process-parallel; nested BLAS/Torch thread pools
    # only oversubscribe the container and amplify long-tail latency.
    torch.set_num_threads(1)


def iter_training_support_rows(
    records: tuple[PathRecord, ...],
    *,
    start_index: int,
    stop_index: int,
    seed: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    ring_catalog: TypedRingCatalog,
    ring_electronic_mode: str,
    workers: int,
    microbatch_size: int,
    prefetch_factor: int = 2,
) -> Iterator[IndexedTrainingSupportRow]:
    """Yield ordered rows while workers dynamically execute small chunks."""

    if start_index < 0 or stop_index <= start_index:
        raise ValueError("training support compilation bounds are invalid")
    if workers < 0 or microbatch_size <= 0 or prefetch_factor <= 0:
        raise ValueError("training support worker settings are invalid")
    warm_ring_system_candidate_indices(ring_catalog)
    dataset = TrainingSupportCompilationDataset(
        records,
        start_index=start_index,
        length=stop_index - start_index,
        seed=seed,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
        ring_catalog=ring_catalog,
        ring_electronic_mode=ring_electronic_mode,
    )
    options: dict[str, Any] = {}
    if workers > 0:
        options.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            worker_init_fn=_initialize_support_worker,
            # The production container is Linux. Explicit fork preserves the
            # parent-loaded path corpus and multi-gigabyte catalog by COW.
            multiprocessing_context="fork",
        )
    loader = DataLoader(
        dataset,
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=_collate_support_rows,
        pin_memory=False,
        drop_last=False,
        **options,
    )
    for microbatch in loader:
        yield from microbatch


def _validated_cached_prefix(
    cache: ShardedTrainingSupportCache,
    *,
    start_index: int,
    stop_index: int,
) -> tuple[int, tuple[Path, ...]]:
    cursor = int(start_index)
    paths = []
    while cursor < stop_index:
        shard_stop = min(cursor + cache.shard_size, cache.total_rows, stop_index)
        path = training_support_shard_path(cache.root, start=cursor, stop=shard_stop)
        if not path.is_file():
            break
        load_training_support_shard(
            path,
            signature=cache.signature,
            expected_start=cursor,
            expected_stop=shard_stop,
        )
        paths.append(path)
        cursor = shard_stop
    return cursor, tuple(paths)


def compile_training_support_shards(
    records: tuple[PathRecord, ...],
    *,
    cache: ShardedTrainingSupportCache,
    start_index: int,
    stop_index: int,
    seed: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    ring_catalog: TypedRingCatalog,
    ring_electronic_mode: str,
    workers: int,
    microbatch_size: int,
    prefetch_factor: int = 2,
    progress_callback: Callable[[Mapping[str, float]], None] | None = None,
    shard_callback: Callable[[Mapping[str, float]], None] | None = None,
) -> tuple[Path, ...]:
    """Resume a contiguous cache range and atomically publish complete shards."""

    if start_index < 0 or stop_index > cache.total_rows or stop_index <= start_index:
        raise ValueError("requested training support range is invalid")
    if start_index % cache.shard_size != 0:
        raise ValueError("training support range must begin at a shard boundary")
    if stop_index != cache.total_rows and stop_index % cache.shard_size != 0:
        raise ValueError("training support range must end at a shard boundary")
    cursor, completed_paths = _validated_cached_prefix(
        cache,
        start_index=start_index,
        stop_index=stop_index,
    )
    if cursor == stop_index:
        return completed_paths

    started = perf_counter()
    published = list(completed_paths)
    buffer: list[TrainingSupportRow] = []
    shard_start = cursor
    compiled = 0
    for indexed in iter_training_support_rows(
        records,
        start_index=cursor,
        stop_index=stop_index,
        seed=seed,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
        ring_catalog=ring_catalog,
        ring_electronic_mode=ring_electronic_mode,
        workers=workers,
        microbatch_size=microbatch_size,
        prefetch_factor=prefetch_factor,
    ):
        expected_index = cursor + compiled
        if indexed.absolute_index != expected_index:
            raise RuntimeError(
                "training support compiler lost deterministic stream order: "
                f"expected {expected_index}, received {indexed.absolute_index}"
            )
        buffer.append(indexed.support)
        compiled += 1
        absolute_stop = indexed.absolute_index + 1
        shard_stop = min(shard_start + cache.shard_size, stop_index)
        if absolute_stop == shard_stop:
            shard = TrainingSupportShard.from_rows(tuple(buffer), start=shard_start)
            path = training_support_shard_path(
                cache.root,
                start=shard.start,
                stop=shard.stop,
            )
            save_training_support_shard(shard, path, signature=cache.signature)
            published.append(path)
            buffer.clear()
            shard_start = shard.stop
            if shard_callback is not None:
                elapsed = max(perf_counter() - started, 1e-12)
                shard_callback(
                    {
                        "shard_start": float(shard.start),
                        "shard_stop": float(shard.stop),
                        "compiled_rows": float(compiled),
                        "elapsed_seconds": float(elapsed),
                        "rows_per_second": float(compiled / elapsed),
                    }
                )
        if progress_callback is not None and (
            compiled % max(microbatch_size, 1) == 0 or absolute_stop == stop_index
        ):
            elapsed = max(perf_counter() - started, 1e-12)
            progress_callback(
                {
                    "compiled_rows": float(compiled),
                    "elapsed_seconds": float(elapsed),
                    "rows_per_second": float(compiled / elapsed),
                    "absolute_stop": float(absolute_stop),
                }
            )
    if buffer or shard_start != stop_index:
        raise RuntimeError("training support compiler ended with a partial shard")
    return tuple(published)


__all__ = [
    "IndexedTrainingSupportRow",
    "TrainingSupportCompilationDataset",
    "compile_training_support_shards",
    "iter_training_support_rows",
]
