"""Parallel deterministic compilation of sparse training-support shards."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Queue
from time import perf_counter

import torch
from torch.utils.data import Dataset

from compose_v4.chem.molecular_graph import MolecularGraph
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
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.ring_system_fiber import (
    clear_semantic_ring_state_caches,
    warm_ring_system_candidate_indices,
)
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


MINIMUM_SUPPORT_REORDER_WINDOW = 16000
WORKER_SUPPORT_CACHE_LIMIT = 512


@dataclass(frozen=True)
class IndexedTrainingSupportRow:
    absolute_index: int
    support: TrainingSupportRow


@dataclass(frozen=True)
class IndexedTrainingSupportRequest:
    """Compact chemistry request sampled by the sole path-owning process."""

    absolute_index: int
    state: MolecularGraph
    full_support: bool


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


class TrainingSupportRequestDataset(Dataset[IndexedTrainingSupportRequest]):
    """Sample states without letting support workers touch the path corpus."""

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
            ring_catalog=None,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> IndexedTrainingSupportRequest:
        example = self.dataset[index]
        return IndexedTrainingSupportRequest(
            absolute_index=self.start_index + int(index),
            state=example.state,
            full_support=example.teacher_rule_name == "ring_system_grow",
        )


_SUPPORT_RING_CATALOG: TypedRingCatalog | None = None
_SUPPORT_RING_ELECTRONIC_MODE = "factorized_local"
_SUPPORT_RING_MODEL: FactorizedTraceletRateModel | None = None
_SUPPORT_REQUESTS_SINCE_RESET = 0


def _initialize_support_worker() -> None:
    # Chemistry workers are process-parallel; nested BLAS/Torch thread pools
    # only oversubscribe the container and amplify long-tail latency.
    torch.set_num_threads(1)
    catalog = _SUPPORT_RING_CATALOG
    if catalog is None:
        raise RuntimeError("support worker was started without its ring catalog")
    global _SUPPORT_REQUESTS_SINCE_RESET, _SUPPORT_RING_MODEL
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _SUPPORT_RING_MODEL = FactorizedTraceletRateModel(
            catalog,
            hidden_dim=1,
            message_passing_steps=1,
            mark_dim=1,
            ring_electronic_mode=_SUPPORT_RING_ELECTRONIC_MODE,
            ring_candidate_cache_limit=WORKER_SUPPORT_CACHE_LIMIT,
        )
    _SUPPORT_REQUESTS_SINCE_RESET = 0


def _compile_support_request(
    request: IndexedTrainingSupportRequest,
) -> IndexedTrainingSupportRow:
    global _SUPPORT_REQUESTS_SINCE_RESET
    model = _SUPPORT_RING_MODEL
    if model is None:
        raise RuntimeError("support worker lacks its chemistry oracle")
    if _SUPPORT_REQUESTS_SINCE_RESET >= WORKER_SUPPORT_CACHE_LIMIT:
        model.clear_ring_candidate_caches()
        clear_semantic_ring_state_caches()
        _SUPPORT_REQUESTS_SINCE_RESET = 0
    if request.full_support:
        mask = model._ring_grow_support(request.state)
    else:
        mask = model._ring_grow_enablement_certificate(request.state)
    _SUPPORT_REQUESTS_SINCE_RESET += 1
    return IndexedTrainingSupportRow(
        absolute_index=int(request.absolute_index),
        support=TrainingSupportRow(
            indices=tuple(
                int(index) for index, supported in enumerate(mask) if supported
            ),
            width=len(mask),
            support_is_exact=bool(request.full_support),
            enablement_is_exact=True,
        ),
    )


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
    """Yield ordered rows from a bounded, dynamically scheduled worker queue.

    A standard ordered DataLoader stops feeding workers when one early row is
    unusually expensive.  Exact ring support has precisely that heavy-tailed
    latency profile.  Workers therefore complete individual indices out of
    order while the parent buffers a bounded window and releases only the next
    deterministic absolute index.
    """

    if start_index < 0 or stop_index <= start_index:
        raise ValueError("training support compilation bounds are invalid")
    if workers < 0 or microbatch_size <= 0 or prefetch_factor <= 0:
        raise ValueError("training support worker settings are invalid")
    warm_ring_system_candidate_indices(ring_catalog)
    if workers == 0:
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
        for index in range(len(dataset)):
            yield dataset[index]
        return

    request_dataset = TrainingSupportRequestDataset(
        records,
        start_index=start_index,
        length=stop_index - start_index,
        seed=seed,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
    )

    # Keep enough independent rows in flight to absorb rare multi-second ring
    # certificates without allowing an unbounded reordering buffer.  The
    # existing microbatch option now controls queue depth, not task grouping;
    # each task remains one row so no fast row is trapped behind a slow peer.
    reorder_window = max(
        MINIMUM_SUPPORT_REORDER_WINDOW,
        int(workers) * int(microbatch_size) * int(prefetch_factor),
    )
    global _SUPPORT_RING_CATALOG, _SUPPORT_RING_ELECTRONIC_MODE
    _SUPPORT_RING_CATALOG = ring_catalog
    _SUPPORT_RING_ELECTRONIC_MODE = str(ring_electronic_mode)
    context = mp.get_context("fork")
    pool = context.Pool(
        processes=workers,
        initializer=_initialize_support_worker,
    )
    completions: Queue[tuple[bool, object]] = Queue()
    pending: dict[int, IndexedTrainingSupportRow] = {}
    next_submit = 0
    next_yield = 0
    in_flight = 0

    def fill_window() -> None:
        nonlocal in_flight, next_submit
        while (
            next_submit < len(request_dataset)
            and next_submit - next_yield < reorder_window
        ):
            request = request_dataset[next_submit]
            pool.apply_async(
                _compile_support_request,
                (request,),
                callback=lambda row: completions.put((True, row)),
                error_callback=lambda error: completions.put((False, error)),
            )
            in_flight += 1
            next_submit += 1

    completed_normally = False
    try:
        fill_window()
        while in_flight:
            succeeded, payload = completions.get()
            in_flight -= 1
            if not succeeded:
                if isinstance(payload, BaseException):
                    raise payload
                raise RuntimeError(f"support worker failed: {payload!r}")
            if not isinstance(payload, IndexedTrainingSupportRow):
                raise TypeError("support worker returned an invalid row payload")
            row = payload
            local_index = int(row.absolute_index) - int(start_index)
            if not 0 <= local_index < len(request_dataset):
                raise RuntimeError(
                    "support worker returned the wrong deterministic index"
                )
            if local_index in pending or local_index < next_yield:
                raise RuntimeError("support worker returned a duplicate row")
            pending[local_index] = row
            while next_yield in pending:
                row = pending.pop(next_yield)
                next_yield += 1
                fill_window()
                yield row
            fill_window()
        if pending or next_yield != len(request_dataset):
            raise RuntimeError("support compiler lost a reordered worker result")
        completed_normally = True
    finally:
        if completed_normally:
            pool.close()
        else:
            pool.terminate()
        pool.join()
        _SUPPORT_RING_CATALOG = None
        _SUPPORT_RING_ELECTRONIC_MODE = "factorized_local"


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
    "IndexedTrainingSupportRequest",
    "TrainingSupportCompilationDataset",
    "TrainingSupportRequestDataset",
    "compile_training_support_shards",
    "iter_training_support_rows",
]
