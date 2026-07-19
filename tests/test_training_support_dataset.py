from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_mark_conditional import FactorizedMarkDataset
from compose_v4.experiments.training_support_cache import (
    ShardedTrainingSupportCache,
    TrainingSupportRow,
    TrainingSupportShard,
    save_training_support_shard,
    training_support_shard_path,
)
from compose_v4.experiments.training_support_compiler import (
    compile_training_support_shards,
    iter_training_support_rows,
)
from compose_v4.model.factorized_tracelet_rate_model import molecular_state_cache_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog_from_paths


def _records_and_catalog() -> tuple[tuple[PathRecord, ...], object]:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(431),
        n_slots=12,
    )
    path = TraceProgressCTMC(
        compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )
    )
    return (PathRecord("benzene", path),), build_typed_ring_catalog_from_paths((path,))


def _dataset(records, catalog, **updates) -> FactorizedMarkDataset:
    arguments = {
        "start_index": 0,
        "length": 8,
        "seed": 47,
        "late_time_fraction": 0.5,
        "operational_horizon": 16.0,
        "progress_stratification_fraction": 0.5,
        "ring_catalog": catalog,
        "ring_electronic_mode": "factorized_local",
    }
    arguments.update(updates)
    return FactorizedMarkDataset(records, **arguments)


def test_factorized_dataset_uses_required_sparse_training_support_cache(tmp_path) -> None:
    records, catalog = _records_and_catalog()
    oracle = _dataset(records, catalog)
    expected = tuple(oracle[index] for index in range(len(oracle)))
    signature = {"format_version": 1, "stream": "unit-test"}
    cache = ShardedTrainingSupportCache(
        tmp_path,
        signature,
        total_rows=len(expected),
        shard_size=len(expected),
    )
    rows = tuple(
        TrainingSupportRow(
            indices=example.ring_grow_support_indices or (),
            width=example.ring_grow_support_width,
            support_is_exact=example.ring_grow_support_is_exact,
            enablement_is_exact=example.ring_grow_enablement_is_exact,
        )
        for example in expected
    )
    shard = TrainingSupportShard.from_rows(rows, start=0)
    save_training_support_shard(
        shard,
        training_support_shard_path(cache.root, start=0, stop=len(expected)),
        signature=signature,
    )

    cached = _dataset(
        records,
        catalog,
        training_support_cache=cache,
        require_cached_support=True,
    )
    observed = tuple(cached[index] for index in range(len(cached)))

    assert cached._ring_support_model is None
    for left, right in zip(expected, observed):
        assert molecular_state_cache_key(left.state) == molecular_state_cache_key(right.state)
        assert left.time == right.time
        assert left.teacher_action == right.teacher_action
        assert left.teacher_rule_name == right.teacher_rule_name
        assert left.teacher_rate == right.teacher_rate
        assert left.importance_weight == right.importance_weight
        assert left.ring_grow_support_indices == right.ring_grow_support_indices
        assert left.ring_grow_support_width == right.ring_grow_support_width
        assert left.ring_grow_support_is_exact == right.ring_grow_support_is_exact
        assert left.ring_grow_enablement_is_exact == right.ring_grow_enablement_is_exact


def test_factorized_dataset_refuses_missing_required_training_support(tmp_path) -> None:
    records, catalog = _records_and_catalog()
    cache = ShardedTrainingSupportCache(
        tmp_path,
        {"format_version": 1, "stream": "missing"},
        total_rows=8,
        shard_size=8,
    )
    dataset = _dataset(
        records,
        catalog,
        training_support_cache=cache,
        require_cached_support=True,
    )
    with pytest.raises(FileNotFoundError, match="missing compiled training support"):
        dataset[0]
    assert dataset._ring_support_model is None


def test_training_support_compiler_publishes_ordered_resumable_shards(tmp_path) -> None:
    records, catalog = _records_and_catalog()
    signature = {"format_version": 1, "stream": "compiled-unit-test"}
    cache = ShardedTrainingSupportCache(
        tmp_path,
        signature,
        total_rows=16,
        shard_size=8,
    )
    published = []
    paths = compile_training_support_shards(
        records,
        cache=cache,
        start_index=0,
        stop_index=16,
        seed=47,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        progress_stratification_fraction=0.5,
        ring_catalog=catalog,
        ring_electronic_mode="factorized_local",
        workers=0,
        microbatch_size=4,
        shard_callback=published.append,
    )
    assert len(paths) == 2
    assert [int(event["shard_stop"]) for event in published] == [8, 16]
    assert all(path.is_file() for path in paths)
    mtimes = tuple(path.stat().st_mtime_ns for path in paths)

    oracle = _dataset(records, catalog, length=16)
    for index in range(16):
        expected = oracle[index]
        observed = cache.require(index)
        assert observed.indices == (expected.ring_grow_support_indices or ())
        assert observed.width == expected.ring_grow_support_width
        assert observed.support_is_exact == expected.ring_grow_support_is_exact
        assert observed.enablement_is_exact == expected.ring_grow_enablement_is_exact

    resumed = compile_training_support_shards(
        records,
        cache=cache,
        start_index=0,
        stop_index=16,
        seed=47,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        progress_stratification_fraction=0.5,
        ring_catalog=catalog,
        ring_electronic_mode="factorized_local",
        workers=0,
        microbatch_size=4,
    )
    assert resumed == paths
    assert tuple(path.stat().st_mtime_ns for path in paths) == mtimes


def test_training_support_compiler_preserves_absolute_indices_for_nonzero_range(
    tmp_path,
) -> None:
    records, catalog = _records_and_catalog()
    signature = {"format_version": 1, "stream": "nonzero-range"}
    cache = ShardedTrainingSupportCache(
        tmp_path,
        signature,
        total_rows=24,
        shard_size=8,
    )
    paths = compile_training_support_shards(
        records,
        cache=cache,
        start_index=8,
        stop_index=16,
        seed=47,
        late_time_fraction=0.5,
        operational_horizon=16.0,
        progress_stratification_fraction=0.5,
        ring_catalog=catalog,
        ring_electronic_mode="factorized_local",
        workers=0,
        microbatch_size=4,
    )
    assert len(paths) == 1
    oracle = _dataset(records, catalog, start_index=8, length=8)
    for offset in range(8):
        expected = oracle[offset]
        observed = cache.require(8 + offset)
        assert observed.indices == (expected.ring_grow_support_indices or ())
        assert observed.width == expected.ring_grow_support_width
        assert observed.support_is_exact == expected.ring_grow_support_is_exact
        assert observed.enablement_is_exact == expected.ring_grow_enablement_is_exact


def test_unordered_worker_scheduler_reconstructs_the_sequential_stream() -> None:
    records, catalog = _records_and_catalog()
    arguments = {
        "start_index": 0,
        "stop_index": 16,
        "seed": 47,
        "late_time_fraction": 0.5,
        "operational_horizon": 16.0,
        "progress_stratification_fraction": 0.5,
        "ring_catalog": catalog,
        "ring_electronic_mode": "factorized_local",
        "microbatch_size": 2,
        "prefetch_factor": 2,
    }
    sequential = tuple(
        iter_training_support_rows(records, workers=0, **arguments)
    )
    parallel = tuple(
        iter_training_support_rows(records, workers=2, **arguments)
    )

    assert parallel == sequential
