from __future__ import annotations

from dataclasses import replace
from threading import Thread
from time import sleep

import pytest
import torch

from compose_v4.experiments.training_support_cache import (
    ShardedTrainingSupportCache,
    TrainingSupportRow,
    TrainingSupportShard,
    load_training_support_shard,
    save_training_support_shard,
    training_support_cache_root,
    training_support_shard_path,
    training_support_signature_fingerprint,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    RingTeacherSemanticCertificate,
)


def _signature(**updates: object) -> dict[str, object]:
    signature: dict[str, object] = {
        "format_version": 1,
        "ring_support_semantics_version": 1,
        "path_fingerprint": "paths-v1",
        "seed": 23,
        "late_time_fraction": 0.5,
        "operational_horizon": 16.0,
        "progress_stratification_fraction": 0.5,
        "ring_catalog_fingerprint": "rings-v1",
        "ring_electronic_mode": "factorized_local",
    }
    signature.update(updates)
    return signature


def _rows() -> tuple[TrainingSupportRow, ...]:
    return (
        TrainingSupportRow(
            (1, 4),
            8,
            True,
            True,
            RingTeacherSemanticCertificate(action_is_valid=True, templates=()),
        ),
        TrainingSupportRow((), 8, False, True),
        TrainingSupportRow((6,), 8, False, True),
    )


def test_training_support_signature_is_stable_and_content_addressed(tmp_path) -> None:
    left = _signature()
    right = dict(reversed(tuple(left.items())))
    assert training_support_signature_fingerprint(left) == (
        training_support_signature_fingerprint(right)
    )
    assert training_support_cache_root(tmp_path, left) != training_support_cache_root(
        tmp_path,
        _signature(seed=24),
    )
    assert training_support_cache_root(tmp_path, left) != training_support_cache_root(
        tmp_path,
        _signature(ring_support_semantics_version=2),
    )


def test_training_support_shard_round_trip_preserves_sparse_rows(tmp_path) -> None:
    signature = _signature()
    shard = TrainingSupportShard.from_rows(_rows(), start=4)
    path = training_support_shard_path(tmp_path, start=4, stop=7)
    save_training_support_shard(shard, path, signature=signature)
    restored = load_training_support_shard(
        path,
        signature=signature,
        expected_start=4,
        expected_stop=7,
    )

    assert tuple(restored.row(index) for index in range(4, 7)) == _rows()
    with pytest.raises(ValueError, match="cache/config mismatch"):
        load_training_support_shard(
            path,
            signature=_signature(seed=25),
        )


def test_training_support_cache_rejects_legacy_shard_without_teacher_certificates(
    tmp_path,
) -> None:
    signature = _signature()
    cache = ShardedTrainingSupportCache(
        tmp_path,
        signature,
        total_rows=3,
        shard_size=3,
    )
    path = training_support_shard_path(cache.root, start=0, stop=3)
    save_training_support_shard(
        TrainingSupportShard.from_rows(_rows(), start=0),
        path,
        signature=signature,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("teacher_semantic_certificates")
    torch.save(payload, path)

    legacy = load_training_support_shard(
        path,
        signature=signature,
        expected_start=0,
        expected_stop=3,
    )
    assert not legacy.teacher_certificates_complete
    assert legacy.row(0).teacher_semantic_certificate is None
    with pytest.raises(FileNotFoundError, match="lacks exact semantic"):
        cache.require(0)


def test_training_support_shard_rejects_unsorted_or_out_of_range_rows() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        TrainingSupportRow((2, 1), 4, True, True)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        TrainingSupportRow((4,), 4, True, True)

    valid = TrainingSupportShard.from_rows(_rows(), start=0)
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(valid, indices=valid.indices.flip(0))


def test_sharded_training_support_cache_is_lazy_and_requires_missing_rows(
    tmp_path,
) -> None:
    signature = _signature()
    cache = ShardedTrainingSupportCache(
        tmp_path,
        signature,
        total_rows=6,
        shard_size=3,
        shard_cache_limit=1,
    )
    assert cache.get(0) is None
    with pytest.raises(FileNotFoundError, match="missing compiled training support"):
        cache.require(0)

    first = TrainingSupportShard.from_rows(_rows(), start=0)
    save_training_support_shard(
        first,
        training_support_shard_path(cache.root, start=0, stop=3),
        signature=signature,
    )
    assert cache.require(0) == _rows()[0]
    assert cache.require(2) == _rows()[2]

    second_rows = tuple(TrainingSupportRow((index,), 8, False, True) for index in (0, 2, 7))
    second = TrainingSupportShard.from_rows(second_rows, start=3)
    save_training_support_shard(
        second,
        training_support_shard_path(cache.root, start=3, stop=6),
        signature=signature,
    )
    assert cache.require(5) == second_rows[2]
    assert tuple(cache._loaded) == (3,)


def test_training_support_cache_waits_for_atomic_concurrent_publication(tmp_path) -> None:
    signature = _signature()
    cache = ShardedTrainingSupportCache(
        tmp_path,
        signature,
        total_rows=3,
        shard_size=3,
        wait_timeout_seconds=1.0,
        poll_interval_seconds=0.01,
    )

    def publish() -> None:
        sleep(0.05)
        shard = TrainingSupportShard.from_rows(_rows(), start=0)
        save_training_support_shard(
            shard,
            training_support_shard_path(cache.root, start=0, stop=3),
            signature=signature,
        )

    publisher = Thread(target=publish)
    publisher.start()
    try:
        assert cache.require(1) == _rows()[1]
    finally:
        publisher.join()


def test_training_support_cache_waits_for_certificate_complete_replacement(tmp_path) -> None:
    signature = _signature()
    cache = ShardedTrainingSupportCache(
        tmp_path,
        signature,
        total_rows=3,
        shard_size=3,
        wait_timeout_seconds=1.0,
        poll_interval_seconds=0.01,
    )
    path = training_support_shard_path(cache.root, start=0, stop=3)
    complete = TrainingSupportShard.from_rows(_rows(), start=0)
    save_training_support_shard(complete, path, signature=signature)
    legacy_payload = torch.load(path, map_location="cpu", weights_only=False)
    legacy_payload.pop("teacher_semantic_certificates")
    torch.save(legacy_payload, path)

    def replace() -> None:
        sleep(0.05)
        save_training_support_shard(complete, path, signature=signature)

    publisher = Thread(target=replace)
    publisher.start()
    try:
        assert cache.require(0) == _rows()[0]
    finally:
        publisher.join()
