from __future__ import annotations

import pytest

from modal_apps.evaluate_rollout_shards import (
    _derive_shard_seed,
    _expected_shard_metadata,
    _merge_shard_payloads,
    _partition_sample_counts,
    _shard_filename,
    _validate_shard_payload,
)


def test_partition_sample_counts_is_exact_and_balanced() -> None:
    counts = _partition_sample_counts(2000, 80)
    assert len(counts) == 80
    assert sum(counts) == 2000
    assert max(counts) - min(counts) <= 1
    assert _partition_sample_counts(10, 3) == (4, 3, 3)


def test_partition_rejects_empty_shards() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _partition_sample_counts(3, 4)


def test_shard_seeds_are_stable_and_distinct() -> None:
    first = tuple(_derive_shard_seed(20260724, index) for index in range(100))
    second = tuple(_derive_shard_seed(20260724, index) for index in range(100))
    assert first == second
    assert len(set(first)) == len(first)
    assert first != tuple(_derive_shard_seed(20260725, index) for index in range(100))


def _payload(index: int, count: int, *, shard_count: int = 3) -> tuple[dict, dict]:
    expected = _expected_shard_metadata(
        source_run_label="source-run",
        checkpoint_name="checkpoint.recovery.pt",
        checkpoint_sha256="abc123",
        shard_index=index,
        shard_count=shard_count,
        samples=count,
        base_seed=7,
    )
    payload = {**expected, "rollouts": tuple(f"{index}:{offset}" for offset in range(count))}
    return payload, expected


def test_validate_shard_payload_rejects_checkpoint_drift() -> None:
    payload, expected = _payload(0, 2)
    payload["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _validate_shard_payload(payload, expected)


def test_merge_is_deterministic_even_when_map_results_are_unordered() -> None:
    pairs = (_payload(0, 2), _payload(1, 1), _payload(2, 2))
    payloads = [pairs[2][0], pairs[0][0], pairs[1][0]]
    expected = tuple(pair[1] for pair in pairs)
    assert _merge_shard_payloads(payloads, expected) == (
        "0:0",
        "0:1",
        "1:0",
        "2:0",
        "2:1",
    )


def test_merge_rejects_duplicate_or_missing_shards() -> None:
    first, expected_first = _payload(0, 1, shard_count=2)
    second, expected_second = _payload(1, 1, shard_count=2)
    with pytest.raises(ValueError, match="duplicate"):
        _merge_shard_payloads((first, first), (expected_first, expected_second))
    with pytest.raises(ValueError, match="one payload"):
        _merge_shard_payloads((first,), (expected_first, expected_second))


def test_shard_filename_encodes_complete_layout() -> None:
    assert _shard_filename(7, 80) == "shard-00007-of-00080.pt"
