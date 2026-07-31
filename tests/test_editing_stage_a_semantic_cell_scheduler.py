"""Contract tests for the deterministic Stage-A cell scheduler."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

from compose_v4.data.successor_fiber_cache import SuccessorFiberCacheAddress
from compose_v4.experiments.editing_stage_a_semantic_cell_scheduler import (
    StageAScheduledBatch,
    StageAScheduledExample,
    StageASemanticCellCursor,
    StageASemanticCellScheduler,
    StageASemanticCellSchedulerError,
)


def _address(index: int, *, terminal: bool = False) -> SuccessorFiberCacheAddress:
    path_length = 2
    return SuccessorFiberCacheAddress(
        packed_shard_content_sha256=f"{index + 1:064x}",
        packed_shard_name=f"shard-{index}.jsonl.gz",
        entry_index=index,
        layer="test-lane",
        partition="train",
        trace_id=f"trace-{index}",
        trace_source_key=f"source-{index}",
        trace_target_key=f"target-{index}",
        progress_index=path_length if terminal else 1,
        path_length=path_length,
    )


def _scheduler(*, seed: int = 17) -> StageASemanticCellScheduler:
    return StageASemanticCellScheduler(
        declared_cells=("cell-c", "cell-a", "cell-b"),
        examples_by_cell={
            "cell-a": tuple(_address(index) for index in range(0, 4)),
            "cell-b": tuple(_address(index) for index in range(4, 7)),
            "cell-c": tuple(_address(index) for index in range(7, 12)),
        },
        seed=seed,
    )


def test_scheduler_has_deterministic_equal_cell_exposure() -> None:
    first = _scheduler(seed=91)
    second = StageASemanticCellScheduler(
        declared_cells=("cell-b", "cell-c", "cell-a"),
        examples_by_cell={
            "cell-c": tuple(_address(index) for index in reversed(range(7, 12))),
            "cell-b": tuple(_address(index) for index in reversed(range(4, 7))),
            "cell-a": tuple(_address(index) for index in reversed(range(0, 4))),
        },
        seed=91,
    )
    first_batch, first_cursor = first.take(first.initial_cursor(), 23)
    second_batch, second_cursor = second.take(second.initial_cursor(), 23)

    assert first.scheduler_sha256 == second.scheduler_sha256
    assert first_batch == second_batch
    assert first_cursor == second_cursor
    counts = Counter(row.semantic_cell_id for row in first_batch.examples)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert first_batch.semantic_cell_multiplicities == (
        ("cell-a", 8),
        ("cell-b", 8),
        ("cell-c", 7),
    )
    assert first_batch.addresses == tuple(row.address for row in first_batch.examples)
    assert [row.semantic_cell_id for row in first_batch.examples[:6]] == [
        "cell-a",
        "cell-b",
        "cell-c",
        "cell-a",
        "cell-b",
        "cell-c",
    ]


def test_scheduler_cycles_each_cell_inventory_with_seeded_shuffles() -> None:
    examples = tuple(_address(index) for index in range(5))
    scheduler = StageASemanticCellScheduler(
        declared_cells=("only-cell",),
        examples_by_cell={"only-cell": examples},
        seed=123,
    )
    batch, _ = scheduler.take(scheduler.initial_cursor(), 10)
    first_cycle = tuple(row.address for row in batch.examples[:5])
    second_cycle = tuple(row.address for row in batch.examples[5:])
    expected = set(examples)
    other_seed = StageASemanticCellScheduler(
        declared_cells=("only-cell",),
        examples_by_cell={"only-cell": examples},
        seed=124,
    )
    other_batch, _ = other_seed.take(other_seed.initial_cursor(), 5)

    assert set(first_cycle) == expected
    assert set(second_cycle) == expected
    assert first_cycle != tuple(row.address for row in other_batch.examples)
    assert _scheduler(seed=1).scheduler_sha256 != _scheduler(seed=2).scheduler_sha256


def test_cursor_serialization_resumes_exact_stream_and_binds_scheduler() -> None:
    scheduler = _scheduler(seed=41)
    full, _ = scheduler.take(scheduler.initial_cursor(), 31)
    prefix, cursor = scheduler.take(scheduler.initial_cursor(), 11)
    encoded = cursor.to_json_bytes()
    restored = StageASemanticCellCursor.from_json_bytes(encoded)
    suffix, terminal_cursor = scheduler.take(restored, 20)

    assert hash(restored) == hash(cursor)
    assert restored.cursor_sha256 == restored.to_payload()["cursor_sha256"]
    assert prefix.examples + suffix.examples == full.examples
    assert terminal_cursor.next_stream_index == 31
    assert encoded == restored.to_json_bytes()
    suffix_counts = Counter(row.semantic_cell_id for row in suffix.examples)
    assert max(suffix_counts.values()) - min(suffix_counts.values()) <= 1

    fresh_scheduler = _scheduler(seed=41)
    fresh_suffix, fresh_terminal_cursor = fresh_scheduler.take(restored, 20)
    assert fresh_suffix == suffix
    assert fresh_terminal_cursor == terminal_cursor

    tampered = json.loads(encoded)
    tampered["next_stream_index"] = 12
    with pytest.raises(StageASemanticCellSchedulerError, match="identity is invalid"):
        StageASemanticCellCursor.from_payload(tampered)

    other = _scheduler(seed=42)
    with pytest.raises(StageASemanticCellSchedulerError, match="another scheduler"):
        other.take(restored, 1)


def test_consecutive_batches_reuse_same_cycle_permutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(seed=73)
    calls: Counter[tuple[str, int]] = Counter()
    original = scheduler._cycle_permutation

    def counted(cell_id: str, cycle_index: int):
        calls[(cell_id, cycle_index)] += 1
        return original(cell_id, cycle_index)

    monkeypatch.setattr(scheduler, "_cycle_permutation", counted)
    first, cursor = scheduler.take(scheduler.initial_cursor(), 3)
    second, second_cursor = scheduler.take(cursor, 3)

    assert first.semantic_cell_multiplicities == (
        ("cell-a", 1),
        ("cell-b", 1),
        ("cell-c", 1),
    )
    assert second.semantic_cell_multiplicities == first.semantic_cell_multiplicities
    assert calls == Counter(
        {
            ("cell-a", 0): 1,
            ("cell-b", 0): 1,
            ("cell-c", 0): 1,
        }
    )
    scheduler.take(second_cursor, 100)
    assert set(scheduler._last_permutation_by_cell) == set(scheduler.declared_cells)


@pytest.mark.parametrize(
    ("declared_cells", "examples_by_cell", "message"),
    (
        (("cell-a", "cell-b"), {"cell-a": (_address(0),)}, "lack examples"),
        (("cell-a",), {"cell-a": ()}, "has no examples"),
        (
            ("cell-a",),
            {"cell-a": (_address(0),), "cell-z": (_address(1),)},
            "unknown semantic cells",
        ),
    ),
)
def test_scheduler_rejects_missing_empty_and_unknown_cells(
    declared_cells,
    examples_by_cell,
    message,
) -> None:
    with pytest.raises(StageASemanticCellSchedulerError, match=message):
        StageASemanticCellScheduler(
            declared_cells=declared_cells,
            examples_by_cell=examples_by_cell,
            seed=3,
        )


def test_scheduler_rejects_terminal_or_multiply_assigned_addresses() -> None:
    with pytest.raises(StageASemanticCellSchedulerError, match="terminal addresses"):
        StageASemanticCellScheduler(
            declared_cells=("cell-a",),
            examples_by_cell={"cell-a": (_address(0, terminal=True),)},
            seed=3,
        )

    shared = _address(1)
    with pytest.raises(StageASemanticCellSchedulerError, match="multiple semantic cells"):
        StageASemanticCellScheduler(
            declared_cells=("cell-a", "cell-b"),
            examples_by_cell={"cell-a": (shared,), "cell-b": (shared,)},
            seed=3,
        )


def test_scheduled_examples_carry_no_scientific_or_importance_weight() -> None:
    scheduler = _scheduler()
    batch, _ = scheduler.take(scheduler.initial_cursor(), 8)

    assert tuple(field.name for field in dataclasses.fields(StageAScheduledExample)) == (
        "stream_index",
        "semantic_cell_id",
        "address",
    )
    assert tuple(field.name for field in dataclasses.fields(StageAScheduledBatch)) == ("examples",)
    multiplicities = dict(batch.semantic_cell_multiplicities)
    represented_cell_count = len(multiplicities)
    objective_coefficients = tuple(
        1.0 / (represented_cell_count * multiplicities[row.semantic_cell_id])
        for row in batch.examples
    )
    coefficient_mass_by_cell = {
        cell_id: sum(
            coefficient
            for row, coefficient in zip(
                batch.examples,
                objective_coefficients,
                strict=True,
            )
            if row.semantic_cell_id == cell_id
        )
        for cell_id in multiplicities
    }
    assert sum(objective_coefficients) == pytest.approx(1.0)
    assert coefficient_mass_by_cell == pytest.approx(
        {cell_id: 1.0 / represented_cell_count for cell_id in multiplicities}
    )
    assert not hasattr(batch.examples[0], "importance_weight")
    assert not hasattr(batch.examples[0], "scientific_weight")
