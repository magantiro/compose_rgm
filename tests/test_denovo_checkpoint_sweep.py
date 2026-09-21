"""Tests for the de-novo checkpoint-sweep row builder.

The two properties worth protecting are the ones that have silently corrupted
de-novo numbers before: the VALIDITY DENOMINATOR (attempts vs committed
endpoints, which differ and have been conflated), and shard identity (two shards
claiming the same trajectory index must raise rather than be quietly merged).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from denovo_checkpoint_sweep import (
    checkpoint_row,
    load_shard_records,
)


def _write_shard(directory: Path, name: str, records: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps({"seed": 1, "records": records}))


def _record(index: int, smiles: str, **overrides) -> dict:
    record = {
        "index": index,
        "smiles": smiles,
        "valid_state": bool(smiles),
        "connected": bool(smiles),
        "exhausted_event_budget": False,
        "event_rules": ["ring_system_grow"],
    }
    record.update(overrides)
    return record


def test_load_shard_records_concatenates_shards_in_sorted_order(tmp_path: Path) -> None:
    _write_shard(tmp_path, "b.json", [_record(2, "c1ccccc1")])
    _write_shard(tmp_path, "a.json", [_record(1, "CCO")])
    records, names = load_shard_records(tmp_path)
    assert names == ["a.json", "b.json"]
    assert [record["index"] for record in records] == [1, 2]


def test_duplicate_trajectory_index_raises_rather_than_merging(tmp_path: Path) -> None:
    _write_shard(tmp_path, "a.json", [_record(1, "CCO")])
    _write_shard(tmp_path, "b.json", [_record(1, "c1ccccc1")])
    with pytest.raises(ValueError, match="appears in both"):
        load_shard_records(tmp_path)


def test_missing_shards_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no shard json"):
        load_shard_records(tmp_path)


def test_validity_denominator_is_attempts_and_is_reported_apart_from_committed() -> None:
    """A trajectory that committed nothing lowers validity-over-ATTEMPTS only."""

    records = [
        _record(0, "c1ccccc1"),
        _record(1, "CCO"),
        _record(2, ""),  # committed nothing
        _record(3, ""),
    ]
    row = checkpoint_row(records, label="t")
    assert row["attempted"] == 4
    assert row["committed"] == 2
    # Two of four attempts produced a molecule.
    assert row["validity_over_attempts"] == 0.5
    # Both committed endpoints are valid states: the architectural guarantee.
    assert row["valid_state_fraction"] == 1.0
    assert row["connected_fraction"] == 1.0


def test_row_carries_the_decomposition_and_the_ring_census() -> None:
    records = [_record(i, smiles) for i, smiles in enumerate(["C1CN1CCCC", "c1ccccc1CCO"])]
    row = checkpoint_row(records, label="t")
    assert set(row["decomposition"]) >= {
        "ring_size_distribution",
        "strain_size_strata",
        "strain_size_regression",
        "ring_event_census",
    }
    # The event census reads the model's own transitions, not the endpoints.
    assert row["decomposition"]["ring_event_census"]["ring_forming_events"] == 2
    assert row["strained_ring_census"]["molecules"] == 2


def test_records_without_an_index_are_kept_and_do_not_collide(tmp_path: Path) -> None:
    _write_shard(tmp_path, "a.json", [{"smiles": "CCO"}, {"smiles": "CCN"}])
    records, _names = load_shard_records(tmp_path)
    assert len(records) == 2


def test_shards_from_a_different_sampling_design_raise(tmp_path: Path) -> None:
    """A shard at a different ``total`` is a different trajectory family."""

    (tmp_path / "a.json").write_text(
        json.dumps({"seed": 20260920, "total": 200, "horizon": 16.0, "records": [_record(0, "CCO")]})
    )
    (tmp_path / "b.json").write_text(
        json.dumps(
            {"seed": 20260920, "total": 1000, "horizon": 16.0, "records": [_record(400, "CCN")]}
        )
    )
    with pytest.raises(ValueError, match="more than one sampling design"):
        load_shard_records(tmp_path)


def test_a_single_consistent_design_is_accepted(tmp_path: Path) -> None:
    for name, index in (("a.json", 0), ("b.json", 1)):
        (tmp_path / name).write_text(
            json.dumps(
                {
                    "seed": 20260920,
                    "total": 200,
                    "horizon": 16.0,
                    "records": [_record(index, "CCO")],
                }
            )
        )
    records, names = load_shard_records(tmp_path)
    assert len(records) == 2 and names == ["a.json", "b.json"]
