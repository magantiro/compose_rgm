"""Guards on the selection-free chunk compile.

The fused compile itself needs a real authenticated source, so these cover the
decisions that can be made without one: that a bounded pilot cannot be asked for
a nonsense size, and that an empty or wrong-role chunk fails loudly instead of
compiling nothing.
"""

from __future__ import annotations

import pytest

from compose_v4.experiments import editing_v2_process_v2_chunk_compile as chunk


def test_a_pilot_limit_must_be_a_positive_integer() -> None:
    for bad in (0, -1, 1.0, "8"):
        with pytest.raises(chunk.ProcessV2ChunkCompileError, match="positive int"):
            chunk.compile_process_v2_chunk_shard(
                object(), object(), task_identity_sha256="a" * 64, limit=bad
            )


def test_an_empty_chunk_refuses_instead_of_compiling_nothing(monkeypatch) -> None:
    """A chunk that yields no rows is a fault, not an empty success."""

    monkeypatch.setattr(
        chunk, "iter_process_v2_role_shards", lambda *a, **k: iter(())
    )
    source = type("S", (), {"active8_run_root": None, "contracts": None, "index": None})()
    with pytest.raises(chunk.ProcessV2ChunkCompileError, match="no accepted"):
        chunk.chunk_transition_rows(source, task_identity_sha256="b" * 64)


def test_rows_are_filtered_to_the_named_chunk(monkeypatch) -> None:
    wanted, other = "c" * 64, "d" * 64
    published = (
        {"task_identity_sha256": wanted, "n": 1},
        {"task_identity_sha256": other, "n": 2},
        {"task_identity_sha256": wanted, "n": 3},
    )
    monkeypatch.setattr(
        chunk, "iter_process_v2_role_shards", lambda *a, **k: iter((({}, published),))
    )
    source = type("S", (), {"active8_run_root": None, "contracts": None, "index": None})()
    rows = chunk.chunk_transition_rows(source, task_identity_sha256=wanted)
    assert [row["n"] for row in rows] == [1, 3]


def test_the_partition_role_is_passed_to_the_reader_not_filtered_after(monkeypatch) -> None:
    """Sealing depends on the role never being opened, so it must be a reader argument."""

    seen: dict[str, object] = {}

    def _reader(run_root, *, contracts, index, partition_role):
        seen["partition_role"] = partition_role
        return iter(((({}), ({"task_identity_sha256": "e" * 64},)),))

    monkeypatch.setattr(chunk, "iter_process_v2_role_shards", _reader)
    source = type("S", (), {"active8_run_root": None, "contracts": None, "index": None})()
    chunk.chunk_transition_rows(
        source, task_identity_sha256="e" * 64, partition_role="train"
    )
    assert seen["partition_role"] == "train"
