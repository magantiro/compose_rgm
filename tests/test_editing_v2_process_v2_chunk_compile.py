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


def test_the_chunk_and_role_are_passed_to_the_reader_not_filtered_after(monkeypatch) -> None:
    """Both selectors must reach the reader.

    The role matters because sealing depends on a held-out shard never being
    opened.  The task matters because a per-chunk fan-out that filters after
    reading opens every shard of the role in every worker -- O(n^2) I/O that a
    single-chunk test cannot reveal.
    """

    seen: dict[str, object] = {}
    wanted = "c" * 64

    def _reader(run_root, *, contracts, index, partition_role, task_identity_sha256=None):
        seen["partition_role"] = partition_role
        seen["task_identity_sha256"] = task_identity_sha256
        return iter((({}, ({"task_identity_sha256": wanted, "n": 1},)),))

    monkeypatch.setattr(chunk, "iter_process_v2_role_shards", _reader)
    source = type("S", (), {"active8_run_root": None, "contracts": None, "index": None})()
    rows = chunk.chunk_transition_rows(source, task_identity_sha256=wanted)

    assert seen["partition_role"] == "train"
    assert seen["task_identity_sha256"] == wanted
    assert [row["n"] for row in rows] == [1]


def test_a_slice_offset_must_be_a_nonnegative_integer() -> None:
    for bad in (-1, 1.0, "0", None):
        with pytest.raises(chunk.ProcessV2ChunkCompileError, match="nonnegative int"):
            chunk.compile_process_v2_chunk_shard(
                object(), object(), task_identity_sha256="a" * 64, offset=bad
            )


def test_an_empty_slice_refuses_rather_than_publishing_nothing(monkeypatch) -> None:
    """A stride past the end of a chunk is a planning error, not an empty success."""

    rows = ({"task_identity_sha256": "b" * 64},) * 4
    monkeypatch.setattr(
        chunk, "iter_process_v2_role_shards", lambda *a, **k: iter((({}, rows),))
    )
    with pytest.raises(chunk.ProcessV2ChunkCompileError, match="selects no transitions"):
        chunk.compile_process_v2_chunk_shard(
            type("S", (), {"active8_run_root": None, "contracts": None, "index": None})(),
            object(),
            task_identity_sha256="b" * 64,
            offset=99,
        )
