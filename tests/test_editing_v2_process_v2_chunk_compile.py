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


# ---- Observability and the deadline ----


def _seamed(monkeypatch, *, entry_count: int, seconds_per_batch: float = 0.0):
    """Drive the compile loop with fake entries, keeping the loop itself real.

    Only the resolver and the per-entry compiler are replaced: the sub-batching,
    the rate projection and the deadline decision under test are the production
    ones.
    """

    import time as _time

    rows = [
        {
            "p50_entry_sha256": f"{index:064d}",
            "model_family": "atom_delete",
            "capability_cell_id": "cell",
            "progress_index": 0,
            "source_state_sha256": "a" * 64,
            "target_state_sha256": "b" * 64,
            "exact_state": {"index": index},
            "teacher_successor_fiber": {},
        }
        for index in range(entry_count)
    ]
    monkeypatch.setattr(chunk, "chunk_transition_rows", lambda *a, **k: tuple(rows))
    monkeypatch.setattr(chunk, "_candidate_from_transition", dict)

    class _Resolved:
        def __init__(self, entry):
            self.panel_entries = [entry]
            self.addressed_trace = self

        @property
        def path(self):
            return self

        def state_at(self, index):
            return object()

    monkeypatch.setattr(
        chunk,
        "resolve_process_v2_t1_entries",
        lambda source, entries: [_Resolved(entry) for entry in entries],
    )
    digests = iter(["a" * 64, "b" * 64] * (entry_count + 1))
    monkeypatch.setattr(chunk, "persistent_slot_state_sha256", lambda state: next(digests))
    monkeypatch.setattr(chunk, "decode_state", lambda payload: payload)
    monkeypatch.setattr(chunk, "FactorizedMarkExample", lambda **kw: kw)

    def _compile(model, sources, targets, entries, **kwargs):
        if seconds_per_batch:
            _time.sleep(seconds_per_batch)
        return [dict(entry) for entry in entries]

    monkeypatch.setattr(chunk, "compile_prepared_entries", _compile)

    class _Collator:
        _chemistry_feature_cache: dict = {}

        def __call__(self, examples):
            return type("B", (), {"batch_size": len(examples)})()

    monkeypatch.setattr(
        chunk.FactorizedMarkCollator, "from_capabilities", classmethod(lambda cls, *a, **k: _Collator())
    )
    return (
        type("S", (), {"active8_run_root": None, "contracts": None, "index": None})(),
        type("M", (), {"operator_capabilities": None, "ring_catalog": None})(),
    )


def test_the_compile_reports_progress_while_it_runs_not_only_when_it_finishes(monkeypatch):
    """A worker that reports only on completion says nothing as it dies.

    The expensive chemistry sits inside ONE batched call, so a whole-slice
    compile emitted no progress at all -- which is why a slice could run an
    entire 45-minute wall and be killed with no rate ever observed.
    """

    source, model = _seamed(monkeypatch, entry_count=20)
    events: list[dict] = []
    result = chunk.compile_process_v2_chunk_shard(
        source,
        model,
        task_identity_sha256="a" * 64,
        compile_batch_size=5,
        progress_callback=events.append,
    )

    assert result["entry_count"] == 20
    assert result["stopped_on_deadline"] is False
    assert result["unfinished_entry_count"] == 0
    assert len(events) == 4
    assert [event["completed"] for event in events] == [5, 10, 15, 20]
    for event in events:
        assert event["total"] == 20
        # The rate and the projection are what make a slow slice actionable
        # before the wall rather than after it.
        assert event["ms_per_entry"] >= 0.0
        assert "projected_remaining_seconds" in event


def test_a_deadline_publishes_the_finished_prefix_and_names_the_remainder(monkeypatch):
    """Degrading to less work beats losing all of it.

    Slices are offset-addressed, so a short slice is a valid published slice.
    The remainder is returned as ``next_offset``/``unfinished_entry_count`` so a
    re-run is a bounded top-up rather than a repeat of the whole slice.
    """

    source, model = _seamed(monkeypatch, entry_count=120, seconds_per_batch=0.05)
    result = chunk.compile_process_v2_chunk_shard(
        source,
        model,
        task_identity_sha256="a" * 64,
        offset=100,
        limit=20,
        compile_batch_size=5,
        deadline_seconds=0.12,
    )

    assert result["stopped_on_deadline"] is True
    assert 0 < result["entry_count"] < 20
    assert result["requested_entry_count"] == 20
    assert result["entry_count"] + result["unfinished_entry_count"] == 20
    # Addressed from the slice's own offset, so the remainder is re-queueable.
    assert result["next_offset"] == 100 + result["entry_count"]


def test_the_deadline_is_precise_to_one_sub_batch_and_says_so_when_it_overruns(monkeypatch):
    """The first sub-batch always runs -- there is no rate to project without it.

    So the deadline is precise to one sub-batch, not to the second. That is a
    real limit, and an overrun is REPORTED rather than left to be inferred from
    a kill, which is the whole reason the original timeout was uninformative.
    """

    source, model = _seamed(monkeypatch, entry_count=20, seconds_per_batch=0.2)
    events: list[dict] = []
    result = chunk.compile_process_v2_chunk_shard(
        source,
        model,
        task_identity_sha256="a" * 64,
        compile_batch_size=20,
        deadline_seconds=0.05,
        progress_callback=events.append,
    )

    # One sub-batch ran and blew the deadline; it is published, and flagged.
    assert result["entry_count"] == 20
    assert result["unfinished_entry_count"] == 0
    assert events and events[-1]["overran_deadline"] is True
    assert events[-1]["elapsed_seconds"] > 0.05


def test_sub_batching_does_not_change_the_published_entry_order(monkeypatch):
    """Concatenated sub-batches must sort exactly as the one-shot path did."""

    outputs = []
    for batch_size in (1, 3, 20):
        source, model = _seamed(monkeypatch, entry_count=20)
        outputs.append(
            [
                row["p50_entry_sha256"]
                for row in chunk.compile_process_v2_chunk_shard(
                    source,
                    model,
                    task_identity_sha256="a" * 64,
                    compile_batch_size=batch_size,
                )["entries"]
            ]
        )
    assert outputs[0] == outputs[1] == outputs[2] == sorted(outputs[0])


@pytest.mark.parametrize("bad", [0, -1, 1.0, "5"])
def test_a_nonsense_compile_batch_size_is_refused(bad):
    with pytest.raises(chunk.ProcessV2ChunkCompileError, match="positive int"):
        chunk.compile_process_v2_chunk_shard(
            object(), object(), task_identity_sha256="a" * 64, compile_batch_size=bad
        )
