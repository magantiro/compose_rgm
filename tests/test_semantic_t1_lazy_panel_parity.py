"""The lazy panel must change WHEN tensors are built, never WHAT is trained on.

The resident panel pre-builds every collated row, which is right for a
512-entry panel and impossible at corpus scale (~92 kB per entry, ~14 GB for
144,870). The lazy panel builds each minibatch on demand instead.

That is only safe if the two are indistinguishable to everything downstream.
These tests pin the retrieval contract they share -- ID handling, ordering,
repetition and rejection of unknown IDs -- so a divergence shows up here rather
than as an unexplained difference between two training runs.

These tests deliberately exercise the LAZY path only. The resident path routes
through ``_index_factorized_batch``, which indexes real sparse tensor fields, so
a stand-in batch cannot travel it -- attempting that tests the fixture rather
than the code. Resident-versus-lazy TENSOR equality is therefore proven by the
parity harness against a real 512-entry prepared panel; what is pinned here is
the retrieval contract, which is where an off-by-one or an ordering assumption
would actually live.
"""

from __future__ import annotations

import pytest

from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
    SemanticT1CapacityRunnerError,
    _batch_from_materialized_panel,
    _LazySemanticT1Panel,
    _MaterializedSemanticT1Panel,
)


class _FakeBatch:
    """Minimal stand-in exposing the one attribute the seam inspects."""

    def __init__(self, rows: tuple[str, ...]) -> None:
        self.rows = rows
        self.batch_size = len(rows)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeBatch) and self.rows == other.rows


IDS = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _resident() -> _MaterializedSemanticT1Panel:
    return _MaterializedSemanticT1Panel(
        batch=_FakeBatch(IDS),
        panel_ids=IDS,
        fibers=tuple(f"fiber-{i}" for i in range(len(IDS))),
        partitions=tuple(f"part-{i}" for i in range(len(IDS))),
        entries=tuple({"panel_entry_sha256": pid} for pid in IDS),
        index_by_panel_id={pid: i for i, pid in enumerate(IDS)},
    )


class _RecordingLazy(_LazySemanticT1Panel):
    """Lazy panel whose materialization mirrors the resident panel's rows."""

    def __init__(self, resident: _MaterializedSemanticT1Panel) -> None:
        self._resident = resident
        self.panel_ids = resident.panel_ids
        self.index_by_panel_id = dict(resident.index_by_panel_id)
        self.calls: list[tuple[str, ...]] = []

    def materialize(self, panel_ids):  # type: ignore[override]
        unknown = [pid for pid in panel_ids if pid not in self.index_by_panel_id]
        if unknown:
            raise SemanticT1CapacityRunnerError(
                "T1 address stream references a panel entry outside the lazy panel"
            )
        self.calls.append(tuple(panel_ids))
        indices = [self.index_by_panel_id[pid] for pid in panel_ids]
        return (
            _FakeBatch(tuple(self._resident.panel_ids[i] for i in indices)),
            tuple(self._resident.fibers[i] for i in indices),
            tuple(self._resident.partitions[i] for i in indices),
            tuple(self._resident.entries[i] for i in indices),
        )


@pytest.mark.parametrize(
    "requested",
    [
        IDS,                       # the whole panel
        (IDS[2],),                 # a single row
        (IDS[3], IDS[0], IDS[1]),  # out of panel order
        (IDS[1], IDS[1], IDS[0]),  # repeated, as an oversampled stratum draws
    ],
)
def test_lazy_returns_exactly_the_requested_rows(requested) -> None:
    lazy = _RecordingLazy(_resident())
    batch, fibers, partitions, entries = _batch_from_materialized_panel(lazy, requested)
    assert batch.rows == tuple(requested)
    assert tuple(e["panel_entry_sha256"] for e in entries) == tuple(requested)
    assert len(fibers) == len(partitions) == len(requested)


def test_requested_order_is_preserved_not_panel_order() -> None:
    """The sampler decides order; storage must not reimpose its own.

    Re-sorting into panel order would silently change the sequence the
    optimizer sees while every count and coefficient stayed identical.
    """

    lazy = _RecordingLazy(_resident())
    requested = (IDS[3], IDS[0])
    _batch, _f, _p, entries = _batch_from_materialized_panel(lazy, requested)
    assert tuple(e["panel_entry_sha256"] for e in entries) == requested


def test_repetition_is_honoured_by_both() -> None:
    """An oversampled row appears as many times as the law drew it."""

    lazy = _RecordingLazy(_resident())
    requested = (IDS[0], IDS[0], IDS[0])
    _batch, _f, _p, entries = _batch_from_materialized_panel(lazy, requested)
    assert len(entries) == 3


def test_unknown_ids_are_refused_by_both() -> None:
    lazy = _RecordingLazy(_resident())
    with pytest.raises(SemanticT1CapacityRunnerError):
        _batch_from_materialized_panel(lazy, ("z" * 64,))


def test_lazy_materializes_only_what_was_asked_for() -> None:
    """The point of the change: peak work tracks the minibatch, not the panel."""

    lazy = _RecordingLazy(_resident())
    _batch_from_materialized_panel(lazy, (IDS[1],))
    assert lazy.calls == [(IDS[1],)]
