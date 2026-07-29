"""A teacher step must be scoreable by the model that scores it.

The scientific run crashed at step ~250 with

    ValueError: factorized grow supports one existing neighbor

because a corruption trace contained an ``atom_insert`` with TWO existing neighbours -- the inverse of
deleting a bridging atom. ``_teacher_action_score`` routes AtomInsert to ``grow_root`` (no neighbour) or
``grow_connected`` (exactly one); there is no head for two.

Measured incidence: 1 in 12,296 packed corruption records (0.008%). The 200-source in-memory corruption
was 375x too small to contain one, so no earlier run hit it -- and the 200-step benchmark returns before
the first evaluation, so it could not have either.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from compose_v4.data.production_edit_corpus import trace_is_representable  # noqa: E402


@dataclass
class _Action:
    neighbors: tuple | None = None


@dataclass
class _Step:
    action: object


@dataclass
class _Trace:
    steps: tuple


def test_rootless_insert_is_representable():
    """grow_root scores an insert with no existing neighbour."""
    assert trace_is_representable(_Trace((_Step(_Action(neighbors=())),)))


def test_single_neighbor_insert_is_representable():
    """grow_connected scores exactly one existing neighbour."""
    assert trace_is_representable(_Trace((_Step(_Action(neighbors=((5, 1),))),)))


def test_two_neighbor_insert_is_rejected():
    """The exact shape that crashed the run: neighbors [[5,1],[16,1]]."""
    assert not trace_is_representable(_Trace((_Step(_Action(neighbors=((5, 1), (16, 1)))),)))


def test_three_neighbor_insert_is_rejected():
    assert not trace_is_representable(
        _Trace((_Step(_Action(neighbors=((1, 1), (2, 1), (3, 1)))),))
    )


def test_one_bad_step_rejects_the_whole_trace():
    """A trace is drawn as a unit, so a single unscoreable step makes the record unusable."""
    trace = _Trace((
        _Step(_Action(neighbors=((5, 1),))),
        _Step(_Action(neighbors=((5, 1), (16, 1)))),
        _Step(_Action(neighbors=())),
    ))
    assert not trace_is_representable(trace)


def test_actions_without_neighbors_are_unaffected():
    """Deletes, restates and bond ops carry no `neighbors` attribute and must pass through."""
    @dataclass
    class _NoNeighbors:
        v: int = 3

    assert trace_is_representable(_Trace((_Step(_NoNeighbors()),)))


def test_the_real_corruption_shape_is_rejected():
    """Reproduce the exact payload observed in packed corruption shard c_0016."""
    observed = ((5, 1), (16, 1))
    assert len(observed) == 2
    assert not trace_is_representable(_Trace((_Step(_Action(neighbors=observed)),)))
