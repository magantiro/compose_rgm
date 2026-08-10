"""The compact fiber must preserve the sufficient statistics of the loss.

Two properties carry the whole representation change:

* **alias aggregation** -- a canonical successor reachable by several marks has
  probability equal to their sum, and MEASURED 13.75% of real groups hold more
  than one mark (up to 18). A schema that assumed one mark per group would look
  correct on most sampled entries and silently corrupt the objective.
* **mark order** -- the scorer associates rows with mark operands positionally,
  so a support holding the right marks in the wrong order is a different input,
  even where the mathematics is permutation invariant after aggregation.

The one-source-two-teachers test exists because the real 512-entry panel has
512 DISTINCT sources and therefore cannot exercise sharing at all -- yet
sharing is the entire reason the teacher was moved out of the fiber.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.data.compact_successor_fiber import (
    COORDINATE_PAD,
    CompactFiberError,
    SourceFiber,
    TrainingEntry,
    build_source_fiber,
    verify_against_partition,
)

FAMILY_IDS = {"atom_insert": 0, "atom_restate": 1, "bond_reroute": 2}
TABLE_IDS = {"grow_connected": 0, "restate": 1, "reroute": 2}


def _mark(family: str, table: str, coordinate: list[int]) -> dict:
    return {
        "action_sha256": "a" * 64,
        "alias": {"coordinate": coordinate, "family_name": family, "table_name": table},
        "successor_state_sha256": "b" * 64,
    }


def _partition(groups: list[tuple[str, list[dict]]]) -> dict:
    return {
        "source_key": "CCO",
        "source_state_sha256": "c" * 64,
        "successor_groups": [
            {"target_key": key, "marks": marks} for key, marks in groups
        ],
        "virtual_marks": [],
    }


def _build(partition):
    return build_source_fiber(partition, family_ids=FAMILY_IDS, table_ids=TABLE_IDS)


def test_many_marks_collapse_into_one_group() -> None:
    """Alias aggregation: three marks, one canonical successor."""

    partition = _partition([
        ("CCO", [
            _mark("atom_insert", "grow_connected", [2, 2, 14]),
            _mark("atom_restate", "restate", [0, 9]),
            _mark("bond_reroute", "reroute", [1]),
        ]),
    ])
    fiber, keys = _build(partition)
    assert fiber.mark_count == 3
    assert fiber.group_count == 1
    assert fiber.mark_group.tolist() == [0, 0, 0]
    assert fiber.group_sizes().tolist() == [3]
    assert keys == ["CCO"]


def test_group_sizes_are_the_alias_multiplicities() -> None:
    partition = _partition([
        ("A", [_mark("atom_insert", "grow_connected", [1, 2, 3])] * 4),
        ("B", [_mark("atom_restate", "restate", [0, 1])]),
        ("C", [_mark("bond_reroute", "reroute", [5])] * 2),
    ])
    fiber, _keys = _build(partition)
    assert fiber.group_sizes().tolist() == [4, 1, 2]
    assert fiber.mark_group.tolist() == [0, 0, 0, 0, 1, 2, 2]


def test_mark_order_is_preserved_exactly() -> None:
    """Order is part of the input, not an incidental detail."""

    partition = _partition([
        ("A", [_mark("bond_reroute", "reroute", [7])]),
        ("B", [_mark("atom_insert", "grow_connected", [1, 2, 3])]),
        ("C", [_mark("atom_restate", "restate", [4, 5])]),
    ])
    fiber, _keys = _build(partition)
    assert fiber.mark_family.tolist() == [2, 0, 1]
    assert fiber.mark_coords.tolist() == [
        [7, COORDINATE_PAD, COORDINATE_PAD],
        [1, 2, 3],
        [4, 5, COORDINATE_PAD],
    ]


def test_short_coordinates_are_padded_not_dropped() -> None:
    fiber, _keys = _build(_partition([("A", [_mark("bond_reroute", "reroute", [9])])]))
    assert fiber.mark_coords.tolist() == [[9, COORDINATE_PAD, COORDINATE_PAD]]
    assert fiber.mark_coords.dtype == np.int16


def test_declared_dtypes_hold() -> None:
    fiber, _keys = _build(_partition([
        ("A", [_mark("atom_insert", "grow_connected", [39, 39, 39])]),
    ]))
    assert fiber.mark_family.dtype == np.uint8
    assert fiber.mark_table.dtype == np.uint8
    assert fiber.mark_coords.dtype == np.int16
    assert fiber.mark_group.dtype == np.int32


def test_one_fiber_serves_two_different_teacher_groups() -> None:
    """The multi-successor case the 512-entry panel cannot exercise.

    29.4% of corpus sources carry more than one successor. The fiber depends on
    x alone, so both transitions must share it while resolving to DIFFERENT
    teacher groups -- which is the whole reason the teacher was moved out.
    """

    partition = _partition([
        ("PRODUCT_ONE", [_mark("atom_insert", "grow_connected", [1, 2, 3]),
                         _mark("atom_insert", "grow_connected", [3, 2, 1])]),
        ("PRODUCT_TWO", [_mark("atom_restate", "restate", [0, 4])]),
    ])
    fiber, keys = _build(partition)

    first = TrainingEntry("e1", "src", keys.index("PRODUCT_ONE"), "atom_insert", "cell")
    second = TrainingEntry("e2", "src", keys.index("PRODUCT_TWO"), "atom_restate", "cell")

    assert first.source_fiber_id == second.source_fiber_id
    assert first.teacher_group != second.teacher_group
    # The shared fiber still reports the aggregation each teacher needs.
    assert fiber.group_sizes()[first.teacher_group] == 2
    assert fiber.group_sizes()[second.teacher_group] == 1


def test_verification_accepts_a_faithful_fiber() -> None:
    partition = _partition([
        ("A", [_mark("atom_insert", "grow_connected", [1, 2, 3]),
               _mark("atom_restate", "restate", [4, 5])]),
        ("B", [_mark("bond_reroute", "reroute", [6])]),
    ])
    fiber, keys = _build(partition)
    verify_against_partition(
        fiber, keys, partition, family_ids=FAMILY_IDS, table_ids=TABLE_IDS
    )


def test_verification_rejects_a_reordered_fiber() -> None:
    """Reordering preserves membership and multiplicities -- and is still wrong."""

    partition = _partition([
        ("A", [_mark("atom_insert", "grow_connected", [1, 2, 3])]),
        ("B", [_mark("bond_reroute", "reroute", [6])]),
    ])
    fiber, keys = _build(partition)
    swapped = SourceFiber(
        mark_family=fiber.mark_family[::-1].copy(),
        mark_table=fiber.mark_table[::-1].copy(),
        mark_coords=fiber.mark_coords[::-1].copy(),
        mark_group=fiber.mark_group,
        group_count=fiber.group_count,
    )
    with pytest.raises(CompactFiberError, match="disagrees with the partition"):
        verify_against_partition(
            swapped, keys, partition, family_ids=FAMILY_IDS, table_ids=TABLE_IDS
        )


def test_unknown_family_is_refused_rather_than_assigned_an_id() -> None:
    """Inventing an id would train a real mark under a wrong operand type."""

    partition = _partition([("A", [_mark("cycle_insert", "reroute", [1])])])
    with pytest.raises(CompactFiberError, match="outside the declared vocabularies"):
        _build(partition)


def test_group_reference_outside_the_fiber_is_refused() -> None:
    with pytest.raises(CompactFiberError, match="outside the fiber"):
        SourceFiber(
            mark_family=np.zeros(2, dtype=np.uint8),
            mark_table=np.zeros(2, dtype=np.uint8),
            mark_coords=np.zeros((2, 3), dtype=np.int16),
            mark_group=np.asarray([0, 5], dtype=np.int32),
            group_count=1,
        )


def test_compact_form_is_far_smaller_than_the_partition() -> None:
    """The reason the representation exists at all."""

    marks = [_mark("atom_insert", "grow_connected", [i % 40, 2, 3]) for i in range(600)]
    partition = _partition([(f"KEY{i}", marks[i: i + 1]) for i in range(600)])
    fiber, _keys = _build(partition)
    assert fiber.nbytes() < 600 * 24
