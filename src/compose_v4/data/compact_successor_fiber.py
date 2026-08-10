"""Compact sufficient statistics for the canonical-successor objective.

    SourceFiber(x)      legal marks and their canonical grouping -- depends on x
    TrainingEntry(x,y)  which group the teacher chose -- depends on the transition

WHY
---
The prepared-input artifact stores a full ``successor_partition`` per training
entry: canonical SMILES per group and two 64-character hex hashes per mark.
MEASURED over 512 real entries, that is a median 223 kB per source and would be
~35.3 GB across the frozen 144,870-entry corpus. None of it is training signal.
The loss needs to know which legal marks share a canonical successor, and which
of those groups the teacher chose; both are integers. The same measurement puts
the compact form at a median 7.7 kB per source -- **29x smaller, ~1.22 GB
corpus-wide** -- which is resident-in-RAM territory and lets the GPU path carry
integer tensors with no RDKit, no SMILES and no enumeration.

WHY THE TEACHER IS NOT IN THE FIBER
-----------------------------------
The legal support and its grouping are functions of the SOURCE. The teacher
successor is a function of the TRANSITION. 29.4% of sources in the corpus carry
more than one successor, so folding the teacher into the fiber would duplicate
the expensive part exactly where sharing pays. Separating them is both smaller
and more honest about what depends on what.

ALIAS AGGREGATION IS LOAD-BEARING, NOT AN EDGE CASE
---------------------------------------------------
A canonical successor is often reachable by several distinct legal marks, and
its probability is the sum over them -- that aggregation IS the canonical
successor likelihood. MEASURED: 13.75% of groups hold more than one mark, up to
18. A representation assuming one mark per group would look right on a sampled
entry and silently corrupt the objective. ``mark_group`` therefore encodes the
many-to-one relation explicitly and is never assumed injective.

DTYPES ARE MEASURED, NOT GUESSED
--------------------------------
From the same 512-entry census: 8 families, 8 tables, coordinates in [0, 39]
(structurally bounded by ``max_atoms``), at most 2,097 marks and 1,828 groups
per source. ``mark_group`` is kept ``int32`` even though the observed maximum
fits ``int16``: that bound is sample-derived rather than structural, and ~2 MB
corpus-wide is not worth an overflow risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

#: Coordinates are padded to this arity. Measured arities are 1, 2 and 3.
COORDINATE_ARITY = 3
#: Fills unused coordinate slots. Distinguishable from a real coordinate,
#: which the census bounds to [0, max_atoms - 1].
COORDINATE_PAD = -1


class CompactFiberError(ValueError):
    """A compact fiber does not faithfully represent its partition."""


@dataclass(frozen=True)
class SourceFiber:
    """Legal marks of one source and their canonical-successor grouping.

    ``mark_group[i]`` is the canonical successor group of mark ``i``. Marks are
    stored in the ORDER THEY APPEAR in the partition, which is exactly the order
    the scorer consumes: it iterates ``CanonicalSuccessorAliasGroup.aliases``,
    a property returning ``sorted(mark.alias for mark in marks)``, and stored
    order already equals that because ``CompiledSuccessorMark`` is ``order=True``
    with ``alias`` as its first field while the group's ``__post_init__``
    enforces sorted-unique marks and unique aliases. VERIFIED over 306,323 real
    groups (42,124 multi-alias): zero disagreements. The coupling is pinned by
    ``test_stored_order_is_the_order_the_scorer_consumes`` -- reordering those
    dataclass fields would silently change every training operand sequence.
    """

    mark_family: np.ndarray      # uint8   [N]
    mark_table: np.ndarray       # uint8   [N]
    mark_coords: np.ndarray      # int16   [N, COORDINATE_ARITY]
    mark_group: np.ndarray       # int32   [N]
    group_count: int

    def __post_init__(self) -> None:
        n = int(self.mark_family.shape[0])
        if not (
            self.mark_table.shape == (n,)
            and self.mark_group.shape == (n,)
            and self.mark_coords.shape == (n, COORDINATE_ARITY)
        ):
            raise CompactFiberError("compact fiber arrays disagree on mark count")
        if n and (int(self.mark_group.min()) < 0
                  or int(self.mark_group.max()) >= self.group_count):
            raise CompactFiberError("mark_group references a group outside the fiber")

    @property
    def mark_count(self) -> int:
        return int(self.mark_family.shape[0])

    def group_sizes(self) -> np.ndarray:
        """Marks per group -- the alias multiplicities the loss sums over."""

        return np.bincount(self.mark_group, minlength=self.group_count).astype(np.int32)

    def nbytes(self) -> int:
        return int(
            self.mark_family.nbytes
            + self.mark_table.nbytes
            + self.mark_coords.nbytes
            + self.mark_group.nbytes
        )


@dataclass(frozen=True)
class TrainingEntry:
    """One (x, y) transition: a source fiber plus the teacher's group.

    Several entries may name the SAME ``source_fiber_id`` with different
    ``teacher_group`` values. That is the multi-successor case, and it is the
    reason the teacher lives here rather than in the fiber.
    """

    entry_id: str
    source_fiber_id: str
    teacher_group: int
    family: str
    capability_cell: str


def build_source_fiber(
    partition: Mapping[str, Any],
    *,
    family_ids: Mapping[str, int],
    table_ids: Mapping[str, int],
) -> tuple[SourceFiber, list[str]]:
    """Compact one ``successor_partition``; also return its group target keys.

    The target keys are returned SEPARATELY rather than stored in the fiber.
    They are provenance -- needed to resolve which group a teacher chose, and
    to verify parity -- but they are strings and must never reach a training
    batch.
    """

    groups: Sequence[Mapping[str, Any]] = partition["successor_groups"]
    families: list[int] = []
    tables: list[int] = []
    coords: list[list[int]] = []
    assignments: list[int] = []
    target_keys: list[str] = []

    for group_index, group in enumerate(groups):
        target_keys.append(str(group["target_key"]))
        for mark in group["marks"]:
            alias = mark["alias"]
            family = str(alias["family_name"])
            table = str(alias["table_name"])
            if family not in family_ids or table not in table_ids:
                raise CompactFiberError(
                    f"mark names family {family!r} / table {table!r} outside the "
                    "declared vocabularies; refusing to invent an id"
                )
            coordinate = [int(value) for value in alias["coordinate"]]
            if len(coordinate) > COORDINATE_ARITY:
                raise CompactFiberError(
                    f"coordinate arity {len(coordinate)} exceeds {COORDINATE_ARITY}"
                )
            coordinate = coordinate + [COORDINATE_PAD] * (COORDINATE_ARITY - len(coordinate))
            families.append(family_ids[family])
            tables.append(table_ids[table])
            coords.append(coordinate)
            # Many marks may carry the same group index. That is alias
            # aggregation and it is expected, not a collision.
            assignments.append(group_index)

    fiber = SourceFiber(
        mark_family=np.asarray(families, dtype=np.uint8),
        mark_table=np.asarray(tables, dtype=np.uint8),
        mark_coords=np.asarray(coords, dtype=np.int16).reshape(-1, COORDINATE_ARITY),
        mark_group=np.asarray(assignments, dtype=np.int32),
        group_count=len(groups),
    )
    return fiber, target_keys


def verify_against_partition(
    fiber: SourceFiber,
    target_keys: Sequence[str],
    partition: Mapping[str, Any],
    *,
    family_ids: Mapping[str, int],
    table_ids: Mapping[str, int],
) -> None:
    """Raise unless the fiber reproduces the partition exactly.

    Checks order, not just membership: a support that contains the right marks
    in the wrong order is a different input to the scorer.
    """

    position = 0
    for group_index, group in enumerate(partition["successor_groups"]):
        if str(group["target_key"]) != target_keys[group_index]:
            raise CompactFiberError(f"group {group_index} target key disagrees")
        for mark in group["marks"]:
            alias = mark["alias"]
            expected = [int(value) for value in alias["coordinate"]]
            expected = expected + [COORDINATE_PAD] * (COORDINATE_ARITY - len(expected))
            if (
                int(fiber.mark_family[position]) != family_ids[str(alias["family_name"])]
                or int(fiber.mark_table[position]) != table_ids[str(alias["table_name"])]
                or [int(v) for v in fiber.mark_coords[position]] != expected
                or int(fiber.mark_group[position]) != group_index
            ):
                raise CompactFiberError(
                    f"compact mark {position} disagrees with the partition"
                )
            position += 1
    if position != fiber.mark_count:
        raise CompactFiberError(
            f"compact fiber holds {fiber.mark_count} marks, partition has {position}"
        )
