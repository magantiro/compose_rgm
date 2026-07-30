from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)


def _state(*, dtype=np.int32) -> MolecularGraph:
    return MolecularGraph(
        atom_types=np.asarray([1, 3, 0, 0], dtype=dtype),
        formal_charges=np.asarray([0, 0, 0, 0], dtype=dtype),
        implicit_h_counts=np.asarray([3, 1, 0, 0], dtype=dtype),
        bonds=np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=dtype,
        ),
    )


def test_digest_is_deterministic_and_integer_dtype_independent() -> None:
    original = _state(dtype=np.int32)
    copied = _state(dtype=np.int64)

    assert persistent_slot_state_sha256(original) == persistent_slot_state_sha256(
        copied
    )
    assert len(persistent_slot_state_sha256(original)) == 64


@pytest.mark.parametrize(
    ("field", "index", "replacement"),
    (
        ("atom_types", (1,), 4),
        ("formal_charges", (1,), 1),
        ("implicit_h_counts", (0,), 2),
        ("bonds", (0, 1), 2),
    ),
)
def test_digest_changes_for_every_semantic_state_component(
    field: str,
    index: tuple[int, ...],
    replacement: int,
) -> None:
    original = _state()
    changed = _state()
    array = getattr(changed, field)
    array[index] = replacement
    if field == "bonds":
        array[index[::-1]] = replacement

    assert persistent_slot_state_sha256(original) != persistent_slot_state_sha256(
        changed
    )


def test_digest_preserves_slot_identity_instead_of_quotienting_relabeling() -> None:
    original = _state()
    permuted = _state()
    permutation = np.asarray([1, 0, 2, 3])
    permuted.atom_types = permuted.atom_types[permutation]
    permuted.formal_charges = permuted.formal_charges[permutation]
    permuted.implicit_h_counts = permuted.implicit_h_counts[permutation]
    permuted.bonds = permuted.bonds[np.ix_(permutation, permutation)]

    assert persistent_slot_state_sha256(original) != persistent_slot_state_sha256(
        permuted
    )


def test_digest_rejects_noninteger_or_out_of_range_arrays() -> None:
    floating = _state()
    floating.atom_types = floating.atom_types.astype(np.float64)
    with pytest.raises(TypeError, match="integer-valued"):
        persistent_slot_state_sha256(floating)

    oversized = _state(dtype=np.int64)
    oversized.atom_types[0] = 1 << 40
    with pytest.raises(ValueError, match="outside signed int32"):
        persistent_slot_state_sha256(oversized)
