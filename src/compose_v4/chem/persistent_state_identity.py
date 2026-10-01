"""Stable identity for an exact persistent-slot molecular state.

Canonical SMILES deliberately quotients atom ordering. Training caches need the
opposite identity: slot coordinates, null/scar occupancy, charges, hydrogens,
and every bond entry must remain distinguishable. This module hashes that exact
integer state without invoking RDKit.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph

PERSISTENT_STATE_DIGEST_SCHEMA = "compose.chem.persistent_slot_state"
PERSISTENT_STATE_DIGEST_VERSION = 1

_FIELDS = (
    ("atom_types", 1),
    ("formal_charges", 1),
    ("implicit_h_counts", 1),
    ("bonds", 2),
)
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def _normalized_integer_array(
    value: object,
    *,
    field: str,
    rank: int,
    n_slots: int,
) -> np.ndarray:
    array = np.asarray(value)
    expected_shape = (n_slots,) if rank == 1 else (n_slots, n_slots)
    if array.shape != expected_shape:
        raise ValueError(
            f"persistent state field {field!r} has shape {array.shape}, "
            f"expected {expected_shape}"
        )
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"persistent state field {field!r} must be integer-valued")
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < _INT32_MIN or maximum > _INT32_MAX:
            raise ValueError(
                f"persistent state field {field!r} lies outside signed int32"
            )
    return np.ascontiguousarray(array, dtype=np.dtype("<i4"))


def persistent_slot_state_sha256(state: MolecularGraph) -> str:
    """Hash one exact padded state under a platform-independent byte contract."""

    if not isinstance(state, MolecularGraph):
        raise TypeError("persistent state identity requires a MolecularGraph")
    n_slots = int(state.n_atoms)
    if n_slots < 0:
        raise ValueError("persistent state slot count cannot be negative")

    digest = hashlib.sha256()
    digest.update(PERSISTENT_STATE_DIGEST_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack("<I", PERSISTENT_STATE_DIGEST_VERSION))
    digest.update(struct.pack("<Q", n_slots))
    for field, rank in _FIELDS:
        normalized = _normalized_integer_array(
            getattr(state, field),
            field=field,
            rank=rank,
            n_slots=n_slots,
        )
        encoded_name = field.encode("ascii")
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<I", normalized.ndim))
        for dimension in normalized.shape:
            digest.update(struct.pack("<Q", int(dimension)))
        raw = normalized.tobytes(order="C")
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


__all__ = [
    "PERSISTENT_STATE_DIGEST_SCHEMA",
    "PERSISTENT_STATE_DIGEST_VERSION",
    "persistent_slot_state_sha256",
]
