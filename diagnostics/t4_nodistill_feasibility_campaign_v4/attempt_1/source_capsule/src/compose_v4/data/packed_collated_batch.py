"""Row-addressable, losslessly narrowed storage for compiled collated batches.

WHY
---
Collation is not chemistry-free. MEASURED on a fresh 32-example batch:

    collate          98.17 s   (97.2%)
    forward + loss    0.24 s
    backward          2.14 s

Nearly all of it is RDKit: 9,310 ``is_rdkit_valid`` and 9,557
``molecular_graph_to_smiles`` calls per 32 molecules, because collation runs the
admission masks over every candidate successor. Re-collating per batch would
re-pay the entire corpus chemistry every epoch -- roughly 107 hours per epoch at
batch 32 -- and none of it moves to a GPU. The compile step already paid this
cost once and stored the result in ``BATCH.pt``; the training path must read it,
not rebuild it.

WHY NOT JUST READ ``BATCH.pt``
------------------------------
It is slice-addressed. A shuffled 32-example minibatch touches ~32 distinct
slices, and each slice file is ~22 MB, so a naive read is ~700 MB per batch --
worse than re-collating. Training needs ROW addressing.

WHY THE RECORDS ARE FIXED SIZE
------------------------------
Every collated field has a fixed per-example shape because ``max_atoms`` is 40:
``(40, 40)`` matrices, ``(40, 40, 3)`` and ``(40, 15)`` masks, ``(40,)`` vectors
and scalars. So a record is a constant number of bytes and row ``i`` lives at
``i * record_bytes``. The store is one flat file, memory-mapped, and a minibatch
is a gather of fixed-size slices.

WHY NARROWING IS FREE
---------------------
MEASURED over the compiled corpus: six ``(40, 40)`` matrices are stored as
``int64`` while holding values in [-1, 313], and eight masks are stored as
``bool`` at one byte per bit. That is 87.85 kB per example of which most is
representation, not information:

    int fields    76.25 kB  ->  11.09 kB   narrowed per measured range
    bool masks    11.60 kB  ->   1.45 kB   bit-packed
    TOTAL         87.85 kB  ->  12.54 kB   (7.0x, 12.66 GB -> 1.81 GB)

Narrowing is chosen from a full-corpus scan and recorded in the header rather
than hardcoded, and the packer asserts every written slice unpacks
``torch.equal`` to its source. Correctness therefore does not depend on the
range scan being right -- a value that does not fit fails loudly at pack time
instead of training as a wrong number.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

#: Fields carrying per-example tensors, in a fixed order. Anything absent here
#: is either a constant of the compile configuration or reconstructible, and is
#: carried in the header instead.
INT_FIELDS = (
    "atom_types",
    "formal_charges",
    "implicit_h_counts",
    "atom_topology",
    "bonds",
    "neural_bonds",
    "ring_system_topology",
    "closure_topology",
    "graft_successor_groups",
    "graft_remove_neighbors",
)
BOOL_FIELDS = (
    "atom_delete_mask",
    "atom_delete_admission_mask",
    "atom_restate_admission_mask",
    "cycle_edge_mask",
    "cycle_open_admission_mask",
    "cyclic_pair_mask",
    "graft_mask",
    "cycle_close_admission_mask",
)
FLOAT_FIELDS = ("times", "teacher_rates", "importance_weights")

#: Per-row fields that are Python objects rather than tensors. These are NOT
#: compile-time constants -- MEASURED, ``ring_restate_actions`` is populated on
#: every row (~402 B) and the descriptors on every row (~241 B) -- so taking
#: them from a template makes the model raise "semantic ring-restatement
#: metadata does not align with the batch". They are stored per row.
#:
#: ``states`` is deliberately absent: at ~7 kB/row it would dominate the store,
#: and it is reconstructed exactly by ``decode_state(entry.exact_state)``,
#: which the corpus loader already does in 19 us.
#:
#: The four currently-empty fields are stored anyway rather than assumed empty
#: corpus-wide, because "it was empty in the slice I looked at" is not a
#: property of the corpus.
META_FIELDS = (
    "teacher_actions",
    "teacher_rule_names",
    "ring_restate_actions",
    "ring_restate_successor_group_ids",
    "ring_restate_successor_group_descriptors",
    "ring_restate_successor_group_multiplicities",
    "ring_delete_actions",
    "ring_teacher_semantic_certificates",
)

SCHEMA = "compose.editing_v2.packed_collated_batch"
SCHEMA_VERSION = 2


class PackedCollatedBatchError(RuntimeError):
    """A packed collated store is malformed or does not round-trip."""


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    kind: str                    # "int" | "bool" | "float"
    shape: tuple[int, ...]       # per example
    stored_dtype: str            # numpy dtype used on disk
    original_dtype: str          # torch dtype to restore
    offset: int                  # byte offset within a record
    nbytes: int                  # bytes within a record

    @property
    def count(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


@dataclass(frozen=True, slots=True)
class PackedLayout:
    fields: tuple[FieldSpec, ...]
    record_bytes: int

    def field(self, name: str) -> FieldSpec:
        for spec in self.fields:
            if spec.name == name:
                return spec
        raise PackedCollatedBatchError(f"packed layout has no field {name!r}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "record_bytes": self.record_bytes,
            "fields": [dataclasses.asdict(spec) for spec in self.fields],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PackedLayout":
        fields = tuple(
            FieldSpec(
                name=str(item["name"]),
                kind=str(item["kind"]),
                shape=tuple(int(v) for v in item["shape"]),
                stored_dtype=str(item["stored_dtype"]),
                original_dtype=str(item["original_dtype"]),
                offset=int(item["offset"]),
                nbytes=int(item["nbytes"]),
            )
            for item in payload["fields"]
        )
        return cls(fields=fields, record_bytes=int(payload["record_bytes"]))


def _narrow_int_dtype(minimum: int, maximum: int) -> str:
    for name in ("int8", "int16", "int32"):
        info = np.iinfo(name)
        if info.min <= minimum and maximum <= info.max:
            return name
    return "int64"


def plan_layout(
    template: Any,
    *,
    int_ranges: Mapping[str, tuple[int, int]],
) -> PackedLayout:
    """Fix the record layout from a template batch and measured value ranges."""

    specs: list[FieldSpec] = []
    offset = 0
    for name in INT_FIELDS:
        tensor = getattr(template, name)
        shape = tuple(int(v) for v in tensor.shape[1:])
        if name not in int_ranges:
            raise PackedCollatedBatchError(f"no measured range for int field {name!r}")
        low, high = int_ranges[name][0], int_ranges[name][1]
        stored = _narrow_int_dtype(low, high)
        nbytes = int(np.prod(shape)) * np.dtype(stored).itemsize
        specs.append(FieldSpec(name, "int", shape, stored, str(tensor.dtype), offset, nbytes))
        offset += nbytes
    for name in BOOL_FIELDS:
        tensor = getattr(template, name)
        shape = tuple(int(v) for v in tensor.shape[1:])
        # Bit-packed: ceil(count / 8) bytes.
        nbytes = (int(np.prod(shape)) + 7) // 8
        specs.append(FieldSpec(name, "bool", shape, "uint8", str(tensor.dtype), offset, nbytes))
        offset += nbytes
    for name in FLOAT_FIELDS:
        tensor = getattr(template, name)
        shape = tuple(int(v) for v in tensor.shape[1:])
        nbytes = max(1, int(np.prod(shape))) * 4
        specs.append(FieldSpec(name, "float", shape, "float32", str(tensor.dtype), offset, nbytes))
        offset += nbytes
    return PackedLayout(fields=tuple(specs), record_bytes=offset)


def pack_slice(batch: Any, layout: PackedLayout) -> np.ndarray:
    """Pack every example of one collated batch into fixed-size records."""

    n = int(batch.batch_size)
    out = np.zeros((n, layout.record_bytes), dtype=np.uint8)
    for spec in layout.fields:
        tensor = getattr(batch, spec.name)
        block = out[:, spec.offset : spec.offset + spec.nbytes]
        if spec.kind == "bool":
            flat = tensor.reshape(n, -1).numpy()
            block[:, :] = np.packbits(flat, axis=1)
        elif spec.kind == "int":
            values = tensor.reshape(n, -1).numpy()
            narrowed = values.astype(spec.stored_dtype, casting="unsafe")
            if not np.array_equal(narrowed.astype(np.int64), values.astype(np.int64)):
                raise PackedCollatedBatchError(
                    f"field {spec.name!r} does not fit {spec.stored_dtype} without loss"
                )
            block[:, :] = narrowed.view(np.uint8).reshape(n, -1)
        else:
            values = tensor.reshape(n, -1).numpy().astype(np.float32)
            block[:, :] = values.view(np.uint8).reshape(n, -1)
    return out


def unpack_rows(
    records: np.ndarray,
    layout: PackedLayout,
    template: Any,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Any:
    """Rebuild a collated batch from packed records, exactly.

    ``records`` is ``(rows, record_bytes)`` uint8. ``template`` supplies the
    non-tensor configuration fields, which are constants of the compile.
    """

    rows = int(records.shape[0])
    if records.shape[1] != layout.record_bytes:
        raise PackedCollatedBatchError("packed records disagree with the layout width")
    values: dict[str, Any] = {}
    for spec in layout.fields:
        block = records[:, spec.offset : spec.offset + spec.nbytes]
        if spec.kind == "bool":
            bits = np.unpackbits(block, axis=1)[:, : spec.count]
            restored = torch.from_numpy(
                np.ascontiguousarray(bits.reshape((rows, *spec.shape)))
            ).to(torch.bool)
        elif spec.kind == "int":
            flat = np.ascontiguousarray(block).view(spec.stored_dtype)
            restored = torch.from_numpy(
                flat.reshape((rows, *spec.shape)).astype(np.int64)
            )
        else:
            flat = np.ascontiguousarray(block).view(np.float32)
            target = spec.shape if spec.shape else ()
            restored = torch.from_numpy(flat.reshape((rows, *target)).copy())
        values[spec.name] = restored
    if extra:
        values.update(extra)
    return dataclasses.replace(template, **values)


def pack_metadata(batch: Any) -> list[bytes]:
    """Serialize the per-row Python metadata, one blob per example."""

    n = int(batch.batch_size)
    columns = {}
    for name in META_FIELDS:
        value = getattr(batch, name)
        if not isinstance(value, (tuple, list)) or len(value) != n:
            raise PackedCollatedBatchError(
                f"metadata field {name!r} is not one entry per row "
                f"({type(value).__name__}, expected {n})"
            )
        columns[name] = value
    return [
        pickle.dumps(
            {name: columns[name][row] for name in META_FIELDS},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for row in range(n)
    ]


def unpack_metadata(blobs: Sequence[bytes]) -> dict[str, tuple[Any, ...]]:
    """Rebuild the per-row metadata columns from row blobs, in order."""

    rows = [pickle.loads(blob) for blob in blobs]
    return {name: tuple(row[name] for row in rows) for name in META_FIELDS}


def assert_roundtrip(batch: Any, layout: PackedLayout) -> None:
    """Raise unless packing and unpacking reproduces every tensor exactly.

    This is what makes the narrowing safe. The dtype choices come from a range
    scan, and a scan can be wrong; this check cannot be, because it compares
    the rebuilt tensors against the source that is being replaced.
    """

    records = pack_slice(batch, layout)
    rebuilt = unpack_rows(records, layout, batch)
    for spec in layout.fields:
        original = getattr(batch, spec.name)
        restored = getattr(rebuilt, spec.name)
        if original.dtype != restored.dtype or not torch.equal(original, restored):
            raise PackedCollatedBatchError(
                f"packed field {spec.name!r} does not round-trip exactly"
            )
    restored_meta = unpack_metadata(pack_metadata(batch))
    for name, column in restored_meta.items():
        if column != tuple(getattr(batch, name)):
            raise PackedCollatedBatchError(
                f"metadata field {name!r} does not round-trip exactly"
            )


class PackedCollatedStore:
    """Memory-mapped random access to packed collated rows."""

    def __init__(self, directory: Path | str) -> None:
        directory = Path(directory)
        header = json.loads((directory / "PACKED_HEADER.json").read_text())
        if header.get("schema") != SCHEMA:
            raise PackedCollatedBatchError("packed store header schema disagrees")
        self.header = header
        self.layout = PackedLayout.from_payload(header["layout"])
        self.row_by_entry_id: dict[str, int] = {
            entry_id: index for index, entry_id in enumerate(header["entry_ids"])
        }
        self._records = np.memmap(
            directory / "PACKED.bin",
            dtype=np.uint8,
            mode="r",
            shape=(len(header["entry_ids"]), self.layout.record_bytes),
        )
        self._meta = np.memmap(directory / "PACKED_META.bin", dtype=np.uint8, mode="r")
        self._meta_offsets = np.load(directory / "PACKED_META_OFFSETS.npy")
        if int(self._meta_offsets.shape[0]) != len(header["entry_ids"]) + 1:
            raise PackedCollatedBatchError(
                "packed metadata offsets disagree with the row count"
            )

    def __len__(self) -> int:
        return int(self._records.shape[0])

    def rows_for(
        self, entry_ids: Sequence[str], template: Any, *, extra: Mapping[str, Any] | None = None
    ) -> Any:
        """Gather these rows, IN THE ORDER GIVEN, into one collated batch."""

        try:
            indices = [self.row_by_entry_id[entry_id] for entry_id in entry_ids]
        except KeyError as error:
            raise PackedCollatedBatchError(
                "training stream references a row outside the packed store"
            ) from error
        records = np.ascontiguousarray(self._records[indices])
        blobs = [
            self._meta[self._meta_offsets[i] : self._meta_offsets[i + 1]].tobytes()
            for i in indices
        ]
        merged: dict[str, Any] = dict(unpack_metadata(blobs))
        if extra:
            merged.update(extra)
        return unpack_rows(records, self.layout, template, extra=merged)


__all__ = [
    "BOOL_FIELDS",
    "FLOAT_FIELDS",
    "INT_FIELDS",
    "META_FIELDS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "FieldSpec",
    "PackedCollatedBatchError",
    "PackedCollatedStore",
    "PackedLayout",
    "assert_roundtrip",
    "pack_metadata",
    "pack_slice",
    "plan_layout",
    "unpack_metadata",
    "unpack_rows",
]
