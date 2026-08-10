"""The packed store replaces the authoritative compiled tensors, so it must be
bit-exact, and it must fail loudly when it cannot be.

Narrowing int64 to int8 and bool to bits is only safe while every value fits.
The dtype choice comes from a range scan, and a scan over part of a corpus can
be wrong -- so the failure mode that matters is a value silently wrapping into
a valid-looking small integer. These tests pin that it raises instead.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from compose_v4.data.packed_collated_batch import (
    BOOL_FIELDS,
    META_FIELDS,
    FLOAT_FIELDS,
    INT_FIELDS,
    PackedCollatedBatchError,
    PackedLayout,
    assert_roundtrip,
    pack_metadata,
    pack_slice,
    plan_layout,
    unpack_metadata,
    unpack_rows,
)

SLOTS = 4
CLASSES = 3


def _int_shape(name: str) -> tuple[int, ...]:
    if name in ("atom_types", "formal_charges", "implicit_h_counts", "atom_topology"):
        return (SLOTS,)
    return (SLOTS, SLOTS)


def _bool_shape(name: str) -> tuple[int, ...]:
    if name == "cycle_close_admission_mask":
        return (SLOTS, SLOTS, CLASSES)
    if name == "atom_restate_admission_mask":
        return (SLOTS, CLASSES)
    if name in ("atom_delete_mask", "atom_delete_admission_mask"):
        return (SLOTS,)
    return (SLOTS, SLOTS)


@dataclass
class _Batch:
    """Stand-in with the real field names; the packer only uses getattr."""

    atom_types: torch.Tensor
    formal_charges: torch.Tensor
    implicit_h_counts: torch.Tensor
    atom_topology: torch.Tensor
    bonds: torch.Tensor
    neural_bonds: torch.Tensor
    ring_system_topology: torch.Tensor
    closure_topology: torch.Tensor
    graft_successor_groups: torch.Tensor
    graft_remove_neighbors: torch.Tensor
    atom_delete_mask: torch.Tensor
    atom_delete_admission_mask: torch.Tensor
    atom_restate_admission_mask: torch.Tensor
    cycle_edge_mask: torch.Tensor
    cycle_open_admission_mask: torch.Tensor
    cyclic_pair_mask: torch.Tensor
    graft_mask: torch.Tensor
    cycle_close_admission_mask: torch.Tensor
    times: torch.Tensor
    teacher_rates: torch.Tensor
    importance_weights: torch.Tensor
    teacher_actions: tuple = ()
    teacher_rule_names: tuple = ()
    ring_restate_actions: tuple = ()
    ring_restate_successor_group_ids: tuple = ()
    ring_restate_successor_group_descriptors: tuple = ()
    ring_restate_successor_group_multiplicities: tuple = ()
    ring_delete_actions: tuple = ()
    ring_teacher_semantic_certificates: tuple = ()
    marker: str = "constant-config-field"

    @property
    def batch_size(self) -> int:
        return int(self.atom_types.shape[0])


def _batch(n: int = 6, *, seed: int = 0, overrides=None) -> _Batch:
    rng = np.random.default_rng(seed)
    values: dict = {}
    for name in INT_FIELDS:
        shape = (n, *_int_shape(name))
        values[name] = torch.from_numpy(
            rng.integers(-1, 5, size=shape).astype(np.int64)
        )
    for name in BOOL_FIELDS:
        shape = (n, *_bool_shape(name))
        values[name] = torch.from_numpy(rng.integers(0, 2, size=shape).astype(bool))
    for name in FLOAT_FIELDS:
        values[name] = torch.from_numpy(rng.random(n).astype(np.float32))
    # Per-row Python metadata, deliberately DISTINCT per row so a reordering
    # or an off-by-one shows up instead of matching by accident.
    values["teacher_actions"] = tuple(None for _ in range(n))
    values["teacher_rule_names"] = tuple(None for _ in range(n))
    values["ring_restate_actions"] = tuple((f"act-{i}",) for i in range(n))
    values["ring_restate_successor_group_ids"] = tuple((i, i + 1) for i in range(n))
    values["ring_restate_successor_group_descriptors"] = tuple(
        ({"ring": i},) for i in range(n)
    )
    values["ring_restate_successor_group_multiplicities"] = tuple((i + 1,) for i in range(n))
    values["ring_delete_actions"] = tuple(() for _ in range(n))
    values["ring_teacher_semantic_certificates"] = tuple(() for _ in range(n))
    if overrides:
        values.update(overrides)
    return _Batch(**values)


def _ranges(batch: _Batch) -> dict[str, tuple[int, int]]:
    return {
        name: (int(getattr(batch, name).min()), int(getattr(batch, name).max()))
        for name in INT_FIELDS
    }


def test_round_trip_is_exact_for_every_field() -> None:
    batch = _batch()
    layout = plan_layout(batch, int_ranges=_ranges(batch))
    assert_roundtrip(batch, layout)


def test_narrowing_picks_the_smallest_lossless_width() -> None:
    batch = _batch()
    ranges = _ranges(batch)
    ranges["graft_successor_groups"] = (-1, 313)  # needs int16
    layout = plan_layout(batch, int_ranges=ranges)
    assert layout.field("bonds").stored_dtype == "int8"
    assert layout.field("graft_successor_groups").stored_dtype == "int16"


def test_a_value_outside_the_declared_range_raises_instead_of_wrapping() -> None:
    """The failure this whole design has to avoid.

    If the range scan under-measures, casting int64 300 to int8 yields 44 --
    a perfectly plausible group id that would train silently as a wrong
    operand. Packing must refuse.
    """

    batch = _batch()
    ranges = _ranges(batch)
    poisoned = batch.graft_successor_groups.clone()
    poisoned[0, 0, 0] = 300  # beyond the int8 the scan would have chosen
    batch = dataclasses.replace(batch, graft_successor_groups=poisoned)
    layout = plan_layout(batch, int_ranges=ranges)  # ranges still say int8
    assert layout.field("graft_successor_groups").stored_dtype == "int8"
    with pytest.raises(PackedCollatedBatchError, match="does not fit"):
        pack_slice(batch, layout)


def test_bitpacking_handles_counts_that_are_not_multiples_of_eight() -> None:
    """(4, 3) is 12 bits; a careless unpack returns 16 and reshape explodes."""

    batch = _batch()
    layout = plan_layout(batch, int_ranges=_ranges(batch))
    spec = layout.field("atom_restate_admission_mask")
    assert spec.count == SLOTS * CLASSES == 12
    assert spec.nbytes == 2
    assert_roundtrip(batch, layout)


def test_rows_come_back_in_the_order_requested_with_repeats() -> None:
    """The sampler owns order and draws repeats; storage must not reorder."""

    batch = _batch(n=6)
    layout = plan_layout(batch, int_ranges=_ranges(batch))
    records = pack_slice(batch, layout)
    order = [4, 0, 4, 2]
    rebuilt = unpack_rows(np.ascontiguousarray(records[order]), layout, batch)
    assert rebuilt.batch_size == len(order)
    for position, source in enumerate(order):
        assert torch.equal(rebuilt.bonds[position], batch.bonds[source])
        assert torch.equal(rebuilt.graft_mask[position], batch.graft_mask[source])


def test_non_tensor_configuration_fields_survive() -> None:
    """Compile-time constants ride along on the template, not the records."""

    batch = _batch()
    layout = plan_layout(batch, int_ranges=_ranges(batch))
    rebuilt = unpack_rows(pack_slice(batch, layout), layout, batch)
    assert rebuilt.marker == "constant-config-field"


def test_record_is_fixed_size_and_offsets_tile_it_exactly() -> None:
    """Fixed-size records are what make row addressing possible at all."""

    batch = _batch()
    layout = plan_layout(batch, int_ranges=_ranges(batch))
    offset = 0
    for spec in layout.fields:
        assert spec.offset == offset
        offset += spec.nbytes
    assert offset == layout.record_bytes
    assert pack_slice(batch, layout).shape == (batch.batch_size, layout.record_bytes)


def test_layout_survives_a_json_round_trip() -> None:
    batch = _batch()
    layout = plan_layout(batch, int_ranges=_ranges(batch))
    assert PackedLayout.from_payload(layout.to_payload()) == layout


def test_packed_form_is_much_smaller_than_the_compiled_tensors() -> None:
    """The reason the store exists: 87.85 kB/example measured, 12.56 packed."""

    batch = _batch()
    layout = plan_layout(batch, int_ranges=_ranges(batch))
    original = sum(
        getattr(batch, spec.name)[0].element_size()
        * getattr(batch, spec.name)[0].nelement()
        for spec in layout.fields
    )
    assert layout.record_bytes < original / 3


def test_per_row_metadata_round_trips_and_is_not_a_constant() -> None:
    """The bug this caught: ring-restate metadata is per ROW, not per compile.

    Taking it from a template made the model raise "semantic ring-restatement
    metadata does not align with the batch" -- but only once a forward ran, and
    only because the template happened to have a different row count. With a
    matching count it would have silently trained on another row's metadata.
    """

    batch = _batch(n=6)
    restored = unpack_metadata(pack_metadata(batch))
    for name in META_FIELDS:
        assert restored[name] == tuple(getattr(batch, name)), name
    assert len({tuple(v) for v in batch.ring_restate_successor_group_ids}) == 6


def test_metadata_blobs_follow_the_requested_row_order() -> None:
    batch = _batch(n=6)
    blobs = pack_metadata(batch)
    order = [3, 0, 3]
    restored = unpack_metadata([blobs[i] for i in order])
    assert restored["ring_restate_actions"] == tuple(
        batch.ring_restate_actions[i] for i in order
    )


def test_metadata_field_with_the_wrong_length_is_refused() -> None:
    """A template's tuples are the wrong length for a gathered minibatch."""

    batch = _batch(n=6)
    batch = dataclasses.replace(batch, ring_restate_actions=(("only-one",),))
    with pytest.raises(PackedCollatedBatchError, match="one entry per row"):
        pack_metadata(batch)
