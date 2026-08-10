#!/usr/bin/env python
"""Repack compiled BATCH.pt slices into one row-addressable, narrowed store.

Training must not re-collate. MEASURED: a fresh 32-example collation costs
98.17 s against 2.38 s of model math, because it runs RDKit over every
candidate successor. The compile already paid that and stored the result; this
turns the slice-addressed result into a row-addressed one so a shuffled
minibatch is a gather rather than ~700 MB of whole-slice reads.

DTYPES ARE STRUCTURAL, AND CHECKED ANYWAY
-----------------------------------------
The narrow widths are not guesses from a sample. ``max_atoms`` is 40 and the
atom vocabulary is 15, so slot indices, bond classes, topology classes, charges
and hydrogen counts are all int8 by construction; only successor-group ids can
exceed a byte, and they are bounded far below int16. MEASURED ranges agree.

Even so, every written slice is unpacked and compared ``torch.equal`` against
its source before it is accepted. A value that does not fit raises rather than
wrapping -- int64 300 cast to int8 is 44, a plausible group id that would train
silently as a wrong operand. If an assert ever fires, widen that field and
rerun; the store is written once and read for every epoch afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from compose_v4.data.durable_path import require_durable_path  # noqa: E402
from compose_v4.data.packed_collated_batch import (  # noqa: E402
    INT_FIELDS,
    SCHEMA,
    SCHEMA_VERSION,
    PackedCollatedBatchError,
    assert_roundtrip,
    pack_slice,
    plan_layout,
)

#: Structural bounds, not sampled ones. max_atoms = 40, vocabulary = 15.
STRUCTURAL_INT_RANGES: dict[str, tuple[int, int]] = {
    "atom_types": (-1, 127),
    "formal_charges": (-8, 8),
    "implicit_h_counts": (-1, 16),
    "atom_topology": (-1, 127),
    "bonds": (-1, 127),
    "neural_bonds": (-1, 127),
    "ring_system_topology": (-1, 127),
    "closure_topology": (-1, 127),
    "graft_remove_neighbors": (-1, 127),
    # Successor-group ids: bounded by pairs of slots, so int16 with headroom.
    "graft_successor_groups": (-1, 32000),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True,
                        help="chunk root holding compiled slices")
    parser.add_argument("--out", required=True)
    parser.add_argument("--role", required=True, help="train | validation")
    parser.add_argument("--split-resolution", default="",
                        help="apply this role's precedence exclusions")
    parser.add_argument("--allow-reapable", action="store_true")
    args = parser.parse_args()

    roots = [
        Path(r) if args.allow_reapable else require_durable_path(Path(r), role="chunk root")
        for r in args.root
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    excluded: set[str] = set()
    if args.split_resolution:
        resolution = json.loads(Path(args.split_resolution).read_text())
        excluded = set(resolution["excluded_source_keys"][args.role])

    slices: list[Path] = []
    for root in roots:
        slices.extend(sorted(p.parent for p in root.rglob("BATCH.pt")))
    if not slices:
        raise SystemExit("no compiled slice found under any root; refusing to write")
    print(f"slices          {len(slices):,}")

    layout = None
    entry_ids: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    dropped = 0
    started = time.perf_counter()
    digest = hashlib.sha256()

    with (out / "PACKED.bin").open("wb") as sink:
        for position, directory in enumerate(slices):
            batch = torch.load(directory / "BATCH.pt", map_location="cpu", weights_only=False)
            entries = json.loads((directory / "ENTRIES.json").read_text())["entries"]
            if len(entries) != int(batch.batch_size):
                raise SystemExit(
                    f"{directory}: {len(entries)} entries but {batch.batch_size} rows; "
                    "the row-to-entry alignment cannot be assumed"
                )
            if layout is None:
                layout = plan_layout(batch, int_ranges=STRUCTURAL_INT_RANGES)
                print(f"record bytes    {layout.record_bytes:,} "
                      f"({layout.record_bytes / 1024:.2f} kB/example)")

            # Reject the whole slice if it does not round-trip. Doing this per
            # slice rather than once keeps a bad field from being discovered
            # after the store is already published.
            assert_roundtrip(batch, layout)
            records = pack_slice(batch, layout)

            keep: list[int] = []
            for row, entry in enumerate(entries):
                entry_id = str(entry["p50_entry_sha256"])
                if entry_id in seen:
                    duplicates += 1
                    continue
                source = entry["teacher_successor_fiber"]
                if source is None:
                    raise SystemExit(f"{directory} row {row} has no teacher fiber")
                if str(source["source_key"]) in excluded:
                    dropped += 1
                    continue
                seen.add(entry_id)
                entry_ids.append(entry_id)
                keep.append(row)

            if keep:
                block = np.ascontiguousarray(records[keep])
                sink.write(block.tobytes())
                digest.update(block.tobytes())

            if (position + 1) % 400 == 0 or position + 1 == len(slices):
                elapsed = time.perf_counter() - started
                print(f"  {position + 1:5,}/{len(slices):,} slices  "
                      f"{len(entry_ids):8,} rows  {elapsed:6.0f}s")

    assert layout is not None
    header = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "PACKED_COLLATED_STORE_EVIDENCE_ONLY_NO_AUTHORITY",
        "role": args.role,
        "roots": [str(r) for r in roots],
        "slice_count": len(slices),
        "row_count": len(entry_ids),
        "duplicate_rows_skipped": duplicates,
        "rows_dropped_by_precedence": dropped,
        "records_sha256": digest.hexdigest(),
        "layout": layout.to_payload(),
        "entry_ids": entry_ids,
    }
    (out / "PACKED_HEADER.json").write_text(json.dumps(header) + "\n")

    written = (out / "PACKED.bin").stat().st_size
    print()
    print(f"rows            {len(entry_ids):,}")
    print(f"duplicates      {duplicates}")
    print(f"dropped         {dropped}")
    print(f"PACKED.bin      {written / 1e9:.3f} GB")
    print(f"records_sha256  {digest.hexdigest()}")
    print(f"wall            {time.perf_counter() - started:.0f}s")
    if written != len(entry_ids) * layout.record_bytes:
        raise SystemExit("written bytes disagree with row count times record size")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackedCollatedBatchError as error:
        print(f"REFUSING: {error}")
        raise SystemExit(2) from error
