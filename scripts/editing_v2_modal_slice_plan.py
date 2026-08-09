"""Emit a Modal slice list covering ONLY records nothing has compiled yet.

WHY THIS EXISTS
---------------
Publication is immutable and write-if-absent, but there is no pre-compute
skip: re-running a slice recompiles every entry and only then discovers the
write is a no-op. The money is spent before the redundancy is noticed. At
~11.5 s/entry on Modal that is a real bill for zero new rows.

Two publishers are in play and neither can see the other -- the local driver
writes to a laptop directory, the fan-out writes to the Modal volume -- so
"what is already done" is the UNION of two sets that must be supplied
separately.

WHY INTERVALS, NOT SLICE NAMES
------------------------------
The two publishers do not share a grid. The local driver publishes 250-entry
slices; a Modal container sub-divides its slice for deadline safety and
publishes 15-entry receipts. So ``000000000-000000250`` and
``000000000-000000015`` describe overlapping work under different names, and
any comparison by directory name silently double-counts. Coverage is therefore
accumulated as half-open record intervals per task, and the emitted plan is the
complement -- which is correct whatever grid either side used.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def _covered_from_local(root: Path) -> dict[str, set[int]]:
    """Record indices already published under a local ``chunks/`` tree."""

    covered: dict[str, set[int]] = {}
    chunks = Path(root) / "chunks"
    if not chunks.is_dir():
        return covered
    for task in chunks.iterdir():
        if not task.is_dir():
            continue
        for slice_dir in task.iterdir():
            if not (slice_dir / "RECEIPT.json").exists():
                continue
            offset_text, _, count_text = slice_dir.name.partition("-")
            offset, count = int(offset_text), int(count_text)
            covered.setdefault(task.name, set()).update(range(offset, offset + count))
    return covered


def _covered_from_listing(path: Path) -> dict[str, set[int]]:
    """Record indices already on the volume, from ``modal volume ls -r`` output.

    Only lines naming a RECEIPT are counted. A slice directory that exists
    without its receipt is an interrupted write, and treating it as done would
    put an unpublished hole in the corpus.
    """

    covered: dict[str, set[int]] = {}
    if not path or not Path(path).exists():
        return covered
    seen_receipt_lines = 0
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if "RECEIPT.json" not in line:
            continue
        seen_receipt_lines += 1
        # The listing may be rooted ("editing_v2/.../chunks/<task>/...") or
        # relative ("chunks/<task>/..."), so anchor on the segment itself
        # rather than on "/chunks/" -- requiring the leading slash silently
        # matched NOTHING against a relative listing, and a zero here is
        # indistinguishable from "the volume is empty". That mistake costs a
        # full recompile of work already paid for.
        marker = "chunks/"
        index = line.find(marker)
        if index < 0:
            continue
        parts = line[index + len(marker):].split("/")
        if len(parts) < 3:
            continue
        task, slice_id = parts[0], parts[1]
        offset_text, _, count_text = slice_id.partition("-")
        if not (offset_text.isdigit() and count_text.isdigit()):
            continue
        offset, count = int(offset_text), int(count_text)
        covered.setdefault(task, set()).update(range(offset, offset + count))
    if seen_receipt_lines and not covered:
        raise SystemExit(
            f"REFUSING TO PLAN: {path} names {seen_receipt_lines} receipts but none "
            "parsed into a task/offset-count. Planning would re-request work that is "
            "already published. Check the listing format."
        )
    return covered


def _runs(missing: list[int], limit: int) -> list[tuple[int, int]]:
    """Contiguous missing indices, chunked to at most ``limit`` each."""

    out: list[tuple[int, int]] = []
    if not missing:
        return out
    start = previous = missing[0]
    for index in missing[1:]:
        if index != previous + 1 or index - start >= limit:
            out.append((start, previous - start + 1))
            start = index
        previous = index
    out.append((start, previous - start + 1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active8-root", required=True)
    parser.add_argument("--manifest", default="diagnostics/editing_v2_v2_selection_manifest.json")
    parser.add_argument(
        "--lanes",
        default="operator_aware_real_endpoint,linker_positional_topology_analogue",
    )
    parser.add_argument("--local-root", action="append", default=[],
                        help="Local chunks/ tree already published. Repeatable.")
    parser.add_argument("--modal-listing", default="",
                        help="Output of `modal volume ls -r <vol> <prefix>`.")
    parser.add_argument("--slice-size", type=int, default=125)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    covered: dict[str, set[int]] = {}
    for root in args.local_root:
        for task, indices in _covered_from_local(Path(root)).items():
            covered.setdefault(task, set()).update(indices)
    for task, indices in _covered_from_listing(Path(args.modal_listing)).items():
        covered.setdefault(task, set()).update(indices)

    manifest = json.loads(Path(args.manifest).read_text())
    active8 = Path(args.active8_root)
    rows: list[dict[str, int | str]] = []
    total_records = total_missing = skipped_non_train = 0

    for lane in [lane for lane in args.lanes.split(",") if lane]:
        for task in sorted(manifest["selection"].get(lane, {})):
            stream = active8 / "tasks" / task / "transitions.jsonl.gz"
            if not stream.exists():
                continue
            with gzip.open(stream, "rt") as handle:
                first = json.loads(handle.readline())
                count = 1 + sum(1 for _ in handle)
            # Held-out roles are not train data. Sending them costs a container
            # that fails with "no accepted train transitions" and returns zero
            # rows -- six such tasks were queued before this filter existed.
            if first.get("partition_role") != "train":
                skipped_non_train += 1
                continue
            total_records += count
            done = covered.get(task, set())
            missing = [index for index in range(count) if index not in done]
            total_missing += len(missing)
            for offset, limit in _runs(missing, int(args.slice_size)):
                rows.append({
                    "task_identity_sha256": task,
                    "entry_offset": offset,
                    "limit": limit,
                })

    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"train records            {total_records:,}")
    print(f"already published        {total_records - total_missing:,}")
    print(f"still missing            {total_missing:,}")
    print(f"non-train tasks skipped  {skipped_non_train}")
    print(f"slices emitted           {len(rows)} (<= {args.slice_size} entries each)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
